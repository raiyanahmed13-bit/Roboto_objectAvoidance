"""End-to-end discrepancy detection against known ground truth.

The experimental design that makes this measurable: we build TWO worlds
that differ in known ways.

    prior world   = OSM footprints, exactly as the GIS pipeline produced
    truth world   = prior + obstacles that OSM does not contain

The robot drives the prior-planned route through the TRUTH world, so its
lidar sees the extra obstacles. Comparing its SLAM map against the prior
should recover exactly the obstacles we injected -- and because we placed
them, precision and recall are computable rather than hand-labelled.

BE CLEAR ABOUT WHAT THIS DOES NOT CLAIM. Ground truth here is a simulated
world we generated, perturbed by known synthetic obstacles. This is not a
claim to detect real errors in OpenStreetMap.

Usage:
    python -m tools.run_discrepancy_offline
"""

from __future__ import annotations

import argparse

import numpy as np

from roboto_core.discrepancy.cluster import cluster
from roboto_core.discrepancy.compare import Cls, DiscrepancyConfig, classify
from roboto_core.frames import GridSpec
from roboto_core.plan.astar import AStarPlanner, TerrainWeights
from roboto_core.sim.odom_corruptor import OdomCorruptor
from roboto_core.sim.raycast_sim import Lidar2D, LidarSpec, RaycastWorld
from roboto_core.sim.scenario import (catalogue_from, describe,
                                      rasterise, sample_placements,
                                      scenario_rng)
from roboto_core.slam.slam import Slam2D
from tools.gis_pipeline.build_slope import TerrainModel
from tools.gis_pipeline.common.crs import load_site
from tools.gis_pipeline.common.paths import derived_dir
from tools.run_slam_offline import trajectory_from_path


def inject(prior_occ, grid: GridSpec, traj, seed: int = 0,
           n_range: tuple[int, int] = (6, 6), types=None):
    """Return (truth_occ, ground_truth, placements) with obstacles added.

    Roadside mode: obstacles sit ALONGSIDE the route rather than across it,
    so the robot passes them and has to detect them without being forced
    to. Drawn per scenario seed, so seeds are independent trials.
    """
    places = sample_placements(scenario_rng(seed), traj, prior_occ, grid,
                               mode="roadside", n_range=n_range,
                               catalogue=catalogue_from(types))
    truth, gts = rasterise(prior_occ, grid, places)
    return truth, gts, places


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--site", default=None)
    ap.add_argument("--goal", nargs=2, type=float, default=None,
                    help="defaults to the site's mission.goal")
    ap.add_argument("--spacing", type=float, default=0.4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--seeds", type=int, default=1,
                    help="run this many consecutive seeds and aggregate")
    ap.add_argument("--scenario-seed", type=int, default=None,
                    help="obstacle layout seed. Omitted, it tracks --seed so "
                         "each trial is an independent scenario; pinned, the "
                         "same obstacles are reused and only noise varies")
    # Six, fixed, matching tools/gis_pipeline/inject_gazebo_obstacles.py.
    # A scenario seed must produce the SAME obstacles offline and in Gazebo;
    # when this defaulted to 2-4 while the injector used 6, that guarantee
    # was quietly broken and the reported numbers came from sparser
    # scenarios than the demo actually showed.
    # Restricting the catalogue is a DEMO convenience, not an experimental
    # one. Every type clears the 0.2 m2 detection floor (worst case 1.41x),
    # so the full catalogue is what the reported numbers should come from.
    # See roboto_core/sim/scenario.py:catalogue_from.
    ap.add_argument("--types", default=None,
                    help='comma-separated obstacle types, e.g. "delivery_truck,fallen_tree". Default: the whole catalogue. Narrowing it makes the scenario easier -- say so if a reported result used it')
    ap.add_argument("--min-obstacles", type=int, default=6)
    ap.add_argument("--max-obstacles", type=int, default=6)
    # Smallest MISSED cluster treated as a real object. This was hardcoded
    # at 0.5 while run_mission_offline.py used 0.2, so the two experiments
    # scored different detectors and the reported recall did not describe
    # the detector the missions actually ran. Same value in both now.
    ap.add_argument("--min-area", type=float, default=0.2)
    # PHANTOM needs a coarser floor than MISSED: a missing building leaves a
    # large hole, whereas an unmapped skip is small, so the thing that makes
    # MISSED sensitive would make PHANTOM fire on ordinary map noise.
    ap.add_argument("--phantom-min-area", type=float, default=2.0)
    ap.add_argument("--truth-poses", action="store_true",
                    help="map from ground-truth poses (isolates detection "
                         "from SLAM error)")
    args = ap.parse_args()

    if args.seeds > 1:
        return _multi(args)
    if args.scenario_seed is None:
        args.scenario_seed = args.seed
    _run(args)
    return 0


def _run(args) -> dict:
    """Run one mission; print a report and return its detection metrics."""
    site = load_site(args.site)
    grid = GridSpec.from_site(site)
    prior_cost = np.load(derived_dir(site) / "prior_cost.npz")["cost"]
    prior_world = RaycastWorld.from_occupancy_yaml(
        derived_dir(site) / "prior_occ.yaml")
    prior_occ = prior_world.occ
    terrain = TerrainModel.load(site)
    spec = LidarSpec.from_site(site)

    # --- plan on the PRIOR, as the robot would -----------------------------
    plan = AStarPlanner(prior_cost, grid, downsample=4, terrain=terrain,
                        weights=TerrainWeights.from_site(site)).plan(
        (site.spawn_x, site.spawn_y),
        tuple(args.goal) if args.goal else site.goal)
    if not plan.found:
        print(f"planning failed: {plan.reason}")
        return 1
    traj = trajectory_from_path(plan.path, args.spacing)
    print(f"route: {plan.length_m:.1f} m, {len(traj)} poses")

    # --- build the truth world --------------------------------------------
    truth_occ, gts, places = inject(
        prior_occ, grid, traj, seed=args.scenario_seed,
        n_range=(args.min_obstacles, args.max_obstacles),
        types=getattr(args, "types", None))
    print(f"scenario seed {args.scenario_seed}: "
          f"{len(gts)} obstacles absent from the prior")
    print(describe(places))

    # --- drive it ----------------------------------------------------------
    world = RaycastWorld(truth_occ, grid)
    lidar = Lidar2D(world, spec, rng=args.seed)
    scans = [lidar.scan(p) for p in traj]
    odom = OdomCorruptor(seed=args.seed).corrupt_trajectory(traj, dt=args.spacing)

    # Matched against prior-or-mapped obstacles. Pure self-map matching
    # erodes 63% of real walls here, which floods PHANTOM detection; see
    # the reference ablation in SlamConfig.
    slam = Slam2D(grid, sensor_range=spec.range_max, initial_pose=traj[0],
                  prior_occ=prior_occ)
    for sc, od, gt in zip(scans, odom, traj):
        slam.update(sc, gt if args.truth_poses else od)

    a = slam.ate(traj)
    ate = a["rmse_m"]
    print(f"\nSLAM: ATE RMSE {a['rmse_m']:.2f} m "
          f"(odometry {slam.odom_ate(traj)['rmse_m']:.2f} m), "
          f"{slam.n_rejected} rejected of {len(scans)}")

    # --- compare against the PRIOR ----------------------------------------
    cfg = DiscrepancyConfig()
    res = classify(slam.map, prior_occ, cfg)
    s_ = res.summary()
    print(f"\nclassification (observed {100 * s_['observed_fraction']:.1f}% of grid, "
          f"registration offset {s_['offset_m'] * 100:.0f} cm / {s_['offset_deg']:+.2f} deg)")
    print(f"  agreement over classified cells : {100 * s_['agreement']:.1f}%")
    print(f"  MISSED  {s_['missed_cells']:7,d} cells ({s_['missed_m2']:8.1f} m2)")
    print(f"  PHANTOM {s_['phantom_cells']:7,d} cells ({s_['phantom_m2']:8.1f} m2)")

    # --- objects and scoring ----------------------------------------------
    missed = cluster(res, Cls.MISSED, min_area_m2=args.min_area,
                     log_odds=slam.map.log_odds)
    phantom = cluster(res, Cls.PHANTOM, min_area_m2=args.phantom_min_area,
                      log_odds=slam.map.log_odds)
    print(f"\nobjects: {len(missed)} MISSED, {len(phantom)} PHANTOM")

    # Distance to the obstacle's actual rotated rectangle, not its bounding
    # box. Fairer than centroid-to-centroid: a lidar sees only the NEAR
    # FACE, so a correct detection sits on the object's boundary, roughly
    # half its depth from its centre. Centroid error charges the detector
    # for the sensor's geometry rather than for being wrong -- and an
    # axis-aligned bound would do the opposite, crediting it for slack.
    print("\nground-truth obstacle recovery:")
    hits, errs, bnd = 0, [], []
    for g in gts:
        best, bd = None, 1e9
        for o in missed:
            d = float(np.hypot(*(o.centroid - g["centroid"])))
            if d < bd:
                best, bd = o, d
        ok = best is not None and bd < 3.0
        hits += ok
        if ok:
            db = g["place"].distance_to(best.centroid)
            errs.append(bd)
            bnd.append(db)
            print(f"   {g['label']:22s} DETECTED  to-boundary {db:4.2f} m, "
                  f"to-centre {bd:4.2f} m, area {best.area_m2:5.1f} m2 "
                  f"(true {g['area_m2']:.1f})")
        else:
            near = f"{bd:.1f} m away" if best is not None else "none found"
            print(f"   {g['label']:22s} MISSED    nearest detection {near}")

    fp = [o for o in missed
          if min((float(np.hypot(*(o.centroid - g['centroid']))) for g in gts),
                 default=1e9) >= 3.0]
    print(f"\n  recall    {hits}/{len(gts)}")
    print(f"  false pos {len(fp)} spurious MISSED objects")
    if fp:
        print(f"    largest: {fp[0].area_m2:.1f} m2 at "
              f"({fp[0].centroid[0]:.0f}, {fp[0].centroid[1]:.0f})")

    return {"ate": ate, "hits": hits, "n_gt": len(gts), "fp": len(fp),
            "errs": errs, "bnd": bnd, "missed_m2": s_["missed_m2"],
            "phantom_m2": s_["phantom_m2"]}


def _multi(args) -> int:
    """Repeat the mission over several seeds and aggregate.

    Single-run numbers have already misled this project once (the score-trim
    sweep looked fine on one seed and was unstable across three), so the
    headline detection figures are reported over several.
    """
    import contextlib
    import io

    rows = []
    base = args.seeds
    for k in range(base):
        one = argparse.Namespace(**vars(args))
        one.seeds, one.seed = 1, args.seed + k
        # Pinning --scenario-seed reuses one obstacle layout across trials,
        # isolating sensor noise. Left alone it advances with the seed, so
        # each trial is an independent scenario.
        one.scenario_seed = (args.seed + k if args.scenario_seed is None
                             else args.scenario_seed)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            metrics = _run(one)
        rows.append((args.seed + k, metrics, buf.getvalue()))

    total_gt = sum(m["n_gt"] for _, m, _ in rows)
    fixed = args.scenario_seed is not None
    print(f"{base} seeds, {total_gt} injected obstacles in total "
          f"({'one fixed scenario' if fixed else 'a fresh scenario each'})")
    print()
    print("  seed   ATE RMSE   recall   false pos   centroid err")
    tot_hit = tot_gt = tot_fp = 0
    errs, bnd = [], []
    for seed, m, _ in rows:
        tot_hit += m["hits"]
        tot_gt += m["n_gt"]
        tot_fp += m["fp"]
        errs.extend(m["errs"])
        bnd.extend(m["bnd"])
        ce = f"{np.mean(m['errs']):.2f} m" if m["errs"] else "-"
        print(f"   {seed:3d}   {m['ate']:7.2f} m    {m['hits']}/{m['n_gt']}"
              f"        {m['fp']:2d}       {ce}")

    prec = tot_hit / (tot_hit + tot_fp) if (tot_hit + tot_fp) else 0.0
    rec = tot_hit / tot_gt if tot_gt else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    print()
    print(f"  recall    {tot_hit}/{tot_gt}  ({100 * rec:.1f} %)")
    print(f"  precision {tot_hit}/{tot_hit + tot_fp}  ({100 * prec:.1f} %)")
    print(f"  F1        {f1:.3f}")
    if errs:
        print(f"  localisation: to-boundary mean {np.mean(bnd):.2f} m "
              f"(max {np.max(bnd):.2f} m), to-centre mean {np.mean(errs):.2f} m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
