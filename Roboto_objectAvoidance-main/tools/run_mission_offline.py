"""Closed-loop mission: plan on the prior, discover, replan, arrive.

This is the experiment the whole project builds towards, and the ablation
that argues for the contribution:

    discrepancy layer OFF  ->  the robot drives into obstacles OSM
                               does not know about
    discrepancy layer ON   ->  it detects them, replans, and arrives

Obstacles are placed ON the planned route, not beside it, so that failing
to react has a consequence that can be counted. They are drawn at random
per scenario seed -- count, position, lateral offset, orientation and size
-- so repeated seeds are independent trials rather than one scenario
measured several times.

    plan over the GIS prior
        |
        v
    step along the path ---> lidar ---> SLAM ---> map
        |                                          |
        |                                          v
        |                              classify vs prior, cluster
        |                                          |
        |<--- replan over fused costmap <--- object blocks the path
        v
    goal (or collision)

Usage:
    python -m tools.run_mission_offline
    python -m tools.run_mission_offline --no-replan      # the ablation
    python -m tools.run_mission_offline --seeds 4
    python -m tools.run_mission_offline --seeds 4 --scenario-seed 0
                                        # ^ one scenario, four noise draws
"""

from __future__ import annotations

import argparse
import time

import numpy as np

from roboto_core.discrepancy.cluster import DiscrepancyTracker, cluster
from roboto_core.discrepancy.compare import Cls, DiscrepancyConfig, classify
from roboto_core.frames import GridSpec
from roboto_core.plan.astar import AStarPlanner, objectives
from roboto_core.plan.cost_fusion import fuse, path_is_blocked
from roboto_core.plan.detour import plan_detour
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


def place_blockers(prior_occ, grid: GridSpec, traj, seed: int = 0,
                   n_range: tuple[int, int] = (6, 6), types=None):
    """Draw obstacles straddling the path. Returns (truth_occ, ground truth).

    Blocking mode: each obstacle lies across the route and spans it, so a
    robot that ignores the discrepancy layer has to hit something.
    """
    places = sample_placements(scenario_rng(seed), traj, prior_occ, grid,
                               mode="blocking", n_range=n_range,
                               catalogue=catalogue_from(types))
    truth, gts = rasterise(prior_occ, grid, places)
    return truth, gts, places


def run(args) -> dict:
    if getattr(args, "scenario_seed", None) is None:
        args.scenario_seed = args.seed
    site = load_site(args.site)
    grid = GridSpec.from_site(site)
    prior_cost = np.load(derived_dir(site) / "prior_cost.npz")["cost"]
    prior_occ = RaycastWorld.from_occupancy_yaml(
        derived_dir(site) / "prior_occ.yaml").occ
    terrain = TerrainModel.load(site)
    spec = LidarSpec.from_site(site)
    r_robot = float(site.raw["robot"]["radius_m"])
    objs = objectives(site)
    if args.objective not in objs:
        raise SystemExit(f"unknown objective {args.objective!r}; "
                         f"choose from {sorted(objs)}")
    objective = objs[args.objective]
    goal = tuple(args.goal) if args.goal else site.goal

    def make_planner(cost):
        return AStarPlanner(cost, grid, downsample=4, terrain=terrain,
                            objective=objective,
                            heuristic_weight=args.heuristic_weight)

    # --- the route the robot believes in ---------------------------------
    plan0 = make_planner(prior_cost).plan((site.spawn_x, site.spawn_y), goal)
    if not plan0.found:
        raise SystemExit(f"initial planning failed: {plan0.reason}")
    traj0 = trajectory_from_path(plan0.path, args.spacing)

    # --- the world as it actually is -------------------------------------
    truth_occ, gts, places = place_blockers(
        prior_occ, grid, traj0, seed=args.scenario_seed,
        n_range=(args.min_obstacles, args.max_obstacles),
        types=getattr(args, "types", None))
    world = RaycastWorld(truth_occ, grid)
    lidar = Lidar2D(world, spec, rng=args.seed)

    # Cells the robot must not enter: obstacle dilated by its own radius.
    from scipy.ndimage import distance_transform_edt
    clearance = distance_transform_edt(~truth_occ) * grid.resolution
    forbidden = clearance <= r_robot

    slam = Slam2D(grid, sensor_range=spec.range_max, initial_pose=traj0[0],
                  prior_occ=prior_occ)
    odo = OdomCorruptor(seed=args.seed)
    odo.reset(traj0[0])
    tracker = DiscrepancyTracker(confirm_after=args.confirm_after)
    dcfg = DiscrepancyConfig(register=False)   # registration once is enough

    traj = traj0
    i = 0
    travelled = 0.0
    collisions = 0
    replans = 0
    objects: list = []
    detections_at = []
    detours = globals_used = expanded = 0
    last_replan = -1e9
    t0 = time.perf_counter()

    max_steps = int(4 * len(traj0) + 500)
    for step in range(max_steps):
        if i >= len(traj):
            break
        pose = traj[i]

        # --- sense and estimate ------------------------------------------
        scan = lidar.scan(pose, stamp=step * args.spacing)
        slam.update(scan, odo.update(pose, dt=args.spacing))

        r, c = grid.world_to_cell(pose[0], pose[1])
        if grid.in_bounds(r, c) and forbidden[int(r), int(c)]:
            collisions += 1

        # --- detect and react --------------------------------------------
        if not args.no_replan and step % args.detect_every == 0 and step > 0:
            res = classify(slam.map, prior_occ, dcfg)
            found = cluster(res, Cls.MISSED, min_area_m2=args.min_area,
                            log_odds=slam.map.log_odds)
            tracker.update(found, stamp=step * args.spacing)
            objects = tracker.confirmed()

            if objects:
                # Plan conservatively, check permissively.
                #
                # A path planned with zero clearance is invalidated the
                # moment the obstacle grows by one cell -- and it grows on
                # every cycle, because each closer look reveals more of it.
                # That is what produced 124 replans in one mission. Planning
                # with extra margin means the obstacle must grow through the
                # margin before the path is blocked again, so the robot
                # commits to a route instead of re-deciding every metre.
                fused = fuse(prior_cost, grid, objects, robot_radius_m=r_robot)
                fused_plan = fuse(prior_cost, grid, objects,
                                  robot_radius_m=r_robot + args.plan_margin_m)

                # Two questions, not one. "Is something in the way NOW?"
                # must always be answered; "will something be in the way
                # eventually?" can wait for a better look. Replanning on the
                # second every cycle is what caused 124 replans in one
                # mission: each new view of an obstacle grows its footprint
                # and invalidates the path planned a metre earlier.
                urgent = path_is_blocked(traj[:, :2], grid, fused,
                                         from_index=i, corridor_m=r_robot,
                                         within_m=args.emergency_m)
                eventual = path_is_blocked(traj[:, :2], grid, fused,
                                           from_index=i, corridor_m=r_robot)
                cooled = (travelled - last_replan) >= args.replan_cooldown_m

                if urgent or (eventual and cooled):
                    new_path = None

                    if args.replan_mode == "detour":
                        # Route around the obstacle and rejoin, rather than
                        # abandoning the remaining route. Falls back to a
                        # global replan when no rejoin is reachable -- if
                        # the obstacle blocks the only way through, staying
                        # local cannot help.
                        wide = make_planner(fused_plan)
                        d = plan_detour(lambda a, b: wide.plan(a, b),
                                        traj[:, :2], pose, grid, fused_plan,
                                        from_index=i, corridor_m=r_robot)
                        expanded += d.expanded
                        if d.found:
                            new_path = d.path
                            detours += 1

                    if new_path is None:
                        new = make_planner(fused_plan).plan(pose[:2], goal)
                        if not new.found:
                            # The margin can make the route infeasible; fall
                            # back to the nominal costmap rather than not
                            # moving.
                            new = make_planner(fused).plan(pose[:2], goal)
                        expanded += new.expanded
                        if new.found and len(new.path) > 1:
                            new_path = new.path
                            globals_used += 1

                    if new_path is not None and len(new_path) > 1:
                        traj = trajectory_from_path(new_path, args.spacing)
                        i = 0
                        replans += 1
                        last_replan = travelled
                        detections_at.append(travelled)
                        continue

        if i + 1 < len(traj):
            travelled += float(np.hypot(*(traj[i + 1, :2] - traj[i, :2])))
        i += 1

    reached = float(np.hypot(*(traj[min(i, len(traj) - 1), :2] - np.array(goal))))
    return {
        "seed": args.seed,
        "scenario_seed": args.scenario_seed,
        "placements": places,
        "collisions": collisions,
        "replans": replans,
        "travelled": travelled,
        "planned": plan0.length_m,
        "goal_gap": reached,
        "success": collisions == 0 and reached < 5.0,
        "objects": len(objects),
        "n_blockers": len(gts),
        "seconds": time.perf_counter() - t0,
        "detect_dists": detections_at,
        "detours": detours,
        "global_replans": globals_used,
        "expanded": expanded,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--site", default=None)
    ap.add_argument("--goal", nargs=2, type=float, default=None,
                    help="defaults to the site's mission.goal")
    ap.add_argument("--spacing", type=float, default=0.4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--seeds", type=int, default=1)
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
    # Poses between discrepancy checks; at 0.4 m spacing, 5 poses is every
    # 2 m. This was 25 (every 10 m), which is slower than obstacles are
    # passed: the robot drove through them and only mapped them afterwards,
    # so the layer scored 0/5 missions while appearing to detect fine --
    # recall is cumulative and does not care whether a detection arrived in
    # time to act on. Swept over 60 missions, 5 seeds, 6 obstacles:
    #
    #   every   min_area 0.2   min_area 0.5
    #       5       4/5            2-3/5
    #      10       2/5            1/5
    #      15       0/5            0/5
    #
    # The live ROS node never had this problem: it checks on a 1 s timer,
    # about every 0.64 m at driving speed, which is why the Gazebo demo
    # reached its goal while this harness did not.
    ap.add_argument("--detect-every", type=int, default=5,
                    help="poses between discrepancy checks")
    ap.add_argument("--confirm-after", type=int, default=2,
                    help="detections required before acting on an object")
    # Smallest cluster treated as a real object. The obstacles being missed
    # are the thin ones -- fence panels at 0.4 m wide, barriers at 0.6-1.2 m
    # -- which a 2D lidar sees as a near face only, so 0.5 m2 was discarding
    # genuine detections. Worth about one mission in five at every detection
    # rate, with no false positives observed (precision stayed 27/27).
    ap.add_argument("--min-area", type=float, default=0.2)
    ap.add_argument("--replan-cooldown-m", type=float, default=0.0,
                    help="minimum travel between non-urgent replans")
    # Kept, but it does nothing at usable detection rates: sweeping 10 vs
    # 20 m gave results identical seed-for-seed and node-for-node in five of
    # six configurations. Checking every 2 m finds a blockage well before
    # any emergency horizon matters. It only separates the arms when
    # detection is slow enough to be late, which is the case now fixed.
    ap.add_argument("--emergency-m", type=float, default=20.0,
                    help="blockage within this distance always replans")
    ap.add_argument("--plan-margin-m", type=float, default=1.0,
                    help="extra clearance used when planning, not when checking")
    ap.add_argument("--objective", default="balanced",
                    help="distance | balanced | time | effort | safe")
    ap.add_argument("--heuristic-weight", type=float, default=1.0,
                    help="0 = Dijkstra, 1 = A*, >1 = weighted A*")
    ap.add_argument("--replan-mode", choices=("global", "detour"),
                    default="global",
                    help="global: replan all the way to the goal. "
                         "detour: route around the obstacle and rejoin the "
                         "original path")
    ap.add_argument("--no-replan", action="store_true",
                    help="the ablation: ignore discrepancies entirely")
    args = ap.parse_args()

    rows = []
    for k in range(max(1, args.seeds)):
        one = argparse.Namespace(**vars(args))
        one.seed = args.seed + k
        # Pinning --scenario-seed holds the obstacles fixed across trials,
        # which isolates sensor noise. Left alone it advances with the seed,
        # so every trial is a different scenario.
        one.scenario_seed = (args.seed + k if args.scenario_seed is None
                             else args.scenario_seed)
        rows.append(run(one))

    mode = "discrepancy layer OFF" if args.no_replan else "discrepancy layer ON"
    fixed = args.scenario_seed is not None
    print(f"{mode}   {len(rows)} trial(s), "
          f"{'one fixed scenario' if fixed else 'a fresh scenario each'}")
    print(f"planned route {rows[0]['planned']:.1f} m")
    print()
    for m in rows:
        print(f"scenario seed {m['scenario_seed']}: "
              f"{m['n_blockers']} obstacle(s) ON the route")
        print(describe(m["placements"]))
    print()
    print("  seed  obstacles  collisions  replans  travelled  goal gap  success")
    for m in rows:
        print(f"   {m['seed']:3d}   {m['n_blockers']:7d}   {m['collisions']:7d}"
              f"   {m['replans']:5d}"
              f"   {m['travelled']:7.1f} m  {m['goal_gap']:6.2f} m"
              f"   {'yes' if m['success'] else 'NO'}")

    n = len(rows)
    print()
    print(f"  missions with a collision : {sum(m['collisions'] > 0 for m in rows)}/{n}")
    print(f"  successful missions       : {sum(m['success'] for m in rows)}/{n}")
    print(f"  obstacles placed          : {sum(m['n_blockers'] for m in rows)}"
          f" over {n} trial(s)")
    print(f"  mean collision steps      : {np.mean([m['collisions'] for m in rows]):.1f}")
    print(f"  mean replans              : {np.mean([m['replans'] for m in rows]):.1f}")
    print(f"  mean detour               : "
          f"{np.mean([m['travelled'] - m['planned'] for m in rows]):+.1f} m")
    print(f"  replan mode               : {args.replan_mode}")
    print(f"  local detours / global    : "
          f"{sum(m['detours'] for m in rows)} / "
          f"{sum(m['global_replans'] for m in rows)}")
    print(f"  mean nodes expanded       : "
          f"{np.mean([m['expanded'] for m in rows]):,.0f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
