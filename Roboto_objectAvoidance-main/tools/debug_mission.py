"""Instrumented single mission: when is each blocker seen, and how late?

Answers the question the aggregate metrics cannot: does the robot fail to
DETECT an obstacle, or detect it too LATE to do anything about it?
"""

from __future__ import annotations

import argparse

import numpy as np
from scipy.ndimage import distance_transform_edt

from roboto_core.discrepancy.cluster import DiscrepancyTracker, cluster
from roboto_core.discrepancy.compare import Cls, DiscrepancyConfig, classify
from roboto_core.frames import GridSpec
from roboto_core.plan.astar import AStarPlanner, TerrainWeights
from roboto_core.plan.cost_fusion import fuse, path_is_blocked
from roboto_core.sim.odom_corruptor import OdomCorruptor
from roboto_core.sim.raycast_sim import Lidar2D, LidarSpec, RaycastWorld
from roboto_core.sim.scenario import describe
from roboto_core.slam.slam import Slam2D
from tools.gis_pipeline.build_slope import TerrainModel
from tools.gis_pipeline.common.crs import load_site
from tools.gis_pipeline.common.paths import derived_dir
from tools.run_mission_offline import place_blockers
from tools.run_slam_offline import trajectory_from_path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--scenario-seed", type=int, default=None,
                    help="obstacle layout seed; defaults to --seed")
    ap.add_argument("--spacing", type=float, default=0.4)
    ap.add_argument("--detect-every", type=int, default=10)
    ap.add_argument("--confirm-after", type=int, default=2)
    args = ap.parse_args()

    site = load_site()
    grid = GridSpec.from_site(site)
    prior_cost = np.load(derived_dir(site) / "prior_cost.npz")["cost"]
    prior_occ = RaycastWorld.from_occupancy_yaml(
        derived_dir(site) / "prior_occ.yaml").occ
    terrain = TerrainModel.load(site)
    spec = LidarSpec.from_site(site)
    r_robot = float(site.raw["robot"]["radius_m"])
    weights = TerrainWeights.from_site(site)
    goal = site.goal

    plan0 = AStarPlanner(prior_cost, grid, 4, terrain=terrain,
                         weights=weights).plan((site.spawn_x, site.spawn_y), goal)
    traj0 = trajectory_from_path(plan0.path, args.spacing)
    scenario_seed = args.seed if args.scenario_seed is None else args.scenario_seed
    truth_occ, gts, places = place_blockers(prior_occ, grid, traj0,
                                            seed=scenario_seed)
    world = RaycastWorld(truth_occ, grid)
    lidar = Lidar2D(world, spec, rng=args.seed)
    clearance = distance_transform_edt(~truth_occ) * grid.resolution
    forbidden = clearance <= r_robot

    slam = Slam2D(grid, sensor_range=spec.range_max, initial_pose=traj0[0],
                  prior_occ=prior_occ)
    odo = OdomCorruptor(seed=args.seed)
    odo.reset(traj0[0])
    tracker = DiscrepancyTracker(confirm_after=args.confirm_after)
    dcfg = DiscrepancyConfig(register=False)

    print(f"route {plan0.length_m:.1f} m, {len(traj0)} poses, "
          f"scenario seed {scenario_seed}, blockers at:")
    print(describe(places))
    print()
    print("  step  travelled   nearest blocker   detected  confirmed  blocked?  event")

    traj, i, travelled = traj0, 0, 0.0
    collisions = 0
    first_hit = {}
    for step in range(4 * len(traj0) + 500):
        if i >= len(traj):
            break
        pose = traj[i]

        scan = lidar.scan(pose, stamp=step * args.spacing)
        slam.update(scan, odo.update(pose, dt=args.spacing))

        r, c = grid.world_to_cell(pose[0], pose[1])
        hit = grid.in_bounds(r, c) and forbidden[int(r), int(c)]
        collisions += bool(hit)

        if step % args.detect_every == 0 and step > 0:
            res = classify(slam.map, prior_occ, dcfg)
            found = cluster(res, Cls.MISSED, min_area_m2=0.5,
                            log_odds=slam.map.log_odds)
            tracker.update(found, stamp=step * args.spacing)
            conf = tracker.confirmed()

            dmin = min(float(np.hypot(*(pose[:2] - g["centroid"]))) for g in gts)
            blocked = False
            event = ""
            if conf:
                fused = fuse(prior_cost, grid, conf, robot_radius_m=r_robot)
                blocked = path_is_blocked(traj[:, :2], grid, fused,
                                          from_index=i, corridor_m=r_robot)
                if blocked:
                    new = AStarPlanner(fused, grid, 4, terrain=terrain,
                                       weights=weights).plan(pose[:2], goal)
                    event = ("REPLAN ok" if new.found
                             else f"REPLAN FAILED: {new.reason}")
                    if new.found and len(new.path) > 1:
                        traj = trajectory_from_path(new.path, args.spacing)
                        i = 0
                        print(f"  {step:4d}  {travelled:7.1f} m   {dmin:8.1f} m"
                              f"        {len(found):2d}        {len(conf):2d}"
                              f"       {'yes':5s}   {event}")
                        continue

            for g in gts:
                d = float(np.hypot(*(pose[:2] - g["centroid"])))
                near = [o for o in found
                        if float(np.hypot(*(o.centroid - g["centroid"]))) < 4.0]
                if near and g["label"] not in first_hit:
                    first_hit[g["label"]] = (travelled, d)

            if step % (args.detect_every * 5) == 0 or conf:
                print(f"  {step:4d}  {travelled:7.1f} m   {dmin:8.1f} m"
                      f"        {len(found):2d}        {len(conf):2d}"
                      f"       {'yes' if blocked else 'no':5s}   {event}")

        if i + 1 < len(traj):
            travelled += float(np.hypot(*(traj[i + 1, :2] - traj[i, :2])))
        i += 1

    print()
    print("first detection of each blocker:")
    for g in gts:
        if g["label"] in first_hit:
            t, d = first_hit[g["label"]]
            print(f"   {g['label']:22s} at {t:6.1f} m travelled, "
                  f"{d:5.1f} m away from it")
        else:
            print(f"   {g['label']:22s} NEVER DETECTED")
    print(f"\ncollision steps: {collisions}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
