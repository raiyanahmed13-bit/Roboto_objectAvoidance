"""Offline SLAM experiment: plan a route, drive it, map it, score it.

Runs the whole stack against the raycast simulator with no Gazebo and no
ROS, which makes a full 300 m mission take seconds instead of minutes. This
is the harness the evaluation matrix will be built on.

    A* over the GIS prior  ->  ground-truth trajectory
                                     |
                    +----------------+----------------+
                    v                                 v
            simulated lidar                   corrupted odometry
                    |                                 |
                    +--------------> SLAM <-----------+
                                      |
                            ATE vs ground truth, map IoU

Usage:
    python -m tools.run_slam_offline
    python -m tools.run_slam_offline --goal 95 110 --seed 3 --no-slam
"""

from __future__ import annotations

import argparse
import time

import numpy as np

from roboto_core.frames import GridSpec
from roboto_core.plan.astar import AStarPlanner, TerrainWeights
from roboto_core.sim.odom_corruptor import OdomCorruptor
from roboto_core.sim.raycast_sim import Lidar2D, LidarSpec, RaycastWorld
from roboto_core.slam.slam import Slam2D, SlamConfig
from roboto_core.slam.transforms2d import wrap_angle
from tools.gis_pipeline.build_slope import TerrainModel
from tools.gis_pipeline.common.crs import load_site
from tools.gis_pipeline.common.paths import derived_dir


def trajectory_from_path(path: np.ndarray, spacing: float = 0.2) -> np.ndarray:
    """Resample an A* path to even spacing and attach a tangent heading.

    A* returns cell-centre waypoints on the coarse search grid; the robot
    needs a dense pose stream at roughly the sensor rate.
    """
    if len(path) < 2:
        return np.zeros((0, 3))

    seg = np.hypot(*np.diff(path, axis=0).T)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    total = float(s[-1])
    if total < spacing:
        return np.zeros((0, 3))

    t = np.arange(0.0, total, spacing)
    x = np.interp(t, s, path[:, 0])
    y = np.interp(t, s, path[:, 1])

    # Heading from a smoothed tangent: raw differences on a grid path are
    # quantised to 45 degree steps and make the robot jitter.
    dx = np.gradient(x)
    dy = np.gradient(y)
    k = min(9, max(3, len(x) // 20) | 1)
    ker = np.ones(k) / k
    dx = np.convolve(dx, ker, mode="same")
    dy = np.convolve(dy, ker, mode="same")
    return np.column_stack([x, y, np.arctan2(dy, dx)])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--site", default=None)
    ap.add_argument("--goal", nargs=2, type=float, default=None,
                    help="defaults to the site's mission.goal")
    ap.add_argument("--spacing", type=float, default=0.2, help="metres between poses")
    ap.add_argument("--seed", type=int, default=0)
    # Matches the live default. The offline sweep and one live run
    # disagree about this value; see SlamConfig.max_correction_m for
    # both sets of measurements and why they do not actually conflict.
    ap.add_argument("--max-correction-m", type=float, default=0.3)
    ap.add_argument("--slope-weight", type=float, default=1.0)
    ap.add_argument("--no-slam", action="store_true",
                    help="map from ground truth instead (isolates mapping)")
    args = ap.parse_args()

    site = load_site(args.site)
    grid = GridSpec.from_site(site)
    prior = np.load(derived_dir(site) / "prior_cost.npz")["cost"]
    terrain = TerrainModel.load(site)

    print(f"site '{site.name}'")

    # --- plan ------------------------------------------------------------
    k = args.slope_weight
    weights = TerrainWeights(uphill=6.0 * k, downhill=1.8 * k, cross=3.6 * k,
                             max_slope_deg=site.raw["slope"]["max_traversable_deg"])
    t0 = time.perf_counter()
    plan = AStarPlanner(prior, grid, downsample=4, terrain=terrain,
                        weights=weights).plan(
        (site.spawn_x, site.spawn_y),
        tuple(args.goal) if args.goal else site.goal)
    if not plan.found:
        print(f"  planning FAILED: {plan.reason}")
        return 1
    print(f"  plan       : {plan.length_m:.1f} m, {plan.expanded} nodes, "
          f"{time.perf_counter() - t0:.2f}s")

    truth = trajectory_from_path(plan.path, args.spacing)
    print(f"  trajectory : {len(truth)} poses at {args.spacing} m spacing")

    # --- simulate --------------------------------------------------------
    world = RaycastWorld.from_occupancy_yaml(derived_dir(site) / "prior_occ.yaml")
    spec = LidarSpec.from_site(site)
    lidar = Lidar2D(world, spec, rng=args.seed)
    odo = OdomCorruptor(seed=args.seed)

    odom = odo.corrupt_trajectory(truth, dt=args.spacing)
    scans = [lidar.scan(p, stamp=i * args.spacing) for i, p in enumerate(truth)]

    odom_final = float(np.hypot(*(odom[-1, :2] - truth[-1, :2])))
    print(f"  odometry   : final drift {odom_final:.2f} m "
          f"({100 * odom_final / plan.length_m:.1f}% of path)")

    # --- SLAM ------------------------------------------------------------
    # The prior is what the default "fused" reference matches against, and
    # Slam2D refuses to start without it. This harness had been left
    # constructing SLAM with no prior, so it raised on every run once
    # "fused" became the default reference.
    slam = Slam2D(grid, sensor_range=spec.range_max,
                  config=SlamConfig(max_correction_m=args.max_correction_m),
                  initial_pose=truth[0], prior_occ=world.occ)
    t0 = time.perf_counter()
    for scan, od, gt in zip(scans, odom, truth):
        slam.update(scan, gt if args.no_slam else od)
    dt = time.perf_counter() - t0

    s = slam.summary()
    print(f"  slam       : {dt:.1f}s for {len(scans)} scans "
          f"({1000 * dt / len(scans):.1f} ms/scan)")
    print(f"               {s['matched']} matched, {s['rejected']} rejected, "
          f"{s['keyframes']} keyframes, mean score {s['mean_score']:.3f}")
    print(f"               map coverage {100 * s['coverage']:.1f}% of the site grid")

    # --- score -----------------------------------------------------------
    a = slam.ate(truth)
    o = slam.odom_ate(truth)
    print()
    print("  trajectory error vs ground truth")
    print(f"    odometry : RMSE {o['rmse_m']:6.2f} m   final {o['final_m']:6.2f} m")
    print(f"    SLAM     : RMSE {a['rmse_m']:6.2f} m   final {a['final_m']:6.2f} m   "
          f"median {a['median_m']:.2f} m   heading RMSE {a['rmse_deg']:.2f} deg")
    if o["rmse_m"] > 0:
        print(f"    improvement factor: {o['rmse_m'] / max(a['rmse_m'], 1e-9):.1f}x")

    iou = slam.map.iou(world.occ)
    print(f"  map IoU vs truth (observed cells only): {iou:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
