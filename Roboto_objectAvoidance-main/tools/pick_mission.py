"""Find a drivable spawn and goal for a site.

Adding a site means choosing two coordinates, and the obvious approach --
reusing the numbers that worked somewhere else -- fails immediately:
(-120, -95) is open street in Pittsburgh and sits inside a building in San
Francisco. build_prior catches a lethal spawn, but only after everything
has been built, and it cannot suggest a replacement.

This searches the finished prior for a pair that is:

  * on a mapped road, so the robot starts and ends somewhere a vehicle
    could plausibly be;
  * clear of obstacles by a margin, not merely non-lethal, since the
    controller brakes at 0.6 m and the planner leaves a 1 m margin;
  * far apart, so the mission is a traverse rather than a hop;
  * actually connected, verified by running the real planner with the real
    terrain -- on a steep site the route can be blocked by grade alone,
    which no amount of looking at the costmap would reveal.

Usage:
    python -m tools.pick_mission --site tools/gis_pipeline/sites/foo.yaml
"""

from __future__ import annotations

import argparse

import numpy as np

from roboto_core.frames import GridSpec
from roboto_core.plan.astar import LETHAL, AStarPlanner, objectives
from roboto_core.plan.cost_fusion import inflate_lethal
from roboto_core.plan.terrain import TerrainGrid
from tools.gis_pipeline.common.crs import load_site
from tools.gis_pipeline.common.paths import derived_dir


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--site", default=None)
    ap.add_argument("--objective", default="balanced")
    ap.add_argument("--plan-margin-m", type=float, default=1.0)
    ap.add_argument("--max-cost", type=int, default=60,
                    help="highest graded cost accepted for an endpoint; "
                         "lower means more clearance")
    ap.add_argument("--edge-margin-m", type=float, default=10.0,
                    help="keep endpoints this far inside the extent; the "
                         "boundary is where the lidar sees least and where "
                         "half the neighbourhood is missing")
    ap.add_argument("--candidates", type=int, default=900,
                    help="road cells sampled before pairing")
    ap.add_argument("--tries", type=int, default=25,
                    help="farthest pairs to test with the real planner")
    args = ap.parse_args()

    site = load_site(args.site)
    grid = GridSpec.from_site(site)
    d = np.load(derived_dir(site) / "prior_cost.npz")
    cost, road = d["cost"], d["road_mask"]

    dem = derived_dir(site) / "dem_local.npz"
    terrain = TerrainGrid.from_npz(dem) if dem.exists() else None
    plan_cost = inflate_lethal(cost, grid, args.plan_margin_m)

    # Endpoints must survive the PLANNING costmap, not just the raw one, or
    # the planner refuses the very point we recommended.
    ok = (plan_cost < LETHAL) & (cost <= args.max_cost) & road.astype(bool)

    m = int(round(args.edge_margin_m / grid.resolution))
    if m > 0:
        edge = np.zeros_like(ok)
        edge[m:-m, m:-m] = True
        ok &= edge
    rows, cols = np.nonzero(ok)
    if len(rows) < 2:
        print(f"no road cell is both drivable and under cost {args.max_cost}; "
              f"raise --max-cost")
        return 1

    rng = np.random.default_rng(0)
    take = rng.choice(len(rows), size=min(args.candidates, len(rows)),
                      replace=False)
    r, c = rows[take], cols[take]
    x, y = grid.cell_to_world(r, c)
    pts = np.column_stack([np.asarray(x), np.asarray(y)])

    # Farthest pairs first.
    d2 = ((pts[:, None, :] - pts[None, :, :]) ** 2).sum(-1)
    order = np.dstack(np.unravel_index(np.argsort(-d2, axis=None), d2.shape))[0]

    obj = objectives(site)[args.objective]
    planner = AStarPlanner(plan_cost, grid, downsample=4, terrain=terrain,
                           objective=obj)

    print(f"site '{site.name}'  {len(rows):,} candidate road cells")
    seen = set()
    tried = 0
    for i, j in order:
        key = (min(i, j), max(i, j))
        if key in seen:
            continue
        seen.add(key)
        start, goal = tuple(pts[i]), tuple(pts[j])
        res = planner.plan(start, goal)
        tried += 1
        straight = float(np.hypot(*(pts[j] - pts[i])))
        if res.found:
            print(f"\n  spawn ({start[0]:.1f}, {start[1]:.1f})  "
                  f"goal ({goal[0]:.1f}, {goal[1]:.1f})")
            print(f"  route {res.length_m:.1f} m "
                  f"(straight line {straight:.1f} m), "
                  f"{res.expanded:,} nodes")
            print("\n  put this in the site yaml:")
            print(f"    robot_spawn:\n      x: {start[0]:.1f}\n"
                  f"      y: {start[1]:.1f}\n      yaw_deg: 0.0")
            print(f"    mission:\n      goal: [{goal[0]:.1f}, {goal[1]:.1f}]")
            return 0
        if tried >= args.tries:
            break
        print(f"  {straight:6.1f} m apart: {res.reason}")

    print(f"\nno connected pair in {tried} attempts. On a steep site the "
          f"route can be blocked by grade alone -- try raising "
          f"slope.max_traversable_deg, or --max-cost for tighter streets.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
