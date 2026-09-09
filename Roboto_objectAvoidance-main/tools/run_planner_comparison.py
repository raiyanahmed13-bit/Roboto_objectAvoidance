"""Compare path planners, on two independent axes.

    SEARCH ALGORITHM -- how the graph is explored
        Dijkstra      no heuristic; expands outward in all directions
        A*            admissible heuristic; optimal, far fewer expansions
        weighted A*   over-trusts the heuristic; fewer still, not optimal

    COST OBJECTIVE -- what "best" means
        distance      pure geometry, the classic shortest path
        balanced      distance + clearance + directional slope (the default)
        time          minimise duration, with speed falling off uphill
        effort        minimise climbing
        safe          maximise clearance from obstacles

The two are orthogonal: the algorithm decides how fast you find the route,
the objective decides which route is the right one. Reporting them
together is what shows that the shortest path is not the best path.

Every route is then scored on the SAME physical quantities -- length,
duration, climb, clearance -- because the planners' own costs are in
different units and cannot be compared directly.

Usage:
    python -m tools.run_planner_comparison
    python -m tools.run_planner_comparison --goal 95 110
"""

from __future__ import annotations

import argparse
import time

import numpy as np

from roboto_core.frames import GridSpec
from roboto_core.plan.astar import AStarPlanner, objectives
from roboto_core.plan.cost_fusion import inflate_lethal
from roboto_core.plan.evaluate import route_metrics
from roboto_core.plan.rrt import RRTConfig, RRTStarPlanner
from roboto_core.plan.terrain import TerrainGrid
from tools.gis_pipeline.common.crs import load_site
from tools.gis_pipeline.common.paths import derived_dir


def _plan(cost, grid, terrain, start, goal, objective, hw, downsample):
    t0 = time.perf_counter()
    planner = AStarPlanner(cost, grid, downsample=downsample, terrain=terrain,
                           objective=objective, heuristic_weight=hw)
    res = planner.plan(start, goal)
    return res, time.perf_counter() - t0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--site", default=None)
    ap.add_argument("--goal", nargs=2, type=float, default=None,
                    help="defaults to the site's mission.goal")
    ap.add_argument("--downsample", type=int, default=4)
    ap.add_argument("--plan-margin-m", type=float, default=1.0)
    args = ap.parse_args()

    site = load_site(args.site)
    grid = GridSpec.from_site(site)
    prior_cost = np.load(derived_dir(site) / "prior_cost.npz")["cost"]
    cost = inflate_lethal(prior_cost, grid, args.plan_margin_m)

    dem = derived_dir(site) / "dem_local.npz"
    terrain = TerrainGrid.from_npz(dem) if dem.exists() else None
    if terrain is None:
        print("no dem_local.npz -- run build_slope; slope columns omitted\n")

    start = (site.spawn_x, site.spawn_y)
    goal = tuple(args.goal) if args.goal else site.goal
    objs = objectives(site)
    v_max = float(site.raw["robot"]["max_speed_mps"])

    def score(res):
        return route_metrics(res.path, grid, prior_cost, terrain,
                             max_speed_mps=v_max)

    # --- axis 1: search algorithm, objective held fixed -------------------
    print(f"site '{site.name}'   {start} -> {goal}")
    print(f"search grid {grid.width // args.downsample} x "
          f"{grid.height // args.downsample} "
          f"@ {grid.resolution * args.downsample:.1f} m\n")

    print("SEARCH ALGORITHM  (objective held at 'balanced')")
    print(f"  {'algorithm':<16}{'expanded':>10}{'time':>9}"
          f"{'length':>10}{'cost':>12}")
    balanced = objs["balanced"]
    base_cost = None
    for name, hw in (("Dijkstra", 0.0), ("A*", 1.0),
                     ("weighted A* (1.5)", 1.5), ("weighted A* (3.0)", 3.0)):
        res, dt = _plan(cost, grid, terrain, start, goal, balanced, hw,
                        args.downsample)
        if not res.found:
            print(f"  {name:<16}{'FAILED: ' + res.reason:>40}")
            continue
        if base_cost is None:
            base_cost = res.cost
        excess = 100.0 * (res.cost / base_cost - 1.0)
        print(f"  {name:<16}{res.expanded:>10,}{dt:>8.2f}s"
              f"{res.length_m:>9.1f}m{res.cost:>10.1f}"
              f"  {excess:+5.1f}%")
    print("  (cost excess is relative to Dijkstra, which is optimal by"
          " construction)\n")

    # --- the sampling planner, compared fairly ---------------------------
    #
    # RRT* cannot take 'balanced': that objective depends on the signed
    # grade, which is asymmetric, and RRT*'s optimality argument assumes a
    # metric cost. So both sides of this comparison use 'distance', which
    # is symmetric and which both planners express exactly.
    print("SAMPLING vs GRID SEARCH  (objective held at 'distance')")
    print(f"  {'planner':<16}{'effort':>12}{'time':>9}{'length':>10}")
    dist_obj = objs["distance"]
    res, dt = _plan(cost, grid, terrain, start, goal, dist_obj, 1.0,
                    args.downsample)
    if res.found:
        print(f"  {'A*':<16}{res.expanded:>8,} exp{dt:>8.2f}s"
              f"{res.length_m:>9.1f}m")
    for budget in (3000, 12000):
        t0 = time.perf_counter()
        rr = RRTStarPlanner(cost, grid, RRTConfig(max_samples=budget),
                            objective=dist_obj).plan(start, goal)
        rdt = time.perf_counter() - t0
        if rr.found:
            excess = 100.0 * (rr.length_m / res.length_m - 1.0)
            print(f"  {'RRT* ' + str(budget):<16}{rr.expanded:>8,} smp"
                  f"{rdt:>8.2f}s{rr.length_m:>9.1f}m  {excess:+5.1f}%")
        else:
            print(f"  {'RRT* ' + str(budget):<16}{'no path: ' + rr.reason:>30}")
    print("  (RRT* samples the free space rather than enumerating cells, so"
          " its cost\n   does not scale with map size -- but it is only"
          " asymptotically optimal)\n")

    # --- axis 2: cost objective, algorithm held fixed ---------------------
    print("COST OBJECTIVE  (plain A*)")
    header = (f"  {'objective':<11}{'length':>9}{'time':>9}{'climb':>9}"
              f"{'descent':>9}{'max up':>8}{'mean cost':>11}{'expanded':>10}")
    print(header)
    rows = {}
    for name, obj in objs.items():
        res, dt = _plan(cost, grid, terrain, start, goal, obj, 1.0,
                        args.downsample)
        if not res.found:
            print(f"  {name:<11}FAILED: {res.reason}")
            continue
        m = score(res)
        rows[name] = m
        print(f"  {name:<11}{m['length_m']:>8.1f}m{m.get('time_s', 0):>8.0f}s"
              f"{m.get('climb_m', 0):>8.1f}m{m.get('descent_m', 0):>8.1f}m"
              f"{m.get('max_uphill_deg', 0):>7.1f}d"
              f"{m.get('mean_cost', 0):>11.1f}{res.expanded:>10,}")

    # --- the headline: nobody wins every column ---------------------------
    if rows:
        print()
        for key, label, better in (("length_m", "shortest route", min),
                                   ("time_s", "fastest route", min),
                                   ("climb_m", "least climbing", min),
                                   ("mean_cost", "most clearance", min)):
            have = {k: v[key] for k, v in rows.items() if key in v}
            if have:
                win = better(have, key=have.get)
                print(f"  {label:<16} {win:<10} ({have[win]:.1f})")
        print("\n  If one objective won every row the comparison would be"
              " pointless -- the\n  point is that it does not.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
