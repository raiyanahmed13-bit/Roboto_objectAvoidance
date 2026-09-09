"""Place obstacles in the Gazebo world that the GIS prior does not contain.

This is what makes the live demo demonstrate the actual contribution
rather than just "a robot follows a path". The obstacles go ON the planned
route, so a robot that ignores them collides and a robot that detects them
has to route around.

Crucially the obstacles are added to the WORLD ONLY. The prior costmap and
prior occupancy grid are untouched, so from the robot's point of view they
simply do not exist until its lidar finds them -- which is exactly the
situation a stale OpenStreetMap extract puts a real robot in.

Obstacles are drawn by roboto_core.sim.scenario, the same sampler the
offline experiments use, rather than from a second hardcoded list that
could drift from the offline one unnoticed. A scenario seed is
reproducible: rerunning with `--scenario-seed N` rebuilds exactly the same
world.

Obstacles are placed as fractions ALONG THE PLANNED ROUTE, so this file
has to plan exactly as roboto_node does -- same inflated costmap, same
terrain gradients, same weights. If the two disagree the obstacles end up
beside the road the robot actually drives and the demo quietly stops
demonstrating anything. That was a real trap while the live node had no
terrain model and this one had to imitate it by weighting slope at zero.

Writes data/site_<name>/world/site_demo.sdf, leaving site.sdf clean.

Usage:
    python -m tools.gis_pipeline.inject_gazebo_obstacles
    python -m tools.gis_pipeline.inject_gazebo_obstacles --scenario-seed 3
"""

from __future__ import annotations

import argparse

import numpy as np

from roboto_core.frames import GridSpec
from roboto_core.plan.astar import AStarPlanner, objectives
from roboto_core.plan.cost_fusion import inflate_lethal
from roboto_core.plan.terrain import TerrainGrid
from roboto_core.sim.raycast_sim import RaycastWorld
from roboto_core.sim.scenario import (CATALOGUE, catalogue_from,
                                      describe,
                                      sample_placements, scenario_rng)

from .common.crs import load_site
from .common.paths import derived_dir, world_dir

# Rendering only -- the geometry comes from the shared sampler. Anything
# not listed falls back to grey.
COLOURS = {
    "construction_barrier": (0.90, 0.45, 0.05),
    "fence_panel_run": (0.90, 0.60, 0.10),
    "parked_van": (0.75, 0.15, 0.15),
    "delivery_truck": (0.20, 0.30, 0.65),
    "dumpster": (0.15, 0.45, 0.25),
    "skip_container": (0.55, 0.40, 0.15),
    "fallen_tree": (0.35, 0.25, 0.12),
}
DEFAULT_COLOUR = (0.5, 0.5, 0.5)


def _model(name, cx, cy, yaw, sx, sy, sz, rgb) -> str:
    r, g, b = rgb
    return f"""
    <model name="{name}">
      <static>true</static>
      <pose>{cx:.3f} {cy:.3f} {sz / 2:.3f} 0 0 {yaw:.6f}</pose>
      <link name="link">
        <collision name="collision">
          <geometry><box><size>{sx:.3f} {sy:.3f} {sz:.3f}</size></box></geometry>
        </collision>
        <visual name="visual">
          <geometry><box><size>{sx:.3f} {sy:.3f} {sz:.3f}</size></box></geometry>
          <material>
            <ambient>{r:.2f} {g:.2f} {b:.2f} 1</ambient>
            <diffuse>{min(r + 0.1, 1):.2f} {min(g + 0.1, 1):.2f} {min(b + 0.1, 1):.2f} 1</diffuse>
          </material>
        </visual>
      </link>
    </model>"""


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--site", default=None)
    ap.add_argument("--goal", nargs=2, type=float, default=None,
                    help="defaults to the site's mission.goal")
    ap.add_argument("--height", type=float, default=1.4)
    ap.add_argument("--scenario-seed", type=int, default=0,
                    help="obstacle layout seed; the same value rebuilds the "
                         "same world")
    # Restrict which obstacle types may be drawn.
    #
    # A 2D lidar's evidence for an object is its VISIBLE WIDTH times one
    # cell of depth, not its footprint, so the smallest types sit closest
    # to the detection floor. Measured worst-case near face against the
    # 0.2 m2 floor, at each type's minimum size and worst yaw:
    #
    #   delivery_truck 2.82x   fallen_tree 2.35x   parked_van   1.97x
    #   fence_panel_run 1.64x  skip_container 1.50x
    #   construction_barrier 1.41x   dumpster 1.41x
    #
    # All of them pass. For a demo where a missed detection wastes a
    # twenty-minute take, restricting to the top three buys a much larger
    # margin. Leave it unset for experiments: narrowing the catalogue makes
    # the scenario easier, and results should not quietly depend on that.
    ap.add_argument("--types", default=None,
                    help="comma-separated obstacle types, e.g. "
                         "delivery_truck,fallen_tree,parked_van. "
                         "Default: the whole catalogue")
    ap.add_argument("--min-obstacles", type=int, default=6)
    ap.add_argument("--max-obstacles", type=int, default=6)
    ap.add_argument("--plan-margin-m", type=float, default=1.0,
                    help="must match the node's plan_margin_m")
    ap.add_argument("--objective", default="balanced",
                    help="MUST match the objective the demo runs with; "
                         "obstacles are placed along the route it produces")
    ap.add_argument("--heuristic-weight", type=float, default=1.0)
    args = ap.parse_args()

    site = load_site(args.site)
    grid = GridSpec.from_site(site)
    prior_cost = np.load(derived_dir(site) / "prior_cost.npz")["cost"]
    prior_occ = RaycastWorld.from_occupancy_yaml(
        derived_dir(site) / "prior_occ.yaml").occ

    # The route the robot will believe in, planned exactly as it will plan
    # it. Every input here has to match roboto_node: the same inflated
    # costmap, the same terrain gradients from the same npz, and the same
    # weights. Obstacles are placed as fractions ALONG THIS ROUTE, so if
    # the two planners disagree the obstacles land beside the road the
    # robot actually drives, and the demo silently stops demonstrating
    # anything.
    dem_npz = derived_dir(site) / "dem_local.npz"
    terrain = TerrainGrid.from_npz(dem_npz) if dem_npz.exists() else None
    if terrain is None:
        print("  warning: no dem_local.npz, planning without slope. "
              "The live node loads it, so the routes may differ. "
              "Run build_slope first.")
    plan_cost = inflate_lethal(prior_cost, grid, args.plan_margin_m)
    objs = objectives(site)
    if args.objective not in objs:
        print(f"unknown objective {args.objective!r}; "
              f"choose from {sorted(objs)}")
        return 1
    plan = AStarPlanner(plan_cost, grid, downsample=4, terrain=terrain,
                        objective=objs[args.objective],
                        heuristic_weight=args.heuristic_weight).plan(
        (site.spawn_x, site.spawn_y),
        tuple(args.goal) if args.goal else site.goal)
    if not plan.found:
        print(f"planning failed: {plan.reason}")
        return 1
    path = plan.path
    print(f"route {plan.length_m:.1f} m over {len(path)} waypoints "
          f"(objective={args.objective})")

    # A* waypoints carry no heading; the sampler differences them.
    # One implementation, shared with both offline harnesses, so a typo is
    # rejected identically everywhere.
    catalogue = catalogue_from(args.types)
    if catalogue != CATALOGUE:
        print(f"restricted to: {', '.join(sorted(t.label for t in catalogue))}")

    places = sample_placements(
        scenario_rng(args.scenario_seed), path, prior_occ, grid,
        mode="blocking", n_range=(args.min_obstacles, args.max_obstacles),
        catalogue=catalogue)
    if not places:
        print("no obstacle placement survived the prior/detour checks")
        return 1

    print(f"scenario seed {args.scenario_seed}: {len(places)} obstacle(s)")
    print(describe(places))

    models = []
    for k, p in enumerate(places):
        length, width = p.size
        models.append(_model(f"{p.label}_{k}", p.cx, p.cy, p.yaw,
                             length, width, args.height,
                             COLOURS.get(p.label, DEFAULT_COLOUR)))

    src = world_dir(site) / "site.sdf"
    if not src.exists():
        print(f"{src} missing -- run build_world first")
        return 1
    world = src.read_text(encoding="utf-8")

    marker = "\n  </world>"
    if marker not in world:
        print("could not find the world closing tag")
        return 1
    out_text = world.replace(
        marker,
        "\n    <!-- Obstacles absent from OSM. Added to the WORLD ONLY: the\n"
        "         prior costmap and occupancy grid are untouched, so the\n"
        "         robot cannot know about these until it sees them. -->"
        + "".join(models) + marker)

    out = world_dir(site) / "site_demo.sdf"
    out.write_text(out_text, encoding="utf-8")
    print(f"\nwrote {out}")
    # The demo launch already prefers site_demo.sdf, so there is nothing
    # to pass. There is no `world:=` argument -- world selection happens
    # before launch arguments resolve, so it is an environment variable.
    print(f"  the demo launch uses this world by default; to force it:")
    print(f"    ROBOTO_WORLD=site_demo.sdf ros2 launch roboto_ros demo.launch.py"
          f" objective:={args.objective}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
