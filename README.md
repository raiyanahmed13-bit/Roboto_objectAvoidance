# roboto — Terrain-Aware Path Planning

A ground robot plans a route over a prior map built from real-world GIS data
(OpenStreetMap footprints + a USGS 3DEP elevation model), then drives it while
a from-scratch SLAM system builds its own occupancy map from lidar. Where the
two disagree — an obstacle the map doesn't have, or a mapped obstacle that
isn't there — the robot detects it, quantifies it, and replans.

The premise: *public GIS data is a powerful but stale prior, and SLAM is what
keeps it honest.*

**Status:** complete and running end to end, offline and live in Gazebo. The
GIS pipeline, the SLAM front end, the planner, discrepancy detection and
closed-loop replanning are all implemented and measured. Loop closure and a
pose-graph back end are not implemented; see [Limitations](#limitations).

The academic write-up is `docs/claudegenreport.md` / `.docx`, with its sourcing
recorded in `docs/claudegenreport_evidence.md`. The older progress report is
`docs/roboto_progress_report.docx`, regenerated with
`python -m tools.make_report_docx`.

---

## Site

Greenfield / Squirrel Hill South, Pittsburgh PA — a 300 m × 300 m patch
chosen for having both things this project needs:

| Property | Value |
|---|---|
| Buildings inside the extent | 134 (236 fetched, including the 50 m buffer) |
| Road polygons | 78 |
| Relief across the planning grid | 48.4 m (315.4 → 363.8 m) |
| Median slope | 8.4° |
| Above the 25° traversability limit | 9.9% of cells |
| Drivable after inflation | 74.3% |
| Planning grid | 3000 × 3000 at 0.10 m |

A plateau in the northwest drops across a steep escarpment to a valley in the
southeast, and that escarpment bisects the map — so slope-aware planning
changes routes rather than decorating them.

---

## Results

All figures below are from the offline harnesses, which run without ROS or
Gazebo. They were measured on 2026-09-09 against commit `85bd161`; reproduce
them with the commands in [Running](#running-the-pipeline).

**SLAM against ground truth** — three seeds per site, each a full mission over
the prior world:

| Site | Odometry RMSE | SLAM RMSE | Ratio | Scans rejected |
|---|---|---|---|---|
| Pittsburgh | 4.81 m | **0.22 m** | 21.9× | 16–18% |
| Chicago | 7.50 m | 2.31 m | 3.2× | 67–81% |
| San Francisco | 6.72 m | 2.27 m | 3.0× | 44–75% |

The odometry baseline varies between seeds because the wheel calibration errors
are drawn per run, so the ratio compares means. On Pittsburgh seed 0: heading
RMSE 0.32°, median position error 0.15 m, map IoU 0.481.

**Chicago and San Francisco are the honest part of this table.** The same
estimator, unchanged, is an order of magnitude worse on a regular street grid,
and the rejection rates say why: on a grid of long parallel walls a scan cannot
distinguish the true pose from one shifted along a wall, so both score 1.000, and
the matcher correctly declines to trust either. One San Francisco seed came out
worse than dead reckoning outright (4.58 m against 4.28 m). This is what a
scan-matching front end with no loop closure looks like on a repeating scene —
see [Limitations](#limitations).

**Discrepancy detection** — four independent randomised scenarios, six obstacles
each, 24 in total:

| | |
|---|---|
| Recall | 24/24 (100%) |
| Precision | 24/24 (100%) |
| F1 | 1.000 |
| Localisation, detection to obstacle boundary | 0.01 m mean, 0.09 m max |
| SLAM ATE across the four runs | 0.15 – 0.54 m |

Localisation is reported as distance to the obstacle's boundary rather than
between centroids. A 2D lidar sees only the near *face* of an object, so a
perfectly correct detection sits on its boundary, roughly half its depth from
its centre; centroid error (0.79 m mean here) charges the detector for the
sensor's geometry rather than for being wrong.

**The ablation that argues for the contribution** — 18 obstacles placed across
the planned route over three independent scenarios. Both arms run the *same*
three scenario seeds, so this is a paired comparison rather than two separate
samples:

| | layer OFF | layer ON |
|---|---|---|
| Missions with a collision | 3/3 | **0/3** |
| Successful missions | 0/3 | **3/3** |
| Mean collision steps | 28.0 | 0.0 |
| Mean replans | 0.0 | 10.0 |
| Mean detour vs the planned route | −3.5 m | +52.3 m |
| Mean nodes expanded | 0 | 780,237 |

With the discrepancy layer off the robot drives straight through everything OSM
does not know about — it travels an identical 340.8 m in all three scenarios,
because it never reacts to anything, and the negative detour is the tell. With
the layer on it detects each obstacle, replans, and arrives, paying 52.3 m of
detour on a 344.4 m route to do it.

**Global vs local replanning** — same three scenarios, both arms 3/3 with no
collisions:

| | global | detour |
|---|---|---|
| Mean replans | 10.0 | 14.0 |
| Mean nodes expanded | 780,237 | **69,107** |
| Mean extra distance | +52.3 m | +80.9 m |
| Local / global replans | 0 / 30 | 41 / 1 |

Eleven times less search for 28.6 m more route. The detour fallback fired once in
42 reactions, so it is not redundant.

**Planners and objectives** (Pittsburgh, `balanced`): Dijkstra expands 192,299
nodes and A* 132,033 for the *same* 344.4 m route; weighted A* at 1.5 costs 1.6%
more route for 39% less search, and at 3.0 costs 56% more for 72% less. Each cost
objective wins the column it minimises and none wins them all — `distance` is
shortest at 337.3 m, `time` fastest at 378 s, `effort` lowest-climbing at 13.6 m,
`safe` highest-clearance at mean cost 1.3 for +31% distance.

**Live in Gazebo**, the same pipeline runs on real simulated lidar: the robot
plans over OSM, discovers the injected obstacles, replans around each, and
reaches the goal. `setup/wait_for_goal.sh` blocks until it does. No live run from
the current build has been recorded, so no live accuracy figure is quoted here.

### Scenarios are randomised

Obstacle position, lateral offset, orientation and size are drawn per scenario
seed by `roboto_core.sim.scenario` (six obstacles per scenario by default), so
repeated seeds are independent trials rather than one layout measured several
times. Scenario randomness is a separate stream from sensor noise, so either can
be held fixed to isolate the other:

```bash
python -m tools.run_mission_offline --seeds 4                     # 4 scenarios
python -m tools.run_mission_offline --seeds 4 --scenario-seed 0   # 1 scenario, 4 noise draws
```

The same sampler places the obstacles in the Gazebo world, so the live demo
and the offline experiments draw from one implementation instead of two
hardcoded lists that can drift apart.

---

## The frame contract

Everything hinges on one invariant:

> **local TM frame ≡ ROS `map` frame ≡ Gazebo world frame**
> x = East, y = North, z = Up, origin at `site.yaml`'s lat/lon.

`tools/gis_pipeline/site.yaml` is the single source of truth for every
coordinate in the project. A coordinate literal anywhere else is a bug.

Three decisions that exist to prevent silent misalignment — the failure mode
where the map looks perfectly plausible and every metric is garbage:

1. **Site-local Transverse Mercator with `k=1`, not UTM.** Zero scale
   distortion at the origin, and coordinates stay near zero. UTM eastings
   near 500,000 m lose ~6 cm to float32 rounding, which corrupts a 0.10 m grid.

2. **Row order is flipped in exactly one place** (`roboto_core/frames.py`,
   re-exported by `common/grid.py`). GDAL and PGM use row 0 = north; ROS
   `OccupancyGrid` uses row 0 = south. Doing that conversion zero times or
   twice produces a mirrored map that still looks like a neighbourhood.

3. **Vector data is GeoPackage, never GeoJSON.** RFC 7946 defines GeoJSON as
   WGS84-only, so GeoPandas silently drops a metric CRS on write and reads it
   back labelled EPSG:4326 — values correct, units wrong, every subsequent
   `.buffer()` and `.area()` quietly meaningless. `load_vector()` refuses a
   geographic CRS outright.

---

## Running the pipeline

Everything runs in WSL (Ubuntu 22.04). Windows Application Control blocks
rasterio's and pyogrio's GDAL DLLs, so the GIS stack cannot import on the
Windows side — edit and commit on Windows, run in WSL.

```bash
python -m venv .venv-gis
.venv-gis/bin/pip install -r requirements-gis.txt

python -m tools.gis_pipeline.fetch_osm                  # OSM -> buildings/roads .gpkg
python -m tools.gis_pipeline.fetch_dem                  # USGS 3DEP -> dem_src.tif
python -m tools.gis_pipeline.build_slope --preview      # reproject + gradients
python -m tools.gis_pipeline.build_prior --preview      # -> prior costmap
python -m tools.gis_pipeline.build_world                # -> Gazebo SDF world
python -m tools.gis_pipeline.inject_gazebo_obstacles    # -> site_demo.sdf
```

Each stage caches to disk; pass `--force` to refetch. Figures land in
`experiments/figures/`. The pipeline only needs rerunning if `site.yaml`
changes — its outputs are committed.

### Experiments

```bash
python -m tools.run_slam_offline                        # SLAM vs odometry
python -m tools.run_discrepancy_offline --seeds 4       # recall / precision
python -m tools.run_mission_offline --seeds 3 --detect-every 3 --confirm-after 1
python -m tools.run_mission_offline --seeds 3 --no-replan        # the ablation
python -m tools.debug_mission                           # instrumented single run
```

### Live demo

```bash
cd ros2_ws && colcon build --packages-select roboto_core roboto_ros
source install/setup.bash
ros2 launch roboto_ros demo.launch.py                            # Gazebo + RViz
ros2 launch roboto_ros demo.launch.py gui:=false rviz:=false     # headless
bash setup/wait_for_goal.sh /tmp/demo.log 10
```

Gazebo needs software rendering here (`LIBGL_ALWAYS_SOFTWARE=1`,
`GALLIUM_DRIVER=llvmpipe`) — `setup/gazebo_env.sh` carries the measurements,
and `setup/gate_a_check.sh` re-verifies that the simulator can actually
produce lidar after any driver change. Hardware ogre1 runs and publishes but
returns `range_min` on every beam, which looks like success and is not.

### Artifacts

```
data/site_<name>/
  raw/      buildings.gpkg  roads.gpkg  dem_src.tif        (as fetched)
  derived/  dem_local.tif                                   (site-local CRS, 1 m)
            prior_occ.pgm + .yaml                           (ROS map_server)
            prior_cost.npz                                  (uint8 costmap, 1.0 MB)
  world/    site.sdf  site_demo.sdf                         (Gazebo)
```

`prior_cost.npz` uses ROS `costmap_2d` encoding: 0–253 graded, 254 lethal.
Slope is deliberately **not** stored — it is a pure function of the committed
DEM, so persisting it would duplicate 36 MB and create a second source of
truth that can drift. `TerrainModel.sample()` interpolates it on demand.

Directional slope cost is likewise not baked into the costmap. Whether a grade
is expensive depends on which way you cross it — climbing is costly,
descending nearly free, side-slope is a rollover risk — so it is an *edge*
cost applied by the planner, which makes the planning graph directed. Only the
direction-independent part (impassably steep) is resolved into the static
lethal mask.

---

## Tests

```bash
.venv-gis/bin/python -m pytest -q          # 246 tests, ~67 s, no ROS needed
```

`tests/test_alignment.py` proves the coordinate machinery on synthetic
fixtures — including mirror and transpose checks a square test region cannot
catch. `tests/test_prior.py` proves the *actual fetched artifacts* are aligned,
which is a separate failure mode: the code can be correct while the committed
data is stale or built from a different `site.yaml`.

**A failure in either file is stop-the-line.** Do not work around it by adding
a flip somewhere else — every downstream metric depends on these holding.

---

## Layout

```
tools/gis_pipeline/     offline GIS pipeline (.venv-gis; heavy geo deps)
tools/                  experiment harnesses and the report generator
ros2_ws/src/
  roboto_core/          pure Python: slam, plan, discrepancy, sim. NO rclpy.
  roboto_ros/           the mission node, launch, RViz config, robot model
  roboto_msgs/          DiscrepancyObject, DiscrepancyReport
setup/                  environment install, rendering config, launch helpers
tests/                  pytest; no ROS, no Gazebo
experiments/            scenarios, bags, results, figures
```

Three deliberate separations:

- **`roboto_core` imports no ROS.** Most of the project is therefore testable
  in seconds without Gazebo or a ROS graph running, which is what makes the
  schedule feasible.
- **The GIS pipeline has its own venv.** Forcing GDAL/rasterio/geopandas into
  a ROS Python environment is a multi-hour dependency fight for no benefit;
  the two sides communicate only through committed data artifacts.
- **The live system is one ROS node, not four.** The shared map is 9 MB as an
  `OccupancyGrid`; passing it between nodes at any useful rate would dominate
  the system. `roboto_node.py` is message plumbing — everything substantive
  lives in `roboto_core` and is tested without ROS.

---

## Limitations

State these plainly; the ground truth here is a simulated world we generated.

1. **SLAM is an order of magnitude worse on a regular street grid.** Pittsburgh
   averages 0.22 m; Chicago 2.31 m and San Francisco 2.27 m, with one SF seed
   worse than dead reckoning. A repeating grid is ambiguous to a scan matcher
   with no loop closure — it rejects 67–81% of scans on Chicago against 16–18%
   on Pittsburgh. The correction limit stops the divergence; it does not fix the
   ambiguity. Treat Chicago as a demonstrated limitation, not a solved site.

2. **This is not a claim to detect real errors in OpenStreetMap.** The truth
   world is our own Gazebo/raycast world perturbed by known synthetic
   obstacles. It measures the detector, not OSM.
3. **The offline controller is a placeholder** — waypoint following with no
   vehicle dynamics. Offline collision counts reflect planning behaviour, not
   vehicle behaviour. The live node uses real pure pursuit.
4. **Discrepancy classification scans the whole 3000 × 3000 grid** per check.
   The dilated prior is cached, which makes a tight detection cycle
   affordable, but a local window around the robot is the proper fix.
5. **A detection is the visible face, not the footprint.** Reported areas are
   well under the true object area for exactly that reason.
6. **Only 1 of 236 buildings carries an OSM height tag**, so the rest use a
   6 m default. Harmless for a 2D lidar.
7. **Ground is flat in Gazebo**; the DEM drives planning cost only. A 2D lidar
   on 3D terrain pitches out of plane and corrupts 2D SLAM.
8. **No loop closure and no pose-graph back end.** The fused matching reference
   (matching against prior-or-mapped obstacles) substitutes for one on
   Pittsburgh, where the prior supplies the globally consistent geometry a
   single-pass self-map cannot. It is not a substitute in general: limitation 1
   is what its absence costs on a repeating street grid, and adding the back end
   is now the highest-value next step rather than a stretch goal.
