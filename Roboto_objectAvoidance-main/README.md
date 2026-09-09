# roboto — GIS-Prior SLAM & Terrain-Aware Path Planning

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

The write-up is `docs/roboto_progress_report.docx`, regenerated with
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
Gazebo. Reproduce them with the commands in [Running](#running-the-pipeline).

**SLAM against ground truth** — one 344 m mission over the prior world:

| | RMSE | final |
|---|---|---|
| Odometry | 3.50 m | 6.23 m |
| SLAM | 0.51 m | 0.25 m |

6.9× improvement; heading RMSE 1.00°, median position error 0.11 m, 1721 of
1722 scans matched with none rejected.

**Discrepancy detection** — four independent randomised scenarios, 12
injected obstacles in total:

| | |
|---|---|
| Recall | 12/12 (100%) |
| Precision | 12/12 (100%) |
| F1 | 1.000 |
| Localisation, detection to obstacle boundary | 0.01 m mean, 0.11 m max |
| SLAM ATE across the four runs | 0.34 – 0.50 m |

Localisation is reported as distance to the obstacle's boundary rather than
between centroids. A 2D lidar sees only the near *face* of an object, so a
perfectly correct detection sits on its boundary, roughly half its depth from
its centre; centroid error (0.79 m mean here) charges the detector for the
sensor's geometry rather than for being wrong.

**The ablation that argues for the contribution** — 8 obstacles placed across
the planned route over three independent scenarios. Both arms run the *same*
three scenario seeds, so this is a paired comparison rather than two separate
samples:

| | layer OFF | layer ON |
|---|---|---|
| Missions with a collision | 3/3 | **0/3** |
| Successful missions | 0/3 | **3/3** |
| Mean collision steps | 12.0 | 0.0 |
| Mean replans | 0.0 | 3.7 |
| Mean detour vs the planned route | −3.5 m | +24.7 m |

With the discrepancy layer off the robot drives straight through everything
OSM does not know about, and the negative detour is the tell: it is taking the
shortest route precisely because it never reacts to anything. With the layer
on it detects each obstacle, replans, and arrives — paying about 25 m of
detour over a 344 m route to do it.

**Live in Gazebo**, the same pipeline runs on real simulated lidar: the robot
plans over OSM, discovers the injected obstacles, replans around each, and
reaches the goal. `setup/wait_for_goal.sh` blocks until it does.

### Scenarios are randomised

Obstacle count, position, lateral offset, orientation and size are drawn per
scenario seed by `roboto_core.sim.scenario`, so repeated seeds are independent
trials rather than one layout measured several times. Scenario randomness is a
separate stream from sensor noise, so either can be held fixed to isolate the
other:

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
.venv-gis/bin/python -m pytest -q          # 148 tests, ~45 s, no ROS needed
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

1. **This is not a claim to detect real errors in OpenStreetMap.** The truth
   world is our own Gazebo/raycast world perturbed by known synthetic
   obstacles. It measures the detector, not OSM.
2. **The offline controller is a placeholder** — waypoint following with no
   vehicle dynamics. Offline collision counts reflect planning behaviour, not
   vehicle behaviour. The live node uses real pure pursuit.
3. **Discrepancy classification scans the whole 3000 × 3000 grid** per check.
   The dilated prior is cached, which makes a tight detection cycle
   affordable, but a local window around the robot is the proper fix.
4. **A detection is the visible face, not the footprint.** Reported areas are
   well under the true object area for exactly that reason.
5. **Only 1 of 236 buildings carries an OSM height tag**, so the rest use a
   6 m default. Harmless for a 2D lidar.
6. **Ground is flat in Gazebo**; the DEM drives planning cost only. A 2D lidar
   on 3D terrain pitches out of plane and corrupts 2D SLAM.
7. **No loop closure and no pose-graph back end.** The fused matching
   reference (matching against prior-or-mapped obstacles) greatly reduces the
   need for one, which is itself a defensible design position, but it is not
   the same thing.
