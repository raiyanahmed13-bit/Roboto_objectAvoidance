# Running the demo in Gazebo and RViz

Everything runs inside WSL. Commands in `bash` blocks are typed **inside
Ubuntu**; the few `powershell` ones are typed on the Windows side.

> **In a hurry?** Jump to [§0 Copy-paste recipes](#0-copy-paste-recipes).
> Every one is a complete sequence from an empty terminal — nothing to
> assemble, nothing to look up.

**Contents**

0. [Copy-paste recipes](#0-copy-paste-recipes)
1. [Where to type things](#1-where-to-type-things)
2. [Start the demo](#2-start-the-demo)
3. [Every option at a glance](#3-every-option-at-a-glance)
4. [Which window to open](#4-which-window-to-open)
5. [Planners — how the route is found](#5-planners--how-the-route-is-found)
6. [Cost objectives — what "best" means](#6-cost-objectives--what-best-means)
7. [Terrain — the elevation model](#7-terrain--the-elevation-model)
8. [Reacting to a blockage](#8-reacting-to-a-blockage)
9. [Maps (sites)](#9-maps-sites)
10. [Obstacles](#10-obstacles)
11. [Following a run](#11-following-a-run)
12. [Stopping](#12-stopping)
13. [Offline experiments](#13-offline-experiments)
14. [Troubleshooting](#14-troubleshooting)

---

## 0. Copy-paste recipes

Each recipe is the **whole sequence**, starting from an empty PowerShell
window. Paste the block, top to bottom. Nothing is left out and nothing
refers you elsewhere.

The first three lines are always the same — they get you into Linux and
load ROS:

```bash
wsl -d Ubuntu-22.04
cd ~/roboto/ros2_ws
source /opt/ros/humble/setup.bash
source install/setup.bash
```

Call that **the preamble**. Every recipe below starts with it.

---

### A. Just run it (Pittsburgh, default everything)

```bash
wsl -d Ubuntu-22.04
cd ~/roboto/ros2_ws
source /opt/ros/humble/setup.bash
source install/setup.bash
ros2 launch roboto_ros demo.launch.py
```

---

### B. Run a different map

All three maps are already built, so this is **one extra word** in front of
the launch. Nothing else changes.

**San Francisco (steep hills):**

```bash
wsl -d Ubuntu-22.04
cd ~/roboto/ros2_ws
source /opt/ros/humble/setup.bash
source install/setup.bash
ROBOTO_SITE=sanfrancisco_russian_hill ros2 launch roboto_ros demo.launch.py
```

**Chicago (flat grid):**

```bash
wsl -d Ubuntu-22.04
cd ~/roboto/ros2_ws
source /opt/ros/humble/setup.bash
source install/setup.bash
ROBOTO_SITE=chicago_west_loop ros2 launch roboto_ros demo.launch.py
```

**Pittsburgh (the default, rolling hills):**

```bash
wsl -d Ubuntu-22.04
cd ~/roboto/ros2_ws
source /opt/ros/humble/setup.bash
source install/setup.bash
ROBOTO_SITE=pittsburgh_greenfield ros2 launch roboto_ros demo.launch.py
```

The three valid names, exactly as typed:

```
pittsburgh_greenfield
sanfrancisco_russian_hill
chicago_west_loop
```

`ROBOTO_SITE=...` goes **before** `ros2`, on the same line, with no
`export`. It is not `site:=` — see §3 for why.

---

### C. Run a different search algorithm

Nothing to rebuild. Just change the number.

```bash
wsl -d Ubuntu-22.04
cd ~/roboto/ros2_ws
source /opt/ros/humble/setup.bash
source install/setup.bash
ros2 launch roboto_ros demo.launch.py heuristic_weight:=0
```

| Instead of `0`, use | You get |
|---|---|
| `0` | Dijkstra |
| `1` | A* (the default) |
| `1.5` or `3` | weighted A* |

For the sampling planner instead:

```bash
wsl -d Ubuntu-22.04
cd ~/roboto/ros2_ws
source /opt/ros/humble/setup.bash
source install/setup.bash
ros2 launch roboto_ros demo.launch.py planner:=rrt objective:=distance
```

`planner:=rrt` **must** be paired with `objective:=distance` or
`objective:=safe`. The others will refuse to start and tell you so.

---

### D. Run a different cost objective

This one needs **two commands**, because the obstacles have to be moved
onto the new route. Skipping the first command is the single most common
mistake — the robot drives past obstacles sitting in open ground and the
demo proves nothing.

```bash
wsl -d Ubuntu-22.04
cd ~/roboto
.venv-gis/bin/python -m tools.gis_pipeline.inject_gazebo_obstacles --objective safe
cd ros2_ws
source /opt/ros/humble/setup.bash
source install/setup.bash
ros2 launch roboto_ros demo.launch.py objective:=safe
```

Replace **both** occurrences of `safe` with whichever you want:
`distance`, `balanced`, `time`, `effort`, `safe`. They must match.

---

### E. Different map *and* different objective

Same idea, but tell the obstacle step which map too:

```bash
wsl -d Ubuntu-22.04
cd ~/roboto
.venv-gis/bin/python -m tools.gis_pipeline.inject_gazebo_obstacles \
    --site tools/gis_pipeline/sites/chicago_west_loop.yaml \
    --objective safe
cd ros2_ws
source /opt/ros/humble/setup.bash
source install/setup.bash
ROBOTO_SITE=chicago_west_loop ros2 launch roboto_ros demo.launch.py objective:=safe
```

For Pittsburgh the `--site` line is omitted entirely — it is the default.

---

### F. Make the robot dodge around obstacles instead of re-routing

```bash
wsl -d Ubuntu-22.04
cd ~/roboto/ros2_ws
source /opt/ros/humble/setup.bash
source install/setup.bash
ros2 launch roboto_ros demo.launch.py replan_mode:=detour
```

Nothing to rebuild.

---

### G. Different obstacles on the same map

```bash
wsl -d Ubuntu-22.04
cd ~/roboto
.venv-gis/bin/python -m tools.gis_pipeline.inject_gazebo_obstacles --scenario-seed 7
cd ros2_ws
source /opt/ros/humble/setup.bash
source install/setup.bash
ros2 launch roboto_ros demo.launch.py
```

Any number after `--scenario-seed` gives a different layout. The same
number always gives the same layout.

---

### H. Stop everything

In the launch terminal press **Ctrl+C**, then:

```bash
pkill -f "ign[ ]gazebo"; pkill -f "lib/roboto[_]ros"
pkill -f parameter_bridge; pkill -f rviz2
```

The simulation does **not** exit on its own when the robot reaches the
goal, and it keeps a CPU core busy until you do this.

---

### Which recipes need a rebuild?

| Changing | Rebuild obstacles first? |
|---|---|
| map (§B) | no — all three are built |
| `heuristic_weight` `0` or `1` (§C) | no |
| `heuristic_weight` above 1 | **yes** |
| `planner:=rrt` (§C) | **yes** |
| `objective` (§D) | **yes** |
| `replan_mode` (§F) | no |
| `gui` / `rviz` | no |

"Rebuild obstacles" always means the one
`inject_gazebo_obstacles` command, with the same options you are about to
launch with.

---

## 1. Where to type things

Two shells are involved and mixing them up is the most common way to get
stuck.

| Prompt looks like | You are in | Use it for |
|---|---|---|
| `PS C:\...>` | Windows PowerShell | starting WSL, `wsl --shutdown` |
| `hem@HEM:~$` | Ubuntu, inside WSL | everything else |

From PowerShell (or the VS Code terminal, **Ctrl+`**):

```powershell
wsl -d Ubuntu-22.04
```

The prompt changes to `hem@HEM:...$`. `exit` goes back.

**Two clones of the repo exist and they are not interchangeable:**

| Path | Role |
|---|---|
| `E:\roboto` | Windows working copy — **edit and commit here** |
| `~/roboto` | inside WSL — **everything runs here** |

Edit on Windows, commit, then `git pull` inside WSL. The ROS build only
exists on the Linux side, and the GIS libraries cannot load on Windows at
all.

---

## 2. Start the demo

```bash
cd ~/roboto/ros2_ws
source /opt/ros/humble/setup.bash
source install/setup.bash
ros2 launch roboto_ros demo.launch.py
```

Both windows open on your normal Windows desktop. Gazebo takes 1–3 minutes
the first time: it rasterises the building meshes on the CPU.

> **Do not `source setup/gazebo_env.sh` before launching.** It exports
> `IGN_HEADLESS_RENDERING=1`, which suppresses the Gazebo window. The
> launch file already sets the software-rendering variables it needs, and
> pins that one to match `gui:=`. The script is for headless runs.

### Rebuild first if you changed any code

```bash
cd ~/roboto && git pull
cd ros2_ws
source /opt/ros/humble/setup.bash
colcon build --packages-select roboto_core roboto_ros
source install/setup.bash
```

---

## 3. Every option at a glance

`ros2 launch roboto_ros demo.launch.py --show-args` lists these.

| Argument | Default | Values | Meaning |
|---|---|---|---|
| `gui` | `true` | `true` / `false` | the Gazebo window |
| `rviz` | `true` | `true` / `false` | the RViz window |
| `planner` | `astar` | `astar` / `rrt` | grid search or sampling |
| `heuristic_weight` | `1.0` | `0`, `1`, `>1` | Dijkstra / A* / weighted A* |
| `rrt_samples` | `6000` | integer | RRT* sample budget |
| `objective` | `balanced` | `distance` `balanced` `time` `effort` `safe` | what to minimise |
| `replan_mode` | `global` | `global` / `detour` | how it reacts to a blockage |
| `goal_x` | *from site* | metres | goal easting |
| `goal_y` | *from site* | metres | goal northing |

Two things are environment variables rather than arguments, because the
world path, model path and bridge topic names are all needed *before*
launch arguments resolve:

| Variable | Default | Meaning |
|---|---|---|
| `ROBOTO_SITE` | `pittsburgh_greenfield` | which site to load |
| `ROBOTO_WORLD` | `site_demo.sdf` | which world file |

There is **no** `world:=` argument. Use `ROBOTO_WORLD=...` instead.

A fully specified run:

```bash
ROBOTO_SITE=sanfrancisco_russian_hill \
ros2 launch roboto_ros demo.launch.py \
    planner:=astar heuristic_weight:=1 objective:=safe replan_mode:=detour
```

---

## 4. Which window to open

Everything renders in software (`llvmpipe`), because hardware GL crashes
OGRE2 under WSL2 — and worse, the OGRE1 fallback runs happily while
returning `range_min` on every lidar beam. So the GUI competes with the
simulation for CPU, and running both windows slows the mission noticeably.

```bash
ros2 launch roboto_ros demo.launch.py gui:=false             # RViz only
ros2 launch roboto_ros demo.launch.py rviz:=false            # Gazebo only
ros2 launch roboto_ros demo.launch.py gui:=false rviz:=false # headless
```

### Using RViz

Preloaded with a top-down view and these layers:

| Layer | What it shows |
|---|---|
| **GIS prior** | what OpenStreetMap thinks is there |
| **SLAM map** | what the lidar actually finds |
| **Discrepancies** | red cylinders on obstacles OSM lacks |
| **Plan** | the current route, redrawn on each replan |
| **Lidar** / **SLAM pose** / **TF** | live scan, estimate, frames |

The moment worth watching: a red cylinder appears and the plan line
immediately snaps to a new route around it.

### Using Gazebo

The camera starts at the world origin looking at empty ground, while the
robot spawns far away. **Right-click `roboto_bot` in the Entity Tree →
Follow** — the camera locks onto the robot and tracks it.

| Action | Result |
|---|---|
| left-drag | orbit |
| middle-drag | pan |
| scroll | zoom |

Right-click any obstacle in the Entity Tree → **Move To** to jump the
camera to it. Bottom-left has pause/step and the real-time factor.

---

## 5. Planners — how the route is found

Two families, and they are genuinely different animals.

### Grid search — `planner:=astar`

Explores a discretised graph. `heuristic_weight` selects the algorithm:

```bash
ros2 launch roboto_ros demo.launch.py heuristic_weight:=0     # Dijkstra
ros2 launch roboto_ros demo.launch.py heuristic_weight:=1     # A* (default)
ros2 launch roboto_ros demo.launch.py heuristic_weight:=1.5   # weighted A*
ros2 launch roboto_ros demo.launch.py heuristic_weight:=3     # weighted A*
```

Measured on Pittsburgh, `balanced` objective:

| Weight | Algorithm | Expanded | Time | Route cost |
|---|---|---|---|---|
| `0` | Dijkstra | 192,299 | 5.3 s | optimal |
| `1` | A* | 132,033 | 3.7 s | optimal |
| `1.5` | weighted A* | 80,385 | 2.5 s | +1.6 % |
| `3` | weighted A* | 37,119 | 1.2 s | **+56 %** |

Dijkstra and A* return the *same route* — A* just looks at 31 % less of the
map. That is what an admissible heuristic buys. Above 1 the heuristic
becomes inadmissible: far less search, and a route allowed to be worse.

### Sampling — `planner:=rrt`

```bash
ros2 launch roboto_ros demo.launch.py planner:=rrt objective:=distance
ros2 launch roboto_ros demo.launch.py planner:=rrt rrt_samples:=12000
```

RRT* builds a tree from random samples of the continuous space instead of
enumerating cells, so its cost does **not** scale with map size. It is only
*asymptotically* optimal. Both sides below use `distance`, since that is
an objective each planner expresses exactly:

| Planner | Effort | Time | Length |
|---|---|---|---|
| A* | 90,296 expansions | 1.05 s | 337.3 m |
| RRT*, 3,000 samples | 3,000 samples | 0.40 s | 405.6 m (+20.3 %) |
| RRT*, 12,000 samples | 12,000 samples | 4.96 s | 384.0 m (+13.9 %) |

Cheaper at a small budget and visibly worse; more samples converge slowly.

> **RRT\* refuses `balanced`, `time` and `effort`.** Those depend on the
> *signed* grade, which is asymmetric, and RRT\*'s asymptotic-optimality
> argument assumes a metric cost. It would still return valid routes, so
> the failure is made explicit rather than left as a quietly weaker
> guarantee:
>
> ```
> objective 'balanced' depends on the signed grade, which is asymmetric...
> Use an objective without slope weighting (distance, safe).
> ```
>
> `distance` and `safe` work fine — integrating the costmap along an edge
> reads the same in either direction.

---

## 6. Cost objectives — what "best" means

```bash
ros2 launch roboto_ros demo.launch.py objective:=safe
```

| Objective | Minimises | Uses slope? | RRT*? |
|---|---|---|---|
| `distance` | pure route length | no | yes |
| `balanced` | length + clearance + slope *(default)* | yes | no |
| `time` | duration; speed falls off uphill | yes | no |
| `effort` | metres climbed | yes | no |
| `safe` | clearance from obstacles | no | yes |

Measured on Pittsburgh with plain A*:

| Objective | Length | Time | Climb | Max up | Mean cost |
|---|---|---|---|---|---|
| `distance` | **337.3 m** | 385 s | 15.9 m | 25.5° | 40.0 |
| `balanced` | 344.4 m | 386 s | 13.8 m | 20.2° | 8.6 |
| `time` | 337.3 m | **378 s** | 13.6 m | 27.3° | 31.9 |
| `effort` | 338.0 m | 379 s | **13.6 m** | 27.3° | 27.8 |
| `safe` | 441.0 m | 498 s | 18.9 m | 21.2° | **1.3** |

Each wins its own column and none wins them all. That is the point: the
shortest path is not the best path. `safe` costs +31 % distance to buy 30×
more clearance.

> ### If you change the objective, regenerate the world
>
> Obstacles are placed as fractions **along the planned route**. `safe`
> plans 441 m where `balanced` plans 344 m — different roads. Obstacles
> laid out for one land in open ground under the other, and the demo
> silently stops demonstrating anything.
>
> ```bash
> cd ~/roboto
> .venv-gis/bin/python -m tools.gis_pipeline.inject_gazebo_obstacles \
>     --objective safe
> ```
>
> The same applies to `--heuristic-weight` above 1, which changes the
> route. Weights `0` and `1` are both optimal and give the same route, so
> those are safe to switch freely.

---

## 7. Terrain — the elevation model

Slope is not a property of a cell, it is a property of a *move*: climbing
a grade is expensive, descending it is nearly free, crossing it sideways
is a rollover risk. That makes the planning graph **directed** — the cost
from A to B differs from the cost from B to A.

Three weights in the site yaml drive it, and a hard limit blocks anything
steeper:

```yaml
slope:
  max_traversable_deg: 25.0   # hard block above this
  w_uphill: 1.0
  w_downhill: 0.3             # descending is cheaper than climbing
  w_crossslope: 0.6
```

The robot reads elevation from `dem_local.npz`, a numpy copy of the DEM
written by `build_slope`. **It must exist or the robot plans with no slope
cost at all** — the node says which at startup:

```
terrain loaded: 400x400 @ 1.0 m, relief 53.3 m
```
```
no dem_npz: planning on distance and obstacles only, slope will be IGNORED
```

To run the **no-terrain ablation**, use an objective with flat weights:

```bash
ros2 launch roboto_ros demo.launch.py objective:=distance   # slope ignored
ros2 launch roboto_ros demo.launch.py objective:=balanced   # slope weighted
```

A 2D lidar at 0.30 m **cannot see a hole** — negative obstacles are below
the scan plane. The robot avoids steep ground because the DEM told it to,
not because it detected anything. SLAM keeps the map honest about
obstacles the map lacks; the map keeps the robot safe from hazards the
sensor cannot see.

---

## 8. Reacting to a blockage

```bash
ros2 launch roboto_ros demo.launch.py replan_mode:=detour
```

| Mode | What it does |
|---|---|
| `global` *(default)* | throws the route away and replans to the goal |
| `detour` | routes around the obstacle and rejoins the original path |

A global replan is correct but looks wrong: an obstacle a quarter of the
way along re-routes the remaining three quarters, so the robot appears to
abandon its plan rather than step around a barrier. A detour keeps the
route recognisable and searches a far smaller area.

It is **not** strictly better. A detour cannot find a better road that
happens to exist, and it fails outright where a global replan succeeds —
if the obstacle blocks the only way through, no rejoin point is reachable.
Both modes fall back, and the log says so:

```
detour #2 (urgent): rejoins at index 431, 18.4 m new, 2104 nodes, 31 ms
detour failed (no rejoin within 80 m); going global
plan #3 (urgent): 189.5 m, 67746 nodes, 779 ms
```

Measured over the same three scenarios, 8 obstacles in total:

| | `global` | `detour` |
|---|---|---|
| Successful missions | 3/3 | 3/3 |
| Collisions | 0 | 0 |
| Replans | 3.7 | 4.0 |
| **Mean nodes expanded** | **310,510** | **16,667** |
| Mean extra distance | +24.7 m | +32.2 m |
| Local / global replans | 0 / 11 | 12 / 0 |

**The search is 19× cheaper and the route is 7.5 m longer.** That is the
whole trade, and it is why neither mode is an obvious default. Note also
that the detour never needed its fallback here — 12 local, 0 global — so
on this site every blockage had a reachable rejoin.

Reproduce with:

```bash
.venv-gis/bin/python -m tools.run_mission_offline \
    --seeds 3 --detect-every 3 --confirm-after 1 --replan-mode detour
```

---

## 9. Maps (sites)

```bash
ROBOTO_SITE=sanfrancisco_russian_hill ros2 launch roboto_ros demo.launch.py
ROBOTO_SITE=chicago_west_loop         ros2 launch roboto_ros demo.launch.py
ROBOTO_SITE=pittsburgh_greenfield     ros2 launch roboto_ros demo.launch.py
```

Three sites, chosen to span the terrain range:

| Site | Relief | Above 25° | Buildings | Character |
|---|---|---|---|---|
| `chicago_west_loop` | 3.5 m | 0.0 % | 154 | flat grid — the control |
| `pittsburgh_greenfield` | 53.3 m | 10.3 % | 236 | rolling, an escarpment |
| `sanfrancisco_russian_hill` | 71.7 m | 16.1 % | 332 | steep, dense |

Chicago is deliberately the **control**: on level ground the slope term
should barely change the route, and it does not. If terrain-aware planning
reroutes there, the weighting is too aggressive.

```
                       length     time   climb  max up  mean cost
chicago    distance    473.2m     484s    3.4m    5.6d       54.3
           balanced    528.8m     533s    1.4m    5.1d        0.1
pittsburgh distance    337.3m     385s   15.9m   25.5d       40.0
           balanced    344.4m     386s   13.8m   20.2d        8.6
sanfran    distance    482.7m     565s   27.3m   21.1d       50.5
           balanced    507.9m     589s   26.9m   20.5d        1.1
```

On flat Chicago the objectives differ in **clearance** but hardly in
climb. In San Francisco every route climbs about 27 m because the terrain
leaves no alternative. That contrast is the argument for more than one
site.

### Adding a site

Needs network. **USGS 3DEP is US-only**, so a non-US site has no elevation.

```bash
cd ~/roboto
CFG=tools/gis_pipeline/sites/<name>.yaml     # copy an existing one, edit origin
.venv-gis/bin/python -m tools.gis_pipeline.fetch_osm    --site $CFG
.venv-gis/bin/python -m tools.gis_pipeline.fetch_dem    --site $CFG
.venv-gis/bin/python -m tools.gis_pipeline.build_slope  --site $CFG
.venv-gis/bin/python -m tools.gis_pipeline.build_prior  --site $CFG
.venv-gis/bin/python -m tools.pick_mission              --site $CFG
# paste the suggested robot_spawn and mission.goal into the yaml, then:
.venv-gis/bin/python -m tools.gis_pipeline.build_prior  --site $CFG
.venv-gis/bin/python -m tools.gis_pipeline.build_world  --site $CFG
.venv-gis/bin/python -m tools.gis_pipeline.inject_gazebo_obstacles --site $CFG
```

**Coordinates do not transfer between sites.** Pittsburgh's spawn
`(-120, -95)` is inside a building in San Francisco; its goal `(95, 110)`
is a lethal cell in Chicago. `pick_mission` finds endpoints that are on a
road, clear of obstacles, inside the extent, and **verified connected by
the real planner with the real terrain** — on a steep site a route can be
blocked by grade alone, which reading the costmap will not tell you.

An unknown `ROBOTO_SITE` fails loudly rather than silently loading one
site's config against another's data.

---

## 10. Obstacles

```bash
cd ~/roboto
.venv-gis/bin/python -m tools.gis_pipeline.inject_gazebo_obstacles \
    --scenario-seed 3 --min-obstacles 6 --max-obstacles 6
```

| Flag | Default | Meaning |
|---|---|---|
| `--scenario-seed` | `0` | layout seed; same value rebuilds the same world |
| `--min-obstacles` / `--max-obstacles` | `6` / `6` | how many |
| `--objective` | `balanced` | must match the demo's |
| `--heuristic-weight` | `1.0` | must match the demo's |
| `--goal` | *site's* | must match the demo's |
| `--site` | default site | which site |

Obstacles go into the **world only**. The prior costmap and occupancy grid
are untouched, so the robot cannot know about them until its lidar finds
them — which is the whole experiment.

---

## 11. Following a run

The launch prints a lot of Ignition transport chatter. The lines that
matter start with `roboto_node-3`:

```bash
ros2 launch roboto_ros demo.launch.py 2>&1 | grep --line-buffered "roboto_node-3"
```

Or keep the full log and watch the interesting part elsewhere:

```bash
ros2 launch roboto_ros demo.launch.py > /tmp/demo.log 2>&1
# in a second terminal:
tail -f /tmp/demo.log | grep --line-buffered -E "plan #|detour|GOAL|ERR|stuck"
```

At startup the node states its configuration:

```
roboto_node up. goal=(95.0, 110.0), grid 3000x3000 @ 0.1 m,
  objective=balanced, search=A*
terrain loaded: 400x400 @ 1.0 m, relief 53.3 m
```

Every 2 s it prints a diagnostic line:

```
slam ( -91.6, -70.3, +60.6d) ERR slam 0.06 m odom 1.13 m pitch 0.0d (max 0.0)
  | score 0.998 matched 458 rejected 0 corr 0.031 m | obj 0 travelled 41.9 m
```

| Field | Meaning |
|---|---|
| `ERR slam` / `ERR odom` | error against Gazebo ground truth |
| `score` | scan-match quality, 0–1 |
| `matched` / `rejected` | cumulative match outcomes |
| `pitch` | robot tilt; should stay at 0 |
| `obj` | confirmed discrepancy objects |

At the goal it reports the live ATE:

```
GOAL REACHED. travelled 514.0 m, 5 plans, 801 s
live ATE over 411 samples: SLAM RMSE 0.79 m (max 2.43, final 1.28)
  | odometry RMSE 12.08 m (max 30.15) | max pitch 0.0 deg | forced replans 0
```

To block until it finishes:

```bash
bash ~/roboto/setup/wait_for_goal.sh /tmp/demo.log 25
```

---

## 12. Stopping

The simulation does **not** exit when the robot reaches the goal, and it
keeps a CPU core busy under software rendering. Ctrl+C in the launch
terminal, then:

```bash
pkill -f "ign[ ]gazebo"; pkill -f "lib/roboto[_]ros"
pkill -f parameter_bridge; pkill -f rviz2
```

---

## 13. Offline experiments

No Gazebo, seconds to minutes. This is where the report's numbers come
from.

```bash
cd ~/roboto

# planners and objectives, all in one table
.venv-gis/bin/python -m tools.run_planner_comparison
.venv-gis/bin/python -m tools.run_planner_comparison \
    --site tools/gis_pipeline/sites/chicago_west_loop.yaml

# SLAM vs odometry
.venv-gis/bin/python -m tools.run_slam_offline

# detection recall / precision
.venv-gis/bin/python -m tools.run_discrepancy_offline --seeds 4

# closed loop, and the ablation that argues for the contribution
.venv-gis/bin/python -m tools.run_mission_offline \
    --seeds 3 --detect-every 3 --confirm-after 1
.venv-gis/bin/python -m tools.run_mission_offline --seeds 3 --no-replan
.venv-gis/bin/python -m tools.run_mission_offline \
    --seeds 3 --replan-mode detour --objective safe

# detection latency: how far out was each obstacle first seen?
.venv-gis/bin/python -m tools.debug_mission

# tests: no ROS, no Gazebo
.venv-gis/bin/python -m pytest tests/ -q
```

`run_mission_offline` takes `--objective`, `--heuristic-weight`,
`--replan-mode`, `--seeds`, `--scenario-seed`, `--min-obstacles` and
`--max-obstacles`, so any combination above can be measured offline before
committing to a slow Gazebo run.

---

## 14. Troubleshooting

**No window appears, or the title says "copy mode".**
WSLg failed to allocate shared memory. Check from Ubuntu:

```bash
ls -d /mnt/shared_memory && echo OK
```

If that says *No such file or directory*, restart WSL from **PowerShell**:

```powershell
wsl --shutdown
```

Wait ~10 s, reopen, relaunch. If it persists, `wsl --update` then shut
down again.

**A window opens but nothing renders.**
You probably sourced `setup/gazebo_env.sh`. Don't — see §2.

**The Gazebo 3D view is black.**
That is the WSL2 rendering problem. Diagnose with
`bash ~/roboto/setup/gate_a_check.sh`.

**`Command 'wsl' not found`.**
You typed a Windows command inside Ubuntu. Anything starting
`wsl -d Ubuntu-22.04 -- bash -lc` belongs in PowerShell.

**`objective ... depends on the signed grade`.**
You asked RRT* for a slope-weighted objective. Use `objective:=distance`
or `objective:=safe`, or switch to `planner:=astar` — see §5.

**The robot drives past obstacles without reacting.**
The world and the demo disagree about the route. Regenerate the obstacles
with the same `--objective`, `--heuristic-weight` and `--goal` the demo
uses — see §6.

**`no dem_npz: ... slope will be IGNORED`.**
Run `build_slope` for that site — see §9.

**`ROBOTO_SITE=... has no config`.**
That site has no `tools/gis_pipeline/sites/<name>.yaml`. The error lists
what is available.

**The buildings are white and flat.**
An old world build. Rerun `build_world` — the current one writes vertex
normals and a material file.
