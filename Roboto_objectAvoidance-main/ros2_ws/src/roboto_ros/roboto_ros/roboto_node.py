"""Live mission node: SLAM, discrepancy detection, replanning, driving.

    /scan, /odom  --->  SLAM  --->  map
                                     |
                          classify against the GIS prior
                                     |
                         MISSED objects block the path?
                                     |
                            replan  --->  pure pursuit  --->  /cmd_vel

WHY ONE NODE AND NOT FOUR
-------------------------
The obvious ROS decomposition is slam / detector / planner / controller.
It is the wrong one here for a concrete reason: the shared map is a
3000 x 3000 grid, which is 9 MB as an OccupancyGrid message. Passing that
between nodes at any useful rate would dominate the entire system, and a
capstone demo that spends its cycles serialising a map is not
demonstrating anything.

So the pipeline runs in one process against the in-memory map, and the
map is published only as a DOWNSAMPLED grid for RViz, at 1 Hz. Everything
substantive still lives in roboto_core and is tested without ROS; this
file is message plumbing.
"""

from __future__ import annotations

import math
import os
import time

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped, Twist
from nav_msgs.msg import Odometry, OccupancyGrid, Path
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy, QoSProfile, QoSReliabilityPolicy
from sensor_msgs.msg import LaserScan
from std_msgs.msg import ColorRGBA
from tf2_ros import TransformBroadcaster
from geometry_msgs.msg import TransformStamped, Point
from tf2_msgs.msg import TFMessage
from visualization_msgs.msg import Marker, MarkerArray

from roboto_core.discrepancy.cluster import DiscrepancyTracker, cluster
from roboto_core.discrepancy.compare import Cls, DiscrepancyConfig, classify
from roboto_core.frames import GridSpec, load_occupancy
from roboto_core.plan.astar import AStarPlanner, TerrainWeights, objectives
from roboto_core.plan.cost_fusion import (LETHAL, fuse, inflate_lethal,
                                          path_is_blocked)
from roboto_core.plan.detour import plan_detour
from roboto_core.plan.pursuit import PurePursuit, PursuitConfig
from roboto_core.plan.rrt import RRTConfig, RRTStarPlanner
from roboto_core.plan.smooth import resample, smooth_path
from roboto_core.plan.terrain import TerrainGrid
from roboto_core.slam.scan import Scan
from roboto_core.slam.slam import Slam2D, SlamConfig
from roboto_core.slam.transforms2d import compose, relative

LATCHED = QoSProfile(depth=1, durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
                     reliability=QoSReliabilityPolicy.RELIABLE)


class RobotoNode(Node):

    def __init__(self):
        super().__init__("roboto_node")

        self.declare_parameter("site_yaml", "")
        self.declare_parameter("prior_yaml", "")
        self.declare_parameter("prior_cost_npz", "")
        self.declare_parameter("dem_npz", "")
        self.declare_parameter("goal_x", 95.0)
        self.declare_parameter("goal_y", 110.0)
        self.declare_parameter("objective", "balanced")
        self.declare_parameter("heuristic_weight", 1.0)
        self.declare_parameter("planner", "astar")
        self.declare_parameter("rrt_samples", 6000)
        self.declare_parameter("max_speed", 0.0)
        self.declare_parameter("max_correction_m", 0.3)
        # Smallest MISSED cluster treated as a real object.
        #
        # This was hardcoded at 0.5, and that is too coarse for the thin
        # obstacles a street actually contains. A 2D lidar returns a
        # ONE-CELL-DEEP strip -- the near face -- so the evidence for an
        # object is its visible width, not its footprint:
        #
        #   delivery_truck_3        6.9 x 2.6 m   face 0.93 m2   passed
        #   construction_barrier_4  3.1 x 1.1 m   face 0.31 m2   DISCARDED
        #
        # The barrier was seen by the lidar every cycle and thrown away
        # before it could become an object, so no replan ever fired and the
        # robot drove into it. An offline sweep over 60 missions had
        # already put this at 0.2 (4/5 missions against 2/5, precision
        # unchanged at 30/30 with no false positives); the harnesses were
        # updated and the live node was not.
        self.declare_parameter("min_area_m2", 0.2)
        self.declare_parameter("lidar_offset_x", 0.15)
        self.declare_parameter("detect_period_s", 1.0)
        self.declare_parameter("map_period_s", 2.0)
        self.declare_parameter("plan_margin_m", 1.0)
        self.declare_parameter("replan_mode", "global")
        # DETECTION RANGE, NOT STOPPING DISTANCE, SETS THIS.
        #
        # A blockage inside this horizon replans immediately, ignoring the
        # cooldown. At 10 m that was too late to be useful: a replan costs
        # about 9 m of travel (a ~10 s A* search at 0.9 m/s) plus 5.2 m to
        # confirm and brake, against detection ranges of 12-17 m. Per
        # obstacle on the Pittsburgh route, detection range minus that
        # 14.2 m of reaction:
        #
        #   truck   visible   detect   margin
        #     0       8.9 m   17.0 m   +2.8 m   cleared
        #     1       6.4 m   12.2 m   -2.0 m   COLLIDED
        #     2       6.6 m   12.7 m   -1.5 m   COLLIDED
        #     3       9.2 m   17.5 m   +3.3 m   cleared
        #     5       7.0 m   13.3 m   -0.9 m   COLLIDED
        #
        # Detection range scales with an obstacle's VISIBLE WIDTH -- beams
        # on target fall as 1/range -- so a smaller or obliquely-approached
        # obstacle is seen later. The three that collided are exactly the
        # three with negative margin; it is arithmetic, not luck.
        #
        # 20 m starts the reaction while there is still room to finish it.
        self.declare_parameter("emergency_m", 20.0)
        # Cooldown removed (0.0), on the reasoning that its premise is gone.
        #
        # It existed to break a feedback loop: a path planned around a
        # partly-seen obstacle was invalidated as soon as the obstacle grew
        # by one cell, and it grew every cycle because each closer look
        # revealed more of it. That produced 124 replans in one mission.
        #
        # Two things changed. plan_margin_m means the obstacle must grow
        # THROUGH the margin before the path counts as blocked, and
        # occlusion filling now marks the whole body from the first
        # sighting rather than accumulating face cells -- truck coverage
        # 72% to 99.7% in one detection. With the obstacle no longer
        # growing, the loop has nothing to feed on.
        #
        # The cost of keeping it was concrete: up to 8 m of travel between
        # knowing an obstacle is on the path and being allowed to act,
        # against detection ranges of 12-17 m and a reaction that already
        # costs 14 m.
        #
        # Kept as a parameter, not deleted: if replan counts climb back
        # toward three figures, this is the first thing to restore.
        self.declare_parameter("replan_cooldown_m", 0.0)
        self.declare_parameter("autostart", True)
        self.declare_parameter("diag_period_s", 2.0)
        self.declare_parameter("stuck_timeout_s", 15.0)
        self.declare_parameter("truth_topic", "")
        self.declare_parameter("truth_frame", "roboto_bot")

        self.goal = (float(self.get_parameter("goal_x").value),
                     float(self.get_parameter("goal_y").value))
        self.lidar_dx = float(self.get_parameter("lidar_offset_x").value)

        self._load_site()

        # --- state --------------------------------------------------------
        self.slam: Slam2D | None = None
        self.odom_pose: np.ndarray | None = None
        self.last_scan: Scan | None = None
        self.path: np.ndarray = np.zeros((0, 2))
        self.path_index = 0
        self.travelled = 0.0
        self.last_replan = -1e9
        self.replans = 0
        self.prev_xy: np.ndarray | None = None
        self.tracker = DiscrepancyTracker(confirm_after=1)
        self.dcfg = DiscrepancyConfig(register=False)
        # Driving speed. 0 means "use the robot's own limit from
        # site.yaml", which is 1.0 m/s -- walking pace, so a 500 m mission
        # takes about ten minutes of simulated time and longer than that on
        # the wall clock, because software rendering runs below real time.
        # Raising it is purely a demo convenience and changes nothing the
        # planner or the estimator does.
        pcfg = PursuitConfig()
        want = float(self.get_parameter("max_speed").value)
        limit = float(self.site["robot"]["max_speed_mps"])
        pcfg.max_speed = min(want, limit) if want > 0 else min(pcfg.max_speed,
                                                               limit)
        self.pursuit = PurePursuit(pcfg)
        self.objects: list = []
        self.finished = False
        self.t_start = time.time()

        # Diagnostics. The previous live failure could only be diagnosed
        # after the fact, from the final pose and a live topic dump, because
        # the node logged nothing between plans -- 32 m of SLAM divergence
        # went unrecorded for four minutes. These pin the estimate down
        # while it is happening.
        self.start_pose: np.ndarray | None = None
        self.first_odom: np.ndarray | None = None
        self.stopped_since: float | None = None
        self.forced_replans = 0

        # Simulator ground truth, for diagnostics ONLY. Nothing in the
        # estimation or control path may read this -- it exists so live
        # error is measurable at all. It stays optional: on a real robot
        # the topic is simply absent and the diagnostics degrade to
        # reporting the estimate without an error column.
        self.truth_pose: np.ndarray | None = None
        self.truth_err: list[float] = []
        self.odom_err: list[float] = []
        self.pitch_deg = 0.0            # current
        self.max_pitch_deg = 0.0        # peak once settled, see on_truth

        # --- ROS interface -------------------------------------------------
        self.create_subscription(LaserScan, "scan", self.on_scan, 10)
        self.create_subscription(Odometry, "odom", self.on_odom, 20)
        truth_topic = str(self.get_parameter("truth_topic").value)
        if truth_topic:
            self.create_subscription(TFMessage, truth_topic, self.on_truth, 10)

        self.pub_cmd = self.create_publisher(Twist, "cmd_vel", 10)
        self.pub_path = self.create_publisher(Path, "plan", LATCHED)
        self.pub_pose = self.create_publisher(PoseStamped, "slam/pose", 10)
        self.pub_map = self.create_publisher(OccupancyGrid, "slam/map", LATCHED)
        self.pub_prior = self.create_publisher(OccupancyGrid, "prior/map", LATCHED)
        self.pub_marks = self.create_publisher(MarkerArray, "discrepancies", 10)
        self.pub_robot = self.create_publisher(MarkerArray, "robot_marker", 10)
        self.pub_ends = self.create_publisher(MarkerArray, "endpoints", LATCHED)
        self.tf = TransformBroadcaster(self)

        self.create_timer(0.1, self.control_tick)
        self.create_timer(float(self.get_parameter("detect_period_s").value),
                          self.detect_tick)
        self.create_timer(float(self.get_parameter("map_period_s").value),
                          self.map_tick)
        self.create_timer(float(self.get_parameter("diag_period_s").value),
                          self.diag_tick)

        alg = ("RRT*" if self.planner_kind == "rrt" else
               "Dijkstra" if self.heuristic_weight == 0 else
               "A*" if self.heuristic_weight == 1 else
               f"weighted A* ({self.heuristic_weight:g})")
        self.get_logger().info(
            f"roboto_node up. goal=({self.goal[0]:.1f}, {self.goal[1]:.1f}), "
            f"grid {self.grid.width}x{self.grid.height} @ {self.grid.resolution} m, "
            f"objective={self.objective.name}, search={alg}, "
            f"max_speed={self.pursuit.cfg.max_speed:.2f} m/s")
        self._publish_prior()

    # ---- setup -----------------------------------------------------------

    def _load_site(self):
        import yaml

        site_path = self.get_parameter("site_yaml").value
        with open(site_path, "r", encoding="utf-8") as fh:
            self.site = yaml.safe_load(fh)

        ext = self.site["extent_m"]
        res = float(self.site["resolution_m"])
        self.grid = GridSpec(
            origin_x=float(ext["xmin"]), origin_y=float(ext["ymin"]),
            resolution=res,
            width=int(round((ext["xmax"] - ext["xmin"]) / res)),
            height=int(round((ext["ymax"] - ext["ymin"]) / res)))

        self.prior_occ, prior_grid = load_occupancy(
            self.get_parameter("prior_yaml").value)
        prior_grid.assert_compatible(self.grid, "prior vs site grid")

        d = np.load(self.get_parameter("prior_cost_npz").value)
        self.prior_cost = d["cost"]

        # Two costmaps, and the distinction is load-bearing.
        #
        #   prior_cost  what is actually impassable -- used to ask "is the
        #               route blocked?", where being permissive avoids
        #               replanning over nothing.
        #   plan_cost   the same, grown by the planning margin -- used for
        #               every plan, so routes keep real clearance.
        #
        # The margin used to be applied on discrepancy replans only, so the
        # INITIAL plan, the one followed for the first couple of hundred
        # metres, could legally graze a wall at 0.36 m while the
        # controller's brake stops dead at 0.6 m. Live, the robot ended up
        # wedged against a building it had known about from the start.
        margin = float(self.get_parameter("plan_margin_m").value)
        self.plan_cost = inflate_lethal(self.prior_cost, self.grid, margin)

        self.r_robot = float(self.site["robot"]["radius_m"])
        # Contact check map: prior inflated by the ROBOT RADIUS, not by the
        # planning margin. The margin exists to keep routes off walls; this
        # asks the narrower question of whether the footprint is in one.
        # Refreshed in detect_tick as discovered obstacles are fused in.
        # CONTACT MAP: the robot's FOOTPRINT, and nothing else.
        #
        # Buildings, inflated by the robot radius once at startup. Detected
        # objects are kept separately as RAW cells (_object_cells) and
        # tested against a small disc at probe time, for two reasons:
        #
        #  * correctness. Reusing fuse() for this was wrong twice over --
        #    it adds unknown_depth_m AND the occlusion shadow, giving a
        #    measured reach of 4.30 m where the comment claimed 0.32 m, so
        #    the robot would have reversed away from obstacles 4 m off. It
        #    also does not re-inflate the prior, so the building contact
        #    margin silently halved from 0.70 m to 0.35 m the moment the
        #    first object was confirmed.
        #  * cost. fuse() runs a distance transform over 9 megacells, 0.62 s
        #    measured, and detect_tick already runs two. A third would have
        #    put ~1.9 s of work in a 1 Hz callback on the single-threaded
        #    executor -- starving the 10 Hz loop that publishes cmd_vel,
        #    which is the exact hazard _halt() exists to prevent.
        self._contact_cost = inflate_lethal(self.prior_cost, self.grid,
                                            self.r_robot)
        self._object_cells = np.zeros(self.grid.shape, dtype=bool)
        self._probe_r = max(1, int(round(self.r_robot / self.grid.resolution)))
        # Metres to the nearest building, computed once. Path smoothing uses
        # it so a shortcut cannot spend the standoff A* paid for -- see
        # roboto_core/plan/smooth.py. Buildings only: they are what the
        # robot wedges between, and recomputing this per replan would add a
        # 9-megacell distance transform to every 5-second plan.
        from scipy.ndimage import distance_transform_edt
        self._clear_field = (distance_transform_edt(self.prior_cost < LETHAL)
                             * self.grid.resolution)
        self.sensor_range = float(self.site["lidar"]["range_max_m"])

        # What "best" means for a route, and how hard the search leans on
        # its heuristic. Both are built from the site's own configuration,
        # via the same objectives() the offline comparison uses, so the
        # live robot and the experiments cannot drift apart.
        avail = objectives(self.site)
        name = str(self.get_parameter("objective").value)
        if name not in avail:
            raise ValueError(
                f"unknown objective {name!r}; choose from {sorted(avail)}")
        self.objective = avail[name]
        self.heuristic_weight = float(
            self.get_parameter("heuristic_weight").value)
        self.weights = self.objective.weights

        # Which planner. RRT* refuses objectives that depend on the signed
        # grade, so validate that here rather than at the first replan --
        # failing at startup is far easier to act on than failing 200 m
        # into a mission.
        self.planner_kind = str(self.get_parameter("planner").value)
        if self.planner_kind not in ("astar", "rrt"):
            raise ValueError(
                f"unknown planner {self.planner_kind!r}; use astar or rrt")
        if self.planner_kind == "rrt":
            RRTStarPlanner(np.zeros((2, 2), dtype=np.uint8), self.grid,
                           objective=self.objective)
        # Terrain, from the numpy copy of the DEM.
        #
        # This used to be unconditionally None, because the DEM lives in a
        # GeoTIFF and rasterio is a GIS-pipeline dependency that is not
        # installed in the ROS environment. The consequence was easy to
        # miss and hard to defend: the offline experiments planned WITH
        # directional slope cost and the live robot planned without it, so
        # the two were not solving the same problem and their routes were
        # not comparable. build_slope now also writes dem_local.npz, which
        # needs nothing but numpy.
        #
        # Still optional. A site built before that existed, or a robot with
        # no elevation data at all, falls back to distance-and-obstacles
        # planning rather than refusing to start -- but it says so, because
        # silently dropping slope is exactly what went wrong before.
        dem_path = str(self.get_parameter("dem_npz").value)
        self.terrain = None
        if dem_path and os.path.exists(dem_path):
            self.terrain = TerrainGrid.from_npz(dem_path)
            self.get_logger().info(
                f"terrain loaded: {self.terrain.grid.width}x"
                f"{self.terrain.grid.height} @ "
                f"{self.terrain.grid.resolution:.1f} m, "
                f"relief {self.terrain.relief:.1f} m")
        else:
            self.get_logger().warn(
                "no dem_npz: planning on distance and obstacles only, "
                "slope will be IGNORED (run build_slope to generate it)")

    # ---- callbacks -------------------------------------------------------

    def on_odom(self, msg: Odometry):
        p = msg.pose.pose
        q = p.orientation
        yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                         1.0 - 2.0 * (q.y * q.y + q.z * q.z))
        self.odom_pose = np.array([p.position.x, p.position.y, yaw])

    def on_truth(self, msg: TFMessage):
        """Simulator ground truth. DIAGNOSTICS ONLY -- never read by SLAM,
        the planner or the controller.

        Pitch is worth tracking as well as position. The ground is flat by
        design precisely so a 2D lidar stays in plane, so any pitch means
        the robot has climbed something and its scans are being taken
        through a tilted plane whatever the estimator does with them.

        The peak ignores the first few seconds. The robot is spawned just
        clear of the ground and settles onto it, which pitches it about
        12 degrees once -- a running max from t=0 latches that immediately
        and then reports it for the rest of the mission, which is how the
        first instrumented run came to claim a 11.8 degree peak before the
        robot had gone anywhere.
        """
        want = str(self.get_parameter("truth_frame").value)
        for tr in msg.transforms:
            if tr.child_frame_id != want:
                continue
            q = tr.transform.rotation
            yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                             1.0 - 2.0 * (q.y * q.y + q.z * q.z))
            pitch = math.asin(max(-1.0, min(1.0, 2.0 * (q.w * q.y - q.z * q.x))))
            self.truth_pose = np.array([tr.transform.translation.x,
                                        tr.transform.translation.y, yaw])
            self.pitch_deg = abs(math.degrees(pitch))
            if time.time() - self.t_start > 5.0:
                self.max_pitch_deg = max(self.max_pitch_deg, self.pitch_deg)
            return

    def on_scan(self, msg: LaserScan):
        n = len(msg.ranges)
        if n == 0:
            return
        angles = msg.angle_min + np.arange(n) * msg.angle_increment
        ranges = np.asarray(msg.ranges, dtype=np.float64)

        # Trim the FOV extremes: with a 270 degree field Ignition stitches
        # several depth-camera faces and the outermost rays land on a seam,
        # disagreeing with their neighbours by metres.
        if n > 8:
            ranges, angles = ranges[1:-1], angles[1:-1]

        self.last_scan = Scan(ranges=ranges, angles=angles,
                              range_min=float(msg.range_min),
                              range_max=float(msg.range_max))

        if self.slam is None and self.odom_pose is not None:
            self._start()
        if self.slam is not None:
            # The lidar sits forward of base_link; SLAM works in the sensor
            # frame so the scan needs no per-beam transform.
            self.slam.update(self.last_scan, self._sensor_pose(self.odom_pose))

    def _sensor_pose(self, base_pose):
        c, s = math.cos(base_pose[2]), math.sin(base_pose[2])
        return np.array([base_pose[0] + self.lidar_dx * c,
                         base_pose[1] + self.lidar_dx * s,
                         base_pose[2]])

    def _start(self):
        # Gazebo's DiffDrive odometry is relative to wherever the robot
        # started, so it reports (0, 0, 0) at spawn. The world pose comes
        # from site.yaml instead -- the same single source of truth the
        # world generator used to place the robot. Slam2D consumes odometry
        # as DELTAS, so this only has to be right once.
        sp = self.site["robot_spawn"]
        start = self._sensor_pose(np.array([
            float(sp["x"]), float(sp["y"]), math.radians(float(sp["yaw_deg"]))]))
        # Overridable per site: a repeating street grid (Chicago) prefers a
        # tighter limit than a robot recovering from an excursion does. See
        # SlamConfig.max_correction_m for the measurements behind 3.0.
        cfg = SlamConfig(max_correction_m=float(
            self.get_parameter("max_correction_m").value))
        self.slam = Slam2D(self.grid, sensor_range=self.sensor_range,
                           config=cfg, initial_pose=start,
                           prior_occ=self.prior_occ)
        self.get_logger().info(
            f"SLAM max_correction_m = {cfg.max_correction_m:.2f}")
        self.start_pose = start.copy()
        self.first_odom = self._sensor_pose(self.odom_pose).copy()
        self.prev_xy = None if self.odom_pose is None else self.odom_pose[:2].copy()
        self.get_logger().info(
            f"SLAM started at ({start[0]:.1f}, {start[1]:.1f}); planning")
        self._publish_endpoints(start)
        if not self._replan(start, self.plan_cost, "initial"):
            self._replan(start, self.prior_cost, "initial (no margin)")

    # ---- planning --------------------------------------------------------

    def _contact_ahead(self, pose) -> bool:
        """Is the map claiming something solid is against the bumper?

        The lidar cannot answer this. Below its 0.15 m minimum range every
        beam is invalid, the brake sees no valid returns and declines to
        brake, and the robot pushes into whatever it is touching -- growing
        more confident the closer it gets, since more beams drop out. This
        is the map's independent opinion, checked one robot radius plus a
        little ahead of the SLAM pose.

        `_contact_cost` is inflated by the robot radius, so a lethal cell at
        that probe point means the footprint is about to overlap something,
        not merely that something is nearby. It costs one array lookup per
        control cycle.
        """
        if self._contact_cost is None:
            return False
        look = self.r_robot + 0.25
        x = float(pose[0]) + look * math.cos(float(pose[2]))
        y = float(pose[1]) + look * math.sin(float(pose[2]))
        r, c = self.grid.world_to_cell(x, y)
        if not self.grid.in_bounds(r, c):
            return False
        r, c = int(r), int(c)
        # Buildings: already inflated by the robot radius at startup.
        if self._contact_cost[r, c] >= LETHAL:
            return True
        # Detected objects: raw cells, dilated here by the footprint. A
        # 7x7 window at 0.1 m resolution, so the cost is negligible and no
        # distance transform is needed on the detection path.
        k = self._probe_r
        r0, r1 = max(0, r - k), min(self.grid.shape[0], r + k + 1)
        c0, c1 = max(0, c - k), min(self.grid.shape[1], c + k + 1)
        return bool(self._object_cells[r0:r1, c0:c1].any())

    def _tidy(self, path, cost):
        """String-pull and evenly resample a freshly planned path.

        8-connected A* on a 4x-downsampled grid can only produce headings
        in 45 degree steps, so a straight road arrives as a staircase with
        2 m treads. Pure pursuit's lookahead is about 2.3 m -- roughly one
        tread -- so its target hops across the true line and the robot
        weaves. The controller was tracking correctly; the path was the
        zigzag. See roboto_core/plan/smooth.py for why shortcuts are capped
        rather than unbounded.
        """
        try:
            out = smooth_path(np.asarray(path, dtype=float), self.grid, cost,
                              clearance=self._clear_field,
                              # _clear_field measures distance to lethal
                              # cells in prior_cost, which is ALREADY
                              # inflated by the robot radius, so adding it
                              # again asked for 1.7 m from the real wall.
                              min_clearance_m=1.0)
            return resample(out, 0.5)
        except Exception as exc:                      # never fail a mission
            self.get_logger().warn(f"path smoothing skipped: {exc}")
            return path

    def _planner(self, cost):
        if self.planner_kind == "rrt":
            # No downsampling: a sampling planner never enumerates cells,
            # so the full-resolution costmap costs it nothing extra. That
            # is its main advantage over the grid search here.
            return RRTStarPlanner(
                cost, self.grid,
                RRTConfig(max_samples=int(
                    self.get_parameter("rrt_samples").value)),
                objective=self.objective)
        return AStarPlanner(cost, self.grid, downsample=4,
                            terrain=self.terrain,
                            objective=self.objective,
                            heuristic_weight=self.heuristic_weight)

    def _halt(self):
        """Stop the wheels, and mean it.

        rclpy.spin() is a SINGLE-THREADED executor, so a planning call
        blocks every other timer, control_tick included, for as long as it
        runs -- 10 to 12 seconds for a 130k-node A* search on this map.
        Nothing republishes cmd_vel in that window, and Gazebo's diff-drive
        plugin holds the last twist it was given rather than timing out, so
        the robot kept driving at its previous velocity, blind, for about
        ten metres. That is what was hitting obstacles the discrepancy
        layer had already found: not a detection failure, an actuation one.
        """
        self.pub_cmd.publish(Twist())

    def _replan(self, pose, cost, why: str) -> bool:
        self._halt()
        t0 = time.perf_counter()
        res = self._planner(cost).plan(pose[:2], self.goal)
        if not res.found:
            self.get_logger().warn(f"replan ({why}) failed: {res.reason}")
            return False

        self.path = self._tidy(res.path, cost)
        self.path_index = 0
        self.last_replan = self.travelled
        self.replans += 1
        self.get_logger().info(
            f"plan #{self.replans} ({why}): {res.length_m:.1f} m, "
            f"{res.expanded} nodes, {1000 * (time.perf_counter() - t0):.0f} ms")
        self._publish_path()
        return True

    # ---- timers ----------------------------------------------------------

    def control_tick(self):
        if self.slam is None or self.finished:
            return
        pose = self.slam.pose

        # Odometry is the odometer. Accumulating SLAM-pose displacement at
        # 10 Hz counts pose jitter as distance: a 345 m route reported as
        # 585 m, which also let the replan cooldown expire early because it
        # is measured in metres travelled.
        if self.odom_pose is not None:
            if self.prev_xy is not None:
                self.travelled += float(np.hypot(*(self.odom_pose[:2] - self.prev_xy)))
            self.prev_xy = self.odom_pose[:2].copy()

        self._publish_pose(pose)
        self._publish_tf(pose)

        if len(self.path) == 0:
            return

        sc = self.last_scan
        cmd = self.pursuit.step(
            pose, self.path,
            scan_ranges=None if sc is None else sc.ranges,
            scan_angles=None if sc is None else sc.angles,
            from_index=self.path_index,
            contact=self._contact_ahead(pose))
        self.path_index = max(self.path_index, cmd.index)

        if cmd.done:
            self.finished = True
            self.pub_cmd.publish(Twist())
            self.get_logger().info(
                f"GOAL REACHED. travelled {self.travelled:.1f} m, "
                f"{self.replans} plans, {time.time() - self.t_start:.0f} s")
            self._report_accuracy()
            return

        t = Twist()
        t.linear.x = cmd.v
        t.angular.z = cmd.omega
        self.pub_cmd.publish(t)
        self._publish_robot_marker(pose)

        # Watchdog. Even with the controller able to turn out of a hard
        # stop, a robot that has not moved for this long is not going to
        # free itself by tracking the same path: the path is what put it
        # there. Force a replan from where it actually is.
        now = time.time()
        # Forward motion only. abs() counted the contact recovery's
        # v = -0.25 as movement, so the watchdog reset on every cycle and
        # could never fire during the exact wedging case it was written for.
        moving = cmd.v > 1e-3
        if moving:
            self.stopped_since = None
        else:
            if self.stopped_since is None:
                self.stopped_since = now
            elif now - self.stopped_since >= float(
                    self.get_parameter("stuck_timeout_s").value):
                self.stopped_since = now
                self.forced_replans += 1
                cost = (self.plan_cost if not self.objects else
                        fuse(self.plan_cost, self.grid, self.objects,
                             robot_radius_m=self.r_robot,
                             viewpoint=pose[:2]))
                self.get_logger().warn(
                    f"stuck for {self.get_parameter('stuck_timeout_s').value:.0f} s "
                    f"at ({pose[0]:.1f}, {pose[1]:.1f}) [{cmd.reason}]; replanning")
                self._replan(pose, cost, "unstick")

    def diag_tick(self):
        """One line of SLAM state, often enough to see a divergence start.

        Reports SLAM and dead reckoning against ground truth SEPARATELY.
        Comparing the two estimates to each other, as this first did, says
        nothing about which one is wrong -- and measured, the assumption
        behind that was backwards: Gazebo's DiffDrive odometry was 20.7 m
        off while SLAM was 12.6 m off, so the difference was mostly
        odometry drift being attributed to SLAM.
        """
        if self.slam is None or self.start_pose is None:
            return

        steps = self.slam.steps
        if not steps:
            return
        last = steps[-1]

        # Where dead reckoning alone says we are, in the map frame.
        dead = compose(self.start_pose,
                       relative(self.first_odom,
                                self._sensor_pose(self.odom_pose)))
        p = self.slam.pose

        recent = steps[-20:]
        corr = float(np.mean([np.hypot(s.correction[0], s.correction[1])
                              for s in recent])) if recent else 0.0

        if self.truth_pose is None:
            err_txt = f"drift-vs-odom {np.hypot(p[0] - dead[0], p[1] - dead[1]):5.2f} m"
        else:
            t = self.truth_pose
            e_slam = float(np.hypot(p[0] - t[0], p[1] - t[1]))
            e_odom = float(np.hypot(dead[0] - t[0], dead[1] - t[1]))
            self.truth_err.append(e_slam)
            self.odom_err.append(e_odom)
            err_txt = (f"ERR slam {e_slam:5.2f} m odom {e_odom:5.2f} m "
                       f"pitch {self.pitch_deg:4.1f}d (max {self.max_pitch_deg:4.1f})")

        self.get_logger().info(
            f"slam ({p[0]:7.1f},{p[1]:7.1f},{math.degrees(p[2]):+6.1f}d) "
            f"{err_txt} | score {last.score:.3f} "
            f"matched {self.slam.n_matched} rejected {self.slam.n_rejected} "
            # Recovery attempts. Without this the log cannot say whether
            # relocalisation ever fired: a run that diverged to 12 m looked
            # identical to one where recovery was never triggered. It also
            # exposes the window limit -- the search is +/-10 m, so once
            # the error passes that, recovery cannot succeed by construction.
            f"reloc {self.slam.n_relocalised}/{self.slam.n_reloc_attempts} "
            f"corr {corr:.3f} m | obj {len(self.objects)} "
            f"travelled {self.travelled:.1f} m")

    def _report_accuracy(self):
        """Live ATE against the simulator, sampled at the diagnostic rate.

        Coarser than the offline ATE, which scores every pose, but it is
        the only live accuracy number the project has -- and the one that
        says whether offline results transfer to Gazebo at all.
        """
        if not self.truth_err:
            self.get_logger().info("no ground-truth topic; live ATE unavailable")
            return
        s = np.asarray(self.truth_err)
        o = np.asarray(self.odom_err)
        self.get_logger().info(
            f"live ATE over {len(s)} samples: "
            f"SLAM RMSE {np.sqrt((s ** 2).mean()):.2f} m "
            f"(max {s.max():.2f}, final {s[-1]:.2f}) | "
            f"odometry RMSE {np.sqrt((o ** 2).mean()):.2f} m "
            f"(max {o.max():.2f}) | max pitch {self.max_pitch_deg:.1f} deg | "
            f"forced replans {self.forced_replans}")

    def detect_tick(self):
        if self.slam is None or self.finished or len(self.path) == 0:
            return

        res = classify(self.slam.map, self.prior_occ, self.dcfg)
        found = cluster(res, Cls.MISSED,
                        min_area_m2=float(
                            self.get_parameter("min_area_m2").value),
                        log_odds=self.slam.map.log_odds)
        self.tracker.update(found, stamp=self.travelled)
        self.objects = self.tracker.confirmed()
        self._publish_marks()
        if not self.objects:
            return

        fused = fuse(self.prior_cost, self.grid, self.objects,
                     robot_radius_m=self.r_robot,
                     viewpoint=self.slam.pose[:2])
        # Discovered obstacles must be able to trigger the contact check
        # too, otherwise it only ever protects against mapped buildings --
        # and unmapped obstacles are the ones the robot actually hits.
        #
        # Raw detected cells only -- no inflation, no shadow, no distance
        # transform. _contact_ahead dilates by the footprint at probe time.
        mask = np.zeros(self.grid.shape, dtype=bool)
        for o in self.objects:
            if o.cells is not None and len(o.cells):
                mask[o.cells[:, 0], o.cells[:, 1]] = True
        self._object_cells = mask
        pose = self.slam.pose
        urgent = path_is_blocked(self.path, self.grid, fused,
                                 from_index=self.path_index,
                                 corridor_m=self.r_robot,
                                 within_m=float(self.get_parameter("emergency_m").value))
        eventual = path_is_blocked(self.path, self.grid, fused,
                                   from_index=self.path_index,
                                   corridor_m=self.r_robot)
        cooled = (self.travelled - self.last_replan) >= float(
            self.get_parameter("replan_cooldown_m").value)

        if urgent or (eventual and cooled):
            # STOP FIRST, THEN THINK.
            #
            # _replan() halts before searching, but by then this callback
            # has already spent a second building `wide` -- a distance
            # transform over 9 megacells -- with the robot still driving at
            # whatever velocity was last published. The whole callback
            # blocks the single-threaded executor, so nothing corrects it.
            #
            # Measured: urgent fired 5.8 m from a 6.6 m truck, the search
            # took about 10 s, and the robot covered 2 m in the meantime.
            # The new path arrived when it was already 3.9 m away and
            # inside the obstacle's inflation, so it made contact anyway.
            # Halting here rather than three costmaps later is most of the
            # difference.
            self._halt()
            margin = float(self.get_parameter("plan_margin_m").value)
            wide = fuse(self.plan_cost, self.grid, self.objects,
                        robot_radius_m=self.r_robot + margin,
                        viewpoint=pose[:2])
            why = "urgent" if urgent else "blocked"

            # Route around the obstacle and rejoin, rather than abandoning
            # the rest of the route. Falls back to a global replan when no
            # rejoin is reachable -- if the obstacle blocks the only way
            # through, staying local cannot help.
            if str(self.get_parameter("replan_mode").value) == "detour":
                # Same single-threaded-executor problem as _replan: this
                # search blocks control_tick, so stop the wheels first
                # rather than coasting into the obstacle we are routing
                # around.
                self._halt()
                planner = self._planner(wide)
                t0 = time.perf_counter()
                d = plan_detour(lambda a, b: planner.plan(a, b),
                                self.path, pose, self.grid, wide,
                                from_index=self.path_index,
                                corridor_m=self.r_robot)
                if d.found:
                    self.path = self._tidy(d.path, wide)
                    self.path_index = 0
                    self.last_replan = self.travelled
                    self.replans += 1
                    self.get_logger().info(
                        f"detour #{self.replans} ({why}): rejoins at "
                        f"index {d.rejoin_index}, {d.detour_m:.1f} m new, "
                        f"{d.expanded} nodes, "
                        f"{1000 * (time.perf_counter() - t0):.0f} ms")
                    self._publish_path()
                    return
                self.get_logger().info(f"detour failed ({d.reason}); going global")

            if not self._replan(pose, wide, why):
                self._replan(pose, fused, "fallback")

    def map_tick(self):
        if self.slam is None:
            return
        self.pub_map.publish(self._grid_msg(self.slam.map.occupied(), "slam/map"))

    # ---- publishing ------------------------------------------------------

    def _grid_msg(self, occ: np.ndarray, _tag: str, step: int = 4) -> OccupancyGrid:
        """Downsampled OccupancyGrid for RViz.

        Full resolution would be 9 MB per message. Reduced by MAX so a thin
        wall never disappears between samples.
        """
        h = (self.grid.height // step) * step
        w = (self.grid.width // step) * step
        red = occ[:h, :w].reshape(h // step, step, w // step, step).max(axis=(1, 3))

        m = OccupancyGrid()
        m.header.frame_id = "map"
        m.header.stamp = self.get_clock().now().to_msg()
        m.info.resolution = self.grid.resolution * step
        m.info.width = red.shape[1]
        m.info.height = red.shape[0]
        m.info.origin.position.x = self.grid.origin_x
        m.info.origin.position.y = self.grid.origin_y
        m.info.origin.orientation.w = 1.0
        m.data = np.where(red, np.int8(100), np.int8(0)).ravel().tolist()
        return m

    def _publish_prior(self):
        self.pub_prior.publish(self._grid_msg(self.prior_occ, "prior/map"))

    def _publish_path(self):
        msg = Path()
        msg.header.frame_id = "map"
        msg.header.stamp = self.get_clock().now().to_msg()
        for x, y in self.path:
            ps = PoseStamped()
            ps.header = msg.header
            ps.pose.position.x = float(x)
            ps.pose.position.y = float(y)
            ps.pose.orientation.w = 1.0
            msg.poses.append(ps)
        self.pub_path.publish(msg)

    def _publish_pose(self, pose):
        ps = PoseStamped()
        ps.header.frame_id = "map"
        ps.header.stamp = self.get_clock().now().to_msg()
        ps.pose.position.x = float(pose[0])
        ps.pose.position.y = float(pose[1])
        ps.pose.orientation.z = math.sin(pose[2] / 2.0)
        ps.pose.orientation.w = math.cos(pose[2] / 2.0)
        self.pub_pose.publish(ps)

    def _publish_tf(self, pose):
        """map -> odom, so RViz can show everything in the map frame.

        Gazebo already publishes odom -> base_link, and map -> odom is the
        SLAM correction: the difference between where odometry thinks the
        robot is and where scan matching says it is.
        """
        if self.odom_pose is None:
            return
        dx = pose[0] - self.odom_pose[0]
        dy = pose[1] - self.odom_pose[1]
        dth = pose[2] - self.odom_pose[2]

        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = "map"
        t.child_frame_id = "odom"
        t.transform.translation.x = float(dx)
        t.transform.translation.y = float(dy)
        t.transform.rotation.z = math.sin(dth / 2.0)
        t.transform.rotation.w = math.cos(dth / 2.0)
        self.tf.sendTransform(t)

    def _publish_robot_marker(self, pose):
        """Where the robot believes it is, big enough to see while filming.

        /slam/pose already carried this, but RViz's Pose display draws a
        thin flat arrow that vanishes against the occupancy grid at the
        zoom needed to see a 340 m route. This is the same information as a
        solid body plus a heading arrow, on its own topic so it can be
        toggled independently of the estimate itself.
        """
        arr = MarkerArray()
        now = self.get_clock().now().to_msg()

        body = Marker()
        body.header.frame_id = "map"
        body.header.stamp = now
        body.ns = "robot"
        body.id = 0
        body.type = Marker.SPHERE
        body.action = Marker.ADD
        body.pose.position.x = float(pose[0])
        body.pose.position.y = float(pose[1])
        body.pose.position.z = 0.4
        body.pose.orientation.w = 1.0
        body.scale.x = body.scale.y = body.scale.z = 2.0 * self.r_robot
        body.color.r, body.color.g, body.color.b, body.color.a = \
            0.1, 0.9, 1.0, 0.95
        arr.markers.append(body)

        head = Marker()
        head.header.frame_id = "map"
        head.header.stamp = now
        head.ns = "robot"
        head.id = 1
        head.type = Marker.ARROW
        head.action = Marker.ADD
        head.pose.position.x = float(pose[0])
        head.pose.position.y = float(pose[1])
        # Sits ABOVE the body sphere, not level with it. At z equal to the
        # body the arrow spends most of its length inside a solid marker of
        # similar size and only the tip shows.
        head.pose.position.z = 1.1
        head.pose.orientation.z = math.sin(float(pose[2]) / 2.0)
        head.pose.orientation.w = math.cos(float(pose[2]) / 2.0)
        head.scale.x, head.scale.y, head.scale.z = 4.0, 0.6, 0.6
        head.color.r, head.color.g, head.color.b, head.color.a = \
            0.0, 0.0, 0.0, 1.0
        arr.markers.append(head)

        self.pub_robot.publish(arr)

    def _publish_endpoints(self, start):
        """Flat discs on the ground at start and goal.

        Latched, so they survive being published once before RViz has
        subscribed -- otherwise they appear only if RViz happened to be up
        first, which it usually is not.
        """
        arr = MarkerArray()
        now = self.get_clock().now().to_msg()
        for k, (xy, rgb, ns) in enumerate((
                (start, (0.15, 1.0, 0.25), "start"),
                (self.goal, (1.0, 0.25, 0.15), "goal"))):
            m = Marker()
            m.header.frame_id = "map"
            m.header.stamp = now
            m.ns = ns
            m.id = k
            m.type = Marker.CYLINDER
            m.action = Marker.ADD
            m.pose.position.x = float(xy[0])
            m.pose.position.y = float(xy[1])
            m.pose.position.z = 0.05
            m.pose.orientation.w = 1.0
            m.scale.x = m.scale.y = 6.0     # 6 m disc, legible when zoomed out
            m.scale.z = 0.1                 # flat on the ground
            m.color.r, m.color.g, m.color.b = rgb
            m.color.a = 0.7
            arr.markers.append(m)
        self.pub_ends.publish(arr)

    def _publish_marks(self):
        arr = MarkerArray()
        clear = Marker()
        clear.action = Marker.DELETEALL
        arr.markers.append(clear)

        for k, o in enumerate(self.objects):
            m = Marker()
            m.header.frame_id = "map"
            m.header.stamp = self.get_clock().now().to_msg()
            m.ns = "missed"
            m.id = k
            m.type = Marker.CYLINDER
            m.action = Marker.ADD
            m.pose.position.x = float(o.centroid[0])
            m.pose.position.y = float(o.centroid[1])
            m.pose.position.z = 0.5
            m.pose.orientation.w = 1.0
            d = max(0.8, 2.0 * o.radius_m)
            m.scale.x = m.scale.y = d
            m.scale.z = 1.0
            m.color = ColorRGBA(r=0.95, g=0.25, b=0.1, a=0.65)
            arr.markers.append(m)
        self.pub_marks.publish(arr)


def main():
    rclpy.init()
    node = RobotoNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.pub_cmd.publish(Twist())
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
