"""Live demo: Gazebo + bridge + the mission node + RViz.

    ros2 launch roboto_ros demo.launch.py
    ros2 launch roboto_ros demo.launch.py gui:=false rviz:=false

Environment notes that are not optional on this machine:

  * LIBGL_ALWAYS_SOFTWARE=1 -- hardware GL crashes OGRE2 under WSL2, and
    since gpu_lidar RENDERS to produce ranges, that failure takes the
    sensor with it, not just the window. setup/gazebo_env.sh carries the
    measurements behind this.
  * use_sim_time -- everything runs off Gazebo's /clock. Without it the
    node timestamps against wall time while the simulator runs at another
    rate, and TF silently falls apart.
"""

import os

import yaml

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (DeclareLaunchArgument, ExecuteProcess,
                            SetEnvironmentVariable)
from launch.conditions import IfCondition, UnlessCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    share = get_package_share_directory("roboto_ros")

    # Repo root, reached from the installed share directory:
    #   <repo>/ros2_ws/install/roboto_ros/share/roboto_ros
    repo = os.path.abspath(os.path.join(share, "..", "..", "..", "..", ".."))
    # Site selection is an environment variable rather than a launch
    # argument because the world path, the model path and the bridge topic
    # names are all needed at description-build time, before a
    # LaunchConfiguration can be resolved.
    site_name = os.environ.get("ROBOTO_SITE", "pittsburgh_greenfield")
    data = os.path.join(repo, "data", f"site_{site_name}")

    world_file = os.environ.get("ROBOTO_WORLD", "site_demo.sdf")
    world = os.path.join(data, "world", world_file)
    if not os.path.exists(world):
        world = os.path.join(data, "world", "site.sdf")

    # Per-site config lives in sites/<name>.yaml. The original single
    # site.yaml is still the default site, so it is accepted only when the
    # name actually matches it -- falling back blindly would pair one
    # site's config with another site's data, and every coordinate in the
    # run would be silently wrong while looking entirely plausible.
    site_yaml = os.path.join(repo, "tools", "gis_pipeline", "sites",
                             f"{site_name}.yaml")
    if not os.path.exists(site_yaml):
        default_yaml = os.path.join(repo, "tools", "gis_pipeline", "site.yaml")
        with open(default_yaml, "r", encoding="utf-8") as fh:
            default_name = yaml.safe_load(fh)["name"]
        if site_name != default_name:
            raise RuntimeError(
                f"ROBOTO_SITE={site_name!r} has no config: expected "
                f"{site_yaml}. Available: "
                f"{sorted(os.path.splitext(f)[0] for f in os.listdir(os.path.join(repo, 'tools', 'gis_pipeline', 'sites')))}"
                f" plus {default_name!r}.")
        site_yaml = default_yaml

    # The goal belongs to the site, not to the launch file: (95, 110) is a
    # drivable point in Pittsburgh and could be inside a building anywhere
    # else.
    with open(site_yaml, "r", encoding="utf-8") as fh:
        site_cfg = yaml.safe_load(fh)
    goal_default = site_cfg.get("mission", {}).get("goal", [95.0, 110.0])
    prior_yaml = os.path.join(data, "derived", "prior_occ.yaml")
    prior_cost = os.path.join(data, "derived", "prior_cost.npz")
    dem_npz = os.path.join(data, "derived", "dem_local.npz")

    models = os.pathsep.join([
        os.path.join(data, "world", "models"),
        os.path.join(repo, "ros2_ws", "src", "roboto_ros", "models"),
    ])

    gui = LaunchConfiguration("gui")
    rviz = LaunchConfiguration("rviz")

    return LaunchDescription([
        DeclareLaunchArgument("gui", default_value="true",
                              description="run the Gazebo GUI"),
        DeclareLaunchArgument("rviz", default_value="true",
                              description="run RViz"),
        DeclareLaunchArgument("goal_x", default_value=str(goal_default[0]),
                              description="goal easting, metres"),
        DeclareLaunchArgument("goal_y", default_value=str(goal_default[1]),
                              description="goal northing, metres"),
        DeclareLaunchArgument(
            "objective", default_value="balanced",
            description="what the planner minimises: "
                        "distance | balanced | time | effort | safe"),
        DeclareLaunchArgument(
            "planner", default_value="astar",
            description="astar (grid search) | rrt (sampling)"),
        DeclareLaunchArgument(
            "rrt_samples", default_value="6000",
            description="RRT* sample budget; more is slower and better"),
        DeclareLaunchArgument(
            "max_speed", default_value="0.0",
            description="drive speed m/s; 0 uses the robot's own limit. "
                        "Raise it to watch a long mission in less time"),
        DeclareLaunchArgument(
            "replan_mode", default_value="global",
            description="global: replan to the goal. "
                        "detour: route around the obstacle and rejoin"),
        DeclareLaunchArgument(
            "heuristic_weight", default_value="1.0",
            description="0 = Dijkstra, 1 = A*, >1 = weighted A*"),
        DeclareLaunchArgument(
            "emergency_m", default_value="20.0",
            description="a blockage within this range replans at once, "
                        "ignoring the cooldown. Must exceed the ~14 m a "
                        "reaction costs (9 m of travel during the search "
                        "plus 5.2 m to confirm and brake) or the replan "
                        "cannot finish before the robot arrives"),
        DeclareLaunchArgument(
            "plan_margin_m", default_value="1.0",
            description="extra clearance the PLANNER keeps beyond the robot "
                        "radius. Measured minimum clearance on the driven "
                        "path: balanced 1.5-5.3 m, safe 2.2-8.3 m, but "
                        "distance only 1.14-1.20 m on every site -- raise "
                        "this to about 1.5 when running objective:=distance"),
        DeclareLaunchArgument(
            "min_area_m2", default_value="0.2",
            description="smallest detected cluster treated as an obstacle. "
                        "A 2D lidar sees only a one-cell-deep near face, so "
                        "a 3.1 m barrier yields 0.31 m2 -- the old 0.5 floor "
                        "discarded it and the robot drove into it"),
        DeclareLaunchArgument(
            "max_correction_m", default_value="0.3",
            description="largest scan-match correction the filter accepts. "
                        "0.3 wins the six-seed offline sweep on every site. "
                        "Pass 3.0 if a run diverges after a drift excursion "
                        "-- one live mission cascaded to 126 m at 0.3 and "
                        "held 0.34 m at 3.0"),

        SetEnvironmentVariable("IGN_GAZEBO_RESOURCE_PATH", models),
        SetEnvironmentVariable("LIBGL_ALWAYS_SOFTWARE", "1"),
        SetEnvironmentVariable("GALLIUM_DRIVER", "llvmpipe"),

        # The window is decided HERE, by gui:=, and nowhere else.
        #
        # setup/gazebo_env.sh exports IGN_HEADLESS_RENDERING=1, which is
        # right for headless runs and silently fatal for a GUI one: the
        # window never appears even though `ign gazebo gui` starts, loads
        # its QML and reports no error. Sourcing that script before
        # launching -- an obvious thing to do, since it is what carries the
        # software-rendering settings -- therefore produced a demo with no
        # visible window and nothing in the log to say why.
        #
        # The two settings that script exists for are already set above, so
        # the launch does not need it. Pinning this variable to match `gui`
        # makes the launch authoritative regardless of what the caller
        # happened to source.
        SetEnvironmentVariable("IGN_HEADLESS_RENDERING", "0",
                               condition=IfCondition(gui)),
        SetEnvironmentVariable("IGN_HEADLESS_RENDERING", "1",
                               condition=UnlessCondition(gui)),

        ExecuteProcess(
            cmd=["ign", "gazebo", "-r", "-v", "2", world],
            output="screen", condition=IfCondition(gui)),
        ExecuteProcess(
            cmd=["ign", "gazebo", "-s", "-r", "-v", "2",
                 "--headless-rendering", world],
            output="screen", condition=UnlessCondition(gui)),

        # '[' is Gazebo -> ROS, ']' is ROS -> Gazebo.
        Node(
            package="ros_gz_bridge", executable="parameter_bridge",
            name="gz_bridge", output="screen",
            arguments=[
                "/clock@rosgraph_msgs/msg/Clock[ignition.msgs.Clock",
                "/scan@sensor_msgs/msg/LaserScan[ignition.msgs.LaserScan",
                "/odom@nav_msgs/msg/Odometry[ignition.msgs.Odometry",
                "/tf@tf2_msgs/msg/TFMessage[ignition.msgs.Pose_V",
                "/cmd_vel@geometry_msgs/msg/Twist]ignition.msgs.Twist",
                # Ground truth from the PosePublisher plugin. Diagnostics
                # only -- nothing in the estimation path may consume it, and
                # the node treats it as optional so the same code runs on a
                # robot that has no such topic. Without it, live SLAM error
                # is not measurable at all: the previous debugging session
                # had to compare against dead reckoning, which turned out to
                # be drifting worse than SLAM was.
                # The WORLD pose topic, not /model/roboto_bot/pose. The
                # model-scoped topic has a Pose_V publisher but its entries
                # carry no usable name, so both the TFMessage and PoseArray
                # conversions come out empty -- at 50 Hz, which looks like a
                # working feed until you read one. The world topic names
                # every model, so the conversion can set child_frame_id and
                # the robot can be picked out by name.
                f"/world/{site_name}/pose/info"
                "@tf2_msgs/msg/TFMessage[ignition.msgs.Pose_V",
            ],
            parameters=[{"use_sim_time": True}]),

        Node(
            package="roboto_ros", executable="roboto_node",
            name="roboto_node", output="screen",
            parameters=[{
                "use_sim_time": True,
                "site_yaml": site_yaml,
                "prior_yaml": prior_yaml,
                "prior_cost_npz": prior_cost,
                # Numpy copy of the DEM, so the robot can weight
                # slope without rasterio. See roboto_core/plan/terrain.py.
                "dem_npz": dem_npz,
                # goal_x/goal_y were declared as launch arguments but read
                # from the environment instead, so `goal_x:=` silently did
                # nothing. A LaunchConfiguration substitutes as a STRING, so
                # numeric parameters need an explicit value_type or the node
                # rejects them as the wrong type.
                "goal_x": ParameterValue(LaunchConfiguration("goal_x"),
                                         value_type=float),
                "goal_y": ParameterValue(LaunchConfiguration("goal_y"),
                                         value_type=float),
                "objective": ParameterValue(LaunchConfiguration("objective"),
                                            value_type=str),
                "replan_mode": ParameterValue(
                    LaunchConfiguration("replan_mode"), value_type=str),
                "planner": ParameterValue(LaunchConfiguration("planner"),
                                          value_type=str),
                "rrt_samples": ParameterValue(
                    LaunchConfiguration("rrt_samples"), value_type=int),
                "max_speed": ParameterValue(LaunchConfiguration("max_speed"),
                                            value_type=float),
                "max_correction_m": ParameterValue(
                    LaunchConfiguration("max_correction_m"), value_type=float),
                "min_area_m2": ParameterValue(
                    LaunchConfiguration("min_area_m2"), value_type=float),
                "plan_margin_m": ParameterValue(
                    LaunchConfiguration("plan_margin_m"), value_type=float),
                "emergency_m": ParameterValue(
                    LaunchConfiguration("emergency_m"), value_type=float),
                "heuristic_weight": ParameterValue(
                    LaunchConfiguration("heuristic_weight"), value_type=float),
                # Diagnostics only. The node defaults to "" -- no ground
                # truth, as on a real robot -- and the simulator supplies it
                # here so live error is measurable.
                "truth_topic": f"/world/{site_name}/pose/info",
            }]),

        Node(
            package="rviz2", executable="rviz2", name="rviz2", output="log",
            arguments=["-d", os.path.join(share, "config", "roboto.rviz")],
            parameters=[{"use_sim_time": True}],
            condition=IfCondition(rviz)),
    ])
