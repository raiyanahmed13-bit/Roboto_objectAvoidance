"""The robot SDF must agree with site.yaml.

The simulator reads its sensor parameters from the SDF; the planner, the
raycast simulator, and the SLAM front end read theirs from site.yaml. If
those two drift apart, everything still runs and every number is quietly
wrong -- the offline simulator would be modelling a different sensor from
the one Gazebo provides, and no test would notice.

Duplicating the values is a deliberate trade (SDF cannot read YAML), so
this file is the mechanism that makes the duplication safe.
"""

import math
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

from roboto_core.sim.odom_corruptor import OdomErrorModel
from tools.gis_pipeline.common.crs import load_site

SDF = Path("ros2_ws/src/roboto_ros/models/roboto_bot/model.sdf")


@pytest.fixture(scope="module")
def root():
    if not SDF.exists():
        pytest.skip(f"{SDF} not present")
    return ET.parse(SDF).getroot()


@pytest.fixture(scope="module")
def lidar(root):
    node = root.find(".//sensor[@type='gpu_lidar']")
    assert node is not None, "robot has no gpu_lidar sensor"
    return node


@pytest.fixture(scope="module")
def site():
    return load_site()


def _f(node, path):
    el = node.find(path)
    assert el is not None, f"missing element: {path}"
    return float(el.text)


# ---------------------------------------------------------------------------
# Sensor agreement
# ---------------------------------------------------------------------------

def test_beam_count_matches_site(lidar, site):
    assert int(_f(lidar, ".//horizontal/samples")) == site.raw["lidar"]["num_beams"]


def test_field_of_view_matches_site(lidar, site):
    lo = _f(lidar, ".//horizontal/min_angle")
    hi = _f(lidar, ".//horizontal/max_angle")
    assert math.degrees(hi - lo) == pytest.approx(site.raw["lidar"]["fov_deg"], abs=0.01)
    assert lo == pytest.approx(-hi, abs=1e-6), "FOV must be symmetric about straight ahead"


def test_range_limits_match_site(lidar, site):
    cfg = site.raw["lidar"]
    assert _f(lidar, ".//range/min") == pytest.approx(cfg["range_min_m"])
    assert _f(lidar, ".//range/max") == pytest.approx(cfg["range_max_m"])


def test_update_rate_matches_site(lidar, site):
    assert _f(lidar, "update_rate") == pytest.approx(site.raw["lidar"]["rate_hz"])


def test_noise_matches_site(lidar, site):
    assert _f(lidar, ".//noise/stddev") == pytest.approx(
        site.raw["lidar"]["range_noise_std_m"])


def test_range_is_long_enough_for_this_site(lidar, site):
    """Guards the specific failure that cost us a debugging cycle.

    At 12 m the robot returned 1/180 beams at the spawn pose, because this
    site's median free-cell clearance is 8.5 m and its p90 is 22 m.
    """
    assert _f(lidar, ".//range/max") >= 25.0, (
        "a short-range lidar leaves the robot blind in open areas here; "
        "see the note in site.yaml"
    )


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------

def test_lidar_is_clear_of_the_chassis(root, lidar):
    """A sensor inside its own collision box returns range_min on every beam.

    This is exactly what happened in the Gate A diagnostic world, where it
    looked like a renderer failure for two rounds of debugging.
    """
    pose = [float(v) for v in lidar.find("pose").text.split()]
    box = root.find(".//link[@name='base_link']/collision/geometry/box/size")
    sx, sy, sz = (float(v) for v in box.text.split())

    outside = (abs(pose[0]) > sx / 2) or (abs(pose[1]) > sy / 2) or (abs(pose[2]) > sz / 2)
    assert outside, (
        f"lidar at {pose[:3]} is inside the {sx}x{sy}x{sz} chassis box; "
        f"every beam will return range_min"
    )


def test_wheel_separation_matches_odom_model(root):
    """The corruptor's kinematics must describe the robot being simulated."""
    sep = float(root.find(".//plugin/wheel_separation").text)
    assert sep == pytest.approx(OdomErrorModel().wheel_baseline_m), (
        "DiffDrive wheel_separation and OdomErrorModel.wheel_baseline_m "
        "describe the same physical robot and must agree"
    )


def test_wheels_are_placed_at_the_declared_separation(root):
    left = [float(v) for v in root.find(".//link[@name='wheel_left']/pose").text.split()]
    right = [float(v) for v in root.find(".//link[@name='wheel_right']/pose").text.split()]
    sep = float(root.find(".//plugin/wheel_separation").text)
    assert abs(left[1] - right[1]) == pytest.approx(sep, abs=1e-6)


def test_robot_rests_on_its_wheels(root):
    """Wheel radius must equal axle height, or the robot spawns clipping the
    ground and physics launches it."""
    radius = float(root.find(".//link[@name='wheel_left']/collision/geometry/"
                             "cylinder/radius").text)
    axle_z = float(root.find(".//link[@name='wheel_left']/pose").text.split()[2])
    assert radius == pytest.approx(axle_z, abs=1e-6)


def test_centre_of_mass_is_inside_the_support_polygon(root):
    """The robot must not be able to sit back on its tail.

    Regression for a live mission failure. The wheels are the rear support
    and the caster the front, so the centre of mass has to lie BETWEEN
    them. It sat exactly over the axle, which meant the caster carried no
    load and nothing at all supported the robot behind the axle: it rocked
    back and settled nose-up by atan2(0.05, 0.25) = 11.3 degrees, measured
    live at 11.8.

    That tilts the lidar with the body. At 30 m range an 11.8 degree tilt
    puts the scan plane 6 m off horizontal, so the beams sail over the
    walls the scan matcher needs and 2D SLAM diverges -- which is exactly
    what happened, twice, at the same point on the route.
    """
    base = root.find(".//link[@name='base_link']")
    body_x = float(base.find("pose").text.split()[0])
    inertial_pose = base.find("inertial/pose")
    com_x = body_x + (0.0 if inertial_pose is None
                      else float(inertial_pose.text.split()[0]))

    axle_x = float(root.find(".//link[@name='wheel_left']/pose").text.split()[0])
    caster_x = float(root.find(".//link[@name='caster']/pose").text.split()[0])

    assert axle_x < com_x < caster_x, (
        f"centre of mass at x={com_x:.3f} is not between the axle "
        f"({axle_x:.3f}) and the caster ({caster_x:.3f}); the robot will "
        f"tip and take the lidar out of plane with it"
    )


def test_pitch_stability_margin_covers_the_drive_acceleration(root):
    """Static balance is not enough -- it must survive braking and
    accelerating at the limit the drive plugin allows.

    The robot stays down while g * d > a * h, for COM offset d ahead of the
    axle and COM height h.
    """
    base = root.find(".//link[@name='base_link']")
    body = [float(v) for v in base.find("pose").text.split()]
    inertial_pose = base.find("inertial/pose")
    off_x = 0.0 if inertial_pose is None else float(inertial_pose.text.split()[0])

    d = (body[0] + off_x) - float(
        root.find(".//link[@name='wheel_left']/pose").text.split()[0])
    h = body[2]
    a_max = float(root.find(".//plugin/max_linear_acceleration").text)

    tolerable = 9.81 * d / h
    assert tolerable > a_max, (
        f"tips at {tolerable:.2f} m/s^2 but the drive is allowed "
        f"{a_max:.2f} m/s^2"
    )


def test_required_plugins_present(root):
    names = {p.get("name") for p in root.findall(".//plugin")}
    for needed in (
        "ignition::gazebo::systems::DiffDrive",
        "ignition::gazebo::systems::PosePublisher",   # ground truth for ATE
    ):
        assert needed in names, f"missing plugin {needed}"


def test_inertias_are_physically_plausible(root):
    """A zero or negative principal inertia makes the solver explode."""
    for link in root.findall(".//link"):
        inertia = link.find("inertial/inertia")
        if inertia is None:
            continue
        for axis in ("ixx", "iyy", "izz"):
            val = float(inertia.find(axis).text)
            assert val > 0, f"{link.get('name')}.{axis} = {val}"
