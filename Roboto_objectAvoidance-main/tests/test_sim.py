"""Raycast simulator and odometry-corruptor behaviour.

These pin the two properties the rest of the project depends on:
  * the simulator returns geometrically correct ranges, and
  * odometry drifts by a realistic amount -- neither negligible (which
    would make SLAM look pointless) nor divergent (which would make scan
    matching untestable).
"""

import numpy as np
import pytest

from roboto_core.frames import GridSpec
from roboto_core.sim.odom_corruptor import OdomCorruptor, OdomErrorModel
from roboto_core.sim.raycast_sim import Lidar2D, LidarSpec, RaycastWorld
from roboto_core.slam.transforms2d import wrap_angle


@pytest.fixture
def boxed_world():
    """20 m x 20 m empty room with walls, at 0.05 m resolution.

    Known geometry: from the centre (0, 0) the walls are 10 m away on
    every side.
    """
    g = GridSpec(origin_x=-10.0, origin_y=-10.0, resolution=0.05, width=400, height=400)
    occ = np.zeros(g.shape, dtype=bool)
    occ[0, :] = occ[-1, :] = True
    occ[:, 0] = occ[:, -1] = True
    return RaycastWorld(occ, g)


# ---------------------------------------------------------------------------
# Raycasting
# ---------------------------------------------------------------------------

def test_ranges_match_known_walls(boxed_world):
    spec = LidarSpec(num_beams=4, fov_deg=360.0, range_max=30.0, range_noise_std=0.0)
    scan = Lidar2D(boxed_world, spec, rng=0).scan((0.0, 0.0, 0.0), noise=False)
    # Beams at 0, 90, 180, 270 deg -- every wall is 10 m from the centre.
    assert np.allclose(scan.ranges, 10.0, atol=0.1)


def test_range_scales_with_offset(boxed_world):
    """Move 3 m east: the east wall is 7 m away, the west wall 13 m.

    Expectations are derived from `spec.angles` rather than hardcoded by
    index -- with fov=360 and endpoint=False, beam 0 points WEST (-pi), and
    assuming otherwise silently inverts the test.
    """
    spec = LidarSpec(num_beams=2, fov_deg=360.0, range_max=30.0, range_noise_std=0.0)
    scan = Lidar2D(boxed_world, spec, rng=0).scan((3.0, 0.0, 0.0), noise=False)

    east = int(np.argmin(np.abs(np.cos(spec.angles) - 1.0)))   # bearing ~0
    west = int(np.argmin(np.cos(spec.angles)))                 # bearing ~pi
    assert scan.ranges[east] == pytest.approx(7.0, abs=0.1)
    assert scan.ranges[west] == pytest.approx(13.0, abs=0.1)


def test_no_return_beyond_max_range(boxed_world):
    spec = LidarSpec(num_beams=4, fov_deg=360.0, range_max=5.0, range_noise_std=0.0)
    scan = Lidar2D(boxed_world, spec, rng=0).scan((0.0, 0.0, 0.0), noise=False)
    assert not scan.valid.any(), "walls at 10 m must not appear with a 5 m sensor"
    assert np.isinf(scan.ranges).all()


def test_rotation_shifts_the_pattern(boxed_world):
    """Rotating the sensor must rotate which wall each beam sees."""
    spec = LidarSpec(num_beams=4, fov_deg=360.0, range_max=30.0, range_noise_std=0.0)
    lid = Lidar2D(boxed_world, spec, rng=0)
    a = lid.scan((3.0, 0.0, 0.0), noise=False).ranges
    b = lid.scan((3.0, 0.0, np.pi), noise=False).ranges
    # 4 beams over 360 deg, so a 180 deg turn shifts the pattern by 2 beams.
    assert np.allclose(a, np.roll(b, 2), atol=0.15)


def test_noise_is_seeded_and_bounded(boxed_world):
    spec = LidarSpec(num_beams=32, fov_deg=270.0, range_max=30.0, range_noise_std=0.02)
    s1 = Lidar2D(boxed_world, spec, rng=7).scan((0.0, 0.0, 0.0))
    s2 = Lidar2D(boxed_world, spec, rng=7).scan((0.0, 0.0, 0.0))
    assert np.allclose(s1.ranges, s2.ranges), "same seed must reproduce exactly"

    clean = Lidar2D(boxed_world, spec, rng=7).scan((0.0, 0.0, 0.0), noise=False)
    err = np.abs(s1.ranges - clean.ranges)
    assert err.max() < 0.2, "2 cm sigma should not produce 20 cm outliers"


def test_out_of_bounds_is_not_a_phantom_wall():
    """Rays leaving the grid must return inf, not a wall at the boundary."""
    g = GridSpec(origin_x=-5.0, origin_y=-5.0, resolution=0.1, width=100, height=100)
    world = RaycastWorld(np.zeros(g.shape, dtype=bool), g)
    spec = LidarSpec(num_beams=8, fov_deg=360.0, range_max=30.0, range_noise_std=0.0)
    scan = Lidar2D(world, spec, rng=0).scan((0.0, 0.0, 0.0), noise=False)
    assert np.isinf(scan.ranges).all()


# ---------------------------------------------------------------------------
# Odometry corruption
# ---------------------------------------------------------------------------

def _square_loop(side=75.0, step=0.1):
    """~300 m closed square loop of (x, y, theta) ground-truth poses."""
    poses, x, y, th = [], 0.0, 0.0, 0.0
    for _ in range(4):
        for _ in range(int(side / step)):
            x += step * np.cos(th)
            y += step * np.sin(th)
            poses.append((x, y, th))
        for _ in range(int((np.pi / 2) / 0.02)):
            th += 0.02
            poses.append((x, y, th))
    return np.array(poses)


def test_drift_is_realistic_over_a_300m_loop():
    """The calibration that makes the whole SLAM evaluation meaningful.

    Too little drift and SLAM has nothing to correct, so the headline
    "SLAM beats odometry" result vanishes. Too much and the scan matcher
    diverges instead of working, so nothing is being measured either.
    Target: a few percent of path length.
    """
    gt = _square_loop()
    path_len = float(np.sum(np.hypot(*np.diff(gt[:, :2], axis=0).T)))
    assert 280 < path_len < 320

    drifts = []
    for seed in range(8):
        odom = OdomCorruptor(seed=seed).corrupt_trajectory(gt, dt=0.1)
        drifts.append(float(np.hypot(*(odom[-1, :2] - gt[-1, :2]))))

    mean_drift = float(np.mean(drifts))
    assert 3.0 < mean_drift < 20.0, (
        f"mean drift {mean_drift:.1f} m over {path_len:.0f} m is outside the "
        f"usable band; see the calibration note in odom_corruptor.py"
    )


def test_drift_grows_with_distance():
    """Systematic error must accumulate, not average out."""
    gt = _square_loop()
    odom = OdomCorruptor(seed=0).corrupt_trajectory(gt, dt=0.1)
    err = np.hypot(*(odom[:, :2] - gt[:, :2]).T)
    early = err[:len(err) // 4].mean()
    late = err[-len(err) // 4:].mean()
    assert late > early * 2, "drift should accumulate over the run"


def test_seeded_and_reproducible():
    gt = _square_loop(side=20.0)
    a = OdomCorruptor(seed=3).corrupt_trajectory(gt)
    b = OdomCorruptor(seed=3).corrupt_trajectory(gt)
    c = OdomCorruptor(seed=4).corrupt_trajectory(gt)
    assert np.allclose(a, b), "same seed must reproduce exactly"
    assert not np.allclose(a, c), "different seeds must differ"


def test_starts_aligned_with_ground_truth():
    od = OdomCorruptor(seed=0)
    start = np.array([12.0, -3.0, 0.7])
    assert np.allclose(od.reset(start), start)


def test_zero_error_model_is_exact():
    """With all error terms off, odometry must reproduce ground truth.

    Guards the kinematics themselves: any bug in the wheel-arc round trip
    shows up here rather than being masked by injected noise.
    """
    gt = _square_loop(side=20.0)
    model = OdomErrorModel(
        wheel_scale_err=0.0, wheel_asymmetry=0.0, baseline_err=0.0,
        gyro_bias_dps=0.0, trans_noise=0.0, rot_noise=0.0,
    )
    odom = OdomCorruptor(model, seed=0).corrupt_trajectory(gt, dt=0.1)
    assert np.abs(odom[-1, :2] - gt[-1, :2]).max() < 1e-6
    assert abs(wrap_angle(odom[-1, 2] - gt[-1, 2])) < 1e-6
