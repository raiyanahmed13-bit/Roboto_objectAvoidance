"""Correlative scan matching.

Gate C in the project plan: the matcher must recover known perturbations to
better than 3 cm and 0.5 deg across a sweep of offsets and noise levels.
These tests are the gate.

They run entirely offline against the raycast simulator -- no Gazebo, no
ROS -- which is what makes it practical to iterate on the matcher at all.
"""

import numpy as np
import pytest

from roboto_core.frames import GridSpec
from roboto_core.sim.raycast_sim import Lidar2D, LidarSpec, RaycastWorld
from roboto_core.slam.likelihood_field import LikelihoodField
from roboto_core.slam.scan_matcher import CorrelativeScanMatcher
from roboto_core.slam.transforms2d import transform_points, wrap_angle


@pytest.fixture(scope="module")
def room():
    """24 m x 24 m room with walls and two offset pillars.

    Asymmetric on purpose: a bare rectangular room is translation-ambiguous
    along neither axis but rotation-ambiguous by 180 deg, and a symmetric
    scene would let a broken matcher pass by luck.
    """
    g = GridSpec(origin_x=-12.0, origin_y=-12.0, resolution=0.05, width=480, height=480)
    occ = np.zeros(g.shape, dtype=bool)
    occ[0, :] = occ[-1, :] = True
    occ[:, 0] = occ[:, -1] = True
    occ[300:340, 320:360] = True      # pillar, NE
    occ[120:150, 260:290] = True      # pillar, SE, different size
    return RaycastWorld(occ, g)


@pytest.fixture(scope="module")
def field(room):
    return LikelihoodField.from_occupancy(room.occ, room.grid, sigma_m=0.15)


@pytest.fixture(scope="module")
def lidar(room):
    return Lidar2D(room, LidarSpec(num_beams=180, fov_deg=270.0, range_max=30.0,
                                   range_noise_std=0.0), rng=0)


TRUTH = (1.0, -2.0, 0.3)


def _err(pose, truth=TRUTH):
    return (float(np.hypot(pose[0] - truth[0], pose[1] - truth[1])),
            float(abs(np.degrees(wrap_angle(pose[2] - truth[2])))))


# ---------------------------------------------------------------------------
# The field itself
# ---------------------------------------------------------------------------

def test_truth_is_the_optimum(field, lidar):
    """The property whose absence caused a real bug.

    Blurring occupancy instead of using a distance transform put the field
    maximum inside walls, so the true pose was not even a local optimum
    (0.498 at truth vs 0.699 half a metre away).
    """
    pts = lidar.scan(TRUTH, noise=False).points()
    s_truth = field.score_points(transform_points(TRUTH, pts))
    assert s_truth > 0.85, f"a noiseless scan should score near 1, got {s_truth:.3f}"

    rng = np.random.default_rng(0)
    for _ in range(15):
        off = np.array([rng.uniform(-1, 1), rng.uniform(-1, 1), rng.uniform(-0.3, 0.3)])
        bad = np.array(TRUTH) + off
        if np.hypot(off[0], off[1]) < 0.1:
            continue
        assert field.score_points(transform_points(bad, pts)) <= s_truth + 1e-6


def test_field_is_one_inside_obstacles(room, field):
    """Distance-transform semantics: obstacles score exactly 1."""
    rows, cols = np.nonzero(room.occ)
    idx = np.linspace(0, len(rows) - 1, 50).astype(int)
    vals = field.field[rows[idx], cols[idx]]
    assert np.allclose(vals, 1.0, atol=1e-5)


def test_field_decays_with_distance(field, room):
    """Far from any obstacle the field must fall off, or search is flat."""
    r, c = room.grid.world_to_cell(0.0, 0.0)      # room centre, far from walls
    assert field.field[int(r), int(c)] < 0.01


# ---------------------------------------------------------------------------
# Recovery -- Gate C
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("dx,dy,dth_deg", [
    (0.0, 0.0, 0.0),
    (0.3, -0.2, 3.0),
    (-0.4, 0.3, -5.0),
    (0.8, 0.6, -9.0),
    (1.2, -1.0, 14.0),
    (-0.9, 0.7, 17.0),
])
def test_recovers_perturbation(field, lidar, dx, dy, dth_deg):
    pts = lidar.scan(TRUTH, noise=False).points()
    init = (TRUTH[0] + dx, TRUTH[1] + dy, TRUTH[2] + np.radians(dth_deg))
    r = CorrelativeScanMatcher(field).match(pts, init)

    pos, ang = _err(r.pose)
    assert pos < 0.03, f"position error {pos * 100:.1f} cm from ({dx}, {dy}, {dth_deg})"
    assert ang < 0.5, f"heading error {ang:.2f} deg from ({dx}, {dy}, {dth_deg})"


@pytest.mark.parametrize("noise", [0.01, 0.03, 0.05])
def test_recovers_under_sensor_noise(room, field, noise):
    """Accuracy must degrade gracefully, not collapse, with range noise."""
    lid = Lidar2D(room, LidarSpec(num_beams=180, fov_deg=270.0, range_max=30.0,
                                  range_noise_std=noise), rng=3)
    pts = lid.scan(TRUTH).points()
    init = (TRUTH[0] + 0.5, TRUTH[1] - 0.4, TRUTH[2] + np.radians(8.0))
    r = CorrelativeScanMatcher(field).match(pts, init)

    pos, ang = _err(r.pose)
    assert pos < 0.10, f"position error {pos * 100:.1f} cm at sigma={noise}"
    assert ang < 1.0, f"heading error {ang:.2f} deg at sigma={noise}"


def test_converges_to_same_answer_from_anywhere(field, lidar):
    """A global search should be insensitive to initialisation.

    This is the property that distinguishes correlative matching from ICP,
    and the reason it was chosen -- so it is worth asserting directly.
    """
    pts = lidar.scan(TRUTH, noise=False).points()
    m = CorrelativeScanMatcher(field)
    poses = [m.match(pts, (TRUTH[0] + dx, TRUTH[1] + dy,
                           TRUTH[2] + np.radians(dth))).pose
             for dx, dy, dth in [(0, 0, 0), (1.0, -0.8, 12.0), (-1.1, 0.9, -15.0)]]
    spread = np.ptp(np.array(poses)[:, :2], axis=0)
    assert spread.max() < 0.06, f"answers differ by {spread} m across initialisations"


# ---------------------------------------------------------------------------
# Covariance and diagnostics
# ---------------------------------------------------------------------------

def test_covariance_is_positive_definite(field, lidar):
    pts = lidar.scan(TRUTH, noise=False).points()
    r = CorrelativeScanMatcher(field).match(pts, TRUTH)
    assert np.all(np.linalg.eigvals(r.covariance) > 0)
    assert np.allclose(r.covariance, r.covariance.T)


def test_information_matrix_is_usable(field, lidar):
    """Pose-graph edges need the inverse covariance to be finite."""
    pts = lidar.scan(TRUTH, noise=False).points()
    r = CorrelativeScanMatcher(field).match(pts, TRUTH)
    assert np.all(np.isfinite(r.information))


def test_corridor_covariance_is_anisotropic():
    """A corridor is under-constrained along its axis, and the covariance
    must say so -- otherwise the pose graph trusts a slid pose completely."""
    g = GridSpec(origin_x=-20.0, origin_y=-4.0, resolution=0.05, width=800, height=160)
    occ = np.zeros(g.shape, dtype=bool)
    occ[0, :] = occ[-1, :] = True                 # two long parallel walls only
    world = RaycastWorld(occ, g)
    fld = LikelihoodField.from_occupancy(occ, g, sigma_m=0.15)
    lid = Lidar2D(world, LidarSpec(num_beams=180, fov_deg=270.0, range_max=30.0,
                                   range_noise_std=0.0), rng=0)

    pose = (0.0, 0.0, 0.0)
    r = CorrelativeScanMatcher(fld).match(lid.scan(pose, noise=False).points(), pose)
    sx, sy = np.sqrt(r.covariance[0, 0]), np.sqrt(r.covariance[1, 1])
    assert sx > sy * 2, (
        f"along-corridor sigma {sx:.3f} m should greatly exceed across-corridor "
        f"{sy:.3f} m; an isotropic covariance here would be a lie"
    )


def test_low_score_signals_a_bad_match(field, lidar):
    """SCORE is the primary reliability signal, not the edge flag.

    Given a hopeless initialisation the score surface is essentially noise,
    so its argmax can land anywhere -- including the window interior, with
    hit_window_edge False. What reliably distinguishes a bad match is that
    the score collapses (~0.04 here versus ~0.93 for a good one), and that
    is what callers must gate on.
    """
    pts = lidar.scan(TRUTH, noise=False).points()
    good = CorrelativeScanMatcher(field).match(pts, TRUTH)
    far = (TRUTH[0] + 6.0, TRUTH[1] + 5.0, TRUTH[2] + np.radians(40.0))
    bad = CorrelativeScanMatcher(field).match(pts, far)

    assert bad.score < 0.3, f"hopeless match scored {bad.score:.3f}"
    assert good.score > 3 * bad.score


def test_window_edge_is_reported(room, field, lidar):
    """When the optimum really is clipped by the window, say so.

    Constructed by shrinking the window so the true pose lies outside it:
    the score then increases all the way to the boundary, and the argmax
    sits on it.
    """
    pts = lidar.scan(TRUTH, noise=False).points()
    m = CorrelativeScanMatcher(
        field,
        coarse_win_xy=0.2, coarse_win_th=np.radians(2.0),
        fine_win_xy=0.1, fine_win_th=np.radians(1.0),
    )
    r = m.match(pts, (TRUTH[0] + 1.5, TRUTH[1] + 1.2, TRUTH[2]))
    assert r.hit_window_edge, "optimum was clipped by the window but not flagged"


def test_empty_scan_is_handled(field):
    """Beam-starved poses happen in open areas; they must not crash."""
    r = CorrelativeScanMatcher(field).match(np.zeros((0, 2)), TRUTH)
    assert r.n_points == 0
    assert r.score == 0.0
    assert np.all(np.isfinite(r.covariance))
