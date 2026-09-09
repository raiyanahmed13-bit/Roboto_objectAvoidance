"""Recovery from a lost matching lock.

Reproduces, offline and in seconds, the failure that ended a live mission:
at 154 m into a run the scan matcher stopped accepting matches during a
~120 degree turnaround, `matched` froze, and the estimate fell back onto
odometry for the remaining 570 m. Nothing could bring it back, because the
local likelihood field is rebuilt around the estimate -- so the further the
estimate drifts, the less chance any match has of scoring. Final error was
134 m.

The test is a kidnapped-robot problem: displace the estimate far enough
that local matching cannot recover, then assert that SLAM finds itself
again.
"""

import numpy as np
import pytest

from roboto_core.frames import GridSpec
from roboto_core.sim.raycast_sim import Lidar2D, LidarSpec, RaycastWorld
from roboto_core.slam.slam import Slam2D, SlamConfig


@pytest.fixture
def world():
    """A 60 m square scattered with blocks, like a coarse city block.

    Deliberately irregular: blocks of differing size at irregular spacing,
    so that every point along the test path sees a locally unique
    arrangement. A regular or sparse layout makes the scene self-similar,
    and then a matcher can legitimately prefer the wrong place -- which
    tests the fixture rather than the code.
    """
    g = GridSpec(origin_x=-30.0, origin_y=-30.0, resolution=0.10,
                 width=600, height=600)
    occ = np.zeros(g.shape, dtype=bool)
    occ[0, :] = occ[-1, :] = True
    occ[:, 0] = occ[:, -1] = True
    for r0, r1, c0, c1 in [
        (100, 150, 60, 170),
        (110, 145, 230, 300),
        (95, 160, 360, 420),
        (105, 155, 470, 560),
        (250, 330, 40, 130),
        (270, 310, 200, 340),
        (240, 350, 420, 500),
        (430, 500, 90, 210),
        (450, 480, 300, 380),
        (420, 520, 470, 580),
    ]:
        occ[r0:r1, c0:c1] = True
    return RaycastWorld(occ, g), g


@pytest.fixture
def spec():
    return LidarSpec(num_beams=180, fov_deg=270.0, range_max=25.0,
                     range_noise_std=0.01)


def _straight_run(n=120, x0=-25.0, y=-25.0, step=0.25):
    """Along the open strip south of the first row of blocks."""
    return np.column_stack([x0 + np.arange(n) * step,
                            np.full(n, y), np.zeros(n)])


def test_recovers_from_a_kidnap(world, spec):
    """The regression test for the live failure.

    Displace the odometry stream by 6 m and 40 degrees partway through --
    far outside the matcher's normal search window, so local matching
    cannot climb back on its own -- and require that SLAM re-acquires.
    """
    w, g = world
    traj = _straight_run()
    lidar = Lidar2D(w, spec, rng=0)

    cfg = SlamConfig(relocalise_after=5, relocalise_win_m=10.0,
                     relocalise_min_matches=40)
    slam = Slam2D(g, sensor_range=spec.range_max, config=cfg,
                  initial_pose=traj[0], prior_occ=w.occ)

    kidnap = np.array([6.0, -4.0, np.radians(40.0)])
    for i, p in enumerate(traj):
        scan = lidar.scan(p)
        odom = p if i < 60 else p + kidnap      # odometry jumps and stays off
        slam.update(scan, odom)

    err = float(np.hypot(slam.pose[0] - traj[-1][0], slam.pose[1] - traj[-1][1]))
    assert slam.n_relocalised >= 1, "never attempted to recover"
    assert err < 2.0, f"did not recover: {err:.2f} m from truth"


def test_no_relocalisation_when_tracking_is_healthy(world, spec):
    """Recovery must not fire on a run that is going fine -- it is an
    expensive wide search, and teleporting a healthy estimate would be far
    worse than the problem it solves."""
    w, g = world
    traj = _straight_run()
    lidar = Lidar2D(w, spec, rng=1)

    slam = Slam2D(g, sensor_range=spec.range_max, config=SlamConfig(),
                  initial_pose=traj[0], prior_occ=w.occ)
    for p in traj:
        slam.update(lidar.scan(p), p)

    assert slam.n_relocalised == 0
    err = float(np.hypot(slam.pose[0] - traj[-1][0], slam.pose[1] - traj[-1][1]))
    assert err < 1.0


def test_relocalisation_is_reported(world, spec):
    """A recovery is a significant event and must be visible in the summary
    and on the step, not silently patched over."""
    w, g = world
    traj = _straight_run()
    lidar = Lidar2D(w, spec, rng=0)

    cfg = SlamConfig(relocalise_after=5, relocalise_min_matches=40)
    slam = Slam2D(g, sensor_range=spec.range_max, config=cfg,
                  initial_pose=traj[0], prior_occ=w.occ)

    kidnap = np.array([6.0, -4.0, np.radians(40.0)])
    for i, p in enumerate(traj):
        slam.update(lidar.scan(p), p if i < 60 else p + kidnap)

    assert slam.summary()["relocalised"] == slam.n_relocalised
    assert any(s.relocalised for s in slam.steps)
    # A relocalised step counts as matched, not rejected: the estimate was
    # corrected, which is what those counters mean.
    reloc = [s for s in slam.steps if s.relocalised]
    assert all(s.matched and not s.rejected for s in reloc)


def test_map_only_slam_does_not_attempt_recovery(world, spec):
    """Recovery matches against the prior. Without one there is nothing
    trustworthy to relocalise against, and it must not fall back to the
    self-built map -- agreeing with a map carved at wrong poses is how the
    estimate got lost in the first place."""
    w, g = world
    traj = _straight_run()
    lidar = Lidar2D(w, spec, rng=0)

    cfg = SlamConfig(reference="map", relocalise_after=5,
                     relocalise_min_matches=40)
    slam = Slam2D(g, sensor_range=spec.range_max, config=cfg,
                  initial_pose=traj[0])
    kidnap = np.array([6.0, -4.0, np.radians(40.0)])
    for i, p in enumerate(traj):
        slam.update(lidar.scan(p), p if i < 60 else p + kidnap)

    assert slam.n_relocalised == 0


def test_does_not_relocalise_before_it_has_ever_had_a_lock(world, spec):
    """Regression for a false recovery observed live.

    A mission opens with the map nearly empty and matches refused -- 19
    consecutive rejections before the first acceptance. A recovery
    triggered inside that bootstrap teleported the estimate 13.5 m and 127
    degrees away while odometry was still good to 8 cm, because the robot
    had never had a lock to lose. The aliased pose then scored 0.78-0.84,
    so the score alone could not have caught it.
    """
    w, g = world
    traj = _straight_run(n=30)
    lidar = Lidar2D(w, spec, rng=0)

    # Eager trigger, but the min-matches guard must still hold it off.
    cfg = SlamConfig(relocalise_after=1, relocalise_min_matches=40)
    slam = Slam2D(g, sensor_range=spec.range_max, config=cfg,
                  initial_pose=traj[0], prior_occ=w.occ)
    for p in traj:
        slam.update(lidar.scan(p), p)

    assert slam.n_matched < cfg.relocalise_min_matches, "premise: no lock yet"
    assert slam.n_relocalised == 0
    err = float(np.hypot(slam.pose[0] - traj[-1][0], slam.pose[1] - traj[-1][1]))
    assert err < 1.0, f"estimate was thrown off during bootstrap: {err:.2f} m"


def test_recovery_must_beat_the_match_it_replaces(world, spec):
    """A recovery scoring no better than the match just refused is not
    evidence of anything, and acting on it is how the false recovery
    above happened."""
    w, g = world
    traj = _straight_run()
    lidar = Lidar2D(w, spec, rng=0)

    # An impossible margin: nothing can ever clear it, so no recovery may
    # be accepted however lost the robot gets.
    cfg = SlamConfig(relocalise_after=5, relocalise_min_matches=40,
                     relocalise_margin=10.0)
    slam = Slam2D(g, sensor_range=spec.range_max, config=cfg,
                  initial_pose=traj[0], prior_occ=w.occ)
    kidnap = np.array([6.0, -4.0, np.radians(40.0)])
    for i, p in enumerate(traj):
        slam.update(lidar.scan(p), p if i < 60 else p + kidnap)

    assert slam.n_relocalised == 0


def test_consecutive_rejections_reset_on_a_good_match(world, spec):
    """The counter must measure a RUN of failures. If isolated rejections
    accumulated across an otherwise healthy mission, the wide search would
    eventually fire on a perfectly good estimate."""
    w, g = world
    traj = _straight_run(n=60)
    lidar = Lidar2D(w, spec, rng=2)

    cfg = SlamConfig(relocalise_after=3, relocalise_min_matches=10)
    slam = Slam2D(g, sensor_range=spec.range_max, config=cfg,
                  initial_pose=traj[0], prior_occ=w.occ)
    for p in traj:
        slam.update(lidar.scan(p), p)

    # Healthy run: whatever isolated rejections occurred, the counter must
    # not have been left sitting at or above the threshold.
    assert slam._consecutive_rejects < cfg.relocalise_after
    assert slam.n_relocalised == 0
