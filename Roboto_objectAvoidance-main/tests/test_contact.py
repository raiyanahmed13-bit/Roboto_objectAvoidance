"""The controller must stop for what the map sees and the scan cannot.

Regression test for a live failure: the robot wedged against an obstacle at
(-1, 78), spun its heading from -25 to -170 degrees over 45 seconds, and
made 2.5 m of progress. Its lidar has a 0.15 m minimum range, so every
forward beam was invalid, _forward_clearance returned None, and the brake
declined to act -- the robot was most confident precisely when it was
already in contact.
"""

import numpy as np
import pytest

from roboto_core.plan.pursuit import PurePursuit, PursuitConfig


def straight_path(n=40, spacing=0.5):
    return np.stack([np.arange(n) * spacing, np.zeros(n)], axis=1)


def test_no_valid_returns_alone_does_not_brake():
    """The scan cannot distinguish 'nothing there' from 'touching it'.

    All-infinite ranges are what an open field looks like, so the scan on
    its own must not be read as contact -- which is exactly why the map has
    to supply the answer.
    """
    p = PurePursuit(PursuitConfig())
    cmd = p.step(np.array([0.0, 0.0, 0.0]), straight_path(),
                 scan_ranges=np.full(180, np.inf),
                 scan_angles=np.linspace(-2.35, 2.35, 180))
    assert cmd.v > 0.0
    assert cmd.reason == "tracking"


def test_contact_stops_forward_motion():
    """Contact must never produce FORWARD motion.

    Originally this asserted v == 0. That was too weak, and the weakness
    cost a run: a robot that merely stops is still touching the obstacle,
    and rotating does not change that. The durable requirement is that it
    must not drive further in.
    """
    p = PurePursuit(PursuitConfig())
    cmd = p.step(np.array([0.0, 0.0, 0.0]), straight_path(),
                 scan_ranges=np.full(180, np.inf),
                 scan_angles=np.linspace(-2.35, 2.35, 180),
                 contact=True)
    assert cmd.v <= 0.0
    assert cmd.reason == "backing off"


def test_contact_still_allows_turning():
    """A stopped robot must still be able to turn out of the wedge.

    Zero v and zero omega is the deadlock that produced the 45-second spin
    with no escape. Differential drive can rotate without translating, so
    rotating is always safe and is the only move left.
    """
    p = PurePursuit(PursuitConfig())
    path = np.stack([np.zeros(20), np.arange(20) * 0.5], axis=1)  # goal to the left
    cmd = p.step(np.array([0.0, 0.0, 0.0]), path,
                 scan_ranges=np.full(180, np.inf),
                 scan_angles=np.linspace(-2.35, 2.35, 180),
                 contact=True)
    assert cmd.v <= 0.0
    assert abs(cmd.omega) > 0.0


def test_contact_holds_when_the_scan_is_blind():
    """Contact must win where the sensor cannot see.

    Inside the 0.15 m minimum range every beam is invalid, so the scan has
    no opinion and the map is the only evidence there is. This is the
    wedged case the contact test exists for.
    """
    p = PurePursuit(PursuitConfig())
    cmd = p.step(np.array([0.0, 0.0, 0.0]), straight_path(),
                 scan_ranges=np.full(180, np.inf),    # nothing valid
                 scan_angles=np.linspace(-2.35, 2.35, 180),
                 contact=True)
    assert cmd.v <= 0.0
    assert cmd.reason == "backing off"


def test_a_seeing_scan_vetoes_a_stale_contact_claim():
    """A positive observation of clear space overrules the map.

    The map's contact claim is only trustworthy where the scan is blind.
    Live, SLAM drifted 3 m, mapped obstacles landed displaced and became
    phantom MISSED clusters -- obj reached 10 with six real obstacles --
    and the robot reversed in a loop in open ground, 3.6 m from the nearest
    truck and 15.5 m from any building. Reversing degraded odometry, which
    grew the error, which made more phantoms.
    """
    p = PurePursuit(PursuitConfig())
    cmd = p.step(np.array([0.0, 0.0, 0.0]), straight_path(),
                 scan_ranges=np.full(180, 25.0),      # 25 m of open road
                 scan_angles=np.linspace(-2.35, 2.35, 180),
                 contact=True)
    assert cmd.v > 0.0, "map overruled a scan that could see"
    assert cmd.reason == "tracking"


def test_a_near_return_does_not_veto_contact():
    """Only clear space vetoes. A close return leaves contact standing."""
    p = PurePursuit(PursuitConfig())
    ranges = np.full(180, 25.0)
    ranges[85:95] = 0.5                 # something genuinely close ahead
    cmd = p.step(np.array([0.0, 0.0, 0.0]), straight_path(),
                 scan_ranges=ranges,
                 scan_angles=np.linspace(-2.35, 2.35, 180),
                 contact=True)
    assert cmd.v <= 0.0


def test_normal_braking_still_works():
    """The scan-based brake must survive the change."""
    p = PurePursuit(PursuitConfig())
    ranges = np.full(180, 25.0)
    ranges[90] = 0.3                      # something 0.3 m dead ahead
    cmd = p.step(np.array([0.0, 0.0, 0.0]), straight_path(),
                 scan_ranges=ranges,
                 scan_angles=np.linspace(-2.35, 2.35, 180))
    assert cmd.v < PursuitConfig().max_speed
    assert cmd.reason in ("braking", "stopped")


@pytest.mark.parametrize("contact", [False, True])
def test_goal_still_wins(contact):
    """Arriving beats everything, contact included."""
    p = PurePursuit(PursuitConfig())
    path = np.array([[0.0, 0.0], [0.1, 0.0]])
    cmd = p.step(np.array([0.1, 0.0, 0.0]), path, contact=contact)
    assert cmd.done


def test_contact_reverses_out():
    """Contact must produce BACKWARD motion, not just rotation.

    Rotating clears a brake trigger, because the brake looks in a cone
    ahead and rotating moves the cone. It cannot clear contact: the
    footprint stays where it is. Live, that cost an entire run -- pinned
    against a truck, four stuck events, ten identical replans, and odometry
    gaining 15 m of motion that never happened while the wheels spun.
    """
    p = PurePursuit(PursuitConfig())
    cmd = p.step(np.array([0.0, 0.0, 0.0]), straight_path(),
                 scan_ranges=np.full(180, np.inf),
                 scan_angles=np.linspace(-2.35, 2.35, 180),
                 contact=True)
    assert cmd.v < 0.0, "contact must reverse, not stall"
    assert cmd.reason == "backing off"


def test_reverse_is_slow():
    """A recovery into space the robot cannot sense should be gentle."""
    cfg = PursuitConfig()
    p = PurePursuit(cfg)
    cmd = p.step(np.array([0.0, 0.0, 0.0]), straight_path(),
                 scan_ranges=np.full(180, np.inf),
                 scan_angles=np.linspace(-2.35, 2.35, 180),
                 contact=True)
    assert abs(cmd.v) <= cfg.reverse_speed + 1e-9
    assert abs(cmd.v) < cfg.max_speed


def test_no_contact_never_reverses():
    """Ordinary driving must stay forward-only."""
    p = PurePursuit(PursuitConfig())
    for ranges in (np.full(180, np.inf), np.full(180, 25.0)):
        cmd = p.step(np.array([0.0, 0.0, 0.0]), straight_path(),
                     scan_ranges=ranges,
                     scan_angles=np.linspace(-2.35, 2.35, 180))
        assert cmd.v >= 0.0


def test_reverse_still_steers():
    """Backing off should also turn, so successive attempts differ."""
    p = PurePursuit(PursuitConfig())
    path = np.stack([np.zeros(20), np.arange(20) * 0.5], axis=1)
    cmd = p.step(np.array([0.0, 0.0, 0.0]), path,
                 scan_ranges=np.full(180, np.inf),
                 scan_angles=np.linspace(-2.35, 2.35, 180),
                 contact=True)
    assert cmd.v < 0.0
    assert abs(cmd.omega) > 0.0


def test_reverse_latches_and_does_not_chatter():
    """Contact must commit to reversing, not bounce on the boundary.

    The probe sits 0.6 m ahead, so a few centimetres of reverse clears it.
    Without a latch the controller drives forward again immediately and
    contact re-fires. Observed live: five consecutive 15 s stuck events at
    the same spot, +/-0.3 m of position jitter, and odometry gaining 3.4 m
    of travel that never happened.
    """
    p = PurePursuit(PursuitConfig())
    path = straight_path()
    pose = np.array([0.0, 0.0, 0.0])

    first = p.step(pose, path, contact=True)
    assert first.v < 0.0

    # Contact clears immediately afterwards; it must keep reversing.
    still = [p.step(pose, path, contact=False) for _ in range(10)]
    assert all(c.v < 0.0 for c in still), "recovery gave up on the boundary"
    assert all(c.reason == "backing off" for c in still)


def test_reverse_eventually_releases():
    """The latch must expire, or the robot reverses forever."""
    cfg = PursuitConfig()
    p = PurePursuit(cfg)
    path = straight_path()
    pose = np.array([0.0, 0.0, 0.0])
    p.step(pose, path, contact=True)
    for _ in range(cfg.reverse_hold_cycles + 2):
        cmd = p.step(pose, path, contact=False)
    assert cmd.v > 0.0, "still reversing after the hold expired"
    assert cmd.reason != "backing off"


def test_persistent_brake_stop_eventually_reverses():
    """The laser brake must not be able to pin the robot indefinitely.

    Reverse was originally wired only to the map-based contact test. The
    brake is the other way the robot comes to rest and had no recovery: it
    zeroed v, the turn-in-place sweep rotated, and the brake re-fired
    because a 6.6 m obstacle fills the cone across most headings. Observed
    live as five consecutive 15 s stuck events with travelled frozen.
    """
    cfg = PursuitConfig()
    p = PurePursuit(cfg)
    path = straight_path()
    pose = np.array([0.0, 0.0, 0.0])
    ranges = np.full(180, 25.0)
    ranges[85:95] = 0.4                       # hard stop: inside brake/2
    angles = np.linspace(-2.35, 2.35, 180)

    seen = []
    for _ in range(cfg.stuck_cycles_before_reverse + 5):
        seen.append(p.step(pose, path, scan_ranges=ranges, scan_angles=angles))
    assert seen[0].v == 0.0 and seen[0].reason == "stopped"
    assert any(c.v < 0.0 for c in seen), "brake pinned the robot forever"


def test_brief_brake_stop_does_not_reverse():
    """A momentary stop must not trigger a reverse -- only a persistent one."""
    cfg = PursuitConfig()
    p = PurePursuit(cfg)
    path = straight_path()
    pose = np.array([0.0, 0.0, 0.0])
    ranges = np.full(180, 25.0)
    ranges[85:95] = 0.4
    angles = np.linspace(-2.35, 2.35, 180)
    for _ in range(3):
        cmd = p.step(pose, path, scan_ranges=ranges, scan_angles=angles)
    assert cmd.v >= 0.0, "reversed after only three cycles"
