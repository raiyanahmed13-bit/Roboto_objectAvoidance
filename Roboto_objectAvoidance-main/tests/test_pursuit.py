"""Pure-pursuit path tracking.

This file exists because the controller had no tests and shipped a deadlock
because of it: nose-in 0.51 m from a building, the brake zeroed `v`, the
in-place recovery turn was skipped by its own guard, and the robot
commanded (0, 0) for twenty minutes while nothing re-triggered planning.

The property that matters is not "the commands are optimal" -- it is
"the controller always leaves itself a way out". Everything below is about
that.
"""

import numpy as np
import pytest

from roboto_core.plan.pursuit import PurePursuit, PursuitConfig


@pytest.fixture
def straight_path():
    """20 m of path running due east from the origin."""
    return np.column_stack([np.linspace(0.0, 20.0, 101), np.zeros(101)])


def blocking_scan(distance_m, n=180, fov_deg=270.0):
    """A scan with a wall straight ahead at `distance_m`, open elsewhere."""
    half = np.radians(fov_deg) / 2.0
    angles = np.linspace(-half, half, n)
    ranges = np.full(n, 30.0)
    ranges[np.abs(angles) <= 0.6] = distance_m
    return ranges, angles


def clear_scan(n=180, fov_deg=270.0):
    half = np.radians(fov_deg) / 2.0
    return np.full(n, 30.0), np.linspace(-half, half, n)


# ---------------------------------------------------------------------------
# Ordinary tracking
# ---------------------------------------------------------------------------

def test_drives_forward_along_a_straight_path(straight_path):
    cmd = PurePursuit().step((0.0, 0.0, 0.0), straight_path)
    assert cmd.v > 0.5
    assert abs(cmd.omega) < 0.1
    assert not cmd.done


def test_turns_towards_a_path_offset_to_the_left(straight_path):
    """Robot below the path, facing east: it should steer left (+omega)."""
    cmd = PurePursuit().step((0.0, -1.0, 0.0), straight_path)
    assert cmd.omega > 0.0


def test_reports_done_inside_the_goal_tolerance(straight_path):
    cfg = PursuitConfig()
    pose = (20.0 - 0.5 * cfg.goal_tolerance_m, 0.0, 0.0)
    cmd = PurePursuit(cfg).step(pose, straight_path)
    assert cmd.done
    assert cmd.v == 0.0


def test_empty_path_is_done_not_a_crash():
    cmd = PurePursuit().step((0.0, 0.0, 0.0), np.zeros((0, 2)))
    assert cmd.done
    assert cmd.v == 0.0


def test_slows_down_near_the_goal(straight_path):
    far = PurePursuit().step((10.0, 0.0, 0.0), straight_path)
    near = PurePursuit().step((18.5, 0.0, 0.0), straight_path)
    assert near.v < far.v


# ---------------------------------------------------------------------------
# The brake
# ---------------------------------------------------------------------------

def test_brakes_for_an_obstacle_the_map_does_not_have(straight_path):
    """The brake reads the laser directly, so it does not depend on the
    obstacle having been mapped or classified first."""
    cfg = PursuitConfig()
    r, a = blocking_scan(0.8 * cfg.brake_distance_m)
    slowed = PurePursuit(cfg).step((0.0, 0.0, 0.0), straight_path,
                                   scan_ranges=r, scan_angles=a)
    clear = PurePursuit(cfg).step((0.0, 0.0, 0.0), straight_path,
                                  *clear_scan())
    assert 0.0 < slowed.v < clear.v


def test_stops_dead_inside_half_the_brake_distance(straight_path):
    cfg = PursuitConfig()
    r, a = blocking_scan(0.4 * cfg.brake_distance_m)
    cmd = PurePursuit(cfg).step((0.0, 0.0, 0.0), straight_path,
                                scan_ranges=r, scan_angles=a)
    assert cmd.v == pytest.approx(0.0, abs=1e-9)
    assert cmd.reason == "stopped"


def test_a_far_obstacle_does_not_brake(straight_path):
    cfg = PursuitConfig()
    r, a = blocking_scan(3.0 * cfg.brake_distance_m)
    cmd = PurePursuit(cfg).step((0.0, 0.0, 0.0), straight_path,
                                scan_ranges=r, scan_angles=a)
    assert cmd.v > 0.5
    assert cmd.reason == "tracking"


# ---------------------------------------------------------------------------
# The deadlock. These are the regression tests.
# ---------------------------------------------------------------------------

def test_a_stopped_robot_can_still_turn(straight_path):
    """The live failure, reduced.

    Stopped by the brake, the controller must still command a rotation.
    A differential drive turning in place does not translate, so this
    cannot drive it further into whatever stopped it -- and it is the only
    move that can change the situation.
    """
    cfg = PursuitConfig()
    r, a = blocking_scan(0.42 * cfg.brake_distance_m)   # 0.5 m, as observed
    cmd = PurePursuit(cfg).step((0.0, 0.0, 0.0), straight_path,
                                scan_ranges=r, scan_angles=a)
    assert cmd.v == pytest.approx(0.0, abs=1e-9)
    assert abs(cmd.omega) > 0.1, "a stopped robot that cannot turn is wedged"


def test_stopped_and_target_dead_ahead_still_turns(straight_path):
    """The nastier variant: the lookahead point is straight through the
    obstacle, so the lateral offset `by` is ~0 and sign(by) is 0. Steering
    proportional to it would leave omega at zero and reproduce the
    deadlock exactly."""
    cfg = PursuitConfig()
    r, a = blocking_scan(0.3 * cfg.brake_distance_m)
    # Dead on the path, facing straight along it: by == 0 by construction.
    cmd = PurePursuit(cfg).step((0.0, 0.0, 0.0), straight_path,
                                scan_ranges=r, scan_angles=a)
    assert cmd.v == pytest.approx(0.0, abs=1e-9)
    assert abs(cmd.omega) > 0.1


def test_never_commands_zero_velocity_and_zero_rotation(straight_path):
    """The invariant, swept across the whole brake range and several
    headings: unless the controller is DONE, it must always leave itself
    some way to act."""
    cfg = PursuitConfig()
    for dist in np.linspace(0.05, 2.0 * cfg.brake_distance_m, 25):
        for yaw in np.linspace(-np.pi, np.pi, 9):
            for lateral in (-1.5, 0.0, 1.5):
                r, a = blocking_scan(float(dist))
                cmd = PurePursuit(cfg).step(
                    (0.0, lateral, float(yaw)), straight_path,
                    scan_ranges=r, scan_angles=a)
                if cmd.done:
                    continue
                assert abs(cmd.v) > 1e-9 or abs(cmd.omega) > 1e-9, (
                    f"wedged at distance {dist:.2f} m, yaw {np.degrees(yaw):.0f} "
                    f"deg, lateral {lateral} m")


def test_turn_direction_follows_the_target_when_it_is_off_axis(straight_path):
    """Turning is not arbitrary: when the path is clearly to one side the
    robot should rotate towards it, not away."""
    cfg = PursuitConfig()
    r, a = blocking_scan(0.3 * cfg.brake_distance_m)
    # Robot south of the path, facing east -> the path is to its left.
    cmd = PurePursuit(cfg).step((0.0, -3.0, 0.0), straight_path,
                                scan_ranges=r, scan_angles=a)
    assert cmd.v == pytest.approx(0.0, abs=1e-9)
    assert cmd.omega > 0.0


def test_omega_stays_within_limits(straight_path):
    cfg = PursuitConfig()
    for dist in (0.1, 0.5, 1.0, 5.0):
        r, a = blocking_scan(float(dist))
        for yaw in np.linspace(-np.pi, np.pi, 13):
            cmd = PurePursuit(cfg).step((0.0, 0.0, float(yaw)), straight_path,
                                        scan_ranges=r, scan_angles=a)
            assert abs(cmd.omega) <= cfg.max_omega + 1e-9
            assert 0.0 <= cmd.v <= cfg.max_speed + 1e-9


# ---------------------------------------------------------------------------
# Progress tracking
# ---------------------------------------------------------------------------

def test_from_index_keeps_progress_monotonic():
    """A path that loops back near itself must not let the robot latch onto
    the earlier leg and drive the route twice."""
    out = np.column_stack([np.linspace(0, 10, 51), np.zeros(51)])
    back = np.column_stack([np.linspace(10, 0, 51), np.full(51, 0.5)])
    path = np.vstack([out, back])

    # Physically near the start, but already 60 waypoints along.
    cmd = PurePursuit().step((1.0, 0.25, 0.0), path, from_index=60)
    assert cmd.index >= 60


def test_missing_scan_is_tolerated(straight_path):
    """The controller has to run before the first scan arrives."""
    cmd = PurePursuit().step((0.0, 0.0, 0.0), straight_path,
                             scan_ranges=None, scan_angles=None)
    assert cmd.v > 0.0


def test_mismatched_scan_arrays_are_ignored(straight_path):
    cmd = PurePursuit().step((0.0, 0.0, 0.0), straight_path,
                             scan_ranges=np.zeros(10), scan_angles=np.zeros(7))
    assert cmd.v > 0.0
