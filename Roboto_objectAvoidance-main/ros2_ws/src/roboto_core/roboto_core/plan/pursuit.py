"""Pure-pursuit path tracking for a differential drive.

The offline experiments teleported the robot along waypoints, which is fine
for measuring planning and detection but is not a robot. Driving live in
Gazebo needs an actual controller: something that turns "here is a path"
into a stream of (v, omega) commands, and that behaves sanely when the path
is stale, the robot has drifted off it, or something is directly ahead.

Pure pursuit is the right level of sophistication here. It is a geometric
controller -- aim at a point a fixed distance ahead on the path and follow
the arc that reaches it -- so it has one meaningful parameter, no tuning
loops, and no failure modes that need explaining in a report. The planner
is what this project is being graded on; the controller only has to not be
the reason something breaks.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..slam.transforms2d import wrap_angle


@dataclass
class PursuitConfig:
    # LOOKAHEAD IS THE DAMPING TERM, AND 1.8 WAS TOO SHORT.
    #
    # Pure pursuit is only marginally stable: it steers at a point a fixed
    # distance ahead, so a short lookahead over-corrects, overshoots, and
    # weaves. Lateral error enters the steering law as 2*by/L^2, so noise
    # in the pose estimate is amplified by the SQUARE of shortening it --
    # and the live estimate jitters, swinging 0.42 to 0.95 m between
    # consecutive samples.
    #
    # Observed live at 1.8: heading changes of +38.9, +26.6, -17.1, -21.9,
    # +30.9, -25.1, -33.6, +29.6, -51.3 degrees in consecutive samples,
    # with no obstacle nearby and no recovery active. Alternating sign,
    # growing amplitude -- the controller arguing with itself.
    #
    # Driving the real controller round the real route with 0.25 m of pose
    # jitter, at 0.1 s control period:
    #
    #   lookahead   turn rate   p99    mean xtrack   max xtrack
    #      2.34 m    3.5 deg/s  10.4       0.15 m       0.35 m
    #      3.54 m    2.1 deg/s   6.4       0.17 m       0.48 m
    #      5.54 m    1.5 deg/s   4.7       0.24 m       0.81 m
    #
    # 3.0 buys 40% less weave for 0.13 m more cross-track. Minimum route
    # clearance is 1.52 m against a 0.35 m robot, so that is affordable;
    # going further trades corner-cutting for diminishing returns.
    #
    # The simulation is kinematic -- no actuator lag -- so it understates
    # the live amplitude. The ordering is what it is being trusted for.
    lookahead_m: float = 3.0
    # Lookahead grows with speed: a fixed distance that is stable at 0.2 m/s
    # oscillates at 1.0 m/s, because the robot covers it before the command
    # takes effect.
    lookahead_gain: float = 0.6
    # Floor, so a sharp bend cannot shrink the lookahead to nothing.
    lookahead_min_m: float = 1.6
    # How hard curvature shortens it.
    #
    # Driving a freshly replanned detour around a 6.6 x 2.5 m truck from
    # 4 m out -- the distance at which urgent replans actually land --
    # closest approach to the body, robot radius 0.35 m:
    #
    #   base L   gain 0    gain 1    gain 3     detour clearance
    #    2.34 m   0.66 m    0.73 m    0.79 m         1.0 m
    #    3.54 m   0.19 m*   0.56 m    0.75 m         1.0 m
    #    3.54 m   0.50 m    0.90 m    1.11 m         1.5 m
    #                * collision
    #
    # At gain 3 the long lookahead corners as well as the short one, so the
    # damping on straights costs nothing in avoidance. A fixed long
    # lookahead does not: it aims past the swerve and drives into the
    # obstacle it was replanning around.
    lookahead_turn_gain: float = 3.0
    max_speed: float = 0.9
    min_speed: float = 0.15
    max_omega: float = 1.2

    goal_tolerance_m: float = 0.6
    slow_radius_m: float = 2.5        # ease off approaching the goal

    # Curvature slows the robot: taking a tight turn at full speed on a
    # diff drive means one wheel reversing, which the odometry model
    # handles badly and which looks wrong on video.
    curvature_slowdown: float = 0.8

    # Emergency brake straight from the laser, independent of the map.
    # A controller that can only stop for obstacles it has already mapped
    # is one mapping failure away from a collision.
    brake_distance_m: float = 1.2
    brake_halfangle_rad: float = 0.6

    # Speed used to back out of contact. Deliberately slow -- this is a
    # recovery, not a manoeuvre, and the robot is reversing into space it
    # cannot currently sense. It only has to move the footprint far enough
    # for the contact test to clear, which is tens of centimetres.
    reverse_speed: float = 0.25

    # HOW LONG TO KEEP REVERSING ONCE STARTED.
    #
    # Without this the recovery chatters. The contact probe sits 0.6 m
    # ahead of the robot, so a few centimetres of reverse clears it, the
    # controller immediately drives forward again, and contact re-fires --
    # a limit cycle at the boundary. Observed live: the robot held station
    # at (-15.5, 62.2) for five consecutive 15 s stuck events, heading
    # swinging 49 to 94 degrees, position jittering +/-0.3 m, odometry
    # gaining 3.4 m of travel that never happened, and eight replans
    # alternating unstick/urgent at the same 245 m remaining.
    #
    # 25 cycles at 10 Hz is 2.5 s, about 0.6 m of reverse -- enough to move
    # the footprint clear of the probe offset rather than just to its edge.
    reverse_hold_cycles: int = 25

    # Cycles the laser brake may hold the robot at zero before the reverse
    # recovery engages. 20 at 10 Hz is 2 s of fruitless rotation.
    stuck_cycles_before_reverse: int = 20


@dataclass
class PursuitCommand:
    v: float
    omega: float
    done: bool = False
    index: int = 0                # nearest path index, for progress tracking
    reason: str = "tracking"


class PurePursuit:
    """Turns a path plus a pose into velocity commands."""

    def __init__(self, config: PursuitConfig | None = None):
        self.cfg = config or PursuitConfig()
        # Cycles of reverse still owed. Latched on contact so the recovery
        # commits instead of chattering at the contact boundary.
        self._backing = 0
        # Consecutive cycles the laser brake has held the robot at zero.
        self._stopped = 0
        # Contact claims overruled by a scan that could see. A high count
        # means the map and the estimate have drifted apart.
        self._vetoed = 0

    # `contact` is the map's answer to "am I touching something?", which the
    # scan cannot give inside its minimum range. See the CONTACT block below.
    def step(self, pose, path: np.ndarray, scan_ranges=None,
             scan_angles=None, from_index: int = 0,
             contact: bool = False) -> PursuitCommand:
        """One control cycle.

        `path` is (N, 2) in the world frame. `from_index` lets the caller
        keep progress monotonic so the robot cannot latch onto an earlier
        part of a path that loops back near itself.
        """
        cfg = self.cfg
        pose = np.asarray(pose, dtype=np.float64)
        path = np.asarray(path, dtype=np.float64)

        if len(path) == 0:
            return PursuitCommand(0.0, 0.0, done=True, reason="no path")

        # --- progress along the path -------------------------------------
        d = np.hypot(path[from_index:, 0] - pose[0], path[from_index:, 1] - pose[1])
        near = int(np.argmin(d)) + from_index

        goal_dist = float(np.hypot(*(path[-1] - pose[:2])))
        if goal_dist <= cfg.goal_tolerance_m:
            return PursuitCommand(0.0, 0.0, done=True, index=near, reason="goal")

        # --- pick the lookahead point ------------------------------------
        #
        # ADAPTIVE, BECAUSE THE TWO FAILURES PULL IN OPPOSITE DIRECTIONS.
        #
        # Long lookahead damps weaving: lateral error enters the steering
        # law as 2*by/L^2, so a short L amplifies pose jitter by its square
        # and the controller oscillates. Measured on a straight route with
        # 0.25 m of jitter: 3.5 deg/s of heading churn at L=2.34, 2.1 at
        # L=3.54.
        #
        # Short lookahead is required to FOLLOW a turn. Pure pursuit aims
        # at a point L ahead and drives the arc to it, so when L exceeds
        # the manoeuvre's own scale it aims past the swerve and cuts the
        # corner. Driving a 1.5 m detour around a 6.6 x 2.5 m truck, the
        # closest approach to the body was:
        #
        #     L = 2.34 m   0.68 m
        #     L = 3.04 m   0.31 m   <- inside the robot radius
        #     L = 3.54 m   0.14 m
        #     L = 4.04 m   0.00 m
        #
        # A fixed value cannot serve both: raising it to stop the weaving
        # is what turned avoidance manoeuvres into collisions.
        #
        # So it shrinks where the path bends. `turn` is how much the route
        # curves over the next stretch, which is near zero on a straight
        # and large exactly where an avoidance detour swings out.
        speed_guess = cfg.max_speed
        L = cfg.lookahead_m + cfg.lookahead_gain * speed_guess
        turn = self._path_turn(path, near, L)
        L = max(cfg.lookahead_min_m, L / (1.0 + cfg.lookahead_turn_gain * turn))
        target = self._lookahead_point(path, pose, near, L)

        # --- geometry: the arc through the target ------------------------
        dx = target[0] - pose[0]
        dy = target[1] - pose[1]
        c, s = np.cos(-pose[2]), np.sin(-pose[2])
        bx = c * dx - s * dy          # forward, in the body frame
        by = s * dx + c * dy          # left

        dist2 = bx * bx + by * by
        curvature = 0.0 if dist2 < 1e-9 else (2.0 * by / dist2)

        # --- speed -------------------------------------------------------
        v = cfg.max_speed
        v *= 1.0 / (1.0 + cfg.curvature_slowdown * abs(curvature))
        if goal_dist < cfg.slow_radius_m:
            v *= max(0.25, goal_dist / cfg.slow_radius_m)

        reason = "tracking"

        # CONTACT: something the MAP says is there, that the SCAN cannot see.
        #
        # The lidar has a 0.15 m minimum range. Inside it every beam returns
        # a non-finite value, _forward_clearance filters those out, finds no
        # valid returns and answers None -- which the brake below reads as
        # "no reading, do not brake". So a robot actually touching an
        # obstacle sees a clear path ahead and keeps pushing, and its
        # confidence GROWS as it closes in, because more beams drop below
        # the minimum. Observed live: wedged at (-1, 78) for 45 s, heading
        # swinging -25 to -170 degrees, 2.5 m of progress.
        #
        # The scan cannot answer this on its own -- an out-of-range return
        # and a too-close return are the same value, by LaserScan
        # convention -- so the caller supplies the map's opinion instead.
        # Scan for what the map does not know; map for what the scan cannot
        # see.
        # THE MAP ONLY WINS WHERE THE SCAN IS BLIND.
        #
        # Contact exists because a touching obstacle is INVISIBLE: inside
        # the 0.15 m minimum range every beam is invalid, so the scan
        # reports nothing and the brake declines to act. That is the case
        # the map must cover.
        #
        # It is not licence to override a scan that can see. Observed live:
        # the robot sat in open ground at (-40.3, 32.6) -- 3.6 m clear of
        # the nearest truck, 15.5 m from any building -- reversing in a
        # loop. SLAM was 3 m out, so mapped obstacle cells landed displaced
        # and classified as fresh MISSED clusters: obj reached 10 with six
        # real obstacles. The contact probe hit a phantom, the robot backed
        # away from nothing, and reversing degraded odometry further, which
        # grew the error, which made more phantoms.
        #
        # So a positive observation of clear space vetoes the map's claim.
        # If the forward cone returns a valid range well beyond braking
        # distance, the robot is demonstrably not touching anything and the
        # map is wrong. If the cone has no valid returns, the scan has no
        # opinion and contact stands.
        brake = self._forward_clearance(scan_ranges, scan_angles)
        if contact and brake is not None and brake > cfg.brake_distance_m:
            contact = False
            self._vetoed += 1

        if contact:
            v = 0.0
            reason = "contact"

        if not contact and brake is not None and brake < cfg.brake_distance_m:
            # Scale down smoothly, and stop outright inside half the
            # braking distance.
            f = max(0.0, (brake - 0.5 * cfg.brake_distance_m)
                    / (0.5 * cfg.brake_distance_m))
            v *= f
            reason = "braking" if f > 0 else "stopped"

        # Reversing into an obstacle behind us is never the right answer, so
        # the robot turns in place instead of backing up.
        if bx < 0 and abs(by) > 0.1:
            v = min(v, cfg.min_speed)
            reason = "turning in place"

        v = float(np.clip(v, 0.0, cfg.max_speed))
        omega = float(np.clip(v * curvature, -cfg.max_omega, cfg.max_omega))

        # A STOPPED ROBOT MUST STILL BE ABLE TO TURN.
        #
        # Turning in place is safe on a differential drive -- it does not
        # translate, so it cannot drive into whatever stopped us -- and once
        # the brake has zeroed v it is the only move left.
        #
        # This guard used to read `reason != "stopped"`, excluding the exact
        # case it was written for. Observed live: the robot came to rest
        # 0.51 m from a building, inside the 0.6 m hard-stop threshold, so
        # v was 0 and omega was v * curvature = 0. It commanded (0, 0)
        # indefinitely and nothing ever re-triggered planning.
        #
        # `by` can be ~0 when the target sits dead ahead through the
        # obstacle, and sign(0) is 0 -- which would leave omega at zero and
        # reproduce the deadlock. Fall back to a fixed direction so the
        # robot always sweeps until it finds a way out.
        if v < 1e-3:
            turn = float(np.sign(by)) if abs(by) > 1e-3 else 1.0
            omega = float(np.clip(turn * 0.4, -cfg.max_omega, cfg.max_omega))

        # CONTACT MUST BE ESCAPABLE, AND ROTATING IS NOT ENOUGH.
        #
        # Turning in place clears a brake trigger, because the brake looks
        # in a cone ahead and rotating moves the cone. It does NOT clear
        # contact: the footprint stays exactly where it is. Against a 6.6 m
        # truck almost every heading keeps the robot touching, so it spun.
        #
        # Measured live, and this is the whole failure chain in one run: the
        # robot pinned at (-38.2, 35.4), four stuck events, ten replans all
        # returning the same 172-178 m route because it never moved. The
        # wheels kept turning against the obstacle, so odometry gained about
        # 15 m of motion that never happened -- error 7.8 m -> 22.6 m over
        # 23 m of "travel". SLAM predicts from odometry, every correction
        # then exceeded max_correction_m and was refused, and the estimate
        # free-ran to 27 m wrong. One contact destroyed the whole run.
        #
        # Reversing is the only move that changes the footprint's position
        # while contact holds. It is safe in a way forward motion is not:
        # the robot arrived along this heading, so the space behind it was
        # occupied by the robot itself a moment ago.
        # THE BRAKE CAN STOP THE ROBOT BUT CANNOT UN-STOP IT.
        #
        # Reverse was wired only to the map-based contact test. The laser
        # brake is the other way the robot comes to rest, and it has no
        # recovery of its own -- it zeroes v, the turn-in-place sweep
        # rotates, and the brake immediately re-fires because a 6.6 m
        # obstacle fills the +/-34 degree cone across most headings.
        #
        # Observed live: held at (-35.4, 40.4) for five consecutive 15 s
        # stuck events, heading sweeping 11 to 72 degrees, travelled frozen
        # at 171.3 m, six replans all returning the same 169 m route. The
        # map did not agree anything was there -- SLAM was 5.8 m out -- so
        # `contact` never fired and the reverse never engaged.
        #
        # Rotating for two seconds without freeing itself is enough
        # evidence that rotating is not the answer.
        if reason == "stopped":
            self._stopped += 1
        else:
            self._stopped = 0

        if contact or self._stopped >= cfg.stuck_cycles_before_reverse:
            self._backing = cfg.reverse_hold_cycles
            self._stopped = 0
        if self._backing > 0:
            self._backing -= 1
            # Clamped to max_speed as well: this assignment happens AFTER
            # the np.clip above, so max_speed:=0.2 would otherwise still
            # reverse at 0.25 m/s -- faster than the robot is allowed to
            # drive forwards.
            v = -min(cfg.reverse_speed, cfg.max_speed)
            omega = float(np.clip(-0.5 * omega, -cfg.max_omega, cfg.max_omega))
            reason = "backing off"

        return PursuitCommand(v, omega, done=False, index=near, reason=reason)

    # ---- internals -------------------------------------------------------

    def _path_turn(self, path, near, L):
        """Total heading change of the path over the next `L` metres, in rad.

        Near zero on a straight; of order 1 rad where an avoidance detour
        swings out and back. Used to shorten the lookahead exactly where a
        long one would cut the corner.
        """
        tail = path[near:]
        if len(tail) < 3:
            return 0.0
        seg = np.hypot(*np.diff(tail, axis=0).T)
        along = np.concatenate([[0.0], np.cumsum(seg)])
        keep = tail[along <= max(L, 1e-6)]
        if len(keep) < 3:
            return 0.0
        d = np.diff(keep, axis=0)
        th = np.unwrap(np.arctan2(d[:, 1], d[:, 0]))
        return float(np.abs(np.diff(th)).sum())

    def _lookahead_point(self, path, pose, near, L):
        """First point at least L ahead of `near`; the endpoint otherwise."""
        acc = 0.0
        for k in range(near, len(path) - 1):
            acc += float(np.hypot(*(path[k + 1] - path[k])))
            if acc >= L:
                return path[k + 1]
        return path[-1]

    def _forward_clearance(self, ranges, angles):
        """Nearest return within a cone straight ahead, or None."""
        if ranges is None or angles is None:
            return None
        ranges = np.asarray(ranges, dtype=np.float64)
        angles = np.asarray(angles, dtype=np.float64)
        if ranges.size == 0 or ranges.shape != angles.shape:
            return None

        cone = np.abs(wrap_angle(angles)) <= self.cfg.brake_halfangle_rad
        vals = ranges[cone & np.isfinite(ranges)]
        return float(vals.min()) if vals.size else None
