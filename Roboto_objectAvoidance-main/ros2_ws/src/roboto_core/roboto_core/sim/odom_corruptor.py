"""Realistic wheel-odometry error for a differential-drive robot.

WHY THIS IS MANDATORY, NOT OPTIONAL
-----------------------------------
Gazebo's DiffDrive `OdometryPublisher` integrates the commanded wheel
velocities, so its odometry is near-perfect. Feed that to a SLAM system and
three things go wrong at once:

  * the scan matcher has nothing to correct, so it looks pointless;
  * loop closure never fires, because there is no drift to close;
  * the SLAM-vs-odometry comparison -- the headline evidence that the SLAM
    works -- shows no improvement, because the baseline was already exact.

So the drift has to be injected deliberately. Doing it here (rather than
just adding noise to the pose) keeps it physically meaningful: the errors
are wheel-level, so they produce the *systematic curvature* drift real
robots exhibit, not a random walk that averages out.

ERROR MODEL
-----------
Per-wheel scale error and a wheel-baseline error, applied to the
differential-drive kinematics:

    d_trans = (s_l + s_r) / 2          s_l, s_r = wheel arc lengths
    d_rot   = (s_r - s_l) / b          b        = wheel baseline

A baseline error `e_b` biases every rotation by a constant factor, so
heading error grows with total turning rather than cancelling out -- this is
the dominant real-world effect and the reason a robot driving a loop comes
back visibly rotated. Per-wheel scale errors add a translation bias, and a
slowly drifting gyro bias plus per-step Gaussian noise supply the rest.

All errors are drawn once per run from a seeded generator, so a given seed
reproduces exactly the same trajectory -- which is what makes the
evaluation ablations comparable.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..slam.transforms2d import compose, relative, wrap_angle


@dataclass(frozen=True)
class OdomErrorModel:
    """Error magnitudes. Defaults are realistic for a mid-quality rover.

    `baseline_err` dominates heading drift; raise it to force loop closures
    to matter, lower it to model a well-calibrated platform.
    """

    wheel_scale_err: float = 0.02      # +2 % systematic travel over-report
    wheel_asymmetry: float = 0.0001    # left/right mismatch -- SEE NOTE BELOW
    baseline_err: float = 0.01         # +1 % -> systematic heading bias
    gyro_bias_dps: float = 0.5         # deg/min of heading random walk
    trans_noise: float = 0.01          # per-step, fraction of distance
    rot_noise: float = 0.004           # per-step, radians per radian turned
    wheel_baseline_m: float = 0.4

    # NOTE ON wheel_asymmetry -- it looks implausibly small and is not.
    #
    # Its effect is amplified by distance/baseline: a left/right scale
    # mismatch `a` bends the path at `a / baseline` radians per metre, so
    # over a 300 m run with a 0.4 m baseline the heading error is
    # 750 * a radians. Calibrated against a 300 m square loop:
    #
    #     asymmetry   final drift over 300 m
    #       0.015       110 m   (37 %)   <- absurd; first attempt
    #       0.003        64 m   (21 %)
    #       0.0003       13 m   (4.5 %)
    #       0.0001         7 m  (2.4 %)  <- chosen
    #
    # Real wheeled odometry drifts a few percent of path length, so 0.0001
    # is the physically correct order of magnitude. Anything larger makes
    # scan matching diverge rather than merely work hard, which would test
    # nothing. tests/test_odom.py pins this band.


class OdomCorruptor:
    """Turns a ground-truth pose stream into a plausible odometry stream.

    Usage::

        od = OdomCorruptor(seed=0)
        odom = od.reset(gt_poses[0])
        for p in gt_poses[1:]:
            odom = od.update(p)
    """

    def __init__(self, model: OdomErrorModel | None = None,
                 seed: int | np.random.Generator | None = None):
        self.model = model or OdomErrorModel()
        self.rng = (seed if isinstance(seed, np.random.Generator)
                    else np.random.default_rng(seed))

        m = self.model
        # Drawn ONCE per run: these are calibration errors, not noise. A
        # value redrawn every step would average away and produce no drift.
        self.scale_l = 1.0 + m.wheel_scale_err + self.rng.normal(0, m.wheel_asymmetry)
        self.scale_r = 1.0 + m.wheel_scale_err + self.rng.normal(0, m.wheel_asymmetry)
        self.baseline_scale = 1.0 + m.baseline_err
        self.gyro_bias = np.radians(m.gyro_bias_dps / 60.0)   # rad per second

        self._gt_prev: np.ndarray | None = None
        self.odom = np.zeros(3)

    def reset(self, gt_pose) -> np.ndarray:
        """Start at `gt_pose`; odometry frame is initialised to match."""
        self._gt_prev = np.asarray(gt_pose, dtype=np.float64).copy()
        self.odom = np.asarray(gt_pose, dtype=np.float64).copy()
        return self.odom.copy()

    def update(self, gt_pose, dt: float = 0.1) -> np.ndarray:
        """Advance odometry by the corrupted version of the true motion."""
        gt_pose = np.asarray(gt_pose, dtype=np.float64)
        if self._gt_prev is None:
            return self.reset(gt_pose)

        # True motion, expressed in the previous body frame.
        d = relative(self._gt_prev, gt_pose)
        self._gt_prev = gt_pose.copy()

        d_trans = float(np.hypot(d[0], d[1]))
        d_rot = float(d[2])
        b = self.model.wheel_baseline_m

        # Forward kinematics -> wheel arcs, corrupt, then invert.
        s_l = d_trans - d_rot * b / 2.0
        s_r = d_trans + d_rot * b / 2.0
        s_l *= self.scale_l
        s_r *= self.scale_r

        d_trans_c = (s_l + s_r) / 2.0
        d_rot_c = (s_r - s_l) / (b * self.baseline_scale)

        m = self.model
        d_trans_c += self.rng.normal(0.0, m.trans_noise * max(abs(d_trans), 1e-3))
        d_rot_c += self.rng.normal(0.0, m.rot_noise * max(abs(d_rot), 1e-3))
        d_rot_c += self.rng.normal(self.gyro_bias * dt, self.gyro_bias * dt * 0.5)

        # Preserve any lateral component (should be ~0 for a diff drive, but
        # discarding it would silently hide a bug in the caller's motion).
        heading = np.arctan2(d[1], d[0]) if d_trans > 1e-9 else 0.0
        step = np.array([
            d_trans_c * np.cos(heading),
            d_trans_c * np.sin(heading),
            wrap_angle(d_rot_c),
        ])
        self.odom = compose(self.odom, step)
        return self.odom.copy()

    def corrupt_trajectory(self, gt_poses, dt: float = 0.1) -> np.ndarray:
        """Convenience: corrupt a whole (N, 3) ground-truth trajectory."""
        gt = np.asarray(gt_poses, dtype=np.float64)
        out = np.empty_like(gt)
        out[0] = self.reset(gt[0])
        for i in range(1, len(gt)):
            out[i] = self.update(gt[i], dt)
        return out
