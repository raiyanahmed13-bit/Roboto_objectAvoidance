"""SE(2) pose algebra.

A pose is a plain ``np.array([x, y, theta])`` with theta in radians. Keeping
it as a bare array (rather than a class) means every function here
vectorizes over a stack of poses for free, which matters when the pose graph
holds a few thousand of them.

Conventions:
  * theta is measured counter-clockwise from +x (East), matching REP-103.
  * ``compose(a, b)`` is "a then b": b expressed in a's frame, mapped to
    a's parent frame. Equivalently the matrix product Ta @ Tb.
"""

from __future__ import annotations

import numpy as np


def wrap_angle(a):
    """Wrap to (-pi, pi]. Vectorized.

    Every angular residual in the project goes through this. Forgetting it
    makes a scan matcher fail only when the robot crosses +/-pi, which is a
    memorably annoying bug to track down.
    """
    return (np.asarray(a) + np.pi) % (2.0 * np.pi) - np.pi


def pose(x=0.0, y=0.0, theta=0.0) -> np.ndarray:
    return np.array([x, y, theta], dtype=np.float64)


def rot(theta) -> np.ndarray:
    """2x2 rotation matrix."""
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s], [s, c]])


def to_matrix(p) -> np.ndarray:
    """3x3 homogeneous transform."""
    x, y, th = np.asarray(p, dtype=np.float64)
    c, s = np.cos(th), np.sin(th)
    return np.array([[c, -s, x], [s, c, y], [0.0, 0.0, 1.0]])


def from_matrix(T) -> np.ndarray:
    return np.array([T[0, 2], T[1, 2], np.arctan2(T[1, 0], T[0, 0])])


def compose(a, b) -> np.ndarray:
    """a ⊕ b -- apply b in a's frame. Equivalent to to_matrix(a) @ to_matrix(b)."""
    ax, ay, ath = np.asarray(a, dtype=np.float64)
    bx, by, bth = np.asarray(b, dtype=np.float64)
    c, s = np.cos(ath), np.sin(ath)
    return np.array([
        ax + c * bx - s * by,
        ay + s * bx + c * by,
        wrap_angle(ath + bth),
    ])


def inverse(p) -> np.ndarray:
    x, y, th = np.asarray(p, dtype=np.float64)
    c, s = np.cos(th), np.sin(th)
    return np.array([-c * x - s * y, s * x - c * y, wrap_angle(-th)])


def relative(a, b) -> np.ndarray:
    """Pose of b expressed in a's frame: a^-1 ⊕ b.

    This is the pose-graph edge measurement and the scan-matching residual.
    """
    return compose(inverse(a), b)


def transform_points(p, pts) -> np.ndarray:
    """Map (N, 2) points from p's frame into p's parent frame."""
    pts = np.asarray(pts, dtype=np.float64)
    if pts.size == 0:
        return pts.reshape(0, 2)
    x, y, th = np.asarray(p, dtype=np.float64)
    c, s = np.cos(th), np.sin(th)
    out = np.empty_like(pts)
    out[:, 0] = c * pts[:, 0] - s * pts[:, 1] + x
    out[:, 1] = s * pts[:, 0] + c * pts[:, 1] + y
    return out
