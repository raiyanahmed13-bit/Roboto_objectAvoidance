"""Compare a live SLAM map against the GIS prior.

This is the project's headline result: finding where reality disagrees with
OpenStreetMap, and doing it in a way that survives contact with a real,
imperfect SLAM map.

    prior (OSM)  +  SLAM map  +  observability  ->  per-cell classification

TWO THINGS MAKE THIS HONEST RATHER THAN EASY
--------------------------------------------
1. **The observability mask.** A cell the robot never sensed says nothing.
   Without masking, every unvisited cell reads as "the prior claims an
   obstacle and I see none" and the headline metric degenerates into "the
   robot did not go there". This is the single most important line here.

2. **Asymmetric thresholds.** Declaring a MISSED obstacle is
   safety-critical and cheap to act on, so be eager. Declaring a PHANTOM
   removes a known obstacle from the costmap on the robot's own evidence,
   which can drive it into a wall -- so be conservative, and demand far
   more observations.

   The asymmetry is also forced by measurement: pose error lets the
   free-space trace erode real walls, so weak "I see free where the prior
   says wall" evidence is exactly the failure mode we must not trust.

REGISTRATION FIRST
------------------
Even a good SLAM map sits centimetres and tenths of a degree off the prior.
Comparing unregistered grids inflates every metric, so a single global
SE(2) correction is estimated first -- reusing the scan matcher, which is
the third job that one implementation does.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

import numpy as np

from ..frames import GridSpec
from ..slam.likelihood_field import LikelihoodField
from ..slam.scan_matcher import CorrelativeScanMatcher
from ..slam.transforms2d import transform_points


class Cls(IntEnum):
    """Per-cell verdict. Values are stable; they index colour tables."""

    UNKNOWN = 0        # not observed, or not confident -- excluded from metrics
    AGREE_FREE = 1
    AGREE_OCC = 2
    MISSED = 3         # SLAM sees an obstacle the prior lacks   <- headline
    PHANTOM = 4        # prior claims an obstacle SLAM cannot find


@dataclass
class DiscrepancyConfig:
    """Thresholds. Deliberately asymmetric -- see the module docstring."""

    # MISSED: an unmapped obstacle. Safety-critical, so react readily.
    l_conf_missed: float = 1.5      # p ~ 0.82
    n_obs_missed: int = 5

    # PHANTOM: clearing a mapped obstacle. Costly if wrong, so demand more.
    l_conf_phantom: float = 3.0     # p ~ 0.05
    n_obs_phantom: int = 15

    # Suppress MISSED within this distance of a prior obstacle.
    #
    # A mapped wall and the prior's version of it never align perfectly --
    # sub-cell registration error plus the endpoint landing a cell short
    # leave a thin sliver of "SLAM occupied where the prior says free"
    # running along every building face. Measured, those slivers were EVERY
    # false positive in a full mission: all 7 sat 0.00-0.10 m from a
    # building, while all 3 true detections were 4.5-14.9 m clear of one.
    #
    # A real unmapped obstacle flush against a wall would be suppressed
    # too, but that is rare and costs nothing operationally: the building
    # is already lethal in the costmap, so the robot avoids that space
    # regardless.
    missed_clearance_m: float = 0.30

    # Registration search, in metres and radians.
    register: bool = True
    register_res: float = 0.40
    register_win_xy: float = 3.0
    register_win_th: float = np.radians(5.0)
    register_max_points: int = 4000


@dataclass
class ComparisonResult:
    labels: np.ndarray               # Cls per cell
    grid: GridSpec
    offset: np.ndarray               # SE(2) prior->slam correction applied
    offset_residual: float           # translation magnitude, metres
    observed_fraction: float         # of the grid, ever sensed

    def count(self, cls: Cls) -> int:
        return int((self.labels == cls).sum())

    def area_m2(self, cls: Cls) -> float:
        return self.count(cls) * self.grid.resolution ** 2

    @property
    def agreement(self) -> float:
        """Fraction of CLASSIFIED cells where prior and SLAM agree.

        Over classified cells only; including UNKNOWN would let a robot
        that never moved score perfectly.
        """
        agree = self.count(Cls.AGREE_FREE) + self.count(Cls.AGREE_OCC)
        total = agree + self.count(Cls.MISSED) + self.count(Cls.PHANTOM)
        return float(agree / total) if total else 0.0

    def summary(self) -> dict:
        return {
            "observed_fraction": self.observed_fraction,
            "agreement": self.agreement,
            "missed_cells": self.count(Cls.MISSED),
            "phantom_cells": self.count(Cls.PHANTOM),
            "missed_m2": self.area_m2(Cls.MISSED),
            "phantom_m2": self.area_m2(Cls.PHANTOM),
            "offset_m": float(self.offset_residual),
            "offset_deg": float(np.degrees(self.offset[2])),
        }


# The dilated prior is a pure function of the prior and the clearance, and
# the prior never changes during a mission -- but dilating 9 million cells
# costs enough to dominate a per-step discrepancy check. Cached so a closed
# loop can afford to run detection often.
_DILATE_CACHE: dict[tuple, np.ndarray] = {}


def _far_from_prior(prior: np.ndarray, grid: GridSpec,
                    clearance_m: float) -> np.ndarray:
    """Cells far enough from any mapped obstacle to judge as MISSED."""
    if clearance_m <= 0:
        return ~prior

    key = (id(prior), prior.shape, float(clearance_m), grid.resolution)
    hit = _DILATE_CACHE.get(key)
    if hit is not None:
        return hit

    from scipy.ndimage import binary_dilation

    k = max(1, int(round(clearance_m / grid.resolution)))
    out = ~binary_dilation(prior, np.ones((3, 3), bool), iterations=k)

    # Bounded: a mission uses one prior, an experiment matrix a handful.
    if len(_DILATE_CACHE) > 4:
        _DILATE_CACHE.clear()
    _DILATE_CACHE[key] = out
    return out

def register_to_prior(slam_map, prior_occ: np.ndarray,
                      cfg: DiscrepancyConfig | None = None) -> np.ndarray:
    """Estimate the SE(2) correction taking the SLAM map onto the prior.

    Treats the SLAM map's occupied cells as a very large "scan" and matches
    it against a coarse likelihood field built from the prior. Reported as
    a diagnostic in its own right: a residual above ~1 m means something
    upstream is broken, not that the world moved.
    """
    cfg = cfg or DiscrepancyConfig()
    grid = slam_map.grid

    occ = slam_map.occupied()
    rows, cols = np.nonzero(occ)
    if rows.size < 50:
        return np.zeros(3)

    # Thin to keep the match cheap; the map has tens of thousands of cells.
    if rows.size > cfg.register_max_points:
        idx = np.linspace(0, rows.size - 1, cfg.register_max_points).astype(int)
        rows, cols = rows[idx], cols[idx]
    x, y = grid.cell_to_world(rows, cols)
    pts = np.column_stack([np.asarray(x), np.asarray(y)])

    field = LikelihoodField.from_occupancy(prior_occ, grid, sigma_m=0.6)
    matcher = CorrelativeScanMatcher(
        field,
        coarse_lin=cfg.register_res, coarse_ang=np.radians(1.0),
        coarse_win_xy=cfg.register_win_xy, coarse_win_th=cfg.register_win_th,
        fine_lin=cfg.register_res / 4, fine_ang=np.radians(0.25),
        fine_win_xy=cfg.register_res, fine_win_th=np.radians(1.0),
        max_points=cfg.register_max_points,
    )
    # Matching from the identity pose: the correction IS the resulting pose.
    return matcher.match(pts, np.zeros(3)).pose


def classify(slam_map, prior_occ: np.ndarray,
             cfg: DiscrepancyConfig | None = None) -> ComparisonResult:
    """Label every cell of the shared grid."""
    cfg = cfg or DiscrepancyConfig()
    grid = slam_map.grid
    prior = np.asarray(prior_occ, dtype=bool)
    if prior.shape != grid.shape:
        raise ValueError(f"prior {prior.shape} != slam grid {grid.shape}")

    offset = np.zeros(3)
    lo = slam_map.log_odds
    counts = slam_map.observed_count

    if cfg.register:
        offset = register_to_prior(slam_map, prior, cfg)
        if np.hypot(offset[0], offset[1]) > 1e-6 or abs(offset[2]) > 1e-9:
            lo = _shift(lo, offset, grid, fill=0.0)
            counts = _shift(counts.astype(np.float32), offset, grid,
                            fill=0.0).astype(np.uint16)

    labels = np.full(grid.shape, Cls.UNKNOWN, dtype=np.uint8)

    seen_missed = counts >= cfg.n_obs_missed
    seen_phantom = counts >= cfg.n_obs_phantom

    slam_occ = lo >= cfg.l_conf_missed
    slam_free_soft = lo <= -cfg.l_conf_missed
    slam_free_hard = lo <= -cfg.l_conf_phantom

    # Cells too close to a mapped obstacle to judge are left UNKNOWN rather
    # than being called MISSED -- see missed_clearance_m.
    far_from_prior = _far_from_prior(prior, grid, cfg.missed_clearance_m)

    labels[seen_missed & prior & slam_occ] = Cls.AGREE_OCC
    labels[seen_missed & ~prior & slam_free_soft] = Cls.AGREE_FREE
    labels[seen_missed & far_from_prior & slam_occ] = Cls.MISSED
    labels[seen_phantom & prior & slam_free_hard] = Cls.PHANTOM

    return ComparisonResult(
        labels=labels,
        grid=grid,
        offset=offset,
        offset_residual=float(np.hypot(offset[0], offset[1])),
        observed_fraction=float((slam_map.observed_count > 0).mean()),
    )


def _shift(arr: np.ndarray, pose, grid: GridSpec, fill=0.0) -> np.ndarray:
    """Apply an SE(2) correction to a grid by resampling.

    Nearest-neighbour on purpose: these are log-odds and observation
    counts, and interpolating them would invent evidence that no sensor
    ever produced.
    """
    from scipy.ndimage import affine_transform

    x, y, th = float(pose[0]), float(pose[1]), float(pose[2])
    c, s = np.cos(-th), np.sin(-th)

    # Rotation about the grid centre, then translation in cells.
    cr, cc = grid.height / 2.0, grid.width / 2.0
    mat = np.array([[c, -s], [s, c]])
    off = np.array([cr, cc]) - mat @ np.array([cr, cc])
    off -= np.array([y, x]) / grid.resolution

    return affine_transform(arr, mat, offset=off, order=0, mode="constant",
                            cval=fill, output=arr.dtype)
