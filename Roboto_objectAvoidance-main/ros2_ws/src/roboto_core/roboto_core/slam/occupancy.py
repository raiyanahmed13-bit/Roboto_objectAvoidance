"""Log-odds occupancy grid mapping.

Textbook inverse-sensor-model mapping, with two details that are easy to
omit and expensive to omit:

1. **`observed_count` is maintained alongside the log-odds.**
   Discrepancy detection is not allowed to compare a cell the robot never
   sensed -- without this mask, every unvisited cell registers as
   "the prior claims an obstacle and I see none", and the headline metric
   degenerates into "the robot did not go there". This grid is what
   separates *disagreement* from *absence of evidence*.

2. **Max-range beams still carve free space.**
   A beam that returns nothing is not missing data: it is positive evidence
   that the space in front of the sensor is empty. Dropping those beams
   entirely leaves the map full of phantom unknown wedges. We trace them as
   free (to 95 % of max range, staying clear of the range boundary) but
   record no hit.

Rays are traced by fixed-step sampling rather than Bresenham. The cells
touched are deduplicated per scan, so each cell receives exactly one update
per scan regardless of how many samples landed in it -- which is the
property that actually matters, and it vectorizes over all beams at once
instead of looping in Python.
"""

from __future__ import annotations

import numpy as np

from ..frames import GridSpec
from .scan import Scan


class OccupancyGrid:
    """Log-odds occupancy over a fixed `GridSpec`.

    The grid geometry is deliberately supplied by the caller rather than
    grown on demand: it must match the GIS prior exactly (same origin, same
    resolution) so discrepancy detection is a direct array comparison with
    no resampling.
    """

    def __init__(self, grid: GridSpec,
                 l_occ: float = 0.85, l_free: float = -0.4,
                 l_min: float = -5.0, l_max: float = 5.0,
                 prior: float = 0.0, free_standoff_m: float = 0.30):
        self.grid = grid
        self.l_occ = float(l_occ)
        self.l_free = float(l_free)
        self.l_min = float(l_min)
        self.l_max = float(l_max)

        # How far short of the endpoint the free-space trace stops.
        #
        # This is a robustness parameter, not a geometric one, and it
        # matters far more than it looks. Free space is carved by EVERY
        # beam of every scan, while a given wall cell is hit only
        # occasionally -- so free evidence accumulates much faster than
        # occupied evidence. Any pose error therefore lets the trace cut
        # into obstacles and erode them. Measured over a 344 m mission,
        # with the stand-off at half a cell:
        #
        #     pose error   true wall cells wrongly marked free
        #        0.00 m                 0.0 %
        #        0.10 m                 3.6 %
        #        0.25 m                34.2 %
        #        0.50 m                62.3 %
        #
        # Eroded walls are catastrophic downstream: every one reads as a
        # PHANTOM obstacle and floods discrepancy detection with false
        # positives on real buildings. The cost of a larger stand-off is a
        # thin unobserved shell around obstacles, which is harmless.
        self.free_standoff_m = float(free_standoff_m)

        # "Sticky obstacle" damping: reduce the free update on cells that
        # are already confident obstacles.
        #
        # DISABLED BY DEFAULT (factor 1.0) BECAUSE IT MEASURABLY HURT.
        # The idea was that free space is carved by every beam while a wall
        # cell is hit only occasionally, so pose error erodes real walls.
        # Damping was meant to protect them. Measured over a 344 m mission:
        #
        #     sticky_factor   ATE RMSE   true walls wrongly cleared
        #          1.00 (off)   0.64 m              72.5 %
        #          0.50         1.55 m              89.4 %
        #          0.25         2.30 m              92.8 %
        #
        # Worse on both counts. The reason is that damping also protects
        # SPURIOUS obstacles created by mis-registered hits: ghost walls
        # accumulate, scan matching degrades against a cluttered map, and
        # the resulting pose error erodes true walls faster than the damping
        # saves them. The dominant failure is not "real walls get cleared",
        # it is "bad poses create obstacles that can never be removed".
        #
        # Kept as a parameter because it is a legitimate technique when pose
        # error is small, and because the negative result is worth showing.
        self.l_sticky = 2.0
        self.sticky_factor = 1.0

        self.log_odds = np.full(grid.shape, float(prior), dtype=np.float32)
        # uint16 saturates at 65535 observations, which no run approaches.
        self.observed_count = np.zeros(grid.shape, dtype=np.uint16)

    # ---- readout ---------------------------------------------------------

    @property
    def prob(self) -> np.ndarray:
        """Occupancy probability in [0, 1]."""
        return 1.0 - 1.0 / (1.0 + np.exp(self.log_odds))

    def occupied(self, l_thresh: float = 2.0) -> np.ndarray:
        """Confidently occupied cells. l=2.0 is p~0.88."""
        return self.log_odds >= l_thresh

    def free(self, l_thresh: float = 2.0) -> np.ndarray:
        """Confidently free cells."""
        return self.log_odds <= -l_thresh

    def observed(self, min_count: int = 5) -> np.ndarray:
        """Cells with enough evidence to make a claim about.

        Every discrepancy metric must be masked by this.
        """
        return self.observed_count >= min_count

    @property
    def coverage(self) -> float:
        """Fraction of the grid ever sensed. Report this beside any map metric."""
        return float((self.observed_count > 0).mean())

    # ---- mapping ---------------------------------------------------------

    def integrate_scan(self, pose, scan: Scan, max_range: float | None = None) -> None:
        """Fold one scan, taken from `pose` = (x, y, theta), into the map."""
        x, y, th = np.asarray(pose, dtype=np.float64)
        rmax = float(max_range if max_range is not None else scan.range_max)
        if not np.isfinite(rmax):
            raise ValueError("integrate_scan needs a finite max range")

        ranges = scan.ranges
        angles = scan.angles + th

        hit = np.isfinite(ranges) & (ranges > scan.range_min) & (ranges <= rmax)
        # No-return beams are evidence of free space, not missing data.
        traced = np.where(hit, ranges, 0.95 * rmax)

        free_idx, hit_idx = self._trace(x, y, angles, traced, hit)

        # Free first, then occupied: a cell that is both (the endpoint) must
        # end up occupied, and applying them in this order guarantees that
        # without needing a set difference.
        flat = self.log_odds.reshape(-1)
        if free_idx.size:
            cur = flat[free_idx]
            damped = np.where(cur >= self.l_sticky,
                              self.l_free * self.sticky_factor, self.l_free)
            flat[free_idx] = cur + damped
        if hit_idx.size:
            flat[hit_idx] += self.l_occ
        np.clip(self.log_odds, self.l_min, self.l_max, out=self.log_odds)

        seen = np.unique(np.concatenate([free_idx, hit_idx])) if hit_idx.size else free_idx
        if seen.size:
            counts = self.observed_count.reshape(-1)
            np.add.at(counts, seen, 1)
            np.minimum(counts, 65535, out=counts)

    def _trace(self, x, y, angles, ranges, hit):
        """Cell indices touched: (free cells, endpoint cells), deduplicated.

        Returns flat indices into the grid so the caller can use fast
        flat-array accumulation.
        """
        g = self.grid
        step = 0.5 * g.resolution
        rmax = float(ranges.max()) if ranges.size else 0.0
        if rmax <= 0:
            return np.empty(0, np.int64), np.empty(0, np.int64)

        n_steps = int(np.ceil(rmax / step))
        t = np.arange(n_steps, dtype=np.float64) * step        # (S,) from 0

        px = x + np.cos(angles)[:, None] * t[None, :]          # (B, S)
        py = y + np.sin(angles)[:, None] * t[None, :]

        # Stop short of the surface so the endpoint cell is not marked free
        # by its own ray, and so modest pose error cannot erode obstacles.
        standoff = max(self.free_standoff_m, 0.5 * g.resolution)
        valid = t[None, :] < (ranges[:, None] - standoff)

        row, col = g.world_to_cell(px, py)
        inside = g.in_bounds(row, col)
        m = valid & inside
        free_idx = np.unique((row[m] * g.width + col[m]).astype(np.int64))

        hit_idx = np.empty(0, np.int64)
        if np.any(hit):
            ex = x + np.cos(angles[hit]) * ranges[hit]
            ey = y + np.sin(angles[hit]) * ranges[hit]
            hr, hc = g.world_to_cell(ex, ey)
            ok = g.in_bounds(hr, hc)
            if np.any(ok):
                hit_idx = np.unique((hr[ok] * g.width + hc[ok]).astype(np.int64))

        return free_idx, hit_idx

    # ---- comparison ------------------------------------------------------

    def iou(self, truth_occ: np.ndarray, l_thresh: float = 2.0,
            observed_only: bool = True) -> float:
        """Intersection-over-union of occupied cells against a ground truth.

        `observed_only` restricts to sensed cells, which is the honest
        comparison: unvisited cells say nothing about map quality.
        """
        pred = self.occupied(l_thresh)
        truth = np.asarray(truth_occ, dtype=bool)
        if truth.shape != pred.shape:
            raise ValueError(f"truth {truth.shape} != map {pred.shape}")
        if observed_only:
            m = self.observed()
            pred, truth = pred & m, truth & m
        union = np.logical_or(pred, truth).sum()
        return float(np.logical_and(pred, truth).sum() / union) if union else 0.0
