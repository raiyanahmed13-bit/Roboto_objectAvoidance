"""Likelihood field: a blurred occupancy map that scan matching maximises.

Scoring a scan directly against a binary occupancy grid gives a piecewise-
constant objective -- it is flat almost everywhere and jumps at cell
boundaries, so there is no gradient to follow and a search finds huge ties.
Blurring the occupancy probability with a Gaussian turns each obstacle cell
into a smooth basin, so a scan endpoint that lands *near* a wall scores
nearly as well as one that lands on it, and the objective becomes
informative between cells.

This is the field used by:
  * the correlative scan matcher (front end),
  * branch-and-bound loop closure, via the max-pooled pyramid,
  * SLAM-map to GIS-prior registration.

The pyramid stores, at level k, the MAXIMUM field value over each 2^k block.
That is what makes branch-and-bound admissible: the coarse score is an
upper bound on any fine score inside the block, so a branch whose upper
bound is below the best-so-far can be discarded without evaluating it.
"""

from __future__ import annotations

import numpy as np

from ..frames import GridSpec


class LikelihoodField:
    """Smoothed occupancy probability with a max-pooled resolution pyramid."""

    def __init__(self, field: np.ndarray, grid: GridSpec, levels: int = 1):
        if field.shape != grid.shape:
            raise ValueError(f"field {field.shape} != grid {grid.shape}")
        self.grid = grid
        self.field = np.ascontiguousarray(field, dtype=np.float32)
        self._pyramid = [self.field]
        for _ in range(max(0, levels - 1)):
            self._pyramid.append(_max_pool2(self._pyramid[-1]))

    # ---- construction ----------------------------------------------------

    @classmethod
    def from_occupancy(cls, occ: np.ndarray, grid: GridSpec,
                       sigma_m: float = 0.15, levels: int = 1) -> "LikelihoodField":
        """Distance-transform likelihood field: exp(-d^2 / 2 sigma^2).

        `d` is the metric distance from each cell to the nearest occupied
        cell, so the field is exactly 1.0 on and inside any obstacle and
        decays smoothly outside it.

        NOT a Gaussian blur of the occupancy grid. Blurring peaks in the
        INTERIOR of a solid building (where the kernel sums many occupied
        cells) and is only about half that on the wall surface -- so a
        perfect, noiseless scan scores ~0.5 while a pose shoved half a metre
        into the walls scores ~0.7, and the matcher happily converges to the
        wrong pose. Measured exactly that before switching: truth 0.498,
        spurious optimum 0.699, offset 0.53 m.

        sigma is in METRES and converted using the grid resolution, so the
        smoothing is a physical length (about the sensor noise) rather than
        a pixel count that changes meaning with resolution.
        """
        from scipy.ndimage import distance_transform_edt

        occupied = np.asarray(occ) > 0.5
        if not occupied.any():
            return cls(np.zeros(grid.shape, np.float32), grid, levels=levels)

        dist = distance_transform_edt(~occupied) * grid.resolution
        sigma = max(float(sigma_m), 1e-6)
        fld = np.exp(-(dist ** 2) / (2.0 * sigma ** 2)).astype(np.float32)
        return cls(fld, grid, levels=levels)

    @classmethod
    def from_slam_map(cls, occ_grid, l_thresh: float = 0.0,
                      sigma_m: float = 0.15, levels: int = 1) -> "LikelihoodField":
        """Build from a live `OccupancyGrid` (its log-odds, thresholded)."""
        return cls.from_occupancy(
            (occ_grid.log_odds > l_thresh).astype(np.float32),
            occ_grid.grid, sigma_m=sigma_m, levels=levels)

    # ---- scoring ---------------------------------------------------------

    def level(self, k: int) -> np.ndarray:
        return self._pyramid[min(k, len(self._pyramid) - 1)]

    def resolution(self, k: int = 0) -> float:
        return self.grid.resolution * (2 ** k)

    def score_points(self, pts: np.ndarray) -> float:
        """Mean field value under world-frame points. Range [0, 1].

        The mean rather than the sum: a sum rewards scans that happen to
        have more valid returns, which would make scores incomparable
        between scans and useless as a loop-closure threshold.
        """
        if len(pts) == 0:
            return 0.0
        row, col = self.grid.world_to_cell(pts[:, 0], pts[:, 1])
        inside = self.grid.in_bounds(row, col)
        if not inside.any():
            return 0.0
        r = np.clip(row, 0, self.grid.height - 1)
        c = np.clip(col, 0, self.grid.width - 1)
        return float(np.where(inside, self.field[r, c], 0.0).mean())

    def cell_indices(self, pts: np.ndarray, k: int = 0):
        """World points -> integer indices into pyramid level `k`."""
        res = self.resolution(k)
        col = np.floor((pts[:, 0] - self.grid.origin_x) / res).astype(np.int64)
        row = np.floor((pts[:, 1] - self.grid.origin_y) / res).astype(np.int64)
        return row, col


def _max_pool2(a: np.ndarray) -> np.ndarray:
    """2x2 max pool, padding with the edge value for odd sizes.

    Max (not mean) is required: the pyramid must upper-bound the fine level
    for branch-and-bound to be admissible. Mean pooling would let the bound
    fall below a real fine-level score and prune the true optimum.
    """
    h, w = a.shape
    ph, pw = h + (h & 1), w + (w & 1)
    if (ph, pw) != (h, w):
        padded = np.zeros((ph, pw), dtype=a.dtype)
        padded[:h, :w] = a
        a = padded
    return a.reshape(ph // 2, 2, pw // 2, 2).max(axis=(1, 3))
