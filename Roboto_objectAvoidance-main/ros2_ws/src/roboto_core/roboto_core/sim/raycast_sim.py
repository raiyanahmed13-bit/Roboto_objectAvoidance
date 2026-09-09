"""A fast 2D lidar simulator that raycasts against an occupancy grid.

This exists for three reasons, and it is worth being explicit about all
three because they justify the effort:

1. **Development loop.** Gazebo-in-the-loop iteration costs 10-30 s per
   cycle. Tuning a scan matcher takes hundreds of cycles. Here a 300 m
   trajectory simulates in under a second, so SLAM work is not gated on the
   simulator being up, fast, or even installed.
2. **Unit-test fixture.** Ground-truth poses are exact and noise is seeded,
   so tests can assert numeric accuracy rather than eyeballing RViz.
3. **Ripcord.** If Gazebo Fortress cannot render under WSL2, this becomes
   the primary simulator and the project continues unharmed -- real OSM
   geometry, real DEM slope, real noise models, real ground truth.

It raycasts against the SAME occupancy grid the GIS pipeline produced, in
the same frame, so results transfer directly.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..frames import GridSpec, load_occupancy
from ..slam.scan import Scan


@dataclass(frozen=True)
class LidarSpec:
    """2D lidar geometry and noise. Mirrors the Gazebo sensor definition."""

    num_beams: int = 180
    fov_deg: float = 270.0
    range_min: float = 0.15
    range_max: float = 12.0
    rate_hz: float = 10.0
    range_noise_std: float = 0.02
    dropout_prob: float = 0.0        # beams that return nothing at random

    @classmethod
    def from_site(cls, site) -> "LidarSpec":
        c = site.raw["lidar"]
        return cls(
            num_beams=int(c["num_beams"]),
            fov_deg=float(c["fov_deg"]),
            range_min=float(c["range_min_m"]),
            range_max=float(c["range_max_m"]),
            rate_hz=float(c["rate_hz"]),
            range_noise_std=float(c["range_noise_std_m"]),
        )

    @property
    def angles(self) -> np.ndarray:
        """Sensor-frame bearings, symmetric about straight ahead."""
        half = np.radians(self.fov_deg) / 2.0
        # endpoint=False so a 360 deg FOV does not duplicate a beam.
        full = np.isclose(self.fov_deg, 360.0)
        return np.linspace(-half, half, self.num_beams, endpoint=not full)


class RaycastWorld:
    """An occupancy grid you can shoot rays at.

    Rays that leave the grid return no measurement (inf) rather than hitting
    a phantom wall at the boundary -- there is no obstacle there, we simply
    stopped modelling. With a 12 m sensor inside a 300 m extent this only
    affects the outermost 12 m, and inventing a wall would corrupt the
    occupancy map far worse.
    """

    def __init__(self, occ: np.ndarray, grid: GridSpec):
        if occ.shape != grid.shape:
            raise ValueError(f"occ shape {occ.shape} != grid shape {grid.shape}")
        self.occ = np.ascontiguousarray(occ, dtype=bool)
        self.grid = grid

    @classmethod
    def from_occupancy_yaml(cls, yaml_path: str | Path) -> "RaycastWorld":
        occ, grid = load_occupancy(yaml_path)
        return cls(occ, grid)

    # ---- queries ---------------------------------------------------------

    def is_occupied(self, x, y) -> np.ndarray:
        """Occupancy lookup. Out-of-bounds reads as free."""
        row, col = self.grid.world_to_cell(x, y)
        inside = self.grid.in_bounds(row, col)
        r = np.clip(row, 0, self.grid.height - 1)
        c = np.clip(col, 0, self.grid.width - 1)
        return self.occ[r, c] & inside

    def raycast(self, x: float, y: float, angles: np.ndarray,
                max_range: float, step: float | None = None) -> np.ndarray:
        """Range to the first occupied cell along each world-frame bearing.

        Uses fixed-step sampling rather than a true DDA: it vectorizes over
        all beams at once in numpy, which is far faster in practice than a
        per-ray Python DDA loop, and the quantization is bounded by `step`.

        Returns inf where nothing was hit within `max_range`.
        """
        if step is None:
            # Quarter-cell sampling: max quantization error 1.25 cm at a
            # 0.10 m grid, comfortably below the 2 cm sensor noise.
            step = 0.25 * self.grid.resolution

        n_steps = int(np.ceil(max_range / step))
        t = (np.arange(1, n_steps + 1) * step)                    # (S,)

        ca = np.cos(angles)[:, None]                              # (B, 1)
        sa = np.sin(angles)[:, None]
        px = x + ca * t[None, :]                                  # (B, S)
        py = y + sa * t[None, :]

        row, col = self.grid.world_to_cell(px, py)
        inside = self.grid.in_bounds(row, col)
        r = np.clip(row, 0, self.grid.height - 1)
        c = np.clip(col, 0, self.grid.width - 1)
        blocked = self.occ[r, c] & inside                         # (B, S)

        any_hit = blocked.any(axis=1)
        first = np.argmax(blocked, axis=1)

        # The true surface lies between samples first-1 and first, so report
        # the midpoint: removes the systematic +step/2 bias of taking t[first].
        rng = np.where(any_hit, t[first] - 0.5 * step, np.inf)
        return rng


class Lidar2D:
    """Turns a pose into a noisy `Scan` against a `RaycastWorld`."""

    def __init__(self, world: RaycastWorld, spec: LidarSpec | None = None,
                 rng: np.random.Generator | int | None = None):
        self.world = world
        self.spec = spec or LidarSpec()
        self.rng = (rng if isinstance(rng, np.random.Generator)
                    else np.random.default_rng(rng))

    def scan(self, pose, stamp: float = 0.0, noise: bool = True) -> Scan:
        """Simulate one scan from `pose` = (x, y, theta)."""
        x, y, th = np.asarray(pose, dtype=np.float64)
        s = self.spec

        ranges = self.world.raycast(x, y, s.angles + th, s.range_max)

        if noise and s.range_noise_std > 0:
            hit = np.isfinite(ranges)
            ranges = ranges.copy()
            ranges[hit] += self.rng.normal(0.0, s.range_noise_std, hit.sum())

        if noise and s.dropout_prob > 0:
            drop = self.rng.random(ranges.shape) < s.dropout_prob
            ranges = np.where(drop, np.inf, ranges)

        # Sub-minimum returns are not reported by real drivers.
        ranges = np.where(ranges < s.range_min, np.inf, ranges)
        ranges = np.where(ranges > s.range_max, np.inf, ranges)

        return Scan(
            ranges=ranges,
            angles=s.angles.copy(),
            stamp=stamp,
            range_min=s.range_min,
            range_max=s.range_max,
        )
