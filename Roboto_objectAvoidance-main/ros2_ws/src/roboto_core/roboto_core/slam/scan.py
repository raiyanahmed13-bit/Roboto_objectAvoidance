"""2D laser scan container.

Deliberately mirrors ``sensor_msgs/LaserScan`` so the ROS shim is a field
copy with no reinterpretation, while staying a plain dataclass that the
offline tests can build without ROS installed.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .transforms2d import transform_points


@dataclass
class Scan:
    """Ranges plus the bearings they were measured along.

    `ranges` may contain inf/nan for no-return beams -- that is normal, not
    an error. Use `.valid` or `.points()`, both of which drop them.
    """

    ranges: np.ndarray
    angles: np.ndarray            # sensor-frame bearings, radians
    stamp: float = 0.0
    range_min: float = 0.0
    range_max: float = np.inf
    meta: dict = field(default_factory=dict)

    def __post_init__(self):
        self.ranges = np.asarray(self.ranges, dtype=np.float64)
        self.angles = np.asarray(self.angles, dtype=np.float64)
        if self.ranges.shape != self.angles.shape:
            raise ValueError(
                f"ranges {self.ranges.shape} and angles {self.angles.shape} differ"
            )

    def __len__(self) -> int:
        return int(self.ranges.size)

    @property
    def valid(self) -> np.ndarray:
        """Beams that returned a usable range."""
        r = self.ranges
        return np.isfinite(r) & (r > self.range_min) & (r < self.range_max)

    def points(self, all_beams: bool = False) -> np.ndarray:
        """(N, 2) endpoints in the SENSOR frame.

        By default only valid returns. `all_beams=True` keeps the array
        aligned with `ranges`, which the free-space raycaster needs since it
        must trace even the no-return beams out to max range.
        """
        m = slice(None) if all_beams else self.valid
        r, a = self.ranges[m], self.angles[m]
        return np.stack([r * np.cos(a), r * np.sin(a)], axis=-1)

    def points_in(self, pose) -> np.ndarray:
        """(N, 2) valid endpoints transformed into `pose`'s parent frame."""
        return transform_points(pose, self.points())

    def subsample(self, n: int) -> "Scan":
        """Evenly thin to at most `n` beams.

        Scan matching against 180 beams is already fast, but the correlative
        matcher's cost is linear in beam count, so thinning is the cheapest
        knob when tuning the accuracy/compute trade-off.
        """
        if n >= len(self) or n <= 0:
            return self
        idx = np.linspace(0, len(self) - 1, n).round().astype(int)
        return Scan(
            ranges=self.ranges[idx],
            angles=self.angles[idx],
            stamp=self.stamp,
            range_min=self.range_min,
            range_max=self.range_max,
            meta=dict(self.meta),
        )
