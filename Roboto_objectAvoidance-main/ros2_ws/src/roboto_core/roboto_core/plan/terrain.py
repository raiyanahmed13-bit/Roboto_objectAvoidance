"""Elevation and its gradients, without a GIS stack.

WHY THIS EXISTS SEPARATELY FROM THE GIS PIPELINE
------------------------------------------------
The planner's terrain cost is directional -- climbing a grade is expensive,
descending it is nearly free, crossing it sideways is a rollover risk --
so it needs the gradient COMPONENTS (dz/dx, dz/dy), not a scalar slope.

Those come from the DEM, which lives in a GeoTIFF and needs rasterio to
read. rasterio is a GIS-pipeline dependency and cannot be imported in the
ROS environment on this machine, so the live node simply ran with
`terrain = None` and planned on distance and obstacles alone. The offline
experiments, which do have rasterio, planned WITH slope. The two systems
were therefore solving different problems and their routes could not be
compared.

The fix is to commit the DEM a second time in a form the robot can read:
`dem_local.npz` is plain numpy, so this module needs nothing but numpy and
scipy. The GeoTIFF stays the source of truth for the GIS pipeline; the npz
is derived from it by build_slope and never edited by hand.

Storage is at the DEM's native ~1 m resolution, NOT the 0.10 m planning
grid. Upsampling here would write three 3000x3000 float arrays -- about
108 MB -- carrying no information the 1 m source does not already have.
`sample()` interpolates onto whatever grid the caller wants, in
milliseconds.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..frames import GridSpec


@dataclass
class TerrainGrid:
    """Elevation and gradients on a metric grid, in ROS convention."""

    z: np.ndarray          # metres, [row, col], row 0 = south
    dzdx: np.ndarray       # rise per metre East
    dzdy: np.ndarray       # rise per metre North
    grid: GridSpec

    # ---- derived ---------------------------------------------------------

    @property
    def slope_deg(self) -> np.ndarray:
        return np.degrees(np.arctan(np.hypot(self.dzdx, self.dzdy)))

    @property
    def relief(self) -> float:
        return float(np.nanmax(self.z) - np.nanmin(self.z))

    # ---- construction ----------------------------------------------------

    @classmethod
    def from_local_dem(cls, z_ros: np.ndarray, grid: GridSpec) -> "TerrainGrid":
        """Differentiate an elevation raster already in ROS convention.

        In ROS convention row increases northward, so np.gradient's first
        output is d/dy directly -- no sign flip. That is a quiet benefit of
        normalising row order once at the I/O boundary.
        """
        dzdy, dzdx = np.gradient(np.asarray(z_ros, dtype=np.float64),
                                 grid.resolution, grid.resolution)
        return cls(z=np.asarray(z_ros), dzdx=dzdx, dzdy=dzdy, grid=grid)

    # ---- numpy-only I/O --------------------------------------------------

    def save_npz(self, path: str | Path) -> Path:
        """Write elevation plus grid geometry. Gradients are NOT stored.

        They are a pure function of z and the resolution, so persisting them
        would triple the file for nothing and create a second thing that can
        go stale against the first.
        """
        path = Path(path)
        np.savez_compressed(
            path,
            z=np.asarray(self.z, dtype=np.float32),
            origin=np.array([self.grid.origin_x, self.grid.origin_y],
                            dtype=np.float64),
            resolution=np.float64(self.grid.resolution),
        )
        return path

    @classmethod
    def from_npz(cls, path: str | Path) -> "TerrainGrid":
        d = np.load(str(path))
        z = d["z"].astype(np.float64)
        grid = GridSpec(
            origin_x=float(d["origin"][0]),
            origin_y=float(d["origin"][1]),
            resolution=float(d["resolution"]),
            width=z.shape[1],
            height=z.shape[0],
        )
        return cls.from_local_dem(z, grid)

    # ---- sampling --------------------------------------------------------

    def sample(self, target: GridSpec):
        """Bilinearly interpolate (dzdx, dzdy, z) onto `target`.

        Returns float32 arrays shaped like `target`, in ROS convention.
        """
        from scipy.ndimage import map_coordinates

        rows = np.arange(target.height)
        cols = np.arange(target.width)
        x, _ = target.cell_to_world(0, cols)
        _, y = target.cell_to_world(rows, 0)

        # Fractional index into this grid. The -0.5 converts a cell-centre
        # world coordinate into map_coordinates' cell-index space.
        fc = (np.asarray(x) - self.grid.origin_x) / self.grid.resolution - 0.5
        fr = (np.asarray(y) - self.grid.origin_y) / self.grid.resolution - 0.5
        RR, CC = np.meshgrid(fr, fc, indexing="ij")
        coords = np.stack([RR, CC])

        out = [
            map_coordinates(a, coords, order=1, mode="nearest").astype(np.float32)
            for a in (self.dzdx, self.dzdy, self.z)
        ]
        return out[0], out[1], out[2]

    def along_path_slope(self, dzdx, dzdy, heading_rad):
        """Signed slope in radians along `heading_rad`. Positive = uphill.

        This is what makes the planning graph directed.
        """
        return np.arctan(dzdx * np.cos(heading_rad) + dzdy * np.sin(heading_rad))
