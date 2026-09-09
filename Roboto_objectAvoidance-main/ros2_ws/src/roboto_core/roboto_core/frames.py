"""Grid geometry and raster I/O -- and the one place row order is flipped.

This module is shared by the offline GIS pipeline and the online ROS nodes,
which is deliberate: the frame contract only holds if there is exactly ONE
implementation of it. `tools.gis_pipeline.common.grid` re-exports from here.

THE CONVENTION
--------------
Every 2D array in this project, once it leaves this module, is in **ROS
convention**:

    arr[row, col]      row 0 = ymin (SOUTH),  col 0 = xmin (WEST)
                       y increases with row,  x increases with col

GDAL/rasterio use the opposite row order (row 0 = NORTH), and PGM images
store row 0 = top = north as well. Those two conversions happen HERE, in
`from_raster` / `to_raster`, and nowhere else in the codebase.

If you ever find yourself adding a `np.flipud` outside this file, you are
about to introduce a bug that will look like a working map and produce
garbage metrics. Fix the caller instead.

Dependencies are numpy + yaml only. rasterio is imported lazily inside
`rasterio_transform` so ROS nodes never need GDAL.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import yaml

if TYPE_CHECKING:                       # pragma: no cover
    from tools.gis_pipeline.common.crs import SiteSpec

# ROS map_server PGM greyscale values.
PGM_OCCUPIED = 0     # black
PGM_FREE = 254       # white
PGM_UNKNOWN = 205    # mid grey


@dataclass(frozen=True)
class GridSpec:
    """A metric raster grid in ROS convention.

    `origin_x`/`origin_y` are the coordinates of the **bottom-left corner**
    of the bottom-left cell -- matching the `origin:` field in a ROS
    map_server YAML, which is a corner and NOT the centre. That distinction
    is worth 150 m of silent offset if you get it wrong.
    """

    origin_x: float
    origin_y: float
    resolution: float
    width: int      # columns, spans x
    height: int     # rows, spans y

    # ---- construction ----------------------------------------------------

    @classmethod
    def from_site(cls, site: "SiteSpec") -> "GridSpec":
        """Planning grid for a site. Duck-typed so this module stays free of
        the pyproj dependency that SiteSpec carries."""
        return cls(
            origin_x=site.xmin,
            origin_y=site.ymin,
            resolution=site.resolution_m,
            width=site.width_cells,
            height=site.height_cells,
        )

    # ---- extent ----------------------------------------------------------

    @property
    def xmax(self) -> float:
        return self.origin_x + self.width * self.resolution

    @property
    def ymax(self) -> float:
        return self.origin_y + self.height * self.resolution

    @property
    def shape(self) -> tuple[int, int]:
        return (self.height, self.width)

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        """(xmin, ymin, xmax, ymax) in metres."""
        return (self.origin_x, self.origin_y, self.xmax, self.ymax)

    # ---- coordinate conversion ------------------------------------------

    def world_to_cell(self, x, y):
        """Metric (x, y) -> integer (row, col). Vectorized over arrays."""
        col = np.floor((np.asarray(x) - self.origin_x) / self.resolution).astype(np.int64)
        row = np.floor((np.asarray(y) - self.origin_y) / self.resolution).astype(np.int64)
        return row, col

    def cell_to_world(self, row, col):
        """Integer (row, col) -> metric (x, y) at the CELL CENTRE."""
        x = self.origin_x + (np.asarray(col) + 0.5) * self.resolution
        y = self.origin_y + (np.asarray(row) + 0.5) * self.resolution
        return x, y

    def in_bounds(self, row, col):
        row = np.asarray(row)
        col = np.asarray(col)
        return (row >= 0) & (row < self.height) & (col >= 0) & (col < self.width)

    # ---- interop ---------------------------------------------------------

    def rasterio_transform(self):
        """North-up Affine for rasterio rasterize/reproject.

        Arrays produced with this transform are in RASTER convention
        (row 0 = north) and must be passed through `from_raster` before
        being used anywhere else.
        """
        from rasterio.transform import from_origin

        return from_origin(self.origin_x, self.ymax, self.resolution, self.resolution)

    def assert_compatible(self, other: "GridSpec", what: str = "grid") -> None:
        """Guard for anything comparing two grids cell-by-cell."""
        if not np.isclose(self.resolution, other.resolution):
            raise ValueError(
                f"{what}: resolution mismatch {self.resolution} vs {other.resolution}"
            )
        if not (np.isclose(self.origin_x, other.origin_x)
                and np.isclose(self.origin_y, other.origin_y)):
            raise ValueError(
                f"{what}: origin mismatch "
                f"({self.origin_x}, {self.origin_y}) vs ({other.origin_x}, {other.origin_y})"
            )
        if self.shape != other.shape:
            raise ValueError(f"{what}: shape mismatch {self.shape} vs {other.shape}")

    def to_map_yaml(self, image_name: str) -> dict:
        """ROS map_server YAML. `origin` is [x, y, yaw] of the BOTTOM-LEFT corner."""
        return {
            "image": image_name,
            "resolution": float(self.resolution),
            "origin": [float(self.origin_x), float(self.origin_y), 0.0],
            "negate": 0,
            "occupied_thresh": 0.65,
            "free_thresh": 0.196,
        }


# ---------------------------------------------------------------------------
# THE FLIP. Both functions below are the only row-order conversions allowed.
# ---------------------------------------------------------------------------

def from_raster(arr: np.ndarray) -> np.ndarray:
    """Raster convention (row 0 = north) -> ROS convention (row 0 = south)."""
    return np.flipud(np.asarray(arr)).copy()


def to_raster(arr: np.ndarray) -> np.ndarray:
    """ROS convention (row 0 = south) -> raster convention (row 0 = north)."""
    return np.flipud(np.asarray(arr)).copy()


# ---------------------------------------------------------------------------
# PGM + YAML I/O (ROS map_server format)
# ---------------------------------------------------------------------------

def save_occupancy(path: str | Path, occ: np.ndarray, spec: GridSpec) -> Path:
    """Write a binary occupancy grid as map_server PGM + YAML.

    `occ` is in ROS convention: True/1 = occupied. Written as P5 binary PGM.
    """
    path = Path(path)
    occ = np.asarray(occ)
    if occ.shape != spec.shape:
        raise ValueError(f"occ shape {occ.shape} != grid shape {spec.shape}")

    img = np.where(occ.astype(bool), PGM_OCCUPIED, PGM_FREE).astype(np.uint8)
    _write_pgm(path, to_raster(img))

    ymlpath = path.with_suffix(".yaml")
    with open(ymlpath, "w", encoding="utf-8") as fh:
        yaml.safe_dump(spec.to_map_yaml(path.name), fh, sort_keys=False)
    return ymlpath


def load_occupancy(yaml_path: str | Path) -> tuple[np.ndarray, GridSpec]:
    """Read a map_server PGM + YAML pair back into ROS convention."""
    yaml_path = Path(yaml_path)
    with open(yaml_path, "r", encoding="utf-8") as fh:
        meta = yaml.safe_load(fh)

    img = from_raster(_read_pgm(yaml_path.parent / meta["image"]))
    ox, oy = meta["origin"][0], meta["origin"][1]
    spec = GridSpec(
        origin_x=float(ox),
        origin_y=float(oy),
        resolution=float(meta["resolution"]),
        width=img.shape[1],
        height=img.shape[0],
    )
    # p_occ = (255 - pixel) / 255, thresholded per map_server semantics.
    p_occ = (255.0 - img.astype(np.float64)) / 255.0
    return p_occ >= float(meta.get("occupied_thresh", 0.65)), spec


def _write_pgm(path: str | Path, img: np.ndarray) -> None:
    """Binary P5 PGM. `img` must already be in raster (top-down) order."""
    img = np.ascontiguousarray(img, dtype=np.uint8)
    h, w = img.shape
    with open(path, "wb") as fh:
        fh.write(f"P5\n{w} {h}\n255\n".encode("ascii"))
        fh.write(img.tobytes())


def _read_pgm(path: str | Path) -> np.ndarray:
    """Binary P5 PGM reader. Returns raster (top-down) order."""
    data = Path(path).read_bytes()
    if not data[:2] == b"P5":
        raise ValueError(f"{path}: not a binary P5 PGM")

    # Header: P5, then width height maxval, with '#' comments allowed between.
    pos, fields = 2, []
    while len(fields) < 3:
        while pos < len(data) and data[pos:pos + 1].isspace():
            pos += 1
        if data[pos:pos + 1] == b"#":
            while pos < len(data) and data[pos:pos + 1] not in (b"\n", b"\r"):
                pos += 1
            continue
        start = pos
        while pos < len(data) and not data[pos:pos + 1].isspace():
            pos += 1
        fields.append(int(data[start:pos]))
    pos += 1  # single whitespace byte after maxval

    w, h, maxval = fields
    if maxval > 255:
        raise ValueError(f"{path}: 16-bit PGM not supported")
    return np.frombuffer(data, dtype=np.uint8, count=w * h, offset=pos).reshape(h, w)
