"""Coordinate reference authority for the whole project.

Every WGS84 <-> local-metric conversion goes through this module. Nothing
else computes a projection, an offset, or a rotation. If you find yourself
writing `x - 150.0` somewhere else, that is the bug.

The invariant:

    local TM frame  ==  ROS `map` frame  ==  Gazebo world frame

with x = East, y = North, z = Up and the origin at (lat0, lon0) from
site.yaml.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from pyproj import CRS, Transformer


@dataclass(frozen=True)
class SiteSpec:
    """Parsed site.yaml. Immutable -- treat it as the project's constants."""

    name: str
    lat0: float
    lon0: float
    alt0: float
    xmin: float
    ymin: float
    xmax: float
    ymax: float
    resolution_m: float
    dem_resolution_m: float
    fetch_buffer_m: float
    spawn_x: float
    spawn_y: float
    spawn_yaw_deg: float
    raw: dict[str, Any]

    # ---- construction ----------------------------------------------------

    @classmethod
    def from_yaml(cls, path: str | Path) -> "SiteSpec":
        with open(path, "r", encoding="utf-8") as fh:
            d = yaml.safe_load(fh)

        origin = d["origin"]
        ext = d["extent_m"]
        spawn = d["robot_spawn"]

        spec = cls(
            name=d["name"],
            lat0=float(origin["lat"]),
            lon0=float(origin["lon"]),
            alt0=float(origin.get("alt_m", 0.0)),
            xmin=float(ext["xmin"]),
            ymin=float(ext["ymin"]),
            xmax=float(ext["xmax"]),
            ymax=float(ext["ymax"]),
            resolution_m=float(d["resolution_m"]),
            dem_resolution_m=float(d["dem_resolution_m"]),
            fetch_buffer_m=float(d.get("fetch_buffer_m", 50.0)),
            spawn_x=float(spawn["x"]),
            spawn_y=float(spawn["y"]),
            spawn_yaw_deg=float(spawn.get("yaw_deg", 0.0)),
            raw=d,
        )

        # The stored proj4 string is redundant with lat0/lon0. If they ever
        # disagree, downstream rasters silently misalign -- so fail loudly.
        stored = d.get("crs_local")
        if stored is not None and not _proj4_matches(stored, spec.proj4):
            raise ValueError(
                f"site.yaml crs_local disagrees with origin lat/lon.\n"
                f"  stored:  {stored}\n"
                f"  derived: {spec.proj4}\n"
                f"Fix one of them; they must describe the same projection."
            )
        return spec

    # ---- projection ------------------------------------------------------

    @property
    def goal(self) -> tuple[float, float]:
        """Where the demo drives to, from the site's own config.

        Every tool used to default this to (95, 110), which is open street
        in Pittsburgh and a lethal cell in Chicago -- so adding a site made
        the tools fail with "goal in lethal cell" while pointing at the
        wrong thing entirely. A goal is site geometry and belongs with the
        rest of it.
        """
        g = self.raw.get("mission", {}).get("goal")
        if g is None:
            raise KeyError(
                f"site '{self.name}' has no mission.goal; add it to the "
                f"site yaml (tools/pick_mission.py will suggest one)")
        return (float(g[0]), float(g[1]))

    @property
    def proj4(self) -> str:
        """Site-local Transverse Mercator, k=1, origin at (lat0, lon0)."""
        return (
            f"+proj=tmerc +lat_0={self.lat0} +lon_0={self.lon0} "
            f"+k=1 +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs"
        )

    @property
    def crs_local(self) -> CRS:
        return CRS.from_proj4(self.proj4)

    @property
    def crs_wgs84(self) -> CRS:
        return CRS.from_epsg(4326)

    def _tf(self, fwd: bool) -> Transformer:
        a, b = (self.crs_wgs84, self.crs_local) if fwd else (self.crs_local, self.crs_wgs84)
        # always_xy keeps argument order (lon, lat) / (x, y) -- without it
        # pyproj follows the CRS axis order and silently swaps lat/lon.
        return Transformer.from_crs(a, b, always_xy=True)

    def lla_to_local(self, lon, lat):
        """(lon, lat) degrees -> (x, y) metres East/North. Accepts arrays."""
        return self._tf(True).transform(lon, lat)

    def local_to_lla(self, x, y):
        """(x, y) metres East/North -> (lon, lat) degrees. Accepts arrays."""
        return self._tf(False).transform(x, y)

    # ---- extent ----------------------------------------------------------

    @property
    def width_m(self) -> float:
        return self.xmax - self.xmin

    @property
    def height_m(self) -> float:
        return self.ymax - self.ymin

    @property
    def width_cells(self) -> int:
        return int(round(self.width_m / self.resolution_m))

    @property
    def height_cells(self) -> int:
        return int(round(self.height_m / self.resolution_m))

    def bbox_local(self, buffer_m: float = 0.0):
        """(xmin, ymin, xmax, ymax) in local metres, optionally buffered."""
        b = buffer_m
        return (self.xmin - b, self.ymin - b, self.xmax + b, self.ymax + b)

    def bbox_wgs84(self, buffer_m: float | None = None):
        """(west, south, east, north) in degrees, for OSM/DEM fetches.

        Projects all four corners and takes the outer hull rather than
        transforming two corners -- a projected rectangle is not a rectangle
        in lat/lon, and using two corners clips the edges.
        """
        b = self.fetch_buffer_m if buffer_m is None else buffer_m
        xmin, ymin, xmax, ymax = self.bbox_local(b)
        xs = np.array([xmin, xmax, xmax, xmin])
        ys = np.array([ymin, ymin, ymax, ymax])
        lons, lats = self.local_to_lla(xs, ys)
        return (float(np.min(lons)), float(np.min(lats)),
                float(np.max(lons)), float(np.max(lats)))


def _proj4_matches(a: str, b: str) -> bool:
    """Compare proj4 strings by parameter set, ignoring order and spacing."""
    def norm(s: str) -> set[str]:
        out = set()
        for tok in s.replace("+", " +").split():
            if not tok.startswith("+"):
                continue
            if "=" in tok:
                k, v = tok.split("=", 1)
                try:
                    out.add(f"{k}={float(v):.9g}")   # 40.4187 == 40.41870
                except ValueError:
                    out.add(f"{k}={v}")
            else:
                out.add(tok)
        return out

    return norm(a) == norm(b)


def load_site(path: str | Path | None = None) -> SiteSpec:
    """Load site.yaml, defaulting to the copy beside this package."""
    if path is None:
        path = Path(__file__).resolve().parent.parent / "site.yaml"
    return SiteSpec.from_yaml(path)
