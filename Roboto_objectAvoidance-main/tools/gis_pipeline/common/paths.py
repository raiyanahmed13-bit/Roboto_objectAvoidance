"""Artifact locations and vector I/O.

Vector data is stored as GeoPackage, NOT GeoJSON.

GeoJSON (RFC 7946) is *defined* to be WGS84 lon/lat. Writing a metric CRS
to it makes GeoPandas silently drop the CRS, and reading it back yields
coordinates that are correct in value but labelled EPSG:4326. Everything
downstream that touches `.area`, `.centroid`, `.buffer` or `.to_crs` then
computes in the wrong units without raising. GeoPackage round-trips an
arbitrary CRS exactly.
"""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd

from .crs import SiteSpec


def site_dir(site: SiteSpec) -> Path:
    return Path("data") / f"site_{site.name}"


def raw_dir(site: SiteSpec) -> Path:
    d = site_dir(site) / "raw"
    d.mkdir(parents=True, exist_ok=True)
    return d


def derived_dir(site: SiteSpec) -> Path:
    d = site_dir(site) / "derived"
    d.mkdir(parents=True, exist_ok=True)
    return d


def world_dir(site: SiteSpec) -> Path:
    d = site_dir(site) / "world"
    d.mkdir(parents=True, exist_ok=True)
    return d


def vector_path(site: SiteSpec, name: str) -> Path:
    """Path to a vector layer, e.g. name='buildings'."""
    return raw_dir(site) / f"{name}.gpkg"


def save_vector(gdf: gpd.GeoDataFrame, site: SiteSpec, name: str) -> Path:
    """Write a layer in the site-local metric CRS, preserving that CRS."""
    if gdf.crs is None:
        raise ValueError(f"{name}: refusing to write a GeoDataFrame with no CRS")
    path = vector_path(site, name)
    gdf.to_file(path, layer=name, driver="GPKG")
    return path


def load_vector(site: SiteSpec, name: str) -> gpd.GeoDataFrame:
    """Read a layer and assert it really is in the site-local metric CRS."""
    path = vector_path(site, name)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} missing -- run `python -m tools.gis_pipeline.fetch_osm` first"
        )
    gdf = gpd.read_file(path, layer=name)

    if gdf.crs is None or gdf.crs.is_geographic:
        raise ValueError(
            f"{path} is in a geographic CRS ({gdf.crs}). Metric CRS required; "
            f"units would be degrees and every area/buffer silently wrong."
        )
    if not gdf.crs.equals(site.crs_local):
        raise ValueError(f"{path} CRS {gdf.crs} != site-local frame")
    return gdf
