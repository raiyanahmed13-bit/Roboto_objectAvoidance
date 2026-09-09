"""Fetch OSM building footprints and the road network for the site.

Writes GeoJSON in the site-local metric CRS to data/site_<name>/raw/.
Results are cached on disk -- Overpass is slow and rate-limited, and there
is no reason to refetch static geometry on every pipeline run.

Usage:
    python -m tools.gis_pipeline.fetch_osm [--site PATH] [--force]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import geopandas as gpd
import osmnx as ox

from .common.crs import SiteSpec, load_site
from .common.paths import load_vector, save_vector, vector_path


def fetch_buildings(site: SiteSpec, force: bool = False) -> gpd.GeoDataFrame:
    """Building footprints, reprojected to local metres and cleaned."""
    out = vector_path(site, "buildings")
    if out.exists() and not force:
        print(f"  buildings: cached  {out}")
        return load_vector(site, "buildings")

    cfg = site.raw["buildings"]
    # osmnx 2.x takes bbox as (left, bottom, right, top) == (W, S, E, N),
    # which is exactly what bbox_wgs84() returns.
    gdf = ox.features_from_bbox(site.bbox_wgs84(), tags={"building": True})

    gdf = gdf[gdf.geometry.type.isin(["Polygon", "MultiPolygon"])].copy()
    gdf = gdf.to_crs(site.crs_local)
    gdf["geometry"] = gdf.geometry.make_valid()
    gdf = gdf.explode(index_parts=False)
    gdf = gdf[gdf.geometry.type == "Polygon"]

    gdf["height_m"] = _building_heights(gdf, cfg)
    gdf["geometry"] = gdf.geometry.simplify(cfg["simplify_tol_m"])
    gdf = gdf[gdf.geometry.area >= cfg["min_area_m2"]]

    # Keep only what is actually needed downstream; OSM tag soup is ~100
    # columns wide and GeoJSON chokes on the mixed types.
    gdf = gdf[["geometry", "height_m"]].reset_index(drop=True)

    save_vector(gdf, site, "buildings")
    print(f"  buildings: {len(gdf):4d} footprints -> {out}")
    return gdf


def _building_heights(gdf: gpd.GeoDataFrame, cfg: dict):
    """Height from `height`, else `building:levels` x storey height, else default."""
    import numpy as np
    import pandas as pd

    n = len(gdf)
    h = pd.Series(np.nan, index=gdf.index, dtype=float)

    if "height" in gdf.columns:
        h = h.fillna(pd.to_numeric(gdf["height"], errors="coerce"))
    if "building:levels" in gdf.columns:
        lv = pd.to_numeric(gdf["building:levels"], errors="coerce")
        h = h.fillna(lv * cfg["levels_to_m"])

    h = h.fillna(cfg["default_height_m"])
    return h.clip(lower=2.0, upper=60.0)


def fetch_roads(site: SiteSpec, force: bool = False) -> gpd.GeoDataFrame:
    """Drivable + walkable network as buffered polygons in local metres."""
    out = vector_path(site, "roads")
    if out.exists() and not force:
        print(f"  roads:     cached  {out}")
        return load_vector(site, "roads")

    import numpy as np
    import pandas as pd

    g = ox.graph_from_bbox(site.bbox_wgs84(), network_type="all", simplify=True)
    edges = ox.graph_to_gdfs(g, nodes=False, edges=True).to_crs(site.crs_local)

    default_w = site.raw["roads"]["default_width_m"]
    width = pd.to_numeric(edges.get("width"), errors="coerce") if "width" in edges else None
    if width is None:
        width = pd.Series(np.nan, index=edges.index, dtype=float)
    if "lanes" in edges.columns:
        lanes = pd.to_numeric(
            edges["lanes"].apply(lambda v: v[0] if isinstance(v, list) else v),
            errors="coerce",
        )
        width = width.fillna(lanes * 3.0)
    width = width.fillna(default_w).clip(2.0, 25.0)

    roads = edges[["geometry", "highway"]].copy()
    roads["width_m"] = width.values
    roads["geometry"] = roads.geometry.buffer(roads["width_m"] / 2.0, cap_style=2)
    roads["highway"] = roads["highway"].apply(
        lambda v: v[0] if isinstance(v, list) else v
    ).astype(str)
    roads = roads.reset_index(drop=True)

    save_vector(roads, site, "roads")
    print(f"  roads:     {len(roads):4d} segments  -> {out}")
    return roads


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--site", default=None, help="path to site.yaml")
    ap.add_argument("--force", action="store_true", help="ignore the disk cache")
    args = ap.parse_args()

    site = load_site(args.site)
    w, s, e, n = site.bbox_wgs84()
    print(f"site '{site.name}'  origin=({site.lat0}, {site.lon0})")
    print(f"  bbox WGS84: W={w:.5f} S={s:.5f} E={e:.5f} N={n:.5f}")

    b = fetch_buildings(site, args.force)
    fetch_roads(site, args.force)

    # Sanity signal on site choice: too few buildings makes for a poor lidar
    # environment and a weak discrepancy story.
    inside = b.cx[site.xmin:site.xmax, site.ymin:site.ymax]
    print(f"\n  {len(inside)} buildings inside the {site.width_m:.0f} m extent")
    if len(inside) < 20:
        print("  WARNING: sparse site -- consider moving the origin.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
