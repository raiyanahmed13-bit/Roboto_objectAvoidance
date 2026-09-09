"""Fetch a Digital Elevation Model covering the site.

Primary source is the USGS 3DEP ImageServer, queried directly over HTTPS.
We deliberately do NOT use py3dep here: its async HTTP stack fails DNS
resolution in some sandboxed/VPN environments even when the host resolves
fine, and the REST call it wraps is a single URL. Fewer moving parts.

The DEM is stored in its native CRS. Reprojection into the site-local grid
happens in build_slope.py -- never assume the source CRS.

Usage:
    python -m tools.gis_pipeline.fetch_dem [--site PATH] [--force]
"""

from __future__ import annotations

import argparse
import json
import math
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np

from .common.crs import SiteSpec, load_site
from .common.paths import raw_dir

USGS_3DEP = (
    "https://elevation.nationalmap.gov/arcgis/rest/services/"
    "3DEPElevation/ImageServer/exportImage"
)

# ArcGIS ImageServer refuses requests above ~4100 px per side.
MAX_SIDE = 4000


def fetch_3dep(site: SiteSpec, out: Path, timeout: float = 180.0) -> Path:
    """Download a float32 GeoTIFF DEM in WGS84 covering the buffered extent.

    Requested in EPSG:4326 and reprojected downstream. Asking the server for
    a custom local TM would mean passing WKT through the REST API, which is
    brittle for no gain -- rasterio reprojects this correctly in one step.
    """
    w, s, e, n = site.bbox_wgs84()

    # Size the request so ground sampling is ~dem_resolution_m.
    span_x_m = site.width_m + 2 * site.fetch_buffer_m
    span_y_m = site.height_m + 2 * site.fetch_buffer_m
    px_w = min(MAX_SIDE, max(64, math.ceil(span_x_m / site.dem_resolution_m)))
    px_h = min(MAX_SIDE, max(64, math.ceil(span_y_m / site.dem_resolution_m)))

    params = {
        "bbox": f"{w},{s},{e},{n}",
        "bboxSR": "4326",
        "imageSR": "4326",
        "size": f"{px_w},{px_h}",
        "format": "tiff",
        "pixelType": "F32",
        "noDataInterpretation": "esriNoDataMatchAny",
        "interpolation": "RSP_BilinearInterpolation",
        "f": "image",
    }
    url = f"{USGS_3DEP}?{urllib.parse.urlencode(params)}"
    print(f"  requesting {px_w}x{px_h} px from USGS 3DEP ...")

    req = urllib.request.Request(url, headers={"User-Agent": "roboto-gis/0.1"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        ctype = resp.headers.get("Content-Type", "")
        data = resp.read()

    # On error the service returns JSON with a 200 status, not an HTTP error.
    if "json" in ctype.lower() or data[:1] == b"{":
        raise RuntimeError(f"3DEP returned an error: {json.loads(data)}")
    if len(data) < 1024:
        raise RuntimeError(f"3DEP returned {len(data)} bytes -- too small to be a DEM")

    out.write_bytes(data)
    return out


def summarize(path: Path, site: SiteSpec) -> dict:
    """Report relief and nodata. Relief is the whole reason for the DEM."""
    import rasterio

    with rasterio.open(path) as ds:
        band = ds.read(1, masked=True)
        crs = ds.crs
        shape = (ds.height, ds.width)

    z = np.ma.filled(band.astype(np.float64), np.nan)
    # 3DEP encodes voids as large negatives rather than a nodata tag.
    z[z < -1e4] = np.nan

    finite = np.isfinite(z)
    stats = {
        "crs": str(crs),
        "shape": shape,
        "nodata_frac": float(1.0 - finite.mean()),
        "zmin": float(np.nanmin(z)),
        "zmax": float(np.nanmax(z)),
        "relief": float(np.nanmax(z) - np.nanmin(z)),
        "zmean": float(np.nanmean(z)),
    }

    print(f"  CRS        : {stats['crs']}")
    print(f"  shape      : {shape[0]} x {shape[1]}")
    print(f"  elevation  : {stats['zmin']:.1f} .. {stats['zmax']:.1f} m "
          f"(mean {stats['zmean']:.1f})")
    print(f"  relief     : {stats['relief']:.1f} m across the fetched area")
    print(f"  nodata     : {stats['nodata_frac'] * 100:.2f} %")

    if stats["relief"] < 10.0:
        print("  WARNING: under 10 m of relief -- the slope-cost feature will be\n"
              "           near-meaningless here. Consider moving the origin.")
    if stats["nodata_frac"] > 0.02:
        print("  WARNING: >2% nodata; build_slope.py will infill, but check coverage.")
    return stats


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--site", default=None)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    site = load_site(args.site)
    out = raw_dir(site) / "dem_src.tif"

    print(f"site '{site.name}'  target resolution {site.dem_resolution_m} m")
    if out.exists() and not args.force:
        print(f"  cached: {out}")
    else:
        fetch_3dep(site, out)
        print(f"  saved -> {out}  ({out.stat().st_size / 1e6:.1f} MB)")

    summarize(out, site)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
