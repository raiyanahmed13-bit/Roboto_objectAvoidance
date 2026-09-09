"""Reproject the DEM into the site-local frame and derive terrain gradients.

Design note: the reprojected DEM is stored at its native ~1 m resolution
(a few hundred KB), NOT upsampled to the 0.10 m planning grid. Upsampling
here would produce three 3000x3000 float32 arrays -- ~108 MB of committed
data carrying no information the 1 m source does not already have. Instead
`TerrainModel.sample(grid)` interpolates onto whatever grid the caller
needs, in milliseconds.

We interpolate the GRADIENT COMPONENTS (dz/dx, dz/dy) rather than the slope
magnitude, because the planner needs *directional* slope: climbing a 15%
grade is expensive, descending it is nearly free, and traversing it
sideways is a rollover risk. A scalar slope field cannot express that.

Usage:
    python -m tools.gis_pipeline.build_slope [--site PATH] [--preview]
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from roboto_core.plan.terrain import TerrainGrid

from .common.crs import SiteSpec, load_site
from .common.grid import GridSpec, from_raster, to_raster
from .common.paths import derived_dir, raw_dir


def dem_grid(site: SiteSpec) -> GridSpec:
    """DEM grid: the buffered extent at native DEM resolution.

    Buffered so that gradients at the working-extent boundary are computed
    from real neighbours rather than edge-clamped values.
    """
    res = site.dem_resolution_m
    xmin, ymin, xmax, ymax = site.bbox_local(site.fetch_buffer_m)
    return GridSpec(
        origin_x=xmin,
        origin_y=ymin,
        resolution=res,
        width=int(round((xmax - xmin) / res)),
        height=int(round((ymax - ymin) / res)),
    )


@dataclass
class TerrainModel(TerrainGrid):
    """The GIS-side terrain model: TerrainGrid plus GeoTIFF loading.

    The maths lives in roboto_core.plan.terrain so the robot can use it
    without rasterio. This subclass adds only the parts that need a GIS
    stack: reading the reprojected DEM, and (via save_npz) writing the
    numpy copy the ROS node reads.
    """

    @classmethod
    def load(cls, site: SiteSpec) -> "TerrainModel":
        import rasterio

        path = derived_dir(site) / "dem_local.tif"
        if not path.exists():
            raise FileNotFoundError(
                f"{path} missing -- run `python -m tools.gis_pipeline.build_slope` first"
            )
        with rasterio.open(path) as ds:
            z = from_raster(ds.read(1)).astype(np.float64)
        return cls.from_local_dem(z, dem_grid(site))


# ---------------------------------------------------------------------------

def reproject_dem(site: SiteSpec, force: bool = False) -> Path:
    """Warp the fetched DEM into the site-local TM grid.

    Always reprojects explicitly -- 3DEP may hand back EPSG:4326, 3857 or
    5070 depending on the endpoint, and assuming the CRS is how a DEM ends
    up silently offset from the vector layers.
    """
    import rasterio
    from rasterio.enums import Resampling
    from rasterio.warp import reproject

    src_path = raw_dir(site) / "dem_src.tif"
    out_path = derived_dir(site) / "dem_local.tif"
    if out_path.exists() and not force:
        print(f"  cached: {out_path}")
        return out_path
    if not src_path.exists():
        raise FileNotFoundError(
            f"{src_path} missing -- run `python -m tools.gis_pipeline.fetch_dem` first"
        )

    grid = dem_grid(site)
    dst = np.full(grid.shape, np.nan, dtype=np.float32)

    with rasterio.open(src_path) as ds:
        src = ds.read(1).astype(np.float32)
        # 3DEP encodes voids as large negatives rather than a nodata tag.
        src[src < -1e4] = np.nan
        print(f"  source CRS {ds.crs} -> {site.crs_local.to_proj4().strip()[:52]}...")
        reproject(
            source=src,
            destination=dst,
            src_transform=ds.transform,
            src_crs=ds.crs,
            dst_transform=grid.rasterio_transform(),
            dst_crs=site.crs_local,
            resampling=Resampling.bilinear,
            src_nodata=np.nan,
            dst_nodata=np.nan,
        )

    dst = _fill_nodata(dst)

    profile = {
        "driver": "GTiff", "height": grid.height, "width": grid.width,
        "count": 1, "dtype": "float32", "crs": site.crs_local,
        "transform": grid.rasterio_transform(), "nodata": np.nan,
        "compress": "deflate",
    }
    with rasterio.open(out_path, "w", **profile) as ds:
        ds.write(dst, 1)   # profile transform is north-up, so write raster order
    return out_path


def _fill_nodata(a: np.ndarray) -> np.ndarray:
    """Nearest-neighbour infill, then a light median to kill speckle."""
    from scipy.ndimage import distance_transform_edt, median_filter

    bad = ~np.isfinite(a)
    if not bad.any():
        return a
    if bad.all():
        raise ValueError("DEM is entirely nodata after reprojection")

    print(f"  infilling {bad.sum()} nodata cells ({bad.mean() * 100:.3f} %)")
    _, idx = distance_transform_edt(bad, return_indices=True)
    filled = a[tuple(idx)]
    smoothed = median_filter(filled, size=3)
    return np.where(bad, smoothed, filled).astype(np.float32)


def save_preview(site: SiteSpec, terr: TerrainModel) -> Path:
    """Hillshade + slope figure. Eyeballing this catches a mirrored DEM."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ext = [terr.grid.origin_x, terr.grid.xmax, terr.grid.origin_y, terr.grid.ymax]
    az, alt = np.radians(315.0), np.radians(45.0)
    hs = (np.sin(alt) / np.sqrt(1 + terr.dzdx**2 + terr.dzdy**2)
          * (1 + np.cos(az) * terr.dzdx + np.sin(az) * terr.dzdy))

    fig, axes = plt.subplots(1, 3, figsize=(16, 5.2), constrained_layout=True)
    for ax, data, cmap, title in [
        (axes[0], terr.z, "terrain", "Elevation (m)"),
        (axes[1], hs, "gray", "Hillshade (NW sun)"),
        (axes[2], terr.slope_deg, "magma", "Slope (deg)"),
    ]:
        # origin="lower" because arrays are in ROS convention (row 0 = south).
        im = ax.imshow(data, origin="lower", extent=ext, cmap=cmap)
        fig.colorbar(im, ax=ax, shrink=0.8)
        ax.set_title(title)
        ax.set_xlabel("East (m)")
        ax.add_patch(plt.Rectangle(
            (site.xmin, site.ymin), site.width_m, site.height_m,
            fill=False, ec="cyan", lw=1.6, ls="--"))
    axes[0].set_ylabel("North (m)")

    out = Path("experiments") / "figures" / "terrain.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=110)
    plt.close(fig)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--site", default=None)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--preview", action="store_true", help="write a hillshade figure")
    args = ap.parse_args()

    site = load_site(args.site)
    print(f"site '{site.name}'")

    path = reproject_dem(site, args.force)
    print(f"  local DEM -> {path}")

    terr = TerrainModel.load(site)
    slope = terr.slope_deg

    # Report over the working extent only; the buffer skews the percentiles.
    work = GridSpec.from_site(site)
    r0, c0 = terr.grid.world_to_cell(site.xmin, site.ymin)
    r1, c1 = terr.grid.world_to_cell(site.xmax, site.ymax)
    s = slope[int(r0):int(r1), int(c0):int(c1)]

    thresh = site.raw["slope"]["max_traversable_deg"]
    print(f"  relief (buffered) : {terr.relief:.1f} m")
    print(f"  slope over extent : median {np.median(s):.1f} deg, "
          f"p90 {np.percentile(s, 90):.1f} deg, max {s.max():.1f} deg")
    print(f"  untraversable     : {(s > thresh).mean() * 100:.2f} % of cells "
          f"above {thresh:.0f} deg")

    dzdx, dzdy, z = terr.sample(work)
    print(f"  sampled onto planning grid {work.shape} "
          f"({dzdx.nbytes * 3 / 1e6:.0f} MB in memory, not on disk)")

    # A numpy copy of the DEM, for the robot. The ROS environment has no
    # rasterio, so without this the live node cannot weight slope at all:
    # it planned on distance and obstacles while the offline experiments
    # planned with terrain, meaning the two were not solving the same
    # problem. Derived from the GeoTIFF, never edited by hand.
    npz = terr.save_npz(derived_dir(site) / "dem_local.npz")
    print(f"  robot-readable DEM -> {npz.name} "
          f"({npz.stat().st_size / 1e6:.1f} MB, no rasterio needed)")

    if args.preview:
        print(f"  preview -> {save_preview(site, terr)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
