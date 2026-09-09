"""Build the GIS prior costmap: the map the robot believes before it drives.

Outputs, all on the planning grid defined by site.yaml:

  prior_occ.pgm/.yaml   binary building occupancy, ROS map_server format
  prior_cost.npz        lethal mask + graded base cost + road mask

Directional slope cost is deliberately NOT baked in here. Whether a grade
is expensive depends on the direction you cross it -- climbing is costly,
descending is nearly free, side-slope is a rollover risk -- so it is an
edge cost, applied by the planner at expansion time from the terrain
gradients. Only the direction-independent part (impassably steep) is
resolved into the static lethal mask.

Usage:
    python -m tools.gis_pipeline.build_prior [--site PATH] [--preview]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import geopandas as gpd
import numpy as np

from .build_slope import TerrainModel
from .common.crs import SiteSpec, load_site
from .common.grid import GridSpec, from_raster, save_occupancy
from .common.paths import derived_dir, load_vector


# ROS costmap_2d-style encoding for the quantized prior.
LETHAL_COST = 254
COST_SCALE = 1.15      # max of base cost: 1.0 inflation + 0.15 off-road penalty


def load_prior(site: SiteSpec) -> dict:
    """Read prior_cost.npz back, restoring the lethal mask and float cost."""
    d = np.load(derived_dir(site) / "prior_cost.npz")
    cost = d["cost"]
    return {
        "cost": cost,
        "lethal": cost >= LETHAL_COST,
        "base_cost": cost.astype(np.float32) / (LETHAL_COST - 1) * COST_SCALE,
        "road_mask": d["road_mask"],
        "grid": GridSpec(
            origin_x=float(d["origin"][0]), origin_y=float(d["origin"][1]),
            resolution=float(d["resolution"]),
            width=cost.shape[1], height=cost.shape[0],
        ),
    }


def _rasterize(geoms, spec: GridSpec) -> np.ndarray:
    """Rasterize geometries into ROS convention. Empty input -> all False."""
    from rasterio.features import rasterize

    geoms = [g for g in geoms if g is not None and not g.is_empty]
    if not geoms:
        return np.zeros(spec.shape, dtype=bool)
    raw = rasterize(
        [(g, 1) for g in geoms],
        out_shape=spec.shape,
        transform=spec.rasterio_transform(),
        fill=0,
        all_touched=True,
        dtype="uint8",
    )
    return from_raster(raw).astype(bool)


def build(site: SiteSpec, preview: bool = False) -> dict:
    grid = GridSpec.from_site(site)
    out = derived_dir(site)

    buildings = load_vector(site, "buildings")
    roads = load_vector(site, "roads")
    print(f"  {len(buildings)} buildings, {len(roads)} road polygons")

    # --- occupancy ------------------------------------------------------
    occ = _rasterize(buildings.geometry, grid)
    road_mask = _rasterize(roads.geometry, grid)
    print(f"  occupancy: {occ.mean() * 100:.1f} % built, {road_mask.mean() * 100:.1f} % road")

    # --- terrain --------------------------------------------------------
    terr = TerrainModel.load(site)
    dzdx, dzdy, z = terr.sample(grid)
    slope_deg = np.degrees(np.arctan(np.hypot(dzdx, dzdy)))

    scfg = site.raw["slope"]
    too_steep = slope_deg > scfg["max_traversable_deg"]
    print(f"  terrain:   {too_steep.mean() * 100:.1f} % steeper than "
          f"{scfg['max_traversable_deg']:.0f} deg")

    # --- lethal + inflation --------------------------------------------
    from scipy.ndimage import distance_transform_edt

    r_robot = site.raw["robot"]["radius_m"]
    hard = occ | too_steep

    # Distance to the nearest hard cell, in metres.
    dist = distance_transform_edt(~hard, sampling=grid.resolution).astype(np.float32)
    lethal = dist <= r_robot          # robot body would intersect an obstacle

    # Exponential decay beyond the inscribed radius: steers the planner away
    # from walls without forbidding tight passages outright.
    decay_m = 1.5
    base = np.exp(-(dist - r_robot) / decay_m, dtype=np.float32)
    np.clip(base, 0.0, 1.0, out=base)
    base[lethal] = 1.0

    # Mild preference for staying on mapped roads/paths.
    base = base + 0.15 * (~road_mask)

    print(f"  lethal:    {lethal.mean() * 100:.1f} % of cells "
          f"(inflated by r_robot={r_robot} m)")
    free_frac = 1.0 - lethal.mean()
    if free_frac < 0.25:
        print("  WARNING: under 25 % of the map is drivable -- planning may be "
              "infeasible. Check the slope threshold.")

    # --- write ----------------------------------------------------------
    yml = save_occupancy(out / "prior_occ.pgm", occ, grid)

    # Quantize to uint8, matching ROS costmap_2d: 0..253 graded, 254 lethal.
    # 254 levels is ample for planning and cuts the artifact ~8x versus
    # float32. slope_deg is deliberately NOT stored -- it is a pure function
    # of the DEM, which is already committed, so persisting it here would
    # duplicate 36 MB and create a second source of truth that can drift.
    cost_u8 = np.clip(base / COST_SCALE * (LETHAL_COST - 1), 0, LETHAL_COST - 1)
    cost_u8 = cost_u8.astype(np.uint8)
    cost_u8[lethal] = LETHAL_COST

    np.savez_compressed(
        out / "prior_cost.npz",
        cost=cost_u8,
        road_mask=road_mask,
        origin=np.array([grid.origin_x, grid.origin_y], dtype=np.float64),
        resolution=np.float64(grid.resolution),
    )
    npz = out / "prior_cost.npz"
    print(f"  wrote {yml.name}, {npz.name} ({npz.stat().st_size / 1e6:.1f} MB)")

    if preview:
        print(f"  preview -> {save_preview(site, grid, occ, lethal, base, buildings)}")

    return {"occ": occ, "lethal": lethal, "base": base, "grid": grid,
            "buildings": buildings, "slope_deg": slope_deg}


def save_preview(site, grid, occ, lethal, base, buildings) -> Path:
    """Overlay rasterized occupancy against the source vectors.

    This is the figure that catches a mirrored or offset raster: the drawn
    footprint outlines must sit exactly on the filled cells.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ext = [grid.origin_x, grid.xmax, grid.origin_y, grid.ymax]
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.4), constrained_layout=True)

    axes[0].imshow(occ, origin="lower", extent=ext, cmap="Greys", interpolation="nearest")
    buildings.boundary.plot(ax=axes[0], color="red", linewidth=0.5)
    axes[0].set_title("Rasterized occupancy vs OSM vectors\n(red outlines must sit on grey cells)")

    axes[1].imshow(lethal, origin="lower", extent=ext, cmap="Reds", interpolation="nearest")
    axes[1].set_title("Lethal (buildings + steep, inflated)")

    im = axes[2].imshow(base, origin="lower", extent=ext, cmap="viridis")
    fig.colorbar(im, ax=axes[2], shrink=0.8)
    axes[2].set_title("Base traversal cost")

    for ax in axes:
        ax.set_xlim(grid.origin_x, grid.xmax)
        ax.set_ylim(grid.origin_y, grid.ymax)
        ax.set_xlabel("East (m)")
        ax.plot(site.spawn_x, site.spawn_y, "c*", ms=16, mec="k", label="spawn")
    axes[0].set_ylabel("North (m)")
    axes[0].legend(loc="upper right")

    out = Path("experiments") / "figures" / "prior_costmap.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=110)
    plt.close(fig)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--site", default=None)
    ap.add_argument("--preview", action="store_true")
    args = ap.parse_args()

    site = load_site(args.site)
    print(f"site '{site.name}'  grid {site.height_cells} x {site.width_cells} "
          f"@ {site.resolution_m} m")
    r = build(site, args.preview)

    # The spawn pose must be drivable, or every experiment fails at t=0.
    row, col = r["grid"].world_to_cell(site.spawn_x, site.spawn_y)
    if r["lethal"][int(row), int(col)]:
        print(f"\n  ERROR: spawn ({site.spawn_x}, {site.spawn_y}) is in a lethal cell. "
              f"Move robot_spawn in site.yaml.")
        return 1
    print(f"\n  spawn ({site.spawn_x}, {site.spawn_y}) is drivable  OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
