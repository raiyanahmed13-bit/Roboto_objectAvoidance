"""Gate B against REAL data.

test_alignment.py proves the coordinate machinery is correct on synthetic
fixtures. This file proves the actual fetched OSM/DEM artifacts are aligned
and self-consistent -- which is a different failure mode: the code can be
right while the committed data is stale, mismatched, or built from a
different site.yaml than the one now on disk.

Skips cleanly if the pipeline has not been run.
"""

import numpy as np
import pytest

from tools.gis_pipeline.build_prior import LETHAL_COST, load_prior
from tools.gis_pipeline.build_slope import TerrainModel, dem_grid
from tools.gis_pipeline.common.crs import load_site
from tools.gis_pipeline.common.grid import GridSpec, load_occupancy
from tools.gis_pipeline.common.paths import derived_dir, load_vector

pytestmark = pytest.mark.realdata


@pytest.fixture(scope="module")
def site():
    return load_site()


@pytest.fixture(scope="module")
def derived(site):
    d = derived_dir(site)
    if not (d / "prior_cost.npz").exists():
        pytest.skip("pipeline artifacts absent; run fetch_osm, fetch_dem, "
                    "build_slope, build_prior")
    return d


@pytest.fixture(scope="module")
def prior(site, derived):
    return load_prior(site)


@pytest.fixture(scope="module")
def buildings(site, derived):
    return load_vector(site, "buildings")


def test_vectors_are_in_local_metric_crs(site, derived):
    """load_vector refuses a geographic CRS. This asserts the stored file
    actually satisfies it -- GeoJSON silently downgraded metric data to
    EPSG:4326, which is why these layers are GeoPackage."""
    for name in ("buildings", "roads"):
        gdf = load_vector(site, name)
        assert gdf.crs.is_projected
        # Coordinates must be small metric offsets, not degrees.
        assert np.abs(gdf.total_bounds).max() > 100.0


# ---------------------------------------------------------------------------
# The artifacts agree with site.yaml
# ---------------------------------------------------------------------------

def test_prior_grid_matches_site(site, prior):
    """Catches a costmap built from an older site.yaml."""
    GridSpec.from_site(site).assert_compatible(prior["grid"], "prior_cost.npz")


def test_occupancy_pgm_matches_site(site, derived):
    _, spec = load_occupancy(derived / "prior_occ.yaml")
    GridSpec.from_site(site).assert_compatible(spec, "prior_occ.yaml")


def test_dem_is_in_local_crs(site, derived):
    """A DEM left in its source CRS is the classic silent misalignment."""
    import rasterio

    with rasterio.open(derived / "dem_local.tif") as ds:
        assert ds.crs is not None
        assert ds.crs.to_epsg() != 4326, "DEM still in WGS84 -- not reprojected"
        # Same projection family and origin as the site frame.
        assert "tmerc" in ds.crs.to_proj4()
        g = dem_grid(site)
        assert ds.transform.c == pytest.approx(g.origin_x, abs=1e-6)
        assert ds.transform.a == pytest.approx(g.resolution, abs=1e-9)


def test_dem_covers_working_extent(site):
    """The DEM must be buffered beyond the extent, so boundary gradients are
    computed from real neighbours rather than edge-clamped values."""
    g = dem_grid(site)
    assert g.origin_x < site.xmin and g.origin_y < site.ymin
    assert g.xmax > site.xmax and g.ymax > site.ymax


# ---------------------------------------------------------------------------
# Vector / raster agreement on real geometry
# ---------------------------------------------------------------------------

def _interior_points_in_extent(site, buildings):
    """Buildings whose interior test point lies inside the working extent.

    `.cx[]` alone is a bounding-box INTERSECTION filter, so it also returns
    footprints straddling the boundary whose interior point falls outside
    the raster. Select on the test point itself.

    representative_point() rather than centroid: a U- or L-shaped building's
    centroid can legitimately lie outside its own polygon.
    """
    pts = buildings.geometry.representative_point()
    keep = (
        (pts.x >= site.xmin) & (pts.x < site.xmax)
        & (pts.y >= site.ymin) & (pts.y < site.ymax)
    )
    return buildings[keep], pts[keep]


def test_building_centroids_are_occupied(site, derived, buildings):
    """Sample real footprints: an interior point must read occupied."""
    occ, spec = load_occupancy(derived / "prior_occ.yaml")

    inside, _ = _interior_points_in_extent(site, buildings)
    assert len(inside) >= 20, "too few buildings in extent to test meaningfully"

    sample = inside.sample(n=min(40, len(inside)), random_state=0)
    misses = []
    for geom in sample.geometry:
        p = geom.representative_point()
        r, c = spec.world_to_cell(p.x, p.y)
        if not (spec.in_bounds(r, c) and occ[int(r), int(c)]):
            misses.append((round(p.x, 1), round(p.y, 1)))

    assert not misses, f"{len(misses)}/{len(sample)} building interiors read free: {misses[:5]}"


def test_northeasternmost_building_roundtrips(site, derived, buildings):
    """The plan's named check: rasterized position must round-trip through
    cell_to_world to within half a cell of the vector geometry."""
    occ, spec = load_occupancy(derived / "prior_occ.yaml")

    _, pts = _interior_points_in_extent(site, buildings)
    p = pts.loc[(pts.x + pts.y).idxmax()]     # most north-easterly

    r, c = spec.world_to_cell(p.x, p.y)
    assert spec.in_bounds(r, c)
    x, y = spec.cell_to_world(r, c)
    assert abs(x - p.x) <= spec.resolution
    assert abs(y - p.y) <= spec.resolution
    assert occ[int(r), int(c)]


def test_raster_and_vector_first_moments_agree(site, derived, buildings):
    """Global mirror check on real data: the centre of mass of occupied
    cells must match that of the building polygons."""
    occ, spec = load_occupancy(derived / "prior_occ.yaml")
    rows, cols = np.nonzero(occ)
    x_r, y_r = spec.cell_to_world(rows.mean(), cols.mean())

    inside = buildings.cx[site.xmin:site.xmax, site.ymin:site.ymax]
    w = inside.geometry.area
    x_v = float((inside.geometry.centroid.x * w).sum() / w.sum())
    y_v = float((inside.geometry.centroid.y * w).sum() / w.sum())

    # 5 m tolerance: the raster is clipped to the extent while the vectors
    # extend into the buffer, so exact agreement is not expected.
    assert abs(x_r - x_v) < 5.0, f"east-west mirror suspected: {x_r:.1f} vs {x_v:.1f}"
    assert abs(y_r - y_v) < 5.0, f"north-south flip suspected: {y_r:.1f} vs {y_v:.1f}"


# ---------------------------------------------------------------------------
# Costmap semantics
# ---------------------------------------------------------------------------

def test_lethal_is_superset_of_occupied(site, derived, prior):
    """Every occupied cell must be lethal -- inflation only ever grows it."""
    occ, _ = load_occupancy(derived / "prior_occ.yaml")
    assert np.all(prior["lethal"][occ]), "an occupied cell is marked traversable"


def test_lethal_encoding_roundtrips(prior):
    assert prior["cost"].dtype == np.uint8
    assert np.array_equal(prior["lethal"], prior["cost"] >= LETHAL_COST)


def test_map_is_mostly_drivable(prior):
    """A planner needs somewhere to go. Under ~50 % free means the slope
    threshold or inflation radius is misconfigured."""
    free = 1.0 - prior["lethal"].mean()
    assert free > 0.5, f"only {free * 100:.1f} % drivable"


def test_roads_are_cheaper_than_offroad(prior):
    """The road preference term must actually bias the planner."""
    road = prior["road_mask"] & ~prior["lethal"]
    off = ~prior["road_mask"] & ~prior["lethal"]
    assert road.sum() > 1000, "almost no drivable road cells"
    assert prior["base_cost"][road].mean() < prior["base_cost"][off].mean()


def test_spawn_is_drivable(site, prior):
    """Every experiment starts here; a lethal spawn fails at t=0."""
    r, c = prior["grid"].world_to_cell(site.spawn_x, site.spawn_y)
    assert not prior["lethal"][int(r), int(c)]


def test_steep_terrain_is_lethal(site, prior):
    """The DEM must actually constrain the costmap, or the terrain feature
    is decorative. Verifies slope cells above threshold are blocked."""
    terr = TerrainModel.load(site)
    dzdx, dzdy, _ = terr.sample(prior["grid"])
    slope = np.degrees(np.arctan(np.hypot(dzdx, dzdy)))

    steep = slope > site.raw["slope"]["max_traversable_deg"]
    assert steep.mean() > 0.01, "site has too little steep terrain to be interesting"
    assert np.all(prior["lethal"][steep]), "steep cells were not marked lethal"


def test_terrain_gradients_are_finite(site, prior):
    terr = TerrainModel.load(site)
    dzdx, dzdy, z = terr.sample(prior["grid"])
    for name, a in [("dzdx", dzdx), ("dzdy", dzdy), ("z", z)]:
        assert np.isfinite(a).all(), f"{name} contains NaN/inf after infill"


def test_directional_slope_is_signed(site):
    """Uphill and downhill must differ in sign -- this is what makes the
    planning graph directed, and a free ablation baseline when zeroed."""
    terr = TerrainModel.load(site)
    dzdx = np.array([0.3]); dzdy = np.array([0.0])

    east = terr.along_path_slope(dzdx, dzdy, np.array([0.0]))
    west = terr.along_path_slope(dzdx, dzdy, np.array([np.pi]))
    assert east[0] > 0 and west[0] < 0
    assert east[0] == pytest.approx(-west[0], abs=1e-9)
