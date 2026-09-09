"""Gate B: coordinate-frame and raster-alignment assertions.

A failure in this file is STOP-THE-LINE. Do not work around it, do not skip
it, do not "fix" it by adding a flip somewhere else. Every downstream metric
in the project -- SLAM accuracy, map IoU, discrepancy precision/recall -- is
meaningless if these do not hold.

The bugs these tests exist to catch, none of which are visually obvious:

  * a north/south row-order flip applied zero times or twice
  * an east/west mirror from a projection sign error
  * treating the map_server `origin` as the grid centre instead of a corner
  * a DEM arriving in a different CRS than the vector layers
"""

import numpy as np
import pytest
from shapely.geometry import Point, Polygon, box

from tools.gis_pipeline.common.crs import SiteSpec, load_site
from tools.gis_pipeline.common.grid import (
    GridSpec,
    from_raster,
    load_occupancy,
    save_occupancy,
    to_raster,
)


@pytest.fixture(scope="module")
def site() -> SiteSpec:
    return load_site()


@pytest.fixture
def small() -> GridSpec:
    """A 20 m x 20 m grid at 0.5 m. Small enough to reason about by hand."""
    return GridSpec(origin_x=-10.0, origin_y=-10.0, resolution=0.5, width=40, height=40)


# ---------------------------------------------------------------------------
# Projection
# ---------------------------------------------------------------------------

def test_origin_projects_to_zero(site):
    """The site origin is (0, 0) by construction -- the whole frame contract."""
    x, y = site.lla_to_local(site.lon0, site.lat0)
    assert abs(x) < 1e-6
    assert abs(y) < 1e-6


def test_lla_roundtrip(site):
    xs = np.array([-150.0, -20.0, 0.0, 37.5, 150.0])
    ys = np.array([-150.0, 80.0, 0.0, -12.25, 150.0])
    lons, lats = site.local_to_lla(xs, ys)
    bx, by = site.lla_to_local(lons, lats)
    assert np.allclose(bx, xs, atol=1e-6)
    assert np.allclose(by, ys, atol=1e-6)


def test_axes_are_east_north(site):
    """+x must be East and +y must be North. Catches a mirrored projection."""
    lon_e, lat_e = site.local_to_lla(100.0, 0.0)
    assert lon_e > site.lon0, "+x must increase longitude (East)"
    assert lat_e == pytest.approx(site.lat0, abs=1e-4)

    lon_n, lat_n = site.local_to_lla(0.0, 100.0)
    assert lat_n > site.lat0, "+y must increase latitude (North)"
    assert lon_n == pytest.approx(site.lon0, abs=1e-4)


def test_local_tm_has_unit_scale(site):
    """k=1 means 100 m in the projection is 100 m on the ground at the origin.

    This is the reason for site-local TM over UTM, so it is worth asserting.
    """
    from pyproj import Geod

    geod = Geod(ellps="WGS84")
    lon1, lat1 = site.local_to_lla(0.0, 0.0)
    lon2, lat2 = site.local_to_lla(100.0, 0.0)
    _, _, dist = geod.inv(lon1, lat1, lon2, lat2)
    assert dist == pytest.approx(100.0, abs=0.02)


def test_proj4_mismatch_is_rejected(tmp_path, site):
    """site.yaml storing a proj4 that disagrees with origin lat/lon must fail
    loudly -- silently misaligned rasters are the alternative."""
    import yaml

    d = dict(site.raw)
    d["crs_local"] = d["crs_local"].replace("lat_0=40.4187", "lat_0=41.0")
    p = tmp_path / "bad.yaml"
    p.write_text(yaml.safe_dump(d), encoding="utf-8")

    with pytest.raises(ValueError, match="disagrees with origin"):
        SiteSpec.from_yaml(p)


def test_bbox_wgs84_contains_extent(site):
    """The fetch bbox must fully contain the working extent, or edge
    buildings get clipped mid-footprint."""
    w, s, e, n = site.bbox_wgs84()
    for x in (site.xmin, site.xmax):
        for y in (site.ymin, site.ymax):
            lon, lat = site.local_to_lla(x, y)
            assert w < lon < e
            assert s < lat < n


# ---------------------------------------------------------------------------
# Grid indexing
# ---------------------------------------------------------------------------

def test_cell_roundtrip_is_stable(small):
    rng = np.random.default_rng(0)
    rows = rng.integers(0, small.height, 200)
    cols = rng.integers(0, small.width, 200)
    x, y = small.cell_to_world(rows, cols)
    r2, c2 = small.world_to_cell(x, y)
    assert np.array_equal(rows, r2)
    assert np.array_equal(cols, c2)


def test_origin_is_bottom_left_corner_not_centre(small):
    """map_server `origin` is the corner of cell (0,0), so that cell's CENTRE
    sits half a resolution up and to the right. Confusing corner for centre
    offsets the entire map by half the extent."""
    x, y = small.cell_to_world(0, 0)
    assert x == pytest.approx(small.origin_x + small.resolution / 2)
    assert y == pytest.approx(small.origin_y + small.resolution / 2)

    r, c = small.world_to_cell(small.origin_x + 1e-9, small.origin_y + 1e-9)
    assert (int(r), int(c)) == (0, 0)


def test_row_increases_northward(small):
    """ROS convention: row 0 is the SOUTH edge."""
    r_south, _ = small.world_to_cell(0.0, small.origin_y + 0.1)
    r_north, _ = small.world_to_cell(0.0, small.ymax - 0.1)
    assert r_south < r_north

    _, c_west = small.world_to_cell(small.origin_x + 0.1, 0.0)
    _, c_east = small.world_to_cell(small.xmax - 0.1, 0.0)
    assert c_west < c_east


def test_map_yaml_origin_is_corner(small):
    meta = small.to_map_yaml("x.pgm")
    assert meta["origin"][:2] == [small.origin_x, small.origin_y]
    assert meta["resolution"] == small.resolution


def test_assert_compatible_catches_drift(small):
    small.assert_compatible(small)
    shifted = GridSpec(small.origin_x + 0.5, small.origin_y, small.resolution,
                       small.width, small.height)
    with pytest.raises(ValueError, match="origin mismatch"):
        small.assert_compatible(shifted)

    coarse = GridSpec(small.origin_x, small.origin_y, 1.0, small.width, small.height)
    with pytest.raises(ValueError, match="resolution mismatch"):
        small.assert_compatible(coarse)


# ---------------------------------------------------------------------------
# The flip
# ---------------------------------------------------------------------------

def test_flip_is_an_involution():
    a = np.arange(12).reshape(3, 4)
    assert np.array_equal(from_raster(to_raster(a)), a)
    assert np.array_equal(to_raster(from_raster(a)), a)


def test_flip_actually_reverses_rows():
    """Guards against someone 'simplifying' from_raster to a no-op."""
    a = np.array([[1, 1], [0, 0]])
    assert np.array_equal(from_raster(a), np.array([[0, 0], [1, 1]]))


# ---------------------------------------------------------------------------
# Rasterization -- the mirror tests
# ---------------------------------------------------------------------------

def _rasterize(geom, spec: GridSpec) -> np.ndarray:
    """Rasterize into ROS convention, exactly as the pipeline does."""
    from rasterio.features import rasterize

    raw = rasterize(
        [(geom, 1)],
        out_shape=spec.shape,
        transform=spec.rasterio_transform(),
        fill=0,
        all_touched=True,
        dtype="uint8",
    )
    return from_raster(raw).astype(bool)


def test_northeast_polygon_lands_northeast(small):
    """THE mirror test.

    A square in the NE quadrant must occupy high rows and high columns. If
    either flip is wrong the map still looks like a plausible neighbourhood
    -- you only find out when discrepancy detection reports total
    disagreement. Assert it instead.
    """
    occ = _rasterize(box(2.0, 2.0, 8.0, 8.0), small)
    rows, cols = np.nonzero(occ)
    assert rows.size > 0

    assert rows.min() >= small.height // 2, "polygon fell in the SOUTH half (N/S flip wrong)"
    assert cols.min() >= small.width // 2, "polygon fell in the WEST half (E/W mirror)"


def test_rasterized_centroid_roundtrips(small):
    """The rasterized centroid must round-trip through cell_to_world to
    within half a cell of the true polygon centroid."""
    poly = box(1.0, -6.0, 5.0, -2.0)      # asymmetric, off-centre, in the SE
    occ = _rasterize(poly, small)
    rows, cols = np.nonzero(occ)
    x, y = small.cell_to_world(rows.mean(), cols.mean())

    assert x == pytest.approx(poly.centroid.x, abs=small.resolution)
    assert y == pytest.approx(poly.centroid.y, abs=small.resolution)


def test_asymmetric_shape_is_not_transposed(small):
    """A wide-and-short rectangle must stay wide and short.

    Catches a row/col transpose, which a square test cannot see.
    """
    occ = _rasterize(box(-8.0, -1.0, 8.0, 1.0), small)
    rows, cols = np.nonzero(occ)
    assert (cols.max() - cols.min()) > (rows.max() - rows.min()) * 3


def test_points_inside_footprint_read_occupied(small):
    poly = Polygon([(-6, -6), (-2, -6), (-2, -1), (-4, -1), (-4, -4), (-6, -4)])  # L-shape
    occ = _rasterize(poly, small)

    rng = np.random.default_rng(7)
    inside = 0
    while inside < 20:
        p = np.array([rng.uniform(-6, -2), rng.uniform(-6, -1)])
        if not poly.contains(Point(*p)):
            continue
        r, c = small.world_to_cell(p[0], p[1])
        assert occ[int(r), int(c)], f"point {p} inside footprint read as free"
        inside += 1


def test_first_moments_match_vector_layer(small):
    """Raster and vector first moments must agree. A cheap global check that
    catches mirroring even when per-point checks are coincidentally fine."""
    poly = box(-9.0, 3.0, -3.0, 9.0)     # NW quadrant, far from centre
    occ = _rasterize(poly, small)
    rows, cols = np.nonzero(occ)
    x, y = small.cell_to_world(rows.mean(), cols.mean())

    assert x == pytest.approx(poly.centroid.x, abs=small.resolution)
    assert y == pytest.approx(poly.centroid.y, abs=small.resolution)
    assert x < 0 and y > 0, "NW polygon did not land in the NW quadrant"


# ---------------------------------------------------------------------------
# PGM round-trip
# ---------------------------------------------------------------------------

def test_pgm_roundtrip_preserves_orientation(tmp_path, small):
    """Write then read must return the same array. Two flips that cancel on a
    symmetric map would pass a weaker test, so use an asymmetric one."""
    occ = _rasterize(box(2.0, 4.0, 9.0, 6.0), small)
    yml = save_occupancy(tmp_path / "m.pgm", occ, small)

    back, spec2 = load_occupancy(yml)
    small.assert_compatible(spec2)
    assert np.array_equal(occ, back)


def test_pgm_header_is_valid_p5(tmp_path, small):
    occ = np.zeros(small.shape, dtype=bool)
    occ[0, 0] = True
    save_occupancy(tmp_path / "m.pgm", occ, small)

    raw = (tmp_path / "m.pgm").read_bytes()
    assert raw.startswith(b"P5")
    assert f"{small.width} {small.height}".encode() in raw[:32]


def test_occupied_cell_survives_as_black_pixel(tmp_path, small):
    """A single occupied cell at a known corner must come back at the same
    cell -- the tightest end-to-end check of the flip pair."""
    occ = np.zeros(small.shape, dtype=bool)
    occ[3, 7] = True                       # low row = south, low col = west
    yml = save_occupancy(tmp_path / "m.pgm", occ, small)

    back, _ = load_occupancy(yml)
    assert back[3, 7]
    assert back.sum() == 1
