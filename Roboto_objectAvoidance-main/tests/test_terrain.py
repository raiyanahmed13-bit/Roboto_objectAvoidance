"""Elevation gradients, and the numpy path the robot uses to read them.

These matter because the live system used to have no terrain at all: the
DEM lives in a GeoTIFF, rasterio is not installed in the ROS environment,
so the node ran with `terrain = None` while the offline experiments planned
WITH slope. The two were solving different problems and nobody noticed,
because a robot that ignores slope still reaches the goal -- it just takes
a different route, and nothing compared them.

The npz path exists to close that gap, so it needs to be provably
equivalent to the GeoTIFF one rather than merely present.
"""

import numpy as np
import pytest

from roboto_core.frames import GridSpec
from roboto_core.plan.terrain import TerrainGrid


@pytest.fixture
def grid():
    """100 m x 100 m at 1 m, like a DEM tile."""
    return GridSpec(origin_x=-50.0, origin_y=-50.0, resolution=1.0,
                    width=100, height=100)


def _ramp(grid, a, b, c=100.0):
    """z = a*x + b*y + c, so dz/dx == a and dz/dy == b everywhere."""
    rows = np.arange(grid.height)
    cols = np.arange(grid.width)
    x, _ = grid.cell_to_world(0, cols)
    _, y = grid.cell_to_world(rows, 0)
    X, Y = np.meshgrid(np.asarray(x), np.asarray(y))
    return a * X + b * Y + c


# ---------------------------------------------------------------------------
# Gradients
# ---------------------------------------------------------------------------

def test_gradients_of_a_known_ramp(grid):
    t = TerrainGrid.from_local_dem(_ramp(grid, 0.2, -0.1), grid)
    # Interior only: np.gradient uses one-sided differences at the edges.
    assert np.allclose(t.dzdx[1:-1, 1:-1], 0.2, atol=1e-9)
    assert np.allclose(t.dzdy[1:-1, 1:-1], -0.1, atol=1e-9)


def test_row_order_is_not_flipped(grid):
    """In ROS convention row increases NORTHWARD, so np.gradient's first
    output is d/dy directly. A sign error here would send the planner
    uphill to save effort, and the route would still look plausible."""
    t = TerrainGrid.from_local_dem(_ramp(grid, 0.0, 0.3), grid)
    # Ground rises towards the north, so z at a high row exceeds z at a low.
    assert t.z[80, 50] > t.z[20, 50]
    assert np.allclose(t.dzdy[1:-1, 1:-1], 0.3, atol=1e-9)


def test_slope_magnitude(grid):
    t = TerrainGrid.from_local_dem(_ramp(grid, 1.0, 0.0), grid)
    # A 1:1 grade is 45 degrees.
    assert t.slope_deg[50, 50] == pytest.approx(45.0, abs=1e-6)


def test_relief_is_the_elevation_span(grid):
    t = TerrainGrid.from_local_dem(_ramp(grid, 0.1, 0.0), grid)
    assert t.relief == pytest.approx(float(t.z.max() - t.z.min()))


# ---------------------------------------------------------------------------
# Directional slope -- the reason gradients are kept as components
# ---------------------------------------------------------------------------

def test_along_path_slope_is_signed(grid):
    """Climbing is positive, descending negative, and crossing the grade
    sideways is zero. A scalar slope field cannot express any of this,
    which is why the planning graph is directed."""
    t = TerrainGrid.from_local_dem(_ramp(grid, 0.5, 0.0), grid)
    dzdx, dzdy = 0.5, 0.0

    uphill = t.along_path_slope(dzdx, dzdy, 0.0)              # due east
    downhill = t.along_path_slope(dzdx, dzdy, np.pi)          # due west
    across = t.along_path_slope(dzdx, dzdy, np.pi / 2)        # due north

    assert uphill > 0
    assert downhill == pytest.approx(-uphill)
    assert across == pytest.approx(0.0, abs=1e-12)


# ---------------------------------------------------------------------------
# The npz path the robot reads
# ---------------------------------------------------------------------------

def test_npz_round_trip_preserves_everything(grid, tmp_path):
    t = TerrainGrid.from_local_dem(_ramp(grid, 0.2, -0.15), grid)
    back = TerrainGrid.from_npz(t.save_npz(tmp_path / "dem.npz"))

    assert back.grid == t.grid
    assert np.allclose(back.z, t.z, atol=1e-3)         # stored as float32
    assert np.allclose(back.dzdx, t.dzdx, atol=1e-4)
    assert np.allclose(back.dzdy, t.dzdy, atol=1e-4)


def test_npz_needs_no_gis_stack(grid, tmp_path, monkeypatch):
    """The whole point: the robot must be able to read this without
    rasterio, which is not installed in the ROS environment."""
    import builtins

    t = TerrainGrid.from_local_dem(_ramp(grid, 0.1, 0.1), grid)
    path = t.save_npz(tmp_path / "dem.npz")

    real_import = builtins.__import__

    def no_gis(name, *a, **kw):
        if name.split(".")[0] in ("rasterio", "geopandas", "pyproj", "fiona"):
            raise ImportError(f"{name} is not available on the robot")
        return real_import(name, *a, **kw)

    monkeypatch.setattr(builtins, "__import__", no_gis)
    back = TerrainGrid.from_npz(path)
    assert back.grid.width == grid.width


def test_gradients_are_not_stored(grid, tmp_path):
    """They are a pure function of z, so persisting them would triple the
    file and create a second thing that can go stale against the first."""
    t = TerrainGrid.from_local_dem(_ramp(grid, 0.2, 0.0), grid)
    keys = set(np.load(str(t.save_npz(tmp_path / "dem.npz"))).keys())
    assert keys == {"z", "origin", "resolution"}


# ---------------------------------------------------------------------------
# Sampling onto the planning grid
# ---------------------------------------------------------------------------

def test_sample_onto_a_finer_grid(grid):
    """The DEM is ~1 m and the planning grid is 0.10 m. Interpolating a
    linear ramp must reproduce it, or every slope cost is subtly wrong."""
    t = TerrainGrid.from_local_dem(_ramp(grid, 0.25, -0.1), grid)
    fine = GridSpec(origin_x=-20.0, origin_y=-20.0, resolution=0.1,
                    width=400, height=400)

    dzdx, dzdy, z = t.sample(fine)
    assert dzdx.shape == fine.shape
    assert np.allclose(dzdx, 0.25, atol=1e-3)
    assert np.allclose(dzdy, -0.1, atol=1e-3)

    # Elevation must land where the ramp says, not merely be smooth.
    r, c = fine.world_to_cell(5.0, 5.0)
    expect = 0.25 * 5.0 + -0.1 * 5.0 + 100.0
    assert z[int(r), int(c)] == pytest.approx(expect, abs=0.05)


def test_sampling_is_georeferenced_not_just_resized(grid):
    """A sample that ignored the origin would still return a plausible
    field, just shifted -- the failure mode this project guards against
    everywhere else."""
    t = TerrainGrid.from_local_dem(_ramp(grid, 1.0, 0.0), grid)
    off = GridSpec(origin_x=10.0, origin_y=0.0, resolution=1.0,
                   width=20, height=20)
    _, _, z = t.sample(off)

    x, _ = off.cell_to_world(0, 0)
    assert z[0, 0] == pytest.approx(1.0 * float(x) + 100.0, abs=0.5)
