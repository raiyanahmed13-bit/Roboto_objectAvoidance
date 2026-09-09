"""A* planning with direction-dependent terrain cost.

The distinctive claim of this planner is that terrain cost belongs on the
EDGE, not the cell: climbing a grade is expensive, descending it is cheap,
crossing it sideways is a rollover risk. That makes the search graph
directed, which a scalar slope field cannot express. These tests pin that
behaviour down, on synthetic terrain where the right answer is obvious.
"""

from dataclasses import dataclass

import numpy as np
import pytest

from roboto_core.frames import GridSpec
from roboto_core.plan.astar import LETHAL, AStarPlanner, TerrainWeights


@dataclass
class HillTerrain:
    """A single Gaussian hill, matching the TerrainModel.sample() interface."""

    height_m: float = 25.0
    sigma_m: float = 12.0

    def sample(self, target: GridSpec):
        cols = np.arange(target.width)
        rows = np.arange(target.height)
        x, _ = target.cell_to_world(0, cols)
        _, y = target.cell_to_world(rows, 0)
        X, Y = np.meshgrid(np.asarray(x), np.asarray(y))
        z = self.height_m * np.exp(-((X ** 2 + Y ** 2) / (2 * self.sigma_m ** 2)))
        dzdy, dzdx = np.gradient(z, target.resolution, target.resolution)
        return (dzdx.astype(np.float32), dzdy.astype(np.float32),
                z.astype(np.float32))


@dataclass
class RampTerrain:
    """Constant grade rising towards +x. Slope is uniform, so any route
    difference must come from DIRECTION, not from position."""

    grade: float = 0.20        # 20 %, about 11 degrees

    def sample(self, target: GridSpec):
        cols = np.arange(target.width)
        rows = np.arange(target.height)
        x, _ = target.cell_to_world(0, cols)
        _, y = target.cell_to_world(rows, 0)
        X, _Y = np.meshgrid(np.asarray(x), np.asarray(y))
        z = (self.grade * X).astype(np.float32)
        return (np.full(target.shape, self.grade, np.float32),
                np.zeros(target.shape, np.float32), z)


@pytest.fixture
def open_grid():
    """80 m x 80 m of open ground at 0.5 m."""
    return GridSpec(origin_x=-40.0, origin_y=-40.0, resolution=0.5,
                    width=160, height=160)


@pytest.fixture
def open_cost(open_grid):
    return np.zeros(open_grid.shape, dtype=np.uint8)


def _climb(path, terrain, grid):
    _, _, z = terrain.sample(grid)
    r, c = grid.world_to_cell(path[:, 0], path[:, 1])
    e = z[np.clip(r, 0, grid.height - 1), np.clip(c, 0, grid.width - 1)]
    d = np.diff(e)
    return float(d[d > 0].sum())


# ---------------------------------------------------------------------------
# Basic behaviour
# ---------------------------------------------------------------------------

def test_straight_line_on_open_ground(open_cost, open_grid):
    r = AStarPlanner(open_cost, open_grid, downsample=1).plan((-20.0, 0.0), (20.0, 0.0))
    assert r.found
    assert r.length_m == pytest.approx(40.0, abs=1.0)


def test_routes_around_an_obstacle(open_cost, open_grid):
    cost = open_cost.copy()
    r0, c0 = open_grid.world_to_cell(-5.0, -10.0)
    r1, c1 = open_grid.world_to_cell(5.0, 10.0)
    cost[int(r0):int(r1), int(c0):int(c1)] = LETHAL

    r = AStarPlanner(cost, open_grid, downsample=1).plan((-20.0, 0.0), (20.0, 0.0))
    assert r.found
    assert r.length_m > 40.0, "path must detour around the wall"
    # And it must not pass through it.
    rr, cc = open_grid.world_to_cell(r.path[:, 0], r.path[:, 1])
    assert not (cost[rr, cc] >= LETHAL).any()


def test_start_and_goal_are_validated(open_cost, open_grid):
    cost = open_cost.copy()
    r0, c0 = open_grid.world_to_cell(-2.0, -2.0)
    cost[int(r0) - 10:int(r0) + 10, int(c0) - 10:int(c0) + 10] = LETHAL

    # A lethal GOAL is unreachable and must be refused.
    assert "lethal" in AStarPlanner(cost, open_grid, 1).plan((20.0, 0.0), (-2.0, -2.0)).reason
    # Start well clear of the lethal box, which spans +/-5 m around (-2, -2).
    assert "bounds" in AStarPlanner(cost, open_grid, 1).plan((20.0, 20.0), (500.0, 0.0)).reason


def test_lethal_start_escapes_instead_of_failing(open_cost, open_grid):
    """A robot can find ITSELF inside a lethal cell.

    It has just discovered an obstacle it was already close to, and the new
    inflation now covers its own position. Refusing to plan is the worst
    possible response: that is precisely the moment a route out is needed.
    In the closed-loop mission this exact refusal turned a recoverable
    situation into a collision.
    """
    cost = open_cost.copy()
    r0, c0 = open_grid.world_to_cell(-2.0, -2.0)
    cost[int(r0) - 10:int(r0) + 10, int(c0) - 10:int(c0) + 10] = LETHAL

    r = AStarPlanner(cost, open_grid, 1).plan((-2.0, -2.0), (20.0, 0.0))
    assert r.found, f"should escape and plan, got: {r.reason}"

    # And the escape route must not run through the obstacle it escaped.
    rr, cc = open_grid.world_to_cell(r.path[:, 0], r.path[:, 1])
    assert not (cost[rr, cc] >= LETHAL).any()


def test_fully_enclosed_start_still_fails(open_grid):
    """Bounded search: a robot walled in everywhere must fail, not hang."""
    cost = np.full(open_grid.shape, LETHAL, dtype=np.uint8)
    r = AStarPlanner(cost, open_grid, 1).plan((0.0, 0.0), (20.0, 0.0))
    assert not r.found
    assert "no escape" in r.reason


def test_downsampling_is_conservative(open_grid):
    """Block-reduce by MAX, never mean: a one-cell wall must not vanish.

    Mean or nearest-neighbour pooling can erase a thin obstacle and produce
    a path straight through a building.
    """
    cost = np.zeros(open_grid.shape, dtype=np.uint8)
    cost[:, 80] = LETHAL                      # single-cell vertical wall

    p = AStarPlanner(cost, open_grid, downsample=4)
    assert p.blocked[:, 20].all(), "thin wall disappeared when downsampling"


# ---------------------------------------------------------------------------
# Terrain -- the directional claim
# ---------------------------------------------------------------------------

def test_terrain_off_goes_straight_over_the_hill(open_cost, open_grid):
    terr = HillTerrain()
    w = TerrainWeights(uphill=0.0, downhill=0.0, cross=0.0)
    r = AStarPlanner(open_cost, open_grid, 1, terrain=terr, weights=w).plan(
        (0.0, -30.0), (0.0, 30.0))
    assert r.found
    assert r.length_m == pytest.approx(60.0, abs=2.0)
    assert _climb(r.path, terr, open_grid) > 20.0, "should climb the whole hill"


def test_terrain_on_routes_around_the_hill(open_cost, open_grid):
    """The headline behaviour: trade distance for elevation gain."""
    terr = HillTerrain()
    w = TerrainWeights(uphill=8.0, downhill=2.4, cross=4.0, max_slope_deg=89.0)
    r = AStarPlanner(open_cost, open_grid, 1, terrain=terr, weights=w).plan(
        (0.0, -30.0), (0.0, 30.0))
    assert r.found
    assert r.length_m > 90.0, "should detour substantially"
    assert _climb(r.path, terr, open_grid) < 5.0, "should avoid nearly all the climb"
    assert np.abs(r.path[:, 0]).max() > 20.0, "detour should swing wide of the summit"


def test_uphill_costs_more_than_downhill(open_cost, open_grid):
    """The graph must be DIRECTED.

    On a uniform ramp, going up must cost more than coming down the same
    line. This is the property a scalar slope field cannot represent.
    """
    terr = RampTerrain(grade=0.20)
    w = TerrainWeights(uphill=8.0, downhill=1.0, cross=0.0, max_slope_deg=89.0)
    p = AStarPlanner(open_cost, open_grid, 1, terrain=terr, weights=w)

    up = p.plan((-20.0, 0.0), (20.0, 0.0))
    down = p.plan((20.0, 0.0), (-20.0, 0.0))
    assert up.found and down.found
    assert up.cost > down.cost * 1.5, (
        f"uphill {up.cost:.1f} should greatly exceed downhill {down.cost:.1f}"
    )


def test_steep_grade_produces_switchbacks(open_cost, open_grid):
    """Emergent behaviour worth documenting: the planner invents switchbacks.

    On a 31 degree fall line with a 25 degree limit, a direct ascent is
    blocked -- but a 45 degree diagonal has an along-path slope of only
    arctan(0.60 * cos 45) = 23 degrees, which is allowed. The planner
    therefore zigzags up the slope, exactly as mountain roads and real
    vehicles do. Nothing special-cases this; it falls out of putting the
    terrain cost on the edge rather than the cell.
    """
    terr = RampTerrain(grade=0.60)             # ~31 degrees
    w = TerrainWeights(uphill=1.0, downhill=1.0, cross=0.0, max_slope_deg=25.0)
    r = AStarPlanner(open_cost, open_grid, 1, terrain=terr, weights=w).plan(
        (-20.0, 0.0), (20.0, 0.0))

    assert r.found, "a diagonal ascent is within the limit, so a path exists"
    assert r.length_m > 40.0 * 1.3, "switchbacking must lengthen the path"
    # Direction reverses repeatedly in y while advancing in x.
    dy = np.diff(r.path[:, 1])
    reversals = int((dy[:-1] * dy[1:] < 0).sum())
    assert reversals > 20, f"expected many zigzags, saw {reversals}"


def test_impassable_grade_is_blocked(open_cost, open_grid):
    """Above the diagonal limit even switchbacks fail, and it must refuse.

    Blocking requires arctan(g * cos 45) > max_slope, i.e. g > 0.66 for a
    25 degree limit -- hence 0.9 here rather than 0.6.
    """
    terr = RampTerrain(grade=0.90)             # ~42 degrees
    w = TerrainWeights(uphill=1.0, downhill=1.0, cross=0.0, max_slope_deg=25.0)
    r = AStarPlanner(open_cost, open_grid, 1, terrain=terr, weights=w).plan(
        (-20.0, 0.0), (20.0, 0.0))
    assert not r.found
    assert r.reason == "no path"


def test_cross_slope_is_penalised(open_cost, open_grid):
    """Traversing a slope sideways is a rollover risk and must cost extra."""
    terr = RampTerrain(grade=0.30)
    p_no = AStarPlanner(open_cost, open_grid, 1, terrain=terr,
                        weights=TerrainWeights(0.0, 0.0, 0.0, 89.0))
    p_cross = AStarPlanner(open_cost, open_grid, 1, terrain=terr,
                           weights=TerrainWeights(0.0, 0.0, 10.0, 89.0))
    # Moving along +y is pure cross-slope on an x-facing ramp.
    a, b = (0.0, -20.0), (0.0, 20.0)
    assert p_cross.plan(a, b).cost > p_no.plan(a, b).cost * 1.5


# ---------------------------------------------------------------------------
# The ablation baseline
# ---------------------------------------------------------------------------

def test_zero_weights_disable_terrain_entirely(open_cost, open_grid):
    """w=0 must be exactly the no-terrain planner, so the ablation is honest.

    If disabling the weights left any residual terrain term, the 'without
    terrain' baseline would not be a true baseline.
    """
    terr = HillTerrain()
    zero = AStarPlanner(open_cost, open_grid, 1, terrain=terr,
                        weights=TerrainWeights(0.0, 0.0, 0.0))
    none = AStarPlanner(open_cost, open_grid, 1, terrain=None)
    assert zero.dzdx is None, "zero weights should skip terrain sampling entirely"

    a, b = (-20.0, -20.0), (20.0, 20.0)
    assert zero.plan(a, b).cost == pytest.approx(none.plan(a, b).cost)
