"""Search algorithm and cost objective, the two axes of the planner.

They are independent and both matter:

  * the ALGORITHM decides how much work finding the route costs, and
    whether the route you get is optimal;
  * the OBJECTIVE decides which route is the right one at all.

The headline these pin down is that the shortest path is not the best
path. If every objective returned the same route the comparison would be
meaningless, so several of these tests assert that the routes actually
DIFFER, and in the expected direction.
"""

import numpy as np
import pytest

from roboto_core.frames import GridSpec
from roboto_core.plan.astar import (LETHAL, AStarPlanner, Objective,
                                    TerrainWeights)
from roboto_core.plan.terrain import TerrainGrid


@pytest.fixture
def grid():
    """60 m x 60 m at 1 m."""
    return GridSpec(origin_x=0.0, origin_y=0.0, resolution=1.0,
                    width=60, height=60)


@pytest.fixture
def empty(grid):
    return np.zeros(grid.shape, dtype=np.uint8)


@pytest.fixture
def slalom(grid):
    """Walls with staggered gaps, so the search has to work.

    On an empty grid A* beams straight at the goal and expands barely more
    than the path length -- there is nothing for a weighted search to skip,
    so the trade-off it is supposed to demonstrate cannot appear.
    """
    cost = np.zeros(grid.shape, dtype=np.uint8)
    for i, col in enumerate((15, 30, 45)):
        cost[:, col:col + 2] = LETHAL
        gap = 5 if i % 2 == 0 else 50
        cost[gap:gap + 8, col:col + 2] = 0
    return cost


@pytest.fixture
def ridge(grid):
    """A hill across the direct line, with a flat pass to the south.

    Going straight means climbing; detouring south is longer but level.
    That is the whole trade the objectives are supposed to express.
    """
    rows = np.arange(grid.height)
    cols = np.arange(grid.width)
    x, _ = grid.cell_to_world(0, cols)
    _, y = grid.cell_to_world(rows, 0)
    X, Y = np.meshgrid(np.asarray(x), np.asarray(y))

    across = np.exp(-(((X - 30.0) / 10.0) ** 2))      # ridge at x = 30
    gate = 1.0 - np.exp(-(((Y - 10.0) / 7.0) ** 2))   # pass at y = 10
    return TerrainGrid.from_local_dem(5.0 * across * gate, grid)


START, GOAL = (5.0, 30.0), (55.0, 30.0)


def _plan(cost, grid, **kw):
    p = AStarPlanner(cost, grid, downsample=1, **kw)
    return p.plan(START, GOAL)


# ---------------------------------------------------------------------------
# Axis 1: the search algorithm
# ---------------------------------------------------------------------------

def test_dijkstra_and_astar_agree_on_cost(empty, grid):
    """Both are optimal, so they must return the same cost. A* just gets
    there having looked at less of the map."""
    dij = _plan(empty, grid, heuristic_weight=0.0)
    ast = _plan(empty, grid, heuristic_weight=1.0)

    assert dij.found and ast.found
    assert ast.cost == pytest.approx(dij.cost, rel=1e-9)
    assert ast.expanded < dij.expanded


def test_weighted_astar_trades_optimality_for_speed(slalom, grid):
    """Over-trusting the heuristic makes it inadmissible: fewer nodes, and
    a route that is allowed to be worse. Both halves must hold, or the
    trade-off being reported is not real."""
    optimal = _plan(slalom, grid, heuristic_weight=1.0)
    greedy = _plan(slalom, grid, heuristic_weight=4.0)

    assert greedy.found
    assert greedy.expanded < optimal.expanded
    assert greedy.cost >= optimal.cost - 1e-9


def test_dijkstra_is_uninformed(empty, grid):
    """Sanity on the mechanism rather than the outcome: with no heuristic
    the frontier grows outward in all directions, so it expands a large
    fraction of the free grid."""
    dij = _plan(empty, grid, heuristic_weight=0.0)
    assert dij.expanded > 0.25 * grid.width * grid.height


# ---------------------------------------------------------------------------
# Axis 2: what "best" means
# ---------------------------------------------------------------------------

def test_distance_ignores_graded_cost_and_safe_does_not(grid):
    """A non-lethal but expensive patch straddling the direct line. The
    shortest route drives through it; the cautious one goes around."""
    cost = np.zeros(grid.shape, dtype=np.uint8)
    cost[20:40, 25:35] = 200                       # costly, not lethal

    dist = _plan(cost, grid, objective=Objective("d", cost_scale=0.0))
    safe = _plan(cost, grid, objective=Objective("s", cost_scale=0.40))
    assert dist.found and safe.found

    def worst(res):
        r, c = grid.world_to_cell(res.path[:, 0], res.path[:, 1])
        return int(cost[np.clip(r, 0, 59), np.clip(c, 0, 59)].max())

    assert worst(dist) > worst(safe)
    assert safe.length_m > dist.length_m       # clearance is not free


def test_time_prefers_the_flatter_line(empty, grid, ridge):
    """Minimising duration should take the level pass, because the speed
    model slows the robot climbing. It must pay for it in distance."""
    flat = TerrainWeights(0.0, 0.0, 0.0, 25.0)
    dist = _plan(empty, grid, terrain=ridge,
                 objective=Objective("d", cost_scale=0.0, weights=flat))
    time = _plan(empty, grid, terrain=ridge,
                 objective=Objective("t", cost_scale=0.0, weights=flat,
                                     time_based=True, max_speed_mps=1.0,
                                     slope_speed_falloff=15.0))
    assert dist.found and time.found

    def max_climb(res):
        r, c = grid.world_to_cell(res.path[:, 0], res.path[:, 1])
        z = ridge.z[np.clip(r, 0, 59), np.clip(c, 0, 59)]
        return float(np.max(z))

    assert max_climb(time) < max_climb(dist)
    assert time.length_m >= dist.length_m - 1e-6


def test_effort_climbs_less_than_distance(empty, grid, ridge):
    w = TerrainWeights(uphill=40.0, downhill=0.5, cross=0.0, max_slope_deg=25.0)
    dist = _plan(empty, grid, terrain=ridge,
                 objective=Objective("d", cost_scale=0.0,
                                     weights=TerrainWeights(0, 0, 0, 25.0)))
    effort = _plan(empty, grid, terrain=ridge,
                   objective=Objective("e", cost_scale=0.0, weights=w))
    assert dist.found and effort.found

    def climb(res):
        r, c = grid.world_to_cell(res.path[:, 0], res.path[:, 1])
        z = ridge.z[np.clip(r, 0, 59), np.clip(c, 0, 59)]
        dz = np.diff(z.astype(float))
        return float(dz[dz > 0].sum())

    assert climb(effort) < climb(dist)


# ---------------------------------------------------------------------------
# Admissibility -- the property that makes A* optimal
# ---------------------------------------------------------------------------

def test_time_heuristic_is_scaled_by_max_speed(empty, grid):
    """The bound must be distance / max_speed. Leaving it in metres would
    overestimate remaining time by a factor of max_speed, which is
    inadmissible and silently returns suboptimal routes."""
    obj = Objective("t", cost_scale=0.0, time_based=True, max_speed_mps=2.0)
    p = AStarPlanner(empty, grid, downsample=1, objective=obj)

    metres = grid.resolution * float(np.hypot(30 - 0, 40 - 0))
    assert p._heuristic(0, 0, 30, 40) == pytest.approx(metres / 2.0)


def test_time_astar_matches_time_dijkstra(empty, grid, ridge):
    """The real admissibility check: with a correct heuristic, A* must
    return the same cost as an uninformed search."""
    obj = Objective("t", cost_scale=0.0,
                    weights=TerrainWeights(0, 0, 0, 25.0),
                    time_based=True, max_speed_mps=1.0)
    ast = _plan(empty, grid, terrain=ridge, objective=obj,
                heuristic_weight=1.0)
    dij = _plan(empty, grid, terrain=ridge, objective=obj,
                heuristic_weight=0.0)
    assert ast.cost == pytest.approx(dij.cost, rel=1e-6)


# ---------------------------------------------------------------------------
# Wiring
# ---------------------------------------------------------------------------

def test_objective_supersedes_loose_arguments(empty, grid):
    obj = Objective("x", cost_scale=0.5,
                    weights=TerrainWeights(1.0, 2.0, 3.0, 20.0))
    p = AStarPlanner(empty, grid, downsample=1, cost_scale=0.02,
                     weights=TerrainWeights(9, 9, 9, 45.0), objective=obj)
    assert p.cost_scale == 0.5
    assert p.weights.max_slope_deg == 20.0


def test_objective_defaults_to_the_standard_slope_weights():
    """Not flat -- it inherits TerrainWeights()'s defaults, matching what
    AStarPlanner already did without an objective. A flat planner has to be
    asked for explicitly."""
    assert Objective("plain").weights == TerrainWeights()
    assert not Objective("plain").weights.disabled
    assert Objective("flat", weights=TerrainWeights(0, 0, 0, 25.0)).weights.disabled


def test_time_objective_still_needs_terrain_sampled(empty, grid, ridge):
    """Its slope weights are all zero, so a `weights.disabled` check would
    skip sampling the gradients -- and then the speed model would see flat
    ground everywhere and the objective would silently do nothing."""
    obj = Objective("t", cost_scale=0.0, time_based=True)
    assert obj.needs_terrain
    p = AStarPlanner(empty, grid, downsample=1, terrain=ridge, objective=obj)
    assert p.dzdx is not None


def test_lethal_is_still_refused_by_every_objective(grid):
    """"Shortest" never means "through a building"."""
    cost = np.zeros(grid.shape, dtype=np.uint8)
    cost[:, 28:32] = LETHAL
    cost[0:5, 28:32] = 0                      # one gap, at the south edge

    res = _plan(cost, grid, objective=Objective("d", cost_scale=0.0))
    assert res.found
    r, c = grid.world_to_cell(res.path[:, 0], res.path[:, 1])
    assert cost[np.clip(r, 0, 59), np.clip(c, 0, 59)].max() < LETHAL
