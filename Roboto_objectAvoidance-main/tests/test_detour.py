"""Local detour, and the sampling planner.

The detour exists because a global replan, while correct, looks wrong: an
obstacle a quarter of the way along re-routes the remaining three quarters,
so the robot appears to abandon its plan rather than step around a barrier.

What matters is not that a detour is always better -- it is not -- but
that it stays local when it can and FAILS CLEANLY when it cannot, because
the caller has to fall back to a global replan in that case. Most of these
tests are about the failure side.
"""

import numpy as np
import pytest

from roboto_core.frames import GridSpec
from roboto_core.plan.astar import LETHAL, AStarPlanner, Objective
from roboto_core.plan.cost_fusion import first_blocked_index
from roboto_core.plan.detour import plan_detour
from roboto_core.plan.rrt import RRTConfig, RRTStarPlanner


@pytest.fixture
def grid():
    """120 m x 120 m at 0.5 m."""
    return GridSpec(origin_x=0.0, origin_y=0.0, resolution=0.5,
                    width=240, height=240)


@pytest.fixture
def empty(grid):
    return np.zeros(grid.shape, dtype=np.uint8)


@pytest.fixture
def straight(grid):
    """A route straight up the middle, west to east at y = 60."""
    x = np.linspace(10.0, 110.0, 201)
    return np.column_stack([x, np.full_like(x, 60.0)])


def _barrier(grid, cx, half_across=6.0, half_along=1.5):
    """A wall across the route at x = cx, with open ground either side."""
    cost = np.zeros(grid.shape, dtype=np.uint8)
    r0, c0 = grid.world_to_cell(cx - half_along, 60.0 - half_across)
    r1, c1 = grid.world_to_cell(cx + half_along, 60.0 + half_across)
    cost[int(r0):int(r1), int(c0):int(c1)] = LETHAL
    return cost


def _planner(cost, grid):
    p = AStarPlanner(cost, grid, downsample=1,
                     objective=Objective("d", cost_scale=0.0))
    return lambda a, b: p.plan(a, b)


# ---------------------------------------------------------------------------
# Locating the blockage
# ---------------------------------------------------------------------------

def test_first_blocked_index_finds_the_barrier(grid, straight):
    cost = _barrier(grid, 60.0)
    i = first_blocked_index(straight, grid, cost)
    assert i is not None
    assert straight[i][0] == pytest.approx(60.0, abs=3.0)


def test_first_blocked_index_is_none_on_a_clear_route(grid, straight, empty):
    assert first_blocked_index(straight, grid, empty) is None


def test_first_blocked_index_respects_from_index(grid, straight):
    """An obstacle already behind the robot is not a reason to react."""
    cost = _barrier(grid, 30.0)
    assert first_blocked_index(straight, grid, cost, from_index=0) is not None
    assert first_blocked_index(straight, grid, cost, from_index=120) is None


# ---------------------------------------------------------------------------
# The detour itself
# ---------------------------------------------------------------------------

def test_detour_rejoins_the_original_route(grid, straight):
    cost = _barrier(grid, 60.0)
    res = plan_detour(_planner(cost, grid), straight, (10.0, 60.0), grid, cost)

    assert res.found, res.reason
    assert res.rejoin_index > 0
    # It ends where the original ended: the goal is unchanged.
    assert np.allclose(res.path[-1], straight[-1])
    # And it actually goes around rather than through.
    assert first_blocked_index(res.path, grid, cost) is None


def test_detour_stays_local(grid, straight):
    """The point of the exercise: most of the original route survives."""
    cost = _barrier(grid, 60.0)
    res = plan_detour(_planner(cost, grid), straight, (10.0, 60.0), grid, cost)

    assert res.found
    kept = len(straight) - res.rejoin_index
    assert kept > 0.3 * len(straight), "the detour replaced most of the route"
    assert res.detour_m < 0.9 * float(
        np.hypot(*np.diff(straight, axis=0).T).sum())


def test_detour_reports_nothing_to_do_on_a_clear_route(grid, straight, empty):
    res = plan_detour(_planner(empty, grid), straight, (10.0, 60.0), grid, empty)
    assert not res.found
    assert res.reason == "not blocked"


def test_detour_fails_cleanly_when_walled_off(grid, straight):
    """A barrier spanning the whole map has no way round. The detour must
    say so rather than return something invalid -- the caller falls back to
    a global replan, which will also fail, and the mission ends honestly."""
    cost = np.zeros(grid.shape, dtype=np.uint8)
    c0, c1 = grid.world_to_cell(59.0, 0.0)[1], grid.world_to_cell(61.0, 0.0)[1]
    cost[:, int(c0):int(c1)] = LETHAL

    res = plan_detour(_planner(cost, grid), straight, (10.0, 60.0), grid, cost)
    assert not res.found
    assert "no rejoin" in res.reason
    assert res.path.shape == (0, 2)


def test_detour_does_not_aim_at_a_blocked_rejoin_point(grid, straight):
    """Rejoin candidates inside the obstacle must be skipped, or the
    planner is asked for a route to a lethal cell and simply fails."""
    cost = _barrier(grid, 60.0, half_along=8.0)     # a deep barrier
    res = plan_detour(_planner(cost, grid), straight, (10.0, 60.0), grid, cost,
                      clear_m=2.0, step_m=2.0)

    if res.found:
        r, c = grid.world_to_cell(*straight[res.rejoin_index])
        assert cost[int(r), int(c)] < LETHAL


def test_detour_gives_up_beyond_the_search_horizon(grid, straight):
    cost = _barrier(grid, 60.0)
    res = plan_detour(_planner(cost, grid), straight, (10.0, 60.0), grid, cost,
                      clear_m=1.0, max_rejoin_m=2.0, step_m=1.0)
    assert not res.found


# ---------------------------------------------------------------------------
# RRT*
# ---------------------------------------------------------------------------

def test_rrt_finds_a_route_around_a_barrier(grid, straight):
    cost = _barrier(grid, 60.0)
    res = RRTStarPlanner(cost, grid, RRTConfig(max_samples=4000)).plan(
        (10.0, 60.0), (110.0, 60.0))

    assert res.found, res.reason
    assert first_blocked_index(res.path, grid, cost) is None


def test_rrt_refuses_a_lethal_endpoint(grid):
    cost = _barrier(grid, 60.0)
    p = RRTStarPlanner(cost, grid, RRTConfig(max_samples=200))
    assert p.plan((60.0, 60.0), (110.0, 60.0)).reason == "start in lethal cell"
    assert p.plan((10.0, 60.0), (60.0, 60.0)).reason == "goal in lethal cell"


def test_rrt_does_not_cut_through_walls(grid, straight):
    """Edges are collision-checked by sampling. Too coarse a step and the
    tree steps straight over a thin wall, which is the classic sampling
    planner bug and is invisible in the returned cost."""
    cost = np.zeros(grid.shape, dtype=np.uint8)
    c0, c1 = grid.world_to_cell(59.0, 0.0)[1], grid.world_to_cell(60.0, 0.0)[1]
    cost[:, int(c0):int(c1)] = LETHAL                 # full-height, 1 m thick

    res = RRTStarPlanner(cost, grid, RRTConfig(max_samples=1500)).plan(
        (10.0, 60.0), (110.0, 60.0))
    assert not res.found, "found a route through a sealed wall"


def test_rrt_improves_with_more_samples(grid, empty):
    """Asymptotic optimality, stated as the property that matters: more
    samples must not give a worse route."""
    few = RRTStarPlanner(empty, grid, RRTConfig(max_samples=400, seed=1)).plan(
        (10.0, 10.0), (110.0, 110.0))
    many = RRTStarPlanner(empty, grid, RRTConfig(max_samples=4000, seed=1)).plan(
        (10.0, 10.0), (110.0, 110.0))

    assert few.found and many.found
    assert many.cost <= few.cost * 1.02


def test_rrt_is_worse_than_astar_on_an_open_map(grid, empty):
    """The honest comparison. A* is optimal on the grid; RRT* samples, so
    on open ground it returns a visibly longer route for the same job."""
    a = AStarPlanner(empty, grid, downsample=1,
                     objective=Objective("d", cost_scale=0.0)).plan(
        (10.0, 10.0), (110.0, 110.0))
    r = RRTStarPlanner(empty, grid, RRTConfig(max_samples=3000)).plan(
        (10.0, 10.0), (110.0, 110.0))

    assert a.found and r.found
    assert r.length_m >= a.length_m - 1e-6


def test_rrt_is_deterministic_for_a_seed(grid, empty):
    kw = dict(max_samples=800, seed=7)
    a = RRTStarPlanner(empty, grid, RRTConfig(**kw)).plan((10.0, 10.0), (100.0, 100.0))
    b = RRTStarPlanner(empty, grid, RRTConfig(**kw)).plan((10.0, 10.0), (100.0, 100.0))
    assert a.cost == pytest.approx(b.cost)


# ---------------------------------------------------------------------------
# What RRT* can and cannot optimise
# ---------------------------------------------------------------------------

def test_rrt_refuses_an_asymmetric_objective(grid, empty):
    """Directional slope is not symmetric, and RRT*'s asymptotic optimality
    assumes a metric cost. It would still return valid routes, so the
    failure has to be explicit rather than a quietly weaker guarantee."""
    from roboto_core.plan.astar import TerrainWeights

    with pytest.raises(ValueError, match="asymmetric"):
        RRTStarPlanner(empty, grid, objective=Objective(
            "effort", weights=TerrainWeights(6.0, 1.5, 3.0, 25.0)))
    with pytest.raises(ValueError, match="asymmetric"):
        RRTStarPlanner(empty, grid, objective=Objective(
            "time", weights=TerrainWeights(0, 0, 0, 25.0), time_based=True))


def test_rrt_accepts_symmetric_objectives(grid, empty):
    """Clearance integrates the costmap along an edge, which reads the same
    in either direction -- so distance and safe are fine."""
    from roboto_core.plan.astar import TerrainWeights

    flat = TerrainWeights(0, 0, 0, 25.0)
    for obj in (Objective("distance", cost_scale=0.0, weights=flat),
                Objective("safe", cost_scale=0.40, weights=flat)):
        assert RRTStarPlanner(empty, grid, objective=obj) is not None


def test_rrt_edge_cost_is_symmetric(grid):
    """The property the whole distinction rests on."""
    from roboto_core.plan.astar import TerrainWeights

    cost = np.zeros(grid.shape, dtype=np.uint8)
    cost[100:140, 100:140] = 180
    p = RRTStarPlanner(cost, grid, objective=Objective(
        "safe", cost_scale=0.40, weights=TerrainWeights(0, 0, 0, 25.0)))

    a, b = (40.0, 40.0), (80.0, 80.0)
    assert p._edge_cost(a, b) == pytest.approx(p._edge_cost(b, a))


def test_rrt_safe_keeps_more_clearance_than_distance(grid):
    """The objective has to change the route, or supporting it is theatre."""
    from roboto_core.plan.astar import TerrainWeights

    flat = TerrainWeights(0, 0, 0, 25.0)
    cost = np.zeros(grid.shape, dtype=np.uint8)
    cost[80:160, 100:140] = 220                     # costly, not lethal

    kw = dict(config=RRTConfig(max_samples=5000, seed=3))
    dist = RRTStarPlanner(cost, grid, objective=Objective(
        "d", cost_scale=0.0, weights=flat), **kw).plan((10.0, 60.0), (110.0, 60.0))
    safe = RRTStarPlanner(cost, grid, objective=Objective(
        "s", cost_scale=0.40, weights=flat), **kw).plan((10.0, 60.0), (110.0, 60.0))
    assert dist.found and safe.found

    def worst(res):
        r, c = grid.world_to_cell(res.path[:, 0], res.path[:, 1])
        return int(cost[np.clip(r, 0, 239), np.clip(c, 0, 239)].max())

    assert worst(safe) <= worst(dist)
