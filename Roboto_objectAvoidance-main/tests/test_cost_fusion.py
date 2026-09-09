"""Folding discovered obstacles into the planning costmap.

The asymmetry here is deliberate and safety-motivated: adding an obstacle
the robot found is cheap to be wrong about (a needless detour), while
clearing one the prior asserts is expensive to be wrong about (driving into
a wall). These tests pin that down, along with the look-ahead horizon that
separates "blocked now" from "blocked eventually".
"""

import numpy as np
import pytest

from roboto_core.discrepancy.cluster import DiscrepancyObject
from roboto_core.discrepancy.compare import Cls
from roboto_core.frames import GridSpec
from roboto_core.plan.cost_fusion import (LETHAL, fuse, inflate_lethal,
                                          path_is_blocked)


@pytest.fixture
def grid():
    """40 m x 40 m at 0.1 m."""
    return GridSpec(origin_x=-20.0, origin_y=-20.0, resolution=0.1,
                    width=400, height=400)


def _object(grid, x, y, klass=Cls.MISSED, half=0.2):
    """A small detection centred on (x, y)."""
    r0, c0 = grid.world_to_cell(x - half, y - half)
    r1, c1 = grid.world_to_cell(x + half, y + half)
    rows, cols = np.mgrid[int(r0):int(r1) + 1, int(c0):int(c1) + 1]
    cells = np.column_stack([rows.ravel(), cols.ravel()])
    return DiscrepancyObject(
        klass=klass, centroid=np.array([x, y]),
        area_m2=len(cells) * grid.resolution ** 2,
        bbox=(x - half, y - half, x + half, y + half),
        cell_count=len(cells), confidence=1.0, cells=cells)


def _straight_path(x0, x1, y=0.0, n=200):
    return np.column_stack([np.linspace(x0, x1, n), np.full(n, y)])


# ---------------------------------------------------------------------------
# Fusing
# ---------------------------------------------------------------------------

def test_detected_obstacle_becomes_lethal(grid):
    prior = np.zeros(grid.shape, dtype=np.uint8)
    out = fuse(prior, grid, [_object(grid, 0.0, 0.0)], robot_radius_m=0.35)
    r, c = grid.world_to_cell(0.0, 0.0)
    assert out[int(r), int(c)] >= LETHAL


def test_lethal_core_covers_unseen_depth(grid):
    """A lidar sees the near FACE; the body extends behind it.

    Inflating by the robot radius alone leaves that body unmarked, and the
    planner routes around the thin arc straight into it -- which is exactly
    what happened in a closed-loop mission.
    """
    prior = np.zeros(grid.shape, dtype=np.uint8)
    obj = _object(grid, 0.0, 0.0, half=0.05)

    shallow = fuse(prior, grid, [obj], robot_radius_m=0.35, unknown_depth_m=0.0)
    deep = fuse(prior, grid, [obj], robot_radius_m=0.35, unknown_depth_m=1.2)
    assert (deep >= LETHAL).sum() > (shallow >= LETHAL).sum() * 4


def test_inflation_decays_outside_the_core(grid):
    prior = np.zeros(grid.shape, dtype=np.uint8)
    out = fuse(prior, grid, [_object(grid, 0.0, 0.0)],
               robot_radius_m=0.35, unknown_depth_m=0.5, inflation_m=2.0)

    near = out[grid.world_to_cell(2.0, 0.0)[0], grid.world_to_cell(2.0, 0.0)[1]]
    far = out[grid.world_to_cell(5.0, 0.0)[0], grid.world_to_cell(5.0, 0.0)[1]]
    assert near > far, "cost must decay with distance from the obstacle"
    assert far == 0, "well clear of the obstacle the prior is untouched"


def test_prior_obstacles_are_preserved(grid):
    """Fusing must never erase what the prior already asserted."""
    prior = np.zeros(grid.shape, dtype=np.uint8)
    r, c = grid.world_to_cell(10.0, 10.0)
    prior[int(r), int(c)] = LETHAL

    out = fuse(prior, grid, [_object(grid, 0.0, 0.0)])
    assert out[int(r), int(c)] >= LETHAL


def test_phantom_reduces_cost_but_never_erases_it(grid):
    """Clearing a mapped obstacle on the robot's own evidence is the
    dangerous direction, so relief is partial by design."""
    prior = np.full(grid.shape, 200, dtype=np.uint8)
    obj = _object(grid, 0.0, 0.0, klass=Cls.PHANTOM)
    out = fuse(prior, grid, [obj], phantom_relief=0.5)

    r, c = grid.world_to_cell(0.0, 0.0)
    val = out[int(r), int(c)]
    assert val < 200, "confident phantom should reduce cost"
    assert val > 0, "but never to zero -- it may still be real"


def test_no_objects_leaves_the_prior_untouched(grid):
    rng = np.random.default_rng(0)
    prior = rng.integers(0, 200, grid.shape, dtype=np.uint8)
    assert np.array_equal(fuse(prior, grid, []), prior)


# ---------------------------------------------------------------------------
# Blockage
# ---------------------------------------------------------------------------

def test_blockage_is_detected_on_the_path(grid):
    prior = np.zeros(grid.shape, dtype=np.uint8)
    cost = fuse(prior, grid, [_object(grid, 5.0, 0.0)])
    assert path_is_blocked(_straight_path(-10.0, 10.0), grid, cost)


def test_obstacle_beside_the_path_is_not_blockage(grid):
    prior = np.zeros(grid.shape, dtype=np.uint8)
    cost = fuse(prior, grid, [_object(grid, 5.0, 12.0)])
    assert not path_is_blocked(_straight_path(-10.0, 10.0), grid, cost)


def test_horizon_separates_now_from_eventually(grid):
    """The distinction that stopped 124 replans in one mission.

    A blockage 15 m ahead can wait for a closer look; the same blockage 2 m
    ahead cannot. Without the horizon the robot re-plans every cycle,
    because each new view grows the obstacle and invalidates the path it
    planned a metre earlier.
    """
    prior = np.zeros(grid.shape, dtype=np.uint8)
    cost = fuse(prior, grid, [_object(grid, 15.0, 0.0)])
    path = _straight_path(0.0, 19.0, n=200)

    assert path_is_blocked(path, grid, cost), "blocked somewhere ahead"
    assert not path_is_blocked(path, grid, cost, within_m=4.0), \
        "but not within the emergency horizon"
    assert path_is_blocked(path, grid, cost, within_m=18.0)


def test_from_index_ignores_path_already_driven(grid):
    """An obstacle behind the robot is not a reason to replan."""
    prior = np.zeros(grid.shape, dtype=np.uint8)
    cost = fuse(prior, grid, [_object(grid, -8.0, 0.0)])
    path = _straight_path(-10.0, 10.0, n=200)

    assert path_is_blocked(path, grid, cost, from_index=0)
    assert not path_is_blocked(path, grid, cost, from_index=150)


def test_corridor_widens_the_test(grid):
    """A route that merely grazes an obstacle should still count."""
    prior = np.zeros(grid.shape, dtype=np.uint8)
    cost = fuse(prior, grid, [_object(grid, 5.0, 2.2)],
                robot_radius_m=0.35, unknown_depth_m=0.5, inflation_m=0.1)
    path = _straight_path(-10.0, 10.0, y=0.0)

    assert not path_is_blocked(path, grid, cost, corridor_m=0.0)
    assert path_is_blocked(path, grid, cost, corridor_m=1.5)


def test_empty_path_is_not_blocked(grid):
    cost = np.zeros(grid.shape, dtype=np.uint8)
    assert not path_is_blocked(np.zeros((0, 2)), grid, cost)


# ---------------------------------------------------------------------------
# Prior inflation: plan conservatively, check permissively
# ---------------------------------------------------------------------------

def test_inflation_grows_the_lethal_region(grid):
    """Regression for a live collision.

    The prior marks a cell lethal within one robot radius (0.35 m), so a
    plan could legally pass 0.36 m from a wall while the controller's
    laser brake stops dead at 0.6 m. The planner was producing routes the
    controller refuses to drive.
    """
    cost = np.zeros(grid.shape, dtype=np.uint8)
    cost[200, 200] = LETHAL

    wide = inflate_lethal(cost, grid, 1.0)
    lethal = wide >= LETHAL

    # A cell 0.5 m away must now be lethal; one 1.5 m away must not.
    assert lethal[200, 205]
    assert not lethal[200, 215]
    assert lethal.sum() > (cost >= LETHAL).sum()


def test_inflation_does_not_mutate_its_input(grid):
    cost = np.zeros(grid.shape, dtype=np.uint8)
    cost[200, 200] = LETHAL
    before = cost.copy()
    inflate_lethal(cost, grid, 1.0)
    assert np.array_equal(cost, before)


def test_zero_margin_is_a_passthrough(grid):
    cost = np.zeros(grid.shape, dtype=np.uint8)
    cost[100:110, 100:110] = LETHAL
    assert np.array_equal(inflate_lethal(cost, grid, 0.0), cost)


def test_inflation_on_an_empty_map_is_a_noop(grid):
    """No lethal cells means nothing to grow -- and the distance transform
    of an all-free grid must not be allowed to mark everything lethal."""
    cost = np.zeros(grid.shape, dtype=np.uint8)
    out = inflate_lethal(cost, grid, 1.0)
    assert not (out >= LETHAL).any()


def test_inflation_preserves_graded_cost_below_lethal(grid):
    """Only the lethal region grows; the graded inflation band the prior
    already carries must survive, or the planner loses its preference for
    staying away from walls."""
    cost = np.zeros(grid.shape, dtype=np.uint8)
    cost[200, 200] = LETHAL
    cost[200, 250] = 100
    out = inflate_lethal(cost, grid, 0.5)
    assert out[200, 250] == 100


def test_margin_keeps_a_path_clear_of_the_brake_distance(grid):
    """The point of the margin, stated as the property that matters: with
    it applied, no non-lethal cell sits closer to an obstacle than the
    distance at which the controller would stop."""
    from scipy.ndimage import distance_transform_edt

    cost = np.zeros(grid.shape, dtype=np.uint8)
    cost[190:210, 190:210] = LETHAL

    brake_stop_m = 0.6
    wide = inflate_lethal(cost, grid, brake_stop_m)
    dist = distance_transform_edt(~(cost >= LETHAL)) * grid.resolution

    drivable = wide < LETHAL
    assert dist[drivable].min() > brake_stop_m
