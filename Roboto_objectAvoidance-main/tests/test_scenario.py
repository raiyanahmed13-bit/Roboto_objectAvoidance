"""Randomised obstacle scenarios.

What these pin down is the experimental design, not just the code. The
whole argument for `--seeds` is that trials are independent; if the sampler
quietly produced the same layout every time, or placed obstacles inside
buildings, or laid a "blocking" obstacle along the route instead of across
it, every headline number downstream would be measuring something other
than what the report claims.
"""

import numpy as np
import pytest

from roboto_core.frames import GridSpec
from roboto_core.sim.scenario import (CATALOGUE, Placement, rasterise,
                                      rect_cells, sample_placements,
                                      scenario_rng)


@pytest.fixture
def grid():
    """120 m x 120 m at 0.10 m, origin at the corner."""
    return GridSpec(origin_x=-60.0, origin_y=-60.0, resolution=0.10,
                    width=1200, height=1200)


@pytest.fixture
def straight_route():
    """100 m of due-east travel from (-50, 0). Heading is exactly 0."""
    x = np.linspace(-50.0, 50.0, 251)
    return np.column_stack([x, np.zeros_like(x), np.zeros_like(x)])


@pytest.fixture
def empty_prior(grid):
    return np.zeros(grid.shape, dtype=bool)


# ---------------------------------------------------------------------------
# Rectangle geometry
# ---------------------------------------------------------------------------

def test_axis_aligned_rect_has_expected_area(grid):
    p = Placement("box", cx=0.0, cy=0.0, yaw=0.0, half_len_m=2.0, half_wid_m=1.0)
    cells = rect_cells(grid, p)
    area = len(cells) * grid.resolution ** 2
    assert area == pytest.approx(p.area_m2, rel=0.05)


def test_rotated_rect_is_not_rasterised_as_its_bounding_box(grid):
    """The bug this replaces: filling the AABB of a rotated rectangle.

    A 8 m x 0.8 m bar at 45 degrees has an AABB of about 6.2 m square --
    nearly 40 m^2 against a true 6.4 m^2. Filling the box would inflate
    ground-truth area six-fold.
    """
    p = Placement("bar", cx=0.0, cy=0.0, yaw=np.pi / 4,
                  half_len_m=4.0, half_wid_m=0.4)
    area = len(rect_cells(grid, p)) * grid.resolution ** 2
    assert area == pytest.approx(p.area_m2, rel=0.05)

    x0, y0, x1, y1 = p.aabb
    aabb_area = (x1 - x0) * (y1 - y0)
    assert aabb_area > 4 * p.area_m2      # the bound really is that loose


def test_rotation_preserves_area(grid):
    """Area must not depend on heading, or detection metrics inherit a bias
    that varies with where on the route an obstacle happened to land."""
    areas = []
    for yaw in np.linspace(0, np.pi, 7):
        p = Placement("bar", cx=0.0, cy=0.0, yaw=float(yaw),
                      half_len_m=3.0, half_wid_m=0.5)
        areas.append(len(rect_cells(grid, p)) * grid.resolution ** 2)
    assert max(areas) - min(areas) < 0.15 * np.mean(areas)


def test_distance_to_rotated_rect(grid):
    p = Placement("bar", cx=0.0, cy=0.0, yaw=np.pi / 2,
                  half_len_m=4.0, half_wid_m=0.5)
    # Rotated 90 deg: the long axis runs along +y, so the near faces in x
    # are half_wid away and in y are half_len away.
    assert p.distance_to((0.0, 0.0)) == 0.0
    assert p.distance_to((2.0, 0.0)) == pytest.approx(1.5)
    assert p.distance_to((0.0, 6.0)) == pytest.approx(2.0)


def test_rect_cells_clipped_at_grid_edge(grid):
    """A placement half off the grid must not raise or wrap around."""
    p = Placement("box", cx=-59.5, cy=0.0, yaw=0.0,
                  half_len_m=3.0, half_wid_m=1.0)
    cells = rect_cells(grid, p)
    assert len(cells) > 0
    assert cells[:, 0].min() >= 0 and cells[:, 1].min() >= 0
    assert cells[:, 0].max() < grid.height and cells[:, 1].max() < grid.width


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------

def test_same_seed_reproduces_the_scenario(grid, straight_route, empty_prior):
    kw = dict(mode="blocking", n_range=(3, 3))
    a = sample_placements(scenario_rng(4), straight_route, empty_prior, grid, **kw)
    b = sample_placements(scenario_rng(4), straight_route, empty_prior, grid, **kw)
    assert [p.label for p in a] == [p.label for p in b]
    assert np.allclose([p.cx for p in a], [p.cx for p in b])
    assert np.allclose([p.yaw for p in a], [p.yaw for p in b])


def test_different_seeds_give_different_scenarios(grid, straight_route,
                                                  empty_prior):
    """The whole point of the change: --seeds must be independent trials.

    Compared over several seeds rather than two, because two draws from a
    small catalogue can legitimately collide.
    """
    sigs = set()
    for seed in range(6):
        pl = sample_placements(scenario_rng(seed), straight_route, empty_prior,
                               grid, mode="blocking", n_range=(2, 4))
        sigs.add(tuple((p.label, round(p.cx, 2), round(p.frac, 4)) for p in pl))
    assert len(sigs) >= 5


def test_scenario_stream_is_independent_of_draw_order(grid, straight_route,
                                                      empty_prior):
    """Scenario randomness must not shift when an unrelated model consumes
    random numbers, or every previously reported figure silently changes."""
    kw = dict(mode="blocking", n_range=(3, 3))
    a = sample_placements(scenario_rng(2), straight_route, empty_prior, grid, **kw)

    noise = np.random.default_rng(2)
    noise.normal(size=1000)               # a sensor model, burning entropy
    b = sample_placements(scenario_rng(2), straight_route, empty_prior, grid, **kw)
    assert [round(p.cx, 6) for p in a] == [round(p.cx, 6) for p in b]


def test_blocking_obstacles_lie_across_the_route(grid, straight_route,
                                                 empty_prior):
    """Across, not along -- otherwise the robot threads past and the
    collision ablation proves nothing."""
    pl = sample_placements(scenario_rng(1), straight_route, empty_prior, grid,
                           mode="blocking", n_range=(4, 4))
    assert pl
    for p in pl:
        # Route heading is 0, so a spanning obstacle is near +/- 90 degrees.
        off = abs(np.degrees(np.arctan2(np.sin(p.yaw), np.cos(p.yaw))))
        assert 60.0 <= off <= 120.0
        assert abs(p.cy) <= 1.0           # sits on the route, not beside it


def test_roadside_obstacles_lie_beside_the_route(grid, straight_route,
                                                 empty_prior):
    pl = sample_placements(scenario_rng(1), straight_route, empty_prior, grid,
                           mode="roadside", n_range=(4, 4), lateral_m=(2.0, 4.5))
    assert pl
    for p in pl:
        off = abs(np.degrees(np.arctan2(np.sin(p.yaw), np.cos(p.yaw))))
        assert off <= 25.0                # aligned with travel
        assert 2.0 <= abs(p.cy) <= 4.5    # offset to one side


def test_blocking_obstacles_span_the_corridor(grid, straight_route,
                                              empty_prior):
    """Each blocker must actually cover the route centreline."""
    pl = sample_placements(scenario_rng(5), straight_route, empty_prior, grid,
                           mode="blocking", n_range=(3, 3))
    assert pl
    for p in pl:
        assert p.distance_to((p.cx, 0.0)) == 0.0


def test_obstacles_are_separated_along_the_route(grid, straight_route,
                                                 empty_prior):
    """Two obstacles on top of each other merge into one detection and
    quietly reduce the effective sample size."""
    pl = sample_placements(scenario_rng(3), straight_route, empty_prior, grid,
                           mode="blocking", n_range=(4, 4),
                           min_separation_m=25.0)
    xs = sorted(p.cx for p in pl)
    assert all(b - a >= 20.0 for a, b in zip(xs, xs[1:]))


def test_obstacles_avoid_the_start_and_the_goal(grid, straight_route,
                                                empty_prior):
    """An obstacle on the spawn traps the robot before it moves; one on the
    goal makes the mission unachievable for reasons unrelated to the layer."""
    pl = sample_placements(scenario_rng(0), straight_route, empty_prior, grid,
                           mode="blocking", n_range=(4, 4),
                           frac_range=(0.15, 0.85))
    for p in pl:
        assert 0.15 <= p.frac <= 0.85


def test_placements_inside_buildings_are_rejected(grid, straight_route):
    """Obstacles the robot can never reach would score as missed detections
    the detector never had a chance at."""
    prior = np.zeros(grid.shape, dtype=bool)
    # Wall off everything north and south of a 6 m corridor along the route.
    r_lo, _ = grid.world_to_cell(0.0, -3.0)
    r_hi, _ = grid.world_to_cell(0.0, 3.0)
    prior[:int(r_lo), :] = True
    prior[int(r_hi):, :] = True

    pl = sample_placements(scenario_rng(7), straight_route, prior, grid,
                           mode="roadside", n_range=(4, 4),
                           lateral_m=(2.0, 4.5))
    for p in pl:
        cells = rect_cells(grid, p)
        assert prior[cells[:, 0], cells[:, 1]].mean() <= 0.05


def test_no_detour_means_no_blocking_placement(grid, straight_route):
    """A barrier in a corridor narrower than the detour requirement has no
    solution; a mission failing there measures the site, not the planner."""
    prior = np.zeros(grid.shape, dtype=bool)
    r_lo, _ = grid.world_to_cell(0.0, -0.6)
    r_hi, _ = grid.world_to_cell(0.0, 0.6)
    prior[:int(r_lo), :] = True
    prior[int(r_hi):, :] = True

    pl = sample_placements(scenario_rng(1), straight_route, prior, grid,
                           mode="blocking", n_range=(3, 3),
                           detour_clearance_m=1.5)
    assert pl == []


def test_requested_count_is_respected_in_open_ground(grid, straight_route,
                                                     empty_prior):
    for n in (1, 2, 3):
        pl = sample_placements(scenario_rng(n), straight_route, empty_prior,
                               grid, mode="blocking", n_range=(n, n))
        assert len(pl) == n


def test_unknown_mode_is_rejected(grid, straight_route, empty_prior):
    with pytest.raises(ValueError, match="unknown mode"):
        sample_placements(scenario_rng(0), straight_route, empty_prior, grid,
                          mode="sideways")


def test_short_route_yields_nothing(grid, empty_prior):
    assert sample_placements(scenario_rng(0), np.zeros((1, 3)), empty_prior,
                             grid) == []


# ---------------------------------------------------------------------------
# Rasterisation into a truth world
# ---------------------------------------------------------------------------

def test_rasterise_adds_obstacles_without_touching_the_prior(grid,
                                                             straight_route):
    prior = np.zeros(grid.shape, dtype=bool)
    prior[100:120, 100:120] = True
    before = prior.copy()

    pl = sample_placements(scenario_rng(2), straight_route, prior, grid,
                           mode="blocking", n_range=(2, 2))
    truth, gts = rasterise(prior, grid, pl)

    assert np.array_equal(prior, before)          # input untouched
    assert truth[before].all()                    # prior preserved
    assert truth.sum() > before.sum()             # obstacles added
    assert len(gts) == len(pl)


def test_ground_truth_area_matches_the_rectangle(grid, straight_route,
                                                 empty_prior):
    pl = sample_placements(scenario_rng(8), straight_route, empty_prior, grid,
                           mode="roadside", n_range=(3, 3))
    _, gts = rasterise(empty_prior, grid, pl)
    for g in gts:
        assert g["area_m2"] == pytest.approx(g["place"].area_m2, rel=0.10)


def test_ground_truth_centroid_sits_on_the_placement(grid, straight_route,
                                                     empty_prior):
    pl = sample_placements(scenario_rng(9), straight_route, empty_prior, grid,
                           mode="roadside", n_range=(3, 3))
    _, gts = rasterise(empty_prior, grid, pl)
    for g in gts:
        assert g["place"].distance_to(g["centroid"]) == 0.0
        assert np.hypot(g["centroid"][0] - g["place"].cx,
                        g["centroid"][1] - g["place"].cy) < 0.2


def test_catalogue_sizes_are_plausible():
    """Guards against a typo turning a dumpster into a building."""
    for t in CATALOGUE:
        assert 0.3 <= t.width_m[0] <= t.width_m[1] <= 3.0
        assert 1.0 <= t.length_m[0] <= t.length_m[1] <= 12.0
        assert t.width_m[1] <= t.length_m[0]      # length is the long axis
