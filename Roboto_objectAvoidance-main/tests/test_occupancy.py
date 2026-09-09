"""Log-odds occupancy mapping.

The headline test builds a map from GROUND-TRUTH poses. That isolates the
mapping from the scan matcher: if this passes and live SLAM is still bad,
the bug is in pose estimation, not in mapping. Establishing that boundary
early is what makes the later debugging tractable.
"""

import numpy as np
import pytest

from roboto_core.frames import GridSpec
from roboto_core.slam.occupancy import OccupancyGrid
from roboto_core.slam.scan import Scan
from roboto_core.sim.raycast_sim import Lidar2D, LidarSpec, RaycastWorld


@pytest.fixture
def room():
    """12 m x 12 m walled room at 0.1 m, with an interior pillar."""
    g = GridSpec(origin_x=-6.0, origin_y=-6.0, resolution=0.1, width=120, height=120)
    occ = np.zeros(g.shape, dtype=bool)
    occ[0, :] = occ[-1, :] = True
    occ[:, 0] = occ[:, -1] = True
    occ[70:80, 70:80] = True                 # pillar, NE of centre
    return RaycastWorld(occ, g)


def _map_from_truth(room, poses, beams=360, standoff=0.05):
    """Map from exact poses.

    Uses a SMALL free-trace stand-off deliberately. The production default
    (0.30 m) exists to stop pose error eroding walls; with exact poses it
    only costs accuracy, because near-wall cells then get no free evidence
    to suppress occasional grazing hits. These tests are about mapping
    geometry, so they isolate it from that robustness margin -- see
    test_standoff_trades_accuracy_for_robustness.
    """
    spec = LidarSpec(num_beams=beams, fov_deg=360.0, range_max=25.0,
                     range_noise_std=0.0)
    lidar = Lidar2D(room, spec, rng=0)
    grid = OccupancyGrid(room.grid, free_standoff_m=standoff)
    for p in poses:
        grid.integrate_scan(p, lidar.scan(p, noise=False))
    return grid


def test_recovers_the_world_from_truth_poses(room):
    """The core acceptance test for mapping."""
    poses = [(x, y, 0.0)
             for x in (-3.0, 0.0, 3.0)
             for y in (-3.0, 0.0, 3.0)]
    grid = _map_from_truth(room, poses)

    iou = grid.iou(room.occ)
    assert iou > 0.80, f"IoU {iou:.3f} -- mapping is not reproducing the world"
    assert grid.coverage > 0.5


def test_pillar_is_found(room):
    """Confident occupancy needs repeated hits: l_occ=0.85 vs threshold 2.0.

    Three views of each face, not one -- a single hit is deliberately not
    enough to call a cell occupied.
    """
    poses = [(-2.0, 1.5, 0.0)] * 3 + [(4.5, 1.5, 0.0)] * 3
    grid = _map_from_truth(room, poses)
    # Pillar spans rows/cols 70..80 (world x,y in 1.0..2.0 m).
    assert grid.occupied()[70:80, 70:80].any(), "pillar faces should read occupied"
    assert grid.log_odds[70:80, 70:80].max() > 0


def test_interior_is_carved_free(room):
    """One scan moves a cell toward free; confidence takes several."""
    r, c = room.grid.world_to_cell(0.5, 0.5)

    one = _map_from_truth(room, [(0.0, 0.0, 0.0)])
    assert one.log_odds[int(r), int(c)] < 0, "one scan must move the cell toward free"
    assert not one.free()[int(r), int(c)], "one scan must NOT yet be confident"

    many = _map_from_truth(room, [(0.0, 0.0, 0.0)] * 6)
    assert many.free()[int(r), int(c)], "repeated observation must reach confidence"


def test_unobserved_cells_stay_unknown(room):
    """The property discrepancy detection depends on absolutely."""
    grid = _map_from_truth(room, [(0.0, 0.0, 0.0)])
    assert not grid.observed().all(), "a single scan cannot observe every cell"

    unseen = ~grid.observed(min_count=1)
    assert np.allclose(grid.log_odds[unseen], 0.0), (
        "cells never sensed must retain the prior, not drift toward free"
    )


def test_observed_count_grows_with_revisits(room):
    one = _map_from_truth(room, [(0.0, 0.0, 0.0)])
    five = _map_from_truth(room, [(0.0, 0.0, 0.0)] * 5)
    assert five.observed_count.max() > one.observed_count.max()


def test_no_return_beams_carve_free_space_without_obstacles():
    """A beam that returns nothing is evidence of emptiness, not missing data."""
    g = GridSpec(origin_x=-10.0, origin_y=-10.0, resolution=0.1, width=200, height=200)
    grid = OccupancyGrid(g)

    angles = np.linspace(-np.pi, np.pi, 72, endpoint=False)
    empty = Scan(ranges=np.full(72, np.inf), angles=angles,
                 range_min=0.15, range_max=10.0)
    grid.integrate_scan((0.0, 0.0, 0.0), empty)

    for _ in range(5):                      # reach the confidence threshold
        grid.integrate_scan((0.0, 0.0, 0.0), empty)

    assert not grid.occupied().any(), "no-return beams must not create obstacles"
    assert grid.free().any(), "no-return beams must still carve free space"
    assert grid.observed().any()


def test_log_odds_are_clamped(room):
    """Without clamping, a long stationary dwell makes cells unupdatable."""
    grid = _map_from_truth(room, [(0.0, 0.0, 0.0)] * 60)
    assert grid.log_odds.max() <= grid.l_max + 1e-6
    assert grid.log_odds.min() >= grid.l_min - 1e-6


def test_iou_is_masked_by_observation(room):
    """Restricting to observed cells must not be gamed by never looking."""
    grid = _map_from_truth(room, [(0.0, 0.0, 0.0)])
    masked = grid.iou(room.occ, observed_only=True)
    raw = grid.iou(room.occ, observed_only=False)
    assert masked >= raw, "unobserved cells can only dilute the score"


def test_grid_geometry_must_match_prior(room):
    """Comparing against a differently-shaped truth must fail loudly."""
    grid = OccupancyGrid(room.grid)
    with pytest.raises(ValueError, match="!="):
        grid.iou(np.zeros((10, 10), dtype=bool))


# ---------------------------------------------------------------------------
# The free-trace stand-off
# ---------------------------------------------------------------------------

def test_standoff_trades_accuracy_for_robustness(room):
    """Larger stand-off costs IoU with exact poses but protects walls without.

    Free space is carved by every beam of every scan while a wall cell is
    hit only occasionally, so any pose error lets the trace cut into
    obstacles and erode them. Stopping the trace short prevents that. The
    price, visible only when poses are exact, is that near-wall cells get no
    free evidence to cancel grazing hits.
    """
    poses = [(x, y, 0.0) for x in (-3.0, 0.0, 3.0) for y in (-3.0, 0.0, 3.0)]
    tight = _map_from_truth(room, poses, standoff=0.05)
    loose = _map_from_truth(room, poses, standoff=0.30)
    assert tight.iou(room.occ) > loose.iou(room.occ)


def test_standoff_prevents_wall_erosion_under_pose_error(room):
    """The behaviour the stand-off exists for.

    With jittered poses a short stand-off lets the free-trace eat real
    walls. This is what made 72% of true wall cells read as free on a full
    mission, which would flood discrepancy detection with phantom obstacles.
    """
    rng = np.random.default_rng(0)
    spec = LidarSpec(num_beams=360, fov_deg=360.0, range_max=25.0,
                     range_noise_std=0.0)
    lidar = Lidar2D(room, spec, rng=0)
    poses = [(x, y, 0.0)
             for x in np.linspace(-4, 4, 9) for y in np.linspace(-4, 4, 9)]
    scans = [lidar.scan(p, noise=False) for p in poses]

    def eroded(standoff):
        m = OccupancyGrid(room.grid, free_standoff_m=standoff)
        r = np.random.default_rng(0)
        for p, sc in zip(poses, scans):
            q = np.array(p, dtype=float)
            q[:2] += r.normal(0.0, 0.25, 2)      # 25 cm pose error
            m.integrate_scan(q, sc)
        walls = room.occ & m.observed()
        return float((m.log_odds[walls] <= -2).mean())

    assert eroded(0.30) < eroded(0.02), (
        "a larger stand-off must reduce wall erosion under pose error"
    )
