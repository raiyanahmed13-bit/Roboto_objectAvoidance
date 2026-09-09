"""Occlusion filling: what the lidar cannot see behind what it can.

A 2D lidar returns the near face of a solid object and nothing else.
Measured on a 6.9 x 2.6 m delivery truck, the scan produced 63 of the
body's 1763 cells -- 4%. Radial inflation by unknown_depth_m covered 28.6%
of the real footprint at the shipped 1.2 m, and only 65.4% at 3.0 m, so the
planner routed around the sliver it knew about and into the rest.
"""

import numpy as np
import pytest

from roboto_core.discrepancy.cluster import DiscrepancyObject
from roboto_core.discrepancy.compare import Cls
from roboto_core.frames import GridSpec
from roboto_core.plan.cost_fusion import LETHAL, fuse, shadow_cells


def make_grid(n=200, res=0.1):
    return GridSpec(width=n, height=n, resolution=res,
                    origin_x=-n * res / 2, origin_y=-n * res / 2)


def box_cells(grid, cx, cy, w, h):
    rr, cc = np.meshgrid(np.arange(grid.shape[0]), np.arange(grid.shape[1]),
                         indexing="ij")
    wx, wy = grid.cell_to_world(rr, cc)
    m = (np.abs(wx - cx) <= w / 2) & (np.abs(wy - cy) <= h / 2)
    return m, wx, wy


def near_face(mask, wx, wy, viewpoint, tol=0.15):
    d = np.hypot(wx - viewpoint[0], wy - viewpoint[1])
    return mask & (d <= d[mask].min() + tol)


def test_shadow_covers_the_body_behind_the_face():
    """The whole box, not just the strip the sensor saw."""
    grid = make_grid()
    body, wx, wy = box_cells(grid, 0.0, 0.0, 1.0, 3.0)
    view = np.array([-6.0, 0.0])
    face = near_face(body, wx, wy, view)

    assert face.sum() < body.sum() / 3, "face should be a small slice"

    hidden = shadow_cells(grid, np.argwhere(face), view, depth_m=4.0)
    covered = (face | hidden)[body].mean()
    assert covered > 0.9, f"only {covered:.0%} of the body covered"


def test_shadow_does_not_reach_around_the_object():
    """Cells beside the object, outside its angular span, stay free.

    A shadow model that floods sideways would block the very route around
    the obstacle that the robot needs.
    """
    grid = make_grid()
    body, wx, wy = box_cells(grid, 0.0, 0.0, 1.0, 2.0)
    view = np.array([-6.0, 0.0])
    hidden = shadow_cells(grid, np.argwhere(near_face(body, wx, wy, view)),
                          view, depth_m=4.0)
    beside = (np.abs(wy) > 2.5) & (np.abs(wx) < 3.0)
    assert not hidden[beside].any(), "shadow leaked sideways"


def test_shadow_does_not_fall_in_front():
    """Nothing between the sensor and the face may be marked."""
    grid = make_grid()
    body, wx, wy = box_cells(grid, 0.0, 0.0, 1.0, 2.0)
    view = np.array([-6.0, 0.0])
    hidden = shadow_cells(grid, np.argwhere(near_face(body, wx, wy, view)),
                          view, depth_m=4.0)
    in_front = wx < -1.0
    assert not hidden[in_front].any(), "shadow fell toward the sensor"


def test_viewpoint_changes_where_the_shadow_falls():
    """Approach from the other side and the hidden region flips with you."""
    grid = make_grid()
    body, wx, wy = box_cells(grid, 0.0, 0.0, 1.0, 2.0)
    left = shadow_cells(grid, np.argwhere(
        near_face(body, wx, wy, np.array([-6.0, 0.0]))),
        np.array([-6.0, 0.0]), depth_m=3.0)
    right = shadow_cells(grid, np.argwhere(
        near_face(body, wx, wy, np.array([6.0, 0.0]))),
        np.array([6.0, 0.0]), depth_m=3.0)
    assert left[wx > 0.6].any() and not left[wx < -0.6].any()
    assert right[wx < -0.6].any() and not right[wx > 0.6].any()


def test_fuse_marks_the_hidden_body_lethal():
    """End to end: fuse() with a viewpoint blocks the whole object.

    The body must be DEEPER than unknown_depth_m (1.2 m) or radial
    inflation already covers it and the test proves nothing -- which is
    exactly why the real 2.6 m deep truck failed and a shallow one would
    not have.
    """
    grid = make_grid(n=300)
    prior = np.zeros(grid.shape, dtype=np.uint8)
    body, wx, wy = box_cells(grid, 0.0, 0.0, 4.0, 3.0)   # 4 m deep
    view = np.array([-8.0, 0.0])
    cells = np.argwhere(near_face(body, wx, wy, view))
    obj = DiscrepancyObject(
        klass=Cls.MISSED, centroid=np.array([wx[body].mean(), wy[body].mean()]),
        area_m2=float(len(cells)) * grid.resolution ** 2,
        bbox=(-0.5, -1.5, 0.5, 1.5), cell_count=len(cells),
        confidence=1.0, cells=cells)

    without = fuse(prior, grid, [obj], robot_radius_m=0.35)
    with_view = fuse(prior, grid, [obj], robot_radius_m=0.35,
                     viewpoint=view, shadow_depth_m=5.0)

    cov_without = (without[body] >= LETHAL).mean()
    cov_with = (with_view[body] >= LETHAL).mean()
    assert cov_with > cov_without
    assert cov_with > 0.95, f"only {cov_with:.0%} lethal"


def test_no_viewpoint_is_unchanged():
    """Callers that cannot supply a viewpoint keep the old behaviour."""
    grid = make_grid()
    prior = np.zeros(grid.shape, dtype=np.uint8)
    body, wx, wy = box_cells(grid, 0.0, 0.0, 1.0, 2.0)
    cells = np.argwhere(near_face(body, wx, wy, np.array([-6.0, 0.0])))
    obj = DiscrepancyObject(
        klass=Cls.MISSED, centroid=np.zeros(2), area_m2=1.0,
        bbox=(-0.5, -1.0, 0.5, 1.0), cell_count=len(cells),
        confidence=1.0, cells=cells)
    a = fuse(prior, grid, [obj], robot_radius_m=0.35)
    b = fuse(prior, grid, [obj], robot_radius_m=0.35, viewpoint=None)
    assert np.array_equal(a, b)


@pytest.mark.parametrize("depth", [1.0, 2.0, 4.0])
def test_shadow_depth_is_bounded(depth):
    """The fill stops at depth_m; it does not run to the map edge."""
    grid = make_grid()
    body, wx, wy = box_cells(grid, 0.0, 0.0, 1.0, 2.0)
    view = np.array([-6.0, 0.0])
    hidden = shadow_cells(grid, np.argwhere(near_face(body, wx, wy, view)),
                          view, depth_m=depth)
    rng = np.hypot(wx - view[0], wy - view[1])
    assert not hidden[rng > rng[body].min() + depth + 0.2].any()


def test_contact_map_must_not_use_planning_inflation():
    """A contact test uses the footprint, not the planner's safety margin.

    fuse() deliberately marks lethal out to robot_radius + unknown_depth_m
    so the PLANNER leaves room around a half-seen obstacle. Probing that
    same map to ask "am I touching something" reported contact at 1.53 m --
    while the robot legitimately drives past obstacles at 1.1-1.5 m. It
    stopped in open space every time it passed anything it had detected,
    and the turn-in-place recovery then oscillated.
    """
    from scipy.ndimage import distance_transform_edt

    grid = make_grid(n=400)
    prior = np.zeros(grid.shape, dtype=np.uint8)
    body, wx, wy = box_cells(grid, 0.0, 0.0, 0.1, 7.0)   # a thin near face
    cells = np.argwhere(body)
    obj = DiscrepancyObject(
        klass=Cls.MISSED, centroid=np.zeros(2), area_m2=0.7,
        bbox=(-0.05, -3.5, 0.05, 3.5), cell_count=len(cells),
        confidence=1.0, cells=cells)

    d = distance_transform_edt(~body) * grid.resolution

    def reach(**kw):
        f = fuse(prior, grid, [obj], **kw)
        lethal = f >= LETHAL
        return float(d[lethal].max()) if lethal.any() else 0.0

    view = np.array([-8.0, 0.0])
    planning = reach(robot_radius_m=0.35, viewpoint=view)

    # WITH the viewpoint, because production passes one. The original
    # version of this test omitted it, so the occlusion shadow never ran
    # and the test reported 0.32 m while the shipped code reached 4.30 m.
    # A guard that does not exercise the production path guards nothing.
    contact = reach(robot_radius_m=0.35, unknown_depth_m=0.0,
                    inflation_m=0.0, shadow_depth_m=0.0, viewpoint=view)

    assert planning > 1.4, "planning inflation should be generous"
    assert contact < 0.5, f"contact reach {contact:.2f} m is too generous"
    # The robot must be able to pass at its measured route clearance.
    assert contact < 1.1, "contact would fire while driving past at 1.1 m"
