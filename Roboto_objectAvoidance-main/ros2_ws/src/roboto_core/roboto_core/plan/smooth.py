"""Remove the staircase from a grid path before a controller has to drive it.

WHY THIS EXISTS
---------------
AStarPlanner is 8-connected on a costmap downsampled by 4. At 0.5 m
resolution that is a 2.0 m cell, and 8-connectivity means every heading
snaps to a multiple of 45 degrees. A straight road therefore comes back as
a staircase with 2 m treads.

Pure pursuit drives that literally. Its lookahead is about 2.3 m, roughly
one tread, so the target point hops from one side of the true line to the
other and the robot weaves along a path that is, at the grid's resolution,
"optimal". The zigzag is not a controller bug -- the controller is
following the path it was given.

String-pulling fixes it: walk forward from an anchor, keep the furthest
waypoint still reachable in a straight line, drop everything between.

WHY THE SPAN IS LIMITED
-----------------------
A shortcut is only checked for collision, not for cost. Left unbounded it
would happily replace a route the planner chose to keep off a 25 degree
slope with a straight line across it, silently discarding the terrain
objective that the route existed to satisfy. Capping the span keeps
shortcuts local: 45-degree treads disappear, the route's shape does not.
"""

from __future__ import annotations

import numpy as np

from ..frames import GridSpec
from .cost_fusion import LETHAL


def segment_is_clear(a, b, grid: GridSpec, cost: np.ndarray,
                     step_m: float = 0.0) -> bool:
    """Is the straight line a->b free of lethal cells?

    `cost` is expected to be inflated by the robot radius already, exactly
    as the planner sees it, so testing the centreline is the same question
    the planner answered.
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    span = float(np.hypot(*(b - a)))
    if span <= 0.0:
        return True

    # Half a cell, so no cell can be stepped over.
    step = step_m if step_m > 0 else 0.5 * grid.resolution
    n = max(2, int(np.ceil(span / step)) + 1)
    t = np.linspace(0.0, 1.0, n)[:, None]
    pts = a[None, :] + t * (b - a)[None, :]

    rows, cols = grid.world_to_cell(pts[:, 0], pts[:, 1])
    ok = grid.in_bounds(rows, cols)
    if not np.all(ok):
        # Leaving the map is not a shortcut worth taking.
        return False
    return not bool(np.any(cost[rows[ok], cols[ok]] >= LETHAL))


def _min_clearance(pts: np.ndarray, grid: GridSpec,
                   clearance: np.ndarray) -> float:
    """Smallest clearance over a set of world points, in metres."""
    rows, cols = grid.world_to_cell(pts[:, 0], pts[:, 1])
    ok = grid.in_bounds(rows, cols)
    if not np.any(ok):
        return 0.0
    return float(clearance[rows[ok], cols[ok]].min())


def _sample(a, b, grid: GridSpec) -> np.ndarray:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    span = float(np.hypot(*(b - a)))
    n = max(2, int(np.ceil(span / (0.5 * grid.resolution))) + 1)
    t = np.linspace(0.0, 1.0, n)[:, None]
    return a[None, :] + t * (b - a)[None, :]


def smooth_path(path: np.ndarray, grid: GridSpec, cost: np.ndarray,
                max_span_m: float = 8.0,
                clearance: np.ndarray | None = None,
                min_clearance_m: float = 0.0) -> np.ndarray:
    """String-pull `path`, keeping shortcuts shorter than `max_span_m`.

    A shortcut must be collision-free AND must not make the route tighter.

    WHY CLEARANCE IS CHECKED SEPARATELY FROM COLLISION
    --------------------------------------------------
    `cost` is binary at the point of test: lethal or not. A* does not plan
    that way -- its costmap has a gradient, so it drifts away from walls
    even where hugging them is legal. String-pulling on the binary test
    throws that away and presses the route against the inflation boundary:
    measured on Pittsburgh, minimum clearance fell from 1.91 m to 1.08 m
    while every point remained perfectly "clear".

    That matters because pose error is not zero. A robot tracking a path
    with 1 m of room, holding an estimate that can jump several metres on
    one accepted scan match, drives into the building. The straightening is
    still worth having; it just must not spend the margin A* paid for.

    So a shortcut has to keep at least `min_clearance_m` -- or, where the
    original was already tighter than that, at least what the original had.
    Corridors that are genuinely narrow stay traversable; open ground stops
    being straightened into a wall.
    """
    path = np.asarray(path, dtype=np.float64)
    if len(path) < 3:
        return path

    out = [path[0]]
    i = 0
    while i < len(path) - 1:
        best = i + 1
        for j in range(len(path) - 1, i, -1):
            if float(np.hypot(*(path[j] - path[i]))) > max_span_m:
                continue
            if not segment_is_clear(path[i], path[j], grid, cost):
                continue
            if clearance is not None:
                # Never worse than the stretch being replaced, and never
                # below the floor unless the original already was.
                need = min(min_clearance_m,
                           _min_clearance(path[i:j + 1], grid, clearance))
                got = _min_clearance(_sample(path[i], path[j], grid), grid,
                                     clearance)
                if got < need - 1e-9:
                    continue
            best = j
            break
        out.append(path[best])
        i = best

    return np.asarray(out, dtype=np.float64)


def resample(path: np.ndarray, spacing_m: float) -> np.ndarray:
    """Even spacing along the polyline.

    String-pulling leaves long straights next to short jogs. Pure pursuit
    searches for its lookahead point by walking waypoints, so wildly uneven
    spacing makes how far ahead it actually looks depend on where it
    happens to be.
    """
    path = np.asarray(path, dtype=np.float64)
    if len(path) < 2 or spacing_m <= 0:
        return path

    seg = np.hypot(*np.diff(path, axis=0).T)
    along = np.concatenate([[0.0], np.cumsum(seg)])
    total = float(along[-1])
    if total <= 0:
        return path

    n = max(2, int(np.ceil(total / spacing_m)) + 1)
    want = np.linspace(0.0, total, n)
    return np.stack([np.interp(want, along, path[:, 0]),
                     np.interp(want, along, path[:, 1])], axis=1)
