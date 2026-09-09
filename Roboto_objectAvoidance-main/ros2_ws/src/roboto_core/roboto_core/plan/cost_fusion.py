"""Fold discovered obstacles into the planning costmap.

The GIS prior says what the world looked like when someone last surveyed
it. The discrepancy detector says how it differs today. This module merges
the two into the costmap the planner actually searches.

    prior costmap  +  MISSED objects (add)  +  PHANTOM objects (clear)
                              |
                              v
                      fused planning costmap

ASYMMETRY, AGAIN
----------------
Adding a discovered obstacle is safety-positive: the worst case is a detour
around something that was not really there. Clearing a mapped obstacle is
safety-negative: the worst case is driving into a wall on the strength of
the robot's own uncertain evidence. So MISSED objects are added at lethal
cost with full inflation, while PHANTOM objects only reduce cost, never to
zero, and only when the detector was confident.
"""

from __future__ import annotations

import numpy as np

from ..discrepancy.cluster import DiscrepancyObject
from ..discrepancy.compare import Cls
from ..frames import GridSpec

LETHAL = 254
INSCRIBED = 253


def shadow_cells(grid: GridSpec, cells: np.ndarray, viewpoint,
                 depth_m: float = 4.0) -> np.ndarray:
    """Cells hidden BEHIND a detected face, as seen from `viewpoint`.

    WHY THIS IS NEEDED
    ------------------
    A 2D lidar returns the near face of a solid object and nothing else.
    Measured on delivery_truck_3 -- a 6.9 x 2.6 m box -- the scan produced
    63 of the body's 1763 cells: FOUR PER CENT. The rest lies in the
    object's own shadow and is never observed at all.

    fuse() used to cover that by inflating radially from the seen cells by
    unknown_depth_m. That guesses a radius where the truth is a shape, and
    it does not converge: at the shipped 1.2 m only 28.6% of the real
    footprint was marked lethal, at 2.0 m 47.6%, at 3.0 m 65.4%. Radial
    growth from a one-cell-deep arc cannot reproduce a rectangle, so the
    planner kept routing confidently around the sliver it knew about and
    into the part it did not. That is what "it detected the obstacle and
    drove into it anyway" looks like from the outside.

    The right question is not "how big might it be" but "what could I not
    have seen". Everything within the face's angular span, and further away
    than the face, is unobserved -- and unobserved directly behind a known
    obstacle is occupied until something proves otherwise. That fills the
    real body whatever its shape or orientation.

    Bounded to a window around the object, so cost is set by the object's
    size rather than the map's.
    """
    if cells is None or len(cells) == 0 or viewpoint is None:
        return np.zeros(grid.shape, dtype=bool)

    vx, vy = float(viewpoint[0]), float(viewpoint[1])
    fx, fy = grid.cell_to_world(cells[:, 0], cells[:, 1])
    fr = np.hypot(fx - vx, fy - vy)
    fa = np.arctan2(fy - vy, fx - vx)
    if fr.size == 0:
        return np.zeros(grid.shape, dtype=bool)

    # Angular span, measured about the face's own mean bearing so a span
    # straddling +/-pi does not wrap into "the whole world".
    mid = np.arctan2(np.sin(fa).mean(), np.cos(fa).mean())
    rel = np.arctan2(np.sin(fa - mid), np.cos(fa - mid))
    a_lo, a_hi = rel.min(), rel.max()
    r_near = float(fr.min())

    r0, r1 = int(cells[:, 0].min()), int(cells[:, 0].max())
    c0, c1 = int(cells[:, 1].min()), int(cells[:, 1].max())
    pad = int(np.ceil(depth_m / grid.resolution)) + 2
    r0 = max(0, r0 - pad); c0 = max(0, c0 - pad)
    r1 = min(grid.shape[0] - 1, r1 + pad); c1 = min(grid.shape[1] - 1, c1 + pad)

    rr, cc = np.meshgrid(np.arange(r0, r1 + 1), np.arange(c0, c1 + 1),
                         indexing="ij")
    wx, wy = grid.cell_to_world(rr, cc)
    rng = np.hypot(wx - vx, wy - vy)
    ang = np.arctan2(wy - vy, wx - vx)
    rel_a = np.arctan2(np.sin(ang - mid), np.cos(ang - mid))

    hidden = ((rel_a >= a_lo) & (rel_a <= a_hi)
              & (rng >= r_near) & (rng <= r_near + depth_m))

    out = np.zeros(grid.shape, dtype=bool)
    out[r0:r1 + 1, c0:c1 + 1] = hidden
    return out


def fuse(prior_cost: np.ndarray, grid: GridSpec,
         objects: list[DiscrepancyObject],
         robot_radius_m: float = 0.35,
         inflation_m: float = 1.0,
         unknown_depth_m: float = 1.2,
         phantom_relief: float = 0.5,
         viewpoint=None,
         shadow_depth_m: float = 4.0) -> np.ndarray:
    """Return a copy of `prior_cost` with discovered obstacles folded in.

    `inflation_m` is the distance over which cost decays away from a new
    obstacle, beyond the lethal core. It exists so the planner leaves
    clearance rather than grazing the obstacle it just discovered.
    """
    from scipy.ndimage import distance_transform_edt

    cost = np.array(prior_cost, dtype=np.uint8, copy=True)

    missed = [o for o in objects if o.klass == Cls.MISSED]
    phantom = [o for o in objects if o.klass == Cls.PHANTOM]

    if missed:
        seed = np.zeros(grid.shape, dtype=bool)
        for o in missed:
            if o.cells is None or len(o.cells) == 0:
                continue
            seed[o.cells[:, 0], o.cells[:, 1]] = True
            # The body behind the face, which the sensor cannot see. Only
            # when a viewpoint is supplied: without one there is no "behind"
            # and the radial guess below is all that is available.
            if viewpoint is not None:
                seed |= shadow_cells(grid, o.cells, viewpoint, shadow_depth_m)

        if seed.any():
            # Distance from every cell to the nearest newly-found obstacle.
            dist = distance_transform_edt(~seed) * grid.resolution

            # robot_radius is not enough on its own. A lidar sees only the
            # NEAR FACE of an obstacle, so a detection is a thin arc while
            # the object is a solid body extending behind it. Inflating by
            # the robot radius alone leaves the body unmarked, and the
            # planner cheerfully routes around the arc and into it -- which
            # is exactly what happened: a replan at 10.5 m still closed to
            # 3.1 m before failing. unknown_depth_m covers the part of the
            # object we cannot see yet.
            lethal = dist <= (robot_radius_m + unknown_depth_m)
            cost[lethal] = LETHAL

            # Exponential decay outside the lethal core, matching the shape
            # ROS costmap_2d uses so the two layers combine sensibly.
            core = robot_radius_m + unknown_depth_m
            band = (~lethal) & (dist <= core + inflation_m)
            if band.any():
                d = (dist[band] - core) / max(inflation_m, 1e-6)
                decay = (INSCRIBED * np.exp(-3.0 * d)).astype(np.uint8)
                cost[band] = np.maximum(cost[band], decay)

    if phantom and phantom_relief > 0:
        # Reduce, never erase. A phantom that is actually real must still
        # look expensive enough for the planner to prefer going round.
        for o in phantom:
            if o.cells is None or len(o.cells) == 0:
                continue
            r, c = o.cells[:, 0], o.cells[:, 1]
            scaled = (cost[r, c].astype(np.float32) * (1.0 - phantom_relief))
            cost[r, c] = np.clip(scaled, 0, INSCRIBED).astype(np.uint8)

    return cost


def inflate_lethal(cost: np.ndarray, grid: GridSpec,
                   extra_m: float) -> np.ndarray:
    """Grow the lethal region of a costmap by `extra_m`.

    PLAN CONSERVATIVELY, CHECK PERMISSIVELY -- applied to the PRIOR, not
    just to newly discovered obstacles.

    The prior costmap marks a cell lethal when the robot body would
    intersect an obstacle, i.e. within one robot radius (0.35 m). A path
    may therefore legally pass 0.36 m from a wall. But the controller's
    laser brake stops dead inside half its braking distance (0.6 m), so
    the planner was routing the robot through gaps the controller refuses
    to drive, and pure pursuit cutting a corner then grazed the wall.

    Observed live: the robot came to rest against a mapped building it had
    known about since before it set off, wheels slipping, odometry error
    growing half a metre a second. The initial plan -- the one followed for
    the first 200 m -- was the only one built without a margin, because the
    margin was applied on discrepancy replans only.
    """
    out = np.array(cost, dtype=np.uint8, copy=True)
    if extra_m <= 0:
        return out

    lethal = out >= LETHAL
    if not lethal.any():
        return out

    from scipy.ndimage import distance_transform_edt

    dist = distance_transform_edt(~lethal) * grid.resolution
    out[dist <= extra_m] = LETHAL
    return out


def first_blocked_index(path: np.ndarray, grid: GridSpec, cost: np.ndarray,
                        from_index: int = 0,
                        corridor_m: float = 0.0) -> int | None:
    """Index of the first point on the remaining path that is blocked.

    `path_is_blocked` answers whether to react; this answers where, which
    is what a local detour needs in order to aim past the obstacle rather
    than abandoning the whole route.
    """
    if len(path) == 0:
        return None
    start = max(0, from_index)
    tail = path[start:]
    if len(tail) == 0:
        return None

    rows, cols = grid.world_to_cell(tail[:, 0], tail[:, 1])
    r = np.clip(rows, 0, grid.height - 1)
    c = np.clip(cols, 0, grid.width - 1)

    hit = cost[r, c] >= LETHAL
    if corridor_m > 0:
        k = max(1, int(round(corridor_m / grid.resolution)))
        for dr in range(-k, k + 1):
            for dc in range(-k, k + 1):
                rr = np.clip(r + dr, 0, grid.height - 1)
                cc = np.clip(c + dc, 0, grid.width - 1)
                hit |= cost[rr, cc] >= LETHAL

    idx = np.nonzero(hit)[0]
    return int(idx[0]) + start if len(idx) else None


def path_is_blocked(path: np.ndarray, grid: GridSpec, cost: np.ndarray,
                    from_index: int = 0, corridor_m: float = 0.0,
                    within_m: float | None = None) -> bool:
    """Does the remaining path pass through a lethal cell?

    `corridor_m` widens the test beyond the path centreline, so a route that
    merely grazes an obstacle also counts as blocked.

    `within_m` limits the check to the first stretch of the remaining path.
    That distinction is what separates "something blocks the route eventually"
    from "something blocks it NOW": the first can wait for better
    information, the second cannot. Without it, a robot approaching an
    obstacle re-plans on every cycle, because each new view of the obstacle
    grows its footprint and invalidates the path planned a metre earlier.
    """
    if len(path) == 0:
        return False
    tail = path[max(0, from_index):]
    if len(tail) == 0:
        return False

    if within_m is not None and len(tail) > 1:
        seg = np.hypot(*np.diff(tail, axis=0).T)
        along = np.concatenate([[0.0], np.cumsum(seg)])
        tail = tail[along <= within_m]
        if len(tail) == 0:
            return False

    rows, cols = grid.world_to_cell(tail[:, 0], tail[:, 1])
    ok = grid.in_bounds(rows, cols)
    if not np.any(ok):
        return False
    r = np.clip(rows[ok], 0, grid.height - 1)
    c = np.clip(cols[ok], 0, grid.width - 1)

    if corridor_m <= 0:
        return bool((cost[r, c] >= LETHAL).any())

    k = max(1, int(round(corridor_m / grid.resolution)))
    for dr in range(-k, k + 1):
        for dc in range(-k, k + 1):
            rr = np.clip(r + dr, 0, grid.height - 1)
            cc = np.clip(c + dc, 0, grid.width - 1)
            if (cost[rr, cc] >= LETHAL).any():
                return True
    return False
