"""Route around the obstacle, not around the route.

When something blocks the path there are two honest reactions:

    GLOBAL REPLAN   throw the route away and search from here to the goal
                    again. Optimal under the current costmap, and free to
                    pick a completely different road.

    LOCAL DETOUR    search only as far as a rejoin point further along the
                    ORIGINAL route, then continue on it. Much cheaper, and
                    the route stays recognisable.

The global replan is what this project did first, and it works, but it
looks wrong: discovering a barrier 25 % of the way along would re-route
the remaining 75 %, so the robot appears to abandon its plan rather than
step around an obstacle. It is also expensive -- a full search from every
blockage, tens of thousands of expansions each time.

The trade is real in both directions and worth measuring rather than
asserting. A detour cannot find a better road that happens to exist, and
it can fail where a global replan would succeed: if the obstacle blocks
the only way through, no rejoin point is reachable. So a detour that
fails must fall back rather than give up, and the caller is expected to do
exactly that.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..frames import GridSpec
from .cost_fusion import LETHAL, first_blocked_index


@dataclass
class DetourResult:
    """A spliced route: new segment, then the tail of the original."""

    path: np.ndarray
    rejoin_index: int = -1      # index into the ORIGINAL path
    expanded: int = 0           # nodes the search popped
    detour_m: float = 0.0       # length of the NEW segment only
    tried: int = 0              # rejoin points attempted
    reason: str = "ok"

    @property
    def found(self) -> bool:
        return len(self.path) > 0


def _arc_length(path: np.ndarray) -> np.ndarray:
    seg = np.hypot(*np.diff(path, axis=0).T)
    return np.concatenate([[0.0], np.cumsum(seg)])


def plan_detour(plan_fn, path: np.ndarray, pose, grid: GridSpec,
                cost: np.ndarray, from_index: int = 0,
                corridor_m: float = 0.35,
                clear_m: float = 8.0,
                max_rejoin_m: float = 80.0,
                step_m: float = 6.0) -> DetourResult:
    """Plan around a blockage and rejoin the original route.

    `plan_fn(start_xy, goal_xy)` returns a PlanResult, so the caller keeps
    control of the costmap, objective and search settings.

    Rejoin points are tried nearest-first, starting `clear_m` past the
    blockage. Nearest-first keeps the detour tight, which is the whole
    point; starting a clear distance past the obstacle stops the robot
    aiming at a point still inside it.
    """
    path = np.asarray(path, dtype=np.float64)
    if len(path) < 2:
        return DetourResult(np.zeros((0, 2)), reason="no path")

    blocked = first_blocked_index(path, grid, cost, from_index, corridor_m)
    if blocked is None:
        return DetourResult(np.zeros((0, 2)), reason="not blocked")

    along = _arc_length(path)
    start_s = along[blocked] + clear_m
    tried = 0

    for target_s in np.arange(start_s, start_s + max_rejoin_m, step_m):
        if target_s > along[-1]:
            # Past the end of the route: the only rejoin left is the goal
            # itself, which is a global replan by another name.
            break
        j = int(np.searchsorted(along, target_s))
        j = min(j, len(path) - 1)

        r, c = grid.world_to_cell(path[j, 0], path[j, 1])
        if not grid.in_bounds(r, c):
            continue
        if cost[int(r), int(c)] >= LETHAL:
            continue                      # rejoin point is itself blocked

        res = plan_fn(pose[:2], tuple(path[j]))
        tried += 1
        if not res.found or len(res.path) < 2:
            continue

        # The new segment must not run back through the obstacle -- checked
        # with NO corridor, because it came from the planner on this very
        # costmap and is lethal-free by construction. Re-testing it with a
        # body-width corridor the planner was never given rejects every
        # valid detour: a route that legitimately passes close to the
        # barrier edge reads as blocked. Clearance is the caller's job, by
        # passing an inflated costmap -- the same "plan conservatively,
        # check permissively" split used everywhere else here.
        if first_blocked_index(res.path, grid, cost, 0, 0.0) is not None:
            continue
        # The stretch just after the rejoin does use the corridor, because
        # that is the original route and the robot's body has to fit:
        # rejoining straight back into what it just avoided is no detour.
        tail = path[j:]
        if first_blocked_index(tail, grid, cost, 0, corridor_m) == 0:
            continue

        spliced = np.vstack([res.path, tail[1:]]) if len(tail) > 1 else res.path
        return DetourResult(path=spliced, rejoin_index=j,
                            expanded=res.expanded, detour_m=res.length_m,
                            tried=tried)

    return DetourResult(np.zeros((0, 2)), tried=tried,
                        reason=f"no rejoin within {max_rejoin_m:.0f} m")
