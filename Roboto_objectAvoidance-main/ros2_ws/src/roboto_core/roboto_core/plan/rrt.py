"""RRT* -- a sampling planner, for contrast with the grid search.

A* and Dijkstra explore a GRAPH: every cell of a discretised map, in an
order the heuristic chooses. RRT* explores a TREE built from random
samples of the continuous space. The difference shows up in the numbers
rather than in the idea:

  * the grid search is resolution-complete -- if a route exists on the
    grid it finds it, and with an admissible heuristic it is optimal;
  * RRT* is only *asymptotically* optimal. Given enough samples it
    converges on the optimum, and given few it returns something valid but
    visibly worse. It stops when told to, which a grid search cannot do.

Its real advantage is that cost does not scale with map size -- it samples
the free space rather than enumerating cells, so a 3000 x 3000 grid costs
it nothing extra.

WHAT IT CAN AND CANNOT OPTIMISE
-------------------------------
Clearance it can do. An edge cost that integrates the graded costmap along
the segment is symmetric -- the same value going either way -- so the
`distance` and `safe` objectives work here exactly as in the grid search,
and are supported.

Directional terrain is the awkward one, and the reason is narrower than it
first looks. The tree is directed root-to-leaf and every edge is evaluated
in its direction of travel, so an asymmetric cost does not break the
mechanics of rewiring. What it breaks is the ARGUMENT: RRT*'s asymptotic
optimality assumes the cost is a metric, and "climbing costs more than
descending" is not symmetric, so that guarantee no longer carries. The
planner would still return valid routes; it just would not be converging
on anything provable.

Rather than quietly weaken the claim, an objective whose cost depends on
the signed grade is refused outright, with a message saying why.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..frames import GridSpec
from .astar import LETHAL, Objective, PlanResult


@dataclass
class RRTConfig:
    max_samples: int = 6000
    step_m: float = 6.0            # how far the tree extends per sample
    goal_bias: float = 0.06        # fraction of samples aimed at the goal
    goal_tol_m: float = 3.0
    # Radius for choosing a better parent and rewiring. The RRT* ball
    # shrinks as the tree grows; this is the ceiling.
    rewire_radius_m: float = 12.0
    # Collision checking is sampled along each edge at this spacing. Too
    # coarse and the tree steps straight through a wall.
    collision_step_m: float = 0.25
    seed: int = 0


class RRTStarPlanner:
    """RRT* over the same costmap the grid planners use.

    Takes the same `Objective` as the grid planners, but only those whose
    cost is symmetric -- see the module docstring.
    """

    def __init__(self, costmap: np.ndarray, grid: GridSpec,
                 config: RRTConfig | None = None,
                 objective: Objective | None = None):
        self.cfg = config or RRTConfig()
        self.grid = grid
        self.cost = np.asarray(costmap)
        self.blocked = self.cost >= LETHAL

        if objective is not None and (objective.time_based
                                      or not objective.weights.disabled):
            raise ValueError(
                f"objective {objective.name!r} depends on the signed grade, "
                f"which is asymmetric. RRT* would still return valid routes, "
                f"but its asymptotic-optimality argument assumes a metric "
                f"cost and no longer holds -- so it is refused rather than "
                f"quietly weakened. Use an objective without slope "
                f"weighting (distance, safe).")
        self.objective = objective

    # ---- cost ------------------------------------------------------------

    def _edge_cost(self, a, b) -> float:
        """Length, plus the graded costmap integrated along the segment.

        Symmetric by construction: traversing the same segment the other
        way visits the same cells.
        """
        d = float(np.hypot(b[0] - a[0], b[1] - a[1]))
        scale = 0.0 if self.objective is None else self.objective.cost_scale
        if scale == 0.0 or d < 1e-9:
            return d

        n = max(2, int(d / self.cfg.collision_step_m) + 1)
        t = np.linspace(0.0, 1.0, n)
        r, c = self.grid.world_to_cell(a[0] + (b[0] - a[0]) * t,
                                       a[1] + (b[1] - a[1]) * t)
        r = np.clip(r, 0, self.grid.height - 1)
        c = np.clip(c, 0, self.grid.width - 1)
        return d * (1.0 + scale * float(self.cost[r, c].mean()))

    # ---- geometry --------------------------------------------------------

    def _free(self, p) -> bool:
        r, c = self.grid.world_to_cell(p[0], p[1])
        if not self.grid.in_bounds(r, c):
            return False
        return not self.blocked[int(r), int(c)]

    def _segment_free(self, a, b) -> bool:
        d = float(np.hypot(b[0] - a[0], b[1] - a[1]))
        n = max(2, int(d / self.cfg.collision_step_m) + 1)
        t = np.linspace(0.0, 1.0, n)
        xs = a[0] + (b[0] - a[0]) * t
        ys = a[1] + (b[1] - a[1]) * t
        r, c = self.grid.world_to_cell(xs, ys)
        ok = self.grid.in_bounds(r, c)
        if not np.all(ok):
            return False
        return not self.blocked[r, c].any()

    # ---- planning --------------------------------------------------------

    def plan(self, start_xy, goal_xy) -> PlanResult:
        cfg = self.cfg
        rng = np.random.default_rng(cfg.seed)
        start = np.asarray(start_xy, dtype=np.float64)
        goal = np.asarray(goal_xy, dtype=np.float64)

        if not self._free(start):
            return PlanResult(np.zeros((0, 2)), np.inf, 0, "start in lethal cell")
        if not self._free(goal):
            return PlanResult(np.zeros((0, 2)), np.inf, 0, "goal in lethal cell")

        xmin, ymin, xmax, ymax = self.grid.bounds
        pts = [start]
        parent = [-1]
        cost = [0.0]
        best_goal = -1
        best_cost = np.inf
        samples = 0

        for samples in range(1, cfg.max_samples + 1):
            if rng.random() < cfg.goal_bias:
                target = goal
            else:
                target = np.array([rng.uniform(xmin, xmax),
                                   rng.uniform(ymin, ymax)])

            arr = np.asarray(pts)
            near_i = int(np.argmin(((arr - target) ** 2).sum(1)))
            near = arr[near_i]

            d = float(np.hypot(*(target - near)))
            if d < 1e-9:
                continue
            new = near + (target - near) * min(cfg.step_m, d) / d
            if not self._free(new) or not self._segment_free(near, new):
                continue

            # --- choose the cheapest reachable parent (the RRT* part) -----
            radius = min(cfg.rewire_radius_m,
                         cfg.step_m * (np.log(len(pts) + 1) / len(pts)) ** 0.5 * 30)
            dists = np.hypot(*(arr - new).T)
            cand = np.nonzero(dists <= max(radius, cfg.step_m))[0]

            best_parent = near_i
            best_new_cost = cost[near_i] + self._edge_cost(near, new)
            for k in cand:
                trial = cost[k] + self._edge_cost(arr[k], new)
                if trial < best_new_cost and self._segment_free(arr[k], new):
                    best_parent, best_new_cost = int(k), trial

            pts.append(new)
            parent.append(best_parent)
            cost.append(best_new_cost)
            added = len(pts) - 1

            # --- rewire neighbours through the new node ------------------
            for k in cand:
                if k == best_parent:
                    continue
                trial = best_new_cost + self._edge_cost(new, arr[k])
                if trial < cost[k] and self._segment_free(new, arr[k]):
                    parent[k] = added
                    cost[k] = trial

            # --- goal check ----------------------------------------------
            if float(np.hypot(*(new - goal))) <= cfg.goal_tol_m:
                total = best_new_cost + self._edge_cost(new, goal)
                if total < best_cost:
                    best_cost, best_goal = total, added

        if best_goal < 0:
            return PlanResult(np.zeros((0, 2)), np.inf, samples,
                              f"no path in {samples} samples")

        chain = []
        node = best_goal
        while node != -1:
            chain.append(pts[node])
            node = parent[node]
        chain.reverse()
        chain.append(goal)
        return PlanResult(np.asarray(chain), float(best_cost), samples)
