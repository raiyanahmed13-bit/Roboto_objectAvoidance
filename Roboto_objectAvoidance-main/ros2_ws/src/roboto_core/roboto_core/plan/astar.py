"""A* over the GIS costmap, with direction-dependent terrain cost.

THE DIRECTED-GRAPH POINT
------------------------
Terrain cost for a ground robot is not a property of a cell, it is a
property of a *move*. Climbing a 15% grade is expensive, descending the
same grade is nearly free, and crossing it sideways is a rollover risk.
So the edge cost depends on heading:

    s = arctan(dzdx*cos(phi) + dzdy*sin(phi))      signed along-path slope

which makes the search graph genuinely directed -- the cost from A to B
differs from B to A. A scalar slope field cannot express that, and neither
can Nav2's NavFn or Smac without a custom cost function, which is one of
the reasons this planner is hand-rolled.

Setting w_uphill = w_downhill = w_cross = 0 recovers a plain distance/
obstacle planner, which is the no-terrain ablation baseline for free.

WHY NOT A NAV2 PLUGIN
---------------------
A nav2_costmap_2d::Layer means pluginlib XML, lifecycle nodes, CMake
linkage and a colcon rebuild on every debug cycle: 1.5-2 days spent on
build systems rather than robotics, for a capstone where implementing the
planner earns more credit than configuring someone else's.
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass

import numpy as np

from ..frames import GridSpec

# ROS costmap_2d convention, matching build_prior.py.
LETHAL = 254
INSCRIBED = 253


@dataclass
class TerrainWeights:
    """Slope penalties, in cost units per radian of grade."""

    uphill: float = 6.0
    downhill: float = 1.5        # descending is cheap but not free
    cross: float = 3.0           # sideways on a slope: rollover risk
    max_slope_deg: float = 25.0  # hard block above this

    @classmethod
    def from_site(cls, site) -> "TerrainWeights":
        c = site_config(site)["slope"]
        return cls(uphill=float(c["w_uphill"]) * 6.0,
                   downhill=float(c["w_downhill"]) * 6.0,
                   cross=float(c["w_crossslope"]) * 6.0,
                   max_slope_deg=float(c["max_traversable_deg"]))

    @property
    def disabled(self) -> bool:
        return self.uphill == 0 and self.downhill == 0 and self.cross == 0


def site_config(site) -> dict:
    """The site's raw config, from either a SiteSpec or a plain dict.

    The GIS tools carry a SiteSpec (which needs pyproj); the ROS node has
    only the parsed YAML, because pyproj is not a robot dependency. Both
    have to be able to build the same weights and objectives, or the live
    robot and the offline experiments end up planning differently -- which
    has already happened once here, with terrain.
    """
    return site.raw if hasattr(site, "raw") else site


@dataclass(frozen=True)
class Objective:
    """What "best" means for a route.

    The planner is the same search either way; only the edge cost changes.
    Separating them makes the point an intro robotics course actually cares
    about: THE SHORTEST PATH IS NOT THE BEST PATH, and which route you get
    depends entirely on what you told the planner to minimise.

    `time_based` switches the units of the edge from metres to seconds, by
    dividing distance by a speed that falls off as the grade steepens.
    Everything else is an additive penalty in the same units as the base
    term. Costs are only ever compared WITHIN one objective, so the mixed
    units across objectives are harmless.
    """

    name: str
    # Weight on the graded prior costmap (0-253). Higher keeps more
    # clearance from obstacles and off-road ground.
    cost_scale: float = 0.02
    # Directional slope penalties. Zero recovers a terrain-blind planner.
    weights: TerrainWeights = None            # type: ignore[assignment]
    # Minimise travel TIME rather than a distance-like cost.
    time_based: bool = False
    max_speed_mps: float = 1.0
    # Speed falls as 1 / (1 + k * tan(uphill grade)). A simple, stated
    # model -- not a measured vehicle curve -- so it belongs in the report
    # as an assumption, not as a result.
    slope_speed_falloff: float = 3.0

    def __post_init__(self):
        if self.weights is None:
            object.__setattr__(self, "weights", TerrainWeights())

    @property
    def needs_terrain(self) -> bool:
        """Time depends on the grade even when no slope PENALTY is applied,
        so the gradients must still be sampled."""
        return self.time_based or not self.weights.disabled


def objectives(site) -> dict[str, Objective]:
    """The comparison set, built from the site's own slope configuration.

    Each of distance / time / effort / safe minimises ONE quantity and
    nothing else. That purity is deliberate: if every objective also
    carried a clearance term, the rows would all converge and the table
    would say nothing. `balanced` is the blend the rest of the project
    actually plans with, and it sits in the middle of the others by
    construction.
    """
    cfg = site_config(site)
    w = TerrainWeights.from_site(site)
    flat = TerrainWeights(0.0, 0.0, 0.0, w.max_slope_deg)
    return {
        # Pure geometry: the classic shortest path. Still refuses lethal
        # cells -- "shortest" has never meant "through a building".
        "distance": Objective("distance", cost_scale=0.0, weights=flat),
        # The project default: distance, clearance and directional slope.
        "balanced": Objective("balanced", cost_scale=0.02, weights=w),
        # Pure duration. Slope enters only through the speed model, so this
        # will accept a longer route if it is a flatter one.
        "time": Objective("time", cost_scale=0.0, weights=flat,
                          time_based=True,
                          max_speed_mps=float(cfg["robot"]["max_speed_mps"])),
        # Pure climbing. Uphill punished hard, descending nearly free, so
        # it trades length for elevation gain.
        "effort": Objective("effort", cost_scale=0.0,
                            weights=TerrainWeights(uphill=6 * w.uphill,
                                                   downhill=0.1 * w.downhill,
                                                   cross=0.0,
                                                   max_slope_deg=w.max_slope_deg)),
        # Maximise clearance: the graded costmap dominates, so the route
        # hugs the middle of open space rather than clipping corners.
        "safe": Objective("safe", cost_scale=0.40, weights=flat),
    }


@dataclass
class PlanResult:
    path: np.ndarray             # (N, 2) world metres, empty if none found
    cost: float
    expanded: int                # nodes popped, for reporting search effort
    reason: str = "ok"

    @property
    def found(self) -> bool:
        return len(self.path) > 0

    @property
    def length_m(self) -> float:
        if len(self.path) < 2:
            return 0.0
        return float(np.hypot(*np.diff(self.path, axis=0).T).sum())


class AStarPlanner:
    """8-connected A* on a downsampled costmap.

    Planning happens on a coarser grid than the 0.10 m map: a 3000 x 3000
    search is needlessly slow, and a path is only ever followed to within
    the controller's tolerance anyway.
    """

    # 8-connected moves as (drow, dcol).
    MOVES = [(-1, 0), (1, 0), (0, -1), (0, 1),
             (-1, -1), (-1, 1), (1, -1), (1, 1)]

    def __init__(self, costmap: np.ndarray, grid: GridSpec, downsample: int = 2,
                 terrain=None, weights: TerrainWeights | None = None,
                 cost_scale: float = 0.02,
                 objective: Objective | None = None,
                 heuristic_weight: float = 1.0):
        # An objective, if given, supersedes the loose weights/cost_scale
        # arguments. Those are kept so existing callers behave identically.
        self.objective = objective
        self.weights = (objective.weights if objective is not None
                        else (weights or TerrainWeights()))
        self.cost_scale = (objective.cost_scale if objective is not None
                           else float(cost_scale))

        # 0 turns A* into DIJKSTRA (uniform-cost search): no goal
        # information, so it expands outward in all directions.
        # 1 is ordinary A*, and the heuristic below is admissible, so the
        # result is optimal.
        # >1 is WEIGHTED A*: it over-trusts the heuristic, expands far
        # fewer nodes, and may return a suboptimal path. That trade is the
        # thing worth measuring.
        self.heuristic_weight = float(heuristic_weight)

        k = max(1, int(downsample))
        self.cost, self.grid = _downsample(costmap, grid, k)

        # Sample terrain gradients onto the SEARCH grid once, up front:
        # interpolating per edge expansion would dominate the runtime.
        self.dzdx = self.dzdy = None
        need_terrain = (objective.needs_terrain if objective is not None
                        else not self.weights.disabled)
        if terrain is not None and need_terrain:
            self.dzdx, self.dzdy, _ = terrain.sample(self.grid)

        self.blocked = self.cost >= LETHAL

    # ---- planning --------------------------------------------------------

    def plan(self, start_xy, goal_xy, max_expansions: int = 4_000_000) -> PlanResult:
        g = self.grid
        sr, sc = (int(v) for v in g.world_to_cell(start_xy[0], start_xy[1]))
        gr, gc = (int(v) for v in g.world_to_cell(goal_xy[0], goal_xy[1]))

        for (r, c), name in (((sr, sc), "start"), ((gr, gc), "goal")):
            if not (0 <= r < g.height and 0 <= c < g.width):
                return PlanResult(np.zeros((0, 2)), np.inf, 0, f"{name} out of bounds")

        # A robot can find ITSELF inside a lethal cell -- it just discovered
        # an obstacle it is already close to, and the new inflation now
        # covers its position. Refusing to plan is the worst possible
        # response: it is precisely the moment a route out is needed.
        # Measured, this turned a recoverable situation into a collision.
        # So escape to the nearest free cell and plan from there.
        if self.blocked[sr, sc]:
            esc = self._nearest_free(sr, sc)
            if esc is None:
                return PlanResult(np.zeros((0, 2)), np.inf, 0,
                                  "start in lethal cell, no escape")
            sr, sc = esc
        if self.blocked[gr, gc]:
            return PlanResult(np.zeros((0, 2)), np.inf, 0, "goal in lethal cell")

        h, w = g.height, g.width
        start, goal = sr * w + sc, gr * w + gc
        res = g.resolution

        gscore = np.full(h * w, np.inf, dtype=np.float64)
        gscore[start] = 0.0
        came = np.full(h * w, -1, dtype=np.int64)
        closed = np.zeros(h * w, dtype=bool)

        heap = [(self._heuristic(sr, sc, gr, gc), start)]
        expanded = 0

        while heap:
            _, cur = heapq.heappop(heap)
            if closed[cur]:
                continue
            if cur == goal:
                return self._reconstruct(came, cur, gscore[cur], expanded)
            closed[cur] = True
            expanded += 1
            if expanded > max_expansions:
                return PlanResult(np.zeros((0, 2)), np.inf, expanded, "expansion limit")

            r, c = divmod(cur, w)
            base = gscore[cur]

            for dr, dc in self.MOVES:
                nr, nc = r + dr, c + dc
                if not (0 <= nr < h and 0 <= nc < w):
                    continue
                nxt = nr * w + nc
                if closed[nxt] or self.blocked[nr, nc]:
                    continue

                step = res * (1.41421356 if dr and dc else 1.0)
                edge = self._edge_cost(r, c, dr, dc, step)
                if not np.isfinite(edge):
                    continue

                tentative = base + edge
                if tentative < gscore[nxt]:
                    gscore[nxt] = tentative
                    came[nxt] = cur
                    heapq.heappush(
                        heap, (tentative + self._heuristic(nr, nc, gr, gc), nxt))

        return PlanResult(np.zeros((0, 2)), np.inf, expanded, "no path")


    def _nearest_free(self, r: int, c: int, max_radius: int = 60):
        """Nearest non-lethal cell to (r, c), searched in rings.

        Bounded so a fully enclosed robot fails fast rather than scanning
        the whole grid.
        """
        g = self.grid
        for rad in range(1, max_radius + 1):
            r0, r1 = max(0, r - rad), min(g.height, r + rad + 1)
            c0, c1 = max(0, c - rad), min(g.width, c + rad + 1)
            sub = self.blocked[r0:r1, c0:c1]
            free = np.argwhere(~sub)
            if len(free) == 0:
                continue
            rr = free[:, 0] + r0
            cc = free[:, 1] + c0
            k = int(np.argmin((rr - r) ** 2 + (cc - c) ** 2))
            return int(rr[k]), int(cc[k])
        return None

    # ---- costs -----------------------------------------------------------

    def _edge_cost(self, r, c, dr, dc, step) -> float:
        """Cost of moving from (r, c) by (dr, dc). inf blocks the move."""
        nr, nc = r + dr, c + dc

        if self.dzdx is None:
            return step + self.cost_scale * step * float(self.cost[nr, nc])

        # Heading of this move, then the signed slope along it.
        norm = np.hypot(dc, dr)
        ch, sh = dc / norm, dr / norm
        gx = 0.5 * (self.dzdx[r, c] + self.dzdx[nr, nc])
        gy = 0.5 * (self.dzdy[r, c] + self.dzdy[nr, nc])

        along = np.arctan(gx * ch + gy * sh)
        cross = np.arctan(-gx * sh + gy * ch)

        # The base term is metres, or SECONDS when minimising time. Speed
        # falls off climbing and is capped at the flat-ground maximum, so
        # descending never makes the robot faster than it can drive.
        obj = self.objective
        if obj is not None and obj.time_based:
            if abs(along) > np.radians(self.weights.max_slope_deg):
                return np.inf
            v = obj.max_speed_mps / (
                1.0 + obj.slope_speed_falloff * max(np.tan(along), 0.0))
            base = step / max(v, 1e-6)
        else:
            base = step

        w = self.weights
        if abs(along) > np.radians(w.max_slope_deg):
            return np.inf

        return (base
                + self.cost_scale * base * float(self.cost[nr, nc])
                + base * (w.uphill * max(along, 0.0)
                          + w.downhill * max(-along, 0.0)
                          + w.cross * abs(cross)))

    def _heuristic(self, r, c, gr, gc) -> float:
        """Lower bound on the remaining cost to the goal.

        Admissibility is what makes A* optimal, and it depends on the
        objective:

          * distance-like costs -- every edge costs at least its geometric
            length, since all other terms are non-negative additions, so
            straight-line distance is a valid bound.
          * TIME -- the bound must be distance / max_speed. The robot can
            never beat its flat-ground top speed, so this cannot
            overestimate. Using plain metres here would be inadmissible by
            a factor of max_speed and would quietly return suboptimal
            routes.

        `heuristic_weight` scales it: 0 gives Dijkstra (admissible but
        uninformed), 1 gives A*, and anything above 1 is deliberately
        INADMISSIBLE -- weighted A*, which trades optimality for a much
        smaller search.
        """
        h = self.grid.resolution * float(np.hypot(gr - r, gc - c))
        obj = self.objective
        if obj is not None and obj.time_based:
            h /= max(obj.max_speed_mps, 1e-6)
        return self.heuristic_weight * h

    def _reconstruct(self, came, node, cost, expanded) -> PlanResult:
        w = self.grid.width
        cells = []
        while node != -1:
            cells.append(node)
            node = came[node]
        cells.reverse()

        rows = np.array([n // w for n in cells])
        cols = np.array([n % w for n in cells])
        x, y = self.grid.cell_to_world(rows, cols)
        return PlanResult(np.column_stack([x, y]), float(cost), expanded)


def _downsample(costmap: np.ndarray, grid: GridSpec, k: int):
    """Block-reduce by MAX, so obstacles never vanish between cells.

    Mean or nearest-neighbour downsampling can erase a one-cell-wide wall
    and produce a path straight through a building. Max is the only safe
    reduction for an obstacle map.
    """
    if k == 1:
        return np.asarray(costmap), grid

    h = (grid.height // k) * k
    w = (grid.width // k) * k
    trimmed = np.asarray(costmap)[:h, :w]
    reduced = trimmed.reshape(h // k, k, w // k, k).max(axis=(1, 3))

    return reduced, GridSpec(
        origin_x=grid.origin_x,
        origin_y=grid.origin_y,
        resolution=grid.resolution * k,
        width=w // k,
        height=h // k,
    )
