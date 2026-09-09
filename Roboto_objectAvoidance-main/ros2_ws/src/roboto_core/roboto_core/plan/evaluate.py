"""Measure a route the same way regardless of what it was planned for.

Each objective minimises a different thing, so their costs are not
comparable -- a "time" cost is in seconds and a "distance" cost is in
metres. To compare routes you have to score them all on the SAME set of
physical quantities, which is what this does.

That is the whole point of the comparison: the shortest route is not the
fastest, the fastest is not the flattest, and the flattest is not the one
with the most clearance. Only a shared scorecard shows the trade.
"""

from __future__ import annotations

import numpy as np

from ..frames import GridSpec


def route_metrics(path: np.ndarray, grid: GridSpec, cost: np.ndarray,
                  terrain=None, max_speed_mps: float = 1.0,
                  slope_speed_falloff: float = 3.0) -> dict:
    """Physical properties of a route.

    `terrain` is optional; without it the elevation columns are omitted
    rather than guessed at.
    """
    path = np.asarray(path, dtype=np.float64)
    out: dict[str, float] = {}
    if len(path) < 2:
        return {"length_m": 0.0}

    seg = np.hypot(*np.diff(path, axis=0).T)
    out["length_m"] = float(seg.sum())

    # Clearance, read off the graded prior costmap. Higher cost means
    # closer to something, so the WORST cell on the route is the number
    # that matters for safety.
    rows, cols = grid.world_to_cell(path[:, 0], path[:, 1])
    ok = grid.in_bounds(rows, cols)
    if np.any(ok):
        vals = cost[np.clip(rows[ok], 0, grid.height - 1),
                    np.clip(cols[ok], 0, grid.width - 1)].astype(float)
        out["mean_cost"] = float(vals.mean())
        out["max_cost"] = float(vals.max())

    if terrain is None:
        return out

    # Elevation along the route, then the signed grade of each segment.
    dzdx, dzdy, z = terrain.sample(grid)
    r = np.clip(np.asarray(rows), 0, grid.height - 1)
    c = np.clip(np.asarray(cols), 0, grid.width - 1)
    elev = z[r, c].astype(np.float64)

    dz = np.diff(elev)
    out["climb_m"] = float(dz[dz > 0].sum())
    out["descent_m"] = float(-dz[dz < 0].sum())

    with np.errstate(divide="ignore", invalid="ignore"):
        grade = np.where(seg > 1e-9, dz / seg, 0.0)
    along = np.arctan(grade)
    out["max_uphill_deg"] = float(np.degrees(along.max())) if len(along) else 0.0
    out["mean_abs_grade_deg"] = float(np.degrees(np.abs(along).mean()))

    # Duration under the same speed model the "time" objective plans with,
    # so a route planned for time can be checked against one planned for
    # distance on equal terms.
    v = max_speed_mps / (1.0 + slope_speed_falloff * np.maximum(np.tan(along), 0.0))
    out["time_s"] = float((seg / np.maximum(v, 1e-6)).sum())
    return out
