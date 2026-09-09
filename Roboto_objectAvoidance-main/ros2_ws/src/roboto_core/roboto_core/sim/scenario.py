"""Randomised placement of obstacles the GIS prior does not contain.

WHY THIS MODULE EXISTS
----------------------
The experiments used to hardcode obstacles at fixed fractions along the
route:

    BLOCKERS = [("construction_barrier", 0.30, 3.0, 0.6), ...]

so `--seeds 3` varied lidar noise and odometry but placed the *same two
obstacles in the same two places* every time. "3 of 3 missions succeeded"
was therefore one scenario measured three times, not three independent
trials, and no amount of seeding fixed that. Randomising the scenario --
count, position, lateral offset, orientation and size -- is what turns a
seed into a genuinely independent sample and lets the headline numbers
carry a variance that means something.

Scenario randomness is drawn from its own stream, independent of the
sensor-noise stream. That separation is deliberate: holding the scenario
fixed while varying noise isolates sensor effects, and holding noise fixed
while varying the scenario isolates geometry. Collapsing both onto one seed
would make those two ablations impossible to run.

THE SAME PLACEMENTS DRIVE THE LIVE DEMO
---------------------------------------
`tools/gis_pipeline/inject_gazebo_obstacles.py` resolves placements through
this module too, so a given scenario seed produces the same obstacles
offline and in Gazebo. Previously the two carried separate hardcoded lists
that merely happened to agree.

GEOMETRY CONVENTION
-------------------
An obstacle is a rectangle with its own long axis. `yaw_offset` is measured
from the route heading at the placement point, so 0 lays the object ALONG
the route (a van parked at the kerb) and pi/2 lays it ACROSS (a barrier
spanning the road).

Rectangles are rasterised properly rather than by their axis-aligned
bounding box. The old code filled the AABB, which for an 8 m x 0.8 m
barrier at 45 degrees marked a 6 m x 6 m square -- inflating ground-truth
area several times over, by a factor that varied with route heading. With
placement now randomised that would be a confound sitting directly
underneath the detection metrics.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..frames import GridSpec

# Scenario draws come from this substream so they cannot be perturbed by a
# change to how many random numbers the sensor models happen to consume.
SCENARIO_STREAM = 7919


@dataclass(frozen=True)
class ObstacleType:
    """A class of real object, with the size range it plausibly spans."""

    label: str
    length_m: tuple[float, float]     # along its own long axis
    width_m: tuple[float, float]


# Ordinary street furniture and vehicles, not adversarial shapes. The point
# of the experiment is that OSM lacks things a real street has.
CATALOGUE: tuple[ObstacleType, ...] = (
    ObstacleType("parked_van", (4.2, 5.6), (1.8, 2.4)),
    ObstacleType("delivery_truck", (6.0, 9.0), (2.3, 2.6)),
    ObstacleType("construction_barrier", (3.0, 7.0), (0.6, 1.2)),
    ObstacleType("fence_panel_run", (3.5, 8.0), (0.4, 0.8)),
    ObstacleType("dumpster", (2.0, 3.2), (1.4, 2.0)),
    ObstacleType("skip_container", (3.2, 4.4), (1.8, 2.4)),
    ObstacleType("fallen_tree", (5.0, 11.0), (0.5, 1.0)),
)


def catalogue_from(names) -> tuple[ObstacleType, ...]:
    """Subset of CATALOGUE by label, for restricting a scenario.

    A 2D lidar's evidence for an object is its VISIBLE WIDTH times one cell
    of depth, not its footprint, so the smallest types sit nearest the
    detection floor. Worst-case near face against a 0.2 m2 floor, at each
    type's minimum size and worst yaw:

        delivery_truck 2.82x   fallen_tree 2.35x   parked_van   1.97x
        fence_panel_run 1.64x  skip_container 1.50x
        construction_barrier 1.41x   dumpster 1.41x

    All of them clear it. Restricting is for demos, where one missed
    detection wastes a twenty-minute take. Experiments should keep the full
    catalogue: a narrower one is an easier scenario, and a result that
    depends on that should say so out loud rather than quietly.

    `names` may be a comma-separated string or an iterable of labels.
    """
    if names is None:
        return CATALOGUE
    if isinstance(names, str):
        names = [n.strip() for n in names.split(",")]
    want = {n for n in names if n}
    if not want:
        return CATALOGUE
    known = {t.label for t in CATALOGUE}
    unknown = want - known
    if unknown:
        raise ValueError(f"unknown obstacle type(s): {sorted(unknown)}. "
                         f"Choose from {sorted(known)}")
    return tuple(t for t in CATALOGUE if t.label in want)


@dataclass(frozen=True)
class Placement:
    """One obstacle resolved onto a concrete route: a rectangle in metres."""

    label: str
    cx: float
    cy: float
    yaw: float                # absolute, world frame
    half_len_m: float         # along the rectangle's own long axis
    half_wid_m: float
    frac: float = 0.0         # arc-length fraction of the route it sits at

    @property
    def size(self) -> tuple[float, float]:
        """(length, width) in metres -- what an SDF <box> wants."""
        return (2.0 * self.half_len_m, 2.0 * self.half_wid_m)

    @property
    def area_m2(self) -> float:
        return 4.0 * self.half_len_m * self.half_wid_m

    def corners(self) -> np.ndarray:
        """(4, 2) world-frame corners, counter-clockwise."""
        c, s = np.cos(self.yaw), np.sin(self.yaw)
        local = np.array([[-self.half_len_m, -self.half_wid_m],
                          [self.half_len_m, -self.half_wid_m],
                          [self.half_len_m, self.half_wid_m],
                          [-self.half_len_m, self.half_wid_m]])
        return np.column_stack([
            self.cx + local[:, 0] * c - local[:, 1] * s,
            self.cy + local[:, 0] * s + local[:, 1] * c,
        ])

    @property
    def aabb(self) -> tuple[float, float, float, float]:
        """(xmin, ymin, xmax, ymax) -- a loose bound on a rotated rectangle."""
        p = self.corners()
        return (float(p[:, 0].min()), float(p[:, 1].min()),
                float(p[:, 0].max()), float(p[:, 1].max()))

    def distance_to(self, pt) -> float:
        """Distance from a point to the rectangle; 0 inside.

        Against the ROTATED rectangle, not its bounding box. A detection
        sits on the near face of the object, so this is the number that
        says whether it was localised correctly -- and using the AABB here
        would quietly credit the detector for the slack in the bound.
        """
        c, s = np.cos(-self.yaw), np.sin(-self.yaw)
        dx, dy = float(pt[0]) - self.cx, float(pt[1]) - self.cy
        u = dx * c - dy * s
        v = dx * s + dy * c
        ou = max(abs(u) - self.half_len_m, 0.0)
        ov = max(abs(v) - self.half_wid_m, 0.0)
        return float(np.hypot(ou, ov))


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------

def scenario_rng(seed: int) -> np.random.Generator:
    """The scenario substream for `seed`, independent of sensor noise."""
    return np.random.default_rng([int(seed), SCENARIO_STREAM])


def _arc_length(traj: np.ndarray) -> np.ndarray:
    seg = np.hypot(*np.diff(np.asarray(traj)[:, :2], axis=0).T)
    return np.concatenate([[0.0], np.cumsum(seg)])


def _pose_at(traj: np.ndarray, along: np.ndarray, target: float):
    """(x, y, heading) at `target` metres along the route."""
    traj = np.asarray(traj)
    i = int(np.clip(np.searchsorted(along, target), 1, len(traj) - 1))
    x, y = traj[i, 0], traj[i, 1]
    if traj.shape[1] >= 3:
        th = float(traj[i, 2])
    else:                                   # bare waypoints: finite difference
        j = min(i + 1, len(traj) - 1)
        k = max(i - 1, 0)
        th = float(np.arctan2(traj[j, 1] - traj[k, 1], traj[j, 0] - traj[k, 0]))
    return float(x), float(y), th


def _overlaps_prior(prior_occ: np.ndarray, grid: GridSpec,
                    place: Placement, tol: float) -> bool:
    """Does the rectangle land on something the prior already contains?

    An obstacle inside a building is not a discrepancy, it is a broken
    scenario: the robot can never reach it, so it would score as a missed
    detection the detector never had a chance at.
    """
    cells = rect_cells(grid, place)
    if len(cells) == 0:
        return True                          # off the grid entirely
    return bool(prior_occ[cells[:, 0], cells[:, 1]].mean() > tol)


def _free_run_m(prior_occ: np.ndarray, grid: GridSpec,
                cx: float, cy: float, ux: float, uy: float,
                max_m: float) -> float:
    """Distance from (cx, cy) along (ux, uy) before the prior turns solid.

    Capped at `max_m`, and the step count is rounded UP: flooring it leaves
    the longest reportable run a cell short of `max_m`, so a caller asking
    `run >= max_m` could never be satisfied even in wide-open space.
    """
    n = max(1, int(np.ceil(max_m / grid.resolution)))
    t = np.arange(1, n + 1) * grid.resolution
    r, c = grid.world_to_cell(cx + ux * t, cy + uy * t)
    ok = grid.in_bounds(r, c)
    blocked = np.ones(n, dtype=bool)          # off-grid counts as blocked
    blocked[ok] = prior_occ[r[ok], c[ok]]
    first = int(np.argmax(blocked)) if blocked.any() else n
    return float(min(first * grid.resolution, max_m))


def _has_detour(prior_occ: np.ndarray, grid: GridSpec, place: Placement,
                span_m: float, need_m: float) -> bool:
    """Is there room to get around this obstacle on at least one side?

    A barrier dropped into an alley narrower than the robot is a scenario
    with no solution, and a mission that fails there measures the site, not
    the planner. Rejecting those keeps the ablation about the discrepancy
    layer. It is a cheap probe, not a connectivity proof -- a route can
    still be unreachable for reasons this does not see, which is a real
    failure and is deliberately left in.

    The probe runs along the obstacle's OWN long axis, because that is
    where its ends are and therefore where a way past has to be. Probing
    perpendicular to it would measure the corridor the obstacle already
    blocks.
    """
    ax, ay = np.cos(place.yaw), np.sin(place.yaw)
    reach = span_m + need_m
    left = _free_run_m(prior_occ, grid, place.cx, place.cy, ax, ay, reach)
    right = _free_run_m(prior_occ, grid, place.cx, place.cy, -ax, -ay, reach)
    return max(left, right) >= reach


def sample_placements(
    rng: np.random.Generator,
    traj: np.ndarray,
    prior_occ: np.ndarray,
    grid: GridSpec,
    *,
    mode: str = "blocking",
    n_range: tuple[int, int] = (2, 4),
    frac_range: tuple[float, float] = (0.15, 0.85),
    min_separation_m: float = 25.0,
    lateral_m: tuple[float, float] = (2.0, 4.5),
    yaw_jitter_rad: float = np.radians(20.0),
    min_span_m: float = 3.0,
    detour_clearance_m: float = 1.5,
    overlap_tol: float = 0.05,
    max_attempts: int = 200,
    catalogue: tuple[ObstacleType, ...] = CATALOGUE,
) -> list[Placement]:
    """Draw a scenario of obstacles along `traj`.

    `mode` selects the experimental geometry:

      "blocking"  the object lies ACROSS the route and spans it, so a robot
                  that ignores the discrepancy layer collides. This is the
                  mission ablation.
      "roadside"  the object lies ALONG the route, offset to one side, so
                  the robot passes it and must detect it without being
                  forced to. This is the detection experiment.

    Placements landing inside a prior building, sitting too close to
    another obstacle, or (in blocking mode) leaving no way past are
    rejected and redrawn. Returns however many survived, which may be fewer
    than requested -- the caller should report the shortfall rather than
    pretending the scenario was as dense as it asked for.
    """
    if mode not in ("blocking", "roadside"):
        raise ValueError(f"unknown mode {mode!r}")

    traj = np.asarray(traj, dtype=np.float64)
    if len(traj) < 2:
        return []
    along = _arc_length(traj)
    total = float(along[-1])
    if total <= 0:
        return []

    usable = catalogue
    if mode == "blocking":
        # Something that cannot span the corridor does not test the
        # ablation: the planner threads past it without ever replanning and
        # the mission "succeeds" for the wrong reason.
        usable = tuple(t for t in catalogue if t.length_m[1] >= min_span_m)
        if not usable:
            raise ValueError("no catalogue entry can span min_span_m")

    want = int(rng.integers(n_range[0], n_range[1] + 1))
    out: list[Placement] = []

    # Draw one obstacle per equal band of the route rather than sampling
    # fractions freely.
    #
    # Free sampling is greedy and corners itself: accept 0.30 and 0.65 on a
    # short route and no third fraction can satisfy the separation
    # constraint against both, so the scenario silently comes up short of
    # what was asked for. Banding also spreads obstacles along the route
    # instead of letting them clump, which means each mission exercises
    # detection at short, medium and long range rather than three times at
    # whatever range the cluster happened to land.
    lo, hi = frac_range
    band = (hi - lo) / want
    tries = max(1, max_attempts // want)

    for b in range(want):
        for _ in range(tries):
            t = usable[int(rng.integers(len(usable)))]
            length = float(rng.uniform(*t.length_m))
            width = float(rng.uniform(*t.width_m))
            frac = float(rng.uniform(lo + b * band, lo + (b + 1) * band))

            if any(abs(frac - p.frac) * total < min_separation_m for p in out):
                continue

            x, y, heading = _pose_at(traj, along, frac * total)

            if mode == "blocking":
                if length < min_span_m:
                    continue
                # Across the route, nudged off centre so the robot does not
                # always meet it head-on at the midpoint.
                yaw_off = np.pi / 2 + float(
                    rng.uniform(-yaw_jitter_rad, yaw_jitter_rad))
                lateral = float(rng.uniform(-0.8, 0.8))
            else:
                yaw_off = float(rng.uniform(-yaw_jitter_rad, yaw_jitter_rad))
                side = 1.0 if rng.random() < 0.5 else -1.0
                lateral = side * float(rng.uniform(*lateral_m))

            yaw = heading + yaw_off
            cx = x - lateral * np.sin(heading)
            cy = y + lateral * np.cos(heading)

            place = Placement(label=t.label, cx=cx, cy=cy, yaw=yaw,
                              half_len_m=length / 2.0, half_wid_m=width / 2.0,
                              frac=frac)

            if _overlaps_prior(prior_occ, grid, place, overlap_tol):
                continue
            if mode == "blocking" and not _has_detour(
                    prior_occ, grid, place, length / 2.0, detour_clearance_m):
                continue

            out.append(place)
            break

    out.sort(key=lambda p: p.frac)
    return out


# ---------------------------------------------------------------------------
# Rasterisation
# ---------------------------------------------------------------------------

def rect_cells(grid: GridSpec, place: Placement) -> np.ndarray:
    """(N, 2) grid cells whose CENTRES fall inside the rotated rectangle."""
    x0, y0, x1, y1 = place.aabb
    r0, c0 = grid.world_to_cell(x0, y0)
    r1, c1 = grid.world_to_cell(x1, y1)
    r0, c0 = int(max(int(r0), 0)), int(max(int(c0), 0))
    r1 = int(min(int(r1) + 1, grid.height))
    c1 = int(min(int(c1) + 1, grid.width))
    if r1 <= r0 or c1 <= c0:
        return np.zeros((0, 2), dtype=np.int64)

    rows = np.arange(r0, r1)
    cols = np.arange(c0, c1)
    X, Y = grid.cell_to_world(rows[:, None], cols[None, :])

    c, s = np.cos(-place.yaw), np.sin(-place.yaw)
    dx = X - place.cx
    dy = Y - place.cy
    u = dx * c - dy * s
    v = dx * s + dy * c
    inside = (np.abs(u) <= place.half_len_m) & (np.abs(v) <= place.half_wid_m)

    rr, cc = np.nonzero(inside)
    return np.column_stack([rr + r0, cc + c0]).astype(np.int64)


def rasterise(prior_occ: np.ndarray, grid: GridSpec,
              placements: list[Placement]) -> tuple[np.ndarray, list[dict]]:
    """Build the truth world: the prior plus these obstacles.

    Returns (truth_occ, ground_truth). Each ground-truth entry carries the
    `Placement` itself, so scoring can measure distance to the real
    rectangle instead of to a bounding box.
    """
    truth = np.array(prior_occ, dtype=bool, copy=True)
    gts: list[dict] = []

    for place in placements:
        cells = rect_cells(grid, place)
        if len(cells) == 0:
            continue
        truth[cells[:, 0], cells[:, 1]] = True
        x, y = grid.cell_to_world(cells[:, 0], cells[:, 1])
        gts.append({
            "label": place.label,
            "place": place,
            "centroid": np.array([float(np.mean(x)), float(np.mean(y))]),
            "bbox": place.aabb,
            "area_m2": len(cells) * grid.resolution ** 2,
            "cell_count": int(len(cells)),
        })
    return truth, gts


def describe(placements: list[Placement]) -> str:
    """One line per obstacle, for experiment logs."""
    if not placements:
        return "   (none placed)"
    return "\n".join(
        f"   {p.label:22s} at ({p.cx:7.1f}, {p.cy:7.1f}) "
        f"{p.size[0]:4.1f} x {p.size[1]:4.1f} m, "
        f"yaw {np.degrees(p.yaw):+6.1f} deg, {100 * p.frac:4.1f}% along"
        for p in placements)
