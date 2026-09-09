"""Turn discrepancy cells into discrete objects.

Raw cell labels are speckled: lidar noise and sub-cell registration error
produce isolated disagreements everywhere. Objects, not cells, are what the
planner can act on and what the report should quote --

    "3 of 3 unmapped obstacles detected, 0 false positives,
     mean centroid error 0.21 m"

is a far stronger claim than a cell-level percentage, and it is the form
that can be scored against ground truth by matching.

Pipeline: opening (despeckle) -> closing (join gaps) -> connected
components -> minimum-area filter.
"""

from __future__ import annotations

from dataclasses import dataclass, field as dc_field

import numpy as np

from ..frames import GridSpec
from .compare import Cls, ComparisonResult


@dataclass
class DiscrepancyObject:
    """One coherent region where the map and the world disagree."""

    klass: Cls
    centroid: np.ndarray             # (x, y) metres
    area_m2: float
    bbox: tuple[float, float, float, float]   # xmin, ymin, xmax, ymax
    cell_count: int
    confidence: float                # mean |log-odds| margin, normalised
    obj_id: int = -1
    first_seen: float = 0.0
    cells: np.ndarray | None = dc_field(default=None, repr=False)

    @property
    def radius_m(self) -> float:
        """Radius of a circle of equal area -- a convenient inflation size."""
        return float(np.sqrt(self.area_m2 / np.pi))

    def iou(self, other: "DiscrepancyObject") -> float:
        """Bounding-box IoU, used for association and for scoring."""
        ax0, ay0, ax1, ay1 = self.bbox
        bx0, by0, bx1, by1 = other.bbox
        ix = max(0.0, min(ax1, bx1) - max(ax0, bx0))
        iy = max(0.0, min(ay1, by1) - max(ay0, by0))
        inter = ix * iy
        if inter <= 0:
            return 0.0
        ua = (ax1 - ax0) * (ay1 - ay0) + (bx1 - bx0) * (by1 - by0) - inter
        return float(inter / ua) if ua > 0 else 0.0


def cluster(result: ComparisonResult, klass: Cls,
            min_area_m2: float = 0.25,
            open_iter: int = 0, close_iter: int = 2,
            log_odds: np.ndarray | None = None) -> list[DiscrepancyObject]:
    """Extract objects of one class from a classification.

    `min_area_m2` defaults to 0.25 m^2, smaller than any real obstacle: a
    parked van, a construction barrier or a dumpster are all far larger,
    while lidar speckle is far smaller.
    """
    from scipy import ndimage

    grid = result.grid
    mask = result.labels == klass
    if not mask.any():
        return []

    # CLOSING FIRST, and opening off by default.
    #
    # A lidar sees only the NEAR FACE of an obstacle, so a detection is a
    # one- or two-cell-thick arc, not a filled blob. Opening with a 3x3
    # element erodes then dilates, which deletes any structure thinner than
    # three cells -- it removed every real detection here: 2,061 MISSED
    # cells yielded zero objects. Closing consolidates the arc instead, and
    # the minimum-area filter below already discards speckle without
    # destroying thin structure.
    if close_iter:
        mask = ndimage.binary_closing(mask, np.ones((3, 3), bool),
                                      iterations=close_iter)
    if open_iter:
        mask = ndimage.binary_opening(mask, np.ones((3, 3), bool),
                                      iterations=open_iter)
    if not mask.any():
        return []

    labelled, n = ndimage.label(mask, structure=np.ones((3, 3), bool))
    if n == 0:
        return []

    cell_area = grid.resolution ** 2
    min_cells = max(1, int(round(min_area_m2 / cell_area)))

    objects: list[DiscrepancyObject] = []
    for idx in range(1, n + 1):
        cells = np.argwhere(labelled == idx)
        if len(cells) < min_cells:
            continue

        rows, cols = cells[:, 0], cells[:, 1]
        x, y = grid.cell_to_world(rows, cols)
        x = np.asarray(x)
        y = np.asarray(y)

        conf = 1.0
        if log_odds is not None:
            m = np.abs(log_odds[rows, cols])
            conf = float(np.clip(m.mean() / 5.0, 0.0, 1.0))

        half = grid.resolution / 2.0
        objects.append(DiscrepancyObject(
            klass=klass,
            centroid=np.array([x.mean(), y.mean()]),
            area_m2=len(cells) * cell_area,
            bbox=(float(x.min() - half), float(y.min() - half),
                  float(x.max() + half), float(y.max() + half)),
            cell_count=len(cells),
            confidence=conf,
            cells=cells,
        ))

    objects.sort(key=lambda o: o.area_m2, reverse=True)
    return objects


class DiscrepancyTracker:
    """Give objects stable ids across successive comparisons.

    Replanning must not be triggered by the same obstacle twice, and
    detection latency can only be measured if an object keeps its identity
    from the frame it first appeared in.
    """

    def __init__(self, iou_threshold: float = 0.3, confirm_after: int = 3,
                 forget_after: float = 25.0):
        self.iou_threshold = float(iou_threshold)
        self.confirm_after = int(confirm_after)
        # Drop a track not re-observed within this much travel.
        #
        # WHY TRACKS MUST EXPIRE
        #
        # They used to be immortal: update() only ever added to self.tracks
        # and nothing removed from it, so the object count could only grow.
        # That is fine while the pose is good and fatal when it is not.
        #
        # Measured live: SLAM drifted 3 m, so mapped obstacle cells landed
        # displaced from their true positions. The prior says that patch of
        # road is free, so the displacement classified as a NEW object
        # rather than the one already tracked -- six real trucks became ten
        # confirmed objects. The planner then routed around obstacles that
        # were not there, and the contact check stopped the robot in open
        # ground 3.6 m clear of anything.
        #
        # A phantom born of a transient pose error is not re-observed once
        # the error changes, so expiry removes it. A real obstacle in front
        # of the robot is re-seen every cycle and survives. Measured in
        # travel rather than time so it does not expire while the robot is
        # stopped and planning.
        self.forget_after = float(forget_after)
        self._next_id = 0
        self.tracks: dict[int, DiscrepancyObject] = {}
        self._hits: dict[int, int] = {}
        self._last_seen: dict[int, float] = {}

    def update(self, objects: list[DiscrepancyObject],
               stamp: float = 0.0) -> list[DiscrepancyObject]:
        """Associate `objects` with existing tracks; return them with ids set."""
        unmatched = dict(self.tracks)
        out: list[DiscrepancyObject] = []

        for obj in objects:
            best_id, best_iou = -1, 0.0
            for tid, track in unmatched.items():
                if track.klass != obj.klass:
                    continue
                score = obj.iou(track)
                if score > best_iou:
                    best_id, best_iou = tid, score

            if best_iou >= self.iou_threshold:
                obj.obj_id = best_id
                obj.first_seen = self.tracks[best_id].first_seen
                self._hits[best_id] += 1
                unmatched.pop(best_id, None)
            else:
                obj.obj_id = self._next_id
                obj.first_seen = stamp
                self._hits[obj.obj_id] = 1
                self._next_id += 1

            self.tracks[obj.obj_id] = obj
            self._last_seen[obj.obj_id] = stamp
            out.append(obj)

        if self.forget_after > 0:
            stale = [tid for tid, seen in self._last_seen.items()
                     if stamp - seen > self.forget_after]
            for tid in stale:
                self.tracks.pop(tid, None)
                self._hits.pop(tid, None)
                self._last_seen.pop(tid, None)
        return out

    def confirmed(self) -> list[DiscrepancyObject]:
        """Objects seen often enough to act on.

        Requiring persistence is what stops a single noisy scan from
        triggering a replan.
        """
        return [o for tid, o in self.tracks.items()
                if self._hits.get(tid, 0) >= self.confirm_after]
