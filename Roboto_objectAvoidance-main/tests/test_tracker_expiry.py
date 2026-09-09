"""Tracks must expire, or a drifting pose manufactures obstacles forever.

DiscrepancyTracker used to only ever add: update() wrote to self.tracks and
nothing removed from it. Measured live, SLAM drifted 3 m, mapped obstacle
cells landed displaced from their true positions, the prior said that patch
of road was free, and each displacement classified as a NEW object. Six
real trucks became ten confirmed ones. The planner routed around obstacles
that did not exist and the robot stopped in open ground 3.6 m clear of
anything.
"""

import numpy as np
import pytest

from roboto_core.discrepancy.cluster import DiscrepancyObject, DiscrepancyTracker
from roboto_core.discrepancy.compare import Cls


def obj_at(x, y, half=1.0):
    return DiscrepancyObject(
        klass=Cls.MISSED, centroid=np.array([float(x), float(y)]),
        area_m2=4 * half * half,
        bbox=(x - half, y - half, x + half, y + half),
        cell_count=100, confidence=1.0)


def test_a_track_not_re_observed_expires():
    t = DiscrepancyTracker(confirm_after=1, forget_after=10.0)
    t.update([obj_at(0, 0)], stamp=0.0)
    assert len(t.confirmed()) == 1
    t.update([], stamp=5.0)
    assert len(t.confirmed()) == 1, "expired too early"
    t.update([], stamp=20.0)
    assert t.confirmed() == [], "stale track survived"


def test_a_re_observed_track_survives():
    """An obstacle the robot is still looking at must not be forgotten."""
    t = DiscrepancyTracker(confirm_after=1, forget_after=10.0)
    for s in range(0, 60, 4):
        t.update([obj_at(0, 0)], stamp=float(s))
    assert len(t.confirmed()) == 1


def test_drifting_detections_do_not_accumulate():
    """The live failure: the same obstacle seen at a drifting pose.

    Each displacement fails the IoU association and opens a new track. With
    immortal tracks the count grew without bound; with expiry only the
    recent ones survive.
    """
    t = DiscrepancyTracker(confirm_after=1, forget_after=10.0)
    for k in range(40):
        t.update([obj_at(k * 3.0, 0)], stamp=float(k))
    assert len(t.confirmed()) <= 12, f"{len(t.confirmed())} phantoms accumulated"


def test_expiry_is_measured_in_travel_not_calls():
    """A robot stopped for a long replan must not forget what stopped it."""
    t = DiscrepancyTracker(confirm_after=1, forget_after=10.0)
    t.update([obj_at(0, 0)], stamp=100.0)
    for _ in range(50):                     # many cycles, no travel
        t.update([], stamp=100.0)
    assert len(t.confirmed()) == 1


def test_forget_after_zero_disables_expiry():
    """Old behaviour remains available."""
    t = DiscrepancyTracker(confirm_after=1, forget_after=0.0)
    t.update([obj_at(0, 0)], stamp=0.0)
    t.update([], stamp=1e6)
    assert len(t.confirmed()) == 1
