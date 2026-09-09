"""SLAM front end: predict from odometry, correct by scan matching, map.

    odom delta  ->  predicted pose  ->  scan match  ->  corrected pose
                                                            |
                                                            v
                                                     integrate into map

THE LOCAL FIELD WINDOW
----------------------
Scan matching needs a likelihood field, and building one over the full
3000 x 3000 site grid takes ~0.66 s -- impossible at 10 Hz. But the sensor
only reaches 30 m, so a global field is almost entirely wasted work.

So the field is built over a window around the robot, roughly twice the
sensor range, and rebuilt only when the robot nears its edge. A 700 x 700
window costs ~25 ms and is rebuilt every few metres rather than every scan.
The window carries its own GridSpec with the correct world origin, so
matching still happens in world coordinates and the frame contract holds.

WHEN THE MATCH IS REJECTED
--------------------------
A low score means the scan could not be placed -- an open area with almost
no returns, or a genuine failure. The system then falls back to the
odometry prediction rather than accepting a confident-looking wrong answer,
and records that it did. Those events are worth reporting: a run with many
fallbacks has a very different error profile from one with none.
"""

from __future__ import annotations

from dataclasses import dataclass, field as dc_field

import numpy as np

from ..frames import GridSpec
from .likelihood_field import LikelihoodField
from .occupancy import OccupancyGrid
from .scan import Scan
from .scan_matcher import CorrelativeScanMatcher, MatchResult
from .transforms2d import compose, relative, wrap_angle


@dataclass
class SlamConfig:
    """Tuning for the front end."""

    # Local field window, as a multiple of the sensor range.
    window_scale: float = 2.2
    # Rebuild once the robot is this fraction of the way to the window edge.
    window_refresh_frac: float = 0.35
    field_sigma_m: float = 0.15

    # What the scan is matched AGAINST.
    #
    #   "map"    pure SLAM: match against the map built so far. Honest, but
    #            in a large outdoor scene without loop closure the map
    #            degrades and drags the estimate with it.
    #   "prior"  match against the GIS prior. The robot genuinely has this
    #            map, and localising against a known map is standard
    #            practice (AMCL does exactly this). Obstacles absent from
    #            the prior simply score as outliers, which is harmless.
    #   "fused"  match against prior OR mapped obstacles, so newly
    #            discovered structure also constrains the pose.
    #
    # This is an ablation axis, not a fixed choice -- see the measured
    # comparison in tools/run_slam_offline.py.
    reference: str = "fused"

    # Below this match score the correction is rejected and odometry used.
    # Optimal value depends on the reference: against a sparse self-built
    # map a low threshold accepts useful corrections, while against an
    # accurate prior a high threshold rejects only genuine failures.
    min_match_score: float = 0.45
    # Reject a correction that disagrees with the odometry prediction by
    # more than this. Typical live corrections are 0.01-0.04 m, so the
    # limit never governs normal driving -- only outliers and recovery.
    #
    # HISTORY, BECAUSE THE OBVIOUS READING OF IT IS WRONG
    #
    # It was 3.0, justified by a sweep in which 1.2 m gave 24.9 m
    # worst-case ATE on Pittsburgh. That sweep ran with the argmax
    # tie-break bug present: ties in the correlation volume resolved to the
    # corner of the search window, so the "good corrections" a tight limit
    # refused included bogus metre-scale jumps. With ties now broken toward
    # the prediction, Pittsburgh measures 0.21-0.27 m at EVERY limit from
    # 0.3 to 3.0. The evidence for 3.0 was an artefact of a defect.
    #
    # OFFLINE EVIDENCE -- six seeds per site, mean ATE, worst single run:
    #
    #   limit   Pittsburgh   San Francisco   Chicago   worst seen
    #     0.3      0.22 m         2.02 m      3.73 m      8.05 m
    #     0.5      0.24 m         1.32 m      4.98 m     10.28 m
    #     1.0      0.25 m         1.08 m     10.40 m     40.93 m
    #     2.0      0.26 m         0.74 m     60-84 m       84 m
    #
    # Anything >= 2.0 is refuted outright by Chicago, whose repeating
    # street grid lets the matcher lock on a block out. Among the tight
    # limits 0.3 wins on both worst case and mean, though only 1.99 m
    # against 0.5's 2.18 m averaged across sites. San Francisco pays for it
    # (2.02 m against 1.32 m); Chicago decides it, being the only site
    # where the wrong value is catastrophic rather than merely worse.
    #
    # LIVE EVIDENCE POINTS THE OTHER WAY
    #
    # One Gazebo mission at 0.3 tracked 0.02-2.0 m for 223 m, then crossed
    # over and ran away to 126.8 m as rejections climbed 199 -> 388 and the
    # map froze. A mission at 3.0 held 0.34 m with zero rejections.
    #
    # Both are real, and they are not in conflict once the mechanism is
    # clear. The limit only matters during RECOVERY, and recovery is what
    # an offline harness cannot exercise: it drives a smooth interpolated
    # trajectory with no wheel slip, so the estimate never has an excursion
    # to climb out of. Live, a robot pinned against an obstacle spins its
    # wheels, odometry gains metres of motion that never happened, and a
    # tight limit then refuses every correction that would fix it.
    #
    # THE CHOICE
    #
    # 0.3, by project decision, on the offline sweep. The live risk is
    # accepted and is bounded by not getting stuck in the first place --
    # which is what the contact check and reverse recovery in
    # plan/pursuit.py address, and what relocalisation below is the last
    # line of defence for.
    #
    # Override without rebuilding:  max_correction_m:=3.0
    #
    # Chicago stays poor at every setting -- 3.73 m mean against
    # Pittsburgh's 0.22 m, and its own odometry baseline is 4.75 m, so SLAM
    # beats dead reckoning there by 1.27x rather than the 13x seen on
    # Pittsburgh. A correction limit stops divergence; it does not fix a
    # repeating grid, which is ambiguous to a matcher with no loop closure.
    # Treat Chicago as a demonstrated limitation, not a solved site.
    max_correction_m: float = 0.3
    max_correction_rad: float = np.radians(25.0)

    # Keyframe policy: only these poses enter the pose graph later.
    keyframe_dist_m: float = 0.5
    keyframe_angle_rad: float = np.radians(15.0)

    # Skip matching until the robot has moved; matching a stationary robot
    # repeatedly just injects noise into the estimate.
    min_motion_m: float = 0.02
    min_motion_rad: float = np.radians(1.0)

    # Weight the scan-match correction by how well the scan observes it --
    # see Slam2D._weighted_pose. Without this the estimate slides freely
    # along a featureless corridor, because every pose along it explains
    # the scan equally well.
    #
    # `predict_sigma` is how far the odometry prediction is trusted for one
    # update. It sets where the crossover sits: a match covariance well
    # under this is applied in full, one well over it is largely ignored in
    # favour of dead reckoning.
    use_covariance_gain: bool = True
    predict_sigma_m: float = 0.05
    predict_sigma_rad: float = np.radians(1.0)

    # --- relocalisation -----------------------------------------------
    #
    # LOSING LOCK USED TO BE PERMANENT. Measured live: during a ~120 degree
    # turnaround at 154 m into a mission, matching stopped being accepted;
    # `matched` froze and never advanced again over the following 570 m.
    # Once matches are rejected the estimate falls back to odometry, the
    # local likelihood field is then rebuilt around an increasingly wrong
    # pose, and every subsequent match has even less chance of scoring --
    # so the failure feeds itself. Error reached 134 m, and no mechanism
    # existed that could ever have brought it back.
    #
    # The recovery is a deliberately wide, coarse search against the GIS
    # PRIOR rather than the self-built map: the prior is fixed and correct,
    # while the map was built through the divergence and cannot be trusted
    # to say where the robot is. This is the third job the correlative
    # matcher does, and the one its docstring anticipates.
    #
    # Only after this many CONSECUTIVE rejections, because the search is
    # expensive and a handful of rejections in a row is normal in open
    # ground where there is nothing to match against.
    relocalise_after: int = 25
    relocalise_win_m: float = 10.0
    relocalise_lin: float = 0.5
    relocalise_ang: float = np.radians(5.0)

    # RECOVERY IS ONLY MEANINGFUL ONCE THERE WAS SOMETHING TO RECOVER.
    # A mission opens with a bootstrap phase in which the map is nearly
    # empty and matches are refused -- measured live, 19 consecutive
    # rejections before the first acceptance. Recovery triggered inside
    # that phase teleported the estimate 13.5 m and 127 degrees away while
    # odometry was still accurate to 8 cm, because the robot had never had
    # a lock to lose. Require an established track first.
    relocalise_min_matches: int = 40

    # Heading search half-width. NOT a full rotation.
    #
    # A suburban block is full of near-repeats, and a wide rotational
    # search finds them: the false recovery above scored 0.78-0.84 at a
    # pose 127 degrees wrong, i.e. the alias genuinely explains the scan
    # better than the truth does on that evidence alone. Odometry heading
    # stays good over the short term even when position matching fails, so
    # a bounded window is both sufficient for the failure this targets --
    # observed heading error at loss of lock was tens of degrees, not
    # hundreds -- and far less prone to aliasing.
    relocalise_win_th: float = np.radians(45.0)


    # Deliberately above min_match_score. Accepting a relocalisation means
    # teleporting the estimate, so it must be a clearly better explanation
    # of the scan than the pose we already have, not a marginal one.
    relocalise_score: float = 0.70
    # ...and it must beat the match it is replacing by this margin, so a
    # recovery never fires on evidence no stronger than what was refused.
    relocalise_margin: float = 0.15


@dataclass
class SlamStep:
    """What happened on one update, for logging and evaluation."""

    pose: np.ndarray
    odom_pose: np.ndarray
    score: float = 0.0
    matched: bool = False           # correction applied
    rejected: bool = False          # match attempted but refused
    is_keyframe: bool = False
    relocalised: bool = False       # recovered by a wide search on the prior
    correction: np.ndarray = dc_field(default_factory=lambda: np.zeros(3))
    covariance: np.ndarray | None = None


class Slam2D:
    """Scan-matching SLAM over a fixed, GIS-aligned grid.

    The grid is supplied by the caller rather than grown on demand: it must
    match the GIS prior exactly so discrepancy detection is a direct array
    comparison with no resampling.
    """

    def __init__(self, grid: GridSpec, sensor_range: float = 30.0,
                 config: SlamConfig | None = None,
                 initial_pose=(0.0, 0.0, 0.0),
                 prior_occ: np.ndarray | None = None):
        self.cfg = config or SlamConfig()
        self.grid = grid
        self.sensor_range = float(sensor_range)
        self.prior_occ = None if prior_occ is None else np.asarray(prior_occ, bool)
        if self.cfg.reference != "map" and self.prior_occ is None:
            raise ValueError(
                f"reference={self.cfg.reference!r} needs prior_occ")

        self.map = OccupancyGrid(grid)
        self.pose = np.asarray(initial_pose, dtype=np.float64).copy()

        self._field: LikelihoodField | None = None
        self._field_center: np.ndarray | None = None
        self._last_odom: np.ndarray | None = None
        self._consecutive_rejects = 0
        self.n_relocalised = 0
        # Attempts, not successes. n_relocalised alone could not distinguish
        # "recovery never fired" from "recovery fired and failed every
        # time", and those need completely different fixes.
        self.n_reloc_attempts = 0

        self.trajectory: list[np.ndarray] = []
        self.odom_trajectory: list[np.ndarray] = []
        self.keyframes: list[tuple[np.ndarray, Scan]] = []
        self.steps: list[SlamStep] = []

    # ---- statistics ------------------------------------------------------

    @property
    def n_matched(self) -> int:
        return sum(s.matched for s in self.steps)

    @property
    def n_rejected(self) -> int:
        return sum(s.rejected for s in self.steps)

    @property
    def mean_score(self) -> float:
        scored = [s.score for s in self.steps if s.matched]
        return float(np.mean(scored)) if scored else 0.0

    def summary(self) -> dict:
        return {
            "steps": len(self.steps),
            "matched": self.n_matched,
            "rejected": self.n_rejected,
            "relocalised": self.n_relocalised,
            "keyframes": len(self.keyframes),
            "mean_score": self.mean_score,
            "coverage": self.map.coverage,
        }

    # ---- main loop -------------------------------------------------------

    def update(self, scan: Scan, odom_pose) -> SlamStep:
        """Fold one (scan, odometry) pair into the estimate."""
        odom = np.asarray(odom_pose, dtype=np.float64)

        # --- predict ------------------------------------------------------
        if self._last_odom is None:
            delta = np.zeros(3)
        else:
            delta = relative(self._last_odom, odom)
        self._last_odom = odom.copy()

        predicted = compose(self.pose, delta)
        moved = (float(np.hypot(delta[0], delta[1])), abs(float(delta[2])))

        step = SlamStep(pose=predicted.copy(), odom_pose=odom.copy())

        # --- correct ------------------------------------------------------
        # With a prior available the field exists from the start, so there
        # is no bootstrap phase in which matching is impossible.
        first_scan = self._field is None and self.cfg.reference == "map"
        worth_matching = (moved[0] >= self.cfg.min_motion_m
                          or moved[1] >= self.cfg.min_motion_rad)

        if not first_scan and worth_matching and len(scan.points()):
            self._ensure_field(predicted)
            result = CorrelativeScanMatcher(self._field).match(
                scan.points(), predicted)
            step.score = result.score
            step.covariance = result.covariance

            if self._accept(result, predicted):
                self.pose = self._weighted_pose(predicted, result)
                step.matched = True
                step.correction = relative(predicted, self.pose)
                self._consecutive_rejects = 0
            else:
                self.pose = predicted
                step.rejected = True
                self._consecutive_rejects += 1

                # Lost. Try to find ourselves again against the prior
                # before odometry carries the estimate somewhere a local
                # search can never come back from.
                if (self.prior_occ is not None
                        and self._consecutive_rejects >= self.cfg.relocalise_after
                        and self.n_matched >= self.cfg.relocalise_min_matches):
                    # THE MARGIN APPLIES ONLY TO SCORE-BASED REFUSALS.
                    #
                    # _relocalise demands the recovery beat the refused
                    # score by relocalise_margin. That is right when the
                    # match was refused for being POOR: replacing weak
                    # evidence with equally weak evidence proves nothing.
                    #
                    # It is unsatisfiable when the match was refused for
                    # being FAR. Measured live: a match scoring 1.000 was
                    # refused for exceeding max_correction_m, which set the
                    # floor to 1.000 + 0.15 = 1.15 against scores that are
                    # normalised to at most 1.0. Recovery fired on 71
                    # consecutive rejections and was guaranteed to return
                    # None every time -- the estimate was told the scan fit
                    # perfectly somewhere else, and the system refused both
                    # to go there and to recover.
                    #
                    # A distance refusal is not doubt about the evidence,
                    # it is doubt about the position, so only the absolute
                    # floor applies.
                    self.n_reloc_attempts += 1
                    found = self._relocalise(scan.points(), predicted,
                                             result.score)
                    if found is not None:
                        self.pose = found.pose
                        step.matched = True
                        step.rejected = False
                        step.relocalised = True
                        step.score = found.score
                        step.correction = relative(predicted, found.pose)
                        self._consecutive_rejects = 0
                        self.n_relocalised += 1
                        # The old local field describes where we thought we
                        # were, which is exactly the wrong place.
                        self._field = None
                        self._field_center = None
        else:
            self.pose = predicted

        step.pose = self.pose.copy()

        # --- map ----------------------------------------------------------
        self.map.integrate_scan(self.pose, scan, max_range=self.sensor_range)

        # The field is stale the moment the map changes; rebuild lazily on
        # the next match rather than eagerly here.
        if first_scan:
            self._ensure_field(self.pose)

        # --- bookkeeping --------------------------------------------------
        self.trajectory.append(self.pose.copy())
        self.odom_trajectory.append(odom.copy())
        if self._is_keyframe():
            self.keyframes.append((self.pose.copy(), scan))
            step.is_keyframe = True
        self.steps.append(step)
        return step

    # ---- internals -------------------------------------------------------

    def _weighted_pose(self, predicted, result: MatchResult) -> np.ndarray:
        """Apply the correction in proportion to how well it is observed.

        THE PROBLEM THIS SOLVES: a long street of parallel walls constrains
        the robot sideways and in heading, and says almost nothing about
        where it is ALONG the street. Every pose slid a few metres up or
        down the corridor explains the scan equally well. Taking the
        matcher's answer wholesale then lets the estimate wander freely in
        that direction -- measured on a regular city grid, error oscillating
        between 2 and 5 m with heading error near zero, which is the
        signature of pure along-corridor sliding.

        The matcher already computes exactly the quantity needed: because it
        evaluated the objective everywhere in the window, the spread of
        high-scoring poses IS the uncertainty, and in a corridor that
        covariance comes out elongated along the corridor. It was being
        thrown away.

        So take a Kalman-style gain, K = P (P + C)^-1, with P the
        (small, roughly isotropic) uncertainty of the odometry prediction
        and C the match covariance. Where the scan pins the pose down,
        C << P and K approaches 1 -- the correction is applied in full.
        Along an unobservable direction C >> P and K approaches 0, so the
        estimate keeps following odometry, which is the better source there.
        """
        if not self.cfg.use_covariance_gain or result.covariance is None:
            return result.pose

        delta = relative(predicted, result.pose)
        delta = np.array([delta[0], delta[1], wrap_angle(delta[2])])

        p = np.diag([self.cfg.predict_sigma_m ** 2,
                     self.cfg.predict_sigma_m ** 2,
                     self.cfg.predict_sigma_rad ** 2])
        c = np.asarray(result.covariance, dtype=np.float64)
        try:
            gain = p @ np.linalg.inv(p + c)
        except np.linalg.LinAlgError:
            return result.pose

        return compose(predicted, gain @ delta)

    def _accept(self, result: MatchResult, predicted) -> bool:
        """Reject implausible corrections rather than trusting the score alone."""
        if result.score < self.cfg.min_match_score:
            return False
        d = relative(predicted, result.pose)
        if np.hypot(d[0], d[1]) > self.cfg.max_correction_m:
            return False
        if abs(wrap_angle(d[2])) > self.cfg.max_correction_rad:
            return False
        return True

    def _relocalise(self, pts, predicted,
                    refused_score: float) -> MatchResult | None:
        """Wide, coarse search against the PRIOR. Returns None if unsure.

        Against the prior and not the fused reference on purpose. By the
        time this runs the estimate is wrong, so the self-built map has
        been carved at wrong poses and agreeing with it is exactly how the
        system got lost. The prior is the only reference still trustworthy.

        `refused_score` is what the ordinary match just scored. A recovery
        that cannot beat it by a clear margin is not evidence of anything,
        and teleporting the estimate on it makes things worse -- which is
        exactly what an earlier, more permissive version of this did.
        """
        if len(pts) == 0:
            return None

        half = self.cfg.relocalise_win_m + self.sensor_range
        field = self._build_local_field(predicted, half, reference="prior")

        matcher = CorrelativeScanMatcher(
            field,
            coarse_lin=self.cfg.relocalise_lin,
            coarse_ang=self.cfg.relocalise_ang,
            coarse_win_xy=self.cfg.relocalise_win_m,
            coarse_win_th=self.cfg.relocalise_win_th,
            fine_lin=self.cfg.relocalise_lin / 5,
            fine_ang=np.radians(0.5),
            fine_win_xy=self.cfg.relocalise_lin,
            fine_win_th=self.cfg.relocalise_ang,
        )
        result = matcher.match(pts, predicted)
        # KNOWN LIMITATION, DELIBERATELY NOT PAPERED OVER.
        #
        # Scores are normalised to at most 1.0, so when a match is refused
        # for DISTANCE rather than quality -- scoring 1.000 and merely
        # further than max_correction_m -- this floor becomes 1.15 and
        # recovery cannot succeed. Measured live: 71 consecutive
        # rejections, recovery attempted every time, guaranteed None. In
        # that state the estimate is told the scan fits perfectly somewhere
        # else and the system refuses both to go there and to recover.
        #
        # Lowering the bar for distance refusals was tried and is WORSE:
        # tests/test_relocalise.py::test_recovers_from_a_kidnap then
        # accepts a bogus recovery and lands 20.7 m from truth instead of
        # inside 2 m. A wide search against a repeating street grid finds
        # convincing wrong answers, which is precisely what this margin
        # exists to refuse. Trading a stuck estimate for a confidently
        # wrong one is not an improvement.
        #
        # The real fix is not a threshold. A match that scores 1.000 and
        # proposes the SAME correction for many consecutive cycles is
        # evidence that odometry is wrong, and that consistency -- not the
        # score of any single frame -- is what should unlock a large jump.
        # That needs designing and testing, not tuning.
        #
        # Until then the mitigation is upstream: do not get wedged. See the
        # contact check and reverse recovery in plan/pursuit.py.
        # n_reloc_attempts distinguishes "never fired" from "fired and
        # failed", which this comment exists because nothing could.
        floor = max(self.cfg.relocalise_score,
                    refused_score + self.cfg.relocalise_margin)
        return result if result.score >= floor else None

    def _ensure_field(self, pose) -> None:
        """(Re)build the local likelihood field if the robot has moved off it."""
        half = self.cfg.window_scale * self.sensor_range / 2.0
        if self._field is not None and self._field_center is not None:
            drift = np.hypot(pose[0] - self._field_center[0],
                             pose[1] - self._field_center[1])
            if drift < half * self.cfg.window_refresh_frac:
                return

        self._field = self._build_local_field(pose, half)
        self._field_center = np.asarray(pose[:2], dtype=np.float64).copy()

    def _build_local_field(self, pose, half: float,
                           reference: str | None = None) -> LikelihoodField:
        g = self.grid
        reference = reference or self.cfg.reference
        r0, c0 = g.world_to_cell(pose[0] - half, pose[1] - half)
        r1, c1 = g.world_to_cell(pose[0] + half, pose[1] + half)
        r0 = int(np.clip(r0, 0, g.height - 1))
        c0 = int(np.clip(c0, 0, g.width - 1))
        r1 = int(np.clip(r1, r0 + 1, g.height))
        c1 = int(np.clip(c1, c0 + 1, g.width))

        mapped = self.map.occupied()[r0:r1, c0:c1]
        if reference == "map" or self.prior_occ is None:
            sub_occ = mapped
        elif reference == "prior":
            sub_occ = self.prior_occ[r0:r1, c0:c1]
        else:                                   # "fused"
            sub_occ = self.prior_occ[r0:r1, c0:c1] | mapped
        # Origin is the corner of cell (r0, c0), not its centre -- the same
        # corner-versus-centre distinction the map_server YAML uses.
        sub_grid = GridSpec(
            origin_x=g.origin_x + c0 * g.resolution,
            origin_y=g.origin_y + r0 * g.resolution,
            resolution=g.resolution,
            width=c1 - c0,
            height=r1 - r0,
        )
        return LikelihoodField.from_occupancy(
            sub_occ, sub_grid, sigma_m=self.cfg.field_sigma_m)

    def _is_keyframe(self) -> bool:
        if not self.keyframes:
            return True
        last = self.keyframes[-1][0]
        d = relative(last, self.pose)
        return (np.hypot(d[0], d[1]) >= self.cfg.keyframe_dist_m
                or abs(d[2]) >= self.cfg.keyframe_angle_rad)

    # ---- evaluation ------------------------------------------------------

    def ate(self, truth) -> dict:
        """Absolute trajectory error against ground truth, unaligned.

        Reported without Umeyama alignment because this system is
        georeferenced: the map frame is a real projected CRS, so absolute
        error in it is meaningful and is the number that matters for
        comparing against the GIS prior.
        """
        est = np.asarray(self.trajectory, dtype=np.float64)
        gt = np.asarray(truth, dtype=np.float64)
        n = min(len(est), len(gt))
        if n == 0:
            return {"n": 0}
        err = np.hypot(est[:n, 0] - gt[:n, 0], est[:n, 1] - gt[:n, 1])
        ang = np.abs(wrap_angle(est[:n, 2] - gt[:n, 2]))
        return {
            "n": int(n),
            "rmse_m": float(np.sqrt((err ** 2).mean())),
            "mean_m": float(err.mean()),
            "median_m": float(np.median(err)),
            "max_m": float(err.max()),
            "final_m": float(err[-1]),
            "rmse_deg": float(np.degrees(np.sqrt((ang ** 2).mean()))),
        }

    def odom_ate(self, truth) -> dict:
        """Same metric for raw odometry -- the baseline SLAM must beat."""
        est = np.asarray(self.odom_trajectory, dtype=np.float64)
        gt = np.asarray(truth, dtype=np.float64)
        n = min(len(est), len(gt))
        if n == 0:
            return {"n": 0}
        err = np.hypot(est[:n, 0] - gt[:n, 0], est[:n, 1] - gt[:n, 1])
        return {
            "n": int(n),
            "rmse_m": float(np.sqrt((err ** 2).mean())),
            "final_m": float(err[-1]),
        }
