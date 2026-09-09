"""Multi-resolution correlative scan matching (Olson 2009).

WHY THIS AND NOT ICP OR GRADIENT DESCENT
----------------------------------------
ICP and Hector-style Gauss-Newton are *local* methods: they follow a
gradient from the initial guess and diverge once the initial yaw error
exceeds roughly 15-20 degrees. After odometry drifts across a suburban
block, that is exactly the regime we are in. Correlative matching instead
searches a bounded window exhaustively, so within that window it cannot get
stuck in a local optimum -- it either finds the global best or the true pose
was outside the window, and the score tells you which.

It also pays for itself three more times:

  * the score surface yields an edge covariance for free, which the pose
    graph needs and ICP does not provide without extra Hessian machinery;
  * the same code with a wider window and branch-and-bound is the
    loop-closure verifier;
  * at coarse resolution it registers the SLAM map against the GIS prior.

Cost is managed by searching coarse-then-fine: a wide window at low
resolution with a thinned scan, then a narrow window at high resolution.
"""

from __future__ import annotations

from dataclasses import dataclass, field as dc_field

import numpy as np

from .likelihood_field import LikelihoodField
from .transforms2d import transform_points, wrap_angle


@dataclass
class MatchResult:
    """Outcome of one scan match."""

    pose: np.ndarray                 # (3,) corrected world pose
    score: float                     # mean field value in [0, 1]
    covariance: np.ndarray           # 3x3, in (x, y, theta)
    n_points: int
    # Optimum sat on the search boundary, so the true pose is probably
    # outside the window. Supplementary only: with a hopeless initialisation
    # the score surface is noise and its argmax can land in the interior, so
    # a LOW SCORE is the reliable signal of an unusable match, not this flag.
    hit_window_edge: bool = False
    meta: dict = dc_field(default_factory=dict)

    @property
    def information(self) -> np.ndarray:
        """Inverse covariance, for pose-graph edges."""
        return np.linalg.inv(self.covariance)


class CorrelativeScanMatcher:
    """Coarse-to-fine exhaustive search over (x, y, theta)."""

    def __init__(
        self,
        field: LikelihoodField,
        coarse_lin: float = 0.20,
        coarse_ang: float = np.radians(2.0),
        coarse_win_xy: float = 1.5,
        coarse_win_th: float = np.radians(20.0),
        fine_lin: float = 0.025,
        fine_ang: float = np.radians(0.25),
        fine_win_xy: float = 0.25,
        fine_win_th: float = np.radians(2.5),
        max_points: int = 180,
        cov_temperature: float = 60.0,
        score_trim: float = 0.10,
    ):
        self.field = field
        self.coarse = (coarse_lin, coarse_ang, coarse_win_xy, coarse_win_th)
        self.fine = (fine_lin, fine_ang, fine_win_xy, fine_win_th)
        self.max_points = int(max_points)

        # Fraction of worst-scoring points discarded before averaging.
        #
        # THIS IS LOAD-BEARING FOR THIS PROJECT. The premise is that the
        # world contains obstacles the reference map does not, so a plain
        # mean is exactly the wrong statistic: points landing on an unmapped
        # obstacle score ~0, and the matcher can raise the mean by SHIFTING
        # THE POSE until those points land on something else. It therefore
        # gets dragged off the truth by the very features we are trying to
        # detect.
        #
        # Measured: injecting 69 m^2 of obstacles (0.1% of the site) into an
        # otherwise perfectly matched run moved ATE from 0.12 m to 13.4 m --
        # worse than using no scan matching at all. Trimming removes the
        # incentive, because outliers are discarded rather than optimised
        # against.
        #
        # 0.10 is chosen from a sweep over three seeds, not one run. More
        # trimming is NOT more robust: at 0.20 the coarse pass (already
        # subsampled to ~60 points) becomes under-constrained, corrections
        # grow, get rejected, and the estimate falls back to drifting
        # odometry. Worst-case ATE over three seeds, in metres:
        #
        #     trim   max_correction 1.2 m   3.0 m
        #     0.05          24.93            0.53
        #     0.10           0.74            0.60      <- stable at both
        #     0.20          39.06           19.81
        self.score_trim = float(np.clip(score_trim, 0.0, 0.9))
        # Sharpness of the softmax used to turn the score surface into a
        # covariance. Higher = more confident (tighter) covariance.
        self.cov_temperature = float(cov_temperature)

    # ---- public API ------------------------------------------------------

    def match(self, pts_body: np.ndarray, init_pose) -> MatchResult:
        """Align body-frame scan points to the field, starting from init_pose.

        `pts_body` is (N, 2) in the sensor/body frame -- typically
        `scan.points()`, which already drops no-return beams.
        """
        pts = np.asarray(pts_body, dtype=np.float64)
        if pts.ndim != 2 or pts.shape[1] != 2:
            raise ValueError(f"expected (N, 2) points, got {pts.shape}")

        init = np.asarray(init_pose, dtype=np.float64)
        if len(pts) == 0:
            return MatchResult(init.copy(), 0.0, np.eye(3) * 1e6, 0,
                               meta={"reason": "no points"})

        # Coarse pass on a thinned cloud: the wide window dominates cost and
        # sub-decimetre accuracy is not needed to land inside the fine window.
        coarse_pts = pts[:: max(1, len(pts) // 60)]
        c_pose, c_score, c_vol, c_edge = self._search(
            coarse_pts, init, *self.coarse)

        # Fine pass on the full cloud, centred on the coarse optimum.
        f_pts = pts[:: max(1, len(pts) // self.max_points)]
        f_pose, f_score, volume, f_edge = self._search(f_pts, c_pose, *self.fine)

        # SHAPE FROM THE COARSE PASS, SCALE FROM THE FINE ONE.
        #
        # The covariance used to come from the fine volume alone, whose
        # window is +/-0.25 m. Corridor ambiguity spans METRES, so it simply
        # is not visible at that scale and the covariance always came out
        # looking confident -- which made it useless for detecting the one
        # situation it most needed to describe.
        #
        # The coarse pass searches +/-1.5 m, which is the scale ambiguity
        # actually lives at, so its score surface shows the elongation.
        # Taking the coarse SHAPE and the fine SCALE keeps both facts: which
        # direction is poorly observed, and how precisely the optimum itself
        # is resolved.
        cov = self._covariance(volume, *self.fine)
        cov = self._reshape_by(cov, self._covariance(c_vol, *self.coarse))
        return MatchResult(
            pose=f_pose,
            score=float(f_score),
            covariance=cov,
            n_points=len(f_pts),
            hit_window_edge=bool(c_edge or f_edge),
            meta={"coarse_score": float(c_score)},
        )

    # ---- search ----------------------------------------------------------

    def _search(self, pts, center, lin, ang, win_xy, win_th):
        """Exhaustive search around `center`. Returns (pose, score, volume, edge)."""
        g = self.field.grid
        fld = self.field.field

        d = np.arange(-win_xy, win_xy + 1e-9, lin)
        thetas = wrap_angle(center[2] + np.arange(-win_th, win_th + 1e-9, ang))

        # scores[t, i, j] for theta t, x offset i, y offset j
        scores = np.empty((len(thetas), len(d), len(d)), dtype=np.float32)

        for t, th in enumerate(thetas):
            c, s = np.cos(th), np.sin(th)
            # Rotate once per angle; translations are then pure offsets.
            rx = c * pts[:, 0] - s * pts[:, 1] + center[0]
            ry = s * pts[:, 0] + c * pts[:, 1] + center[1]

            # (N, D) candidate coordinates per axis, then broadcast to
            # (N, D, D) so every (dx, dy) pair is scored in one shot.
            col = np.floor((rx[:, None] + d[None, :] - g.origin_x) / g.resolution)
            row = np.floor((ry[:, None] + d[None, :] - g.origin_y) / g.resolution)
            col = col.astype(np.int64)[:, :, None]        # (N, D, 1)
            row = row.astype(np.int64)[:, None, :]        # (N, 1, D)

            ok = ((row >= 0) & (row < g.height) & (col >= 0) & (col < g.width))
            vals = fld[np.clip(row, 0, g.height - 1), np.clip(col, 0, g.width - 1)]
            vals = np.where(ok, vals, 0.0)                     # (N, D, D)

            if self.score_trim > 0 and vals.shape[0] > 4:
                keep = max(1, int(round(vals.shape[0] * (1.0 - self.score_trim))))
                cut = vals.shape[0] - keep
                # Partition so the `keep` best values sit at the top, then
                # average only those: outliers are dropped, not optimised.
                scores[t] = np.partition(vals, cut, axis=0)[cut:].mean(axis=0)
            else:
                scores[t] = vals.mean(axis=0)

        # TIE-BREAK TOWARDS THE PREDICTION.
        #
        # Ties are not rare, they are the normal case in regular geometry: a
        # scan taken in a street of long parallel walls lands every kept
        # point on an occupied cell at the true pose AND at a pose shifted
        # along the wall, so both score exactly 1.0. Plain argmax returns
        # the FIRST maximum, and since `d` runs from -win_xy upwards that is
        # the most negative offset in the window -- the corner.
        #
        # Measured on a regular city grid, that bias destroyed the estimate
        # within a metre of the start: corrections of 1.8 m and heading
        # errors of 20 degrees against a +/-1.5 m, +/-20 degree window, at a
        # reported score of 1.000, while odometry was still accurate to a
        # centimetre. On irregular geometry exact ties are rare and the bug
        # stayed hidden.
        #
        # The penalty is a weak motion prior: among poses the scan cannot
        # distinguish, prefer the one closest to where odometry says we are.
        # At 1e-5 it is ~500x smaller than the smallest meaningful score
        # difference (one point of ~180 entering or leaving the field is
        # worth ~0.005), so it can only ever separate genuine ties.
        off_xy = np.abs(d) / max(win_xy, 1e-9)
        off_th = np.abs(wrap_angle(thetas - center[2])) / max(win_th, 1e-9)
        bias = (off_th[:, None, None] + off_xy[None, :, None]
                + off_xy[None, None, :])
        t, i, j = np.unravel_index(
            int(np.argmax(scores.astype(np.float64) - 1e-5 * bias)),
            scores.shape)
        pose = np.array([center[0] + d[i], center[1] + d[j], thetas[t]])

        # An optimum on the boundary means the true pose is probably outside
        # the window -- the caller should treat the result as unreliable
        # rather than trusting a clipped answer.
        edge = (t in (0, len(thetas) - 1)
                or i in (0, len(d) - 1)
                or j in (0, len(d) - 1))
        return pose, float(scores[t, i, j]), (scores, d, thetas), edge

    # ---- covariance ------------------------------------------------------

    @staticmethod
    def _reshape_by(fine_cov: np.ndarray, coarse_cov: np.ndarray) -> np.ndarray:
        """Give `fine_cov` the anisotropy of `coarse_cov`, keeping its size.

        Only the translation block is reshaped. The coarse surface says
        which direction the scan fails to pin down; the fine surface says
        how sharp the optimum is. Rescaling the coarse block so its
        SMALLEST eigenvalue matches the fine one transfers the first without
        inflating the second -- so a well-constrained match stays confident
        and near-isotropic, while a corridor stays confident across the
        street and uncertain along it.
        """
        out = np.array(fine_cov, dtype=np.float64, copy=True)
        a = np.asarray(coarse_cov, dtype=np.float64)[:2, :2]
        try:
            eig = np.linalg.eigvalsh(a)
        except np.linalg.LinAlgError:
            return out
        lo = float(eig.min())
        if not np.isfinite(lo) or lo <= 1e-12:
            return out

        fine_lo = float(np.linalg.eigvalsh(out[:2, :2]).min())
        if not np.isfinite(fine_lo) or fine_lo <= 0:
            return out
        out[:2, :2] = a * (fine_lo / lo)
        return out

    def _covariance(self, volume, lin, ang, win_xy, win_th) -> np.ndarray:
        """Weighted second moment of the score surface.

        This is the reason for choosing correlative matching: the search
        already evaluated the objective everywhere in the window, so the
        spread of high-scoring poses *is* the uncertainty. A corridor gives
        a covariance elongated along the corridor, which is both correct and
        exactly what the pose graph needs.
        """
        scores, d, thetas = volume
        w = np.exp(self.cov_temperature * (scores - scores.max())).astype(np.float64)
        tot = w.sum()
        if not np.isfinite(tot) or tot <= 0:
            return np.diag([1.0, 1.0, 0.25])

        TH, DX, DY = np.meshgrid(thetas, d, d, indexing="ij")
        # Use relative theta so wrapping cannot distort the moment.
        TH = wrap_angle(TH - thetas[len(thetas) // 2])
        stack = np.stack([DX.ravel(), DY.ravel(), TH.ravel()])
        wv = w.ravel()

        mean = (stack * wv).sum(axis=1) / tot
        dev = stack - mean[:, None]
        cov = (dev * wv) @ dev.T / tot

        # Floor at the search discretisation: the surface cannot resolve
        # detail finer than one step, so a tighter covariance would be a
        # fiction that makes the pose graph overconfident.
        floor = np.diag([(lin / 2) ** 2, (lin / 2) ** 2, (ang / 2) ** 2])
        return cov + floor


def match_scan(field: LikelihoodField, scan, init_pose, **kw) -> MatchResult:
    """Convenience wrapper taking a `Scan` rather than raw points."""
    return CorrelativeScanMatcher(field, **kw).match(scan.points(), init_pose)
