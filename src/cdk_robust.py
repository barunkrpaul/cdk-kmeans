"""
Contamination-robust merge test for CDK.

Failure mode being addressed (conference paper, Sec. VII-C): under uniform
contamination the background points fill the gap between two genuine
clusters; the union of the two bridged leaves then looks like one connected
mass, the energy statistic is unremarkable, the merge is accepted, and the
two clusters fuse.

Repair: trim the sparse points from BOTH sides of the test.  For a point
set Z let d_k(z) be the distance from z to its k-th nearest neighbour
within Z (k = TRIM_K = 5).  The trimming operator T_gamma removes the
ceil(gamma * |Z|) points with the largest d_k.  The observed union is
trimmed before the bisection mechanism and the statistic are applied, and
every reference draw is trimmed by the same operator before ITS bisection
and statistic, so the mechanism-matching principle of the paper is
preserved: observed and reference statistics are produced by the identical
pipeline, now including the trim.

gamma is an explicit robustness parameter (default 0.10), reported and
swept rather than concealed; gamma = 0 recovers the untrimmed test
exactly.  Bridge points between clusters are sparse relative to cluster
cores, so they are removed preferentially, and the gap the contamination
hid becomes visible to the statistic again.  On clean data the trim
removes the outer shell of the union on both sides symmetrically, which
should leave calibration approximately unchanged; this is measured, not
assumed.
"""
from __future__ import annotations
import numpy as np
from scipy.spatial.distance import cdist
import cdk as _cdk
from cdk_geom import CDKGeo2
from cdk import mean_interpoint_distance, draw_bisection

TRIM_K = 5


def trim_sparse(Z, gamma, rng=None):
    """Remove the ceil(gamma*|Z|) points with the largest k-NN radius."""
    n = len(Z)
    if gamma <= 0 or n < 3 * TRIM_K:
        return Z
    k = min(TRIM_K, n - 1)
    D = cdist(Z, Z)
    np.fill_diagonal(D, np.inf)
    dk = np.partition(D, k - 1, axis=1)[:, k - 1]
    drop = int(np.ceil(gamma * n))
    if drop >= n - 4:
        return Z
    keep = np.argsort(dk)[: n - drop]
    return Z[keep]


class CDKRobust(CDKGeo2):
    """knn+2ec family with the gamma-trimmed merge test."""

    def __init__(self, gamma=0.10, **kw):
        kw.setdefault("family", "knn")
        super().__init__(**kw)
        self.gamma = gamma

    def _certify(self, X, lab_obs, rng, level=None):
        """Identical to CDK._certify except that the trimming operator is
        applied to the observed union and to every reference draw."""
        if len(X) > self.cert_cap:
            sel = rng.choice(len(X), self.cert_cap, replace=False)
            X = X[sel]
        # The reference is fitted to the UNTRIMMED union and each draw has the
        # union's size, so that after one application of the same trimming
        # operator the observed and the reference statistics are computed at
        # the same sample size: |trim(X)| = |trim(Z_r)| = (1-gamma)|X|.
        Xt = trim_sparse(X, self.gamma)
        mean_d = mean_interpoint_distance(Xt, self.sub_cap, rng)
        if mean_d <= 0:
            return 1.0
        lab_o = draw_bisection(Xt, self.L, self.tau, rng, mean_d)
        if lab_o is None:
            return 1.0
        t_obs = self._T(Xt, lab_o, mean_d, rng)
        hits = done = 0
        ceiling = max(5.0 * self.q, 0.25)
        gen = _cdk.make_reference(X, self.null)
        for _ in range(self.R):
            Xr = gen(rng)
            Xr = trim_sparse(Xr, self.gamma)
            md = mean_interpoint_distance(Xr, self.sub_cap, rng)
            lr = draw_bisection(Xr, self.L, self.tau, rng, md)
            t_r = 0.0 if lr is None else self._T(Xr, lr, md, rng)
            hits += int(t_r >= t_obs); done += 1
            if done >= 25 and (1.0 + hits) / (1.0 + done) > ceiling:
                break
        return (1.0 + hits) / (1.0 + done)
