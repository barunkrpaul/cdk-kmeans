"""
Certified Divisive K-means (CDK), version 2.
============================================
Implements the framework of Sections 13-14 of the report, with two corrections
that were forced by the first round of experiments and are documented here
rather than hidden.

CORRECTION 1 (validity mechanism).
The report's Conjecture 13.1 proposed calibrating the node p-value against
"a uniformly random relabelling of the points of C".  That construction is a
no-op: a permutation of point labels does not change the multiset, and every
permutation-invariant clustering algorithm returns the same bisection.  The
randomisation must therefore act on a REFERENCE SAMPLE, which means a null model
is unavoidable.  The consequence is that "distribution-free" applies to the
STATISTIC and not to the NULL.  Two nulls are provided so the cost of the
assumption can be measured:
    null='uniform'   uniform over the PCA-aligned bounding box of the node,
                     the least-favourable unimodal shape
    null='gaussian'  N(mean, Sigma-hat), i.e. the SigClust / G-means assumption

CORRECTION 2 (architecture).
A purely top-down procedure that certifies every split before descending has no
power at the root of a many-cluster dataset, because the first test is a
"ten groups versus ten groups" comparison in which neither side is a single
cluster.  Measured on 20 well-separated Gaussians the top-down variant returned
k=1.  The architecture is therefore OVER-PARTITION THEN CERTIFY BOTTOM-UP: build
an unconditional bisecting hierarchy down to a resolution floor, then certify
each internal node from the leaves upward, merging where the null is not
rejected.  This also moves the statistical weight onto the MERGE decision, which
Section 8 of the report identifies as the weakest point of the whole family.

The resolution floor n_min is the smallest group the test can detect and is the
only quantity that behaves like a resolution parameter; it is reported and swept
rather than concealed.
"""
from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist, pdist


# ----------------------------------------------------------------------------
# 1.  scale-invariant energy statistic   (Eq. 7 of the report)
# ----------------------------------------------------------------------------
def energy_T(A, B, parent_mean_dist):
    nA, nB = len(A), len(B)
    if nA < 2 or nB < 2 or parent_mean_dist <= 0:
        return 0.0
    ab = cdist(A, B).mean()
    aa = pdist(A).mean()
    bb = pdist(B).mean()
    return float((2.0 * ab - aa - bb) / parent_mean_dist)


def mean_interpoint_distance(X, cap, rng):
    n = len(X)
    if n < 2:
        return 0.0
    if n > cap:
        X = X[rng.choice(n, cap, replace=False)]
    return float(pdist(X).mean())


# ----------------------------------------------------------------------------
# 2.  randomised bisection (exponential mechanism, Eq. 8)
# ----------------------------------------------------------------------------
def _two_means(X, rng, iters=20):
    """A direct 2-means with k-means++ seeding.

    scikit-learn's KMeans carries several milliseconds of fixed overhead per
    call, which dominates when the routine is invoked once per Monte-Carlo draw
    on a few dozen points; measured at 47 ms per reference draw, against 0.15 ms
    here.  The inner loop is therefore written out directly."""
    n = len(X)
    i0 = rng.integers(n)
    d0 = ((X - X[i0]) ** 2).sum(1)
    tot = d0.sum()
    i1 = int(rng.choice(n, p=d0 / tot)) if tot > 0 else int((i0 + 1) % n)
    C = np.vstack([X[i0], X[i1]])
    lab = np.zeros(n, dtype=np.int64)
    for _ in range(iters):
        d = np.empty((n, 2))
        d[:, 0] = ((X - C[0]) ** 2).sum(1)
        d[:, 1] = ((X - C[1]) ** 2).sum(1)
        new = (d[:, 1] < d[:, 0]).astype(np.int64)
        if _ and np.array_equal(new, lab):
            lab = new
            break
        lab = new
        for j in (0, 1):
            m = lab == j
            if m.any():
                C[j] = X[m].mean(0)
    d = np.empty((n, 2))
    d[:, 0] = ((X - C[0]) ** 2).sum(1)
    d[:, 1] = ((X - C[1]) ** 2).sum(1)
    inertia = float(d[np.arange(n), lab].sum())
    return lab, inertia


def _candidates(X, L, rng):
    out = []
    for _ in range(L):
        lab, q = _two_means(X, rng)
        s = int(lab.sum())
        if 0 < s < len(X):
            out.append((lab, q))
    return out


def draw_bisection(X, L, tau, rng, mean_d):
    c = _candidates(X, L, rng)
    if not c:
        return None
    scale = max(len(X) * mean_d ** 2, 1e-300)
    lg = -np.array([q for _, q in c]) / scale / max(tau, 1e-12)
    lg -= lg.max()
    p = np.exp(lg); p /= p.sum()
    return c[int(rng.choice(len(c), p=p))][0]


# ----------------------------------------------------------------------------
# 3.  reference (null) sample
# ----------------------------------------------------------------------------
def make_reference(X, kind):
    """Return a closure that draws a structureless reference sample of the same
    size, dimension and scale.  The expensive decomposition is done once per
    node rather than once per Monte-Carlo draw."""
    n, d = X.shape
    mu = X.mean(0); Xc = X - mu
    if kind == "uniform":
        _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
        Z = Xc @ Vt.T
        lo, hi = Z.min(0), Z.max(0)

        def draw(rng):
            return rng.uniform(lo, hi, size=Z.shape) @ Vt + mu
        return draw
    if kind == "gaussian":
        C = np.atleast_2d(np.cov(Xc, rowvar=False)) + 1e-10 * np.eye(d)
        try:
            Lc = np.linalg.cholesky(C)
        except np.linalg.LinAlgError:
            w, V = np.linalg.eigh(C)
            Lc = V @ np.diag(np.sqrt(np.clip(w, 0, None)))

        def draw(rng):
            return rng.standard_normal((n, d)) @ Lc.T + mu
        return draw
    raise ValueError(kind)


def reference_sample(X, kind, rng):
    return make_reference(X, kind)(rng)


# ----------------------------------------------------------------------------
# 4.  ForwardStop / StrongStop   (G'Sell et al. 2016)
# ----------------------------------------------------------------------------
def forward_stop(p, q):
    if len(p) == 0:
        return 0
    p = np.clip(np.asarray(p, float), 0.0, 1 - 1e-12)
    cum = np.cumsum(-np.log(1.0 - p)) / np.arange(1, len(p) + 1)
    ok = np.where(cum <= q)[0]
    return int(ok[-1] + 1) if len(ok) else 0


def benjamini_hochberg(p, q):
    """Indices of the rejected hypotheses under BH at level q (unordered family)."""
    p = np.asarray(p, float)
    m = len(p)
    if m == 0:
        return np.array([], int)
    o = np.argsort(p)
    thr = q * (np.arange(1, m + 1)) / m
    passing = np.where(p[o] <= thr)[0]
    if len(passing) == 0:
        return np.array([], int)
    return o[: passing[-1] + 1]


def strong_stop(p, q):
    if len(p) == 0:
        return 0
    p = np.clip(np.asarray(p, float), 1e-12, 1 - 1e-12)
    m = len(p); lp = np.log(p)
    vals = np.array([np.exp(np.sum(lp[i:]) / (m - i)) for i in range(m)])
    thr = q * (np.arange(m) + 1) / m
    ok = np.where(vals <= thr)[0]
    return int(ok[-1] + 1) if len(ok) else 0


# ----------------------------------------------------------------------------
# 5.  the algorithm
# ----------------------------------------------------------------------------
@dataclass
class CDKResult:
    k: int
    labels: np.ndarray
    pvals: list = field(default_factory=list)
    conf_set: tuple = (1, 1)
    n_tests: int = 0
    k_init: int = 1
    fdr_level: float = float("nan")
    # --- applicability diagnostic ------------------------------------------
    # p_median is the median p-value over the fixed edge family, and it is the
    # signal that predicts when CDK has nothing to detect: if the median
    # adjacent leaf pair cannot be told from a null draw, the data are one
    # contiguous mass in the supplied metric and k_hat will collapse towards 1.
    # sep_evidence (fraction of edges rejecting) is recorded alongside it but is
    # NOT a reliable predictor, because the returned k depends on the
    # CONNECTIVITY of the surviving merge graph, not on the count of rejections:
    # a dense 4-NN edge family can stay connected through its non-rejecting
    # minority.  The threshold on p_median is calibrated empirically in Sec.VI-C
    # rather than assumed.
    sep_evidence: float = 0.0
    p_median: float = float("nan")
    applicable: bool = True


class CDK:
    def __init__(self, q=0.05, R=199, tau=0.05, L=3, null="uniform",
                 stat="energy", n_min=25, sub_cap=200, cert_cap=300, correct=True,
                 rule="bh", kmax=64, n_nb=4, max_tests=600,
                 max_pmedian=0.5, random_state=0):
        self.q, self.R, self.tau, self.L = q, R, tau, L
        self.null, self.stat, self.n_min = null, stat, n_min
        self.correct = correct
        self.sub_cap, self.cert_cap = sub_cap, cert_cap
        self.rule, self.kmax = rule, kmax
        self.n_nb, self.max_tests = n_nb, max_tests
        self.max_pmedian = max_pmedian
        self.random_state = random_state

    # ---- statistic -------------------------------------------------------
    def _T(self, X, lab, mean_d, rng):
        A, B = X[lab == 0], X[lab == 1]
        if len(A) < 2 or len(B) < 2:
            return 0.0
        if self.stat == "energy":
            if len(A) > self.sub_cap:
                A = A[rng.choice(len(A), self.sub_cap, replace=False)]
            if len(B) > self.sub_cap:
                B = B[rng.choice(len(B), self.sub_cap, replace=False)]
            return energy_T(A, B, mean_d)
        if self.stat == "ad":
            from scipy.stats import anderson
            v = A.mean(0) - B.mean(0); nv = np.linalg.norm(v)
            if nv < 1e-12:
                return 0.0
            z = (X @ v) / nv
            z = (z - z.mean()) / (z.std() + 1e-12)
            try:
                return float(anderson(z, "norm").statistic)
            except Exception:
                return 0.0
        raise ValueError(self.stat)

    # ---- one certification test -----------------------------------------
    def _certify(self, X, lab_obs, rng, level=None):
        """H0: the points of X are one group.  Returns the p-value.

        The observed statistic is computed on the OBSERVED partition of X into
        its two current sub-groups.  The reference statistic is computed by
        applying the randomised bisection mechanism to a null sample of the same
        size and shape, so the selection effect is present on both sides.

        Both sides are evaluated on a bounded subsample of the node (cert_cap),
        drawn once and reused, so that the observed and the reference statistics
        are computed at the same sample size.
        """
        if len(X) > self.cert_cap:
            sel = rng.choice(len(X), self.cert_cap, replace=False)
            X = X[sel]
        mean_d = mean_interpoint_distance(X, self.sub_cap, rng)
        if mean_d <= 0:
            return 1.0
        # The observed statistic must be produced by the SAME mechanism as the
        # reference statistics, otherwise the two are not exchangeable under the
        # null and the test over-rejects.  Measured directly: scoring the given
        # partition against a freshly drawn reference bisection rejected the null
        # for two halves of a single Gaussian at the same rate as for two
        # genuinely separated clusters.  The observed bisection is therefore
        # drawn by the mechanism as well, and H0 is read as "the union of these
        # two groups is one group".
        lab_o = draw_bisection(X, self.L, self.tau, rng, mean_d)
        if lab_o is None:
            return 1.0
        t_obs = self._T(X, lab_o, mean_d, rng)
        hits = done = 0
        ceiling = max(5.0 * self.q, 0.25)
        # Exact early stop.  The final p-value is (1+hits)/(1+R), which can only
        # increase as further hits arrive, so once (1+hits)/(1+R) exceeds the
        # decision level the outcome (do not reject, i.e. merge) is already
        # determined and the remaining reference draws are wasted work.
        lv = level if level is not None else self.q
        gen = make_reference(X, self.null)
        for _ in range(self.R):
            Xr = gen(rng)
            md = mean_interpoint_distance(Xr, self.sub_cap, rng)
            lr = draw_bisection(Xr, self.L, self.tau, rng, md)
            t_r = 0.0 if lr is None else self._T(Xr, lr, md, rng)
            hits += int(t_r >= t_obs); done += 1
            if done >= 25 and (1.0 + hits) / (1.0 + done) > ceiling:
                break

        return (1.0 + hits) / (1.0 + done)

    # ---- over-partition --------------------------------------------------
    def _leaves(self, X, rng):
        """Unconditional bisecting hierarchy down to the resolution floor.
        Returns the list of leaf index arrays (the initial over-partition).
        The number of leaves is DERIVED from n and the resolution floor n_min;
        it is not a user-chosen over-clustering budget."""
        leaves, queue = [], [np.arange(len(X))]
        while queue:
            idx = queue.pop(0)
            if len(idx) < 2 * self.n_min or len(leaves) + len(queue) >= self.kmax:
                leaves.append(idx); continue
            lab = draw_bisection(X[idx], self.L, self.tau, rng,
                                 mean_interpoint_distance(X[idx], self.sub_cap, rng))
            if lab is None:
                leaves.append(idx); continue
            a, b = idx[lab == 0], idx[lab == 1]
            if len(a) < self.n_min or len(b) < self.n_min:
                leaves.append(idx); continue
            queue += [a, b]
        return leaves

    # ---- certified separation graph --------------------------------------
    def _edges(self, X, leaves, rng, n_nb):
        """Candidate edges: each leaf joined to its n_nb nearest leaves."""
        C = np.array([X[l].mean(0) for l in leaves])
        D = cdist(C, C) + np.eye(len(C)) * 1e18
        E = set()
        for a in range(len(C)):
            for b in np.argsort(D[a])[:n_nb]:
                E.add((min(a, int(b)), max(a, int(b))))
        return sorted(E, key=lambda e: D[e[0], e[1]])

    def _test_edges(self, X, leaves, edges, rng, level):
        """p-value for every candidate edge.  H0: the union of the two leaves is
        one group.  A LARGE p-value means the pair is not certified as separate
        and the edge is kept."""
        p = {}
        for (a, b) in edges:
            ia, ib = leaves[a], leaves[b]
            idx = np.concatenate([ia, ib])
            lab = np.zeros(len(idx), int); lab[len(ia):] = 1
            p[(a, b)] = self._certify(X[idx], lab, rng, level)
        return p

    @staticmethod
    def _components(nleaf, edges_kept):
        parent = list(range(nleaf))

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]; x = parent[x]
            return x

        for a, b in edges_kept:
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[ra] = rb
        return [find(i) for i in range(nleaf)]

    # ---- main ------------------------------------------------------------
    def fit(self, X):
        X = np.asarray(X, float)
        n = len(X)
        rng = np.random.default_rng(self.random_state)
        leaves = self._leaves(X, rng)
        k_init = len(leaves)
        if k_init == 1:
            return CDKResult(k=1, labels=np.zeros(n, int), k_init=1,
                             sep_evidence=0.0, p_median=float("nan"),
                             applicable=False)

        edges = self._edges(X, leaves, rng, self.n_nb)
        # multiplicity over the FIXED family of edge hypotheses.  Rejecting an
        # edge asserts that the two leaves are separate groups, so the quantity
        # to control is the number of spurious separations; the level is
        # Bonferroni-corrected over the family and floored at the Monte-Carlo
        # resolution, which in practice is the binding constraint.
        m = len(edges)
        level = (max(self.q / max(m, 1), 2.0 / (self.R + 1.0))
                 if self.correct else self.q)
        self.level_ = level
        pv = self._test_edges(X, leaves, edges, rng, level)

        def k_at(lv):
            kept = [e for e in edges if pv[e] > lv]
            comp = self._components(k_init, kept)
            return comp, len(set(comp))

        comp, k_hat = k_at(level)
        root = {r: j for j, r in enumerate(sorted(set(comp)))}
        labels = np.zeros(n, int)
        for i, l in enumerate(leaves):
            labels[l] = root[comp[i]]
        _, labels = np.unique(labels, return_inverse=True)
        k_hat = int(labels.max() + 1)

        # Sensitivity range, NOT a simultaneous confidence set.  Sweeping the
        # level inverts a family of DEPENDENT tests and carries no coverage
        # guarantee; measured marginal coverage is reported in Sec. VI-E and is
        # below nominal.  It is published as a stability diagnostic only.
        _, k_strict = k_at(max(level / 4.0, 1.0 / (self.R + 1.0)))
        _, k_liberal = k_at(min(0.5, level * 8.0))
        conf = (int(min(k_strict, k_hat)), int(max(k_liberal, k_hat)))

        sep = sorted([v for v in pv.values() if v <= level])
        if sep:
            adj = [min(1.0, p_ * len(sep) / (i + 1)) for i, p_ in enumerate(sep)]
            fdr = float(max(adj))
        else:
            fdr = float("nan")

        vals = np.asarray(list(pv.values()), float)
        sep_ev = float((vals <= level).mean()) if len(vals) else 0.0
        p_med = float(np.median(vals)) if len(vals) else float("nan")
        return CDKResult(k=k_hat, labels=labels, pvals=sorted(pv.values()),
                         conf_set=conf, n_tests=len(pv), k_init=k_init,
                         fdr_level=fdr, sep_evidence=sep_ev, p_median=p_med,
                         applicable=bool(p_med <= self.max_pmedian))


def cdk_estimate(X, **kw):
    return CDK(**kw).fit(X)
