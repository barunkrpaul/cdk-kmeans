"""
UniForCE (Vardakas, Kalogeratos & Likas, Pattern Recognition 172:112357, 2026).

Reimplemented from the paper (arXiv:2312.11323) for use as a benchmark
baseline.  Pipeline:

1. Over-cluster with global k-means++ to K' subclusters (default K'=50);
   eliminate subclusters smaller than M=25 points and reassign their points
   to the nearest surviving centre.
2. For each candidate pair, test unimodality of the union: project the points
   of both subclusters onto the direction joining the two centres (signed
   distance to the perpendicular bisecting hyperplane), balance the sample
   sizes by subsampling the larger side to the smaller, apply Hartigan's dip
   test, and repeat L=11 times with majority vote at level alpha=0.001.
3. Assemble the unimodality spanning forest with a modified Kruskal run over
   centre-distance-sorted pairs: an edge joins two trees only when the pair
   passes the unimodality test.  Pairs already in one tree are skipped.
4. k = number of trees in the forest.

Checked against the official implementation (github.com/gvardakas/UniForCE):
the small-subcluster rule reassigns the SAMPLES of subclusters below min_size
to the nearest active subcluster and recomputes centres (it does not drop the
data), the larger side of a tested pair is subsampled to the smaller, and the
vote is a majority over n_tests=11 dip tests at alpha=0.001.

Deviations from the paper, stated rather than hidden:
- Global k-means++ uses a small candidate pool (3 candidates per added
  centre) rather than the official n_init=100 pool, for cost.
- The official default K'=50 is kept, with a documented fallback for small
  datasets: if fewer than 2 subclusters reach min_size (impossible for the
  official code, which crashes there), the over-clustering is rerun with
  K' = max(2, n // (2*min_size)).
- The official code sorts candidate edges by the distance of the first two
  coordinates only; this implementation uses the full Euclidean distance.
"""
from __future__ import annotations
import numpy as np
from scipy.spatial.distance import cdist
import diptest


def _lloyd(X, C, iters=50):
    for _ in range(iters):
        lab = cdist(X, C).argmin(1)
        newC = np.array([X[lab == j].mean(0) if (lab == j).any() else C[j]
                         for j in range(len(C))])
        if np.allclose(newC, C):
            break
        C = newC
    return C, cdist(X, C).argmin(1)


def global_kmeanspp(X, Kp, rng, candidates=3):
    """Incremental k-means: add one centre at a time by k-means++ sampling,
    refining with Lloyd after each addition."""
    n = len(X)
    C = X[[rng.integers(n)]]
    C, lab = _lloyd(X, C)
    while len(C) < min(Kp, n):
        d2 = ((X - C[cdist(X, C).argmin(1)]) ** 2).sum(1)
        tot = d2.sum()
        if tot <= 0:
            break
        best = None
        for _ in range(candidates):
            i = int(rng.choice(n, p=d2 / tot))
            C2, lab2 = _lloyd(X, np.vstack([C, X[i]]), iters=30)
            q = ((X - C2[lab2]) ** 2).sum()
            if best is None or q < best[0]:
                best = (q, C2, lab2)
        _, C, lab = best
    return C, lab


def unimodal_pair(Xi, Xj, mu_i, mu_j, rng, alpha=0.001, L=11):
    r = mu_j - mu_i
    nr = np.linalg.norm(r)
    if nr < 1e-12:
        return True
    r = r / nr
    mid = (mu_i + mu_j) / 2.0
    pi = (Xi - mid) @ r
    pj = (Xj - mid) @ r
    s = min(len(pi), len(pj))
    if s < 4:
        return True
    votes = 0
    for _ in range(L):
        a = pi if len(pi) == s else pi[rng.choice(len(pi), s, replace=False)]
        b = pj if len(pj) == s else pj[rng.choice(len(pj), s, replace=False)]
        _, pval = diptest.diptest(np.concatenate([a, b]))
        votes += int(pval >= alpha)
    return votes >= (L // 2 + 1)


def uniforce(X, Kp=None, M=25, alpha=0.001, L=11, random_state=0):
    """Kp=None applies the protocol adaptation K' = min(50, max(2, n // M)),
    which keeps the official K'=50 for n >= 1250 and scales it down where the
    official default would make the min_size elimination degenerate (measured:
    at n=800, K'=50 leaves 7 active subclusters and the reassignment step
    builds a 309-point subcluster straddling two true clusters).  This is
    generous to the baseline relative to the published default."""
    X = np.asarray(X, float)
    n = len(X)
    rng = np.random.default_rng(random_state)
    if Kp is None:
        Kp = min(50, max(2, n // M))
    Kp = min(Kp, max(1, n // 2))
    C, lab = global_kmeanspp(X, Kp, rng)
    sizes = np.bincount(lab, minlength=len(C))
    keep = np.where(sizes >= M)[0]
    if len(keep) < 2:
        # small-n fallback: rerun over-clustering at a feasible resolution
        Kp2 = max(2, n // (2 * M))
        C, lab = global_kmeanspp(X, Kp2, rng)
        sizes = np.bincount(lab, minlength=len(C))
        keep = np.where(sizes >= M)[0]
        if len(keep) < 2:
            keep = np.argsort(sizes)[-2:] if len(C) >= 2 else np.array([0])
    if len(keep) == 1:
        return 1, np.zeros(n, int)
    C = C[keep]
    lab = cdist(X, C).argmin(1)
    C = np.array([X[lab == j].mean(0) if (lab == j).any() else C[j]
                  for j in range(len(C))])
    lab = cdist(X, C).argmin(1)
    K = len(C)
    if K == 1:
        return 1, np.zeros(n, int)

    # Kruskal over centre-distance-sorted pairs with the unimodality test
    D = cdist(C, C)
    pairs = sorted(((a, b) for a in range(K) for b in range(a + 1, K)),
                   key=lambda e: D[e[0], e[1]])
    parent = list(range(K))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]; x = parent[x]
        return x

    members = {j: np.where(lab == j)[0] for j in range(K)}
    for a, b in pairs:
        ra, rb = find(a), find(b)
        if ra == rb:
            continue
        if unimodal_pair(X[members[a]], X[members[b]], C[a], C[b], rng,
                         alpha=alpha, L=L):
            parent[ra] = rb
    roots = [find(j) for j in range(K)]
    remap = {r: i for i, r in enumerate(sorted(set(roots)))}
    out = np.array([remap[roots[lab[i]]] for i in range(n)])
    return len(remap), out
