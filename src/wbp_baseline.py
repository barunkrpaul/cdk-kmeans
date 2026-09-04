"""
Wu, Bien & Panigrahi, "Hierarchical Clustering With Confidence"
(arXiv:2512.06522) as a k-estimation baseline, wrapping the authors' own
implementation (github.com/judywu4800/SI_HierarchicalClustering, src/).

The method randomises the agglomerative merge sequence with a softmax over
linkage distances (temperature tau* times the mean pairwise linkage), computes
a selective p-value at each dendrogram node by numerical integration of the
F-statistic density weighted by the merge-path probabilities, and estimates K
with an adaptive alpha-spending walk over the merge sequence, with a
probabilistic guarantee against overestimation.

Protocol adaptations, documented rather than hidden:
- The authors' code costs ~1.5 min at n=60 and grows superlinearly, so the
  baseline is run on a uniform random subsample of at most n_cap points
  (default 80) and the estimated K is read from the subsample.  Labels for
  the full dataset are then obtained by k-means at that K, so external scores
  remain comparable.
- Defaults follow the paper: tau*=0.1, alpha=0.05 total, exponentially
  decaying spending sequence, complete linkage.
"""
from __future__ import annotations
import sys, os
import numpy as np

_SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "wbp_repo", "src")


def wbp_estimate(X, rep=0, n_cap=80, tau=0.1, alpha=0.05, linkage="complete",
                 ngrid=800, ncoarse=20, grid_width=300):
    if _SRC not in sys.path:
        sys.path.insert(0, _SRC)
    from find_best_K import find_best_K_F, generate_alpha_list_exp

    X = np.asarray(X, float)
    n = len(X)
    rng = np.random.default_rng(rep)
    if n > n_cap:
        idx = rng.choice(n, n_cap, replace=False)
        Xs = X[idx]
    else:
        Xs = X
    ns = len(Xs)
    al = generate_alpha_list_exp(n=ns, total_alpha=alpha)
    K_hat, pv, aseq, lab_s = find_best_K_F(
        Xs, tau=tau, alpha_list=al, linkage=linkage, total_alpha=alpha,
        seed=int(rng.integers(1e9)), method="interpolation",
        ngrid=ngrid, ncoarse=ncoarse, grid_width=grid_width)
    K_hat = int(K_hat)
    # labels for the full dataset at the estimated K
    if K_hat <= 1:
        return 1, np.zeros(n, int)
    from sklearn.cluster import KMeans
    km = KMeans(n_clusters=K_hat, n_init=10, random_state=rep).fit(X)
    return K_hat, km.labels_
