"""Baseline estimators of the number of clusters.

Every estimator returns (k_hat, labels).  Range-based selectors receive the SAME
candidate range, fixed a priori and never tuned per method, following the
convention of Yerbury et al. (2024): [max(2,k*-5), max(12,k*+5)].
"""
from __future__ import annotations
import numpy as np, warnings
from sklearn.cluster import KMeans, BisectingKMeans, HDBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.metrics import (silhouette_score, calinski_harabasz_score,
                             davies_bouldin_score)
from scipy.spatial.distance import pdist, cdist
from scipy.stats import anderson
import diptest

warnings.filterwarnings("ignore")
NI = 10                      # restarts for every k-means call in the baselines


def _km(X, k, seed=0, n_init=NI):
    return KMeans(n_clusters=k, n_init=n_init, random_state=seed).fit(X)


# ----------------------------------------------------------- index sweeps ----
def _sweep(X, ks, score, seed, sign):
    best, bk, bl = -np.inf, ks[0], None
    for k in ks:
        if k >= len(X):
            continue
        km = _km(X, k, seed)
        if len(np.unique(km.labels_)) < 2:
            continue
        s = sign * score(X, km.labels_)
        if s > best:
            best, bk, bl = s, k, km.labels_
    return bk, bl


def silhouette(X, ks, seed=0):
    return _sweep(X, ks, silhouette_score, seed, +1)

def calinski(X, ks, seed=0):
    return _sweep(X, ks, calinski_harabasz_score, seed, +1)

def daviesbouldin(X, ks, seed=0):
    return _sweep(X, ks, davies_bouldin_score, seed, -1)


# ------------------------------------------------------------ elbow / jump ---
def _inertias(X, ks, seed):
    out = {}
    for k in ks:
        if k < len(X):
            out[k] = _km(X, k, seed).inertia_
    return out


def elbow(X, ks, seed=0):
    """Kneedle: maximum distance from the chord joining the endpoints."""
    ks = [k for k in ks if k < len(X)]
    W = np.array([_km(X, k, seed).inertia_ for k in ks])
    x = np.array(ks, float)
    xn = (x - x.min()) / max(np.ptp(x), 1e-12)
    yn = (W - W.min()) / max(np.ptp(W), 1e-12)
    d = np.abs(yn - (1 - xn))       # distance from the descending chord
    k = int(ks[int(np.argmax(d))])
    return k, _km(X, k, seed).labels_


def jump(X, ks, seed=0):
    """Sugar & James (2003) transformed-distortion jump, Y = d/2."""
    d = X.shape[1]
    Y = d / 2.0
    ks_full = [1] + [k for k in ks if k < len(X)]
    dist = {}
    for k in ks_full:
        km = _km(X, k, seed) if k > 1 else None
        w = ((X - X.mean(0)) ** 2).sum() if k == 1 else km.inertia_
        dist[k] = (w / (len(X) * d)) ** (-Y)
    js = {k: dist[k] - dist[kp] for kp, k in zip(ks_full[:-1], ks_full[1:])}
    k = int(max(js, key=js.get))
    return k, _km(X, k, seed).labels_


# ------------------------------------------------------------ gap statistic --
def gap(X, ks, seed=0, B=20):
    rng = np.random.default_rng(seed)
    mu = X.mean(0); Xc = X - mu
    _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
    Z = Xc @ Vt.T
    lo, hi = Z.min(0), Z.max(0)
    ks_full = [k for k in ks if k < len(X)]
    logW, Estar, sk = {}, {}, {}
    for k in ks_full:
        logW[k] = np.log(max(_km(X, k, seed, 5).inertia_, 1e-300))
        vals = []
        for b in range(B):
            Zr = rng.uniform(lo, hi, size=Z.shape) @ Vt + mu
            vals.append(np.log(max(_km(Zr, k, seed + b, 1).inertia_, 1e-300)))
        vals = np.array(vals)
        Estar[k] = vals.mean()
        sk[k] = vals.std() * np.sqrt(1 + 1.0 / B)
    G = {k: Estar[k] - logW[k] for k in ks_full}
    chosen = ks_full[-1]
    for i, k in enumerate(ks_full[:-1]):
        kn = ks_full[i + 1]
        if G[k] >= G[kn] - sk[kn]:
            chosen = k
            break
    return chosen, _km(X, chosen, seed).labels_


# ---------------------------------------------------- split-based estimators --
def _bic_spherical(X, labels, k):
    n, d = X.shape
    cent = np.array([X[labels == j].mean(0) for j in range(k)])
    ss = sum(((X[labels == j] - cent[j]) ** 2).sum() for j in range(k))
    if n - k <= 0:
        return -np.inf
    var = ss / (n - k)
    if var <= 0:
        return np.inf
    ll = 0.0
    for j in range(k):
        nj = (labels == j).sum()
        if nj == 0:
            continue
        ll += (-nj / 2 * np.log(2 * np.pi) - nj * d / 2 * np.log(var)
               - (nj - k) / 2 + nj * np.log(nj) - nj * np.log(n))
    p = (k - 1) + d * k + 1
    return ll - p / 2 * np.log(n)


def xmeans(X, ks, seed=0):
    """Pelleg & Moore (2000): split a centroid when the local BIC prefers it."""
    kmin, kmax = min(ks), max(ks)
    km = _km(X, kmin, seed); labels = km.labels_; k = kmin
    changed = True
    while changed and k < kmax:
        changed = False
        for j in range(k):
            idx = np.where(labels == j)[0]
            if len(idx) < 4 or k >= kmax:
                continue
            sub = X[idx]
            b_par = _bic_spherical(sub, np.zeros(len(sub), int), 1)
            km2 = _km(sub, 2, seed, 3)
            if len(np.unique(km2.labels_)) < 2:
                continue
            b_ch = _bic_spherical(sub, km2.labels_, 2)
            if b_ch > b_par:
                labels[idx[km2.labels_ == 1]] = k
                k += 1
                changed = True
    km = _km(X, k, seed)
    return k, km.labels_


def gmeans(X, ks, seed=0, alpha_crit=1.8692, kmax=None):
    """Hamerly & Elkan (2003): Anderson-Darling on the bisector projection."""
    kmax = kmax or max(ks)
    rng = np.random.default_rng(seed)
    centers = [X.mean(0)]
    for _ in range(30):
        k = len(centers)
        km = KMeans(n_clusters=k, init=np.array(centers), n_init=1,
                    random_state=seed).fit(X)
        labels, centers = km.labels_, list(km.cluster_centers_)
        new = []
        for j in range(k):
            sub = X[labels == j]
            if len(sub) < 8 or len(centers) + len(new) >= kmax:
                new.append(centers[j]); continue
            km2 = _km(sub, 2, seed, 3)
            if len(np.unique(km2.labels_)) < 2:
                new.append(centers[j]); continue
            v = km2.cluster_centers_[0] - km2.cluster_centers_[1]
            nv = np.linalg.norm(v)
            if nv < 1e-12:
                new.append(centers[j]); continue
            z = sub @ v / nv
            z = (z - z.mean()) / (z.std() + 1e-12)
            try:
                A2 = anderson(z, "norm").statistic
            except Exception:
                A2 = 0.0
            n = len(z)
            A2s = A2 * (1 + 4.0 / n - 25.0 / n ** 2)
            if A2s > alpha_crit:
                new.extend(list(km2.cluster_centers_))
            else:
                new.append(centers[j])
        if len(new) == len(centers):
            break
        centers = new
    k = len(centers)
    km = KMeans(n_clusters=k, init=np.array(centers), n_init=1,
                random_state=seed).fit(X)
    return k, km.labels_


def dipmeans(X, ks, seed=0, alpha=0.0, vthd=0.01, kmax=None, cap=400):
    """Kalogeratos & Likas (2012): dip-dist viewer voting."""
    kmax = kmax or max(ks)
    rng = np.random.default_rng(seed)
    labels = np.zeros(len(X), int); k = 1
    while k < kmax:
        best_score, best_j = 0.0, -1
        for j in range(k):
            idx = np.where(labels == j)[0]
            if len(idx) < 20:
                continue
            sub = X[idx]
            view = idx if len(idx) <= cap else rng.choice(idx, cap, replace=False)
            D = cdist(X[view], sub)
            dips, votes = [], 0
            for row in D:
                dv, pv = diptest.diptest(np.sort(row))
                if pv <= max(alpha, 1e-9):
                    votes += 1; dips.append(dv)
            if votes / len(view) >= vthd and dips:
                s = float(np.mean(dips))
                if s > best_score:
                    best_score, best_j = s, j
        if best_j < 0:
            break
        idx = np.where(labels == best_j)[0]
        km2 = _km(X[idx], 2, seed, 3)
        if len(np.unique(km2.labels_)) < 2:
            break
        labels[idx[km2.labels_ == 1]] = k
        k += 1
        labels = _km(X, k, seed, 3).labels_ if k > 1 else labels
    return k, labels


def isodata(X, ks, seed=0, n_min=None, sigma_max=None, L_min=None, I_max=12):
    """Ball & Hall (1965) with the conventional threshold defaults."""
    n, d = X.shape
    n_min = n_min or max(5, n // (4 * max(ks)))
    k_init = max(2, (min(ks) + max(ks)) // 2)
    gsd = X.std(0).mean()
    sigma_max = sigma_max or 0.7 * gsd
    L_min = L_min or 0.5 * gsd
    km = _km(X, k_init, seed, 3); C = list(km.cluster_centers_)
    for _ in range(I_max):
        km = KMeans(n_clusters=len(C), init=np.array(C), n_init=1,
                    random_state=seed).fit(X)
        lab, C = km.labels_, list(km.cluster_centers_)
        keep = [j for j in range(len(C)) if (lab == j).sum() >= n_min]
        if not keep:
            break
        C = [C[j] for j in keep]
        km = KMeans(n_clusters=len(C), init=np.array(C), n_init=1,
                    random_state=seed).fit(X)
        lab, C = km.labels_, list(km.cluster_centers_)
        newC = []
        for j in range(len(C)):
            sub = X[lab == j]
            if len(sub) > 2 * (n_min + 1) and len(sub) > 2 and \
               sub.std(0).max() > sigma_max and len(C) + len(newC) < max(ks):
                ax = int(np.argmax(sub.std(0)))
                off = np.zeros(d); off[ax] = 0.5 * sub.std(0)[ax]
                newC += [C[j] + off, C[j] - off]
            else:
                newC.append(C[j])
        C = newC
        merged, used = [], set()
        for a in range(len(C)):
            if a in used:
                continue
            grp = [C[a]]
            for b in range(a + 1, len(C)):
                if b not in used and np.linalg.norm(C[a] - C[b]) <= L_min:
                    grp.append(C[b]); used.add(b)
            merged.append(np.mean(grp, 0)); used.add(a)
        C = merged
        if len(C) < 1:
            break
    k = max(1, len(C))
    km = KMeans(n_clusters=k, init=np.array(C), n_init=1, random_state=seed).fit(X)
    return k, km.labels_


def mdl_kmeans(X, ks, seed=0, kmax=None):
    """A description-length split/merge procedure in the style of k*-means
    (Mahon & Lapata 2025): split if the residual reduction exceeds 2N/(k+1),
    merge the closest pair if half the reduction is below N/k.  No candidate
    range, no significance level."""
    kmax = kmax or max(ks) + 5
    N = len(X)
    labels = np.zeros(N, int); k = 1
    for _ in range(60):
        moved = False
        # ---- split
        gains = []
        for j in range(k):
            idx = np.where(labels == j)[0]
            if len(idx) < 4:
                gains.append((-np.inf, j, None)); continue
            sub = X[idx]
            Q = ((sub - sub.mean(0)) ** 2).sum()
            km2 = _km(sub, 2, seed, 3)
            gains.append((Q - km2.inertia_, j, (idx, km2.labels_)))
        gains.sort(key=lambda t: -t[0])
        if gains and gains[0][0] > 2.0 * N / (k + 1) and k < kmax:
            _, j, pack = gains[0]
            idx, l2 = pack
            labels[idx[l2 == 1]] = k
            k += 1; moved = True
        # ---- merge closest pair
        if k > 1:
            C = np.array([X[labels == j].mean(0) for j in range(k)])
            D = cdist(C, C) + np.eye(k) * 1e18
            a, b = np.unravel_index(np.argmin(D), D.shape)
            idx = np.where((labels == a) | (labels == b))[0]
            sub = X[idx]
            Q = ((sub - sub.mean(0)) ** 2).sum()
            Qa = ((X[labels == a] - X[labels == a].mean(0)) ** 2).sum()
            Qb = ((X[labels == b] - X[labels == b].mean(0)) ** 2).sum()
            if 0.5 * (Q - (Qa + Qb)) - N / k < 0:
                labels[labels == b] = a
                _, labels = np.unique(labels, return_inverse=True)
                k = labels.max() + 1; moved = True
        if not moved:
            break
        labels = _km(X, k, seed, 3).labels_
    return int(k), labels


def gmm_ic(X, ks, seed=0, crit="bic"):
    best, bk, bl = np.inf, min(ks), None
    for k in ks:
        if k >= len(X):
            continue
        try:
            g = GaussianMixture(k, covariance_type="full", n_init=2,
                                random_state=seed, reg_covar=1e-4).fit(X)
        except Exception:
            continue
        lab = g.predict(X)
        val = g.bic(X)
        if crit == "icl":
            P = np.clip(g.predict_proba(X), 1e-12, 1)
            val = val + 2.0 * (-(P * np.log(P)).sum())
        if val < best:
            best, bk, bl = val, k, lab
    if bl is None:
        bl = _km(X, bk, seed).labels_
    return bk, bl


def hdbscan_est(X, ks, seed=0, min_cluster_size=None):
    m = min_cluster_size or max(5, len(X) // (4 * max(ks)))
    h = HDBSCAN(min_cluster_size=m).fit(X)
    lab = h.labels_
    k = len(set(lab[lab >= 0]))
    if k == 0:
        return 1, np.zeros(len(X), int)
    out = lab.copy()
    if (lab == -1).any():          # attach noise to the nearest cluster centre
        C = np.array([X[lab == j].mean(0) for j in range(k)])
        out[lab == -1] = np.argmin(cdist(X[lab == -1], C), 1)
    _, out = np.unique(out, return_inverse=True)
    return k, out


def bisecting_sil(X, ks, seed=0):
    best, bk, bl = -np.inf, min(ks), None
    for k in ks:
        if k >= len(X):
            continue
        lab = BisectingKMeans(n_clusters=k, random_state=seed,
                              n_init=3).fit_predict(X)
        if len(np.unique(lab)) < 2:
            continue
        s = silhouette_score(X, lab)
        if s > best:
            best, bk, bl = s, k, lab
    return bk, bl


REGISTRY = {
    "Silhouette":      silhouette,
    "Calinski-Har.":   calinski,
    "Davies-Bouldin":  daviesbouldin,
    "Elbow (Kneedle)": elbow,
    "Jump statistic":  jump,
    "Gap statistic":   gap,
    "X-means":         xmeans,
    "G-means":         gmeans,
    "dip-means":       dipmeans,
    "ISODATA":         isodata,
    "MDL split/merge": mdl_kmeans,
    "GMM+BIC":         lambda X, ks, seed=0: gmm_ic(X, ks, seed, "bic"),
    "GMM+ICL":         lambda X, ks, seed=0: gmm_ic(X, ks, seed, "icl"),
    "HDBSCAN":         hdbscan_est,
    "Bisecting+Sil.":  bisecting_sil,
}
