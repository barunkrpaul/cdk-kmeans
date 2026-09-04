"""Synthetic families and real benchmarks, following Section 16 of the report."""
from __future__ import annotations
import zlib
import numpy as np
from sklearn.datasets import (make_blobs, load_iris, load_wine,
                              load_breast_cancer, load_digits)
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


def _sep_blobs(n, k, d, sep, rng, scale=1.0):
    """k isotropic Gaussians whose centres are placed so that the minimum
    centre separation is `sep` standard deviations."""
    C = rng.standard_normal((k, d))
    # spread the centres out until the minimum pairwise distance is >= sep
    for _ in range(400):
        D = np.linalg.norm(C[:, None] - C[None], axis=2) + np.eye(k) * 1e9
        m = D.min()
        if m >= sep:
            break
        i, j = np.unravel_index(np.argmin(D), D.shape)
        v = C[i] - C[j]
        nv = np.linalg.norm(v)
        v = rng.standard_normal(d) if nv < 1e-9 else v / nv
        C[i] += 0.12 * sep * v
        C[j] -= 0.12 * sep * v
    sizes = np.full(k, n // k); sizes[: n - sizes.sum()] += 1
    X = np.vstack([C[j] + scale * rng.standard_normal((sizes[j], d))
                   for j in range(k)])
    y = np.repeat(np.arange(k), sizes)
    return X, y


def make(spec, rep):
    """Return (X, y, true_k) for a dataset specification and replicate index."""
    fam, name, par = spec
    # Python's built-in hash() is randomised per process for strings
    # (PYTHONHASHSEED), so it cannot be used to seed a reproducible generator.
    # zlib.crc32 is a fixed function of the bytes and gives the same dataset on
    # every machine and every run.
    rng = np.random.default_rng(zlib.crc32(f"{name}|{rep}".encode()))
    n = par.get("n", 600)

    if fam == "sep":
        k, d, s = par["k"], par["d"], par.get("sep", 8.0)
        X, y = _sep_blobs(n, k, d, s, rng)
    elif fam == "overlap":
        k, d, s = par["k"], par["d"], par["sep"]
        X, y = _sep_blobs(n, k, d, s, rng)
    elif fam == "unbalanced":
        k, d, r = par["k"], par["d"], par["ratio"]
        w = np.geomspace(1.0, 1.0 / r, k); w /= w.sum()
        sizes = np.maximum((w * n).astype(int), 25)
        Xc, _ = _sep_blobs(k * 10, k, d, 9.0, rng)
        C = np.array([Xc[i * 10:(i + 1) * 10].mean(0) for i in range(k)])
        X = np.vstack([C[j] + rng.standard_normal((sizes[j], d))
                       for j in range(k)])
        y = np.repeat(np.arange(k), sizes)
    elif fam == "unequal_var":
        k, d, r = par["k"], par["d"], par["ratio"]
        sc = np.geomspace(1.0, r, k)
        Xc, _ = _sep_blobs(k * 10, k, d, 12.0, rng)
        C = np.array([Xc[i * 10:(i + 1) * 10].mean(0) for i in range(k)])
        sizes = np.full(k, n // k); sizes[: n - sizes.sum()] += 1
        X = np.vstack([C[j] + sc[j] * rng.standard_normal((sizes[j], d))
                       for j in range(k)])
        y = np.repeat(np.arange(k), sizes)
    elif fam == "elongated":
        k, d, cond = par["k"], par["d"], par["cond"]
        Xc, _ = _sep_blobs(k * 10, k, d, 10.0, rng)
        C = np.array([Xc[i * 10:(i + 1) * 10].mean(0) for i in range(k)])
        sizes = np.full(k, n // k); sizes[: n - sizes.sum()] += 1
        Xs = []
        for j in range(k):
            A = rng.standard_normal((d, d)); Q, _ = np.linalg.qr(A)
            s = np.geomspace(cond, 1.0, d)
            Xs.append(C[j] + (rng.standard_normal((sizes[j], d)) * s) @ Q.T)
        X = np.vstack(Xs); y = np.repeat(np.arange(k), sizes)
    elif fam == "nongauss":
        k, d, kind = par["k"], par["d"], par["kind"]
        Xc, _ = _sep_blobs(k * 10, k, d, 11.0, rng)
        C = np.array([Xc[i * 10:(i + 1) * 10].mean(0) for i in range(k)])
        sizes = np.full(k, n // k); sizes[: n - sizes.sum()] += 1
        Xs = []
        for j in range(k):
            if kind == "t3":
                e = rng.standard_t(3, (sizes[j], d)) / np.sqrt(3.0)
            elif kind == "uniform":
                e = rng.uniform(-1.7, 1.7, (sizes[j], d))
            elif kind == "mixed":
                r = j % 3
                e = (rng.standard_normal((sizes[j], d)) if r == 0 else
                     rng.standard_t(3, (sizes[j], d)) / np.sqrt(3.0) if r == 1
                     else rng.uniform(-1.7, 1.7, (sizes[j], d)))
            elif kind == "skew":
                e = rng.exponential(1.0, (sizes[j], d)) - 1.0
            Xs.append(C[j] + e)
        X = np.vstack(Xs); y = np.repeat(np.arange(k), sizes)
    elif fam == "noise":
        k, d, f = par["k"], par["d"], par["frac"]
        ns = int(n * (1 - f))
        X0, y0 = _sep_blobs(ns, k, d, 9.0, rng)
        nn = n - ns
        lo, hi = X0.min(0) - 1, X0.max(0) + 1
        Xn = rng.uniform(lo, hi, (nn, d))
        X = np.vstack([X0, Xn]); y = np.concatenate([y0, np.full(nn, -1)])
    elif fam == "nested":
        d = par["d"]
        Xc, _ = _sep_blobs(30, 3, d, 26.0, rng)
        C = np.array([Xc[i * 10:(i + 1) * 10].mean(0) for i in range(3)])
        Xs, ys, t = [], [], 0
        for j in range(3):
            sub, sl = _sep_blobs(n // 3, 2, d, 5.0, rng)
            Xs.append(sub + C[j])
            ys.append(np.asarray(sl, int) + t)
            t += 2
        X = np.vstack(Xs); y = np.concatenate(ys)
    elif fam == "null":
        d, kind = par["d"], par["kind"]
        if kind == "gaussian":
            X = rng.standard_normal((n, d))
        elif kind == "uniform":
            X = rng.uniform(-1, 1, (n, d))
        elif kind == "t3":
            X = rng.standard_t(3, (n, d))
        elif kind == "aniso":
            A = rng.standard_normal((d, d)); Q, _ = np.linalg.qr(A)
            X = (rng.standard_normal((n, d)) * np.geomspace(6, 1, d)) @ Q.T
        y = np.zeros(n, int)
    else:
        raise ValueError(fam)

    true_k = 1 if fam == "null" else int(len(np.unique(y[y >= 0])))
    return np.asarray(X, float), np.asarray(y), true_k


# --------------------------------------------------------------- the suite ---
SPECS = []
A = SPECS.append
# 1 well separated
for k in [2, 3, 5, 10, 20]:
    for d in [2, 10, 50]:
        A(("sep", f"sep_k{k}_d{d}", dict(k=k, d=d, sep=8.0, n=800)))
# 2 overlapping (separation swept)
for s in [6.0, 4.0, 3.0, 2.0]:
    A(("overlap", f"overlap_s{s}", dict(k=5, d=2, sep=s, n=800)))
# 3 unbalanced
for r in [5, 20, 100]:
    A(("unbalanced", f"unbal_r{r}", dict(k=5, d=2, ratio=r, n=900)))
# 4 unequal variance
for r in [3, 10]:
    A(("unequal_var", f"uvar_r{r}", dict(k=4, d=2, ratio=r, n=800)))
# 5 elongated
for c in [5, 20]:
    A(("elongated", f"elong_c{c}", dict(k=4, d=2, cond=c, n=800)))
# 6 non-Gaussian components
for kind in ["t3", "uniform", "mixed", "skew"]:
    A(("nongauss", f"nongauss_{kind}", dict(k=5, d=2, kind=kind, n=800)))
A(("nongauss", "nongauss_t3_d10", dict(k=5, d=10, kind="t3", n=800)))
# 7 noise
for f in [0.1, 0.3, 0.5]:
    A(("noise", f"noise_{int(f*100)}", dict(k=4, d=2, frac=f, n=900)))
# 8 nested / hierarchical
A(("nested", "nested_d2", dict(d=2, n=900)))
# 9 null data  (true k = 1)
for kind in ["gaussian", "uniform", "t3", "aniso"]:
    for d in [2, 10]:
        A(("null", f"null_{kind}_d{d}", dict(d=d, kind=kind, n=600)))


def real_datasets():
    """Real benchmark data.

    Two groups. (a) LABELLED datasets, where the class count is the
    conventional -- and, as Iris shows, not always the geometric -- target.
    (b) SINGLE-CLASS extracts, where one labelled class is used on its own.
    These are the closest thing the standard repositories offer to a real
    k*=1 benchmark: genuine high-dimensional, non-Gaussian, correlated data
    that is known to have been drawn from one labelled population.  They are
    NOT a guarantee of unimodality -- a labelled class can carry real
    substructure (handwriting styles within one digit) -- so they are reported
    separately and read as a directional signal, not as ground truth.
    """
    out = []
    for nm, ld in [("Iris", load_iris), ("Wine", load_wine),
                   ("Breast cancer", load_breast_cancer)]:
        D = ld(); X = StandardScaler().fit_transform(D.data)
        out.append((nm, X, D.target, len(np.unique(D.target))))

    D = load_digits()
    Xd = StandardScaler().fit_transform(D.data)
    yd = D.target
    out.append(("Digits", Xd, yd, 10))
    rng = np.random.default_rng(0)
    idx = rng.choice(len(Xd), 700, replace=False)
    out.append(("Digits-700", Xd[idx], yd[idx], 10))

    # subsets: an easier 4-class problem and a 5-class problem
    for tag, cls in [("Digits-0168", (0, 1, 6, 8)), ("Digits-0to4", (0, 1, 2, 3, 4))]:
        m = np.isin(yd, cls)
        out.append((tag, Xd[m], yd[m], len(cls)))

    # standard PCA preprocessing, 10 components (retains ~59% variance here)
    P = PCA(n_components=10, random_state=0).fit_transform(Xd)
    out.append(("Digits-PCA10", StandardScaler().fit_transform(P), yd, 10))

    # ---- single-class extracts: real data with one labelled population -----
    Db = load_breast_cancer()
    Xb = StandardScaler().fit_transform(Db.data)
    out.append(("BC-benign(1cls)", Xb[Db.target == 1], np.zeros((Db.target == 1).sum(), int), 1))
    out.append(("BC-malignant(1cls)", Xb[Db.target == 0], np.zeros((Db.target == 0).sum(), int), 1))
    for c in (1, 3, 7):
        m = yd == c
        out.append((f"Digit-{c}(1cls)", Xd[m], np.zeros(m.sum(), int), 1))
    Dw = load_wine(); Xw = StandardScaler().fit_transform(Dw.data)
    m = Dw.target == 1
    out.append(("Wine-c1(1cls)", Xw[m], np.zeros(m.sum(), int), 1))
    Di = load_iris(); Xi = StandardScaler().fit_transform(Di.data)
    m = Di.target == 0
    out.append(("Iris-setosa(1cls)", Xi[m], np.zeros(m.sum(), int), 1))
    return out
