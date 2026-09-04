"""
Convex-hull reference for the CDK merge test.

Measured defect (see propositions_2ec.tex, calibration section): in d=2 the
per-edge rejection rate on structureless data is 0.32-0.48 at nominal 0.01,
for every null kind including the matched uniform one.  The union of two
adjacent cells of a two-means partition is a polygonal region, not its
principal-axis bounding box; a reference drawn uniform over the box has mass
in corners the observed union cannot occupy, which inflates the observed
statistic relative to the reference distribution.  In d=10 the effect is
absent at the measured sample sizes.

Repair: draw the reference uniform over the CONVEX HULL of the observed
union.  The hull is the tightest convex support estimate available without a
tuning parameter, and sampling is exact via the Delaunay triangulation: the
simplices tile the hull, a simplex is chosen with probability proportional
to its volume, and a point is drawn uniformly inside it with Dirichlet(1)
barycentric weights.  Above HULL_DMAX dimensions (default 6) the box
reference is kept: qhull's cost grows exponentially with d, and the box is
measured to be near-calibrated there.

The mechanism-matching principle of the paper is preserved: observed and
reference draws are bisected by the same randomised two-means and the
statistic is the same; only the null support changes.
"""
from __future__ import annotations
import numpy as np
from scipy.spatial import Delaunay
from scipy.special import gammaln
import cdk as _cdk
from cdk_geom import CDKGeo2

HULL_DMAX = 6

_orig_make_reference = _cdk.make_reference


def _hull_sampler(X):
    """Return a draw(rng)->sample closure, uniform over conv(X), or None."""
    n, d = X.shape
    if d > HULL_DMAX or n <= d + 1:
        return None
    try:
        tri = Delaunay(X)
    except Exception:
        return None
    S = X[tri.simplices]                       # (ns, d+1, d)
    V = S[:, 1:, :] - S[:, :1, :]              # (ns, d, d)
    vol = np.abs(np.linalg.det(V))             # ~ d! * volume
    tot = vol.sum()
    if not np.isfinite(tot) or tot <= 0:
        return None
    prob = vol / tot

    def draw(rng):
        idx = rng.choice(len(S), size=n, p=prob)
        w = rng.dirichlet(np.ones(S.shape[1]), size=n)   # (n, d+1)
        return np.einsum('ij,ijk->ik', w, S[idx])
    return draw


def make_reference(X, kind):
    if kind == "hull":
        drawer = _hull_sampler(np.asarray(X, float))
        if drawer is not None:
            return drawer
        return _orig_make_reference(X, "uniform")   # fallback: box
    return _orig_make_reference(X, kind)


# route the patched reference through the cdk module so _certify picks it up
_cdk.make_reference = make_reference


class CDKHull(CDKGeo2):
    """knn+2ec family with the convex-hull reference (null='hull')."""

    def __init__(self, **kw):
        kw.setdefault("family", "knn")
        kw.setdefault("null", "hull")
        super().__init__(**kw)
