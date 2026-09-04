"""
Geometric candidate families for CDK.

The default CDK candidate family joins each leaf to its n_nb nearest leaves.
That rule carries a free parameter (n_nb) and gives no structural guarantee:
a k-nearest-neighbour graph on the leaf centroids can be DISCONNECTED, in
which case two groups of leaves are declared separate populations without any
edge between them ever being tested.  The separation is then asserted by the
family construction rather than certified by a rejected test, which is exactly
the kind of untested assertion the method exists to remove.

This module replaces the family with classical proximity graphs from
computational geometry, computed on the leaf centroids:

  gabriel   (a,b) is an edge iff no centroid c lies in the open ball with
            diameter [a,b]:   d(a,c)^2 + d(c,b)^2 < d(a,b)^2  for no c.
  rng       relative neighbourhood graph: (a,b) is an edge iff no centroid c
            is closer to both endpoints than they are to each other:
            max(d(a,c), d(c,b)) < d(a,b) for no c.

Both are parameter-free.  Both satisfy, for points in general position,

        MST  <=  RNG  <=  Gabriel  <=  Delaunay,

so both contain the Euclidean minimum spanning tree of the centroids and are
therefore CONNECTED.  The consequence for CDK is structural: every separation
in the output corresponds to at least one candidate edge that was tested and
rejected.  With the k-NN family that statement is false.

Both graphs are computed from pairwise distances alone (O(m^3) with m leaves,
m <= 64 here), so unlike the Delaunay triangulation itself they carry no
curse-of-dimensionality construction cost and are well defined at any d.
"""
from __future__ import annotations
import numpy as np
from scipy.spatial.distance import cdist
from cdk import CDK


def gabriel_edges(C):
    """Edge list of the Gabriel graph of the rows of C (squared-distance test)."""
    m = len(C)
    D2 = cdist(C, C, metric="sqeuclidean")
    E = []
    for a in range(m):
        for b in range(a + 1, m):
            ok = True
            for c in range(m):
                if c != a and c != b and D2[a, c] + D2[c, b] < D2[a, b] - 1e-12:
                    ok = False
                    break
            if ok:
                E.append((a, b))
    return E


def rng_edges(C):
    """Edge list of the relative neighbourhood graph of the rows of C."""
    m = len(C)
    D = cdist(C, C)
    E = []
    for a in range(m):
        for b in range(a + 1, m):
            ok = True
            for c in range(m):
                if c != a and c != b and max(D[a, c], D[c, b]) < D[a, b] - 1e-12:
                    ok = False
                    break
            if ok:
                E.append((a, b))
    return E


class CDKGeo(CDK):
    """CDK with a parameter-free geometric candidate family.

    family='gabriel' | 'rng' | 'knn' (knn reproduces the parent class exactly).
    """

    def __init__(self, family="gabriel", **kw):
        super().__init__(**kw)
        self.family = family

    def _edges(self, X, leaves, rng, n_nb):
        if self.family == "knn":
            return super()._edges(X, leaves, rng, n_nb)
        C = np.array([X[l].mean(0) for l in leaves])
        if self.family == "gabriel":
            E = gabriel_edges(C)
        elif self.family == "rng":
            E = rng_edges(C)
        else:
            raise ValueError(self.family)
        D = cdist(C, C)
        return sorted(E, key=lambda e: D[e[0], e[1]])


# ---------------------------------------------------------------------------
# Bridge-augmented Gabriel family.
#
# In a sparse candidate family a BRIDGE edge is a single point of failure: one
# false rejection disconnects the surviving graph and fabricates a component,
# so the per-edge type-I error propagates to k at first order.  Augmenting the
# graph until it is 2-edge-connected means every spurious separation requires
# at least two simultaneous false rejections across the same cut (second-order
# in the per-edge level).  The augmentation rule is deterministic and
# parameter-free: while a bridge exists, add the shortest centroid pair that
# crosses the corresponding cut and is not yet an edge.
# ---------------------------------------------------------------------------
def _bridges(m, edges):
    """Tarjan bridge finding; returns the set of bridge edges."""
    adj = [[] for _ in range(m)]
    for i, (a, b) in enumerate(edges):
        adj[a].append((b, i)); adj[b].append((a, i))
    disc = [-1] * m; low = [0] * m; out = set(); t = [0]

    def dfs(root):
        stack = [(root, -1, iter(adj[root]))]
        disc[root] = low[root] = t[0]; t[0] += 1
        while stack:
            u, pe, it = stack[-1]
            adv = False
            for v, ei in it:
                if ei == pe:
                    continue
                if disc[v] == -1:
                    disc[v] = low[v] = t[0]; t[0] += 1
                    stack.append((v, ei, iter(adj[v])))
                    adv = True
                    break
                low[u] = min(low[u], disc[v])
            if not adv:
                stack.pop()
                if stack:
                    pu = stack[-1][0]
                    low[pu] = min(low[pu], low[u])
                    if low[u] > disc[pu]:
                        a, b = edges[pe]
                        out.add((min(a, b), max(a, b)))
    for r in range(m):
        if disc[r] == -1:
            dfs(r)
    return out


def two_edge_connect(C, edges):
    """Augment edge list until 2-edge-connected (or no addable pair remains)."""
    from scipy.spatial.distance import cdist as _cdist
    m = len(C)
    if m <= 2:
        return edges
    D = _cdist(C, C)
    E = set(edges)
    for _ in range(m * m):
        br = _bridges(m, sorted(E))
        if not br:
            break
        a, b = min(br, key=lambda e: D[e[0], e[1]])
        # component split induced by removing this bridge
        E2 = set(E) - {(a, b)}
        seen = {a}; stack = [a]
        adj = {i: [] for i in range(m)}
        for x, y in E2:
            adj[x].append(y); adj[y].append(x)
        while stack:
            u = stack.pop()
            for v in adj[u]:
                if v not in seen:
                    seen.add(v); stack.append(v)
        side = np.array([i in seen for i in range(m)])
        best, bd = None, np.inf
        for i in range(m):
            for j in range(m):
                if side[i] and not side[j]:
                    e = (min(i, j), max(i, j))
                    if e not in E and D[i, j] < bd:
                        best, bd = e, D[i, j]
        if best is None:
            break
        E.add(best)
    return sorted(E)


class CDKGeo2(CDKGeo):
    """Gabriel (or RNG) family augmented to 2-edge-connectivity."""

    def _edges(self, X, leaves, rng, n_nb):
        E = super()._edges(X, leaves, rng, n_nb)
        C = np.array([X[l].mean(0) for l in leaves])
        from scipy.spatial.distance import cdist as _cdist
        D = _cdist(C, C)
        return sorted(two_edge_connect(C, E), key=lambda e: D[e[0], e[1]])
