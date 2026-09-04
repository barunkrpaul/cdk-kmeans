"""Compare CDK candidate-family constructions: knn (default), gabriel, rng.

Records k_hat, family size m, whether the candidate graph is connected,
number of tests, and runtime, over null and structured specs.
"""
import sys, os, time, warnings, itertools
warnings.filterwarnings("ignore")
sys.path.insert(0, "cdk/src")
import numpy as np, pandas as pd
from multiprocessing import Pool
from datasets import SPECS, make
from cdk_geom import CDKGeo
from cdk import CDK

NULL_SPECS = [s for s in SPECS if s[0] == "null"]
STRUCT = ["sep_k2_d2", "sep_k5_d10", "sep_k10_d50", "sep_k20_d10",
          "overlap_s4.0", "unbal_r20", "elong_c5", "nested_d2",
          "nongauss_t3", "noise_30"]
STRUCT_SPECS = [s for s in SPECS if s[1] in STRUCT]
REPS_NULL = 5
REPS_STRUCT = 3

def connected(m, edges):
    parent = list(range(m))
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]; x = parent[x]
        return x
    for a, b in edges:
        ra, rb = find(a), find(b)
        if ra != rb: parent[ra] = rb
    return len({find(i) for i in range(m)}) == 1

def run(job):
    spec, rep, fam = job
    X, y, tk = make(spec, rep)
    est = CDKGeo(family=fam, random_state=rep)
    # capture the candidate family
    rng = np.random.default_rng(rep)
    t0 = time.time()
    r = est.fit(X)
    el = time.time() - t0
    # rebuild leaves+edges deterministically for connectivity bookkeeping
    rng2 = np.random.default_rng(rep)
    leaves = est._leaves(X, rng2)
    edges = est._edges(X, leaves, rng2, est.n_nb) if len(leaves) > 1 else []
    return dict(dataset=spec[1], family=spec[0], rep=rep, fam=fam,
                true_k=tk, k_hat=r.k, k_init=r.k_init, m_edges=len(edges),
                cand_connected=connected(len(leaves), edges) if leaves else True,
                n_tests=r.n_tests, secs=round(el, 2))

jobs = [(s, r, f) for s in NULL_SPECS for r in range(REPS_NULL)
        for f in ("knn", "gabriel", "rng")]
jobs += [(s, r, f) for s in STRUCT_SPECS for r in range(REPS_STRUCT)
         for f in ("knn", "gabriel", "rng")]
print(len(jobs), "jobs")

if __name__ == "__main__":
    with Pool(2) as p:
        rows = p.map(run, jobs)
    df = pd.DataFrame(rows)
    df.to_csv("family_sweep.csv", index=False)
    print(df.groupby(["family", "fam"]).agg(
        abs_err=("k_hat", lambda s: np.abs(s - df.loc[s.index, "true_k"]).mean()),
        exact=("k_hat", lambda s: (s == df.loc[s.index, "true_k"]).mean()),
        m=("m_edges", "mean"), conn=("cand_connected", "mean"),
        secs=("secs", "mean")).round(3))
