import sys, time, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, "cdk/src")
import numpy as np, pandas as pd
from multiprocessing import Pool
from datasets import SPECS, make
from cdk_geom import CDKGeo2

NULL_SPECS = [s for s in SPECS if s[0] == "null"]
STRUCT = ["sep_k2_d2", "sep_k5_d10", "sep_k10_d50", "sep_k20_d10",
          "overlap_s4.0", "unbal_r20", "elong_c5", "nested_d2",
          "nongauss_t3", "noise_30"]
STRUCT_SPECS = [s for s in SPECS if s[1] in STRUCT]

def run(job):
    spec, rep, fam = job
    X, y, tk = make(spec, rep)
    t0 = time.time()
    r = CDKGeo2(family=fam, random_state=rep).fit(X)
    return dict(dataset=spec[1], family=spec[0], rep=rep, fam=fam + "2ec",
                true_k=tk, k_hat=r.k, k_init=r.k_init, m_edges=r.n_tests,
                cand_connected=True, n_tests=r.n_tests,
                secs=round(time.time() - t0, 2))

jobs = [(s, r, f) for s in NULL_SPECS for r in range(5) for f in ("gabriel", "rng")]
jobs += [(s, r, f) for s in STRUCT_SPECS for r in range(3) for f in ("gabriel", "rng")]
print(len(jobs), "jobs")
if __name__ == "__main__":
    with Pool(2) as p:
        rows = p.map(run, jobs)
    pd.DataFrame(rows).to_csv("family_sweep2.csv", index=False)
    print("done")
