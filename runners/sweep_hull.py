import sys, time, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, "cdk/src")
import numpy as np, pandas as pd
from datasets import SPECS, make
from cdk_hull import CDKHull

NULLS = [s for s in SPECS if s[0] == "null"]
STRUCT = ["sep_k2_d2", "sep_k5_d2", "sep_k20_d2", "overlap_s4.0", "elong_c5",
          "nested_d2", "noise_30", "unbal_r20", "sep_k5_d10", "sep_k10_d50"]
SS = [s for s in SPECS if s[1] in STRUCT]

rows = []
for spec in NULLS + SS:
    reps = 5 if spec[0] == "null" else 3
    for rep in range(reps):
        X, y, tk = make(spec, rep)
        t0 = time.time()
        r = CDKHull(random_state=rep).fit(X)
        rows.append(dict(dataset=spec[1], family=spec[0], rep=rep, true_k=tk,
                         k_hat=r.k, n_tests=r.n_tests,
                         sep_evidence=r.sep_evidence, p_median=r.p_median,
                         secs=round(time.time()-t0, 2)))
    print("done", spec[1], flush=True)
pd.DataFrame(rows).to_csv("hull_sweep.csv", index=False)
print("saved", len(rows))
