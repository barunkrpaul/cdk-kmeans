import sys, time, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, "cdk/src")
import numpy as np, pandas as pd
from datasets import SPECS, make
from cdk_robust import CDKRobust
from sklearn.metrics import adjusted_rand_score

NOISE = [s for s in SPECS if s[0] == "noise"]
NULLS = [s for s in SPECS if s[0] == "null"]
CLEAN = [s for s in SPECS if s[1] in
         ["sep_k5_d10", "sep_k20_d10", "elong_c5", "overlap_s4.0", "unbal_r20"]]

rows = []
for spec, reps in [(s, 5) for s in NOISE] + [(s, 3) for s in NULLS] + [(s, 3) for s in CLEAN]:
    for rep in range(reps):
        X, y, tk = make(spec, rep)
        for g in (0.0, 0.1, 0.2):
            t0 = time.time()
            r = CDKRobust(gamma=g, random_state=rep).fit(X)
            m = y >= 0
            ari = adjusted_rand_score(y[m], r.labels[m]) if m.sum() > 1 else np.nan
            rows.append(dict(dataset=spec[1], family=spec[0], rep=rep,
                             true_k=tk, gamma=g, k_hat=r.k, ARI=ari,
                             sep_evidence=r.sep_evidence,
                             secs=round(time.time() - t0, 1)))
    print("done", spec[1], flush=True)
    pd.DataFrame(rows).to_csv("robust_sweep.csv", index=False)
pd.DataFrame(rows).to_csv("robust_sweep.csv", index=False)
print("saved", len(rows))
