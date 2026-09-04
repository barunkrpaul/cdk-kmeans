"""Large-n scaling of the repaired CDK default. Resume-capable.

Grid: separated blobs (k* in {5,20}, d in {2,10}) and Gaussian nulls (k*=1,
d=2), at n in {2000, 10000, 50000, 200000}, 2 replicates each = 40 fits.
Records k_hat and wall time so the cost analysis is measured rather than
asserted.
"""
import sys, os, time, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, "cdk/src")
import numpy as np, pandas as pd
from datasets import make
import cdk_hull
from cdk_robust import CDKRobust

NS = [2000, 10000, 50000, 200000]
CASES = []
for n in NS:
    for k, d in [(5, 2), (5, 10), (20, 2), (20, 10)]:
        CASES.append((f"sep_k{k}_d{d}_n{n}", ("sep", "x", dict(k=k, d=d, sep=8.0, n=n)), k))
    CASES.append((f"null_d2_n{n}", ("null", "x", dict(d=2, kind="gaussian", n=n)), 1))

done = set()
rows = []
if os.path.exists("scaling.csv.ckpt"):
    prev = pd.read_csv("scaling.csv.ckpt")
    rows = prev.to_dict("records")
    done = set(zip(prev.case, prev.rep))
    print("resuming past", len(done), "fits", flush=True)

for name, spec, tk in CASES:
    for rep in range(2):
        if (name, rep) in done:
            continue
        X, y, _ = make(spec, rep)
        t0 = time.time()
        r = CDKRobust(gamma=0.10, null="hull", random_state=rep).fit(X)
        el = time.time() - t0
        rows.append(dict(case=name, rep=rep, n=len(X), d=X.shape[1],
                         true_k=tk, k_hat=r.k, k_init=r.k_init,
                         n_tests=r.n_tests, secs=round(el, 1)))
        pd.DataFrame(rows).to_csv("scaling.csv.ckpt", index=False)
        print(f"{name} rep{rep}: k_hat={r.k} (true {tk})  {el:.0f}s", flush=True)

pd.DataFrame(rows).to_csv("scaling.csv", index=False)
print("COMPLETE", len(rows), "fits")
