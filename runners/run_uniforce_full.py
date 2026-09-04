"""Full-protocol run of the UniForCE baseline (1010 blocks)."""
import sys, time, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, "cdk/src")
import numpy as np, pandas as pd
from multiprocessing import Pool
from datasets import SPECS, make, real_datasets
from uniforce import uniforce
from sklearn.metrics import (adjusted_rand_score, adjusted_mutual_info_score,
                             normalized_mutual_info_score)

REPS, REPS_REAL = 20, 10

def score(y, lab):
    m = y >= 0
    if m.sum() < 2 or len(np.unique(lab[m])) < 1:
        return dict(ARI=np.nan, AMI=np.nan, NMI=np.nan)
    return dict(ARI=adjusted_rand_score(y[m], lab[m]),
                AMI=adjusted_mutual_info_score(y[m], lab[m]),
                NMI=normalized_mutual_info_score(y[m], lab[m]))

def run_one(job):
    try:
        spec, rep, tag = job
        if tag == "syn":
            X, y, tk = make(spec, rep); name, fam = spec[1], spec[0]
        else:
            name, X, y, tk = spec; fam = "real"
        t0 = time.time()
        k, lab = uniforce(X, random_state=rep)
        return [dict(dataset=name, family=fam, rep=rep, n=len(X), d=X.shape[1],
                     true_k=tk, method="UniForCE", k_hat=k,
                     runtime=time.time() - t0, **score(y, lab))]
    except Exception as e:
        import traceback
        sys.stderr.write(f"FAILED {job}: {e}\n" + traceback.format_exc())
        return []

if __name__ == "__main__":
    jobs = [(s, r, "syn") for s in SPECS for r in range(REPS)]
    jobs += [(s, r, "real") for s in real_datasets() for r in range(REPS_REAL)]
    print(len(jobs), "jobs", flush=True)
    t0 = time.time(); out = []
    with Pool(2) as pool:
        for i, rows in enumerate(pool.imap_unordered(run_one, jobs, chunksize=4)):
            out.extend(rows)
            if (i + 1) % 100 == 0:
                pd.DataFrame(out).to_csv("uniforce_full.csv.ckpt", index=False)
                print(f"{i+1}/{len(jobs)} {(time.time()-t0)/60:.1f} min", flush=True)
    pd.DataFrame(out).to_csv("uniforce_full.csv", index=False)
    print("saved", len(out), f"rows in {(time.time()-t0)/60:.1f} min")
