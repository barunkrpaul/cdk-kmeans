"""Wu-Bien-Panigrahi baseline over a documented representative subset.

The authors' implementation costs minutes per fit even at n<=60-80, so the
full 1010-block protocol is out of reach here.  Subset: all 8 null specs x 2
reps, 12 structured specs x 2 reps, 6 real datasets x 1 rep = 46 blocks, with
the subsample cap n_cap=60.
"""
import sys, time, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, "cdk/src")
import numpy as np, pandas as pd
from multiprocessing import Pool
from datasets import SPECS, make, real_datasets
from wbp_baseline import wbp_estimate
from sklearn.metrics import (adjusted_rand_score, adjusted_mutual_info_score,
                             normalized_mutual_info_score)

STRUCT = ["sep_k2_d2", "sep_k3_d2", "sep_k5_d10", "sep_k10_d50", "sep_k20_d10",
          "overlap_s4.0", "unbal_r20", "elong_c5", "nested_d2",
          "nongauss_t3", "noise_30", "uvar_r3"]
REAL = ["Iris", "Wine", "Breast cancer", "Digits-PCA10",
        "BC-benign(1cls)", "Iris-setosa(1cls)"]

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
        k, lab = wbp_estimate(X, rep=rep, n_cap=60, ngrid=500)
        return [dict(dataset=name, family=fam, rep=rep, n=len(X), d=X.shape[1],
                     true_k=tk, method="WBP (HC-confidence)", k_hat=k,
                     runtime=time.time() - t0, **score(y, lab))]
    except Exception as e:
        import traceback
        sys.stderr.write(f"FAILED {job}: {e}\n" + traceback.format_exc())
        return []

if __name__ == "__main__":
    nulls = [s for s in SPECS if s[0] == "null"]
    struct = [s for s in SPECS if s[1] in STRUCT]
    reals = [r for r in real_datasets() if r[0] in REAL]
    jobs = [(s, r, "syn") for s in nulls for r in range(2)]
    jobs += [(s, r, "syn") for s in struct for r in range(2)]
    jobs += [(s, 0, "real") for s in reals]
    print(len(jobs), "jobs", flush=True)
    t0 = time.time(); out = []
    with Pool(2) as pool:
        for i, rows in enumerate(pool.imap_unordered(run_one, jobs, chunksize=1)):
            out.extend(rows)
            pd.DataFrame(out).to_csv("wbp_subset.csv.ckpt", index=False)
            print(f"{i+1}/{len(jobs)} {(time.time()-t0)/60:.1f} min", flush=True)
    pd.DataFrame(out).to_csv("wbp_subset.csv", index=False)
    print("saved", len(out), f"rows in {(time.time()-t0)/60:.1f} min")
