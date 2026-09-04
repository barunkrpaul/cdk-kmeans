"""Full-protocol run of CDK with the knn + 2-edge-connectivity candidate family.

Same job grid as run_experiments.py (43 synthetic specs x 20 reps + 15 real
datasets x 10 reps = 1010 blocks); only the corrected-family CDK is run, for
comparison against the existing raw_results.csv rows of 'CDK (proposed)'.
"""
import sys, os, time, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, "cdk/src")
import numpy as np, pandas as pd
from multiprocessing import Pool
from datasets import SPECS, make, real_datasets
from cdk_geom import CDKGeo2
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
        r = CDKGeo2(family="knn", random_state=rep).fit(X)
        return [dict(dataset=name, family=fam, rep=rep, n=len(X), d=X.shape[1],
                     true_k=tk, method="CDK knn+2ec", k_hat=r.k,
                     runtime=time.time() - t0, n_tests=r.n_tests,
                     k_init=r.k_init, conf_lo=r.conf_set[0],
                     conf_hi=r.conf_set[1], sep_evidence=r.sep_evidence,
                     p_median=r.p_median, **score(y, r.labels))]
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
        for i, rows in enumerate(pool.imap_unordered(run_one, jobs, chunksize=1)):
            out.extend(rows)
            if (i + 1) % 50 == 0:
                pd.DataFrame(out).to_csv("knn2ec_full.csv.ckpt", index=False)
                el = time.time() - t0
                print(f"{i+1}/{len(jobs)} {el/60:.1f} min, "
                      f"eta {el/(i+1)*(len(jobs)-i-1)/60:.1f} min", flush=True)
    pd.DataFrame(out).to_csv("knn2ec_full.csv", index=False)
    print("saved", len(out), f"rows in {(time.time()-t0)/60:.1f} min")
