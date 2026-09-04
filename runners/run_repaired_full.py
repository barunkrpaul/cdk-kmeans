"""Full-protocol run of the repaired CDK default:
knn+2ec family, trimmed test gamma=0.1, hull reference (d<=6, box above)."""
import sys, time, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, "cdk/src")
import numpy as np, pandas as pd
from multiprocessing import Pool
from datasets import SPECS, make, real_datasets
import cdk_hull  # patches the reference; provides 'hull' kind
from cdk_robust import CDKRobust
from sklearn.metrics import (adjusted_rand_score, adjusted_mutual_info_score,
                             normalized_mutual_info_score)

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
        r = CDKRobust(gamma=0.10, null="hull", random_state=rep).fit(X)
        return [dict(dataset=name, family=fam, rep=rep, n=len(X), d=X.shape[1],
                     true_k=tk, method="CDK repaired", k_hat=r.k,
                     runtime=time.time() - t0, n_tests=r.n_tests,
                     k_init=r.k_init, sep_evidence=r.sep_evidence,
                     p_median=r.p_median, **score(y, r.labels))]
    except Exception as e:
        import traceback
        sys.stderr.write(f"FAILED {job}: {e}\n" + traceback.format_exc())
        return []

if __name__ == "__main__":
    import os
    done = set()
    prev = None
    if os.path.exists("repaired_full.csv.ckpt"):
        prev = pd.read_csv("repaired_full.csv.ckpt")
        done = set(zip(prev.dataset, prev.rep))
        print("resuming past", len(done), "blocks", flush=True)
    jobs = [(s, r, "syn") for s in SPECS for r in range(20)]
    jobs += [(s, r, "real") for s in real_datasets() for r in range(10)]
    jobs = [j for j in jobs if (j[0][1] if j[2]=="syn" else j[0][0], j[1]) not in done]
    print(len(jobs), "jobs", flush=True)
    t0 = time.time(); out = []
    with Pool(2) as pool:
        for i, rows in enumerate(pool.imap_unordered(run_one, jobs, chunksize=1)):
            out.extend(rows)
            if (i + 1) % 25 == 0:
                cum = pd.DataFrame(out) if prev is None else pd.concat(
                    [prev, pd.DataFrame(out)], ignore_index=True)
                cum.to_csv("repaired_full.csv.ckpt", index=False)
                el = time.time() - t0
                print(f"{i+1}/{len(jobs)} {el/60:.1f} min, "
                      f"eta {el/(i+1)*(len(jobs)-i-1)/60:.1f} min", flush=True)
    df = pd.DataFrame(out)
    if prev is not None:
        df = pd.concat([prev, df], ignore_index=True).drop_duplicates(subset=["dataset","rep"])
    df.to_csv("repaired_full.csv", index=False)
    print("COMPLETE", len(df), f"rows in {(time.time()-t0)/60:.1f} min")
