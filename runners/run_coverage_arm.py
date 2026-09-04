"""Coverage arm: repaired config with R=1999 (nominal level operative).
Half-replicates: 10 reps synthetic, 5 real = 505 blocks. Resume-capable."""
import sys, os, time, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, "cdk/src")
import numpy as np, pandas as pd
from multiprocessing import Pool
from datasets import SPECS, make, real_datasets
import cdk_hull
from cdk_robust import CDKRobust
from sklearn.metrics import adjusted_rand_score

def run_one(job):
    try:
        spec, rep, tag = job
        if tag == "syn":
            X, y, tk = make(spec, rep); name, fam = spec[1], spec[0]
        else:
            name, X, y, tk = spec; fam = "real"
        t0 = time.time()
        r = CDKRobust(gamma=0.10, null="hull", R=1999, random_state=rep).fit(X)
        m = y >= 0
        ari = adjusted_rand_score(y[m], r.labels[m]) if m.sum() > 1 else np.nan
        return [dict(dataset=name, family=fam, rep=rep, true_k=tk,
                     method="CDK repaired R1999", k_hat=r.k, ARI=ari,
                     n_tests=r.n_tests, sep_evidence=r.sep_evidence,
                     runtime=time.time() - t0)]
    except Exception as e:
        import traceback
        sys.stderr.write(f"FAILED {job}: {e}\n" + traceback.format_exc())
        return []

if __name__ == "__main__":
    done = set(); prev = None
    if os.path.exists("coverage_arm.csv.ckpt"):
        prev = pd.read_csv("coverage_arm.csv.ckpt")
        done = set(zip(prev.dataset, prev.rep))
        print("resuming past", len(done), "blocks", flush=True)
    jobs = [(s, r, "syn") for s in SPECS for r in range(5)]
    jobs += [(s, r, "real") for s in real_datasets() for r in range(3)]
    jobs = [j for j in jobs if (j[0][1] if j[2]=="syn" else j[0][0], j[1]) not in done]
    print(len(jobs), "jobs", flush=True)
    t0 = time.time(); out = []
    with Pool(2) as pool:
        for i, rows in enumerate(pool.imap_unordered(run_one, jobs, chunksize=1)):
            out.extend(rows)
            if (i + 1) % 10 == 0:
                cum = pd.DataFrame(out) if prev is None else pd.concat(
                    [prev, pd.DataFrame(out)], ignore_index=True)
                cum.to_csv("coverage_arm.csv.ckpt", index=False)
                el = time.time() - t0
                print(f"{i+1}/{len(jobs)} {el/60:.1f} min, "
                      f"eta {el/(i+1)*(len(jobs)-i-1)/60:.1f} min", flush=True)
    df = pd.DataFrame(out)
    if prev is not None:
        df = pd.concat([prev, df], ignore_index=True).drop_duplicates(subset=["dataset","rep"])
    df.to_csv("coverage_arm.csv", index=False)
    print("COMPLETE", len(df), f"rows in {(time.time()-t0)/60:.1f} min")
