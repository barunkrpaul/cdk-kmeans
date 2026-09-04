"""Ablation grid rerun with the knn+2ec candidate family as the base.

Same 32-spec x 10-rep grid as raw_ablation.csv, same five variants plus the
new base, so the journal Table V can be quoted for the corrected default.
"""
import sys, time, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, "cdk/src")
import numpy as np, pandas as pd
from multiprocessing import Pool
from datasets import SPECS, make
from cdk_geom import CDKGeo2
from sklearn.metrics import adjusted_rand_score

GRID = ['nongauss_mixed', 'nongauss_skew', 'nongauss_t3', 'nongauss_t3_d10',
        'nongauss_uniform', 'null_aniso_d10', 'null_aniso_d2',
        'null_gaussian_d10', 'null_gaussian_d2', 'null_t3_d10', 'null_t3_d2',
        'null_uniform_d10', 'null_uniform_d2', 'overlap_s2.0', 'overlap_s3.0',
        'overlap_s4.0', 'overlap_s6.0', 'sep_k10_d10', 'sep_k10_d2',
        'sep_k10_d50', 'sep_k20_d10', 'sep_k20_d2', 'sep_k20_d50',
        'sep_k2_d10', 'sep_k2_d2', 'sep_k2_d50', 'sep_k3_d10', 'sep_k3_d2',
        'sep_k3_d50', 'sep_k5_d10', 'sep_k5_d2', 'sep_k5_d50']
REPS = 10
VARIANTS = [("CDK 2ec (default)", {}),
            ("CDK 2ec / Gaussian null", dict(null="gaussian")),
            ("CDK 2ec / AD statistic", dict(stat="ad")),
            ("CDK 2ec / no multiplicity corr.", dict(correct=False)),
            ("CDK 2ec / n_min=15", dict(n_min=15)),
            ("CDK 2ec / n_min=40", dict(n_min=40))]

def run_one(job):
    spec, rep = job
    rows = []
    try:
        X, y, tk = make(spec, rep)
        for label, kw in VARIANTS:
            t0 = time.time()
            try:
                r = CDKGeo2(family="knn", random_state=rep, **kw).fit(X)
                kk, lab = r.k, r.labels
            except Exception:
                kk, lab = np.nan, np.zeros(len(X), int)
            m = y >= 0
            ari = adjusted_rand_score(y[m], lab[m]) if m.sum() > 1 else np.nan
            rows.append(dict(dataset=spec[1], family=spec[0], rep=rep,
                             true_k=tk, method=label, k_hat=kk, ARI=ari,
                             runtime=time.time() - t0))
    except Exception as e:
        sys.stderr.write(f"FAILED {job}: {e}\n")
    return rows

if __name__ == "__main__":
    import os
    done = set()
    if os.path.exists("ablation_2ec.csv.ckpt"):
        prev = pd.read_csv("ablation_2ec.csv.ckpt")
        done = set(zip(prev.dataset, prev.rep))
        print("resuming past", len(done)//1, "checkpointed blocks", flush=True)
    else:
        prev = None
    specs = [s for s in SPECS if s[1] in GRID]
    jobs = [(s, r) for s in specs for r in range(REPS)
            if (s[1], r) not in done]
    MAXJOBS = int(os.environ.get("MAXJOBS", "40"))
    total_remaining = len(jobs)
    jobs = jobs[:MAXJOBS]
    print("remaining", total_remaining, "-> this slice", len(jobs), flush=True)
    print(len(jobs), "jobs x", len(VARIANTS), "variants", flush=True)
    t0 = time.time(); out = []
    with Pool(2) as pool:
        for i, rows in enumerate(pool.imap_unordered(run_one, jobs, chunksize=1)):
            out.extend(rows)
            if (i + 1) % 10 == 0:
                cum = pd.DataFrame(out) if prev is None else pd.concat(
                    [prev, pd.DataFrame(out)], ignore_index=True)
                cum.to_csv("ablation_2ec.csv.ckpt", index=False)
                el = time.time() - t0
                print(f"{i+1}/{len(jobs)} {el/60:.1f} min, "
                      f"eta {el/(i+1)*(len(jobs)-i-1)/60:.1f} min", flush=True)
    df = pd.DataFrame(out)
    if prev is not None:
        df = pd.concat([prev, df], ignore_index=True).drop_duplicates(
            subset=["dataset", "rep", "method"])
    df.to_csv("ablation_2ec.csv.ckpt", index=False)
    if total_remaining <= MAXJOBS:
        df.to_csv("ablation_2ec.csv", index=False)
        print("COMPLETE:", len(df), "rows")
    print("slice done,", len(df), f"rows total, {(time.time()-t0)/60:.1f} min")
