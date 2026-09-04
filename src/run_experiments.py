from __future__ import annotations
import os, sys, sys, time, json, warnings, itertools
import numpy as np, pandas as pd
from multiprocessing import Pool
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(__file__))
from datasets import SPECS, make, real_datasets
from baselines import REGISTRY, _km
from cdk import CDK
from sklearn.metrics import (adjusted_rand_score, adjusted_mutual_info_score,
                             normalized_mutual_info_score)

REPS = int(os.environ.get("REPS", 20))
REPS_REAL = int(os.environ.get("REPS_REAL", 10))
CDK_R = int(os.environ.get("CDK_R", 199))


def krange(true_k):
    lo = max(2, true_k - 5)
    hi = max(12, true_k + 5)
    return list(range(lo, hi + 1))


def score(y, lab):
    m = y >= 0                      # noise points excluded from external scores
    if m.sum() < 2 or len(np.unique(lab[m])) < 1:
        return dict(ARI=np.nan, AMI=np.nan, NMI=np.nan)
    return dict(ARI=adjusted_rand_score(y[m], lab[m]),
                AMI=adjusted_mutual_info_score(y[m], lab[m]),
                NMI=normalized_mutual_info_score(y[m], lab[m]))


def run_one(job):
    try:
        return _run_one(job)
    except Exception as e:                    # one bad spec must not kill the sweep
        import traceback
        sys.stderr.write(f"JOB FAILED {job[0][:2] if isinstance(job[0],(list,tuple)) else job[0]} "
                         f"rep={job[1]}: {e}\n" + traceback.format_exc() + "\n")
        sys.stderr.flush()
        return []


def _run_one(job):
    spec, rep, tag = job
    if tag == "syn":
        X, y, tk = make(spec, rep)
        name, fam = spec[1], spec[0]
    else:
        name, X, y, tk = spec
        fam = "real"
    ks = krange(tk)
    rows = []

    # oracle: k-means at the true k (quality reference only)
    t0 = time.time()
    lab = _km(X, max(tk, 1), rep).labels_ if tk >= 1 else np.zeros(len(X), int)
    rows.append(dict(dataset=name, family=fam, rep=rep, n=len(X), d=X.shape[1],
                     true_k=tk, method="ORACLE k-means(k*)", k_hat=tk,
                     runtime=time.time() - t0, **score(y, lab)))

    # the proposal
    variants = [("CDK (proposed)", {})]
    if os.environ.get("ABLATE"):
        variants += [("CDK / Gaussian null", dict(null="gaussian")),
                     ("CDK / AD statistic", dict(stat="ad")),
                     ("CDK / no multiplicity corr.", dict(correct=False)),
                     ("CDK / n_min=15", dict(n_min=15)),
                     ("CDK / n_min=40", dict(n_min=40))]
    for label, kw in variants:
        t0 = time.time()
        try:
            r = CDK(R=CDK_R, random_state=rep, **kw).fit(X)
            kk, lab = r.k, r.labels
            extra = dict(conf_lo=r.conf_set[0], conf_hi=r.conf_set[1],
                         n_tests=r.n_tests, k_init=r.k_init,
                         sep_evidence=r.sep_evidence, p_median=r.p_median)
        except Exception as e:
            kk, lab, extra = np.nan, np.zeros(len(X), int), {}
        rows.append(dict(dataset=name, family=fam, rep=rep, n=len(X),
                         d=X.shape[1], true_k=tk, method=label, k_hat=kk,
                         runtime=time.time() - t0, **extra, **score(y, lab)))

    # baselines
    for nm, fn in REGISTRY.items():
        t0 = time.time()
        try:
            kk, lab = fn(X, ks, rep)
        except Exception:
            kk, lab = np.nan, np.zeros(len(X), int)
        rows.append(dict(dataset=name, family=fam, rep=rep, n=len(X),
                         d=X.shape[1], true_k=tk, method=nm, k_hat=kk,
                         runtime=time.time() - t0, **score(y, lab)))
    return rows


if __name__ == "__main__":
    only = os.environ.get("ONLY")
    specs = [s for s in SPECS if (not only or s[0] in only.split(","))]
    jobs = [(s, r, "syn") for s in specs for r in range(REPS)]
    if not only:
        jobs += [(s, r, "real") for s in real_datasets() for r in range(REPS_REAL)]
    CKPT = os.environ.get("OUT","../results/raw_results.csv") + ".ckpt"
    print(f"{len(jobs)} jobs, REPS={REPS}, CDK_R={CDK_R}", flush=True)
    t0 = time.time()
    out = []
    with Pool(2) as pool:
        for i, rows in enumerate(pool.imap_unordered(run_one, jobs, chunksize=1)):
            out.extend(rows)
            if (i + 1) % 20 == 0:
                pd.DataFrame(out).to_csv(CKPT, index=False)
                el = time.time() - t0
                print(f"  {i+1}/{len(jobs)}  {el/60:.1f} min elapsed, "
                      f"eta {el/(i+1)*(len(jobs)-i-1)/60:.1f} min", flush=True)
    df = pd.DataFrame(out)
    df["abs_err"] = (df.k_hat - df.true_k).abs()
    df["signed_err"] = df.k_hat - df.true_k
    df["exact"] = (df.k_hat == df.true_k).astype(float)
    df.to_csv(os.environ.get("OUT","../results/raw_results.csv"), index=False)
    print("saved", df.shape, f"in {(time.time()-t0)/60:.1f} min")
