# CDK: certified divisive determination of the number of clusters

Code, raw results and reproduction material for the journal extension of
the CDK paper ("Determining the number of clusters without a candidate
range: a certified divisive procedure with graph-theoretic error
control").

## Layout

- `src/` — the procedure and every variant examined in the paper:
  - `cdk.py` — the conference implementation (over-partition, energy
    statistic, matched-mechanism Monte Carlo, multiplicity threshold).
  - `cdk_geom.py` — candidate merge families as graphs: Gabriel and
    relative neighbourhood families, bridge finding, and the
    2-edge-connectivity augmentation (the certified family).
  - `cdk_hull.py` — the convex-hull reference (exact uniform sampling
    over the union's hull via Delaunay simplices, d <= 6).
  - `cdk_robust.py` — the trimmed merge test (gamma; the same trimming
    operator applied to the observed union and to every reference draw).
  - `uniforce.py` — reimplementation of UniForCE (Vardakas, Kalogeratos
    and Likas, Pattern Recognition 172:112357, 2026), checked against the
    official repository; deviations documented in the module docstring.
  - `wbp_baseline.py` — wrapper over the authors' implementation of Wu,
    Bien and Panigrahi (arXiv:2512.06522); clone
    github.com/judywu4800/SI_HierarchicalClustering as `wbp_repo/` beside
    this directory to run it.
  - `baselines.py`, `datasets.py`, `run_experiments.py` — the conference
    benchmark harness (16 methods, 43 synthetic specifications, 15 real
    datasets, fixed seeds).
- `results/` — every raw result file the paper's numbers are computed
  from: the conference runs (`raw_results.csv`, `raw_ablation.csv`) and
  the extension runs (certified family, UniForCE, WBP subset, family
  study, hull and trim sweeps, ablation rerun, repaired default,
  coverage arm at R=1999, large-n scaling).
- `runners/` — the scripts that produced each extension result file;
  all are seed-fixed and resume from checkpoints.
- `CDK_extension_verify.ipynb` — a self-contained verification notebook
  (Google Colab): Section A recomputes every reported number from the
  result files with assertions; Section B re-runs the algorithms live.

## Requirements

Python 3.10+, numpy, scipy, scikit-learn, pandas; `diptest` for the
UniForCE baseline. The WBP baseline additionally needs the authors'
repository as described above.

## Reproduction

The notebook is the quickest check. The full protocol is reproduced by
`src/run_experiments.py` (conference pool) and the scripts in `runners/`
(extension), and takes several CPU-hours in total.
