# P6 Calibration Design

## Old Flow Summary

The frozen calibration path is centered on `fit_tre_parameters_from_runs.py`, a 1939-line script that loads per-window run CSVs, filters warmup or contaminated rows, rebuilds latency SLO labels, computes TRS-like signals, scores candidate parameters with Spearman health correlation plus AUROC, then fits `theta_m` from healthy quantiles or a legacy reliability scan. A smaller `fit_theta.py` path fits a publishable threshold from filtered CSVs using the same higher-TRS-is-healthier convention and scenario-family coverage checks.

## Split Target

The refactor starts with small, testable modules:

- `dataset.py` owns calibration window records and scenario-level train/test splitting, so scenario IDs never leak across sets.
- `fit.py` owns threshold fitting. The first implementation fits a separable synthetic boundary where higher signal means healthier service. Later slices will add healthy-quantile and reliability/coverage selection.
- `evaluate.py` owns no-dependency rank and threshold metrics: AUROC, Spearman health correlation, and balanced accuracy.

## First Verification Fixture

The first synthetic fixture uses violating windows below TRS 100 and healthy windows above TRS 100. The expected fit is `theta_m = 100`, AUROC is 1.0, Spearman health direction is 1.0, and threshold classification has no false healthy or false violation windows. This gives P6 a deterministic baseline before porting CSV loading and old-parameter search.


## CSV Loading and Publish Gate

The second slice adds filtered CSV loading for the same row classes: warmup rows, contaminated rows, rows with a filter reason, rows without finite signal/SLO fields, and rows with no token traffic are excluded. Active latency SLOs derive both the hard `slo_met` label and the continuous health score `1 / (1 + max_p95_ratio)`.

The reliability fitter mirrors the archived publish gate at small scale: scan candidate thresholds from low to high, select the first threshold whose `signal >= theta` subset has enough support and SLO attainment, then publish only if scenario-family coverage and confidence pass. The output keeps structured reject reasons for later CLI/profile emission.


## Signal Recompute

The third slice extracts the archived TRS formula into `signals.py`. `SignalInputs` carries token totals, queue depths, replica visibility, and KV cache hit rate. `compute_trs()` emits both floor-protected and raw TRS variants using the old formula: prompt tokens are discounted by cache hit rate and weighted by `w_p`, waiting queue depth is scaled by `lambda_wait`, queue floor is bounded by `qmin`, and scores are multiplied by assigned/routable replica ratio.

Candidate scoring currently uses raw TRS (`trs_no_floor`), matching the archived parameter objective path, and evaluates direction with AUROC plus Spearman health correlation. Later slices can grid-search over this helper without mixing formula code into CSV loading or threshold publishing.


## Parameter Search and Profile Patch

The fourth slice adds a deterministic grid-search wrapper over candidate `w_p`, `lambda_wait`, and `qmin` values. Candidate ranking uses objective first, then AUROC and Spearman health direction, with smaller parameters as final tie-breakers for reproducibility. The profile patch builder is intentionally separate from registry mutation: it returns a stable payload containing publish status, theta fit gates, selected TRS parameters, and scoring metrics. A later CLI can decide whether to write this payload beside a run or apply it to `registry.yaml`.


## CLI Artifact

The initial CLI is deliberately artifact-only. It reads a filtered window CSV, applies latency SLO labels, fits theta with the reliability gate, evaluates the CSV signal direction, and writes a sorted JSON profile patch. It does not mutate `registry.yaml`; applying or rejecting the patch remains an explicit operator step.
