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
