# Ranking metrics of the pressure signal (2026-09-30)

Status: implemented (`tre_calibration.ranking`, `dline_refit accept`, `scripts.ranking_report`).
**Disclosure only.** No acceptance criterion reads these numbers; no preregistration makes
them a gate. Every JSON block carries `"gating": false` and the definitions below
(`definition` fields, `tre_calibration.ranking.DEFINITIONS`).

## Why

The acceptance criteria (plan §6.9f A-D) judge the *thresholded* classifier: BA at the
published theta, dwell-confirmed CRITICAL recall / false alarm. They say nothing about
whether Z orders windows by how stressed they are, which is what the controller's
cross-model comparison relies on. These metrics disclose that ordering.

## Definitions

Notation: a window w of model m has Z = `tre_calibration.fit.signal_z(signal, theta_m,
direction)` (higher is healthier for both orientations), **pressure** = -Z, label
`slo_met`, and **severity** = the label's `ratio_max` (max over the label's metrics of
p95 / SLO under the frozen label definition; a window violated only through unserved
requests gets >= `UNSERVED_MIN_RATIO` = 2.0). Windows are the ones the fit loader keeps
(`SignalSpec.load` with the frozen signal spec, label and trim; TSS recomputed with the
frozen EMA).

1. **AUROC (pooled-Z AUROC).** Positives = violated windows (not `slo_met`), score =
   pressure. AUROC = U / (n_pos n_neg), U the Mann-Whitney statistic with average ranks,
   so a positive-negative tie counts 1/2. Undefined (null) when a class is empty.
   Reported per model and **pooled** over the windows of all models - pooling is the
   point: a single Z scale is what makes the models comparable. Windows with a
   non-finite Z are dropped and counted (`dropped_nonfinite_z`).
2. **Kendall tau-b (i), window level.** tau_b(pressure, severity) =
   (C - D) / sqrt((n0 - n1)(n0 - n2)) over all pairs of windows, n0 = n(n-1)/2, n1 / n2 the
   pairs tied in pressure / severity. Per model and pooled over all windows of all models.
   Positive = Z ranks pressure correctly. Windows without a finite severity are dropped
   and counted (`dropped_no_severity`). Computed in O(n log n) (Knight 1966).
3. **Kendall tau-b (ii), cross-model.** Only pairs of windows of **different** models at
   the **same instant** (the window's `window_end_ms`). Aggregated as a stratified tau-b:
   sum over instants of (C - D) / sqrt(sum over instants of pairs not tied in pressure x
   sum of pairs not tied in severity). Reported with the number of instants holding at
   least one such pair and the number of pairs. Null with the reason
   "needs multi-model concurrent data (E1 or concurrent campaigns)" when no instant holds
   two models' windows (e.g. a sequential campaign, or single-model cells). Also reported
   with instants rounded to the nearest re-window step (`bin_<step>ms`), for data whose
   windows are not cut on one absolute grid. The function takes records
   `(model, cell, instant, z, severity, violated)` (`ranking.RankRecord`), so E1 timeline
   data can be fed in unchanged.
4. **Confidence intervals.** Cell bootstrap exactly as for the BA: cells (scenario ids)
   drawn with replacement, `n_resamples` (accept: 1000) draws from `random.Random(seed)`
   (accept: `SEED`), 95 % percentile interval with the `_ci95` index rule
   (sorted v: `[v[int(.025 n)], v[int(.975 n) - 1]]`). Pooled metrics draw each model's
   cells from that model's cells (**stratified by model**); models in sorted order, cells
   sorted. A resample on which a metric is undefined is skipped (`resamples_used`). With
   one model the draw sequence is identical to `acceptance_bootstrap`'s, so the per-model
   ranking intervals come from the same resamples as the BA interval.
   **Caveat for tau_b (ii):** its pairs join cells of different models that ran at the
   same time, but the bootstrap resamples each model's cells independently, so it ignores
   the dependence the shared time span creates (e.g. a common gateway or node effect).
   Its interval is therefore likely too narrow; read it as a lower bound on the
   uncertainty. Resampling time-overlap clusters would be the fix if (ii) ever gates.

### Exact matrix bootstrap

A resample with cell multiplicities w (cell c drawn k times = k copies of its windows)
gives, over the expanded window list,
`sum_{ordered pairs} sgn(dx) sgn(dy) = w'Kw`, pairs not tied in x (y) `= w'Aw` (`w'Bw`),
`2U = w'Uw`, `n_pos = w.P`, `n_neg = w.N`, with cell x cell integer matrices K, A, B, U
built once (a window paired with its own copy is tied in both coordinates and adds
nothing). Hence tau_b = w'Kw / sqrt(w'Aw w'Bw) and AUROC = w'Uw / (2 n_pos n_neg) -
equal to recomputing on the expanded resample (tested). For tau-b (ii) K, A, B hold only
the cross-model same-instant pairs. All entries are integers, so the numpy path and the
pure-Python fallback are bit-identical (tested); numpy is used when importable. Cost:
O(windows^2) to build the matrices, O(cells^2) per resample. Run 2 (4.8 k windows, 408
cells, 1000 resamples): 5 s with numpy; the pure-Python path takes 27 s at 200 resamples.

## Where it is reported

* `dline_refit accept` (result `format_revision` 2): per model `ranking_disclosure`
  (AUROC + tau_b (i) with CIs and counts) and a top-level pooled `ranking_disclosure`
  (pooled AUROC, pooled tau_b (i), tau_b (ii) exact and binned), plus a table on stdout.
  `--recheck` of a revision-1 result (the D22 one) drops the keys revision 2 added
  (`ranking_disclosure`, per-model `windowing`) before comparing; everything else must
  still match.
* `python -m scripts.ranking_report`: any standard dataset(s), a freeze (or explicit
  per-model parameters), row filters (`--split`, `--cell-status` default `valid`,
  `--role`), `--resamples` / `--seed`, `--instant-bin-ms`; writes `NAME.json` and
  `NAME.md`. Same functions as accept.
* The other AUROC implementations delegate to `ranking.auroc`: `evaluate._auc` (keeps its
  0.5 for a single class), `e5_timeline_auroc.auroc` and `calibration_resplit.auroc`
  (None). `v1_lambda_fit.auc` is left alone on purpose: it is a faithful port of v1's
  objective, whose selections it must reproduce.

## Reading the numbers

AUROC measures separation of violated from healthy windows; tau_b (i) also asks whether
*more* pressure goes with *more* severe violations, inside both classes; tau_b (ii) asks
whether, at one moment, the model with the higher pressure is the one in worse shape -
the comparison the controller makes. tau_b values are much smaller than AUROC by
construction (all within-class pairs count), and many healthy windows share severities
near their floor, so ties are common: tau-b (not tau-a) is used for that reason.

## Windowing parameters (same change)

`dline_refit` stages take `--window-ms` / `--step-ms` / `--dt-ref-s` / `--horizon-ms` /
`--dwell-windows` (defaults 30 s / 10 s / 10 s / 30 s / 2, the former constants), thread
them explicitly, record them as `windowing` in alpha / wp / final outputs and in the
freeze (format revision 2; revision 1 freezes still verify and accept at the defaults),
and accept applies the recorded dwell / window length. `alpha_fit` gains `--window-ms`
and `--episode-margin-ms` (defaults unchanged). A new window length is an offline
re-run with flags; no code change.
