# R3 / S2 calibration write-back evidence (2026-07-09)

Final calibration landing for the three-model fleet on the frozen control window
**W=30000 sliding** (ADR-0012). Signal = `trs` column of the R3 slide CSVs, produced by the
same shared `TRSComputer` the controller runs (doc15 §4 offline re-window). This directory holds
the small fit/eval JSONs that back the `registry.yaml` write; the large sliding CSVs are **not**
checked in — their paths on node10 (`76`) are listed below.

## Decision (doc15 §2 5% criterion, §1.4 write-back)

| model | w_p | lambda_wait | qmin | theta_m (published) | S2 recommendation |
|---|---|---|---|---|---|
| dsqwen-7b  | 0.02 (was 0.08)   | 3.0 (was 1.875) | 1.0 | **715.2709967116108** (was 738.67) | adopt_refit |
| dsllama-8b | 0.02 (was 0.08)   | 3.0 (was 1.875) | 1.0 | **209.6470588235294** (was 738)    | adopt_refit |
| dsqwen-14b | 0.0575 (kept)     | 3.0 (kept)      | 1.0 | **247.4172794117647** (was 534)    | keep_inherited |

- **7b / llama**: S2 refit (`refit_*_report.json`) picks `w_p=0.02, lambda_wait=3.0` as best on
  the `trs` signal. Spearman-health improvement over the inherited params exceeds the 5% bar
  (7b +8.61%, llama +18.84%); AUROC change is within noise (7b +1.17%, llama +0.31%). Per doc15 §2
  the 5% Spearman gate is met on both ⇒ **adopt**. `theta_m` is then re-fit on the **new-parameter**
  `trs` column (reliability 0.9, e2e SLO 12000) → the published values above (`theta_fit_*_v2.json`).
- **14b**: S2 recommends `keep_inherited` (w_p 0.0575 / λ3.0 already best-tier); `theta_m` written
  from the enriched (supplemental-cell) fit `theta_fit_14b.json` = 247.4172794117647.

All other `trs` fields (tau_*, ema_*, qsat/epsat/hsat, w_d) are unchanged.

## theta fits (full-CSV, reliability 0.9)

| model | theta_m | publish | coverage_pass | AUROC | spearman_health | support |
|---|---|---|---|---|---|---|
| dsqwen-7b (v2)  | 715.271 | true | true | 0.9287 | 0.7542 | 2619 |
| dsllama-8b (v2) | 209.647 | true | true | 0.9235 | 0.6948 | 1671 |
| dsqwen-14b      | 247.417 | true | true | 0.9802 | 0.8209 | 2543 |

## Ranking separation (train/test scenario holdout, two seeds)

Terminal `eval_ranking_separation.py`, test_fraction 0.2. `theta` reported is the **train-split**
reliability fit (differs from the published full-fit theta by design — this is a generalisation
diagnostic, not the published value).

| model | seed | train AUROC | test AUROC | test viol windows | false_healthy | generalizes |
|---|---|---|---|---|---|---|
| dsqwen-7b  | tre-v2-ranking   | 0.957 | 0.817 | 50 | 50 | False |
| dsqwen-7b  | tre-v2-ranking-b | 0.913 | 0.972 | 62 | 62 | False |
| dsllama-8b | tre-v2-ranking   | 0.928 | 0.898 | 16 | 16 | False |
| dsllama-8b | tre-v2-ranking-b | 0.964 | 0.771 | 19 | 19 | False |
| dsqwen-14b | tre-v2-ranking   | 0.979 | 0.500 | 0  | 0  | True  |
| dsqwen-14b | tre-v2-ranking-b | 0.977 | 0.500 | 0  | 0  | True  |

**Known split artifact**: 14b's test AUROC pins at 0.500 for both seeds because the held-out
scenarios contain **zero** SLO-violation windows (test_viol=0) — AUROC is undefined with one class,
so the evaluator returns 0.5. This is a split-composition artifact, not a signal failure: the
full-fit 14b AUROC is 0.9802 and train AUROC ~0.978 both seeds. `false_healthy=0` on test both seeds,
so `generalizes=True`.

## Violation rates (window-level p95 > SLO on the slide CSV; independent of trs params)

| model | e2e SLO (ms) | windows | violations | pct | breakdown |
|---|---|---|---|---|---|
| dsqwen-7b  | 12000 | 2979 | 221 | 7.42% | ttft 190 / e2e 65 / tpot 0 |
| dsllama-8b | 12000 | 1671 | 105 | 6.28% | ttft 55 / e2e 56 / tpot 0 |
| dsqwen-14b | 15000 | 2873 | 129 | 4.49% | e2e 116 / ttft 28 / tpot 0 |

## Files here

- `theta_fit_7b_v2.json`, `theta_fit_llama_v2.json` — theta fit on the **new-param** (w_p0.02/λ3.0) slide CSV.
- `theta_fit_14b.json` — enriched 14b fit (params unchanged), source of the 247.417 write.
- `refit_{7b,llama,14b}_report.json` — S2 grid-search reports (doc15 §2).
- `ranking_{7b,llama,14b}_final[_seedB].json` — ranking separation, two seeds each.
- `summary_3models.json` — roll-up (theta, S2 rec, ranking, violation stats).

## Large artifacts (not in repo — on node10 `76`, `/root/tre-experiments/`)

- `r3_7b_slide_v2.csv` (30000/5000 slide, bucket_upper, instant 10000; w_p0.02/λ3.0 trs) — 2979 windows.
- `r3_llama_slide_v2.csv` — same config — 1984 windows.
- `r3_14b_slide.csv` — 14b slide (params unchanged) — 2873 windows.
- Raw per-request + instant JSONL: `/root/tre-experiments/r3_raw/{r3_7b_sweep,r3_llama_sweep,r3_14b_sweep}/`.
- Re-window used a temp registry `/root/tre-experiments/registry_v2_7b_llama.yaml` (7b/llama w_p0.02/λ3.0)
  since `rewindow_from_raw.py` reads trs params from the registry `spec`.

## Follow-up: 7b capacity monotonicity supplement (A3 dependency)

Supplemental sweep  (4 cells x 300s) appended to 
(540 -> 580 rows), then  re-fit (, ttft 500 / tpot 75).
C(512, o) four points: o128 **11.733** > o256 **9.600** > o384 **8.533** > o512 **6.400** rps —
strictly decreasing, monotonicity RESOLVED (was 7.467 / 5.333 for o256/o384 with the
low-concurrency-only support, which dipped below o512).
