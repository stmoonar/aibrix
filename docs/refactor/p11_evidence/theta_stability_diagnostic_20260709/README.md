# theta_m stability diagnostics — dsllama-8b scrutiny (2026-07-09)

Follow-up to `../r3_calibration_20260709/` (the R3/S2 write-back). An architect review (Fable 5)
flagged `dsllama-8b`'s published `theta_m = 209.647` as suspicious: it swung ~3.5x from its
"inherited" prior (738) under the same reparameterization (w_p 0.08→0.02, λ 1.875→3.0) that barely
moved `dsqwen-7b` (738.67→715.27, −3%), and llama's 738 prior is *identical* to 7b's inherited
738.67 — i.e. it looks like a copied prior that was never independently fit on llama's own data.
Llama also has the smallest window support (1671) and lowest Spearman (0.6948) of the three, and
its load-scan grid was never validity-checked or supplemented (7b/14b both were).

Two cheap diagnostics were built to test this. Both reuse the **production**
`fit_theta_by_reliability` verbatim (reliability 0.9, min_support 3, min_confidence 0.9,
min_scenario_families 2, max_single_scenario_ratio 0.7) — no fit logic was reimplemented.

## Diagnostic 1 — cell-level bootstrap CI on `theta_m`

New permanent QA tool `tre_calibration.bootstrap.bootstrap_theta` (+ CLI
`deploy/scripts/bootstrap_theta_ci.py`). Resampling unit is the distinct `scenario_id`
(load-scan **grid cell**), NOT the individual sliding window — windows inside a cell overlap
~5/6 of their raw requests (30s window / 5s step, ADR-0012) so window-level bootstrap would
understate variance. Each iteration draws `n_cells` cells with replacement (seeded
`random.Random`, reused across iterations) and re-fits on the concatenated window blocks.
n_resamples = 2000, seed = 42, same SLO/params as the published R3 fit.

| model | cells | support | point θ | boot median | 95% CI [p2.5, p97.5] | mean | std | CV (std/point) | publish_rate |
|---|---|---|---|---|---|---|---|---|---|---|
| dsqwen-7b  | 49 | 2619 | 715.271 | 766.541 | [715.271, 1202.131] | 858.36 | 170.87 | **0.24** | 1.00 |
| dsllama-8b | 33 | 1671 | 209.647 | 209.647 | [209.647, 1351.199] | 648.07 | 494.74 | **2.36** | 1.00 |
| dsqwen-14b | 46 | 2543 | 247.417 | 247.417 | [247.417, 1020.715] | 428.56 | 251.70 | **1.02** | 1.00 |

(All three models' point θ equals their own p2.5 — a structural consequence of the reliability
fit picking the *smallest* θ that clears the gates, so resampling can only push θ up. The
informative quantity is therefore the **width / upper tail / std**, not the lower bound.)

**Reading**: llama's bootstrap distribution is by far the loosest. Its CI spans **1141.6** units
on a point of 210 (a 5.4x upper spread), its std (494.74) is **2.9x** 7b's and **2x** 14b's, and
its coefficient of variation relative to the published value (2.36) is **~10x** 7b's (0.24) and
2.3x 14b's (1.02). The published 209.647 sits at the very floor of a distribution whose *mean*
is 648 and whose upper tail reaches 1351 — i.e. the "typical" resample of llama's own cells
yields a θ roughly 3x higher than the number that was actually published. 7b, by contrast, is
tight: its whole CI lives in 715–1202 and its mean (858) is within 20% of its point. All three
publish on 100% of resamples (the fit never collapses), so this is pure θ-*location* instability,
not a publish/coverage failure.

## Diagnostic 2 — refit llama under its OLD inherited weights, on its OWN real data

The 738 prior was never actually fit against llama's data. So we ran that fit for the first time:
re-windowed llama's raw per-request logs under the **old** weights (`w_p=0.08, λ=1.875, qmin=1.0`)
and ran the production fit CLI.

- Re-window: `deploy/scripts/rewindow_from_raw.py --model dsllama-8b --raw-dir
  /root/tre-experiments/r3_raw/r3_llama_sweep --registry
  /root/tre-experiments/registry_llama_inherited.yaml --window-ms 30000 --step-ms 5000 --output
  /root/tre-experiments/r3_llama_slide_inherited.csv` → 1984 windows (same 30000/5000 config as
  the published `_v2` CSVs, only the trs weights differ). `registry_llama_inherited.yaml` is a
  copy of `registry_v2_7b_llama.yaml` with only dsllama-8b's `trs.w_p`/`trs.lambda_wait` set back
  to the inherited values.
- Fit (`theta_fit_llama_inherited.json`): **`theta_m = 296.47`**, publish=True, support=1671,
  attainment=0.937, coverage_pass=True, all 6 scenario families covered, AUROC 0.926,
  spearman 0.589.

| llama θ_m | source |
|---|---|
| **738** | assumed "inherited" — identical to 7b's 738.67; **never fit on llama's data** |
| **296.47** | llama's OWN data under those exact old weights (this diagnostic) |
| **209.647** | published, adopted weights (w_p 0.02 / λ 3.0) |

**Reading**: the 738 was fiction. Llama's own data under the old weights fits **296**, not 738 —
so the historical prior was indeed a copied 7b number with zero grounding in llama's measurements.
Crucially, though, 296 and the adopted 209.647 are the **same order of magnitude** (~200–300),
~41% apart — comparable to how 7b's own θ moves (715→766) across resamples. So the published
209.647 is *not* an artifact of the reparameterization: two independent weight settings both put
llama's real capacity threshold in the 200–300 band. It is a low draw within that band, but it is
not fabricated.

## Verdict

**CONFIRMED.** dsllama-8b's calibration is unambiguously the weakest of the three and the most
under-supported by data:

1. **Fewest cells (33 vs 46/49), smallest window support (1671 vs 2543/2619), lowest Spearman
   (0.6948).** Already known; now quantified downstream.
2. **Widest bootstrap CI by a large margin** — std 494.74 (2.9x 7b), CV 2.36 (~10x 7b's 0.24).
   Its published θ is the *floor* of a distribution whose typical value is ~3x higher. llama's θ
   is genuinely unstable to which grid cells the scan happened to hit; 7b's is rock-solid.
3. **Its 738 prior was never grounded in its own data** — refitting llama's real data under those
   exact old weights yields 296, not 738. The 3.5x "swing" the architect saw was a swing away from
   a fictitious number, not a real regression.

**One nuance that does NOT rescue the calibration but does bound the damage**: the *direction* of
209.647 is corroborated — llama's own data says ~200–300 under both weight settings, so the
adopted value is a plausible (if low) point in the right band, not a wild artifact. But it should
be treated as a **soft lower bound with high uncertainty**, not a precise threshold: the bootstrap
says the "expected" θ for llama is closer to 300–650. Before any paper leans on llama's Z_m
cross-model comparison, its load-scan grid should be **supplemented with more cells** (as 7b's and
14b's were), which is the single cheapest way to collapse that CI. Trust llama's θ as "≈200–300,
wide error bars"; do NOT trust 209.647 as a sharp number.

## Files here

- `bootstrap_theta_{7b,llama,14b}.json` — full bootstrap reports (point fit + distribution).
- `theta_fit_llama_inherited.json` — Diagnostic 2 fit (llama, old weights, own data → θ=296.47).

## Large artifacts (not in repo — on node10 `76`, `/root/tre-experiments/`)

- `r3_{7b,llama_v2 / 14b}_slide*.csv` — the published R3 slide CSVs the bootstrap resampled
  (`r3_7b_slide_v2.csv`, `r3_llama_slide_v2.csv`, `r3_14b_slide.csv`).
- `r3_llama_slide_inherited.csv` — Diagnostic-2 re-window (llama, old weights) — 1984 windows.
- `registry_llama_inherited.yaml` — temp registry (llama `w_p=0.08, λ=1.875`) used for the
  re-window. Not part of the deployed config.
- Raw per-request + instant JSONL: `/root/tre-experiments/r3_raw/r3_llama_sweep/`.
