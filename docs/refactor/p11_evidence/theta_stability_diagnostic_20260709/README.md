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

## dsllama-8b broad grid supplement

Run date: 2026-07-09. Safety gates before load generation were clean: Redis `tre:v2:controller:mode` was `observe`, and service-manager `/v2/state` showed one awake, non-hidden `dsllama-8b` binding (`nscc-ds-4a100-node9`, GPU0). No controller mode, registry, pod, deployment, or routing state was changed.

Supplement command added exactly the sibling-model first-round broad grid: every llama family got concurrency 48, 64, and 96. The resumable checkpoint and sweep CSV ended at exactly **54 cells** (old 36 + new 18), with no unexpected scenario IDs. The live R3 run emitted 10 tumbling windows per new cell to `/root/tre-experiments/r3_llama_sweep.csv`; offline re-windowing at the published sliding config (`window_ms=30000`, `step_ms=5000`) produced **2985 CSV rows** from the full raw directory, and the production loader retained **2672 fit windows / 51 latency-valid cells**.

| new cell | R3 tumbling windows | sliding windows | violation windows |
|---|---:|---:|---:|
| i1024_o128_c48 | 10 | 55 | 8 |
| i1024_o128_c64 | 10 | 55 | 2 |
| i1024_o128_c96 | 10 | 55 | 55 |
| i1024_o512_c48 | 10 | 57 | 57 |
| i1024_o512_c64 | 10 | 57 | 57 |
| i1024_o512_c96 | 10 | 57 | 57 |
| i128_o128_c48 | 10 | 55 | 0 |
| i128_o128_c64 | 10 | 55 | 1 |
| i128_o128_c96 | 10 | 55 | 45 |
| i128_o512_c48 | 10 | 56 | 0 |
| i128_o512_c64 | 10 | 56 | 10 |
| i128_o512_c96 | 10 | 55 | 55 |
| i512_o128_c48 | 10 | 55 | 19 |
| i512_o128_c64 | 10 | 55 | 25 |
| i512_o128_c96 | 10 | 55 | 18 |
| i512_o512_c48 | 10 | 55 | 55 |
| i512_o512_c64 | 10 | 57 | 57 |
| i512_o512_c96 | 10 | 56 | 56 |

Re-window command:

```bash
cd /data/nfs_shared_data/xxy/aibrix/tre
PYTHONPATH=common:controller:service-manager:replayer:deploy \
/root/miniconda3/bin/python deploy/scripts/rewindow_from_raw.py \
  --model dsllama-8b \
  --raw-dir /root/tre-experiments/r3_raw/r3_llama_sweep \
  --output /root/tre-experiments/r3_llama_slide_supp.csv \
  --window-ms 30000 --step-ms 5000 \
  --percentile-mode bucket_upper --instant-sample-ms 10000 \
  --registry /root/tre-experiments/registry_v2_7b_llama.yaml
```

Production refit on the supplemented sliding CSV publishes cleanly:

| dataset | cells used by bootstrap | fit windows | support | point theta | boot median | 95% CI [p2.5, p97.5] | mean | std | CV (std/point) | publish_rate |
|---|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|
| published llama v2 | 33 | 1671 | 1671 | 209.647 | 209.647 | [209.647, 1351.199] | 648.07 | 494.74 | **2.36** | 1.00 |
| broad supplement | 51 | 2672 | 1913 | 1290.915 | 1290.269 | [981.843, 1384.117] | 1266.78 | 106.88 | **0.083** | 1.00 |

The supplement shifted the point estimate upward by **1081.27** TRS units (**+516%**, 6.16x the published value) and collapsed the relative bootstrap instability from CV **2.36** to **0.083** (about **28.5x lower**). The point no longer sits at the floor of a huge distribution; the bootstrap mass is tightly centered around ~1.27k, with a 95% span of 402.27 TRS units instead of 1141.55.

Violation concentration was recomputed on `/root/tre-experiments/r3_llama_slide_supp.csv` using the same window-level rule as the sibling diagnostic: a violation is any window where `p95_ttft > 500`, `p95_tpot > 75`, or `p95_e2e > 12000`.

| concentration metric | value |
|---|---:|
| latency-valid cells | 51 |
| cells with >=1 violation | 24 |
| total violation windows | 682 |
| max single-cell share | 8.36% (57 / 682) |
| top-2 share | 16.72% (114 / 682) |
| top-3 share | 25.07% (171 / 682) |
| top-5 share | 41.64% (284 / 682) |
| top-10 share | 80.65% (550 / 682) |

Top cells are high-concurrency, long-output cases, but none dominates alone:

| rank | cell | violation windows | share | main failing SLOs |
|---:|---|---:|---:|---|
| 1 | i1024_o512_c48 | 57 | 8.36% | e2e, sparse ttft |
| 2 | i1024_o512_c64 | 57 | 8.36% | e2e, sparse ttft |
| 3 | i1024_o512_c96 | 57 | 8.36% | e2e, ttft |
| 4 | i512_o512_c64 | 57 | 8.36% | e2e, ttft |
| 5 | i1024_o512_c32 | 56 | 8.21% | e2e, sparse ttft |
| 6 | i512_o512_c96 | 56 | 8.21% | e2e, ttft |
| 7 | i1024_o128_c96 | 55 | 8.06% | tpot, sparse ttft/e2e |
| 8 | i128_o512_c96 | 55 | 8.06% | e2e, ttft |
| 9 | i512_o512_c48 | 55 | 8.06% | e2e, sparse ttft |
| 10 | i128_o128_c96 | 45 | 6.60% | ttft |

**Verdict**: the broad supplement worked for stability. Llama no longer has the original thin-data / giant-CV failure mode, and it does **not** reveal the sibling model's cheap targeted fix pattern (that sibling had 87.6% of violations concentrated in 2 cells; llama's top 2 are only 16.7%). The new theta is much higher than the published 209.647, so this should be treated as a material calibration update candidate rather than a minor confidence tweak. On data quality alone, the supplemented llama calibration is now publishable enough for human review; if more work is done, it should be a confirmatory repeat or a few boundary cells around the high-concurrency long-output front, not a narrow two-cell concentration repair.

New files in this evidence directory:

- `theta_fit_llama_supp.json` -- production fit on `/root/tre-experiments/r3_llama_slide_supp.csv`.
- `bootstrap_theta_llama_supp.json` -- cell-level bootstrap, n_resamples=2000, seed=42.

Additional large artifacts left only on node10:

- `/root/tre-experiments/r3_llama_sweep.csv` -- full 54-cell sweep CSV.
- `/root/tre-experiments/r3_llama_sweep.checkpoint.json` -- exactly 54 completed cells.
- `/root/tre-experiments/r3_llama_slide_supp.csv` -- supplemented 30s/5s sliding-window CSV.

## 14b violation-densification supplement (follow-up)

Follow-up to the bootstrap-CI diagnostic above: 14b's CV=1.02 was traced to violation
concentration, not a fit-code issue. Of 129 violating windows in the published `r3_14b_slide.csv`
(46-cell sweep, 52 cells checkpointed), **113 (87.6%) came from exactly 2 cells**:
`i1024_o512_c192` (57/57 windows violating) and `i512_o512_c192` (56/56) — both sitting at the
very top of their concurrency ladders, with the next-richest evidence (`i1024_o512_c128`, 9/57)
two rungs down and a large gap (128→192) never sampled in between. `i128_o512` was also never
extended past c32, unlike its two sibling o512 families.

**Fix**: densify the onset region rather than add more cells overall. Ran 7 new cells at 300s
each (`--cell-seconds 300 --window-ms 30000`, same `r3_grid.py` invocation pattern as the
original supplement) — safety gates confirmed clean beforehand (`tre:v2:controller:mode=observe`,
service-manager `/v2/state` showed `dsqwen-14b` awake on `nscc-ds-4a100-node10` GPU0-1):

| new cell | R3 tumbling windows |
|---|---:|
| i1024_o512_c112 | 10 |
| i1024_o512_c160 | 10 |
| i512_o512_c112 | 10 |
| i512_o512_c160 | 10 |
| i128_o512_c96 | 10 |
| i128_o512_c128 | 10 |
| i128_o512_c192 | 10 |

Checkpoint went from 52 -> 59 cells with no unexpected scenario IDs re-driven. Re-window
(`rewindow_from_raw.py --model dsqwen-14b --raw-dir /root/tre-experiments/r3_raw/r3_14b_sweep
--registry deploy/registry.yaml --window-ms 30000 --step-ms 5000`, same config as the published
`r3_14b_slide.csv`) produced **3263 sliding windows** -> `/root/tre-experiments/r3_14b_slide_supp2.csv`.

Production refit (`tre_calibration.cli`, same fit-config as always: reliability 0.9, min_support 3,
min_confidence 0.9, min_scenario_families 2, max_single_scenario_ratio 0.7, w_p=0.0575,
lambda_wait=3.0, qmin=1.0, SLO ttft=500/tpot=75/e2e=15000):

| dataset | cells | fit windows | support | point theta | boot median | 95% CI [p2.5, p97.5] | mean | std | CV (std/point) | publish_rate |
|---|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|
| published (46 cells) | 46 | 2543 | 2543 | 247.417 | 247.417 | [247.417, 1020.715] | 428.56 | 251.70 | **1.02** | 1.00 |
| densified supplement (59 cells) | 53 | 2933 | 2913 | **934.996** | 905.895 | [247.417, 1147.889] | 737.10 | 354.41 | **0.379** | 1.00 |

Violation concentration on the 59-cell dataset (`p95_e2e>15000` / `p95_ttft>500` / `p95_tpot>75`,
same rule as the original diagnostic):

| concentration metric | before (46 cells, 129 violations) | after (59 cells, 309 violations) |
|---|---:|---:|
| single-cell max share | 44.2% (57/129, `i1024_o512_c192`) | 18.4% (57/309, tied `i1024_o512_c160`/`c192`) |
| two-cell combined share | 87.6% (113/129) | 36.9% (114/309) |
| five-cell combined share | n/a (only 2 cells drove almost everything) | 90.6% (280/309: `i1024_o512_c160/c192`, `i512_o512_c160/c192`, `i128_o512_c192`, each 55-57 windows, ~18% each) |

**Reading**: the densification fix worked exactly as predicted. theta_m jumped 3.78x
(247.417 -> 934.996) and bootstrap CV collapsed 2.7x (1.02 -> 0.379) — the old point estimate was
an artifact of two sparse, saturated cells clearing the reliability gate too easily; once the
128->192 gap was filled, the fit discovered the true onset sits far higher, and violations are now
spread across 5 cells at a near-uniform ~18% each instead of concentrated in 2. The new 934.996
lands almost exactly inside the *old* bootstrap's own upper mode (97.5th pct was 1020.7) —
convergence with the old distribution's blow-up half, not a new instability.

New files in this evidence directory: `theta_fit_14b_supp2.json`, `bootstrap_theta_14b_supp2.json`.
Large artifact: `/root/tre-experiments/r3_14b_slide_supp2.csv` (node10 only).

## 14b convergence probe (Fable 5 follow-up, step 1)

Architect review (Fable 5) on the above result: the theta jump is a **safety improvement**
(higher theta_m => lower Z_m for the same real TRS => controller reads 14b as *more*
capacity-tense, i.e. the old 247.417 was too permissive), not a regression — corroborated by the
934.996 landing inside the old CI's upper mode. Fable's prediction: theta is set by the
onset/boundary region, not by how deep saturation goes, so extending the ladder into deeper
saturation shouldn't move theta much further (those cells sit *below* theta, they don't define it).

Tested with 3 more cells at the same 300s/window-30000ms convention: a new boundary midpoint
`c144` for both rich o512 families, plus a deep-saturation control `i1024_o512_c256`. Safety gates
re-checked and still clean before this run (`observe` mode, `dsqwen-14b` binding still awake,
ClusterIPs unchanged: `dsqwen-14b` 10.98.154.51, `tre-v2-redis` 10.109.196.83). Checkpoint went
59 -> 62 cells. Re-window (same command, now against all 62 cells) produced 3431 sliding windows
-> `/root/tre-experiments/r3_14b_slide_supp3.csv`.

| dataset | cells | fit windows | support | point theta | boot median | CV | 97.5pct |
|---|---:|---:|---:|---:|---:|---:|---:|
| densified supp2 (59 cells) | 53 | 2933 | 2913 | 934.996 | 905.895 | 0.379 | 1147.889 |
| convergence-probe supp3 (62 cells) | 56 | 3101 | 3008 | **1020.235** | 1004.209 | **0.321** | 1166.606 |

theta_m moved **+9.12%** (934.996 -> 1020.235) — under Fable's 15% "converged, freeze it"
threshold. **Verdict: converged.** Bootstrap CV kept improving (0.379 -> 0.321), consistent with
more evidence tightening the fit rather than the estimate still hunting for a ceiling.

New-cell TRS values and violation status (direct check of Fable's "deep saturation sits below
theta" prediction):

| cell | windows | TRS min | TRS median | TRS max | violations |
|---|---:|---:|---:|---:|---:|
| i512_o512_c144 | 56 | 1102.26 | 1189.11 | 1436.06 | 0/56 |
| i1024_o512_c144 | 57 | 1157.56 | 1244.64 | 1712.81 | 27/57 |
| i1024_o512_c256 | 55 | 755.18 | 994.57 | 1255.38 | 55/55 |

**Nuance, reported plainly rather than forcing Fable's clean prediction**: `c144` (the new
boundary midpoint) actually has *higher* TRS than `c256` and is only partially violating —
consistent with it being a genuine boundary-region point, and its own high-TRS/non-violating
windows are what pushed theta up the further +9.1%, not saturation. `c256` (deep saturation, 100%
violating) has median TRS 994.57, only ~2.5% below the new theta (1020.235) — directionally
correct (below theta, in the violating set) but not a dramatically wide gap; the control cell
confirms the *sign* of the prediction (saturation windows sit at/under theta, not above it) but
the margin is narrower than "clearly below" would suggest. Overall this is best read as: the
onset region (c112-c160) is what defines theta and is now well localized; deeper saturation
doesn't drive theta further up, it just confirms it's on the correct side of the boundary.

Violation concentration re-checked on the final 62-cell / 3431-window dataset (391 total
violations, up from 309 with the addition of the new cells):

| concentration metric | 59-cell (309 viol.) | 62-cell final (391 viol.) |
|---|---:|---:|
| single-cell max share | 18.4% | 14.58% (57/391) |
| two-cell combined share | 36.9% | 29.16% (114/391) |
| five-cell combined share | 90.6% | 71.61% (280/391; a 6th cell, `i1024_o512_c256`, adds another 14.1%, ~85.7% for 6 cells) |

Concentration kept flattening as more onset-region data arrived — further confirmation the
densification fix is real and not an artifact of which 7 cells happened to be picked.

New files in this evidence directory: `theta_fit_14b_supp3.json`, `bootstrap_theta_14b_supp3.json`.
Large artifact: `/root/tre-experiments/r3_14b_slide_supp3.csv` (node10 only, 62 cells / 3431 rows).

**Per Fable's explicit guidance (step 4), no broad llama-style re-scan was run for 14b** — the
boundary is now localized to c112-c256 and these 3 targeted probes were judged to strictly
dominate a broad re-scan; that path was deliberately not taken.

## TRS / tau_high shrink-eligibility check (Fable 5 follow-up, step 2)

Concern: shrink-eligibility requires `Z_m > tau_high (1.26)`, i.e. `TRS > tau_high * theta_m`. At
the old theta (247.417) that bar is 311.75; at the converged new theta (1020.235) it is **1285.50**
— nearly 4.1x higher. Could 14b become permanently un-shrinkable?

**(a) Calibration-side ceiling.** Across the final 62-cell / 3431-window dataset, low-concurrency
cells (c<=8, where queueing is near-floor and TRS = throughput/queue should sit at its structural
peak) give:

| stat | low-concurrency (c<=8), n=1320 | all windows, n=3431 |
|---|---:|---:|
| min | 143.34 | 143.34 |
| p50 | 2180.96 | 1987.90 |
| p95 | 2961.65 | 2818.65 |
| max | 4206.09 | 4206.09 |

Even the **median** low-concurrency TRS (2180.96) clears the new tau_high bar (1285.50) by
**1.70x**, and the max clears it by **3.27x**. **14b is not at risk of becoming permanently
un-shrinkable at theta=1020.235** — there is ample structural TRS headroom above tau_high; the
model simply needs to actually be at low/idle concurrency when the slow loop evaluates
shrink-eligibility (normal — that is the intended condition for a shrink check to fire).

**(b) Real-trace check.** HANDOFF.md's open item #1 references a `timeline_tre.csv` for trial t5
as evidence for "14b's Z_m/theta not crossing delta_high" during an unexplained scale-up failure.
That file exists at `docs/refactor/p11_evidence/exp3_comparison_20260709/timeline_tre.csv` (458
rows) but its schema is only `ts, awake, submitted, loop, actions` — **no per-window Z_m/TRS value
was ever persisted**, so it cannot be used to literally reconstruct 14b's historical Z_m
trajectory. The only real numeric evidence available is the t5 trial-level summary
(`summary_tre_t5.json`): 14b was in deep saturation during that trace (`violation_time_frac=0.84`,
`p95_e2e_ms=35833`), i.e. that trial exercised the **low-Z_m / scale-up boundary** (`delta_high`),
not the **high-Z_m / shrink boundary** (`tau_high`) this check is about — so t5 is simply not
informative for the tau_high question either way, and the two open items (t5's scale-up gap,
this task's tau_high risk) are about different, unrelated thresholds despite both citing 14b's
theta_m. Flagging as a logging gap: no per-window Z_m/TRS is persisted anywhere for any real
serving trial, which is worth fixing so this class of question can be answered directly next time
rather than only via calibration-side proxies.

**Verdict**: based on (a), 14b is **not** at material risk of becoming permanently un-shrinkable
under theta=1020.235 — the structural TRS ceiling clears the raised tau_high bar with 1.7-3.3x
headroom. (b) neither confirms nor refutes this from production data since the needed signal was
never logged; it is a gap, not a contradiction. `tau_high` does not need to move purely on this
evidence, though re-verifying with an actual Z_m timeline (once logged) would be worthwhile before
fully closing this question.

## Backlog: theta-stability acceptance gate + 7b/llama grid-gap audit (Fable 5 follow-up, step 3)

**Backlog note.** This round's whole finding (14b's original theta was 3.78x too low because of a
masked ladder gap) was only caught by an after-the-fact bootstrap-CV audit. Future calibration
rounds should run the cell-level bootstrap CI (`tre_calibration.bootstrap.bootstrap_theta` /
`deploy/scripts/bootstrap_theta_ci.py`, already built and reused unchanged throughout this whole
diagnostic) as a **go/no-go acceptance gate before writing to `registry.yaml`**, not just as a
follow-up diagnostic after publishing. A reasonable starting gate: reject (or require ladder
densification before) any fit with CV > ~0.5-0.6, given 7b's CV=0.24 is comfortably stable and
14b's original 1.02 / llama's original 2.36 were both clearly unstable by inspection.

**Gap audit — does the same masked-boundary pattern exist for 7b / llama?** (No new load-scan
cells run for either model — this is purely a read of the existing sweep/slide CSVs already on
disk, same method used to find 14b's original gap: last thin-violation point vs. top of tested
ladder, per family.)

*dsqwen-7b* (ladders per family, from `r3_7b_sweep.csv`; violations from `r3_7b_slide_v2.csv`,
e2e SLO 12000, 221 total violations):

| family | ladder top | violations at top-of-ladder cell | share of total |
|---|---:|---:|---:|
| i1024_o512 (top=c64) | 64 | 56 | 25.3% |
| i1024_o128 (top=c64) | 64 | 40 | 18.1% |
| i512_o512 (top=c64) | 64 | 15 (peak is c48=27, i.e. *before* the top) | 6.8% |

Two of three violating families (`i1024_o512`, `i1024_o128`) have their **single worst violation
count sitting exactly at the top of the tested ladder** (c64), with no cell beyond it to confirm
the onset has actually plateaued — the same qualitative signature 14b had before this fix.
Combined, these two top-of-ladder cells hold 96/221 = **43.4%** of all 7b violations. The
`i512_o512` family, by contrast, already shows its peak one rung *before* the top (c48 > c64),
suggesting that family's boundary is already captured mid-ladder — a healthier sign. Despite this,
7b's own bootstrap CV (0.24) is by far the tightest of the three models, so if a masked boundary
exists here it has not yet destabilized the fit the way it did for 14b/llama — but the structural
gap (`i1024_o512`/`i1024_o128` never extended past c64) is real and worth a cheap follow-up probe
(e.g. c80/c96) if 7b's calibration is revisited.

*dsllama-8b* (ladders from `r3_llama_sweep.csv`, **all six families capped at c32**, before this
task's already-committed broad-grid supplement above; violations from `r3_llama_slide_v2.csv`,
e2e SLO 12000, 105 total violations):

| family | ladder top (pre-supplement) | violations at top-of-ladder cell | share of total |
|---|---:|---:|---:|
| i1024_o512 (top=c32) | 32 | 56 | 53.3% |
| i512_o128 (top=c32) | 32 | 15 | 14.3% |
| i512_o512 (top=c32) | 32 | 15 | 14.3% |

llama's single worst cell (`i1024_o512_c32`, at the absolute top of a ladder that **never
extended past c32 in any family**) held **53.3%** of all violations by itself — a more extreme
single-cell concentration than 14b's original 44.2% — and the top-of-ladder cells across all three
violating families combined for 86/105 = **81.9%**. This is the *same* masked-boundary pattern as
14b's, and structurally worse (no family got extended at all, vs. 14b where some families reached
c96-c192 before this task). This lines up exactly with why llama's original bootstrap CV (2.36) was
by far the worst of the three in the diagnostic above, and independently corroborates it via a
different signal (violation concentration vs. the tested ladder). Note: llama's broad-grid
supplement (committed separately, `28443563`, see the `## dsllama-8b broad grid supplement`
section above) already addressed this — its post-supplement CV of 0.083 confirms the same
densification approach that worked for 14b also worked for llama.

**Takeaway**: this is a **systemic pattern**, not a 14b-specific fluke — every model's calibration
grid had at least one family whose top-of-ladder cell held an outsized share of total violations
with no cell beyond it to confirm a plateau. 7b's is present but has not (yet) visibly destabilized
its fit; llama's was the most severe and has already been fixed; 14b's has now also been fixed and
converged. Recommend making the "densify past the last thin-violation point" check a standard part
of grid design for any future model onboarding, informed by the acceptance-gate backlog item above.

## dsqwen-7b two-round convergence probe + registry adoption (2026-07-09/10, Fable 5 follow-up)

Follow-up to the gap audit above, which flagged `i1024_o512`/`i1024_o128` as having the same
masked-boundary pattern as llama/14b (43.4% of 7b's violations sitting at the untested top of a
c64 ladder), though 7b's original bootstrap CV (0.24) had not yet visibly destabilized.

**Round 1**: added `c80`/`c96` to both flagged families (4 cells, 300s each, same load-driver
convention as llama/14b). Refit on the resulting 62-cell dataset:

| dataset | cells | fit windows | point theta | CI95 | std | CV |
|---|---:|---:|---:|---|---:|---:|
| published (58 cells) | 49 | 2619 | 715.271 | [715.271, 1202.131] | 170.87 | 0.24 |
| round-1 (62 cells) | 57 | 3062 | **993.469** | [463.043, 1212.399] | 231.85 | 0.233 |

theta_m moved **+38.9%**, above Fable 5's 15% convergence threshold (set during the 14b
diagnostic) — the CV stayed roughly flat rather than collapsing the way llama/14b's did, and the
CI floor (previously always equal to the point, a structural property of the reliability fit) for
the first time sat *below* the point, traced to `i1024_o512_c80`'s lowest-TRS window being the
cell's chronologically-first window (rank 0 of 56) — a ramp-up/pre-steady-state artifact, not a
real operating point.

**Fable 5's round-2 prescription** (architect consult, not a broad re-scan): 3 cells, all in
`i1024_o128` only (`i1024_o512` was already cleanly saturated at 100% since c64, no more boundary
information available there) — a fresh **`c64` re-run** as a control (to separate a suspected
cross-run confound from a real regime change behind a 40→17 violation-count dip between the
original session and round 1), plus **`c128`** and **`c160`** to find the actual plateau.

**Round 2 result**: theta_m came back **bit-identical to round 1**: `993.4687597800112`, a 0.0%
move, on a differently-composed cell pool (57 vs 59 cells). The `c64` re-run gave 21/55 violations
— between the original session's 40 and round-1's 17 — confirming the dip was a cross-run
confound (warmup/prefix-cache state), not a real regime change. `c128`/`c160` both saturate
cleanly to 55/55 (100%) violations with TRS collapsing to a median of 375/287, giving `i1024_o128`
the same clean plateau shape `i1024_o512` already had — the open ladder edge the gap audit
flagged is now closed.

| dataset | cells | fit windows | point theta | CI95 | std | CV |
|---|---:|---:|---:|---:|---|---:|---:|
| round-1 (62 cells) | 57 | 3062 | 993.469 | [463.043, 1212.399] | 231.85 | 0.233 |
| round-2 (64 cells) | 59 | 3172 | **993.469 (unchanged)** | [188.906, 1207.525] | 300.42 | **0.302** |

The bootstrap CV widened further (0.233→0.302) despite the point being frozen — Fable 5's read,
after reviewing both rounds: this is a *different* mechanism than llama/14b's CV collapse. Llama/14b
improved because thin boundary-adjacent data was replaced with dense boundary data (genuine
location uncertainty shrinking). 7b's CV widened because the new cells are unambiguously
*non-boundary* (100%-violating, TRS 130-186) — adding them to the cell-resampled bootstrap pool
mechanically increases resample diversity: on the rare draws where the true boundary-anchoring
cell doesn't get sampled, the fit substitutes new, very-low-TRS material instead, pulling the
floor down. This is an expected side effect of confirming the plateau exists, not new evidence
that the point location is uncertain — and it's structurally distinct from the earlier ramp-edge
artifact (verified separately: the actual anchor window defining `993.4687597800112`, found by
exact-match on the TRS value, sits at chronological rank 11 of 56 in its cell — solidly mid-cell,
not an edge case).

**Verdict (Fable 5): converged.** 0% point movement (not just under the 15% threshold) plus a
confirmed plateau plus a mechanistically-explained (not just observed) CV trend together clear the
bar; another round would burn probe budget re-confirming an already-stable number. The ramp-edge
artifact does not gate the write (both rounds' points are demonstrably insensitive to it) but was
flagged as a fast-follow: trim the first window per cell in `rewindow_from_raw.py` / the driver's
ramp phase, then re-verify all three models' bootstrap CIs (not just refit their points) — this
became more urgent after the exp2 rerun below, since the same raw CSVs feed both the bootstrap and
the ranking-separation train/test splits, and 7b's post-recalibration seed-A split explicitly draws
`i1024_o512_c80` (the confirmed ramp-edge cell) into its test fold.

**Registry adoption**: per user decision, all three models' candidate thetas were written together
as one batch — `dsqwen-7b` 715.271→993.4687597800112, `dsllama-8b` 209.647→1290.915, `dsqwen-14b`
247.417→1020.235 — committed `a4fe6e17`, pushed to the live `tre-v2-registry` ConfigMap via
`/api/params` (not `kubectl apply -k`; the git-tracked `deploy/overlays/tre-v2/params.yaml` overlay
is a stale bootstrap copy and is NOT the live update path — confirmed the live ConfigMap already
diverged from it before this change, via the console's PUT/restart API instead), and the controller
was explicitly restarted and verified: both the running pod's mounted `/etc/tre/registry.yaml` and
the live `/api/decision/latest` output show the new theta_m values.

New files in this evidence directory: `theta_fit_7b_convprobe.json`, `bootstrap_theta_7b_convprobe.json`
(round 1), `theta_fit_7b_convprobe2.json`, `bootstrap_theta_7b_convprobe2.json` (round 2).
Large artifacts (node10 only): `/root/tre-experiments/r3_7b_slide_convprobe.csv`,
`/root/tre-experiments/r3_7b_slide_convprobe2.csv`, raw per-request logs under
`/root/tre-experiments/r3_raw/r3_7b_sweep/` (64 cells; original `i1024_o128_c64` raw data backed up
as `*.bak_round1` before the round-2 control re-run overwrote it).

## Post-recalibration experiment reruns (2026-07-10)

With the new registry live, experiments 2 (ranking separation) and 3 (TRE vs APA full comparison)
were rerun to validate the recalibration end-to-end. Full results and Fable 5's analysis:
`docs/refactor/p11_evidence/exp2_ranking_20260710_posttheta/README.md` and
`docs/refactor/p11_evidence/exp3_comparison_20260710_posttheta/README.md`.

Headline: experiment 3's `t5` trace — previously a mutual TRE/APA failure (both ~55.3% violation
rate) attributed to 14b's Z_m never crossing the fairness-loop's receiver-eligibility band — is now
a clean TRE win (31.78% vs APA's 55.35%, -23.57pp) because 14b's peak replica count during t5 rose
1→3, confirmed at the action-log level (`source_loop=fairness`, `reason=low_fairness_sleeping_capacity`).
Experiment 2's AUROC numbers look similar before/after, but the operationally important number
changed: pre-recalibration, the published theta caught **zero** real violations in every held-out
test split that contained any (14b's test splits had none at all, making its "generalizes=True"
vacuous); post-recalibration, 5 of 6 model/seed pairs get nonzero recall for the first time.
