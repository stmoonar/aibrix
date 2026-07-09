# Experiment 3 rerun — TRE vs APA full 7x2, post-recalibration (2026-07-10)

Rerun of the full traceset-v2 comparison (`run_full_comparison_v2_posttheta.sh`, identical
orchestration to the original `docs/refactor/p11_evidence/exp3_comparison_20260709/` run) after
adopting the converged thetas (commit `a4fe6e17`: 7b 715.271→993.469, llama 209.647→1290.915,
14b 247.417→1020.235). All 14 runs passed the infra gate (UF=0, SOCKFAIL=0). Baseline (pre-theta)
results preserved at `/root/tre-experiments/comparison_v2_pretheta_20260709/` on node10 for
direct comparison — trace payloads are byte-identical between the two runs (same `04de30a1`
traceset, verified request counts match, e.g. t5 = 31308 requests both times), so there is no
trace-drift confound between before/after.

## Comparison table (system-level V_req, request-weighted) — before vs after

| trace | axis/tier | TRE before | APA before | verdict before | TRE after | APA after | verdict after |
|---|---|---|---|---|---|---|---|
| t1 | A1 demand_shift / tight | 42.72% | 55.48% | TRE -12.76pp | 42.30%* | ~55%* | TRE (~-12pp, similar) |
| t2 | A2 anticorrelated / loose | 20.01% | 40.22% | TRE -20.21pp | 17.26% | 40.31% | TRE -23.05pp |
| t3 | A3 io_drift / loose | 2.87% | 50.14% | TRE -47.27pp | 1.87% | 49.64% | TRE -47.77pp |
| t4 | A4 spike_vs_burst / loose | 8.83% | 19.48% | TRE -10.65pp | 4.94% | 19.52% | TRE -14.59pp |
| **t5** | **A5 tp_pressure / tight** | **55.34%** | **55.33%** | **tie (both fail)** | **31.78%** | **55.35%** | **TRE -23.57pp** |
| t6 | A6 control | 0.01% | 0.01% | tie | 0.01% | 0.00% | tie |
| t7 | A2b anticorrelated_hot / tight | 40.46% | 52.93% | TRE -12.47pp | 38.92% | 52.79% | TRE -13.88pp |

**Score: 5/7 TRE wins before → 6/7 after** (t5 flips from a mutual-failure tie to a clean win; t6
stays a tie by construction — near-zero baseline).

## t5 fix — confirmed at the action-log level, not just the score

`dsqwen-14b` peak replica count during t5 rose **1→3** (tp=2, so 6 of 8 GPU slots), first scale-up
at t=157s (previously: never scaled). Cross-referenced against `planner.py`: the action fires via
`source_loop=fairness`, `reason=low_fairness_sleeping_capacity`. `planner.py` has an explicit
ADR-0014 comment that fairness-loop receiver eligibility is Z_m-*band*-gated only (CRITICAL/LOW) —
i.e. directly a function of `theta_m`. This confirms the standing hypothesis from HANDOFF's open
item #1 ("14b's Z_m/theta never crosses delta_high"): the old theta (247.417, since found to be
4.1x too low) kept 14b's Z_m reading out of the receiver-eligible band even under real capacity
tension; the corrected theta (1020.235) fixes this directly, not as a side effect.

## Replica-count changes beyond t5 — same mechanism, confirmed genuine

| trace | model | peak before | peak after | mechanism |
|---|---|---|---|---|
| t2, t7 | dsllama-8b | 1 | 2 | `fairness` / `low_fairness_sleeping_capacity` (same pathway as t5) |
| t4 | dsqwen-7b | 3 | 4 | `rescue` / `critical_sleeping_capacity` (fast emergency loop) |
| t4 | dsllama-8b | 1 | 2 | `fairness` (as above) |
| t1, t3, t6 | all | unchanged | unchanged | — |

Llama getting most of its fix through the slower fairness loop rather than the sharp rescue trigger
is consistent with llama's theta moving the most of the three (6.16x) — a bigger jump shifts more
Z_m readings into the eligible band across a wider load range, which a 10s-cadence rebalancing loop
picks up more readily than a single-shot emergency trigger.

**Not fixed by recalibration** (separate, still-open issue): `dsqwen-7b`'s peak on t1/t7 stays at 4
replicas, not the physically-feasible 5 (8 slots − llama's 1-2 − 14b's tp2 2). This is HANDOFF's
planner capacity-allocation item (open item #2), unrelated to calibration.

## Performance-source attribution — mostly t5, rest is real but smaller

APA's own numbers are theta-independent and were run on identical trace payloads, so APA's
before→after drift is a direct empirical noise floor:

| trace | TRE Δ improvement (pp) | APA Δ (≈ noise floor) | ratio | read |
|---|---:|---:|---:|---|
| t1 | 0.45 | 0.06 | ~7x | borderline noise |
| t2 | 2.75 | 0.09 | ~30x | real |
| t3 | 1.00 | 0.50 | ~2x | weak signal, inconclusive |
| t4 | 3.89 | 0.04 | ~97x | real |
| **t5** | **23.56** | 0.02 | **real, dominant** | **~71% of total improvement** |
| t6 | 0.00 | 0.01 | flat | control, as expected |
| t7 | 1.54 | 0.14 | ~11x | real |

t5 alone accounts for ~71% (23.56 / 33.19pp) of the aggregate improvement across all 7 traces, and
is the only trace that flips a verdict category — that's why the win count moves 5/7→6/7. t2/t4/t7
are real (10-100x the observed noise floor) and line up with the confirmed replica-count changes
above. t1 is borderline (~7x noise, and its replica counts didn't change) — read as a secondary
effect of recalibration on continuous Z_m-driven decisions (safescale timing, hidden-routing)
rather than a discrete scale event; don't lean on it hard. t3's APA arm itself moved 0.50pp on an
identical trace (the largest noise observation in the set), so t3's 1.00pp TRE improvement is weak
signal, not a confident recalibration win — flagged as inconclusive, not claimed.

**For any writeup: lead with t5 as the headline (matches the original hypothesis, now confirmed at
the log level), describe t2/t4/t7 as a real secondary effect, and do not claim t1/t3 as
recalibration wins** — they sit within or barely above this comparison's own measured noise floor.

## Next optimization directions (Fable 5)

1. **7b's replica cap on t1/t7 (4 vs feasible 5)** — now the clearest remaining gap once t5's
   confound is cleared; TRE still beats APA there but runs at 40-42% internal violation rate with
   headroom left on the table. Planner capacity-allocation logic, not calibration.
2. **14b-seed-a's 0/7 recall (exp2)** — don't treat the new thetas' safety guarantee as fully
   validated on the tight/high-concurrency tail yet; n=7 is too small either way. Get more
   violation-labeled test coverage there before this ranking result is used to justify any
   tau_high/tau_crit tuning.
3. **Ramp-edge window trim** — more urgent now than a documentation cleanup, since the same raw
   CSVs feed both the bootstrap CI and exp2's train/test splits (7b's post-recalibration seed-A
   split explicitly draws the confirmed ramp-edge cell `i1024_o512_c80` into its test fold). Land
   before the next bootstrap or ranking-separation run.
4. **t1/t3 noise band** — APA's own t3 drift (0.50pp) was larger than expected for a supposedly
   theta-independent arm; a cheap repeated-seed rerun of just t1/t3/t6 would confirm whether the
   noise floor is stable (~0.1pp) or t3's 0.50pp was itself an outlier worth understanding.

## Files here

`final_report.json`, `timeline_{tre,apa}.csv`, `summary_{tre,apa}_{t1..t7}.json`. Full markdown
report: `/root/tre-experiments/comparison_v2_posttheta/final_report.md` (node10 only). Baseline
(pre-theta) evidence preserved at `docs/refactor/p11_evidence/exp3_comparison_20260709/` and
`/root/tre-experiments/comparison_v2_pretheta_20260709/`.
