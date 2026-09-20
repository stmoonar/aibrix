# Calibration refit, 2026-09-20 — theta_m and the delta margins

Offline refit only. No cluster run, no deployment, no redis or k8s change was made to
produce this. The numbers here were written into `tre/deploy/registry.yaml` (and its
`tre/deploy/overlays/tre-v2/params.yaml` ConfigMap copy); applying them to the live
cluster is a separate, deliberate step.

## What changed and why

`fit_theta_by_reliability` walks the signal axis until the *cumulative upper set*
reaches 90 % SLO attainment. That is a containment rule, and on the R3 load scans it
lands far below the point where healthy and violating windows actually separate:
**local** attainment at the published theta is only 0.09–0.23.

`fit_theta_by_balanced_accuracy` instead maximises balanced accuracy of the rule
`signal >= theta  =>  SLO met` over quantiles of the healthy-window signal
distribution. The recall floor is **off** (`min_healthy_recall = 0.0`): a 0.90 floor
vetoes the optimum on all three models (its `recall_good` is 0.80–0.85) and reproduces
the same low-theta bias.

`fit_delta_margins` fits `tau_crit = 1 - delta_crit` and `tau_high = 1 + delta_high`
per model from severity-labelled windows, so the control bands are model-specific
instead of the generic 0.2 / 0.25.

The acceptance floors (`min_critical_recall = 0.85`, `min_surplus_precision = 0.80`) are
now **soft**: candidates are ranked by balanced accuracy and the floor only breaks ties
(`delta_floor_mode = "soft"`, recorded in every artifact). They used to be a hard
pre-filter that outranked the objective, which is what collapsed dsqwen-14b's LOW band —
see *The 14b clamp* below. The old behaviour is still reachable with
`--delta-floor-mode strict` and is reported per model under
`comparison.delta_with_strict_floor`.

## Files

| file | contents |
|---|---|
| `summary.json` | all three models side by side, plus the E1 validation block |
| `fit_dsqwen-7b.json`, `fit_dsllama-8b.json`, `fit_dsqwen-14b.json` | full per-model artifact: method string, every knob, fit metrics, delta fit, input provenance |

Each artifact records `method.theta_m_method`, `method.delta_method`, the complete
`fit_config` (including `trim_ramp_windows`), and `inputs` with the CSV path, its
SHA-256, and the window/scenario counts.

## Inputs

| model | CSV | sha256 | windows | cells |
|---|---|---|---|---|
| dsqwen-7b | `/root/tre-experiments/r3_7b_slide_convprobe2.csv` | `91674fa7…39584` | 3172 | 59 |
| dsllama-8b | `/root/tre-experiments/r3_llama_slide_supp.csv` | `49efbc46…a99c4` | 2672 | 51 |
| dsqwen-14b | `/root/tre-experiments/r3_14b_slide_supp3.csv` | `98df06e2…8150c` | 3101 | 56 |

Provenance was pinned by matching `window.count` / `n_cells` / `family_counts` /
`theta` against the shipped `bootstrap_theta_*.json` reports for the deployed values.

**`trim_ramp_windows = 0`, explicitly.** The deployed thetas reproduce bit-exactly only
with 0; `dataset.trim_scenario_ramp_windows` post-dates those fits and the CLI default
of 1 shifts theta by ~3 %. The setting is recorded in every artifact rather than left
to a default.

## Results

| model | theta (was) | theta (now) | x live | healthy quantile | in-sample BA | delta_crit | delta_high |
|---|---|---|---|---|---|---|---|
| dsqwen-7b | 993.469 | 1718.237 | 1.730 | 0.20 | 0.867 | 0.2087 | 0.2967 |
| dsllama-8b | 1290.915 | 1494.662 | 1.158 | 0.20 | 0.861 | 0.2112 | 0.3976 |
| dsqwen-14b | 1020.235 | 1414.082 | 1.386 | 0.15 | 0.902 | 0.1727 | 0.4448 |

`theta` is unchanged by the soft floor — only the margins move. Against the hard-filter
fit that this directory first shipped (and against the pre-refit deployed bands):

| model | delta_crit deployed | delta_crit hard filter | delta_crit now | delta_high deployed | delta_high hard filter | delta_high now |
|---|---|---|---|---|---|---|
| dsqwen-7b | 0.2515 | 0.0881 | **0.2087** | 0.6296 | 0.2967 | **0.2967** |
| dsllama-8b | 0.2515 | 0.1113 | **0.2112** | 0.6296 | 0.3976 | **0.3976** |
| dsqwen-14b | 0.44 | 1.0e-06 | **0.1727** | 0.26 | 0.6930 | **0.4448** |

Every side of every fit now also carries `clamped` / `clamp_reason`; all six are
`clamped: false` in this refit.

Reproduction check: running `fit_theta_by_reliability` on the same loaded windows
returns the deployed thetas bit-exactly for all three models, so the difference is the
criterion and nothing else.

### The 14b clamp

An earlier version of this README said dsqwen-14b's `delta_crit` collapsed to
`tau_crit = 0.999999` because "every candidate quantile of its critically-labelled `z`
sits above 1.0". **That is wrong.** Only 2 of the 19 candidate quantiles (q = 0.90 and
0.95) are above 1.0; the other 17 sit between 0.594 and 0.938, and the severe violations
are spread right across that range.

The real cause was the acceptance floor being applied as a hard pre-filter, ahead of the
objective. With 137 critically-labelled windows the 0.85 floor needs 117 of them
recalled. The candidate quantiles give:

| q | tau | critical windows recalled | balanced accuracy |
|---|---|---|---|
| 0.75 | 0.8273 | 103 / 137 (0.752) | **0.8418** (best) |
| 0.80 | 0.8838 | 109 / 137 (0.796) | 0.8373 |
| 0.85 | 0.9377 | 116 / 137 (0.847) | 0.8343 |
| 0.90 | 1.0344 → clipped to 0.999999 | 119 / 137 (0.869) | 0.8230 |
| 0.95 | 1.1239 → clipped to 0.999999 | 119 / 137 (0.869) | 0.8230 |

q = 0.85 misses the floor **by one window** (116, needs 117). The only candidates that
clear it are the two that got clipped to the `tau_low` bound, so the filter handed the
fit to a clamped candidate: `tau_crit = tau_low - 1e-6`, an empty LOW band, and 0.02 of
balanced accuracy given away for 0.12 of recall. Nothing in the artifact said so —
`used_fallback` was `false` and `reject_reason` was `null`.

Both halves are fixed: the floor is now a tie-break (soft mode) and any `tau` that lands
on or within `1e-5` of a bound is reported as `clamped` with a reason. Under the soft
criterion 14b takes the best-BA candidate, q = 0.75, so `delta_crit = 0.1727` and the LOW
band is a real band again.

## Validation on E1

`summary.json.e1_validation` scores a single shared threshold `Z < 1` on 3968 E1 windows
(label: `vrate > 0.10`):

| rule | pooled BA | recall | precision |
|---|---|---|---|
| refit theta, `Z < 1` | 0.933 | 0.934 | 0.646 |
| deployed theta, `Z < 0.8` | 0.740 | 0.482 | — |

Per model at `Z < 1`: 7b BA 0.881 / recall 0.905, 8b 0.958 / 0.964, 14b 0.946 / 0.910.

**Do not quote 14b's 0.910 as validation.** It rests on 78 violating windows, and all 78
come from a single run, `t5_tre_seed1` — one arm of one experiment, not an independent
sample. 70.5 % of them sit inside the last 5 % below theta (`0.95 <= Z < 1.0`), so the
number is a statement about where that run happened to stop, not about separation: move
the threshold to `theta x 0.95` and 14b's recall falls from 0.910 to **0.205**. The 7b and
8b figures rest on broader support; 14b's needs a second E1 run before it means anything.
This is the first configuration in which `Z = 1` means what the design says it means.

## What this refit does *not* settle

* **`lambda_wait` is left at 3.0 and is unvalidated.** It is not identifiable from this
  data: `avg_waiting` is exactly zero in 81.5 % / 90.7 % / 100.0 % of windows
  (7b / 8b / 14b), because the R3 driver is closed-loop on concurrency and never lets a
  backlog form in vLLM's waiting queue. Sweeping it over [1, 4] moves 14b's signal not
  at all. Re-deriving it needs an open-loop (fixed-rate) sweep.
* **The replica factor is extrapolated, not calibrated.** R3 is a single-instance sweep
  (`assigned_replicas = routable_pods = 1`), while E1 runs at 8 / 8 / 4.
* **`surplus_queue_quantile`'s half of the `delta_high` labels** is driven by
  `queue_raw`, which under a closed-loop driver is essentially `avg_running` alone.
* **`alt_thresholds`** (queue_len / decode_tps / prefill_tps) are still fitted by the
  containment criterion and were not touched here.
