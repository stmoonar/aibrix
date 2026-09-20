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

| model | theta (was) | theta (now) | x live | healthy quantile | in-sample BA | delta_crit (was 0.2515 / 0.44) | delta_high (was 0.6296 / 0.26) |
|---|---|---|---|---|---|---|---|
| dsqwen-7b | 993.469 | 1718.237 | 1.730 | 0.20 | 0.867 | 0.0881 | 0.2967 |
| dsllama-8b | 1290.915 | 1494.662 | 1.158 | 0.20 | 0.861 | 0.1113 | 0.3976 |
| dsqwen-14b | 1020.235 | 1414.082 | 1.386 | 0.15 | 0.902 | 1.0e-06 | 0.6930 |

Reproduction check: running `fit_theta_by_reliability` on the same loaded windows
returns the deployed thetas bit-exactly for all three models, so the difference is the
criterion and nothing else.

dsqwen-14b's `delta_crit` collapses to the clamp (`tau_crit = 0.999999`). That is not a
fit failure — every candidate quantile of its critically-labelled `z` sits above 1.0,
i.e. in R3 its severe violations do not occur at low `Z`. Its LOW band is therefore
empty and anything under `Z = 1` is CRITICAL.

## Validation on E1

`summary.json.e1_validation` scores a single shared threshold `Z < 1` on 3968 E1 windows
(label: `vrate > 0.10`):

| rule | pooled BA | recall | precision |
|---|---|---|---|
| refit theta, `Z < 1` | 0.933 | 0.934 | 0.646 |
| deployed theta, `Z < 0.8` | 0.740 | 0.482 | — |

Per model at `Z < 1`: 7b BA 0.881 / recall 0.905, 8b 0.958 / 0.964, 14b 0.946 / 0.910.
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
