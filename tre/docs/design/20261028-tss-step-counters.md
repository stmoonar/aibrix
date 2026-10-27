# TSS numerator from vLLM step counters, 20 s window (2026-10-28)

Status: implemented on `feat/tss-counter-20261028` (not deployed). User decisions
(2026-10-28):

1. The TSS numerator switches from the completion histograms (`request_prompt_tokens` /
   `request_generation_tokens` `_sum`: tokens of the requests that COMPLETED in the
   window) to vLLM's per-step counters `vllm:prompt_tokens_total` /
   `vllm:generation_tokens_total` (tokens the engine PROCESSED in the window).
2. The metrics window goes from 30 s to 20 s.
3. The EMA stays (`trs.ema_tau_ms` 10 s, unchanged).
4. The recalibration on the new definition skips the acceptance procedure (no M rounds,
   no onset gate): the fitted theta / lambda / tau / w_p are used directly.

## Definition

Per pod and window `(start, end]` (`MetricsStore`):

    tokens = (C(newest hist doc) - C(baseline doc)) / (scraped_ms(newest) - scraped_ms(baseline)) x (end - start)

- baseline = the newest histogram doc at or before the window start (else the first in it),
  as for every other delta; `scraped_ms` is the gateway's observation time.
- The numerator stays a WINDOW TOTAL (tokens per window). theta is therefore tied to the
  window length and to the numerator source.
- Counter reset (a counter lower than in the previous doc: the engine restarted): the delta
  restarts at the first doc after it, the pod is flagged (`sample_incomplete_pods`). vLLM
  keeps its counters across sleep / wake, so a sleeping pod reads flat counters (no reset).
- A counter missing at the baseline or the newest doc: the pod's tokens are unknown. The pod
  is left out of queue AND tokens and flagged; a model with no pod left has unknown tokens
  (no signal), never zero.
- Requests stay a completion measure: `request_count(_observed)` = the
  `request_prompt_tokens` count delta (O1 `min_evidence_requests`, SafeScale). The mean
  prompt length (SafeScale's length-dependent TTFT threshold) = `completed_prompt_tokens`
  (the histogram sum) / that count. It never uses the counter total.
- vLLM adds generated tokens every engine step and prompt tokens once at the first token
  (`vllm/v1/metrics/stats.py` `update_from_output`): chunked prefill and preemption recompute
  are not counted twice (10-27 check: counter / client = decode 0.9996-1.015, prompt
  1.006-1.016).

The gateway writes the two counters on every histogram doc under `model_counter_metrics`
(`pkg/cache/cache_tre_redis.go`; same boundary and `scraped_ms` as the histograms; a
non-finite value is left out). It already scraped them (`counterGaugeMetricNames`).

## Offline check (before implementing)

Replay of the 16 TRE runs of the 10-26 campaign (`campaign-v2-20261026{,-tre2,-tp-seeds}`,
1 Hz `pod_metrics_1s` counters, scripts of the 10-27 replay, open loop). Counter theta =
live theta x rho (rho = counter / completion steady ratio .995 / .994 / 1.013), numerator in
30 s units so one theta serves both windows. 23 arrival steps for crossings.

| Variant | HIGH entries | Receiver entries | False-receiver share | Jitter med / p90 | Cross tau_low med (p25-p75) | Cross tau_crit med (n/23) | Z/settled after scale-up 0-6 s | after scale-down 0-6 s |
|---|---|---|---|---|---|---|---|---|
| completion, 30 s, EMA (today) | 168 | 130 | .54 | .014 / .074 | 29 s (23.5-31) | 37 s (16) | .61 | 1.21 |
| counter, 30 s, EMA | 135 | 87 | .39 | .008 / .045 | 34.5 s (27-39) | 68.5 s (12) | .83 | 1.08 |
| **counter, 20 s, EMA (chosen)** | 155 | 86 | **.30** | .009 / .049 | **32 s (25.5-37.5)** | 64 s (12) | .75 | 1.15 |
| completion, 30 s, raw (ref) | 315 | 165 | .57 | .030 / .100 | 19 s | 24 s (19) | .50 | 1.41 |
| counter, 20 s, raw (ref) | 238 | 131 | .44 | .015 / .065 | 21 s | 29 s (14) | .69 | 1.21 |

- Paired with today: tau_low crossing +3 s (median; later in 21/23 steps); counter 30 s:
  +6.5 s. Harm (first violations) median 18.5 s after the step.
- The chosen config is less noisy than today and has half the O1 bias, and crosses
  tau_low about 3 s later. Not a correctness problem, so the decision stands.
- tau_crit crossings are much later and rarer (12/23 vs 16/23). The completion numerator's
  early dip at a step (queue up, completions late) is gone by construction. The
  recalibrated tau_crit will move. Watch the fast-loop (CRITICAL) path after the release.
- Open-loop, no feedback. Steps come from 2 traces only.

## Inventory

(a) Numerator source

| Place | Before | Now |
|---|---|---|
| gateway `pkg/cache/cache_tre_redis.go` | hist docs: 5 histograms | + `model_counter_metrics` (2 counters) |
| controller `store/metrics_store.py` | `_hist_sum_delta` of the token histograms; reset on histogram sums | `_counter_delta`; reset on the counters; missing = pod left out; `completed_prompt_tokens` |
| `tre_common/metrics_schema.py`, `window_pods.py` | - | `completed_prompt_tokens`, `mean_completed_prompt_tokens()` |
| SafeScale mean prompt length (`safescale_task.py`, `planning/safescale_evidence.py`) | `prompt_tokens / request_count` | `completed_prompt_tokens / request_count_observed` |
| `tre_common/tss.py` | completion, `V2_WINDOW_MS` 30 s | step counters; `theta_basis_errors`; `V2_WINDOW_MS` removed (`convert_window_total_theta` takes the window) |
| registry `trs` (`registry.py`, `registry.yaml`, `params.yaml`, `registry_params.py`) | - | stamps `numerator`, `theta_window_ms`; editable in `params_set.py` |
| calibration `deploy/scripts/l3_numerator.py` | L3 = unscaled boundary delta, 2 s interior-gap void | span-scaled (online rule); interior holes <= 10 s and failed scrapes no void |
| `run_freeze_m2.sh`, `dline_refit.py` | gateway numerator, 30 s | `NUMERATOR` (default `vllm_counter`), `WINDOW_MS` 20000, `STEP_MS` 10000 through every stage; window mismatch refused; freeze `registry` block stamps `numerator` / `theta_window_ms` |
| `signals/saturation.py` `numerator_zero` | "nothing completed" (every onset) | "no token processed" (rare) |
| alt signals `decode_tps` / `prefill_tps` | completion tokens per s per replica | counter tokens per s per replica; their `alt_thresholds` were fitted on completion (ablation arms only) |
| unchanged | | replayer / eval scoring windows (request windows, not TSS); baselines (own counter scrapes); `tre:v2:*` keys (`rediskeys.py`); the v1 legacy schema has no counters (tokens unknown: v2 only) |

(b) Window length

- Config: `TRE_METRICS_WINDOW_MS` (controller env, `overlays/tre-v2/controller.yaml`)
  = 20000; config default 20_000. The decision layer reads each window's own bounds. The
  guard `test_no_period_in_decision_layer.py` now also forbids `metrics_window` /
  `METRICS_WINDOW` there.
- theta coupling: each model's `trs.numerator` / `trs.theta_window_ms` must equal what the
  controller computes, else it refuses to start (`create_controller_dependencies`). An old
  registry with the new image therefore fails loudly instead of shifting every Z by 2/3.
- EMA: tau (10 s) is not tied to the window. Two things are:
  - the EMA idle-gap reset threshold is the window, so it is now 20 s;
  - `registry_params.py` locks `ema_tau_ms` as "window-coupled; frozen with W". The coupling
    is to the 10 s grid (alpha .632) and the onset gate's 20 s lag budget
    (`dline_refit.ONSET_TAU_S`), which is skipped now. Not changed.
- O1 `scaling.min_evidence_ms` = 20000: a suffix is strictly shorter than the 20 s window,
  so it never reaches 20 s. The model then waits for the first whole window after the
  breakpoint, about the same 20-22 s as today. The suffix branch is effectively dead.
  **Not changed; user decision** (keep, or lower to e.g. 10 s).
- `scaling.min_evidence_requests` = 3 was added for the completion numerator. Kept;
  **user decision** whether to drop it.
- Relative rules adapt by themselves: token span [0.67, 1.5] x window, coverage
  `sample_min_ratio`, SafeScale tail suffix, H3, hold fallback 50 s.
- Calibration: dataset `--window-ms`, dline_refit `windowing`, accept check C independence
  = window / step (2 at 20/10).

## Registry values on this branch

`registry.yaml` / `params.yaml` carry PROVISIONAL thetas: theta(completion, 30 s) x 20/30 x
rho, tau unchanged, stamped `step_counters` / 20000. This is a unit conversion, not a
calibration. It keeps the branch consistent and testable. The recalibration below replaces
it before any data-producing run. (The repo copy is the 10-05 freeze; live is the 10-25
freeze.)

## Recalibration (offline refit; user 2026-10-28: no acceptance round)

**No new cluster data is needed for theta / lambda / tau / w_p.**

1. Data: `calib_20261003`. Every training cell has the 1 Hz per-pod capture
   `cells/*/vllm_metrics_1hz/`: run1 229, run2 398, run2b 182, supp 28, p1 40, p1r2 18
   cells (M 103). The capture holds every `vllm:*_total` counter, the
   `kv_cache_usage_perc` gauge and the TTFT / ITL / e2e / token histograms per pod. T14 has
   no 1 Hz capture; it is not used (no acceptance).
2. Void windows at 20 s (1 Hz gaps), counted over the capture spans:
   - before the 10-28 rule change: run1 / run2 / run2b / supp 0-0.6 %, M 5 %, p1 12.5 %,
     p1r2 21 %. The voided windows are deep overload, i.e. violated ones, so the fit was
     biased;
   - after it (only boundary age <= 2 s and holes > 10 s void): p1 2.8 %, p1r2 3.1 %,
     others <= 0.3 %;
   - dataset build check on run1 / dsqwen-7b: 77 / 1650 windows `no_sample` (cell edges),
     counter / gateway numerator median 1.005 (p10 .87, p90 1.27).
3. The 20 s windowing and the EMA are recomputed offline from the raw series (dataset
   builder `--window-ms 20000`, then dline_refit alpha (tau 10 s) / wp / final). This
   needs no cluster, no exclusive window and no GPU. It is CPU only: about 20 s per
   dataset and minutes per fit.
4. Run from a clean checkout of the merged sha:

       cd tre/deploy && CALIB_ROOT=/data/nfs_shared_data/xxy/calib_20261003 \
         OUT_DIR=/data/nfs_shared_data/xxy/calib_20261003/freeze-ctr20-<date> MODE=final \
         NUMERATOR=vllm_counter WINDOW_MS=20000 STEP_MS=10000 QUEUE_ESTIMATOR=raw_mean \
         FIT_METHOD=m2_unique_grid_plateau_midpoint PUBLISH_TAU_S=10 \
         scripts/calibration/run_freeze_m2.sh        # no ACCEPT_M_ROOT

   Before the real run, check the cell list: at 20 s one probe cell's verdict flips
   (`i256_o128_c1175`, rho 1.75).
5. Outputs per model: theta_m, lambda_wait, w_p, tau_crit, tau_high, and the stamps
   `numerator: step_counters`, `theta_window_ms: 20000` in the freeze `registry` block.
   - Unchanged: c / b (idle TTFT fit; the label, independent of the numerator), tau_low
     1.0, ema_tau_ms 10 s.
6. Alt-signal thresholds (queue_len, decode_tps, prefill_tps, kv / ttft_mean): only for
   ablation arms. `windows.csv` has no KV-usage or per-pod mean-TTFT column, but the 1 Hz
   capture has both, so a dataset-builder extension can add them offline (not done). A new
   capture is not needed for them either.
7. Validation instead of acceptance: replay the frozen values on the 10-26 campaign
   `pod_metrics_1s` (1 Hz counters, 0 resets / 0 gaps in 16 runs), with the metrics of
   the offline check above. Report; no gate.

### Atomic release (after the freeze; user approval, two batches apart)

1. Back up to `backups/<name>/` with `rollback.sh`, and notify parallel sessions.
2. `set_run_mode.sh observe observe`.
3. Build `gateway-plugins`, `service-manager` and `controller` from the merged sha (tag
   `<date>-<sha>`). The overlay tags and `test_kustomize_overlays.py` change together.
   `controller.yaml` carries `TRE_METRICS_WINDOW_MS=20000`.
4. gateway-plugins first. Check that the `tre:v2:hist:*` docs carry `model_counter_metrics`.
   The old controller ignores the field.
5. `params_set.py --set <m>.trs.{theta_m,lambda_wait,w_p,tau_crit,tau_high}=... --set
   <m>.trs.numerator=step_counters --set <m>.trs.theta_window_ms=20000 --apply
   --backup-dir ...`. Use one `--apply` for all models and do not restart the old
   controller after it.
6. SM (Recreate), then controller (new image + overlay env). The new controller refuses
   to start if a stamp is missing. That is the check that steps 5-6 happened together.
7. `set_run_mode.sh observe active` (canonical), then a few-minute smoke: no
   `samples_incomplete` storm, Z defined while busy, O1 holds about 20 s.

## Not done / follow-ups

- Live registry ConfigMap, images, cluster: untouched.
- `min_evidence_ms` / `min_evidence_requests`: see above (user decision).
- `numerator_zero` saturation branch and the O1 suffix branch are now mostly dead code.
  They are candidates for the controller simplification.
- Alt-threshold refit for decode_tps / prefill_tps if an ablation arm uses them.
