# Next calibration round: hybrid attribution interface (2026-10-05)

Contract between the four branches of the next round (`calib/next-20261005` = attribution +
integration, `calib/next-fit-gate-20261005`, `calib/next-m2-20261005`,
`calib/next-t14-20261005`). Design: local workspace
`docs/calib-next-round-design-20261005.md`. Code against this file; anything not listed here
is unchanged.

## 1. The attribution

| value | label | TTFT sample + min-n count (`completed_requests`, `ttft_len_samples`) | TPOT / e2e samples | unserved |
|---|---|---|---|---|
| `completion` | v1 (every label so far) | window of `done_ts_ms` | window of `done_ts_ms` | window of `send_ts_ms` |
| `hybrid` | v2 | window of the first token (`recv_first_token_ts_ms`) | window of `done_ts_ms` | window of `send_ts_ms` |

Window membership is unchanged (`(start, end]` on the grid-aligned fitting windows). The TSS
token numerator (gateway, count at completion) and every queue / signal column do **not**
depend on the attribution.

Constants: `tre_common.slo_labels.ATTRIBUTION_COMPLETION = "completion"`,
`ATTRIBUTION_HYBRID = "hybrid"`, `ATTRIBUTIONS`, `ATTRIBUTION_RULES` (text per value).

## 2. Label definition (`tre_common.slo_labels.LabelDefinition`)

* New field `attribution: str = "completion"` (last field, keyword). Property `.hybrid`.
* `as_dict()`:
  * completion: **byte-identical** to the v1 record (no new key; name
    `p95_ttft_slowdown_tpot_plus_unserved_v1` / `p95_ttft_tpot_plus_unserved_v1`), so every
    existing `label_def_sha256` still matches;
  * hybrid: name `p95_ttft_slowdown_tpot_plus_unserved_v2` (fixed mode:
    `p95_ttft_tpot_plus_unserved_v2`) plus keys `attribution: "hybrid"`, `attribution_rule`,
    `min_n_counts`.
* Identity / sha: unchanged rule, `dline_refit.canonical_sha256(label.as_dict())` (the
  `label_def_sha256` of verdicts, freezes and M / T14 manifests). A hybrid label therefore
  has a different sha, and an M / T14 manifest sealed under it refuses a v1 freeze and vice
  versa (existing `check_m_manifest` logic, no change needed).
* `LabelDefinition.from_dict(record)` reads `attribution` (absent = completion).
* `label_arms(primary)` keeps the primary's attribution on every arm.
* Builders: `slo_labels.label_def_for_model(..., attribution=None)` (None = completion);
  `label_def_from_args` reads `args.label_attribution`; CLI flag
  `--label-attribution {completion,hybrid}` in `add_label_arguments` (so `r3_grid`, the fit
  CLIs and `rewindow_from_raw` accept it); `label_cli_args` emits it only for hybrid (a v1
  command line is unchanged).
* Campaign: `calibration_campaign --fit-label-attribution {completion,hybrid}` (default
  completion) -> `campaign.primary_label(args, model)` builds the probe / fit label with it.
  **For the fit-gate / M2 / T14 branches**: `dline_refit.label_for(...)` and every
  "our label == frozen label" check must pass the attribution through
  (`label_def_for_model(..., attribution=...)`); take it from the freeze's
  `label_def["attribution"]` (absent = completion) or from the datasets (section 3), never
  default silently to completion when the freeze says hybrid.

## 3. Datasets (`scripts.calibration_dataset`)

* `python -m scripts.calibration_dataset RUN --attribution hybrid` (default `completion`).
* Default output dir: `RUN/dataset_hybrid/` (with `--numerator vllm_counter`:
  `RUN/dataset_l3_hybrid/`); completion stays `RUN/dataset/` (`dataset_l3/`).
  `calibration_dataset.default_dataset_dirname(settings)` returns the name.
* Manifest: top-level `"attribution": {"value": "completion" | "hybrid", "rule": <text>}`.
  A manifest without the key = `completion`. Helper:
  `calibration_dataset.dataset_attribution(directory) -> str` (dataset dir or run root).
  `label` / `label_by_model` carry the label record of section 2 (so also `attribution`).
* `windows.csv`: **no column change** (same `WINDOW_COLUMNS`); the attribution lives in the
  manifest and the label record only. Under hybrid, `completed_requests`,
  `ttft_len_samples` and `p95_ttft_client_ms` are over the TTFT set; `p95_tpot_client_ms`,
  `p95_e2e_client_ms`, token totals and signal columns are as in the completion dataset.
* `requests.csv`: unchanged; the per-request first-token instant is already its
  `first_token_ts_ms` column (raw `recv_first_token_ts_ms`), next to `send_ts_ms` and
  `done_ts_ms`.
* Downstream checks expected from the other branches: `dline_refit trainset` refuses
  sources of two attributions (like two numerators) and records `attribution` in
  `trainset.json`; `accept` and the T14 scorer refuse a dataset whose
  `dataset_attribution(...)` differs from the frozen label's attribution.
* End of a collection run: `campaign.finalize_run` builds `dataset/` as before and, when
  the run's `plan.json` label (`plan["label"]["label_def"]`, else `plan["label_def"]`) is
  hybrid, also `dataset_hybrid/` (`campaign.planned_attribution(out_dir)`).

## 4. Re-windower (`scripts.rewindow_from_raw`)

* `label_cell(...)` reads the attribution from the label it is given
  (`rewindow_from_raw.label_attribution(arms)`); no new argument for callers.
* Lower level: `aggregate_window(..., attribution=)`, `window_request_evidence(...,
  attribution=)`, `rewindow_cell(..., attribution=)`, `ttft_requests(records, start, end,
  closed_right=)` (served requests whose first token is in the window).

## 5. Not changed

Controller / online path (reads no label), gateway numerator, v1-lambda, the w_p 1-SE rule,
EMA tau 10 s, dwell 1, window length / step, the slowdown TTFT SLO, min-n = 20.
