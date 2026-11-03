# PreServe mLSTM predictors trained on each trace's own past, W = 10 s

Created 2026-11-03T19:59:24+00:00 by `preserve_train/trace_train.py finish` (aibrix d581d5bd56c503ed56b57a92d8d0282ceb1c09ee, dirty False); trained 2026-11-03T20:01:44+00:00 on nscc-ds-4a100-node9 (aibrix d581d5bd56c503ed56b57a92d8d0282ceb1c09ee, dirty False), torch 2.13.0+cu130, numpy 2.2.6, CPU. Config `tre/baselines/preserve_train/configs/trace-train-20261104.json` (sha256 `c239c7e2aeaa076c1d0f4c13db2631e7c0b2ff9accafd84540cc8233b23c2ecc`).

## What

The paper trains a service's predictor on that service's own past (section 5.3 RQ2: first two days -> next two). Here the past of an evaluation trace = independent realizations of the same service made by the trace's own generator and spec with seeds no evaluation uses (101, 102, 103, 104; evaluation seed 1 is never trained on): 4 x 72 windows of 10 s = 288 windows (paper: 2 days of 10 min windows = 288).

- **V1 per trace** (`v1/<trace>.<model>.npz`, 21 files): training series = the trace's 4 realizations concatenated in seed order (288 windows, 263 sequences).
- **Real_* (user 2026-11-04)**: V1 of Real_conv / Real_code is trained on the REAL past of the slice instead of seed realizations: per model the 4 Azure intervals right before the slice's own source interval, each through the slice's r3e transform and generator (table below); training order past4 .. past1, so the history before t0 (end of the training series) is the true pre-context of the slice.
- **V2 concat** (`v2/concat.<model>.npz`, 3 files): training series = all 7 traces concatenated (config order: Alternating_hot_model_periodic_A, Simultaneous_spike_ramp_twice_tps1o2, Prefill_mixed_corner_decode_mix, Decode_heavy_burst, Sinusoidal_demand, Real_conv_2023_slice_a_tok70, Real_code_2024_slice_a_tok70; each trace's realizations in seed order; 2016 windows; sequences run across trace boundaries).
- Hyper-parameters: the 11-03 protocol (seq_len 25, hidden 64, 50 epochs, batch 32, Adam 0.001, MSE, seed 42) except normalization = **min-max (paper Alg. 1)** over the training series; min / max are stored in the file (`<kind>.norm_min`, `<kind>.norm_max`, `normalization` = `minmax`) and applied at inference; the policy refuses a file whose normalization differs from `params.predictor_normalization` (default `per_sequence_mean`, the 11-03 weights).
- History before t0: `preserve_history_w10_train-tail-20261104.json` (policy `history_variant: train-tail-20261104`), method `train-realization-tail-v1-20261104`: the last 25 windows of the trace's training series (= the end of the last training realization; the evaluation realization is the next one). Files under `/data/nfs_shared_data/xxy/datasets/preserve-train-trace-20261104/history/` (sha256 in manifest.json).

## One-step-ahead MAPE (%) on the evaluation realization (seed 1)

Every window i of the 72 is predicted from the last 25 of [history before t0, observed windows < i] (the policy's Tier-1: observed windows enter the history, predictions never do; prediction clamped at 0). MAPE = mean |y - yhat| / y over windows with y > 0. `current` = the shipped `preserve-mlstm-20261103-paper-w10` set the policy picks (Real_code -> code, Real_conv -> conv, else pooled; per_sequence_mean) with the trace's current `preserve_history_w10.json`; `naive` = the previous window (window 0: the last row of the new history). Series: P = prompt_length, D = max_output_tokens (null = loadgen config max_tokens) by arrival, windows from the first arrival.

`*_2step` = the two-step forecast of window j made at the start of window j - 1 (observed windows < j - 1, predict j - 1, append it, predict j: paper Alg. 2, what the policy's Tier-1 now uses to set N_{i+1}); scored on windows 1..71 (window 0 would be forecast before t0). Naive two-step = window j - 2.

D here is the request's max_output_tokens: under the runs' ignore_eos the actual output tokens equal max_tokens, so this is also the paper's D (completed output tokens attributed to the arrival window).

### Total (P + D) per trace, mean over the 3 models

| trace | V1 | V2 | current | naive | V1 2-step | V2 2-step | current 2-step | naive 2-step |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Alternating_hot_model_periodic_A | 5.2 | 7.6 | 26.3 | 10.0 | 7.6 | 14.2 | 34.5 | 17.9 |
| Simultaneous_spike_ramp_twice_tps1o2 | 9.1 | 13.5 | 22.4 | 10.9 | 11.5 | 25.7 | 29.1 | 19.9 |
| Prefill_mixed_corner_decode_mix | 6.8 | 8.1 | 25.4 | 8.5 | 11.6 | 15.6 | 32.6 | 15.0 |
| Decode_heavy_burst | 2.7 | 4.5 | 7.1 | 1.7 | 4.0 | 8.9 | 9.8 | 2.7 |
| Sinusoidal_demand | 1.2 | 3.0 | 15.5 | 8.5 | 1.1 | 5.4 | 21.1 | 16.7 |
| Real_conv_2023_slice_a_tok70 | 10.9 | 9.9 | 32.0 | 8.5 | 18.2 | 17.3 | 49.4 | 12.8 |
| Real_code_2024_slice_a_tok70 | 14.2 | 12.9 | 36.4 | 12.0 | 22.2 | 21.6 | 48.7 | 16.9 |
| **all 21** | **7.1** | **8.5** | **23.6** | **8.6** | **10.9** | **15.5** | **32.2** | **14.6** |

### Per trace and model (P / D / total)

| trace | model | current set | V1 | V2 | current | naive | V1 2-step | V2 2-step | current 2-step |
|---|---|---|---|---|---|---|---|---|---|
| Alternating_hot_model_periodic_A | dsqwen-7b | pooled | 4.4 / 4.4 / 4.4 | 7.8 / 6.3 / 6.2 | 25.8 / 28.6 / 27.0 | 8.9 / 8.9 / 8.9 | 6.9 / 6.9 / 6.9 | 15.1 / 11.0 / 11.8 | 34.4 / 36.5 / 35.3 |
| Alternating_hot_model_periodic_A | dsllama-8b | pooled | 5.8 / 5.8 / 5.8 | 10.0 / 7.2 / 8.6 | 24.9 / 27.7 / 26.1 | 11.3 / 11.3 / 11.3 | 7.2 / 7.2 / 7.2 | 18.4 / 11.3 / 14.5 | 33.8 / 35.5 / 34.5 |
| Alternating_hot_model_periodic_A | dsqwen-14b | pooled | 5.4 / 5.4 / 5.4 | 10.6 / 7.0 / 8.2 | 24.2 / 27.7 / 25.7 | 10.0 / 9.9 / 9.9 | 8.5 / 8.6 / 8.6 | 22.3 / 10.4 / 16.2 | 32.4 / 35.2 / 33.6 |
| Simultaneous_spike_ramp_twice_tps1o2 | dsqwen-7b | pooled | 8.8 / 8.8 / 8.8 | 11.4 / 10.7 / 11.1 | 19.3 / 21.7 / 20.2 | 9.7 / 9.7 / 9.7 | 11.3 / 11.3 / 11.3 | 19.6 / 19.3 / 19.1 | 25.7 / 26.7 / 26.0 |
| Simultaneous_spike_ramp_twice_tps1o2 | dsllama-8b | pooled | 8.4 / 8.4 / 8.4 | 14.6 / 11.7 / 13.0 | 20.4 / 23.1 / 21.4 | 10.7 / 10.7 / 10.7 | 10.5 / 10.5 / 10.5 | 29.0 / 20.2 / 24.7 | 27.7 / 28.9 / 28.1 |
| Simultaneous_spike_ramp_twice_tps1o2 | dsqwen-14b | pooled | 10.1 / 10.1 / 10.1 | 18.6 / 14.4 / 16.6 | 24.2 / 27.5 / 25.5 | 12.4 / 12.4 / 12.4 | 12.6 / 12.6 / 12.6 | 36.9 / 27.6 / 33.3 | 32.5 / 34.6 / 33.3 |
| Prefill_mixed_corner_decode_mix | dsqwen-7b | pooled | 7.9 / 7.0 / 6.5 | 7.4 / 8.0 / 7.3 | 38.3 / 21.8 / 31.6 | 10.4 / 9.3 / 10.1 | 13.7 / 10.1 / 10.5 | 14.3 / 11.8 / 12.8 | 48.9 / 26.9 / 39.9 |
| Prefill_mixed_corner_decode_mix | dsllama-8b | pooled | 9.5 / 6.1 / 6.6 | 12.5 / 7.4 / 8.9 | 32.9 / 14.4 / 25.3 | 9.3 / 7.2 / 8.1 | 18.7 / 7.5 / 10.7 | 26.0 / 10.0 / 17.2 | 43.5 / 17.6 / 33.0 |
| Prefill_mixed_corner_decode_mix | dsqwen-14b | pooled | 7.1 / 8.5 / 7.2 | 10.5 / 7.6 / 8.1 | 18.6 / 18.3 / 19.4 | 6.0 / 8.4 / 7.2 | 14.3 / 13.8 / 13.8 | 24.0 / 12.1 / 16.8 | 25.1 / 22.0 / 24.8 |
| Decode_heavy_burst | dsqwen-7b | pooled | 2.6 / 2.5 / 2.5 | 4.5 / 3.7 / 3.3 | 6.5 / 7.4 / 7.0 | 1.7 / 1.6 / 1.6 | 3.8 / 3.8 / 3.8 | 8.7 / 7.5 / 6.1 | 9.4 / 9.8 / 9.6 |
| Decode_heavy_burst | dsllama-8b | pooled | 2.5 / 2.5 / 2.5 | 9.7 / 2.7 / 6.2 | 6.0 / 6.9 / 6.4 | 1.8 / 1.7 / 1.8 | 3.6 / 3.6 / 3.6 | 21.6 / 5.0 / 13.1 | 8.7 / 9.0 / 8.8 |
| Decode_heavy_burst | dsqwen-14b | pooled | 3.0 / 3.0 / 3.0 | 5.3 / 4.3 / 3.9 | 7.5 / 8.5 / 8.0 | 1.9 / 1.8 / 1.8 | 4.5 / 4.5 / 4.5 | 10.5 / 7.8 / 7.4 | 10.8 / 11.2 / 11.0 |
| Sinusoidal_demand | dsqwen-7b | pooled | 1.1 / 1.1 / 1.1 | 3.1 / 2.5 / 2.6 | 15.1 / 17.0 / 15.9 | 8.6 / 8.7 / 8.6 | 1.0 / 1.0 / 1.0 | 5.6 / 3.6 / 4.2 | 21.3 / 22.0 / 21.6 |
| Sinusoidal_demand | dsllama-8b | pooled | 1.0 / 1.0 / 1.0 | 3.5 / 2.0 / 2.7 | 15.9 / 17.8 / 16.6 | 9.7 / 9.7 / 9.7 | 0.9 / 0.9 / 0.9 | 7.2 / 3.1 / 5.2 | 22.5 / 23.2 / 22.8 |
| Sinusoidal_demand | dsqwen-14b | pooled | 1.5 / 1.5 / 1.5 | 4.3 / 3.1 / 3.6 | 13.3 / 15.0 / 14.0 | 7.3 / 7.3 / 7.3 | 1.3 / 1.4 / 1.4 | 8.3 / 5.5 / 6.8 | 18.8 / 19.1 / 18.9 |
| Real_conv_2023_slice_a_tok70 | dsqwen-7b | conv | 14.7 / 15.0 / 14.7 | 13.4 / 13.5 / 12.9 | 32.8 / 37.6 / 34.0 | 13.2 / 14.4 / 13.2 | 20.9 / 21.1 / 20.9 | 18.9 / 19.8 / 18.3 | 50.6 / 53.0 / 51.1 |
| Real_conv_2023_slice_a_tok70 | dsllama-8b | conv | 7.3 / 7.9 / 7.3 | 9.4 / 9.8 / 9.2 | 19.2 / 22.4 / 20.0 | 5.9 / 6.5 / 5.7 | 12.5 / 12.6 / 12.5 | 19.6 / 18.3 / 19.0 | 26.5 / 28.6 / 27.0 |
| Real_conv_2023_slice_a_tok70 | dsqwen-14b | conv | 10.7 / 11.7 / 10.8 | 7.9 / 7.7 / 7.6 | 40.5 / 46.9 / 42.2 | 6.7 / 7.5 / 6.6 | 21.0 / 21.7 / 21.2 | 15.8 / 13.4 / 14.7 | 69.0 / 72.9 / 70.1 |
| Real_code_2024_slice_a_tok70 | dsqwen-7b | code | 12.3 / 11.2 / 12.0 | 8.8 / 10.9 / 8.5 | 28.1 / 29.7 / 28.4 | 9.1 / 8.1 / 8.7 | 21.9 / 20.4 / 21.5 | 15.4 / 19.7 / 14.8 | 39.5 / 39.3 / 39.4 |
| Real_code_2024_slice_a_tok70 | dsllama-8b | code | 10.1 / 9.5 / 9.8 | 12.4 / 14.0 / 12.1 | 25.1 / 26.0 / 25.2 | 9.4 / 7.4 / 8.9 | 13.3 / 13.4 / 13.0 | 20.1 / 23.8 / 20.0 | 33.4 / 32.0 / 33.1 |
| Real_code_2024_slice_a_tok70 | dsqwen-14b | code | 21.1 / 19.2 / 20.7 | 18.3 / 18.2 / 18.0 | 56.5 / 51.6 / 55.6 | 18.8 / 17.4 / 18.5 | 32.5 / 30.6 / 32.1 | 30.0 / 31.7 / 29.8 | 75.0 / 66.9 / 73.5 |

## Real_* training intervals (real Azure past of each model's slice)

Real_* (user 2026-11-04): train on the real Azure past of the slice. Per model, the n intervals right before
    the slice's own source interval (same length, contiguous going back; an interval that overlaps a used one moves
    back on the 60 s grid to the nearest earlier free position), each through the r3e transform of the slice
    (``r3e_select.materialize``: x8 compression, 121 s envelope, plateau-clip min-max to the model's lo..hi target x
    the slice's beta, tok70 lengths, thinning / copies) and the r3e generator (``gen_r3e.run_unit``, the slice's
    seed), into ``<out>/real-past/<set>/past<k>/``.

| trace | model | key | source interval (UTC) | slice starts | shifted back (s) | collisions skipped |
|---|---|---|---|---|---:|---|
| Real_conv_2023_slice_a_tok70 | dsllama-8b | past1 | 2024-05-16T19:08:00+00:00 .. 2024-05-16T20:44:00+00:00 | 2024-05-16T22:13:00+00:00 | 5340 | history traces-icse-ktrace-20261031/Real_code_2024_slice_a_tok70/dsllama-8b |
| Real_conv_2023_slice_a_tok70 | dsllama-8b | past2 | 2024-05-16T17:32:00+00:00 .. 2024-05-16T19:08:00+00:00 | 2024-05-16T22:13:00+00:00 | 0 | none |
| Real_conv_2023_slice_a_tok70 | dsllama-8b | past3 | 2024-05-16T14:37:00+00:00 .. 2024-05-16T16:13:00+00:00 | 2024-05-16T22:13:00+00:00 | 4740 | r3c Real_conv_2023_slice_a_tok70/dsqwen-7b |
| Real_conv_2023_slice_a_tok70 | dsllama-8b | past4 | 2024-05-16T13:01:00+00:00 .. 2024-05-16T14:37:00+00:00 | 2024-05-16T22:13:00+00:00 | 0 | none |
| Real_conv_2023_slice_a_tok70 | dsqwen-7b | past1 | 2024-05-17T11:18:00+00:00 .. 2024-05-17T12:54:00+00:00 | 2024-05-17T15:23:00+00:00 | 8940 | tre-calibrated-suite-20261028 / tre-suite-r3-20261029 source_start_utc 2024-05-17T13:54:00+00:00 |
| Real_conv_2023_slice_a_tok70 | dsqwen-7b | past2 | 2024-05-17T09:42:00+00:00 .. 2024-05-17T11:18:00+00:00 | 2024-05-17T15:23:00+00:00 | 0 | none |
| Real_conv_2023_slice_a_tok70 | dsqwen-7b | past3 | 2024-05-17T08:06:00+00:00 .. 2024-05-17T09:42:00+00:00 | 2024-05-17T15:23:00+00:00 | 0 | none |
| Real_conv_2023_slice_a_tok70 | dsqwen-7b | past4 | 2024-05-17T06:30:00+00:00 .. 2024-05-17T08:06:00+00:00 | 2024-05-17T15:23:00+00:00 | 0 | none |
| Real_conv_2023_slice_a_tok70 | dsqwen-14b | past1 | 2024-05-18T19:22:00+00:00 .. 2024-05-18T20:58:00+00:00 | 2024-05-18T20:58:00+00:00 | 0 | none |
| Real_conv_2023_slice_a_tok70 | dsqwen-14b | past2 | 2024-05-18T17:46:00+00:00 .. 2024-05-18T19:22:00+00:00 | 2024-05-18T20:58:00+00:00 | 0 | none |
| Real_conv_2023_slice_a_tok70 | dsqwen-14b | past3 | 2024-05-18T16:10:00+00:00 .. 2024-05-18T17:46:00+00:00 | 2024-05-18T20:58:00+00:00 | 0 | none |
| Real_conv_2023_slice_a_tok70 | dsqwen-14b | past4 | 2024-05-18T14:34:00+00:00 .. 2024-05-18T16:10:00+00:00 | 2024-05-18T20:58:00+00:00 | 0 | none |
| Real_code_2024_slice_a_tok70 | dsllama-8b | past1 | 2024-05-14T08:37:00+00:00 .. 2024-05-14T10:13:00+00:00 | 2024-05-14T10:47:00+00:00 | 2040 | history traces-icse-half400-20261101/Alternating_hot_model_periodic_A/dsqwen-14b; history traces-icse-ktrace-20261031/Sinusoidal_demand/dsqwen-7b |
| Real_code_2024_slice_a_tok70 | dsllama-8b | past2 | 2024-05-14T03:16:00+00:00 .. 2024-05-14T04:52:00+00:00 | 2024-05-14T10:47:00+00:00 | 13500 | tre-calibrated-suite-20261028 / tre-suite-r3-20261029 source_start_utc 2024-05-14T06:50:00+00:00; history traces-icse-ktrace-20261031/Prefill_mixed_corner_decode_mix/dsqwen-7b; history traces-icse-ktrace-20261031/Simultaneous_spike_ramp_twice_tps1o2/dsqwen-14b |
| Real_code_2024_slice_a_tok70 | dsllama-8b | past3 | 2024-05-14T01:40:00+00:00 .. 2024-05-14T03:16:00+00:00 | 2024-05-14T10:47:00+00:00 | 0 | none |
| Real_code_2024_slice_a_tok70 | dsllama-8b | past4 | 2024-05-13T22:54:00+00:00 .. 2024-05-14T00:30:00+00:00 | 2024-05-14T10:47:00+00:00 | 4200 | history traces-icse-ktrace-20261031/Real_code_2024_slice_a_tok70/dsqwen-14b |
| Real_code_2024_slice_a_tok70 | dsqwen-7b | past1 | 2024-05-11T23:55:00+00:00 .. 2024-05-12T01:31:00+00:00 | 2024-05-12T03:17:00+00:00 | 6360 | history traces-icse-ktrace-20261031/Sinusoidal_demand/dsqwen-14b |
| Real_code_2024_slice_a_tok70 | dsqwen-7b | past2 | 2024-05-11T21:42:00+00:00 .. 2024-05-11T23:18:00+00:00 | 2024-05-12T03:17:00+00:00 | 2220 | history traces-icse-ktrace-20261031/Prefill_mixed_corner_decode_mix/dsqwen-14b |
| Real_code_2024_slice_a_tok70 | dsqwen-7b | past3 | 2024-05-11T20:06:00+00:00 .. 2024-05-11T21:42:00+00:00 | 2024-05-12T03:17:00+00:00 | 0 | none |
| Real_code_2024_slice_a_tok70 | dsqwen-7b | past4 | 2024-05-11T18:30:00+00:00 .. 2024-05-11T20:06:00+00:00 | 2024-05-12T03:17:00+00:00 | 0 | none |
| Real_code_2024_slice_a_tok70 | dsqwen-14b | past1 | 2024-05-16T17:28:00+00:00 .. 2024-05-16T19:04:00+00:00 | 2024-05-16T19:04:00+00:00 | 0 | none |
| Real_code_2024_slice_a_tok70 | dsqwen-14b | past2 | 2024-05-16T15:52:00+00:00 .. 2024-05-16T17:28:00+00:00 | 2024-05-16T19:04:00+00:00 | 0 | none |
| Real_code_2024_slice_a_tok70 | dsqwen-14b | past3 | 2024-05-16T14:16:00+00:00 .. 2024-05-16T15:52:00+00:00 | 2024-05-16T19:04:00+00:00 | 0 | none |
| Real_code_2024_slice_a_tok70 | dsqwen-14b | past4 | 2024-05-16T11:01:00+00:00 .. 2024-05-16T12:37:00+00:00 | 2024-05-16T19:04:00+00:00 | 5940 | history traces-icse-half400-20261101/Real_code_2024_slice_a_tok70/dsqwen-7b |

## How different the training realizations are from the evaluation one

Mean over windows of |x_seed - x_eval| / x_eval (%), mean (max) over the 4 training seeds. Near 0 means the generator fixes the window sums and the seed only moves arrivals inside a second / prompt text: the predictor then has seen (almost) this very series.

| trace | model | P | D |
|---|---|---:|---:|
| Alternating_hot_model_periodic_A | dsqwen-7b | 1.38 (1.47) | 1.37 (1.44) |
| Alternating_hot_model_periodic_A | dsllama-8b | 1.48 (1.62) | 1.46 (1.60) |
| Alternating_hot_model_periodic_A | dsqwen-14b | 2.12 (2.19) | 2.03 (2.12) |
| Simultaneous_spike_ramp_twice_tps1o2 | dsqwen-7b | 1.57 (1.73) | 1.50 (1.68) |
| Simultaneous_spike_ramp_twice_tps1o2 | dsllama-8b | 1.94 (2.29) | 1.92 (2.27) |
| Simultaneous_spike_ramp_twice_tps1o2 | dsqwen-14b | 2.17 (2.64) | 2.07 (2.52) |
| Prefill_mixed_corner_decode_mix | dsqwen-7b | 1.32 (1.57) | 1.27 (1.53) |
| Prefill_mixed_corner_decode_mix | dsllama-8b | 1.42 (1.79) | 1.39 (1.80) |
| Prefill_mixed_corner_decode_mix | dsqwen-14b | 1.64 (1.92) | 1.65 (1.98) |
| Decode_heavy_burst | dsqwen-7b | 0.58 (0.84) | 0.50 (0.79) |
| Decode_heavy_burst | dsllama-8b | 0.59 (0.79) | 0.55 (0.75) |
| Decode_heavy_burst | dsqwen-14b | 0.57 (0.80) | 0.47 (0.71) |
| Sinusoidal_demand | dsqwen-7b | 0.53 (0.76) | 0.46 (0.70) |
| Sinusoidal_demand | dsllama-8b | 0.42 (0.51) | 0.38 (0.48) |
| Sinusoidal_demand | dsqwen-14b | 0.65 (0.88) | 0.60 (0.84) |
| Real_conv_2023_slice_a_tok70 | dsqwen-7b | 122.56 (146.69) | 120.84 (143.91) |
| Real_conv_2023_slice_a_tok70 | dsllama-8b | 116.73 (164.36) | 118.15 (165.16) |
| Real_conv_2023_slice_a_tok70 | dsqwen-14b | 77.21 (136.37) | 77.40 (136.79) |
| Real_code_2024_slice_a_tok70 | dsqwen-7b | 100.39 (159.13) | 97.66 (156.49) |
| Real_code_2024_slice_a_tok70 | dsllama-8b | 118.28 (159.60) | 120.67 (168.36) |
| Real_code_2024_slice_a_tok70 | dsqwen-14b | 152.09 (172.77) | 152.52 (173.46) |

## Files

| file | trace | model | sha256 |
|---|---|---|---|
| `v1/Alternating_hot_model_periodic_A.dsllama-8b.npz` | Alternating_hot_model_periodic_A | dsllama-8b | `7cf8613ca338faaf77bc59a135a318a9d9335df4a3b580982a400a15e1a397de` |
| `v1/Alternating_hot_model_periodic_A.dsqwen-14b.npz` | Alternating_hot_model_periodic_A | dsqwen-14b | `4f935df2e8533acea7643ed829548e667dce0aa67ab9b14d211c1661318d3f1c` |
| `v1/Alternating_hot_model_periodic_A.dsqwen-7b.npz` | Alternating_hot_model_periodic_A | dsqwen-7b | `ea31034af6183361b47c3c6c3f467b4573404b0d3ae26b8b4dc565a02719efc0` |
| `v1/Decode_heavy_burst.dsllama-8b.npz` | Decode_heavy_burst | dsllama-8b | `20af262f0e6a6440308c0761a1fb5f4b238b0422f9fbb5b0365580a999979a35` |
| `v1/Decode_heavy_burst.dsqwen-14b.npz` | Decode_heavy_burst | dsqwen-14b | `5cba19ed0fcf9ce28064e33e65010792af7468f198b659d9ea34376ad6c1cc35` |
| `v1/Decode_heavy_burst.dsqwen-7b.npz` | Decode_heavy_burst | dsqwen-7b | `47b42951f20808b0f7ec7dae5b209e47accebf8b2a8e939479ca3e4b56b7b951` |
| `v1/Prefill_mixed_corner_decode_mix.dsllama-8b.npz` | Prefill_mixed_corner_decode_mix | dsllama-8b | `e9d3ad347ddfcf6cb8b6547cf5ddf56690aef38a47bfb21af856a7eba3d4369b` |
| `v1/Prefill_mixed_corner_decode_mix.dsqwen-14b.npz` | Prefill_mixed_corner_decode_mix | dsqwen-14b | `f4bb765e123915576a7f557bbbdf9826f405bd54392d8661926ea72b82d1aa1d` |
| `v1/Prefill_mixed_corner_decode_mix.dsqwen-7b.npz` | Prefill_mixed_corner_decode_mix | dsqwen-7b | `6895df8ec224687ac6ec73fea0535b08e37cadaf6627d1afad4f6b28a4dbb491` |
| `v1/Real_code_2024_slice_a_tok70.dsllama-8b.npz` | Real_code_2024_slice_a_tok70 | dsllama-8b | `a3e0716aded448e8f1f09790a0caf2244845176723029006bb99ecf35e02aa9c` |
| `v1/Real_code_2024_slice_a_tok70.dsqwen-14b.npz` | Real_code_2024_slice_a_tok70 | dsqwen-14b | `e29df732db1a0b08a246c10283c97dec4dbeff54bee6e410fa70cc2defe0ed23` |
| `v1/Real_code_2024_slice_a_tok70.dsqwen-7b.npz` | Real_code_2024_slice_a_tok70 | dsqwen-7b | `21e14528ac1f920134b4a7f12c66a52070e1f4707d029c830acdd05ae915cf50` |
| `v1/Real_conv_2023_slice_a_tok70.dsllama-8b.npz` | Real_conv_2023_slice_a_tok70 | dsllama-8b | `396ab0d9e14f37c9f275bfcd12929c899005a890a59ed013db1fbd9e803e0f1d` |
| `v1/Real_conv_2023_slice_a_tok70.dsqwen-14b.npz` | Real_conv_2023_slice_a_tok70 | dsqwen-14b | `e49bebbd25cb9741c4d79e880a64e26a8319b6f3f8ca09c4d8be4eb22822bf95` |
| `v1/Real_conv_2023_slice_a_tok70.dsqwen-7b.npz` | Real_conv_2023_slice_a_tok70 | dsqwen-7b | `5c9cdbbaceee37faa2b25e0d8e613be0dc10f54ce15e253fcda4d5136d9cc832` |
| `v1/Simultaneous_spike_ramp_twice_tps1o2.dsllama-8b.npz` | Simultaneous_spike_ramp_twice_tps1o2 | dsllama-8b | `33545127a51a926f0c6cb0c67cbd80ea1e502dbf3cdeecb36155749a6c4217f3` |
| `v1/Simultaneous_spike_ramp_twice_tps1o2.dsqwen-14b.npz` | Simultaneous_spike_ramp_twice_tps1o2 | dsqwen-14b | `18c79a416cd977a7d24a309b41d8770406261a2c5531a28fc256cd0d9258a520` |
| `v1/Simultaneous_spike_ramp_twice_tps1o2.dsqwen-7b.npz` | Simultaneous_spike_ramp_twice_tps1o2 | dsqwen-7b | `b3330f2f7f15badb0dffe6b2fa41fd7058e7bdbe6e18b92e25882fd403194092` |
| `v1/Sinusoidal_demand.dsllama-8b.npz` | Sinusoidal_demand | dsllama-8b | `850fe9164111b6d0f444ce75b23a94f55c5313aecb2984d6ca3c51adffd2fa84` |
| `v1/Sinusoidal_demand.dsqwen-14b.npz` | Sinusoidal_demand | dsqwen-14b | `8e6385ca1bfa38bce81b77dd527b4169f7204d71114f5533f15792eb9d3c5b37` |
| `v1/Sinusoidal_demand.dsqwen-7b.npz` | Sinusoidal_demand | dsqwen-7b | `1eb7c6b636820cf4229ce612b59118e424d9bc2bca6d015933558d5ffd15b148` |
| `v2/concat.dsllama-8b.npz` | * | dsllama-8b | `3f0b2db93e7b3a8123562c9644fd53c0668e728cfa6316263891515dff705e59` |
| `v2/concat.dsqwen-14b.npz` | * | dsqwen-14b | `cfa8e88f6e3a89ffdeee3b7a55bd3e42a77dd3774f648a5101beafea60bc2f76` |
| `v2/concat.dsqwen-7b.npz` | * | dsqwen-7b | `b057f4b5ab9ec94bae881186c13f159ed810e13d7bbc65f680fdc01d5ce69612` |

## Policy params (example)

```yaml
# V1 (one set per trace, a file per model); V2: one set 'concat' with predictor_rules [['', concat]]
window_s: 10
predictor_normalization: minmax
history_variant: train-tail-20261104
predictors:
  v1-Alternating_hot_model_periodic_A:
    models:
      dsqwen-7b: {path: weights/preserve-mlstm-20261104-trace-w10/v1/Alternating_hot_model_periodic_A.dsqwen-7b.npz, sha256: ea31034af6183361b47c3c6c3f467b4573404b0d3ae26b8b4dc565a02719efc0}
      dsllama-8b: {path: weights/preserve-mlstm-20261104-trace-w10/v1/Alternating_hot_model_periodic_A.dsllama-8b.npz, sha256: 7cf8613ca338faaf77bc59a135a318a9d9335df4a3b580982a400a15e1a397de}
      dsqwen-14b: {path: weights/preserve-mlstm-20261104-trace-w10/v1/Alternating_hot_model_periodic_A.dsqwen-14b.npz, sha256: 4f935df2e8533acea7643ed829548e667dce0aa67ab9b14d211c1661318d3f1c}
  # ... one set per trace
predictor_rules:
- [Alternating_hot_model_periodic_A, v1-Alternating_hot_model_periodic_A]
```

## Run notes (added by hand)

- realize (synthetic traces; also the r3e seed realizations, now unused): 76, nice 19, taskset 56-63, 4 processes,
  29.6 min including the seed-1 reproduction runs. Every seed-1 rerun and every rebuilt design input is
  byte-identical to the evaluation set (SOURCES.md of the data directory).
- realize-real-past (Real_*, user 2026-11-04): 76, nice 19, taskset 56-63, 4 processes, 6 min. Code path:
  `r3e_select.materialize` with the past interval as its `start_s` (the slice's beta, lo / hi, tok70 targets; hot-block
  fields recomputed for the record only), then `gen_r3e.run_unit` with seed 1, from the sha256-checked `_src` copy.
  The only difference from the slice is the interval, plus a new design dir per past (`real-past/<set>/past<k>.design`).
- Overlap: 9 of the 24 intervals hit a used interval and moved back on the 60 s grid (table above). The 14b pasts
  touch their slices; for 7b and 8b (both families) past1 ends 34-149 min before the slice, so their history is the
  nearest free real past, not the contiguous pre-context.
- series 76; train 75 (clock about 160 s ahead), nice 10, taskset 48-63, 12 jobs x 1 thread, 42 s (48 nets); finish
  76 with OMP_NUM_THREADS=1. The 15 synthetic V1 files are identical to the previous run (376ae36b); the 6 Real_* V1
  files and the 3 V2 files changed (replaced in place; the earlier files remain in git at 376ae36b).
- D = max_output_tokens: the runs use ignore_eos, so the actual output tokens equal max_tokens; D is therefore also
  the paper's D (completed output tokens attributed to the arrival window).
- Synthetic traces: the training realizations differ from the evaluation one by 0.4-2.2 % per window (the
  generators fix the per-second counts), a best case. Real_*: 77-152 % per window (each real interval has its own
  plateau timing); MAPE there is the honest "unseen day" number.
