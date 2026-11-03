# PreServe mLSTM predictors trained on each trace's own past, W = 10 s

Created 2026-11-03T19:38:33+00:00 by `preserve_train/trace_train.py finish` (aibrix 5ea90950157b64b62b427a94064759080c9018b6, dirty False); trained 2026-11-03T19:40:56+00:00 on nscc-ds-4a100-node9 (aibrix 5ea90950157b64b62b427a94064759080c9018b6, dirty False), torch 2.13.0+cu130, numpy 2.2.6, CPU. Config `tre/baselines/preserve_train/configs/trace-train-20261104.json` (sha256 `2b24c600450627c37ead1ab29343de7da077c9c7775f58b8b55b57027cdab194`).

## What

The paper trains a service's predictor on that service's own past (section 5.3 RQ2: first two days -> next two). Here the past of an evaluation trace = independent realizations of the same service made by the trace's own generator and spec with seeds no evaluation uses (101, 102, 103, 104; evaluation seed 1 is never trained on): 4 x 72 windows of 10 s = 288 windows (paper: 2 days of 10 min windows = 288).

- **V1 per trace** (`v1/<trace>.<model>.npz`, 21 files): training series = the trace's 4 realizations concatenated in seed order (288 windows, 263 sequences).
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
| Alternating_hot_model_periodic_A | 5.2 | 7.5 | 26.3 | 10.0 | 7.6 | 14.2 | 34.5 | 17.9 |
| Simultaneous_spike_ramp_twice_tps1o2 | 9.1 | 13.3 | 22.4 | 10.9 | 11.5 | 24.4 | 29.1 | 19.9 |
| Prefill_mixed_corner_decode_mix | 6.8 | 7.3 | 25.4 | 8.5 | 11.6 | 13.4 | 32.6 | 15.0 |
| Decode_heavy_burst | 2.7 | 5.1 | 7.1 | 1.7 | 4.0 | 10.8 | 9.8 | 2.7 |
| Sinusoidal_demand | 1.2 | 3.3 | 15.5 | 8.5 | 1.1 | 6.8 | 21.1 | 16.7 |
| Real_conv_2023_slice_a_tok70 | 7.8 | 7.8 | 32.0 | 7.9 | 11.2 | 13.5 | 49.4 | 12.7 |
| Real_code_2024_slice_a_tok70 | 10.5 | 8.3 | 36.4 | 11.9 | 14.4 | 12.5 | 48.7 | 17.1 |
| **all 21** | **6.2** | **7.5** | **23.6** | **8.5** | **8.8** | **13.7** | **32.2** | **14.6** |

### Per trace and model (P / D / total)

| trace | model | current set | V1 | V2 | current | naive | V1 2-step | V2 2-step | current 2-step |
|---|---|---|---|---|---|---|---|---|---|
| Alternating_hot_model_periodic_A | dsqwen-7b | pooled | 4.4 / 4.4 / 4.4 | 6.0 / 6.7 / 5.9 | 25.8 / 28.6 / 27.0 | 8.9 / 8.9 / 8.9 | 6.9 / 6.9 / 6.9 | 12.1 / 11.7 / 11.2 | 34.4 / 36.5 / 35.3 |
| Alternating_hot_model_periodic_A | dsllama-8b | pooled | 5.8 / 5.8 / 5.8 | 9.0 / 7.2 / 7.9 | 24.9 / 27.7 / 26.1 | 11.3 / 11.3 / 11.3 | 7.2 / 7.2 / 7.2 | 15.5 / 11.2 / 13.4 | 33.8 / 35.5 / 34.5 |
| Alternating_hot_model_periodic_A | dsqwen-14b | pooled | 5.4 / 5.4 / 5.4 | 12.7 / 5.5 / 8.6 | 24.2 / 27.7 / 25.7 | 10.0 / 9.9 / 9.9 | 8.5 / 8.6 / 8.6 | 27.9 / 7.8 / 17.9 | 32.4 / 35.2 / 33.6 |
| Simultaneous_spike_ramp_twice_tps1o2 | dsqwen-7b | pooled | 8.8 / 8.8 / 8.8 | 14.3 / 11.2 / 12.4 | 19.3 / 21.7 / 20.2 | 9.7 / 9.7 / 9.7 | 11.3 / 11.3 / 11.3 | 26.4 / 20.9 / 22.5 | 25.7 / 26.7 / 26.0 |
| Simultaneous_spike_ramp_twice_tps1o2 | dsllama-8b | pooled | 8.4 / 8.4 / 8.4 | 13.2 / 11.0 / 11.8 | 20.4 / 23.1 / 21.4 | 10.7 / 10.7 / 10.7 | 10.5 / 10.5 / 10.5 | 24.0 / 17.2 / 20.7 | 27.7 / 28.9 / 28.1 |
| Simultaneous_spike_ramp_twice_tps1o2 | dsqwen-14b | pooled | 10.1 / 10.1 / 10.1 | 19.3 / 13.9 / 15.8 | 24.2 / 27.5 / 25.5 | 12.4 / 12.4 / 12.4 | 12.6 / 12.6 / 12.6 | 39.8 / 23.7 / 30.1 | 32.5 / 34.6 / 33.3 |
| Prefill_mixed_corner_decode_mix | dsqwen-7b | pooled | 7.9 / 7.0 / 6.5 | 5.7 / 7.3 / 6.2 | 38.3 / 21.8 / 31.6 | 10.4 / 9.3 / 10.1 | 13.7 / 10.1 / 10.5 | 9.5 / 11.2 / 10.4 | 48.9 / 26.9 / 39.9 |
| Prefill_mixed_corner_decode_mix | dsllama-8b | pooled | 9.5 / 6.1 / 6.6 | 7.1 / 7.7 / 6.4 | 32.9 / 14.4 / 25.3 | 9.3 / 7.2 / 8.1 | 18.7 / 7.5 / 10.7 | 13.0 / 11.2 / 10.6 | 43.5 / 17.6 / 33.0 |
| Prefill_mixed_corner_decode_mix | dsqwen-14b | pooled | 7.1 / 8.5 / 7.2 | 12.3 / 8.2 / 9.2 | 18.6 / 18.3 / 19.4 | 6.0 / 8.4 / 7.2 | 14.3 / 13.8 / 13.8 | 27.5 / 13.4 / 19.2 | 25.1 / 22.0 / 24.8 |
| Decode_heavy_burst | dsqwen-7b | pooled | 2.6 / 2.5 / 2.5 | 6.5 / 4.3 / 5.2 | 6.5 / 7.4 / 7.0 | 1.7 / 1.6 / 1.6 | 3.8 / 3.8 / 3.8 | 13.8 / 9.2 / 11.0 | 9.4 / 9.8 / 9.6 |
| Decode_heavy_burst | dsllama-8b | pooled | 2.5 / 2.5 / 2.5 | 6.4 / 3.7 / 4.9 | 6.0 / 6.9 / 6.4 | 1.8 / 1.7 / 1.8 | 3.6 / 3.6 / 3.6 | 14.0 / 7.1 / 10.6 | 8.7 / 9.0 / 8.8 |
| Decode_heavy_burst | dsqwen-14b | pooled | 3.0 / 3.0 / 3.0 | 7.9 / 3.8 / 5.2 | 7.5 / 8.5 / 8.0 | 1.9 / 1.8 / 1.8 | 4.5 / 4.5 / 4.5 | 17.2 / 6.9 / 10.7 | 10.8 / 11.2 / 11.0 |
| Sinusoidal_demand | dsqwen-7b | pooled | 1.1 / 1.1 / 1.1 | 3.2 / 2.5 / 2.7 | 15.1 / 17.0 / 15.9 | 8.6 / 8.7 / 8.6 | 1.0 / 1.0 / 1.0 | 7.2 / 3.4 / 5.4 | 21.3 / 22.0 / 21.6 |
| Sinusoidal_demand | dsllama-8b | pooled | 1.0 / 1.0 / 1.0 | 3.8 / 2.2 / 3.0 | 15.9 / 17.8 / 16.6 | 9.7 / 9.7 / 9.7 | 0.9 / 0.9 / 0.9 | 8.5 / 3.1 / 6.1 | 22.5 / 23.2 / 22.8 |
| Sinusoidal_demand | dsqwen-14b | pooled | 1.5 / 1.5 / 1.5 | 6.9 / 2.5 / 4.3 | 13.3 / 15.0 / 14.0 | 7.3 / 7.3 / 7.3 | 1.3 / 1.4 / 1.4 | 15.2 / 3.5 / 9.1 | 18.8 / 19.1 / 18.9 |
| Real_conv_2023_slice_a_tok70 | dsqwen-7b | conv | 7.2 / 8.9 / 7.3 | 7.3 / 9.4 / 7.1 | 32.8 / 37.6 / 34.0 | 8.7 / 10.0 / 8.7 | 12.0 / 13.6 / 12.1 | 13.4 / 16.4 / 12.7 | 50.6 / 53.0 / 51.1 |
| Real_conv_2023_slice_a_tok70 | dsllama-8b | conv | 8.6 / 8.7 / 8.5 | 8.6 / 8.5 / 7.8 | 19.2 / 22.4 / 20.0 | 7.8 / 8.3 / 7.6 | 11.3 / 11.2 / 11.2 | 13.9 / 13.5 / 13.0 | 26.5 / 28.6 / 27.0 |
| Real_conv_2023_slice_a_tok70 | dsqwen-14b | conv | 8.3 / 7.2 / 7.7 | 11.0 / 7.2 / 8.4 | 40.5 / 46.9 / 42.2 | 7.4 / 8.1 / 7.3 | 11.5 / 9.4 / 10.4 | 20.9 / 11.2 / 14.9 | 69.0 / 72.9 / 70.1 |
| Real_code_2024_slice_a_tok70 | dsqwen-7b | code | 8.5 / 7.7 / 8.2 | 7.1 / 8.6 / 6.7 | 28.1 / 29.7 / 28.4 | 9.0 / 8.1 / 8.6 | 12.7 / 11.8 / 12.4 | 12.0 / 14.4 / 11.2 | 39.5 / 39.3 / 39.4 |
| Real_code_2024_slice_a_tok70 | dsllama-8b | code | 9.3 / 8.2 / 9.0 | 8.2 / 11.5 / 8.1 | 25.1 / 26.0 / 25.2 | 9.5 / 7.4 / 9.1 | 10.7 / 10.2 / 10.4 | 9.9 / 18.5 / 10.5 | 33.4 / 32.0 / 33.1 |
| Real_code_2024_slice_a_tok70 | dsqwen-14b | code | 14.9 / 13.4 / 14.5 | 10.5 / 12.6 / 10.3 | 56.5 / 51.6 / 55.6 | 18.5 / 16.9 / 18.1 | 20.8 / 19.2 / 20.4 | 16.5 / 19.8 / 15.9 | 75.0 / 66.9 / 73.5 |

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
| Real_conv_2023_slice_a_tok70 | dsqwen-7b | 0.39 (0.74) | 0.36 (0.77) |
| Real_conv_2023_slice_a_tok70 | dsllama-8b | 0.39 (0.69) | 0.35 (0.66) |
| Real_conv_2023_slice_a_tok70 | dsqwen-14b | 0.39 (0.67) | 0.33 (0.69) |
| Real_code_2024_slice_a_tok70 | dsqwen-7b | 0.73 (0.98) | 0.62 (0.88) |
| Real_code_2024_slice_a_tok70 | dsllama-8b | 0.75 (0.88) | 0.69 (0.86) |
| Real_code_2024_slice_a_tok70 | dsqwen-14b | 0.73 (0.86) | 0.62 (0.78) |

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
| `v1/Real_code_2024_slice_a_tok70.dsllama-8b.npz` | Real_code_2024_slice_a_tok70 | dsllama-8b | `f654469d7e887a04942de0fac29a98dac9b62aca76eff9cf9a167bc4384fbde6` |
| `v1/Real_code_2024_slice_a_tok70.dsqwen-14b.npz` | Real_code_2024_slice_a_tok70 | dsqwen-14b | `d138480cf24876eff31cc2b4360f6f9e89c6962d01c3a26e323b3f804f6a0ef9` |
| `v1/Real_code_2024_slice_a_tok70.dsqwen-7b.npz` | Real_code_2024_slice_a_tok70 | dsqwen-7b | `70208e76636187d8cfa014a601231a86d572f61a83439bf23929beff886963cb` |
| `v1/Real_conv_2023_slice_a_tok70.dsllama-8b.npz` | Real_conv_2023_slice_a_tok70 | dsllama-8b | `473be5095e7613accfcf99c916ad598148520587489e25be370605b246279389` |
| `v1/Real_conv_2023_slice_a_tok70.dsqwen-14b.npz` | Real_conv_2023_slice_a_tok70 | dsqwen-14b | `9f4df486782cce3362ab52b5f049a3809d78c7cc998cdf728ada0634b6fa9268` |
| `v1/Real_conv_2023_slice_a_tok70.dsqwen-7b.npz` | Real_conv_2023_slice_a_tok70 | dsqwen-7b | `024f3b00de6e648dcbe344a710966300f04378370d5a80b4706001940c7d50c6` |
| `v1/Simultaneous_spike_ramp_twice_tps1o2.dsllama-8b.npz` | Simultaneous_spike_ramp_twice_tps1o2 | dsllama-8b | `33545127a51a926f0c6cb0c67cbd80ea1e502dbf3cdeecb36155749a6c4217f3` |
| `v1/Simultaneous_spike_ramp_twice_tps1o2.dsqwen-14b.npz` | Simultaneous_spike_ramp_twice_tps1o2 | dsqwen-14b | `18c79a416cd977a7d24a309b41d8770406261a2c5531a28fc256cd0d9258a520` |
| `v1/Simultaneous_spike_ramp_twice_tps1o2.dsqwen-7b.npz` | Simultaneous_spike_ramp_twice_tps1o2 | dsqwen-7b | `b3330f2f7f15badb0dffe6b2fa41fd7058e7bdbe6e18b92e25882fd403194092` |
| `v1/Sinusoidal_demand.dsllama-8b.npz` | Sinusoidal_demand | dsllama-8b | `850fe9164111b6d0f444ce75b23a94f55c5313aecb2984d6ca3c51adffd2fa84` |
| `v1/Sinusoidal_demand.dsqwen-14b.npz` | Sinusoidal_demand | dsqwen-14b | `8e6385ca1bfa38bce81b77dd527b4169f7204d71114f5533f15792eb9d3c5b37` |
| `v1/Sinusoidal_demand.dsqwen-7b.npz` | Sinusoidal_demand | dsqwen-7b | `1eb7c6b636820cf4229ce612b59118e424d9bc2bca6d015933558d5ffd15b148` |
| `v2/concat.dsllama-8b.npz` | * | dsllama-8b | `bce6ad04a92eb5e6d224d1392838ddaa65f1ae69961534f633d7568f44b979df` |
| `v2/concat.dsqwen-14b.npz` | * | dsqwen-14b | `31d5dad4db387455410214de6e92ee531783afea3aface19589c592e3c0ebaad` |
| `v2/concat.dsqwen-7b.npz` | * | dsqwen-7b | `365c276e7419c7627ce3c9d1fd7a2c0cc7d5deee7871434aafe7a628c5436b24` |

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

- realize: 76, nice 19, taskset 56-63, 4 generator processes, 2026-11-04 03:01:49-03:31:25 (29.6 min, of which the
  seed-1 reproduction runs take about 9 min). Every seed-1 rerun (traces_tre.effective.json and the generator's own
  preserve_history.json) and every rebuilt design input (r3c trace.json / r3c.json, r3e real_traces/*.json / r3e.json)
  is byte-identical to the evaluation set (SOURCES.md of the data directory).
- series: 76, 11 s, by the working tree of this branch before its first commit; trace_train.py did not change
  between that run and commit 5ea90950 (the committed file is the file that ran), so series_w10.json records
  56146f78 + dirty.
- train: 75 (clock about 160 s ahead of 76), nice 10, taskset 48-63, 12 jobs x 1 torch thread, 42 s for the 48 nets
  (24 files x prompt / response). A first run at the uncommitted tree gave identical arrays in all 24 files.
- finish: 76, numpy with OMP_NUM_THREADS=1 (15 s; with the default BLAS threads 3 min).
- D = max_output_tokens: the runs use ignore_eos, so the actual output tokens equal max_tokens; D is therefore also
  the paper's D (completed output tokens attributed to the arrival window).
- The training realizations are close to the evaluation one (table above: 0.3-2.2 % per window): the generators fix
  the per-second counts (synthetic: carry-round(k x rps(t)); Real_*: the design lists) and the seed changes only the
  arrival inside a second and the sampled prompt lengths. For Real_* the seed changes no request length (the lists
  carry them), only the arrival inside the second; the shape is the same fixed Azure interval. So V1 has seen nearly
  the evaluated series: a best case for "the same service's past", not a forecast of an unseen day.
