# traces-icse-r3c-20261101: sources and checksums (ICSE r3c, 2026-11-01)

Five ICSE traces redesigned so that the 8-GPU pool is a binding constraint and the models' peaks are staggered (moving GPUs between models is needed, already in the first 400 s), while the ICSE shape semantics stay. Decode_heavy_burst and Sinusoidal_demand are NOT in this set (unchanged: use traces-icse-ktrace-20261031). Seed 1 only. Trace files are never committed; scripts: local workspace `docs/scripts/trace_suite_r3c_20261101/` (copy in `_src/`). Design dir (v1 generator tree with the edited inputs): `../traces-icse-r3c-20261101.design/`.

Offline audit (demand model, contention, fixed-layout failure, orientation sims): `audit.md` / `audit.json`. Design search grids: `_checks/design_search.txt`. Real interval selection: `_checks/selection.json`. PreServe history causality: `_checks/check_history_causal.{json,out}`.

## Rules that fixed every parameter (no method result was read)

- R-rise (realism): steepest 30 s rise per model (arrivals in [t, t+30) / mean rate in [t-60, t)) <= 2.16 = the steepest whole-service rise in a week of Azure 2024 code traffic (conv 1.80; ramp_azure.py). Where ICSE is steeper, the quiet-period floor is raised: smallest floor on a 0.05 grid that meets it.
- R-feas (not everybody overloaded): rho* <= 0.95 in every 30 s window (some layout inside 8 GPUs keeps every model <= 95 % of its knee), demand model r3c_model.py with 14b kappa 1.0.
- R-k: the rate factor is the largest on the grid that meets R-feas (the resulting contention is measured, not tuned). Real_*: per-model factor to fixed peak targets, then one common beta (largest with R-feas).
- Why not 'tailored to TRE': every rule is a property of the load (realistic steepness, feasibility, pool pressure, staggered peaks), stated before any run; the same traces go to every method; where the result favours a fixed layout (Real_conv 2/2/2 at the knee) it is reported, not changed.

## Per-trace changes

### Alternating_hot_model_periodic_A

k = 1.65 (count rule carry-round(k x rps(t))), edit `alt` {"floor_frac": 0.4, "ramp": 36}, max_tokens rule default. r3c trace.json sha256 `c60e8f68010a974ee51923501e44ef654b19c303930429acad870c3902dab64b`; ICSE trace.json sha256 `30833fd5cfc6cfcfdfd4ee4ad29bc6320a59087488456cacb28850b9546d2dff`.

- floor 0.40 x hot (ICSE 0.255): smallest floor on the 0.05 grid with the steepest 30 s rise <= 2.16 (0.35 -> 2.26-2.63, 0.40 -> 2.05; ICSE 2.7-2.9 with the r2 ramp, 3.4-3.9 with ICSE 3 s steps)
- k 1.65: largest on the 0.05 grid with rho_star <= 0.95 (1.70 -> 0.955)
- unchanged: 10 slots x 72 s, rotation 8b -> 7b -> 14b, hot rps ratios, 500 / 400, r2 36 s ramps

### Prefill_mixed_corner_decode_mix

k = 1.7 (count rule carry-round(k x rps(t))), edit `pfm` {"corner_s": 60, "work_mult": 2.0, "d_delay": 30}, max_tokens rule generator. r3c trace.json sha256 `e42fddb55a0de8ba47fe116cb785aa24c4d42bdcd9346cb4915193e24d736cc5`; ICSE trace.json sha256 `9e7a5386990ccd0a842bcf13db7d1523b3ef0c2101b8e2a5be43fc2098d97f45`.

- corners 60 s (ICSE 40 s): a 40 s corner is over before any scaler has a full 30 s window of it
- corner work = 2 x base work for every corner (ICSE 1.4-1.8x, unequal across models): the prefill corner keeps its long prompts / short outputs and the decode corner its short prompts / long outputs; the count rise is 1.23-1.72 (<= 2.16)
- decode corner starts 30 s after its paired prefill corner (ICSE: same second): the pair overlaps 30 s, so demand shifts from one model to another inside a corner event
- k 1.7: largest on the 0.1 grid with rho_star <= 0.95 (1.8 -> 0.998)
- unchanged: base 500 / 400 at ICSE rps ratios, corner schedule and model pairing, per-segment max_tokens (r2 generator rule)

### Real_code_2024_slice_a_tok70

Real Azure 2024 data only: each model replays its OWN independent 720 s interval of the code week (rules below); per-model rate factor (beta 0.83); ICSE tok70 length transform; time order and per-request length pairing kept.

| model | interval (UTC) | offset in CSV (s) | raw rps | burst 30 s | t_peak (s) | rate factor | requests (copies / dropped) | design list sha256 |
|---|---|---:|---:|---:|---:|---:|---|---|
| dsllama-8b | 2024-05-13T01:18:00+00:00 .. 01:30:00 | 263880 | 18.34 | 1.337 | 320 | 1.869125 | 24682 (11477 / 0) | `404fca6b577ada0a25e6587fa1ab3038d9f4044774f42be1b1d0c71bcb7175db` |
| dsqwen-7b | 2024-05-12T20:32:00+00:00 .. 20:44:00 | 246720 | 17.731 | 1.378 | 90 | 1.896767 | 24214 (11448 / 0) | `fdb197d75bb1006ca45aaaeb5ba8c9f1c2ee9eeaacd08a4e536c16881da39570` |
| dsqwen-14b | 2024-05-13T02:42:00+00:00 .. 02:54:00 | 268920 | 16.908 | 1.461 | 470 | 0.815069 | 9923 (0 / 2251) | `730acc17b18b162545450cd7d7fdd31da95aa44051f8018718ef2017aa66f644` |

Source CSV `2024/AzureLLMInferenceTrace_code_1week.csv` sha256 `71de5c55cbc35f8f1ed0b6b7806b4cd1e9764b0058469725a6aac98023a1448f` (verified at generation). Candidates 7764; busy-hour rate >= 16.58 rps; burst threshold (p90 of busy candidates) 1.319; peak targets {'dsqwen-7b': 2.55, 'dsllama-8b': 2.55, 'dsqwen-14b': 1.35}; peak bands {'dsqwen-7b': [30, 230], 'dsllama-8b': [230, 430], 'dsqwen-14b': [430, 690]}.

Selection rule (real_select.py):

```
Real_* r3c: one independent real Azure 2024 interval (720 s) per model, chosen by a fixed rule; per-model rate
factor; demand check. Design inputs only: per-second counts of the week CSVs (azure_counts.py) and the rule.

Rule (fixed before looking at any method; the same for code and conv):
1. candidates: 720 s intervals on a 60 s grid inside the week, not in the first day (kept for history / dev, as the
   10-28 independent-interval plan), overlapping no EXCLUDE zone (ICSE Real_* v1 windows + their pre-slice history,
   the six 10-28 independent intervals, 1 h guards: preserve_history_azure.EXCLUDE) and no r3c history segment.
2. statistic per candidate: 30 s windows (10 s step) of the raw count over [30, 720) s of the interval (the runner
   trims the first 30 s): burst = max window / mean window, t_peak = start of the max window.
3. qualifying: raw mean rate >= the family's median candidate rate (busy-hour traffic: a burst statistic on a
   quiet interval is mostly Poisson noise, and scaling a quiet interval up means mostly copied requests), and
   burst >= the 90th percentile of burst among those busy candidates (the burstier tenth of real
   traffic: a 720 s run can only test scaling if demand moves inside it; v1/ICSE got movement by min-max stretching
   the counts, which r3c does not do) AND t_peak inside the model's band: 7b [30, 230), 8b [230, 430),
   14b [430, 690) (peaks staggered by construction of the rule, two of them inside the first 400 s + the run's
   first 30 s trim).
4. pick: candidates in sha256("r3c-real|<family>|<model>|<start>") order; the first qualifying one that overlaps no
   interval already picked (any model, both traces) wins.
5. rate factor: f_m = P_m / (peak window rho at factor 1), P = 7b 2.55 / 8b 2.55 / 14b 1.35: a TP1 model needs
   3 replicas at its own peak (U 0.85, as the synthetic hot targets); the TP2 model needs its 2nd replica (a 2-GPU
   block) only at its own peak (1.35 on one replica) while its interval mean (peak / burst ~ 0.9) fits one replica
   (with 14b at its peak target 1.7 it holds 4 GPUs all the time and the trace degenerates to a fixed 2/2/2);
   then one common beta (0.01 steps, <= 1): the largest with rho_star <= 0.95 in every window (R-feas, as the
   synthetic traces).
6. requests: each model's raw requests of its interval, time order and per-request (ContextTokens, GeneratedTokens)
   pairing kept; lengths by the ICSE Real tok70 transform (v9: min-max of the interval's own lengths onto the v1
   token targets of the family, preserve_history_azure.TOKEN_TARGETS); count per second = carry-round(f_m x raw count):
   fewer -> the requests with the smallest sha256(model|row) are kept; more -> all raw requests + copies of
   random requests of the same second, input and output x U(0.85, 1.15) (v1 step-4 rule, ICSE Real_*).
   Arrival inside the second: v1 RealTraceAdapter (uniform, seeded), as ICSE.
```

### Real_conv_2023_slice_a_tok70

Real Azure 2024 data only: each model replays its OWN independent 720 s interval of the conv week (rules below); per-model rate factor (beta 0.79); ICSE tok70 length transform; time order and per-request length pairing kept.

| model | interval (UTC) | offset in CSV (s) | raw rps | burst 30 s | t_peak (s) | rate factor | requests (copies / dropped) | design list sha256 |
|---|---|---:|---:|---:|---:|---:|---|---|
| dsllama-8b | 2024-05-13T19:03:00+00:00 .. 19:15:00 | 154980 | 52.785 | 1.19 | 350 | 0.647834 | 24621 (0 / 13384) | `9b5a175544c1dfd3c35d33bc7cbe314080f27536cf9f70ac9d898a0b982e39a1` |
| dsqwen-7b | 2024-05-16T16:13:00+00:00 .. 16:25:00 | 403980 | 55.76 | 1.228 | 220 | 0.601109 | 24133 (0 / 16014) | `4095459c18cf6b6d627418db4349653ee0051f2e5603116f46d67df100460690` |
| dsqwen-14b | 2024-05-13T19:22:00+00:00 .. 19:34:00 | 156120 | 52.089 | 1.222 | 620 | 0.28418 | 10658 (0 / 26846) | `20f3ab6b48c71e8fa080c81d7a173b2377ce0f4206c326b6978fb0b7de8e6167` |

Source CSV `2024/AzureLLMInferenceTrace_conv_1week.csv` sha256 `a0cc9b969a9bbf0fd811802cbf4323edd3a209ace791e3799ad4f9207f213941` (verified at generation). Candidates 7723; busy-hour rate >= 49.13 rps; burst threshold (p90 of busy candidates) 1.187; peak targets {'dsqwen-7b': 2.55, 'dsllama-8b': 2.55, 'dsqwen-14b': 1.35}; peak bands {'dsqwen-7b': [30, 230], 'dsllama-8b': [230, 430], 'dsqwen-14b': [430, 690]}.

Selection rule (real_select.py):

```
Real_* r3c: one independent real Azure 2024 interval (720 s) per model, chosen by a fixed rule; per-model rate
factor; demand check. Design inputs only: per-second counts of the week CSVs (azure_counts.py) and the rule.

Rule (fixed before looking at any method; the same for code and conv):
1. candidates: 720 s intervals on a 60 s grid inside the week, not in the first day (kept for history / dev, as the
   10-28 independent-interval plan), overlapping no EXCLUDE zone (ICSE Real_* v1 windows + their pre-slice history,
   the six 10-28 independent intervals, 1 h guards: preserve_history_azure.EXCLUDE) and no r3c history segment.
2. statistic per candidate: 30 s windows (10 s step) of the raw count over [30, 720) s of the interval (the runner
   trims the first 30 s): burst = max window / mean window, t_peak = start of the max window.
3. qualifying: raw mean rate >= the family's median candidate rate (busy-hour traffic: a burst statistic on a
   quiet interval is mostly Poisson noise, and scaling a quiet interval up means mostly copied requests), and
   burst >= the 90th percentile of burst among those busy candidates (the burstier tenth of real
   traffic: a 720 s run can only test scaling if demand moves inside it; v1/ICSE got movement by min-max stretching
   the counts, which r3c does not do) AND t_peak inside the model's band: 7b [30, 230), 8b [230, 430),
   14b [430, 690) (peaks staggered by construction of the rule, two of them inside the first 400 s + the run's
   first 30 s trim).
4. pick: candidates in sha256("r3c-real|<family>|<model>|<start>") order; the first qualifying one that overlaps no
   interval already picked (any model, both traces) wins.
5. rate factor: f_m = P_m / (peak window rho at factor 1), P = 7b 2.55 / 8b 2.55 / 14b 1.35: a TP1 model needs
   3 replicas at its own peak (U 0.85, as the synthetic hot targets); the TP2 model needs its 2nd replica (a 2-GPU
   block) only at its own peak (1.35 on one replica) while its interval mean (peak / burst ~ 0.9) fits one replica
   (with 14b at its peak target 1.7 it holds 4 GPUs all the time and the trace degenerates to a fixed 2/2/2);
   then one common beta (0.01 steps, <= 1): the largest with rho_star <= 0.95 in every window (R-feas, as the
   synthetic traces).
6. requests: each model's raw requests of its interval, time order and per-request (ContextTokens, GeneratedTokens)
   pairing kept; lengths by the ICSE Real tok70 transform (v9: min-max of the interval's own lengths onto the v1
   token targets of the family, preserve_history_azure.TOKEN_TARGETS); count per second = carry-round(f_m x raw count):
   fewer -> the requests with the smallest sha256(model|row) are kept; more -> all raw requests + copies of
   random requests of the same second, input and output x U(0.85, 1.15) (v1 step-4 rule, ICSE Real_*).
   Arrival inside the second: v1 RealTraceAdapter (uniform, seeded), as ICSE.
```

### Simultaneous_spike_ramp_twice_tps1o2

k = 1.325 (count rule carry-round(k x rps(t))), edit `sim` {"base_frac": 0.5, "plateaus": [30, 50, 70]}, max_tokens rule default. r3c trace.json sha256 `2776cdaf15884f1f7e9625637002ed6decf84f41604ed02f6f52719da589640c`; ICSE trace.json sha256 `ed7a822f159c366e2d14430dacd1f8f39b0570f14b79b9e467a4a6cf4150f0ad`.

- off-spike floor 0.5 x spike rps (ICSE 0.37-0.44): smallest floor on the 0.05 grid with rise <= 2.16 (0.45 -> 2.19-2.21, 0.50 -> 1.98; ICSE 2.47-2.94)
- plateaus 30 / 50 / 70 s rotating (ICSE 40 / 40 / 40): the spikes still start together; they end at different times, so the GPUs of the model whose burst ends first must move to the others while the pool is full (with equal plateaus every model needs its GPUs at the same time and releases at the same time: no move is ever needed, oracle moves 0 at any k)
- k 1.325: largest on the 0.025 grid with rho_star <= 0.95 (1.35 -> 0.955)
- unchanged: onsets (45, 189, 333, 477, 621), spike rps per model and spike (8b / 7b lead alternates), 1000 / 600, 3 s ramps

## Outputs

| trace/seed | requests | traces_tre.effective.json | preserve_history.json | manifest.json |
|---|---:|---|---|---|
| Alternating_hot_model_periodic_A/s1 | 23319 | `574ab1b5eb03fdb76c991fe3efb99b2476d0df32a8b7c1a6dade2c798cd1bbbf` | `58aaac45aebf03f20864239963ce246ec8a103810d69bd9ffbd4d8589d177120` | `2284881c0d1fcf2d47e09c0c48e9d9355437fc51c656c96362accc33b77a9e8f` |
| Prefill_mixed_corner_decode_mix/s1 | 19594 | `91821e6f5d92660566e252cf7086a6ed4712d542fbefa8fbcec844bf2ff8c42e` | `26165dfd7bcaf3e015f3e6e5bb8ad45937ba90567d4e4c0f467bcc53bb62fdf1` | `f6151f1d916fceceb0dcb5b7c8065ad905826a56dea3a3e09a875caab9f7db39` |
| Real_code_2024_slice_a_tok70/s1 | 58819 | `713351163a57766251f195cbe9420c4b4a352c03a7e056b87df428ca89d691fd` | `9d8abee284a4f467c233e9fd56f73d313ff12723592e4cae87f6eab56f1b9df6` | `fc60a0ea6723ead0c073fad17c885446e0f3c74360703ba252647980d0d765c8` |
| Real_conv_2023_slice_a_tok70/s1 | 59412 | `f71edf48a54054fc373fc613dc332afa6c5d22d8e5c74b413b64a02f9233c2d2` | `0236877e9cd953ce41f4cd74a61aedbf0488c1dba3377f8cad24c21801908239` | `5289672c2dcab0293c46ab59e522898dfaaeacf10442415f297911a056b5abf8` |
| Simultaneous_spike_ramp_twice_tps1o2/s1 | 8479 | `6a2ce8f782b82a6a0daa938aa8aa3e245525c3a8caf3c3f9d9755d31be8c4d93` | `3f696a82c57c82d9ab782a6839e3664e2b06833e995865b0dcadaa5823903728` | `d5ec667a5109db69545b45d31c27a9361534c83558f82b4151496a6ec56ec2a9` |

## PreServe history (ktrace method, user 10-31: causal, real Azure segments)

Per model a real Azure 2024 750 s segment (25 x 30 s windows) at a sha256-derived offset, excluding every Real_* interval of this and earlier suites (+- 1 h); v9 tok70 lengths; P and D levelled to the trace's PLANNED mean per 30 s window (design inputs only, never the generated file). check_history_causal.py: code check, runtime open() audit, garbage-trace rebuild (see _checks).

| trace | model | Azure segment (UTC) | raw req/s | scale P / D |
|---|---|---|---:|---|
| Alternating_hot_model_periodic_A | dsllama-8b | conv 2024-05-13T23:34:30+00:00 .. 23:47:00 | 43.995 | 0.407389 / 0.957337 |
| Alternating_hot_model_periodic_A | dsqwen-7b | code 2024-05-12T14:00:00+00:00 .. 14:12:30 | 8.317 | 2.201748 / 7.903358 |
| Alternating_hot_model_periodic_A | dsqwen-14b | code 2024-05-14T10:45:30+00:00 .. 10:58:00 | 9.636 | 0.931159 / 3.227021 |
| Prefill_mixed_corner_decode_mix | dsllama-8b | conv 2024-05-15T06:05:00+00:00 .. 06:17:30 | 46.385 | 0.357827 / 0.718213 |
| Prefill_mixed_corner_decode_mix | dsqwen-7b | code 2024-05-12T07:49:00+00:00 .. 08:01:30 | 3.723 | 5.142199 / 17.350207 |
| Prefill_mixed_corner_decode_mix | dsqwen-14b | code 2024-05-11T02:00:30+00:00 .. 02:13:00 | 13.257 | 0.618811 / 2.231622 |
| Real_code_2024_slice_a_tok70 | dsllama-8b | conv 2024-05-15T10:20:00+00:00 .. 10:32:30 | 51.859 | 0.74355 / 0.481739 |
| Real_code_2024_slice_a_tok70 | dsqwen-7b | code 2024-05-16T12:37:30+00:00 .. 12:50:00 | 32.851 | 0.931668 / 1.022488 |
| Real_code_2024_slice_a_tok70 | dsqwen-14b | code 2024-05-10T04:10:00+00:00 .. 04:22:30 | 13.704 | 0.957025 / 1.041003 |
| Real_conv_2023_slice_a_tok70 | dsllama-8b | conv 2024-05-13T23:53:30+00:00 .. 00:06:00 | 45.66 | 0.726768 / 0.722486 |
| Real_conv_2023_slice_a_tok70 | dsqwen-7b | code 2024-05-11T16:47:30+00:00 .. 17:00:00 | 15.869 | 1.864884 / 2.864267 |
| Real_conv_2023_slice_a_tok70 | dsqwen-14b | code 2024-05-13T12:58:30+00:00 .. 13:11:00 | 28.716 | 0.43684 / 0.691842 |
| Simultaneous_spike_ramp_twice_tps1o2 | dsllama-8b | conv 2024-05-16T11:57:00+00:00 .. 12:09:30 | 52.341 | 0.270123 / 0.470739 |
| Simultaneous_spike_ramp_twice_tps1o2 | dsqwen-7b | code 2024-05-15T21:01:00+00:00 .. 21:13:30 | 69.36 | 0.157211 / 0.494927 |
| Simultaneous_spike_ramp_twice_tps1o2 | dsqwen-14b | code 2024-05-16T05:21:30+00:00 .. 05:34:00 | 12.625 | 0.501711 / 1.329093 |

## Code

- `_src/GENERATE.sh` 44bda512cf4dc2c6d1960a28c8e5ee28785f5e9017104f5f8a661659bd2d40c0
- `_src/audit_r3c.py` 52807cec1c7bf39663da0df1f628179a872ba093f4f9b91d57f293e7517c6d81
- `_src/azure_counts.py` 3ac62866e3762ae362d46234d2f637d95c1a1049f4670940fc44a2695e7be5f2
- `_src/build_icse_suite.py` c592e1f90b2e079a4e3bfe41c5872819b38cc83c97dacb2c4f38699bc9bad4fc
- `_src/campaign/campaign-icse-r3c-half-20261101.yaml` ef9384c0f0269add55deb441fb8813fe4e44bfdcd4c254907a5a25c33a25a9a1
- `_src/campaign/run_r3c_half.sh` 6cc37f721f2afea792ea9f6146b92c21b73d3fce02fa98baaf7475c03192029e
- `_src/check_final.py` 80f7d0584b269a8cf188c1f94c5c6fb4a4123ebf591d2a2e33c4b8666bef6cd1
- `_src/check_history_causal.py` b33842a8b6ca9cadbde4e31957ac963890832bf8af2dd31303f1b1ebf966d697
- `_src/contend_lib.py` d1735ad34d3ed7995aae4b5d4b81d7f8a9c8e9e8e8946514a12da114e12c165c
- `_src/cut_prefix.py` edd0bdfe0e86fdc522bde133925679eea8c5a41f481faad3b593a43f2f76da1e
- `_src/explore_current.py` 52e2e688d04c08069ddc190a8c3d6dc1871bf3d9e473fa6382c793875691380a
- `_src/explore_grid.py` f35a2f36b88555cf878d1c7220835bc167163a8497717e4aa445e05c7d089d0a
- `_src/gen_r3c.py` 860b52c6305ebccf3ccf4312a2a382a1b2683ef846236ba3eab67e00e43051b0
- `_src/preserve_history_azure.py` 6dac1a1ed7eace6ac6a8a4ff79820874bbc841f61ee23c9a276694cb7ed95614
- `_src/r2_designs.py` 4e2619f336f04a8e63556c21df330b98ab3cfc4089d6dce66eb638da5d631385
- `_src/r3c_designs.py` d6508c67dedf60ca2b08e8e9ecb89d6c50fb88697426b828015133afd48d483e
- `_src/r3c_model.py` 2c438c78254b40488d6009600eaf5d37979cac666b4aa8e7da96927e4ff9b7fa
- `_src/r3c_plan.py` b26dbf27b0cbba8e715bd958ef2f7e041a3d41299722f5d06e280e584f0ca7e6
- `_src/real_select.py` 11598b92dd84f70c61b7d212b5c4423d1a2b6498304e477bed9f6cb9d325cc36
- `_src/scale_icse_suite.py` 11e9ac3cc93b4bd26cb9a7807aee807f4fc70027871d29016bb0363357a5b58b
- `_src/verify_cut_r3c.py` fa902121c90b04f54f4e34cc54e7ec543dc2f33d52e1d0b4717a16fa1cb15ae1
- `_src/write_half_sources.py` bc3ef383b5fcff0927038fca92fe2a229bfeb75ab19736e7b1f6938e35577c99
- `_src/write_sources.py` 3d645c17661677eb738869b4d80c677e01fd4fb5ac3e58b703c7744831970a30
- v1 generator `/root/aibrix-main/CustomTraceGenerator` git aeb2cf42ba6c0c81680a918c3f7a98695498bafc (dirty src: none)

## Use

Campaign: `trace_root` = this directory, `loadgen_config_root` = `${TRE_REPO}/loadgen_v1/configs/traces_v14` (trace names unchanged; no r3c edit changes a config field the client reads: null max_output_tokens = config max_tokens as before). Half runs: `../traces-icse-r3c-half400-20261101/` (requests with timestamp < 400 s).

## Notes added after generation (11-01)

- Simultaneous reason text in r3c.json says equal plateaus give "oracle moves 0 at any k": precisely, 0 on the whole
  searched grid up to k 1.6 with the 0.5 floor (design_search.txt); at k 1.7 equal plateaus give 3, but rho* is 1.07 there.
- Plan vs generated: the generated synthetic files have exactly the planned per-second counts (checked); prompt_length
  is in v1 units (~1.5 % below the segment input_tokens), so the audited demand is slightly below the plan
  (Alternating oracle moves 10 -> 4, need > 8 22 % -> 13 %). Move counts are threshold counts and move by a few
  per cent of demand; read them as "present / absent", not as precise numbers.
- 14b kappa sensitivity (1.3 instead of 1.0) weakens Alternating (need peak 8, moves 2) and Simultaneous (moves 0);
  Prefill_mixed and the Real_* traces keep their moves.
