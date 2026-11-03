# traces-icse-r3e-20261101: sources and checksums (ICSE r3e = Real_* round 3, 2026-11-01)

Real_code_2024_slice_a_tok70 and Real_conv_2023_slice_a_tok70 redesigned so that demand SHIFTS between the models in WIDE, staggered hot phases (7b, then 14b, then 8b; real Azure 2024 traffic, plateau-clipped min-max), sized with the demand model CALIBRATED on observed runs, under a load cap that keeps APA runnable. Replaces the r3d versions (traces-icse-r3d-20261101: no contention under the calibrated model). Seed 1 only. Trace files are never committed; scripts: local workspace `docs/scripts/trace_suite_r3e_20261101/` (copy in `_src/`). Design dir: `../traces-icse-r3e-20261101.design/`.

Offline audit: `audit.md` / `audit.json` (vs r3d and ktrace; gates full 720 s and first 400 s). r3d diagnosis and capacity calibration: `_checks/r3d_diagnosis.txt`, `_checks/cap_calib.{txt,json}` (finished runs used only as engine-capacity samples and for arm layouts; no latency comparison between methods). Design search record: `_checks/design_search.txt`. Selection: `_checks/selection.{json,out}`. PreServe history causality: `_checks/check_history_causal.{json,out}`.

## Rules

- The full rule (compression, candidates, plateau clipping, staggered bands, joint pick, levels, beta) is quoted per trace below (r3e_select.py docstring); the gates are in audit.md.

## Per-trace design

### Real_code_2024_slice_a_tok70

Real Azure 2024 traffic only: each model replays its OWN independent interval of the code week, time-compressed x8 (96 real minutes in 720 s); its smoothed (121 s) real rate envelope is plateau-clipped (top 0.2 of the seconds = hot plateau, bottom 0.3 = trough) and min-max mapped to the target load lo..hi (replicas at the calibrated knee, kappa {'dsqwen-7b': 0.85, 'dsllama-8b': 0.855, 'dsqwen-14b': 1.2}) {'dsqwen-7b': 0.6, 'dsllama-8b': 0.6, 'dsqwen-14b': 0.45} .. {'dsqwen-7b': 2.6, 'dsllama-8b': 2.6, 'dsqwen-14b': 2.45} x beta 0.79; requests = the interval's own real requests (thinned or v1 step-4 copies, time order and per-request length pairing kept), ICSE tok70 lengths.

| model | interval (UTC, real) | offset in CSV (s) | raw rps | hot block (s) | head | target req/s min..max | requests (copies / dropped) | design list sha256 |
|---|---|---:|---:|---|---:|---|---|---|
| dsllama-8b | 2024-05-14T10:47:00+00:00 .. 12:23:00 | 384420 | 14.592 | [576, 720] | 0.0 | 9.29 .. 40.27 | 14829 (0 / 69223) | `f9e12911800946825ab1612ed3e034493cf01bec63532411d0fc67edaf91c78c` |
| dsqwen-7b | 2024-05-12T03:17:00+00:00 .. 04:53:00 | 184620 | 11.325 | [69, 198] | 0.488 | 10.19 .. 44.18 | 16132 (0 / 49102) | `2169fed74e9f4f033ed41e901de1764935e01850fc5d09fa40ebf31d88f0c90b` |
| dsqwen-14b | 2024-05-16T19:04:00+00:00 .. 20:40:00 | 587040 | 77.375 | [273, 395] | 0.361 | 7.61 .. 41.45 | 15044 (0 / 430634) | `c33e9aefa3ad3d0f7f48b580eef34cb6ed11abc1d5ec557a9453a77241ad3ff4` |

Source CSV `2024/AzureLLMInferenceTrace_code_1week.csv` sha256 `71de5c55cbc35f8f1ed0b6b7806b4cd1e9764b0058469725a6aac98023a1448f` (verified at generation). Candidates 4637; qualifying {'dsqwen-7b': 25, 'dsqwen-14b': 19, 'dsllama-8b': 806, 'lists': {'dsqwen-7b': 25, 'dsqwen-14b': 19, 'dsllama-8b': 806}, 'pick': {'ranks': [22, 0, 41], 'beta_cap': 0.794, 'evaluated': 3000, 'gates_passed': 6, 'beta': 0.79}}; bands (hot block inside) {'dsqwen-7b': [60, 270], 'dsqwen-14b': [200, 420], 'dsllama-8b': [400, 720]}; load reference {'set': 'traces-icse-ktrace-r2-20261031 (Real_code x1.25)', 'factor': 1.0, 'rps_mean': 74.27, 'rps_peak30': 79.57, 'tok_mean': 34826.4, 'tok_peak30': 38030.8}.

Plan gates (design lists, calibrated model): {"static_full": true, "static_400": true, "feasible": true, "moves_full": false, "moves_400": false, "cap": true, "rise": true}; {"static_layout": "3/3/1", "static_over": 0.418, "static_over_400": 0.314, "moves": 2, "moves_400": 1, "feasible": 1.0, "rise_max": 1.78, "load_ratio": {"rps_mean": 0.867, "rps_peak30": 0.995, "tok_mean": 0.885, "tok_peak30": 0.999}}

Rule (r3e_select.py):

```
Real_* r3e (2026-11-01, round 3 of the r3c Real_* traces): WIDE, staggered hot phases from real Azure 2024
traffic, sized with the demand model calibrated on observed runs, under an APA-safe load cap.

Why a round 3 (r3d diagnosis, _checks/cap_calib.txt, _checks/r3d_diagnosis.txt):
* the r3d contention numbers (oracle pool-full 54 % / 25 %) counted GPUs at U = 0.85 of an UNCALIBRATED knee; the
  arms do not keep that headroom (TokenScale scales out at rho_rep 0.8-1.05, TRE at 0.9-1.5, old units) and the
  r3d peaks were narrow, so every arm absorbed them in place. Measured knees (request violation > 5 %, TTFT 500 /
  TPOT 75 ms): code 7b 1.0 / 8b 0.9 x the old capacity, conv 7b 1.3 / 8b 1.1, 14b >= 1.15 (both families).
  With the calibrated model the best fixed full layout of r3d (3/3/1) is over the knee in 0 % of windows and the
  knee oracle needs 0 cross-model moves: r3d had no contention to find.
* so r3e (i) uses r3c_model KAPPA real_code / real_conv; (ii) makes each model's hot phase a PLATEAU (quantile
  clipping of the real envelope; >= ~120 s at peak) instead of a single-second argmax; (iii) puts the three plateaus
  one after another (7b, then 14b, then 8b) so one model's peak meets the others' troughs; (iv) caps the total load
  (APA must still produce a result).

Rule (fixed before generation; identical for code and conv; seed 1; design inputs = per-second counts of the week
CSVs, the calibrated demand model, the two reference load files; the search that led here: _checks/design_search.txt):
1. time compression C = 8 (as r3d): model m replays a real interval of 8 x 720 s of its family's week in 720 s;
   trace second t holds the real requests of real seconds [s + 8 t, s + 8 t + 8).
2. candidates: intervals on a 60 s grid (a real hump can be placed to 7.5 s), not in the first day, no overlap with
   EXCLUDE_BASE, the r3c and the r3d Real_* intervals (each +- 1 h).
3. envelope e(t) = 121 s centred moving mean of the compressed counts; plateau clipping (WIDE hot phases):
   u(t) = clip((e - Q_lo) / (Q_hi - Q_lo), 0, 1), Q_hi = quantile(e, 0.8), Q_lo = quantile(e, 0.3): the busiest
   20 % of the interval's seconds are the hot plateau (u = 1), the quietest 30 % the trough (u = 0), the real ramps
   in between stay in time order. hot block = the longest contiguous run of u = 1.
4. qualifying for model m: hot block >= 120 s and >= 75 % of the plateau seconds, lying inside the model's band
   (one after another: 7b [60, 270), 14b [200, 420), 8b [400, 720); 7b and 14b plateaus fall inside the 400 s half
   runs); head <= 0.5 (mean u over [0, 60): no model opens at its peak); R-rise on the target series of rule 5
   (steepest 30 s rise vs the previous 60 s <= the family's Azure week max: code 2.16, conv 1.80).
   Pick (joint, total demand steady): over all triples of qualifying, mutually non-overlapping intervals (+- 1 h,
   same file), ranked by the scale-down their total load needs to meet the cap at beta = 1 (steadiest total first;
   ties: sha256("r3e-real|C8|<family>|<model>|<start>") ranks), the first whose plan passes every gate below; when
   none of the first 3000 does, the one passing the most gates (then the higher beta).
5. target load (v1 min-max onto designed levels, units = replicas at the CALIBRATED knee):
   rho_m(t) = beta x (LO_m + (HI_m - LO_m) u_m(t)); req/s(t) = rho_m(t) x mu_m(family mean tok70 shape).
   LO / HI, same for both families: TP1 0.6 / 2.6 (trough 1 replica, plateau 3 replicas at the knee, 4 at U 0.85);
   14b 0.45 / 2.45 (1 -> 3 replicas = 6 GPUs: the 14b plateau is what fills the pool cheaply in req/s).
   beta (0.01 steps, <= 1) = the largest with (a) the load cap: total req/s and total token rate (in + out tokens
   per s), mean over [30, 720) and peak 30 s window, <= the family reference (code: the ICSE Real_code x1.25 version
   traces-icse-ktrace-r2-20261031, APA 8 % client failures there; conv: 1.6 x the ICSE Real_conv x1 version
   traces-icse-20261030), and (b) R-feas: rho* <= 1 in every 30 s window. On the family mean shape (plan), then
   finished on the design lists' own lengths (refine()).
6. requests (r3d rule 6 unchanged): count(t) = carry-round(req/s(t)); the real requests of the compressed second t,
   thinned by smallest sha256(model|start|row), or all of them + copies (input and output x U(0.85, 1.15), v1
   step-4 rule); real time order kept; ICSE tok70 lengths (min-max of the interval's own lengths onto the family's
   v1 token targets).

Gates (design targets, checked on the plan in rule 4 and audited on the generated files, full 720 s and first
400 s; audit_r3e.py): best fixed FULL-pool layout (min over all of them) over the knee in >= 25 % of windows;
rho* <= 1 in >= 95 % of windows; r3d oracle cross-model moves >= 2 (400 s) and >= 4 (720 s); load cap; rise.
```

### Real_conv_2023_slice_a_tok70

Real Azure 2024 traffic only: each model replays its OWN independent interval of the conv week, time-compressed x8 (96 real minutes in 720 s); its smoothed (121 s) real rate envelope is plateau-clipped (top 0.2 of the seconds = hot plateau, bottom 0.3 = trough) and min-max mapped to the target load lo..hi (replicas at the calibrated knee, kappa {'dsqwen-7b': 1.105, 'dsllama-8b': 1.045, 'dsqwen-14b': 1.2}) {'dsqwen-7b': 0.6, 'dsllama-8b': 0.6, 'dsqwen-14b': 0.45} .. {'dsqwen-7b': 2.6, 'dsllama-8b': 2.6, 'dsqwen-14b': 2.45} x beta 0.78; requests = the interval's own real requests (thinned or v1 step-4 copies, time order and per-request length pairing kept), ICSE tok70 lengths.

| model | interval (UTC, real) | offset in CSV (s) | raw rps | hot block (s) | head | target req/s min..max | requests (copies / dropped) | design list sha256 |
|---|---|---:|---:|---|---:|---|---|---|
| dsllama-8b | 2024-05-16T22:13:00+00:00 .. 23:49:00 | 425580 | 36.824 | [573, 700] | 0.249 | 10.34 .. 44.81 | 16716 (0 / 195393) | `ec50c9d0c838bfc5d79aba9b2097eda3e3b5181c2bbd057663ed8f2979dbdafe` |
| dsqwen-7b | 2024-05-17T15:23:00+00:00 .. 16:59:00 | 487380 | 52.944 | [78, 219] | 0.407 | 12.36 .. 53.55 | 19592 (0 / 285365) | `cd4c41e9604a2baec2f8d2e0a05d8d407d84dd298a34b970af67bb15415bf38e` |
| dsqwen-14b | 2024-05-18T20:58:00+00:00 .. 22:34:00 | 593880 | 27.501 | [271, 415] | 0.39 | 7.18 .. 39.11 | 15873 (0 / 142535) | `cf8e0a285a8403b4cdff7c911d44d6020d644703904851cf260e36eb35cfaef5` |

Source CSV `2024/AzureLLMInferenceTrace_conv_1week.csv` sha256 `a0cc9b969a9bbf0fd811802cbf4323edd3a209ace791e3799ad4f9207f213941` (verified at generation). Candidates 5438; qualifying {'dsqwen-7b': 35, 'dsqwen-14b': 18, 'dsllama-8b': 275, 'lists': {'dsqwen-7b': 35, 'dsqwen-14b': 18, 'dsllama-8b': 275}, 'pick': {'ranks': [2, 3, 153], 'beta_cap': 0.93, 'evaluated': 285, 'gates_passed': 7, 'beta': 0.82}}; bands (hot block inside) {'dsqwen-7b': [60, 270], 'dsqwen-14b': [200, 420], 'dsllama-8b': [400, 720]}; load reference {'set': 'traces-icse-20261030 (Real_conv x1)', 'factor': 1.6, 'rps_mean': 62.76, 'rps_peak30': 69.87, 'tok_mean': 27750.4, 'tok_peak30': 30757.2}.

Plan gates (design lists, calibrated model): {"static_full": true, "static_400": true, "feasible": true, "moves_full": false, "moves_400": true, "cap": true, "rise": true}; {"static_layout": "1/3/2", "static_over": 0.522, "static_over_400": 0.257, "moves": 3, "moves_400": 3, "feasible": 1.0, "rise_max": 1.79, "load_ratio": {"rps_mean": 0.727, "rps_peak30": 0.842, "tok_mean": 0.748, "tok_peak30": 0.87}}

Rule (r3e_select.py):

```
Real_* r3e (2026-11-01, round 3 of the r3c Real_* traces): WIDE, staggered hot phases from real Azure 2024
traffic, sized with the demand model calibrated on observed runs, under an APA-safe load cap.

Why a round 3 (r3d diagnosis, _checks/cap_calib.txt, _checks/r3d_diagnosis.txt):
* the r3d contention numbers (oracle pool-full 54 % / 25 %) counted GPUs at U = 0.85 of an UNCALIBRATED knee; the
  arms do not keep that headroom (TokenScale scales out at rho_rep 0.8-1.05, TRE at 0.9-1.5, old units) and the
  r3d peaks were narrow, so every arm absorbed them in place. Measured knees (request violation > 5 %, TTFT 500 /
  TPOT 75 ms): code 7b 1.0 / 8b 0.9 x the old capacity, conv 7b 1.3 / 8b 1.1, 14b >= 1.15 (both families).
  With the calibrated model the best fixed full layout of r3d (3/3/1) is over the knee in 0 % of windows and the
  knee oracle needs 0 cross-model moves: r3d had no contention to find.
* so r3e (i) uses r3c_model KAPPA real_code / real_conv; (ii) makes each model's hot phase a PLATEAU (quantile
  clipping of the real envelope; >= ~120 s at peak) instead of a single-second argmax; (iii) puts the three plateaus
  one after another (7b, then 14b, then 8b) so one model's peak meets the others' troughs; (iv) caps the total load
  (APA must still produce a result).

Rule (fixed before generation; identical for code and conv; seed 1; design inputs = per-second counts of the week
CSVs, the calibrated demand model, the two reference load files; the search that led here: _checks/design_search.txt):
1. time compression C = 8 (as r3d): model m replays a real interval of 8 x 720 s of its family's week in 720 s;
   trace second t holds the real requests of real seconds [s + 8 t, s + 8 t + 8).
2. candidates: intervals on a 60 s grid (a real hump can be placed to 7.5 s), not in the first day, no overlap with
   EXCLUDE_BASE, the r3c and the r3d Real_* intervals (each +- 1 h).
3. envelope e(t) = 121 s centred moving mean of the compressed counts; plateau clipping (WIDE hot phases):
   u(t) = clip((e - Q_lo) / (Q_hi - Q_lo), 0, 1), Q_hi = quantile(e, 0.8), Q_lo = quantile(e, 0.3): the busiest
   20 % of the interval's seconds are the hot plateau (u = 1), the quietest 30 % the trough (u = 0), the real ramps
   in between stay in time order. hot block = the longest contiguous run of u = 1.
4. qualifying for model m: hot block >= 120 s and >= 75 % of the plateau seconds, lying inside the model's band
   (one after another: 7b [60, 270), 14b [200, 420), 8b [400, 720); 7b and 14b plateaus fall inside the 400 s half
   runs); head <= 0.5 (mean u over [0, 60): no model opens at its peak); R-rise on the target series of rule 5
   (steepest 30 s rise vs the previous 60 s <= the family's Azure week max: code 2.16, conv 1.80).
   Pick (joint, total demand steady): over all triples of qualifying, mutually non-overlapping intervals (+- 1 h,
   same file), ranked by the scale-down their total load needs to meet the cap at beta = 1 (steadiest total first;
   ties: sha256("r3e-real|C8|<family>|<model>|<start>") ranks), the first whose plan passes every gate below; when
   none of the first 3000 does, the one passing the most gates (then the higher beta).
5. target load (v1 min-max onto designed levels, units = replicas at the CALIBRATED knee):
   rho_m(t) = beta x (LO_m + (HI_m - LO_m) u_m(t)); req/s(t) = rho_m(t) x mu_m(family mean tok70 shape).
   LO / HI, same for both families: TP1 0.6 / 2.6 (trough 1 replica, plateau 3 replicas at the knee, 4 at U 0.85);
   14b 0.45 / 2.45 (1 -> 3 replicas = 6 GPUs: the 14b plateau is what fills the pool cheaply in req/s).
   beta (0.01 steps, <= 1) = the largest with (a) the load cap: total req/s and total token rate (in + out tokens
   per s), mean over [30, 720) and peak 30 s window, <= the family reference (code: the ICSE Real_code x1.25 version
   traces-icse-ktrace-r2-20261031, APA 8 % client failures there; conv: 1.6 x the ICSE Real_conv x1 version
   traces-icse-20261030), and (b) R-feas: rho* <= 1 in every 30 s window. On the family mean shape (plan), then
   finished on the design lists' own lengths (refine()).
6. requests (r3d rule 6 unchanged): count(t) = carry-round(req/s(t)); the real requests of the compressed second t,
   thinned by smallest sha256(model|start|row), or all of them + copies (input and output x U(0.85, 1.15), v1
   step-4 rule); real time order kept; ICSE tok70 lengths (min-max of the interval's own lengths onto the family's
   v1 token targets).

Gates (design targets, checked on the plan in rule 4 and audited on the generated files, full 720 s and first
400 s; audit_r3e.py): best fixed FULL-pool layout (min over all of them) over the knee in >= 25 % of windows;
rho* <= 1 in >= 95 % of windows; r3d oracle cross-model moves >= 2 (400 s) and >= 4 (720 s); load cap; rise.
```

## Outputs

| trace/seed | requests | traces_tre.effective.json | preserve_history.json | manifest.json |
|---|---:|---|---|---|
| Real_code_2024_slice_a_tok70/s1 | 46005 | `457df889b8bd051b78f745ee7d9a7f86af729a9f8ce6f72415d320313990289c` | `48e7d0b604a84db35aab6ecbdabddddf27a48dcda84f72b9a31bc6827209468f` | `0bee15714bd7a70fd9b698ff3bfddd582b13718f6216dfbf04d7d5d9b7fd969b` |
| Real_conv_2023_slice_a_tok70/s1 | 52181 | `431c96e7ff7b9961241c3915d3200fa3a6033bcc51ae1d215f7ce55eb51653d5` | `b5690a64770f22aea1819d2b81c4765c60f61ada07eb9e897715634db736aaaf` | `d67b30021a5f26f57be130c2b23b17739f346ecd4cca54a4f1ff482dbd15a9e6` |

## PreServe history (r3c / ktrace method unchanged, user 10-31: causal, real Azure segments)

Per model a real Azure 2024 750 s segment (25 x 30 s windows) at a sha256-derived offset, excluding every Real_* interval of this and earlier suites (+- 1 h; the r3d and r3e intervals over their whole 96 min real span); v9 tok70 lengths; P and D levelled to the trace's PLANNED mean per 30 s window (design inputs only, never the generated file). check_history_causal.py: code check, runtime open() audit, garbage-trace rebuild (see _checks).

| trace | model | Azure segment (UTC) | raw req/s | scale P / D |
|---|---|---|---:|---|
| Real_code_2024_slice_a_tok70 | dsllama-8b | conv 2024-05-15T10:20:00+00:00 .. 10:32:30 | 51.859 | 0.464716 / 0.287452 |
| Real_code_2024_slice_a_tok70 | dsqwen-7b | code 2024-05-16T12:37:30+00:00 .. 12:50:00 | 32.851 | 0.655451 / 0.66991 |
| Real_code_2024_slice_a_tok70 | dsqwen-14b | code 2024-05-10T04:10:00+00:00 .. 04:22:30 | 13.704 | 1.485512 / 1.52199 |
| Real_conv_2023_slice_a_tok70 | dsllama-8b | conv 2024-05-13T23:53:30+00:00 .. 00:06:00 | 45.66 | 0.496245 / 0.506952 |
| Real_conv_2023_slice_a_tok70 | dsqwen-7b | code 2024-05-11T16:47:30+00:00 .. 17:00:00 | 15.869 | 1.520979 / 2.33137 |
| Real_conv_2023_slice_a_tok70 | dsqwen-14b | code 2024-05-13T12:58:30+00:00 .. 13:11:00 | 28.716 | 0.656338 / 1.071251 |

## Code

- `_src/GENERATE.sh` e9fe0cef5c05161d015ba8c3c0c73007bbf83b52f474811e6013b3206c542913
- `_src/audit_r3d.py` ac3c934ece728c3e60c877e00a43859e442e427b2b555ccf1e6f0630bbc5bb6c
- `_src/audit_r3e.py` 58498fab68cd1e8a70d39431be33ccae7eec7ac3c6ba0f661a2c3e8a91dc40f2
- `_src/azure_counts.py` 3ac62866e3762ae362d46234d2f637d95c1a1049f4670940fc44a2695e7be5f2
- `_src/beta_scan.py` 08581ae06c7489e18f00a7f9beaefc06ad8ecd8a3d283d0c23349eb8fa8f43b8
- `_src/build_icse_suite.py` c592e1f90b2e079a4e3bfe41c5872819b38cc83c97dacb2c4f38699bc9bad4fc
- `_src/cap_calib.py` d577ef61aa0ed73cd8838ee7910b08acf0133f96a6e8dab2eeb68cb2ec06d3d2
- `_src/cap_check.py` ac9cad268e4856913eb0b2af25219a50b92c92ea49204e0c821c1c531bf141a9
- `_src/check_history_causal.py` b7c63b358a92021ea51abb2a9415d036828891ae4f370aada5af35253f0f1fc3
- `_src/contend_lib.py` d1735ad34d3ed7995aae4b5d4b81d7f8a9c8e9e8e8946514a12da114e12c165c
- `_src/contend_runs.py` 72b25ac50945728638c75e5370b751a8b14f7c8ae1ab5caa35f1bb9976af18e7
- `_src/cut_prefix.py` edd0bdfe0e86fdc522bde133925679eea8c5a41f481faad3b593a43f2f76da1e
- `_src/diag_r3e.py` bb567049fb09aa5cd7659afb21b78f84e714e9d8f57018998630bb29d7fbe50f
- `_src/explore_compress.py` e351850116d754b61c2fe70a57b176813dbd6b1c17c634af266638f6dfddf358
- `_src/explore_grid_r3d.py` 20deb1f0cf4514517dc5626f3c0d9698795c9115fb75b00115fd88afb94965fe
- `_src/explore_r3e.py` 0262b964515d26245f67b5e94724df30d30ad2407394b8841cd5b9af55bce862
- `_src/gen_r3d.py` 242665f6887c4de357eb7de20eeae27a263fead867e3407c0bc40718aadd1844
- `_src/gen_r3e.py` 4d0aabd2473f8130de60e7b96b340c058b13ac7d72e31370968ecdc61345acd7
- `_src/ideal_r3e.py` 06c24db3c64fbc5a381ecc988b00a991f73ff412b0b1b8a4ef8ec9fdd3ba6be0
- `_src/preserve_history_azure.py` ab053d6746e27214ae8fecaad534dd27d6822d33e7e0c42f1d028f5eb89231a6
- `_src/r3c_model.py` 668c73a28c654a49bfc6c81c91e634fba0be7dd153f26b3d00788b4df221a69b
- `_src/r3d_select.py` 2820fa2b3e3145e9f297db9ab21cbd39bdda62871678120f4d0050bb66918875
- `_src/r3e_select.py` da94d1d093dd5d533b481c80d4d445e3c4a9eb0ef0edf769df7152394a1322eb
- `_src/real_select.py` 11598b92dd84f70c61b7d212b5c4423d1a2b6498304e477bed9f6cb9d325cc36
- `_src/refload.py` 7ca459815845425fc95605719daaed9e3d35009744e4d7080eec12fe01303bb2
- `_src/scaleout_rho.py` 4647732e101b9eab1d99e5c63a34db18e70ff18bba8d71d43315b2a4ea0a07ba
- `_src/series_r3e.py` 81a10d02244d5a8edcc72209e6ba8f394d633de370b9946b76576b391fcecaca
- `_src/verify_cut_r3c.py` fa902121c90b04f54f4e34cc54e7ec543dc2f33d52e1d0b4717a16fa1cb15ae1
- `_src/write_half_sources.py` 3710a277048c75d24c32543ab54de481fedb2299b528f5d78207f898b8c4780c
- `_src/write_sources.py` cdd7a98e302020bbddf6d4964420b9b71dee7532c46abf9390c1c20a156172ad
- v1 generator `/data/nfs_shared_data/xxy/tools/r3d-genenv-20261101/v1gen` git  (dirty src: none)

## Use

Campaign: `trace_root` = this directory, `loadgen_config_root` = `${TRE_REPO}/loadgen_v1/configs/traces_v14` (trace names unchanged). Half runs: `../traces-icse-r3e-half400-20261101/` (requests with timestamp < 400 s). Specs (not run): `/data/nfs_shared_data/xxy/campaign-icse-r3e-20261101/`.
