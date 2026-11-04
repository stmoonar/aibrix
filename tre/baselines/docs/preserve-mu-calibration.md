# PreServe mu calibration: the chain from raw data to `mu.yaml`

PreServe's per-replica serving capability mu_p / mu_d / mu_t (paper arXiv 2504.03702v2, Alg. 1 lines 8-11) by
the method the owner chose on 2026-11-04:

* Alg. 1 on **dataset A**, the TSS / theta training set (the sources and window filters of the 2026-11-03 TSS
  fit), 10 s windows of the datasets' own grid;
* a request belongs to the window its send time falls in; rate = the arrived requests' token sum / 10 s;
* a window is clean when no request failed and every request has NL = e2e / output tokens <= NL_SLO;
* mu = max over clean windows, per term;
* **envelope c**: only windows whose mean input length and mean output length each lie within [min, max] of the
  evaluation traces' 10 s arrival windows are judged.

`R` below is the data root (the directory that holds the calibration roots and the trace sets; a site keeps it
in its env file, e.g. `TRE_FIT_DATA_ROOT`). Commands run from the `tre/` tree with
`PYTHONPATH=common:baselines` unless noted. Steps a, b and the calibration runs of d drive load on GPUs:
they are documented here, not re-run. Every other step is CPU only and reads its inputs read-only.

| step | what | GPU | output |
|---|---|---|---|
| a | idle TTFT / TPOT capture, c/b fit | capture: yes; fit: no | `cb_idle/<model>/raw/idle/i0_o16_c1.jsonl`, `ttft_idle_fit.json`, registry c/b |
| b | isolated run (mix shape, isolated NL) | yes (buckets report: no) | `bl-calib-20261026/isolated/` |
| c | NL_SLO per model | no | `nl_slo.yaml` |
| d | dataset A | calibration runs: yes; datasets: no | 18 `dataset_hybrid/` dirs |
| e | envelope from the evaluation traces | no | `envelope_c.yaml` (+ a, b) |
| f | Alg. 1 with envelope c | no | `mu.yaml`, `result.json` |
| g | mu into the policy config | no | ConfigMap block (freeze step) |

## a. Idle TTFT and TPOT (2026-10-03; GPU)

Procedure: `tre/deploy/RUN-calib-theta-20261003.md` sections B (capture + fit) and C (c/b into the registry).
Engine `vllm-openai-tre:0.30.0-ts-8dc0f2a7`; chat API, streaming, one request in flight, >= 500 ms between
requests, L = 128 ... 4096 (7 lengths x 30), 16 output tokens, one pod.

```bash
cd tre/deploy; CAL=$PWD/scripts/calibration; C=$R/calib_20261003/cb_idle    # + the RUN doc's section 0 env
for m in dsqwen-7b dsllama-8b dsqwen-14b; do OUT_ROOT=$C bash $CAL/run_cb_idle.sh $m; done   # scripts.ttft_idle_capture
python3 -m scripts.ttft_idle_fit --root $C --models dsqwen-7b,dsllama-8b,dsqwen-14b --min-gap-ms 500 \
    --out $C/ttft_idle_fit.json --registry-patch                      # Huber fit; c/b rounded into registry slo
```

| file (under `R/calib_20261003/cb_idle/`) | sha256 |
|---|---|
| `dsqwen-7b/raw/idle/i0_o16_c1.jsonl` | `08182bf45376efdf70ce5f04329d3270e63381b58c7f93b6342d705882ce413e` |
| `dsllama-8b/raw/idle/i0_o16_c1.jsonl` | `d00af04cd0df59d1bff9c78df87b63a63e7884b951c338c44cfa16738a13f705` |
| `dsqwen-14b/raw/idle/i0_o16_c1.jsonl` | `e121869bda6ae970ab55541c8c325742d320c8b7e9c9f81127c7082f94f2b2d4` |
| `ttft_idle_fit.json` | `e31cb00088331fe70369faead2587631889440be7210722fdd78993bf96623dd` |

Registry `models[].slo.ttft_idle_c_ms` / `ttft_idle_b_ms_per_token`: 41.1 / 0.0694, 48.1 / 0.0717, 45.3 / 0.0879
(fit 41.092 / 0.06944, 48.149 / 0.07170, 45.280 / 0.08792). Offline part re-run 2026-11-04: `ttft_idle_fit` on
the raw files gives a `ttft_idle_fit.json` equal to the original in every key except `generated_utc`.

## b. Isolated run (2026-10-26; GPU)

The mix shape and the nine cell centres come from the buckets report of the 720 s trace set (CPU); the isolated
run sends one request at a time per shape to an idle pod (5 measured runs, 2 warm-up runs of the mix shape).

```bash
python3 -m tre_baselines.tools.tokenscale_buckets --trace-root $R/traces-v2-720-20261026 \
    --traces Decode_burst,Prefill_mix,Real_code,Real_conv,Simul_spike,Sinusoidal,Tight_pool \
    --out $R/bl-calib-20261026/buckets-report.yaml
python3 -m tre_baselines.tools.velocity_ladder isolated --report $R/bl-calib-20261026/buckets-report.yaml \
    --sm-url <service-manager URL> --out-dir $R/bl-calib-20261026/isolated --i-have-user-approval   # tools 5b14b41d
```

| file (under `R/bl-calib-20261026/`) | sha256 |
|---|---|
| `buckets-report.yaml` | `c29e33ef9b79e9e600ebdcd863c2597eb3190e306e5cfdabd90d60acbc1b1c6b` |
| `isolated/isolated-meta.json` | `10418983464299cbafd847edfbc20abf1d6e21144a5d727ac99797f317d8af43` |
| `isolated/dsqwen-7b.jsonl` | `cf4e31e886816b9d71d088cd13e2c2cf38bac0d2d93e589fd8a16eb5bb1ac71d` |
| `isolated/dsllama-8b.jsonl` | `20d3296e0ae13683114cdec5417176610626ecd26fdb3f54880f814057e1687c` |
| `isolated/dsqwen-14b.jsonl` | `72d5d15685e28bd282c6112163d851369f7217b145854df56ef4c91202f21329` |

Mix shape (L x out): 7b 459 x 271, 8b 442 x 243, 14b 463 x 294. The buckets report re-run 2026-11-04 is
byte-identical.

## c. NL_SLO (CPU)

`tools/preserve_nl_slo.py`, config `tools/preserve_nl_slo.yaml` (models, input paths). Since 2026-11-04 the SLO
parameters are the registry's `models[].slo.idle_multiple` (`tre_common.slo` mode `idle_multiple`: ttft_k / tpot_k
7b, 8b 3 / 5, 14b 5 / 3, ttft_floor_ms 450, tpot_idle_ms), not the config's:

* TTFT_slo(L) = max(floor, ttft_k (c + b L)), c / b from the registry;
* TPOT_slo = tpot_k x tpot_idle_ms; the tool recomputes TPOT_idle = median `tpot_ms` of the `outcome == ok` idle
  records (step a) and refuses a registry whose `tpot_idle_ms` is not that median;
* NL_SLO = (TTFT_slo(L_mix) + (out_mix - 1) TPOT_slo) / out_mix at the mix shape (step b), rounded to 0.01 ms.

```bash
python3 -m tre_baselines.tools.preserve_nl_slo --config baselines/tre_baselines/tools/preserve_nl_slo.yaml \
    --data-root $R --registry deploy/registry.yaml --out $OUT/nl_slo
```

Output `$OUT/nl_slo/nl_slo.yaml` (every intermediate value, inputs with sha256, no time stamp):

| model | TPOT_idle ms | TTFT_slo(L_mix) ms | TPOT_slo ms | NL_SLO ms | / isolated NL |
|---|---|---|---|---|---|
| dsqwen-7b | 12.188 | 450 | 60.94 | **62.38** | 5.00 |
| dsllama-8b | 12.675 | 450 | 63.37 | **64.97** | 5.03 |
| dsqwen-14b | 13.982 | 450 | 41.95 | **43.34** | 3.04 |

The registry used on 2026-11-04 (`deploy/registry.yaml`, sha256 `e262765b…`) is recorded in the file. The alg1
config pins these three values (`nl_slo_expect_ms`); alg1 refuses an `nl_slo.yaml` with other values.

## d. Dataset A (calibration runs: GPU; datasets: CPU)

Sources per model (`tools/preserve_mu_alg1.yaml` `sources`, 6 per model, 18 dirs of `windows.csv` /
`requests.csv` / `cells.csv`):

* **old** (`run1`, `run2` (14b: `run2b`), `supp`, `p1` (7b: `p1r2`)): the 2026-10-03 calibration runs
  `R/calib_20261003/<stage>/<model>` (`tre/deploy/RUN-calib-theta-20261003.md` section D, accepted attempts in
  section E and `calib_20261003/ATTEMPTS.md`). Datasets built 2026-10-30 by `run_alphafit_w10.sh` step 1
  (`WINDOW_MS=10000 STEP_MS=10000`, builder main `9f2856a1`: `scripts.calibration_dataset --attribution hybrid
  --queue-estimator raw_mean --numerator vllm_counter --label-registry <registry of e891d29b> --window-ms 10000
  --step-ms 10000`) into `R/analysis-tss-params-20261030/win10/alphafit-w10000-s10000/datasets/`.
* **suppA, suppB**: the 2026-11-03 calibration supplement blocks A and B
  (`tre/deploy/scripts/calibration/run_chain_supp.sh`) in `R/calib_supp_20261103/{A,B}`, datasets built by
  `build_datasets.sh` (builder `97a4d4ab`) into `R/calib_supp_20261103/datasets/w10000/{A,B}/`.

Rebuild from raw (TSS fit toolchain `tre/calibration/scripts/fit_tss/`, its README; site env file with
`TRE_FIT_DATA_ROOT` and the exclusive-window values):

```bash
. <site env>/fit_tss.env
FT=tre/calibration/scripts/fit_tss
bash $FT/build_datasets.sh --old $OUT/build/old 10000                        # the 12 old sources
bash $FT/build_datasets.sh $R/calib_supp_20261103/A A $OUT/build 10000
bash $FT/build_datasets.sh $R/calib_supp_20261103/B B $OUT/build 10000
```

Evidence that the offline part reproduces:

* 2026-11-03 rebuild (builder `ff6398f9`) into `R/calib_supp_20261103/fit/adhoc/fit-tss-repo-checks-20261103/build/`
  (fit_tss README, "Equivalence check"): A / B `windows.csv`, `requests.csv`, `cells.csv` byte-identical; old the
  same except `windows.csv`, which has alternative-signal columns added after `9f2856a1` (build with `TRE` = a
  checkout of `9f2856a1` for byte-identical old roots).
* 2026-11-04: step f on those rebuilt datasets (the alg1 config with the 18 paths replaced) gives the same
  `mu.yaml` values, `models` / `mu` / `rule` of `result.json` and the same `windows.csv.gz` content as on the
  original datasets (alg1 reads the columns it needs by name).

The 54 input files and their sha256 are in `result.json` `provenance.files` (610 MB in all).

## e. Envelope (CPU)

`tools/preserve_eval_envelope.py`, config `tools/preserve_mu_envelope.yaml`: the 7 traces of
`eval/runner/campaigns/campaign-bl-w10-preserve-20261103.yaml` (seed 1, `traces_tre.effective.json`, sha256
pinned; the tool refuses another file), 10 s arrival windows, output length = `max_output_tokens` or the
model's `max_tokens` of the trace's loadgen config; `default: c`.

```bash
python3 -m tre_baselines.tools.preserve_eval_envelope --config baselines/tre_baselines/tools/preserve_mu_envelope.yaml \
    --data-root $R --out $OUT/envelope
```

| output | sha256 |
|---|---|
| `envelope_c.yaml` (lengths-minmax, the default) | `7c4e05f62cf899fba4e7178cb9196f9df45e13ebe23d9345c36c5a0a9f612a52` |
| `envelope_a.yaml` (share-minmax, sensitivity) | `839bcf48c2c49251e5cfe57896f38f4c45e5ca1f52d9e5badc1788fb1ae151d9` |
| `envelope_b.yaml` (share-p5p95, sensitivity) | `fbfa37f914a44edcec21cbf75ee353837c8da96d245e8d40f2849c529e581040` |
| `eval_stats.json` | `d61e1828fd357f6c0206b79f85e6990b0e4e6eb19e91e59ab25bcdbd1910e23b` |

`provenance.json` records the tool sha and the config / trace / loadgen config sha256.

## f. Alg. 1 with envelope c (CPU, about 20 s)

```bash
python3 -m tre_baselines.tools.preserve_mu alg1 --config baselines/tre_baselines/tools/preserve_mu_alg1.yaml \
    --data-root $R --nl-slo $OUT/nl_slo/nl_slo.yaml --envelope $OUT/envelope/envelope_c.yaml --out $OUT/alg1
```

alg1 refuses: an NL file off the pins; no `--envelope` (the config names `envelope_default`; `--no-envelope`
for the unfiltered row); an envelope of another name (`--envelope-variant` for rows a / b); an envelope whose
window or trace sha256 differ from `preserve_mu_envelope.yaml`; an existing `result.json`.

Result (tok/s per replica; `windows.csv.gz` content sha256 `a3d6089459b4b1dc77c32d7e2ba15a2614ec526e40d205cfbf46fa84b4383dfd`):

| model | mu_p | mu_d | mu_t | clean / judged (outside envelope) |
|---|---|---|---|---|
| dsqwen-7b | 10137.6 | 7424.0 | 12672.0 | 1296 / 2035 (2834) |
| dsllama-8b | 8140.8 | 5056.0 | 10176.0 | 1423 / 1946 (2779) |
| dsqwen-14b | 6374.4 | 4480.0 | 7968.0 | 1254 / 1881 (2836) |

`mu.yaml` (policy format) and `result.json` carry the tool sha, the config sha256, the NL and envelope files with
sha256 and every dataset file with sha256.

## g. Where mu goes

The policy parameter `mu` (`{model: {p, d, t}}`, `examples/preserve.yaml`; required for every managed model).
The deployed block of the `tre-v2-baseline-preserve` ConfigMap in `deploy/baselines/tre/policy-configmaps.yaml`
is rendered from `mu.yaml` by the freeze step (`python3 -m tre_baselines.tools.freeze_params preserve --mu
<mu.yaml> --write`, README "Freeze step"); a change of the live config follows the deployment rules.
