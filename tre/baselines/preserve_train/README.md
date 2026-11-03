# preserve_train: offline trainer of the PreServe Tier-1 mLSTM

Produces `tre/baselines/weights/preserve-mlstm-*/`: the numpy `.npz` weights that the PreServe baseline
(`tre_baselines/policies/preserve_predictor.py`) loads. Offline and CPU only; needs torch
(`requirements-train.txt`). Nothing in `tre_baselines` imports this package, so the baseline-scaler image
(no torch) never runs it; the Dockerfile copies `baselines/` whole, so these files are in the image but inert.

## One command

```bash
cd tre/baselines
pip install -r preserve_train/requirements-train.txt          # torch 2.13.0, numpy 2.2.6 (CPU is enough)
PYTHONPATH=. python3 -m preserve_train.pipeline all \
  --azure-root <datasets/azure-llm-inference> \
  --window-s 10 --file-suffix=-paper-w10 --jobs 12 --threads 1 \
  --train-dir <new training dir> \
  --weights-out weights/preserve-mlstm-<date>-paper-w10 \
  [--check-cache code=<30 s cache> --check-cache conv=<30 s cache>] [--ref-train-dir <30 s training dir>]
```

Steps (each is also a subcommand, see `pipeline.py --help`):

1. `windows`: Azure CSV -> per-window sums `windows_<stream>_<W>s.npz` in the training dir (no pandas).
   `--check-cache` first reproduces given caches at `--check-window-s` (default 30) exactly, e.g. the v1
   caches of the 30 s run; any difference stops the pipeline.
2. `train`: `train_mlstm.py --protocol paper_past --window-s W` (all other arguments = the trainer's
   defaults = the v1 protocol below); writes `<set><suffix>.npz` (with `window_s`), `manifest.json`,
   `REPORT.md` in the training dir.
3. `finish`: copies the npz files to the weights dir, checks they load with the policy's loader and
   declare W, adds the naive last-window MAPE on the same held-out targets (and the reference window's
   with `--ref-train-dir`), writes `REPORT.md` + `manifest.json` there.

`compare --a X.npz --b Y.npz` checks that two weight files hold identical arrays (`window_s` aside).

Datasets are not in git: the Azure LLM inference 2024 week traces (`AzureLLMInferenceTrace_code_1week.csv`
sha256 `71de5c55cbc35f8f1ed0b6b7806b4cd1e9764b0058469725a6aac98023a1448f`,
`AzureLLMInferenceTrace_conv_1week.csv` sha256 `a0cc9b969a9bbf0fd811802cbf4323edd3a209ace791e3799ad4f9207f213941`;
the trainer checks both) are expected under `<azure-root>/2024/`.

## Protocol (v1 `paper_past`, unchanged)

Real-time windows of W s from 00:00 UTC of each file's first day (P = sum ContextTokens, D = sum
GeneratedTokens); the v1 Real_* slice intervals excluded (conv 2024-05-13 09:00-11:00, code 2024-05-11
00:00-00:30 UTC); `code` trains on code windows before 2024-05-11 00:00, `conv` on conv windows before
2024-05-13 09:00, `pooled` on the first half of each stream; seq_len 25, hidden 64, 50 epochs, batch 32,
Adam lr 1e-3, MSE, seed 42, eps 1.0; shipped normalization `per_sequence_mean` (global min-max trained for
comparison only); held-out MAPE as upstream `train_predictor.py`. Upstream model: PreServe
`mLSTM_predict.py` (`model.py` docstring).

## Provenance

Copied from the TRE v1 repository (`github.com/stmoonar/Token-service-share_Rebalancing_Engine`, local
`tre_code`), commit `80571402c94b326e3e7d34fbdbbdacfc3a791784` (2026-10-24, "tre_preserve: mLSTM protocol
paper_past"), directory `python/tre_preserve/train/`: the trainer of `weights/preserve-mlstm-20261024-paper`
(its REPORT.md cites this commit).

| file here | original (80571402) | original sha256 | change |
|---|---|---|---|
| `data.py` | `train/data.py` | `1c24acb6e2c6f7aa6be1366ad6ceb0dd159c052a81abacc8976d111eb77efcbc` | none (byte-identical) |
| `model.py` | `train/model.py` | `ebf1591c2fa7faad3d7fdbbb606422c131d2f542e371ca7995b7637cc48785af` | none (byte-identical) |
| `report.py` | `train/report.py` | `9bf3995150fb9a9baa83b485b07d42c8d538817ce0167f4bd2136fec855d7121` | none (byte-identical) |
| `export.py` | `train/export.py` | `e14b97b744bc510f26601e77396765cc1d4c33da9b2ce6302654f18afb7069d5` | constants from `tre_baselines.policies.preserve_predictor` (the v2 port of `tre_preserve/predictor.py`, original sha256 `c1b122ae96d339099cc27a7c84fd2cb4c806732132eb20ff4b8c6e21891e8ce3`); writes the scalar `window_s`; 2026-11-04: optional `normalization` (`minmax` + per-kind `norm_min` / `norm_max`) and scalar `meta` keys (default output unchanged) |
| `train_mlstm.py` | `train/train_mlstm.py` | `0d73924238090c9c68c9bb112412479b817e4a8b279294222886cbe9b09ce47f` | imports (`preserve_train.*`, the v2 predictor for the numpy vs torch check); passes `window_s` to the export |
| `pipeline.py` | - | - | new (2026-11-03): pandas-free window caches, naive MAPE, REPORT, one command |
| `trace_train.py`, `configs/trace-train-20261104.json` | - | - | new (2026-11-04): trained on each trace's own past, see below |

Later tre_code change not taken: 40c6bbc (2026-10-24 22:12, item 7) lets `data.iso_to_ns` accept a trailing
`Z`. `data.py` stays the copy of 80571402, the commit that trained `preserve-mlstm-20261024-paper`; the
trainer passes it only `+00:00` times (exclusions, cut-offs), where both versions agree.

Not copied: `eval_pre_t0.py`, `eval_traces.py` (v1 evaluations on v1 traces; they do not produce weights).
The manifest key `dirty_tre_preserve_train` keeps its v1 name; here it means "preserve_train has
uncommitted changes".

## Trained on the trace's own past (`trace_train.py`, 2026-11-04)

The paper trains a service's predictor on that service's past (section 5.3 RQ2). `trace_train.py` makes that past
for each evaluation trace: the trace's own generator (a copy of its set's `_src/`, sha256-checked against the
evaluation manifest) and spec, run with seeds no evaluation uses (config `train_seeds`), 4 realizations x 72
windows of 10 s = 288 windows. V1 = one predictor per (trace, model) on its 4 realizations; V2 = one per model on
all 7 traces concatenated. Same hyper-parameters as above, normalization min-max over the training series
(paper Alg. 1; constants stored in the file). History before t0 = the last 25 windows of the training series,
`preserve_history_w10_<history_variant>.json` (policy `history_variant`). Generated data (realizations, series,
histories) stay outside git (`--out`, with `SOURCES.md`); the generators are not copied into the repository.

```bash
cd tre/baselines
X=<data root with the trace sets>; OUT=<datasets>/preserve-train-trace-20261104
PYTHONPATH=. python3 -m preserve_train.trace_train realize --data-root $X --out $OUT \
  --v1-gen <v1 CustomTraceGenerator> --azure-root <datasets/azure-llm-inference> --cpus 56-63   # 76: v1 generator
PYTHONPATH=. python3 -m preserve_train.trace_train series --data-root $X --out $OUT
PYTHONPATH=. python3 -m preserve_train.trace_train train --series $OUT/series_w10.json --train-dir <dir>   # torch
PYTHONPATH=. python3 -m preserve_train.trace_train finish --data-root $X --out $OUT --series $OUT/series_w10.json \
  --train-dir <dir> --weights-out weights/preserve-mlstm-20261104-trace-w10 \
  --history-sha256-out histories/preserve_history_w10_train-tail-20261104.sha256
```
