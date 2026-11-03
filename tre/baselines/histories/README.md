# PreServe history files (data, not in git)

The trace sets live outside the repository; this directory records which history files a run must find.
Check from the data root (the directory that holds the trace sets):

```bash
cd <data root> && sha256sum -c <aibrix>/tre/baselines/histories/preserve_history_w10-20261103.sha256
```

## `preserve_history_w10-20261103.sha256`

The 10 s pre-t0 histories (`window_s` 10, 25 windows) of the seven ICSE-suite traces, seed 1 (the rep2 set:
r3c Alternating / Simultaneous / Prefill, ktrace Decode_heavy / Sinusoidal, r3e Real_code / Real_conv). Written
2026-11-03 on 76 by `tre_baselines/tools/preserve_history_azure.py` at aibrix 1c661670 (the files record
the tool's path, profile and sha256), one call per set:

| set | `--profile` | `--v1-gen` |
|---|---|---|
| `traces-icse-r3c-20261101` | `r3c` | `traces-icse-r3c-20261101.design` |
| `traces-icse-ktrace-20261031` | `icse` | v1 `CustomTraceGenerator` |
| `traces-icse-r3e-20261101` | `r3e` | `traces-icse-r3e-20261101.design` |

with `--suite <set> --k-map <set>/_checks/k_map.json --v9 <azure>/v1-retrace/traces_v9 --azure-root <azure>
--window-s 10 --seeds 1 --trace ...`. The same tool at the same commit reproduced every 30 s
`preserve_history.json` of these traces (seeds 1-3 where present, 11 files) byte for byte once the two fields
naming the generator (path, sha256) are set to the old copy's values (`--verify`). The rows of the 10 s files
equal those of a first 10 s run made earlier that day by a wrapper around the old copies (no run used it).

## `preserve_history_w10_train-tail-20261104.sha256`

The 10 s pre-t0 histories of the same seven traces for the predictors trained on each trace's own past
(`weights/preserve-mlstm-20261104-trace-w10`, policy `history_variant: train-tail-20261104`): the last 25 windows
of the trace's training series = the end of its last training realization (seed 104), method
`train-realization-tail-v1-20261104`, written by `preserve_train/trace_train.py finish`. Paths are relative to the
generated-data directory (`<datasets>/preserve-train-trace-20261104`, see its SOURCES.md), not to the trace sets:
these files are not next to the evaluation traces yet (a run copies the file next to the trace, or sets
`history_path`).
