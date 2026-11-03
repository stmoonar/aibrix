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
