# Trace-set configs of the 7 traces in use (2026-11-03)

Trace data and generators are NOT in the repo (user decision 2026-11-03). What is here is what a campaign spec points to plus the
checksums that pin the data:

* The loadgen config each spec uses is `tre/loadgen_v1/configs/traces_v14/<trace>/config.yaml` (`loadgen_config_root`), already in
  main; `traces.tsv` lists its path and sha256 (equal to `config_sha256` in every `trace_manifest.json` of the runs).
* `<set>/<trace>/manifest.json`: the generator's record of the trace (rate factor k, seeds, generator script and its sha256, source
  sha256s, PreServe history method). Copy of `<data root>/<set>/<trace>/seed1/manifest.json`.
* `<set>/<trace>/preserve_history.json`: the PreServe history file the baseline arm reads next to the trace (small: ~9 KB, causal,
  from the Azure counts before the slice). Copy; sha256 in `traces.tsv`.
* `<set>/<trace>/gen/config.yaml`, `gen/trace.json`: the generator-time loadgen config and the rate timeline of the trace set
  (`<data root>/<set>/_gen/<trace>/seed1/`); the Real_* sets also name per-model `real_traces/*.json` design lists (about 0.5-1 MB
  each, not copied; sha256 in `TRACE_FILES.sha256`). The copies keep the cluster paths they were generated with.
* `TRACE_FILES.sha256`: sha256 of every file above plus the trace files `traces_tre.effective.json` (45-89 MB), relative to the data
  root. Check on a machine that has the data: `cd "$CAMPAIGN_DATA_ROOT" && sha256sum -c <repo>/tre/loadgen_v1/configs/trace_sets/TRACE_FILES.sha256`.

| trace | set | requests | trace sha256 (prefix) |
|---|---|---:|---|
| Alternating_hot_model_periodic_A | traces-icse-r3c-20261101 (k 1.65) | 23319 | 574ab1b5eb03 |
| Simultaneous_spike_ramp_twice_tps1o2 | traces-icse-r3c-20261101 (k 1.325) | 8479 | 6a2ce8f782b8 |
| Prefill_mixed_corner_decode_mix | traces-icse-r3c-20261101 (k 1.7) | 19594 | 91821e6f5d92 |
| Decode_heavy_burst | traces-icse-ktrace-20261031 (k 1.2) | 26091 | f4ed951263fc |
| Sinusoidal_demand | traces-icse-ktrace-20261031 (k 1.3) | 28122 | 5e1e3233bd0a |
| Real_conv_2023_slice_a_tok70 | traces-icse-r3e-20261101 | 52181 | 431c96e7ff7b |
| Real_code_2024_slice_a_tok70 | traces-icse-r3e-20261101 | 46005 | 457df889b8bd |

Not included: the `SOURCES.md` / `audit.md` prose of the three sets (design rules and audits; they cite workspace scripts), the
other traces of those sets, and the older sets the first-batch specs name (traces-icse-20261030, the ktrace set's own
Alternating / Prefill_mixed / Simultaneous / Real_* that the 10-31 batch ran).
