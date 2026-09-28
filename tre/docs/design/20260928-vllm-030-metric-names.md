# vLLM 0.30 metric names: where TRE resolves them

Date: 2026-09-28. Branch `tre/pd-metrics`.

The model pods moved to the vLLM 0.30 fork image (plan D9). vLLM deprecated several
Prometheus families in 0.10.x and dropped the old names in 0.11+. A reader that names only
the old family reads nothing, and it fails silently: the AIBrix APA fetcher treats a missing
metric as 0 and never scales, least-gpu-cache routes at random, and a histogram stays empty.

## Renamed families

| Quantity | vLLM 0.10.1 exports | vLLM 0.30 exports |
|---|---|---|
| KV-cache fill (0..1) | `vllm:gpu_cache_usage_perc` **and** `vllm:kv_cache_usage_perc` | `vllm:kv_cache_usage_perc` |
| Per-token latency histogram | `vllm:time_per_output_token_seconds` | `vllm:inter_token_latency_seconds` (same semantics) |
| Prefix-cache queries / hits | `vllm:gpu_prefix_cache_{queries,hits}_total` **and** `vllm:prefix_cache_{queries,hits}_total` | `vllm:prefix_cache_{queries,hits}_total` |

These are unchanged: `num_requests_running`, `num_requests_waiting` (0.30 counts sleep-paused
requests in it; `num_requests_paused` is a subset of it), `time_to_first_token_seconds`,
`e2e_request_latency_seconds`, `request_prompt_tokens`, `request_generation_tokens`.
`num_requests_swapped` does not exist on the V1 engine, so readers treat it as 0.

Samples: `tre/deploy/tests/fixtures/vllm_metrics/vllm-0.30.0.prom` is a live pod capture.
`vllm-0.10.1.prom` is the stock 0.10.1 stat logger rendered without a GPU; its header gives
the generator.

## One name table per language, newest name first

- **Go**: `engineMetricEquivalents` in `pkg/metrics/engine_fetcher.go`. For a metric
  definition that maps any member of a group, the engine fetcher reads the first member the
  engine exports. This covers every gateway reader that goes through the fetcher: the
  `pkg/cache` pod metrics, the least-gpu-cache and least-kv-cache routers, prefix-cache
  metrics, the TRE Redis instant and histogram docs, and the APA `RestMetricsFetcher` in
  our fork. The PromQL TPOT metrics in `pkg/metrics/metrics.go` use
  `inter_token_latency_seconds or time_per_output_token_seconds`. They are inactive in
  tre-v2 because `PROMETHEUS_ENDPOINT` is unset. `AvgTPOT5mPod` also divided `_sum` by
  `_sum`; it now divides by `_count`.
- **Python**: `tre/common/tre_common/vllm_metrics.py`.
  - `VLLM_METRICS` holds the raw `vllm:` names. The SM sleep drain check
    (`sleep_primitive.parse_vllm_load`) and the openloop / r3 / calibration-ladder pod
    sidecar (`deploy/scripts/openloop.py`) use it.
  - `GATEWAY_DOC_KEYS` holds the Redis doc keys. The controller `MetricsStore` resolves
    through it for TPOT and the KV-cache fill.
  - The gateway resolves engine names itself. It still writes the AIBrix identifiers
    `time_per_output_token_seconds` and `gpu_cache_usage_perc` into `tre:v2:*` docs, so
    the doc schema did not change. Readers prefer `inter_token_latency_seconds` and
    `kv_cache_usage_perc` when those keys are present.
- No other TRE component parses vLLM metrics. The UI, replayer, loadgen_v1 and calibration
  measure latency on the client side or read the controller's windows. The reissue sidecar
  proxies `/metrics` unchanged.

Guard tests:
- `tre/deploy/tests/test_vllm_metric_names.py` checks that every table entry, the APA
  `targetMetric` and the text parsers resolve against both samples.
- `tre/controller/tests/test_metrics_store_metric_names.py` checks the old and new doc keys
  and the e2e p95.
- `pkg/metrics/engine_fetcher_samples_test.go` checks the Go fetcher against both samples.

## APA baseline `targetMetric` (for the operator to apply later)

`tre/deploy/baselines/apa/*-apa.yaml` now use `targetMetric: kv_cache_usage_perc`. The
aibrix-system controller-manager (7d3535b1) maps it to `vllm:kv_cache_usage_perc`. Both
0.10.1 and 0.30 export that name, so the change is safe for either image. No
PodAutoscaler was in the cluster on 2026-09-28. The next `deploy/scripts/toggle_tre_apa.sh apa`
applies the new name (it runs `kubectl -n default apply -f` on the anchors and CRs of this
directory). A CR applied from an older checkout still carries `gpu_cache_usage_perc`. On a
0.30 pod APA then reads 0 and never scales. Never run APA alongside TRE.

## Not changed here (policy)

The aibrix-system gateway-plugins (7d3535b1) has no name fallback, and ADR-0008 means we
do not touch it. Any traffic routed through the aibrix-system gateway with least-gpu-cache
reads `vllm:gpu_cache_usage_perc`, which 0.30 pods lack, and so it routes at random. The
tre-v2 gateway (our fork, 27923798 and this change) is not affected.

## e2e histogram check (2026-09-28)

A report said the Redis docs lacked the e2e histogram. The live state does not reproduce
this:
- All 44 `tre:v2:hist:*` zsets carry `<model>/e2e_request_latency_seconds` in every doc.
- The live 0.30 `/metrics` exports `vllm:e2e_request_latency_seconds`. The name did not
  change.

A SafeScale probe shows `latency_source: avg_ttft` (its `p95_e2e_ms` field then holds the
TTFT mean). That comes from the `TRE_MIN_LATENCY_SAMPLES=10` guard: under 10 completed
requests per pod per window gives no p95. The metric is present; the fallback triggers
because too few requests completed in the window.
