# P3 Metrics Pipeline

Date: 2026-07-04
Environment: remote server 76, `/data/nfs_shared_data/xxy/aibrix`

## Writer Contract

Source: `pkg/cache/cache_tre_redis.go` (`TRE-PATCH(P2-GW-003)`). The gateway writes TRE pod metrics every `RequestTraceWriteInterval`, rounded down to the interval boundary in milliseconds.

Redis v2 keys shared with `tre/common/tre_common/rediskeys.py`:

| Key | Type | Score / Value | Retention |
| --- | --- | --- | --- |
| `tre:v2:hist:{pod}` | Sorted Set | `score=timestamp_ms`, `member=JSON` histogram snapshot | `ZREMRANGEBYSCORE` older than 30 minutes plus 2h TTL |
| `tre:v2:inst:{pod}` | Sorted Set | `score=timestamp_ms`, `member=JSON` instant snapshot | same |
| `tre:v2:pods:{model}` | Set | pod keys reporting that model | 2h TTL refreshed by writer |

Histogram members contain `model_histogram_metrics` keyed as `{model}/{metric}`. Values are cumulative snapshots with `sum`, `count`, and `buckets` where bucket values are cumulative counts. Instant members contain `model_metrics` keyed as `{model}/{metric}` with gauge values.

## Field Semantics

| Output field | Source metric | Unit | Aggregation |
| --- | --- | --- | --- |
| `prompt_tokens` | `request_prompt_tokens.sum` | tokens/window | per-pod `max(0, last.sum - first.sum)`, then sum across pods |
| `generation_tokens` | `request_generation_tokens.sum` | tokens/window | same as prompt tokens |
| `avg_waiting` | `num_requests_waiting` | requests | per-pod sum of samples divided by expected sample count; then sum across pods |
| `avg_running` | `num_requests_running` | requests | same as `avg_waiting` |
| `avg_swapping` | `num_requests_swapped` | requests | same as `avg_waiting` |
| `kv_cache_hit_rate` | `kv_cache_hit_rate` | ratio | per-pod window average; then average across pods with samples |
| `ttft_p95_ms` | `time_to_first_token_seconds.buckets` | ms | per-pod cumulative bucket delta; default percentile mode is `bucket_upper` |
| `tpot_p95_ms` | `time_per_output_token_seconds.buckets` | ms | same as TTFT |
| `e2e_p95_ms` | `e2e_request_latency_seconds.buckets` | ms | same as TTFT |
| `routable_pods` | `tre:v2:pods:{model}` plus window data | count | pods with histogram or instant docs in the requested window |
| `assigned_replicas` | temporary P3 fallback | count | equals `routable_pods` until service-manager v2 state is available in P4/P5 |

Window semantics for this slice: callers pass explicit `[window_start_ms, window_end_ms]`; P5 `metrics_task` will choose the last complete interval-aligned window. The store caches completed `(model, start, end)` windows in process memory so fast-loop reads in the same window do not re-read Redis.

## Legacy Compatibility Notes

The first P3 implementation preserves the old collector formulas from `/root/aibrix-main/python/tre/monitor/collector.py`: histogram counters use first/last deltas, gauge values divide by expected sample count rather than observed sample count, and p95 defaults to bucket upper bounds to keep old fitted thresholds valid.

## Verification Log

### P3-METRICS-001 v2 window store

RED:

```bash
PYTHONPATH=tre/common:tre/controller python3 -m pytest -q tre/controller/tests/test_metrics_store.py
```

Result: failed during collection because `tre_controller.store.metrics_store` did not exist.

GREEN:

```bash
PYTHONPATH=tre/common:tre/controller python3 -m pytest -q tre/controller/tests/test_metrics_store.py
cd tre && make check
cd tre && make smoke
```

Result: all passed on server 76. The tests cover v2 sorted-set window reads, legacy histogram first/last deltas, instant gauge expected-sample averaging, default bucket-upper p95, and completed-window cache reuse.


### P3-METRICS-002 v1 compatibility reads

RED:

```bash
PYTHONPATH=tre/common:tre/controller python3 -m pytest -q tre/controller/tests/test_metrics_store.py -k v1_legacy
```

Result: failed because `MetricsStore.__init__()` did not accept a `schema` mode and had no legacy-key read path.

GREEN:

```bash
PYTHONPATH=tre/common:tre/controller python3 -m pytest -q tre/controller/tests/test_metrics_store.py
cd tre && make check
cd tre && make smoke
```

Result: all passed on server 76. The v1 path scans only `aibrix:pod_histogram_metrics_*` and `aibrix:pod_instant_metrics_*`, normalizes timestamp suffixes to milliseconds, filters docs for the requested model, and reuses the v2 aggregation semantics.


### P3-METRICS-003 fixture edge cases and snapshots

RED:

```bash
PYTHONPATH=tre/common:tre/controller:tre/controller/tests python3 -m pytest -q tre/controller/tests/test_metrics_store.py -k snapshot
```

Result: failed because `MetricsStore` did not expose `read_snapshot()`.

GREEN:

```bash
PYTHONPATH=tre/common:tre/controller:tre/controller/tests python3 -m pytest -q tre/controller/tests/test_metrics_store.py
cd tre && make check
cd tre && make smoke
```

Result: all passed on server 76. The fixture helper covers out-of-order v2 writes, missing instant samples, and counter reset clamping. `read_snapshot()` now returns `MetricsSnapshot` for every model in the registry, including zero-valued models with no data in the requested window.

## Remaining P3 Work

- Add multi-window scenarios to the fixture generator and benchmark the 3 model x 8 pod x 30 minute fixture target.
- Compare the new store against the frozen old collector on the same fixture and document any remaining differences.
