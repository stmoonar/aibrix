# Gateway Plugin Environment Variables

This document covers all environment variables used in the `pkg/plugins/gateway` package and its sub-packages.

---

## General Gateway

| Variable | Type | Default | Description | Source |
|---|---|---|---|---|
| `POD_NAME` | string | `""` | Kubernetes pod name. Used for logging and metric label tagging. | [gateway.go](gateway.go), [util.go](util.go) |
| `ROUTING_ALGORITHM` | string | _(none)_ | Default routing algorithm when no per-request override is set. | [types.go](types.go), [util.go](util.go) |

---

## TRE transparent sleep (`tre_transparent_sleep.go`)

Gateway side of the TRE sleep/wake handshake with the service-manager. Redis keys
(`tre:v2:gw:instances`, `tre:v2:gw:seen:<pod>`, `tre:v2:gw:inflight:<pod>`, all keyed by pod
name), the pod annotation `tre.aibrix.io/route-gen` and the request header
`x-tre-exclude-pod` (comma separated; repeated headers are merged) are fixed by the
cross-component contract and are not configurable.

| Variable | Type | Default | Description | Source |
|---|---|---|---|---|
| `TRE_GW_COORDINATION` | bool | value of `TRE_ROUTABLE_LABEL_FILTER` | Enable the instance heartbeat, route-gen acks and inflight mirror in Redis. Fail-closed: when wanted but it cannot start (no `TRE_ROUTABLE_LABEL_FILTER=true`, no Redis, no Kubernetes API, or Redis/apiserver unreachable for `TRE_GW_STARTUP_TIMEOUT`) the plugin exits non-zero instead of routing without it. | [tre_transparent_sleep.go](tre_transparent_sleep.go) |
| `TRE_GW_INSTANCE_ID` | string | `POD_NAME`, then hostname | Instance id used as ZSET member / hash field. | [tre_transparent_sleep.go](tre_transparent_sleep.go) |
| `TRE_GW_HEARTBEAT_INTERVAL` | duration | `2s` | Heartbeat period of `tre:v2:gw:instances`. The score is the Redis server time (`TIME`), not the gateway host clock; the `ts` fields of seen/inflight use the same clock. | [tre_transparent_sleep.go](tre_transparent_sleep.go) |
| `TRE_GW_INSTANCE_RETENTION` | duration | `10m` | Heartbeat entries older than this are pruned from the ZSET. | [tre_transparent_sleep.go](tre_transparent_sleep.go) |
| `TRE_GW_KEY_TTL` | duration | `300s` | TTL of the seen and inflight hashes, renewed on every write. | [tre_transparent_sleep.go](tre_transparent_sleep.go) |
| `TRE_GW_REFRESH_INTERVAL` | duration | `30s` | Period of the full re-ack / inflight rewrite (a restarted instance re-acks at startup). | [tre_transparent_sleep.go](tre_transparent_sleep.go) |
| `TRE_GW_STARTUP_TIMEOUT` | duration | `60s` | Bound on the retries of each startup step (quorum pod LIST, reset of own inflight fields, first ack/inflight refresh, first heartbeat). | [tre_transparent_sleep.go](tre_transparent_sleep.go) |
| `TRE_GW_REDIS_RETRY_MAX` | duration | `10s` | Cap of the exponential backoff (from 250ms) after failed Redis writes; failures are logged and counted in `tre_gateway_redis_errors_total{op}`. | [tre_transparent_sleep.go](tre_transparent_sleep.go) |
| `TRE_GW_RETRY_AFTER_SECONDS` | int | `1` | `Retry-After` on 503s from pod selection (no routable pod, all candidates excluded, commit race, instance shutting down). | [tre_transparent_sleep.go](tre_transparent_sleep.go) |
| `TRE_DEFAULT_ROUTING_STRATEGY` | string | off | Optional last-resort strategy when neither the `routing-strategy` header, the config profile nor `ROUTING_ALGORITHM` names one. Prefer the upstream `ROUTING_ALGORITHM` (older images honour it too). `""`, `none` or `off` disable it. | [tre_transparent_sleep.go](tre_transparent_sleep.go), [util.go](util.go) |
| `TRE_ROUTABLE_LABEL_FILTER` | bool | `false` | Only pods labelled `tre.aibrix.io/routable=true` are routing candidates. | [pkg/utils/pod.go](../../utils/pod.go) |
| `TRE_ROUTE_MODEL_HEADER` | bool | `false` | Stamp the body model onto the `model` request header. | [tre_route_model_header.go](tre_route_model_header.go) |

Behaviour with coordination on:

- Every counted request holds its inflight slot until its ext_proc stream ends (response
  `end_of_stream`, error, client disconnect, shutdown). Usage in a stream chunk
  (`stream_options.continuous_usage_stats`) does not end it.
- A request that resolves to no routing strategy is routed per pod with `random` instead of
  the HTTPRoute/Service path, so it is counted and bound to the acked route table.
- Routers that do not select from this request's candidate list synchronously are refused
  with 400: queue routers (`slo*`) and `pd` (its prefill pod would not be counted).
- On startup the route table is seeded from a quorum pod LIST (ResourceVersion `""`,
  label `tre.aibrix.io/routable`), and per pod object the route-gen never goes back, so a
  restarted instance with the same id does not ack or route on a stale watch-cache state.
  This needs `list` on pods cluster-wide, which the pod informer already requires.
- On pod deletion this instance's fields in `seen:<pod>` / `inflight:<pod>` are deleted;
  a same-name replacement starts from zero.
- On graceful shutdown new commits get 503 + `Retry-After` before the instance leaves
  `tre:v2:gw:instances` and clears its inflight fields.

Continuation classification (`non_continuable` in the inflight value, D6): requests that
cannot be resumed from their emitted tokens and must be drained rather than aborted are
`n>1`, `best_of>1`, any logprobs output, `echo`, beam search, tool/function calling
(streaming or not, unless `tool_choice`/`function_call` is `"none"`), structured output /
guided decoding (`response_format` other than `text`, `guided_json`, `guided_regex`,
`guided_choice`, `guided_grammar`, `guided_json_object`, `structural_tag`,
`structured_outputs`) and every endpoint other than completions and chat completions.
Known semantic drift that is *not* classified as non-continuable: in a continuation the
tokens generated before the seam are part of the prompt, so `presence_penalty` and
`frequency_penalty` (which count generated tokens only) no longer penalise them
(`repetition_penalty` covers prompt and output and is unaffected), and a fixed `seed`
restarts its random stream at the seam. A continued sampled output is therefore valid but
not bit-identical to an uninterrupted one; greedy decoding without these penalties is
unaffected. `max_tokens` / `min_tokens` are the continuation sender's responsibility.

---|---|---|---|---|
| `TRE_DEFAULT_ROUTING_STRATEGY` | string | `least-gpu-cache` | Strategy for requests that name none (no `routing-strategy` header, no config profile, no `ROUTING_ALGORITHM`), so every request goes through ext_proc pod selection. `""`, `none` or `off` disables it (upstream HTTPRoute/Service path). | [tre_transparent_sleep.go](tre_transparent_sleep.go), [util.go](util.go) |
| `TRE_GW_COORDINATION` | bool | value of `TRE_ROUTABLE_LABEL_FILTER` | Enable the instance heartbeat, route-gen acks and inflight mirror in Redis. Refuses to start without `TRE_ROUTABLE_LABEL_FILTER=true`. | [tre_transparent_sleep.go](tre_transparent_sleep.go) |
| `TRE_GW_INSTANCE_ID` | string | `POD_NAME`, then hostname | Instance id used as ZSET member / hash field. | [tre_transparent_sleep.go](tre_transparent_sleep.go) |
| `TRE_GW_HEARTBEAT_INTERVAL` | duration | `2s` | Heartbeat period of `tre:v2:gw:instances`. | [tre_transparent_sleep.go](tre_transparent_sleep.go) |
| `TRE_GW_INSTANCE_RETENTION` | duration | `10m` | Heartbeat entries older than this are pruned from the ZSET. | [tre_transparent_sleep.go](tre_transparent_sleep.go) |
| `TRE_GW_KEY_TTL` | duration | `300s` | TTL of the seen and inflight hashes, renewed on every write. | [tre_transparent_sleep.go](tre_transparent_sleep.go) |
| `TRE_GW_REFRESH_INTERVAL` | duration | `30s` | Period of the full re-ack / inflight rewrite (a restarted instance re-acks at startup). | [tre_transparent_sleep.go](tre_transparent_sleep.go) |
| `TRE_GW_RETRY_AFTER_SECONDS` | int | `1` | `Retry-After` on 503s from pod selection (no routable pod, all candidates excluded). | [tre_transparent_sleep.go](tre_transparent_sleep.go) |
| `TRE_ROUTABLE_LABEL_FILTER` | bool | `false` | Only pods labelled `tre.aibrix.io/routable=true` are routing candidates. | [pkg/utils/pod.go](../../utils/pod.go) |
| `TRE_ROUTE_MODEL_HEADER` | bool | `false` | Stamp the body model onto the `model` request header. | [tre_route_model_header.go](tre_route_model_header.go) |

---

## Response Processing

| Variable | Type | Default | Description | Source |
|---|---|---|---|---|
| `AIBRIX_TTFT_THRESHOLD_S` | int (seconds) | `1` | Time-to-first-token threshold in seconds. Requests exceeding this are flagged in response processing. | [gateway_rsp_body.go](gateway_rsp_body.go) |

---

## Redis Sync (`statesync/`)

| Variable | Type | Default | Description | Source |
|---|---|---|---|---|
| `AIBRIX_STATESYNC_ENABLED` | bool | `false` | Enable cross-replica state sync via Redis. Must be `true` to activate the statesync manager. | [cmd/plugins/main.go](../../../cmd/plugins/main.go) |
| `AIBRIX_STATESYNC_SYNC_PERIOD` | duration | `10s` | Interval at which gateway state is synced to Redis across replicas. | [statesync/redissync.go](statesync/redissync.go) |

---

## Prefix Cache Router (`algorithms/prefix_cache.go`)

| Variable | Type | Default | Description | Source |
|---|---|---|---|---|
| `AIBRIX_PREFIX_CACHE_TOKENIZER_TYPE` | string | `"character"` | Tokenizer type for prefix cache hashing. Options: `character`, `tiktoken`, `remote`. | [algorithms/prefix_cache.go](algorithms/prefix_cache.go) |
| `AIBRIX_PREFIX_CACHE_POD_RUNNING_REQUEST_IMBALANCE_ABS_COUNT` | int | `8` | Absolute running-request count difference threshold that triggers load-imbalance routing. | [algorithms/prefix_cache.go](algorithms/prefix_cache.go) |
| `AIBRIX_PREFIX_CACHE_STANDARD_DEVIATION_FACTOR` | int | `1` | Factor multiplied by the standard deviation of pod loads during imbalance calculation. | [algorithms/prefix_cache.go](algorithms/prefix_cache.go) |
| `AIBRIX_PREFIX_CACHE_USE_REMOTE_TOKENIZER` | bool | `false` | Use a remote HTTP tokenizer service instead of the local tokenizer. Requires `AIBRIX_PREFIX_CACHE_TOKENIZER_TYPE=remote`. | [algorithms/prefix_cache.go](algorithms/prefix_cache.go) |
| `AIBRIX_PREFIX_CACHE_KV_EVENT_SYNC_ENABLED` | bool | `false` | Enable KV cache event synchronization across gateway replicas. When `true`, also requires `AIBRIX_PREFIX_CACHE_USE_REMOTE_TOKENIZER=true`. | [algorithms/prefix_cache.go](algorithms/prefix_cache.go) |
| `AIBRIX_PREFIX_CACHE_REMOTE_TOKENIZER_ENDPOINT` | string | `""` | Remote tokenizer service endpoint URL. Required when `AIBRIX_PREFIX_CACHE_KV_EVENT_SYNC_ENABLED=true`. | [pkg/constants/kv_event_sync.go](../../constants/kv_event_sync.go) |
| `AIBRIX_PREFIX_CACHE_KV_EVENT_PUBLISH_ADDR` | string | `""` | ZMQ publish address for KV cache events. Used when KV event sync is enabled. | [pkg/constants/kv_event_sync.go](../../constants/kv_event_sync.go) |
| `AIBRIX_PREFIX_CACHE_KV_EVENT_SUBSCRIBE_ADDRS` | string | `""` | ZMQ subscribe addresses for KV cache events (comma-separated). Used when KV event sync is enabled. | [pkg/constants/kv_event_sync.go](../../constants/kv_event_sync.go) |

### Remote Tokenizer Pool (`algorithms/prefix_cache.go`)

These configure the pool of remote tokenizer connections used when `AIBRIX_PREFIX_CACHE_USE_REMOTE_TOKENIZER=true`.

| Variable | Type | Default | Description |
|---|---|---|---|
| `AIBRIX_VLLM_TOKENIZER_ENDPOINT_TEMPLATE` | string | `"http://%s:8000"` | HTTP endpoint template for tokenizer pods. `%s` is replaced with the pod name. |
| `AIBRIX_TOKENIZER_HEALTH_CHECK_PERIOD` | duration | `30s` | How often to health-check tokenizer pool members. |
| `AIBRIX_TOKENIZER_TTL` | duration | `300s` | TTL for cached tokenizer connections in the pool. |
| `AIBRIX_MAX_TOKENIZERS_PER_POOL` | int | `100` | Maximum number of tokenizer connections in the pool. |
| `AIBRIX_TOKENIZER_REQUEST_TIMEOUT` | duration | `5s` | Timeout for individual remote tokenizer requests. |

---

## Preble (Prefix Cache with Histogram) Router (`algorithms/prefix_cache_preble.go`)

| Variable | Type | Default | Description |
|---|---|---|---|
| `AIBRIX_ROUTER_PREBLE_TARGET_GPU` | string | `"V100"` | GPU model used for hardware-specific latency estimates in the Preble algorithm. |
| `AIBRIX_ROUTER_PREBLE_DECODING_LENGTH` | int | `45` | Expected decode sequence length used for cache allocation decisions. |
| `AIBRIX_ROUTER_PREBLE_SLIDING_WINDOW_PERIOD` | int (minutes) | `3` | Sliding window length in minutes for histogram metrics collection. |
| `AIBRIX_ROUTER_PREBLE_EVICTION_LOOP_INTERVAL` | int (ms) | `1000` | Interval in milliseconds between cache eviction loop executions. |

---

## VTC (Virtual Token Counter) Router

### Token Tracker (`algorithms/vtc/token_tracker.go`)

| Variable | Type | Default | Description |
|---|---|---|---|
| `AIBRIX_ROUTER_VTC_TOKEN_TRACKER_WINDOW_SIZE` | int | `5` | Sliding window size (in `TIME_UNIT` units) for token usage tracking. |
| `AIBRIX_ROUTER_VTC_TOKEN_TRACKER_TIME_UNIT` | string | `"minutes"` | Time unit for the sliding window. Options: `minutes`, `seconds`, `milliseconds`. |
| `AIBRIX_ROUTER_VTC_TOKEN_TRACKER_MIN_TOKENS` | float64 | `1000.0` | Minimum token count threshold for adaptive load normalization. |
| `AIBRIX_ROUTER_VTC_TOKEN_TRACKER_MAX_TOKENS` | float64 | `8000.0` | Maximum token count threshold for adaptive load normalization. |

### VTC Basic Scorer (`algorithms/vtc/vtc_basic.go`)

Scoring formula: `score = (fairnessWeight * normFairness + utilizationWeight * normUtilization) / normFreeGPU`

| Variable | Type | Default | Description |
|---|---|---|---|
| `AIBRIX_ROUTER_VTC_BASIC_MAX_POD_LOAD` | float64 | `100.0` | Load value at which a pod is considered fully saturated. |
| `AIBRIX_ROUTER_VTC_BASIC_INPUT_TOKEN_WEIGHT` | float64 | `1.0` | Weight applied to input tokens when computing pod load. |
| `AIBRIX_ROUTER_VTC_BASIC_OUTPUT_TOKEN_WEIGHT` | float64 | `2.0` | Weight applied to output tokens when computing pod load (typically higher than input). |
| `AIBRIX_ROUTER_VTC_BASIC_FAIRNESS_WEIGHT` | float64 | `1.0` | Weight of the fairness component in the routing score. |
| `AIBRIX_ROUTER_VTC_BASIC_UTILIZATION_WEIGHT` | float64 | `1.0` | Weight of the utilization component in the routing score. |

---

## PD (Prefill-Decode) Disaggregation Router (`algorithms/pd_disaggregation.go`)

| Variable | Type | Default | Description |
|---|---|---|---|
| `AIBRIX_PREFILL_REQUEST_TIMEOUT` | int (seconds) | `30` | HTTP request timeout for prefill pod calls. |
| `AIBRIX_PREFILL_LOAD_IMBALANCE_MIN_SPREAD` | int32 | `16` | Minimum (max − min) running-request spread across prefill pods to trigger load-imbalance routing. |
| `AIBRIX_DECODE_LOAD_IMBALANCE_MIN_SPREAD` | float64 | `16.0` | Minimum (max − min) running-request spread across decode pods to trigger load-imbalance routing. |
| `AIBRIX_DECODE_THROUGHPUT_IMBALANCE_MIN_SPREAD` | float64 | `2048.0` | Minimum (max − min) token-throughput spread (tokens/s) across decode pods to trigger throughput-imbalance routing. |
| `AIBRIX_DECODE_SCORE_RATIO_THRESHOLD` | float64 | `1.5` | Max/min drain-rate score ratio above which the slowest decode pod is excluded from selection. |
| `AIBRIX_PROMPT_LENGTH_BUCKETING` | bool | `false` | Route requests to prefill pods whose prompt-length bucket matches the request length. |
| `AIBRIX_KV_CONNECTOR_TYPE` | string | `"shfs"` | KV cache transfer backend. Options: `shfs` (GPU shared memory), `nixl` (Neuron). |
| `AIBRIX_PREFILL_SCORE_POLICY` | string | `"prefix_cache"` | Strategy for selecting the prefill pod. Options: `prefix_cache`, `least_request`. |
| `AIBRIX_DECODE_SCORE_POLICY` | string | `"load_balancing"` | Strategy for selecting the decode pod. Options: `load_balancing`, `least_request`. |

### Decode Load Balancer Scorer (`algorithms/pd/decode_scorer.go`)

Scoring formula: `score = (wRun × normRunning + wThroughput × normInvThroughput) / normFreeGPU`

| Variable | Type | Default | Description |
|---|---|---|---|
| `AIBRIX_DECODE_LB_WEIGHT_RUNNING` | float64 | `1.0` | Weight for the normalized running-request term in the decode LB score. |
| `AIBRIX_DECODE_LB_WEIGHT_THROUGHPUT` | float64 | `1.0` | Weight for the normalized inverse-throughput term in the decode LB score. |

---

## Utilities (`algorithms/util.go`)

| Variable | Type | Default | Description |
|---|---|---|---|
| `AIBRIX_TRT_MACHINE_ID` | int64 | `0` | 10-bit machine ID (0–1023) used in Snowflake-style disaggregation request ID generation: `[timestamp:41b][machineID:10b][counter:12b]`. Panics on init if out of range. |

---

## Variable Dependency Notes

The following variables have interdependencies that must be satisfied together:

- **KV event sync** requires all three to be set consistently:
  ```
  AIBRIX_PREFIX_CACHE_KV_EVENT_SYNC_ENABLED=true
  AIBRIX_PREFIX_CACHE_USE_REMOTE_TOKENIZER=true
  AIBRIX_PREFIX_CACHE_TOKENIZER_TYPE=remote
  AIBRIX_PREFIX_CACHE_REMOTE_TOKENIZER_ENDPOINT=<url>
  ```

- **Remote tokenizer pool** is initialized whenever `AIBRIX_PREFIX_CACHE_USE_REMOTE_TOKENIZER=true`. The pool variables (`AIBRIX_VLLM_TOKENIZER_ENDPOINT_TEMPLATE`, `AIBRIX_TOKENIZER_*`, `AIBRIX_MAX_TOKENIZERS_PER_POOL`) all apply in that case.

## OpenTelemetry

The gateway plugins feature built-in support for distributed tracing via OpenTelemetry (OTel), empowering you to monitor and trace end-to-end requests across the entire external processing pipeline.

**Tracing is opt-in by default.** Telemetry components will only initialize if you explicitly configure either ``OTEL_EXPORTER_OTLP_ENDPOINT`` or ``OTEL_EXPORTER_OTLP_TRACES_ENDPOINT``. If both are omitted, all tracing capabilities remain disabled to conserve system resources.

| Variable | Type | Default | Description                                                                                                                                                                                                                       | Source                                                                                                             |
|---|---|---------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------|
| `OTEL_EXPORTER_OTLP_PROTOCOL` | string | `grpc`  | The transport protocol for OTLP data. Valid options are `grpc`, `http`, or `http/protobuf`.                                                                                                                                       | [cmd/plugins/main.go](../../../cmd/plugins/main.go)                                                                |
| `OTEL_EXPORTER_OTLP_ENDPOINT` | string | `""`    | Base URL for all OTLP signals. **⚠️ Note:** ensure your ENDPOINT URLs include the `http://` or `https://` prefix. The SDK automatically appends signal-specific paths to this URL (e.g., appending `/v1/traces` for HTTP exports). | [cmd/plugins/main.go](../../../cmd/plugins/main.go)                                                                |
| `OTEL_EXPORTER_OTLP_TRACES_ENDPOINT` | string | `""`    | Target URL specifically for traces. Takes precedence over the global `ENDPOINT`. **⚠️ Note:** This URL is used exactly **as-is**. The SDK will **NOT** automatically append `/v1/traces` to it.                                   | [cmd/plugins/main.go](../../../cmd/plugins/main.go)                                                                |
| `OTEL_EXPORTER_OTLP_HEADERS` | string | `""`    | Key-value pairs used as headers for OTLP requests (e.g., `key1=value1,key2=value2`). Useful for passing auth tokens to backends like Datadog or Honeycomb.                                                                        | [cmd/plugins/main.go](../../../cmd/plugins/main.go)                                                                |
| `OTEL_EXPORTER_OTLP_TIMEOUT` | string | `10s`   | Maximum time the OTLP exporter will wait for each batch export.                                                                                                                                                                   | [cmd/plugins/main.go](../../../cmd/plugins/main.go)                                                                |
| `OTEL_EXPORTER_OTLP_INSECURE` | bool | `false` | Set to `true` to disable TLS/HTTPS for the exporter. Force downgrade to HTTP.                                                                                                                                                     | [cmd/plugins/main.go](../../../cmd/plugins/main.go)                                                                |
| `OTEL_EXPORTER_OTLP_INSECURE_SKIP_VERIFY` | bool | `false` |  Keeps TLS active but skips server certificate validation (useful for self-signed certs). Set to "true" to enable. | [cmd/plugins/main.go](../../../cmd/plugins/main.go)                                                                      |

> **Note on OpenTelemetry Configuration:**
> Aibrix supports the standard OpenTelemetry Protocol Exporter environment variables. For advanced OTLP configurations (such as `_CERTIFICATE`, `_CLIENT_KEY`, or specific `_COMPRESSION` settings), please refer to the [OpenTelemetry Protocol Exporter](https://opentelemetry.io/docs/specs/otel/protocol/exporter/)
