# tre_replayer — trace replay, calibration load, and the one sending client

Since 2026-09-30 every request the tree puts on a gateway goes through one client, the
sending core in `tre_replayer/engine/`. The calibration drivers (`deploy/scripts/openloop.py`,
`r3_grid.py`, `calibration_campaign.py`), the trace replays (`tre_replayer.run_trace`,
`campaign_queue`) and the E1 client (`python3 -m tre_loadgen_v1`, now a shell) differ only
in their **profile**.

| module | role |
|---|---|
| `engine/api.py` | the fixed-length request (body, headers, endpoint checks) |
| `engine/profiles.py` | `calib` / `replay` / `e1_v1`; v1's request parameters (`V1ChatOptions`); client provenance |
| `engine/transport.py` | the wire: pooled async `httpx` (calib, replay) and v1's `openai.AsyncOpenAI` (e1_v1); `send_sync` |
| `engine/stream.py` | `StreamParser`, the only SSE parser: strict and v1 view of one answer; `StreamResult` |
| `engine/metrics.py` | both metric bases per request, summaries |
| `engine/http_sender.py` | `StreamingHttpSender`: one request = build, send, record (+ send lateness) |
| `engine/dispatcher.py` | `dispatch_open_loop`: fire each request at `base + offset` (asyncio) |
| `engine/procpool.py` | `ProcessPoolRunner`: N worker processes, pre-sharded schedule, shared start; `StopGate` |

## Profiles

| | `calib` | `replay` | `e1_v1` |
|---|---|---|---|
| endpoint | `/v1/chat/completions` | `/v1/completions` | `/v1/chat/completions` (SDK) |
| body | `api.request_body`: `model`, `messages`=[user], `max_tokens`, `temperature: 0`, `ignore_eos: true`, `stream`, `stream_options.include_usage`, `seed` if given | same with `prompt` | v1's `create()` kwargs: `model`, `messages`=[user], `temperature` from the model config (unset → `null`), `stream`, `stream_options.include_usage`, `max_tokens` = trace's, else config's, else absent; **no** `ignore_eos` |
| headers | `Content-Type`, `Accept: text/event-stream`, `model`, `routing-strategy` if set | same | SDK headers (`Authorization: Bearer dummy-key-for-local-gateway`, `X-Stainless-*`, UA `AsyncOpenAI/Python`), `routing-strategy` (config, `least-gpu-cache`) |
| prompt | materialised natural prompt, exact templated length | materialised | the trace's text, verbatim |
| transport | `httpx.AsyncClient`, keep-alive, sharded pools (64 connections, 16 idle kept per shard; shards up to `max_in_flight`), `Accept-Encoding: identity` | same | `openai.DefaultAsyncHttpxClient` (1000 / 100), as v1 |
| retries | none | none | SDK `max_retries` (default 2; `run_arm.sh` passes 0) |
| timeout | `max(30, max_tokens/4)` s per connect / read / write | same | config `timeout` (300 s) |
| record | calibration row (unchanged; `dual_metrics=True` adds both bases) | same | v1's `performance_metrics.json` line + audit + strict + lateness |
| processes | `--sender-processes` (default `DEFAULT_SENDER_PROCESSES` = 4) | 1 (in-process) | config `process_count` (v14 configs: 8) |

All three recognise the reissue sidecar: `x-tre-retried` (header), `x-tre-continued`
(header, `tre_continued` on the final chunk, `: x-tre-continued: N` SSE comment).

## Concurrency and timing

N worker processes × one asyncio loop each × one pooled client each (v1's model). The
schedule is sorted and sharded round-robin (request *k* → worker *k mod N*) before the
fork; every worker builds its client, reports ready, and after one "go" carrying a shared
`time.monotonic()` start fires its own requests at `start + offset`. Nothing blocks a
worker's loop (v1's `Queue.get(timeout=0.1)` froze it for up to 100 ms whenever its queue
was empty). Records stream back over a pipe as requests finish, so the parent can apply
the calibration stop rules live (`StopGate`: admission-overflow truncation, backlog
ceiling; same semantics as `openloop.TruncateOnProxyShed` / `StopOnBacklog`).

Every record carries its **send lateness** — `on_wire_delay_ms` (calib / replay) /
`send_lateness_ms` (e1): the scheduled instant to the transport call — decomposed into
`schedule_delay_ms`, `pool_wait_ms`, `body_build_ms`.

## Two metric bases (column names say which)

| | v1 basis (`ttft`, `tpot`, `e2e_latency`, `success`; `v1_*` on calibration rows) | strict basis (`*_strict*`; calibration rows' own `ttft_ms` / raw `tpot_ms`; `slo_labels`) |
|---|---|---|
| TTFT | first chunk with `choices[0].delta.content is not None` — the role-only chunk included | first chunk carrying `text` / `content` / `reasoning_content` / `reasoning` |
| TPOT | `(end − first) / completion_tokens`, end = end of body | `(E2E − TTFT) / (completion_tokens − 1)`, E2E at `[DONE]` |
| success | headers of a 2xx arrived: a cut stream, an in-stream error chunk, zero output all count | 2xx, complete (`[DONE]` or a finish reason), no in-stream error, ≥ 1 completion token (`failure_strict` says why not) |
| missing TTFT | dropped from percentiles | a violation (`ttft_missing_strict`) |
| retries | inside TTFT / E2E | excluded: timed from the last attempt; `retries`, `retry_wait_s` reported |

Calibration labels stay on the strict basis (`test_probe_label_parity`).

## Equivalence (2026-09-30, `scripts/verify_unified_client.py`, local fake server)

Old = `dcb8d5f3` (git archive), new = this branch; both driven by the same harness.

* **calib** (`/v1/chat/completions`, seed 7) and **replay** (`/v1/completions`,
  `routing-strategy: least-gpu-cache`), 10 scenario cells × 8 requests
  (normal, stop, reasoning field, no text, `x-tre-retried`, continued, Envoy 503 overflow,
  JSON 500, in-stream error, connection reset mid-body): request bodies **byte-identical**
  (80/80 each), explicit headers identical; sender records and raw rows identical in every
  non-clock field and field order, durations (`ttft_ms`, `e2e_ms`) within 1–3.5 ms
  (median); **one intended difference**: a stream whose connection is reset mid-body —
  old: urllib ended the iteration silently, the row was `http_status 200`, outcome `ok`
  (a served request); new: `http_status 0`, `error: RemoteProtocolError`, outcome
  `model_error`.
* **e1_v1** (old `python3 -m tre_loadgen_v1` vs new shell, 39 requests over 13 scenarios
  incl. retries 503×1/×2/×5, 500, mid-stream reset, read stall, header timeout, error
  chunk, reasoning, continued): the server received the same attempts with **byte-identical
  bodies and identical headers** (all SDK headers included); every non-clock field of
  every v1 record identical (success, status, tokens, finish reason, error message,
  `stream_interrupted` / `stream_error`, attempts, target pod); v1 formulas hold on every
  new record. Clock fields: the old client's TTFT was longer by 125 ms median (max 0.27 s /
  0.67 s in two runs) and E2E by ~220 ms median — its blocking hand-off, not the server.

* **gateway check** (3 single requests to dsqwen-7b through the tre-v2 gateway, no load
  running): new `calib` and `e1_v1` and the old calibration sender all answered 200 with
  `usage.prompt_tokens` 21 and 16 completion tokens (`ignore_eos` honoured), first token
  in `delta.content`, `target-pod` named; the role-only chunk and the first content chunk
  arrived 0.02 ms apart.

## Send timing benchmark (`scripts/bench_sender.py`, local fake server)

Local only: `fake_openai_server.py --mode bench` (4 worker processes, vLLM-shaped: 256
generating at once, the rest queued with headers sent; one chunk per decode step), client
and server on node 76 (64 cores, shared with the running fleet). Rates are taken from the
committed calibration schedules (`replayer/traces_v2/calibration/INDEX.json`):

* **burst** - the highest offered rate of any committed schedule, the `bursts` primitive's
  2 s spike: **202 rps** (1x) and 303 rps (1.5x), 10 s each, 256-token-ish prompts, 128
  output tokens, 20 ms decode steps (256 streams x 50 chunks/s: harsher than a real 7B
  batch of 256). In flight reached **2238** (the campaign's ceiling is
  `DEFAULT_MAX_IN_FLIGHT` 4096 / `burst_request_cap` 3840).
* **P1 deep overload** - 3.0 x rho* x C for the fastest P1 shape (dsqwen-7b S2, C = 8.2
  rps): **25 rps** (1x) and 37.5 rps (1.5x), 30 s, 768-token prompts, 192 output tokens,
  capacity ~12 rps (a growing queue, as in P1).
* **E1 peak** - the highest 1 s rate in the v14 E1 traces (Real_code_2024: 100 rps) and
  1.5x.

*send lateness* = scheduled instant -> the client's transport call, as the client
records it (`on_wire_delay_ms` / `send_lateness_ms`; for the old E1 client
`start_time - (base_time + timestamp)`). *arrival p99* = the server's arrival time minus
the scheduled instant; *TTFT overhead* = client TTFT minus the server's own
(first write - arrival). Both of those also contain the fake server's own accept/queue
delay (4 Python workers juggling up to ~2k connections), so they are upper bounds on the
client's share. CPU = user+sys of all client processes / wall.

| client | rate | send lateness p50 / p99 / max (ms) | arrival p99 (ms) | TTFT overhead p50 / p99 (ms) | CPU (cores) |
|---|---|---|---|---|---|
| **new, calib, 4 procs** (default) | burst 202 | 0.41 / **4.2** / 69 | 21 | 2.2 / 12 | 1.44 |
| | burst 303 | 0.39 / **6.8** / 59 | 25 | 2.0 / 15 | 1.60 |
| | P1 25 | 0.74 / **2.0** / 2.8 | 9.7 | 2.9 / 8.6 | 0.47 |
| | P1 37.5 | 0.76 / **1.8** / 3.6 | 9.3 | 2.9 / 8.2 | 0.52 |
| | E1 100 / 150 | 0.52 / 2.1 / 8.1 ; 0.49 / 1.6 / 9.5 | 9.7 ; 9.1 | 2.9 / 23 ; 2.7 / 8.3 | 1.53 ; 1.49 |
| new, calib, 1 proc | burst 202 / 303 | 42 / 215 / 259 ; 89 / 399 / 450 | 1408 ; 3556 | 338 / 1403 ; 736 / 3455 | 0.93 ; 0.94 (saturated) |
| | P1 25 / 37.5 | 0.65 / 6.6 / 119 ; 0.61 / 107 / 239 | 21 ; 60 | 3.4 / 14 ; 3.0 / 24 | 0.38 ; 0.45 |
| old calib sender (`dcb8d5f3`: asyncio + 4096 urllib threads) | burst 202 / 303 | 3553 / 7372 / 7524 ; 6844 / 15363 / 15682 | 7487 ; 15405 | 77 / 453 ; 82 / 503 | 1.26 ; 1.16 |
| | P1 25 / 37.5 | 1.5 / 5.3 / 71 ; 1.5 / 5.2 / 13 | 49 ; 27 | 1.2 / 39 ; 1.3 / 25 | 0.21 ; 0.26 |
| | E1 100 / 150 | 181 / 473 / 498 ; 1645 / 2995 / 3011 | 590 ; 3149 | 60 / 293 ; 79 / 423 | 1.09 ; 1.16 |
| **new, e1_v1, 8 procs** | burst 202 / 303 | 0.55 / 28 / 106 ; 0.56 / 36 / 78 | 77 ; 73 | 4.9 / 73 ; 5.1 / 62 | 2.72 ; 2.71 |
| | P1 25 / 37.5 | 0.80 / **2.5** / 3.8 ; 0.72 / **2.1** / 3.7 | 39 ; 77 | 4.2 / 38 ; 4.3 / 75 | 0.50 ; 0.41 |
| | E1 100 / 150 | 0.60 / **2.5** / 42 ; 0.56 / 21 / 75 | 25 ; 75 | 4.1 / 31 ; 4.4 / 75 | 2.67 ; 2.69 |
| old E1 client (`tre_loadgen_v1` @ `dcb8d5f3` = v1), 8 procs | burst 202 / 303 | 175 / 681 / 918 ; 263 / 1740 / 1806 | 7303 ; 7772 | 1117 / 7200 ; 1092 / 7107 | 2.68 ; 2.34 |
| | P1 25 / 37.5 | 69 / 143 / 168 ; 74 / 176 / 206 | 1495 ; 1868 | 327 / 1477 ; 346 / 1840 | 0.61 ; 0.62 |
| | E1 100 / 150 | 90 / 207 / 239 ; 122 / 375 / 422 | 6645 ; 11765 | 590 / 6573 ; 622 / 11497 | 2.16 ; 2.35 |

Reading it:

* The calibration profile meets **p99 send lateness < 10 ms at 1.5x every peak with 4
  processes** (`DEFAULT_SENDER_PROCESSES`); one process saturates a core at the burst
  peak (256 streams at 50 chunks/s), so bursts need >= 4. The old calibration sender kept
  up at P1 rates (p99 5 ms) but fell seconds behind at the burst peak - its cells there
  would have failed the 50 ms guard (`CALIBRATION_MAX_P99_DELAY_MS`).
* The e1_v1 profile meets it at the E1 traces' peak (100 rps: p99 2.5 ms) and at P1; at
  150-300 rps its p99 is 21-36 ms: the OpenAI SDK's own per-request work on each loop
  (the same SDK client as v1). Sharding the SDK clients (`V1ChatOptions.pool_shards`)
  only brought 28/36 ms to 21/29 ms. Above ~100 rps give E1 more processes (16).
* The old E1 client added **hundreds of ms to seconds** of client-made delay to every
  request at these rates: its worker loop blocks 100 ms in `Queue.get` whenever its queue
  is empty, and every stream on that loop waits with it (TTFT overhead p50 0.3-1.1 s,
  arrival p99 1.5-12 s). E1 latencies measured with it are not comparable with the new
  client's.
* httpcore's connection pool rescans all its connections (and, per idle one, all again)
  on every request start and end; a single pool of ~1-2k connections cost more CPU than
  the streams themselves (profile 2026-09-30). The calib/replay transport therefore uses
  pool shards of 64 connections (least-loaded, opened on demand, 16 idle kept each).

## Differences from the v1 client (CustomTraceGenerator `src/client_dispatcher.py`)

Against v1 = `/root/aibrix-main/CustomTraceGenerator/src/client_dispatcher.py` (its port
`tre_loadgen_v1` @ `dcb8d5f3` was line-for-line the same). "Impact" is measured on the
local fake server unless it says otherwise.

| item | v1 | unified client | impact |
|---|---|---|---|
| processes / coroutines | `process_count` (v14 configs: 8) processes, each `asyncio.run` + one `AsyncOpenAI` | same model: `e1_v1` uses the config's `process_count`; `calib` 4 (`--sender-processes`); `replay` 1 | none |
| hand-off | the main process puts the next 5 s window of requests on the least-loaded worker's `mp.Queue` every 0.5 s (first window split evenly); the worker reads it with a **blocking `Queue.get(timeout=0.1)` inside its event loop** | schedule pre-sharded round-robin before the fork; one "go" with a shared monotonic start; each worker fires at absolute times; nothing blocks a loop; records return over a pipe | v1: TTFT +125 ms median (max 0.27-0.67 s), E2E +~220 ms median on 39 scenario requests; under load 0.3-1.1 s TTFT overhead (table above) |
| start instant | `base_time = time.time()` right after starting the workers | 0.25 s after every worker built its client | v1's first requests could fall before its workers were ready |
| HTTP library / pool | openai SDK -> httpx, 1000 connections / 100 keep-alive / 5 s expiry, one pool per process | `e1_v1`: the same SDK client (optionally sharded, off by default); `calib`/`replay`: raw httpx, keep-alive, 64-connection pool shards up to `max_in_flight` (the old calibration sender: urllib, a new connection per request, `Connection: close`) | request bytes unchanged; transport headers differ for calib (`User-Agent: python-httpx`, `Connection: keep-alive`; `Accept-Encoding: identity` kept) |
| request, `e1_v1` | `chat.completions.create(model, messages=[user prompt], temperature=config (null), stream, stream_options.include_usage, max_tokens = trace's else config's)`, no `ignore_eos`; headers `routing-strategy: least-gpu-cache`, `Authorization: Bearer dummy-key-for-local-gateway`, SDK `X-Stainless-*` | identical (same SDK call) | server saw byte-identical bodies and identical headers |
| request, `calib` | (v1's calibration also sent chat) | chat, `temperature 0`, `ignore_eos`, fixed `max_tokens`, `stream` + `include_usage`, `seed` when given; = `dcb8d5f3` byte for byte | vs v1's experiment request: temperature 0 vs null, `ignore_eos`, synthetic fitted prompts |
| request, `replay` | - | completions, same fields; unchanged bytes | - |
| retries | SDK `max_retries=2` (hard-coded) | `e1_v1`: configurable, default 2 (`run_arm.sh` passes 0); `calib`/`replay`: never | same default |
| timeout | httpx 300 s per connect / read / write / pool (config `timeout`) | `e1_v1` same; `calib` `max(30, max_tokens/4)` s per operation (was a urllib socket timeout: same meaning) | none |
| TTFT | first chunk with `delta.content is not None` - the role-only chunk (`content: ""`) included | both: `ttft` (v1 basis) and `ttft_strict_s` / calibration `ttft_ms` (first non-empty text / content / reasoning) | strict >= v1; on the fleet vLLM sends both in one step (<1 ms apart, gateway check); one decode step when the first token's text is empty |
| TPOT | `(end - first) / completion_tokens`, end = end of body | v1 basis kept; strict `(E2E - TTFT) / (n - 1)`, E2E at `[DONE]` | gateway check, 16 tokens: 11.6 vs 12.4 ms |
| success | `create()` returned = success: a cut stream, an in-stream error chunk, zero output all succeed; only a `create()` exception fails | v1 column unchanged; `success_strict` / `failure_strict` (http_error, client_timeout, transport_error, stream_error, incomplete_stream, zero_output) | 39 scenario requests: v1 30 successes, strict 21 |
| retried requests | TTFT / E2E include the failed attempts and back-off | v1 basis unchanged; strict timed from the last attempt, `retries` / `retry_wait_s` | e.g. one 503 retry: +0.5-1 s in v1 E2E only |
| reissue sidecar marks | ignored (the SDK drops SSE comments and unknown fields) | every profile records `tre_continued` (header / final chunk / `: x-tre-continued` comment) and `tre_retried` (header) | new columns |
| random seed | dispatch has no randomness (`random.seed(None)` is in trace generation); SDK retry jitter uses `random` | same | none |
| prompt source | traces.json text | `e1_v1`: the same; `calib`/`replay`: materialised synthetic prompts (unchanged) | none for E1 |
| output | `performance_metrics.json` (17 fields), `process_log/worker_process_N.log`, `client_dispatcher_main.log`, `process_load_over_time.png` (worker-reported coroutines), `actual_send_rps_by_model.png` | the same 17 fields first (v1 basis) + the port's 5 audit fields + 15 new (`*_strict*`, `retries`, `retry_wait_s`, `http_status_strict`, `first_token_field`, `send_lateness_ms`, `schedule_delay_ms`, `in_flight_at_send`, `tre_continued`, `tre_retried`); `loadgen_run_meta.json` adds `client` provenance, `workers` (CPU), `metrics` (both bases); no `process_log/`; the process-load plot is rebuilt from the records | `smoke-e1-20260930/tools/analyze.py` runs unchanged on old and new files; strict columns added (None on old files) |
| send lateness | not recorded (log lines only) | per request | new column |
| calibration: stream reset mid-body | (old calibration sender) urllib ended the iteration silently: `http_status 200`, outcome `ok` | `http_status 0`, `RemoteProtocolError`, outcome `model_error` | the one record difference of the calibration path |

## Tools

* `scripts/fake_openai_server.py` — synthetic OpenAI-compatible SSE server (scenario /
  vLLM-shaped bench mode); never a model.
* `scripts/verify_unified_client.py` — `calib-run` / `e1-run` / `compare` (old vs new tree).
* `scripts/bench_sender.py` — `plan` / `run-calib` / `timed` / `analyze`.
