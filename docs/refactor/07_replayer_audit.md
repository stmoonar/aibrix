# P7 Replayer Audit

## Initial Dispatcher Audit

The frozen replayer dispatcher lives in `/root/aibrix-main/CustomTraceGenerator/src/client_dispatcher.py`. It already uses absolute request timestamps (`base_time + request.timestamp`) inside worker coroutines, but scheduling, OpenAI request handling, multiprocessing load balancing, metrics plotting, and result persistence are coupled in one large module. That makes the P7 offline precision contract hard to test directly.

Observed first-slice risks:

- Open-loop behavior depends on workers creating per-request tasks without awaiting individual responses. This needs a small pure dispatcher test so a future refactor cannot accidentally serialize requests.
- Scheduled vs actual timestamps are collected as incidental request fields instead of a reusable timing report. P7 needs a report with P99 delay and RPS error for the offline stub test.
- Arrival schedules are generated elsewhere as per-second RPS segments. P7 needs a deterministic pre-generated schedule module before adding Poisson arrivals and trace config loading.

## First Refactor Slice

Added `tre/replayer/tre_replayer/engine/schedule.py` and `dispatcher.py`:

- `build_deterministic_schedule()` converts half-open RPS segments into absolute-offset request events.
- `dispatch_open_loop()` sleeps to each scheduled absolute timestamp, starts sender tasks without waiting for earlier responses, and returns scheduled-vs-actual timing records.
- Tests use injected clock/sleep hooks for deterministic precision checks without contacting the cluster.

Remaining P7 work: Poisson schedules, trace config loading for existing trace files, 60s aiohttp-stub precision test, capacity/design/lint/oracle tooling, coverage matrix, and final lint/oracle reports for existing traces.
