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


## Trace Loading and Poisson Schedules

Existing `trace.json` files under the frozen `config/traces_v14/*/` directories are model-keyed JSON objects. Each model maps to segment dictionaries with `start_time`, `end_time`, `rps`, `input_tokens`, and `max_tokens`. The new loader keeps that format intact and maps it to `RpsSegment` records without requiring YAML config parsing.

`build_poisson_schedule()` now pre-generates arrival timestamps with an explicit seed. This satisfies P7's requirement that the dispatcher consumes a fixed schedule rather than deriving arrivals from response progress. Token controls from the trace segments are copied into every scheduled request for later prompt/token construction.


## Trace Set Discovery

`discover_trace_set()` reads `INDEX.json` when present and then scans every immediate child directory containing `trace.json`. Indexed workloads are listed first and marked `indexed=True`; additional trace folders are retained and marked `indexed=False` instead of being silently ignored.

Read-only check against `/root/aibrix-main/CustomTraceGenerator/config/traces_v14` parsed 5 cases: `Simultaneous_spike_ramp_twice_tps1o2` from the index plus unindexed `Alternating_hot_model_periodic_A`, `Decode_heavy_burst`, `Prefill_mixed_corner_decode_mix`, and `Sinusoidal_demand`.


## Capacity Surface Foundation

Added `tre_calibration.capacity` as the first capacity-surface building block for P7 trace linting. It fits the max SLO-safe RPS at each `(model, input_tokens, output_tokens)` grid point and marks out-of-grid lookups as `nearest_extrapolated` with `low_confidence=True`. This is intentionally conservative until real training-grid interpolation is added.


## Lint Foundations

Added `tre_replayer.lint` with the first C1/C2/C3 checks from section 12. C1 computes normalized occupancy from `rho = rps / C_m(i,o)` times model slot width and rejects traces above 95% of total slots. C2 accumulates time where any model exceeds `rho > 1.2` and rejects traces that do not trigger scaling for at least three slow-loop periods. C3 checks the declared headroom tier (`loose`, `medium`, or `tight`) with the plan's +/-0.05 tolerance.

The current C1 implementation is the instantaneous feasibility bound; the full oracle violation-rate check remains to be added in `oracle.py`.


## Oracle Lower Bound Foundation

Added `tre_replayer.oracle.compute_oracle_lower_bound()` as the first hand-checkable oracle metric. It partitions the trace by segment boundaries, computes required slots from normalized demand and model slot width, and reports total duration, unavoidable overcapacity duration, violation fraction, and max required slots. This is a conservative lower-bound foundation; future slices can add warm-switch timing and integer slot-shape constraints.


## Oracle in Lint Reports

`TraceLintReport` now includes `oracle_violation_fraction`, and C1 combines both plan checks: instantaneous headroom must stay within 95% of total slots and the oracle lower-bound violation fraction must stay below 1%. Short spikes still fail C1 if their instantaneous demand exceeds the hard headroom bound, matching section 12.3.
