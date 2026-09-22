#!/usr/bin/env python3
"""R3 load-grid driver (endgame plan 6.2): sweep input x output x concurrency
against a model, emit a window-level CSV consumable by the calibration `fit` CLI
(columns: scenario_id, scenario_family, prompt_tokens_total, generation_tokens_total,
p95_ttft, p95_tpot, trs). Reuses the controller MetricsStore for window aggregation
and TRSComputer for the trs column, so metric parsing is not reimplemented.

Checkpoints per cell (resumable). The full grid is a ~10h/model run (R3, wall-clock);
validate on a single cell (--only-first-cell) before the long run.

RAW LOGGING (S4, doc15 §4): R3 is the most expensive step (~10h/model). To make the data
re-windowable offline (e.g. re-fit theta at 20s after capturing at 60s) WITHOUT a re-run,
`drive_cell` streams each request and appends one per-request line to a local-disk JSONL
`<raw-dir>/<cell_id>.jsonl` (schema: send_ts_ms/recv_first_token_ts_ms/done_ts_ms/
input_tokens/output_tokens/ttft_ms/tpot_ms/e2e_ms/http_status/cell_id) plus an instant
queue sidecar `<cell_id>.instant.jsonl` (ts_ms/waiting/running/swapping — queue is an
instant sample, never in the per-request record). `rewindow_from_raw.py` re-aggregates
those into window CSVs at any --window-ms/--step-ms, reusing this module's window_row +
compute_window_results so the trs column is byte-identical to the online path.

CAVEAT (authoritative R3): the online CSV below re-windows TUMBLING at window_ms (dt=window_ms),
whereas live control uses a SLIDING window refreshed every ~5s. For the authoritative trs
column, re-aggregate from the raw log with a sliding window at the live refresh step
(`rewindow_from_raw --window-ms=<W> --step-ms=<refresh>`); the tumbling series here is the
online quick-look. The streaming raw-logger reuses the replayer's http_sender SSE/usage
parser (`tre_replayer.engine.http_sender`), so there is a single sender/parse implementation.

OPEN-LOOP MODE (`--schedule <file>`): the worker-pool driver above is closed-loop, so it
can never offer more load than the engine drains and never produces a waiting queue (see
`scripts/openloop.py` for the measurements that forced this). With `--schedule` the cell is
driven instead from a replayer trace file through `dispatch_open_loop`: requests fire at
wall-clock offsets regardless of completions. The raw JSONL / instant sidecar schemas and
the window CSV are unchanged, so `rewindow_from_raw.py` and `tre_calibration` consume both
modes identically -- except that the sidecar samples at `--instant-sample-ms` (1000 ms for
the calibration campaign, vs the live 10 s gateway grid), so an offline re-window must be
given the matching `--instant-sample-ms`.
"""
from __future__ import annotations

import argparse
import csv
import itertools
import json
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Optional, Sequence

from tre_common.rediskeys import SCRAPE_INTERVAL_MS

from scripts import openloop


@dataclass(frozen=True)
class GridCell:
    input_tokens: int
    output_tokens: int
    concurrency: int

    @property
    def scenario_id(self) -> str:
        return f"i{self.input_tokens}_o{self.output_tokens}_c{self.concurrency}"

    @property
    def scenario_family(self) -> str:
        return f"i{self.input_tokens}_o{self.output_tokens}"

    @classmethod
    def from_scenario_id(cls, scenario_id: str) -> "GridCell":
        """Parse ``i<in>_o<out>_c<conc>`` back to a GridCell (used by rewindow_from_raw
        to reconstruct the cell that a raw file belongs to)."""
        try:
            i_part, o_part, c_part = scenario_id.split("_")
            return cls(int(i_part[1:]), int(o_part[1:]), int(c_part[1:]))
        except (ValueError, IndexError) as exc:  # noqa: BLE001
            raise ValueError(f"not a grid scenario_id: {scenario_id!r}") from exc


def enumerate_cells(
    input_buckets: Iterable[int],
    output_buckets: Iterable[int],
    concurrency_levels: Iterable[int],
) -> list[GridCell]:
    # Materialize first: callers may pass generators, and reusing a generator in a
    # nested loop silently exhausts it after the first outer iteration (truncated grid).
    inputs = [int(i) for i in input_buckets]
    outputs = [int(o) for o in output_buckets]
    concs = [int(c) for c in concurrency_levels]
    cells: list[GridCell] = []
    for i in inputs:
        for o in outputs:
            for c in concs:
                cells.append(GridCell(i, o, c))
    return cells


def compute_window_results(windows: list, spec) -> list:
    """EMA'd TRS series over a sequence of windows, using the SAME shared time-constant
    EMA the live controller uses (ema_tau_ms + window_end_ms deltas, ADR-0011), so theta
    is fit on the signal the controller actually sees (S1.4). One TRSComputer across the
    cell's windows (state persists), mirroring the live shared per-model computer.

    Shared by the online path (main) and the offline `rewindow_from_raw` path, so both
    produce a byte-identical trs column from the same ModelWindowMetrics sequence.
    """
    from tre_controller.signals.trs import TRSComputer, TRSInput

    computer = TRSComputer(ema_alpha=spec.trs.ema_alpha, ema_tau_ms=spec.trs.ema_tau_ms)
    results = []
    for wm in windows:
        inp = TRSInput.from_metrics(wm, spec.trs)
        results.append(computer.compute(inp, theta_m=spec.trs.theta_m, window_end_ms=wm.window_end_ms))
    return results


def window_row(cell: GridCell, window_metrics, trs: float, queue_control: float) -> dict:
    """Assemble one calibration CSV row from an aggregated window + its trs.
    Pure: window_metrics is a ModelWindowMetrics-like object.

    Includes the queue observables S2 (grid_search w_p/lambda/qmin) and S3 (qsat fit)
    need — avg_waiting/running/swapping + queue_control (Q_ctl) + p95_e2e — so those
    refits are recoverable from the R3 CSV without a re-run (B2).
    """
    return {
        "scenario_id": cell.scenario_id,
        "scenario_family": cell.scenario_family,
        "input_tokens": cell.input_tokens,
        "output_tokens": cell.output_tokens,
        "concurrency": cell.concurrency,
        "window_start_ms": window_metrics.window_start_ms,
        "window_end_ms": window_metrics.window_end_ms,
        "prompt_tokens_total": window_metrics.prompt_tokens,
        "generation_tokens_total": window_metrics.generation_tokens,
        "avg_waiting": window_metrics.avg_waiting,
        "avg_running": window_metrics.avg_running,
        "avg_swapping": window_metrics.avg_swapping,
        "queue_control": queue_control,
        "p95_ttft": window_metrics.ttft_p95_ms,
        "p95_tpot": window_metrics.tpot_p95_ms,
        "p95_e2e": window_metrics.e2e_p95_ms,
        "trs": trs,
        # Requests that went unserved inside this window, and the resulting verdict. A
        # failed request contributes no latency sample, so a window whose slowest work
        # all errored out otherwise shows a comfortable p95 and is scored as healthy.
        # model_errors is the ENGINE failing; proxy_transient_errors is a connection
        # under the request dying, which is not evidence about the engine but is still a
        # request nobody served. They are separate columns so the second can never be
        # read as an engine fault. All three default to "none seen";
        # openloop.mark_unserved_request_windows fills them in.
        "model_errors": 0,
        "proxy_transient_errors": 0,
        "slo_violated": False,
    }


#: What a voided cell's raw capture is renamed to. It falls outside
#: ``rewindow_from_raw``'s ``*.jsonl`` glob, which is the point: a voided cell must not
#: reach a fit, and leaving the file in place is how it would.
VOID_RAW_SUFFIX = ".void"


CSV_COLUMNS = [
    "scenario_id", "scenario_family", "input_tokens", "output_tokens", "concurrency",
    "window_start_ms", "window_end_ms", "prompt_tokens_total", "generation_tokens_total",
    "avg_waiting", "avg_running", "avg_swapping", "queue_control",
    "p95_ttft", "p95_tpot", "p95_e2e", "trs",
    "model_errors", "proxy_transient_errors", "slo_violated",
]

# S4 per-request raw JSONL schema (doc15 §4). Queue observables are NOT here (they are an
# instant sample, recorded separately in the .instant.jsonl sidecar).
RAW_COLUMNS = [
    "send_ts_ms", "recv_first_token_ts_ms", "done_ts_ms",
    "input_tokens", "output_tokens", "ttft_ms", "tpot_ms", "e2e_ms",
    "http_status", "cell_id",
    # Pod that served the request, when the serving path names one. Per-pod attribution
    # has to be captured here or not at all: nothing downstream can reconstruct it.
    "target_pod",
]

# S4 disk estimate: each per-request line is ~200 bytes of JSON. Warn if a full run is
# projected to exceed this many bytes on the (local, not NFS) raw disk.
RAW_BYTES_PER_REQUEST = 200
DEFAULT_DISK_WARN_BYTES = 2 * 1024**3  # 2 GiB


def write_csv(rows: list[dict], path: Path) -> None:
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def load_existing_rows(path: Path, keep_scenarios: set) -> list[dict]:
    """Reload already-written window rows for the checkpoint-completed cells so a resumed
    run APPENDS to (rather than truncates) prior output.

    The bug this fixes: ``main`` rewrites the whole CSV (``write_csv`` opens "w") after
    every cell from the in-memory ``rows`` list. On resume, done cells are ``continue``d
    and never re-added to ``rows``, so the first ``write_csv`` after a resume truncated the
    file down to only the newly-driven cells and silently dropped every previously-captured
    window row. Seeding ``rows`` from the on-disk CSV (filtered to the checkpoint's done
    set, so a not-yet-done cell can never be duplicated) preserves them.

    Rows are filtered to ``keep_scenarios`` (the checkpoint done set): only cells the
    checkpoint says are complete are trusted from disk; anything else is re-driven and would
    otherwise duplicate. An empty ``keep_scenarios`` (fresh run) reloads nothing, preserving
    the original truncate-and-restart behaviour."""
    if not path.exists() or not keep_scenarios:
        return []
    with path.open("r", newline="") as fh:
        reader = csv.DictReader(fh)
        return [row for row in reader if row.get("scenario_id") in keep_scenarios]


@dataclass
class Checkpoint:
    path: Path
    done: set = field(default_factory=set)

    @classmethod
    def load(cls, path: Path) -> "Checkpoint":
        done: set = set()
        if path.exists():
            done = set(json.loads(path.read_text()).get("done", []))
        return cls(path=path, done=done)

    def mark(self, cell: GridCell) -> None:
        self.done.add(cell.scenario_id)
        self.path.write_text(json.dumps({"done": sorted(self.done)}))

    def is_done(self, cell: GridCell) -> bool:
        return cell.scenario_id in self.done


#: Mirror of ``tre_replayer.engine.prompts.DEFAULT_MODE`` and ``MODES``, repeated here so
#: importing this module never requires the replayer package (guarded by a test).
PROMPT_MODE_DEFAULT = "natural"
PROMPT_MODES = ("token_ids", "text", "natural")


def _make_prompt(
    input_tokens: int, seed_key: str, mode: str = PROMPT_MODE_DEFAULT, model: str | None = None
):
    """One request's prompt: ``input_tokens`` long and unique to ``seed_key``.

    The grid used to send one constant prompt for a whole cell. On an engine with
    prefix caching enabled that serves every request after the first from cache, so
    prefill costs nothing and the measured capacity *rises* with prompt length - the
    calibration built on it is then meaningless. Uniqueness and the seed policy live in
    :mod:`tre_replayer.engine.prompts`; the import is lazy for the same reason as
    :func:`_default_stream_call`.
    """
    from tre_replayer.engine.prompts import build_prompt

    return build_prompt(input_tokens, seed_key, mode=mode, model=model)


def build_raw_record(cell_id: str, send_ts_ms: int, res) -> dict:
    """Map one streamed completion (a StreamResult-like object with .status,
    .first_token_ms, .done_ms, .prompt_tokens, .completion_tokens) to the S4 raw schema.

    Pure and network-free so it is unit-testable. Absolute epoch timestamps are derived
    from ``send_ts_ms`` + the seam's request-relative durations. tpot is the mean
    inter-token latency ((e2e-ttft)/(completion_tokens-1)); anything unavailable is null,
    never fabricated (doc15 §4). input/output tokens are the vLLM usage counts.
    """
    ttft_ms = res.first_token_ms
    e2e_ms = res.done_ms
    completion = res.completion_tokens
    recv_first = None if ttft_ms is None else int(send_ts_ms + ttft_ms)
    done_ts = None if e2e_ms is None else int(send_ts_ms + e2e_ms)
    tpot_ms: Optional[float] = None
    if ttft_ms is not None and e2e_ms is not None and completion is not None and completion > 1:
        tpot_ms = (e2e_ms - ttft_ms) / (completion - 1)
    return {
        "send_ts_ms": int(send_ts_ms),
        "recv_first_token_ts_ms": recv_first,
        "done_ts_ms": done_ts,
        "input_tokens": res.prompt_tokens,
        "output_tokens": res.completion_tokens,
        "ttft_ms": ttft_ms,
        "tpot_ms": tpot_ms,
        "e2e_ms": e2e_ms,
        "http_status": res.status,
        "cell_id": cell_id,
        "target_pod": getattr(res, "target_pod", None),
    }


def _request_headers(model: str, routing_strategy: Optional[str] = None) -> dict:
    """Same rule as the replayer's sender; lazy import for the same reason."""
    from tre_replayer.engine.http_sender import build_request_headers

    return build_request_headers(model, routing_strategy)


def _default_stream_call():
    """Lazy import of the replayer's streaming SSE/usage seam so importing this module
    (e.g. in tests) never requires the replayer package or the network."""
    from tre_replayer.engine.http_sender import _default_stream_call as seam

    return seam


def _append_jsonl(path: Path, records: list[dict]) -> None:
    with path.open("a", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(record, separators=(",", ":")) + "\n")


def drive_cell(
    gateway_url: str,
    model: str,
    cell: GridCell,
    duration_s: float,
    *,
    raw_path: Optional[Path] = None,
    instant_path: Optional[Path] = None,
    instant_sampler: Optional[Callable[[int], dict]] = None,
    instant_interval_s: float = 5.0,
    stream_call: Optional[Callable] = None,
    prompt_mode: str = PROMPT_MODE_DEFAULT,
    routing_strategy: Optional[str] = None,
    run_key: str = "r3",
    now_ms: Callable[[], int] = lambda: int(time.time() * 1000),
) -> tuple[int, int]:
    """Drive cell.concurrency workers against the model for duration_s.
    Returns (start_ms, end_ms). Fixed output length via max_tokens + ignore_eos.

    Every request gets its own prompt, keyed by ``run_key``/cell/sequence (see
    :func:`_make_prompt`); the sequence counter is shared by the workers, so the *set*
    of prompts a cell sends is reproducible even though which worker sends which is not.

    S4: when ``raw_path`` is given, each request is streamed (via ``stream_call``, default
    the replayer SSE seam) and its per-request record appended to that JSONL. When
    ``instant_path`` + ``instant_sampler`` are given, an instant queue snapshot is sampled
    every ``instant_interval_s`` into that sidecar. Both writes go to local disk.
    """
    stop = threading.Event()
    call = stream_call or _default_stream_call()
    records: list[dict] = []
    instants: list[dict] = []
    lock = threading.Lock()
    cell_id = cell.scenario_id
    timeout = max(30.0, cell.output_tokens / 4.0)
    headers = _request_headers(model, routing_strategy)
    # next() on an itertools.count is atomic under CPython, so the workers can share one
    # sequence without a lock; each value is used by exactly one request.
    sequence = itertools.count()

    def request_body(seq: int) -> bytes:
        prompt = _make_prompt(cell.input_tokens, f"{run_key}|{cell_id}|{seq}", prompt_mode, model)
        return json.dumps({
            "model": model, "prompt": prompt, "max_tokens": cell.output_tokens,
            "temperature": 0, "ignore_eos": True,
            "stream": True, "stream_options": {"include_usage": True},
        }).encode()

    def worker() -> None:
        while not stop.is_set():
            body = request_body(next(sequence))
            send_ts = now_ms()
            try:
                res = call(gateway_url, headers, body, timeout)
            except Exception:  # noqa: BLE001 - a failed send must not kill the worker
                continue
            if raw_path is not None:
                with lock:
                    records.append(build_raw_record(cell_id, send_ts, res))

    def sampler() -> None:
        while not stop.is_set():
            try:
                snap = instant_sampler(now_ms())  # type: ignore[misc]
            except Exception:  # noqa: BLE001
                snap = None
            if snap is not None:
                with lock:
                    instants.append({
                        "ts_ms": int(now_ms()),
                        "waiting": snap.get("waiting", 0.0),
                        "running": snap.get("running", 0.0),
                        "swapping": snap.get("swapping", 0.0),
                    })
            stop.wait(instant_interval_s)

    start_ms = now_ms()
    threads = [threading.Thread(target=worker, daemon=True) for _ in range(cell.concurrency)]
    if instant_path is not None and instant_sampler is not None:
        threads.append(threading.Thread(target=sampler, daemon=True))
    for t in threads:
        t.start()
    time.sleep(duration_s)
    stop.set()
    for t in threads:
        t.join(timeout=5)
    end_ms = now_ms()

    if raw_path is not None:
        _append_jsonl(raw_path, records)
    if instant_path is not None and instants:
        _append_jsonl(instant_path, instants)
    return start_ms, end_ms


def estimate_capture_bytes(cells: list[GridCell], cell_seconds: float, assumed_rps_per_worker: float) -> int:
    """Projected raw-log size for the whole grid: sum over cells of
    concurrency * cell_seconds * rps_per_worker requests, ~RAW_BYTES_PER_REQUEST each."""
    total_requests = sum(cell.concurrency * cell_seconds * assumed_rps_per_worker for cell in cells)
    return int(total_requests * RAW_BYTES_PER_REQUEST)


def _make_live_instant_sampler(store, model: str, lookback_ms: int) -> Callable[[int], dict]:
    """Instant queue snapshot from the live MetricsStore, reusing its instant read seam.

    Returns the LATEST scrape value (via store.read_latest_instant), not a windowed
    average. The gateway writes the instant buckets on a boundary-aligned ~10s ticker
    (SCRAPE_INTERVAL_MS), so a naive [now-5000, now] read misses the current bucket and
    records 0 (r3 SMOKE_FINDINGS defect 1). ``lookback_ms`` must be >= ~2x the scrape
    cadence so the last-written bucket is always captured."""

    def sample(now: int) -> dict:
        return store.read_latest_instant(model, now, lookback_ms)

    return sample


def discover_pod_metrics_endpoints(model: str, namespace: str, port: int) -> list[str]:
    """/metrics URLs of the model's routable pods, via kubectl.

    Only routable pods are scraped: a sleeping/hidden resident on the same card is not
    serving this load and its (zero) gauges would dilute the queue average, which is the
    one observable the open-loop primitives exist to measure.
    """
    import subprocess

    out = subprocess.run(
        [
            "kubectl", "-n", namespace, "get", "pods",
            "-l", f"model.aibrix.ai/name={model},tre.aibrix.io/routable=true",
            "-o", "jsonpath={range .items[*]}{.status.podIP}{\"\\n\"}{end}",
        ],
        capture_output=True, text=True, check=True,
    ).stdout
    ips = [line.strip() for line in out.splitlines() if line.strip()]
    if not ips:
        raise RuntimeError(
            f"no routable pods found for model {model!r} in namespace {namespace!r}; "
            "the sidecar would have nothing to sample"
        )
    return [f"http://{ip}:{port}/metrics" for ip in ips]


def drain_start_from_index(schedule_path: Path, model: str) -> Optional[float]:
    """The offset of a schedule's drain segment, read from its generated INDEX.json.

    A cell truncated by a gateway shed jumps to that offset instead of stopping dead, so
    the recovery tail is still captured. The lookup is best effort: a schedule run from
    outside a generated set simply has no drain segment to jump to, and truncation then
    means "stop sending", which is still correct - just less informative.
    """
    index_path = Path(schedule_path).resolve().parent.parent / "INDEX.json"
    if not index_path.exists():
        return None
    try:
        index = json.loads(index_path.read_text(encoding="utf-8"))
    except (ValueError, OSError):
        return None
    wanted = f"{model}/{Path(schedule_path).name}"
    for entry in index.get("schedules", []):
        if entry.get("path") == wanted or Path(str(entry.get("path", ""))).name == Path(schedule_path).name:
            value = entry.get("drain_start_s")
            return None if value is None else float(value)
    return None


def count_slo_windows(rows: Sequence[dict], *, ttft_slo_ms: float, tpot_slo_ms: float) -> int:
    """Windows whose p95 latency is above the SLO.

    This is the evidence a cell exists to produce: theta is a threshold on the signal at
    the moment the model stops meeting its SLO, so a cell that never crossed measured
    nothing about it. A window with no p95 at all (too few samples) is not a crossing.
    """
    crossed = 0
    for row in rows:
        if row.get("slo_violated"):
            # Marked by openloop.mark_unserved_request_windows: a request in this
            # window went unserved, which is a violation even when the p95 of the
            # requests that did survive looks fine.
            crossed += 1
            continue
        ttft = row.get("p95_ttft")
        tpot = row.get("p95_tpot")
        if ttft is not None and float(ttft) > ttft_slo_ms:
            crossed += 1
        elif tpot is not None and float(tpot) > tpot_slo_ms:
            crossed += 1
    return crossed


def censor_after(rows: Sequence[dict], truncated_at_ts_ms: Optional[int]) -> tuple[list, int]:
    """Drop the windows that start at or after a truncation. Returns (kept, dropped).

    After a gateway shed the offered load is no longer what the schedule says: the driver
    has stopped sending and the engine is draining. Those windows describe the recovery,
    not the operating point, so they must not reach the fit.
    """
    if truncated_at_ts_ms is None:
        return list(rows), 0
    kept = [row for row in rows if int(row["window_start_ms"]) < int(truncated_at_ts_ms)]
    return kept, len(rows) - len(kept)


def run_schedule_cell(args, store, spec) -> tuple[list, "openloop.CellGuard"]:
    """Drive one open-loop cell from --schedule and return (window rows, guard)."""
    from tre_replayer.traces.loader import load_trace_segments

    segments = [
        seg for seg in load_trace_segments(args.schedule) if seg.model == args.model
    ]
    if not segments:
        raise SystemExit(
            f"schedule {args.schedule} has no segments for model {args.model!r}"
        )
    cell_id = args.cell_id or _cell_id_from_schedule(Path(args.schedule), args.model, segments)
    cell = GridCell.from_scenario_id(cell_id)  # fail now, not in rewindow_from_raw

    if args.instant_source == "pod":
        endpoints = args.pod_endpoint or discover_pod_metrics_endpoints(
            args.model, args.namespace, args.pod_metrics_port
        )
        sampler = openloop.make_pod_metrics_sampler(endpoints)
        print(f"sidecar: {len(endpoints)} pod /metrics endpoint(s) @ {args.instant_sample_ms}ms")
    else:
        sampler = _make_live_instant_sampler(store, args.model, 2 * SCRAPE_INTERVAL_MS)

    raw_dir = None if args.no_raw else Path(args.raw_dir) / Path(args.output).stem
    if raw_dir is not None:
        raw_dir.mkdir(parents=True, exist_ok=True)
    raw_path = raw_dir / f"{cell_id}.jsonl" if raw_dir is not None else None
    instant_path = raw_dir / f"{cell_id}.instant.jsonl" if raw_dir is not None else None
    failures_path = raw_dir / f"{cell_id}.failures.jsonl" if raw_dir is not None else None
    rps_path = raw_dir / f"{cell_id}.rps.csv" if raw_dir is not None else None

    # Where this run's prompts are materialised. Never the committed schedule tree: the
    # schedules are a few kB of segments and the prompt text of one cell is tens of MB.
    # Defaults to the raw capture directory, which is already this run's own output
    # directory; the campaign points it at <out-dir>/prompts instead.
    prompt_dir = Path(args.prompt_dir) if args.prompt_dir else raw_dir
    if prompt_dir is None:
        print(
            "WARNING: no --prompt-dir and raw logging is off, so prompts will be built "
            "on the send path; the cell's on-wire delay will include a tokenizer fit per "
            "request"
        )
    else:
        prompt_dir.mkdir(parents=True, exist_ok=True)

    drain_start_s = args.drain_start_s
    if drain_start_s is None:
        drain_start_s = drain_start_from_index(Path(args.schedule), args.model)
    if args.truncate_on_proxy_shed:
        where = "stop sending" if drain_start_s is None else f"jump to {drain_start_s:.1f}s (drain)"
        print(f"truncation armed: first gateway shed -> {where}")

    ttft_slo_ms = args.ttft_slo_ms if args.ttft_slo_ms is not None else spec.slo.ttft_p95_ms
    tpot_slo_ms = args.tpot_slo_ms if args.tpot_slo_ms is not None else spec.slo.tpot_p95_ms

    sentinel = None
    if args.envoy_stats_url:
        # Validity only. A non-zero delta says the capture was shaped by the proxy; it
        # never feeds a controller or a fit.
        sentinel = openloop.PendingOverflowSentinel(
            read=openloop.make_envoy_stats_reader(args.envoy_stats_url),
            cluster_filter=args.envoy_cluster_filter or "",
        )

    sender_records: list[dict] = []
    start_ms, end_ms, guard = openloop.drive_cell_schedule(
        args.gateway_url, args.model, cell_id, segments,
        records_out=sender_records,
        seed=args.schedule_seed,
        raw_path=raw_path, instant_path=instant_path,
        instant_sampler=sampler,
        instant_interval_s=args.instant_sample_ms / 1000.0,
        prompt_mode=args.prompt_mode,
        prompt_dir=prompt_dir,
        prompt_workers=args.prompt_workers,
        rps_timeline_path=rps_path,
        routing_strategy=args.routing_strategy,
        max_in_flight=args.max_in_flight,
        truncate_on_proxy_shed=args.truncate_on_proxy_shed,
        drain_start_s=drain_start_s,
        failures_path=failures_path,
        overflow_sentinel=sentinel,
        guard_kwargs={
            "max_p99_delay_ms": args.max_p99_delay_ms,
            "max_p99_pool_wait_ms": args.max_p99_pool_wait_ms,
            "max_model_error_rate": args.max_model_error_rate,
            "max_proxy_transient_rate": args.max_proxy_transient_rate,
            "proxy_transient_allowance": args.proxy_transient_allowance,
            "min_slo_windows": args.min_slo_windows,
            "max_routing_imbalance": args.max_routing_imbalance,
            "shed_policy": args.shed_policy,
            "ttft_slo_ms": ttft_slo_ms,
            "tpot_slo_ms": tpot_slo_ms,
        },
    )

    windows = []
    w = start_ms
    while w + args.window_ms <= end_ms:
        windows.append(store.read_model_window(args.model, w, w + args.window_ms))
        w += args.window_ms
    results = compute_window_results(windows, spec)
    rows = [
        # An undefined TSS (idle rule, tre_common.tss) is written blank, never as 0.
        window_row(cell, wm, result.TRS if result.defined else None, result.Q_ctl)
        for wm, result in zip(windows, results)
    ]
    # A window holding a model error is a violation and is KEPT. Dropping it would remove
    # exactly the overloaded windows and pull theta towards health.
    rows = openloop.mark_unserved_request_windows(rows, sender_records)
    if guard.shed_policy == openloop.SHED_POLICY_VOID and guard.voided:
        # Nothing from a voided cell may reach the fit - not even the windows taken
        # before the shed, which are precisely the healthy ones.
        censored_windows = len(rows)
        rows = []
    else:
        rows, censored_windows = censor_after(rows, guard.truncated_at_ts_ms)
    guard = guard.with_slo_windows(
        count_slo_windows(rows, ttft_slo_ms=ttft_slo_ms, tpot_slo_ms=tpot_slo_ms)
    )

    artifact = guard.as_dict()
    artifact.update({
        "schedule": str(args.schedule),
        "model": args.model,
        "drain_start_s": drain_start_s,
        "censored_windows": censored_windows,
        "windows": len(rows),
        "ttft_slo_ms": ttft_slo_ms,
        "tpot_slo_ms": tpot_slo_ms,
        "start_ms": start_ms,
        "end_ms": end_ms,
        "instant_sample_ms": args.instant_sample_ms,
        # How the load was actually generated and routed. Recorded per cell because a
        # capacity number is only comparable to another one made the same way.
        "prompt_mode": args.prompt_mode,
        "routing_strategy": args.routing_strategy,
        "prompt_file": (
            None
            if prompt_dir is None
            else str(openloop.prompt_file_path_for(prompt_dir, cell_id))
        ),
        "rps_timeline": None if rps_path is None else str(rps_path),
    })
    if raw_dir is not None:
        (raw_dir / f"{cell_id}.guard.json").write_text(
            json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    print(f"cell {cell_id} guard: {json.dumps(artifact, sort_keys=True)}")
    if guard.unrecognised_proxy_failures:
        # Never silent. A wording no rule matched was counted as transient, which is the
        # forgiving side; if Envoy changed its wording for a real admission rejection,
        # this line is the only place it shows before theta quietly moves.
        print(
            f"WARNING: cell {cell_id} saw {guard.unrecognised_proxy_failures} proxy "
            f"failure(s) whose wording matched no rule and were counted as transient: "
            f"{json.dumps(guard.proxy_failure_reasons or {}, sort_keys=True)}"
        )
    if guard.sentinel_contradiction:
        print(f"WARNING: cell {cell_id} sentinel disagreement: {guard.sentinel_contradiction}")
    if guard.voided:
        print(
            f"cell {cell_id} is VOID ({', '.join(guard.void_reasons)}): "
            f"{censored_windows} window(s) discarded, nothing from this cell may be fitted; "
            "re-run it"
        )
        # ... and nothing from it may reach the fit through the back door either. The
        # fitting re-window globs every <cell>.jsonl under the raw root and filters only
        # by cell id, so a voided capture left in place is silently re-windowed - and a
        # re-run, which writes the same cell id again, would pool both attempts into one
        # fit. Renaming it out of the glob is the whole of the fix; the bytes are kept,
        # under a name that says what they are.
        if raw_path is not None and Path(raw_path).exists():
            quarantined = Path(str(raw_path) + VOID_RAW_SUFFIX)
            Path(raw_path).replace(quarantined)
            print(f"cell {cell_id} raw capture quarantined -> {quarantined}")
    if guard.truncated:
        print(
            f"cell {cell_id} was TRUNCATED by an admission overflow at offset "
            f"{guard.truncated_at_offset_s}s: {guard.censored} request(s) censored, "
            f"{censored_windows} window(s) dropped, {guard.slo_windows} window(s) above "
            f"the SLO kept"
        )
    if args.guard_mode == "fail":
        openloop.raise_on_guard(guard)
    elif not guard.ok:
        print(f"WARNING: cell {cell_id} guard failed (continuing on --guard-mode warn)")
    return rows, guard


def _cell_id_from_schedule(path: Path, model: str, segments: list) -> str:
    """Default scenario id for a schedule file: ``i<in>_o<out>_c<load-code>``.

    The load code is 100x the primitive's characteristic rho, as written by
    gen_calibration_schedules.py, and is recovered here from the file name so a schedule
    run from the committed tree needs no extra flag. A mixture schedule (several token
    shapes) records i0_o0, which r3_capacity then correctly declines to fit.
    """
    from scripts.gen_calibration_schedules import HOLD_PRIMITIVE, LOAD_CODE

    stem = path.stem  # <shape>_<primitive>, or <shape>_hold<load-code>
    primitive = stem.rsplit("_", 1)[-1]
    if primitive in LOAD_CODE:
        code = LOAD_CODE[primitive]
    elif primitive.startswith(HOLD_PRIMITIVE) and primitive[len(HOLD_PRIMITIVE):].isdigit():
        # A boundary-search hold cell carries its own rho in the file name, because its
        # rho is decided at campaign time and there is no fixed code to look up.
        code = int(primitive[len(HOLD_PRIMITIVE):])
    else:
        raise SystemExit(
            f"cannot derive a cell id from {path.name!r}; pass --cell-id explicitly"
        )
    shapes = {(s.input_tokens, s.max_output_tokens) for s in segments}
    if len(shapes) == 1:
        i, o = next(iter(shapes))
    else:
        i, o = 0, 0
    return f"i{i or 0}_o{o or 0}_c{code}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--gateway-url", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--input-buckets", default="128,512,1024")
    ap.add_argument("--output-buckets", default="128,512")
    ap.add_argument("--concurrency", default="1,2,4,8,16")
    ap.add_argument("--cell-seconds", type=float, default=60.0)
    # MUST equal the frozen control window W (S1.2; provisional 30000). theta fit on a
    # different window is invalid (S1.4 hard gate).
    ap.add_argument("--window-ms", type=int, default=30000)
    ap.add_argument("--redis-url", default="redis://tre-v2-redis:6379/0")
    ap.add_argument("--metrics-schema", default="v1")
    # Sidecar sampling cadence (how often WE sample the queue into the .instant.jsonl).
    # This is NOT the divisor MetricsStore uses: the redis buckets it reads are written by
    # the gateway on its own 10 s ticker, so that divisor is --store-instant-sample-ms and
    # must stay at SCRAPE_INTERVAL_MS. An offline rewindow_from_raw of this capture must be
    # passed --instant-sample-ms equal to the value used here.
    # Closed-loop default stays the gateway cadence. Schedule mode defaults to 1 s: the
    # gateway grid gives 3 samples per 30 s window, which aliases away the short-lived
    # waiting queue the bursts primitive exists to produce (see scripts/openloop.py).
    ap.add_argument("--instant-sample-ms", type=int, default=None)
    # expected_samples divisor for the redis windows: the gateway's cadence, not ours.
    ap.add_argument("--store-instant-sample-ms", type=int, default=SCRAPE_INTERVAL_MS)
    ap.add_argument("--percentile-mode", default="bucket_upper")
    ap.add_argument("--min-latency-samples", type=int, default=10)  # align with live TRE_MIN_LATENCY_SAMPLES
    ap.add_argument("--registry", default=None)
    ap.add_argument("--namespace", default="default", help="namespace holding the model pods")
    ap.add_argument("--only-first-cell", action="store_true")
    # S4: raw per-request log lands on local disk (NOT NFS: doc15 §4.3). Default is 76's
    # local experiments dir; a subdir per output stem keeps concurrent runs separate.
    ap.add_argument("--raw-dir", default="/root/tre-experiments/r3_raw")
    ap.add_argument("--no-raw", action="store_true", help="disable S4 raw logging")
    ap.add_argument("--disk-warn-gib", type=float, default=DEFAULT_DISK_WARN_BYTES / 1024**3)
    ap.add_argument("--assumed-rps-per-worker", type=float, default=2.0,
                    help="only used for the pre-run raw disk estimate")
    # Prompt synthesis. token_ids sends an explicit token-id list, so the realised
    # prompt length is exact; text is the fallback for an endpoint that only accepts a
    # string. Either way each request gets its own prompt (no prefix-cache freebies).
    ap.add_argument("--prompt-dir", default=None,
                    help="directory this run materialises its prompts into, before the "
                         "cell starts (default: alongside the raw capture). Never the "
                         "committed schedule tree")
    ap.add_argument("--prompt-workers", type=int, default=None,
                    help="processes used to pre-build prompts (default: one per core, "
                         "capped); the tokenizer holds the GIL, so threads do not help")
    ap.add_argument("--prompt-mode", default=PROMPT_MODE_DEFAULT, choices=list(PROMPT_MODES),
                    help="natural: English prose cut to the exact token count with the "
                         "model's own tokenizer (default). token_ids: uniformly random "
                         "ids - exact, but not language. text: nominal length only.")
    ap.add_argument("--routing-strategy", default=None,
                    help="Route via the AIBrix gateway plugin with this strategy (e.g. "
                         "least-request) instead of the per-model HTTPRoute. This is the "
                         "only way the answers name a serving pod, so it is what makes "
                         "the per-pod routing-balance check in the guard artifact "
                         "non-empty - but it also changes who picks the pod, so runs made "
                         "with and without it are not comparable.")
    ap.add_argument("--max-routing-imbalance", type=float,
                    default=openloop.DEFAULT_MAX_ROUTING_IMBALANCE,
                    help="Fail a cell whose busiest pod served more than this multiple of "
                         "its quietest pod's requests. Unset by default: the balance is "
                         "reported in the guard artifact but never gates a cell.")
    # Seeds prompt content. Defaults to the output stem so two runs writing different
    # CSVs differ, and re-running the same output reproduces the same prompts.
    ap.add_argument("--run-key", default=None)
    # ---- open-loop (schedule-driven) mode ----
    ap.add_argument("--schedule", default=None,
                    help="replayer trace file; switches this cell to the open-loop driver")
    ap.add_argument("--cell-id", default=None,
                    help="scenario id for the schedule cell (default: from the schedule INDEX "
                         "convention i<in>_o<out>_c<load-code>); must parse as a GridCell or "
                         "rewindow_from_raw will skip the raw file")
    ap.add_argument("--schedule-seed", type=int, default=1234)
    ap.add_argument("--max-in-flight", type=int, default=openloop.DEFAULT_MAX_IN_FLIGHT)
    # Sidecar source. "pod" scrapes the model pods' /metrics directly at
    # --instant-sample-ms (1 s for the campaign); "store" is the legacy redis read, which
    # cannot resolve faster than the gateway's 10 s grid.
    ap.add_argument("--instant-source", default="pod", choices=["pod", "store"])
    ap.add_argument("--pod-metrics-port", type=int, default=8000)
    ap.add_argument("--pod-endpoint", action="append", default=[],
                    help="explicit http://ip:port/metrics endpoint; repeatable. Default: "
                         "discovered from the routable pods of --model")
    ap.add_argument("--max-p99-delay-ms", type=float, default=openloop.DEFAULT_MAX_P99_DELAY_MS)
    ap.add_argument("--max-p99-pool-wait-ms", type=float,
                    default=openloop.DEFAULT_MAX_P99_POOL_WAIT_MS)
    # Only MODEL errors count against this budget. An admission overflow is the
    # campaign hitting the admission ceiling, not the engine failing, and it truncates or
    # voids the cell instead.
    ap.add_argument("--max-model-error-rate", type=float,
                    default=openloop.DEFAULT_MAX_MODEL_ERROR_RATE)
    # Transient proxy errors - a connection carrying the request died - get their own,
    # tighter budget. They are not an admission decision, so one of them must not void a
    # cell; a path that keeps dropping connections still must.
    ap.add_argument("--max-proxy-transient-rate", type=float,
                    default=openloop.DEFAULT_MAX_PROXY_TRANSIENT_RATE,
                    help="a cell whose TRANSIENT proxy error rate exceeds this is void "
                         "and must be re-run; the individual windows are marked as "
                         "violations and kept either way")
    ap.add_argument("--proxy-transient-allowance", type=int,
                    default=openloop.DEFAULT_PROXY_TRANSIENT_ALLOWANCE,
                    help="transient proxy errors a cell is never voided on, whatever its "
                         "size: one dropped connection is not a rate")
    ap.add_argument("--truncate-on-proxy-shed", action="store_true", default=True,
                    help="on the first admission overflow, jump to the schedule drain "
                         "segment and censor the windows after it (default: on). A "
                         "transient proxy error never truncates; under --shed-policy "
                         "void the drain is skipped because the cell is void anyway")
    ap.add_argument("--no-truncate-on-proxy-shed", action="store_false",
                    dest="truncate_on_proxy_shed",
                    help="keep offering load after a gateway shed (the cell then measures "
                         "the circuit breaker, not the engine)")
    ap.add_argument("--drain-start-s", type=float, default=None,
                    help="offset a truncated cell jumps to (default: the schedule's "
                         "drain_start_s from its generated INDEX.json)")
    ap.add_argument("--shed-policy", default=openloop.DEFAULT_SHED_POLICY,
                    choices=list(openloop.SHED_POLICIES),
                    help="what a gateway shed does to the cell. truncate: keep the "
                         "windows from before it (replay default). void: discard the "
                         "whole cell - the only correct choice for a calibration cell, "
                         "because the windows before a shed are exactly the healthy ones "
                         "and keeping them biases theta towards health.")
    ap.add_argument("--envoy-stats-url", default=None,
                    help="Envoy stats endpoint - in this cluster "
                         "http://<envoy-pod-ip>:19001/stats/prometheus, because the admin "
                         "listener on :19000 is bound inside the container and the envoy "
                         "container has no curl. When set, the cell records the change in "
                         "upstream_rq_pending_overflow across it, is voided if it moved, "
                         "and cross-checks it against the requests the client classified "
                         "as admission overflow. Validity sentinel only: it never enters "
                         "a fit or a control law.")
    ap.add_argument("--envoy-cluster-filter", default=None,
                    help="only count overflow counters whose stat name contains this "
                         "(e.g. the model's cluster name)")
    ap.add_argument("--min-slo-windows", type=int, default=openloop.DEFAULT_MIN_SLO_WINDOWS,
                    help="windows above the SLO a truncated cell must already have "
                         "collected to still pass")
    ap.add_argument("--ttft-slo-ms", type=float, default=None,
                    help="p95 TTFT SLO for the window evidence count (default: registry)")
    ap.add_argument("--tpot-slo-ms", type=float, default=None,
                    help="p95 TPOT SLO for the window evidence count (default: registry)")
    ap.add_argument("--guard-mode", default="fail", choices=["fail", "warn"],
                    help="fail: a cell that did not deliver its load aborts the run")
    args = ap.parse_args()
    if args.schedule is None and args.instant_source == "pod":
        # The closed-loop path historically reads the store; keep that default intact.
        args.instant_source = "store"
    if args.instant_sample_ms is None:
        args.instant_sample_ms = (
            int(openloop.DEFAULT_SIDECAR_INTERVAL_S * 1000)
            if args.schedule is not None
            else SCRAPE_INTERVAL_MS
        )

    cells = enumerate_cells(
        (int(x) for x in args.input_buckets.split(",")),
        (int(x) for x in args.output_buckets.split(",")),
        (int(x) for x in args.concurrency.split(",")),
    )
    if args.only_first_cell:
        cells = cells[:1]
    if args.schedule is not None:
        # The schedule replaces the grid entirely; keep one nominal cell only so the raw
        # disk estimate below has something to size against.
        cells = cells[:1]

    out = Path(args.output)
    run_key = args.run_key or out.stem
    out.parent.mkdir(parents=True, exist_ok=True)
    ckpt = Checkpoint.load(out.with_suffix(".checkpoint.json"))

    raw_dir: Optional[Path] = None
    if not args.no_raw:
        raw_dir = Path(args.raw_dir) / out.stem
        raw_dir.mkdir(parents=True, exist_ok=True)
        projected = estimate_capture_bytes(cells, args.cell_seconds, args.assumed_rps_per_worker)
        warn_bytes = int(args.disk_warn_gib * 1024**3)
        note = "  !! EXCEEDS WARN THRESHOLD" if projected > warn_bytes else ""
        print(
            f"S4 raw log -> {raw_dir} (local disk); projected ~{projected / 1024**2:.1f} MiB "
            f"for {len(cells)} cells x {args.cell_seconds:.0f}s{note}"
        )

    import redis  # type: ignore[import-not-found]
    from tre_common.registry import load_registry
    from tre_controller.store.metrics_store import MetricsStore

    registry = load_registry(args.registry)
    spec = registry.model(args.model)
    redis_client = redis.Redis.from_url(args.redis_url)
    store = MetricsStore(
        redis_client, registry,
        instant_sample_interval_ms=args.store_instant_sample_ms,
        percentile_mode=args.percentile_mode,
        schema=args.metrics_schema,
        min_latency_samples=args.min_latency_samples,  # align p95 with the live N1 guard
    )
    # Freshness lookback = 2x scrape cadence so the last-written 10s bucket is always in
    # range even with scrape/write lag (r3 SMOKE_FINDINGS defect 1); read_latest_instant
    # then takes the freshest bucket, not a lookback-wide average.
    instant_sampler = _make_live_instant_sampler(store, args.model, 2 * SCRAPE_INTERVAL_MS)

    if args.schedule is not None:
        rows, _guard = run_schedule_cell(args, store, spec)
        write_csv(rows, out)
        print(f"wrote {len(rows)} rows to {out}")
        return 0

    # Resume-safe: seed rows from the rows already on disk for checkpoint-done cells, so the
    # per-cell full rewrite below appends instead of truncating away prior captures.
    rows: list[dict] = load_existing_rows(out, ckpt.done)
    for cell in cells:
        if ckpt.is_done(cell):
            continue
        raw_path = raw_dir / f"{cell.scenario_id}.jsonl" if raw_dir is not None else None
        instant_path = raw_dir / f"{cell.scenario_id}.instant.jsonl" if raw_dir is not None else None
        start_ms, end_ms = drive_cell(
            args.gateway_url, args.model, cell, args.cell_seconds,
            raw_path=raw_path, instant_path=instant_path,
            instant_sampler=instant_sampler, instant_interval_s=args.instant_sample_ms / 1000.0,
            prompt_mode=args.prompt_mode, routing_strategy=args.routing_strategy,
            run_key=run_key,
        )
        windows = []
        w = start_ms
        while w + args.window_ms <= end_ms:
            windows.append(store.read_model_window(args.model, w, w + args.window_ms))
            w += args.window_ms
        results = compute_window_results(windows, spec)  # shared time-constant EMA (S1.4)
        for wm, result in zip(windows, results):
            rows.append(window_row(cell, wm, result.TRS if result.defined else None, result.Q_ctl))
        cell_windows = len(windows)
        ckpt.mark(cell)
        write_csv(rows, out)
        print(f"cell {cell.scenario_id}: {cell_windows} windows")
    print(f"wrote {len(rows)} rows to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
