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
"""
from __future__ import annotations

import argparse
import csv
import json
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Optional

from tre_common.rediskeys import SCRAPE_INTERVAL_MS


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
    cells: list[GridCell] = []
    for i in input_buckets:
        for o in output_buckets:
            for c in concurrency_levels:
                cells.append(GridCell(int(i), int(o), int(c)))
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
    }


CSV_COLUMNS = [
    "scenario_id", "scenario_family", "input_tokens", "output_tokens", "concurrency",
    "window_start_ms", "window_end_ms", "prompt_tokens_total", "generation_tokens_total",
    "avg_waiting", "avg_running", "avg_swapping", "queue_control",
    "p95_ttft", "p95_tpot", "p95_e2e", "trs",
]

# S4 per-request raw JSONL schema (doc15 §4). Queue observables are NOT here (they are an
# instant sample, recorded separately in the .instant.jsonl sidecar).
RAW_COLUMNS = [
    "send_ts_ms", "recv_first_token_ts_ms", "done_ts_ms",
    "input_tokens", "output_tokens", "ttft_ms", "tpot_ms", "e2e_ms",
    "http_status", "cell_id",
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


def _make_prompt(input_tokens: int) -> str:
    return " ".join(["token"] * max(1, input_tokens))


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
    }


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
    now_ms: Callable[[], int] = lambda: int(time.time() * 1000),
) -> tuple[int, int]:
    """Drive cell.concurrency workers against the model for duration_s.
    Returns (start_ms, end_ms). Fixed output length via max_tokens + ignore_eos.

    S4: when ``raw_path`` is given, each request is streamed (via ``stream_call``, default
    the replayer SSE seam) and its per-request record appended to that JSONL. When
    ``instant_path`` + ``instant_sampler`` are given, an instant queue snapshot is sampled
    every ``instant_interval_s`` into that sidecar. Both writes go to local disk.
    """
    stop = threading.Event()
    prompt = _make_prompt(cell.input_tokens)
    call = stream_call or _default_stream_call()
    records: list[dict] = []
    instants: list[dict] = []
    lock = threading.Lock()
    cell_id = cell.scenario_id
    timeout = max(30.0, cell.output_tokens / 4.0)
    headers = {
        "Content-Type": "application/json",
        "Accept": "text/event-stream",
        "model": model,
    }
    body = json.dumps({
        "model": model, "prompt": prompt, "max_tokens": cell.output_tokens,
        "temperature": 0, "ignore_eos": True,
        "stream": True, "stream_options": {"include_usage": True},
    }).encode()

    def worker() -> None:
        while not stop.is_set():
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
    # Instant sampler cadence + expected_samples divisor. MUST match the gateway scrape
    # cadence (SCRAPE_INTERVAL_MS=10000): a smaller value doubles expected_samples and
    # halves the offline queue average vs. the online path (r3 SMOKE_FINDINGS defect 2).
    ap.add_argument("--instant-sample-ms", type=int, default=SCRAPE_INTERVAL_MS)
    ap.add_argument("--percentile-mode", default="bucket_upper")
    ap.add_argument("--min-latency-samples", type=int, default=10)  # align with live TRE_MIN_LATENCY_SAMPLES
    ap.add_argument("--registry", default=None)
    ap.add_argument("--only-first-cell", action="store_true")
    # S4: raw per-request log lands on local disk (NOT NFS: doc15 §4.3). Default is 76's
    # local experiments dir; a subdir per output stem keeps concurrent runs separate.
    ap.add_argument("--raw-dir", default="/root/tre-experiments/r3_raw")
    ap.add_argument("--no-raw", action="store_true", help="disable S4 raw logging")
    ap.add_argument("--disk-warn-gib", type=float, default=DEFAULT_DISK_WARN_BYTES / 1024**3)
    ap.add_argument("--assumed-rps-per-worker", type=float, default=2.0,
                    help="only used for the pre-run raw disk estimate")
    args = ap.parse_args()

    cells = enumerate_cells(
        (int(x) for x in args.input_buckets.split(",")),
        (int(x) for x in args.output_buckets.split(",")),
        (int(x) for x in args.concurrency.split(",")),
    )
    if args.only_first_cell:
        cells = cells[:1]

    out = Path(args.output)
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
        instant_sample_interval_ms=args.instant_sample_ms,
        percentile_mode=args.percentile_mode,
        schema=args.metrics_schema,
        min_latency_samples=args.min_latency_samples,  # align p95 with the live N1 guard
    )
    # Freshness lookback = 2x scrape cadence so the last-written 10s bucket is always in
    # range even with scrape/write lag (r3 SMOKE_FINDINGS defect 1); read_latest_instant
    # then takes the freshest bucket, not a lookback-wide average.
    instant_sampler = _make_live_instant_sampler(store, args.model, 2 * SCRAPE_INTERVAL_MS)

    rows: list[dict] = []
    for cell in cells:
        if ckpt.is_done(cell):
            continue
        raw_path = raw_dir / f"{cell.scenario_id}.jsonl" if raw_dir is not None else None
        instant_path = raw_dir / f"{cell.scenario_id}.instant.jsonl" if raw_dir is not None else None
        start_ms, end_ms = drive_cell(
            args.gateway_url, args.model, cell, args.cell_seconds,
            raw_path=raw_path, instant_path=instant_path,
            instant_sampler=instant_sampler, instant_interval_s=args.instant_sample_ms / 1000.0,
        )
        windows = []
        w = start_ms
        while w + args.window_ms <= end_ms:
            windows.append(store.read_model_window(args.model, w, w + args.window_ms))
            w += args.window_ms
        results = compute_window_results(windows, spec)  # shared time-constant EMA (S1.4)
        for wm, result in zip(windows, results):
            rows.append(window_row(cell, wm, result.TRS, result.Q_ctl))
        cell_windows = len(windows)
        ckpt.mark(cell)
        write_csv(rows, out)
        print(f"cell {cell.scenario_id}: {cell_windows} windows")
    print(f"wrote {len(rows)} rows to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
