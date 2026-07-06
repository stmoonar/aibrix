#!/usr/bin/env python3
"""R3 load-grid driver (endgame plan 6.2): sweep input x output x concurrency
against a model, emit a window-level CSV consumable by the calibration `fit` CLI
(columns: scenario_id, scenario_family, prompt_tokens_total, generation_tokens_total,
p95_ttft, p95_tpot, trs). Reuses the controller MetricsStore for window aggregation
and TRSComputer for the trs column, so metric parsing is not reimplemented.

Checkpoints per cell (resumable). The full grid is a ~10h/model run (R3, wall-clock);
validate on a single cell (--only-first-cell) before the long run.
"""
from __future__ import annotations

import argparse
import csv
import json
import threading
import time
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable


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

    CAVEAT (authoritative R3): live uses a SLIDING window refreshed every ~5s (the EMA
    advances every ~5s with dt~=5s), whereas this driver re-windows TUMBLING at window_ms
    (dt=window_ms). Same tau, different advance cadence -> a coarser approximation. For the
    authoritative R3 trs column, re-aggregate from the raw per-request log with a SLIDING
    window at the live refresh step (S4 `rewindow_from_raw --window-ms=<W> --step-ms=<refresh>`).
    This tumbling series is kept for the online quick-look CSV.
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


def drive_cell(gateway_url: str, model: str, cell: GridCell, duration_s: float) -> tuple[int, int]:
    """Drive cell.concurrency workers against the model for duration_s.
    Returns (start_ms, end_ms). Fixed output length via max_tokens + ignore_eos.
    """
    stop = threading.Event()
    prompt = _make_prompt(cell.input_tokens)

    def worker() -> None:
        body = json.dumps({
            "model": model, "prompt": prompt, "max_tokens": cell.output_tokens,
            "temperature": 0, "ignore_eos": True,
        }).encode()
        while not stop.is_set():
            req = urllib.request.Request(gateway_url, data=body,
                                         headers={"Content-Type": "application/json", "model": model})
            try:
                urllib.request.urlopen(req, timeout=max(30.0, cell.output_tokens / 5.0)).read()
            except Exception:
                pass

    start_ms = int(time.time() * 1000)
    threads = [threading.Thread(target=worker, daemon=True) for _ in range(cell.concurrency)]
    for t in threads:
        t.start()
    time.sleep(duration_s)
    stop.set()
    for t in threads:
        t.join(timeout=5)
    end_ms = int(time.time() * 1000)
    return start_ms, end_ms


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
    ap.add_argument("--instant-sample-ms", type=int, default=5000)
    ap.add_argument("--percentile-mode", default="bucket_upper")
    ap.add_argument("--min-latency-samples", type=int, default=10)  # align with live TRE_MIN_LATENCY_SAMPLES
    ap.add_argument("--registry", default=None)
    ap.add_argument("--only-first-cell", action="store_true")
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

    rows: list[dict] = []
    for cell in cells:
        if ckpt.is_done(cell):
            continue
        start_ms, end_ms = drive_cell(args.gateway_url, args.model, cell, args.cell_seconds)
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
