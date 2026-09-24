"""Run a trace against a gateway and score it (audit blocker B3).

Glue: trace.json -> seeded arrival schedule -> open-loop streaming sender -> per-request
JSONL -> per-model V_sys. Use --dry-run (fake sender, no network) to exercise the whole
pipeline in tests/CI. Live runs are driven by the executor, not here.

Note (out of scope, see docstring): generating the real 7 trace.json from R3 capacity surfaces
(gen_traces.py) is deferred until R3 capacity data exists; this driver replays any trace.json.
"""
from __future__ import annotations

import argparse
import asyncio
import json
from typing import Any

from tre_replayer.engine import rps_timeline
from tre_replayer.engine.dispatcher import dispatch_open_loop
from tre_replayer.engine.http_sender import DEFAULT_ROUTING_STRATEGY, StreamResult, StreamingHttpSender
from tre_replayer.engine.prompt_store import materialize_prompts
from tre_replayer.engine.schedule import build_poisson_schedule
from tre_replayer.scoring import compute_v_sys
from tre_replayer.traces.loader import load_trace_segments


def _dry_stream_call(url: str, headers: dict[str, str], body: bytes, timeout_s: float) -> StreamResult:
    out = json.loads(body).get("max_tokens", 128)
    # deterministic synthetic response so a dry run is reproducible and within SLO.
    return StreamResult(status=200, first_token_ms=80.0, done_ms=80.0 + out * 3.0, prompt_tokens=64, completion_tokens=out)


def run_trace(
    trace_path: str,
    *,
    gateway_url: str,
    out_path: str | None = None,
    registry_path: str | None = None,
    seed: int = 0,
    dry_run: bool = False,
    window_ms: int = 30_000,
    step_ms: int = 5_000,
    max_in_flight: int = 512,
    trim_ramp_windows: int = 1,
    sleep: Any = None,
    prompt_path: str | None = None,
    rps_timeline_path: str | None = None,
    prompt_workers: int | None = None,
    routing_strategy: str | None = DEFAULT_ROUTING_STRATEGY,
) -> dict[str, Any]:
    from tre_common.registry import load_registry

    segments = load_trace_segments(trace_path)
    schedule = build_poisson_schedule(segments, seed=seed)
    # Prompts are built here, before the loop, not inside each send: see
    # tre_replayer.engine.prompt_store. Without a path there is nowhere to put them and
    # the sender falls back to fitting inline.
    prompt_store = (
        None
        if prompt_path is None
        else materialize_prompts(schedule, path=prompt_path, processes=prompt_workers)
    )
    sender = StreamingHttpSender(
        gateway_url,
        stream_call=_dry_stream_call if dry_run else None,
        max_in_flight=max_in_flight,
        prompt_store=prompt_store,
        routing_strategy=routing_strategy or None,
    )
    dispatch_kwargs = {"sleep": sleep} if sleep is not None else {}
    try:
        report = asyncio.run(dispatch_open_loop(schedule, sender, **dispatch_kwargs))
    finally:
        sender.close()
    if out_path:
        sender.write_jsonl(out_path)

    registry = load_registry(registry_path)
    slos = {m.name: m.slo for m in registry.models()}
    per_model: dict[str, Any] = {}
    by_model: dict[str, list[dict]] = {}
    trace_timestamps = [
        record.get("actual_send_ts_ms")
        for record in sender.records
        if record.get("actual_send_ts_ms") is not None
    ]
    trace_start_ms = min(trace_timestamps) if trace_timestamps else None
    for rec in sender.records:
        by_model.setdefault(rec["model"], []).append(rec)
    for model, recs in sorted(by_model.items()):
        slo = slos.get(model)
        if slo is None:
            continue
        per_model[model] = compute_v_sys(
            recs,
            ttft_slo_ms=slo.ttft_p95_ms,
            tpot_slo_ms=slo.tpot_p95_ms,
            e2e_slo_ms=slo.e2e_p95_ms,
            window_ms=window_ms,
            step_ms=step_ms,
            trim_ramp_windows=trim_ramp_windows,
            trace_start_ms=trace_start_ms,
        )
    # Achieved against nominal arrivals, per model, from the instants the requests
    # really went on the wire. This is the evidence that the replay applied the intensity
    # the trace describes; schedule_rps_error is its headline.
    timelines = {
        model: rps_timeline.build_rps_timeline(
            [event.scheduled_offset_s for event in schedule if event.model == model],
            _achieved_offsets(recs),
            window_s=rps_timeline.DEFAULT_WINDOW_S,
        )
        for model, recs in by_model.items()
    }
    if rps_timeline_path:
        rps_timeline.write_rps_timeline_csv(rps_timeline_path, timelines)
    rps_error = {
        model: round(
            rps_timeline.max_relative_rps_error(
                rps_timeline.build_rps_timeline(
                    [event.scheduled_offset_s for event in schedule if event.model == model],
                    _achieved_offsets(by_model[model]),
                    window_s=rps_timeline.ERROR_WINDOW_S,
                )
            ),
            4,
        )
        for model in sorted(by_model)
    }
    # Which pod served each request, per model, as the plugin reported it (routed path
    # only; None rows are the Service path or failures that never reached a pod). The
    # quickest check that least-gpu-cache spreads load and never lands on a sleeping pod.
    target_pods: dict[str, dict[str, int]] = {}
    for rec in sender.records:
        pod = rec.get("target_pod")
        if pod:
            by_pod = target_pods.setdefault(rec["model"], {})
            by_pod[pod] = by_pod.get(pod, 0) + 1
    return {
        "trace": trace_path,
        "routing_strategy": routing_strategy or None,
        "target_pods": target_pods,
        "requests": len(sender.records),
        "schedule_p99_delay_ms": round(report.p99_delay_ms, 2),
        "schedule_rps_error": round(report.actual_rps_error_ratio, 4),
        "rps_error_by_model": rps_error,
        "rps_timeline": rps_timeline_path,
        "max_pool_wait_ms": round(sender.max_pool_wait_ms(), 2),  # F5: high -> sender pool starved
        # Scheduled instant -> socket call, the whole of it. Unlike the two above it
        # includes whatever the worker did before the send, which is where an inline
        # prompt fit used to hide.
        "max_on_wire_delay_ms": round(sender.max_on_wire_delay_ms(), 2),
        "prompt_store_misses": sender.prompt_store_misses,
        "trim_ramp_windows": trim_ramp_windows,
        "per_model": per_model,
    }


def _achieved_offsets(records: list[dict]) -> list[float]:
    """On-wire instants in the schedule's own time base (offset + on-wire lateness)."""
    return [
        float(record["scheduled_offset_s"])
        + float(record.get("on_wire_delay_ms", 0.0) or 0.0) / 1000.0
        for record in records
        if record.get("scheduled_offset_s") is not None
    ]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace", required=True)
    # Both arms go through the tre-v2 gateway (NodePort 31094) since 2026-09-24.
    ap.add_argument("--gateway-url", default="http://192.168.223.76:31094/v1/completions")
    ap.add_argument("--out", default=None)
    ap.add_argument("--registry", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--window-ms", type=int, default=30_000)
    ap.add_argument("--step-ms", type=int, default=5_000)
    ap.add_argument("--max-in-flight", type=int, default=512)  # sender thread pool (F5)
    ap.add_argument("--trim-ramp-windows", type=int, default=1)
    ap.add_argument("--prompt-file", default=None,
                    help="materialise every prompt here before the replay starts, so no "
                         "prompt is built on the send path")
    ap.add_argument("--prompt-workers", type=int, default=None)
    ap.add_argument("--rps-timeline", default=None,
                    help="CSV of nominal vs achieved requests per second, per model")
    ap.add_argument("--routing-strategy", default=DEFAULT_ROUTING_STRATEGY,
                    help="routing-strategy request header (default %(default)s, what the v1 "
                         "client sent); '' or 'none' sends none and uses the per-model "
                         "Service path instead")
    args = ap.parse_args(argv)
    routing_strategy = None if args.routing_strategy.strip().lower() in ("", "none") else args.routing_strategy.strip()
    summary = run_trace(
        args.trace, gateway_url=args.gateway_url, out_path=args.out, registry_path=args.registry,
        seed=args.seed, dry_run=args.dry_run, window_ms=args.window_ms, step_ms=args.step_ms,
        max_in_flight=args.max_in_flight, trim_ramp_windows=args.trim_ramp_windows,
        prompt_path=args.prompt_file, prompt_workers=args.prompt_workers,
        rps_timeline_path=args.rps_timeline,
        routing_strategy=routing_strategy,
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
