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

from tre_replayer.engine.dispatcher import dispatch_open_loop
from tre_replayer.engine.http_sender import StreamResult, StreamingHttpSender
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
    sleep: Any = None,
) -> dict[str, Any]:
    from tre_common.registry import load_registry

    segments = load_trace_segments(trace_path)
    schedule = build_poisson_schedule(segments, seed=seed)
    sender = StreamingHttpSender(gateway_url, stream_call=_dry_stream_call if dry_run else None)
    dispatch_kwargs = {"sleep": sleep} if sleep is not None else {}
    report = asyncio.run(dispatch_open_loop(schedule, sender, **dispatch_kwargs))
    if out_path:
        sender.write_jsonl(out_path)

    registry = load_registry(registry_path)
    slos = {m.name: m.slo for m in registry.models()}
    per_model: dict[str, Any] = {}
    by_model: dict[str, list[dict]] = {}
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
        )
    return {
        "trace": trace_path,
        "requests": len(sender.records),
        "schedule_p99_delay_ms": round(report.p99_delay_ms, 2),
        "schedule_rps_error": round(report.actual_rps_error_ratio, 4),
        "per_model": per_model,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace", required=True)
    ap.add_argument("--gateway-url", default="http://10.103.92.7/v1/completions")
    ap.add_argument("--out", default=None)
    ap.add_argument("--registry", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--window-ms", type=int, default=30_000)
    ap.add_argument("--step-ms", type=int, default=5_000)
    args = ap.parse_args(argv)
    summary = run_trace(
        args.trace, gateway_url=args.gateway_url, out_path=args.out, registry_path=args.registry,
        seed=args.seed, dry_run=args.dry_run, window_ms=args.window_ms, step_ms=args.step_ms,
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
