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
from pathlib import Path
from typing import Any

from tre_replayer.engine import rps_timeline
from tre_replayer.engine.corpus import effective_zh_ratio
from tre_replayer.engine.dispatcher import dispatch_open_loop
from tre_replayer.engine.http_sender import DEFAULT_ROUTING_STRATEGY, StreamResult, StreamingHttpSender
from tre_replayer.engine.profiles import PROFILE_E1_V1, PROFILE_REPLAY, V1ChatOptions
from tre_replayer.engine.stream import TTFT_BASIS
from tre_replayer.engine.prompt_store import materialize_prompts
from tre_replayer.engine.prompts import CORPUS_LANGS, DEFAULT_CORPUS_LANG, DEFAULT_ZH_RATIO
from tre_replayer.engine.schedule import build_poisson_schedule
from tre_replayer.scoring import compute_v_sys
from tre_replayer.traces.loader import load_trace_segments


#: The E1 arms' client (2026-10-01): v1's request (chat, stream, no ignore_eos,
#: max_tokens from the trace, temperature unset), v1's process count, and the retries
#: smoke-E1's run_arm.sh runs the same client with.
E1_DEFAULT_PROCESSES = 8
E1_DEFAULT_MAX_RETRIES = 0
E1_TIMEOUT_S = 300.0


def _unit_interval(text: str) -> float:
    """argparse type: a float within [0, 1]."""
    value = float(text)
    if not 0.0 <= value <= 1.0:
        raise argparse.ArgumentTypeError(f"must be within [0, 1], got {text}")
    return value


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
    corpus_lang: str = DEFAULT_CORPUS_LANG,
    zh_ratio: float = DEFAULT_ZH_RATIO,
    client_profile: str = PROFILE_REPLAY,
    sender_processes: int | None = None,
    max_retries: int = E1_DEFAULT_MAX_RETRIES,
) -> dict[str, Any]:
    from tre_common.registry import load_registry

    segments = load_trace_segments(trace_path)
    schedule = build_poisson_schedule(segments, seed=seed)
    if client_profile == PROFILE_E1_V1:
        if dry_run:
            raise ValueError("--dry-run has no e1_v1 form (the SDK transport has no synchronous seam)")
        records, report, client = _run_e1(
            schedule, gateway_url=gateway_url, prompt_path=prompt_path, prompt_workers=prompt_workers,
            routing_strategy=routing_strategy, corpus_lang=corpus_lang, zh_ratio=zh_ratio,
            processes=E1_DEFAULT_PROCESSES if sender_processes is None else sender_processes,
            max_retries=max_retries,
        )
        if out_path:
            _write_jsonl(out_path, records)
        return _summarise(
            trace_path, schedule, records, report, registry_path=registry_path, window_ms=window_ms,
            step_ms=step_ms, trim_ramp_windows=trim_ramp_windows, rps_timeline_path=rps_timeline_path,
            routing_strategy=routing_strategy, corpus_lang=corpus_lang, zh_ratio=zh_ratio,
            prompt_store_misses=0, client=client,
        )
    if client_profile != PROFILE_REPLAY:
        raise ValueError(f"run_trace sends the replay or the e1_v1 profile, not {client_profile!r}")
    # Prompts are built here, before the loop, not inside each send: see
    # tre_replayer.engine.prompt_store. Without a path there is nowhere to put them and
    # the sender falls back to fitting inline.
    prompt_store = (
        None
        if prompt_path is None
        else materialize_prompts(
            schedule, path=prompt_path, processes=prompt_workers,
            corpus_lang=corpus_lang, zh_ratio=zh_ratio,
        )
    )
    sender = StreamingHttpSender(
        gateway_url,
        stream_call=_dry_stream_call if dry_run else None,
        max_in_flight=max_in_flight,
        prompt_store=prompt_store,
        routing_strategy=routing_strategy or None,
        corpus_lang=corpus_lang,
        zh_ratio=zh_ratio,
    )
    dispatch_kwargs = {"sleep": sleep} if sleep is not None else {}

    async def _dispatch_then_close():
        # The replay sends from this process: one asyncio loop, the pooled async
        # transport (tre_replayer.engine.transport), closed on the loop that used it.
        try:
            return await dispatch_open_loop(schedule, sender, **dispatch_kwargs)
        finally:
            await sender.aclose()

    try:
        report = asyncio.run(_dispatch_then_close())
    finally:
        sender.close()
    if out_path:
        sender.write_jsonl(out_path)
    return _summarise(
        trace_path, schedule, sender.records, report, registry_path=registry_path, window_ms=window_ms,
        step_ms=step_ms, trim_ramp_windows=trim_ramp_windows, rps_timeline_path=rps_timeline_path,
        routing_strategy=routing_strategy, corpus_lang=corpus_lang, zh_ratio=zh_ratio,
        prompt_store_misses=sender.prompt_store_misses, client=sender.provenance(processes=1),
    )


def e1_base_url(gateway_url: str) -> str:
    """The gateway base the SDK appends ``/v1/...`` to, from an endpoint URL."""
    url = gateway_url.rstrip("/")
    for suffix in ("/v1/chat/completions", "/v1/completions", "/v1"):
        if url.endswith(suffix):
            return url[: -len(suffix)]
    return url


def _run_e1(schedule, *, gateway_url, prompt_path, prompt_workers, routing_strategy, corpus_lang, zh_ratio,
            processes, max_retries):
    """Send ``schedule`` with the e1_v1 profile from ``processes`` workers; rows are the
    e1 record plus the replay row's keys (strict basis) the scoring reads."""
    import tempfile
    from dataclasses import replace

    from tre_replayer.engine.procpool import ProcessPoolRunner

    # The trace's input_tokens is the length of the prompt *text* (v1's prompt_length),
    # so the prompts are fitted as plain text and the chat template comes on top, as it
    # did for v1. Built before anything forks, as for every other profile.
    path = prompt_path or str(Path(tempfile.mkdtemp(prefix="tre-e1-prompts-")) / "prompts.jsonl")
    store = materialize_prompts(schedule, path=path, processes=prompt_workers, corpus_lang=corpus_lang,
                                zh_ratio=zh_ratio, api="completions")
    events = [replace(event, prompt=store.get(event.request_id)) for event in schedule]
    options = V1ChatOptions(
        # every model: max_tokens from the trace, temperature unset (JSON null) - v1's config
        model_params={event.model: {"max_tokens": None, "temperature": None} for event in events},
        max_retries=max_retries, timeout_s=E1_TIMEOUT_S, routing_strategy=routing_strategy or None,
    )
    base = e1_base_url(gateway_url)

    def make(index, in_flight, on_record):
        return StreamingHttpSender(base, profile=PROFILE_E1_V1, v1_options=options, in_flight=in_flight,
                                   on_record=on_record, process_id=index)

    client = make(0, None, None).provenance(processes=processes)
    by_id = {event.request_id: event for event in events}

    def validate(event) -> None:
        if not event.prompt:
            raise ValueError(f"{event.request_id}: no prompt was materialised")

    with ProcessPoolRunner(events, make, processes=processes, validate=validate) as runner:
        run = runner.run()
    return [e1_replay_row(rec, by_id[rec["request_id"]]) for rec in run.records], run.report, client


def e1_replay_row(record: dict, event) -> dict:
    """An e1_v1 record with the replay row's keys on the strict basis (what the scoring,
    the request-health gate and the arrival series read). The v1 fields stay as they
    are; where a name is taken, the v1 value moves to ``*_v1``.

    Token fields: ``input_tokens`` is the prompt length the trace asked for (the replay
    row's meaning; v1's usage value is kept as ``input_tokens_v1`` and ``prompt_tokens``);
    ``output_tokens`` is the ACTUAL completion length (usage; e1_v1 sends no
    ``ignore_eos``, so it may end below the bound), also kept as ``output_tokens_v1`` and
    ``completion_tokens``; the trace's upper bound (``max_tokens``) is
    ``max_output_tokens``. The scoring reads ``completion_tokens`` / ``ttft_ms`` /
    ``e2e_ms`` / ``http_status`` / ``error``, the request-health gate ``http_status``."""
    ms = (lambda s: None if s is None else s * 1000.0)
    strict_ok = bool(record.get("success_strict"))
    row = dict(record)
    row.update({
        "model": record["model_name"],
        "scheduled_offset_s": float(event.scheduled_offset_s),
        "actual_send_ts_ms": int(round(float(record["start_time"]) * 1000.0)),
        "on_wire_delay_ms": record.get("send_lateness_ms"),
        "ttft_ms": ms(record.get("ttft_strict_s")),
        "e2e_ms": ms(record.get("e2e_strict_s")),
        "input_tokens_v1": record.get("input_tokens"),
        "output_tokens_v1": record.get("output_tokens"),
        "prompt_tokens": record.get("input_tokens"),
        "completion_tokens": record.get("output_tokens"),
        "input_tokens": event.prompt_tokens,
        "output_tokens": record.get("output_tokens"),
        "max_output_tokens": event.max_output_tokens,
        "http_status_v1": record.get("http_status"),
        "http_status": record.get("http_status_strict"),
        "error": None if strict_ok else (
            f"{record.get('failure_strict')}: {record.get('error_message') or record.get('stream_error') or ''}"),
        "ttft_basis": TTFT_BASIS,
        "api": "chat",
        "client_profile": PROFILE_E1_V1,
    })
    return row


def _write_jsonl(path: str, records: list[dict]) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(record, separators=(",", ":")) + "\n")


def _summarise(trace_path, schedule, records, report, *, registry_path, window_ms, step_ms, trim_ramp_windows,
               rps_timeline_path, routing_strategy, corpus_lang, zh_ratio, prompt_store_misses, client):
    from tre_common.registry import load_registry

    registry = load_registry(registry_path)
    slos = {m.name: m.slo for m in registry.models()}
    per_model: dict[str, Any] = {}
    by_model: dict[str, list[dict]] = {}
    trace_timestamps = [
        record.get("actual_send_ts_ms")
        for record in records
        if record.get("actual_send_ts_ms") is not None
    ]
    trace_start_ms = min(trace_timestamps) if trace_timestamps else None
    for rec in records:
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
    for rec in records:
        pod = rec.get("target_pod")
        if pod:
            by_pod = target_pods.setdefault(rec["model"], {})
            by_pod[pod] = by_pod.get(pod, 0) + 1
    reissue = reissue_summary(records)
    return {
        "trace": trace_path,
        "routing_strategy": routing_strategy or None,
        # What the prompts were written in (tre_replayer.engine.corpus): runs made with
        # different corpora are not comparable, so the summary says which it was.
        "corpus_lang": corpus_lang,
        "zh_ratio": effective_zh_ratio(corpus_lang, zh_ratio),
        "target_pods": target_pods,
        "requests": len(records),
        "schedule_p99_delay_ms": round(report.p99_delay_ms, 2),
        "schedule_rps_error": round(report.actual_rps_error_ratio, 4),
        "rps_error_by_model": rps_error,
        "rps_timeline": rps_timeline_path,
        "max_pool_wait_ms": round(max((r.get("pool_wait_ms") or 0.0 for r in records), default=0.0), 2),
        # Scheduled instant -> socket call, the whole of it. Unlike the two above it
        # includes whatever the worker did before the send, which is where an inline
        # prompt fit used to hide.
        "max_on_wire_delay_ms": round(max((r.get("on_wire_delay_ms") or 0.0 for r in records), default=0.0), 2),
        "prompt_store_misses": prompt_store_misses,
        # The client that sent it: profile (replay = completions + ignore_eos; e1_v1 =
        # v1's request), wire, processes, code.
        "client": client,
        "trim_ramp_windows": trim_ramp_windows,
        # Reissue sidecar outcomes (plan 2026-09-27 P5): a run with continue > 0 had
        # requests stitched across pods and is flagged as contaminated.
        "reissue": reissue,
        "reissue_contaminated": reissue["continue"] > 0,
        "per_model": per_model,
    }


def reissue_summary(records: list[dict]) -> dict[str, Any]:
    """Counts of what the reissue sidecar did, from the per-request records: ``abort``
    (the client got finish_reason=abort), ``retry`` (resent through the gateway before it
    started), ``continue`` (stitched from several pods) and the segments stitched in;
    the same per model."""

    def count(recs: list[dict]) -> dict[str, int]:
        return {
            "abort": sum(1 for r in recs if r.get("finish_reason") == "abort"),
            "retry": sum(1 for r in recs if (r.get("tre_retried") or 0) > 0),
            "continue": sum(1 for r in recs if (r.get("tre_continued") or 0) > 0),
            "continued_segments": sum(int(r.get("tre_continued") or 0) for r in recs),
        }

    by_model: dict[str, list[dict]] = {}
    for record in records:
        by_model.setdefault(record.get("model"), []).append(record)
    out: dict[str, Any] = count(records)
    out["by_model"] = {model: count(recs) for model, recs in sorted(by_model.items(), key=lambda kv: str(kv[0]))}
    return out


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
    ap.add_argument("--max-in-flight", type=int, default=512)  # sender connection pool (F5)
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
    ap.add_argument("--corpus-lang", default=DEFAULT_CORPUS_LANG, choices=list(CORPUS_LANGS),
                    help="text of the natural prompts: mix (default; English and Chinese "
                         "sentences, --zh-ratio of the tokens Chinese), en, zh")
    ap.add_argument("--zh-ratio", type=_unit_interval, default=DEFAULT_ZH_RATIO,
                    help="Chinese share of each prompt's tokens under --corpus-lang mix, "
                         "counted with the model's own tokenizer (default %(default)s)")
    ap.add_argument("--client-profile", default=PROFILE_REPLAY, choices=[PROFILE_REPLAY, PROFILE_E1_V1],
                    help="replay (default): /v1/completions, ignore_eos, fixed lengths; e1_v1: v1's "
                         "request - chat, stream, no ignore_eos, max_tokens from the trace "
                         "(tre_replayer.engine.profiles)")
    ap.add_argument("--sender-processes", type=int, default=None,
                    help="worker processes (e1_v1 only; default %d, v1's process_count)" % E1_DEFAULT_PROCESSES)
    ap.add_argument("--max-retries", type=int, default=E1_DEFAULT_MAX_RETRIES,
                    help="OpenAI SDK retries (e1_v1 only; default %(default)s, as run_arm.sh)")
    args = ap.parse_args(argv)
    routing_strategy = None if args.routing_strategy.strip().lower() in ("", "none") else args.routing_strategy.strip()
    summary = run_trace(
        args.trace, gateway_url=args.gateway_url, out_path=args.out, registry_path=args.registry,
        seed=args.seed, dry_run=args.dry_run, window_ms=args.window_ms, step_ms=args.step_ms,
        max_in_flight=args.max_in_flight, trim_ramp_windows=args.trim_ramp_windows,
        prompt_path=args.prompt_file, prompt_workers=args.prompt_workers,
        rps_timeline_path=args.rps_timeline,
        routing_strategy=routing_strategy,
        corpus_lang=args.corpus_lang, zh_ratio=args.zh_ratio,
        client_profile=args.client_profile, sender_processes=args.sender_processes,
        max_retries=args.max_retries,
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
