#!/usr/bin/env python3
"""Equivalence check of the unified client against the senders it replaced.

Run against ``fake_openai_server.py --mode scenario`` (never a gateway). Each side runs
in its own interpreter with its own tree on PYTHONPATH (the old tree is a
``git archive`` of the commit before the merge), so both are driven by the same code
here:

``calib-run``
    One ``scripts.openloop.drive_cell_schedule`` cell per scenario model (``ok``,
    ``reasoning``, ``notext``, ``retried``, ``continued``, ``shed``, ``e500``,
    ``sseerror``, ``cut``; ``--api chat`` = the calibration request, ``completions`` = the
    replay request), prompts made deterministic by patching the sender's ``build_prompt``.
    Dumps the sender records, the raw rows and the server's view of every request.
``e1-run``
    ``python3 -m tre_loadgen_v1 --stage dispatch`` on a traces.json covering the E1
    client's scenarios; dumps performance_metrics.json and the server's view.
``compare``
    Diff of two dumps: request bytes and explicit headers per request id, record and raw
    fields that do not depend on the clock (exactly), clock fields (within a tolerance),
    and the transport-level headers that differ.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import urllib.request
from pathlib import Path

CALIB_MODELS = ("ok", "ok-stop", "reasoning", "notext", "retried", "continued", "shed", "e500", "sseerror", "cut")

#: Record / raw fields that are wall-clock instants or scheduling artefacts of one run:
#: never comparable across two runs, only their presence (None or not) is.
ABSOLUTE_FIELDS = {
    "scheduled_offset_ms", "actual_send_ts_ms", "send_ts_ms", "recv_first_token_ts_ms", "done_ts_ms",
    "scheduled_send_ts_ms", "schedule_delay_ms", "pool_wait_ms", "body_build_ms", "on_wire_delay_ms",
    "in_flight_at_send", "start_time", "end_time", "attempt_log", "process_id",
}
#: Durations measured on the answer: compared within a tolerance, and summarised.
DURATION_FIELDS = {"ttft_ms", "e2e_ms", "tpot_ms", "ttft", "tpot", "e2e_latency"}
CLOCK_FIELDS = ABSOLUTE_FIELDS | DURATION_FIELDS
E1_CLOCK_FIELDS = CLOCK_FIELDS
#: Headers every HTTP client adds on its own (not the request's own headers).
TRANSPORT_HEADERS = {"host", "user-agent", "connection", "accept-encoding", "content-length"}


def _records(url: str) -> list[dict]:
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(url.rstrip("/") + "/_records", timeout=30) as resp:
        return json.loads(resp.read())["records"]


def calib_run(args) -> dict:
    from scripts import openloop
    from tre_replayer.engine import http_sender
    from tre_replayer.engine.schedule import RpsSegment

    http_sender.build_prompt = lambda n, key, **kw: f"{key.split('|')[-1]} deterministic prompt {n}"
    path = "/v1/chat/completions" if args.api == "chat" else "/v1/completions"
    out: dict = {"impl": args.impl, "api": args.api, "cells": {}}
    tmp = Path(tempfile.mkdtemp(prefix=f"verify-{args.impl}-"))
    for model in CALIB_MODELS:
        segments = [RpsSegment(model=model, start_s=0.0, end_s=1.0, rps=8.0, input_tokens=16, max_output_tokens=6)]
        records: list = []
        raw = tmp / f"{model}.jsonl"
        kwargs = dict(seed=11, raw_path=raw, prompt_mode="natural" if args.api == "chat" else "text",
                      api=args.api, request_seed=7 if args.api == "chat" else None,
                      routing_strategy=args.routing or None, records_out=records, request_key=f"k{model}")
        if args.impl == "new":
            kwargs["sender_processes"] = args.processes
        try:
            openloop.drive_cell_schedule(args.url + path, model, f"c-{model}", segments, **kwargs)
            error = None
        except Exception as exc:  # noqa: BLE001 - recorded as the outcome
            error = f"{type(exc).__name__}: {exc}"
        rows = [json.loads(line) for line in raw.read_text().splitlines()] if raw.exists() else []
        out["cells"][model] = {"records": records, "raw": rows, "error": error}
    out["server"] = _records(args.url)
    return out


def e1_run(args) -> dict:
    tmp = Path(tempfile.mkdtemp(prefix=f"verify-e1-{args.impl}-"))
    traces = []
    t = 0.3
    for model in ("ok", "ok-stop", "reasoning", "notext", "retried", "continued", "fail503x1", "fail503x2",
                  "fail503x5", "e500", "sseerror", "cut", "hang"):
        for i in range(3):
            rid = f"req_{model}_{i}"
            traces.append({"request_id": rid, "timestamp": round(t, 3), "model_name": model,
                           "prompt": f"{rid} e1 prompt", "prompt_length": 3, "phase_type": "stable",
                           "max_output_tokens": (5 if i == 1 else None)})
            t += 0.05
    (tmp / "traces.json").write_text(json.dumps(traces), encoding="utf-8")
    cfg = {"custom_load_test": {
        "duration_seconds": 10, "gateway_endpoint": "http://localhost:8888", "generate_mode": "custom",
        "output": {"base_dir": str(tmp), "sub_dir_name": "run", "files": {
            "load_timeline_rps": "load_timeline_rps.json", "load_timeline_token_rate": "load_timeline_token_rate.json",
            "trace_data": "traces.json", "trace_plots": "traces.png", "performance_metrics": "performance_metrics.json",
            "performance_plots": "plot_analysis/"}},
        "random_seed": 1,
        "client": {"api_key": "dummy", "timeout": 2.0, "enable_streaming": True, "log_level": "INFO",
                   "routing_algorithm": "least-gpu-cache", "process_count": 4, "max_coroutines_per_process": 100,
                   "task_batch_window": 5.0, "load_monitor_interval": 1.0},
        "models": [{"name": m, "modelscope_url": f"fake/{m}", "max_tokens": 6}
                   for m in ("ok", "ok-stop", "reasoning", "notext", "retried", "continued", "fail503x1",
                             "fail503x2", "fail503x5", "e500", "sseerror", "cut")],
        "load_mode": "rps", "total_load": 4, "stable_period": {"min_duration": 30, "max_duration": 60},
        "transition_period": {"min_duration": 2, "max_duration": 5}, "noise": {"enabled": False},
        "custom_trace_json": "trace.json", "input_token_config": {}, "analysis": {"percentiles": [50, 90, 99]},
    }}
    import yaml

    (tmp / "config.yaml").write_text(yaml.safe_dump(cfg), encoding="utf-8")
    out_dir = tmp / "out"
    env = dict(os.environ, PYTHONPATH=args.loadgen_path)
    proc = subprocess.run([sys.executable, "-m", "tre_loadgen_v1", "--stage", "dispatch", "--config",
                           str(tmp / "config.yaml"), "--trace-file", str(tmp / "traces.json"), "--base-url",
                           args.url, "--output", str(out_dir)], env=env, capture_output=True, text=True,
                          timeout=600)
    metrics = out_dir / "performance_metrics.json"
    lines = [json.loads(line) for line in metrics.read_text().splitlines()] if metrics.exists() else []
    return {"impl": args.impl, "rc": proc.returncode, "stderr": proc.stderr[-2000:], "traces": traces,
            "records": lines, "server": _records(args.url)}


# ------------------------------------------------------------------ comparison


def _server_by_rid(server: list[dict]) -> dict:
    out: dict = {}
    for entry in server:
        out.setdefault(entry["rid"], []).append(entry)
    return out


def _compare_server(old: list[dict], new: list[dict], report: dict) -> None:
    o, n = _server_by_rid(old), _server_by_rid(new)
    report["requests"] = {"old": len(old), "new": len(new), "ids_equal": sorted(o) == sorted(n)}
    body_diff, header_diff, attempt_diff = [], [], []
    transport = {"old": {}, "new": {}}
    for rid in sorted(set(o) & set(n)):
        if len(o[rid]) != len(n[rid]):
            attempt_diff.append((rid, len(o[rid]), len(n[rid])))
        for a, b in zip(o[rid], n[rid]):
            if a["raw_b64"] != b["raw_b64"]:
                body_diff.append(rid)
            ha = {k: v for k, v in a["headers"].items() if k not in TRANSPORT_HEADERS}
            hb = {k: v for k, v in b["headers"].items() if k not in TRANSPORT_HEADERS}
            if ha != hb:
                header_diff.append({"rid": rid, "old": ha, "new": hb})
            for side, entry in (("old", a), ("new", b)):
                for key in TRANSPORT_HEADERS - {"host", "content-length"}:
                    transport[side].setdefault(key, set()).add(entry["headers"].get(key))
    report["request_bytes_identical"] = not body_diff
    report["body_diff"] = body_diff[:10]
    report["explicit_headers_identical"] = not header_diff
    report["header_diff"] = header_diff[:5]
    report["attempt_count_diff"] = attempt_diff
    report["transport_headers"] = {side: {k: sorted(map(str, v)) for k, v in d.items()} for side, d in transport.items()}


def _cmp_rows(old_rows: list[dict], new_rows: list[dict], key: str, clock: set, tolerance: float) -> dict:
    o = {r[key]: r for r in old_rows}
    n = {r[key]: r for r in new_rows}
    exact_diff, clock_diff, order_diff = [], [], []
    durations: dict = {}
    for rid in sorted(set(o) & set(n)):
        a, b = o[rid], n[rid]
        if list(a) != list(b)[:len(a)]:
            order_diff.append(rid)
        for field in a:
            if field in clock:
                x, y = a.get(field), b.get(field)
                if (x is None) != (y is None):
                    clock_diff.append((rid, field, x, y))
                elif field in DURATION_FIELDS and x is not None:
                    diff = abs(float(x) - float(y))
                    durations.setdefault(field, []).append(diff)
                    if diff > tolerance:
                        clock_diff.append((rid, field, x, y))
            elif a.get(field) != b.get(field):
                exact_diff.append((rid, field, a.get(field), b.get(field)))
    return {"rows": [len(o), len(n)], "ids_equal": sorted(o) == sorted(n),
            "new_only_fields": sorted(set().union(*(set(r) for r in new_rows)) - set().union(*(set(r) for r in old_rows)))
            if old_rows and new_rows else [],
            "field_order_equal": not order_diff, "exact_field_diffs": exact_diff[:20],
            "n_exact_field_diffs": len(exact_diff), "clock_field_diffs_over_tolerance": clock_diff[:20],
            "n_clock_diffs": len(clock_diff),
            "duration_abs_diff": {f: {"n": len(v), "median": sorted(v)[len(v) // 2], "max": max(v)}
                                  for f, v in durations.items()}}


def compare(args) -> dict:
    old = json.loads(Path(args.old).read_text())
    new = json.loads(Path(args.new).read_text())
    report: dict = {"kind": args.kind}
    _compare_server(old["server"], new["server"], report)
    if args.kind == "calib":
        cells = {}
        for model in CALIB_MODELS:
            a, b = old["cells"].get(model, {}), new["cells"].get(model, {})
            cells[model] = {
                "old_error": a.get("error"), "new_error": b.get("error"),
                "records": _cmp_rows(a.get("records", []), b.get("records", []), "request_id", CLOCK_FIELDS,
                                     args.tolerance_ms),
                "raw": _cmp_rows(a.get("raw", []), b.get("raw", []), "request_id", CLOCK_FIELDS, args.tolerance_ms),
            }
        report["cells"] = cells
    else:
        report["rc"] = [old["rc"], new["rc"]]
        report["records"] = _cmp_rows(old["records"], new["records"], "request_id", E1_CLOCK_FIELDS,
                                      args.tolerance_ms / 1000.0)
        # the v1 formulas, recomputed on the new records
        bad = []
        for r in new["records"]:
            if r["success"] and r["ttft"] is not None and r["output_tokens"]:
                expect = (r["end_time"] - (r["start_time"] + r["ttft"])) / r["output_tokens"]
                if abs(expect - r["tpot"]) > 1e-5:
                    bad.append(r["request_id"])
            if abs((r["end_time"] - r["start_time"]) - r["e2e_latency"]) > 1e-5:
                bad.append(r["request_id"])
        report["v1_formula_violations"] = bad
        o = {r["request_id"]: r for r in old["records"]}
        deltas = {"ttft": [], "tpot": [], "e2e_latency": []}
        for r in new["records"]:
            a = o.get(r["request_id"])
            if a and a["success"] and r["success"]:
                for f in deltas:
                    if a[f] is not None and r[f] is not None:
                        deltas[f].append(abs(a[f] - r[f]) * 1000.0)
        report["v1_clock_abs_diff_ms"] = {
            f: {"n": len(v), "median": sorted(v)[len(v) // 2] if v else None, "max": max(v) if v else None}
            for f, v in deltas.items()}
    return report


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("calib-run")
    c.add_argument("--url", required=True)
    c.add_argument("--impl", choices=["old", "new"], required=True)
    c.add_argument("--api", choices=["chat", "completions"], default="chat")
    c.add_argument("--routing", default="")
    c.add_argument("--processes", type=int, default=2)
    c.add_argument("--out", required=True)
    e = sub.add_parser("e1-run")
    e.add_argument("--url", required=True)
    e.add_argument("--impl", choices=["old", "new"], required=True)
    e.add_argument("--loadgen-path", required=True, help="the tree's tre/loadgen_v1 (PYTHONPATH of the run)")
    e.add_argument("--out", required=True)
    m = sub.add_parser("compare")
    m.add_argument("--kind", choices=["calib", "e1"], required=True)
    m.add_argument("--old", required=True)
    m.add_argument("--new", required=True)
    m.add_argument("--tolerance-ms", type=float, default=150.0)
    args = ap.parse_args(argv)
    if args.cmd == "calib-run":
        result = calib_run(args)
    elif args.cmd == "e1-run":
        result = e1_run(args)
    else:
        print(json.dumps(compare(args), indent=1, default=str))
        return 0
    Path(args.out).write_text(json.dumps(result, default=str), encoding="utf-8")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
