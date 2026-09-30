#!/usr/bin/env python3
"""Send-timing benchmark of the load clients, against ``fake_openai_server.py --mode bench``.

Local only (the fake server on 127.0.0.1); never a gateway. Subcommands:

``plan``
    A Poisson schedule (``--rate`` rps for ``--duration`` s, seeded) with explicit prompts
    whose first word is the request id; also written as a v1 ``traces.json`` + config so
    the E1 clients replay the same requests.
``run-calib``
    Send the plan with the calibration sender of the tree on PYTHONPATH: ``--impl new``
    (the unified client, ``--processes`` workers) or ``--impl old`` (the pre-2026-09-30
    sender: one asyncio dispatcher + a 4096-thread urllib pool). Writes the sender rows.
``timed -- CMD``
    Run CMD, report its wall time and the CPU (user+sys) of it and every process it
    waited for.
``analyze``
    Send lateness as the client measured it, arrival lateness as the server saw it (the
    common ground truth: server arrival - scheduled instant), and the TTFT timing
    overhead (client TTFT - server TTFT on the same request).
"""
from __future__ import annotations

import argparse
import json
import math
import random
import re
import resource
import subprocess
import sys
import time
from pathlib import Path


def _pct(values, p):
    xs = sorted(v for v in values if v is not None)
    if not xs:
        return None
    k = (len(xs) - 1) * p / 100.0
    lo = math.floor(k)
    hi = min(lo + 1, len(xs) - 1)
    return round(xs[lo] + (xs[hi] - xs[lo]) * (k - lo), 3)


def _stats(values) -> dict:
    xs = [v for v in values if v is not None]
    return {"n": len(xs), "p50": _pct(xs, 50), "p99": _pct(xs, 99), "max": round(max(xs), 3) if xs else None}


def plan(args) -> None:
    rng = random.Random(args.seed)
    t = 0.0
    rows = []
    filler = ("lorem ipsum dolor sit amet " * (args.in_chars // 27 + 1))[: args.in_chars]
    while True:
        t += rng.expovariate(args.rate)
        if t >= args.duration:
            break
        rid = f"b{len(rows):06d}"
        rows.append({"request_id": rid, "offset": round(t, 6), "prompt": f"{rid} {filler}",
                     "max_tokens": args.out_tokens, "model": args.model})
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "plan.json").write_text(json.dumps(rows), encoding="utf-8")
    traces = [{"request_id": r["request_id"], "timestamp": r["offset"], "model_name": r["model"],
               "prompt": r["prompt"], "prompt_length": args.in_chars // 4, "phase_type": "stable",
               "max_output_tokens": r["max_tokens"]} for r in rows]
    (out / "traces.json").write_text(json.dumps(traces), encoding="utf-8")
    import yaml

    cfg = {"custom_load_test": {
        "duration_seconds": int(args.duration), "gateway_endpoint": "http://unused", "generate_mode": "custom",
        "output": {"base_dir": str(out), "sub_dir_name": "e1", "files": {
            "load_timeline_rps": "load_timeline_rps.json", "load_timeline_token_rate": "load_timeline_token_rate.json",
            "trace_data": "traces.json", "trace_plots": "traces.png", "performance_metrics": "performance_metrics.json",
            "performance_plots": "plot_analysis/"}},
        "client": {"api_key": "dummy", "timeout": 300.0, "enable_streaming": True, "log_level": "INFO",
                   "routing_algorithm": "least-gpu-cache", "process_count": args.e1_processes,
                   "max_coroutines_per_process": 100, "task_batch_window": 5.0, "load_monitor_interval": 1.0},
        "models": [{"name": args.model, "modelscope_url": "fake/x", "max_tokens": args.out_tokens}],
        "load_mode": "rps", "total_load": args.rate, "stable_period": {"min_duration": 30, "max_duration": 60},
        "transition_period": {"min_duration": 2, "max_duration": 5}, "noise": {"enabled": False},
        "custom_trace_json": "x", "input_token_config": {}, "analysis": {"percentiles": [50, 90, 99]},
    }}
    (out / "config.yaml").write_text(yaml.safe_dump(cfg), encoding="utf-8")
    print(f"{len(rows)} requests over {args.duration} s")


def run_calib(args) -> None:
    import asyncio

    from tre_replayer.engine.dispatcher import dispatch_open_loop
    from tre_replayer.engine.http_sender import StreamingHttpSender
    from tre_replayer.engine.schedule import ScheduledRequest

    rows = json.loads(Path(args.plan).read_text())
    events = [ScheduledRequest(r["request_id"], r["model"], r["offset"], prompt=r["prompt"],
                               max_output_tokens=r["max_tokens"]) for r in rows]
    url = args.url.rstrip("/") + "/v1/chat/completions"
    if args.impl == "new":
        from tre_replayer.engine.procpool import ProcessPoolRunner

        def make(index, in_flight, on_record):
            return StreamingHttpSender(url, api="chat", max_in_flight=4096, in_flight=in_flight,
                                       on_record=on_record, process_id=index)

        records = ProcessPoolRunner(events, make, processes=args.processes).run().records
    else:
        sender = StreamingHttpSender(url, api="chat", max_in_flight=4096)
        try:
            asyncio.run(dispatch_open_loop(events, sender))
        finally:
            sender.close()
        records = sender.records
    with open(args.out, "w", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(record) + "\n")
    print(f"{len(records)} records")


def timed(args) -> None:
    before = resource.getrusage(resource.RUSAGE_CHILDREN)
    start = time.time()
    proc = subprocess.run(args.cmd)
    wall = time.time() - start
    after = resource.getrusage(resource.RUSAGE_CHILDREN)
    cpu = (after.ru_utime - before.ru_utime) + (after.ru_stime - before.ru_stime)
    report = {"rc": proc.returncode, "wall_s": round(wall, 2), "cpu_s": round(cpu, 2),
              "cpu_cores_avg": round(cpu / wall, 3) if wall else None}
    if args.out:
        Path(args.out).write_text(json.dumps(report), encoding="utf-8")
    print(json.dumps(report))


def _server_log(prefix: str) -> dict:
    out = {}
    for path in sorted(Path(prefix).parent.glob(Path(prefix).name + ".[0-9]*")):
        for line in path.read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                out.setdefault(row["rid"], row)  # first attempt
    return out


def analyze(args) -> None:
    plan_rows = {r["request_id"]: r for r in json.loads(Path(args.plan).read_text())}
    server = _server_log(args.server_log)
    records = [json.loads(line) for line in Path(args.records).read_text().splitlines() if line.strip()]
    lateness, arrival, overhead = [], [], []
    base_epoch = None
    if args.kind == "lg-old":
        text = Path(args.lg_log).read_text(encoding="utf-8", errors="replace")
        base_epoch = float(re.search(r"基准时间: ([0-9.]+)", text).group(1))
    for rec in records:
        rid = rec.get("request_id")
        srv = server.get(rid)
        if args.kind == "calib":
            late = rec.get("on_wire_delay_ms")
            sched_epoch = (rec["actual_send_ts_ms"] - (late or 0.0)) / 1000.0
            ttft_ms = rec.get("ttft_ms")
        elif args.kind == "e1-new":
            late = rec.get("send_lateness_ms")
            sched_epoch = rec["start_time"] - (late or 0.0) / 1000.0
            ttft_ms = None if rec.get("ttft") is None else rec["ttft"] * 1000.0
        else:
            sched_epoch = base_epoch + float(rec["timestamp"])
            late = (rec["start_time"] - sched_epoch) * 1000.0
            ttft_ms = None if rec.get("ttft") is None else rec["ttft"] * 1000.0
        lateness.append(late)
        if srv is not None:
            arrival.append((srv["t"] - sched_epoch) * 1000.0)
            if ttft_ms is not None and srv.get("t_first") is not None:
                overhead.append(ttft_ms - (srv["t_first"] - srv["t"]) * 1000.0)
    in_flight = [r.get("in_flight_at_send") for r in records if r.get("in_flight_at_send") is not None]
    out = {"planned": len(plan_rows), "records": len(records), "server_seen": len(server),
           "client_send_lateness_ms": _stats(lateness), "server_arrival_lateness_ms": _stats(arrival),
           "ttft_overhead_ms": _stats(overhead),
           "max_in_flight_at_send": max(in_flight) if in_flight else None}
    if args.cpu:
        out["cpu"] = json.loads(Path(args.cpu).read_text())
    print(json.dumps(out))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd_name", required=True)
    p = sub.add_parser("plan")
    p.add_argument("--rate", type=float, required=True)
    p.add_argument("--duration", type=float, required=True)
    p.add_argument("--in-chars", type=int, default=1024)
    p.add_argument("--out-tokens", type=int, default=128)
    p.add_argument("--model", default="bench")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--e1-processes", type=int, default=8)
    p.add_argument("--out", required=True)
    r = sub.add_parser("run-calib")
    r.add_argument("--impl", choices=["new", "old"], required=True)
    r.add_argument("--processes", type=int, default=4)
    r.add_argument("--url", required=True)
    r.add_argument("--plan", required=True)
    r.add_argument("--out", required=True)
    t = sub.add_parser("timed")
    t.add_argument("--out", default=None)
    t.add_argument("cmd", nargs=argparse.REMAINDER)
    a = sub.add_parser("analyze")
    a.add_argument("--kind", choices=["calib", "e1-new", "lg-old"], required=True)
    a.add_argument("--plan", required=True)
    a.add_argument("--records", required=True)
    a.add_argument("--server-log", required=True)
    a.add_argument("--lg-log", default=None)
    a.add_argument("--cpu", default=None)
    args = ap.parse_args(argv)
    if args.cmd_name == "timed" and args.cmd and args.cmd[0] == "--":
        args.cmd = args.cmd[1:]
    {"plan": plan, "run-calib": run_calib, "timed": timed, "analyze": analyze}[args.cmd_name](args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
