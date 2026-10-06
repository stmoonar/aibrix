#!/usr/bin/env python3
"""Summarise a tre_controller.profile_dump CSV: per-phase p50/p95/p99 (ms), CPU%, RSS.
Usage: profile_summary.py profile.csv [--from-ms A --to-ms B] [--json out.json]"""
import csv
import json
import sys
from collections import defaultdict


def pct(vals, q):
    if not vals:
        return None
    v = sorted(vals)
    k = (len(v) - 1) * q
    lo, hi = int(k), min(int(k) + 1, len(v) - 1)
    return v[lo] + (v[hi] - v[lo]) * (k - lo)


def stats(vals):
    return {"n": len(vals), "p50": pct(vals, .5), "p95": pct(vals, .95), "p99": pct(vals, .99),
            "max": max(vals) if vals else None, "mean": (sum(vals) / len(vals)) if vals else None}


def main(argv):
    path = argv[0]
    a = int(argv[argv.index("--from-ms") + 1]) if "--from-ms" in argv else None
    b = int(argv[argv.index("--to-ms") + 1]) if "--to-ms" in argv else None
    out = argv[argv.index("--json") + 1] if "--json" in argv else None
    series = defaultdict(list)
    for row in csv.DictReader(open(path)):
        ts = int(float(row.get("ts_ms") or 0))
        if (a is not None and ts < a) or (b is not None and ts > b):
            continue
        kind = row.get("kind")

        def ms(col):
            v = row.get(col)
            return None if v in (None, "") else float(v) / 1e6

        if kind == "tick":
            loop = row.get("loop") or "?"
            for col in ("signals_ns", "plan_ns", "safescale_ns", "submit_ns", "tick_total_ns"):
                v = ms(col)
                if v is not None:
                    series[f"tick.{loop}.{col[:-3]}_ms"].append(v)
            for col in ("cpu_user_ms_delta", "cpu_sys_ms_delta"):
                if row.get(col):
                    series[f"tick.{loop}.{col}"].append(float(row[col]))
        elif kind == "decision":
            v = ms("decision_write_ns")
            if v is not None:
                series[f"decision_write.{row.get('loop')}_ms"].append(v)
        elif kind == "poll":
            v = ms("fetch_ns")
            if v is not None:
                series["poll.fetch_ms"].append(v)
        elif kind == "dispatch":
            v = ms("http_ns")
            if v is not None:
                series["dispatch.sm_http_ms"].append(v)
        elif kind == "proc":
            series["proc.cpu_percent"].append(float(row["cpu_percent"]))
            series["proc.rss_mib"].append(float(row["rss_mib"]))
    res = {k: stats(v) for k, v in sorted(series.items())}
    for k, s in res.items():
        f = lambda x: "-" if x is None else f"{x:.2f}"
        print(f"{k:40} n={s['n']:6} p50={f(s['p50']):>9} p95={f(s['p95']):>9} p99={f(s['p99']):>9} "
              f"max={f(s['max']):>9} mean={f(s['mean']):>9}")
    if out:
        json.dump(res, open(out, "w"), indent=1)


if __name__ == "__main__":
    main(sys.argv[1:])
