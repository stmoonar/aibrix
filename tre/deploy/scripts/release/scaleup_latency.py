#!/usr/bin/env python3
"""Scale-up latency of a verification / smoke run, read-only, from its evidence directory.

  scaleup_latency.py <evidence dir> [<evidence dir> ...] [--target 4] [--json]

Per evidence directory and model: the awake replica count when the load started, the
first instant it rose above that (first scale-up) and the first instant it reached
``--target`` (default 4, the replica cap), in seconds after ``load_start_epoch``.
Inputs: ``load_start_epoch`` (and ``load_end_epoch`` if present) and ``layout.jsonl``
(the read-only sampler: one line per ~1 s, ``models.<m>.awake`` = awake binding ids).
Both are stamped on the same host clock (the sampler and the load driver run on the same
node), so no cross-node clock correction is needed. Awake = in the SM's store (a woken
binding, hidden or not); a run without any rise reports None.
"""
from __future__ import annotations

import argparse
import json
import os
import sys


def _read_epoch(path: str):
    try:
        return float(open(path, encoding="utf-8").read().strip())
    except (OSError, ValueError):
        return None


def analyse(evidence: str, target: int) -> dict:
    start = _read_epoch(os.path.join(evidence, "load_start_epoch"))
    end = _read_epoch(os.path.join(evidence, "load_end_epoch"))
    if start is None:
        return {"evidence": evidence, "error": "no load_start_epoch"}
    rows = []
    with open(os.path.join(evidence, "layout.jsonl"), encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except ValueError:
                    continue
    rows.sort(key=lambda r: r.get("ts", 0))
    models = sorted({m for r in rows for m in (r.get("models") or {})})
    out = {"evidence": os.path.basename(evidence.rstrip("/")), "load_start": start,
           "load_s": None if end is None else round(end - start, 1), "models": {}}
    for model in models:
        series = [(r["ts"], len(((r.get("models") or {}).get(model) or {}).get("awake") or []))
                  for r in rows if "ts" in r]
        before = [n for ts, n in series if ts <= start]
        base = before[-1] if before else (series[0][1] if series else 0)
        first_up = reach = None
        peak = base
        for ts, n in series:
            if ts < start or (end is not None and ts > end + 120):
                continue
            peak = max(peak, n)
            if first_up is None and n > base:
                first_up = round(ts - start, 1)
            if reach is None and n >= target:
                reach = round(ts - start, 1)
        out["models"][model] = {"at_load_start": base, "peak": peak, "first_scale_up_s": first_up,
                                f"reach_{target}_s": reach}
    return out


def main(argv) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("evidence", nargs="+")
    parser.add_argument("--target", type=int, default=4)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    results = [analyse(path, args.target) for path in args.evidence]
    if args.json:
        print(json.dumps(results, indent=1, sort_keys=True))
        return 0
    for res in results:
        if "error" in res:
            print(f"{res['evidence']}: {res['error']}")
            continue
        print(f"{res['evidence']}  (load {res['load_s']} s)")
        for model, m in res["models"].items():
            print(f"  {model:12s} start={m['at_load_start']} peak={m['peak']} "
                  f"first_up={m['first_scale_up_s']} s  reach_{args.target}={m[f'reach_{args.target}_s']} s")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
