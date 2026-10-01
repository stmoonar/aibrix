#!/usr/bin/env python3
"""Scale-up / scale-down timing of a verification / smoke run, read-only, from its
evidence directory.

  scaleup_latency.py <evidence dir> [...] [--from load_start_epoch] [--target 4] [--json]

Per evidence directory and model, from ``layout.jsonl`` (the read-only sampler: one line
per ~1 s, ``models.<m>.awake`` = awake binding ids, i.e. in the SM's store, hidden or not):

* ``at_t0``: the awake count at the reference instant ``t0``;
* ``first_scale_up_s``: first instant the count rose above ``at_t0``;
* ``reach_<target>_s``: first instant it reached ``--target`` (default 4, the cap);
* ``first_scale_down_s``: first instant after the first scale-up the count fell below
  the maximum reached so far; ``first_scale_down_after_load_end_s`` the same instant
  relative to ``load_end_epoch`` (None without that file).

All in seconds after ``t0`` = the epoch in the evidence file named by ``--from``
(default ``load_start_epoch``; a C-crit run uses ``t_active_epoch``, the instant the
controller went active, because its load starts in observe). The sampler and the load
driver stamp on the same host clock, so no cross-node correction is needed. A quantity
that never happened is None.
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


def analyse(evidence: str, target: int, t0_file: str = "load_start_epoch") -> dict:
    t0 = _read_epoch(os.path.join(evidence, t0_file))
    end = _read_epoch(os.path.join(evidence, "load_end_epoch"))
    if t0 is None:
        return {"evidence": evidence, "error": f"no {t0_file}"}
    rows = []
    with open(os.path.join(evidence, "layout.jsonl"), encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except ValueError:
                    continue
    rows = sorted((r for r in rows if "ts" in r), key=lambda r: r["ts"])
    models = sorted({m for r in rows for m in (r.get("models") or {})})
    out = {"evidence": os.path.basename(evidence.rstrip("/")), "t0": t0_file,
           "load_s": None if end is None else round(end - t0, 1), "models": {}}
    for model in models:
        series = [(r["ts"], len(((r.get("models") or {}).get(model) or {}).get("awake") or [])) for r in rows]
        before = [n for ts, n in series if ts <= t0]
        base = before[-1] if before else (series[0][1] if series else 0)
        first_up = reach = first_down = None
        peak = base
        for ts, n in series:
            if ts < t0:
                continue
            if first_up is None and n > base:
                first_up = round(ts - t0, 1)
            if reach is None and n >= target:
                reach = round(ts - t0, 1)
            if first_up is not None and first_down is None and n < peak:
                first_down = ts
            peak = max(peak, n)
        out["models"][model] = {
            "at_t0": base, "peak": peak, "first_scale_up_s": first_up, f"reach_{target}_s": reach,
            "first_scale_down_s": None if first_down is None else round(first_down - t0, 1),
            "first_scale_down_after_load_end_s": (
                None if first_down is None or end is None else round(first_down - end, 1)),
        }
    return out


def main(argv) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("evidence", nargs="+")
    parser.add_argument("--from", dest="t0_file", default="load_start_epoch",
                        help="evidence file holding the reference epoch (default load_start_epoch)")
    parser.add_argument("--target", type=int, default=4)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    results = [analyse(path, args.target, args.t0_file) for path in args.evidence]
    if args.json:
        print(json.dumps(results, indent=1, sort_keys=True))
        return 0
    for res in results:
        if "error" in res:
            print(f"{res['evidence']}: {res['error']}")
            continue
        print(f"{res['evidence']}  (t0={res['t0']}, load {res['load_s']} s)")
        for model, m in res["models"].items():
            print(f"  {model:12s} t0={m['at_t0']} peak={m['peak']} first_up={m['first_scale_up_s']} s "
                  f"reach_{args.target}={m[f'reach_{args.target}_s']} s first_down={m['first_scale_down_s']} s "
                  f"(after load end {m['first_scale_down_after_load_end_s']} s)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
