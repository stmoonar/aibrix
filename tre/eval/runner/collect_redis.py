#!/usr/bin/env python3
"""collect_redis.py <arm dir> <controller restart epoch s> <redis url>: end-of-arm Redis dumps (read-only).

Run by run_arm_pilot.sh after the load, next to the signal_log / SafeScale dump:

* ``decision_hist.jsonl`` - the controller's per-model decision history
  (``tre:v2:decision:hist:<model>``, ZSET scored by window_end_ms, kept ~24 h): one line per
  member from 60 s before the controller restart on, sorted by (score, model). The same model
  states as the ``trs_calc_result`` lines of controller.log (Z, TSS, queue, O1 window,
  ``signal_hold_reason``, ``sample_quality_hold``, saturation), kept as a second copy when the
  container log is rotated or truncated. Models: ``MODELS`` (comma list).
* ``sm_operations.jsonl`` - the SM operations journal (HASH ``tre:v2:sm:operations``: kind,
  actor, status, phases_ms of every wake / sleep / target change), the records started 60 s
  before the controller restart or later, sorted by ``started_at``.

Nothing is written to Redis. A key that is missing gives an empty file and a note on stdout.
"""
from __future__ import annotations

import datetime as dt
import json
import os
import sys

DECISION_HIST_PREFIX = "tre:v2:decision:hist:"
SM_OPERATIONS_KEY = "tre:v2:sm:operations"
LEAD_S = 60.0


def iso_epoch(stamp) -> float | None:
    """``2026-10-18T18:50:04.111039+00:00`` (SM ``started_at``) -> epoch s; None when unparsable."""
    try:
        t = dt.datetime.fromisoformat(str(stamp).replace("Z", "+00:00"))
    except ValueError:
        return None
    if t.tzinfo is None:
        t = t.replace(tzinfo=dt.timezone.utc)
    return t.timestamp()


def operations_since(raw: dict, t0_s: float) -> list[dict]:
    """SM journal records (HASH values, JSON) started at t0_s - 60 s or later, oldest first.
    Records without a parsable ``started_at`` are kept (fail open: never lose an operation)."""
    out = []
    for value in raw.values():
        try:
            rec = json.loads(value)
        except (TypeError, ValueError):
            continue
        if not isinstance(rec, dict):
            continue
        t = iso_epoch(rec.get("started_at"))
        if t is None or t >= t0_s - LEAD_S:
            out.append(rec)
    out.sort(key=lambda r: (iso_epoch(r.get("started_at")) or 0.0, str(r.get("operation_id"))))
    return out


def main(argv: list[str]) -> int:
    import redis  # noqa: PLC0415

    d, t0_s, url = argv[1], float(argv[2]), argv[3]
    r = redis.Redis.from_url(url, decode_responses=True, socket_timeout=10.0)
    models = [m for m in os.environ.get("MODELS", "").split(",") if m]
    n_dec = 0
    rows = []
    for m in models:
        for member, score in r.zrangebyscore(DECISION_HIST_PREFIX + m, int((t0_s - LEAD_S) * 1000), "+inf",
                                             withscores=True):
            try:
                rec = json.loads(member)
            except ValueError:
                rec = {"raw": member[:2000]}
            rows.append((score, m, rec))
    rows.sort(key=lambda x: (x[0], x[1]))
    with open(os.path.join(d, "decision_hist.jsonl"), "w") as f:
        for score, m, rec in rows:
            f.write(json.dumps({"score_ms": int(score), "model": m, **rec}, separators=(",", ":")) + "\n")
            n_dec += 1
    raw = {}
    for k, v in r.hscan_iter(SM_OPERATIONS_KEY, count=1000):
        raw[k] = v
    ops = operations_since(raw, t0_s)
    with open(os.path.join(d, "sm_operations.jsonl"), "w") as f:
        for rec in ops:
            f.write(json.dumps(rec, separators=(",", ":"), sort_keys=True) + "\n")
    print(f"decision_hist {n_dec} rows ({','.join(models) or 'no MODELS'}); sm_operations {len(ops)} of {len(raw)} records")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
