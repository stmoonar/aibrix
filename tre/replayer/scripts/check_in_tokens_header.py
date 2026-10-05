#!/usr/bin/env python3
"""Check the sent ``x-tre-bl-in-tokens`` header against ``usage.prompt_tokens``.

Reads per-request records of a run made with ``--send-in-tokens`` (``tre_loadgen_v1``'s
``performance_metrics.json``, ``run_trace --out`` JSONL; one JSON object per line) and
compares each row's ``in_tokens_header`` (the value sent) with the engine's
``usage.prompt_tokens`` (``prompt_tokens`` on replayer rows, ``input_tokens`` on loadgen
v1 lines). Rows without a usage count (failed requests) and rows sent without the header
are counted, not compared.

    python3 check_in_tokens_header.py <records.jsonl> [more ...] [--limit 100] [--show 20]

Prints one JSON summary; exit 0 when every compared row matches, 1 on any mismatch,
2 when nothing could be compared (no row carries the field, or none has usage).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Iterable, Optional

FIELD = "in_tokens_header"


def usage_prompt_tokens(row: dict) -> Optional[int]:
    """The engine's ``usage.prompt_tokens`` on a record (None: the request has none)."""
    value = row["prompt_tokens"] if "prompt_tokens" in row else row.get("input_tokens")
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        return None  # a failed request: no usage (v1 lines record 0)
    return value


def check_rows(rows: Iterable[dict], *, limit: Optional[int] = None, show: int = 20) -> dict[str, Any]:
    out: dict[str, Any] = {"rows": 0, "without_field": 0, "header_omitted": 0, "no_usage": 0,
                           "compared": 0, "matched": 0, "mismatched": 0, "mismatches": []}
    by_model: dict[str, dict[str, int]] = {}
    for row in rows:
        out["rows"] += 1
        if FIELD not in row:
            out["without_field"] += 1
            continue
        sent = row[FIELD]
        if sent is None:
            out["header_omitted"] += 1
            continue
        usage = usage_prompt_tokens(row)
        if usage is None:
            out["no_usage"] += 1
            continue
        if limit is not None and out["compared"] >= limit:
            continue
        model = str(row.get("model") or row.get("model_name"))
        stats = by_model.setdefault(model, {"compared": 0, "mismatched": 0})
        out["compared"] += 1
        stats["compared"] += 1
        if int(sent) == usage:
            out["matched"] += 1
            continue
        out["mismatched"] += 1
        stats["mismatched"] += 1
        if len(out["mismatches"]) < show:
            out["mismatches"].append({"request_id": row.get("request_id"), "model": model, "sent": int(sent),
                                      "usage_prompt_tokens": usage, "diff": int(sent) - usage})
    out["by_model"] = dict(sorted(by_model.items()))
    return out


def read_rows(paths: Iterable[str]) -> Iterable[dict]:
    for path in paths:
        with Path(path).open(encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    yield json.loads(line)


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("records", nargs="+", help="per-request JSONL record files")
    ap.add_argument("--limit", type=int, default=None, help="compare at most this many rows (default: all)")
    ap.add_argument("--show", type=int, default=20, help="mismatches to list (default %(default)s)")
    args = ap.parse_args(argv)
    summary = check_rows(read_rows(args.records), limit=args.limit, show=args.show)
    print(json.dumps(summary, indent=2))
    if summary["mismatched"]:
        return 1
    return 0 if summary["compared"] else 2


if __name__ == "__main__":
    sys.exit(main())
