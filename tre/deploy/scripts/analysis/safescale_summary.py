#!/usr/bin/env python3
"""Summarise the SafeScale probes of one run: rollback rate and rollback reasons.

Input: the ``safescale.json`` a campaign run writes (``campaign_queue.safescale_snapshot``:
``{"probes": {request_id: <json record>}, "journals": {...}}``), or a JSON object / list
of probe records. A probe record is what the controller keeps in
``tre:v2:controller:safescale:probes`` (``planning.safescale._probe_record``); the
2026-09-29 evidence fields (``window_terms`` / ``terminal_details``) are read when present.

Resolved probes are garbage-collected from that hash one hour after they resolve, so for
runs longer than an hour the snapshot taken at the end misses the early ones; the
controller log (``safescale_rollback_reason:`` / ``safescale_evidence:`` events) is the
complete record.

    python3 -m scripts.analysis.safescale_summary <run_dir>/safescale.json
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping


def load_probe_records(raw: Any) -> list[dict[str, Any]]:
    """Probe records from a campaign snapshot, a {id: record} mapping or a list."""
    if isinstance(raw, Mapping) and "probes" in raw:
        raw = raw["probes"]
    values: Iterable[Any] = raw.values() if isinstance(raw, Mapping) else (raw or [])
    records: list[dict[str, Any]] = []
    for value in values:
        if isinstance(value, (str, bytes)):
            try:
                value = json.loads(value)
            except (TypeError, ValueError):
                continue
        if isinstance(value, Mapping):
            records.append(dict(value))
    return records


def _resolution(record: Mapping[str, Any]) -> str | None:
    """commit / rollback for a resolved (or committing) probe; None while probing."""
    resolution = record.get("resolution")
    if resolution in ("commit", "rollback"):
        return str(resolution)
    return None


def _rollback_code(record: Mapping[str, Any]) -> str:
    for section in ("terminal_details", "window_terms"):
        reason = (record.get(section) or {}).get("rollback_reason")
        if isinstance(reason, Mapping) and reason.get("code"):
            return str(reason["code"])
    # Records of controllers before 2026-09-29: the plain terminal reason.
    return str(record.get("terminal_reason") or record.get("resolution_reason") or "unknown")


def summarize(records: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    records = list(records)
    decided = [record for record in records if _resolution(record) is not None]
    rollbacks = [record for record in decided if _resolution(record) == "rollback"]
    reasons = Counter(_rollback_code(record) for record in rollbacks)
    gates = Counter()
    formal_gates = Counter()
    extensions = []
    clamped = 0
    pre_hide = []
    samples = []
    modes = Counter()
    for record in decided:
        terms = record.get("window_terms") or {}
        if terms.get("latency_gate"):
            gates[str(terms["latency_gate"]) + (
                f":{terms['latency_skip_reason']}" if terms.get("latency_skip_reason") else ""
            )] += 1
        if terms.get("threshold_mode"):
            modes[str(terms["threshold_mode"])] += 1
        if terms.get("extensions") is not None:
            extensions.append(int(terms["extensions"]))
        if terms.get("clamped"):
            clamped += 1
        if terms.get("tail_pre_hide_fraction") is not None and terms.get("latency_source") == "evidence":
            pre_hide.append(float(terms["tail_pre_hide_fraction"]))
        if terms.get("latency_samples") is not None:
            samples.append(float(terms["latency_samples"]))
        reason = terms.get("rollback_reason")
        if isinstance(reason, Mapping) and reason.get("code") == "formal_commit_gate_failed":
            formal_gates.update(str(gate) for gate in reason.get("gates") or ())
    return {
        "probes": len(records),
        "decided": len(decided),
        "commits": len(decided) - len(rollbacks),
        "rollbacks": len(rollbacks),
        "rollback_rate": (len(rollbacks) / len(decided)) if decided else None,
        "rollback_reasons": dict(reasons.most_common()),
        "formal_gate_failures": dict(formal_gates.most_common()),
        "latency_gate": dict(gates.most_common()),
        "threshold_mode": dict(modes.most_common()),
        "extensions_total": sum(extensions),
        "extensions_max": max(extensions) if extensions else None,
        "clamped": clamped,
        "latency_samples_median": sorted(samples)[len(samples) // 2] if samples else None,
        # Regression assertion: the latency evidence never predates the hide.
        "evidence_pre_hide_fraction_max": max(pre_hide) if pre_hide else None,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("path", help="safescale.json of a run (or a JSON list / mapping of probe records)")
    args = parser.parse_args(argv)
    raw = json.loads(Path(args.path).read_text(encoding="utf-8"))
    print(json.dumps(summarize(load_probe_records(raw)), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
