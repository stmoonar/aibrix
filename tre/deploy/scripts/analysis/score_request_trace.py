#!/usr/bin/env python3
"""Re-score one request JSONL with a trace-start ramp trim."""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

from tre_common.registry import load_registry
from tre_replayer.scoring import compute_v_sys, trim_trace_start, window_violations


def load_request_jsonl(path: str | Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"line {line_number} is not a JSON object")
            records.append(value)
    return records


def score_request_trace(
    records: Sequence[dict[str, Any]],
    *,
    registry_path: str | None = None,
    window_ms: int = 30_000,
    step_ms: int = 5_000,
    min_samples: int = 5,
    trim_ramp_windows: int = 1,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    records = list(records)
    timestamps = [
        record.get("actual_send_ts_ms")
        for record in records
        if record.get("actual_send_ts_ms") is not None
    ]
    trace_start_ms = min(timestamps) if timestamps else None

    registry = load_registry(registry_path)
    slos = {model.name: model.slo for model in registry.models()}
    by_model: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        model = str(record.get("model") or "")
        if model not in slos:
            raise ValueError(f"request record uses unknown model {model!r}")
        by_model[model].append(record)

    per_model: dict[str, dict[str, Any]] = {}
    window_rows: list[dict[str, Any]] = []
    for model, model_records in sorted(by_model.items()):
        slo = slos[model]
        score = compute_v_sys(
            model_records,
            ttft_slo_ms=slo.ttft_p95_ms,
            tpot_slo_ms=slo.tpot_p95_ms,
            e2e_slo_ms=slo.e2e_p95_ms,
            window_ms=window_ms,
            step_ms=step_ms,
            min_samples=min_samples,
            trim_ramp_windows=trim_ramp_windows,
            trace_start_ms=trace_start_ms,
        )
        per_model[model] = score

        kept, scoring_start_ms = trim_trace_start(
            model_records,
            trim_ramp_windows=trim_ramp_windows,
            step_ms=step_ms,
            trace_start_ms=trace_start_ms,
        )
        windows = window_violations(
            kept,
            ttft_slo_ms=slo.ttft_p95_ms,
            tpot_slo_ms=slo.tpot_p95_ms,
            e2e_slo_ms=slo.e2e_p95_ms,
            window_ms=window_ms,
            step_ms=step_ms,
            t0_ms=scoring_start_ms,
            min_samples=min_samples,
        )
        for window in windows:
            window_rows.append(
                {
                    "model": model,
                    "trim_ramp_windows": trim_ramp_windows,
                    **window,
                }
            )

    scored_requests = sum(score["n_requests"] for score in per_model.values())
    successful_requests = sum(score["n_success"] for score in per_model.values())

    def weighted(metric: str) -> float | None:
        present = [
            (score[metric], score["n_requests"])
            for score in per_model.values()
            if score[metric] is not None
        ]
        denominator = sum(weight for _value, weight in present)
        if denominator == 0:
            return None
        return sum(value * weight for value, weight in present) / denominator

    summary = {
        "scoring": {
            "window_ms": window_ms,
            "step_ms": step_ms,
            "min_samples": min_samples,
            "trim_ramp_windows": trim_ramp_windows,
            "trim_scope": "trace_start_only",
            "trace_start_ms": trace_start_ms,
        },
        "system": {
            "n_requests_total": len(records),
            "n_requests": scored_requests,
            "n_requests_trimmed": len(records) - scored_requests,
            "n_success": successful_requests,
            "success_rate": (
                successful_requests / scored_requests if scored_requests else None
            ),
            "violation_request_frac": weighted("violation_request_frac"),
            "violation_time_frac": weighted("violation_time_frac"),
        },
        "per_model": per_model,
    }
    return summary, window_rows


def write_window_csv(path: str | Path, rows: Sequence[dict[str, Any]]) -> None:
    fieldnames = [
        "model",
        "trim_ramp_windows",
        "window_end_ms",
        "n_requests",
        "violated",
        "ttft_p95",
        "tpot_p95",
        "e2e_p95",
        "errors",
    ]
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Per-request JSONL")
    parser.add_argument("--output", required=True, help="Summary JSON")
    parser.add_argument("--windows-output", help="Optional per-window flags CSV")
    parser.add_argument("--registry")
    parser.add_argument("--window-ms", type=int, default=30_000)
    parser.add_argument("--step-ms", type=int, default=5_000)
    parser.add_argument("--min-samples", type=int, default=5)
    parser.add_argument("--trim-ramp-windows", type=int, default=1)
    args = parser.parse_args(argv)

    summary, windows = score_request_trace(
        load_request_jsonl(args.input),
        registry_path=args.registry,
        window_ms=args.window_ms,
        step_ms=args.step_ms,
        min_samples=args.min_samples,
        trim_ramp_windows=args.trim_ramp_windows,
    )
    summary["source"] = str(Path(args.input))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.windows_output:
        write_window_csv(args.windows_output, windows)
    print(
        f"requests={summary['system']['n_requests']} "
        f"trimmed={summary['system']['n_requests_trimmed']} "
        f"V_req={summary['system']['violation_request_frac']} "
        f"success={summary['system']['success_rate']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
