from __future__ import annotations

import argparse
import bisect
import csv
import re
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

MODELS = ("dsllama-8b", "dsqwen-7b", "dsqwen-14b")
NODES = ("nscc-ds-4a100-node9", "nscc-ds-4a100-node10")


@dataclass(frozen=True)
class PowerEvent:
    timestamp: float
    serve_id: str
    model: str
    node: str
    gpu_ids: tuple[int, ...]
    awake: bool


@dataclass(frozen=True)
class RunBound:
    trace: str
    arm: str
    start_ts: float
    end_ts: float


class PowerTimeline:
    def __init__(self, events: Iterable[PowerEvent]) -> None:
        grouped: dict[str, list[PowerEvent]] = defaultdict(list)
        for event in events:
            grouped[event.serve_id].append(event)
        self._events = {
            serve_id: sorted(rows, key=lambda row: row.timestamp)
            for serve_id, rows in grouped.items()
        }
        self._timestamps = {
            serve_id: [row.timestamp for row in rows]
            for serve_id, rows in self._events.items()
        }

    def state_at(self, timestamp: float) -> tuple[PowerEvent, ...]:
        awake: list[PowerEvent] = []
        for serve_id, rows in self._events.items():
            index = bisect.bisect_right(self._timestamps[serve_id], timestamp) - 1
            if index >= 0 and rows[index].awake:
                awake.append(rows[index])
        return tuple(awake)


def _parse_ts(value: str) -> float:
    normalized = value.strip().replace("Z", "+00:00")
    normalized = re.sub(r"(\.\d{6})\d+(?=[+-]\d{2}:\d{2}$)", r"\1", normalized)
    return datetime.fromisoformat(normalized).timestamp()


def load_power_events(path: Path) -> list[PowerEvent]:
    events: list[PowerEvent] = []
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            events.append(
                PowerEvent(
                    timestamp=_parse_ts(row["timestamp"]),
                    serve_id=row["serve_id"],
                    model=row["model"],
                    node=row["node"],
                    gpu_ids=tuple(int(value) for value in re.findall(r"\d+", row["gpu_ids"])),
                    awake=row["state"] == "awake",
                )
            )
    return events


def load_run_bounds(path: Path) -> list[RunBound]:
    bounds: list[RunBound] = []
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            bounds.append(
                RunBound(
                    trace=row["trace"],
                    arm=row["arm"],
                    start_ts=_parse_ts(row["start_ts"]),
                    end_ts=_parse_ts(row["end_ts"]),
                )
            )
    return bounds


def load_timeline(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def _parse_awake_counts(value: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in value.split(","):
        model, separator, count = item.partition("=")
        if separator:
            counts[model] = int(count)
    return counts


def audit_placement(
    *,
    timelines: dict[str, list[dict[str, str]]],
    bounds: list[RunBound],
    power_events: list[PowerEvent],
    output_dir: Path,
    max_share_diff: float = 0.10,
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    power = PowerTimeline(power_events)
    placement_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []

    expected_runs = {(f"t{index}", arm) for index in range(1, 8) for arm in ("tre", "apa")}
    actual_runs = {(bound.trace, bound.arm) for bound in bounds}
    if actual_runs != expected_runs:
        missing = sorted(expected_runs - actual_runs)
        extra = sorted(actual_runs - expected_runs)
        raise ValueError(f"run bounds must cover 7x2 runs; missing={missing}, extra={extra}")

    for bound in sorted(bounds, key=lambda item: (int(item.trace[1:]), item.arm)):
        timeline = timelines.get(bound.arm)
        if timeline is None:
            raise ValueError(f"missing timeline for arm {bound.arm}")
        samples = [
            row
            for row in timeline
            if bound.start_ts <= _parse_ts(row["ts"]) <= bound.end_ts
        ]
        if not samples:
            raise ValueError(f"no timeline samples for {bound.trace}/{bound.arm}")

        node10_shares: list[float] = []
        max_node10_coresidency = 0
        mismatch_count = 0
        for sample in samples:
            timestamp = _parse_ts(sample["ts"])
            awake = power.state_at(timestamp)
            expected = _parse_awake_counts(sample["awake"])
            counts: dict[tuple[str, str], int] = defaultdict(int)
            for event in awake:
                counts[(event.model, event.node)] += 1

            for model in MODELS:
                event_total = sum(counts[(model, node)] for node in NODES)
                if model in expected and expected[model] != event_total:
                    mismatch_count += 1
                placement_rows.append(
                    {
                        "trace": bound.trace,
                        "arm": bound.arm,
                        "window_ts": sample["ts"],
                        "model": model,
                        "node9_awake": counts[(model, NODES[0])],
                        "node10_awake": counts[(model, NODES[1])],
                    }
                )

            node10_awake = sum(counts[(model, NODES[1])] for model in MODELS)
            total_awake = sum(counts.values())
            node10_shares.append(node10_awake / total_awake if total_awake else 0.0)
            max_node10_coresidency = max(max_node10_coresidency, node10_awake)

        summary_rows.append(
            {
                "trace": bound.trace,
                "arm": bound.arm,
                "mean_node10_share": sum(node10_shares) / len(node10_shares),
                "max_coresidency_node10": max_node10_coresidency,
                "window_count": len(samples),
                "aggregate_count_mismatches": mismatch_count,
            }
        )

    by_trace: dict[str, dict[str, dict[str, object]]] = defaultdict(dict)
    for row in summary_rows:
        by_trace[str(row["trace"])][str(row["arm"])] = row

    verdict_rows: list[dict[str, object]] = []
    for trace in sorted(by_trace, key=lambda value: int(value[1:])):
        tre = by_trace[trace]["tre"]
        apa = by_trace[trace]["apa"]
        share_diff = abs(float(tre["mean_node10_share"]) - float(apa["mean_node10_share"]))
        coresidency_equal = tre["max_coresidency_node10"] == apa["max_coresidency_node10"]
        passed = share_diff <= max_share_diff and coresidency_equal
        reasons: list[str] = []
        if share_diff > max_share_diff:
            reasons.append("node10_share_diff")
        if not coresidency_equal:
            reasons.append("node10_coresidency_asymmetry")
        verdict_rows.append(
            {
                "trace": trace,
                "verdict": "PASS" if passed else "FLAG",
                "mean_node10_share_diff": share_diff,
                "tre_max_coresidency_node10": tre["max_coresidency_node10"],
                "apa_max_coresidency_node10": apa["max_coresidency_node10"],
                "tre_count_mismatches": tre["aggregate_count_mismatches"],
                "apa_count_mismatches": apa["aggregate_count_mismatches"],
                "reason": ";".join(reasons) if reasons else "placement_symmetric",
            }
        )

    _write_csv(
        output_dir / "placement_audit.csv",
        placement_rows,
        ("trace", "arm", "window_ts", "model", "node9_awake", "node10_awake"),
    )
    _write_csv(
        output_dir / "placement_summary.csv",
        summary_rows,
        (
            "trace",
            "arm",
            "mean_node10_share",
            "max_coresidency_node10",
            "window_count",
            "aggregate_count_mismatches",
        ),
    )
    _write_csv(
        output_dir / "placement_verdicts.csv",
        verdict_rows,
        (
            "trace",
            "verdict",
            "mean_node10_share_diff",
            "tre_max_coresidency_node10",
            "apa_max_coresidency_node10",
            "tre_count_mismatches",
            "apa_count_mismatches",
            "reason",
        ),
    )
    return placement_rows, summary_rows, verdict_rows


def _write_csv(path: Path, rows: list[dict[str, object]], fields: tuple[str, ...]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit node placement in exp3 timelines")
    parser.add_argument("--timeline-tre", type=Path, required=True)
    parser.add_argument("--timeline-apa", type=Path, required=True)
    parser.add_argument("--run-bounds", type=Path, required=True)
    parser.add_argument("--pod-events", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-share-diff", type=float, default=0.10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    audit_placement(
        timelines={
            "tre": load_timeline(args.timeline_tre),
            "apa": load_timeline(args.timeline_apa),
        },
        bounds=load_run_bounds(args.run_bounds),
        power_events=load_power_events(args.pod_events),
        output_dir=args.output_dir,
        max_share_diff=args.max_share_diff,
    )


if __name__ == "__main__":
    main()
