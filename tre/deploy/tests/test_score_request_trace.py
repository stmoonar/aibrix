from __future__ import annotations

import csv
import json
from pathlib import Path

from scripts.analysis import score_request_trace


def _record(timestamp_ms: int, *, status: int, e2e_ms: float) -> dict:
    return {
        "model": "dsqwen-7b",
        "actual_send_ts_ms": timestamp_ms,
        "ttft_ms": 50.0,
        "e2e_ms": e2e_ms,
        "completion_tokens": 10,
        "http_status": status,
        "error": None,
    }


def test_cli_trims_first_sliding_window_and_writes_flags(tmp_path) -> None:
    source = tmp_path / "requests.jsonl"
    summary_path = tmp_path / "summary.json"
    windows_path = tmp_path / "windows.csv"
    records = [
        *[_record(i * 1000, status=503, e2e_ms=900.0) for i in range(6)],
        *[_record(6000 + i * 1000, status=200, e2e_ms=400.0) for i in range(55)],
    ]
    source.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )
    registry = Path(__file__).parents[1] / "registry.yaml"

    result = score_request_trace.main(
        [
            "--input",
            str(source),
            "--output",
            str(summary_path),
            "--windows-output",
            str(windows_path),
            "--registry",
            str(registry),
            "--trim-ramp-windows",
            "1",
        ]
    )

    assert result == 0
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["scoring"]["trim_ramp_windows"] == 1
    assert summary["scoring"]["trim_scope"] == "trace_start_only"
    assert summary["system"]["n_requests_trimmed"] == 6
    assert summary["system"]["violation_request_frac"] == 0.0
    assert summary["system"]["success_rate"] == 1.0
    with windows_path.open(encoding="utf-8", newline="") as source_file:
        windows = list(csv.DictReader(source_file))
    assert windows
    assert int(windows[0]["window_end_ms"]) == 35_000
    assert all(row["violated"] == "False" for row in windows)
