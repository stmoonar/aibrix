from __future__ import annotations

import csv

from tre_calibration.dataset import load_windows_from_csv


def test_load_windows_from_csv_filters_and_labels_slo_health(tmp_path) -> None:
    src = tmp_path / "windows.csv"
    rows = [
        {
            "scenario_id": "steady-a",
            "scenario_family": "steady",
            "trs": "120",
            "p95_ttft": "90",
            "p95_tpot": "40",
            "prompt_tokens_total": "100",
            "generation_tokens_total": "50",
        },
        {
            "scenario_id": "burst-b",
            "scenario_family": "burst",
            "trs": "80",
            "p95_ttft": "130",
            "p95_tpot": "45",
            "prompt_tokens_total": "80",
            "generation_tokens_total": "30",
        },
        {
            "scenario_id": "warmup",
            "scenario_family": "steady",
            "trs": "200",
            "p95_ttft": "70",
            "p95_tpot": "20",
            "prompt_tokens_total": "1",
            "generation_tokens_total": "1",
            "is_warmup": "1",
        },
        {
            "scenario_id": "filtered",
            "scenario_family": "steady",
            "trs": "210",
            "p95_ttft": "70",
            "p95_tpot": "20",
            "prompt_tokens_total": "1",
            "generation_tokens_total": "1",
            "filter_reason": "contaminated",
        },
        {
            "scenario_id": "empty-tokens",
            "scenario_family": "steady",
            "trs": "220",
            "p95_ttft": "70",
            "p95_tpot": "20",
            "prompt_tokens_total": "0",
            "generation_tokens_total": "0",
        },
    ]
    with src.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=sorted({key for row in rows for key in row}))
        writer.writeheader()
        writer.writerows(rows)

    windows = load_windows_from_csv(src, latency_slo_ms={"ttft_p95": 100.0, "tpot_p95": 50.0})

    assert [window.scenario_id for window in windows] == ["steady-a", "burst-b"]
    assert windows[0].signal == 120.0
    assert windows[0].slo_met is True
    assert round(windows[0].health_score or 0.0, 6) == round(1.0 / 1.9, 6)
    assert windows[1].signal == 80.0
    assert windows[1].slo_met is False
    assert round(windows[1].health_score or 0.0, 6) == round(1.0 / 2.3, 6)
