from __future__ import annotations

import csv
import json

from tre_calibration.cli import main


def test_cli_writes_profile_patch_from_synthetic_csv(tmp_path) -> None:
    src = tmp_path / "windows.csv"
    out = tmp_path / "patch.json"
    rows = [
        _row("steady-low", "steady", 60.0, 130.0, 45.0),
        _row("burst-low", "burst", 80.0, 125.0, 45.0),
        _row("steady-good", "steady", 105.0, 80.0, 35.0),
        _row("burst-good", "burst", 120.0, 85.0, 35.0),
        _row("steady-high", "steady", 140.0, 70.0, 30.0),
    ]
    with src.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    rc = main([
        "--input", str(src),
        "--output", str(out),
        "--model-name", "dsqwen-7b",
        "--trim-ramp-windows", "0",
        "--ttft-p95-ms", "100",
        "--tpot-p95-ms", "50",
        "--reliability-target", "0.9",
        "--min-support", "3",
        "--min-confidence", "0.9",
        "--min-scenario-families", "2",
        "--max-single-scenario-ratio", "0.7",
        "--w-p", "0.04",
        "--lambda-wait", "2.625",
        "--qmin", "1.0",
        "--generated-at", "2026-07-04T00:00:00+00:00",
    ])

    assert rc == 0
    patch = json.loads(out.read_text(encoding="utf-8"))
    assert patch["model_name"] == "dsqwen-7b"
    assert patch["publish"] is True
    assert patch["trs"]["theta_m"] == 105.0
    assert patch["trs"]["w_p"] == 0.04
    assert patch["metrics"]["auroc"] == 1.0
    assert patch["fit"]["support"] == 3
    assert patch["fit"]["coverage_pass"] is True


def _row(scenario_id: str, family: str, trs: float, p95_ttft: float, p95_tpot: float) -> dict[str, str]:
    return {
        "scenario_id": scenario_id,
        "scenario_family": family,
        "trs": str(trs),
        "p95_ttft": str(p95_ttft),
        "p95_tpot": str(p95_tpot),
        "prompt_tokens_total": "100",
        "generation_tokens_total": "50",
    }
