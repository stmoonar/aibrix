from __future__ import annotations

import csv
import json

from tre_calibration.cli import main
from tre_calibration.fit import (
    THETA_METHOD_BALANCED_ACCURACY,
    THETA_METHOD_RELIABILITY,
)

_BASE_ARGS = [
    "--trim-ramp-windows", "0",
    "--ttft-slo-mode", "fixed", "--ttft-p95-ms", "100",
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
]


def test_cli_defaults_to_the_balanced_accuracy_criterion(tmp_path) -> None:
    src = _write_csv(tmp_path)
    out = tmp_path / "patch.json"

    rc = main([
        "--input", str(src),
        "--output", str(out),
        "--model-name", "dsqwen-7b",
        "--no-fit-delta",
        *_BASE_ARGS,
    ])

    assert rc == 0
    patch = json.loads(out.read_text(encoding="utf-8"))
    assert patch["model_name"] == "dsqwen-7b"
    assert patch["method"]["theta_m_method"] == THETA_METHOD_BALANCED_ACCURACY
    # The criterion and every knob that moves theta are recorded in the artifact.
    assert patch["fit_config"]["theta_criterion"] == "balanced_accuracy"
    assert patch["fit_config"]["min_healthy_recall"] == 0.0
    assert patch["fit_config"]["trim_ramp_windows"] == 0
    assert patch["inputs"]["csv_path"] == str(src)
    assert len(patch["inputs"]["csv_sha256"]) == 64
    assert patch["inputs"]["window_count"] == 5
    # Healthy signals are 105/120/140. Every candidate quantile in [0.05, 0.50] gives
    # the same confusion matrix here, so the tie-break takes the largest theta.
    assert patch["trs"]["theta_m"] == 120.0
    assert patch["fit"]["healthy_quantile"] == 0.50
    assert patch["fit"]["recall_good"] == 2 / 3
    assert patch["fit"]["specificity_bad"] == 1.0
    assert patch["publish"] is True


def test_cli_can_still_fit_the_cumulative_attainment_baseline(tmp_path) -> None:
    src = _write_csv(tmp_path)
    out = tmp_path / "patch.json"

    rc = main([
        "--input", str(src),
        "--output", str(out),
        "--model-name", "dsqwen-7b",
        "--theta-criterion", "reliability",
        "--no-fit-delta",
        *_BASE_ARGS,
    ])

    assert rc == 0
    patch = json.loads(out.read_text(encoding="utf-8"))
    assert patch["method"]["theta_m_method"] == THETA_METHOD_RELIABILITY
    assert patch["fit_config"]["theta_criterion"] == "reliability"
    assert patch["publish"] is True
    assert patch["trs"]["theta_m"] == 105.0
    assert patch["trs"]["w_p"] == 0.04
    assert patch["metrics"]["auroc"] == 1.0
    assert patch["fit"]["support"] == 3
    assert patch["fit"]["coverage_pass"] is True


def _write_csv(tmp_path):
    src = tmp_path / "windows.csv"
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
    return src


def _row(scenario_id: str, family: str, trs: float, p95_ttft_client_ms: float, p95_tpot_client_ms: float) -> dict[str, str]:
    return {
        "scenario_id": scenario_id,
        "scenario_family": family,
        "trs": str(trs),
        "p95_ttft_client_ms": str(p95_ttft_client_ms),
        "p95_tpot_client_ms": str(p95_tpot_client_ms),
        "prompt_tokens_total": "100",
        "generation_tokens_total": "50",
    }
