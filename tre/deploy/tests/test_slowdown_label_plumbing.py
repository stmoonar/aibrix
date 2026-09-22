"""The slowdown TTFT label reaches every fit (plan 2026-09-21 §6.11 D6)."""
from __future__ import annotations

from pathlib import Path

from scripts import calibration_campaign as campaign
from scripts import r3_grid
from scripts.rewindow_from_raw import window_request_evidence
from scripts.theta_verdict import violation_class_breakdown
from tre_calibration.dataset import CalibrationWindow
from tre_calibration.labels import LabelDefinition, parse_ttft_len_samples
from tre_common.registry import load_registry


class _Args:
    out_dir = Path("/out")
    window_ms = 30000
    fit_step_ms = 5000
    instant_sample_ms = 1000
    ttft_slo_ms = 500.0
    tpot_slo_ms = 75.0
    max_model_error_rate = 0.05
    envoy_stats_url = None


def _value(cmd: list[str], flag: str) -> str:
    return cmd[cmd.index(flag) + 1]


def test_fit_plan_defaults_to_the_slowdown_label_with_the_registry_idle_fit() -> None:
    models = ["dsqwen-7b", "dsqwen-14b"]
    plan = campaign.fit_plan(models, Path("/out"), Path("/raw"), _Args())
    reg = load_registry()
    for model in models:
        d = plan["label_def_by_model"][model]
        assert d["mode"] == "slowdown" and d["k"] == 3.0 and d["floor_ms"] == 150.0 and d["min_n"] == 20
        assert d["c_ms"] == reg.model(model).slo.ttft_idle_c_ms
        assert d["b_ms_per_token"] == reg.model(model).slo.ttft_idle_b_ms_per_token
        # every per-model fit rebuilds exactly that label
        for step in ("theta", "verdict", "ablation"):
            for entry in plan[step]:
                if entry["model"] != model:
                    continue
                cmd = entry["command"]
                assert _value(cmd, "--ttft-slo-mode") == "slowdown"
                assert float(_value(cmd, "--ttft-idle-c-ms")) == d["c_ms"]
                assert float(_value(cmd, "--ttft-idle-b-ms-per-token")) == d["b_ms_per_token"]
                assert _value(cmd, "--min-completed-requests") == "20"
    assert plan["label_def"] == plan["label_def_by_model"]["dsqwen-7b"]
    # the multi-model alt fit resolves each model's own c/b from the registry
    for entry in plan["alt"]:
        cmd = entry["command"]
        assert _value(cmd, "--ttft-slo-mode") == "slowdown"
        assert "--ttft-idle-c-ms" not in cmd and "--ttft-idle-b-ms-per-token" not in cmd


def test_fit_plan_keeps_the_fixed_label_as_an_option() -> None:
    class Fixed(_Args):
        fit_ttft_slo_mode = "fixed"

    plan = campaign.fit_plan(["dsqwen-7b"], Path("/out"), Path("/raw"), Fixed())
    assert plan["label_def"]["mode"] == "fixed" and plan["label_def"]["ttft_p95_ms"] == 500.0
    for entry in plan["theta"]:
        assert _value(entry["command"], "--ttft-slo-mode") == "fixed"
        assert "--ttft-idle-c-ms" not in entry["command"]


def test_rewindow_writes_the_per_request_evidence() -> None:
    records = [
        {"done_ts_ms": 1000, "ttft_ms": 50.0, "input_tokens": 256},
        {"done_ts_ms": 2000, "ttft_ms": 150.0, "input_tokens": 2048},
        {"done_ts_ms": 2500, "ttft_ms": None, "input_tokens": 2048},  # no TTFT: not counted
        {"done_ts_ms": 40000, "ttft_ms": 70.0, "input_tokens": 256},  # outside the window
    ]
    ev = window_request_evidence(records, 0, 30000)
    assert ev["completed_requests"] == 2
    assert parse_ttft_len_samples(ev["ttft_len_samples"]) == [(50.0, 256.0), (150.0, 2048.0)]
    assert {"completed_requests", "ttft_len_samples"} <= set(r3_grid.CSV_COLUMNS)


def test_violation_class_breakdown_reports_critical_recall_per_class() -> None:
    def win(signal: float, cls: str | None) -> CalibrationWindow:
        return CalibrationWindow("c", "f", signal, cls is None, violation_class=cls)

    windows = [win(10.0, "ttft_only"), win(90.0, "ttft_only"), win(10.0, "both"), win(200.0, None)]
    out = violation_class_breakdown(windows, theta=100.0, tau_crit=0.5, direction="higher_is_healthier")
    assert out["ttft_only"] == {"windows": 2, "critical_recall": 0.5}
    assert out["both"] == {"windows": 1, "critical_recall": 1.0}
    assert out["tpot_only"] == {"windows": 0, "critical_recall": None}


def test_holdout_rebuilds_the_label_from_the_verdict() -> None:
    label = LabelDefinition(500.0, 75.0, ttft_slo_mode="slowdown", ttft_idle_c_ms=36.4,
                            ttft_idle_b_ms_per_token=0.0527, ttft_slowdown_k=2.0)
    assert LabelDefinition.from_dict(label.as_dict()) == label
