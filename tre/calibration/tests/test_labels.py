"""The shared SLO label (plan 2026-09-21 §6.3 B2/B4), now ``tre_common.slo_labels``.

Unserved evidence is the three count columns; ``slo_violated`` is the label's output and
is never read back as "unserved" (the merge guard below)."""
from __future__ import annotations

import csv
import random
from pathlib import Path

import pytest

from tre_calibration.bootstrap import bootstrap_theta, bootstrap_theta_and_delta_crit
from tre_calibration.dataset import CalibrationWindow, load_windows_from_csv
from tre_calibration.fit import DEFAULT_HEALTHY_QUANTILE_CANDIDATES, ThetaFitConfig
from tre_common.slo_labels import UNSERVED_MIN_RATIO, LabelDefinition, label_window

LABEL = LabelDefinition(500.0, 75.0)


def test_label_is_ttft_and_tpot_only() -> None:
    assert LABEL.latency_slo_ms() == {"ttft_p95": 500.0, "tpot_p95": 75.0}
    assert LABEL.as_dict()["e2e"] == "excluded"
    # an e2e far over any SLO does not violate
    lab = label_window({"p95_ttft": 100, "p95_tpot": 20, "p95_e2e": 99999}, LABEL.latency_slo_ms())
    assert lab is not None and lab.slo_met


@pytest.mark.parametrize("column", ["model_errors", "proxy_transient_errors", "client_timeouts"])
def test_unserved_forces_violation_even_with_good_latency(column) -> None:
    lab = label_window({"p95_ttft": 100, "p95_tpot": 20, column: "1"}, LABEL.latency_slo_ms())
    assert lab is not None and not lab.slo_met and lab.unserved
    lab = label_window({"p95_ttft": 100, "p95_tpot": 20, column: "1"}, LABEL)
    assert lab is not None and not lab.slo_met and lab.unserved


def test_missing_latency_is_dropped_unless_unserved() -> None:
    assert label_window({"p95_ttft": "", "p95_tpot": 20}, LABEL.latency_slo_ms()) is None
    lab = label_window({"p95_ttft": "", "p95_tpot": "", "client_timeouts": 2}, LABEL.latency_slo_ms())
    assert lab is not None and not lab.slo_met and lab.ratio_max == UNSERVED_MIN_RATIO


def test_label_rejects_non_positive_slo() -> None:
    with pytest.raises(ValueError):
        LabelDefinition(0.0, 75.0)


def test_dataset_loader_reads_the_unserved_counts(tmp_path: Path) -> None:
    path = tmp_path / "w.csv"
    fields = ["scenario_id", "scenario_family", "window_start_ms", "window_end_ms",
              "prompt_tokens_total", "generation_tokens_total", "p95_ttft_client_ms",
              "p95_tpot_client_ms", "trs", "model_errors", "proxy_transient_errors",
              "client_timeouts", "slo_violated"]
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerow(dict(scenario_id="a", scenario_family="f", window_start_ms=0, window_end_ms=30000,
                        prompt_tokens_total=100, generation_tokens_total=100, p95_ttft_client_ms=100,
                        p95_tpot_client_ms=20, trs=50, model_errors=0, proxy_transient_errors=0,
                        client_timeouts=1, slo_violated="True"))
        w.writerow(dict(scenario_id="a", scenario_family="f", window_start_ms=5000, window_end_ms=35000,
                        prompt_tokens_total=100, generation_tokens_total=100, p95_ttft_client_ms="",
                        p95_tpot_client_ms="", trs=40, model_errors=2, proxy_transient_errors=0,
                        client_timeouts=0, slo_violated="True"))
        w.writerow(dict(scenario_id="a", scenario_family="f", window_start_ms=10000, window_end_ms=40000,
                        prompt_tokens_total=100, generation_tokens_total=100, p95_ttft_client_ms=100,
                        p95_tpot_client_ms=20, trs=60, model_errors=0, proxy_transient_errors=0,
                        client_timeouts=0, slo_violated="False"))
    windows = load_windows_from_csv(path, latency_slo_ms=LABEL.latency_slo_ms())
    assert [w.slo_met for w in windows] == [False, False, True]
    assert [w.violation_class for w in windows] == ["unserved", "unserved", None]


def test_a_main_format_violation_is_not_read_as_unserved(tmp_path: Path) -> None:
    """Merge guard (2026-09-23): in a 09-23 window CSV ``slo_violated`` is the LABEL, so a
    latency-violated window carries slo_violated=True with all three counts 0. Read as
    "unserved" (the 09-22 meaning of the column) it would make every violation unserved,
    grade it UNSERVED_MIN_RATIO, and - on a CSV whose windows all violate - report 100 %
    violations without one error. It must be read from the counts only."""
    path = tmp_path / "main.csv"
    fields = ["scenario_id", "scenario_family", "window_start_ms", "window_end_ms",
              "prompt_tokens_total", "generation_tokens_total", "p95_ttft_client_ms",
              "p95_tpot_client_ms", "trs", "model_errors", "proxy_transient_errors",
              "client_timeouts", "slo_label", "slo_violated"]
    rows = [
        # violated through TPOT only: stays violated, but NOT unserved, graded by latency
        dict(p95_ttft_client_ms=100, p95_tpot_client_ms=150, slo_label="violated", slo_violated="True"),
        # a stale / foreign slo_violated=True on healthy latency: healthy, not unserved
        dict(p95_ttft_client_ms=100, p95_tpot_client_ms=20, slo_label="violated", slo_violated="True"),
        dict(p95_ttft_client_ms=100, p95_tpot_client_ms=20, slo_label="healthy", slo_violated="False"),
    ]
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for i, extra in enumerate(rows):
            w.writerow(dict(scenario_id="a", scenario_family="f", window_start_ms=i * 10000,
                            window_end_ms=i * 10000 + 30000, prompt_tokens_total=100,
                            generation_tokens_total=100, trs=50 + i, model_errors=0,
                            proxy_transient_errors=0, client_timeouts=0, **extra))
    for spec in (LABEL, LABEL.latency_slo_ms()):
        windows = load_windows_from_csv(path, latency_slo_ms=spec)
        assert [w.slo_met for w in windows] == [False, True, True]
        assert [w.violation_class for w in windows] == ["tpot_only", None, None]
        assert windows[0].latency_ratio_p95 == pytest.approx(2.0)  # 150/75, not the 2.0 stand-in
        assert not any(label_window(r, LABEL).unserved for r in csv.DictReader(path.open()))


def test_healthy_quantile_grid_reaches_down_to_one_percent() -> None:
    q = DEFAULT_HEALTHY_QUANTILE_CANDIDATES
    assert q[:5] == (0.01, 0.02, 0.03, 0.04, 0.05) and q[-1] == 0.50


def _synthetic(seed: int = 5) -> list[CalibrationWindow]:
    rng = random.Random(seed)
    out: list[CalibrationWindow] = []
    for cell in range(24):
        load = 0.4 + 0.05 * cell
        for k in range(12):
            signal = 100.0 / load * rng.uniform(0.9, 1.1)
            ratio = load * rng.uniform(0.8, 1.2)
            out.append(CalibrationWindow(
                scenario_id=f"c{cell}", scenario_family="S1" if cell % 2 else "S4",
                signal=signal, slo_met=ratio <= 1.0, health_score=1 / (1 + ratio),
                latency_ratio_p95=ratio,
            ))
    return out


def test_joint_bootstrap_keeps_the_theta_interval_and_adds_delta() -> None:
    windows = _synthetic()
    cfg = ThetaFitConfig()
    plain = bootstrap_theta(windows, n_resamples=40, seed=11, config=cfg)
    joint, delta = bootstrap_theta_and_delta_crit(windows, n_resamples=40, seed=11, config=cfg)
    assert joint.theta_values == plain.theta_values
    assert delta.n_fitted > 0
    assert 0.0 <= delta.delta_p2_5 <= delta.delta_p50 <= delta.delta_p97_5 <= 0.5
