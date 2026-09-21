"""The alternative signals are thresholded by the same criterion as TSS.

E2 asks whether TSS separates healthy from violating windows better than queue length or
the per-replica token rates. If TSS were fitted by the balanced-accuracy criterion while
its alternatives kept the cumulative-attainment containment rule, the answer would be a
statement about the two criteria and not about the signals. These tests pin the three
things that keeps honest:

* every alternative signal goes through the balanced-accuracy criterion by default;
* the two criteria really do disagree on this fixture, so the parity above is not
  vacuous -- a fixture on which they agreed would prove nothing;
* orientation survives the change. The alternatives are ``lower_is_healthier`` (a long
  queue is bad); running them as ``higher_is_healthier`` calls every violating window
  healthy, and the fit must not do that.

Fixture shape (mirroring ``test_balanced_accuracy_fit`` on the other orientation): a
clean boundary at 300, every violating window above it, and twenty of the hundred healthy
windows forming a high tail that overlaps the violating range -- what a real load scan
looks like, and what makes the choice of criterion matter.
"""
from __future__ import annotations

import csv
import importlib.util
import sys
from pathlib import Path

import pytest
import yaml

from tre_calibration.alt_signals import (
    ALT_SIGNALS,
    alt_signal_direction,
    alt_signal_names,
    threshold_curve,
)
from tre_calibration.dataset import CalibrationWindow
from tre_calibration.fit import (
    DEFAULT_THETA_CRITERION,
    fit_theta,
    fit_theta_by_reliability,
    threshold_balanced_accuracy,
)
from tre_common.registry import EXPECTED_SIGNAL_DIRECTIONS

BOUNDARY = 300.0
#: The 0.20 healthy quantile measured from the unhealthy (high) end straddles the
#: boundary: 310 -> 290 interpolates to 294.
EXPECTED_BALANCED_ACCURACY_THETA = 294.0
#: The largest threshold whose cumulative lower set is still 90% healthy. It sits deep
#: inside the violating range and admits ten violating windows as healthy.
EXPECTED_RELIABILITY_THETA = 400.0

_RELIABILITY_KNOBS = {
    "reliability_target": 0.9,
    "min_support": 3,
    "min_confidence": 0.9,
}


def _signal_values() -> list[tuple[float, bool]]:
    rows: list[tuple[float, bool]] = []
    # 30 violating windows, all above the boundary.
    rows.extend((BOUNDARY + 10.0 * (i + 1), False) for i in range(30))
    # 20 healthy windows in the overlap region, 80 below the boundary.
    rows.extend((BOUNDARY + 10.0 * (i + 1), True) for i in range(20))
    rows.extend((BOUNDARY - 10.0 - 2.0 * i, True) for i in range(80))
    return rows


def _windows() -> list[CalibrationWindow]:
    return [
        CalibrationWindow(
            scenario_id=f"cell-{index}",
            scenario_family="steady" if index % 2 == 0 else "burst",
            signal=signal,
            slo_met=slo_met,
        )
        for index, (signal, slo_met) in enumerate(_signal_values())
    ]


def _mirrored(windows: list[CalibrationWindow]) -> list[CalibrationWindow]:
    return [
        CalibrationWindow(
            scenario_id=window.scenario_id,
            scenario_family=window.scenario_family,
            signal=-window.signal,
            slo_met=window.slo_met,
        )
        for window in windows
    ]


# --------------------------------------------------------------------------------
# The criterion itself, on a lower-is-healthier signal.
# --------------------------------------------------------------------------------


def test_balanced_accuracy_criterion_finds_the_boundary_of_a_lower_is_healthier_signal() -> None:
    fit = fit_theta(_windows(), direction="lower_is_healthier")

    assert fit.publish is True
    assert fit.direction == "lower_is_healthier"
    assert fit.healthy_quantile == 0.20
    assert fit.theta == pytest.approx(EXPECTED_BALANCED_ACCURACY_THETA)
    # Every violating window is above theta; 80 of 100 healthy windows below it.
    assert fit.specificity_bad == 1.0
    assert fit.recall_good == 0.80
    assert fit.balanced_accuracy == pytest.approx(0.90)


def test_the_two_criteria_disagree_on_this_fixture() -> None:
    """Without this the parity tests below would be vacuous."""
    windows = _windows()

    balanced = fit_theta(windows, criterion="balanced_accuracy", direction="lower_is_healthier")
    reliability = fit_theta(
        windows, criterion="reliability", direction="lower_is_healthier", **_RELIABILITY_KNOBS
    )

    assert reliability.publish is True
    assert reliability.theta == pytest.approx(EXPECTED_RELIABILITY_THETA)
    assert reliability.theta != pytest.approx(balanced.theta)
    # The containment rule is biased towards the unhealthy side under this orientation:
    # ten violating windows fall inside its "healthy" set, and it scores worse on the
    # very data it was fitted on.
    reliability_metrics = threshold_balanced_accuracy(
        windows, theta=reliability.theta, direction="lower_is_healthier"
    )
    assert reliability_metrics["specificity_bad"] < 1.0
    assert reliability_metrics["balanced_accuracy"] < balanced.balanced_accuracy


def test_orientation_is_not_assumed_the_tss_way() -> None:
    windows = _windows()

    healthy_low = fit_theta(windows, direction="lower_is_healthier")
    healthy_high = fit_theta(windows, direction="higher_is_healthier")

    assert healthy_high.theta != pytest.approx(healthy_low.theta)
    # Reading a queue-like signal as higher-is-healthier declares every violating
    # window healthy. That is the failure the direction argument exists to prevent.
    assert healthy_high.specificity_bad == 0.0
    assert healthy_low.specificity_bad == 1.0


def test_the_two_orientations_are_one_criterion_on_a_reflected_axis() -> None:
    windows = _windows()

    healthy_low = fit_theta(windows, direction="lower_is_healthier")
    healthy_high = fit_theta(_mirrored(windows), direction="higher_is_healthier")

    assert healthy_high.theta == pytest.approx(-healthy_low.theta)
    assert healthy_high.healthy_quantile == healthy_low.healthy_quantile
    assert healthy_high.balanced_accuracy == pytest.approx(healthy_low.balanced_accuracy)
    assert healthy_high.recall_good == pytest.approx(healthy_low.recall_good)
    assert healthy_high.specificity_bad == pytest.approx(healthy_low.specificity_bad)
    assert healthy_high.family_counts == healthy_low.family_counts


def test_threshold_balanced_accuracy_rejects_an_unknown_orientation() -> None:
    with pytest.raises(ValueError, match="direction"):
        threshold_balanced_accuracy(_windows(), theta=1.0, direction="sideways")


def test_fit_by_reliability_rejects_an_unknown_orientation() -> None:
    with pytest.raises(ValueError, match="direction"):
        fit_theta_by_reliability(
            [],
            direction="sideways",
            **_RELIABILITY_KNOBS,
            min_scenario_families=2,
            max_single_scenario_ratio=0.7,
        )


def test_fit_theta_rejects_an_unknown_criterion() -> None:
    with pytest.raises(ValueError, match="criterion"):
        fit_theta(_windows(), criterion="eyeball")


def test_threshold_curve_scores_the_criterion_that_selects_the_threshold() -> None:
    windows = _windows()
    rows = threshold_curve(windows, direction="lower_is_healthier")

    assert {row["theta"] for row in rows} == {window.signal for window in windows}
    for row in rows:
        expected = threshold_balanced_accuracy(
            windows, theta=row["theta"], direction="lower_is_healthier"
        )
        assert row["balanced_accuracy"] == pytest.approx(expected["balanced_accuracy"])
        assert row["recall_good"] == pytest.approx(expected["recall_good"])


# --------------------------------------------------------------------------------
# The driver: every alternative signal, end to end from a window CSV.
# --------------------------------------------------------------------------------


def _load_driver():
    path = Path(__file__).resolve().parents[1] / "scripts" / "fit_alt_thresholds.py"
    spec = importlib.util.spec_from_file_location("tre_fit_alt_thresholds", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


_MODEL = "dsqwen-7b"
_SLO = {"ttft_p95_ms": 500.0, "tpot_p95_ms": 75.0, "e2e_p95_ms": 12000.0}


def _write_registry(path: Path) -> Path:
    path.write_text(
        yaml.safe_dump(
            {
                "cluster": {"nodes": []},
                "models": [
                    {
                        "name": _MODEL,
                        "weights_path": "/weights",
                        "tp_size": 1,
                        "min_replicas": 1,
                        "max_replicas": 4,
                        "vllm_image": "vllm:test",
                        "slo": _SLO,
                        "trs": {
                            "w_p": 0.02,
                            "w_d": 1.0,
                            "lambda_wait": 2.625,
                            "qmin": 1.0,
                            "ema_alpha": 0.3,
                            "tau_crit": 0.8,
                            "tau_low": 1.0,
                            "tau_high": 1.25,
                            "qsat": 40.0,
                            "epsat": 0.05,
                            "hsat": 3,
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    return path


def _write_window_csv(path: Path) -> Path:
    """One CSV carrying the fixture on all three signals at once.

    Each row lasts exactly one second on one replica and puts the same number in the
    queue column and in both token counters, so ``queue_len``, ``decode_tps`` and
    ``prefill_tps`` all read the same fixture. Any difference between the three fits is
    then a difference in how the driver treats the signal, which is what is under test.
    """
    fields = [
        "scenario_id",
        "scenario_family",
        "window_start_ms",
        "window_end_ms",
        "assigned_replicas",
        "queue_control",
        "prompt_tokens_total",
        "generation_tokens_total",
        "p95_ttft",
        "p95_tpot",
        "p95_e2e",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for index, (signal, slo_met) in enumerate(_signal_values()):
            writer.writerow(
                {
                    "scenario_id": f"cell-{index}",
                    "scenario_family": "steady" if index % 2 == 0 else "burst",
                    "window_start_ms": 0,
                    "window_end_ms": 1000,
                    "assigned_replicas": 1,
                    "queue_control": signal,
                    "prompt_tokens_total": signal,
                    "generation_tokens_total": signal,
                    "p95_ttft": 400.0 if slo_met else 600.0,
                    "p95_tpot": 50.0,
                    "p95_e2e": 1000.0,
                }
            )
    return path


@pytest.mark.parametrize("signal", alt_signal_names())
def test_every_alternative_signal_is_fitted_by_the_balanced_accuracy_criterion(
    signal: str, tmp_path: Path
) -> None:
    driver = _load_driver()
    registry = _write_registry(tmp_path / "registry.yaml")
    csv_path = _write_window_csv(tmp_path / "windows.csv")

    payload, curve = driver.fit_model(
        _MODEL,
        csv_path,
        registry_path=str(registry),
        signal=signal,
        trim_ramp_windows=0,
    )

    assert payload["theta_criterion"] == DEFAULT_THETA_CRITERION == "balanced_accuracy"
    threshold = payload["alt_thresholds"][signal]
    assert threshold["direction"] == "lower_is_healthier"
    assert threshold["theta"] == pytest.approx(EXPECTED_BALANCED_ACCURACY_THETA)
    # The fit block is the balanced-accuracy one, not the containment one.
    assert payload["fit"]["balanced_accuracy"] == pytest.approx(0.90)
    assert payload["fit"]["healthy_quantile"] == 0.20
    assert payload["fit"]["specificity_bad"] == 1.0
    assert "attainment" not in payload["fit"]
    assert curve and "balanced_accuracy" in curve[0]


@pytest.mark.parametrize("signal", alt_signal_names())
def test_every_alternative_signal_moves_when_the_criterion_is_switched(
    signal: str, tmp_path: Path
) -> None:
    """The containment rule is still reachable -- and still gives a different answer."""
    driver = _load_driver()
    registry = _write_registry(tmp_path / "registry.yaml")
    csv_path = _write_window_csv(tmp_path / "windows.csv")

    payload, _curve = driver.fit_model(
        _MODEL,
        csv_path,
        registry_path=str(registry),
        signal=signal,
        trim_ramp_windows=0,
        criterion="reliability",
        **_RELIABILITY_KNOBS,
    )

    assert payload["theta_criterion"] == "reliability"
    theta = payload["alt_thresholds"][signal]["theta"]
    assert theta == pytest.approx(EXPECTED_RELIABILITY_THETA)
    assert theta != pytest.approx(EXPECTED_BALANCED_ACCURACY_THETA)


def test_the_driver_and_the_registry_agree_on_every_signals_orientation() -> None:
    assert set(alt_signal_names()) == set(EXPECTED_SIGNAL_DIRECTIONS)
    for signal in alt_signal_names():
        assert alt_signal_direction(signal) == EXPECTED_SIGNAL_DIRECTIONS[signal]
        assert ALT_SIGNALS[signal][1] == "lower_is_healthier"
