from __future__ import annotations

import math

import pytest

from golden.legacy_trs import (
    LegacyTRSComputer,
    LegacyTRSInput,
    legacy_compute_eta_m,
    legacy_compute_z_m,
)
from tre_common.metrics_schema import ModelWindowMetrics
from tre_common.registry import TrsParams
from tre_controller.signals.trs import (
    TRSComputer,
    TRSInput,
    compute_eta_m,
    compute_z_m,
)


def _assert_float_or_none_equal(actual: float | None, expected: float | None) -> None:
    if expected is None:
        assert actual is None
    elif math.isinf(expected):
        assert actual == expected
    else:
        assert actual == pytest.approx(expected)


def _assert_trs_result_equal(actual, expected) -> None:
    assert actual.Y_m == pytest.approx(expected.Y_m)
    assert actual.y_m == pytest.approx(expected.y_m)
    assert actual.Q == pytest.approx(expected.Q)
    assert actual.Q_ctl == pytest.approx(expected.Q_ctl)
    assert actual.TRS_raw == pytest.approx(expected.TRS_raw)
    assert actual.TRS == pytest.approx(expected.TRS)
    _assert_float_or_none_equal(actual.eta_m, expected.eta_m)
    _assert_float_or_none_equal(actual.Z_m, expected.Z_m)
    assert actual.ema_alpha == pytest.approx(expected.ema_alpha)
    _assert_float_or_none_equal(actual.prev_Y, expected.prev_Y)
    _assert_float_or_none_equal(actual.prev_Q_ctl, expected.prev_Q_ctl)


def test_trs_computer_matches_legacy_sequence() -> None:
    legacy = LegacyTRSComputer(ema_alpha=0.4)
    migrated = TRSComputer(ema_alpha=0.4)
    inputs = [
        LegacyTRSInput(
            prompt_tokens_total=900.0,
            generation_tokens_total=1800.0,
            avg_waiting=0.2,
            avg_running=1.0,
            avg_swapping=0.0,
            routable_pods=2,
            assigned_replicas=4,
            kv_cache_hit_rate=0.25,
        ),
        LegacyTRSInput(
            prompt_tokens_total=1200.0,
            generation_tokens_total=2200.0,
            avg_waiting=2.0,
            avg_running=2.0,
            avg_swapping=1.0,
            routable_pods=3,
            assigned_replicas=3,
            kv_cache_hit_rate=0.5,
        ),
        LegacyTRSInput(
            prompt_tokens_total=0.0,
            generation_tokens_total=0.0,
            avg_waiting=0.0,
            avg_running=0.0,
            avg_swapping=0.0,
            routable_pods=0,
            assigned_replicas=0,
            kv_cache_hit_rate=0.0,
        ),
    ]

    for item in inputs:
        expected = legacy.compute(item, theta_m=750.0)
        actual = migrated.compute(TRSInput(**item.__dict__), theta_m=750.0)
        _assert_trs_result_equal(actual, expected)
        assert migrated.snapshot() == pytest.approx(legacy.snapshot())


def test_trs_restore_matches_legacy_state() -> None:
    legacy = LegacyTRSComputer(ema_alpha=0.25)
    migrated = TRSComputer(ema_alpha=0.25)
    legacy.restore(ema=20.0, prev_Y=100.0, prev_Q_ctl=4.0)
    migrated.restore(ema=20.0, prev_Y=100.0, prev_Q_ctl=4.0)

    item = LegacyTRSInput(
        prompt_tokens_total=500.0,
        generation_tokens_total=100.0,
        avg_waiting=1.5,
        avg_running=0.5,
        avg_swapping=0.0,
        routable_pods=1,
        assigned_replicas=1,
    )

    _assert_trs_result_equal(
        migrated.compute(TRSInput(**item.__dict__), theta_m=25.0),
        legacy.compute(item, theta_m=25.0),
    )


# ADR-0014: SaturationGuard was removed (z_m threshold bands are the sole scaling
# trigger), so the former test_saturation_guard_matches_legacy_sequence parity test
# was deleted. The legacy reference class survives only in golden/legacy_trs.py.


def test_trs_input_can_be_built_from_model_metrics_and_registry_params() -> None:
    params = TrsParams(
        w_p=0.08,
        w_d=1.2,
        lambda_wait=3.0,
        qmin=2.0,
        ema_alpha=0.6,
        theta_m=42.0,
        tau_crit=0.7,
        tau_low=1.0,
        tau_high=1.3,
        qsat=5.0,
        epsat=0.01,
        hsat=4,
    )
    metrics = ModelWindowMetrics(
        model="m1",
        window_start_ms=0,
        window_end_ms=10_000,
        prompt_tokens=1000.0,
        generation_tokens=3000.0,
        avg_waiting=2.5,
        avg_running=1.5,
        avg_swapping=0.5,
        kv_cache_hit_rate=0.25,
        ttft_p95_ms=100.0,
        tpot_p95_ms=20.0,
        e2e_p95_ms=500.0,
        routable_pods=3,
        assigned_replicas=4,
        per_pod={},
    )

    inp = TRSInput.from_metrics(metrics, params)

    assert inp.prompt_tokens_total == 1000.0
    assert inp.generation_tokens_total == 3000.0
    assert inp.avg_waiting == 2.5
    assert inp.avg_running == 1.5
    assert inp.avg_swapping == 0.5
    assert inp.routable_pods == 3
    assert inp.assigned_replicas == 4
    assert inp.w_p == 0.08
    assert inp.w_d == 1.2
    assert inp.lambda_wait == 3.0
    assert inp.qmin == 2.0
    assert inp.kv_cache_hit_rate == 0.25


@pytest.mark.parametrize("trs,routable", [(0.0, 3), (float("inf"), 2), (float("nan"), 1)])
def test_eta_helper_matches_legacy_unavailable_values(trs: float, routable: int) -> None:
    assert compute_eta_m(trs, routable) == legacy_compute_eta_m(trs, routable)


@pytest.mark.parametrize("theta", [None, 0.0, -1.0])
def test_z_helper_matches_legacy_unavailable_theta(theta: float | None) -> None:
    assert compute_z_m(100.0, theta) == legacy_compute_z_m(100.0, theta)
