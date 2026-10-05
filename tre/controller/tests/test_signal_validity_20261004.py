"""Signal validity (2026-10-04, I3 / I4).

I3: every observation carries its own validity; data older than the window is
UNKNOWN - not 0 and not the last value. I4: a model is a donor (scale-down or
release) only on a level computed from tokens observed in the current window."""

from __future__ import annotations

import pytest

from tre_controller.loops.tick import PaperStateCache
from tre_controller.planning.classify import classify_all_models
from tre_controller.planning.planner import PlanConfig, ScaleAction, build_plan


def _ctx(z_m: float | None, y_m: float, q: float) -> dict:
    return {
        "trs": 100.0 * (z_m or 0.0), "z_m": z_m, "trs_z_m": z_m, "theta_m": 100.0, "eta_m": 500.0,
        "tss_defined": True, "Y_m": y_m, "y_m": y_m, "Q": q, "Q_ctl": max(q, 1.0),
        "signal_source": "zm", "signal_unavailable_reason": None, "request_rate_rps": 5.0 if y_m else 0.0,
        "routable_pods": 3, "assigned_replicas": 3, "awake_replicas": 3,
        "signal_warm": True, "signal_full_window": True, "signal_hold_reason": None,
    }


def _missing_ctx() -> dict:
    """The tokens-missing context of a window without token data (tick._model_contexts)."""
    return {
        "trs": 0.0, "z_m": None, "signal_source": "zm", "signal_unavailable_reason": "tokens_missing",
        "Y_m": None, "y_m": None, "Q": 0.0, "Q_ctl": 1.0, "theta_m": 100.0,
        "routable_pods": 3, "assigned_replicas": 3, "awake_replicas": 3, "request_rate_rps": None,
    }


def _plan(donor_ctx: dict):
    receiver = _ctx(0.1, 500.0, 50.0)  # CRITICAL, warm, whole window
    contexts = {"r": receiver, "d": donor_ctx}
    return build_plan(
        model_contexts=contexts,
        classifications=classify_all_models(contexts),
        model_replicas={"r": 1, "d": 3},
        idle_gpus=0,
        cfg=PlanConfig(min_replicas_per_model=1, max_replicas_per_model=8),
    )


def _gives(plan, model: str) -> bool:
    released = any(
        isinstance(a, ScaleAction) and a.model == model and a.delta < 0 for a in plan.actions
    )
    return released or model in plan.probe_upscale_plans


@pytest.mark.parametrize(
    "fresh",
    [
        _ctx(3.0, 900.0, 3.0),  # HIGH: immediate donor release
        _ctx(1.1, 600.0, 5.0),  # HEALTHY: middle-zone SafeScale scale-down
        _ctx(0.0, 0.0, 0.0),  # IDLE: immediate donor release
    ],
    ids=["high", "middle_zone", "idle"],
)
def test_held_context_is_never_donor_evidence(fresh):
    """D: a window without tokens holds the model's last context (PaperStateCache); the
    held level is not current-window evidence, so the model gives nothing up."""
    assert _gives(_plan(dict(fresh)), "d")  # control: the same level from a fresh window donates

    cache = PaperStateCache(max_stale_windows=3)
    cache.apply("d", dict(fresh), tokens_available=True)
    held, events = cache.apply("d", _missing_ctx(), tokens_available=False)
    assert events == ("paper_state_stale_hold:d",)

    plan = _plan(held)
    assert not _gives(plan, "d")
    assert "donor_suppressed_breakpoint_window:d" in plan.events

