"""F-onset warmup guard (architect-ruled): at load onset the sliding window is still
filling with traffic -> TRS structurally low -> false CRITICAL/LOW. Suppress receiver
scale-ups until the window lies fully inside the traffic period. ADR-0014 removed the
former saturation bypass, so warmup suppression is now unconditional (a genuine flash
crowd in the warmup window is delayed at most one window)."""

from __future__ import annotations

from tre_controller.planning.classify import ModelClassification, ModelRole, ModelState, TauThresholds
from tre_controller.planning.planner import PlanConfig, ScaleAction, build_plan
from tre_controller.signals.trs import SignalState


# --- SignalState.observe_traffic unit tests ---

def test_observe_traffic_not_warm_until_window_clears_onset() -> None:
    s = SignalState(warmup_ms=-1)  # auto
    # onset recorded at first traffic window_end=60000; window still straddles onset.
    assert s.observe_traffic("m", has_traffic=True, window_start_ms=30000, window_end_ms=60000) is False
    assert s.observe_traffic("m", has_traffic=True, window_start_ms=35000, window_end_ms=65000) is False
    # once window_start >= onset (window fully inside the traffic period) -> warm.
    assert s.observe_traffic("m", has_traffic=True, window_start_ms=60000, window_end_ms=90000) is True


def test_observe_traffic_idle_resets_onset() -> None:
    s = SignalState(warmup_ms=-1)
    s.observe_traffic("m", has_traffic=True, window_start_ms=30000, window_end_ms=60000)  # onset=60000
    assert s.observe_traffic("m", has_traffic=True, window_start_ms=60000, window_end_ms=90000) is True
    s.observe_traffic("m", has_traffic=False, window_start_ms=90000, window_end_ms=120000)  # idle -> reset
    # traffic resumes -> new onset -> not warm again.
    assert s.observe_traffic("m", has_traffic=True, window_start_ms=120000, window_end_ms=150000) is False


def test_observe_traffic_duplicate_window_idempotent() -> None:
    s = SignalState(warmup_ms=-1)
    a = s.observe_traffic("m", has_traffic=True, window_start_ms=30000, window_end_ms=60000)
    b = s.observe_traffic("m", has_traffic=True, window_start_ms=30000, window_end_ms=60000)  # 2nd loop, same window
    assert a is False and b is False
    assert s._onset_ms["m"] == 60000  # onset not advanced by the duplicate read


def test_observe_traffic_disabled_always_warm() -> None:
    s = SignalState(warmup_ms=0)  # ablation / pre-fix behaviour
    assert s.observe_traffic("m", has_traffic=True, window_start_ms=0, window_end_ms=30000) is True


def test_observe_traffic_explicit_span() -> None:
    s = SignalState(warmup_ms=20000)
    s.observe_traffic("m", has_traffic=True, window_start_ms=0, window_end_ms=30000)  # onset=30000
    assert s.observe_traffic("m", has_traffic=True, window_start_ms=10000, window_end_ms=45000) is False  # 15000 < 20000
    assert s.observe_traffic("m", has_traffic=True, window_start_ms=20000, window_end_ms=50000) is True  # 20000 >= 20000


# --- planner enforcement (unconditional warmup suppression, ADR-0014) ---

def _crit(model: str, z: float = 0.5) -> ModelClassification:
    return ModelClassification(
        model_name=model, state=ModelState.CRITICAL, role=ModelRole.RECEIVER,
        Z_m=z, eta_m=None, trs=0.0, theta_m=1.0, tau=TauThresholds.from_control(), donor_tier=None,
    )


def _ctx(*, warm: bool, assigned: int = 3, routable: int = 1) -> dict:
    return {"assigned_replicas": assigned, "routable_pods": routable, "signal_warm": warm}


def _cfg() -> PlanConfig:
    return PlanConfig(min_replicas_per_model=1, max_replicas_per_model=4)


def _upscales(plan) -> list[ScaleAction]:
    return [a for a in plan.actions if isinstance(a, ScaleAction) and a.delta > 0]


def test_build_plan_suppresses_unwarm_critical_receiver() -> None:
    plan = build_plan(
        model_contexts={"m": _ctx(warm=False)},
        classifications=[_crit("m")], model_replicas={"m": 3}, idle_gpus=0, cfg=_cfg(),
    )
    assert _upscales(plan) == []
    assert "receiver_suppressed_signal_warmup:m" in plan.events


def test_build_plan_warmup_suppression_has_no_saturation_bypass() -> None:
    # ADR-0014 (behaviour change): the former saturation bypass is gone. An unwarm
    # CRITICAL receiver is suppressed for this tick regardless of queue depth / any
    # would-be saturation -- the context no longer carries an is_saturated escape hatch.
    plan = build_plan(
        model_contexts={"m": _ctx(warm=False)},
        classifications=[_crit("m")], model_replicas={"m": 3}, idle_gpus=0, cfg=_cfg(),
    )
    assert _upscales(plan) == []
    assert "receiver_suppressed_signal_warmup:m" in plan.events


def test_build_plan_warm_critical_receiver_scales_first_tick() -> None:
    plan = build_plan(
        model_contexts={"m": _ctx(warm=True)},
        classifications=[_crit("m")], model_replicas={"m": 3}, idle_gpus=0, cfg=_cfg(),
    )
    assert any(a.model == "m" and a.delta > 0 for a in _upscales(plan))


def test_build_plan_default_context_warm_is_backcompat() -> None:
    # contexts WITHOUT signal_warm default to warm=True -> no suppression (golden-safe).
    plan = build_plan(
        model_contexts={"m": {"assigned_replicas": 3, "routable_pods": 1}},
        classifications=[_crit("m")], model_replicas={"m": 3}, idle_gpus=0, cfg=_cfg(),
    )
    assert any(a.model == "m" and a.delta > 0 for a in _upscales(plan))
    assert not any(e.startswith("receiver_suppressed_signal_warmup") for e in plan.events)
