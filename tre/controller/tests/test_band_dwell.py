"""D8 band dwell: the offline helper (``tre_common.dwell``) used by the calibration tools.

The controller's band dwell (``TRE_DWELL_WINDOWS``, off since the v1/paper alignment A5)
was removed in the timer cleanup (2026-10-02); the last test checks it is gone.
"""
from __future__ import annotations

import pytest

from tre_common.dwell import DwellCounter, dwell_confirmed_series
from tre_common.metrics_schema import MetricsSnapshot, ModelWindowMetrics
from tre_common.registry import (
    ClusterTopology, ModelSpec, NodeSpec, Registry, SloSpec, TrsParams,
)
from tre_controller.loops.rescue_task import run_rescue_tick
from tre_controller.planning.classify import ModelState
from tre_controller.planning.planner import ScaleAction
from tre_controller.signals.trs import SignalState

E = 1_790_000_030_000
P = 10_000
THETA = 100.0


# --- DwellCounter ------------------------------------------------------------------------

def test_counter_needs_consecutive_new_windows() -> None:
    c = DwellCounter(required=2, max_gap_ms=30_000)
    assert c.update(E, True) is False
    assert c.update(E, True) is False  # re-read of the same window
    assert c.update(E - P, True) is False  # regressed window
    assert c.run == 1
    assert c.update(E + P, True) is True
    assert c.update(E + 2 * P, False) is False and c.run == 0


def test_counter_gap_and_eligibility_restart_the_run() -> None:
    c = DwellCounter(required=2, max_gap_ms=30_000)
    c.update(E, True)
    assert c.update(E + 30_001, True) is False and c.run == 1  # gap > one window
    c.update(E + 40_001, True, eligible=False)
    assert c.run == 0
    assert c.update(E + 50_001, True) is False


def test_required_one_is_dwell_off() -> None:
    assert DwellCounter(required=1).update(E, True) is True


def test_offline_series_matches_streaming_with_duplicate_rows() -> None:
    flags = [True, True, True, False, True, True, True]
    ends = [E, E, E + P, E + 2 * P, E + 3 * P, E + 3 * P, E + 4 * P]
    assert dwell_confirmed_series(flags, ends, required=2, max_gap_ms=30_000) == [
        False, False, True, False, False, False, True,
    ]
    with pytest.raises(ValueError):
        dwell_confirmed_series([True], [E, E + P])


# --- controller: rescue / fairness ticks ----------------------------------------------------

class _Queue:
    def __init__(self, cooldown: dict | None = None) -> None:
        self._cooldown = cooldown or {}

    def inflight_models(self) -> set[str]:
        return set()

    def submit(self, actions) -> object:
        return object()

    def last_actions(self):
        return self._cooldown


def _registry(scaling=None) -> Registry:
    slo = SloSpec(ttft_p95_ms=500.0, tpot_p95_ms=75.0, e2e_p95_ms=10_000.0)
    trs = TrsParams(
        w_p=0.0, w_d=1.0, lambda_wait=0.0, qmin=1.0, ema_alpha=0.5, theta_m=THETA,
        tau_crit=0.8, tau_low=1.0, tau_high=1.25, qsat=4.0, epsat=0.05, hsat=1,
        # tau << step: exp(-10 s / 1 ms) == 0.0, so the EMA is the raw TSS and Z is exactly
        # generation / running / theta (the dwell, not the smoothing, is under test)
        ema_tau_ms=1.0,
    )
    return Registry(
        ClusterTopology(nodes=(NodeSpec(name="node-a", gpus=4, two_gpu_slots=((0, 1), (2, 3))),)),
        [ModelSpec(name="m", weights_path="/w", tp_size=1, min_replicas=0, max_replicas=4,
                   vllm_image="img", slo=slo, trs=trs)],
        scaling=scaling,
    )


def _snap(end: int, z: float | None, *, idle: bool = False) -> MetricsSnapshot:
    tokens = None if z is None else (0.0 if idle else z * THETA)
    metrics = ModelWindowMetrics(
        model="m", window_start_ms=end - 30_000, window_end_ms=end,
        prompt_tokens=None if z is None else 0.0, generation_tokens=tokens,
        avg_waiting=0.0, avg_running=0.0 if idle else 1.0, avg_swapping=0.0, kv_cache_hit_rate=0.0,
        ttft_p95_ms=100.0, tpot_p95_ms=10.0, e2e_p95_ms=1000.0,
        routable_pods=1, assigned_replicas=1, per_pod={}, request_count=None if idle else 30.0,
    )
    return MetricsSnapshot(ts_ms=end, models={"m": metrics}, stale=False)


def _tick(state: SignalState, snap: MetricsSnapshot, *, loop=run_rescue_tick, queue=None):
    return loop(snap, queue=queue or _Queue(), registry=_registry(), signal_state=state,
                action_cooldown=queue is not None)


def _ups(result) -> list:
    return [a for a in result.actions if isinstance(a, ScaleAction) and a.delta > 0]


def test_the_controller_has_no_band_dwell_any_more() -> None:
    # Timer cleanup (2026-10-02): CRITICAL / HIGH act on the first window that shows them,
    # whatever TRE_DWELL_WINDOWS says (it is ignored).
    state = SignalState(warmup_ms=0)
    r = _tick(state, _snap(E, 0.3))
    assert r.classifications["m"].state == ModelState.CRITICAL and _ups(r)
    assert not any(e.startswith(("dwell_hold", "receiver_suppressed_dwell")) for e in r.events)
    assert _tick(SignalState(warmup_ms=0), _snap(E, 2.0)).classifications["m"].state == ModelState.HIGH
