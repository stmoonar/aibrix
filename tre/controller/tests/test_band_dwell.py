"""D8: band dwell in the controller, counted by distinct window_end_ms.

CRITICAL / LOW / HIGH act only after ``dwell_windows`` consecutive NEW metrics windows;
the rescue loop's repeated reads of one snapshot never advance the count. The offline
helper (``tre_common.dwell.dwell_confirmed_series``) gives the same verdicts.
"""
from __future__ import annotations

import pytest

from tre_common.dwell import DwellCounter, dwell_confirmed_series
from tre_common.metrics_schema import MetricsSnapshot, ModelWindowMetrics
from tre_common.registry import ClusterTopology, ModelSpec, NodeSpec, Registry, SloSpec, TrsParams
from tre_controller.loops.fairness_task import run_fairness_tick
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


def _registry() -> Registry:
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


def test_critical_needs_two_new_windows_and_rereads_do_not_count() -> None:
    state = SignalState(warmup_ms=0, dwell_windows=2)
    first = _snap(E, 0.3)
    r1 = _tick(state, first)
    assert "dwell_hold:m:critical:1/2" in r1.events
    assert "receiver_suppressed_dwell:m" in r1.events
    assert _ups(r1) == []
    # the 5 s rescue loop and the fairness loop re-read the same snapshot: no advance
    for loop in (run_rescue_tick, run_fairness_tick, run_rescue_tick):
        again = _tick(state, first, loop=loop)
        assert "dwell_hold:m:critical:1/2" in again.events and _ups(again) == []
    assert state.dwell_run("m", "critical") == 1
    r2 = _tick(state, _snap(E + P, 0.3))
    assert not any(e.startswith("dwell_hold") for e in r2.events)
    assert r2.classifications["m"].state == ModelState.CRITICAL
    assert _ups(r2)


def test_dwell_one_acts_on_the_first_window() -> None:
    state = SignalState(warmup_ms=0, dwell_windows=1)
    assert _ups(_tick(state, _snap(E, 0.3)))


def test_unconfirmed_critical_with_confirmed_receiver_band_is_low() -> None:
    state = SignalState(warmup_ms=0, dwell_windows=2)
    _tick(state, _snap(E, 0.9))  # LOW
    r = _tick(state, _snap(E + P, 0.3))  # CRITICAL, receiver band held for 2 windows
    assert r.classifications["m"].state == ModelState.LOW
    assert "dwell_hold:m:critical:1/2" in r.events
    assert "receiver_suppressed_dwell:m" not in r.events


def test_high_needs_dwell_and_is_healthy_meanwhile() -> None:
    state = SignalState(warmup_ms=0, dwell_windows=2)
    r1 = _tick(state, _snap(E, 2.0))
    assert r1.classifications["m"].state == ModelState.HEALTHY
    assert "dwell_hold:m:high:1/2" in r1.events
    r2 = _tick(state, _snap(E + P, 2.0))
    assert r2.classifications["m"].state == ModelState.HIGH


def test_dwell_states_select_the_gated_bands() -> None:
    state = SignalState(warmup_ms=0, dwell_windows=2, dwell_states=("critical", "low"))
    assert _tick(state, _snap(E, 2.0)).classifications["m"].state == ModelState.HIGH
    with pytest.raises(ValueError):
        SignalState(dwell_windows=2, dwell_states=("bogus",))


def test_warmup_windows_do_not_pre_confirm_a_critical() -> None:
    # auto warmup: onset at E; warm once window_start >= E, i.e. from E + 30 s.
    state = SignalState(warmup_ms=-1, dwell_windows=2)
    for k in range(3):
        r = _tick(state, _snap(E + k * P, 0.3))
        assert "receiver_suppressed_signal_warmup:m" in r.events and _ups(r) == []
    assert state.dwell_run("m", "critical") == 0
    r_warm = _tick(state, _snap(E + 3 * P, 0.3))
    assert "receiver_suppressed_dwell:m" in r_warm.events and _ups(r_warm) == []
    assert _ups(_tick(state, _snap(E + 4 * P, 0.3)))


def test_idle_window_resets_the_dwell() -> None:
    state = SignalState(warmup_ms=0, dwell_windows=2)
    _tick(state, _snap(E, 0.3))
    _tick(state, _snap(E + P, 0.0, idle=True))
    assert state.dwell_run("m", "critical") == 0
    r = _tick(state, _snap(E + 2 * P, 0.3))
    assert "dwell_hold:m:critical:1/2" in r.events


def test_gap_longer_than_a_window_restarts_the_dwell() -> None:
    state = SignalState(warmup_ms=0, dwell_windows=2)
    _tick(state, _snap(E, 0.3))
    r = _tick(state, _snap(E + 40_000, 0.3))
    assert "dwell_hold:m:critical:1/2" in r.events


def test_missing_tokens_neither_count_nor_reset() -> None:
    state = SignalState(warmup_ms=0, dwell_windows=2)
    _tick(state, _snap(E, 0.3))
    _tick(state, _snap(E + P, None))
    assert state.dwell_run("m", "critical") == 1
    assert _ups(_tick(state, _snap(E + 2 * P, 0.3)))


def test_dwell_keeps_counting_under_cooldown() -> None:
    state = SignalState(warmup_ms=0, dwell_windows=2)
    queue = _Queue(cooldown={"m": (E + 10 * P, "up")})  # last scale-up not yet reflected
    _tick(state, _snap(E, 0.3), queue=queue)
    r = _tick(state, _snap(E + P, 0.3), queue=queue)
    assert state.dwell_run("m", "critical") == 2
    assert "cooldown_hold:m" in r.events and _ups(r) == []
    assert _ups(_tick(state, _snap(E + 2 * P, 0.3)))  # cooldown over -> acts at once


def test_online_dwell_equals_offline_series() -> None:
    zs = [0.3, 0.3, 0.9, 0.3, 0.3, 0.3, 1.1, 0.3]
    ends = [E + k * P for k in range(len(zs))]
    state = SignalState(warmup_ms=0, dwell_windows=2)
    online = []
    for z, end in zip(zs, ends):
        snap = _snap(end, z)
        _tick(state, snap)  # rescue
        r = _tick(state, snap, loop=run_fairness_tick)  # fairness re-read
        online.append(r.classifications["m"].state == ModelState.CRITICAL and "receiver_suppressed_dwell:m" not in r.events)
    offline = dwell_confirmed_series([z < 0.8 for z in zs], ends, required=2, max_gap_ms=30_000)
    assert online == offline
