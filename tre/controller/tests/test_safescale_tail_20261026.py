"""Round 5 review (H1, L1; docs/design/20261025-instant-sampling.md "Evidence, not
ticks"): the commit gate's tail and the immediate SLO rollback both judge ONE value over
the probe's post-hide span ``[S, E]`` of the newest snapshot (``tail_*`` of the
observation), and both need ``min_commit_samples`` requests in it.

* H1: too few requests in a tail shorter than one window is insufficient evidence -
  the gate extends (to the W ceiling), it does not fail on a noisy Z;
* L1: the immediate rollback judges the post-hide span on every new snapshot (also one
  whose window starts before the hide or overlaps one already judged), once the
  request floor is met.
"""
from __future__ import annotations

from tre_controller.config import SafeScaleConfig
from tre_controller.planning.safescale import ProbeObservation, SafeScaleStateMachine

MODEL = "d"
HIDE = 100_000  # no evidence source: the planned start is the post-hide start


def _machine() -> SafeScaleStateMachine:
    machine = SafeScaleStateMachine(config=SafeScaleConfig(
        ttft_p95_slo_ms=500.0, tpot_p95_slo_ms=75.0, min_window_ms=20_000.0, window_ceiling_ms=60_000.0,
        tau_low=1.0, min_commit_samples=20,
    ))
    machine.start_probe(model=MODEL, pods=("d-1",), now_ms=HIDE)
    return machine


def _obs(end: int, *, requests: float, z: float = 2.0, ttft: float = 100.0) -> ProbeObservation:
    """A snapshot whose 30 s window starts before the hide; its post-hide tail starts
    at the hide (shorter than the window) and carries the values judged."""
    return ProbeObservation(
        ts_ms=end, window_start_ms=end - 30_000, window_end_ms=end,
        z_m=2.0, has_traffic=True, ttft_p95_ms=100.0, tpot_p95_ms=10.0,  # the whole window: fine
        tail_start_ms=HIDE, tail_observed_ms=end - HIDE, tail_requests=requests,
        tail_z_m=z, tail_has_traffic=True, tail_ttft_p95_ms=ttft, tail_tpot_p95_ms=10.0,
    )


def test_too_few_tail_requests_extend_the_probe_instead_of_failing_the_z_gate():
    machine = _machine()
    assert machine.observe(MODEL, _obs(110_000, requests=5.0, z=0.4), now_ms=110_000).status == "probing"
    # The deadline (120 s): Z 0.4 < tau_low on 8 requests - not judged, extended.
    decision = machine.observe(MODEL, _obs(120_000, requests=8.0, z=0.4), now_ms=120_000)
    assert (decision.status, decision.reason) == ("probing", "evidence_extended")
    assert decision.details["extend_reason"] == "tail_low_evidence"
    assert decision.details["tail_requests"] == 8.0 and decision.details["tail_evidence_ok"] is False
    # 25 requests: the same Z is evidence now - the gate fails.
    decision = machine.observe(MODEL, _obs(130_000, requests=25.0, z=0.4), now_ms=130_000)
    assert (decision.status, decision.reason) == ("rollback", "formal_commit_gate_failed")
    assert decision.details["rollback_reason"]["gates"] == ["z_below_tau_low"]


def test_the_immediate_rollback_judges_the_post_hide_span_once_it_has_enough_requests():
    machine = _machine()
    # Slow in the post-hide span, but 5 requests: not judged.
    assert machine.observe(MODEL, _obs(104_000, requests=5.0, ttft=5_000.0), now_ms=104_000).status == "probing"
    # The next snapshot overlaps it and its window starts before the hide (round 4
    # judged neither): 25 requests in [S, E] -> rolled back now.
    decision = machine.observe(MODEL, _obs(108_000, requests=25.0, ttft=5_000.0), now_ms=108_000)
    assert (decision.status, decision.reason) == ("rollback", "slo_violation")
    reason = decision.details["rollback_reason"]
    assert (reason["metrics"], reason["window_start_ms"], reason["evidence_start_ms"]) == (["ttft"], HIDE, HIDE)
