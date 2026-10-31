"""2026-11-01: a receiver-less SafeScale probe (a HIGH model's proactive shrink) commits
only if the model is still HIGH after it - tail Z at or above the tau_high it was
classified with. It has no beneficiary: leaving HIGH would trade latency for a GPU
nobody asked for. A probe for a receiver keeps the tau_low floor."""
from __future__ import annotations

import pytest

from relay_view import fleet_view
from section_ticks import run_rescue_tick
from test_b8_observe_probes import _probe_snapshot
from test_loop_ticks import FakeQueue, _metrics_with_pods, _registry_with_models

from tre_common.metrics_schema import MetricsSnapshot
from tre_controller.config import SafeScaleConfig
from tre_controller.planning.safescale import ProbeObservation, SafeScaleStateMachine

TAU_HIGH = 1.25  # the test registry's trs.tau_high (tau_low 1.0)


def _tail(probe, z: float) -> ProbeObservation:
    end = probe.deadline_ms
    return ProbeObservation(
        ts_ms=end, window_start_ms=end - 30_000, window_end_ms=end,
        z_m=z, has_traffic=True, ttft_p95_ms=100.0, tpot_p95_ms=10.0,
        tail_start_ms=probe.start_ms, tail_observed_ms=end - probe.start_ms, tail_requests=25.0,
        tail_z_m=z, tail_has_traffic=True, tail_ttft_p95_ms=100.0, tail_tpot_p95_ms=10.0,
    )


@pytest.mark.parametrize(
    ("receiver", "z", "status", "gates"),
    [
        (False, 1.1, "rollback", ["z_below_tau_high"]),  # tau_low <= Z < tau_high: left HIGH
        (False, TAU_HIGH, "commit", None),  # still HIGH
        (True, 1.1, "commit", None),  # for a receiver: the tau_low floor, unchanged
    ],
)
def test_a_receiverless_shrink_commits_only_if_the_model_stays_high(receiver, z, status, gates):
    machine = SafeScaleStateMachine(config=SafeScaleConfig())
    if receiver:
        model, snapshot = "donor", _probe_snapshot()
        registry = _registry_with_models("critical", "donor")
        view = fleet_view(registry, "donor", ("donor-a", "donor-b"), "critical")
    else:
        model, registry, view = "high", _registry_with_models("high"), None
        snapshot = MetricsSnapshot(ts_ms=1, stale=False, models={
            "high": _metrics_with_pods("high", generation=400.0, waiting=0.0, running=1.0, pods=("h-a", "h-b")),
        })
    run_rescue_tick(snapshot, queue=FakeQueue(), registry=registry, safescale=machine, cluster_view=view)
    probe = machine.active_probe(model)
    assert probe is not None and bool(probe.pending_upscales) is receiver

    decision = machine.observe(model, _tail(probe, z), now_ms=probe.deadline_ms)
    assert decision.status == status
    if gates is not None:
        assert decision.details["rollback_reason"]["gates"] == gates
