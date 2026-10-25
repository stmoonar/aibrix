"""TRE_SAFESCALE_SCOPE (2026-10-26, user decision): ``receiverless_shrink`` keeps
SafeScale only for the HIGH proactive shrink that has no receiver (without SafeScale
those releases oscillated: e1 Real_*, 47/47 had no receiver and scaled back up); a
donor released for a receiver goes immediately, as with SafeScale off.
``TRE_ABLATION_DISABLE_SAFESCALE=true`` stays the alias of ``off``."""

from __future__ import annotations

from dataclasses import replace

import pytest

from tre_common.metrics_schema import MetricsSnapshot
from tre_controller.config import ControllerConfig, SafeScaleConfig
from tre_controller.loops.rescue_task import run_rescue_tick
from tre_controller.planning.planner import TransferIntent
from tre_controller.planning.safescale import SafeScaleStateMachine

from test_b8_observe_probes import _probe_snapshot
from test_loop_ticks import FakeQueue, _metrics_with_pods, _registry_with_models

SCOPE = "receiverless_shrink"


def test_only_the_receiverless_high_shrink_runs_as_a_probe():
    # A HIGH model with no receiver: the proactive shrink starts a probe.
    machine = SafeScaleStateMachine(config=SafeScaleConfig())
    alone = MetricsSnapshot(ts_ms=1, stale=False, models={
        "high": _metrics_with_pods("high", generation=400.0, waiting=0.0, running=1.0, pods=("h-a", "h-b")),
    })
    result = run_rescue_tick(alone, queue=FakeQueue(), registry=_registry_with_models("high"),
                             safescale=machine, safescale_scope=SCOPE)
    assert result.classifications["high"].state.value == "high"
    assert machine.active_probe("high") is not None

    # A HIGH donor for a CRITICAL receiver: an immediate relay, no probe.
    machine = SafeScaleStateMachine(config=SafeScaleConfig())
    snapshot = _probe_snapshot()
    donor = _metrics_with_pods("donor", generation=200.0, waiting=0.0, running=1.0, pods=("donor-a", "donor-b"))
    snapshot = replace(snapshot, models={**snapshot.models, "donor": donor})
    queue = FakeQueue()
    run_rescue_tick(snapshot, queue=queue, registry=_registry_with_models("critical", "donor"),
                    safescale=machine, safescale_scope=SCOPE)
    assert machine.active_probe("donor") is None
    assert queue.submitted == [
        (TransferIntent("donor", "critical", 1, "critical_high_donor_safescale_nosafescale", "rescue"),)
    ]


@pytest.mark.parametrize(("env", "scope"), [
    ({}, "all"),
    ({"TRE_ABLATION_DISABLE_SAFESCALE": "true"}, "off"),
    ({"TRE_SAFESCALE_SCOPE": SCOPE, "TRE_ABLATION_DISABLE_SAFESCALE": "false"}, SCOPE),
    ({"TRE_SAFESCALE_SCOPE": "off", "TRE_ABLATION_DISABLE_SAFESCALE": "true"}, "off"),
    ({"TRE_SAFESCALE_SCOPE": SCOPE, "TRE_ABLATION_DISABLE_SAFESCALE": "true"}, None),
    ({"TRE_SAFESCALE_SCOPE": "off", "TRE_ABLATION_DISABLE_SAFESCALE": "false"}, None),
    ({"TRE_SAFESCALE_SCOPE": "some"}, None),
])
def test_the_scope_and_its_alias_must_agree(env, scope):
    if scope is None:
        with pytest.raises(ValueError):
            ControllerConfig.from_env(env)
        return
    config = ControllerConfig.from_env(env)
    assert (config.safescale_scope, config.ablation_disable_safescale) == (scope, scope == "off")
