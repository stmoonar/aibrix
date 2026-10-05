"""Timer cleanup item 2 (2026-10-02, docs/design/20261002-timer-cleanup.md): a model
whose SafeScale probe rolled back is re-probed on new evidence, not after a fixed 60 s.

A capacity rollback (SLO violation, formal commit gate, donor health) keeps the Z and
routable count of the decision that started the probe; the next receiver-less HIGH
probe needs a metrics window ending after the rollback AND a different routable count
or a Z at least ``rollback_retry_z_margin`` higher. Other rollbacks need the new
window only. ``TRE_SAFESCALE_ROLLBACK_BACKOFF_MS`` still parses and is ignored."""

from __future__ import annotations

from dataclasses import replace

import pytest

from tre_common.metrics_schema import MetricsSnapshot
from tre_common.registry import SafeScaleRegistryConfig, parse_safescale_config
from tre_controller.config import ControllerConfig, SafeScaleConfig
from tre_controller.loops.rescue_task import run_rescue_tick
from tre_controller.planning.planner import HideAction
from tre_controller.planning.safescale import (
    CAPACITY_ROLLBACK_CODES,
    ProbeWindowInputs,
    SafeScaleStateMachine,
    _probe_from_record,
    _probe_record,
)

from test_loop_ticks import FakeQueue, _metrics_with_pods, _registry_with_models


def _machine(**config) -> SafeScaleStateMachine:
    return SafeScaleStateMachine(config=SafeScaleConfig(**config))


def _rolled_back(machine, *, reason="slo_violation", z=1.5, routable=3, now_ms=10_000, model="donor"):
    machine.start_probe(model=model, pods=(f"{model}-a",), now_ms=0,
                        window_inputs=ProbeWindowInputs(z_m=z, routable_pods=routable))
    machine.resolve(model, status="rollback", reason=reason, now_ms=now_ms)
    return machine


def test_capacity_rollback_waits_for_a_higher_z_or_another_replica_count():
    machine = _rolled_back(_machine())
    evidence = machine.rollback_evidence()["donor"]
    assert (evidence.capacity, evidence.z_m, evidence.routable, evidence.reason) == (True, 1.5, 3, "slo_violation")
    assert machine.rollback_retry_holds({}) == {"donor": "no_signal"}
    assert machine.rollback_retry_holds({"donor": (3.0, 4, 10_000)}) == {"donor": "no_new_window"}
    for z in (1.5, 1.6, 1.74):  # below start + margin: the same evidence, whatever the time
        assert machine.rollback_retry_holds({"donor": (z, 3, 3_600_000)}) == {"donor": "same_evidence"}
    assert machine.rollback_retry_holds({"donor": (1.75, 3, 20_000)}) == {}
    assert machine.rollback_retry_holds({"donor": (1.0, 2, 20_000)}) == {}  # n changed
    assert machine.rollback_retry_holds({"donor": (None, 3, 20_000)}) == {"donor": "same_evidence"}


def test_the_margin_is_configurable():
    machine = _rolled_back(_machine(rollback_retry_z_margin=0.0))
    assert machine.rollback_retry_holds({"donor": (1.5, 3, 20_000)}) == {}
    machine = _rolled_back(_machine(rollback_retry_z_margin=1.0))
    assert machine.rollback_retry_holds({"donor": (2.4, 3, 20_000)}) == {"donor": "same_evidence"}
    assert machine.rollback_retry_holds({"donor": (2.5, 3, 20_000)}) == {}


@pytest.mark.parametrize("code", ["evidence_incomplete:no_baseline", "sm_maintenance", "observe_entered",
                                  "hide_failed", "probe_pods_gone"])
def test_non_capacity_rollback_needs_only_a_new_window(code):
    machine = _rolled_back(_machine(), reason=code)
    assert code not in CAPACITY_ROLLBACK_CODES
    assert machine.rollback_evidence()["donor"].capacity is False
    assert machine.rollback_retry_holds({"donor": (1.5, 3, 10_000)}) == {"donor": "no_new_window"}
    assert machine.rollback_retry_holds({"donor": (1.5, 3, 10_001)}) == {}


def test_structured_rollback_code_wins_over_the_queue_reason():
    machine = _machine()
    machine.start_probe(model="donor", pods=("donor-a",), now_ms=0,
                        window_inputs=ProbeWindowInputs(z_m=1.5, routable_pods=3))
    probe = machine.active_probe("donor")
    machine._probes["donor"] = replace(probe, terminal_details={"rollback_reason": {"code": "donor_health"}})
    machine.resolve("donor", status="rollback", reason="unhide_done", now_ms=1_000)
    assert machine.rollback_evidence()["donor"].reason == "donor_health"
    assert machine.rollback_evidence()["donor"].capacity is True


def test_unknown_starting_z_takes_the_first_window_after_the_rollback():
    machine = _machine()
    machine.start_probe(model="donor", pods=("donor-a",), now_ms=0)  # no window inputs
    machine.resolve("donor", status="rollback", reason="formal_commit_gate_failed", now_ms=10_000)
    assert machine.rollback_retry_holds({"donor": (1.4, 3, 20_000)}) == {"donor": "same_evidence"}
    assert machine.rollback_evidence()["donor"].z_m == 1.4
    assert machine.rollback_retry_holds({"donor": (1.6, 3, 30_000)}) == {"donor": "same_evidence"}
    assert machine.rollback_retry_holds({"donor": (1.65, 3, 30_000)}) == {}


def test_commit_clears_and_preemption_never_records_evidence():
    machine = _rolled_back(_machine())
    machine.start_probe(model="donor", pods=("donor-a",), now_ms=20_000)
    machine.resolve("donor", status="commit", reason="formal_commit_gate_passed", now_ms=30_000)
    assert machine.rollback_evidence() == {}
    machine.start_probe(model="donor", pods=("donor-a",), now_ms=40_000)
    machine.request_preemption("donor")
    machine.resolve("donor", status="rollback", reason="receiver_need_upscale", now_ms=41_000)
    assert machine.rollback_evidence() == {}


def test_probe_record_keeps_the_starting_evidence():
    machine = _machine()
    machine.start_probe(model="donor", pods=("donor-a",), now_ms=0,
                        window_inputs=ProbeWindowInputs(z_m=1.5, routable_pods=3))
    record = _probe_record(machine.active_probe("donor"), terminal_reason=None)
    assert (record["start_z_m"], record["start_routable"]) == (1.5, 3)

    class _Store:
        def load_probe_journal(self, _request_id):
            return []

    restored = _probe_from_record(record, _Store())
    assert (restored.start_z_m, restored.start_routable) == (1.5, 3)
    bare = _machine()
    bare.start_probe(model="donor", pods=("donor-a",), now_ms=0)
    assert "start_z_m" not in _probe_record(bare.active_probe("donor"), terminal_reason=None)


def test_rescue_tick_re_probes_once_the_replica_count_changed():
    registry = _registry_with_models("hot")
    two = _metrics_with_pods("hot", generation=1000.0, waiting=0.0, running=1.0, pods=("hot-a", "hot-b"))
    snapshot = MetricsSnapshot(ts_ms=100_000, stale=False, models={"hot": two})
    safescale = _machine(min_window_ms=60_000.0)
    probed = run_rescue_tick(snapshot, queue=FakeQueue(), registry=registry, safescale=safescale)
    assert any(isinstance(a, HideAction) for a in probed.actions)
    assert safescale.active_probe("hot").start_routable == 2
    safescale.resolve("hot", status="rollback", reason="slo_violation", now_ms=90_000)

    later = {"window_start_ms": 100_000, "window_end_ms": 160_000}
    same = MetricsSnapshot(ts_ms=200_000, stale=False, models={"hot": replace(two, **later)})
    held = run_rescue_tick(same, queue=FakeQueue(), registry=registry, safescale=safescale)
    assert not any(isinstance(a, HideAction) for a in held.actions)
    assert "safescale_rollback_hold:hot:same_evidence" in held.events

    three = _metrics_with_pods("hot", generation=1500.0, waiting=0.0, running=1.0, pods=("hot-a", "hot-b", "hot-c"))
    grown = MetricsSnapshot(ts_ms=200_000, stale=False, models={"hot": replace(three, **later)})
    freed = run_rescue_tick(grown, queue=FakeQueue(), registry=registry, safescale=safescale)
    assert any(isinstance(a, HideAction) for a in freed.actions), freed.events
    assert not any(event.startswith("safescale_rollback_hold") for event in freed.events)


def test_a_rollback_is_held_by_evidence_whatever_the_elapsed_time():
    assert ControllerConfig.from_env({}).safescale.rollback_retry_z_margin == 0.25
    machine = _rolled_back(_machine())
    assert machine.rollback_retry_holds({"donor": (1.5, 3, 10_000_000)}) == {"donor": "same_evidence"}

def test_registry_margin_key():
    assert SafeScaleRegistryConfig().rollback_retry_z_margin == 0.25
    assert parse_safescale_config({"rollback_retry_z_margin": 0.5}).rollback_retry_z_margin == 0.5
    for bad in (-0.1, "x", True, float("nan")):
        with pytest.raises(ValueError):
            parse_safescale_config({"rollback_retry_z_margin": bad})
