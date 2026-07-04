from __future__ import annotations

from tre_controller.config import SafeScaleConfig
from tre_controller.planning.safescale import (
    ProbeObservation,
    SafeScaleCommand,
    SafeScaleDecision,
    SafeScaleStateMachine,
)


class FakeProbeStore:
    def __init__(self, unresolved: list[dict] | None = None, journal: dict[str, list[dict]] | None = None) -> None:
        self.records: dict[str, dict] = {}
        self.deleted: list[str] = []
        self.unresolved = unresolved or []
        self.journal = journal or {}

    def save_probe(self, request_id: str, record: dict) -> None:
        self.records[request_id] = dict(record)

    def delete_probe(self, request_id: str) -> None:
        self.deleted.append(request_id)
        self.records.pop(request_id, None)

    def list_unresolved_probes(self) -> list[dict]:
        return [dict(item) for item in self.unresolved]

    def append_probe_journal(self, request_id: str, record: dict) -> None:
        self.journal.setdefault(request_id, []).append(dict(record))

    def load_probe_journal(self, request_id: str) -> list[dict]:
        return [dict(item) for item in self.journal.get(request_id, [])]


def _cfg() -> SafeScaleConfig:
    return SafeScaleConfig(
        ttft_p95_slo_ms=1000.0,
        tpot_p95_slo_ms=100.0,
        default_window_ms=60_000.0,
        min_window_ms=15_000.0,
        max_window_ms=300_000.0,
        hq=0.5,
        tau_low=1.0,
    )


def _healthy_observation(ts_ms: int = 61_000) -> ProbeObservation:
    return ProbeObservation(
        ts_ms=ts_ms,
        ttft_p95_ms=500.0,
        tpot_p95_ms=50.0,
        z_m=1.2,
        q_ctl=0.0,
        has_traffic=True,
        avg_gpu_cache_norm=0.5,
    )


def test_safescale_starts_probe_and_persists_hidden_pods() -> None:
    store = FakeProbeStore()
    machine = SafeScaleStateMachine(config=_cfg(), store=store)

    decision = machine.start_probe(
        model="donor",
        pods=("pod-a", "pod-b"),
        now_ms=1_000,
        pending_upscales={"receiver": 2},
    )

    assert decision.status == "probing"
    assert decision.reason == "probe_started"
    assert decision.commands == (
        SafeScaleCommand(kind="hide", model="donor", pods=("pod-a", "pod-b"), reason="probe_started"),
    )
    probe = machine.active_probe("donor")
    assert probe is not None
    assert probe.deadline_ms == 61_000
    assert probe.request_id in store.records
    assert store.records[probe.request_id]["pending_upscales"] == {"receiver": 2}


def test_safescale_rolls_back_immediately_on_slo_violation() -> None:
    store = FakeProbeStore()
    machine = SafeScaleStateMachine(config=_cfg(), store=store)
    machine.start_probe(model="donor", pods=("pod-a",), now_ms=1_000)

    decision = machine.observe(
        "donor",
        ProbeObservation(
            ts_ms=2_000,
            ttft_p95_ms=1200.0,
            tpot_p95_ms=50.0,
            z_m=1.5,
            q_ctl=0.0,
            has_traffic=True,
        ),
        now_ms=2_000,
    )

    assert decision == SafeScaleDecision(
        status="rollback",
        reason="slo_violation",
        commands=(SafeScaleCommand(kind="unhide", model="donor", pods=("pod-a",), reason="slo_violation"),),
    )
    assert machine.active_probe("donor") is None
    assert store.deleted


def test_safescale_commits_after_deadline_when_tail_is_healthy() -> None:
    store = FakeProbeStore()
    machine = SafeScaleStateMachine(config=_cfg(), store=store)
    machine.start_probe(model="donor", pods=("pod-a",), now_ms=1_000, pending_upscales={"receiver": 1})
    assert machine.observe("donor", _healthy_observation(ts_ms=20_000), now_ms=20_000).status == "probing"

    decision = machine.observe("donor", _healthy_observation(ts_ms=61_000), now_ms=61_000)

    assert decision.status == "commit"
    assert decision.reason == "formal_commit_gate_passed"
    assert decision.commands == (
        SafeScaleCommand(kind="scale_down", model="donor", delta=-1, reason="formal_commit_gate_passed"),
        SafeScaleCommand(kind="scale_up", model="receiver", delta=1, reason="safescale_followup_upscale"),
    )
    assert machine.active_probe("donor") is None
    assert store.deleted


def test_safescale_rolls_back_at_deadline_when_tail_health_fails() -> None:
    machine = SafeScaleStateMachine(config=_cfg(), store=FakeProbeStore())
    machine.start_probe(model="donor", pods=("pod-a",), now_ms=1_000)

    decision = machine.observe(
        "donor",
        ProbeObservation(
            ts_ms=61_000,
            ttft_p95_ms=500.0,
            tpot_p95_ms=50.0,
            z_m=0.8,
            q_ctl=0.0,
            has_traffic=True,
        ),
        now_ms=61_000,
    )

    assert decision.status == "rollback"
    assert decision.reason == "formal_commit_gate_failed"
    assert decision.commands == (
        SafeScaleCommand(kind="unhide", model="donor", pods=("pod-a",), reason="formal_commit_gate_failed"),
    )


def test_safescale_restores_unresolved_probe_and_commits() -> None:
    unresolved = [
        {
            "model": "donor",
            "request_id": "probe-1",
            "pods": ["pod-a"],
            "start_ms": 1_000,
            "deadline_ms": 61_000,
            "status": "probing",
            "pending_upscales": {"receiver": 1},
        }
    ]
    journal = {
        "probe-1": [
            {
                "last_observation": {
                    "ts_ms": 20_000,
                    "ttft_p95_ms": 500.0,
                    "tpot_p95_ms": 50.0,
                    "z_m": 1.2,
                    "q_ctl": 0.0,
                    "has_traffic": True,
                }
            }
        ]
    }
    store = FakeProbeStore(unresolved=unresolved, journal=journal)
    machine = SafeScaleStateMachine(config=_cfg(), store=store)

    assert machine.restore() == 1
    decision = machine.observe("donor", _healthy_observation(ts_ms=61_000), now_ms=61_000)

    assert decision.status == "commit"
    assert decision.commands[0] == SafeScaleCommand(
        kind="scale_down",
        model="donor",
        delta=-1,
        reason="formal_commit_gate_passed",
    )


def test_safescale_restored_tail_blocks_commit_on_prior_latency_violation() -> None:
    unresolved = [
        {
            "model": "donor",
            "request_id": "probe-violation",
            "pods": ["pod-a"],
            "start_ms": 1_000,
            "deadline_ms": 61_000,
            "status": "probing",
        }
    ]
    journal = {
        "probe-violation": [
            {
                "last_observation": {
                    "ts_ms": 20_000,
                    "ttft_p95_ms": 1200.0,
                    "tpot_p95_ms": 50.0,
                    "z_m": 1.4,
                    "q_ctl": 0.0,
                    "has_traffic": True,
                }
            }
        ]
    }
    store = FakeProbeStore(unresolved=unresolved, journal=journal)
    machine = SafeScaleStateMachine(config=_cfg(), store=store)
    assert machine.restore() == 1

    decision = machine.observe("donor", _healthy_observation(ts_ms=61_000), now_ms=61_000)

    assert decision.status == "rollback"
    assert decision.reason == "formal_commit_gate_failed"
