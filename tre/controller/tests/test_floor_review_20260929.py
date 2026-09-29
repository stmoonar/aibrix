"""Review fixes of the SafeScale window / replica-floor commit (2026-09-29).

* P1-2 (audit only): every commit-gate decision records how much of the judged tail
  evidence predates the hide (``tail_pre_hide_fraction_mean`` / ``_max`` and
  ``tail_observation_count``), from the metrics-window timestamps each observation
  now carries.
* P2-6: an SM 409 floor_violation holds the refused donor out of scale-down planning
  for TRE_FLOOR_VIOLATION_COOLDOWN_TICKS fast-loop ticks (no per-tick livelock).
* P2-7: the probe-window floor moved to SAFE_SCALE_WINDOW_FLOOR_MS; the legacy
  SAFE_SCALE_MIN_WINDOW_MS stays at 60000 in the overlay for older images.
* P3-11: W is capped at 2 x registry gateway.route_timeout_s.
* Fix C relaxed: a same-slot shrink is counted in the plan's deltas only - a
  3-replica donor with floor 1 serves two receivers of one tick, a 2-replica one
  only one.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path

import pytest
import yaml

from tre_common.metrics_schema import MetricsSnapshot
from tre_common.registry import ClusterTopology, NodeSpec
from tre_controller.config import ControllerConfig, SafeScaleConfig
from tre_controller.loops.action_queue import ActionQueue
from tre_controller.loops.rescue_task import run_rescue_tick
from tre_controller.loops.safescale_task import _observation_from_metrics, run_safescale_observation_tick
from tre_controller.planning.classify import ModelClassification, ModelRole, ModelState, TauThresholds
from tre_controller.planning.planner import ClusterView, PlanConfig, ScaleAction, ShrinkForSlotAction, build_plan
from tre_controller.planning.safescale import (
    ProbeObservation,
    ProbeWindowInputs,
    SafeScaleStateMachine,
    calc_probe_window_details,
    format_window_event,
)
from tre_sm.allocator.slots import Binding, Slot

from test_action_queue_review3 import ScriptedSM
from test_loop_ticks import _metrics as _tick_metrics
from test_loop_ticks import _registry_with_model_bounds
from test_safescale import FakeProbeStore
from test_safescale_task import FakeQueue as ObservationQueue
from test_safescale_task import _metrics as _observation_snapshot
from test_safescale_task import _registry as _donor_registry

TRE_DIR = Path(__file__).resolve().parents[2]


# --------------------------------------------------------------- P1-2 tail audit
def _obs(ts_ms: int, window_ms: int = 30_000) -> ProbeObservation:
    return ProbeObservation(
        ts_ms=ts_ms,
        ttft_p95_ms=100.0,
        tpot_p95_ms=10.0,
        z_m=2.0,
        has_traffic=True,
        window_start_ms=ts_ms - window_ms,
        window_end_ms=ts_ms,
    )


def test_commit_gate_records_the_pre_hide_share_of_the_judged_tail() -> None:
    store = FakeProbeStore()
    machine = SafeScaleStateMachine(
        config=SafeScaleConfig(ttft_p95_slo_ms=500.0, tpot_p95_slo_ms=75.0, min_window_ms=25_000.0, hq=0.5),
        store=store,
    )
    machine.start_probe(model="donor", pods=("donor-0",), now_ms=100_000)  # hide at 100 s
    assert machine.observe("donor", _obs(110_000), now_ms=110_000).reason == "probe_pending"
    assert machine.observe("donor", _obs(120_000), now_ms=120_000).reason == "probe_pending"
    decision = machine.observe("donor", _obs(130_000), now_ms=130_000)

    assert decision.status == "commit"  # the audit never changes the criterion
    # hq 0.5 of 3 observations -> the last 2 are judged: windows [90, 120] s and
    # [100, 130] s -> pre-hide shares 10/30 and 0/30.
    expected = {
        "tail_pre_hide_fraction_mean": pytest.approx(1 / 6),
        "tail_pre_hide_fraction_max": pytest.approx(1 / 3),
        "tail_observation_count": 2,
    }
    probe = machine.active_probe("donor")
    for record in (decision.details, probe.terminal_details, probe.window_terms, probe.terminal_details["tail"]):
        assert {key: record[key] for key in expected} == expected
    # Persisted with the decision (committing record) and the observation journal.
    machine.mark_committing("donor", status="commit", reason=decision.reason, now_ms=130_000)
    saved = store.records[probe.request_id]
    assert saved["terminal_details"]["tail_pre_hide_fraction_max"] == pytest.approx(1 / 3)
    assert saved["window_terms"]["tail_observation_count"] == 2
    journal = store.journal[probe.request_id][-1]["last_observation"]
    assert (journal["window_start_ms"], journal["window_end_ms"]) == (100_000, 130_000)


def test_observations_carry_the_metrics_window_they_read() -> None:
    snapshot = _observation_snapshot(ts_ms=70_000)
    metrics = snapshot.models["donor"]  # window [0, 60 s + ts] (one snapshot per ts)
    observation = _observation_from_metrics(70_000, metrics, _donor_registry().model("donor"), "zm")
    assert (observation.window_start_ms, observation.window_end_ms) == (0, 130_000)

    queue = ObservationQueue()
    machine = SafeScaleStateMachine(
        config=SafeScaleConfig(ttft_p95_slo_ms=1000.0, tpot_p95_slo_ms=100.0, min_window_ms=1000.0, hq=0.5)
    )
    machine.start_probe(model="donor", pods=("pod-a",), now_ms=45_000)  # hide 45 s into the window
    result = run_safescale_observation_tick(
        _observation_snapshot(ts_ms=60_000), queue=queue, registry=_donor_registry(), safescale=machine
    )
    # window [0, 120 s], hide at 45 s -> 45/120 of it predates the hide.
    assert "safescale_tail_pre_hide:donor:mean=0.375:max=0.375:n=1" in result.events


# ------------------------------------------------------ P2-6 floor-violation hold
FLOOR_REFUSAL = {
    "ok": False,
    "error": "HTTP 409: replica floor",
    "status": 409,
    "retriable": False,
    "floor_violation": {"model": "warm", "floor": 1, "routable_after": 0},
}


def _idle_warm_snapshot() -> MetricsSnapshot:
    # "warm" is IDLE with 2 routable replicas and min_replicas 1: the fast loop plans
    # an immediate (urgent) scale-down of one replica every tick.
    return MetricsSnapshot(
        ts_ms=1,
        stale=False,
        models={"warm": _tick_metrics("warm", generation=0.0, waiting=0.0, running=0.0, assigned=2, routable=2)},
    )


def test_a_floor_violation_holds_the_donor_for_n_ticks_then_releases_it() -> None:
    registry = _registry_with_model_bounds({"warm": (1, 4)})
    clock = {"ms": 1_000_000}
    ticks, interval_s = 6, 5.0
    sm = ScriptedSM(results={"scale:warm": [FLOOR_REFUSAL]})
    queue = ActionQueue(sm, now_ms=lambda: clock["ms"], floor_violation_hold_ms=ticks * interval_s * 1000.0)

    first = run_rescue_tick(_idle_warm_snapshot(), queue=queue, registry=registry)
    assert [(a.model, a.delta) for a in first.actions] == [("warm", -1)]
    [refused] = asyncio.run(queue.drain_once())
    assert (refused.ok, refused.floor_violation) == (False, True)

    # The next tick does not pick the refused donor again (no livelock) ...
    clock["ms"] += int(interval_s * 1000)
    second = run_rescue_tick(_idle_warm_snapshot(), queue=queue, registry=registry)
    assert second.actions == () and "floor_violation_hold:warm" in second.events
    # ... for the whole hold ...
    clock["ms"] = 1_000_000 + int((ticks * interval_s - 1) * 1000)
    assert run_rescue_tick(_idle_warm_snapshot(), queue=queue, registry=registry).actions == ()
    # ... and plans it again once N ticks have passed.
    clock["ms"] = 1_000_000 + int(ticks * interval_s * 1000)
    third = run_rescue_tick(_idle_warm_snapshot(), queue=queue, registry=registry)
    assert [(a.model, a.delta) for a in third.actions] == [("warm", -1)]
    assert queue.stats()["floor_violation_total"] == 1


def test_only_floor_refusals_hold_and_zero_ticks_disables_the_hold() -> None:
    registry = _registry_with_model_bounds({"warm": (1, 4)})
    busy = {"ok": False, "error": "HTTP 409: writer busy", "status": 409, "retriable": True}
    queue = ActionQueue(ScriptedSM(results={"scale:warm": [busy]}), floor_violation_hold_ms=30_000.0)
    run_rescue_tick(_idle_warm_snapshot(), queue=queue, registry=registry)
    asyncio.run(queue.drain_once())
    assert queue.floor_held_models() == set()

    off = ActionQueue(ScriptedSM(results={"scale:warm": [FLOOR_REFUSAL]}), floor_violation_hold_ms=0.0)
    run_rescue_tick(_idle_warm_snapshot(), queue=off, registry=registry)
    asyncio.run(off.drain_once())
    assert off.floor_held_models() == set()
    assert run_rescue_tick(_idle_warm_snapshot(), queue=off, registry=registry).actions != ()


def test_floor_violation_cooldown_ticks_config() -> None:
    assert ControllerConfig.from_env({}).floor_violation_cooldown_ticks == 6
    assert ControllerConfig.from_env({"TRE_FLOOR_VIOLATION_COOLDOWN_TICKS": "0"}).floor_violation_cooldown_ticks == 0
    with pytest.raises(ValueError, match="TRE_FLOOR_VIOLATION_COOLDOWN_TICKS"):
        ControllerConfig.from_env({"TRE_FLOOR_VIOLATION_COOLDOWN_TICKS": "-1"})


# ------------------------------------------------ P2-7 window-floor env rename
def test_new_window_floor_env_wins_over_the_legacy_one(caplog) -> None:
    both = ControllerConfig.from_env({"SAFE_SCALE_WINDOW_FLOOR_MS": "20000", "SAFE_SCALE_MIN_WINDOW_MS": "60000"})
    assert both.safescale.min_window_ms == 20_000.0
    with caplog.at_level(logging.WARNING, logger="tre_controller.config"):
        legacy = ControllerConfig.from_env({"SAFE_SCALE_MIN_WINDOW_MS": "60000"})
    assert legacy.safescale.min_window_ms == 60_000.0
    assert any("legacy SAFE_SCALE_MIN_WINDOW_MS" in record.getMessage() for record in caplog.records)
    assert ControllerConfig.from_env({}).safescale.min_window_ms == 20_000.0


def _overlay_controller_env() -> dict[str, str]:
    doc = yaml.safe_load((TRE_DIR / "deploy" / "overlays" / "tre-v2" / "controller.yaml").read_text())
    [container] = [c for c in doc["spec"]["template"]["spec"]["containers"] if c["name"] == "controller"]
    return {item["name"]: str(item["value"]) for item in container["env"] if "value" in item}


def test_overlay_env_gives_the_new_floor_and_keeps_the_legacy_value_for_old_images() -> None:
    env = _overlay_controller_env()
    # Older images read only the legacy name; their N2 startup guard needs >= 60000.
    assert float(env["SAFE_SCALE_MIN_WINDOW_MS"]) >= 60_000.0
    env["TRE_REGISTRY_PATH"] = str(TRE_DIR / "deploy" / "registry.yaml")
    assert ControllerConfig.from_env(env).safescale.min_window_ms == 20_000.0


# ------------------------------------------------------- P3-11 window ceiling
def test_window_is_capped_at_twice_the_route_timeout() -> None:
    cfg = SafeScaleConfig(min_window_ms=20_000.0, window_ceiling_ms=300_000.0)
    capped = calc_probe_window_details(ProbeWindowInputs(p95_e2e_ms=200_000.0), hidden_count=1, config=cfg)
    assert (capped["W"], capped["W_max"], capped["clamped"], capped["dominant"]) == (
        300_000.0, 300_000.0, True, "ceiling"
    )
    assert ":max=300000:clamped=1" in format_window_event("m", capped)
    free = calc_probe_window_details(ProbeWindowInputs(p95_e2e_ms=50_000.0), hidden_count=1, config=cfg)
    assert (free["W"], free["clamped"], free["dominant"]) == (100_000.0, False, "e2e")
    # A ceiling below the floor never shortens W under the floor.
    low = SafeScaleConfig(min_window_ms=20_000.0, window_ceiling_ms=10_000.0)
    floor = calc_probe_window_details(ProbeWindowInputs(), hidden_count=1, config=low)
    assert (floor["W"], floor["W_max"], floor["clamped"]) == (20_000.0, 20_000.0, False)


def test_window_ceiling_comes_from_the_registry_safescale_section(tmp_path, caplog) -> None:
    # 2026-09-29: registry safescale.window_ceiling_s (60 s) replaced 2 x route timeout.
    assert ControllerConfig.from_env({}).safescale.window_ceiling_ms == 60_000.0
    raw = yaml.safe_load((TRE_DIR / "deploy" / "registry.yaml").read_text(encoding="utf-8"))
    raw["safescale"]["window_ceiling_s"] = 45
    raw["gateway"]["route_timeout_s"] = 90  # no longer part of the ceiling
    tuned = tmp_path / "tuned.yaml"
    tuned.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    assert ControllerConfig.from_env({"TRE_REGISTRY_PATH": str(tuned)}).safescale.window_ceiling_ms == 45_000.0
    # Section absent (a live registry not merged yet) -> the built-in 60 s.
    raw.pop("safescale")
    absent = tmp_path / "absent.yaml"
    absent.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    assert ControllerConfig.from_env({"TRE_REGISTRY_PATH": str(absent)}).safescale.window_ceiling_ms == 60_000.0
    # Registry unreadable -> the built-in defaults, with a warning.
    with caplog.at_level(logging.WARNING, logger="tre_controller.config"):
        missing = ControllerConfig.from_env({"TRE_REGISTRY_PATH": str(tmp_path / "missing.yaml")})
    assert missing.safescale.window_ceiling_ms == 60_000.0
    assert any("built-in defaults" in record.getMessage() for record in caplog.records)


def test_probe_record_keeps_w_max_and_clamped() -> None:
    store = FakeProbeStore()
    machine = SafeScaleStateMachine(config=SafeScaleConfig(window_ceiling_ms=300_000.0), store=store)
    machine.start_probe(
        model="m", pods=("m-0",), now_ms=0, window_inputs=ProbeWindowInputs(p95_e2e_ms=400_000.0)
    )
    probe = machine.active_probe("m")
    assert (probe.window_ms, probe.deadline_ms) == (300_000.0, 300_000)
    terms = store.records[probe.request_id]["window_terms"]
    assert (terms["W_max"], terms["clamped"]) == (300_000.0, True)


# ------------------------------------------- fix C relaxed: floor via deltas only
def _cls(model: str, state: ModelState, role: ModelRole, z: float | None, tier: str | None = None):
    return ModelClassification(
        model_name=model,
        state=state,
        role=role,
        Z_m=z,
        eta_m=None,
        trs=0.0,
        theta_m=1.0,
        tau=TauThresholds.from_control(),
        donor_tier=tier,
    )


def _taken(plan, model: str) -> list[str]:
    """Pods the plan takes from ``model`` (same-slot shrinks + negative scales)."""
    pods = [a.serve_id for a in plan.actions if isinstance(a, ShrinkForSlotAction) and a.donor == model]
    for action in plan.actions:
        if isinstance(action, ScaleAction) and action.model == model and action.delta < 0:
            pods.extend(action.pods or ("?",) * -action.delta)
    return pods


def _two_receiver_plan(high_replicas: int):
    """tp2 (TP=2, CRITICAL) can only be served by a same-slot shrink of a HIGH
    replica on node-a; crit1 (TP=1, CRITICAL) can only be served by the donor loop
    with high-b1 (node-b gpu 1, on which crit1 has its sleeping binding)."""
    high_node_a = [Binding("high-0", "high", Slot("node-a", (0,)), awake=True)]
    if high_replicas >= 3:
        high_node_a.append(Binding("high-2", "high", Slot("node-a", (2,)), awake=True))
    bindings = tuple(high_node_a) + (
        Binding("high-b1", "high", Slot("node-b", (1,)), awake=True),
        Binding("crit1-b0", "crit1", Slot("node-b", (0,)), awake=True),
        Binding("crit1-b1", "crit1", Slot("node-b", (1,)), awake=False),
    )
    view = ClusterView(
        topology=ClusterTopology(
            nodes=(
                NodeSpec(name="node-a", gpus=4, two_gpu_slots=((0, 1), (2, 3))),
                NodeSpec(name="node-b", gpus=2, two_gpu_slots=((0, 1),)),
            )
        ),
        bindings=bindings,
    )
    return build_plan(
        model_contexts={
            "tp2": {"assigned_replicas": 0, "routable_pods": 0},
            "crit1": {"assigned_replicas": 2, "routable_pods": 1, "awake_replicas": 1},
            "high": {"assigned_replicas": high_replicas, "routable_pods": high_replicas},
        },
        classifications=[
            _cls("tp2", ModelState.CRITICAL, ModelRole.RECEIVER, 0.5),
            _cls("crit1", ModelState.CRITICAL, ModelRole.RECEIVER, 0.5),
            _cls("high", ModelState.HIGH, ModelRole.DONOR, 1.4, "surplus"),
        ],
        model_replicas={"tp2": 0, "crit1": 2, "high": high_replicas},
        idle_gpus=0,
        cfg=PlanConfig(
            min_replicas_per_model=1,
            max_replicas_per_model=4,
            model_tp_sizes={"tp2": 2, "crit1": 1, "high": 1},
        ),
        cluster_view=view,
    )


def test_three_replica_donor_with_floor_one_serves_two_receivers_in_one_tick() -> None:
    plan = _two_receiver_plan(3)
    shrinks = [(a.donor, a.beneficiary) for a in plan.actions if isinstance(a, ShrinkForSlotAction)]
    assert shrinks == [("high", "tp2")]
    immediate = [a for a in plan.actions if isinstance(a, ScaleAction) and a.model == "high"]
    assert [(a.delta, a.reason, a.pods) for a in immediate] == [(-1, "critical_donor_immediate", ("high-b1",))]
    taken = _taken(plan, "high")
    assert len(taken) == 2 and len(set(taken)) == 2  # two different pods, 1 left = floor


def test_two_replica_donor_with_floor_one_is_taken_once() -> None:
    plan = _two_receiver_plan(2)
    assert [(a.donor, a.beneficiary) for a in plan.actions if isinstance(a, ShrinkForSlotAction)] == [
        ("high", "tp2")
    ]
    assert not [a for a in plan.actions if isinstance(a, ScaleAction) and a.model == "high"]
    assert len(_taken(plan, "high")) == 1


def test_one_same_slot_probe_per_donor_model_per_tick() -> None:
    # Two TP=2 receivers and a 3-replica HIGH donor: the SafeScale state machine holds
    # one probe per model, so only the first receiver gets a same-slot shrink.
    view = ClusterView(
        topology=ClusterTopology(nodes=(NodeSpec(name="node-a", gpus=4, two_gpu_slots=((0, 1), (2, 3))),)),
        bindings=(
            Binding("high-0", "high", Slot("node-a", (0,)), awake=True),
            Binding("high-2", "high", Slot("node-a", (2,)), awake=True),
            Binding("high-3", "high", Slot("node-a", (3,)), awake=True),
        ),
    )
    plan = build_plan(
        model_contexts={
            "tpa": {"assigned_replicas": 0, "routable_pods": 0},
            "tpb": {"assigned_replicas": 0, "routable_pods": 0},
            "high": {"assigned_replicas": 3, "routable_pods": 3},
        },
        classifications=[
            _cls("tpa", ModelState.CRITICAL, ModelRole.RECEIVER, 0.5),
            _cls("tpb", ModelState.CRITICAL, ModelRole.RECEIVER, 0.5),
            _cls("high", ModelState.HIGH, ModelRole.DONOR, 1.4, "surplus"),
        ],
        model_replicas={"tpa": 0, "tpb": 0, "high": 3},
        idle_gpus=0,
        cfg=PlanConfig(
            min_replicas_per_model=1, max_replicas_per_model=2, model_tp_sizes={"tpa": 2, "tpb": 2, "high": 1}
        ),
        cluster_view=view,
    )
    assert [(a.donor, a.beneficiary) for a in plan.actions if isinstance(a, ShrinkForSlotAction)] == [
        ("high", "tpa")
    ]
