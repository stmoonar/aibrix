"""Utilisation-gated scale-down probe (TRE_UTIL_SCALE_DOWN)."""
from __future__ import annotations

import math

import pytest

from tre_common.metrics_schema import MetricsSnapshot, ModelWindowMetrics
from tre_common.registry import ClusterTopology, ModelSpec, NodeSpec, Registry, load_registry
from tre_controller.app import create_controller_dependencies
from tre_controller.config import ControllerConfig, SafeScaleConfig
from tre_controller.loops.fairness_task import run_fairness_tick
from tre_controller.planning.classify import ModelClassification, ModelRole, ModelState, TauThresholds
from tre_controller.planning.planner import ClusterView, HideAction, PlanConfig, ScaleAction, build_plan
from tre_controller.planning.safescale import SafeScaleStateMachine
from tre_controller.planning.util_scale_down import (
    DEFAULT_Q_PER_REPLICA,
    UtilScaleDown,
    UtilWindow,
    parse_q_per_replica,
    util_scale_down_ready,
)
from tre_sm.allocator.slots import Binding, Slot

from test_controller_app import REGISTRY_PATH, EmptyRedis
from test_loop_ticks import FakeQueue, _registry


def _windows(*q_raw: float, routable: int = 3) -> tuple[UtilWindow, ...]:
    return tuple(UtilWindow(5_000 * (i + 1), q, routable) for i, q in enumerate(q_raw))


# ------------------------------------------------------------------ pure signal


def test_ready_after_sustained_low_per_replica_load() -> None:
    assert util_scale_down_ready(_windows(1, 1, 1, 1, 1, 1), routable=3, threshold=2.0, windows=6) == 0.5


def test_not_ready_with_too_few_windows_rising_load_or_high_load() -> None:
    assert util_scale_down_ready(_windows(1, 1, 1, 1, 1), routable=3, threshold=2.0, windows=6) is None
    assert util_scale_down_ready(_windows(1, 1, 1, 1, 1, 1.5), routable=3, threshold=2.0, windows=6) is None
    assert util_scale_down_ready(_windows(1, 1, 1, 1, 1, 4.2), routable=3, threshold=2.0, windows=6) is None


def test_not_ready_when_replica_count_changed_inside_the_windows() -> None:
    history = _windows(1, 1, 1, routable=4) + tuple(UtilWindow(40_000 + i, 1, 3) for i in range(3))
    assert util_scale_down_ready(history, routable=3, threshold=2.0, windows=6) is None


def test_tracker_dedupes_windows_by_window_end() -> None:
    util = UtilScaleDown(windows=2)
    util.observe("m", window_end_ms=10, q_raw=1.0, routable=2)
    util.observe("m", window_end_ms=10, q_raw=9.0, routable=2)  # same window, other loop
    util.observe("m", window_end_ms=5, q_raw=9.0, routable=2)  # stale window
    util.observe("m", window_end_ms=15, q_raw=2.0, routable=2)
    assert util.history()["m"] == (UtilWindow(10, 1.0, 2), UtilWindow(15, 2.0, 2))


def test_threshold_precedence_env_model_then_env_global_then_registry_then_default() -> None:
    base = _registry().model("critical")
    with_registry = ModelSpec(**{**base.__dict__, "scale_down_q_per_replica": 7.0})
    assert UtilScaleDown().threshold_for(base) == DEFAULT_Q_PER_REPLICA
    assert UtilScaleDown().threshold_for(with_registry) == 7.0
    assert UtilScaleDown(q_overrides={"*": 3.0}).threshold_for(with_registry) == 3.0
    assert UtilScaleDown(q_overrides={"*": 3.0, "critical": 1.5}).threshold_for(with_registry) == 1.5


def test_parse_q_per_replica_env() -> None:
    assert parse_q_per_replica(None) == {}
    assert parse_q_per_replica("2.5") == {"*": 2.5}
    assert parse_q_per_replica("dsqwen-7b=20, dsllama-8b=12") == {"dsqwen-7b": 20.0, "dsllama-8b": 12.0}
    for bad in ("x", "m=0", "m=-1", "m=nan"):
        with pytest.raises(ValueError):
            parse_q_per_replica(bad)


# ------------------------------------------------------------------ config / registry wiring


def test_config_env_knobs() -> None:
    default = ControllerConfig.from_env({})
    assert (default.util_scale_down, default.util_scale_down_windows, default.util_scale_down_q_per_replica) == (
        True,
        6,
        {},
    )
    cfg = ControllerConfig.from_env(
        {
            "TRE_UTIL_SCALE_DOWN": "false",
            "TRE_UTIL_SCALE_DOWN_WINDOWS": "3",
            "TRE_UTIL_SCALE_DOWN_Q_PER_REPLICA": "dsqwen-7b=4",
        }
    )
    assert (cfg.util_scale_down, cfg.util_scale_down_windows, cfg.util_scale_down_q_per_replica) == (
        False,
        3,
        {"dsqwen-7b": 4.0},
    )


def test_app_builds_tracker_from_config_only_when_enabled() -> None:
    env = {"TRE_REGISTRY_PATH": str(REGISTRY_PATH)}
    on = create_controller_dependencies(
        ControllerConfig.from_env({**env, "TRE_UTIL_SCALE_DOWN_WINDOWS": "4", "TRE_UTIL_SCALE_DOWN_Q_PER_REPLICA": "9"}),
        redis_client=EmptyRedis(),
    )
    assert isinstance(on.util_scale_down, UtilScaleDown)
    assert (on.util_scale_down.windows, on.util_scale_down.q_overrides) == (4, {"*": 9.0})

    off = create_controller_dependencies(
        ControllerConfig.from_env({**env, "TRE_UTIL_SCALE_DOWN": "0"}), redis_client=EmptyRedis()
    )
    assert off.util_scale_down is None


def test_deploy_registry_carries_r3_derived_thresholds() -> None:
    """Every model reaches the controller with a usable scale_down_q_per_replica.

    Shape, not values: the numbers are re-derived by every R3 calibration, and the
    live ones come from the console ``/api/params`` PUT into the registry ConfigMap --
    ``deploy/registry.yaml`` is only the bootstrap copy. Pinning the numbers here would
    turn the next recalibration into a false failure while proving nothing about the
    pipeline this test exists to check: that the field survives load_registry at all.
    """
    registry = load_registry(str(REGISTRY_PATH))
    thresholds = {spec.name: spec.scale_down_q_per_replica for spec in registry.models()}

    assert set(thresholds) == {"dsqwen-7b", "dsllama-8b", "dsqwen-14b"}
    for name, value in thresholds.items():
        assert isinstance(value, float), name
        assert math.isfinite(value) and value > 0.0, (name, value)


# ------------------------------------------------------------------ tick level


def _registry_m(*, min_replicas: int = 1, q: float | None = None) -> Registry:
    base = _registry().model("critical")
    spec = ModelSpec(**{**base.__dict__, "name": "m", "min_replicas": min_replicas, "scale_down_q_per_replica": q})
    return Registry(ClusterTopology(nodes=(NodeSpec(name="node-a", gpus=4, two_gpu_slots=((0, 1), (2, 3))),)), [spec])


def _view(registry: Registry, awake: int) -> ClusterView:
    return ClusterView(
        registry.topology(),
        tuple(Binding(f"m-{i}", "m", Slot("node-a", (i,)), awake=i < awake) for i in range(4)),
    )


def _snapshot(index: int, *, running: float, pods: int) -> MetricsSnapshot:
    end = 60_000 + 5_000 * index
    return MetricsSnapshot(
        ts_ms=end,
        stale=False,
        models={
            # HIGH (fast tokens per queued request), lightly loaded.
            "m": ModelWindowMetrics(
                model="m",
                window_start_ms=end - 30_000,
                window_end_ms=end,
                prompt_tokens=0.0,
                generation_tokens=100_000.0 * 30.0,  # TSS is a rate: total = rate x 30 s window
                avg_waiting=0.0,
                avg_running=running,
                avg_swapping=0.0,
                kv_cache_hit_rate=0.0,
                ttft_p95_ms=100.0,
                tpot_p95_ms=10.0,
                e2e_p95_ms=1000.0,
                routable_pods=pods,
                assigned_replicas=pods,
                per_pod={},
            )
        },
    )


def _run(
    loads: list[float],
    *,
    util: UtilScaleDown | None,
    registry: Registry | None = None,
    awake: int = 3,
    safescale: bool = True,
):
    registry = registry or _registry_m()
    machine = SafeScaleStateMachine(config=SafeScaleConfig()) if safescale else None
    queue = FakeQueue()
    results = []
    for index, running in enumerate(loads):
        results.append(
            run_fairness_tick(
                _snapshot(index, running=running, pods=awake),
                queue=queue,
                registry=registry,
                cluster_view=_view(registry, awake),
                active_probe_models={p.model for p in machine.active_probes()} if machine else set(),
                safescale=machine,
                util_scale_down=util,
            )
        )
    hides = [a for batch in queue.submitted for a in batch if isinstance(a, HideAction)]
    return results, hides, machine


def test_sustained_low_load_proposes_exactly_one_probe() -> None:
    results, hides, machine = _run([1.0] * 9, util=UtilScaleDown())

    # m-0/m-1/m-2 awake on gpu 0/1/2, m-3 sleeping on gpu 3: hiding m-2 is the only
    # shrink that hands back an aligned pair (2,3) for a tp=2 model, so release order
    # puts it first (a lexicographic order would have probed m-0 and freed nothing).
    assert hides == [HideAction("m", ("m-2",), "probe_started", "fairness")]
    assert [i for i, r in enumerate(results) if any(e.startswith("util_scale_down_proposed:m") for e in r.events)] == [5]
    assert "util_scale_down_proposed:m:q_after=0.50" in results[5].events
    assert machine.active_probe("m").pending_upscales == {}


def test_rising_load_or_too_few_windows_proposes_nothing() -> None:
    assert _run([1.0] * 5, util=UtilScaleDown())[1] == []
    assert _run([0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1], util=UtilScaleDown())[1] == []


def test_min_replicas_and_serving_floor_are_respected() -> None:
    assert _run([0.0] * 8, util=UtilScaleDown(), registry=_registry_m(min_replicas=3))[1] == []
    assert _run([0.0] * 8, util=UtilScaleDown(), registry=_registry_m(min_replicas=0), awake=1)[1] == []
    assert len(_run([0.0] * 8, util=UtilScaleDown(), registry=_registry_m(min_replicas=0), awake=2)[1]) == 1


def test_registry_threshold_and_env_override_reach_the_planner() -> None:
    # q_after = 1.0 / (3 - 1) = 0.5
    assert _run([1.0] * 6, util=UtilScaleDown(), registry=_registry_m(q=0.4))[1] == []
    assert len(_run([1.0] * 6, util=UtilScaleDown(), registry=_registry_m(q=0.6))[1]) == 1
    assert _run([1.0] * 6, util=UtilScaleDown(q_overrides={"m": 0.4}), registry=_registry_m(q=0.6))[1] == []


def test_windows_knob_reaches_the_planner() -> None:
    results, hides, _ = _run([1.0] * 2, util=UtilScaleDown(windows=2))
    assert len(hides) == 1


def test_flag_off_or_no_safescale_is_identical_to_legacy() -> None:
    legacy, legacy_hides, _ = _run([1.0] * 8, util=None)
    no_safescale, _, _ = _run([1.0] * 8, util=UtilScaleDown(), safescale=False)
    assert legacy_hides == []
    for index in range(8):
        baseline = run_fairness_tick(
            _snapshot(index, running=1.0, pods=3),
            queue=FakeQueue(),
            registry=_registry_m(),
            cluster_view=_view(_registry_m(), 3),
            safescale=SafeScaleStateMachine(config=SafeScaleConfig()),
        )
        assert (legacy[index].actions, legacy[index].events) == (baseline.actions, baseline.events)
        assert legacy[index].model_contexts == baseline.model_contexts
        assert not any("util_scale_down" in event for event in no_safescale[index].events)
        assert not any(getattr(a, "reason", "") == "util_scale_down_safescale" for a in no_safescale[index].actions)


def test_planner_path_is_skipped_for_receivers_and_active_probes() -> None:
    history = {"m": _windows(0, 0, 0, 0, 0, 0)}

    def plan(state: ModelState, *, active: set[str] = frozenset()):
        return build_plan(
            model_contexts={"m": {"routable_pods": 3, "assigned_replicas": 3}},
            classifications=[
                ModelClassification(
                    model_name="m", state=state,
                    role=ModelRole.RECEIVER if state in (ModelState.CRITICAL, ModelState.LOW) else ModelRole.NEUTRAL,
                    Z_m=1.1, eta_m=None, trs=0.0, theta_m=1.0, tau=TauThresholds.from_control(),
                )
            ],
            model_replicas={"m": 3},
            idle_gpus=0,
            cfg=PlanConfig(
                min_replicas_per_model=1, max_replicas_per_model=4, rescue_due=False,
                util_scale_down=True, scale_down_q_per_replica_by_model={"m": 2.0},
            ),
            active_probe_models=set(active),
            util_windows=history,
        )

    ok = plan(ModelState.HEALTHY)
    assert [(a.model, a.delta, a.reason, a.requires_safescale) for a in ok.actions if isinstance(a, ScaleAction)] == [
        ("m", -1, "util_scale_down_safescale", True)
    ]
    for state in (ModelState.CRITICAL, ModelState.LOW, ModelState.IDLE):
        assert not any(getattr(a, "reason", "") == "util_scale_down_safescale" for a in plan(state).actions)
    assert plan(ModelState.HEALTHY, active={"m"}).actions == []
