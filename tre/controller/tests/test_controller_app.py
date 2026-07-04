from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from tre_common.registry import ClusterTopology, NodeSpec
from tre_controller.app import ControllerDependencies, build_controller_task_specs
from tre_controller.loops.cluster_view_task import ClusterViewBox
from tre_controller.loops.metrics_task import SnapshotBox
from tre_controller.loops.decision_snapshot import DecisionSnapshotWriter
from tre_controller.planning.safescale import SafeScaleStateMachine

TRE_ROOT = Path(__file__).resolve().parents[2]
REGISTRY_PATH = TRE_ROOT / "deploy" / "registry.yaml"


class FakeQueue:
    async def run(self) -> None:
        return None


class FakeDecisionWriter:
    def write(self, loop_name, snapshot, result) -> None:
        return None


class FakeSafeScale:
    pass


class FakeServiceManagerClient:
    async def get_state(self) -> dict:
        return {"bindings": []}


class FakeRegistry:
    def topology(self) -> ClusterTopology:
        return ClusterTopology(
            nodes=(NodeSpec(name="node-a", gpus=4, two_gpu_slots=((0, 1), (2, 3))),)
        )


class EmptyRedis:
    def smembers(self, key):
        return set()

    def zrangebyscore(self, key, minimum, maximum):
        return []


def _cfg(
    *,
    enable_tre_scaling: bool = True,
    ablation_disable_fast_loop: bool = False,
) -> SimpleNamespace:
    return SimpleNamespace(
        enable_tre_scaling=enable_tre_scaling,
        ablation_disable_fast_loop=ablation_disable_fast_loop,
        metrics_window_ms=60_000,
        monitor_interval_s=20.0,
        rescue_interval_s=5.0,
        fairness_interval_s=10.0,
    )


def _deps() -> ControllerDependencies:
    return ControllerDependencies(
        store=object(),
        snapshot_box=SnapshotBox(),
        queue=FakeQueue(),
        sm_client=FakeServiceManagerClient(),
        cluster_view_box=ClusterViewBox(),
        decision_writer=FakeDecisionWriter(),
        safescale=FakeSafeScale(),
        registry=FakeRegistry(),
    )


def test_build_controller_task_specs_includes_all_runtime_tasks_by_default() -> None:
    specs = build_controller_task_specs(_deps(), _cfg())

    assert tuple(spec.name for spec in specs) == (
        "metrics",
        "cluster_view",
        "rescue",
        "fairness",
        "action_queue",
    )


def test_build_controller_task_specs_honors_fast_loop_ablation() -> None:
    specs = build_controller_task_specs(_deps(), _cfg(ablation_disable_fast_loop=True))

    assert tuple(spec.name for spec in specs) == ("metrics", "cluster_view", "fairness", "action_queue")


def test_build_controller_task_specs_disables_scaling_tasks_but_keeps_metrics() -> None:
    specs = build_controller_task_specs(_deps(), _cfg(enable_tre_scaling=False))

    assert tuple(spec.name for spec in specs) == ("metrics",)


def test_create_controller_dependencies_wires_configured_components() -> None:
    from tre_controller.app import create_controller_dependencies
    from tre_controller.config import ControllerConfig
    from tre_controller.loops.action_queue import ActionQueue
    from tre_controller.sm_client import ServiceManagerClient
    from tre_controller.store.metrics_store import MetricsStore

    cfg = ControllerConfig.from_env(
        {
            "TRE_REGISTRY_PATH": str(REGISTRY_PATH),
            "TRE_REDIS_URL": "redis://example:6379/0",
            "TRE_SERVICE_MANAGER_URL": "http://service-manager:8001/",
            "TRE_INSTANT_SAMPLE_INTERVAL_MS": "7000",
            "TRE_PERCENTILE_MODE": "interpolated",
        }
    )
    redis = EmptyRedis()

    deps = create_controller_dependencies(cfg, redis_client=redis)

    assert isinstance(deps.store, MetricsStore)
    assert deps.store._redis is redis
    assert deps.store._instant_sample_interval_ms == 7000
    assert deps.store._percentile_mode == "interpolated"
    assert isinstance(deps.snapshot_box, SnapshotBox)
    assert isinstance(deps.queue, ActionQueue)
    assert deps.queue._client is deps.sm_client
    assert isinstance(deps.sm_client, ServiceManagerClient)
    assert deps.cluster_view_box.get() is None
    assert isinstance(deps.decision_writer, DecisionSnapshotWriter)
    assert deps.decision_writer._redis is redis
    assert isinstance(deps.safescale, SafeScaleStateMachine)
    assert deps.registry.model("dsqwen-7b").tp_size == 1


def test_main_builds_config_from_env_and_runs_controller() -> None:
    import asyncio

    from tre_controller.app import main

    seen = {}

    async def fake_runner(deps, cfg):
        seen["redis"] = deps.store._redis
        seen["service_manager_url"] = cfg.service_manager_url
        seen["registry_models"] = [spec.name for spec in deps.registry.models()]

    asyncio.run(
        main(
            env={
                "TRE_REGISTRY_PATH": str(REGISTRY_PATH),
                "TRE_REDIS_URL": "redis://example:6379/0",
                "TRE_SERVICE_MANAGER_URL": "http://service-manager:8001/",
            },
            redis_client_factory=lambda url: ("redis", url),
            runner=fake_runner,
        )
    )

    assert seen == {
        "redis": ("redis", "redis://example:6379/0"),
        "service_manager_url": "http://service-manager:8001",
        "registry_models": ["dsqwen-7b", "dsllama-8b", "dsqwen-14b"],
    }



def test_main_uses_default_controller_runner_when_runner_not_injected(monkeypatch) -> None:
    import asyncio

    import tre_controller.app as app

    seen = {}

    async def fake_run_controller(deps, cfg):
        seen["redis"] = deps.store._redis
        seen["service_manager_url"] = cfg.service_manager_url

    monkeypatch.setattr(app, "run_controller", fake_run_controller)

    asyncio.run(
        app.main(
            env={
                "TRE_REGISTRY_PATH": str(REGISTRY_PATH),
                "TRE_REDIS_URL": "redis://example:6379/0",
                "TRE_SERVICE_MANAGER_URL": "http://service-manager:8001/",
            },
            redis_client_factory=lambda url: ("redis", url),
        )
    )

    assert seen == {
        "redis": ("redis", "redis://example:6379/0"),
        "service_manager_url": "http://service-manager:8001",
    }
