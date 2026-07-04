from __future__ import annotations

import pytest

from tre_common.metrics_schema import MetricsSnapshot, ModelWindowMetrics
from tre_common.rediskeys import DECISION_LATEST_KEY
from tre_common.registry import ClusterTopology, ModelSpec, NodeSpec, Registry, SloSpec, TrsParams
from tre_controller.loops.action_queue import ActionQueue
from tre_controller.loops.decision_snapshot import DecisionSnapshotWriter
from tre_controller.offline_integration import run_offline_integration_step
from tre_controller.planning.planner import ClusterView, DefragAction, ScaleAction
from tre_controller.sm_client import ServiceManagerClient
from tre_sm.allocator.slots import Binding, Slot
from tre_sm.api.v2 import ServiceManagerV2, create_app
from tre_sm.state.store import StateStore


class FakeRedis:
    def __init__(self) -> None:
        self.values = {}
        self.hashes = {}

    def get(self, key):
        value = self.values.get(key)
        return None if value is None else str(value).encode("utf-8")

    def set(self, key, value):
        self.values[key] = str(value)

    def delete(self, key):
        self.hashes.pop(key, None)

    def hgetall(self, key):
        return dict(self.hashes.get(key, {}))

    def hset(self, key, mapping=None, **kwargs):
        bucket = self.hashes.setdefault(key, {})
        values = dict(mapping or {})
        values.update(kwargs)
        for field, value in values.items():
            bucket[str(field).encode("utf-8")] = str(value).encode("utf-8")


class FixtureSnapshotStore:
    def __init__(self, snapshot: MetricsSnapshot) -> None:
        self.snapshot = snapshot
        self.windows: list[tuple[int, int]] = []

    def read_snapshot(self, window_start_ms: int, window_end_ms: int) -> MetricsSnapshot:
        self.windows.append((window_start_ms, window_end_ms))
        return self.snapshot


class AppTransport:
    def __init__(self, app) -> None:
        from fastapi.testclient import TestClient

        self.client = TestClient(app)
        self.calls: list[tuple[str, str, dict | None]] = []

    async def request(self, method: str, url: str, *, json: dict | None = None, timeout_s: float) -> dict:
        del timeout_s
        path = "/" + url.split("/", 3)[3]
        self.calls.append((method, path, json))
        response = self.client.request(method, path, json=json)
        response.raise_for_status()
        return response.json()


def _registry() -> Registry:
    trs = TrsParams(
        w_p=0.04,
        w_d=1.0,
        lambda_wait=2.625,
        qmin=1.0,
        ema_alpha=0.0,
        theta_m=100.0,
        tau_crit=0.8,
        tau_low=1.0,
        tau_high=1.25,
        qsat=4.0,
        epsat=0.05,
        hsat=1,
    )
    return Registry(
        ClusterTopology(nodes=(NodeSpec(name="node-a", gpus=4, two_gpu_slots=((0, 1), (2, 3))),)),
        [
            ModelSpec(
                name="m1",
                weights_path="/weights/m1",
                tp_size=1,
                min_replicas=0,
                max_replicas=2,
                vllm_image="image",
                slo=SloSpec(ttft_p95_ms=1200, tpot_p95_ms=100, e2e_p95_ms=10000),
                trs=trs,
            ),
            ModelSpec(
                name="tp2",
                weights_path="/weights/tp2",
                tp_size=2,
                min_replicas=0,
                max_replicas=1,
                vllm_image="image",
                slo=SloSpec(ttft_p95_ms=1200, tpot_p95_ms=100, e2e_p95_ms=10000),
                trs=trs,
            ),
        ],
    )


def _critical_snapshot() -> MetricsSnapshot:
    return MetricsSnapshot(
        ts_ms=300_000,
        stale=False,
        models={
            "m1": ModelWindowMetrics(
                model="m1",
                window_start_ms=240_000,
                window_end_ms=300_000,
                prompt_tokens=0.0,
                generation_tokens=50.0,
                avg_waiting=10.0,
                avg_running=1.0,
                avg_swapping=0.0,
                kv_cache_hit_rate=0.0,
                ttft_p95_ms=100.0,
                tpot_p95_ms=10.0,
                e2e_p95_ms=1000.0,
                routable_pods=1,
                assigned_replicas=1,
                per_pod={},
            )
        },
    )


def _critical_tp2_snapshot() -> MetricsSnapshot:
    return MetricsSnapshot(
        ts_ms=300_000,
        stale=False,
        models={
            "tp2": ModelWindowMetrics(
                model="tp2",
                window_start_ms=240_000,
                window_end_ms=300_000,
                prompt_tokens=0.0,
                generation_tokens=50.0,
                avg_waiting=10.0,
                avg_running=1.0,
                avg_swapping=0.0,
                kv_cache_hit_rate=0.0,
                ttft_p95_ms=100.0,
                tpot_p95_ms=10.0,
                e2e_p95_ms=1000.0,
                routable_pods=0,
                assigned_replicas=0,
                per_pod={},
            )
        },
    )


@pytest.mark.asyncio
async def test_p9_offline_integration_step_closes_metrics_decision_dispatch_chain() -> None:
    registry = _registry()
    redis = FakeRedis()
    sm_store = StateStore(redis)
    sm_store.save(
        [
            Binding("serve-a", "m1", Slot("node-a", (0,)), awake=True),
            Binding("serve-b", "m1", Slot("node-a", (1,)), awake=False),
        ],
        expected_version=0,
    )
    transport = AppTransport(create_app(ServiceManagerV2(registry, sm_store)))
    queue = ActionQueue(ServiceManagerClient("http://service-manager", transport=transport))

    result = await run_offline_integration_step(
        store=FixtureSnapshotStore(_critical_snapshot()),
        queue=queue,
        decision_writer=DecisionSnapshotWriter(redis),
        registry=registry,
        now_ms=360_000,
        window_ms=60_000,
    )

    assert result.metrics.stale is False
    assert [action.reason for action in result.decision.actions] == ["critical_idle_capacity"]
    assert [(item.action_kind, item.model, item.ok) for item in result.dispatches] == [("scale", "m1", True)]
    assert transport.calls == [
        ("GET", "/v2/state", None),
        ("PUT", "/v2/models/m1/target", {"wake_replicas": 2}),
    ]
    assert sm_store.load().bindings[1].awake is True
    decision_hash = redis.hgetall(DECISION_LATEST_KEY)
    assert decision_hash[b"loop"] == b"rescue"
    assert b"critical_idle_capacity" in decision_hash[b"actions"]


@pytest.mark.asyncio
async def test_p9_offline_integration_defrags_fragmented_capacity_then_expands_tp2() -> None:
    registry = _registry()
    redis = FakeRedis()
    sm_store = StateStore(redis)
    bindings = (
        Binding("serve-a", "m1", Slot("node-a", (0,)), awake=True),
        Binding("serve-b", "m1", Slot("node-a", (2,)), awake=True),
    )
    sm_store.save(bindings, expected_version=0)
    transport = AppTransport(create_app(ServiceManagerV2(registry, sm_store)))
    queue = ActionQueue(ServiceManagerClient("http://service-manager", transport=transport))

    result = await run_offline_integration_step(
        store=FixtureSnapshotStore(_critical_tp2_snapshot()),
        queue=queue,
        decision_writer=DecisionSnapshotWriter(redis),
        registry=registry,
        now_ms=360_000,
        window_ms=60_000,
        cluster_view=ClusterView(registry.topology(), bindings),
    )

    assert [type(action) for action in result.decision.actions] == [DefragAction, ScaleAction]
    assert [action.reason for action in result.decision.actions] == ["critical_tp_defrag", "critical_tp_defrag"]
    assert [(item.action_kind, item.model, item.ok) for item in result.dispatches] == [
        ("defrag", "__cluster__", True),
        ("scale", "tp2", True),
    ]
    assert transport.calls == [
        ("POST", "/v2/defrag", {"tp_size": 2}),
        ("GET", "/v2/state", None),
        ("PUT", "/v2/models/tp2/target", {"wake_replicas": 1}),
    ]
    assert sm_store.load().bindings == [
        Binding("serve-a", "m1", Slot("node-a", (0,)), awake=True),
        Binding("serve-b", "m1", Slot("node-a", (1,)), awake=True),
        Binding("tp2-1", "tp2", Slot("node-a", (2, 3)), awake=True),
    ]
