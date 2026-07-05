from __future__ import annotations

from fastapi.testclient import TestClient

from tre_common.registry import ClusterTopology, ModelSpec, NodeSpec, Registry, SloSpec, TrsParams
from tre_sm.allocator.slots import Binding, Slot
from tre_sm.api.v2 import ServiceManagerV2, create_app
from tre_sm.allocator.topology import K8sPodSnapshot
from tre_sm.state.store import StateStore


class FakeRedis:
    def __init__(self):
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

    def hset(self, key, mapping):
        bucket = self.hashes.setdefault(key, {})
        for field, value in mapping.items():
            bucket[str(field).encode("utf-8")] = str(value).encode("utf-8")


class FakeRuntimeOps:
    def __init__(self):
        self.calls = []
        self.snapshots = {
            "serve-a": K8sPodSnapshot("serve-a", "m1", "node-a", {"CUDA_VISIBLE_DEVICES": "0"}, pod_ip="10.0.0.1"),
            "serve-b": K8sPodSnapshot("serve-b", "m1", "node-a", {"CUDA_VISIBLE_DEVICES": "2"}, pod_ip="10.0.0.2"),
        }
        self.ready_snapshot = K8sPodSnapshot(
            "serve-b-new",
            "m1",
            "node-a",
            {"CUDA_VISIBLE_DEVICES": "1"},
            annotations={"tre.aibrix.io/gpu-ids": "1"},
            pod_ip="10.0.0.3",
        )

    def list_pod_snapshots(self, *, model=None):
        snapshots = list(self.snapshots.values())
        if model is None:
            return snapshots
        return [snapshot for snapshot in snapshots if snapshot.model == model]

    def write_binding_annotations(self, binding, *, state):
        self.calls.append(("annotate", binding.serve_id, state))

    def wait_pod_unroutable(self, binding):
        self.calls.append(("wait_unroutable", binding.serve_id))

    def ensure_model_httproute(self, model):
        self.calls.append(("ensure_route", model))

    def delete_model_deployment(self, binding):
        self.calls.append(("delete_deployment", binding.serve_id, binding.slot.gpu_ids))
        self.snapshots.pop(binding.serve_id, None)

    def wait_pod_deleted(self, serve_id):
        self.calls.append(("wait_deleted", serve_id))

    def create_model_deployment(self, model, slot):
        self.calls.append(("create_deployment", model, slot.gpu_ids))
        self.snapshots[self.ready_snapshot.name] = self.ready_snapshot
        return self.ready_snapshot.name

    def wait_pod_ready(self, serve_id):
        self.calls.append(("wait_ready", serve_id))
        return self.snapshots[serve_id]


class FakeVllmOps:
    def __init__(self):
        self.calls = []

    def wait_until_ready(self, pod_ip, *, port=None):
        self.calls.append(("wait_until_ready", pod_ip, port))
        return type("Result", (), {"success": True, "message": ""})()

    def sleep(self, pod_ip, *, port=None):
        self.calls.append(("sleep", pod_ip, port))
        return type("Result", (), {"success": True, "message": ""})()

    def wake_up(self, pod_ip, *, port=None):
        self.calls.append(("wake_up", pod_ip, port))
        return type("Result", (), {"success": True, "message": ""})()


def registry() -> Registry:
    topology = ClusterTopology(nodes=(NodeSpec(name="node-a", gpus=4, two_gpu_slots=((0, 1), (2, 3))),))
    trs = TrsParams(
        w_p=0.04,
        w_d=1.0,
        lambda_wait=2.625,
        qmin=1.0,
        ema_alpha=0.5,
        theta_m=0.0,
        tau_crit=0.8,
        tau_low=1.0,
        tau_high=1.25,
        qsat=4.0,
        epsat=0.05,
        hsat=3,
    )
    slo = SloSpec(ttft_p95_ms=1200, tpot_p95_ms=100, e2e_p95_ms=10000)
    return Registry(
        topology,
        [
            ModelSpec(
                name="m1",
                weights_path="/m1",
                tp_size=1,
                min_replicas=0,
                max_replicas=2,
                vllm_image="image",
                slo=slo,
                trs=trs,
            )
        ],
    )


def test_v2_defrag_moves_one_gpu_serve_and_frees_two_gpu_slot():
    store = StateStore(FakeRedis())
    store.save(
        [
            Binding("serve-a", "m1", Slot("node-a", (0,)), awake=True),
            Binding("serve-b", "m1", Slot("node-a", (2,)), awake=True),
        ],
        expected_version=0,
    )
    client = TestClient(create_app(ServiceManagerV2(registry(), store)))

    response = client.post("/v2/defrag", json={"tp_size": 2})

    assert response.status_code == 200
    assert response.json() == {
        "version": 2,
        "migrations": [
            {
                "serve_id": "serve-b",
                "from_slot": {"node": "node-a", "gpu_ids": [2]},
                "to_slot": {"node": "node-a", "gpu_ids": [1]},
            }
        ],
        "actions": [
            {"action": "hide", "serve_id": "serve-b"},
            {"action": "sleep", "serve_id": "serve-b"},
            {"action": "recreate", "serve_id": "serve-b", "node": "node-a", "gpu_ids": [1]},
            {"action": "wake", "serve_id": "serve-b"},
            {"action": "unhide", "serve_id": "serve-b"},
        ],
    }
    assert client.get("/v2/state").json()["bindings"] == [
        {"serve_id": "serve-a", "model": "m1", "node": "node-a", "gpu_ids": [0], "awake": True, "hidden": False},
        {"serve_id": "serve-b", "model": "m1", "node": "node-a", "gpu_ids": [1], "awake": True, "hidden": False},
    ]


def test_v2_defrag_returns_409_without_partial_state_change_when_no_plan_exists():
    store = StateStore(FakeRedis())
    initial = [
        Binding("serve-a", "m1", Slot("node-a", (0,)), awake=True),
        Binding("serve-b", "m1", Slot("node-a", (1,)), awake=True),
        Binding("serve-c", "m1", Slot("node-a", (2,)), awake=True),
        Binding("serve-d", "m1", Slot("node-a", (3,)), awake=True),
    ]
    store.save(initial, expected_version=0)
    client = TestClient(create_app(ServiceManagerV2(registry(), store)))

    response = client.post("/v2/defrag", json={"tp_size": 2})

    assert response.status_code == 409
    assert response.json() == {"detail": {"reason": "no_feasible_defrag"}}
    assert store.load().bindings == initial


def test_v2_defrag_uses_runtime_delete_create_path_and_is_idempotent():
    store = StateStore(FakeRedis())
    store.save(
        [
            Binding("serve-a", "m1", Slot("node-a", (0,)), awake=True),
            Binding("serve-b", "m1", Slot("node-a", (2,)), awake=True),
        ],
        expected_version=0,
    )
    runtime_ops = FakeRuntimeOps()
    vllm_ops = FakeVllmOps()
    service = ServiceManagerV2(registry(), store, runtime_ops=runtime_ops, vllm_ops=vllm_ops)

    first = service.defrag(tp_size=2)
    second = service.defrag(tp_size=2)

    assert first["actions"] == [
        {"action": "hide", "serve_id": "serve-b"},
        {"action": "sleep", "serve_id": "serve-b"},
        {"action": "delete_deployment", "serve_id": "serve-b"},
        {"action": "create_deployment", "serve_id": "serve-b-new", "node": "node-a", "gpu_ids": [1]},
        {"action": "wake", "serve_id": "serve-b-new"},
        {"action": "unhide", "serve_id": "serve-b-new"},
    ]
    assert second["actions"] == []
    assert runtime_ops.calls == [
        ("ensure_route", "m1"),
        ("annotate", "serve-b", "hidden"),
        ("wait_unroutable", "serve-b"),
        ("annotate", "serve-b", "sleeping"),
        ("delete_deployment", "serve-b", (2,)),
        ("wait_deleted", "serve-b"),
        ("create_deployment", "m1", (1,)),
        ("ensure_route", "m1"),
        ("wait_ready", "serve-b-new"),
        ("annotate", "serve-b-new", "awake"),
    ]
    assert vllm_ops.calls == [
        ("sleep", "10.0.0.2", 8000),
        ("wait_until_ready", "10.0.0.3", 8000),
        ("wake_up", "10.0.0.3", 8000),
    ]
    assert store.load().bindings == [
        Binding("serve-a", "m1", Slot("node-a", (0,)), awake=True),
        Binding("serve-b-new", "m1", Slot("node-a", (1,)), awake=True),
    ]
