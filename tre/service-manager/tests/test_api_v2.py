from tre_common.registry import ClusterTopology, ModelSpec, NodeSpec, Registry, SloSpec, TrsParams
from tre_sm.allocator.slots import Binding, Slot
from fastapi.testclient import TestClient

from tre_sm.api.v2 import ServiceManagerV2, create_app
from tre_sm.state.reconcile import PodRecord
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
    def __init__(self, snapshots):
        self._snapshots = list(snapshots)
        self.annotations = []

    def list_pod_snapshots(self, *, model=None):
        if model is None:
            return list(self._snapshots)
        return [snapshot for snapshot in self._snapshots if snapshot.model == model]

    def write_binding_annotations(self, binding, *, state):
        self.annotations.append((binding.serve_id, state))


class FakeVllmOps:
    def __init__(self):
        self.calls = []

    def sleep(self, pod_ip, *, port=None):
        self.calls.append(("sleep", pod_ip, port))
        return type("Result", (), {"success": True, "message": ""})()

    def wake_up(self, pod_ip, *, port=None):
        self.calls.append(("wake_up", pod_ip, port))
        return type("Result", (), {"success": True, "message": ""})()

class FakeK8sClient:
    def __init__(self, pods):
        self._pods = list(pods)

    def list_pods(self):
        return list(self._pods)


def registry():
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


def registry_with_tp2():
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
            ),
            ModelSpec(
                name="tp2",
                weights_path="/tp2",
                tp_size=2,
                min_replicas=0,
                max_replicas=1,
                vllm_image="image",
                slo=slo,
                trs=trs,
            ),
        ],
    )


def test_v2_state_exposes_version_and_bindings():
    store = StateStore(FakeRedis())
    store.save(
        [
            Binding("serve-a", "m1", Slot("node-a", (0,)), awake=True),
            Binding("serve-b", "m1", Slot("node-a", (1,)), awake=False),
        ],
        expected_version=0,
    )
    service = ServiceManagerV2(registry(), store)

    state = service.get_state()

    assert state["version"] == 1
    assert state["models"]["m1"] == {"awake": 1, "bound": 2}
    assert state["bindings"] == [
        {"serve_id": "serve-a", "model": "m1", "node": "node-a", "gpu_ids": [0], "awake": True, "hidden": False},
        {"serve_id": "serve-b", "model": "m1", "node": "node-a", "gpu_ids": [1], "awake": False, "hidden": False},
    ]


def test_v2_put_target_is_idempotent_and_returns_diff_actions():
    store = StateStore(FakeRedis())
    store.save(
        [
            Binding("serve-a", "m1", Slot("node-a", (0,)), awake=True),
            Binding("serve-b", "m1", Slot("node-a", (1,)), awake=False),
        ],
        expected_version=0,
    )
    service = ServiceManagerV2(registry(), store)

    first = service.put_model_target("m1", wake_replicas=2)
    second = service.put_model_target("m1", wake_replicas=2)

    assert first == {
        "model": "m1",
        "wake_replicas": 2,
        "version": 2,
        "actions": [{"action": "wake", "serve_id": "serve-b"}],
    }
    assert second == {
        "model": "m1",
        "wake_replicas": 2,
        "version": 2,
        "actions": [],
    }
    assert store.load().bindings == [
        Binding("serve-a", "m1", Slot("node-a", (0,)), awake=True),
        Binding("serve-b", "m1", Slot("node-a", (1,)), awake=True),
    ]


def test_v2_put_target_rejects_target_above_model_max_replicas():
    store = StateStore(FakeRedis())
    store.save([Binding("serve-a", "m1", Slot("node-a", (0,)), awake=False)], expected_version=0)
    service = ServiceManagerV2(registry(), store)

    try:
        service.put_model_target("m1", wake_replicas=3)
    except ValueError as exc:
        assert "max_replicas" in str(exc)
    else:
        raise AssertionError("expected target above max_replicas to fail")


def test_v2_put_target_allocates_new_binding_when_free_slot_exists():
    store = StateStore(FakeRedis())
    store.save([Binding("serve-a", "m1", Slot("node-a", (0,)), awake=True)], expected_version=0)
    service = ServiceManagerV2(registry_with_tp2(), store)

    result = service.put_model_target("tp2", wake_replicas=1)

    assert result == {
        "model": "tp2",
        "wake_replicas": 1,
        "version": 2,
        "actions": [{"action": "create", "serve_id": "tp2-1", "node": "node-a", "gpu_ids": [2, 3]}],
    }
    assert store.load().bindings == [
        Binding("serve-a", "m1", Slot("node-a", (0,)), awake=True),
        Binding("tp2-1", "tp2", Slot("node-a", (2, 3)), awake=True),
    ]


def test_v2_put_target_rejects_state_only_create_when_runtime_ops_are_enabled():
    from tre_sm.allocator.topology import K8sPodSnapshot

    store = StateStore(FakeRedis())
    original = [Binding("serve-a", "m1", Slot("node-a", (0,)), awake=True)]
    store.save(original, expected_version=0)
    service = ServiceManagerV2(
        registry(),
        store,
        runtime_ops=FakeRuntimeOps(
            [
                K8sPodSnapshot(
                    name="serve-a",
                    model="m1",
                    node="node-a",
                    env={"CUDA_VISIBLE_DEVICES": "0"},
                    pod_ip="10.0.0.1",
                )
            ]
        ),
        vllm_ops=FakeVllmOps(),
    )

    try:
        service.put_model_target("m1", wake_replicas=2)
    except ValueError as exc:
        assert "runtime create is not implemented" in str(exc)
    else:
        raise AssertionError("expected runtime create to fail")
    assert store.load().bindings == original


def test_v2_fastapi_routes_delegate_to_service_layer():
    store = StateStore(FakeRedis())
    store.save(
        [
            Binding("serve-a", "m1", Slot("node-a", (0,)), awake=True),
            Binding("serve-b", "m1", Slot("node-a", (1,)), awake=False),
        ],
        expected_version=0,
    )
    client = TestClient(create_app(ServiceManagerV2(registry(), store)))

    assert client.get("/healthz").json() == {"ok": True}
    assert client.get("/v2/state").json()["version"] == 1

    first = client.put("/v2/models/m1/target", json={"wake_replicas": 2})
    second = client.put("/v2/models/m1/target", json={"wake_replicas": 2})

    assert first.status_code == 200
    assert first.json()["actions"] == [{"action": "wake", "serve_id": "serve-b"}]
    assert second.status_code == 200
    assert second.json()["actions"] == []


def test_v2_put_routable_is_idempotent_and_persists_hidden_pods():
    store = StateStore(FakeRedis())
    store.save(
        [
            Binding("serve-a", "m1", Slot("node-a", (0,)), awake=True),
            Binding("serve-b", "m1", Slot("node-a", (1,)), awake=True),
        ],
        expected_version=0,
    )
    service = ServiceManagerV2(registry(), store)

    first = service.put_model_routable("m1", hidden_pods=["serve-b"])
    second = service.put_model_routable("m1", hidden_pods=["serve-b"])

    assert first == {
        "model": "m1",
        "hidden_pods": ["serve-b"],
        "version": 2,
        "actions": [{"action": "hide", "serve_id": "serve-b"}],
    }
    assert second == {
        "model": "m1",
        "hidden_pods": ["serve-b"],
        "version": 2,
        "actions": [],
    }
    state = service.get_state()
    assert state["bindings"][1]["hidden"] is True


def test_v2_routable_route_delegates_to_service_layer():
    store = StateStore(FakeRedis())
    store.save(
        [
            Binding("serve-a", "m1", Slot("node-a", (0,)), awake=True),
            Binding("serve-b", "m1", Slot("node-a", (1,)), awake=True),
        ],
        expected_version=0,
    )
    client = TestClient(create_app(ServiceManagerV2(registry(), store)))

    first = client.put("/v2/models/m1/routable", json={"hidden_pods": ["serve-b"]})
    second = client.put("/v2/models/m1/routable", json={"hidden_pods": ["serve-b"]})

    assert first.status_code == 200
    assert first.json()["actions"] == [{"action": "hide", "serve_id": "serve-b"}]
    assert second.status_code == 200
    assert second.json()["actions"] == []


def test_v2_put_routable_writes_runtime_labels_when_enabled():
    store = StateStore(FakeRedis())
    store.save(
        [
            Binding("serve-a", "m1", Slot("node-a", (0,)), awake=True),
            Binding("serve-b", "m1", Slot("node-a", (1,)), awake=True),
        ],
        expected_version=0,
    )
    runtime_ops = FakeRuntimeOps([])
    service = ServiceManagerV2(registry(), store, runtime_ops=runtime_ops)

    hidden = service.put_model_routable("m1", hidden_pods=["serve-b"])
    unhidden = service.put_model_routable("m1", hidden_pods=[])

    assert hidden["actions"] == [{"action": "hide", "serve_id": "serve-b"}]
    assert unhidden["actions"] == [{"action": "unhide", "serve_id": "serve-b"}]
    assert runtime_ops.annotations == [("serve-b", "hidden"), ("serve-b", "awake")]


def test_v2_reconcile_updates_state_from_pod_reality():
    store = StateStore(FakeRedis())
    store.save(
        [Binding("serve-a", "m1", Slot("node-a", (1,)), awake=False)],
        expected_version=0,
    )
    service = ServiceManagerV2(
        registry(),
        store,
        k8s_client=FakeK8sClient(
            [
                PodRecord(
                    serve_id="serve-a",
                    model="m1",
                    node="node-a",
                    cuda_visible_devices="0",
                    state="awake",
                )
            ]
        ),
    )

    result = service.reconcile()

    assert result["version"] == 2
    assert result["warnings"] == ["serve-a: pod reality overrides persisted binding"]
    assert store.load().bindings == [Binding("serve-a", "m1", Slot("node-a", (0,)), awake=True)]


def test_v2_reconcile_route_delegates_to_service_layer():
    store = StateStore(FakeRedis())
    store.save(
        [Binding("serve-a", "m1", Slot("node-a", (1,)), awake=False)],
        expected_version=0,
    )
    client = TestClient(
        create_app(
            ServiceManagerV2(
                registry(),
                store,
                k8s_client=FakeK8sClient(
                    [
                        PodRecord(
                            serve_id="serve-a",
                            model="m1",
                            node="node-a",
                            cuda_visible_devices="0",
                            state="awake",
                        )
                    ]
                ),
            )
        )
    )

    response = client.post("/v2/reconcile")

    assert response.status_code == 200
    assert response.json()["version"] == 2
    assert response.json()["warnings"] == ["serve-a: pod reality overrides persisted binding"]


def test_v2_put_target_calls_vllm_and_pod_annotations_for_existing_bindings():
    from tre_sm.allocator.topology import K8sPodSnapshot

    store = StateStore(FakeRedis())
    store.save(
        [
            Binding("serve-a", "m1", Slot("node-a", (0,)), awake=True),
            Binding("serve-b", "m1", Slot("node-a", (1,)), awake=False),
        ],
        expected_version=0,
    )
    runtime_ops = FakeRuntimeOps(
        [
            K8sPodSnapshot(
                name="serve-a",
                model="m1",
                node="node-a",
                env={"CUDA_VISIBLE_DEVICES": "0"},
                pod_ip="10.0.0.1",
            ),
            K8sPodSnapshot(
                name="serve-b",
                model="m1",
                node="node-a",
                env={"CUDA_VISIBLE_DEVICES": "1"},
                pod_ip="10.0.0.2",
            ),
        ]
    )
    vllm_ops = FakeVllmOps()
    service = ServiceManagerV2(registry(), store, runtime_ops=runtime_ops, vllm_ops=vllm_ops)

    down = service.put_model_target("m1", wake_replicas=0)
    up = service.put_model_target("m1", wake_replicas=2)

    assert down["actions"] == [{"action": "sleep", "serve_id": "serve-a"}]
    assert up["actions"] == [
        {"action": "wake", "serve_id": "serve-a"},
        {"action": "wake", "serve_id": "serve-b"},
    ]
    assert vllm_ops.calls == [
        ("sleep", "10.0.0.1", 8000),
        ("wake_up", "10.0.0.1", 8000),
        ("wake_up", "10.0.0.2", 8000),
    ]
    assert runtime_ops.annotations == [
        ("serve-a", "sleeping"),
        ("serve-a", "awake"),
        ("serve-b", "awake"),
    ]
    assert store.load().bindings == [
        Binding("serve-a", "m1", Slot("node-a", (0,)), awake=True),
        Binding("serve-b", "m1", Slot("node-a", (1,)), awake=True),
    ]


def test_v2_put_target_treats_matching_state_conflict_after_runtime_action_as_success():
    from tre_sm.allocator.topology import K8sPodSnapshot
    from tre_sm.state.store import StateConflict, StateSnapshot

    class ConflictingStore:
        def __init__(self):
            self.current = StateSnapshot(
                version=1,
                bindings=[Binding("serve-a", "m1", Slot("node-a", (0,)), awake=True)],
            )

        def load(self):
            return self.current

        def save(self, bindings, *, expected_version):
            self.current = StateSnapshot(
                version=2,
                bindings=[Binding("serve-a", "m1", Slot("node-a", (0,)), awake=False)],
            )
            raise StateConflict(expected_version=expected_version, current_version=2)

    runtime_ops = FakeRuntimeOps(
        [
            K8sPodSnapshot(
                name="serve-a",
                model="m1",
                node="node-a",
                env={"CUDA_VISIBLE_DEVICES": "0"},
                pod_ip="10.0.0.1",
            )
        ]
    )
    vllm_ops = FakeVllmOps()
    service = ServiceManagerV2(registry(), ConflictingStore(), runtime_ops=runtime_ops, vllm_ops=vllm_ops)

    result = service.put_model_target("m1", wake_replicas=0)

    assert result == {
        "model": "m1",
        "wake_replicas": 0,
        "version": 2,
        "actions": [{"action": "sleep", "serve_id": "serve-a"}],
    }
    assert vllm_ops.calls == [("sleep", "10.0.0.1", 8000)]
    assert runtime_ops.annotations == [("serve-a", "sleeping")]
