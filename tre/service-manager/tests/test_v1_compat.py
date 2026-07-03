from fastapi.testclient import TestClient

from tre_common.registry import ClusterTopology, ModelSpec, NodeSpec, Registry, SloSpec, TrsParams
from tre_sm.allocator.slots import Binding, Slot
from tre_sm.api.v2 import ServiceManagerV2, create_app
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


def registry():
    topology = ClusterTopology(nodes=(NodeSpec(name="node-a", gpus=4, two_gpu_slots=((0, 1), (2, 3))),))
    trs = TrsParams(0.04, 1.0, 2.625, 1.0, 0.5, 0.0, 0.8, 1.0, 1.25, 4.0, 0.05, 3)
    slo = SloSpec(1200, 100, 10000)
    return Registry(topology, [ModelSpec("m1", "/m1", 1, 0, 3, "image", slo, trs)])


def app_with_state(bindings):
    store = StateStore(FakeRedis())
    store.save(bindings, expected_version=0)
    return create_app(ServiceManagerV2(registry(), store))


def test_v1_models_replicas_returns_awake_counts_for_apa():
    client = TestClient(
        app_with_state(
            [
                Binding("serve-a", "m1", Slot("node-a", (0,)), awake=True),
                Binding("serve-b", "m1", Slot("node-a", (1,)), awake=False),
            ]
        )
    )

    response = client.post("/models_replicas", params={"models": "m1"})

    assert response.status_code == 200
    assert response.json() == {"m1": 1}


def test_v1_scale_service_applies_delta_through_v2_target():
    client = TestClient(
        app_with_state(
            [
                Binding("serve-a", "m1", Slot("node-a", (0,)), awake=True),
                Binding("serve-b", "m1", Slot("node-a", (1,)), awake=False),
                Binding("serve-c", "m1", Slot("node-a", (2,)), awake=False),
            ]
        )
    )

    response = client.post(
        "/scale_service",
        params={"model_name": "m1", "scale_type": "up", "scale_value": "2"},
    )

    assert response.status_code == 200
    assert response.json() == {"requested": 2, "actual": 2}
    assert client.post("/models_replicas", params={"models": "m1"}).json() == {"m1": 3}


def test_v1_wake_up_wakes_one_sleeping_replica_for_gateway():
    client = TestClient(
        app_with_state(
            [
                Binding("serve-a", "m1", Slot("node-a", (0,)), awake=True),
                Binding("serve-b", "m1", Slot("node-a", (1,)), awake=False),
            ]
        )
    )

    response = client.post("/wake_up", params={"model_name": "m1", "kind": "0", "queue_len": "7"})

    assert response.status_code == 200
    assert response.json()["success"] is True
    assert response.json()["delayed"] is False
    assert response.json()["strategy_type"] == "wake_up"
    assert response.json()["strategy"]["serves_to_wakeup"] == ["serve-b"]
