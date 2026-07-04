from __future__ import annotations

from fastapi.testclient import TestClient

from tre_common.registry import ClusterTopology, ModelSpec, NodeSpec, Registry, SloSpec, TrsParams
from tre_ui.app import create_ui_app


class FakeRedis:
    def hgetall(self, key):
        assert key == "tre:v2:decision:latest"
        return {
            b"ts_ms": b"123",
            b"payload": b'{"actions":[{"model":"m1","delta":1}],"reason":"critical"}',
        }


class FakeServiceManagerClient:
    def get_state(self):
        return {
            "version": 7,
            "bindings": [
                {"serve_id": "m1-a", "model": "m1", "slot": {"node": "node-a", "gpu_ids": [0]}, "awake": True},
            ],
        }


def _registry() -> Registry:
    topology = ClusterTopology(nodes=(NodeSpec("node-a", 4, ((0, 1), (2, 3))),))
    trs = TrsParams(0.04, 1.0, 2.625, 1.0, 0.5, 0.0, 0.8, 1.0, 1.25, 4.0, 0.05, 3)
    slo = SloSpec(1200, 100, 10000)
    return Registry(topology, [ModelSpec("m1", "/weights/m1", 1, 0, 2, "image", slo, trs)])


def test_ui_backend_aggregates_cluster_state_from_service_manager_and_registry() -> None:
    client = TestClient(create_ui_app(_registry(), FakeRedis(), FakeServiceManagerClient()))

    payload = client.get("/api/cluster").json()

    assert payload["service_manager"]["version"] == 7
    assert payload["topology"]["nodes"] == [{"name": "node-a", "gpus": 4, "two_gpu_slots": [[0, 1], [2, 3]]}]


def test_ui_backend_exposes_model_parameters_and_latest_decision() -> None:
    client = TestClient(create_ui_app(_registry(), FakeRedis(), FakeServiceManagerClient()))

    models = client.get("/api/models").json()
    decision = client.get("/api/decision/latest").json()
    experiments = client.get("/api/experiments").json()

    assert models["models"][0]["name"] == "m1"
    assert models["models"][0]["trs"]["lambda_wait"] == 2.625
    assert decision == {"ts_ms": 123, "payload": {"actions": [{"model": "m1", "delta": 1}], "reason": "critical"}}
    assert experiments == {"available": False, "reason": "orchestrate integration is stubbed in P8"}
