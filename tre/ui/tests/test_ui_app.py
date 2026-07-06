from __future__ import annotations

from fastapi.testclient import TestClient

from tre_common.registry import ClusterTopology, ModelSpec, NodeSpec, Registry, SloSpec, TrsParams
from tre_ui.app import create_ui_app


class FakeRedis:
    def __init__(self) -> None:
        self.hist = {
            "tre:v2:decision:hist:m1": [
                b'{"ts":1000,"model":"m1","z_m":1.5,"state":"HEALTHY"}',
                b'{"ts":2000,"model":"m1","z_m":0.6,"state":"CRITICAL"}',
            ]
        }
        self.gpu = {"tre:gpu_truth:node-a": b'{"node":"node-a","gpus":[{"uuid":"G0","used_mib":37000,"total_mib":40960}]}'}

    def hgetall(self, key):
        assert key == "tre:v2:decision:latest"
        return {
            b"ts_ms": b"123",
            b"loop": b"rescue",
            b"submitted": b"1",
            b"stale": b"false",
            b"model_states": b'{"m1":{"z_m":0.6,"state":"CRITICAL"}}',
            b"actions": b'[{"kind":"scale","model":"m1","delta":1}]',
            b"events": b'["critical_sleeping_capacity"]',
        }

    def zrangebyscore(self, key, minimum, maximum):
        return self.hist.get(key, [])

    def scan_iter(self, match):
        return iter(self.gpu.keys())

    def get(self, key):
        return self.gpu.get(key)


class FakeServiceManagerClient:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str, dict | None]] = []

    def get_state(self):
        return {"version": 7, "models": {"m1": {"awake": 1, "bound": 2}}, "bindings": []}

    def request(self, method, path, payload=None):
        self.calls.append((method, path, payload))
        return {"ok": True, "version": 8, "warnings": []}


def _registry() -> Registry:
    topology = ClusterTopology(nodes=(NodeSpec("node-a", 4, ((0, 1), (2, 3))),))
    trs = TrsParams(0.04, 1.0, 2.625, 1.0, 0.5, 0.0, 0.8, 1.0, 1.25, 4.0, 0.05, 3)
    slo = SloSpec(1200, 100, 10000)
    return Registry(topology, [ModelSpec("m1", "/weights/m1", 1, 0, 2, "image", slo, trs)])


def _client() -> tuple[TestClient, FakeServiceManagerClient]:
    sm = FakeServiceManagerClient()
    return TestClient(create_ui_app(_registry(), FakeRedis(), sm)), sm


def test_ui_cluster_aggregates_sm_and_topology() -> None:
    client, _ = _client()
    payload = client.get("/api/cluster").json()
    assert payload["service_manager"]["version"] == 7
    assert payload["topology"]["nodes"] == [{"name": "node-a", "gpus": 4, "two_gpu_slots": [[0, 1], [2, 3]]}]


def test_ui_models_and_latest_decision() -> None:
    client, _ = _client()
    models = client.get("/api/models").json()
    decision = client.get("/api/decision/latest").json()
    assert models["models"][0]["name"] == "m1"
    assert models["models"][0]["trs"]["tau_crit"] == 0.8
    assert decision["ts_ms"] == 123 and decision["loop"] == "rescue"
    assert decision["model_states"]["m1"]["state"] == "CRITICAL"
    assert decision["actions"][0]["model"] == "m1"


def test_ui_signal_history_reads_zset() -> None:
    client, _ = _client()
    body = client.get("/api/signal/history?model=m1").json()
    assert body["model"] == "m1"
    assert [p["state"] for p in body["points"]] == ["HEALTHY", "CRITICAL"]
    assert client.get("/api/signal/history?model=nope").status_code == 404


def test_ui_gputruth_reads_agent_keys() -> None:
    client, _ = _client()
    body = client.get("/api/gputruth").json()
    assert body["nodes"][0]["node"] == "node-a"
    assert body["nodes"][0]["gpus"][0]["used_mib"] == 37000


def test_ui_operate_proxies_to_service_manager_and_audits() -> None:
    client, sm = _client()
    assert client.post("/api/ops/models/m1/target", json={"wake_replicas": 2}).json()["ok"] is True
    client.post("/api/ops/reconcile")
    assert ("PUT", "/v2/models/m1/target", {"wake_replicas": 2}) in sm.calls
    assert ("POST", "/v2/reconcile", None) in sm.calls
    assert client.post("/api/ops/models/nope/target", json={"wake_replicas": 1}).status_code == 404


def test_ui_serves_local_single_page_app_without_runtime_cdn() -> None:
    client, _ = _client()
    response = client.get("/")
    assert response.status_code == 200
    assert "TRE Console" in response.text
    assert "https://" not in response.text
    assert "cdn" not in response.text.lower()
