from __future__ import annotations

from fastapi.testclient import TestClient

from tre_common.registry import ClusterTopology, ModelSpec, NodeSpec, Registry, SloSpec, TrsParams
from tre_ui.app import create_ui_app


class FakeRedis:
    def __init__(self) -> None:
        self.hist = {
            "tre:v2:decision:hist:m1": [
                b'{"ts":1000,"window_end_ms":1000,"model":"m1","z_m":1.5,"state":"HEALTHY"}',
                b'{"ts":2000,"window_end_ms":2000,"model":"m1","z_m":0.6,"state":"CRITICAL"}',
            ]
        }
        self.gpu = {"tre:gpu_truth:node-a": b'{"node":"node-a","gpus":[{"uuid":"G0","used_mib":37000,"total_mib":40960}]}'}
        self.kv: dict[str, str] = {}
        self.stream = [(
            "1-0",
            {b"window_id": b"2000", b"model": b"m1", b"z_m": b"0.6", b"queue_len": b"4",
             b"decode_tps": b"90", b"prefill_tps": b"310", b"replicas_awake": b"1",
             b"replicas_target": b"2", b"tier": b"critical", b"action": b"scale_up"},
        )]

    def hgetall(self, key):
        if key == "tre:v2:decision:latest":
            return {
                b"ts_ms": b"123",
                b"loop": b"rescue",
                b"submitted": b"1",
                b"stale": b"false",
                b"model_states": b'{"m1":{"z_m":0.6,"state":"CRITICAL"}}',
                b"actions": b'[{"kind":"scale","model":"m1","delta":1,"reason":"critical_sleeping_capacity"}]',
                b"events": b'["critical_sleeping_capacity"]',
            }
        return {}  # safescale probes key etc.

    def zrangebyscore(self, key, minimum, maximum):
        return self.hist.get(key, [])

    def scan_iter(self, match):
        return iter(self.gpu.keys())

    def get(self, key):
        if key in self.gpu:
            return self.gpu[key]
        return self.kv.get(key)

    def set(self, key, value):
        self.kv[key] = value
        return True

    def rpush(self, key, value):
        self.kv.setdefault(key, []).append(value)
        return len(self.kv[key])

    def xrevrange(self, key, max="+", min="-", count=None):
        return list(reversed(self.stream))[:count] if count else list(reversed(self.stream))

    def xrange(self, key, min="-", max="+", count=None):
        start = str(min).lstrip("(")
        out = [e for e in self.stream if e[0] > start]
        return out[:count] if count else out


class FakeServiceManagerClient:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str, dict | None]] = []

    def get_state(self):
        return {"version": 7, "models": {"m1": {"awake": 1, "bound": 2}}, "bindings": []}

    def request(self, method, path, payload=None):
        self.calls.append((method, path, payload))
        if path == "/v2/fleet/state":
            return {"desired_version": 4, "observed_version": 4,
                    "desired": [], "observed": [], "mismatches": []}
        if path == "/v2/supervisor":
            return {"enabled": True, "running": True, "drift_observations": 0}
        if path.startswith("/v2/operations"):
            return {"operations": [{"id": "op-9", "status": "succeeded"}]}
        if path == "/v2/audit":
            return {"healthy": True, "version": 12, "issues": []}
        return {"ok": True, "version": 8, "warnings": []}


_REGISTRY_YAML = """
cluster:
  nodes: [{name: node-a, gpus: 4, two_gpu_slots: [[0, 1], [2, 3]]}]
models:
  - name: m1
    weights_path: /w/m1
    tp_size: 1
    min_replicas: 1
    max_replicas: 4
    vllm_image: img:1
    slo: {ttft_p95_ms: 500.0, tpot_p95_ms: 75.0, e2e_p95_ms: 12000.0}
    alt_thresholds: {queue_len: {theta: 6.5, direction: lower_is_healthier}}
    trs: {w_p: 0.08, w_d: 1.0, lambda_wait: 1.875, qmin: 1.0, ema_alpha: 0.25,
          theta_m: 738.0, tau_crit: 0.75, tau_low: 1.0, tau_high: 1.63, qsat: 4.0,
          epsat: 0.1, hsat: 4, ema_tau_ms: 20000}
"""


class FakeK8s:
    def __init__(self) -> None:
        self.cm = {"metadata": {"name": "tre-v2-registry", "resourceVersion": "100"},
                   "data": {"registry.yaml": _REGISTRY_YAML}}
        self.deploy = {"metadata": {"name": "tre-v2-controller", "generation": 3},
                       "spec": {"replicas": 1, "template": {"metadata": {"annotations": {}}}},
                       "status": {"observedGeneration": 3, "updatedReplicas": 1, "readyReplicas": 1, "conditions": []}}
        self.patches: list = []

    def get_configmap(self, name):
        return self.cm

    def replace_configmap(self, name, data, resource_version):
        assert resource_version == self.cm["metadata"]["resourceVersion"]
        self.cm = {"metadata": {"name": name, "resourceVersion": str(int(resource_version) + 1)}, "data": data}
        return self.cm

    def get_deployment(self, name):
        return self.deploy

    def patch_deployment(self, name, patch):
        self.patches.append(patch)
        anns = patch["spec"]["template"]["metadata"]["annotations"]
        self.deploy["spec"]["template"]["metadata"]["annotations"].update(anns)
        return self.deploy


def _registry() -> Registry:
    topology = ClusterTopology(nodes=(NodeSpec("node-a", 4, ((0, 1), (2, 3))),))
    trs = TrsParams(0.04, 1.0, 2.625, 1.0, 0.5, 0.0, 0.8, 1.0, 1.25, 4.0, 0.05, 3)
    slo = SloSpec(1200, 100, 10000)
    return Registry(topology, [ModelSpec("m1", "/weights/m1", 1, 0, 2, "image", slo, trs)])


def _client(k8s: FakeK8s | None = None) -> tuple[TestClient, FakeServiceManagerClient]:
    sm = FakeServiceManagerClient()
    app = create_ui_app(_registry(), FakeRedis(), sm, k8s)
    # Populate the sampler cache deterministically without spawning the background thread
    # (TestClient is used without the lifespan context, so startup does not fire).
    app.state.sampler.sample_once()
    return TestClient(app), sm


# ---- legacy per-request endpoints (unchanged) ----

def test_ui_cluster_aggregates_sm_and_topology() -> None:
    client, _ = _client()
    payload = client.get("/api/cluster").json()
    assert payload["service_manager"]["version"] == 7
    assert payload["topology"]["nodes"] == [{"name": "node-a", "gpus": 4, "two_gpu_slots": [[0, 1], [2, 3]]}]


def test_ui_operate_proxies_to_service_manager_and_audits() -> None:
    client, sm = _client()
    assert client.post("/api/ops/models/m1/target", json={"wake_replicas": 2}).json()["ok"] is True
    client.post("/api/ops/reconcile")
    assert ("PUT", "/v2/models/m1/target", {"wake_replicas": 2}) in sm.calls
    assert ("POST", "/v2/reconcile", None) in sm.calls
    assert client.post("/api/ops/models/nope/target", json={"wake_replicas": 1}).status_code == 404


# ---- sampler-backed live endpoints ----

def test_ui_snapshot_is_composite_and_cached() -> None:
    client, _ = _client()
    response = client.get("/api/snapshot")
    snap = response.json()
    assert snap["version"] >= 1
    assert snap["decision"]["latest"]["ts_ms"] == 123
    assert snap["models"]["m1"]["state"] == {"z_m": 0.6, "state": "CRITICAL"}
    assert [p["state"] for p in snap["models"]["m1"]["hist_tail"]] == ["HEALTHY", "CRITICAL"]
    assert snap["gpu_truth"]["nodes"][0]["node"] == "node-a"
    assert snap["sm"]["state"]["version"] == 7
    # ETag round-trip: unchanged version returns 304 with no body.
    etag = response.headers["etag"]
    assert client.get("/api/snapshot", headers={"if-none-match": etag}).status_code == 304


def test_ui_events_surfaced_from_decisions() -> None:
    client, _ = _client()
    events = client.get("/api/events").json()["events"]
    texts = [e["text"] for e in events]
    assert "critical_sleeping_capacity" in texts
    assert any(e["kind"] == "scale" and e["model"] == "m1" for e in events)


def test_ui_meta_exposes_registry_params() -> None:
    client, _ = _client()
    meta = client.get("/api/meta").json()
    assert meta["models"][0]["name"] == "m1"
    assert meta["models"][0]["trs"]["tau_crit"] == 0.8
    assert meta["topology"]["nodes"][0]["name"] == "node-a"


def test_ui_controller_mode_toggle() -> None:
    client, _ = _client()
    assert client.get("/api/ops/controller/mode").json()["mode"] == "active"
    assert client.post("/api/ops/controller/mode", json={"mode": "observe"}).json()["mode"] == "observe"
    assert client.get("/api/ops/controller/mode").json()["mode"] == "observe"
    assert client.post("/api/ops/controller/mode", json={"mode": "bogus"}).status_code == 400


def test_ui_serves_local_single_page_app_without_runtime_cdn() -> None:
    client, _ = _client()
    response = client.get("/")
    assert response.status_code == 200
    assert "TRE Console" in response.text
    assert "https://" not in response.text
    assert "cdn" not in response.text.lower()


# ---- param editing (P0-4B) ----

def test_params_unavailable_without_k8s() -> None:
    client, _ = _client()  # no k8s client
    assert client.get("/api/params").status_code == 503


def test_get_params_view_and_pending_restart() -> None:
    client, _ = _client(FakeK8s())
    body = client.get("/api/params").json()
    assert body["models"]["m1"]["editable"]["trs.theta_m"]["value"] == 738.0
    assert body["models"]["m1"]["editable"]["alt_thresholds.queue_len.theta"]["value"] == 6.5
    assert body["resource_version"] == "100"
    assert body["pending_restart"] is False  # no applied hash annotation yet


def test_put_params_validates_writes_and_restart_flow() -> None:
    k8s = FakeK8s()
    client, _ = _client(k8s)
    # first read seeds the applied baseline -> in sync
    assert client.get("/api/params").json()["pending_restart"] is False
    # invalid: out of bounds -> 422, no write
    bad = client.put("/api/params", json={"models": {"m1": {"trs": {"tau_crit": 9.9}}}})
    assert bad.status_code == 422
    assert k8s.cm["metadata"]["resourceVersion"] == "100"
    # valid edit -> CM rewritten, rv bumped, and now pending a restart
    ok = client.put(
        "/api/params",
        json={
            "expected_resource_version": "100",
            "models": {
                "m1": {
                    "trs": {"theta_m": 800.0},
                    "alt_thresholds": {
                        "queue_len": {
                            "theta": 7.25,
                            "direction": "lower_is_healthier",
                        }
                    },
                }
            },
        },
    )
    assert ok.status_code == 200
    assert "800.0" in k8s.cm["data"]["registry.yaml"]
    assert "7.25" in k8s.cm["data"]["registry.yaml"]
    assert ok.json()["pending_restart"] is True
    # stale rv -> 409
    stale = client.put("/api/params", json={"expected_resource_version": "100", "models": {"m1": {"trs": {"theta_m": 810.0}}}})
    assert stale.status_code == 409
    # restart stamps the hash annotation + applied baseline -> pending clears
    r = client.post("/api/ops/controller/restart", json={"reason": "apply theta"})
    assert r.status_code == 200 and r.json()["ok"] is True
    assert k8s.patches and "tre.dev/params-hash" in k8s.patches[0]["spec"]["template"]["metadata"]["annotations"]
    assert client.get("/api/params").json()["pending_restart"] is False


def test_rollout_state_reports_ready() -> None:
    client, _ = _client(FakeK8s())
    assert client.get("/api/ops/controller/rollout").json()["state"] == "ready"


# ---- timeline endpoint (Task 2) ----

def test_signal_timeline_returns_points_for_a_known_model() -> None:
    client, _ = _client()

    payload = client.get("/api/signal/timeline?model=m1").json()

    assert payload["model"] == "m1"
    assert payload["points"][0]["queue_len"] == 4.0
    assert payload["points"][0]["action"] == "scale_up"


def test_signal_timeline_rejects_an_unknown_model() -> None:
    client, _ = _client()

    assert client.get("/api/signal/timeline?model=nope").status_code == 404


def test_signal_timeline_honours_since_ms() -> None:
    client, _ = _client()

    assert client.get("/api/signal/timeline?model=m1&since_ms=9999").json()["points"] == []


# ---- manual audit (Task 3) ----

def test_audit_runs_only_when_explicitly_posted() -> None:
    """/v2/audit lists k8s pods and probes every vLLM pod, so it must be manual."""
    client, sm = _client()

    assert client.get("/api/ops/audit").json()["ran_at_ms"] is None
    assert not any("audit" in path for _, path, _ in sm.calls)

    posted = client.post("/api/ops/audit")

    assert posted.status_code == 200
    assert posted.json()["result"]["healthy"] is True
    assert posted.json()["ran_at_ms"] is not None
    assert client.get("/api/ops/audit").json()["result"]["version"] == 12


def test_audit_surfaces_service_manager_failures() -> None:
    class FailingSm(FakeServiceManagerClient):
        def request(self, method, path, payload=None):
            if path == "/v2/audit":
                raise RuntimeError("service-manager unreachable")
            return super().request(method, path, payload)

    app = create_ui_app(_registry(), FakeRedis(), FailingSm(), None)
    app.state.sampler.sample_once()

    assert TestClient(app).post("/api/ops/audit").status_code == 502


# ---- static mount (Task 4) ----

def test_static_assets_are_served_from_the_static_mount() -> None:
    client, _ = _client()

    assert client.get("/static/style.css").status_code == 200
    assert client.get("/static/js/main.js").status_code == 200


def test_index_still_serves_the_console_shell() -> None:
    client, _ = _client()

    response = client.get("/")

    assert response.status_code == 200
    assert "/static/" in response.text

