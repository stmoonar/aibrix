from __future__ import annotations

import json
import threading
from dataclasses import replace

import pytest
from fastapi.testclient import TestClient

from tre_common.registry import ClusterTopology, ModelSpec, NodeSpec, Registry, SloSpec, TrsParams
from tre_sm.allocator.slots import Binding, Slot
from tre_sm.allocator.topology import GPU_IDS_ANNOTATION, K8sPodSnapshot, STATE_ANNOTATION
from tre_sm.api.v2 import ServiceManagerV2, create_app
from tre_sm.app import create_service_app
from tre_sm.ops.drain import (
    DrainConfig,
    SleepDrainer,
    drain_timeout_s,
    parse_e2e_p95_s,
    parse_sleep_snapshot,
    parse_vllm_load,
)
from tre_sm.ops.k8s_ops import ModelDeploymentRecord
from tre_sm.ops.vllm_ops import VllmOps
from tre_sm.state.fleet_repair import FleetRepairExecutor
from tre_sm.state.operations import _CURRENT_OPERATION
from tre_sm.state.store import StateStore


SNAPSHOT_TEXT = json.dumps(
    [
        {
            "request_id": "chatcmpl-req_000016",
            "priority": 1,
            "status": 9,
            "stop_reason": None,
            "arrival_time": 1768118540.27,
            "client_index": 0,
            "all_token_len": 2349,
            "original_prompt_len": 2051,
            "generated_len": 298,
        },
        {"request_id": "chatcmpl-req_000017", "generated_len": 3},
    ]
)


def metrics_text(running: int, waiting: int, *, histogram: dict | None = None) -> str:
    lines = [
        "# HELP vllm:num_requests_running Number of requests in model execution batches.",
        "# TYPE vllm:num_requests_running gauge",
        f'vllm:num_requests_running{{engine="0",model_name="m1"}} {float(running)}',
        f'vllm:num_requests_waiting{{engine="0",model_name="m1"}} {float(waiting)}',
    ]
    for le, count in (histogram or {}).items():
        lines.append(
            f'vllm:e2e_request_latency_seconds_bucket{{engine="0",le="{le}",model_name="m1"}} {float(count)}'
        )
    return "\n".join(lines) + "\n"


class FakeClock:
    def __init__(self):
        self.now = 1000.0
        self.sleeps: list[float] = []
        self._lock = threading.Lock()

    def monotonic(self) -> float:
        with self._lock:
            return self.now

    def sleep(self, seconds: float) -> None:
        with self._lock:
            self.sleeps.append(seconds)
            self.now += seconds


def cfg_on(**overrides) -> DrainConfig:
    """Hide + drain on (drain requires hide)."""
    return DrainConfig(enabled=True, hide_before_sleep=True, **overrides)


def cfg_hide_only(**overrides) -> DrainConfig:
    return DrainConfig(enabled=False, hide_before_sleep=True, **overrides)


class Result:
    def __init__(self, message="", status_code=200, success=True):
        self.message = message
        self.status_code = status_code
        self.success = success


class FakeRuntimeOps:
    def __init__(self, events, snapshots, *, unroutable_timeout=False):
        self.events = events
        self.snapshots = {snapshot.name: snapshot for snapshot in snapshots}
        self.unroutable_timeout = unroutable_timeout
        self.unroutable_kwargs = []

    def list_pod_snapshots(self, *, model=None):
        values = list(self.snapshots.values())
        if model is not None:
            values = [item for item in values if item.model == model]
        return values

    def write_binding_annotations(self, binding, *, state):
        self.events.append(("annotate", binding.serve_id, state))

    def wait_pod_unroutable(self, binding, *, timeout_s=30.0, interval_s=0.5):
        self.events.append(("wait_unroutable", binding.serve_id))
        self.unroutable_kwargs.append(timeout_s)
        if self.unroutable_timeout:
            raise TimeoutError(f"pod {binding.serve_id} remained routable before timeout")


class FakeVllmOps:
    """vLLM fake with scripted /metrics pages per pod IP."""

    def __init__(self, events, *, pages=None, sleep_message=""):
        self.events = events
        self.pages = {ip: list(items) for ip, items in (pages or {}).items()}
        self.sleep_message = sleep_message
        self.sleeping: dict[str, bool] = {}
        self.sleep_kwargs: list[dict] = []

    def metrics(self, pod_ip, *, port=None):
        self.events.append(("metrics", pod_ip))
        items = self.pages.get(pod_ip)
        if not items:
            return None
        return items.pop(0) if len(items) > 1 else items[0]

    def sleep(self, pod_ip, *, port=None, **kwargs):
        # kwargs is how the header travels: hidden=True <=> X-TRE-Hidden: 1.
        self.events.append(("sleep", pod_ip))
        self.sleep_kwargs.append(kwargs)
        self.sleeping[pod_ip] = True
        return Result(self.sleep_message)

    def wake_up(self, pod_ip, *, port=None):
        self.events.append(("wake_up", pod_ip))
        self.sleeping[pod_ip] = False
        return Result()

    def wait_until_ready(self, pod_ip, *, port=None):
        self.events.append(("wait_until_ready", pod_ip))
        return Result()

    def is_sleeping(self, pod_ip, *, port=None):
        return self.sleeping.get(pod_ip, False)


class NoMetricsVllmOps(FakeVllmOps):
    metrics = None  # type: ignore[assignment]


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


class FakeOperation:
    def __init__(self):
        self.phases = []

    def assert_active(self):
        return None

    def advance(self, phase, *, details=None):
        self.phases.append((phase, details))


def registry() -> Registry:
    topology = ClusterTopology(
        nodes=(NodeSpec(name="node-a", gpus=4, two_gpu_slots=((0, 1), (2, 3))),)
    )
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
                max_replicas=4,
                vllm_image="image",
                slo=slo,
                trs=trs,
            )
        ],
    )


def _pod(name, gpu, ip):
    return K8sPodSnapshot(
        name=name,
        model="m1",
        node="node-a",
        env={"CUDA_VISIBLE_DEVICES": str(gpu)},
        pod_ip=ip,
    )


def _two_awake_store():
    store = StateStore(FakeRedis())
    store.save(
        [
            Binding("serve-a", "m1", Slot("node-a", (0,)), awake=True),
            Binding("serve-b", "m1", Slot("node-a", (1,)), awake=True),
        ],
        expected_version=0,
    )
    return store


def _service(events, *, pages=None, enabled=True, clock=None, vllm_cls=FakeVllmOps,
             unroutable_timeout=False, sleep_message="", store=None, cfg=None):
    runtime = FakeRuntimeOps(
        events,
        [_pod("serve-a", 0, "10.0.0.1"), _pod("serve-b", 1, "10.0.0.2")],
        unroutable_timeout=unroutable_timeout,
    )
    vllm = vllm_cls(events, pages=pages, sleep_message=sleep_message)
    cfg = cfg or (cfg_on() if enabled else DrainConfig())
    drainer = None
    if enabled:
        clock = clock or FakeClock()
        drainer = SleepDrainer(
            runtime, vllm, cfg, monotonic=clock.monotonic, sleep=clock.sleep
        )
    service = ServiceManagerV2(
        registry(),
        store or _two_awake_store(),
        runtime_ops=runtime,
        vllm_ops=vllm,
        drain_config=cfg,
        sleep_drainer=drainer,
    )
    return service, runtime, vllm


# ---------------------------------------------------------------- flag off


def test_flag_off_put_target_call_order_is_unchanged():
    events: list = []
    service, _runtime, _vllm = _service(
        events,
        pages={"10.0.0.1": [metrics_text(5, 5)], "10.0.0.2": [metrics_text(5, 5)]},
        enabled=False,
        sleep_message=SNAPSHOT_TEXT,
    )

    result = service.put_model_target("m1", wake_replicas=1)

    (slept,) = result["actions"]
    ip = {"serve-a": "10.0.0.1", "serve-b": "10.0.0.2"}[slept["serve_id"]]
    # Exactly the legacy sequence: no hide, no unroutable wait, no scrape,
    # and the legacy sleep(pod_ip, port=8000) call without a header.
    assert events == [("sleep", ip), ("annotate", slept["serve_id"], "sleeping")]
    assert _vllm.sleep_kwargs == [{}]
    assert "sleep_outcomes" not in result
    audit = service.get_sleep_audit()
    assert audit["enabled"] is False
    assert audit["hide_before_sleep"] is False
    (record,) = audit["records"]
    assert record["drain_enabled"] is False
    assert record["aborted_count"] == 2
    assert record["sleep_snapshot"][0]["request_id"] == "chatcmpl-req_000016"
    assert "drained" not in record


def test_flag_off_does_not_journal_even_inside_operation():
    events: list = []
    service, _runtime, _vllm = _service(events, enabled=False, sleep_message="[]")
    operation = FakeOperation()
    token = _CURRENT_OPERATION.set(operation)
    try:
        service.put_binding_power("serve-a", awake=False)
    finally:
        _CURRENT_OPERATION.reset(token)
    assert operation.phases == []
    assert events == [("sleep", "10.0.0.1"), ("annotate", "serve-a", "sleeping")]


def test_disabled_config_builds_no_drainer():
    events: list = []
    runtime = FakeRuntimeOps(events, [_pod("serve-a", 0, "10.0.0.1")])
    vllm = FakeVllmOps(events)
    service = ServiceManagerV2(
        registry(),
        _two_awake_store(),
        runtime_ops=runtime,
        vllm_ops=vllm,
        drain_config=DrainConfig(enabled=False),
    )
    assert service.get_sleep_audit() == {
        "enabled": False,
        "hide_before_sleep": False,
        "drain_before_sleep": False,
        "records": [],
    }
    assert service._drain_markers is None
    assert "draining" not in service.get_state()


# ---------------------------------------------------------------- flag on


def test_flag_on_put_target_hides_all_first_then_drains_each_before_sleep():
    events: list = []
    pages = {
        "10.0.0.1": [metrics_text(1, 0), metrics_text(0, 0)],
        "10.0.0.2": [metrics_text(0, 0)],
    }
    service, runtime, _vllm = _service(events, pages=pages)

    result = service.put_model_target("m1", wake_replicas=0)

    assert sorted(
        action["serve_id"] for action in result["actions"] if action["action"] == "sleep"
    ) == ["serve-a", "serve-b"]
    assert [a["action"] for a in result["actions"]].count("hide") == 2
    hides = [i for i, event in enumerate(events) if event[0] == "annotate" and event[2] == "hidden"]
    waits = [i for i, event in enumerate(events) if event[0] == "wait_unroutable"]
    assert len(hides) == 2 and len(waits) == 2
    assert max(hides) < min(waits)
    for serve_id, ip in (("serve-a", "10.0.0.1"), ("serve-b", "10.0.0.2")):
        wait = events.index(("wait_unroutable", serve_id))
        scrape = events.index(("metrics", ip))
        sleep = events.index(("sleep", ip))
        annotate = events.index(("annotate", serve_id, "sleeping"))
        assert wait < scrape < sleep < annotate
    assert runtime.unroutable_kwargs == [30.0, 30.0]
    assert _vllm.sleep_kwargs == [{"hidden": True}, {"hidden": True}]
    records = service.get_sleep_audit()["records"]
    assert [record["drained"] for record in records] == [True, True]
    assert [item["outcome"] for item in result["sleep_outcomes"]] == ["slept", "slept"]


def test_drain_completes_early_when_queue_empties():
    events: list = []
    clock = FakeClock()
    pages = {"10.0.0.1": [metrics_text(2, 1), metrics_text(1, 0), metrics_text(0, 0)]}
    service, _runtime, _vllm = _service(events, pages=pages, clock=clock)

    service.put_binding_power("serve-a", awake=False)

    (record,) = service.get_sleep_audit()["records"]
    assert record["drained"] is True
    assert record["interrupted_running"] == 0
    assert record["interrupted_waiting"] == 0
    assert clock.sleeps == [1.0, 1.0]
    assert record["waited_s"] == pytest.approx(2.0)
    assert record["drain_timeout_s"] == 60.0
    assert events.count(("metrics", "10.0.0.1")) == 3


def test_drain_timeout_records_interrupted_counts_and_still_sleeps():
    events: list = []
    clock = FakeClock()
    pages = {"10.0.0.1": [metrics_text(3, 2)]}
    service, _runtime, vllm = _service(events, pages=pages, clock=clock)

    service.put_binding_power("serve-a", awake=False)

    (record,) = service.get_sleep_audit()["records"]
    assert record["drained"] is False
    assert record["interrupted_running"] == 3
    assert record["interrupted_waiting"] == 2
    assert record["waited_s"] == pytest.approx(60.0)
    assert sum(clock.sleeps) == pytest.approx(60.0)
    assert ("sleep", "10.0.0.1") in events
    assert vllm.sleeping["10.0.0.1"] is True


def test_drain_timeout_uses_twice_p95_from_histogram():
    events: list = []
    clock = FakeClock()
    # 100 requests, 95th lands mid-way in the (20, 40] bucket -> ~ 30 s.
    histogram = {"10.0": 50, "20.0": 90, "40.0": 100, "+Inf": 100}
    pages = {"10.0.0.1": [metrics_text(1, 0, histogram=histogram)]}
    service, _runtime, _vllm = _service(events, pages=pages, clock=clock)

    service.put_binding_power("serve-a", awake=False)

    (record,) = service.get_sleep_audit()["records"]
    assert record["p95_e2e_s"] == pytest.approx(30.0)
    assert record["drain_timeout_s"] == pytest.approx(60.0)
    assert record["waited_s"] == pytest.approx(60.0)


def test_timeout_clamp():
    cfg = cfg_on()
    assert drain_timeout_s(10.0, cfg) == 30.0
    assert drain_timeout_s(200.0, cfg) == 300.0
    assert drain_timeout_s(40.0, cfg) == 80.0
    assert drain_timeout_s(None, cfg) == 60.0
    assert drain_timeout_s(None, DrainConfig(default_timeout_s=10.0)) == 30.0


def test_metrics_unavailable_does_not_wait_blindly():
    for vllm_cls, pages in ((NoMetricsVllmOps, None), (FakeVllmOps, {})):
        events: list = []
        clock = FakeClock()
        service, _runtime, _vllm = _service(
            events, pages=pages, clock=clock, vllm_cls=vllm_cls
        )

        service.put_binding_power("serve-a", awake=False)

        (record,) = service.get_sleep_audit()["records"]
        assert record["metrics_available"] is False
        assert record["drained"] is False
        assert record["interrupted_running"] is None
        assert record["waited_s"] == 0.0
        assert clock.sleeps == []
        assert ("sleep", "10.0.0.1") in events


def test_metrics_lost_mid_drain_stops_polling():
    events: list = []
    clock = FakeClock()
    runtime = FakeRuntimeOps(events, [_pod("serve-a", 0, "10.0.0.1")])

    class Flaky(FakeVllmOps):
        def __init__(self, events):
            super().__init__(events)
            self.pages_left = [metrics_text(4, 0), None]

        def metrics(self, pod_ip, *, port=None):
            self.events.append(("metrics", pod_ip))
            return self.pages_left.pop(0)

    drainer = SleepDrainer(
        runtime, Flaky(events), cfg_on(),
        monotonic=clock.monotonic, sleep=clock.sleep,
    )
    record = drainer.drain(Binding("serve-a", "m1", Slot("node-a", (0,)), awake=True), "10.0.0.1")
    assert record["metrics_available"] is False
    assert record["interrupted_running"] == 4
    assert clock.sleeps == [1.0]


def test_wait_unroutable_timeout_is_tolerated():
    events: list = []
    pages = {"10.0.0.1": [metrics_text(0, 0)]}
    service, _runtime, _vllm = _service(events, pages=pages, unroutable_timeout=True)

    service.put_binding_power("serve-a", awake=False)

    (record,) = service.get_sleep_audit()["records"]
    assert record["unroutable_confirmed"] is False
    assert record["drained"] is True
    assert ("sleep", "10.0.0.1") in events


def test_sleep_snapshot_goes_into_audit_and_operation_journal():
    events: list = []
    pages = {"10.0.0.1": [metrics_text(0, 0)]}
    service, _runtime, _vllm = _service(events, pages=pages, sleep_message=SNAPSHOT_TEXT)
    operation = FakeOperation()
    token = _CURRENT_OPERATION.set(operation)
    try:
        service.put_binding_power("serve-a", awake=False)
    finally:
        _CURRENT_OPERATION.reset(token)

    (record,) = service.get_sleep_audit()["records"]
    assert record["action"] == "sleep"
    assert record["sleep_status_code"] == 200
    assert record["aborted_count"] == 2
    assert record["sleep_snapshot"][0]["generated_len"] == 298
    assert record["binding_id"] == "m1/node-a/0"
    assert record["ts"]
    phases = [phase for phase, _details in operation.phases]
    assert phases == ["sleep_draining", "sleep_drained", "sleep_committed"]
    assert operation.phases[1][1] == record


def test_non_json_sleep_response_is_kept_truncated():
    events: list = []
    pages = {"10.0.0.1": [metrics_text(0, 0)]}
    service, _runtime, _vllm = _service(events, pages=pages, sleep_message="x" * 5000)

    service.put_binding_power("serve-a", awake=False)

    (record,) = service.get_sleep_audit()["records"]
    assert record["sleep_snapshot"] == "x" * 4096
    assert record["aborted_count"] is None


def test_binding_power_sleep_path_hides_and_drains():
    events: list = []
    pages = {"10.0.0.1": [metrics_text(0, 0)]}
    service, _runtime, _vllm = _service(events, pages=pages)

    service.put_binding_power("serve-a", awake=False)

    assert events == [
        ("annotate", "serve-a", "hidden"),
        ("wait_unroutable", "serve-a"),
        ("metrics", "10.0.0.1"),
        ("sleep", "10.0.0.1"),
        ("annotate", "serve-a", "sleeping"),
    ]
    assert _vllm.sleep_kwargs == [{"hidden": True}]


def test_defrag_path_does_not_wait_unroutable_twice():
    events: list = []
    pages = {"10.0.0.2": [metrics_text(0, 0)]}

    class DefragRuntime(FakeRuntimeOps):
        def ensure_model_httproute(self, model):
            pass

        def delete_model_deployment(self, binding):
            self.events.append(("delete_deployment", binding.serve_id))
            self.snapshots.pop(binding.serve_id, None)

        def wait_pod_deleted(self, serve_id):
            pass

        def create_model_deployment(self, model, slot):
            new = K8sPodSnapshot(
                "serve-b-new", "m1", "node-a", {"CUDA_VISIBLE_DEVICES": "1"},
                annotations={"tre.aibrix.io/gpu-ids": "1"}, pod_ip="10.0.0.3",
            )
            self.snapshots[new.name] = new
            return new.name

        def wait_pod_ready(self, serve_id):
            return self.snapshots[serve_id]

    store = StateStore(FakeRedis())
    store.save(
        [
            Binding("serve-a", "m1", Slot("node-a", (0,)), awake=True),
            Binding("serve-b", "m1", Slot("node-a", (2,)), awake=True),
        ],
        expected_version=0,
    )
    runtime = DefragRuntime(events, [_pod("serve-a", 0, "10.0.0.1"), _pod("serve-b", 2, "10.0.0.2")])
    vllm = FakeVllmOps(events, pages=pages)
    clock = FakeClock()
    cfg = cfg_on()
    service = ServiceManagerV2(
        registry(), store, runtime_ops=runtime, vllm_ops=vllm, drain_config=cfg,
        sleep_drainer=SleepDrainer(runtime, vllm, cfg, monotonic=clock.monotonic, sleep=clock.sleep),
    )

    service.defrag(tp_size=2)

    assert events.count(("wait_unroutable", "serve-b")) == 1
    # Per-call drain: defrag is a direct sleep and never drains, even with the
    # TRE_SM_DRAIN_BEFORE_SLEEP default on (no /metrics scrape before /sleep).
    assert ("metrics", "10.0.0.2") not in events
    assert events.index(("wait_unroutable", "serve-b")) < events.index(("sleep", "10.0.0.2"))
    (record,) = service.get_sleep_audit()["records"]
    assert record["unroutable_confirmed"] is True
    assert record["drain_enabled"] is False and record["drain_budget_source"] == "call"
    assert vllm.sleep_kwargs == [{"hidden": True}]


# ---------------------------------------------------------------- fleet repair


class FleetRuntime(FakeRuntimeOps):
    def __init__(self, events, deployments, snapshots):
        super().__init__(events, snapshots)
        self.deployments = list(deployments)

    def list_model_deployments(self):
        return list(self.deployments)

    def write_binding_annotations(self, binding, *, state):
        super().write_binding_annotations(binding, state=state)
        snapshot = self.snapshots[binding.serve_id]
        annotations = dict(snapshot.annotations)
        annotations[STATE_ANNOTATION] = state
        self.snapshots[binding.serve_id] = replace(snapshot, annotations=annotations)

    def scale_model_deployment(self, name, *, replicas):
        raise AssertionError("no repair expected")

    def wait_deployment_pods_deleted(self, deployment_name):
        raise AssertionError("no repair expected")

    def wait_pod_ready(self, serve_id):
        raise AssertionError("no repair expected")


class FakeSafety:
    def wait_until_healthy(self, operation):
        return None


def test_fleet_repair_sleep_path_hides_then_sleeps_without_draining():
    events: list = []
    deployment = ModelDeploymentRecord("m1-node-a-gpu-0", "m1", "node-a", (0,), replicas=1)
    pod = K8sPodSnapshot(
        name="m1-node-a-gpu-0-pod",
        model="m1",
        node="node-a",
        env={"CUDA_VISIBLE_DEVICES": "0"},
        annotations={GPU_IDS_ANNOTATION: "0", STATE_ANNOTATION: "awake"},
        pod_ip="10.0.0.9",
        routable=True,
        ready=True,
    )
    runtime = FleetRuntime(events, [deployment], [pod])
    vllm = FakeVllmOps(events, pages={"10.0.0.9": [metrics_text(1, 0), metrics_text(0, 0)]})
    clock = FakeClock()
    from tre_sm.ops.drain import SleepAuditLog

    audit_log = SleepAuditLog()
    executor = FleetRepairExecutor(
        runtime_ops=runtime,
        vllm_ops=vllm,
        safety_gate=FakeSafety(),
        poll_interval_s=0,
        sleep=lambda _seconds: None,
        drainer=SleepDrainer(runtime, vllm, cfg_on(), monotonic=clock.monotonic, sleep=clock.sleep),
        sleep_audit=audit_log,
    )
    executor.run(
        FakeOperation(),
        awake_binding_ids=[],
        reconcile=lambda _strict: {},
        set_binding_power=lambda _binding_id, _awake: {},
        audit=lambda: {"healthy": True},
    )

    serve = pod.name
    # Per-call drain: fleet repair is a direct sleep (budget 0) even with the
    # TRE_SM_DRAIN_BEFORE_SLEEP default on - hide, wait unroutable, sleep.
    assert events[:4] == [
        ("annotate", serve, "hidden"),
        ("wait_unroutable", serve),
        ("sleep", "10.0.0.9"),
        ("annotate", serve, "sleeping"),
    ]
    (record,) = audit_log.records()
    assert record["drained"] is False and record["drain_enabled"] is False
    assert record["drain_budget_source"] == "call"
    assert vllm.sleep_kwargs == [{"hidden": True}]


def test_service_passes_drainer_to_fleet_repair():
    class Runtime(FleetRuntime):
        pass

    events: list = []
    runtime = Runtime(events, [], [])
    vllm = FakeVllmOps(events)
    service = ServiceManagerV2(
        registry(), _two_awake_store(), runtime_ops=runtime, vllm_ops=vllm,
        safety_gate=FakeSafety(), drain_config=cfg_on(),
    )
    assert service._fleet_repair is not None
    assert service._fleet_repair._drainer is service._drainer
    assert service._drainer is not None
    off = ServiceManagerV2(
        registry(), _two_awake_store(), runtime_ops=runtime, vllm_ops=vllm,
        safety_gate=FakeSafety(),
    )
    assert off._fleet_repair._drainer is None


# ---------------------------------------------------------------- config / parsers


def test_from_env_parsing():
    assert DrainConfig.from_env({}) == DrainConfig()
    assert DrainConfig.from_env({}).enabled is False
    assert DrainConfig.from_env({}).hide_before_sleep is False
    assert DrainConfig().sleep_deadline_s == 240.0
    hide = {"TRE_SM_HIDE_BEFORE_SLEEP": "1", "TRE_SM_ALLOW_DEFAULT_DRAIN": "1"}
    for truthy in ("1", "true", "YES", " on "):
        assert DrainConfig.from_env({**hide, "TRE_SM_DRAIN_BEFORE_SLEEP": truthy}).enabled is True
        assert DrainConfig.from_env({"TRE_SM_HIDE_BEFORE_SLEEP": truthy}).hide_before_sleep is True
    for falsy in ("0", "false", "off", ""):
        assert DrainConfig.from_env({**hide, "TRE_SM_DRAIN_BEFORE_SLEEP": falsy}).enabled is False
    cfg = DrainConfig.from_env(
        {
            **hide,
            "TRE_SM_DRAIN_BEFORE_SLEEP": "true",
            "TRE_SM_DRAIN_DEFAULT_S": "45",
            "TRE_SM_DRAIN_MIN_S": "10",
            "TRE_SM_DRAIN_MAX_S": "120.5",
            "TRE_SM_DRAIN_POLL_S": "0.5",
            "TRE_SM_SLEEP_DEADLINE_S": "200",
            "TRE_SM_UNROUTABLE_TIMEOUT_S": "20",
        }
    )
    assert cfg == DrainConfig(
        enabled=True,
        hide_before_sleep=True,
        default_timeout_s=45.0,
        min_timeout_s=10.0,
        max_timeout_s=120.5,
        poll_interval_s=0.5,
        sleep_deadline_s=200.0,
        unroutable_timeout_s=20.0,
    )
    with pytest.raises(ValueError):
        DrainConfig.from_env({"TRE_SM_DRAIN_MIN_S": "400"})
    with pytest.raises(ValueError):
        DrainConfig.from_env({"TRE_SM_DRAIN_POLL_S": "0"})
    with pytest.raises(ValueError):
        DrainConfig.from_env({"TRE_SM_DRAIN_MAX_S": "abc"})


def test_parse_vllm_load_sums_series_and_detects_absence():
    text = (
        'vllm:num_requests_running{engine="0",model_name="m1"} 2.0\n'
        'vllm:num_requests_running{engine="1",model_name="m1"} 1.0\n'
        'vllm:num_requests_waiting{engine="0",model_name="m1"} 4.0\n'
        'vllm:num_requests_waiting_by_reason{reason="x"} 9.0\n'
    )
    assert parse_vllm_load(text) == (3, 4)
    assert parse_vllm_load('vllm:num_requests_running{a="b"} 1\n') == (1, 0)
    assert parse_vllm_load("# nothing\nprocess_cpu_seconds_total 3\n") is None
    assert parse_vllm_load("") is None


def test_parse_e2e_p95_sums_label_sets_and_handles_edges():
    text = (
        'vllm:e2e_request_latency_seconds_bucket{engine="0",le="1.0"} 40\n'
        'vllm:e2e_request_latency_seconds_bucket{engine="0",le="2.0"} 50\n'
        'vllm:e2e_request_latency_seconds_bucket{engine="0",le="+Inf"} 50\n'
        'vllm:e2e_request_latency_seconds_bucket{engine="1",le="1.0"} 40\n'
        'vllm:e2e_request_latency_seconds_bucket{engine="1",le="2.0"} 50\n'
        'vllm:e2e_request_latency_seconds_bucket{engine="1",le="+Inf"} 50\n'
        'vllm:e2e_request_latency_seconds_count{engine="0"} 50\n'
    )
    # 100 total, target 95: bucket (1, 2] has 80 -> 100 -> 1 + 15/20.
    assert parse_e2e_p95_s(text) == pytest.approx(1.75)
    tail = (
        'vllm:e2e_request_latency_seconds_bucket{le="5.0"} 10\n'
        'vllm:e2e_request_latency_seconds_bucket{le="+Inf"} 100\n'
    )
    assert parse_e2e_p95_s(tail) == 5.0
    empty = 'vllm:e2e_request_latency_seconds_bucket{le="+Inf"} 0\n'
    assert parse_e2e_p95_s(empty) is None
    assert parse_e2e_p95_s(metrics_text(1, 1)) is None


def test_parse_sleep_snapshot():
    assert len(parse_sleep_snapshot(SNAPSHOT_TEXT)) == 2
    assert parse_sleep_snapshot("[]") == []
    assert parse_sleep_snapshot("") is None
    assert parse_sleep_snapshot("OK") is None
    assert parse_sleep_snapshot('{"a": 1}') is None


def test_vllm_ops_metrics_uses_transport_and_swallows_failures():
    class Response:
        def __init__(self, status_code, text):
            self.status_code = status_code
            self.text = text

    class Transport:
        def __init__(self, response=None, error=None):
            self.response = response
            self.error = error
            self.urls = []

        def get(self, url, *, timeout):
            self.urls.append(url)
            if self.error is not None:
                raise self.error
            return self.response

        def post(self, url, *, timeout):
            raise AssertionError("metrics must not POST")

    ok = Transport(Response(200, "vllm:num_requests_running 0\n"))
    assert VllmOps(http=ok).metrics("10.0.0.1") == "vllm:num_requests_running 0\n"
    assert ok.urls == ["http://10.0.0.1:8000/metrics"]
    assert VllmOps(http=Transport(Response(503, "busy"))).metrics("10.0.0.1") is None
    assert VllmOps(http=Transport(error=ConnectionError("down"))).metrics("10.0.0.1", port=9000) is None


# ---------------------------------------------------------------- endpoint


def test_sleep_audit_endpoint_reports_records_newest_last():
    events: list = []
    pages = {"10.0.0.1": [metrics_text(0, 0)], "10.0.0.2": [metrics_text(0, 0)]}
    service, _runtime, _vllm = _service(events, pages=pages, sleep_message="[]")
    client = TestClient(create_app(service))

    assert client.get("/v2/sleep-audit").json() == {
        "enabled": True,
        "hide_before_sleep": True,
        "drain_before_sleep": True,
        "records": [],
    }
    service.put_binding_power("serve-a", awake=False)
    service.put_binding_power("serve-b", awake=False)

    body = client.get("/v2/sleep-audit").json()
    assert body["enabled"] is True
    assert [record["serve_id"] for record in body["records"]] == ["serve-a", "serve-b"]
    last = client.get("/v2/sleep-audit", params={"limit": 1}).json()
    assert [record["serve_id"] for record in last["records"]] == ["serve-b"]
    assert client.get("/v2/sleep-audit", params={"limit": 0}).status_code == 400


def test_create_service_app_threads_drain_config():
    events: list = []
    runtime = FakeRuntimeOps(events, [_pod("serve-a", 0, "10.0.0.1")])
    vllm = FakeVllmOps(events)
    app = create_service_app(
        registry(),
        _two_awake_store(),
        runtime_ops=runtime,
        vllm_ops=vllm,
        drain_config=cfg_on(),
    )
    assert TestClient(app).get("/v2/sleep-audit").json()["enabled"] is True
    default_app = create_service_app(
        registry(), _two_awake_store(), runtime_ops=runtime, vllm_ops=vllm
    )
    assert TestClient(default_app).get("/v2/sleep-audit").json() == {
        "enabled": False,
        "hide_before_sleep": False,
        "drain_before_sleep": False,
        "records": [],
    }
