"""Async SM operations (TRE_SM_ASYNC_OPS): 202 + operation id, per-model latest-wins,
restart recovery. Design note 20260924-reissue-sidecar.md section 3.4."""
from __future__ import annotations

import importlib.util
import threading
import time
from contextlib import contextmanager
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from tre_sm.allocator.slots import Binding, Slot
from tre_sm.api.v2 import ServiceManagerV2, create_app
from tre_sm.ops.drain import DrainConfig, SleepDrainer
from tre_sm.state.async_ops import (
    AsyncOperationManager,
    AsyncOpJournal,
    AsyncOpsConfig,
)
from tre_sm.state.drain_markers import DrainMarker, DrainMarkerStore
from tre_sm.state.operations import OperationBusy
from tre_sm.state.supervisor import FleetSupervisor

_HERE = Path(__file__).resolve().parent


def _load(name: str):
    spec = importlib.util.spec_from_file_location(f"_async_helpers_{name}", _HERE / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_drain = _load("test_drain_before_sleep")
_staged = _load("test_staged_sleep")
FakeClock = _drain.FakeClock
FakeRuntimeOps = _drain.FakeRuntimeOps
FakeVllmOps = _drain.FakeVllmOps
FakeRedis = _drain.FakeRedis
FakeOperation = _drain.FakeOperation
metrics_text = _drain.metrics_text
registry = _drain.registry
_pod = _drain._pod
CallbackVllm = _staged.CallbackVllm
FailingSleepVllm = _staged.FailingSleepVllm
_store = _staged._store
_bindings = _staged._bindings

WAIT_S = 10.0
ASYNC_ON = AsyncOpsConfig(enabled=True, lock_retry_s=0.01, lock_wait_s=5.0)


def _three_bindings(redis=None):
    return _store(
        [
            Binding("serve-a", "m1", Slot("node-a", (0,)), awake=True),
            Binding("serve-b", "m1", Slot("node-a", (1,)), awake=True),
            Binding("serve-c", "m1", Slot("node-a", (2,)), awake=False),
        ],
        redis,
    )


def _amake(events, *, hide=True, drain=False, pages=None, vllm=None, store=None, coordinator=None,
           async_config=ASYNC_ON, journal=None, markers=None, **cfg):
    runtime = FakeRuntimeOps(
        events,
        [_pod("serve-a", 0, "10.0.0.1"), _pod("serve-b", 1, "10.0.0.2"), _pod("serve-c", 2, "10.0.0.3")],
    )
    vllm = vllm or FakeVllmOps(events, pages=pages)
    config = DrainConfig(enabled=drain, hide_before_sleep=hide, **cfg)
    drainer = None
    if hide:
        clock = FakeClock()
        drainer = SleepDrainer(runtime, vllm, config, monotonic=clock.monotonic, sleep=clock.sleep)
    service = ServiceManagerV2(
        registry(),
        store or _three_bindings(),
        runtime_ops=runtime,
        vllm_ops=vllm,
        drain_config=config,
        sleep_drainer=drainer,
        operation_coordinator=coordinator,
        async_config=async_config,
        async_journal=journal,
        drain_markers=markers,
    )
    return service, runtime, vllm


def _wait(service, operation_id):
    assert service.wait_async_operation(operation_id, timeout_s=WAIT_S), "operation did not finish"
    return service.get_operation(operation_id)


# ------------------------------------------------------------------- flag off


def test_flag_off_async_request_is_served_synchronously_like_main():
    events: list = []
    service, _runtime, vllm = _amake(events, hide=False, async_config=AsyncOpsConfig())
    client = TestClient(create_app(service))

    response = client.put("/v2/bindings/serve-b/power?async=1", json={"awake": False})

    assert response.status_code == 200
    body = response.json()
    assert "async_operation" not in body
    assert body["actions"] == [{"action": "sleep", "serve_id": "serve-b"}]
    assert vllm.sleep_kwargs == [{}]
    assert client.get("/v2/async-operations").json() == {"enabled": False, "operations": []}
    assert service.async_enabled is False
    with pytest.raises(ValueError):
        service.submit_model_target("m1", wake_replicas=1)


def test_async_config_from_env():
    assert AsyncOpsConfig.from_env({}) == AsyncOpsConfig()
    assert AsyncOpsConfig.from_env({}).enabled is False
    cfg = AsyncOpsConfig.from_env({"TRE_SM_ASYNC_OPS": "true", "TRE_SM_ASYNC_LOCK_WAIT_S": "30"})
    assert cfg.enabled is True and cfg.lock_wait_s == 30.0
    with pytest.raises(ValueError):
        AsyncOpsConfig(orphan_after_s=1.0, heartbeat_s=5.0)


# ------------------------------------------------------------ 202 + lifecycle


def test_202_then_status_lifecycle_with_per_binding_results():
    events: list = []
    pages = {"10.0.0.2": [metrics_text(2, 0), metrics_text(0, 0)]}
    service, _runtime, vllm = _amake(events, pages=pages)
    client = TestClient(create_app(service))

    response = client.put(
        "/v2/bindings/serve-b/power", params={"async": "1"}, json={"awake": False, "drain_s": 20}
    )

    assert response.status_code == 202, response.text
    accepted = response.json()
    assert accepted["async_operation"] is True and accepted["status"] == "pending"
    assert accepted["plan"]["action"] == "sleep" and accepted["model"] == "m1"
    record = _wait(service, accepted["operation_id"])
    status = client.get(accepted["status_url"]).json()
    assert status["status"] == record["status"] == "succeeded"
    (binding,) = status["bindings"]
    assert binding["serve_id"] == "serve-b" and binding["outcome"] == "slept"
    assert binding["drained"] is True and binding["drained_s"] > 0 and binding["interrupted"] == 0
    assert binding["drain_budget_s"] == 20.0
    assert status["summary"]["slept"] == ["serve-b"]
    assert status["latency_s"] >= 0 and status["finished_at"]
    assert vllm.sleep_kwargs == [{"hidden": True}]
    assert _bindings(service)["serve-b"].awake is False


def test_prefer_header_and_drain_zero_model_target():
    events: list = []
    pages = {"10.0.0.2": [metrics_text(5, 0)]}
    service, _runtime, _vllm = _amake(events, drain=True, pages=pages)
    client = TestClient(create_app(service))

    response = client.put(
        "/v2/models/m1/target",
        headers={"Prefer": "respond-async"},
        json={"wake_replicas": 1, "drain_s": 0},
    )

    assert response.status_code == 202
    plan = response.json()["plan"]
    assert plan["direction"] == "down" and plan["serving"] == 2 and len(plan["sleep"]) == 1
    record = _wait(service, response.json()["operation_id"])
    assert record["status"] == "succeeded"
    (binding,) = record["bindings"]
    assert binding["drained"] is False and binding["drain_budget_s"] is None
    assert events.count(("metrics", "10.0.0.2")) == 0  # drain 0: no scrape at all


def test_validation_errors_are_synchronous_and_create_no_operation():
    events: list = []
    service, _runtime, _vllm = _amake(events)
    client = TestClient(create_app(service))

    assert client.put("/v2/models/m1/target?async=1", json={"wake_replicas": -1}).status_code == 400
    assert client.put("/v2/models/nope/target?async=1", json={"wake_replicas": 1}).status_code == 400
    assert client.put("/v2/bindings/ghost/power?async=1", json={"awake": True}).status_code == 400
    assert client.put(
        "/v2/bindings/serve-b/power?async=1", json={"awake": False, "drain_s": -2}
    ).status_code == 400
    assert service.list_async_operations() == []


def test_infeasible_wake_is_a_synchronous_409():
    events: list = []
    store = _store(
        [
            Binding("serve-a", "m1", Slot("node-a", (0,)), awake=True),
            Binding("serve-x", "m1", Slot("node-a", (0,)), awake=False),
        ]
    )
    service, _runtime, _vllm = _amake(events, store=store)
    client = TestClient(create_app(service))

    response = client.put("/v2/bindings/serve-x/power?async=1", json={"awake": True})

    assert response.status_code == 409
    assert service.list_async_operations() == []


def test_failed_sleep_marks_the_operation_failed_with_rolled_back_binding():
    events: list = []
    vllm = FailingSleepVllm(events, mode="refused", pages={"10.0.0.2": [metrics_text(0, 0)]})
    service, _runtime, _ = _amake(events, vllm=vllm)

    accepted = service.submit_binding_power("serve-b", awake=False, drain_s=0)
    record = _wait(service, accepted["operation_id"])

    assert record["status"] == "failed" and record["error_code"] == "sleep_commit_failed"
    (binding,) = record["bindings"]
    assert binding["outcome"] == "rolled_back"
    assert record["summary"]["errors"][0]["serve_id"] == "serve-b"
    assert _bindings(service)["serve-b"].awake is True
    assert _bindings(service)["serve-b"].hidden is False


# ------------------------------------------------ per-model rules / latest wins


def test_newer_target_during_a_drain_starts_at_once_and_reclaims_latest_wins():
    events: list = []
    holder: dict = {}

    def during_drain(_ip):
        service = holder["service"]
        newer = service.submit_model_target("m1", wake_replicas=2, drain_s=0)
        holder["newer"] = newer
        # It must not wait behind the drain: it completes while we are still in it.
        holder["newer_done"] = service.wait_async_operation(newer["operation_id"], timeout_s=WAIT_S)

    vllm = CallbackVllm(
        events,
        pages={"10.0.0.1": [metrics_text(3, 0)], "10.0.0.2": [metrics_text(3, 0)]},
        on_metrics=during_drain,
    )
    service, _runtime, _ = _amake(events, vllm=vllm)
    holder["service"] = service

    older = service.submit_model_target("m1", wake_replicas=1, drain_s=60)
    older_record = _wait(service, older["operation_id"])

    assert holder["newer_done"] is True
    newer_record = service.get_operation(holder["newer"]["operation_id"])
    assert newer_record["status"] == "succeeded"
    victim = older_record["summary"]["abandoned"][0]
    assert newer_record["summary"]["reclaimed"] == [victim]
    assert older_record["status"] == "superseded"
    assert [item["outcome"] for item in older_record["bindings"]] == ["abandoned_reclaimed"]
    assert not any(event[0] == "sleep" for event in events)
    assert service.get_state()["models"]["m1"]["awake"] == 2
    assert service._load_markers() == {}


def test_feasible_wake_does_not_wait_behind_another_bindings_drain():
    events: list = []
    release = threading.Event()
    seen: dict = {}

    def during_drain(_ip):
        service = seen["service"]
        wake = service.submit_binding_power("serve-c", awake=True)
        seen["wake_done_during_drain"] = service.wait_async_operation(
            wake["operation_id"], timeout_s=WAIT_S
        )
        seen["wake"] = service.get_operation(wake["operation_id"])
        release.set()

    vllm = CallbackVllm(
        events,
        pages={"10.0.0.2": [metrics_text(1, 0), metrics_text(0, 0)]},
        trigger_ip="10.0.0.2",
        on_metrics=during_drain,
    )
    service, _runtime, _ = _amake(events, vllm=vllm)
    seen["service"] = service

    sleep_op = service.submit_binding_power("serve-b", awake=False, drain_s=30)
    record = _wait(service, sleep_op["operation_id"])

    assert release.is_set()
    assert seen["wake_done_during_drain"] is True
    assert seen["wake"]["status"] == "succeeded" and seen["wake"]["summary"]["woken"] == ["serve-c"]
    assert record["status"] == "succeeded" and record["summary"]["slept"] == ["serve-b"]
    # the wake happened before the drained binding was put to sleep
    assert events.index(("wake_up", "10.0.0.3")) < events.index(("sleep", "10.0.0.2"))


def test_queued_operations_collapse_latest_wins():
    started = threading.Event()
    release = threading.Event()
    ran: list[str] = []

    def executor(record, progress):
        ran.append(record["request"]["n"])
        if record["request"]["n"] == 1:
            started.set()
            assert release.wait(WAIT_S)
        return "succeeded", {"summary": {"n": record["request"]["n"]}}

    manager = AsyncOperationManager(
        AsyncOpJournal(), executor, config=ASYNC_ON, instance="sm-1"
    )
    first = manager.submit(kind="model_target", model="m1", target_key="model:m1", request={"n": 1})
    assert started.wait(WAIT_S)
    second = manager.submit(kind="model_target", model="m1", target_key="model:m1", request={"n": 2})
    other = manager.submit(kind="binding_power", model="m1", target_key="binding:x", request={"n": 9})
    third = manager.submit(kind="model_target", model="m1", target_key="model:m1", request={"n": 3})
    assert manager.get(second["operation_id"])["status"] == "superseded"
    assert manager.get(second["operation_id"])["superseded_by"] == third["operation_id"]
    assert third["supersedes"] == [second["operation_id"]]
    release.set()
    for item in (first, other, third):
        assert manager.wait(item["operation_id"], timeout_s=WAIT_S)
    assert ran == [1, 9, 3]  # FIFO per model, the collapsed one never ran
    assert manager.get(third["operation_id"])["status"] == "succeeded"
    assert manager.active_models() == set()


def test_busy_writer_lock_before_phase_one_is_retried():
    events: list = []

    class BusyFirst:
        def __init__(self, busy):
            self.busy = busy

        @contextmanager
        def operation(self, kind, *, request=None):
            if self.busy > 0:
                self.busy -= 1
                raise OperationBusy("other-writer")
            yield FakeOperation()

    service, _runtime, _ = _amake(events, coordinator=BusyFirst(3), pages={"10.0.0.2": [metrics_text(0, 0)]})

    accepted = service.submit_binding_power("serve-b", awake=False, drain_s=0)
    record = _wait(service, accepted["operation_id"])

    assert record["status"] == "succeeded"
    assert _bindings(service)["serve-b"].awake is False


def test_busy_writer_lock_until_lock_wait_fails_the_operation():
    events: list = []

    class AlwaysBusy:
        @contextmanager
        def operation(self, kind, *, request=None):
            raise OperationBusy("other-writer")
            yield  # pragma: no cover

    service, _runtime, _ = _amake(
        events,
        coordinator=AlwaysBusy(),
        async_config=AsyncOpsConfig(enabled=True, lock_retry_s=0.01, lock_wait_s=0.05),
    )

    accepted = service.submit_binding_power("serve-b", awake=False, drain_s=0)
    record = _wait(service, accepted["operation_id"])

    assert record["status"] == "failed" and record["error_code"] == "writer_busy"
    assert record["phase1_done"] is False
    assert _bindings(service)["serve-b"].awake is True
    assert not any(event[0] in ("annotate", "sleep") for event in events)


def test_legacy_path_runs_async_too_when_hide_is_off():
    events: list = []
    service, _runtime, vllm = _amake(events, hide=False)

    accepted = service.submit_model_target("m1", wake_replicas=1)
    record = _wait(service, accepted["operation_id"])

    assert record["status"] == "succeeded"
    assert len(record["summary"]["slept"]) == 1
    assert vllm.sleep_kwargs == [{}]


# ---------------------------------------------------------- restart recovery


def _dead_record(operation_id, *, phase, instance="dead-sm", age_s=600.0):
    now = time.time()
    return {
        "operation_id": operation_id,
        "async": True,
        "kind": "binding_power",
        "model": "m1",
        "target_key": "binding:serve-a",
        "request": {"serve_id": "serve-a", "awake": False, "drain_s": 30},
        "status": "running" if phase != "queued" else "pending",
        "phase": phase,
        "instance": instance,
        "created_ts": now - age_s,
        "updated_ts": now - age_s,
        "heartbeat_ts": now - age_s,
    }


def test_restart_recovery_finishes_orphaned_draining_operation():
    events: list = []
    redis = FakeRedis()
    journal = AsyncOpJournal(redis)
    markers = DrainMarkerStore(redis)
    journal.save(_dead_record("op-draining", phase="draining"))
    journal.save(_dead_record("op-queued", phase="queued"))
    journal.save(_dead_record("op-live", phase="draining", instance="other-live-sm", age_s=1.0))
    markers.save(
        {
            "m1/node-a/0": DrainMarker(
                binding_id="m1/node-a/0", serve_id="serve-a", model="m1", token="t-dead",
                instance="dead-sm", started_at=time.time() - 10, deadline_at=time.time() + 200,
                reason="binding_power", async_op_id="op-draining",
            )
        }
    )
    store = _store(
        [
            Binding("serve-a", "m1", Slot("node-a", (0,)), awake=True, hidden=True),
            Binding("serve-b", "m1", Slot("node-a", (1,)), awake=True),
        ],
        redis,
    )
    service, _runtime, vllm = _amake(events, store=store, journal=journal, markers=markers)
    # Not stale by deadline + grace yet: only the orphaned operation makes it so.
    assert service._marker_is_stale(markers.load()["m1/node-a/0"]) is False

    result = service.recover_orphaned_async_ops()

    assert sorted(result["orphaned"]) == ["op-draining", "op-queued"]
    assert [item["outcome"] for item in result["drain_recovery"]["recovered"]] == ["slept"]
    draining = service.get_operation("op-draining")
    assert draining["status"] == "failed" and draining["phase"] == "orphaned"
    assert draining["orphaned_in_phase"] == "draining"
    queued = service.get_operation("op-queued")
    assert queued["status"] == "failed" and "never started" in queued["error"]
    assert service.get_operation("op-live")["status"] == "running"  # other live instance
    # the staged sleep of the dead instance was finished (desired state wins)
    assert _bindings(service)["serve-a"].awake is False
    assert vllm.sleep_kwargs == [{"hidden": True}]
    assert markers.load() == {}
    assert service._orphaned_async_ops == set()


def test_supervisor_tick_recovers_orphaned_async_ops_before_stale_drains():
    calls: list[str] = []

    class Service:
        def converge_startups(self):
            calls.append("converge")

        def recover_orphaned_async_ops(self):
            calls.append("async")
            raise OperationBusy("x")  # tolerated like the drain recovery

        def recover_stale_drains(self):
            calls.append("drains")

        def recover_stale_fleet_repairs(self):
            return None

        def detect_fleet_drift(self):
            return []

    FleetSupervisor(Service()).run_once()
    assert calls == ["converge", "async", "drains"]


def test_reconcile_marks_orphans_first():
    events: list = []
    redis = FakeRedis()
    journal = AsyncOpJournal(redis)
    journal.save(_dead_record("op-x", phase="running"))
    service, _runtime, _vllm = _amake(events, journal=journal)
    service._mark_orphaned_async_ops()
    assert service.get_operation("op-x")["status"] == "failed"
    assert "op-x" in service._orphaned_async_ops


def test_marker_to_dict_omits_async_op_id_when_unset_for_rollback_compat():
    marker = DrainMarker(
        binding_id="b", serve_id="s", model="m1", token="t", instance="i",
        started_at=1.0, deadline_at=2.0, reason="r",
    )
    assert "async_op_id" not in marker.to_dict()
    assert DrainMarker(**{**marker.to_dict(), "async_op_id": "op"}).to_dict()["async_op_id"] == "op"
