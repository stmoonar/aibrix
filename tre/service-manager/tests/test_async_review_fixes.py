"""Review fixes of the async / per-call-drain SM work (H2, H3, M1, M4, L1, L3) plus the
missing scenarios the review named (rolling update, wake of a draining binding)."""
from __future__ import annotations

import importlib.util
import threading
import time
from contextlib import contextmanager
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from tre_sm.api.v2 import create_app
from tre_sm.state.async_ops import AsyncOperationManager, AsyncOpJournal, AsyncOpsConfig
from tre_sm.state.drain_markers import DrainMarker, DrainMarkerStore
from tre_sm.state.operations import OperationBusy

_HERE = Path(__file__).resolve().parent


def _load(name: str):
    spec = importlib.util.spec_from_file_location(f"_review_helpers_{name}", _HERE / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_async = _load("test_async_ops")
_amake = _async._amake
_wait = _async._wait
_bindings = _async._bindings
_three_bindings = _async._three_bindings
_dead_record = _async._dead_record
ASYNC_ON = _async.ASYNC_ON
FakeRedis = _async.FakeRedis
FakeOperation = _async.FakeOperation
CallbackVllm = _async.CallbackVllm
metrics_text = _async.metrics_text
WAIT_S = 10.0


class BusyFirst:
    def __init__(self, busy):
        self.busy = busy
        self.attempts = 0

    @contextmanager
    def operation(self, kind, *, request=None):
        self.attempts += 1
        if self.busy > 0:
            self.busy -= 1
            raise OperationBusy("async-worker")
        yield FakeOperation()


def _fake_lock_clock(service):
    state = {"t": 0.0, "sleeps": []}

    def sleep(seconds):
        state["sleeps"].append(seconds)
        state["t"] += seconds

    service._lock_clock = lambda: state["t"]
    service._lock_sleep = sleep
    return state


# ------------------------------------------------------------------ H2


def test_h2_sync_routable_waits_for_a_busy_writer_lock_instead_of_409():
    events: list = []
    coordinator = BusyFirst(3)
    service, _runtime, _vllm = _amake(events, coordinator=coordinator)
    clock = _fake_lock_clock(service)

    result = service.put_model_routable("m1", hidden_pods=["serve-b"])

    assert result["actions"] == [{"action": "hide", "serve_id": "serve-b"}]
    assert coordinator.attempts == 4 and len(clock["sleeps"]) == 3


def test_h2_api_routable_answers_200_while_the_lock_is_briefly_busy():
    events: list = []
    service, _runtime, _vllm = _amake(events, coordinator=BusyFirst(2))
    _fake_lock_clock(service)
    response = TestClient(create_app(service)).put(
        "/v2/models/m1/routable", json={"hidden_pods": ["serve-b"]}
    )
    assert response.status_code == 200, response.text


def test_h2_lock_wait_is_bounded():
    events: list = []
    config = AsyncOpsConfig(enabled=True, sync_lock_wait_s=1.0, sync_lock_poll_s=0.2)
    service, _runtime, _vllm = _amake(events, coordinator=BusyFirst(10**6), async_config=config)
    clock = _fake_lock_clock(service)

    with pytest.raises(OperationBusy):
        service.put_model_routable("m1", hidden_pods=[])
    assert sum(clock["sleeps"]) == pytest.approx(1.0)


def test_h2_flags_off_keeps_mains_single_attempt():
    events: list = []
    coordinator = BusyFirst(1)
    service, _runtime, _vllm = _amake(
        events, hide=False, coordinator=coordinator, async_config=AsyncOpsConfig()
    )
    clock = _fake_lock_clock(service)
    with pytest.raises(OperationBusy):
        service.put_model_routable("m1", hidden_pods=[])
    assert coordinator.attempts == 1 and clock["sleeps"] == []


# ------------------------------------------------------------------ H3


class _ExplodingProgressJournal(AsyncOpJournal):
    def save(self, record):
        if record.get("phase") == "draining":
            raise ConnectionError("redis blip")
        super().save(record)


def test_h3_failing_progress_bookkeeping_does_not_leak_drain_tokens():
    events: list = []
    service, _runtime, _vllm = _amake(
        events, journal=_ExplodingProgressJournal(), pages={"10.0.0.2": [metrics_text(0, 0)]}
    )

    accepted = service.submit_binding_power("serve-b", awake=False, drain_s=0)
    record = _wait(service, accepted["operation_id"])

    assert record["status"] == "succeeded", record
    assert service._inflight_tokens == set()
    assert service._load_markers() == {}
    assert _bindings(service)["serve-b"].awake is False


def test_h3_own_token_leaked_past_deadline_plus_grace_is_stale():
    events: list = []
    service, _runtime, _vllm = _amake(events)
    now = time.time()

    def marker(deadline):
        return DrainMarker(
            binding_id="m1/node-a/1", serve_id="serve-b", model="m1", token="t-leaked",
            instance=service._instance_id, started_at=now - 600, deadline_at=deadline,
            reason="binding_power",
        )

    service._inflight_tokens.add("t-leaked")
    assert service._marker_is_stale(marker(now - 300)) is True
    assert service._marker_is_stale(marker(now + 300)) is False


# ------------------------------------------------------------------ M1


def _manager(journal, executor=None):
    return AsyncOperationManager(
        journal,
        executor or (lambda record, progress: ("succeeded", {})),
        config=ASYNC_ON,
        instance="sm-self",
    )


def test_m1_recover_orphans_never_touches_records_of_its_own_instance():
    journal = AsyncOpJournal()
    journal.save(_dead_record("mine", phase="draining", instance="sm-self"))
    assert _manager(journal).recover_orphans() == []
    assert journal.get("mine")["status"] == "running"


class _RacingJournal(AsyncOpJournal):
    """list() still shows the stale heartbeat; get() (re-read under the lock) the fresh one."""

    def list(self):
        records = super().list()
        for record in records:
            record["heartbeat_ts"] = record["updated_ts"] = 1.0  # stale snapshot
        return records


def test_m1_record_refreshed_between_scan_and_lock_is_left_alone():
    journal = _RacingJournal()
    fresh = _dead_record("live", phase="draining", instance="other-sm", age_s=1.0)
    journal.save(fresh)
    assert _manager(journal).recover_orphans() == []
    assert journal.get("live")["status"] == "running"


# ------------------------------------------------------------------ L1


class _FlakyFinalSaveJournal(AsyncOpJournal):
    def __init__(self, failures):
        super().__init__()
        self.failures = failures

    def save(self, record):
        if record.get("status") in ("succeeded", "failed", "superseded") and self.failures > 0:
            self.failures -= 1
            raise ConnectionError("redis down")
        super().save(record)


def test_l1_failed_final_save_is_kept_and_rewritten_not_left_running():
    journal = _FlakyFinalSaveJournal(failures=3)
    manager = _manager(journal)

    record = manager.submit(kind="model_target", model="m1", target_key="model:m1", request={})

    assert manager.wait(record["operation_id"], timeout_s=WAIT_S) is True
    assert manager.get(record["operation_id"])["status"] == "succeeded"
    assert journal.get(record["operation_id"])["status"] == "running"  # write failed
    assert manager.active_models() == set()
    with manager._lock:
        manager._flush_unsaved_locked()  # what the heartbeat loop does
    assert journal.get(record["operation_id"])["status"] == "succeeded"
    assert manager._unsaved == {}


# ------------------------------------------------------------------ L3


class _TimeRedis:
    def __init__(self, offset_s=0.0, broken=False):
        self.offset_s = offset_s
        self.broken = broken

    def time(self):
        if self.broken:
            raise ConnectionError("no redis")
        now = time.time() + self.offset_s
        return int(now), int((now % 1) * 1e6)


def test_l3_clock_skew_keeps_async_ops_off():
    from tre_sm.state.async_ops import enforce_clock_skew

    on = AsyncOpsConfig(enabled=True)
    assert enforce_clock_skew(on, _TimeRedis(0.5)).enabled is True
    assert enforce_clock_skew(on, _TimeRedis(12.0)).enabled is False
    assert enforce_clock_skew(on, _TimeRedis(-12.0)).enabled is False
    assert enforce_clock_skew(on, _TimeRedis(broken=True)).enabled is False
    off = AsyncOpsConfig()
    assert enforce_clock_skew(off, _TimeRedis(broken=True)) is off


# ------------------------------------------------------------------ M4 (SM side)


def test_m4_caller_meta_is_stored_and_active_operations_are_listed():
    events: list = []
    service, _runtime, _vllm = _amake(events, pages={"10.0.0.2": [metrics_text(0, 0)]})
    client = TestClient(create_app(service))
    meta = {"controller": True, "action_id": "a-1", "rollback_unhide": ["serve-b"]}

    response = client.put(
        "/v2/bindings/serve-b/power?async=1", json={"awake": False, "drain_s": 0, "meta": meta}
    )
    assert response.status_code == 202
    operation_id = response.json()["operation_id"]
    record = _wait(service, operation_id)
    assert record["meta"] == meta
    assert client.get("/v2/async-operations?active=1").json()["operations"] == []
    assert len(client.get("/v2/async-operations").json()["operations"]) == 1
    too_big = client.put(
        "/v2/bindings/serve-a/power?async=1", json={"awake": False, "meta": {"x": "y" * 5000}}
    )
    assert too_big.status_code == 400


def test_m4_active_listing_shows_running_operations():
    release = threading.Event()

    def executor(record, progress):
        assert release.wait(WAIT_S)
        return "succeeded", {}

    manager = _manager(AsyncOpJournal(), executor)
    record = manager.submit(kind="model_target", model="m1", target_key="model:m1", request={}, meta={"a": 1})
    active = manager.list(active=True)
    assert [item["operation_id"] for item in active] == [record["operation_id"]]
    assert active[0]["meta"] == {"a": 1}
    release.set()
    assert manager.wait(record["operation_id"], timeout_s=WAIT_S)
    assert manager.list(active=True) == []


# ------------------------------------------------------ review-named scenarios


def test_rolling_update_two_sm_instances_do_not_orphan_each_other():
    events: list = []
    redis = FakeRedis()
    store = _three_bindings(redis)
    seen: dict = {}

    def during_drain(_ip):
        other = seen["b"]
        seen["b_recovered"] = other.recover_orphaned_async_ops()
        seen["b_drains"] = other.recover_stale_drains()
        seen["markers_during"] = dict(other._load_markers())

    vllm = CallbackVllm(
        events, pages={"10.0.0.2": [metrics_text(1, 0), metrics_text(0, 0)]},
        trigger_ip="10.0.0.2", on_metrics=during_drain,
    )
    a, _runtime_a, _ = _amake(
        events, vllm=vllm, store=store, journal=AsyncOpJournal(redis), markers=DrainMarkerStore(redis)
    )
    b, _runtime_b, _ = _amake(
        events, store=store, journal=AsyncOpJournal(redis), markers=DrainMarkerStore(redis)
    )
    seen["b"] = b

    accepted = a.submit_binding_power("serve-b", awake=False, drain_s=30)
    record = _wait(a, accepted["operation_id"])

    # B (new pod of a rolling update) saw A's live operation and its marker and left
    # both alone; A finished its own sleep.
    assert seen["b_recovered"] is None and seen["b_drains"] is None
    assert list(seen["markers_during"]) == ["m1/node-a/1"]
    assert record["status"] == "succeeded" and record["summary"]["slept"] == ["serve-b"]
    assert b.get_operation(accepted["operation_id"])["status"] == "succeeded"
    # Later A is gone with an operation still marked running: B orphans it, A (if it
    # were still there) would never orphan its own records.
    AsyncOpJournal(redis).save({**_dead_record("a-dead", phase="running"), "instance": a._instance_id})
    assert a.recover_orphaned_async_ops() is None
    assert b.recover_orphaned_async_ops()["orphaned"] == ["a-dead"]


def test_wake_of_the_same_binding_while_it_drains_reclaims_it():
    events: list = []
    seen: dict = {}

    def during_drain(_ip):
        service = seen["service"]
        wake = service.submit_binding_power("serve-b", awake=True)
        seen["done"] = service.wait_async_operation(wake["operation_id"], timeout_s=WAIT_S)
        seen["wake"] = service.get_operation(wake["operation_id"])

    vllm = CallbackVllm(
        events, pages={"10.0.0.2": [metrics_text(2, 0)]}, trigger_ip="10.0.0.2", on_metrics=during_drain,
    )
    service, _runtime, _ = _amake(events, vllm=vllm)
    seen["service"] = service

    sleep_op = service.submit_binding_power("serve-b", awake=False, drain_s=60)
    record = _wait(service, sleep_op["operation_id"])

    assert seen["done"] is True
    assert seen["wake"]["status"] == "succeeded" and seen["wake"]["summary"]["reclaimed"] == ["serve-b"]
    assert record["status"] == "superseded"
    assert [item["outcome"] for item in record["bindings"]] == ["abandoned_reclaimed"]
    final = _bindings(service)["serve-b"]
    assert final.awake is True and final.hidden is False
    assert not any(event[0] == "sleep" for event in events)
