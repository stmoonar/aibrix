"""Staged (lock-releasing) sleep: TRE_SM_HIDE_BEFORE_SLEEP / TRE_SM_DRAIN_BEFORE_SLEEP
after review H1/H3/M1/M4/M5 (tre/docs/design/20260924-reissue-sidecar.md section 3.3)."""
from __future__ import annotations

import importlib.util
import time
from contextlib import contextmanager
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import pytest

from tre_common import rediskeys
from tre_common.registry import ReissueSidecarSpec, Registry
from tre_sm.allocator.slots import Binding, Slot
from tre_sm.allocator.topology import K8sPodSnapshot
from tre_sm.api.v2 import ServiceManagerV2, SleepCommitFailed
from tre_sm.ops.drain import DrainConfig, SleepConfigError, SleepDrainer, check_reissue_coupling
from tre_sm.state.drain_markers import _SAVE_MARKERS_SCRIPT, DrainMarker
from tre_sm.state.fleet_store import DesiredBinding, DesiredSnapshot
from tre_sm.state.operations import OperationBusy, OperationCoordinator
from tre_sm.state.store import StateStore

_HERE = Path(__file__).resolve().parent


def _load(name: str):
    spec = importlib.util.spec_from_file_location(f"_staged_helpers_{name}", _HERE / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_drain = _load("test_drain_before_sleep")
_ops = _load("test_operations")
FakeClock = _drain.FakeClock
FakeRuntimeOps = _drain.FakeRuntimeOps
FakeVllmOps = _drain.FakeVllmOps
FakeRedis = _drain.FakeRedis
FakeOperation = _drain.FakeOperation
Result = _drain.Result
metrics_text = _drain.metrics_text
registry = _drain.registry
_pod = _drain._pod


class MarkerScriptRedis(_ops.ScriptRedis):
    """ScriptRedis plus the draining-marker Lua script."""

    def eval(self, script, numkeys, *keys_and_args):
        if script == _SAVE_MARKERS_SCRIPT:
            key, lock_key = keys_and_args[:numkeys]
            lock_value, document = keys_and_args[numkeys:]
            if lock_value and self.values.get(lock_key) != lock_value:
                return -1
            self.values[key] = document
            return 1
        return super().eval(script, numkeys, *keys_and_args)


class CallbackVllm(FakeVllmOps):
    """Runs ``on_metrics`` once, at the first /metrics scrape of ``trigger_ip`` - i.e.
    in phase 2, while the service holds no writer lock."""

    def __init__(self, events, *, pages=None, sleep_message="", trigger_ip=None, on_metrics=None):
        super().__init__(events, pages=pages, sleep_message=sleep_message)
        self.trigger_ip = trigger_ip
        self.on_metrics = on_metrics
        self.fired = False

    def metrics(self, pod_ip, *, port=None):
        if not self.fired and self.on_metrics is not None and (self.trigger_ip in (None, pod_ip)):
            self.fired = True
            self.on_metrics(pod_ip)
        return super().metrics(pod_ip, port=port)


def hidden_calls(vllm):
    """Whether each /sleep carried X-TRE-Hidden (hidden=True kwarg)."""
    return [bool(kwargs.get("hidden", False)) for kwargs in vllm.sleep_kwargs]


def _store(bindings, redis=None):
    store = StateStore(redis if redis is not None else FakeRedis())
    store.save(bindings, expected_version=0)
    return store


def _two_awake(redis=None):
    return _store(
        [
            Binding("serve-a", "m1", Slot("node-a", (0,)), awake=True),
            Binding("serve-b", "m1", Slot("node-a", (1,)), awake=True),
        ],
        redis,
    )


def _make(events, *, hide=True, drain=False, pages=None, vllm=None, runtime=None, store=None,
          coordinator=None, real_clock=False, fleet_store=None, gpu_leases=None, reg=None, **cfg):
    runtime = runtime or FakeRuntimeOps(events, [_pod("serve-a", 0, "10.0.0.1"), _pod("serve-b", 1, "10.0.0.2")])
    vllm = vllm or FakeVllmOps(events, pages=pages)
    config = DrainConfig(enabled=drain, hide_before_sleep=hide, **cfg)
    drainer = None
    if hide and not real_clock:
        clock = FakeClock()
        drainer = SleepDrainer(runtime, vllm, config, monotonic=clock.monotonic, sleep=clock.sleep)
    service = ServiceManagerV2(
        reg or registry(),
        store or _two_awake(),
        runtime_ops=runtime,
        vllm_ops=vllm,
        drain_config=config,
        sleep_drainer=drainer,
        operation_coordinator=coordinator,
        fleet_store=fleet_store,
        gpu_leases=gpu_leases,
    )
    return service, runtime, vllm


def _bindings(service):
    return {binding.serve_id: binding for binding in service._store.load().bindings}


# ---------------------------------------------------------------- flags off == main


def test_flags_off_is_the_legacy_path_without_header_or_markers():
    events: list = []
    redis = FakeRedis()
    service, _runtime, vllm = _make(events, hide=False, store=_two_awake(redis))
    state_before = service.get_state()
    assert set(state_before) == {"version", "models", "bindings"}
    assert state_before["models"]["m1"] == {"awake": 2, "bound": 2}
    assert all("draining" not in item for item in state_before["bindings"])

    result = service.put_model_target("m1", wake_replicas=1)
    (slept,) = result["actions"]
    ip = {"serve-a": "10.0.0.1", "serve-b": "10.0.0.2"}[slept["serve_id"]]
    assert events == [("sleep", ip), ("annotate", slept["serve_id"], "sleeping")]
    assert hidden_calls(vllm) == [False]
    assert set(result) == {"model", "wake_replicas", "version", "actions"}

    service.put_binding_power("serve-b" if slept["serve_id"] == "serve-a" else "serve-a", awake=False)
    assert hidden_calls(vllm) == [False, False]
    assert "tre:v2:sm:draining" not in redis.values
    assert not any(event[0] in {"wait_unroutable", "metrics"} for event in events)


# ---------------------------------------------------------------- H3: config


def test_drain_without_hide_and_sidecar_without_hide_fail_at_startup():
    with pytest.raises(SleepConfigError):
        DrainConfig(enabled=True)
    with pytest.raises(SleepConfigError):
        DrainConfig.from_env({"TRE_SM_DRAIN_BEFORE_SLEEP": "true"})
    base = registry()
    with_sidecar = Registry(base.topology(), base.models(), ReissueSidecarSpec(enabled=True))
    with pytest.raises(SleepConfigError):
        check_reissue_coupling(with_sidecar, DrainConfig())
    check_reissue_coupling(with_sidecar, DrainConfig(hide_before_sleep=True))
    check_reissue_coupling(base, DrainConfig())
    events: list = []
    with pytest.raises(SleepConfigError):
        ServiceManagerV2(
            with_sidecar, _two_awake(),
            runtime_ops=FakeRuntimeOps(events, []), vllm_ops=FakeVllmOps(events),
        )


def test_hide_only_mode_waits_unroutable_then_sleeps_with_header():
    events: list = []
    service, _runtime, vllm = _make(events, hide=True, drain=False, pages={"10.0.0.1": [metrics_text(9, 9)]})
    result = service.put_binding_power("serve-a", awake=False)
    assert events == [
        ("annotate", "serve-a", "hidden"),
        ("wait_unroutable", "serve-a"),
        ("sleep", "10.0.0.1"),
        ("annotate", "serve-a", "sleeping"),
    ]
    assert hidden_calls(vllm) == [True]
    assert [item["outcome"] for item in result["sleep_outcomes"]] == ["slept"]
    (record,) = service.get_sleep_audit()["records"]
    assert record["hide_enabled"] is True and record["drain_enabled"] is False
    assert _bindings(service)["serve-a"].awake is False
    assert service._load_markers() == {}


# ---------------------------------------------------------------- H1: no lock while draining


def test_other_sm_operations_succeed_during_a_drain_instead_of_409():
    events: list = []
    redis = MarkerScriptRedis()
    coordinator = OperationCoordinator(redis, owner="sm-test")
    seen: dict = {}
    holder: dict = {}

    def during_drain(_ip):
        service = holder["service"]
        seen["lock_held"] = rediskeys.SM_WRITER_LOCK_KEY in redis.values
        state = service.get_state()
        seen["state_models"] = state["models"]["m1"]
        seen["draining_flags"] = {item["serve_id"]: item["draining"] for item in state["bindings"]}
        # a lock-taking operation must not hit OperationBusy (the old 409)
        seen["routable"] = service.put_model_routable("m1", hidden_pods=[])

    vllm = CallbackVllm(
        events, pages={"10.0.0.1": [metrics_text(1, 0), metrics_text(0, 0)]},
        trigger_ip="10.0.0.1", on_metrics=during_drain,
    )
    service, _runtime, _ = _make(
        events, drain=True, vllm=vllm, store=_two_awake(redis), coordinator=coordinator,
    )
    holder["service"] = service

    result = service.put_binding_power("serve-a", awake=False)

    assert seen["lock_held"] is False
    assert seen["state_models"] == {"awake": 1, "bound": 2, "draining": 1}
    assert seen["draining_flags"] == {"serve-a": True, "serve-b": False}
    assert "model" in seen["routable"] or seen["routable"]
    assert [item["outcome"] for item in result["sleep_outcomes"]] == ["slept"]
    assert _bindings(service)["serve-a"].awake is False
    assert service._load_markers() == {}
    assert rediskeys.SM_WRITER_LOCK_KEY not in redis.values


def test_growing_target_during_drain_reclaims_the_draining_binding():
    events: list = []
    holder: dict = {}

    def during_drain(_ip):
        holder["grow"] = holder["service"].put_model_target("m1", wake_replicas=2)

    vllm = CallbackVllm(
        events,
        pages={"10.0.0.1": [metrics_text(3, 0)], "10.0.0.2": [metrics_text(3, 0)]},
        on_metrics=during_drain,
    )
    service, _runtime, _ = _make(events, drain=True, vllm=vllm)
    holder["service"] = service

    result = service.put_model_target("m1", wake_replicas=1)

    (outcome,) = result["sleep_outcomes"]
    assert outcome["outcome"] == "abandoned_reclaimed"
    victim = outcome["serve_id"]
    assert {"action": "reclaim", "serve_id": victim} in holder["grow"]["actions"]
    final = _bindings(service)[victim]
    assert final.awake is True and final.hidden is False
    assert ("annotate", victim, "awake") in events
    assert not any(event[0] == "sleep" for event in events)
    assert service._load_markers() == {}
    assert service.get_state()["models"]["m1"]["awake"] == 2


def test_wake_request_for_a_draining_binding_cancels_its_sleep():
    events: list = []
    holder: dict = {}

    def during_drain(_ip):
        holder["wake"] = holder["service"].put_binding_power("serve-a", awake=True)

    vllm = CallbackVllm(events, pages={"10.0.0.1": [metrics_text(2, 0)]}, on_metrics=during_drain)
    service, _runtime, _ = _make(events, drain=True, vllm=vllm)
    holder["service"] = service

    result = service.put_binding_power("serve-a", awake=False)

    assert [item["outcome"] for item in result["sleep_outcomes"]] == ["abandoned_reclaimed"]
    assert holder["wake"]["actions"] == [{"action": "reclaim", "serve_id": "serve-a"}]
    assert _bindings(service)["serve-a"].awake is True
    assert not any(event[0] == "sleep" for event in events)


def test_changed_fencing_token_abandons_the_sleep():
    events: list = []
    holder: dict = {}

    def during_drain(_ip):
        service = holder["service"]
        markers = service._load_markers()
        marker = markers["m1/node-a/0"]
        markers["m1/node-a/0"] = DrainMarker(**{**marker.to_dict(), "token": "someone-else"})
        service._drain_markers.save(markers)

    vllm = CallbackVllm(events, pages={"10.0.0.1": [metrics_text(0, 0)]}, on_metrics=during_drain)
    service, _runtime, _ = _make(events, drain=True, vllm=vllm)
    holder["service"] = service

    result = service.put_binding_power("serve-a", awake=False)

    assert [item["outcome"] for item in result["sleep_outcomes"]] == ["abandoned_reclaimed"]
    assert not any(event[0] == "sleep" for event in events)
    assert _bindings(service)["serve-a"].awake is True  # the new owner decides


def test_draining_gpu_is_not_free_capacity():
    events: list = []
    holder: dict = {}
    store = _store(
        [
            Binding("serve-a", "m1", Slot("node-a", (0,)), awake=True),
            Binding("serve-b", "m1", Slot("node-a", (1,)), awake=True),
            # another model's sleeping binding on the draining binding's GPU
            Binding("serve-c", "m2", Slot("node-a", (0,)), awake=False),
        ]
    )
    pod_c = K8sPodSnapshot(name="serve-c", model="m2", node="node-a", env={"CUDA_VISIBLE_DEVICES": "0"},
                           pod_ip="10.0.0.3")
    runtime = FakeRuntimeOps(events, [_pod("serve-a", 0, "10.0.0.1"), _pod("serve-b", 1, "10.0.0.2"), pod_c])
    base = registry()
    two_models = Registry(base.topology(), [*base.models(), replace(base.models()[0], name="m2")])

    def during_drain(_ip):
        try:
            holder["service"].put_binding_power("serve-c", awake=True)
        except Exception as exc:  # noqa: BLE001 - record whatever refused it
            holder["refused"] = f"{type(exc).__name__}: {exc}"

    vllm = CallbackVllm(events, pages={"10.0.0.1": [metrics_text(0, 0)]}, on_metrics=during_drain)
    service, _runtime, _ = _make(events, drain=True, vllm=vllm, runtime=runtime, store=store, reg=two_models)
    holder["service"] = service

    service.put_binding_power("serve-a", awake=False)

    assert "refused" in holder, "a wake onto the draining binding's GPU must be refused"
    assert holder["refused"].startswith("WakeConflict"), holder["refused"]
    assert ("wake_up", "10.0.0.3") not in events
    assert _bindings(service)["serve-a"].awake is False


# ---------------------------------------------------------------- M4: one deadline, parallel


def test_multi_binding_drain_is_parallel_and_bounded_by_one_deadline():
    events: list = []
    pages = {"10.0.0.1": [metrics_text(5, 0)], "10.0.0.2": [metrics_text(5, 0)]}
    service, _runtime, vllm = _make(
        events, drain=True, pages=pages, real_clock=True,
        sleep_deadline_s=1.0, commit_reserve_s=0.3, poll_interval_s=0.05, unroutable_timeout_s=0.5,
    )
    started = time.monotonic()
    result = service.put_model_target("m1", wake_replicas=0)
    elapsed = time.monotonic() - started

    assert elapsed < 1.0, elapsed           # one deadline for the whole call ...
    assert elapsed > 0.5, elapsed           # ... and the drains really waited
    assert sorted(item["outcome"] for item in result["sleep_outcomes"]) == ["slept", "slept"]
    records = service.get_sleep_audit()["records"]
    assert all(r["deadline_capped"] and r["drained"] is False and r["interrupted_running"] == 5 for r in records)
    assert hidden_calls(vllm) == [True, True]


# ---------------------------------------------------------------- M5: rollback


class FailingSleepVllm(FakeVllmOps):
    def __init__(self, events, *, mode, pages=None):
        super().__init__(events, pages=pages)
        self.mode = mode

    def sleep(self, pod_ip, *, port=None, **kwargs):
        self.events.append(("sleep", pod_ip))
        self.sleep_kwargs.append(kwargs)
        if self.mode == "refused":
            return Result("engine refused", status_code=500, success=False)
        raise TimeoutError("read timed out")

    def is_sleeping(self, pod_ip, *, port=None):
        return None if self.mode == "unknown" else False


@pytest.mark.parametrize("mode", ["refused", "timeout"])
def test_failed_sleep_rolls_back_to_awake_and_routable(mode):
    events: list = []
    vllm = FailingSleepVllm(events, mode=mode, pages={"10.0.0.1": [metrics_text(0, 0)]})
    service, _runtime, _ = _make(events, drain=True, vllm=vllm)

    with pytest.raises(SleepCommitFailed) as info:
        service.put_binding_power("serve-a", awake=False)

    assert [item["outcome"] for item in info.value.outcomes] == ["rolled_back"]
    binding = _bindings(service)["serve-a"]
    assert binding.awake is True and binding.hidden is False
    assert events.index(("annotate", "serve-a", "hidden")) < events.index(("annotate", "serve-a", "awake"))
    assert service._load_markers() == {}


def test_unverifiable_sleep_stays_hidden_for_reconcile():
    events: list = []
    vllm = FailingSleepVllm(events, mode="unknown", pages={"10.0.0.1": [metrics_text(0, 0)]})
    service, _runtime, _ = _make(events, drain=True, vllm=vllm)

    with pytest.raises(SleepCommitFailed) as info:
        service.put_binding_power("serve-a", awake=False)

    assert [item["outcome"] for item in info.value.outcomes] == ["sleep_unverified"]
    binding = _bindings(service)["serve-a"]
    assert binding.hidden is True  # fail closed: not routable while its power is unknown
    assert ("annotate", "serve-a", "awake") not in events
    assert service._load_markers() == {}


def test_drain_error_rolls_back():
    events: list = []

    class BrokenRuntime(FakeRuntimeOps):
        def wait_pod_unroutable(self, binding, *, timeout_s=30.0, interval_s=0.5):
            raise RuntimeError("apiserver unavailable")

    runtime = BrokenRuntime(events, [_pod("serve-a", 0, "10.0.0.1"), _pod("serve-b", 1, "10.0.0.2")])
    service, _runtime, _ = _make(events, drain=True, runtime=runtime, pages={"10.0.0.1": [metrics_text(0, 0)]})

    with pytest.raises(SleepCommitFailed) as info:
        service.put_binding_power("serve-a", awake=False)

    (outcome,) = info.value.outcomes
    assert outcome["outcome"] == "rolled_back" and "apiserver unavailable" in outcome["error"]
    assert _bindings(service)["serve-a"].hidden is False
    assert not any(event[0] == "sleep" for event in events)


def test_commit_lock_busy_until_deadline_leaves_marker_for_recovery():
    events: list = []

    class BusyAfterFirst:
        def __init__(self):
            self.calls = 0

        @contextmanager
        def operation(self, kind, *, request=None):
            self.calls += 1
            if self.calls > 1:
                raise OperationBusy("other-writer")
            yield FakeOperation()

    service, _runtime, _ = _make(
        events, drain=True, coordinator=BusyAfterFirst(), pages={"10.0.0.1": [metrics_text(0, 0)]},
        sleep_deadline_s=2.0, commit_reserve_s=1.0, lock_retry_interval_s=0.25,
    )
    with pytest.raises(OperationBusy):
        service.put_binding_power("serve-a", awake=False)
    markers = service._load_markers()
    assert list(markers) == ["m1/node-a/0"]
    # the owner gave up, so for this instance the marker is stale right away
    assert service._marker_is_stale(markers["m1/node-a/0"]) is True


# ---------------------------------------------------------------- stale recovery


def _stale_marker(service, *, instance="dead-sm"):
    return DrainMarker(
        binding_id="m1/node-a/0", serve_id="serve-a", model="m1", token="t-old", instance=instance,
        started_at=time.time() - 600, deadline_at=time.time() - 300, reason="model_target",
    )


def test_stale_marker_is_completed_when_desired_still_sleeping():
    events: list = []
    store = _store(
        [
            Binding("serve-a", "m1", Slot("node-a", (0,)), awake=True, hidden=True),
            Binding("serve-b", "m1", Slot("node-a", (1,)), awake=True),
        ]
    )
    service, _runtime, vllm = _make(events, drain=True, store=store)
    service._drain_markers.save({"m1/node-a/0": _stale_marker(service)})

    result = service.recover_stale_drains()

    assert [item["outcome"] for item in result["recovered"]] == ["slept"]
    assert hidden_calls(vllm) == [True]
    assert _bindings(service)["serve-a"].awake is False
    assert service._load_markers() == {}


def test_stale_marker_is_unhidden_when_desired_is_awake_again():
    events: list = []
    now = datetime.now(timezone.utc).isoformat()

    class FleetStore:
        def load_desired(self):
            return DesiredSnapshot(1, [
                DesiredBinding("m1/node-a/0", "m1", "node-a", (0,), "resident", "awake", False, 1, now, "t", "t"),
            ])

    store = _store(
        [
            Binding("serve-a", "m1", Slot("node-a", (0,)), awake=True, hidden=True),
            Binding("serve-b", "m1", Slot("node-a", (1,)), awake=True),
        ]
    )
    service, _runtime, _vllm = _make(events, drain=True, store=store, fleet_store=FleetStore())
    service._drain_markers.save({"m1/node-a/0": _stale_marker(service)})

    result = service.recover_stale_drains()

    assert [item["outcome"] for item in result["recovered"]] == ["unhidden"]
    assert _bindings(service)["serve-a"].hidden is False
    assert ("annotate", "serve-a", "awake") in events
    assert not any(event[0] == "sleep" for event in events)


def test_live_marker_of_another_instance_is_left_alone():
    events: list = []
    service, _runtime, _vllm = _make(events, drain=True)
    live = DrainMarker(
        binding_id="m1/node-a/0", serve_id="serve-a", model="m1", token="t", instance="other-sm",
        started_at=time.time(), deadline_at=time.time() + 200, reason="model_target",
    )
    service._drain_markers.save({"m1/node-a/0": live})
    assert service.recover_stale_drains() is None
    assert list(service._load_markers()) == ["m1/node-a/0"]


# ---------------------------------------------------------------- startup admission / converge


def test_startup_admission_suspends_resident_with_hide_and_header():
    from tre_sm.ops.k8s_ops import StartupPodRecord

    now = datetime.now(timezone.utc).isoformat()
    events: list = []

    class FleetStore:
        def load_desired(self):
            return DesiredSnapshot(1, [
                DesiredBinding("m1/node-a/0", "m1", "node-a", (0,), "resident", "sleeping", False, 1, now, "t", "t"),
                DesiredBinding("m2/node-a/0", "m2", "node-a", (0,), "resident", "awake", False, 1, now, "t", "t"),
            ])

    class Coordinator:
        def active_operation(self, *, kind=None):
            return None

        @contextmanager
        def operation(self, *_args, **_kwargs):
            handle = FakeOperation()
            handle.operation_id = "startup-op"
            yield handle

    class Safety:
        def assert_no_pressure(self):
            return None

    class Leases:
        def acquire(self, binding, *, phase):
            events.append(("lease_acquire", binding.binding_id, phase))

        def release(self, binding):
            events.append(("lease_release", binding.binding_id))

        def load(self):
            return []

    resident = K8sPodSnapshot(
        name="m2-old", model="m2", node="node-a", env={"CUDA_VISIBLE_DEVICES": "0"},
        annotations={"tre.aibrix.io/gpu-ids": "0", "tre.aibrix.io/state": "awake"},
        pod_ip="10.0.0.2", ready=True, pod_uid="old-uid",
    )

    class Runtime(FakeRuntimeOps):
        def get_startup_pod(self, name):
            return StartupPodRecord(
                name=name, uid="new-uid", model="m1", node="node-a", gpu_ids=(0,), annotations={},
                labels={}, pod_ip=None, phase="Pending", ready=False,
            )

        def list_startup_resident_snapshots(self):
            return self.list_pod_snapshots()

        def admit_startup_pod(self, name, **kwargs):
            events.append(("admit", name))

    registry2 = _api_registry()
    runtime = Runtime(events, [resident])
    vllm = FakeVllmOps(events, pages={"10.0.0.2": [metrics_text(0, 0)]})
    config = DrainConfig(enabled=True, hide_before_sleep=True)
    clock = FakeClock()
    service = ServiceManagerV2(
        registry2, _store([Binding("m2-old", "m2", Slot("node-a", (0,)), awake=True)]),
        runtime_ops=runtime, vllm_ops=vllm, operation_coordinator=Coordinator(), safety_gate=Safety(),
        fleet_store=FleetStore(), gpu_leases=Leases(), drain_config=config,
        sleep_drainer=SleepDrainer(runtime, vllm, config, monotonic=clock.monotonic, sleep=clock.sleep),
    )

    result = service.admit_startup(pod_name="m1-new", pod_uid="new-uid")

    assert result["suspended_binding_ids"] == ["m2/node-a/0"]
    hide = events.index(("annotate", "m2-old", "hidden"))
    assert hide < events.index(("wait_unroutable", "m2-old")) < events.index(("sleep", "10.0.0.2"))
    assert hidden_calls(vllm) == [True]
    assert events.index(("sleep", "10.0.0.2")) < events.index(("admit", "m1-new"))


def test_startup_converge_sleep_uses_hide_and_header():
    now = datetime.now(timezone.utc).isoformat()
    events: list = []

    class FleetStore:
        def load_desired(self):
            return DesiredSnapshot(1, [
                DesiredBinding("m1/node-a/0", "m1", "node-a", (0,), "resident", "sleeping", False, 1, now, "t", "t"),
            ])

    class Leases:
        def acquire(self, binding, *, phase):
            events.append(("lease_acquire", binding.binding_id, phase))

        def release(self, binding):
            events.append(("lease_release", binding.binding_id))

    class Runtime(FakeRuntimeOps):
        def clear_startup_admission(self, name):
            events.append(("clear_admission", name))

    started = K8sPodSnapshot(
        name="serve-a", model="m1", node="node-a", env={"CUDA_VISIBLE_DEVICES": "0"},
        annotations={"tre.aibrix.io/gpu-ids": "0", "tre.aibrix.io/startup-admitted-uid": "u1"},
        pod_ip="10.0.0.1", ready=True, pod_uid="u1",
    )
    runtime = Runtime(events, [started])
    vllm = FakeVllmOps(events, pages={"10.0.0.1": [metrics_text(0, 0)]})
    service, _runtime, _ = _make(
        events, drain=True, vllm=vllm, runtime=runtime, fleet_store=FleetStore(), gpu_leases=Leases(),
        store=_store([]),
    )
    service._reconcile_unlocked = lambda **_kwargs: {}

    service._converge_startup(started, False)

    assert events.index(("annotate", "serve-a", "hidden")) < events.index(("sleep", "10.0.0.1"))
    assert ("wait_unroutable", "serve-a") in events
    assert hidden_calls(vllm) == [True]
    assert ("lease_release", "m1/node-a/0") in events


def _api_registry():
    return _load("test_api_v2").registry()


# ---------------------------------------------------------------- M1: audit


def test_empty_sleep_snapshot_is_marked_unreliable_not_zero():
    events: list = []
    service, _runtime, _ = _make(events, drain=True, pages={"10.0.0.1": [metrics_text(0, 0)]})
    service._vllm_ops.sleep_message = "[]"
    service.put_binding_power("serve-a", awake=False)
    (record,) = service.get_sleep_audit()["records"]
    assert record["sleep_snapshot"] == []
    assert record["aborted_count"] is None
    assert record["aborted_count_reported"] == 0
    assert record["aborted_count_source"] == "vllm_sleep_response_unreliable"
    assert record["interrupted_estimate"] == 0
    assert record["authoritative_count_source"] == "reissue_sidecar_tre_reissue_metrics"
