"""The whole-lock service-manager (2026-10-02, design 20261002-sm-wholelock.md):
every write holds the writer lock from start to end; requests queue on it.

One test per risk: concurrent requests serialize and keep one awake engine per
GPU, a hung /sleep or /wake_up leaves the books fenced and releases the lock for
the recovery, a replayed relative target is harmless (converted at arrival), the transfer
primitive (POST /v2/transfers) and its failure / crash paths, /v2/state's
routable view, the floor and routable_unknown answers, supervisor isolation.

Concurrency is made deterministic with events (a request is held inside its
vLLM call until the other one is seen waiting for the lock), never with timing.
"""

from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import threading

import pytest
from fastapi.testclient import TestClient

from tre_common.registry import ClusterTopology, ModelSpec, NodeSpec, Registry, ServiceManagerConfig, SloSpec
from tre_sm.api.v2 import ServiceManagerV2, TransferFailed, create_app
from tre_sm.ops.sleep_primitive import GatewayState, SleepJournal
from tre_sm.state.fleet_store import DesiredBinding, FleetStateStore
from tre_sm.state.gpu_leases import GpuLeaseStore
from tre_sm.state.operations import OperationBusy, _CURRENT_OPERATION
from tre_sm.state.store import StateStore
from tre_sm.state.supervisor import FleetSupervisor
from tre_sm.state.wake_journal import WakeJournal

from sm_test_fakes import (
    FakeGateway,
    FakeHandle,
    FakeRedis,
    FakeRuntime,
    FakeSafety,
    FakeVllm,
    LegacyRedis,
    Result,
    TickingClock,
    binding_of,
    fence,
    pod,
    policy,
    trs,
)

SLO = SloSpec(ttft_p95_ms=1200, tpot_p95_ms=100, e2e_p95_ms=10000)


class BlockingCoordinator:
    """A real (blocking, first-come) writer lock like the Redis one: a caller
    waits up to ``wait_s`` for it, then OperationBusy. Records who held it and
    who is waiting, and every handle (their notes = the operation records)."""

    owner = "sm-test"

    def __init__(self, redis):
        self.redis = redis
        self._lock = threading.Lock()
        self.kinds: list[str] = []
        self.handles: list[tuple[str, FakeHandle]] = []
        self.waiting = threading.Event()
        self.active = None

    @contextmanager
    def operation(self, kind, *, request=None, wait_s=0.0):
        if not self._lock.acquire(blocking=False):
            self.waiting.set()
            if not self._lock.acquire(timeout=max(0.01, float(wait_s))):
                raise OperationBusy(f"held by {self.active}")
        self.kinds.append(kind)
        handle = FakeHandle(f"{kind}-{len(self.kinds)}")
        self.handles.append((kind, handle))
        self.active = {"operation_id": handle.operation_id, "kind": kind, "status": "running"}
        token = _CURRENT_OPERATION.set(handle)
        try:
            with fence(self.redis, handle.operation_id):
                yield handle
        finally:
            _CURRENT_OPERATION.reset(token)
            self.active = None
            self._lock.release()

    @property
    def held(self) -> bool:
        return self._lock.locked()

    def active_operation(self, *, kind=None):
        return self.active

    def stale_running_operations(self, *, kind=None):
        return []

    def list_operations(self, *, limit=100):
        return []


def _registry(*, d_min=0):
    topology = ClusterTopology(
        nodes=(NodeSpec("node-a", 4, ((0, 1), (2, 3)), ("GPU-0", "GPU-1", "GPU-2", "GPU-3")),)
    )
    models = [
        ModelSpec("d", "/d", 1, d_min, 4, "image", SLO, trs()),
        ModelSpec("r", "/r", 1, 0, 4, "image", SLO, trs()),
        ModelSpec("x", "/x", 1, 0, 4, "image", SLO, trs()),
        ModelSpec("t", "/t", 2, 0, 2, "image", SLO, trs()),
    ]
    return Registry(topology, models, service_manager=ServiceManagerConfig(sleep=policy()))


#: (serve id, model, gpus, state): d (donor) and r (receiver) share GPUs 0 / 1,
#: x is a third resident of GPU 0, r-2 / x-2 share the free GPU 2.
DEFAULT_PODS = (
    ("d-0", "d", (0,), "awake"),
    ("r-0", "r", (0,), "sleeping"),
    ("x-0", "x", (0,), "sleeping"),
    ("d-1", "d", (1,), "awake"),
    ("r-1", "r", (1,), "sleeping"),
    ("r-2", "r", (2,), "sleeping"),
    ("x-2", "x", (2,), "sleeping"),
)


class World:
    def __init__(self, pods=DEFAULT_PODS, *, d_min=0):
        self.redis = FakeRedis()
        self.registry = _registry(d_min=d_min)
        self.ip_of, self.gpus_of_ip, snapshots = {}, {}, []
        for index, (name, model, gpus, state) in enumerate(pods, start=1):
            ip = f"10.0.0.{index}"
            self.ip_of[name], self.gpus_of_ip[ip] = ip, tuple(gpus)
            snapshots.append(pod(name, model, gpus, ip=ip, state=state))
        self.runtime = FakeRuntime(snapshots)
        self.vllm = FakeVllm()
        self.vllm.sleeping.update({self.ip_of[n]: s == "sleeping" for n, _m, _g, s in pods})
        self.violations: list[tuple] = []
        self._guard_one_awake_engine_per_gpu()
        self.gateway = FakeGateway(self.redis, self.runtime)
        self.gateway.heartbeat("gw-1")
        self.gateway.auto_ack.add("gw-1")
        self.clock = TickingClock(lambda now: self.gateway.tick())
        self.store = StateStore(LegacyRedis())
        bindings = [binding_of(s) for s in snapshots]
        self.store.save(bindings, expected_version=0)
        self.fleet = FleetStateStore(self.redis)
        self.leases = GpuLeaseStore(self.redis)
        now = datetime.now(timezone.utc).isoformat()
        with fence(self.redis):
            self.fleet.save_desired(
                [
                    DesiredBinding(b.binding_id, b.model, "node-a", tuple(b.slot.gpu_ids), "resident",
                                   "awake" if b.awake else "sleeping", False, 1, now, "t", "t")
                    for b in bindings
                ],
                expected_version=0,
            )
            self.leases.rebuild_awake(bindings)
        self.coordinator = BlockingCoordinator(self.redis)
        self.service = self.make_service()
        self.client = TestClient(create_app(self.service))

    def make_service(self, **extra) -> ServiceManagerV2:
        """A service-manager process on this world's state (a restart = a new one)."""
        return ServiceManagerV2(
            self.registry,
            self.store,
            runtime_ops=self.runtime,
            vllm_ops=self.vllm,
            operation_coordinator=self.coordinator,
            safety_gate=FakeSafety(),
            fleet_store=self.fleet,
            gpu_leases=self.leases,
            gateway_state=GatewayState(self.redis, monotonic=lambda: self.clock.monotonic()),
            sleep_journal=SleepJournal(self.redis),
            wake_journal=WakeJournal(self.redis),
            sleep_clock=self.clock,
            **extra,
        )

    def _guard_one_awake_engine_per_gpu(self) -> None:
        """Invariant 1, checked after every /wake_up the fake engine accepts."""
        original = self.vllm.wake_up

        def wake_up(pod_ip, *, port=None):
            result = original(pod_ip, port=port)
            for gpu in self.gpus_of_ip.get(pod_ip, ()):
                awake = [ip for ip, gpus in self.gpus_of_ip.items() if gpu in gpus and self.vllm.sleeping.get(ip) is False]
                if len(awake) > 1:
                    self.violations.append((gpu, sorted(awake)))
            return result

        self.vllm.wake_up = wake_up

    def awake(self, name) -> bool:
        return self.vllm.sleeping.get(self.ip_of[name]) is False

    def stored_awake(self, name) -> bool:
        return next(b for b in self.store.load().bindings if b.serve_id == name).awake

    def lease(self, name):
        binding_id = next(b.binding_id for b in self.store.load().bindings if b.serve_id == name)
        return next((lease.phase for lease in self.leases.load() if lease.binding_id == binding_id), None)

    def state(self, name) -> str:
        return self.runtime.snapshots[name].annotations["tre.aibrix.io/state"]

    def journals(self):
        return self.service._sleep_primitive.journal.entries(), self.service._wake_journal.entries()

    def hold_inside(self, call: str, ip: str):
        """Hold the first ``call`` (sleep | wake_up) of ``ip`` until ``release``
        is set; ``entered`` is set once it is inside."""
        entered, release = threading.Event(), threading.Event()
        original = getattr(self.vllm, call)

        def held(pod_ip, **kwargs):
            if pod_ip == ip and not entered.is_set():
                entered.set()
                assert release.wait(5)
            return original(pod_ip, **kwargs)

        setattr(self.vllm, call, held)
        return entered, release


def _run(target):
    box = {}

    def run():
        try:
            box["result"] = target()
        except BaseException as exc:  # noqa: BLE001 - reported to the test
            box["error"] = exc

    thread = threading.Thread(target=run)
    thread.start()
    return thread, box


# ------------------------------------------------------------ serialization
def test_two_concurrent_requests_serialize_on_the_lock_and_keep_one_engine_per_gpu():
    """A sleep of d-0 (GPU 0) holds the lock through its /sleep; a wake of x-0 on
    the same GPU arriving meanwhile waits for the lock, then finds GPU 0 free."""
    world = World()
    entered, release = world.hold_inside("sleep", world.ip_of["d-0"])
    first, first_box = _run(lambda: world.service.put_binding_power("d-0", awake=False))
    assert entered.wait(5)
    second, second_box = _run(lambda: world.service.put_model_target("x", wake_replicas=1))
    assert world.coordinator.waiting.wait(5)  # the second request queues on the lock
    assert not world.awake("x-0")
    release.set()
    first.join(5)
    second.join(5)

    assert "error" not in first_box and "error" not in second_box, (first_box, second_box)
    assert world.coordinator.kinds == ["put_binding_power", "put_model_target"]
    assert not world.awake("d-0") and world.awake("x-0")
    assert world.lease("d-0") is None and world.lease("x-0") == "awake"
    assert world.violations == []


def test_a_replayed_scale_up_from_one_base_is_harmless():
    """I2 (2026-10-04): APA's /scale_service up 1, sent again (a retry after its
    client timeout) while the first still runs, converts at arrival from the same
    awake count - the target is base + 1 both times, not base + 2."""
    world = World(pods=(("x-2", "x", (2,), "sleeping"), ("x-3", "x", (3,), "sleeping")))
    call = lambda: world.client.post("/scale_service", params={"model_name": "x", "scale_type": "up", "scale_value": 1})
    entered, release = threading.Event(), threading.Event()
    original = world.vllm.wake_up

    def hold_first(pod_ip, *, port=None):  # whichever binding the SM picks
        if not entered.is_set():
            entered.set()
            assert release.wait(5)
        return original(pod_ip, port=port)

    world.vllm.wake_up = hold_first
    first, first_box = _run(call)
    assert entered.wait(5)
    second, second_box = _run(call)
    assert world.coordinator.waiting.wait(5)  # converted, now queued on the lock
    release.set()
    first.join(5)
    second.join(5)

    assert [box["result"].status_code for box in (first_box, second_box)] == [200, 200]
    assert [world.awake("x-2"), world.awake("x-3")].count(True) == 1
    assert world.violations == []


def test_a_target_during_a_wake_queues_instead_of_retry_later():
    world = World()
    entered, release = world.hold_inside("wake_up", world.ip_of["r-2"])
    first, first_box = _run(lambda: world.client.put("/v2/models/r/target", json={"wake_replicas": 1}))
    assert entered.wait(5)
    second, second_box = _run(lambda: world.client.put("/v2/models/r/target", json={"wake_replicas": 1}))
    assert world.coordinator.waiting.wait(5)
    release.set()
    first.join(5)
    second.join(5)

    assert first_box["result"].status_code == 200
    assert second_box["result"].status_code == 200  # never 409 "retry later"
    assert second_box["result"].json()["actions"] == []  # r already has one awake
    assert world.awake("r-2")


# ------------------------------------------------------------ hung engines
class Unanswered:
    success = False
    status_code = None
    message = "read timed out"


def test_a_sleep_without_an_answer_stays_hidden_releases_the_lock_and_is_recovered():
    world = World()
    real_sleep = world.vllm.sleep

    def hung(pod_ip, **kwargs):
        real_sleep(pod_ip, **kwargs)  # the engine does go to sleep ...
        return Unanswered()  # ... but the HTTP call times out

    world.vllm.sleep = hung
    response = world.client.put("/v2/bindings/d-0/power", json={"awake": False})

    assert response.status_code == 409
    assert response.json()["outcomes"][0]["status"] == "unconfirmed"
    assert not world.coordinator.held  # released at once
    assert world.state("d-0") == "hidden"  # routing not re-opened
    sleep_journal, _ = world.journals()
    assert sleep_journal["d-0"]["phase"] == "sleep_unconfirmed"
    assert world.lease("d-0") == "awake"  # its GPU stays taken until settled

    result = world.service.recover_sleep_journal()

    assert result["resolved"] == [{"serve_id": "d-0", "result": "slept"}]
    assert world.lease("d-0") is None and not world.stored_awake("d-0")
    assert world.journals() == ({}, {})


def test_a_wake_without_an_answer_keeps_its_fence_releases_the_lock_and_is_recovered():
    world = World()

    def hung(pod_ip, *, port=None):
        world.vllm.sleeping[pod_ip] = False  # the engine wakes after all ...
        raise TimeoutError("read timed out")  # ... the call does not answer

    world.vllm.wake_up = hung
    response = world.client.put("/v2/bindings/r-2/power", json={"awake": True})

    assert response.status_code == 409 and response.json()["error"] == "wake_failed"
    assert not world.coordinator.held
    _, wake_journal = world.journals()
    assert set(wake_journal) == {"r/node-a/2"}
    assert world.lease("r-2") == "awake"  # GPU 2 stays fenced
    assert world.service.recover_wake_journal() == {"resolved": [], "kept": []}  # recheck later

    world.service._wake_journal.update("r/node-a/2", recover_after_ms=0)
    result = world.service.recover_wake_journal()

    assert result["resolved"] == [{"binding_id": "r/node-a/2", "result": "completed"}]
    assert world.stored_awake("r-2") and world.state("r-2") == "awake"
    assert world.journals() == ({}, {})


# ------------------------------------------------------------ transfers
def test_transfer_sleeps_the_donor_and_wakes_the_receiver_in_one_lock_hold():
    world = World()

    response = world.client.post("/v2/transfers", json={"donor_model": "d", "receiver_model": "r", "count": 1})

    assert response.status_code == 200, response.text
    body = response.json()
    [pair] = body["pairs"]
    assert pair["status"] == "done"
    assert body["done"] == body["taken"] == 1
    assert body["unfilled"] == 0 and body["clamped_by_floor"] is False
    assert set(body["phases_ms"]) == {"select", "donor_sleep", "receiver_wake", "total"}
    assert not world.awake(pair["donor"]) and world.awake(pair["receiver"])
    assert world.lease(pair["donor"]) is None and world.lease(pair["receiver"]) == "awake"
    assert world.coordinator.kinds == ["transfer"]
    [(_kind, handle)] = world.coordinator.handles
    assert handle.notes["transfer"]["pairs"][0]["status"] == "done"  # GET /v2/operations
    assert "phases_ms" in handle.notes
    assert world.violations == [] and world.journals() == ({}, {})


def test_transfer_tp2_receiver_is_not_woken_when_one_donor_fails_to_sleep():
    world = World(pods=(("d-0", "d", (0,), "awake"), ("d-1", "d", (1,), "awake"), ("t-01", "t", (0, 1), "sleeping")))
    world.vllm.fail_sleep_for.add(world.ip_of["d-1"])

    response = world.client.post("/v2/transfers", json={"donor_model": "d", "receiver_model": "t", "count": 2})

    assert response.status_code == 409 and response.json()["error"] == "partial"
    assert response.json()["pairs"][0]["status"] == "donor_sleep_failed"
    assert not any(call == ("wake_up", world.ip_of["t-01"]) for call in world.vllm.calls)
    assert world.awake("d-1") and world.state("d-1") == "awake"  # rolled back, routable
    assert world.lease("t-01") is None
    assert world.violations == []


def test_transfer_receiver_wake_failure_sleeps_it_back_and_keeps_the_donor_asleep():
    world = World(pods=(("d-0", "d", (0,), "awake"), ("r-0", "r", (0,), "sleeping")))
    real_wake = world.vllm.wake_up

    def late(pod_ip, *, port=None):
        real_wake(pod_ip, port=port)  # it woke ...
        return Result(False, "engine error")  # ... and reported a failure

    world.vllm.wake_up = late

    with pytest.raises(TransferFailed) as info:
        world.service.transfer(donor_model="d", receiver_model="r")

    [pair] = info.value.response["pairs"]
    assert pair["status"] == "receiver_wake_failed"
    assert pair["compensating_sleep"]["done"] is True
    assert not world.awake("r-0") and world.lease("r-0") is None
    assert not world.awake("d-0") and not world.stored_awake("d-0")  # the donor is not rolled back


def test_transfer_receiver_pod_list_failure_after_the_donor_slept_still_answers():
    """Review 2026-10-06 P3-3: the receivers' Pod LIST fails after the donor
    slept. The caller still gets the transfer answer (409 partial with pairs and
    taken), the receiver is untouched and the donor stays asleep."""
    world = World(pods=(("d-0", "d", (0,), "awake"), ("r-0", "r", (0,), "sleeping")))
    real_list = world.runtime.list_pod_snapshots

    def flaky(*, model=None):
        if model == "r" and world.vllm.sleeping.get(world.ip_of["d-0"]):
            raise RuntimeError("api server timeout")
        return real_list(model=model)

    world.runtime.list_pod_snapshots = flaky

    response = world.client.post("/v2/transfers", json={"donor_model": "d", "receiver_model": "r"})

    assert response.status_code == 409
    body = response.json()
    assert body["error"] == "partial" and body["taken"] == 1 and body["done"] == 0
    [pair] = body["pairs"]
    assert pair["status"] == "receiver_wake_failed"
    assert "api server timeout" in pair["error"]["detail"]
    assert not any(call == ("wake_up", world.ip_of["r-0"]) for call in world.vllm.calls)
    assert not world.awake("r-0") and world.lease("r-0") is None
    assert not world.awake("d-0") and not world.stored_awake("d-0") and world.lease("d-0") is None
    assert world.violations == [] and world.journals() == ({}, {})


def test_transfer_crash_after_the_donor_slept_is_settled_by_the_journals():
    world = World(pods=(("d-0", "d", (0,), "awake"), ("r-0", "r", (0,), "sleeping")))

    class Crash(BaseException):
        pass

    def crash(*_args, **_kwargs):
        raise Crash()  # the SM dies right after the donor's physical confirmation

    world.service._record_sleep_outcomes = crash
    with pytest.raises(Crash):
        world.service.transfer(donor_model="d", receiver_model="r")
    assert not world.awake("d-0") and world.lease("d-0") == "awake"  # books not written yet

    world.service = world.make_service()  # a new SM process (the dead one's lock lease expired)
    world.service.recover_sleep_journal()
    world.service.recover_wake_journal()

    assert not world.stored_awake("d-0") and world.lease("d-0") is None
    assert next(d.power for d in world.fleet.load_desired().bindings if d.binding_id == "d/node-a/0") == "sleeping"
    assert not world.awake("r-0") and world.lease("r-0") is None  # untouched, no orphan lease
    assert world.leases.load() == []
    assert world.journals() == ({}, {})


def test_transfer_held_back_by_the_donor_floor_is_200_clamped():
    world = World(d_min=2)

    response = world.client.post("/v2/transfers", json={"donor_model": "d", "receiver_model": "r", "count": 1})

    assert response.status_code == 200
    body = response.json()
    assert body["clamped_by_floor"] is True and body["taken"] == 0 and body["unfilled"] == 1
    assert world.awake("d-0") and world.awake("d-1")


def test_transfer_on_the_safescale_commit_path_is_a_400():
    world = World()
    response = world.client.post(
        "/v2/transfers", json={"donor_model": "d", "receiver_model": "r", "sleep_path": "safescale_commit"}
    )
    assert response.status_code == 400


# ------------------------------------------------------------ routable view / floor
def test_v2_state_carries_the_routable_view_and_the_v1_api_lists_no_pods():
    world = World()
    lists = []
    original = world.runtime.list_pod_snapshots
    world.runtime.list_pod_snapshots = lambda **kwargs: lists.append(kwargs) or original(**kwargs)

    state = world.client.get("/v2/state").json()

    assert len(lists) == 1  # one Pod LIST for every model
    assert isinstance(state["fetched_ms"], int) and state["floor_enforced"] is True
    by_name = {b["serve_id"]: b["routable"] for b in state["bindings"]}
    assert by_name["d-0"] is True and by_name["r-0"] is False
    assert state["models"]["d"]["routable"] == 2 and state["models"]["d"]["floor_headroom"] == 2

    lists.clear()
    assert world.client.post("/models_replicas", params={"models": "d,r"}).json() == {"d": 2, "r": 0}
    assert lists == []  # the APA arm's read path is as light as before


def test_a_shrink_with_an_unreadable_routable_view_is_409_before_any_hide():
    world = World()
    world.runtime.list_pod_snapshots = lambda **kwargs: (_ for _ in ()).throw(ConnectionError("apiserver"))

    response = world.client.put("/v2/models/d/target", json={"wake_replicas": 0})

    assert response.status_code == 409 and response.json()["error"] == "routable_unknown"
    assert not [p for p in world.runtime.patches if p[1] == "hidden"]
    assert world.awake("d-0") and world.awake("d-1")


# ------------------------------------------------------------ supervisor
def test_a_failing_recovery_step_never_stops_the_others():
    calls = []

    class Service:
        def __getattr__(self, name):
            def record(*args, **kwargs):
                calls.append(name)
                if name == "recover_sleep_journal":
                    raise RuntimeError("redis down")
                return [] if name != "converge_startups" else {}
            return record

        def recover_stale_fleet_repairs(self, **kwargs):
            return None

        def detect_fleet_drift(self):
            return []

        def actuation_observe(self):
            return False

    supervisor = FleetSupervisor(Service())
    supervisor.run_once()

    assert "recover_wake_journal" in calls and "converge_startups" in calls
    assert "recover_sleep_journal: RuntimeError: redis down" in supervisor.snapshot().last_error


def test_stale_operation_records_are_superseded_except_fleet_repairs():
    world = World()
    superseded = []

    class Stale(BlockingCoordinator):
        def stale_running_operations(self, *, kind=None):
            return [{"operation_id": "op-dead", "kind": "put_model_target"},
                    {"operation_id": "op-repair", "kind": "fleet_repair"}]

    world.service._operation_coordinator = Stale(world.redis)
    original = FakeHandle.supersede
    FakeHandle.supersede = lambda self, operation_id: superseded.append(operation_id)
    try:
        assert world.service.supersede_stale_operations() == ["op-dead"]
    finally:
        FakeHandle.supersede = original
    assert superseded == ["op-dead"]
