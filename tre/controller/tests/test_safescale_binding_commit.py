"""Review F2/F3: a safescale probe must hide a *serving* pod and its commit must sleep
exactly that pod (binding-level power), never a different serving pod."""
from __future__ import annotations

import asyncio

from tre_common.metrics_schema import MetricsSnapshot, ModelWindowMetrics
from tre_common.registry import ClusterTopology, ModelSpec, NodeSpec, Registry, SloSpec, TrsParams
from tre_controller.app import _active_probe_models
from tre_controller.config import SafeScaleConfig
from tre_controller.loops.action_queue import ActionQueue
from tre_controller.loops.cluster_view_task import ClusterViewBox, cluster_view_from_state
from tre_controller.loops.metrics_task import SnapshotBox
from tre_controller.loops.rescue_task import rescue_task
from tre_controller.loops.safescale_task import run_safescale_observation_tick
from tre_controller.loops.tick import _commands_to_actions, _idle_gpus, _pods_to_probe
from tre_controller.planning.planner import ClusterView, TransferIntent
from tre_controller.planning.safescale import SafeScaleStateMachine
from tre_sm.allocator.slots import Binding, Slot, natural_key
from tre_sm.api.v2 import ServiceManagerV2
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


class InProcessServiceManager:
    """Controller SM client backed by the real ServiceManagerV2 service layer."""

    def __init__(self, service: ServiceManagerV2) -> None:
        self.service = service
        self.calls: list[tuple] = []

    async def scale_model(self, model, delta, **_kwargs):
        self.calls.append(("scale_model", model, delta))
        awake = self.service.get_state()["models"][model]["awake"]
        # Forward the sleep path / drain budget like the HTTP client does.
        self.service.put_model_target(model, wake_replicas=max(0, awake + int(delta)), **_kwargs)
        return {"ok": True}

    async def set_routable(self, model, hidden_pods):
        self.calls.append(("set_routable", model, tuple(hidden_pods)))
        self.service.put_model_routable(model, hidden_pods=list(hidden_pods))
        return {"ok": True}

    async def set_binding_power(self, serve_id, *, awake, **_kwargs):
        self.calls.append(("set_binding_power", serve_id, awake))
        self.service.put_binding_power(serve_id, awake=awake, **_kwargs)
        return {"ok": True}

    async def defrag(self, migrations):
        return {"ok": True}

    async def transfer(self, donor_model, receiver_model, count, *, sleep_path="urgent"):
        """Fake ``POST /v2/transfers`` (2026-10-02) on the in-process service, whose
        service-manager has no transfer primitive: the SM's pair rule (a sleeping, not
        hidden receiver binding all of whose GPUs awake, not hidden donor bindings hold),
        then a binding-level donor sleep and receiver wake per pair; the response has
        the SM's shape (pairs, done, taken, unfilled, picked)."""
        self.calls.append(("transfer", donor_model, receiver_model, int(count)))
        state = self.service.get_state()
        awake = {}
        for item in state["bindings"]:
            if item.get("awake"):
                for gpu in item.get("gpu_ids") or ():
                    awake[(item["node"], int(gpu))] = item
        used: set[str] = set()
        pairs, taken = [], 0
        receivers = sorted(
            (item for item in state["bindings"]
             if item["model"] == receiver_model and not item.get("awake") and not item.get("hidden")),
            key=lambda item: natural_key(item["serve_id"]),
        )
        for item in receivers:
            gpus = [(item["node"], int(gpu)) for gpu in item.get("gpu_ids") or ()]
            if not gpus or not all(gpu in awake for gpu in gpus):
                continue
            occupants = {awake[gpu]["serve_id"]: awake[gpu] for gpu in gpus}
            if any(o["model"] != donor_model or o.get("hidden") or sid in used for sid, o in occupants.items()):
                continue
            if taken + len(occupants) > int(count):
                continue
            for serve_id in occupants:
                self.service.put_binding_power(serve_id, awake=False, sleep_path=sleep_path)
            self.service.put_binding_power(item["serve_id"], awake=True)
            used.update(occupants)
            taken += len(occupants)
            pairs.append({"donor": next(iter(occupants)), "donors": sorted(occupants),
                          "receiver": item["serve_id"], "node": item["node"],
                          "gpu_ids": list(item.get("gpu_ids") or ()), "status": "done"})
        body = {"transfer_id": f"tr-fake-{len(self.calls)}", "donor_model": donor_model,
                "receiver_model": receiver_model, "count": int(count), "pairs": pairs,
                "done": len(pairs), "taken": taken, "clamped_by_floor": False,
                "unfilled": int(count) - taken, "refusals": [], "skipped": {},
                "picked": [{"serve_id": pair["receiver"]} for pair in pairs]}
        if not pairs:
            return {"ok": False, "error": "HTTP 409: partial", "status": 409, "retriable": False,
                    "partial": True, "response": body}
        return {"ok": True, "response": body}


def _trs() -> TrsParams:
    return TrsParams(
        w_p=0.04,
        w_d=1.0,
        lambda_wait=2.0,
        qmin=1.0,
        ema_alpha=0.0,
        theta_m=100.0,
        tau_crit=0.8,
        tau_low=1.0,
        tau_high=1.25,
        qsat=4.0,
        epsat=0.05,
        hsat=1,
    )


def _registry(*names: str) -> Registry:
    slo = SloSpec(ttft_p95_ms=1000.0, tpot_p95_ms=100.0, e2e_p95_ms=10_000.0)
    topology = ClusterTopology(
        nodes=(NodeSpec(name="node-a", gpus=4, two_gpu_slots=((0, 1), (2, 3))),)
    )
    return Registry(
        topology,
        [
            ModelSpec(
                name=name,
                weights_path="/weights",
                tp_size=1,
                min_replicas=0,
                max_replicas=4,
                vllm_image="image",
                slo=slo,
                trs=_trs(),
            )
            for name in names
        ],
    )


def _window(model: str, *, generation: float, waiting: float, running: float, pods: int) -> ModelWindowMetrics:
    return ModelWindowMetrics(
        model=model,
        window_start_ms=0,
        window_end_ms=60_000,
        prompt_tokens=0.0,
        generation_tokens=generation,
        avg_waiting=waiting,
        avg_running=running,
        avg_swapping=0.0,
        kv_cache_hit_rate=0.0,
        ttft_p95_ms=100.0,
        tpot_p95_ms=10.0,
        e2e_p95_ms=1000.0,
        routable_pods=pods,
        assigned_replicas=pods,
        per_pod={},
    )


def test_probe_hides_serving_pod_and_commit_sleeps_exactly_that_pod() -> None:
    registry = _registry("donor")
    store = StateStore(FakeRedis())
    store.save(
        [Binding(f"p{i}", "donor", Slot("node-a", (i - 1,)), awake=True) for i in range(1, 5)]
        + [Binding("s0", "donor", Slot("node-a", (0,)), awake=False)],
        expected_version=0,
    )
    service = ServiceManagerV2(registry, store)
    sm = InProcessServiceManager(service)
    queue = ActionQueue(sm)
    machine = SafeScaleStateMachine(
        config=SafeScaleConfig(
            ttft_p95_slo_ms=1000.0, tpot_p95_slo_ms=100.0, min_window_ms=1000.0, hq=0.5
        )
    )
    snapshot = MetricsSnapshot(
        ts_ms=0,
        stale=False,
        models={"donor": _window("donor", generation=120.0, waiting=0.0, running=1.0, pods=4)},
    )

    view = cluster_view_from_state(service.get_state(), registry.topology())
    pods = _pods_to_probe(snapshot, "donor", 1, cluster_view=view)
    # awake && !hidden only -- the sleeping s0 is never probed. p1..p4 fill the node, so
    # no single release merges upwards; the tie goes to the highest address, p4.
    assert pods == ("p4",)
    assert "s0" not in pods

    started = machine.start_probe(model="donor", pods=pods, now_ms=0)
    queue.submit(_commands_to_actions(started.commands, source_loop="rescue"))
    asyncio.run(queue.drain_once())
    assert {b.serve_id for b in store.load().bindings if b.hidden} == {"p4"}

    for ts in (500, 1000):
        run_safescale_observation_tick(
            MetricsSnapshot(ts_ms=ts, stale=False, models=snapshot.models),
            queue=queue,
            registry=registry,
            safescale=machine,
        )
    asyncio.run(queue.drain_once())

    assert ("set_binding_power", "p4", False) in sm.calls
    assert not any(call[0] == "scale_model" for call in sm.calls)
    bindings = {b.serve_id: b for b in store.load().bindings}
    assert bindings["p4"].awake is False and bindings["p4"].hidden is False
    assert all(bindings[p].awake and not bindings[p].hidden for p in ("p1", "p2", "p3"))
    assert not any(b.hidden for b in bindings.values())  # no hidden orphan


def test_probe_order_frees_an_aligned_pair_before_a_stranded_gpu() -> None:
    """Shrinking by one must give back a pair a tp=2 model can wake into."""
    registry = _registry("donor")
    view = ClusterView(
        registry.topology(),
        (
            Binding("d-0", "donor", Slot("node-a", (2,)), awake=True),
            Binding("d-1", "donor", Slot("node-a", (3,)), awake=True),
            Binding("d-2", "donor", Slot("node-a", (0,)), awake=True),
        ),
    )
    snapshot = MetricsSnapshot(ts_ms=0, stale=False, models={})

    # gpu1 is already free, so sleeping d-2 (gpu0) completes the aligned pair (0,1),
    # while sleeping d-0 or d-1 only strands another single GPU. Lexicographic order
    # would have probed d-0 and freed nothing a tp=2 model can use.
    assert _pods_to_probe(snapshot, "donor", 1, cluster_view=view) == ("d-2",)
    # Second pick is scored with gpu0 already released: neither remaining GPU merges,
    # so the tie goes to the highest address.
    assert _pods_to_probe(snapshot, "donor", 2, cluster_view=view) == ("d-2", "d-1")


def test_idle_gpus_counts_gpus_without_awake_binding_from_cluster_view() -> None:
    registry = _registry("a", "b")
    view = ClusterView(
        registry.topology(),
        (
            Binding("a-0", "a", Slot("node-a", (0,)), awake=True),
            Binding("a-1", "a", Slot("node-a", (1,)), awake=True, hidden=True),
            Binding("b-1", "b", Slot("node-a", (1,)), awake=False),
            Binding("b-2", "b", Slot("node-a", (2,)), awake=False),
        ),
    )
    snapshot = MetricsSnapshot(ts_ms=0, stale=False, models={})

    # gpu0 awake, gpu1 hidden-but-awake (still occupied), gpu2 only sleeping, gpu3 empty.
    assert _idle_gpus(snapshot, registry, view) == 2


class _StopLoop(Exception):
    pass


async def _stop_sleep(_seconds):
    raise _StopLoop()


class _Queue:
    def __init__(self):
        self.submitted = []

    def inflight_models(self):
        return set()

    def submit(self, actions):
        self.submitted.append(tuple(actions))
        return object()


def _run_rescue_once(*, probed: bool) -> list:
    registry = _registry("critical", "donor")
    snapshot = MetricsSnapshot(
        ts_ms=1,
        stale=False,
        models={
            "critical": _window("critical", generation=50.0, waiting=10.0, running=1.0, pods=1),  # + sleeping critical-1 under donor-1
            "donor": _window("donor", generation=100_000.0, waiting=0.0, running=1.0, pods=3),
        },
    )
    view = ClusterView(
        registry.topology(),
        (
            Binding("critical-0", "critical", Slot("node-a", (0,)), awake=True),
            Binding("critical-1", "critical", Slot("node-a", (1,)), awake=False),
            Binding("donor-1", "donor", Slot("node-a", (1,)), awake=True),
            Binding("donor-2", "donor", Slot("node-a", (2,)), awake=True),
            Binding("donor-3", "donor", Slot("node-a", (3,)), awake=True),
        ),
    )
    machine = SafeScaleStateMachine(config=SafeScaleConfig())
    if probed:
        machine.start_probe(model="donor", pods=("donor-3",), now_ms=0)
    queue = _Queue()
    try:
        asyncio.run(
            rescue_task(
                SnapshotBox(snapshot),
                queue=queue,
                registry=registry,
                cfg=type("Cfg", (), {"rescue_interval_s": 5.0})(),
                sleep=_stop_sleep,
                cluster_view_box=ClusterViewBox(view),
                active_probe_models=lambda: _active_probe_models(machine),
                safescale=machine,
            )
        )
    except _StopLoop:
        pass
    return [action for batch in queue.submitted for action in batch]


def test_rescue_task_excludes_actively_probed_model_from_immediate_donors() -> None:
    control = _run_rescue_once(probed=False)
    # 2026-10-02: the immediate relay is one TransferIntent (donor -> receiver, a count).
    assert any(
        isinstance(action, TransferIntent) and action.donor_model == "donor" and action.count >= 1
        for action in control
    )

    probed = _run_rescue_once(probed=True)
    assert not any(
        getattr(action, "model", None) == "donor" or getattr(action, "donor_model", None) == "donor"
        for action in probed
    )
