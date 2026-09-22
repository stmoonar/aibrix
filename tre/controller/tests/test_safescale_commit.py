from __future__ import annotations

import asyncio
import json

import pytest

from tre_common import rediskeys
from tre_common.metrics_schema import MetricsSnapshot, ModelWindowMetrics
from tre_common.registry import ClusterTopology, ModelSpec, NodeSpec, Registry, SloSpec, TrsParams
from tre_controller.config import SafeScaleConfig
from tre_controller.loops.action_queue import ActionQueue, SubmitResult
from tre_controller.loops.safescale_task import run_safescale_observation_tick
from tre_controller.planning.safescale import SafeScaleStateMachine
from tre_controller.reconcile.hidden_orphans import HiddenOrphanDetector
from tre_controller.store.state_store import ControllerStateStore


class FakeRedis:
    def __init__(self):
        self.hashes = {}
        self.lists = {}

    def hset(self, name, key=None, value=None, mapping=None):
        bucket = self.hashes.setdefault(name, {})
        values = mapping if mapping is not None else {key: value}
        for field, payload in values.items():
            bucket[str(field).encode("utf-8")] = str(payload).encode("utf-8")
        return len(values)

    def hgetall(self, name):
        return dict(self.hashes.get(name, {}))

    def hdel(self, name, *keys):
        bucket = self.hashes.setdefault(name, {})
        removed = 0
        for key in keys:
            removed += int(bucket.pop(str(key).encode("utf-8"), None) is not None)
        return removed

    def rpush(self, name, *values):
        bucket = self.lists.setdefault(name, [])
        for value in values:
            bucket.append(str(value).encode("utf-8"))
        return len(bucket)

    def lrange(self, name, start, end):
        values = self.lists.get(name, [])
        end_index = None if int(end) == -1 else int(end) + 1
        return list(values[int(start):end_index])


class OutcomeQueue:
    def __init__(self, outcome):
        self.outcome = outcome
        self.submitted = []

    def submit(self, actions):
        self.submitted.append(tuple(actions))
        if isinstance(self.outcome, Exception):
            raise self.outcome
        return SubmitResult(accepted=self.outcome)


class FakeServiceManager:
    async def scale_model(self, model, delta):
        return {"ok": True}

    async def set_routable(self, model, hidden_pods):
        return {"ok": True}

    async def defrag(self, migrations):
        return {"ok": True}


def _registry() -> Registry:
    spec = ModelSpec(
        name="donor",
        weights_path="/weights",
        tp_size=1,
        min_replicas=0,
        max_replicas=4,
        vllm_image="image",
        slo=SloSpec(
            ttft_p95_ms=1000.0,
            tpot_p95_ms=100.0,
            e2e_p95_ms=10_000.0,
        ),
        trs=TrsParams(
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
        ),
    )
    topology = ClusterTopology(
        nodes=(NodeSpec(name="node-a", gpus=4, two_gpu_slots=((0, 1), (2, 3))),)
    )
    return Registry(topology, [spec])


def _metrics(ts_ms: int) -> MetricsSnapshot:
    return MetricsSnapshot(
        ts_ms=ts_ms,
        stale=False,
        models={
            "donor": ModelWindowMetrics(
                model="donor",
                window_start_ms=0,
                window_end_ms=60_000,
                prompt_tokens=0.0,
                generation_tokens=120.0,
                avg_waiting=0.0,
                avg_running=1.0,
                avg_swapping=0.0,
                kv_cache_hit_rate=0.0,
                ttft_p95_ms=500.0,
                tpot_p95_ms=50.0,
                e2e_p95_ms=1000.0,
                routable_pods=1,
                assigned_replicas=1,
                per_pod={},
            )
        },
    )


def _machine(store):
    return SafeScaleStateMachine(
        config=SafeScaleConfig(
            ttft_p95_slo_ms=1000.0,
            tpot_p95_slo_ms=100.0,
            default_window_ms=1000.0,
            hq=0.5,
            tau_low=1.0,
        ),
        store=store,
    )


def _start_and_prime(machine):
    machine.start_probe(
        model="donor",
        pods=("pod-a",),
        now_ms=0,
        pending_upscales={"receiver": 1},
    )
    run_safescale_observation_tick(
        _metrics(500),
        queue=OutcomeQueue(0),
        registry=_registry(),
        safescale=machine,
    )


def _probe_records(redis):
    return {
        key.decode("utf-8"): json.loads(value)
        for key, value in redis.hgetall(
            rediskeys.CONTROLLER_SAFESCALE_PROBES_KEY
        ).items()
    }


def test_commit_enqueue_success_marks_probe_resolved_without_deleting_it():
    redis = FakeRedis()
    store = ControllerStateStore(redis)
    machine = _machine(store)
    _start_and_prime(machine)
    queue = OutcomeQueue(2)

    result = run_safescale_observation_tick(
        _metrics(1000), queue=queue, registry=_registry(), safescale=machine
    )

    assert result.submitted == 2
    assert len(queue.submitted) == 1
    record = next(iter(_probe_records(redis).values()))
    assert record["status"] == "resolved"
    assert record["resolution"] == "commit"
    assert record["resolved_ts"] == 1.0
    assert machine.active_probe("donor") is None


@pytest.mark.parametrize("outcome", [0, RuntimeError("queue unavailable")])
def test_commit_enqueue_failure_leaves_probe_restorable(outcome):
    redis = FakeRedis()
    store = ControllerStateStore(redis)
    machine = _machine(store)
    _start_and_prime(machine)

    result = run_safescale_observation_tick(
        _metrics(1000),
        queue=OutcomeQueue(outcome),
        registry=_registry(),
        safescale=machine,
    )

    assert result.submitted == 0
    assert next(iter(_probe_records(redis).values()))["status"] == "probing"
    assert machine.active_probe("donor") is not None
    restored = _machine(store)
    assert restored.restore() == 1
    assert restored.active_probe("donor") is not None


def test_gc_deletes_only_resolved_probes_older_than_one_hour():
    redis = FakeRedis()
    store = ControllerStateStore(redis)
    base = {
        "model": "donor",
        "pods": ["pod-a"],
        "start_ms": 0,
        "deadline_ms": 1000,
    }
    store.save_probe(
        "old", {**base, "status": "resolved", "resolved_ts": 100.0}
    )
    store.save_probe(
        "fresh", {**base, "status": "resolved", "resolved_ts": 200.0}
    )
    store.save_probe(
        "probing", {**base, "status": "probing", "resolved_ts": 0.0}
    )

    assert store.gc_resolved_probes(now_ts=3701.0) == ("old",)
    assert set(_probe_records(redis)) == {"fresh", "probing"}


def test_observe_mode_holds_commit_and_orphan_detector_stays_quiet():
    redis = FakeRedis()
    store = ControllerStateStore(redis)
    machine = _machine(store)
    _start_and_prime(machine)
    queue = ActionQueue(FakeServiceManager(), is_observe=lambda: True)

    result = run_safescale_observation_tick(
        _metrics(1000), queue=queue, registry=_registry(), safescale=machine
    )
    assert result.submitted == 2
    assert next(iter(_probe_records(redis).values()))["status"] == "resolved"

    assert asyncio.run(queue.drain_once()) == ()
    assert len(queue.pending_actions()) == 2
    redis.hset(
        rediskeys.SM_STATE_KEY,
        mapping={
            "pod-a": json.dumps(
                {"model": "donor", "awake": True, "hidden": True}
            )
        },
    )
    detector = HiddenOrphanDetector(redis, grace_s=600)

    assert detector.scan(now=2.0) == ()
    assert redis.hgetall(rediskeys.CONTROLLER_HIDDEN_ORPHAN_ALERTS_KEY) == {}
