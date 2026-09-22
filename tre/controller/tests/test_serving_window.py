"""Sleeping pods write gateway docs too; the decision window must count only serving pods.

Live 09-22: the metrics window of dsqwen-7b reported routable_pods=8 (1 awake + 7 asleep,
each asleep pod writing zero-gauge instant docs and flat histogram docs) while the fleet
state said 1. ``restrict_to_serving`` (via ``tick.serving_window``) drops the docs of
pods the fleet state reports asleep and takes the pod count from it, for the planner tick
and the safescale observation alike. A window of 1 awake + N sleeping pods must then give
the same Z and the same per-replica alternative signals as 1 awake pod alone - which is
also what the single-awake-pod calibration capture measured offline.
"""
from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from tre_common.alt_signals import queue_len_per_replica
from tre_common.metrics_schema import MetricsSnapshot
from tre_common.window_pods import restrict_to_serving
from tre_controller.loops.rescue_task import run_rescue_tick
from tre_controller.loops.safescale_task import run_safescale_observation_tick
from tre_controller.loops.tick import serving_window
from tre_controller.planning.planner import ClusterView
from tre_controller.store.metrics_store import MetricsStore
from tre_sm.allocator.slots import Binding, Slot

from test_band_dwell import _Queue, _registry

END = 1_790_000_030_000
P = 10_000
AWAKE = "m-node-a-gpu-0-abc"
ASLEEP = [f"m-node-a-gpu-{i}-zz{i}" for i in range(1, 8)]


class _Redis:
    def __init__(self) -> None:
        self.sets: dict = {}
        self.zsets: dict = {}

    def smembers(self, key):
        return set(self.sets.get(key, set()))

    def zrangebyscore(self, key, lo, hi):
        return [m for s, m in self.zsets.get(key, []) if float(lo) <= s <= float(hi)]

    def add(self, key, ts, doc):
        self.zsets.setdefault(key, []).append((float(ts), json.dumps(dict(doc, timestamp=ts))))
        self.zsets[key].sort(key=lambda item: item[0])


def _pod_docs(redis: _Redis, pod: str, *, busy: bool) -> None:
    redis.sets.setdefault("tre:v2:pods:m", set()).add(f"default/{pod}")
    for k, tick in enumerate(range(END - 40_000, END + 1, P)):
        redis.add(f"tre:v2:inst:default/{pod}", tick, {"pod_name": pod, "model_metrics": {
            "m/num_requests_running": 4.0 if busy else 0.0,
            "m/num_requests_waiting": 1.0 if busy else 0.0,
        }})
        gen = 50.0 * k if busy else 777.0  # a sleeping pod's counters are flat (stale total)
        redis.add(f"tre:v2:hist:default/{pod}", tick, {"pod_name": pod, "model_histogram_metrics": {
            "m/request_generation_tokens": {"sum": gen, "count": k if busy else 9, "buckets": {}},
            "m/request_prompt_tokens": {"sum": gen / 5, "count": k if busy else 9, "buckets": {}},
        }})


def _store(pods: dict[str, bool]) -> MetricsStore:
    redis = _Redis()
    for pod, busy in pods.items():
        _pod_docs(redis, pod, busy=busy)
    registry = SimpleNamespace(models=lambda: [SimpleNamespace(name="m")])
    return MetricsStore(redis, registry, instant_sample_interval_ms=P, schema="v2")


def _view(awake: list[str], asleep: list[str]) -> ClusterView:
    registry = _registry()
    bindings = [Binding(pod, "m", Slot("node-a", (i % 4,)), awake=True) for i, pod in enumerate(awake)]
    bindings += [Binding(pod, "m", Slot("node-a", (i % 4,)), awake=False) for i, pod in enumerate(asleep)]
    return ClusterView(topology=registry.topology(), bindings=tuple(bindings))


def _read(store: MetricsStore):
    return store.read_snapshot(END - 30_000, END, use_cache=False, start_exclusive=True)


def test_raw_window_counts_sleeping_pods() -> None:
    raw = _read(_store({AWAKE: True, **{p: False for p in ASLEEP}})).models["m"]
    alone = _read(_store({AWAKE: True})).models["m"]
    assert raw.routable_pods == 8 and alone.routable_pods == 1
    # the sums are not diluted (sleeping gauges are 0, flat counters give 0 tokens) ...
    assert (raw.generation_tokens, raw.avg_running, raw.avg_waiting) == (
        alone.generation_tokens, alone.avg_running, alone.avg_waiting)
    # ... but anything per replica is divided by 8 instead of 1
    assert queue_len_per_replica(raw.avg_running, raw.avg_waiting, raw.routable_pods) == pytest.approx(
        queue_len_per_replica(alone.avg_running, alone.avg_waiting, alone.routable_pods) / 8)


def test_restrict_to_serving_equals_the_awake_pod_alone() -> None:
    raw = _read(_store({AWAKE: True, **{p: False for p in ASLEEP}})).models["m"]
    alone = _read(_store({AWAKE: True})).models["m"]
    served = restrict_to_serving(raw, sleeping_pods=ASLEEP, routable_pods=1)
    assert served == alone
    assert served.instant_ticks_ms == raw.instant_ticks_ms
    assert serving_window(raw, _view([AWAKE], ASLEEP)) == alone


def test_unknown_pods_are_kept_and_no_view_is_a_noop() -> None:
    raw = _read(_store({AWAKE: True, **{p: False for p in ASLEEP}})).models["m"]
    assert serving_window(raw, None) is raw
    # the view does not list the sleepers (just replaced): keep their docs, fix the count
    kept = serving_window(raw, _view([AWAKE], []))
    assert len(kept.per_pod) == 8 and kept.routable_pods == 1


def test_planner_z_is_the_same_with_or_without_sleeping_pods() -> None:
    registry = _registry()
    with_sleepers = _read(_store({AWAKE: True, **{p: False for p in ASLEEP}}))
    alone = _read(_store({AWAKE: True}))
    a = run_rescue_tick(with_sleepers, queue=_Queue(), registry=registry, cluster_view=_view([AWAKE], ASLEEP))
    b = run_rescue_tick(alone, queue=_Queue(), registry=registry, cluster_view=_view([AWAKE], []))
    ca, cb = a.model_contexts["m"], b.model_contexts["m"]
    for key in ("z_m", "trs", "Q", "Y_m", "y_m", "eta_m", "routable_pods", "decode_tps", "prefill_tps"):
        assert ca[key] == cb[key], key
    assert ca["routable_pods"] == 1


class _Probe:
    model = "m"


class _Observer:
    def __init__(self) -> None:
        self.observations = []

    def active_probes(self):
        return (_Probe(),)

    def observe(self, model, observation, *, now_ms):
        self.observations.append(observation)
        return SimpleNamespace(reason="observing", commands=(), status="observing")

    def resolve(self, *args, **kwargs):
        return True


def test_safescale_observation_uses_the_serving_window() -> None:
    registry = _registry()
    with_sleepers = _read(_store({AWAKE: True, **{p: False for p in ASLEEP}}))
    alone = _read(_store({AWAKE: True}))
    obs_a, obs_b = _Observer(), _Observer()
    run_safescale_observation_tick(with_sleepers, queue=_Queue(), registry=registry, safescale=obs_a,
                                   cluster_view=_view([AWAKE], ASLEEP))
    run_safescale_observation_tick(alone, queue=_Queue(), registry=registry, safescale=obs_b,
                                   cluster_view=_view([AWAKE], []))
    assert obs_a.observations[0] == obs_b.observations[0]


def test_snapshot_level_equality_after_restriction_keeps_freshness_ticks() -> None:
    raw = _read(_store({AWAKE: True, **{p: False for p in ASLEEP}}))
    assert isinstance(raw, MetricsSnapshot)
    assert raw.models["m"].instant_ticks_ms == (END - 20_000, END - 10_000, END)
