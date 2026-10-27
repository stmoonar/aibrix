"""The writer-lock hold must not grow with the number of targets (review
2026-10-06 P2-1): the Kubernetes calls a sleep or a wake makes one
after the other stay within the K8S_CALLS_* the worst-case formulas count,
whatever the number of bindings (per-target calls run in parallel)."""

import contextvars
from concurrent.futures import ThreadPoolExecutor

from tre_common.registry import K8S_CALLS_SLEEP, K8S_CALLS_WAKE

import tre_sm.api.v2 as v2_module
import tre_sm.ops.sleep_primitive as primitive_module
from test_wholelock_20261002 import World

#: Kubernetes calls per runtime call (an annotation write is a read + a patch).
WEIGHT = {"write_binding_annotations": 2, "set_pod_routable": 2}

#: The counter of the call path running in this context (None = not counted).
_PATH: contextvars.ContextVar[list | None] = contextvars.ContextVar("k8s_call_path", default=None)


def _count_sequential_k8s_calls(world, monkeypatch) -> list[int]:
    """The Kubernetes calls on the longest path through the call, whatever
    thread makes them (2026-10-28 review: counting only the caller's thread
    missed everything a plan does on its wave threads). Every ``_parallel``
    fork-join adds the LONGEST of its branches to the path that forked it, as
    the wall clock does; the caller calls the service directly (a TestClient
    request runs on another thread without this context)."""
    root = [0]
    _PATH.set(root)
    original_parallel = primitive_module._parallel

    def critical_path_parallel(fn, items):
        parent = _PATH.get()
        if parent is None or len(items) <= 1:
            return original_parallel(fn, items)
        branches: dict[int, list] = {}

        def branch(item):
            counter = [0]
            branches[id(item)] = counter
            _PATH.set(counter)
            return fn(item)

        context = contextvars.copy_context()
        with ThreadPoolExecutor(max_workers=len(items)) as pool:
            futures = {id(item): pool.submit(context.copy().run, branch, item) for item in items}
            results = {key: future.result() for key, future in futures.items()}
        parent[0] += max(counter[0] for counter in branches.values())
        return results

    monkeypatch.setattr(primitive_module, "_parallel", critical_path_parallel)
    monkeypatch.setattr(v2_module, "_parallel", critical_path_parallel)
    runtime = world.runtime
    for name in ("list_pod_snapshots", "write_binding_annotations", "set_pod_routable",
                 "list_live_model_pod_binding_ids", "list_ready_pod_names"):
        original = getattr(runtime, name, None)
        if not callable(original):
            continue

        def wrapped(*args, _original=original, _name=name, **kwargs):
            path = _PATH.get()
            if path is not None:
                path[0] += WEIGHT.get(_name, 1)
            return _original(*args, **kwargs)

        setattr(runtime, name, wrapped)
    return root


def test_a_model_shrink_of_two_replicas_stays_within_one_sleeps_k8s_budget(monkeypatch):
    world = World()
    counted = _count_sequential_k8s_calls(world, monkeypatch)
    world.service.put_model_target("d", wake_replicas=0)
    assert not world.awake("d-0") and not world.awake("d-1")
    assert counted[0] <= K8S_CALLS_SLEEP


def test_a_model_growth_of_two_replicas_stays_within_one_wakes_k8s_budget(monkeypatch):
    world = World(pods=(("r-2", "r", (2,), "sleeping"), ("r-3", "r", (3,), "sleeping")))
    counted = _count_sequential_k8s_calls(world, monkeypatch)
    world.service.put_model_target("r", wake_replicas=2)
    assert world.awake("r-2") and world.awake("r-3")
    assert counted[0] <= K8S_CALLS_WAKE

