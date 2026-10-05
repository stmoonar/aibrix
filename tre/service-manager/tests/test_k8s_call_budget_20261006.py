"""The writer-lock hold must not grow with the number of targets (review
2026-10-06 P2-1): the Kubernetes calls a sleep, a wake or a transfer makes one
after the other stay within the K8S_CALLS_* the worst-case formulas count,
whatever the number of bindings (per-target calls run in parallel)."""

import threading

from tre_common.registry import K8S_CALLS_SLEEP, K8S_CALLS_TRANSFER_SELECT, K8S_CALLS_WAKE

from test_wholelock_20261002 import World

#: Kubernetes calls per runtime call (an annotation write is a read + a patch).
WEIGHT = {"write_binding_annotations": 2, "set_pod_routable": 2}


def _count_sequential_k8s_calls(world) -> list[int]:
    counted = [0]
    main = threading.current_thread()
    runtime = world.runtime
    for name in ("list_pod_snapshots", "write_binding_annotations", "set_pod_routable",
                 "list_live_model_pod_binding_ids", "list_ready_pod_names"):
        original = getattr(runtime, name, None)
        if not callable(original):
            continue

        def wrapped(*args, _original=original, _name=name, **kwargs):
            if threading.current_thread() is main:
                counted[0] += WEIGHT.get(_name, 1)
            return _original(*args, **kwargs)

        setattr(runtime, name, wrapped)
    return counted


def test_a_model_shrink_of_two_replicas_stays_within_one_sleeps_k8s_budget():
    world = World()
    counted = _count_sequential_k8s_calls(world)
    world.service.put_model_target("d", wake_replicas=0)
    assert not world.awake("d-0") and not world.awake("d-1")
    assert counted[0] <= K8S_CALLS_SLEEP


def test_a_model_growth_of_two_replicas_stays_within_one_wakes_k8s_budget():
    world = World(pods=(("r-2", "r", (2,), "sleeping"), ("r-3", "r", (3,), "sleeping")))
    counted = _count_sequential_k8s_calls(world)
    world.service.put_model_target("r", wake_replicas=2)
    assert world.awake("r-2") and world.awake("r-3")
    assert counted[0] <= K8S_CALLS_WAKE


def test_a_two_pair_transfer_stays_within_the_transfer_k8s_budget():
    world = World()
    counted = _count_sequential_k8s_calls(world)
    body = world.service.transfer(donor_model="d", receiver_model="r", count=2)
    assert body["done"] == 2
    assert counted[0] <= K8S_CALLS_TRANSFER_SELECT + K8S_CALLS_SLEEP + K8S_CALLS_WAKE
