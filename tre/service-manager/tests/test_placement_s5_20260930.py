"""S5 (2026-09-30), service-manager side: a pure-capacity wake is placed by the SM
(registry placement policy) with the caller's hints first and a substitute for a
hint it cannot wake; ``picked`` reports where the replicas went. ``/v2/state``
reports per GPU whether a wake could go there (``gpus[].wakeable``) and per node
the gpu-truth health."""

import json

from fastapi.testclient import TestClient

from tre_sm.api.v2 import create_app
from tre_sm.allocator.slots import Binding, Slot
from tre_sm.gpu_truth import RedisGpuTruth
from tre_sm.state.gpu_leases import GpuLeaseStore
from tre_sm.state.wake_journal import WakeJournal

from sm_test_fakes import fence, pod
from test_review2_sleep import World, _desired

TRUTH_KEY = "tre:gpu_truth:node-a"


def _world(*, truth=None):
    """node-a: m1 bindings asleep on GPUs 0, 1, 2 (m1 awake nowhere)."""
    snapshots = [pod(f"pod-{g}", "m1", (g,), ip=f"10.0.0.{g + 1}", state="sleeping") for g in (0, 1, 2)]
    desired = [_desired(f"m1/node-a/{g}", "m1", (g,), "sleeping") for g in (0, 1, 2)]
    world = World(snapshots, desired)
    world.leases = GpuLeaseStore(world.redis)
    world.service._gpu_leases = world.leases
    world.service._wake_journal = WakeJournal(world.redis)
    if truth is not None:
        world.redis.values[TRUTH_KEY] = json.dumps(
            {"node": "node-a", "timestamp": 1.0, "seq": 1, "refresh_seq": 0, "gpus": [
                {"uuid": f"GPU-{g}", "used_mib": truth.get(g, 500), "total_mib": 40960} for g in range(4)
            ]}
        )
        world.service._gpu_truth = RedisGpuTruth(world.redis)
        world.redis.incr = lambda key: 0  # the refresh counter (never served here)
    return world


def test_hints_are_woken_first_when_feasible():
    world = _world()

    result = world.service.put_model_target("m1", wake_replicas=1, hints=["pod-2"])

    assert result["actions"] == [{"action": "wake", "serve_id": "pod-2"}]
    (picked,) = result["picked"]
    assert (picked["serve_id"], picked["node"], picked["gpu_ids"], picked["hinted"]) == ("pod-2", "node-a", [2], True)


def test_an_unwakeable_hint_is_substituted_by_the_sm_choice():
    # GPU 2 shows 30 GiB used that no binding explains (a leak): the gate refuses it.
    world = _world(truth={2: 30_000})

    result = world.service.put_model_target("m1", wake_replicas=1, hints=["pod-2"])

    (picked,) = result["picked"]
    assert picked["hinted"] is False and picked["serve_id"] != "pod-2"
    assert world.vllm.sleeping["10.0.0.3"] is True  # the hint stayed asleep
    assert world.desired()["m1/node-a/2"][0] == "sleeping"
    assert world.desired()[picked["binding_id"]][0] == "awake"


def test_hints_over_http_and_an_unknown_hint_is_ignored():
    world = _world()
    client = TestClient(create_app(world.service))

    answer = client.put("/v2/models/m1/target", json={"wake_replicas": 2, "at_least": True, "hints": ["nope", "pod-1"]})

    assert answer.status_code == 200
    picked = answer.json()["picked"]
    assert len(picked) == 2 and picked[0]["serve_id"] == "pod-1" and picked[0]["hinted"] is True


def test_state_reports_wakeable_gpus_and_why_not():
    world = _world(truth={3: 30_000})
    with fence(world.redis):
        world.leases.acquire(Binding("new", "tp2", Slot("node-a", (0, 1)), awake=False), phase="starting")
    world.service.put_binding_power("pod-2", awake=True)

    state = world.service.get_state()
    gpus = {entry["gpu"]: entry for entry in state["gpus"]}

    assert (gpus[0]["wakeable"], gpus[0]["reason"], gpus[0]["loading_binding_id"]) == (False, "loading", "tp2/node-a/0,1")
    assert gpus[1]["reason"] == "loading"
    assert (gpus[2]["wakeable"], gpus[2]["reason"], gpus[2]["awake_binding_id"]) == (False, "awake", "m1/node-a/2")
    assert (gpus[3]["wakeable"], gpus[3]["reason"], gpus[3]["used_mib"]) == (False, "gpu_truth_used", 30_000)
    node = state["nodes"]["node-a"]
    assert node["truth_available"] is True and node["truth_seq"] == 1 and node["truth_age_s"] is not None
    # the waking lease of an in-flight wake would read "waking"; the wake is done here
    assert all(entry["waking_binding_id"] is None for entry in state["gpus"])


def test_state_without_gpu_truth_marks_free_gpus_wakeable_by_probe():
    world = _world()
    state = world.service.get_state()
    assert [entry["wakeable"] for entry in state["gpus"]] == [True, True, True, True]
    assert {entry["truth_source"] for entry in state["gpus"]} == {"none"}
    assert state["nodes"]["node-a"]["truth_configured"] is False
