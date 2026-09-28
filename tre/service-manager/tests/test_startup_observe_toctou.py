"""Startup admission vs SM actuation observe: the observe check is repeated
under the prepare writer lock of the resident sleep (2026-09-28 TOCTOU fix).

The unlocked pre-check in admit_startup can pass (actuation active) and the
actuation be switched to observe before the residents are hidden; the check
right before the hide, under the writer lock, must still refuse (retriable
409, recorded as suppressed) with nothing hidden or slept.
"""
from __future__ import annotations

import pytest

from tre_sm.api.v2 import RetryLater

from test_review2_sleep import _startup_world
from test_review4_sm import _cold_start_world


def _flip_to_observe_after_the_precheck(world):
    service = world.service
    original = service._assert_unrequested_startup_allowed

    def precheck_then_flip(pod):
        original(pod)  # actuation active: passes
        service._safety_gate.actuation = "observe"  # the operator flips it now

    service._assert_unrequested_startup_allowed = precheck_then_flip


def test_observe_switched_after_the_precheck_is_refused_under_the_writer_lock():
    world = _startup_world()  # tp2 awake on GPUs 0,1; the m1 Pod starts on GPU 0
    _flip_to_observe_after_the_precheck(world)
    under_lock = []
    safety = world.service._safety_gate
    record = safety.record_suppressed

    def spy(action, detail):
        under_lock.append(world.coordinator.active and world.coordinator.active["kind"])
        return record(action, detail)

    safety.record_suppressed = spy

    with pytest.raises(RetryLater, match="SM actuation is observe"):
        world.service.admit_startup(pod_name="m1-new", pod_uid="new-uid")

    assert under_lock == ["startup_admit_sleep"]  # refused while holding the lock
    assert world.vllm.sleeping["10.0.0.5"] is False  # the resident keeps serving
    assert world.state("pod-tp2") != "hidden"  # and was never hidden
    assert world.coordinator.kinds == ["startup_admit_sleep"]  # no commit, no admission
    assert world.desired()["tp2/node-a/0,1"][0] == "awake"
    assert safety.suppressed == [
        ("startup_admission_sleep",
         {"pod": "m1-new", "binding_id": "m1/node-a/0", "would_sleep": ["tp2/node-a/0,1"]})
    ]


def test_active_actuation_still_sleeps_the_resident_through_the_locked_check():
    world = _startup_world()
    result = world.service.admit_startup(pod_name="m1-new", pod_uid="new-uid")
    assert result["suspended_binding_ids"] == ["tp2/node-a/0,1"]
    assert world.service._safety_gate.suppressed == []


def test_an_owned_admission_is_unchanged_in_observe():
    # A cold start owns its Pod (phase starting_binding): pre-authorized, it
    # never reaches the unrequested-sleep checks.
    world = _cold_start_world()
    world.service._safety_gate.actuation = "observe"
    world.service.put_model_target("m1", wake_replicas=2)
    assert len(world.runtime.created) == 1
    assert world.service._safety_gate.suppressed == []
