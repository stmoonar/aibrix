"""I1 (2026-10-04): a GPU lease is released only on fresh physical evidence -
never on a transport timeout, an HTTP error, an accepted k8s delete, a
recovery that "gave up" or a service-manager restart.

Every test drives the real VllmOps over a fake pod port (reissue sidecar in
front of vLLM). Evidence for "no wake in flight": the sidecar's ``waking`` is
0, read before /is_sleeping says asleep - or, with no sidecar (404), the
weaker /is_sleeping alone (logged).
"""

from __future__ import annotations

import dataclasses
import json
import logging
from urllib.parse import urlsplit

import pytest
import requests
from fastapi.testclient import TestClient

from tre_sm.api.v2 import create_app
from tre_sm.ops.vllm_ops import VllmOps
from tre_sm.server import rebuild_gpu_leases

from sm_test_fakes import binding_of, fence
from test_wholelock_20261002 import World

ABSENT = "absent"  # no sidecar in front of the engine: /tre-reissue/state is a 404


class _Response:
    def __init__(self, status_code, payload=None):
        self.status_code = status_code
        self._payload = payload
        self.text = ""

    def json(self):
        return self._payload


class PodPorts:
    """HTTP of the model pods' port 8000, backed by the world's fake engines.
    /wake_up never answers and the engine stays asleep (the wake is still
    queued). ``sidecar[ip]``: the sidecar's ``waking`` (default 0), None = no
    answer, ABSENT = no sidecar. ``engine_down``: /is_sleeping answers 502."""

    def __init__(self, engines):
        self.engines = engines
        self.sidecar: dict[str, int | str | None] = {}
        self.engine_down: set[str] = set()
        self.calls: list[tuple[str, str]] = []

    def post(self, url, *, timeout, headers=None):
        parts = urlsplit(url)
        self.calls.append((parts.hostname, parts.path))
        if parts.path == "/wake_up":
            raise requests.Timeout("read timed out")
        if parts.path == "/sleep":
            self.engines.sleeping[parts.hostname] = True
            return _Response(200)
        return _Response(404)

    def get(self, url, *, timeout):
        parts = urlsplit(url)
        ip = parts.hostname
        self.calls.append((ip, parts.path))
        sidecar = self.sidecar.get(ip, 0)
        if parts.path == "/tre-reissue/state":
            if sidecar is None:
                raise requests.ConnectionError("sidecar unreachable")
            if sidecar == ABSENT:
                return _Response(404)
            return _Response(200, {"waking": sidecar})
        if parts.path == "/is_sleeping":
            if ip in self.engine_down:
                return _Response(502)
            return _Response(200, {"is_sleeping": self.engines.sleeping.get(ip)})
        if parts.path == "/version":
            return _Response(200, {"version": "0.30.0"})
        return _Response(404)


def _service(world, ports, *, wake_timeout_s=10.0, **extra):
    """A service-manager process on the world's state with the real VllmOps
    (production: one /wake_up attempt with service_manager.wake.call_timeout_s)."""
    fakes = world.vllm
    world.vllm = VllmOps(http=ports, timeout_s=1.0, max_attempts=3, wake_timeout_s=wake_timeout_s)
    try:
        world.service = world.make_service(**extra)
    finally:
        world.vllm = fakes
    world.client = TestClient(create_app(world.service))
    return world.service


def _world(*, wake_timeout_s=10.0, **sm_config):
    world = World()
    # No recheck floor: the recovery may look at once (the floor is not the proof).
    config = dataclasses.replace(world.registry.service_manager(), wake_transport_recheck_s=0.0, **sm_config)
    world.registry = type(world.registry)(world.registry.topology(), world.registry.models(), service_manager=config)
    ports = PodPorts(world.vllm)
    _service(world, ports, wake_timeout_s=wake_timeout_s)
    return world, ports


def _unanswered_wake(world, ports):
    response = world.client.put("/v2/bindings/r-2/power", json={"awake": True})
    assert response.status_code == 409 and response.json()["error"] == "wake_failed"
    assert [call for call in ports.calls if call[1] == "/wake_up"] == [(world.ip_of["r-2"], "/wake_up")]
    assert world.lease("r-2") == "awake"  # uncertain, not failed: the GPU stays fenced


def _journaled(world):
    return set(world.client.get("/v2/wake").json()["in_progress"])


def _recover(world):
    result = world.service.recover_wake_journal()
    return [item["result"] for item in result["resolved"] + result["kept"]]


@pytest.mark.parametrize("wake_timeout_s", [10.0, None], ids=["single_shot", "retrying_ops"])
def test_an_unanswered_wake_keeps_its_lease_until_the_sidecar_and_the_engine_agree(wake_timeout_s):
    world, ports = _world(wake_timeout_s=wake_timeout_s)
    ip = world.ip_of["r-2"]
    _unanswered_wake(world, ports)  # never retried, also by the retrying construction

    ports.sidecar[ip] = 1  # asleep, but the sidecar still forwards the wake
    assert _recover(world) == ["wake_unsettled"]
    assert world.lease("r-2") == "awake" and _journaled(world) == {"r/node-a/2"}

    ports.sidecar[ip] = None  # no answer: unknown, and the engine is not probed
    ports.calls.clear()
    assert _recover(world) == ["physical_state_unknown"]
    assert (ip, "/is_sleeping") not in ports.calls
    assert world.lease("r-2") == "awake"

    ports.sidecar[ip] = 0  # the wake ended, the engine is asleep: released
    assert _recover(world) == ["rolled_back"]
    assert world.lease("r-2") is None and _journaled(world) == set()


def test_without_a_sidecar_is_sleeping_alone_settles_the_wake_with_a_warning(caplog):
    world, ports = _world()
    ports.sidecar[world.ip_of["r-2"]] = ABSENT  # reissue disabled: the path reaches vLLM (404)
    _unanswered_wake(world, ports)

    with caplog.at_level(logging.WARNING, logger="tre_sm.api.v2"):
        assert _recover(world) == ["rolled_back"]

    assert world.lease("r-2") is None
    events = [json.loads(r.getMessage()).get("event") for r in caplog.records if r.getMessage().startswith("{")]
    assert "lease_release_weak_evidence" in events


def test_a_terminating_pod_keeps_the_lease_of_its_journaled_wake_until_it_is_gone():
    world, ports = _world()
    _unanswered_wake(world, ports)
    # Out of the Running snapshots (deletionTimestamp set), still a Pod object.
    gone = world.runtime.snapshots.pop("r-2")
    world.runtime.terminating_uids.add(gone.pod_uid)

    assert _recover(world) == ["pod_still_present"]
    assert world.lease("r-2") == "awake"

    world.runtime.terminating_uids.clear()
    assert _recover(world) == ["pod_gone"]
    assert world.lease("r-2") is None


def test_a_restart_keeps_the_lease_of_a_suspect_until_it_is_settled():
    world, ports = _world(wake_recovery_unknown_attempts=1)
    ip = world.ip_of["r-2"]
    _unanswered_wake(world, ports)
    ports.engine_down.add(ip)
    assert _recover(world) == ["gave_up"]
    assert world.lease("r-2") == "awake"  # giving up is no evidence: a suspect now

    # The service-manager restarts (restart-to-apply): the bootstrap rebuild.
    world.runtime.list_live_model_pod_binding_ids = lambda: {
        binding_of(s).binding_id for s in world.runtime.snapshots.values()
    }
    with fence(world.redis):
        suspects = rebuild_gpu_leases(
            world.leases, world.runtime, world.store.load().bindings, starting_bindings=[], waking_bindings=[]
        )
    service = _service(world, ports, restored_suspects=suspects)
    assert world.lease("r-2") == "awake"  # survived the restart

    ports.engine_down.clear()
    ports.sidecar[ip] = 1  # asleep, a wake in flight: kept
    service.guard_container_restarts()
    assert world.lease("r-2") == "awake"

    ports.sidecar[ip] = 0
    service.guard_container_restarts()
    assert world.lease("r-2") is None
