"""I1 (2026-10-04): a GPU lease is released only on fresh physical evidence -
never on a transport timeout, an HTTP error, an accepted k8s delete or a
recovery that "gave up".

One regression test per bug: an unanswered /wake_up through the real VllmOps
keeps the lease until the sidecar reports no wake in flight AND the engine
reads asleep in the same recovery pass.
"""

from __future__ import annotations

from urllib.parse import urlsplit

import requests
from fastapi.testclient import TestClient

from tre_sm.api.v2 import create_app
from tre_sm.ops.vllm_ops import VllmOps

from test_wholelock_20261002 import World


class _Response:
    def __init__(self, status_code, payload=None):
        self.status_code = status_code
        self._payload = payload
        self.text = ""

    def json(self):
        return self._payload


class PodPorts:
    """HTTP of the model pods' port 8000 (reissue sidecar in front of vLLM),
    backed by the world's fake engines. /wake_up never answers (the engine is
    still waking); the sidecar counts that wake in ``waking`` until told
    otherwise (None = /tre-reissue/state unreadable)."""

    def __init__(self, engines):
        self.engines = engines
        self.waking: dict[str, int | None] = {}
        self.wake_posts: list[str] = []

    def post(self, url, *, timeout, headers=None):
        parts = urlsplit(url)
        if parts.path == "/wake_up":
            self.wake_posts.append(parts.hostname)
            self.waking[parts.hostname] = 1
            raise requests.Timeout("read timed out")
        if parts.path == "/sleep":
            self.engines.sleeping[parts.hostname] = True
            return _Response(200)
        return _Response(404)

    def get(self, url, *, timeout):
        parts = urlsplit(url)
        ip = parts.hostname
        if parts.path == "/is_sleeping":
            return _Response(200, {"is_sleeping": self.engines.sleeping.get(ip)})
        if parts.path == "/tre-reissue/state":
            if self.waking.get(ip, 0) is None:
                raise requests.ConnectionError("sidecar unreachable")
            return _Response(200, {"waking": self.waking.get(ip, 0), "sleeping": True})
        if parts.path == "/version":
            return _Response(200, {"version": "0.30.0"})
        return _Response(404)


def test_an_unanswered_wake_keeps_its_lease_until_the_sidecar_and_the_engine_agree():
    world = World()
    ports = PodPorts(world.vllm)
    fakes, world.vllm = world.vllm, VllmOps(http=ports, timeout_s=1.0, max_attempts=3)
    world.service = world.make_service()  # the real VllmOps over HTTP
    world.client = TestClient(create_app(world.service))
    world.vllm = fakes
    ip, binding_id = world.ip_of["r-2"], "r/node-a/2"

    response = world.client.put("/v2/bindings/r-2/power", json={"awake": True})

    assert response.status_code == 409 and response.json()["error"] == "wake_failed"
    assert ports.wake_posts == [ip]  # never retried after no answer
    assert world.lease("r-2") == "awake"  # an unanswered wake is uncertain, not failed
    _, wake_journal = world.journals()
    assert set(wake_journal) == {binding_id}

    world.service._wake_journal.update(binding_id, recover_after_ms=0)  # past the recheck floor
    # Asleep, but the sidecar still forwards the wake: kept.
    assert world.service.recover_wake_journal()["kept"] == [{"binding_id": binding_id, "result": "wake_unsettled"}]
    assert world.lease("r-2") == "awake"
    # Asleep, the sidecar cannot be read: kept.
    ports.waking[ip] = None
    assert world.service.recover_wake_journal()["kept"] == [{"binding_id": binding_id, "result": "wake_unsettled"}]
    assert world.lease("r-2") == "awake" and set(world.journals()[1]) == {binding_id}

    # The wake ended (engine still asleep): released in the same pass.
    ports.waking[ip] = 0
    result = world.service.recover_wake_journal()

    assert result["resolved"] == [{"binding_id": binding_id, "result": "rolled_back"}]
    assert world.lease("r-2") is None
    assert world.journals()[1] == {}
