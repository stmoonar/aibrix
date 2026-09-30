"""Controller side of the SM replica floor (2026-09-29).

The SM answers 409 ``error: floor_violation`` when a hide / sleep would take a
model below its min_replicas. The controller does not retry it (a plain 409 -
writer lock busy, sleep reservation - stays retriable): the action fails for this
tick and the planner re-plans from a fresh view on the next one. A refused
SafeScale hide rolls its probe back; a refused one-shot commit gives the donor
pods their routing back instead of backing off and retrying.
"""

from __future__ import annotations

import asyncio

from tre_controller.loops.action_queue import ActionQueue
from tre_controller.planning.planner import HideAction
from tre_controller.sm_client import ServiceManagerClient, ServiceManagerError

from test_action_queue_review3 import ScriptedSM, _calls, _commit

FLOOR_BODY = {
    "detail": "replica floor: safescale_hide would leave m with 0 routable replica(s) < min_replicas 1",
    "error": "floor_violation",
    "path": "safescale_hide",
    "floor": {"model": "m", "floor": 1, "routable": ["m-0"], "removing": ["m-0"], "routable_after": 0},
}


def test_floor_violation_is_not_retriable_but_other_409s_are():
    floor = ServiceManagerError("HTTP 409", status=409, body=FLOOR_BODY)
    assert floor.floor_violation and not floor.retriable
    result = floor.result()
    assert result["retriable"] is False and result["floor_violation"]["routable_after"] == 0
    busy = ServiceManagerError("HTTP 409", status=409, body={"detail": "writer busy"})
    assert not busy.floor_violation and busy.retriable
    assert ServiceManagerError("HTTP 409", status=409).retriable


class RaisingTransport:
    def __init__(self, error):
        self.error = error
        self.calls = []

    async def request(self, method, url, *, json=None, timeout_s):
        self.calls.append((method, url, json))
        raise self.error


def test_client_calls_report_the_floor_violation_as_permanent():
    transport = RaisingTransport(ServiceManagerError("HTTP 409", status=409, body=FLOOR_BODY))
    client = ServiceManagerClient("http://sm", transport=transport)

    routable = asyncio.run(client.set_routable("m", ("m-0",)))
    power = asyncio.run(client.set_binding_power("m-0", awake=False, sleep_path="urgent"))

    for result in (routable, power):
        assert result["ok"] is False and result["retriable"] is False
        assert result["floor_violation"]["model"] == "m"


def test_a_refused_hide_marks_its_probe_failed_once_without_retry():
    marks = []

    async def scenario():
        sm = ScriptedSM(
            results={"routable:m": [{"ok": False, "error": "HTTP 409: floor", "retriable": False}]}
        )
        queue = ActionQueue(sm, on_hide_failed=lambda *args: marks.append(args))
        queue.submit((HideAction("m", ("m-0",), "probe_started", "rescue"),))
        [result] = await queue.drain_once()
        assert (result.ok, result.retriable) == (False, False)
        assert _calls(sm) == [("m", "routable", ("m-0",))]

    asyncio.run(scenario())
    assert marks == [("m", ("m-0",), "hide_failed: HTTP 409: floor")]


def test_a_refused_one_shot_commit_is_not_retried_and_unhides_the_donor():
    slept = []

    async def no_backoff(seconds):
        slept.append(seconds)

    async def scenario():
        sm = ScriptedSM(
            results={"7b-1": [{"ok": False, "error": "HTTP 409: floor", "retriable": False}]}
        )
        queue = ActionQueue(sm, sleep=no_backoff)
        queue.submit((_commit(),))
        await queue.drain_once()
        # one sleep attempt, no backoff / retry, then the donor pod gets its routing back
        assert _calls(sm) == [("7b-1", "sleep"), ("7b", "routable", ())]
        assert queue.stats()["oneshot_retries_total"] == 0

    asyncio.run(scenario())
    assert slept == []
