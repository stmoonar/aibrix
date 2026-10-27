from __future__ import annotations

import json
import socket
import threading
import time

import pytest

from bl_fakes import FakeCluster, FakeRedis, FakeSource, ScriptedPolicy, StubSM, limits, make_config, wait_until
from tre_baselines.loop import BaselineShell
from tre_baselines.sm_client import Dispatcher, SMClient, SMResult, parse_sm_error, plan_item, plan_item_results


def _json(status, doc):
    return lambda body: (status, json.dumps(doc).encode(), "application/json")


PLAN = {"plan_id": "bl-x", "sleeps": [], "wakes": [{"model": "m", "at_least": 3}]}


def test_plan_is_posted_with_the_actor_header() -> None:
    with StubSM() as sm:
        result = SMClient(sm.url, actor="tre-baseline-scaler/test").post_plan(PLAN)
    assert result.ok and result.code == 200 and result.raw["wakes"][0]["status"] == "done"
    method, path, sent, headers = sm.requests[0]
    assert (method, path, sent) == ("POST", "/v2/plans", PLAN)
    assert headers.get("x-tre-actor") == "tre-baseline-scaler/test"


def test_items_grow_only_up_absolute_down_on_the_abort_path_never_a_drain() -> None:
    assert plan_item("m", "up", 3) == {"model": "m", "at_least": 3}
    assert plan_item("m", "down", 1) == {"model": "m", "at_most": 1, "sleep_path": "urgent"}
    assert plan_item("m", "down", 2, abort_sleep_path="apa") == {"model": "m", "at_most": 2, "sleep_path": "apa"}
    with pytest.raises(ValueError):
        plan_item("m", "sideways", 1)


def test_structured_409_is_parsed() -> None:
    doc = {"detail": "no free GPU pair", "error": "placement", "reason": "no_pair", "node": "node-a",
           "gpu_ids": [2, 3], "scope": "node", "binding_id": "m/node-a/2,3",
           "blocking_binding_id": "x/node-a/2", "retry_after_s": 4.5, "future_key": {"ignored": True}}
    with StubSM(_json(409, doc)) as sm:
        result = SMClient(sm.url).post_plan(PLAN)
    assert not result.ok and result.code == 409
    assert (result.error, result.reason, result.node, result.gpu_ids, result.scope, result.retry_after_s) == (
        "placement", "no_pair", "node-a", (2, 3), "node", 4.5)
    assert result.detail == "no free GPU pair"
    d = result.as_dict()
    assert d["binding_id"] == "m/node-a/2,3" and d["blocking_binding_id"] == "x/node-a/2"


def test_structured_409_nested_under_detail_is_parsed() -> None:
    # FastAPI HTTPException(detail={...}) nests the object under "detail".
    result = parse_sm_error(409, json.dumps({"detail": {"reason": "cap", "message": "above cap"}}))
    assert result.reason == "cap" and result.detail == "above cap" and result.error == "http_error"


def test_plain_detail_409_current_main() -> None:
    with StubSM(_json(409, {"detail": "model m has a binding reserved for sleep"})) as sm:
        result = SMClient(sm.url).post_plan(PLAN)
    assert not result.ok and result.code == 409
    assert result.error == "http_error" and result.reason is None and result.gpu_ids is None
    assert result.detail == "model m has a binding reserved for sleep"


def test_plain_text_409_falls_back_to_text() -> None:
    with StubSM(lambda b: (409, b"writer lock busy", "text/plain")) as sm:
        result = SMClient(sm.url).post_plan(PLAN)
    assert not result.ok and result.code == 409 and result.detail == "writer lock busy"


def test_floor_violation_body() -> None:
    result = parse_sm_error(409, json.dumps({"detail": "below floor", "error": "floor_violation",
                                             "path": "scale_down", "floor": {"floor": 1}}))
    assert result.error == "floor_violation" and result.detail == "below floor"


def test_timeout() -> None:
    with StubSM(delay_s=1.0) as sm:
        started = time.monotonic()
        result = SMClient(sm.url, timeout_s=0.2).post_plan(PLAN)
        assert time.monotonic() - started < 0.9
    assert not result.ok and result.error == "timeout" and result.code is None


def test_transport_error() -> None:
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    result = SMClient(f"http://127.0.0.1:{port}", timeout_s=1.0).post_plan(PLAN)
    assert not result.ok and result.error in {"transport", "timeout"}


def test_get_state_tolerates_extra_fields() -> None:
    state = {"version": 7, "models": {"m": {"awake": 2, "bound": 4, "new": 1}}, "bindings": [], "fleet": {},
             "placement": {"free_gpus": 3}}
    with StubSM(state=state) as sm:
        assert SMClient(sm.url).get_state()["models"]["m"]["awake"] == 2


def test_a_whole_plan_answer_without_items_is_every_models_answer() -> None:
    busy = parse_sm_error(409, json.dumps({"detail": "writer busy", "error": "writer_busy", "retry_after_s": 1.0}))
    answers = plan_item_results(busy, ["a", "b"])
    assert answers["a"] is busy and answers["b"] is busy
    transport = SMResult(ok=False, error="timeout")
    assert plan_item_results(transport, ["a"])["a"] is transport


def test_one_item_plan_reached_is_the_items_awake_after() -> None:
    """Design s8 item 13: a scale-up is ONE wake item ``at_least = target`` (no steps); the
    SM fills it best effort. Woke 2 of 3 (``partial``): reached = ``awake_after``, located
    by the item's refusal; a scale-down is one sleep item ``at_most``."""
    cluster = FakeCluster(awake={"a": 3, "b": 1}, place_limit={"b": 3})
    d = Dispatcher(cluster.post_plan)
    assert d.dispatch({"a": ("down", 1, None), "b": ("up", 4, 1)}) == {"a": "sent", "b": "sent"}
    assert wait_until(lambda: d.inflight_count() == 0)
    done = {c.model: c for c in d.drain_results()}
    (plan,) = cluster.plans
    assert plan["sleeps"] == [{"model": "a", "at_most": 1, "sleep_path": "urgent"}]
    assert plan["wakes"] == [{"model": "b", "at_least": 4}]
    b = done["b"]
    assert (b.start, b.reached, b.partial_fill, b.result.ok, b.result.code) == (1, 3, True, False, 200)
    assert (b.result.reason, b.result.node, b.result.gpu_ids, b.result.blocking_binding_id) == (
        "slot_occupied", "n", (0,), "x/n/0")
    assert b.as_dict()["unfilled"] == 1 and b.as_dict()["plan_id"] == plan["plan_id"]
    a = done["a"]
    assert a.result.ok and a.as_dict()["taken"] == 2 and cluster.awake == {"a": 1, "b": 3}
    assert a.plan_id == b.plan_id and a.plan_size == b.plan_size == 2
    d.close(join_s=1.0)


def test_coalescing_one_plan_in_flight_latest_decision_per_model_sent_next() -> None:
    """A plan in flight + new ticks: no second concurrent POST; a model in the plan in
    flight is not taken; the other models' latest decisions go in the next plan, which
    is sent as soon as the one in flight returns."""
    gate = threading.Event()
    cluster = FakeCluster(awake={"a": 1, "b": 1, "c": 2}, gate=gate)
    d = Dispatcher(cluster.post_plan)
    assert d.dispatch({"a": ("up", 2, 1)}) == {"a": "sent"}
    assert wait_until(lambda: len(cluster.plans) == 1)
    # two more ticks while the first plan is held
    assert d.dispatch({"a": ("up", 3, 1), "b": ("up", 2, 1)}) == {"a": "inflight", "b": "queued"}
    assert d.dispatch({"b": ("up", 3, 1), "c": ("down", 1, None)}) == {"b": "queued", "c": "queued"}
    time.sleep(0.05)
    assert len(cluster.plans) == 1 and d.inflight_count() == 3
    gate.set()
    assert wait_until(lambda: d.inflight_count() == 0)
    assert len(cluster.plans) == 2 and cluster.max_in_flight == 1
    assert cluster.plans[1]["wakes"] == [{"model": "b", "at_least": 3}]          # latest decision of b
    assert cluster.plans[1]["sleeps"] == [{"model": "c", "at_most": 1, "sleep_path": "urgent"}]
    assert sorted((c.model, c.target) for c in d.drain_results()) == [("a", 2), ("b", 3), ("c", 1)]
    # a tick whose decisions no longer include a waiting model drops it (latest wins)
    gate.clear()
    d.dispatch({"a": ("up", 3, 2)})
    assert wait_until(lambda: len(cluster.plans) == 3)
    d.dispatch({"b": ("up", 4, 3)})
    d.dispatch({})
    gate.set()
    assert wait_until(lambda: d.inflight_count() == 0)
    assert len(cluster.plans) == 3
    d.close(join_s=1.0)


def test_the_guard_drops_the_whole_plan() -> None:
    cluster = FakeCluster(awake={"a": 1, "b": 2})
    d = Dispatcher(cluster.post_plan, guard=lambda: "owner_lost")
    d.dispatch({"a": ("up", 2, 1), "b": ("down", 1, None)})
    assert wait_until(lambda: d.inflight_count() == 0)
    done = d.drain_results()
    assert cluster.plans == [] and {(c.model, c.result.error, c.result.reason) for c in done} == {
        ("a", "dropped", "owner_lost"), ("b", "dropped", "owner_lost")}
    assert not any(c.progressed for c in done)
    d.close(join_s=1.0)


def test_slow_sm_call_no_duplicate_and_tick_not_blocked(tmp_path) -> None:
    """An SM call slower than two ticks: later ticks log inflight_skip, send nothing, and
    return immediately; the result shows up on the first tick after it completed."""
    with StubSM(delay_s=0.6) as sm:
        config = make_config(tmp_path, {"a": limits("a")}, dry_run=False, sm_url=sm.url, tick_s=0.2)
        cluster = FakeCluster(awake={"a": 1})
        redis = FakeRedis()
        client = SMClient(sm.url, timeout_s=5.0)
        dispatcher = Dispatcher(client.post_plan)
        policy = ScriptedPolicy({t: {"a": 2} for t in range(10)})
        shell = BaselineShell(config, FakeSource(config, cluster, redis), policy, dispatcher, redis)
        actions = []
        for _ in range(3):
            started = time.monotonic()
            actions.append(shell.tick_once()[0]["action"])
            assert time.monotonic() - started < 0.15
            time.sleep(0.2)
        assert actions == ["up", "inflight_skip", "inflight_skip"]
        assert len(sm.requests) == 1
        assert wait_until(lambda: dispatcher.inflight_count() == 0)
        line = shell.tick_once()[0]
        assert line["sm_result"]["ok"] is True and line["sm_result"]["target"] == 2
        # FakeCluster was not told about the stub's success, so the policy asks again.
        assert line["action"] == "up"
        assert wait_until(lambda: dispatcher.inflight_count() == 0)
        assert len(sm.requests) == 2
        dispatcher.close(join_s=1.0)


def test_plan_outcomes_that_are_not_full_successes_are_logged() -> None:
    """A clamp at the floor (200, item ``clamped``) and a partial wake must show in the log."""
    plan = SMResult(ok=True, code=200, raw={"plan_id": "p", "status": "partial", "sleeps": [
        {"model": "a", "status": "clamped", "taken": 0, "clamped_by_floor": True}], "wakes": [
        {"model": "b", "status": "partial", "awake_before": 1, "awake_after": 2, "woke": [{"serve_id": "b-1"}],
         "unfilled": 1, "refusals": [{"reason": "gpu_busy"}]}]})
    answers = plan_item_results(plan, ["a", "b"])
    a, b = answers["a"].as_dict(), answers["b"].as_dict()
    assert answers["a"].ok and a["clamped_by_floor"] is True and a["taken"] == 0 and a["status"] == "clamped"
    assert not answers["b"].ok and b["unfilled"] == 1 and b["refusals"] == [{"reason": "gpu_busy"}]
    assert (b["awake_after"], b["woke"], b["plan_id"], b["plan_status"]) == (2, ["b-1"], "p", "partial")
    refused = plan_item_results(SMResult(ok=False, code=409, error="partial", raw={"status": "failed", "wakes": [
        {"model": "b", "status": "refused", "error": {"error": "gpu_busy", "reason": "slot_occupied",
                                                      "binding_id": "b/n/0"}}]}), ["b"])["b"]
    assert (refused.ok, refused.code, refused.error, refused.reason) == (False, 409, "gpu_busy", "slot_occupied")
    assert refused.as_dict()["binding_id"] == "b/n/0"
