from __future__ import annotations

import json
import socket
import time

import pytest

from bl_fakes import FakeCluster, FakeRedis, FakeSource, ScriptedPolicy, StubSM, limits, make_config, wait_until
from tre_baselines.loop import BaselineShell
from tre_baselines.sm_client import Dispatcher, SMClient, parse_sm_error, target_body


def _json(status, doc):
    return lambda model, body: (status, json.dumps(doc).encode(), "application/json")


def test_200_up_body_is_grow_only_and_actor_header_sent() -> None:
    with StubSM(_json(200, {"model": "m", "actions": [{"action": "wake"}]})) as sm:
        client = SMClient(sm.url, actor="tre-baseline-scaler/test")
        body = target_body("up", 3)
        result = client.put_target("m", body)
    assert result.ok and result.code == 200
    assert result.raw["actions"] == [{"action": "wake"}]
    method, path, sent, headers = sm.requests[0]
    assert (method, path) == ("PUT", "/v2/models/m/target")
    assert sent == {"wake_replicas": 3, "at_least": True}
    assert headers.get("x-tre-actor") == "tre-baseline-scaler/test"
    assert result.as_dict()["actions"] == [{"action": "wake"}]


def test_down_body_is_absolute_on_the_abort_path_and_never_asks_for_a_drain() -> None:
    assert target_body("down", 1) == {"wake_replicas": 1, "sleep_path": "urgent"}
    assert target_body("down", 2, abort_sleep_path="apa") == {"wake_replicas": 2, "sleep_path": "apa"}
    with pytest.raises(ValueError):
        target_body("sideways", 1)


def test_structured_409_is_parsed() -> None:
    doc = {"detail": "no free GPU pair", "error": "placement", "reason": "no_pair", "node": "node-a",
           "gpu_ids": [2, 3], "scope": "node", "binding_id": "m/node-a/2,3",
           "blocking_binding_id": "x/node-a/2", "retry_after_s": 4.5, "future_key": {"ignored": True}}
    with StubSM(_json(409, doc)) as sm:
        result = SMClient(sm.url).put_target("m", {"wake_replicas": 2, "at_least": True})
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
        result = SMClient(sm.url).put_target("m", {"wake_replicas": 1, "sleep_path": "urgent"})
    assert not result.ok and result.code == 409
    assert result.error == "http_error" and result.reason is None and result.gpu_ids is None
    assert result.detail == "model m has a binding reserved for sleep"


def test_plain_text_409_falls_back_to_text() -> None:
    with StubSM(lambda m, b: (409, b"writer lock busy", "text/plain")) as sm:
        result = SMClient(sm.url).put_target("m", {"wake_replicas": 1})
    assert not result.ok and result.code == 409 and result.detail == "writer lock busy"


def test_floor_violation_body() -> None:
    result = parse_sm_error(409, json.dumps({"detail": "below floor", "error": "floor_violation",
                                             "path": "scale_down", "floor": {"floor": 1}}))
    assert result.error == "floor_violation" and result.detail == "below floor"


def test_timeout() -> None:
    with StubSM(delay_s=1.0) as sm:
        started = time.monotonic()
        result = SMClient(sm.url, timeout_s=0.2).put_target("m", {"wake_replicas": 1, "at_least": True})
        assert time.monotonic() - started < 0.9
    assert not result.ok and result.error == "timeout" and result.code is None


def test_transport_error() -> None:
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    result = SMClient(f"http://127.0.0.1:{port}", timeout_s=1.0).put_target("m", {"wake_replicas": 1})
    assert not result.ok and result.error in {"transport", "timeout"}


def test_get_state_tolerates_extra_fields() -> None:
    state = {"version": 7, "models": {"m": {"awake": 2, "bound": 4, "new": 1}}, "bindings": [], "fleet": {},
             "placement": {"free_gpus": 3}}
    with StubSM(state=state) as sm:
        assert SMClient(sm.url).get_state()["models"]["m"]["awake"] == 2


def test_dispatcher_one_inflight_per_model() -> None:
    cluster = FakeCluster(awake={"a": 1, "b": 1}, delay_s=0.3)
    d = Dispatcher(cluster.put_target)
    assert d.submit("a", "up", 2)
    assert not d.submit("a", "up", 3)       # still in flight: refused, not queued
    assert d.submit("b", "down", 1)          # another model is independent
    assert wait_until(lambda: d.inflight_count() == 0)
    done = d.drain_results()
    assert sorted((c.model, c.direction, c.target) for c in done) == [("a", "up", 2), ("b", "down", 1)]
    assert [c[0] for c in cluster.calls].count("a") == 1
    assert d.submit("a", "up", 3)            # free again
    assert wait_until(lambda: d.inflight_count() == 0)
    d.close(join_s=1.0)


def test_slow_sm_call_no_duplicate_and_tick_not_blocked(tmp_path) -> None:
    """An SM call slower than two ticks: later ticks log inflight_skip, send nothing, and
    return immediately; the result shows up on the first tick after it completed."""
    with StubSM(_json(200, {"model": "a"}), delay_s=0.6) as sm:
        config = make_config(tmp_path, {"a": limits("a")}, dry_run=False, sm_url=sm.url, tick_s=0.2)
        cluster = FakeCluster(awake={"a": 1})
        redis = FakeRedis()
        client = SMClient(sm.url, timeout_s=5.0)
        dispatcher = Dispatcher(client.put_target)
        policy = ScriptedPolicy({t: {"a": 3} for t in range(10)})
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
        assert line["sm_result"]["ok"] is True and line["sm_result"]["target"] == 3
        # FakeCluster was not told about the stub's success, so the policy asks again.
        assert line["action"] == "up"
        assert wait_until(lambda: dispatcher.inflight_count() == 0)
        assert len(sm.requests) == 2
        dispatcher.close(join_s=1.0)
