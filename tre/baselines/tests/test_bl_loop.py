from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request

from bl_fakes import FakeCluster, FakeRedis, FakeSource, ScriptedPolicy, StubSM, limits, make_config, wait_until
from tre_baselines.keys import CONTROLLER_MODE_KEY, DECISIONS_STREAM, OWNER_KEY, decision_key
from tre_baselines.loop import RELEASE_LUA, RENEW_LUA, BaselineShell, OwnerLock, clamp, make_http_server
from tre_baselines.policies.static import StaticPolicy
from tre_baselines.sm_client import Backoff, Dispatcher, SMClient, SMResult

MODELS = {"a": limits("a", 1, 4), "b": limits("b", 1, 4), "c": limits("c", 1, 3, tp=2)}


def _shell(tmp_path, policy, *, dry_run, awake=None, lock=True, redis=None, sm_put=None, **cfg):
    config = make_config(tmp_path, MODELS, dry_run=dry_run, **cfg)
    cluster = FakeCluster(awake=dict(awake or {"a": 2, "b": 1, "c": 1}))
    redis = redis or FakeRedis()
    dispatcher = Dispatcher(sm_put or cluster.put_target, sleep_path=config.sleep_path)
    owner = OwnerLock(redis, config.lock_ttl_s, token="me") if lock else None
    shell = BaselineShell(config, FakeSource(config, cluster, redis), policy, dispatcher, redis, lock=owner)
    return shell, cluster, redis, dispatcher


def _run(shell, dispatcher, ticks):
    lines = []
    for _ in range(ticks):
        lines.append(shell.tick_once())
        assert wait_until(lambda: dispatcher.inflight_count() == 0)
    return lines


SCRIPT = {
    0: {"a": 1, "b": 3, "c": 99},      # a down, b up, c clamped to 3 (up)
    3: {"a": 0, "b": 3},               # a clamped to min 1 = awake -> none
    5: {"a": 4, "b": 1, "c": 1},       # a up, b down, c down
    9: {"b": 2},
}


def test_clamp() -> None:
    assert clamp(0, 1, 4) == 1 and clamp(9, 1, 4) == 4 and clamp(2, 1, 4) == 2


def test_dry_run_twenty_ticks_sends_nothing(tmp_path) -> None:
    for i, policy in enumerate((StaticPolicy(), ScriptedPolicy(SCRIPT))):
        with StubSM() as sm:
            shell, cluster, redis, dispatcher = _shell(
                tmp_path, policy, dry_run=True, sm_put=SMClient(sm.url).put_target,
                sm_url=sm.url, log_dir=str(tmp_path / f"logs{i}"))
            lines = _run(shell, dispatcher, 20)
            assert sm.requests == [] and dispatcher.submitted == []
        flat = [line for tick in lines for line in tick]
        assert len(flat) == 60
        assert {line["action"] for line in flat} <= {"none", "dry_run"}
        assert all(line["dry_run"] for line in flat)
        assert OWNER_KEY not in redis.kv          # a dry-run shell never takes the lock
    first = lines[0]
    assert [(l["model"], l["action"], l["clamped"]) for l in first] == [
        ("a", "dry_run", 1), ("b", "dry_run", 3), ("c", "dry_run", 3)]
    logs = list((tmp_path / "logs1").glob("decisions-scripted-*.jsonl"))
    assert len(logs) == 1
    rows = [json.loads(x) for x in logs[0].read_text().splitlines()]
    assert len(rows) == 60 and {"ts_ms", "policy", "model", "awake", "raw_desired", "clamped", "action",
                                "reason", "inputs"} <= set(rows[0])
    stored = json.loads(redis.kv[decision_key("a")])
    assert stored["model"] == "a" and stored["tick"] == 19
    assert any(call[2]["ex"] == 3600 for call in redis.set_calls if call[0] == decision_key("a"))


def test_active_put_sequence_downs_before_ups(tmp_path) -> None:
    shell, cluster, redis, dispatcher = _shell(tmp_path, ScriptedPolicy(SCRIPT), dry_run=False)
    lines = _run(shell, dispatcher, 20)
    assert redis.kv[OWNER_KEY] == "me"
    # tick 0: a down first, then b and c up (c clamped to its cap 3)
    assert dispatcher.submitted[:3] == [(1, "a", "down", 1), (2, "b", "up", 3), (3, "c", "up", 3)]
    # bodies (the arrival order at the SM is up to the worker threads)
    assert sorted(cluster.calls[:3], key=lambda c: c[0]) == [
        ("a", {"wake_replicas": 1, "sleep_path": "scale_down"}),
        ("b", {"wake_replicas": 3, "at_least": True}),
        ("c", {"wake_replicas": 3, "at_least": True}),
    ]
    # tick 3: a asks for 0, clamped to min 1 == awake -> nothing sent
    assert [l["action"] for l in lines[3]] == ["none", "none", "none"]
    assert lines[3][0]["raw_desired"] == 0 and lines[3][0]["clamped"] == 1
    # tick 5: downs (b, c) before the up (a)
    assert [(m, d, t) for _, m, d, t in dispatcher.submitted[3:6]] == [
        ("b", "down", 1), ("c", "down", 1), ("a", "up", 4)]
    # tick 9: b up to 2
    assert [(m, d, t) for _, m, d, t in dispatcher.submitted[6:]] == [("b", "up", 2)]
    assert cluster.awake == {"a": 4, "b": 2, "c": 1}
    # results come back on the following tick's line
    assert lines[1][0]["sm_result"]["ok"] is True
    metrics = shell.metrics_text()
    assert 'tre_bl_actions_total{policy="scripted",model="a",action="down"} 1' in metrics
    assert 'tre_bl_actions_total{policy="scripted",model="a",action="up"} 1' in metrics
    # a: down at t0, up at t5 (10 s later) -> reversal; b: up, down, up -> 2 reversals
    assert 'tre_bl_direction_reversals_60s_total{policy="scripted",model="a"} 1' in metrics
    assert 'tre_bl_direction_reversals_60s_total{policy="scripted",model="b"} 2' in metrics
    dispatcher.close(join_s=1.0)


def test_static_active_sends_nothing(tmp_path) -> None:
    shell, cluster, redis, dispatcher = _shell(tmp_path, StaticPolicy(), dry_run=False)
    _run(shell, dispatcher, 20)
    assert dispatcher.submitted == [] and cluster.calls == []


def test_sm_refusal_is_logged_not_retried_immediately(tmp_path) -> None:
    shell, cluster, redis, dispatcher = _shell(tmp_path, ScriptedPolicy({0: {"b": 3}}), dry_run=False)
    cluster.refuse.add("b")
    lines = _run(shell, dispatcher, 2)
    assert len(cluster.calls) == 1
    b1 = [l for l in lines[1] if l["model"] == "b"][0]
    assert b1["sm_result"]["ok"] is False and b1["sm_result"]["code"] == 409
    assert shell.stats.sm_failures == 1


def test_not_lock_owner_forces_dry_run(tmp_path) -> None:
    redis = FakeRedis()
    redis.set(OWNER_KEY, "someone-else", px=30_000)
    shell, cluster, redis, dispatcher = _shell(
        tmp_path, ScriptedPolicy({t: {"b": 3} for t in range(20)}), dry_run=False, redis=redis)
    lines = _run(shell, dispatcher, 3)
    assert dispatcher.submitted == [] and cluster.calls == []
    assert all(l["dry_run"] and not l["owner"] for tick in lines for l in tick)
    assert [l["action"] for l in lines[0] if l["model"] == "b"] == ["dry_run"]
    assert "tre_bl_dry_run{policy=\"scripted\"} 1" in shell.metrics_text()
    redis.delete(OWNER_KEY)                 # the other shell went away
    line = [l for l in shell.tick_once() if l["model"] == "b"][0]
    assert line["action"] == "up" and line["owner"] and not line["dry_run"]
    assert redis.kv[OWNER_KEY] == "me" and redis.ttl_ms[OWNER_KEY] == 30_000
    assert wait_until(lambda: dispatcher.inflight_count() == 0)
    dispatcher.close(join_s=1.0)


def test_write_redis_can_be_disabled(tmp_path) -> None:
    shell, cluster, redis, dispatcher = _shell(tmp_path, StaticPolicy(), dry_run=True, write_redis=False)
    _run(shell, dispatcher, 2)
    assert not any(k.startswith("tre:v2:bl:decision:") for k in redis.kv)


def _get(url):
    try:
        with urllib.request.urlopen(url, timeout=2) as resp:
            return resp.status, resp.read().decode()
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read().decode()


def test_failing_ticks_turn_healthz_503(tmp_path) -> None:
    failing = {"on": True}
    config = make_config(tmp_path, MODELS, dry_run=True, max_tick_failures=3)
    cluster = FakeCluster(awake={"a": 1, "b": 1, "c": 1})
    redis = FakeRedis()
    source = FakeSource(config, cluster, redis, fail=lambda tick: failing["on"])
    dispatcher = Dispatcher(cluster.put_target)
    shell = BaselineShell(config, source, StaticPolicy(), dispatcher, redis)
    server = make_http_server(shell, 0, host="127.0.0.1")
    threading.Thread(target=server.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        assert _get(base + "/healthz")[0] == 200
        for _ in range(2):
            assert shell.safe_tick() is None
        assert _get(base + "/healthz")[0] == 200
        assert shell.safe_tick() is None
        code, body = _get(base + "/healthz")
        assert code == 503 and json.loads(body)["consecutive_failures"] == 3
        assert "source failure" in json.loads(body)["last_error"]
        failing["on"] = False
        assert shell.safe_tick() is not None
        assert _get(base + "/healthz")[0] == 200
        code, metrics = _get(base + "/metrics")
        assert code == 200 and 'tre_bl_tick_failures_total{policy="scripted"} 3' in metrics
        assert 'tre_bl_ticks_total{policy="scripted"} 1' in metrics
        assert _get(base + "/nope")[0] == 404
    finally:
        server.shutdown()
        server.server_close()


def test_run_loop_stops(tmp_path) -> None:
    shell, cluster, redis, dispatcher = _shell(tmp_path, StaticPolicy(), dry_run=True)
    shell.run(max_ticks=5)
    assert shell.stats.ticks == 5
    t = threading.Thread(target=shell.run)
    t.start()
    shell.stop()
    t.join(2.0)
    assert not t.is_alive()


def test_owner_lock_renew_and_release() -> None:
    redis = FakeRedis()
    lock = OwnerLock(redis, 10, token="t1")
    other = OwnerLock(redis, 10, token="t2")
    assert lock.ensure() and redis.ttl_ms[OWNER_KEY] == 10_000
    assert not other.ensure()
    redis.ttl_ms[OWNER_KEY] = 1
    assert lock.ensure() and redis.ttl_ms[OWNER_KEY] == 10_000   # renewed
    other.release()
    assert redis.kv[OWNER_KEY] == "t1"
    lock.release()
    assert OWNER_KEY not in redis.kv
    assert other.ensure()


def test_owner_lock_renewal_never_touches_another_shells_lock() -> None:
    redis = FakeRedis()
    mine = OwnerLock(redis, 10, token="t1")
    assert mine.ensure()
    redis.kv[OWNER_KEY] = "t2"            # expired and taken by another shell in between
    redis.ttl_ms[OWNER_KEY] = 5_000
    assert not mine.ensure()              # compare-and-pexpire: no renewal, SET NX refused
    assert redis.ttl_ms[OWNER_KEY] == 5_000 and redis.kv[OWNER_KEY] == "t2"
    mine.release()                        # compare-and-delete: the other lock stays
    assert redis.kv[OWNER_KEY] == "t2"
    assert all(call[0] in (RENEW_LUA, RELEASE_LUA) for call in redis.eval_calls)


# ------------------------------------------------------------------ controller-mode guard


def test_controller_not_in_observe_forces_dry_run(tmp_path) -> None:
    redis = FakeRedis()
    redis.kv[CONTROLLER_MODE_KEY] = "active"
    shell, cluster, redis, dispatcher = _shell(
        tmp_path, ScriptedPolicy({t: {"b": 3} for t in range(20)}), dry_run=False, redis=redis)
    lines = _run(shell, dispatcher, 2)
    assert dispatcher.submitted == [] and cluster.calls == []
    b = [l for l in lines[0] if l["model"] == "b"][0]
    assert b["action"] == "guard_controller_active" and b["dry_run"] and b["controller_mode"] == "active"
    assert [l["action"] for l in lines[0] if l["model"] != "b"] == ["none", "none"]
    metrics = shell.metrics_text()
    assert 'tre_bl_controller_guard{policy="scripted"} 1' in metrics
    assert 'tre_bl_controller_guard_ticks_total{policy="scripted"} 2' in metrics
    assert 'action="guard_controller_active"} 2' in metrics
    assert shell.health_doc()["controller_guard"] is True
    for mode in ("weird", "ACTIVE"):     # anything but observe is a guard
        redis.kv[CONTROLLER_MODE_KEY] = mode
        assert [l["action"] for l in shell.tick_once() if l["model"] == "b"] == ["guard_controller_active"]
    redis.kv[CONTROLLER_MODE_KEY] = "observe"
    line = [l for l in shell.tick_once() if l["model"] == "b"][0]
    assert line["action"] == "up" and not line["dry_run"] and line["controller_mode"] == "observe"
    assert 'tre_bl_controller_guard{policy="scripted"} 0' in shell.metrics_text()
    assert wait_until(lambda: dispatcher.inflight_count() == 0)
    dispatcher.close(join_s=1.0)


def test_controller_mode_missing_is_observe_and_read_failure_guards(tmp_path) -> None:
    shell, cluster, redis, dispatcher = _shell(tmp_path, ScriptedPolicy({0: {"b": 3}, 1: {"b": 4}}), dry_run=False)
    assert CONTROLLER_MODE_KEY not in redis.kv
    line = [l for l in shell.tick_once() if l["model"] == "b"][0]
    assert line["action"] == "up" and "controller_mode" not in line
    assert wait_until(lambda: dispatcher.inflight_count() == 0)
    real_get = redis.get

    def flaky_get(key):
        if key == CONTROLLER_MODE_KEY:
            raise ConnectionError("redis down")
        return real_get(key)

    redis.get = flaky_get
    line = [l for l in shell.tick_once() if l["model"] == "b"][0]
    assert line["action"] == "guard_controller_active" and line["controller_mode"] == "<read failed>"
    dispatcher.close(join_s=1.0)


def test_dry_run_shell_does_not_read_the_controller_mode(tmp_path) -> None:
    redis = FakeRedis()
    redis.kv[CONTROLLER_MODE_KEY] = "active"
    shell, cluster, redis, dispatcher = _shell(tmp_path, ScriptedPolicy({0: {"b": 3}}), dry_run=True, redis=redis)
    line = [l for l in shell.tick_once() if l["model"] == "b"][0]
    assert line["action"] == "dry_run" and "controller_mode" not in line


# ------------------------------------------------------------------ decision stream


def test_decision_lines_are_mirrored_to_the_stream(tmp_path) -> None:
    shell, cluster, redis, dispatcher = _shell(tmp_path, ScriptedPolicy(SCRIPT), dry_run=True)
    lines = [l for tick in _run(shell, dispatcher, 3) for l in tick]
    entries = redis.streams[DECISIONS_STREAM]
    assert [json.loads(f["line"]) for _, f in entries] == [json.loads(json.dumps(l, sort_keys=True, default=str))
                                                           for l in lines]
    assert set(redis.xadd_calls) == {(DECISIONS_STREAM, 100_000, True)}
    shell, cluster, redis, dispatcher = _shell(tmp_path, StaticPolicy(), dry_run=True, decision_stream=False,
                                               log_dir=str(tmp_path / "l2"))
    _run(shell, dispatcher, 2)
    assert DECISIONS_STREAM not in redis.streams


# ------------------------------------------------------------------ SM refusal backoff


def _refusing_put(cluster, state, refuse_until_tick, retry_after_s=None):
    calls = []

    def put(model, body):
        calls.append(state["tick"])
        if state["tick"] < refuse_until_tick:
            return SMResult(ok=False, code=409, error="gpu_busy", retry_after_s=retry_after_s)
        return cluster.put_target(model, body)

    return put, calls


def _drive(shell, dispatcher, state, ticks):
    lines = []
    for t in range(ticks):
        state["tick"] = t
        lines.append({l["model"]: l for l in shell.tick_once()})
        assert wait_until(lambda: dispatcher.inflight_count() == 0)
    return lines


def test_sm_refusals_back_off_exponentially(tmp_path) -> None:
    state = {"tick": 0}
    cluster = FakeCluster(awake={"a": 2, "b": 1, "c": 1})
    put, calls = _refusing_put(cluster, state, refuse_until_tick=10)
    config = make_config(tmp_path, MODELS, dry_run=False, tick_s=2.0)  # = FakeSource 2 s step
    redis = FakeRedis()
    dispatcher = Dispatcher(put)
    shell = BaselineShell(config, FakeSource(config, cluster, redis), ScriptedPolicy({t: {"b": 3} for t in range(30)}),
                          dispatcher, redis, lock=OwnerLock(redis, 30, token="me"))
    lines = _drive(shell, dispatcher, state, 30)
    # refused at ticks 0, 2, 5 (waits 2 s, 4 s, 8 s from the tick that saw the result), ok at 10
    assert calls == [0, 2, 5, 10]
    assert [lines[t]["b"]["action"] for t in range(11)] == [
        "up", "backoff", "up", "backoff", "backoff", "up", "backoff", "backoff", "backoff", "backoff", "up"]
    assert lines[6]["b"]["backoff_delay_s"] == 8.0 and lines[6]["b"]["backoff_s"] == 8.0
    assert cluster.awake["b"] == 3 and all(lines[t]["b"]["action"] == "none" for t in range(12, 30))
    assert shell.backoff.delay_s("b") is None                       # reset by the success
    assert 'tre_bl_backoff_skips_total{policy="scripted"} 7' in shell.metrics_text()
    dispatcher.close(join_s=1.0)


def test_backoff_honours_retry_after_and_resets_when_desired_is_awake(tmp_path) -> None:
    state = {"tick": 0}
    cluster = FakeCluster(awake={"a": 2, "b": 1, "c": 1})
    put, calls = _refusing_put(cluster, state, refuse_until_tick=1, retry_after_s=9.0)
    config = make_config(tmp_path, MODELS, dry_run=False, tick_s=2.0)
    redis = FakeRedis()
    dispatcher = Dispatcher(put)
    shell = BaselineShell(config, FakeSource(config, cluster, redis), ScriptedPolicy({t: {"b": 3} for t in range(8)}),
                          dispatcher, redis, lock=OwnerLock(redis, 30, token="me"))
    _drive(shell, dispatcher, state, 8)
    assert calls == [0, 6]   # the result is seen at tick 1 (t=2 s); 9 s later is tick 6 (t=12 s)
    # a model whose desired count goes back to awake forgets its backoff
    state2 = {"tick": 0}
    cluster2 = FakeCluster(awake={"a": 2, "b": 1, "c": 1})
    put2, calls2 = _refusing_put(cluster2, state2, refuse_until_tick=1, retry_after_s=9.0)
    dispatcher2 = Dispatcher(put2)
    redis2 = FakeRedis()
    shell2 = BaselineShell(config, FakeSource(config, cluster2, redis2),
                           ScriptedPolicy({0: {"b": 3}, 1: {"b": 1}, 2: {"b": 3}}), dispatcher2, redis2,
                           lock=None)
    lines = _drive(shell2, dispatcher2, state2, 3)
    assert [lines[t]["b"]["action"] for t in range(3)] == ["up", "none", "up"] and calls2 == [0, 2]
    dispatcher.close(join_s=1.0)
    dispatcher2.close(join_s=1.0)


def test_backoff_schedule_and_cap() -> None:
    b = Backoff(tick_s=2.0, max_s=60.0)
    delays = [b.failed("m", 0) for _ in range(7)]
    assert delays == [2.0, 4.0, 8.0, 16.0, 32.0, 60.0, 60.0]
    assert b.failed("m", 0, retry_after_s=90.0) == 90.0          # the SM retry_after wins
    b.reset("m")
    assert b.remaining_s("m", 0) is None and b.failed("m", 1000, retry_after_s=0.5) == 2.0
    assert b.remaining_s("m", 2000) == 1.0 and b.remaining_s("m", 3000) is None


# ------------------------------------------------------------------ liveness vs readiness


def test_livez_follows_the_loop_not_the_ticks(tmp_path) -> None:
    config = make_config(tmp_path, MODELS, dry_run=True, max_tick_failures=2, tick_s=0.02, liveness_stall_s=5.0)
    cluster = FakeCluster(awake={"a": 1, "b": 1, "c": 1})
    redis = FakeRedis()
    source = FakeSource(config, cluster, redis, fail=lambda tick: True)   # e.g. the SM is down
    shell = BaselineShell(config, source, StaticPolicy(), Dispatcher(cluster.put_target), redis)
    server = make_http_server(shell, 0, host="127.0.0.1")
    threading.Thread(target=server.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        assert _get(base + "/livez")[0] == 503                          # loop not started
        loop = threading.Thread(target=shell.run, daemon=True)
        loop.start()
        assert wait_until(lambda: shell.stats.tick_failures >= 3)
        assert _get(base + "/healthz")[0] == 503                        # not ready ...
        assert _get(base + "/livez")[0] == 200                          # ... but alive
        shell.stop()
        loop.join(2.0)
        assert _get(base + "/livez")[0] == 503
    finally:
        server.shutdown()
        server.server_close()


def test_livez_fails_when_the_loop_is_wedged(tmp_path) -> None:
    config = make_config(tmp_path, MODELS, dry_run=True, tick_s=0.02, liveness_stall_s=5.0)
    shell, cluster, redis, dispatcher = _shell(tmp_path, StaticPolicy(), dry_run=True)
    shell.config = config
    import time as _time

    shell._running, shell._beat = True, _time.monotonic() - 1000.0  # running, no tick for 1000 s
    assert not shell.alive()
    shell._beat = _time.monotonic()
    assert shell.alive()
