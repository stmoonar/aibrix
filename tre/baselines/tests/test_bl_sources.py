from __future__ import annotations

import json
from pathlib import Path

import pytest

from bl_fakes import FakeRedis, limits, make_config
from tre_baselines.keys import REPLAY_T0_KEY, req_stream_key
from tre_baselines.snapshot import COUNTER_KEYS
from tre_baselines.sources import (
    COUNTER_SOURCES,
    EventReader,
    LiveSource,
    PodEndpoint,
    endpoints_from_pod_list,
    entry_ts_ms,
    parse_event,
    parse_prometheus_text,
    pod_snapshot_from_metrics,
    read_replay_info,
    serving_bindings,
)

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "vllm030_metrics.txt"


def test_counter_table_covers_every_standard_key() -> None:
    assert set(COUNTER_SOURCES) == set(COUNTER_KEYS)


def test_live_vllm030_scrape_fixture_maps_every_key() -> None:
    text = FIXTURE.read_text(encoding="utf-8")
    snap = pod_snapshot_from_metrics(text, pod="p", model="dsqwen-7b", node="n", gpu_ids=[0], scraped_at_ms=5)
    assert set(snap.counters) == set(COUNTER_KEYS)
    assert snap.counters["gen_tokens"] == pytest.approx(7.706686e06)
    assert snap.counters["prompt_tokens"] == pytest.approx(1.092763e07)
    assert snap.counters["itl_count"] == pytest.approx(7.686447e06)
    assert snap.counters["ttft_count"] == 20239.0
    assert snap.counters["e2e_count"] == 20239.0
    assert snap.counters["req_success"] == 20239.0  # stop 2 + length 20237 (+ zero reasons)
    assert (snap.running, snap.waiting, snap.kv_usage) == (0.0, 0.0, 0.0)
    assert (snap.num_gpu_blocks, snap.block_size) == (20381, 16)
    assert snap.gpu_ids == (0,) and snap.scraped_at_ms == 5


def test_old_names_and_multi_engine_sum() -> None:
    text = "\n".join([
        '# HELP x',
        'vllm:num_requests_running{engine="0",model_name="m"} 3',
        'vllm:num_requests_running{engine="1",model_name="m"} 2',
        'vllm:num_requests_waiting{engine="0",model_name="m"} 1',
        'vllm:gpu_cache_usage_perc{engine="0",model_name="m"} 0.2',
        'vllm:gpu_cache_usage_perc{engine="1",model_name="m"} 0.4',
        'vllm:time_per_output_token_seconds_sum{model_name="m"} 1.5',
        'vllm:time_per_output_token_seconds_count{model_name="m"} 30',
        'vllm:generation_tokens_total{model_name="m"} 42',
        'vllm:cache_config_info{block_size="16",num_gpu_blocks="None",x="a \\"q\\" b"} 1.0',
        'garbage line without value',
    ])
    snap = pod_snapshot_from_metrics(text, pod="p", model="m", node=None, gpu_ids=(), scraped_at_ms=0)
    assert snap.running == 5 and snap.waiting == 1
    assert snap.kv_usage == pytest.approx(0.3)
    assert snap.counters == {"itl_sum": 1.5, "itl_count": 30.0, "gen_tokens": 42.0}
    assert snap.block_size == 16 and snap.num_gpu_blocks is None


def test_prometheus_parser_handles_escapes_and_timestamps() -> None:
    samples = parse_prometheus_text('a{l="x}y",m="\\"z\\""} 1 1700000000000\nb 2.5\n')
    assert samples[0].labels == {"l": "x}y", "m": '"z"'} and samples[0].value == 1.0
    assert samples[1].name == "b" and samples[1].value == 2.5


def test_parse_event_fields_and_defaults() -> None:
    ev = parse_event("m", "1700000000123-4", {"kind": "arr", "pod": "p1", "req_id": "r1", "in_tokens": "812",
                                                "in_src": "header", "max_tokens": "256"})
    assert ev.ts_ms == 1700000000123 and ev.entry_id == "1700000000123-4"
    assert (ev.kind, ev.pod, ev.in_tokens, ev.in_src, ev.max_tokens) == ("arr", "p1", 812, "header", 256)
    assert ev.out_tokens is None and ev.status is None and ev.reissue == "none"
    done = parse_event("m", b"5-0", {b"kind": b"done", b"req_id": b"r1", b"out_tokens": b"", b"status": b"200",
                                     b"reissue": b"continued"})
    assert done.out_tokens is None and done.status == "200" and done.reissue == "continued" and done.pod is None
    assert parse_event("m", "5-1", {"kind": "weird"}) is None
    assert parse_event("m", "5-2", {}) is None
    assert entry_ts_ms("42-7") == 42


def test_event_token_sentinels_become_none() -> None:
    # the gateway writes out_tokens=-1 when the response carried no usage
    done = parse_event("m", "5-0", {"kind": "done", "req_id": "r", "out_tokens": "-1", "in_tokens": "0"})
    assert done.out_tokens is None and done.in_tokens is None
    ok = parse_event("m", "5-1", {"kind": "done", "req_id": "r", "out_tokens": "0", "in_tokens": "1"})
    assert ok.out_tokens == 0 and ok.in_tokens == 1  # 0 output tokens is a real (empty) answer
    assert parse_event("m", "5-2", {"kind": "arr", "req_id": "r", "in_tokens": "-5"}).in_tokens is None


def test_event_reader_starts_at_now_and_keeps_cursor() -> None:
    redis = FakeRedis(now_ms=10_000)
    key = req_stream_key("m")
    redis.xadd(key, {"kind": "arr", "req_id": "old"})          # before the shell started
    reader = EventReader(redis, ["m", "n"], batch=2)
    redis.advance(1)
    assert reader.read(redis.now_ms) == {"m": (), "n": ()}
    for i in range(5):
        redis.advance(1)
        redis.xadd(key, {"kind": "ft" if i % 2 else "arr", "req_id": f"r{i}"})
    redis.xadd(key, {"kind": "bogus"})
    got = reader.read(redis.now_ms)
    assert [e.req_id for e in got["m"]] == ["r0", "r1", "r2", "r3", "r4"]   # paged by batch=2
    assert got["n"] == ()
    assert reader.parse_errors == 1 and reader.events_total == 5
    assert reader.read(redis.now_ms)["m"] == ()                          # cursor persisted
    assert reader.last_event_ms["m"] == got["m"][-1].ts_ms


def test_replay_info() -> None:
    redis = FakeRedis()
    assert read_replay_info(redis) is None
    redis.set(REPLAY_T0_KEY, json.dumps({"t0_ms": 123, "trace_path": "traces/a.jsonl"}))
    info = read_replay_info(redis)
    assert (info.t0_ms, info.trace_path) == (123, "traces/a.jsonl")
    assert info.seed is None  # additive field: markers without a seed still parse
    redis.set(REPLAY_T0_KEY, json.dumps({"t0_ms": 5, "trace_path": "t", "seed": 42}))
    assert read_replay_info(redis).seed == 42
    redis.set(REPLAY_T0_KEY, json.dumps({"t0_ms": 5, "trace_path": "t", "seed": "7"}))
    assert read_replay_info(redis).seed == 7
    redis.set(REPLAY_T0_KEY, "not json")
    assert read_replay_info(redis) is None


def _pod(name, model, ip, *, phase="Running", routable="true", port="8000", deleting=False):
    meta = {"name": name, "labels": {"model.aibrix.ai/name": model, "model.aibrix.ai/port": port,
                                     "tre.aibrix.io/routable": routable}}
    if deleting:
        meta["deletionTimestamp"] = "2026-01-01T00:00:00Z"
    return {"metadata": meta, "status": {"phase": phase, "podIP": ip}}


def test_endpoints_from_pod_list() -> None:
    doc = {"items": [
        _pod("a", "m", "10.0.0.1"), _pod("b", "m", None), _pod("c", "m", "10.0.0.3", phase="Pending"),
        _pod("d", "m", "10.0.0.4", deleting=True), _pod("e", "m", "10.0.0.5", port="9000"),
    ]}
    eps = {e.name: e for e in endpoints_from_pod_list(doc)}
    assert eps["a"].ready and eps["a"].port == 8000
    assert not eps["b"].ready and not eps["c"].ready and not eps["d"].ready
    assert eps["e"].port == 9000
    assert endpoints_from_pod_list(doc, port_override=8001)[0].port == 8001


STATE = {
    "version": 3,
    "models": {"m": {"awake": 3, "bound": 4}, "n": {"awake": 0, "bound": 2}},
    "bindings": [
        {"serve_id": "m-b", "model": "m", "node": "node-a", "gpu_ids": [1], "awake": True, "hidden": False},
        {"serve_id": "m-a", "model": "m", "node": "node-a", "gpu_ids": [0], "awake": True, "hidden": False},
        {"serve_id": "m-h", "model": "m", "node": "node-b", "gpu_ids": [0], "awake": True, "hidden": True},
        {"serve_id": "m-s", "model": "m", "node": "node-b", "gpu_ids": [1], "awake": False, "hidden": False},
        {"serve_id": "x-1", "model": "x", "node": "node-b", "gpu_ids": [2], "awake": True, "hidden": False},
    ],
}


def test_serving_bindings_excludes_hidden_and_sleeping() -> None:
    got = serving_bindings(STATE, ["m", "n"])
    assert [b.pod for b in got["m"]] == ["m-a", "m-b"]
    assert got["n"] == []


def test_live_source_gather(tmp_path) -> None:
    config = make_config(tmp_path, {"m": limits("m", 1, 4), "n": limits("n", 0, 2, tp=2)})
    redis = FakeRedis(now_ms=50_000)
    fixture = FIXTURE.read_text(encoding="utf-8")
    fetched = []

    def fetch(url, timeout):
        fetched.append((url, timeout))
        if "10.0.0.2" in url:
            raise OSError("connection refused")
        return fixture

    pods = [PodEndpoint("m-a", "m", "10.0.0.1", 8000, True), PodEndpoint("m-b", "m", "10.0.0.2", 8000, True),
            PodEndpoint("m-h", "m", "10.0.0.3", 8000, True)]
    source = LiveSource(config, redis, lambda: STATE, lambda: pods, fetch_text=fetch)
    snap = source.gather(tick=7)
    redis.advance(100)
    redis.xadd(req_stream_key("m"), {"kind": "arr", "req_id": "r", "pod": "m-a"})
    snap2 = source.gather(tick=8)
    source.close()

    assert snap.now_ms == 50_000 and snap.tick == 7 and snap.replay is None
    m = snap.models["m"]
    assert m.awake == 3                                   # SM count, not the pods seen
    assert [p.pod for p in m.pods] == ["m-a"] and m.pods[0].node == "node-a" and m.pods[0].gpu_ids == (0,)
    assert m.unscraped == ("m-b",)
    assert m.max_replicas == 4 and m.gpus_per_replica == 1
    assert snap.models["n"].pods == () and snap.models["n"].gpus_per_replica == 2
    assert sorted({u for u, _ in fetched}) == ["http://10.0.0.1:8000/metrics", "http://10.0.0.2:8000/metrics"]  # hidden m-h never scraped
    assert all(t == config.scrape_timeout_s for _, t in fetched)
    assert source.scrape_failures == 2 and snap.extra["scrape_failed"] == 1
    assert [e.req_id for e in snap2.models["m"].events] == ["r"]
    assert snap2.extra["event_lag_s"] == {"m": 0.0}


def test_live_source_missing_pod_counts_unscraped(tmp_path) -> None:
    config = make_config(tmp_path, {"m": limits("m")})
    source = LiveSource(config, FakeRedis(), lambda: STATE, lambda: [], fetch_text=lambda u, t: "")
    snap = source.gather()
    source.close()
    assert snap.models["m"].pods == () and snap.models["m"].unscraped == ("m-a", "m-b")


def test_live_source_bad_state_raises(tmp_path) -> None:
    config = make_config(tmp_path, {"zz": limits("zz")})
    source = LiveSource(config, FakeRedis(), lambda: STATE, lambda: [], fetch_text=lambda u, t: "")
    with pytest.raises(ValueError):
        source.gather()
    source.close()


def _metrics(running, waiting=0.0) -> str:
    return (f'vllm:num_requests_running{{model_name="m"}} {running}\n'
            f'vllm:num_requests_waiting{{model_name="m"}} {waiting}\n')


def test_missing_gauges_are_unknown_not_zero() -> None:
    snap = pod_snapshot_from_metrics('vllm:generation_tokens_total{model_name="m"} 5\n'
                                     'vllm:num_requests_waiting{model_name="m"} NaN\n',
                                     pod="p", model="m", node=None, gpu_ids=(), scraped_at_ms=0)
    assert snap.running is None and snap.waiting is None and snap.queued is None


ONE_POD = {"version": 1, "models": {"m": {"awake": 1, "bound": 1}},
           "bindings": [{"serve_id": "m-a", "model": "m", "node": "n", "gpu_ids": [0], "awake": True}]}


def test_event_gap_cohort_blocks_scale_down_until_drained(tmp_path) -> None:
    """Shell start / lost events: requests the engine runs but the stream never showed
    are the cohort; scale-down evidence is incomplete until they are gone (state, not time)."""
    from tre_baselines.snapshot import evidence_gaps

    config = make_config(tmp_path, {"m": limits("m")})
    redis = FakeRedis(now_ms=100_000)
    engine = {"q": 2}  # two requests in flight before the shell started
    source = LiveSource(config, redis, lambda: ONE_POD,
                        lambda: [PodEndpoint("m-a", "m", "10.0.0.1", 8000, True)],
                        fetch_text=lambda u, t: _metrics(engine["q"]))
    key = req_stream_key("m")

    def gaps():
        snap = source.gather()
        redis.advance(2000)
        return evidence_gaps(snap.models["m"], snap.now_ms, 2.0, events=True)

    assert "event_gap" in gaps()                         # cohort of 2, nothing tracked
    redis.xadd(key, {"kind": "arr", "req_id": "new", "pod": "m-a"})
    engine["q"] = 3                                      # cohort still running + the new one
    assert "event_gap" in gaps()
    engine["q"] = 1                                      # cohort drained, "new" is tracked
    assert gaps() == ()
    redis.xadd(key, {"kind": "done", "req_id": "new", "pod": "m-a"})
    assert gaps() == ()                                  # verified until the next gap
    # the stream is trimmed past the cursor: a new cohort, unverified again
    for i in range(3):
        redis.advance(1)
        redis.xadd(key, {"kind": "arr", "req_id": f"lost{i}", "pod": "m-a"})
    redis.streams[key] = redis.streams[key][-1:]
    engine["q"] = 3
    assert "event_gap" in gaps()
    engine["q"] = 0                                      # an idle engine has nothing unknown
    assert "event_gap" not in gaps()
    source.close()


def test_trimmed_stream_restarts_the_event_history(tmp_path) -> None:
    redis = FakeRedis(now_ms=10_000)
    key = req_stream_key("m")
    reader = EventReader(redis, ["m"])
    redis.xadd(key, {"kind": "arr", "req_id": "a", "pod": "p"})
    redis.advance(1)
    reader.read(redis.now_ms)
    assert reader.history_since_ms("m") == 10_001                 # shell start
    redis.advance(5000)
    for i in range(3):
        redis.advance(1)
        redis.xadd(key, {"kind": "arr", "req_id": f"b{i}", "pod": "p"})
    redis.streams[key] = redis.streams[key][-1:]          # MAXLEN trimmed past our cursor
    reader.read(redis.now_ms)
    assert reader.history_since_ms("m") == redis.now_ms and reader.gaps["m"] == 1


def test_a_gap_forgets_pre_gap_tracked_requests(tmp_path) -> None:
    """Review P2-4: a request tracked before the gap whose done was trimmed away must not
    cover an unknown request after the gap."""
    from tre_baselines.snapshot import evidence_gaps

    config = make_config(tmp_path, {"m": limits("m")})
    redis = FakeRedis(now_ms=100_000)
    engine = {"q": 0}
    source = LiveSource(config, redis, lambda: ONE_POD,
                        lambda: [PodEndpoint("m-a", "m", "10.0.0.1", 8000, True)],
                        fetch_text=lambda u, t: _metrics(engine["q"]))
    key = req_stream_key("m")

    def gaps():
        snap = source.gather()
        redis.advance(2000)
        return evidence_gaps(snap.models["m"], snap.now_ms, 2.0, events=True)

    assert "event_gap" not in gaps()                      # idle engine: verified
    redis.xadd(key, {"kind": "arr", "req_id": "old", "pod": "m-a"})
    engine["q"] = 1
    assert gaps() == ()                                   # "old" is tracked and running
    for i in range(3):                                    # old's done and a new arr get trimmed
        redis.advance(1)
        redis.xadd(key, {"kind": "done" if i == 0 else "arr", "req_id": "old" if i == 0 else f"n{i}",
                         "pod": "m-a"})
    redis.streams[key] = redis.streams[key][-1:]          # only n2's arr survives
    engine["q"] = 2                                       # n1 (lost) and n2 run; "old" is gone
    assert "event_gap" in gaps()                          # stale "old" no longer covers n1
    source.close()


def test_wakeable_slots_from_the_sm_gpu_view() -> None:
    from tre_baselines.sources import wakeable_slots

    state = {"bindings": [
        {"serve_id": "a0", "model": "a", "node": "n", "gpu_ids": [0], "awake": True},
        {"serve_id": "a1", "model": "a", "node": "n", "gpu_ids": [1], "awake": False},
        {"serve_id": "a2", "model": "a", "node": "n", "gpu_ids": [2], "awake": False},
        {"serve_id": "t0", "model": "t", "node": "n", "gpu_ids": [1, 2], "awake": False},
        {"serve_id": "t1", "model": "t", "node": "n", "gpu_ids": [2, 3], "awake": False},
    ], "gpus": [{"node": "n", "gpu": g, "wakeable": g in (1, 2)} for g in range(4)]}
    assert wakeable_slots(state, "a") == 2
    assert wakeable_slots(state, "t") == 1          # TP=2: only (1,2) is fully free
    assert wakeable_slots({"bindings": state["bindings"]}, "a") is None  # no gpus[]: unknown
    from tre_baselines.sources import wakeable_gpus
    assert wakeable_gpus(state) == ["n/1", "n/2"] and wakeable_gpus({}) is None


def test_parse_event_stream_flag() -> None:
    assert parse_event("m", "1-0", {"kind": "arr", "req_id": "r", "stream": "false"}).stream is False
    assert parse_event("m", "1-0", {"kind": "arr", "req_id": "r", "stream": "true"}).stream is True
    assert parse_event("m", "1-0", {"kind": "arr", "req_id": "r"}).stream is None
