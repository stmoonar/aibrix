from __future__ import annotations

import asyncio
import csv
import json
from types import SimpleNamespace

from tre_common.metrics_schema import MetricsSnapshot
from tre_controller import profile_dump
from tre_controller.loops.action_queue import ActionQueue
from tre_controller.loops.rescue_task import run_rescue_tick
from tre_controller.planning.planner import ScaleAction
from tre_controller.profiling import PROFILE_STREAM_KEY, TickProfiler, build_profiler

from test_action_queue import FakeServiceManagerClient
from test_loop_ticks import FakeQueue, _metrics, _registry


class _Stop(Exception):
    pass


class FakePipeline:
    def __init__(self, redis: "FakeStreamRedis") -> None:
        self._redis = redis
        self._ops: list[tuple] = []

    def xadd(self, key, fields, maxlen=None, approximate=True):
        self._ops.append((key, fields, maxlen))
        return self

    def execute(self):
        for key, fields, maxlen in self._ops:
            self._redis.xadd(key, fields, maxlen=maxlen, approximate=True)
        self._ops = []
        return []


class FakeStreamRedis:
    def __init__(self) -> None:
        self.streams: dict[str, list[tuple[str, dict]]] = {}
        self._seq = 0

    def xadd(self, key, fields, maxlen=None, approximate=True, id="*"):
        self._seq += 1
        entry_id = f"{self._seq}-0"
        stream = self.streams.setdefault(key, [])
        stream.append((entry_id, dict(fields)))
        if maxlen is not None and len(stream) > maxlen:
            del stream[: len(stream) - maxlen]
        return entry_id

    def xrange(self, key, min="-", max="+"):
        return list(self.streams.get(key, []))

    def xlen(self, key):
        return len(self.streams.get(key, []))

    def pipeline(self):
        return FakePipeline(self)


def _critical_snapshot() -> MetricsSnapshot:
    return MetricsSnapshot(
        ts_ms=1,
        stale=False,
        models={"critical": _metrics("critical", generation=50.0, waiting=10.0, running=1.0, assigned=2)},
    )


# --- build_profiler toggle ---
def test_build_profiler_returns_none_when_disabled() -> None:
    cfg = SimpleNamespace(profile_enabled=False, profile_stream_maxlen=10, profile_flush_interval_s=1.0)
    assert build_profiler(cfg, FakeStreamRedis()) is None


def test_build_profiler_returns_profiler_when_enabled() -> None:
    cfg = SimpleNamespace(profile_enabled=True, profile_stream_maxlen=7, profile_flush_interval_s=0.5)
    prof = build_profiler(cfg, FakeStreamRedis())
    assert isinstance(prof, TickProfiler)
    assert prof._maxlen == 7
    assert prof._flush_interval_s == 0.5


# --- record + flush ---
def test_record_and_flush_xadds_events_to_stream() -> None:
    redis = FakeStreamRedis()
    prof = TickProfiler(redis)
    prof.record({"kind": "tick", "ts_ms": 1})
    prof.record({"kind": "poll", "ts_ms": 2})
    flushed = prof.flush()
    assert flushed == 2
    entries = redis.xrange(PROFILE_STREAM_KEY)
    assert len(entries) == 2
    decoded = [json.loads(fields["data"]) for _id, fields in entries]
    assert {e["kind"] for e in decoded} == {"tick", "poll"}
    # buffer drained, second flush is a no-op
    assert prof.flush() == 0


def test_flush_respects_maxlen_bound() -> None:
    redis = FakeStreamRedis()
    prof = TickProfiler(redis, maxlen=3)
    for i in range(5):
        prof.record({"kind": "tick", "seq": i})
    prof.flush()
    assert redis.xlen(PROFILE_STREAM_KEY) == 3


def test_flush_loop_runs_one_iteration_then_stops() -> None:
    redis = FakeStreamRedis()
    prof = TickProfiler(redis, flush_interval_s=0.01)
    prof.record({"kind": "tick", "ts_ms": 1})

    calls = {"n": 0}

    async def fake_sleep(_seconds: float) -> None:
        calls["n"] += 1
        if calls["n"] >= 2:
            raise _Stop()

    try:
        asyncio.run(prof.flush_loop(sleep=fake_sleep))
    except _Stop:
        pass
    assert redis.xlen(PROFILE_STREAM_KEY) == 1


def test_proc_sampler_loop_primes_then_records_one_sample() -> None:
    prof = TickProfiler(FakeStreamRedis())
    calls = {"n": 0}

    async def fake_sleep(_seconds: float) -> None:
        calls["n"] += 1
        if calls["n"] >= 2:
            raise _Stop()

    try:
        asyncio.run(prof.proc_sampler_loop(interval_s=0.01, sleep=fake_sleep))
    except _Stop:
        pass
    procs = [e for e in prof._buffer if e["kind"] == "proc"]
    assert len(procs) == 1
    assert procs[0]["rss_mib"] > 0
    assert "cpu_percent" in procs[0]


# --- run_planner_tick prof wiring ---
def test_run_planner_tick_without_prof_is_unchanged() -> None:
    queue = FakeQueue()
    result = run_rescue_tick(_critical_snapshot(), queue=queue, registry=_registry())
    assert result.submitted == 1
    assert len(queue.submitted) == 1


def test_run_planner_tick_with_prof_records_tick_event() -> None:
    redis = FakeStreamRedis()
    prof = TickProfiler(redis)
    queue = FakeQueue()

    result = run_rescue_tick(_critical_snapshot(), queue=queue, registry=_registry(), prof=prof)

    ticks = [e for e in prof._buffer if e["kind"] == "tick"]
    assert len(ticks) == 1
    tick = ticks[0]
    assert tick["loop"] == "rescue"
    assert tick["seq"] == 1
    assert tick["n_models"] == 1
    assert tick["n_actions"] == result.submitted == 1
    assert tick["n_pods"] == 2  # assigned_replicas of the single model
    assert tick["tick_total_ns"] > 0
    for field in ("signals_ns", "plan_ns", "safescale_ns", "submit_ns"):
        assert tick[field] >= 0
    assert "cpu_user_ms_delta" in tick and "cpu_sys_ms_delta" in tick

    # seq increments across repeated calls on the same loop
    run_rescue_tick(_critical_snapshot(), queue=FakeQueue(), registry=_registry(), prof=prof)
    ticks = [e for e in prof._buffer if e["kind"] == "tick"]
    assert [t["seq"] for t in ticks] == [1, 2]


def test_run_planner_tick_stale_snapshot_records_nothing() -> None:
    prof = TickProfiler(FakeStreamRedis())
    queue = FakeQueue()
    stale = MetricsSnapshot(ts_ms=1, models={}, stale=True)
    run_rescue_tick(stale, queue=queue, registry=_registry(), prof=prof)
    assert prof._buffer == []


# --- ActionQueue dispatch record ---
def test_action_queue_records_dispatch_event() -> None:
    prof = TickProfiler(FakeStreamRedis())
    queue = ActionQueue(FakeServiceManagerClient(), prof=prof)
    queue.submit((ScaleAction("m", 1, "critical", "rescue"),))
    asyncio.run(queue.drain_once())
    dispatch = [e for e in prof._buffer if e["kind"] == "dispatch"]
    assert len(dispatch) == 1
    assert dispatch[0]["n_actions"] == 1
    assert dispatch[0]["http_ns"] >= 0


def test_action_queue_observe_mode_records_no_dispatch() -> None:
    prof = TickProfiler(FakeStreamRedis())
    queue = ActionQueue(FakeServiceManagerClient(), is_observe=lambda: True, prof=prof)
    queue.submit((ScaleAction("m", 1, "critical", "rescue"),))
    asyncio.run(queue.drain_once())
    assert [e for e in prof._buffer if e["kind"] == "dispatch"] == []


def test_action_queue_observe_held_safescale_records_no_dispatch() -> None:
    prof = TickProfiler(FakeStreamRedis())
    queue = ActionQueue(FakeServiceManagerClient(), is_observe=lambda: True, prof=prof)
    queue.submit((ScaleAction("m", -1, "release", "safescale"),))
    results = asyncio.run(queue.drain_once())
    # safescale action is held (not dispatched, not observe_skipped) so it stays pending
    assert results == ()
    assert len(queue.pending_actions()) == 1
    assert [e for e in prof._buffer if e["kind"] == "dispatch"] == []


# --- profile_dump CLI helpers ---
def test_profile_dump_reads_stream_and_writes_union_csv(tmp_path) -> None:
    redis = FakeStreamRedis()
    redis.xadd("k", {"data": json.dumps({"kind": "tick", "loop": "rescue", "seq": 1, "ts_ms": 100, "tick_total_ns": 5})})
    redis.xadd("k", {"data": json.dumps({"kind": "poll", "ts_ms": 200, "fetch_ns": 3, "n_models": 2, "stale": False})})
    redis.xadd("k", {"data": json.dumps({"kind": "proc", "ts_ms": 300, "cpu_percent": 1.5, "rss_mib": 40.0})})

    events = profile_dump.read_events(redis, stream_key="k")
    assert len(events) == 3

    out = tmp_path / "run.csv"
    n = profile_dump.write_csv(events, str(out))
    assert n == 3

    with open(out, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        header = set(reader.fieldnames or [])
        rows = list(reader)
    assert len(rows) == 3
    # union of all fields across kinds appears as columns
    for col in ("kind", "loop", "seq", "ts_ms", "tick_total_ns", "fetch_ns", "n_models", "stale", "cpu_percent", "rss_mib"):
        assert col in header
    # a field absent from a given kind is blank for that row
    poll_row = next(r for r in rows if r["kind"] == "poll")
    assert poll_row["tick_total_ns"] == ""


def test_profile_dump_since_filters_on_event_ts_ms() -> None:
    redis = FakeStreamRedis()
    redis.xadd("k", {"data": json.dumps({"kind": "tick", "ts_ms": 100})})
    redis.xadd("k", {"data": json.dumps({"kind": "tick", "ts_ms": 200})})
    events = profile_dump.read_events(redis, stream_key="k", since_ms=150)
    assert [e["ts_ms"] for e in events] == [200]
