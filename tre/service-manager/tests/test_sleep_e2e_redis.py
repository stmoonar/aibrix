"""End-to-end sleep primitive against a REAL Redis, a fake gateway plugin and a fake
vLLM HTTP server (plan 2026-09-27 D1-D4, integration of the SM and gateway branches).

What is real here: redis-py against a real Redis server (the reservation Lua script,
the journal / stats hashes, the gateway contract keys), the SM's ``VllmOps`` over
HTTP, ``GatewayState`` with the real monotonic clock. What is fake: k8s (the
``FakeRuntime`` pod patches), the gateway plugin (a thread that mimics the Go plugin
in pkg/plugins/gateway/tre_transparent_sleep.go: heartbeat ZSET score = epoch ms,
inflight written BEFORE seen in one MULTI pipeline, JSON field names of the Go
structs) and the vLLM engine.

Opt-in only, so ``make check`` stays hermetic: set ``TRE_SM_E2E_REDIS_URL`` to a
THROWAWAY Redis (e.g. ``docker run -d --rm -p 127.0.0.1:16399:6379 redis:7``). The
test FLUSHes that database; it refuses a non-empty database it did not mark itself.
"""

from __future__ import annotations

import json
import os
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

import pytest

from tre_common import rediskeys
from tre_sm.ops import sleep_primitive as sp
from tre_sm.ops.sleep_primitive import (
    GatewayAckTimeout,
    GatewayState,
    SleepFailed,
    SleepJournal,
    SleepPrimitive,
    SleepTarget,
)
from tre_sm.ops.vllm_ops import VllmOps
from tre_sm.state.sleep_reservations import SleepReservations

from sm_test_fakes import FakeRuntime, binding_of, pod, policy

REDIS_URL = os.environ.get("TRE_SM_E2E_REDIS_URL", "").strip()
MARKER_KEY = "tre:e2e:throwaway"
POD = "pod-e2e"
INSTANCE = "tre-gateway-plugins-e2e-0"

pytestmark = pytest.mark.skipif(not REDIS_URL, reason="TRE_SM_E2E_REDIS_URL not set (opt-in e2e)")


# ----------------------------------------------------------------------------- redis
@pytest.fixture()
def redis_client():
    redis = pytest.importorskip("redis")
    client = redis.Redis.from_url(REDIS_URL, socket_timeout=2)  # bytes, like server.py
    try:
        client.ping()
    except Exception as exc:  # pragma: no cover - environment dependent
        pytest.skip(f"redis at {REDIS_URL} unreachable: {exc}")
    if client.dbsize() and not client.exists(MARKER_KEY):
        pytest.skip(f"{REDIS_URL} is not empty and not a TRE e2e throwaway db; refusing to flush")
    client.flushdb()
    client.set(MARKER_KEY, "1")
    yield client
    client.flushdb()


# ------------------------------------------------------------------------- event log
class Events:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.items: list[tuple] = []

    def add(self, *event) -> None:
        with self._lock:
            self.items.append((time.monotonic(),) + event)

    def index(self, predicate) -> int:
        with self._lock:
            for i, item in enumerate(self.items):
                if predicate(item[1:]):
                    return i
        raise AssertionError(f"event not found; log={self.items}")

    def kinds(self) -> list[tuple]:
        with self._lock:
            return [item[1:] for item in self.items]


class RecordingRuntime(FakeRuntime):
    """FakeRuntime whose pod patches land in the shared, timestamped event log."""

    def __init__(self, snapshots, events: Events) -> None:
        super().__init__(snapshots)
        self._log = events
        self._lock = threading.Lock()

    def write_binding_annotations(self, binding, *, state):
        with self._lock:
            gen = super().write_binding_annotations(binding, state=state)
        self._log.add("patch", binding.serve_id, state, gen)
        return gen

    def route_state(self, name: str) -> tuple[int, bool] | None:
        with self._lock:
            snapshot = self.snapshots.get(name)
            if snapshot is None:
                return None
            return self.gen.get(name, 0), bool(snapshot.routable)


# ---------------------------------------------------------------------- fake engine
class FakeEngine:
    """vLLM HTTP surface the SM uses: /version /metrics /is_sleeping /sleep /wake_up."""

    def __init__(self, events: Events) -> None:
        self.events = events
        self.running = 0
        self.sleeping = False
        self.sleep_calls: list[dict] = []
        engine = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):  # silence
                return

            def _send(self, code, body, ctype="application/json"):
                data = body.encode() if isinstance(body, str) else body
                self.send_response(code)
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def do_GET(self):
                path = urlparse(self.path).path
                if path == "/version":
                    self._send(200, json.dumps({"version": "0.30.0"}))
                elif path == "/metrics":
                    self._send(
                        200,
                        "# HELP vllm:num_requests_running x\n"
                        f'vllm:num_requests_running{{model_name="m1"}} {float(engine.running)}\n'
                        f'vllm:num_requests_waiting{{model_name="m1"}} 0.0\n',
                        "text/plain",
                    )
                elif path == "/is_sleeping":
                    self._send(200, json.dumps({"is_sleeping": engine.sleeping}))
                else:
                    self._send(404, "{}")

            def do_POST(self):
                parsed = urlparse(self.path)
                if parsed.path == "/sleep":
                    call = {
                        "mode": (parse_qs(parsed.query).get("mode") or [None])[0],
                        "hidden": self.headers.get("X-TRE-Hidden"),
                        "running": engine.running,
                    }
                    engine.sleep_calls.append(call)
                    engine.events.add("vllm_sleep", call["mode"], call["hidden"], call["running"])
                    engine.sleeping = True
                    self._send(200, "{}")
                elif parsed.path == "/wake_up":
                    engine.events.add("vllm_wake_up")
                    engine.sleeping = False
                    self._send(200, "{}")
                else:
                    self._send(404, "{}")

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.port = self.server.server_address[1]
        self._thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self.server.shutdown()
        self.server.server_close()


# ---------------------------------------------------------------------- fake plugin
class FakePlugin:
    """Mimics the Go plugin's Redis writes (tre_transparent_sleep.go)."""

    def __init__(self, redis, runtime: RecordingRuntime, events: Events, *, instance=INSTANCE,
                 ack_delay_s=0.3, ack=True, beat_s=0.1) -> None:
        self.redis = redis
        self.runtime = runtime
        self.events = events
        self.instance = instance
        self.ack_delay_s = ack_delay_s
        self.ack = ack
        self.beat_s = beat_s
        self.total = 0
        self.non_continuable = 0
        self._applied: dict[str, int] = {}
        self._observed: dict[str, tuple[int, float]] = {}
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self.lock = threading.Lock()

    def set_inflight(self, total: int, non_continuable: int = 0) -> None:
        with self.lock:
            self.total, self.non_continuable = total, non_continuable
            self._write(POD, ack=None)
        self.events.add("gw_inflight", total, non_continuable)

    def _write(self, name: str, *, ack: tuple[int, bool] | None) -> None:
        now_ms = int(time.time() * 1000)
        pipe = self.redis.pipeline(transaction=True)
        # Go order: inflight of the pod BEFORE its seen field, same pipeline.
        pipe.hset(rediskeys.gw_inflight_key(name), self.instance, json.dumps(
            {"total": self.total, "non_continuable": self.non_continuable, "ts": now_ms}))
        pipe.expire(rediskeys.gw_inflight_key(name), 300)
        if ack is not None:
            pipe.hset(rediskeys.gw_seen_key(name), self.instance, json.dumps(
                {"gen": ack[0], "routable": ack[1], "ts": now_ms}))
            pipe.expire(rediskeys.gw_seen_key(name), 300)
        pipe.execute()

    def _run(self) -> None:
        while not self._stop.is_set():
            self.redis.zadd(rediskeys.GW_INSTANCES_KEY, {self.instance: int(time.time() * 1000)})
            state = self.runtime.route_state(POD)
            if state is not None and self.ack:
                gen, routable = state
                now = time.monotonic()
                seen = self._observed.get(POD)
                if seen is None or seen[0] != gen:
                    self._observed[POD] = (gen, now)  # informer event arrives
                elif self._applied.get(POD) != gen and now - seen[1] >= self.ack_delay_s:
                    with self.lock:
                        self._write(POD, ack=(gen, routable))
                    self._applied[POD] = gen
                    self.events.add("gw_ack", POD, gen, routable)
            self._stop.wait(self.beat_s)

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._thread.join(timeout=2)


# ------------------------------------------------------------------------- harness
def _primitive(redis, runtime, **overrides) -> SleepPrimitive:
    base = dict(
        ack_timeout_s=3.0,
        instance_staleness_s=2.0,
        no_plugin_grace_s=1.0,
        poll_interval_s=0.05,
        sleep_call_timeout_s=5.0,
        physical_confirm_timeout_s=3.0,
        vllm_sleep_mode_param="auto",  # probes GET /version over HTTP
        hard_cap_s=10.0,
    )
    base.update(overrides)
    return SleepPrimitive(
        runtime_ops=runtime,
        vllm_ops=VllmOps(timeout_s=2.0, max_attempts=1),
        policy=policy(**base),
        gateway=GatewayState(redis),
        journal=SleepJournal(redis),
        reservations=SleepReservations(redis),
        # Real time (conftest swaps DEFAULT_CLOCK for a virtual one): the plugin and
        # engine here are threads that only make progress in wall time.
        clock=sp.Clock(),
    )


@pytest.fixture()
def world(redis_client, monkeypatch):
    events = Events()
    snapshot = pod(POD, "m1", (0,), ip="127.0.0.1")
    runtime = RecordingRuntime([snapshot], events)
    with FakeEngine(events) as engine:
        monkeypatch.setattr(sp, "VLLM_PORT", engine.port)
        yield redis_client, runtime, engine, events, SleepTarget(binding_of(snapshot), "127.0.0.1")


def _hash_json(redis, key):
    return {k.decode(): json.loads(v) for k, v in redis.hgetall(key).items()}


# --------------------------------------------------------------------------- tests
def test_hide_ack_drain_sleep_ordering_over_real_redis_and_http(world):
    redis, runtime, engine, events, target = world
    engine.running = 2
    with FakePlugin(redis, runtime, events, ack_delay_s=0.3) as plugin:
        plugin.set_inflight(2)
        time.sleep(0.3)  # a few heartbeats: the SM must see the score ADVANCE

        def finish_requests():
            # both requests end ~0.8 s after the plugin acked the hide
            deadline = time.monotonic() + 5
            while time.monotonic() < deadline:
                if any(e[0] == "gw_ack" and e[3] is False for e in events.kinds()):
                    break
                time.sleep(0.02)
            time.sleep(0.8)
            engine.running = 0
            plugin.set_inflight(0)

        finisher = threading.Thread(target=finish_requests, daemon=True)
        finisher.start()
        [outcome] = _primitive(redis, runtime).sleep([target], path="scale_down")
        finisher.join(timeout=5)

    hide = events.index(lambda e: e[:3] == ("patch", POD, "hidden"))
    hidden_gen = events.kinds()[hide][3]
    ack = events.index(lambda e: e[0] == "gw_ack" and e[2] == hidden_gen and e[3] is False)
    drained = events.index(lambda e: e == ("gw_inflight", 0, 0))
    slept = events.index(lambda e: e[0] == "vllm_sleep")
    committed = events.index(lambda e: e[:3] == ("patch", POD, "sleeping"))
    assert hide < ack < drained < slept < committed, events.kinds()

    assert engine.sleep_calls == [{"mode": "wait", "hidden": "1", "running": 0}]
    assert engine.sleeping is True
    assert outcome["ack_mode"] == "plugin"
    assert outcome["drained"] is True and outcome["forced_abort"] is False
    assert outcome["ack_latency_ms"] >= 250  # waited for the (delayed) plugin ack
    # SM state in real Redis: counters, ack latency, journal and reservation cleared.
    stats = {k.decode(): int(v) for k, v in redis.hgetall(rediskeys.SM_SLEEP_STATS_KEY).items()}
    assert stats.get("sleeps_total") == 1 and "rollback_total" not in stats
    assert redis.llen(rediskeys.SM_SLEEP_ACK_LATENCY_KEY) == 1
    assert redis.hlen(rediskeys.SM_SLEEP_OPS_KEY) == 0
    assert redis.hlen(rediskeys.SM_SLEEP_RESERVATIONS_KEY) == 0
    # The plugin's contract keys as the SM read them (Go field names).
    assert _hash_json(redis, rediskeys.gw_seen_key(POD))[INSTANCE]["gen"] >= hidden_gen
    assert set(_hash_json(redis, rediskeys.gw_inflight_key(POD))[INSTANCE]) == {"total", "non_continuable", "ts"}


def test_missing_plugin_ack_rolls_back_without_touching_the_engine(world):
    redis, runtime, engine, events, target = world
    with FakePlugin(redis, runtime, events, ack=False):  # heartbeats, never acks
        time.sleep(0.3)
        with pytest.raises(GatewayAckTimeout, match=INSTANCE):
            _primitive(redis, runtime, ack_timeout_s=1.0).sleep([target], path="scale_down")

    patches = [e for e in events.kinds() if e[0] == "patch"]
    assert [p[2] for p in patches] == ["hidden", "awake"]
    assert patches[1][3] > patches[0][3]  # restored under a NEW route-gen
    assert engine.sleep_calls == [] and engine.sleeping is False
    stats = {k.decode(): int(v) for k, v in redis.hgetall(rediskeys.SM_SLEEP_STATS_KEY).items()}
    assert stats.get("ack_timeout_total") == 1 and stats.get("rollback_total") == 1
    assert redis.hlen(rediskeys.SM_SLEEP_OPS_KEY) == 0
    assert redis.hlen(rediskeys.SM_SLEEP_RESERVATIONS_KEY) == 0


def test_non_continuable_inflight_past_the_hard_cap_rolls_back(world):
    redis, runtime, engine, events, target = world
    engine.running = 1
    with FakePlugin(redis, runtime, events, ack_delay_s=0.1) as plugin:
        plugin.set_inflight(1, non_continuable=1)  # e.g. a beam-search request: never aborted
        time.sleep(0.3)
        with pytest.raises(SleepFailed):
            _primitive(redis, runtime, hard_cap_s=1.5, budgets_s={"scale_down": 0.3}).sleep(
                [target], path="scale_down"
            )

    kinds = events.kinds()
    assert any(e[0] == "gw_ack" and e[3] is False for e in kinds)  # it was acked ...
    patches = [e for e in kinds if e[0] == "patch"]
    assert [p[2] for p in patches] == ["hidden", "awake"]  # ... but never slept
    assert engine.sleep_calls == [] and engine.sleeping is False
    stats = {k.decode(): int(v) for k, v in redis.hgetall(rediskeys.SM_SLEEP_STATS_KEY).items()}
    assert stats.get("non_continuable_rollback_total") == 1
    assert "forced_abort_total" not in stats
    assert redis.hlen(rediskeys.SM_SLEEP_OPS_KEY) == 0
    assert redis.hlen(rediskeys.SM_SLEEP_RESERVATIONS_KEY) == 0
