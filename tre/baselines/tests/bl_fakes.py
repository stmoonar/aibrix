"""Test doubles for the baseline shell: in-memory Redis, a stub SM HTTP server, a fake
cluster that applies target calls, and a scripted policy. No network beyond 127.0.0.1."""
from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Callable, Mapping, Optional

from tre_baselines.config import Config, ModelLimits
from tre_baselines.policies.base import Decision
from tre_baselines.sm_client import SMResult
from tre_baselines.snapshot import ClusterSnapshot, ModelSnapshot


class FakeRedis:
    """Just the commands the shell uses (decode_responses=True semantics)."""

    def __init__(self, now_ms: int = 1_700_000_000_000) -> None:
        self.now_ms = now_ms
        self.kv: dict[str, str] = {}
        self.ttl_ms: dict[str, int] = {}
        self.streams: dict[str, list[tuple[str, dict]]] = {}
        self._seq: dict[int, int] = {}
        self.set_calls: list[tuple[str, Any, dict]] = []
        self.eval_calls: list[tuple] = []
        self.xadd_calls: list[tuple] = []

    def time(self):
        return (self.now_ms // 1000, (self.now_ms % 1000) * 1000)

    def advance(self, ms: int) -> None:
        self.now_ms += ms

    def get(self, key):
        return self.kv.get(key)

    def set(self, key, value, nx=False, px=None, ex=None, xx=False):
        self.set_calls.append((key, value, {"nx": nx, "px": px, "ex": ex}))
        if nx and key in self.kv:
            return None
        if xx and key not in self.kv:
            return None
        self.kv[key] = value
        if px is not None:
            self.ttl_ms[key] = int(px)
        elif ex is not None:
            self.ttl_ms[key] = int(ex) * 1000
        return True

    def pexpire(self, key, ms):
        if key not in self.kv:
            return False
        self.ttl_ms[key] = int(ms)
        return True

    def delete(self, key):
        existed = key in self.kv
        self.kv.pop(key, None)
        self.ttl_ms.pop(key, None)
        return int(existed)

    def eval(self, script, numkeys, *args):
        """The owner-lock scripts of ``tre_baselines.loop`` (compare-and-pexpire / -delete)."""
        from tre_baselines.loop import RELEASE_LUA, RENEW_LUA

        self.eval_calls.append((script, numkeys, args))
        key, token = args[0], args[1]
        if self.kv.get(key) != token:
            return 0
        if script == RENEW_LUA:
            return int(self.pexpire(key, int(args[2])))
        if script == RELEASE_LUA:
            return self.delete(key)
        raise NotImplementedError(script)

    def xadd(self, key, fields, id: Optional[str] = None, maxlen=None, approximate=True):
        self.xadd_calls.append((key, maxlen, approximate))
        if id is None or id == "*":
            seq = self._seq.get(self.now_ms, 0)
            self._seq[self.now_ms] = seq + 1
            id = f"{self.now_ms}-{seq}"
        self.streams.setdefault(key, []).append((id, dict(fields)))
        return id

    @staticmethod
    def _id_key(entry_id: str) -> tuple[int, int]:
        ms, _, seq = entry_id.partition("-")
        return int(ms), int(seq or 0)

    def xrange(self, key, min="-", max="+", count=None):
        entries = list(self.streams.get(key, []))
        return entries[:count] if count is not None else entries

    def xread(self, streams: Mapping[str, str], count: Optional[int] = None, block=None):
        out = []
        for key, cursor in streams.items():
            entries = [e for e in self.streams.get(key, []) if self._id_key(e[0]) > self._id_key(cursor)]
            if count is not None:
                entries = entries[:count]
            if entries:
                out.append([key, entries])
        return out


@dataclass
class FakeCluster:
    """Awake counts per model; ``put_target`` applies a target body like the SM does."""

    awake: dict[str, int]
    calls: list[tuple[str, dict]] = field(default_factory=list)
    delay_s: float = 0.0
    refuse: set = field(default_factory=set)
    lock: threading.Lock = field(default_factory=threading.Lock)
    #: ``/v2/state`` version: bumped by every applied target.
    version: int = 1
    #: (model, "start" | "end") in the order the calls ran.
    trace: list = field(default_factory=list)

    def put_target(self, model: str, body: Mapping[str, Any]) -> SMResult:
        with self.lock:
            self.calls.append((model, dict(body)))
            self.trace.append((model, "start"))
        if self.delay_s:
            time.sleep(self.delay_s)
        if model in self.refuse:
            with self.lock:
                self.trace.append((model, "end"))
            return SMResult(ok=False, code=409, error="http_error", detail="refused")
        with self.lock:
            target = int(body["wake_replicas"])
            if body.get("at_least"):
                self.awake[model] = max(self.awake[model], target)
            else:
                self.awake[model] = target
            self.version += 1
            self.trace.append((model, "end"))
        return SMResult(ok=True, code=200, raw={"model": model})


def limits(name: str, lo: int = 1, hi: int = 4, tp: int = 1) -> ModelLimits:
    return ModelLimits(name=name, min_replicas=lo, max_replicas=hi, gpus_per_replica=tp,
                       ttft_slo_ms=500.0, tpot_slo_ms=75.0, max_num_seqs=256)


def make_config(tmp_path, models: Mapping[str, ModelLimits], **overrides) -> Config:
    base = dict(
        sm_url="http://127.0.0.1:1", redis_url="redis://127.0.0.1:1/0", policy="scripted",
        dry_run=True, tick_s=0.01, log_dir=str(tmp_path / "logs"), models=dict(models),
        max_tick_failures=3, lock_ttl_s=30.0,
    )
    base.update(overrides)
    return Config(**base)


class FakeSource:
    """Snapshot from a FakeCluster's awake counts and a FakeRedis clock (+2 s per gather)."""

    def __init__(self, config: Config, cluster: FakeCluster, redis: FakeRedis, *, step_ms: int = 2000,
                 fail: Callable[[int], bool] = lambda tick: False) -> None:
        self.config = config
        self.cluster = cluster
        self.redis = redis
        self.step_ms = step_ms
        self.fail = fail
        self.gathers = 0

    def gather(self, tick: int = 0) -> ClusterSnapshot:
        self.gathers += 1
        self.redis.advance(self.step_ms)
        if self.fail(tick):
            raise RuntimeError(f"source failure at tick {tick}")
        with self.cluster.lock:
            awake = dict(self.cluster.awake)
            version = self.cluster.version
        models = {
            name: ModelSnapshot(
                model=name, awake=awake[name], min_replicas=lim.min_replicas,
                max_replicas=lim.max_replicas, gpus_per_replica=lim.gpus_per_replica,
                ttft_slo_ms=lim.ttft_slo_ms, tpot_slo_ms=lim.tpot_slo_ms,
                max_num_seqs=lim.max_num_seqs, pods=(), events=(),
            )
            for name, lim in sorted(self.config.models.items())
        }
        return ClusterSnapshot(now_ms=self.redis.now_ms, tick_s=self.config.tick_s, models=models,
                               tick=tick, extra={"event_lag_s": {}, "scrape_failed": 0,
                                                 "sm_state_version": version})


class ScriptedPolicy:
    """``script[tick][model] = desired`` (missing tick/model = no decision)."""

    name = "scripted"

    def __init__(self, script: Mapping[int, Mapping[str, int]]) -> None:
        self.script = script
        self.seen: list[int] = []

    def decide(self, snap: ClusterSnapshot) -> Mapping[str, Decision]:
        self.seen.append(snap.tick)
        return {m: Decision(desired=d, reason="scripted", inputs={"tick": snap.tick})
                for m, d in self.script.get(snap.tick, {}).items()}


class StubSM:
    """Stub SM HTTP server on 127.0.0.1: records PUT bodies; the reply per call comes from
    ``responder(model, body) -> (status, body_bytes, content_type)``; optional delay."""

    def __init__(self, responder=None, delay_s: float = 0.0,
                 state: Optional[dict] = None) -> None:
        self.requests: list[tuple[str, str, dict, dict]] = []
        #: time.monotonic() at which each PUT arrived (same order as ``requests``).
        self.arrivals: list[float] = []
        self.responder = responder or (lambda model, body: (200, json.dumps({"model": model}).encode(), "application/json"))
        self.delay_s = delay_s
        self.state = state or {"version": 1, "models": {}, "bindings": []}
        stub = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):  # noqa: N802
                if self.path == "/v2/state":
                    self._send(200, json.dumps(stub.state).encode(), "application/json")
                else:
                    self._send(404, b"{}", "application/json")

            def do_PUT(self):  # noqa: N802
                length = int(self.headers.get("content-length") or 0)
                body = json.loads(self.rfile.read(length) or b"{}")
                model = self.path.split("/")[3] if self.path.startswith("/v2/models/") else ""
                stub.arrivals.append(time.monotonic())
                stub.requests.append(("PUT", self.path, body, {k.lower(): v for k, v in self.headers.items()}))
                if stub.delay_s:
                    time.sleep(stub.delay_s)
                status, payload, ctype = stub.responder(model, body)
                try:
                    self._send(status, payload, ctype)
                except (BrokenPipeError, ConnectionResetError):
                    pass

            def _send(self, code, payload, ctype):
                self.send_response(code)
                self.send_header("content-type", ctype)
                self.send_header("content-length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

            def log_message(self, *args):
                pass

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.server.daemon_threads = True
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}"
        self._thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    def __enter__(self) -> "StubSM":
        self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self.server.shutdown()
        self.server.server_close()


def wait_until(pred: Callable[[], bool], timeout_s: float = 5.0) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if pred():
            return True
        time.sleep(0.005)
    return pred()
