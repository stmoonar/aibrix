"""Test doubles for the baseline shell: in-memory Redis, a stub SM HTTP server, a fake
cluster that applies plans (``POST /v2/plans``), and a scripted policy. No network beyond
127.0.0.1."""
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


def error_body(result: SMResult) -> dict:
    """The structured 409 body an SM item ``error`` carries, from an :class:`SMResult`."""
    body = {"detail": result.detail, "error": result.error, "reason": result.reason, "node": result.node,
            "gpu_ids": list(result.gpu_ids) if result.gpu_ids is not None else None, "scope": result.scope,
            "retry_after_s": result.retry_after_s, "blocking_binding_id": result.blocking_binding_id}
    return {k: v for k, v in body.items() if v is not None}


#: The refusal of a wake the fake cluster cannot place (``FakeCluster.place_limit``).
SLOT_OCCUPIED = SMResult(ok=False, code=409, error="gpu_busy", reason="slot_occupied", node="n", gpu_ids=(0,),
                         scope="gpu", retry_after_s=30.0, blocking_binding_id="x/n/0")


@dataclass
class FakeCluster:
    """Awake counts per model; ``post_plan`` applies a plan like the SM does (one item per
    model: wake ``at_least`` grow-only, sleep ``at_most``) and answers with its items."""

    awake: dict[str, int]
    #: (model, item) per plan item, in the order the plans ran.
    calls: list[tuple[str, dict]] = field(default_factory=list)
    #: Every plan body received.
    plans: list[dict] = field(default_factory=list)
    delay_s: float = 0.0
    refuse: set = field(default_factory=set)
    #: ``hook(model, item)`` -> an SMResult refusal for that item (its fields become the
    #: item's ``error``), or None to apply it.
    hook: Optional[Callable[[str, dict], Optional[SMResult]]] = None
    #: Per model, the most replicas the fake can place; a wake past it is ``unfilled``
    #: (``partial`` when some woke, else ``refused``), located by ``SLOT_OCCUPIED``.
    place_limit: dict = field(default_factory=dict)
    #: Set to hold every plan until it is set again (a plan in flight).
    gate: Optional[threading.Event] = None
    lock: threading.Lock = field(default_factory=threading.Lock)
    #: ``/v2/state`` version: bumped by every change.
    version: int = 1
    #: ("plan", "start" | "end") in the order the plans ran.
    trace: list = field(default_factory=list)
    in_flight: int = 0
    max_in_flight: int = 0

    def post_plan(self, body: Mapping[str, Any]) -> SMResult:
        with self.lock:
            self.plans.append(json.loads(json.dumps(body)))
            self.trace.append(("plan", "start"))
            self.in_flight += 1
            self.max_in_flight = max(self.max_in_flight, self.in_flight)
        if self.gate is not None:
            self.gate.wait(5.0)
        if self.delay_s:
            time.sleep(self.delay_s)
        out: dict[str, list] = {"sleeps": [], "wakes": []}
        changed = problems = False
        for kind in ("sleeps", "wakes"):
            for index, item in enumerate(body.get(kind) or ()):
                model = item["model"]
                refusal = self.hook(model, item) if self.hook else None
                if refusal is None and model in self.refuse:
                    refusal = SMResult(ok=False, code=409, error="http_error", detail="refused")
                with self.lock:
                    self.calls.append((model, dict(item)))
                    before = self.awake[model]
                    entry: dict[str, Any] = {"index": index, "model": model}
                    if refusal is not None:
                        after = before
                        entry.update(status="refused", error=error_body(refusal))
                    elif kind == "wakes":
                        want = int(item["at_least"])
                        after = max(before, min(want, int(self.place_limit.get(model, want))))
                        unfilled = max(0, want - after)
                        entry.update(status="noop" if want <= before else
                                     "done" if not unfilled else "partial" if after > before else "refused",
                                     unfilled=unfilled, refusals=[error_body(SLOT_OCCUPIED)] if unfilled else [])
                        if unfilled:
                            entry["error"] = error_body(SLOT_OCCUPIED)
                    else:
                        after = min(before, int(item["at_most"]))
                        entry.update(status="done" if after < before else "noop", taken=before - after,
                                     clamped_by_floor=False)
                    if after != before:
                        self.awake[model] = after
                        self.version += 1
                        changed = True
                    if kind == "wakes":
                        entry.update(awake_before=before, awake_after=after,
                                     woke=[{"serve_id": f"{model}-{i}"} for i in range(before, after)])
                    problems = problems or entry["status"] in ("refused", "failed")
                out[kind].append(entry)
        with self.lock:
            self.trace.append(("plan", "end"))
            self.in_flight -= 1
        status = ("failed" if problems else "noop") if not changed else ("partial" if problems else "done")
        response = {"plan_id": body.get("plan_id"), "version": self.version, "status": status, **out}
        if status == "failed":  # 409 partial: nothing changed, an item refused (full body)
            return SMResult(ok=False, code=409, error="partial", reason="partial",
                            raw={**response, "error": "partial", "reason": "partial"})
        return SMResult(ok=True, code=200, raw=response)


def plan_done(body: Mapping[str, Any]) -> dict:
    """A 200 plan answer with every item ``done`` (the stub SM's default)."""
    return {"plan_id": body.get("plan_id"), "status": "done",
            "sleeps": [{"index": i, "model": it["model"], "status": "done", "taken": 1}
                       for i, it in enumerate(body.get("sleeps") or ())],
            "wakes": [{"index": i, "model": it["model"], "status": "done", "unfilled": 0, "refusals": []}
                      for i, it in enumerate(body.get("wakes") or ())]}


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
    """Stub SM HTTP server on 127.0.0.1: records ``POST /v2/plans`` bodies; the reply per call
    comes from ``responder(body) -> (status, body_bytes, content_type)`` (default: every item
    done); optional delay."""

    def __init__(self, responder=None, delay_s: float = 0.0,
                 state: Optional[dict] = None) -> None:
        self.requests: list[tuple[str, str, dict, dict]] = []
        #: time.monotonic() at which each POST arrived (same order as ``requests``).
        self.arrivals: list[float] = []
        self.responder = responder or (lambda body: (200, json.dumps(plan_done(body)).encode(), "application/json"))
        self.delay_s = delay_s
        self.state = state or {"version": 1, "models": {}, "bindings": []}
        stub = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):  # noqa: N802
                if self.path == "/v2/state":
                    self._send(200, json.dumps(stub.state).encode(), "application/json")
                else:
                    self._send(404, b"{}", "application/json")

            def do_POST(self):  # noqa: N802
                length = int(self.headers.get("content-length") or 0)
                body = json.loads(self.rfile.read(length) or b"{}")
                stub.arrivals.append(time.monotonic())
                stub.requests.append(("POST", self.path, body, {k.lower(): v for k, v in self.headers.items()}))
                if self.path != "/v2/plans":
                    self._send(404, b"{}", "application/json")
                    return
                if stub.delay_s:
                    time.sleep(stub.delay_s)
                status, payload, ctype = stub.responder(body)
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
