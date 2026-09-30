"""End-to-end fake-environment test of the baseline shell, against a REAL Redis.

Runs under ``make check-redis`` (``TRE_TEST_REDIS_URL`` points at the throwaway Redis
container; every test id contains ``real-redis``); skipped in a plain ``make check``.

Fakes (nothing else is faked, the shell code is the production one):

* a stub service manager over HTTP (``bl_fakes.StubSM``): records ``PUT /v2/models/{m}/target``
  and serves ``GET /v2/state`` consistent with the targets it applied;
* one fake vLLM ``/metrics`` HTTP server per pod, rendered from the real 0.30 fixture, its
  gauges and cumulative counters advancing with the simulated load;
* a request-event generator XADDing ``tre:v2:bl:req:<model>`` entries in exactly the Go
  writer's field format (``pkg/plugins/gateway/tre_bl_req_events.go``: every value a
  string, kinds arr / ft / done, ``out_tokens=-1`` without usage, ``reissue``);
* pod discovery through ``LiveSource``'s ``list_pods`` injection point.

The shell runs in real time (tick 0.3 s, three 5 s load phases: low -> high -> low), the
clock is the Redis ``TIME`` of the real container.
"""
from __future__ import annotations

import json
import os
import re
import threading
import time
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Callable, Optional

import pytest

from bl_fakes import StubSM, limits, make_config, wait_until
from tre_baselines.keys import CONTROLLER_MODE_KEY, DECISIONS_STREAM, OWNER_KEY, REPLAY_T0_KEY, req_stream_key
from tre_baselines.loop import ACTIONS, BaselineShell, DecisionLog, OwnerLock
from tre_baselines.policies import build_policy
from tre_baselines.sm_client import Dispatcher, SMClient
from tre_baselines.sources import LiveSource, PodEndpoint
from tre_baselines.tools import arm

REDIS_URL = os.environ.get("TRE_TEST_REDIS_URL")
FIXTURE = Path(__file__).resolve().parent / "fixtures" / "vllm030_metrics.txt"

A, B = "model-a", "model-b"
MIN_R, MAX_R = 1, 3
TICK_S = 0.3
PHASE_S = 5.0


@pytest.fixture(params=[pytest.param(REDIS_URL, id="real-redis")])
def redis_url(request):
    if not request.param:
        pytest.skip("TRE_TEST_REDIS_URL not set (runs under make check-redis)")
    return request.param


@pytest.fixture
def redis(redis_url):
    import redis as redis_lib

    r = redis_lib.Redis.from_url(redis_url, decode_responses=True)
    _clean(r)
    yield r
    _clean(r)
    r.close()


def _clean(r) -> None:
    keys = r.keys("tre:v2:bl:*")
    if keys:
        r.delete(*keys)
    r.delete(CONTROLLER_MODE_KEY)  # missing = observe (the tests that need "active" set it)


# ------------------------------------------------------------------------ the simulated world


@dataclass(frozen=True)
class Load:
    rps: float   # request arrivals per second (events)
    conc: float  # requests in flight over all pods of the model (engine gauges)


LOW = Load(0.5, 0.3)
HIGH = Load(5.0, 12.0)


@dataclass
class Counters:
    gen: float = 0.0
    prompt: float = 0.0
    itl_sum: float = 0.0
    itl_cnt: float = 0.0
    ttft_sum: float = 0.0
    ttft_cnt: float = 0.0
    e2e_sum: float = 0.0
    e2e_cnt: float = 0.0


class World:
    """Awake counts (what the stub SM applied), the offered load and per-pod engine state."""

    CAP = 4.0          # running requests one pod holds before the rest waits
    TOK_PER_S = 30.0   # decode tokens/s of one running request
    ITL_S = 0.02

    def __init__(self, models, max_pods: int) -> None:
        self.lock = threading.RLock()
        self.models = list(models)
        self.max_pods = max_pods
        self.awake = {m: MIN_R for m in models}
        self.load = {m: LOW for m in models}
        self.version = 1
        self.counters = {(m, i): Counters() for m in models for i in range(max_pods)}

    @staticmethod
    def pod_name(model: str, i: int) -> str:
        return f"{model}-{i}"

    def awake_pods(self, model: str) -> list[str]:
        with self.lock:
            return [self.pod_name(model, i) for i in range(self.awake[model])]

    def state_doc(self) -> dict:
        with self.lock:
            return {
                "version": self.version,
                "models": {m: {"awake": self.awake[m], "bound": self.max_pods} for m in self.models},
                "bindings": [
                    {"serve_id": self.pod_name(m, i), "model": m, "node": "node-a", "gpu_ids": [i],
                     "awake": i < self.awake[m], "hidden": False}
                    for m in self.models for i in range(self.max_pods)
                ],
            }

    def apply_target(self, model: str, body: dict) -> None:
        with self.lock:
            target = min(int(body["wake_replicas"]), self.max_pods)
            self.awake[model] = max(self.awake[model], target) if body.get("at_least") else target
            self.version += 1

    def running_waiting(self, model: str, i: int) -> tuple[float, float]:
        with self.lock:
            if i >= self.awake[model]:
                return 0.0, 0.0
            per_pod = self.load[model].conc / self.awake[model]
            return min(per_pod, self.CAP), max(0.0, per_pod - self.CAP)

    def step(self, dt: float) -> None:
        with self.lock:
            for (m, i), c in self.counters.items():
                running, _ = self.running_waiting(m, i)
                tokens = running * self.TOK_PER_S * dt
                c.gen += tokens
                c.itl_cnt += tokens
                c.itl_sum += tokens * self.ITL_S
                c.prompt += running * 2.0 * dt * 100
                done = running * dt / 2.0
                c.ttft_cnt += done
                c.ttft_sum += done * 0.1
                c.e2e_cnt += done
                c.e2e_sum += done * 2.0

    def view(self, model: str, i: int) -> dict:
        with self.lock:
            running, waiting = self.running_waiting(model, i)
            c = self.counters[(model, i)]
            return {
                "vllm:num_requests_running": running,
                "vllm:num_requests_waiting": waiting,
                "vllm:kv_cache_usage_perc": min(1.0, running / self.CAP * 0.5),
                "vllm:prompt_tokens_total": c.prompt,
                "vllm:generation_tokens_total": c.gen,
                "vllm:inter_token_latency_seconds_sum": c.itl_sum,
                "vllm:inter_token_latency_seconds_count": c.itl_cnt,
                "vllm:time_to_first_token_seconds_sum": c.ttft_sum,
                "vllm:time_to_first_token_seconds_count": c.ttft_cnt,
                "vllm:e2e_request_latency_seconds_sum": c.e2e_sum,
                "vllm:e2e_request_latency_seconds_count": c.e2e_cnt,
            }


def render_metrics(template: list[str], view: dict, num_gpu_blocks: Optional[int]) -> str:
    out = []
    for line in template:
        if line.strip() and not line.startswith("#"):
            name = re.split(r"[{ ]", line, maxsplit=1)[0]
            if name in view:
                line = line[: line.rindex(" ") + 1] + repr(float(view[name]))
            elif name == "vllm:cache_config_info" and num_gpu_blocks:
                line = re.sub(r'num_gpu_blocks="\d+"', f'num_gpu_blocks="{num_gpu_blocks}"', line)
        out.append(line)
    return "\n".join(out) + "\n"


class MetricsFarm:
    """One fake vLLM ``/metrics`` HTTP server per pod (127.0.0.1, ephemeral ports)."""

    def __init__(self, world: World, num_gpu_blocks: Optional[int]) -> None:
        self._template = FIXTURE.read_text(encoding="utf-8").splitlines()
        self.servers: dict[str, ThreadingHTTPServer] = {}
        farm = self
        for m in world.models:
            for i in range(world.max_pods):
                farm.servers[world.pod_name(m, i)] = self._make(world, m, i, num_gpu_blocks)

    def _make(self, world: World, model: str, i: int, blocks: Optional[int]) -> ThreadingHTTPServer:
        farm = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):  # noqa: N802
                body = render_metrics(farm._template, world.view(model, i), blocks).encode()
                self.send_response(200)
                self.send_header("content-type", "text/plain; version=0.0.4")
                self.send_header("content-length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *args):
                pass

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        server.daemon_threads = True
        threading.Thread(target=server.serve_forever, daemon=True).start()
        return server

    def port(self, pod: str) -> int:
        return self.servers[pod].server_address[1]

    def close(self) -> None:
        for s in self.servers.values():
            s.shutdown()
            s.server_close()


#: Field names per event kind, in the Go writer's order (tre_bl_req_events.go).
GO_FIELDS = {
    "arr": ["kind", "req_id", "reissue", "gw_local_ms", "pod", "in_tokens", "in_src", "max_tokens", "stream"],
    "ft": ["kind", "req_id", "reissue", "gw_local_ms", "pod", "dt_arr_ms"],
    "done": ["kind", "req_id", "reissue", "gw_local_ms", "pod", "status", "out_tokens", "out_src", "dt_arr_ms"],
}


def go_event(kind: str, n: int, req_id: str, pod: str) -> dict[str, str]:
    """One event as the Go writer writes it: every value a string. ``n`` (request number)
    drives the deterministic mix of odd cases the shell must survive."""
    reissue = "continued" if n % 15 == 7 else "none"
    common = {"kind": kind, "req_id": req_id, "reissue": reissue,
              "gw_local_ms": str(int(time.time() * 1000)), "pod": pod}
    return _ordered(kind, _go_values(kind, n, common))


def _ordered(kind: str, values: dict[str, str]) -> dict[str, str]:
    assert sorted(values) == sorted(GO_FIELDS[kind])
    return {k: values[k] for k in GO_FIELDS[kind]}


def _go_values(kind: str, n: int, common: dict) -> dict[str, str]:
    if kind == "arr":
        return {**common,
                "in_tokens": "0" if n % 20 == 3 else "100",          # 0 = count failed
                "in_src": "header" if n % 2 else "token_ids",
                "max_tokens": "" if n % 10 == 5 else "100",           # "" = absent
                "stream": "false"}
    if kind == "ft":
        return {**common, "dt_arr_ms": "12"}
    usage = n % 4 != 0
    return {**common, "status": "200", "out_tokens": "100" if usage else "-1",
            "out_src": "usage" if usage else "none", "dt_arr_ms": "400"}


class EventGen(threading.Thread):
    """Poisson-free (fixed interval) arrivals per model at the phase's rps; each request
    writes arr now, ft 0.1 s later, done 0.4 s later, on a pod that is awake."""

    def __init__(self, redis, world: World) -> None:
        super().__init__(daemon=True)
        self._redis = redis
        self._world = world
        self._stop_evt = threading.Event()
        self.written: dict[str, int] = {"arr": 0, "ft": 0, "done": 0}
        self._n = 0

    def stop(self) -> None:
        self._stop_evt.set()
        self.join(5)

    def run(self) -> None:
        w = self._world
        due: list[tuple[float, str, dict]] = []
        next_arr = {m: time.monotonic() for m in w.models}
        while not self._stop_evt.is_set():
            now = time.monotonic()
            for m in w.models:
                interval = 1.0 / w.load[m].rps
                next_arr[m] = min(next_arr[m], now + interval)
                while next_arr[m] <= now:
                    next_arr[m] += interval
                    pods = w.awake_pods(m)
                    self._n += 1
                    n, pod = self._n, pods[self._n % len(pods)]
                    req = f"{m}-req-{n}"
                    for delay, kind in ((0.0, "arr"), (0.1, "ft"), (0.4, "done")):
                        due.append((now + delay, req_stream_key(m), go_event(kind, n, req, pod)))
            due.sort(key=lambda d: d[0])
            while due and due[0][0] <= now:
                _, key, fields = due.pop(0)
                self._redis.xadd(key, fields)
                self.written[fields["kind"]] += 1
            time.sleep(0.005)


# ------------------------------------------------------------------------ the harness


class Harness:
    def __init__(self, tmp_path, redis_url: str, redis, policy: str, params: dict, *, dry_run: bool,
                 hook: Optional[Callable[[int, str, dict], Optional[tuple]]] = None, sm_delay_s: float = 0.0,
                 num_gpu_blocks: Optional[int] = None) -> None:
        self.redis = redis
        self.redis_url = redis_url
        self.policy_name = policy
        self.world = World([A, B], MAX_R)
        self.puts: list[tuple[str, dict, int]] = []  # (model, body, http status)
        self._hook = hook
        self._put_seq = 0
        self.sm = StubSM(responder=self._responder, delay_s=sm_delay_s, state=self.world.state_doc())
        self.farm = MetricsFarm(self.world, num_gpu_blocks)
        self.config = make_config(
            tmp_path, {A: limits(A, MIN_R, MAX_R), B: limits(B, MIN_R, MAX_R)}, policy=policy,
            dry_run=dry_run, tick_s=TICK_S, sm_url=self.sm.url, redis_url=redis_url, policy_params=params,
            lock_ttl_s=5.0, max_tick_failures=3,
        )
        client = SMClient(self.sm.url, timeout_s=10.0)
        self.dispatcher = Dispatcher(client.put_target, sleep_path=self.config.sleep_path)
        self.source = LiveSource(self.config, redis, client.get_state, self._list_pods)
        self.shell = BaselineShell(
            self.config, self.source, build_policy(policy, self.config), self.dispatcher, redis,
            lock=OwnerLock(redis, self.config.lock_ttl_s), decision_log=DecisionLog(self.config.log_dir, policy),
        )
        self.events = EventGen(redis, self.world)
        self._sim_stop = threading.Event()
        self._shell_thread = threading.Thread(target=self.shell.run, daemon=True)
        self.marks: dict[str, int] = {}

    # stub SM: apply like the real one, then refresh the state it serves
    def _responder(self, model: str, body: dict):
        self._put_seq += 1
        override = self._hook(self._put_seq, model, body) if self._hook else None
        if override is not None:
            self.puts.append((model, body, override[0]))
            return override
        self.world.apply_target(model, body)
        self.sm.state = self.world.state_doc()
        self.puts.append((model, body, 200))
        return 200, json.dumps({"model": model}).encode(), "application/json"

    def _list_pods(self) -> list[PodEndpoint]:
        return [PodEndpoint(name=p, model=m, ip="127.0.0.1", port=self.farm.port(p), ready=True)
                for m in self.world.models for p in self.world.awake_pods(m)]

    def _simulate(self) -> None:
        last = time.monotonic()
        while not self._sim_stop.is_set():
            time.sleep(0.05)
            now = time.monotonic()
            self.world.step(now - last)
            last = now

    def now_ms(self) -> int:
        seconds, micros = self.redis.time()
        return int(seconds) * 1000 + int(micros) // 1000

    def __enter__(self) -> "Harness":
        self.sm.__enter__()
        threading.Thread(target=self._simulate, daemon=True).start()
        self.events.start()
        self._shell_thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self.shell.stop()
        self._shell_thread.join(10)
        self.events.stop()
        self._sim_stop.set()
        self.dispatcher.close()
        self.source.close()
        self.farm.close()
        self.sm.__exit__()

    def phase(self, name: str, seconds: float, loads: dict) -> None:
        self.marks[name] = self.now_ms()
        with self.world.lock:
            self.world.load.update(loads)
        time.sleep(seconds)

    def profile(self) -> None:
        """low -> high -> low; model B stays low."""
        self.phase("low1", PHASE_S, {A: LOW, B: LOW})
        self.phase("high", PHASE_S, {A: HIGH, B: LOW})
        self.phase("low2", PHASE_S, {A: LOW, B: LOW})
        time.sleep(1.0)  # let the last SM call finish and the next tick log its result

    def lines(self) -> list[dict]:
        out = []
        for path in sorted(Path(self.config.log_dir).glob(f"decisions-{self.policy_name}-*.jsonl")):
            out.extend(json.loads(ln) for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip())
        return out


# ------------------------------------------------------------------------ scenarios / params


def _policy_params(policy: str, tmp_path) -> dict:
    if policy == "chiron":
        return {"b_init": 4, "b_max": 4, "theta": {"*": 0.5}}
    if policy == "tokenscale":
        vel = {"buckets": [[200.0] * 3] * 3, "v_prefill": 1000.0}
        return {"models": [A, B], "bucket_edges": {"*": {"in": [50, 150], "out": [50, 150]}},
                "velocity": {A: vel, B: vel}, "misbucket_rate": 0.0, "window_s": 3, "seed": 0}
    if policy == "preserve":
        # Windows of PHASE_S s: w0 low, w1 high (N=3), w2 low. mu in tok/s per replica.
        reqs = [{"timestamp": t, "model_name": A, "prompt_length": 50, "max_output_tokens": 50}
                for t in (1.0, 2.0, 3.0, 4.0)]
        reqs += [{"timestamp": 5.5 + k, "model_name": A, "prompt_length": 250, "max_output_tokens": 250}
                 for k in range(5)]
        reqs += [{"timestamp": t, "model_name": A, "prompt_length": 50, "max_output_tokens": 50}
                 for t in (11.0, 12.0, 13.0, 14.0)]
        path = tmp_path / "traces" / "e2e" / "trace.json"
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps(reqs), encoding="utf-8")
        mu = {"p": 100.0, "d": 100.0, "t": 200.0}
        return {"trace_path": str(path), "window_s": PHASE_S, "tier1": "oracle", "mu": {A: mu, B: dict(mu)}}
    raise AssertionError(policy)


def _mark_replay(h: Harness, trace: str) -> None:
    """The campaign's replay marker, written with the arm tool through the real Redis."""
    assert arm.main(["mark-replay", "--trace", trace, "--seed", "0", "--execute"],
                    redis=arm.DirectRedis(h.redis_url)) == 0


REQUIRED = {"ts_ms", "tick", "policy", "model", "awake", "raw_desired", "clamped", "direction", "reason",
            "inputs", "action", "dry_run", "owner", "inflight"}


def _check_lines(lines: list[dict], policy: str) -> None:
    assert lines, "no decision lines"
    for ln in lines:
        assert REQUIRED <= set(ln), REQUIRED - set(ln)
        assert ln["policy"] == policy and ln["model"] in (A, B)
        assert MIN_R <= ln["clamped"] <= MAX_R and MIN_R <= ln["awake"] <= MAX_R
        assert ln["action"] in ACTIONS
        assert isinstance(ln["inputs"], dict)
        json.dumps(ln)
    assert len({(ln["model"], ln["tick"]) for ln in lines}) == len(lines)  # one line per model per tick


# ------------------------------------------------------------------------ tests


def test_go_event_format_in_real_redis(redis_url, redis) -> None:
    """The generator's entries carry exactly the Go writer's fields, all strings, and parse."""
    from tre_baselines.sources import EventReader

    reader = EventReader(redis, [A])
    reader.start(0)
    for n, kind in ((3, "arr"), (4, "done"), (7, "ft"), (5, "arr")):
        redis.xadd(req_stream_key(A), go_event(kind, n, f"r{n}", "pod-0"))
    entries = redis.xrange(req_stream_key(A))
    for _, fields in entries:
        assert list(fields) == GO_FIELDS[fields["kind"]]
        assert all(isinstance(v, str) for v in fields.values())
    got = reader.read(10**15)[A]
    assert [e.kind for e in got] == ["arr", "done", "ft", "arr"]
    arr0, done, ft, arr1 = got
    assert arr0.in_tokens is None and arr0.max_tokens == 100 and arr0.in_src == "header"   # in_tokens "0"
    assert done.out_tokens is None and done.status == "200"                               # out_tokens "-1"
    assert ft.reissue == "continued" and arr1.max_tokens is None                          # max_tokens ""


@pytest.mark.parametrize("policy", ["chiron", "tokenscale", "preserve"])
def test_low_high_low_scales_up_then_down(tmp_path, redis_url, redis, policy) -> None:
    params = _policy_params(policy, tmp_path)
    with Harness(tmp_path, redis_url, redis, policy, params, dry_run=False) as h:
        if policy == "preserve":
            _mark_replay(h, params["trace_path"])
            assert json.loads(redis.get(REPLAY_T0_KEY))["seed"] == 0
        h.profile()
        assert h.shell.healthy()
    lines = h.lines()
    _check_lines(lines, policy)
    mine = [ln for ln in lines if ln["model"] == A]
    other = [ln for ln in lines if ln["model"] == B]
    high, low2 = h.marks["high"], h.marks["low2"]
    # up during the high phase, down after the load dropped
    assert any(ln["action"] == "up" and high <= ln["ts_ms"] < low2 for ln in mine), [
        (ln["ts_ms"] - high, ln["action"], ln["reason"]) for ln in mine if ln["action"] != "none"]
    assert any(ln["action"] == "down" and ln["ts_ms"] >= low2 - 500 for ln in mine), [
        (ln["ts_ms"] - high, ln["action"], ln["reason"]) for ln in mine if ln["action"] != "none"]
    assert max(ln["awake"] for ln in mine) >= 2
    assert not any(ln["action"] == "up" for ln in other)          # the quiet model is left alone
    assert all(ln["owner"] and not ln["dry_run"] for ln in lines)
    # every dispatched action reached the stub SM once, with the contract's body
    dispatched = [ln for ln in lines if ln["action"] in ("up", "down")]
    assert len(h.puts) == len(dispatched)
    for model, body, status in h.puts:
        assert 1 <= body["wake_replicas"] <= MAX_R and status == 200
        assert body.get("at_least") is True or body.get("sleep_path") == "scale_down"
    assert any(b.get("at_least") for _, b, _ in h.puts) and any(not b.get("at_least") for _, b, _ in h.puts)
    # the decision is mirrored to Redis (TTL) and the stream saw all three kinds
    assert redis.ttl(f"tre:v2:bl:decision:{A}") > 0
    # every decision line is also in the stream (evidence that survives the pod)
    streamed = [json.loads(fields["line"]) for _, fields in redis.xrange(DECISIONS_STREAM)]
    assert len(streamed) == len(lines) and streamed[-1] == lines[-1]
    assert all(h.events.written[k] > 0 for k in ("arr", "ft", "done"))
    if policy == "tokenscale":
        # Go-format events reached the policy: lambda > 0, odd events were counted, not fatal
        assert any(ln["inputs"].get("lambda_in", 0) > 0 and ln["inputs"].get("new_events", 0) > 0 for ln in mine)
        last = mine[-1]["inputs"]
        assert last["skipped_reissue"] > 0 and last["skipped_missing_in"] > 0 and last["missing_out"] >= 0
        assert any(ln["inputs"].get("missing_out", 0) > 0 for ln in mine)
    if policy == "chiron":
        assert any(ln["inputs"].get("IBP") == 1.0 for ln in mine)
    if policy == "preserve":
        reasons = {ln["reason"] for ln in mine}
        assert any("tier1_window" in r for r in reasons), reasons
        assert any(ln["inputs"].get("tier2") for ln in mine)  # Tier-2 fed by the events


@pytest.mark.parametrize("policy", ["chiron", "tokenscale"])
def test_dry_run_sends_nothing(tmp_path, redis_url, redis, policy) -> None:
    params = _policy_params(policy, tmp_path)
    with Harness(tmp_path, redis_url, redis, policy, params, dry_run=True) as h:
        h.phase("high", 4.0, {A: HIGH, B: LOW})
        assert h.shell.healthy()
    lines = h.lines()
    _check_lines(lines, policy)
    assert h.sm.requests == [] and h.puts == []
    assert h.world.awake == {A: MIN_R, B: MIN_R}
    assert any(ln["action"] == "dry_run" for ln in lines if ln["model"] == A)
    assert all(ln["dry_run"] and not ln["owner"] for ln in lines)
    assert redis.get(OWNER_KEY) is None  # a dry-run shell never takes the actuation lock


def test_sm_refusals_do_not_stop_the_loop(tmp_path, redis_url, redis) -> None:
    def hook(seq: int, model: str, body: dict):
        if seq == 1:  # structured (T1-style) refusal
            payload = {"detail": {"error": "gpu_busy", "reason": "no_free_gpu", "node": "node-a",
                                  "gpu_ids": [0], "retry_after_s": 2}}
            return 409, json.dumps(payload).encode(), "application/json"
        if seq == 2:  # plain-text detail
            return 409, json.dumps({"detail": "writer lock busy"}).encode(), "application/json"
        return None

    params = _policy_params("chiron", tmp_path)
    with Harness(tmp_path, redis_url, redis, "chiron", params, dry_run=False, hook=hook) as h:
        # refusal 1 (retry_after 2 s) then refusal 2 (backoff 4 s): the third call ~7 s in
        h.phase("high", 10.0, {A: HIGH, B: LOW})
        assert h.shell.healthy()
        ticks_before_stop = h.shell.stats.ticks
    lines = h.lines()
    _check_lines(lines, "chiron")
    results = [ln["sm_result"] for ln in lines if "sm_result" in ln]
    assert results[0]["ok"] is False and results[0]["code"] == 409 and results[0]["error"] == "gpu_busy"
    assert results[0]["reason"] == "no_free_gpu" and results[0]["retry_after_s"] == 2.0
    assert results[1]["ok"] is False and results[1]["code"] == 409 and results[1]["error"] == "http_error"
    assert results[1]["detail"] == "writer lock busy"
    assert any(r["ok"] for r in results[2:])                       # it kept trying and got through
    mine = [ln for ln in lines if ln["model"] == A]
    assert any(ln["action"] == "backoff" for ln in mine)            # but not every tick
    a_puts = [p for p in h.puts if p[0] == A]
    assert len(a_puts) < sum(ln["direction"] == "up" for ln in mine)
    assert max(ln["awake"] for ln in lines if ln["model"] == A) >= 2
    assert ticks_before_stop >= 12 and h.shell.stats.tick_failures == 0
    assert h.shell.stats.sm_failures == 2


def test_slow_sm_call_is_not_duplicated(tmp_path, redis_url, redis) -> None:
    delay = 3 * TICK_S  # a call spanning more than two ticks
    params = _policy_params("chiron", tmp_path)
    with Harness(tmp_path, redis_url, redis, "chiron", params, dry_run=False, sm_delay_s=delay) as h:
        h.phase("high", 5.0, {A: HIGH, B: LOW})
        time.sleep(delay + 0.5)
        wait_until(lambda: h.dispatcher.inflight_count() == 0, 5)
    lines = h.lines()
    _check_lines(lines, "chiron")
    mine = [ln for ln in lines if ln["model"] == A]
    assert any(ln["action"] == "inflight_skip" for ln in mine)     # the model wanted more while busy
    arrivals = h.sm.arrivals
    assert len(arrivals) == len(h.puts) >= 2
    assert len(arrivals) == sum(ln["action"] in ("up", "down") for ln in lines)
    a_puts = [t for t, (m, _, _) in zip(arrivals, h.puts) if m == A]
    assert all(b - a >= delay * 0.9 for a, b in zip(a_puts, a_puts[1:]))  # never two calls at once


def test_preserve_tier2_overload_scales_up(tmp_path, redis_url, redis) -> None:
    """No replay marker (Tier-1 inactive); a tiny KV capacity makes the look-ahead maps
    overload under the high phase, which must add instances."""
    params = _policy_params("preserve", tmp_path)
    params["down_grace_s"] = 3600
    with Harness(tmp_path, redis_url, redis, "preserve", params, dry_run=False, num_gpu_blocks=12) as h:
        h.phase("high", 6.0, {A: HIGH, B: LOW})
        assert h.shell.healthy()
    lines = h.lines()
    _check_lines(lines, "preserve")
    mine = [ln for ln in lines if ln["model"] == A]
    assert any(ln["inputs"]["tier1"].get("inactive") == "tier1_no_replay" for ln in mine)
    assert any("tier2_overload" in ln["reason"] for ln in mine)
    assert any(ln["action"] == "up" for ln in mine) and max(ln["awake"] for ln in mine) >= 2


def test_owner_lock_scripts_in_real_redis(redis_url, redis) -> None:
    a, b = OwnerLock(redis, 5.0, token="a"), OwnerLock(redis, 5.0, token="b")
    assert a.ensure() and not b.ensure()
    assert 0 < redis.pttl(OWNER_KEY) <= 5000
    redis.pexpire(OWNER_KEY, 100)
    assert a.ensure() and redis.pttl(OWNER_KEY) > 1000          # renewed (compare-and-pexpire)
    redis.set(OWNER_KEY, "b", px=3000)                           # expired and taken over
    assert not a.ensure() and redis.get(OWNER_KEY) == "b" and redis.pttl(OWNER_KEY) <= 3000
    a.release()                                                  # compare-and-delete: not ours
    assert redis.get(OWNER_KEY) == "b"
    b.release()
    assert redis.get(OWNER_KEY) is None


def test_controller_active_suspends_actuation(tmp_path, redis_url, redis) -> None:
    redis.set(CONTROLLER_MODE_KEY, "active")
    params = _policy_params("chiron", tmp_path)
    with Harness(tmp_path, redis_url, redis, "chiron", params, dry_run=False) as h:
        h.phase("high", 3.0, {A: HIGH, B: LOW})
        assert h.shell.healthy()
    lines = h.lines()
    _check_lines(lines, "chiron")
    assert h.puts == [] and h.world.awake == {A: MIN_R, B: MIN_R}
    assert "guard_controller_active" in [ln["action"] for ln in lines if ln["model"] == A]
    assert all(ln["dry_run"] and ln["controller_mode"] == "active" for ln in lines)
