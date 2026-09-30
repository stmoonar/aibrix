"""Shared fakes for the transparent-sleep tests (fake Redis, k8s, vLLM, gateway)."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import replace
import json

from tre_common import rediskeys
from tre_common.registry import (
    ClusterTopology,
    ModelSpec,
    NodeSpec,
    Registry,
    ServiceManagerConfig,
    SleepPolicy,
    SloSpec,
    TrsParams,
)
from tre_sm.allocator.slots import Binding, Slot
from tre_sm.allocator.topology import GPU_IDS_ANNOTATION, STATE_ANNOTATION, K8sPodSnapshot
from tre_sm.ops.k8s_ops import ModelDeploymentRecord, StartupPodRecord
from tre_sm.state.fleet_store import _SAVE_HASH_SCRIPT
from tre_sm.state.operations import WriterFence, _CURRENT_FENCE
from tre_sm.state import sleep_reservations as _res
from tre_sm.state import gpu_leases as _leases
from tre_sm.state import safety as _safety


def _b(value) -> bytes:
    return value if isinstance(value, bytes) else str(value).encode("utf-8")


def _s(value) -> str:
    return value.decode("utf-8") if isinstance(value, bytes) else str(value)


_MAINTENANCE_SCRIPTS = (
    _safety._MAINTENANCE_ACQUIRE_SCRIPT,
    _safety._MAINTENANCE_RENEW_SCRIPT,
    _safety._MAINTENANCE_RELEASE_SCRIPT,
)


def maintenance_lua(values: dict, ttls_ms: dict, script, keys_and_args):
    """Python model of the maintenance-lock Lua scripts (same argument layout).

    ``values`` holds the string keys, ``ttls_ms`` the PX of keys set with a
    TTL (a key missing there has none, like a pre-TTL lock or a hand-set key).
    Returns NotImplemented for any other script."""
    if script not in _MAINTENANCE_SCRIPTS:
        return NotImplemented
    key, *args = [_s(item) for item in keys_and_args]
    raw = values.get(key)
    cur = None if raw is None else _s(raw)
    try:
        rec = json.loads(cur) if cur is not None else None
    except ValueError:
        rec = None
    holder = rec.get("operation_id") if isinstance(rec, dict) else None
    if script == _safety._MAINTENANCE_ACQUIRE_SCRIPT:
        value, ttl_ms, *allowed = args
        if cur is not None:
            takeover = key not in ttls_ms or not isinstance(rec, dict) or holder in allowed
            if not takeover:
                return [0, _b(cur)]
        values[key] = value
        ttls_ms[key] = int(ttl_ms)
        return [2, _b(cur)] if cur is not None else [1, b""]
    if cur is None or not isinstance(rec, dict) or holder != args[0]:
        return 0
    if script == _safety._MAINTENANCE_RENEW_SCRIPT:
        ttls_ms[key] = int(args[1])
        return 1
    values.pop(key, None)
    ttls_ms.pop(key, None)
    return 1


class FakeRedis:
    """In-memory Redis with the commands the SM uses (plus the fleet-state Lua script)."""

    def __init__(self, *, now_ms: int = 1_700_000_000_000) -> None:
        self.values: dict[str, str] = {}
        self.hashes: dict[str, dict[str, str]] = {}
        self.zsets: dict[str, dict[str, float]] = {}
        self.lists: dict[str, list[str]] = {}
        self.now_ms = now_ms
        self.fail_reads = False
        #: PX of string keys set by a Lua model with a TTL (maintenance lock)
        self.ttls_ms: dict[str, int] = {}

    def _check(self) -> None:
        if self.fail_reads:
            raise ConnectionError("redis down")

    # strings
    def get(self, key):
        self._check()
        value = self.values.get(key)
        return None if value is None else _b(value)

    def set(self, key, value):
        self.values[key] = _s(value)
        self.ttls_ms.pop(key, None)

    def delete(self, key):
        self.values.pop(key, None)
        self.hashes.pop(key, None)
        self.ttls_ms.pop(key, None)

    def time(self):
        self._check()
        return (self.now_ms // 1000, (self.now_ms % 1000) * 1000)

    # hashes
    def hgetall(self, key):
        self._check()
        return {_b(k): _b(v) for k, v in self.hashes.get(key, {}).items()}

    def hget(self, key, field):
        value = self.hashes.get(key, {}).get(_s(field))
        return None if value is None else _b(value)

    def hset(self, key, field=None, value=None, mapping=None):
        bucket = self.hashes.setdefault(key, {})
        if field is not None:
            bucket[_s(field)] = _s(value)
        for k, v in (mapping or {}).items():
            bucket[_s(k)] = _s(v)

    def hdel(self, key, *fields):
        bucket = self.hashes.get(key, {})
        for field in fields:
            bucket.pop(_s(field), None)

    def hlen(self, key):
        return len(self.hashes.get(key, {}))

    def hincrby(self, key, field, amount=1):
        bucket = self.hashes.setdefault(key, {})
        bucket[_s(field)] = str(int(bucket.get(_s(field), "0")) + int(amount))
        return int(bucket[_s(field)])

    # sorted sets
    def zadd(self, key, mapping):
        self.zsets.setdefault(key, {}).update({_s(k): float(v) for k, v in mapping.items()})

    def zrange(self, key, start, stop, withscores=False):
        self._check()
        members = sorted(self.zsets.get(key, {}).items(), key=lambda kv: kv[1])
        stop = len(members) - 1 if stop == -1 else stop
        chosen = members[start : stop + 1]
        if withscores:
            return [(_b(m), float(score)) for m, score in chosen]
        return [_b(m) for m, _score in chosen]

    def zrangebyscore(self, key, low, high):
        self._check()
        low = float("-inf") if low == "-inf" else float(low)
        high = float("inf") if high == "+inf" else float(high)
        members = self.zsets.get(key, {})
        return [_b(m) for m, score in sorted(members.items(), key=lambda kv: kv[1]) if low <= score <= high]

    # lists
    def lpush(self, key, *values):
        bucket = self.lists.setdefault(key, [])
        for value in values:
            bucket.insert(0, _s(value))

    def ltrim(self, key, start, stop):
        bucket = self.lists.get(key, [])
        self.lists[key] = bucket[start : stop + 1]

    def lrange(self, key, start, stop):
        return [_b(v) for v in self.lists.get(key, [])[start : stop + 1]]

    # Lua scripts (Python re-implementations with the same argument layout)
    def eval(self, script, numkeys, *keys_and_args):
        modelled = maintenance_lua(self.values, self.ttls_ms, script, keys_and_args)
        if modelled is not NotImplemented:
            return modelled
        if script == _res._ACQUIRE_SCRIPT:
            return self._reservation_acquire(keys_and_args[0], [_s(a) for a in keys_and_args[numkeys:]])
        if script == _res._RENEW_SCRIPT:
            return self._reservation_renew(keys_and_args[0], [_s(a) for a in keys_and_args[numkeys:]])
        if script == _res._RELEASE_SCRIPT:
            return self._reservation_release(keys_and_args[0], [_s(a) for a in keys_and_args[numkeys:]])
        if script in (_leases._ACQUIRE_GPU_SCRIPT, _leases._RELEASE_GPU_SCRIPT, _leases._REBUILD_GPU_SCRIPT):
            return self._gpu_lease_script(script, keys_and_args[:numkeys], [_s(a) for a in keys_and_args[numkeys:]])
        assert script == _SAVE_HASH_SCRIPT, "unknown Lua script"
        state_key, version_key, lock_key = keys_and_args[:numkeys]
        expected, next_version, lock_value, *pairs = keys_and_args[numkeys:]
        current = int(self.values.get(version_key, 0))
        if current != int(expected):
            return [0, current]
        if self.values.get(lock_key) != lock_value:
            return [-1, current]
        self.hashes[state_key] = dict(zip(pairs[::2], pairs[1::2]))
        self.values[version_key] = str(next_version)
        return [1, int(next_version)]


    def _gpu_lease_script(self, script, keys, argv):
        """Python model of the GPU lease Lua scripts (tre_sm.state.gpu_leases)."""
        leases_key, lock_key = (_s(key) for key in keys)
        bucket = self.hashes.setdefault(leases_key, {})
        if script == _leases._ACQUIRE_GPU_SCRIPT:
            if self.values.get(lock_key) != argv[0]:
                return [-1, "writer_fence_lost", ""]
            binding_id, node, owner, token = argv[1], argv[2], argv[3], int(argv[4])
            gpu_ids, ttl_ms, count = json.loads(argv[5]), int(argv[6]), int(argv[7])
            fields, phase = argv[8 : 8 + count], argv[8 + count]
            for field_name in fields:
                raw = bucket.get(field_name)
                if raw is None:
                    continue
                existing = json.loads(raw)
                expires = int(existing["expires_at_ms"])
                if (expires == 0 or expires > self.now_ms) and existing["binding_id"] != binding_id:
                    return [0, field_name, existing["binding_id"]]
            expires = 0 if ttl_ms == 0 else self.now_ms + ttl_ms
            record = json.dumps({
                "binding_id": binding_id, "node": node, "gpu_ids": gpu_ids, "owner": owner,
                "fencing_token": token, "phase": phase, "expires_at_ms": expires,
            })
            for field_name in fields:
                bucket[field_name] = record
            return [1, str(expires), ""]
        if script == _leases._RELEASE_GPU_SCRIPT:
            if self.values.get(lock_key) != argv[0]:
                return -1
            binding_id, token, count = argv[1], int(argv[2]), int(argv[3])
            for field_name in argv[4 : 4 + count]:
                raw = bucket.get(field_name)
                if raw is None:
                    continue
                existing = json.loads(raw)
                if existing["binding_id"] == binding_id and int(existing["fencing_token"]) <= token:
                    bucket.pop(field_name)
            return 1
        if self.values.get(lock_key) != argv[0]:
            return -1
        bucket.clear()
        for index in range(1, len(argv), 2):
            bucket[argv[index]] = argv[index + 1]
        return 1

    def _reservation_acquire(self, key, argv):
        token, owner, operation_id, ttl_ms, count = argv[0], argv[1], argv[2], int(argv[3]), int(argv[4])
        wanted = []
        for i in range(count):
            base = 5 + i * 4
            wanted.append(
                {"id": argv[base], "node": argv[base + 1], "gpus": json.loads(argv[base + 2]), "serve": argv[base + 3]}
            )
        bucket = self.hashes.setdefault(key, {})
        for field_name, raw in list(bucket.items()):
            record = json.loads(raw)
            if int(record["expires_at_ms"]) <= self.now_ms:
                bucket.pop(field_name)
            elif record["token"] != token:
                for w in wanted:
                    if record["binding_id"] == w["id"]:
                        return [0, record["binding_id"]]
        expires = self.now_ms + ttl_ms
        for w in wanted:
            bucket[w["id"]] = json.dumps(
                {
                    "binding_id": w["id"], "serve_id": w["serve"], "node": w["node"],
                    "gpu_ids": w["gpus"], "token": token, "owner": owner,
                    "operation_id": operation_id, "expires_at_ms": expires,
                }
            )
        return [1, str(expires)]

    def _reservation_renew(self, key, argv):
        token, ttl_ms, ids = argv[0], int(argv[1]), argv[2:]
        bucket = self.hashes.setdefault(key, {})
        for binding_id in ids:
            raw = bucket.get(binding_id)
            if raw is None:
                return 0
            record = json.loads(raw)
            if record["token"] != token or int(record["expires_at_ms"]) <= self.now_ms:
                return 0
        for binding_id in ids:
            record = json.loads(bucket[binding_id])
            record["expires_at_ms"] = self.now_ms + ttl_ms
            bucket[binding_id] = json.dumps(record)
        return 1

    def _reservation_release(self, key, argv):
        token, ids = argv[0], argv[1:]
        bucket = self.hashes.setdefault(key, {})
        released = 0
        for binding_id in ids:
            raw = bucket.get(binding_id)
            if raw is not None and json.loads(raw)["token"] == token:
                bucket.pop(binding_id)
                released += 1
        return released


class LegacyRedis:
    """Legacy StateStore backend without EVAL (the store's in-memory test path)."""

    def __init__(self) -> None:
        self.values: dict[str, str] = {}
        self.hashes: dict[str, dict] = {}

    def get(self, key):
        value = self.values.get(key)
        return None if value is None else _b(value)

    def set(self, key, value):
        self.values[key] = _s(value)

    def delete(self, key):
        self.hashes.pop(key, None)

    def hgetall(self, key):
        return dict(self.hashes.get(key, {}))

    def hset(self, key, mapping):
        bucket = self.hashes.setdefault(key, {})
        for field, value in mapping.items():
            bucket[_b(field)] = _b(value)


@contextmanager
def fence(redis: FakeRedis, operation_id: str = "op-test"):
    """Hold the SM writer fence (what an OperationCoordinator operation does)."""
    held = WriterFence(operation_id, "sm-test", 1, "sm-test:1")
    redis.values[rediskeys.SM_WRITER_LOCK_KEY] = held.lock_value
    token = _CURRENT_FENCE.set(held)
    try:
        yield held
    finally:
        _CURRENT_FENCE.reset(token)


class FakeHandle:
    def __init__(self, operation_id: str) -> None:
        self.operation_id = operation_id
        self.phases: list[tuple[str, dict | None]] = []

    def advance(self, phase, *, details=None):
        self.phases.append((phase, details))

    def note(self, **fields):
        self.notes = {**getattr(self, "notes", {}), **fields}

    def assert_active(self):
        return None

    def supersede(self, _operation_id):
        return None


class FakeCoordinator:
    """Synchronous OperationCoordinator: sets the writer fence like the real one."""

    owner = "sm-test"

    def __init__(self, redis: FakeRedis) -> None:
        self.redis = redis
        self.active: dict | None = None
        self.submitted: list[str] = []

    @contextmanager
    def operation(self, kind, *, request=None, wait_s=0.0):
        operation_id = f"{kind}-{len(self.submitted) + 1}"
        self.submitted.append(operation_id)
        handle = FakeHandle(operation_id)
        previous = self.active
        self.active = {"operation_id": operation_id, "kind": kind, "phase": "executing", "status": "running"}
        from tre_sm.state.operations import _CURRENT_OPERATION

        op_token = _CURRENT_OPERATION.set(handle)
        try:
            with fence(self.redis, operation_id):
                yield handle
        finally:
            _CURRENT_OPERATION.reset(op_token)
            self.active = previous

    def submit(self, kind, target, *, request=None):
        with self.operation(kind, request=request) as handle:
            target(handle)
        return handle.operation_id

    def active_operation(self, *, kind=None):
        if self.active is None:
            return None
        if kind is not None and self.active.get("kind") != kind:
            return None
        return self.active

    def stale_running_operations(self, *, kind=None):
        return []

    def list_operations(self, *, limit=100):
        return []


class FakeSafety:
    def __init__(self, actuation: str = "active") -> None:
        #: SM actuation switch (tre:v2:sm:actuation) the service sees.
        self.actuation = actuation
        self.suppressed: list[tuple[str, dict]] = []
        self.maintenance_calls: list[tuple] = []

    def assert_no_pressure(self):
        return None

    def wait_until_healthy(self, operation):
        return None

    def acquire_maintenance(self, operation_id, *, kind, owner="", takeover_operation_ids=()):
        self.maintenance_calls.append(("acquire", operation_id, kind))

    def release_maintenance(self, operation_id):
        self.maintenance_calls.append(("release", operation_id))

    def assert_maintenance_held(self, operation_id):
        return None

    def maintenance(self):
        return None

    def actuation_mode(self):
        return self.actuation

    def actuation_state(self):
        return {"mode": self.actuation, "source": "sm", "suppressed": []}

    def record_suppressed(self, action, detail):
        self.suppressed.append((action, detail))
        return True


class FakeLeases:
    def __init__(self) -> None:
        self.calls: list[tuple] = []

    def acquire(self, binding, *, phase):
        self.calls.append(("acquire", binding.binding_id, phase))

    def release(self, binding):
        self.calls.append(("release", binding.binding_id))

    def load(self):
        return []


class Result:
    def __init__(self, success=True, message=""):
        self.success = success
        self.message = message


class FakeVllm:
    """vLLM engines keyed by pod IP: sleep state, /metrics load, call log."""

    def __init__(self) -> None:
        self.sleeping: dict[str, bool] = {}
        self.load: dict[str, int] = {}
        self.calls: list[tuple] = []
        self.sleep_results: list[Result] = []
        self.events: list[tuple] | None = None
        #: pod IPs whose /sleep always fails (engine stays awake)
        self.fail_sleep_for: set[str] = set()
        #: pod IPs whose /metrics is unavailable (None)
        self.metrics_down: set[str] = set()
        #: pod IP -> vLLM version string (default 0.30.0); None = /version fails
        self.versions: dict[str, str | None] = {}
        self.version_calls: list[str] = []
        #: pod IP -> /is_sleeping answer override (e.g. None = unreachable)
        self.physical_override: dict[str, bool | None] = {}

    def _log(self, *event):
        self.calls.append(event)
        if self.events is not None:
            self.events.append(("vllm",) + event)

    def sleep(self, pod_ip, *, port=None, mode=None, timeout_s=None, hidden=False):
        self._log("sleep", pod_ip, mode, hidden)
        if pod_ip in self.fail_sleep_for:
            return Result(False, "injected failure")
        if self.sleep_results:
            result = self.sleep_results.pop(0)
            if not result.success:
                return result
        self.sleeping[pod_ip] = True
        self.load[pod_ip] = 0
        return Result()

    def wake_up(self, pod_ip, *, port=None):
        self._log("wake_up", pod_ip)
        self.sleeping[pod_ip] = False
        return Result()

    def is_sleeping(self, pod_ip, *, port=None):
        if pod_ip in self.physical_override:
            return self.physical_override[pod_ip]
        return self.sleeping.get(pod_ip)

    def version(self, pod_ip, *, port=None):
        self.version_calls.append(pod_ip)
        return self.versions.get(pod_ip, "0.30.0")

    def wait_until_ready(self, pod_ip, *, port=None):
        self._log("wait_until_ready", pod_ip)
        return Result()

    def metrics(self, pod_ip, *, port=None):
        if pod_ip in self.metrics_down:
            return None
        load = self.load.get(pod_ip, 0)
        return (
            "# HELP vllm:num_requests_running x\n"
            f'vllm:num_requests_running{{model_name="m"}} {float(load)}\n'
            f'vllm:num_requests_waiting{{model_name="m"}} 0.0\n'
        )


class FakeRuntime:
    """k8s Pods with a routable label and a route-gen annotation, like K8sOps."""

    def __init__(self, snapshots=(), deployments=()) -> None:
        self.snapshots: dict[str, K8sPodSnapshot] = {s.name: s for s in snapshots}
        self.deployments = list(deployments)
        self.gen: dict[str, int] = {}
        self.patches: list[tuple[str, str, int]] = []
        self.events: list[tuple] | None = None
        self.fail_hide_for: set[str] = set()

    def list_pod_snapshots(self, *, model=None):
        values = list(self.snapshots.values())
        return [s for s in values if model is None or s.model == model]

    def write_binding_annotations(self, binding, *, state):
        if state == "hidden" and binding.serve_id in self.fail_hide_for:
            raise RuntimeError("patch refused")
        gen = self.gen.get(binding.serve_id, 0) + 1
        self.gen[binding.serve_id] = gen
        self.patches.append((binding.serve_id, state, gen))
        if self.events is not None:
            self.events.append(("patch", binding.serve_id, state, gen))
        snapshot = self.snapshots.get(binding.serve_id)
        if snapshot is not None:
            annotations = dict(snapshot.annotations)
            annotations[STATE_ANNOTATION] = state
            self.snapshots[binding.serve_id] = replace(
                snapshot, annotations=annotations, routable=state == "awake"
            )
        return gen

    def set_pod_routable(self, serve_id, *, routable):
        gen = self.gen.get(serve_id, 0) + 1
        self.gen[serve_id] = gen
        return gen

    def list_model_deployments(self):
        return list(self.deployments)

    def ensure_model_httproute(self, model):
        return None


class FakeGateway:
    """The gateway plugin's side of the contract, written into FakeRedis.

    Instances in ``alive`` re-heartbeat on every tick (their score advances);
    ``auto_ack`` instances ack a route-gen change after ``ack_after_polls``
    ticks, like an informer with some watch latency.
    """

    def __init__(self, redis: FakeRedis, runtime: FakeRuntime | None = None) -> None:
        self.redis = redis
        self.runtime = runtime
        self.instances: list[str] = []
        self.alive: set[str] = set()
        self.auto_ack: set[str] = set()
        self.ack_after_polls = 0
        self._polls = 0
        self._beats = 0

    def heartbeat(self, instance: str, *, age_ms: int = 0, alive: bool | None = None) -> None:
        """Write a heartbeat; ``alive`` (default: age_ms == 0) keeps it beating."""
        self.redis.zadd(rediskeys.GW_INSTANCES_KEY, {instance: self.redis.now_ms - age_ms})
        if instance not in self.instances:
            self.instances.append(instance)
        if alive if alive is not None else age_ms == 0:
            self.alive.add(instance)
        else:
            self.alive.discard(instance)

    def beat(self) -> None:
        """One heartbeat period for every alive instance (scores advance)."""
        self._beats += 1
        for instance in self.alive:
            self.redis.zadd(rediskeys.GW_INSTANCES_KEY, {instance: self.redis.now_ms + self._beats * 1000})

    def seen(self, pod: str, instance: str, *, gen: int, routable: bool) -> None:
        self.redis.hset(
            rediskeys.gw_seen_key(pod),
            instance,
            json.dumps({"gen": gen, "routable": routable, "ts": self.redis.now_ms}),
        )

    def inflight(self, pod: str, instance: str, *, total: int, non_continuable: int = 0) -> None:
        self.redis.hset(
            rediskeys.gw_inflight_key(pod),
            instance,
            json.dumps({"total": total, "non_continuable": non_continuable, "ts": self.redis.now_ms}),
        )

    def tick(self) -> None:
        """Heartbeat, then let auto-ack instances observe the route state."""
        self.beat()
        if self.runtime is None:
            return
        self._polls += 1
        if self._polls <= self.ack_after_polls:
            return
        for pod, gen in self.runtime.gen.items():
            snapshot = self.runtime.snapshots.get(pod)
            routable = bool(snapshot.routable) if snapshot is not None else False
            for instance in self.auto_ack:
                self.seen(pod, instance, gen=gen, routable=routable)


class TickingClock:
    """Virtual clock that also lets fakes react between polls."""

    def __init__(self, *hooks) -> None:
        self.now = 1000.0
        self.hooks = list(hooks)
        self.slept: list[float] = []

    def monotonic(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.slept.append(seconds)
        self.now += max(float(seconds), 0.001)
        for hook in self.hooks:
            hook(self.now)


def policy(**overrides) -> SleepPolicy:
    base = dict(
        ack_timeout_s=10.0,
        instance_staleness_s=10.0,
        no_plugin_grace_s=5.0,
        poll_interval_s=0.5,
        sleep_call_timeout_s=60.0,
        physical_confirm_timeout_s=15.0,
        vllm_sleep_mode_param="true",
        hard_cap_s=150.0,
        plugin_label_selector=None,
        # The mechanism tests exercise the draining protocol on every path; the
        # no-drain paths (production default) are covered by test_sleep_no_drain.
        no_drain_paths=(),
    )
    budgets = overrides.pop("budgets_s", None)
    base.update(overrides)
    result = SleepPolicy(**base)
    if budgets:
        result.budgets_s.update(budgets)
    return result


def trs() -> TrsParams:
    return TrsParams(
        w_p=0.04, w_d=1.0, lambda_wait=2.625, qmin=1.0, ema_alpha=0.5, theta_m=0.0,
        tau_crit=0.8, tau_low=1.0, tau_high=1.25, qsat=4.0, epsat=0.05, hsat=3,
    )


def registry(*, sleep: SleepPolicy | None = None, **config) -> Registry:
    """node-a (4 GPUs): m1 tp1 on GPUs 0,1 (max_replicas 2) + tp2 on (0,1)."""
    topology = ClusterTopology(
        nodes=(NodeSpec("node-a", 4, ((0, 1), (2, 3)), ("GPU-0", "GPU-1", "GPU-2", "GPU-3")),)
    )
    slo = SloSpec(ttft_p95_ms=1200, tpot_p95_ms=100, e2e_p95_ms=10000)
    models = [
        ModelSpec("m1", "/m1", 1, 0, 2, "image", slo, trs()),
        ModelSpec("tp2", "/tp2", 2, 0, 1, "image", slo, trs()),
    ]
    sm_config = ServiceManagerConfig(sleep=sleep or policy(), **config)
    return Registry(topology, models, service_manager=sm_config)


def pod(name, model, gpus, *, ip, state="awake", node="node-a", uid=None, admitted=False):
    gpu_text = ",".join(str(g) for g in gpus)
    annotations = {GPU_IDS_ANNOTATION: gpu_text, STATE_ANNOTATION: state}
    if admitted:
        annotations["tre.aibrix.io/startup-admitted-uid"] = uid or f"uid-{name}"
    return K8sPodSnapshot(
        name=name,
        model=model,
        node=node,
        env={"CUDA_VISIBLE_DEVICES": gpu_text},
        annotations=annotations,
        pod_ip=ip,
        routable=state == "awake",
        ready=True,
        pod_uid=uid or f"uid-{name}",
    )


def binding_of(snapshot: K8sPodSnapshot) -> Binding:
    gpus = tuple(int(g) for g in snapshot.annotations[GPU_IDS_ANNOTATION].split(","))
    state = snapshot.annotations.get(STATE_ANNOTATION, "awake")
    return Binding(
        snapshot.name, snapshot.model, Slot(snapshot.node, gpus),
        awake=state != "sleeping", hidden=state == "hidden",
    )


def deployment(model, gpus, *, node="node-a"):
    name = f"{model}-{node}-gpu-{'-'.join(str(g) for g in gpus)}"
    return ModelDeploymentRecord(name, model, node, tuple(gpus), replicas=1)


def startup_pod(name, model, gpus, *, uid, node="node-a"):
    return StartupPodRecord(
        name=name, uid=uid, model=model, node=node, gpu_ids=tuple(gpus),
        annotations={}, labels={}, pod_ip=None, phase="Pending", ready=False,
    )
