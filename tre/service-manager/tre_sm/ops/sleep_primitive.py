"""The one way the service-manager puts a vLLM pod to sleep (plan 2026-09-27 D1-D4).

Sleeping a vLLM engine cuts off every request it is still serving, which a
client sees as an abort / truncated stream / error. To keep sleep invisible to
clients, every sleep - SafeScale commit, controller scale-downs (fast and
slow loop), APA scale-downs, defrag, fleet repair, startup admission - runs
the same ordered protocol, and callers cannot skip a step:

1. **hide**: one Pod patch sets ``tre.aibrix.io/routable=false``, the
   ``hidden`` state annotation and bumps ``tre.aibrix.io/route-gen``.
2. **gateway ack**: wait until every *live* gateway plugin instance
   (heartbeat in ``tre:v2:gw:instances`` newer than ``instance_staleness_s``)
   reports ``gen >= target`` and ``routable=false`` for the pod in
   ``tre:v2:gw:seen:<pod>``. Timeout -> roll back. With no live instance at all
   (plugin not deployed / fallback routing) wait for the k8s label plus a
   grace delay instead, and log a warning.
3. **drain**: wait until the plugin's in-flight count for the pod (summed over
   live instances, ``tre:v2:gw:inflight:<pod>``) and vLLM's running + waiting
   gauges are both zero, up to the caller's soft budget. Requests the plugin
   marks ``non_continuable`` are waited for up to the hard cap (the gateway
   route timeout) even past the soft budget.
4. **/sleep**: ``mode=wait`` when drained; ``mode=abort`` only once the budget
   is exhausted, counted as a forced abort (``forced_abort_total``,
   ``forced_abort_requests_total``) and logged. vLLM's ``mode=wait`` takes no
   timeout of its own (0.30), so the drain is bounded here and the HTTP
   timeout bounds the call.

Any failure rolls the pod back to its previous routing state (a new
route-gen). A journal entry per pod (``tre:v2:sm:sleep_ops``) lives from the
hide to the end, so a crash in between is visible to the audit
(``hidden_without_operation`` / ``sleep_operation_orphaned``).
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
import json
import logging
import re
import time
from typing import Callable, Mapping

from tre_common import rediskeys
from tre_common.registry import SleepPolicy
from tre_sm.allocator.slots import Binding
from tre_sm.state.reconcile import POD_STATE_AWAKE, POD_STATE_HIDDEN, POD_STATE_SLEEPING

LOG = logging.getLogger("tre_sm.sleep")

VLLM_PORT = 8000
HIDDEN_SLEEP_HEADER = "X-TRE-Hidden"
RECENT_OUTCOMES = 256

_RUNNING_METRIC = "vllm:num_requests_running"
_WAITING_METRIC = "vllm:num_requests_waiting"
_SAMPLE = re.compile(
    r"^(?P<name>[A-Za-z_:][A-Za-z0-9_:]*)(?:\{(?P<labels>[^}]*)\})?\s+(?P<value>\S+)"
)


class Clock:
    """Time seams (tests substitute a virtual clock)."""

    def monotonic(self) -> float:
        return time.monotonic()

    def sleep(self, seconds: float) -> None:
        if seconds > 0:
            time.sleep(seconds)


#: Used when a SleepPrimitive gets no clock; read at call time so tests can patch it.
DEFAULT_CLOCK: Clock = Clock()


class GatewayAckTimeout(RuntimeError):
    pass


class SleepFailed(RuntimeError):
    pass


@dataclass(frozen=True)
class SleepTarget:
    binding: Binding
    pod_ip: str


def parse_vllm_load(metrics_text: str | None) -> int | None:
    """running + waiting requests from vLLM's Prometheus text (None = no gauges)."""
    if not metrics_text:
        return None
    total = 0.0
    seen = False
    for line in metrics_text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        match = _SAMPLE.match(line)
        if match is None or match.group("name") not in (_RUNNING_METRIC, _WAITING_METRIC):
            continue
        try:
            total += float(match.group("value"))
        except ValueError:
            continue
        seen = True
    return int(round(total)) if seen else None


def _text(value: object) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


class GatewayState:
    """Read side of the gateway-plugin contract (keys in tre_common.rediskeys).

    Live plugin instances = fresh heartbeats UNION Ready plugin pods from k8s
    (plugin_pods): an instance with Redis trouble may keep routing while its
    heartbeat goes stale, so it must still ack. Errors propagate; the
    primitive treats them as "not converged" (fail closed).
    """

    def __init__(
        self,
        redis_client,
        *,
        plugin_pods: Callable[[], set[str]] | None = None,
        wall_ms: Callable[[], int] | None = None,
    ) -> None:
        self._redis = redis_client
        self._plugin_pods = plugin_pods
        self._wall_ms = wall_ms

    def now_ms(self) -> int:
        """Redis server time: one clock for SM and every plugin instance."""
        if self._wall_ms is not None:
            return int(self._wall_ms())
        redis_time = getattr(self._redis, "time", None)
        if callable(redis_time):
            try:
                seconds, micros = redis_time()
                return int(seconds) * 1000 + int(micros) // 1000
            except Exception:  # pragma: no cover - fall back to the local clock
                pass
        return int(time.time() * 1000)

    def live_instances(self, staleness_s: float) -> list[str]:
        cutoff = self.now_ms() - int(staleness_s * 1000)
        raw = self._redis.zrangebyscore(rediskeys.GW_INSTANCES_KEY, cutoff, "+inf")
        live = {_text(item) for item in raw or []}
        if self._plugin_pods is not None:
            live |= {str(name) for name in self._plugin_pods()}
        return sorted(live)

    def seen(self, pod: str) -> dict[str, dict | None]:
        return self._json_hash(rediskeys.gw_seen_key(pod))

    def inflight(self, pod: str) -> dict[str, dict | None]:
        return self._json_hash(rediskeys.gw_inflight_key(pod))

    def _json_hash(self, key: str) -> dict[str, dict | None]:
        result: dict[str, dict | None] = {}
        for field_name, raw in (self._redis.hgetall(key) or {}).items():
            try:
                payload = json.loads(_text(raw))
            except (TypeError, ValueError):
                payload = None
            result[_text(field_name)] = payload if isinstance(payload, dict) else None
        return result


class SleepJournal:
    """Per-pod record of a sleep in progress plus counters (Redis, or memory)."""

    def __init__(self, redis_client=None) -> None:
        self._redis = redis_client
        self._ops: dict[str, dict] = {}
        self._stats: dict[str, int] = {}
        self._ack_latencies: deque[int] = deque(maxlen=rediskeys.SM_SLEEP_ACK_LATENCY_MAX)

    def begin(self, pod: str, record: dict) -> None:
        self._write(pod, dict(record))

    def update(self, pod: str, **fields) -> None:
        # This process is the only writer of its own entries: update from memory.
        record = dict(self._ops.get(pod) or self.get(pod) or {})
        record.update(fields)
        self._write(pod, record)

    def end(self, pod: str) -> None:
        self._ops.pop(pod, None)
        if self._redis is not None:
            self._redis.hdel(rediskeys.SM_SLEEP_OPS_KEY, pod)

    def get(self, pod: str) -> dict | None:
        return self.entries().get(pod)

    def entries(self) -> dict[str, dict]:
        if self._redis is None:
            return {pod: dict(record) for pod, record in self._ops.items()}
        result: dict[str, dict] = {}
        for field_name, raw in (self._redis.hgetall(rediskeys.SM_SLEEP_OPS_KEY) or {}).items():
            try:
                result[_text(field_name)] = json.loads(_text(raw))
            except (TypeError, ValueError):
                result[_text(field_name)] = {"corrupt": True}
        return result

    def incr(self, name: str, amount: int = 1) -> None:
        self._stats[name] = self._stats.get(name, 0) + int(amount)
        if self._redis is not None:
            self._redis.hincrby(rediskeys.SM_SLEEP_STATS_KEY, name, int(amount))

    def record_ack_latency(self, latency_ms: int) -> None:
        self._ack_latencies.appendleft(int(latency_ms))
        if self._redis is not None:
            self._redis.lpush(rediskeys.SM_SLEEP_ACK_LATENCY_KEY, int(latency_ms))
            self._redis.ltrim(
                rediskeys.SM_SLEEP_ACK_LATENCY_KEY, 0, rediskeys.SM_SLEEP_ACK_LATENCY_MAX - 1
            )

    def stats(self) -> dict[str, int]:
        if self._redis is None:
            return dict(self._stats)
        raw = self._redis.hgetall(rediskeys.SM_SLEEP_STATS_KEY) or {}
        return {_text(key): int(_text(value)) for key, value in raw.items()}

    def ack_latencies_ms(self) -> list[int]:
        if self._redis is None:
            return list(self._ack_latencies)
        raw = self._redis.lrange(
            rediskeys.SM_SLEEP_ACK_LATENCY_KEY, 0, rediskeys.SM_SLEEP_ACK_LATENCY_MAX - 1
        )
        return [int(_text(item)) for item in raw or []]

    def _write(self, pod: str, record: dict) -> None:
        self._ops[pod] = record
        if self._redis is not None:
            self._redis.hset(
                rediskeys.SM_SLEEP_OPS_KEY,
                mapping={pod: json.dumps(record, sort_keys=True, separators=(",", ":"))},
            )


@dataclass
class _PodSleep:
    target: SleepTarget
    previous_state: str
    gen: int | None = None
    hidden: bool = False
    done: bool = False
    last_load: dict = field(default_factory=dict)

    @property
    def pod(self) -> str:
        return self.target.binding.serve_id


class SleepPrimitive:
    def __init__(
        self,
        *,
        runtime_ops,
        vllm_ops,
        policy: SleepPolicy,
        gateway: GatewayState | None = None,
        journal: SleepJournal | None = None,
        clock: Clock | None = None,
        owner: str = "service-manager",
        operation_id: Callable[[], str | None] | None = None,
    ) -> None:
        self._runtime = runtime_ops
        self._vllm = vllm_ops
        self._policy = policy
        self._gateway = gateway
        self._journal = journal or SleepJournal()
        self._clock = clock
        self._owner = owner
        self._operation_id = operation_id or (lambda: None)
        self._recent: deque[dict] = deque(maxlen=RECENT_OUTCOMES)

    @property
    def policy(self) -> SleepPolicy:
        return self._policy

    @property
    def journal(self) -> SleepJournal:
        return self._journal

    def recent(self) -> list[dict]:
        return list(self._recent)

    # ------------------------------------------------------------------ public
    def sleep(
        self,
        targets: list[SleepTarget],
        *,
        path: str,
        drain_budget_s: float | None = None,
    ) -> list[dict]:
        """hide -> ack -> drain -> /sleep for every target; see module docstring.

        Several targets are hidden together and drain concurrently (one soft
        budget for the call). Raises after rolling back every target that did
        not reach sleep.
        """
        if not targets:
            return []
        clock = self._clock or DEFAULT_CLOCK
        soft_s = self._policy.soft_budget_s(path, drain_budget_s)
        hard_s = self._policy.hard_cap_s
        pods = [_PodSleep(target, _previous_state(target.binding)) for target in targets]
        started = clock.monotonic()
        try:
            self._hide(pods, path=path, soft_s=soft_s)
            ack = self._await_ack(pods, clock)
        except BaseException as exc:
            self._rollback(pods, reason=f"{type(exc).__name__}: {exc}")
            raise
        soft_deadline = started + soft_s
        hard_deadline = started + max(hard_s, soft_s)
        outcomes: list[dict] = []
        failure: BaseException | None = None
        pending = list(pods)
        try:
            while pending:
                now = clock.monotonic()
                for pod in list(pending):
                    load = self._load(pod, ack)
                    pod.last_load = load
                    must_wait_nc = load["non_continuable"] > 0 and now < hard_deadline
                    budget_left = now < soft_deadline or must_wait_nc
                    if load["drained"] or not budget_left:
                        pending.remove(pod)
                        outcomes.append(
                            self._sleep_one(
                                pod,
                                load=load,
                                drained=load["drained"],
                                path=path,
                                ack=ack,
                                waited_s=now - started,
                                hard_deadline=hard_deadline,
                                clock=clock,
                            )
                        )
                if pending:
                    now = clock.monotonic()
                    next_deadline = min(
                        hard_deadline
                        if pod.last_load.get("non_continuable", 0) > 0
                        else soft_deadline
                        for pod in pending
                    )
                    clock.sleep(
                        max(0.0, min(self._policy.poll_interval_s, next_deadline - now))
                        or min(self._policy.poll_interval_s, 0.01)
                    )
        except BaseException as exc:
            failure = exc
        if failure is not None:
            self._rollback([pod for pod in pods if not pod.done], reason=str(failure))
            raise failure
        return outcomes

    # ------------------------------------------------------------------- steps
    def _hide(self, pods: list[_PodSleep], *, path: str, soft_s: float) -> None:
        for pod in pods:
            binding = pod.target.binding
            self._journal.begin(
                pod.pod,
                {
                    "binding_id": binding.binding_id,
                    "serve_id": binding.serve_id,
                    "path": path,
                    "soft_budget_s": soft_s,
                    "hard_cap_s": self._policy.hard_cap_s,
                    "phase": "hiding",
                    "previous_state": pod.previous_state,
                    "owner": self._owner,
                    "operation_id": self._operation_id(),
                    "started_at_ms": int(time.time() * 1000),
                },
            )
            gen = self._runtime.write_binding_annotations(binding, state=POD_STATE_HIDDEN)
            pod.hidden = True
            pod.gen = gen if isinstance(gen, int) and not isinstance(gen, bool) else None
            self._journal.update(pod.pod, phase="awaiting_ack", target_gen=pod.gen)

    def _await_ack(self, pods: list[_PodSleep], clock: Clock) -> dict:
        policy = self._policy
        started = clock.monotonic()
        deadline = started + policy.ack_timeout_s
        live: list[str] = []
        while True:
            error: str | None = None
            unacked: list[tuple[str, str]] = []
            try:
                live = (
                    self._gateway.live_instances(policy.instance_staleness_s)
                    if self._gateway is not None
                    else []
                )
                if live:
                    unacked = [
                        (pod.pod, instance)
                        for pod in pods
                        for instance in live
                        if not _acked(self._gateway.seen(pod.pod).get(instance), pod.gen)
                    ]
            except Exception as exc:  # Redis / k8s read error: not converged
                error = f"{type(exc).__name__}: {exc}"
                unacked = [(pod.pod, "<read error>") for pod in pods]
            if error is None and not live:
                # Neither a fresh heartbeat nor a Ready plugin pod: no plugin at all.
                return self._fallback_ack(pods, clock)
            now = clock.monotonic()
            if not unacked:
                latency_ms = int(round((now - started) * 1000))
                self._journal.record_ack_latency(latency_ms)
                for pod in pods:
                    self._journal.update(pod.pod, phase="draining", ack="plugin")
                return {"mode": "plugin", "instances": live, "latency_ms": latency_ms}
            if now >= deadline:
                self._journal.incr("ack_timeout_total")
                raise GatewayAckTimeout(
                    f"gateway plugin did not ack hide within {policy.ack_timeout_s}s: "
                    f"{sorted(unacked)}" + (f" (last error: {error})" if error else "")
                )
            clock.sleep(min(policy.poll_interval_s, max(0.0, deadline - now)))

    def _fallback_ack(self, pods: list[_PodSleep], clock: Clock) -> dict:
        policy = self._policy
        LOG.warning(
            "no live gateway plugin instance: hiding %s on the k8s label and a %.1fs grace "
            "delay (no in-flight view from the gateway)",
            [pod.pod for pod in pods],
            policy.no_plugin_grace_s,
        )
        self._journal.incr("ack_fallback_total")
        wait_unroutable = getattr(self._runtime, "wait_pod_unroutable", None)
        if callable(wait_unroutable):
            for pod in pods:
                wait_unroutable(pod.target.binding, timeout_s=policy.ack_timeout_s)
        clock.sleep(policy.no_plugin_grace_s)
        for pod in pods:
            self._journal.update(pod.pod, phase="draining", ack="fallback_no_plugin")
        return {"mode": "fallback_no_plugin", "instances": [], "latency_ms": None}

    def _load(self, pod: _PodSleep, ack: dict) -> dict:
        gateway_total: int | None = None
        gateway_error = False
        non_continuable = 0
        if ack["mode"] == "plugin" and self._gateway is not None:
            # Read only after the ack: the plugin writes a pod's inflight before its
            # seen field in one pipeline, so this value is at least as new as the ack.
            # A live instance without a field has nothing in flight (0).
            try:
                live = set(self._gateway.live_instances(self._policy.instance_staleness_s))
                entries = self._gateway.inflight(pod.pod)
            except Exception:  # unreadable: not drained (fail closed)
                gateway_error = True
                entries = {}
                live = set()
            gateway_total = 0
            for instance, entry in entries.items():
                if instance not in live:
                    continue
                if entry is None:
                    # Unreadable entry of a live instance: assume one request.
                    gateway_total += 1
                    continue
                gateway_total += max(0, int(entry.get("total", 0) or 0))
                non_continuable += max(0, int(entry.get("non_continuable", 0) or 0))
        engine_load: int | None = None
        metrics = getattr(self._vllm, "metrics", None)
        if callable(metrics):
            try:
                engine_load = parse_vllm_load(metrics(pod.target.pod_ip, port=VLLM_PORT))
            except Exception:  # an unreachable engine reports nothing
                engine_load = None
        drained = (
            not gateway_error and (gateway_total or 0) == 0 and (engine_load or 0) == 0
        )
        return {
            "gateway_read_error": gateway_error,
            "gateway_inflight": gateway_total,
            "non_continuable": non_continuable,
            "engine_load": engine_load,
            "drained": drained,
        }

    def _sleep_one(
        self,
        pod: _PodSleep,
        *,
        load: dict,
        drained: bool,
        path: str,
        ack: dict,
        waited_s: float,
        hard_deadline: float,
        clock: Clock,
    ) -> dict:
        target = pod.target
        policy = self._policy
        self._journal.update(pod.pod, phase="sleeping", drained=drained)
        in_flight = max(load.get("gateway_inflight") or 0, load.get("engine_load") or 0)
        forced = not drained
        forced_count = in_flight if forced else 0
        if policy.vllm_sleep_mode_param:
            mode: str | None = "wait" if drained else "abort"
        else:
            mode = None
        timeout = policy.sleep_call_timeout_s
        if mode == "wait":
            timeout += max(0.0, hard_deadline - clock.monotonic())
        result = self._vllm.sleep(
            target.pod_ip, port=VLLM_PORT, mode=mode, timeout_s=timeout, hidden=True
        )
        if not _success(result) and mode == "wait" and self._physical(target) is not True:
            # A straggler kept mode=wait from finishing: the budget is spent.
            forced = True
            forced_count = max(1, self._current_in_flight(pod))
            mode = "abort"
            result = self._vllm.sleep(
                target.pod_ip,
                port=VLLM_PORT,
                mode="abort",
                timeout_s=policy.sleep_call_timeout_s,
                hidden=True,
            )
        if not _success(result):
            message = getattr(result, "message", "") or "operation failed"
            raise SleepFailed(f"vLLM sleep failed for {target.binding.serve_id}: {message}")
        self._await_physical_sleep(target, clock)
        self._runtime.write_binding_annotations(target.binding, state=POD_STATE_SLEEPING)
        pod.done = True
        self._journal.end(pod.pod)
        self._journal.incr("sleeps_total")
        self._journal.incr(f"sleeps_path_{path}")
        if forced:
            self._journal.incr("forced_abort_total")
            self._journal.incr("forced_abort_requests_total", forced_count)
        outcome = {
            "serve_id": target.binding.serve_id,
            "binding_id": target.binding.binding_id,
            "path": path,
            "ack_mode": ack["mode"],
            "ack_latency_ms": ack.get("latency_ms"),
            "drained": drained,
            "sleep_mode": mode,
            "forced_abort": forced,
            "forced_abort_requests": forced_count,
            "non_continuable_at_sleep": load.get("non_continuable", 0),
            "waited_s": round(waited_s, 3),
        }
        self._recent.append(outcome)
        log = LOG.warning if forced else LOG.info
        log("sleep %s", json.dumps(outcome, sort_keys=True))
        return outcome

    def _current_in_flight(self, pod: _PodSleep) -> int:
        metrics = getattr(self._vllm, "metrics", None)
        if not callable(metrics):
            return 0
        try:
            return parse_vllm_load(metrics(pod.target.pod_ip, port=VLLM_PORT)) or 0
        except Exception:
            return 0

    def _physical(self, target: SleepTarget) -> bool | None:
        probe = getattr(self._vllm, "is_sleeping", None)
        if not callable(probe):
            return None
        return probe(target.pod_ip, port=VLLM_PORT)

    def _await_physical_sleep(self, target: SleepTarget, clock: Clock) -> None:
        if not callable(getattr(self._vllm, "is_sleeping", None)):
            return
        deadline = clock.monotonic() + self._policy.sleep_call_timeout_s
        while True:
            if self._physical(target) is True:
                return
            if clock.monotonic() >= deadline:
                raise SleepFailed(
                    f"vLLM sleep did not physically converge for {target.binding.serve_id}"
                )
            clock.sleep(self._policy.poll_interval_s)

    def _rollback(self, pods: list[_PodSleep], *, reason: str) -> None:
        for pod in pods:
            if pod.done or not pod.hidden:
                if not pod.done:
                    self._journal.end(pod.pod)
                continue
            try:
                physical = self._physical(pod.target)
                if physical is True:
                    # It did go to sleep; record the truth instead of re-routing.
                    self._runtime.write_binding_annotations(
                        pod.target.binding, state=POD_STATE_SLEEPING
                    )
                else:
                    self._runtime.write_binding_annotations(
                        pod.target.binding, state=pod.previous_state
                    )
                self._journal.end(pod.pod)
                self._journal.incr("rollback_total")
                LOG.warning(
                    "sleep of %s rolled back to %s: %s",
                    pod.pod,
                    POD_STATE_SLEEPING if physical is True else pod.previous_state,
                    reason,
                )
            except Exception:  # keep the original failure; the journal stays as evidence
                LOG.exception("rollback of %s failed; journal entry kept", pod.pod)


def _previous_state(binding: Binding) -> str:
    if binding.hidden or not binding.awake:
        return POD_STATE_HIDDEN
    return POD_STATE_AWAKE


def _acked(entry: Mapping | None, gen: int | None) -> bool:
    """Contract: seen gen > target (a later SM patch superseded it), or seen gen ==
    target with routable == false. A missing field is not an ack."""
    if not isinstance(entry, Mapping):
        return False
    if gen is None:  # runtime could not report the generation: routable=false only
        return entry.get("routable") is False
    try:
        seen_gen = int(entry.get("gen", -1))
    except (TypeError, ValueError):
        return False
    if seen_gen > gen:
        return True
    return seen_gen == gen and entry.get("routable") is False


def _success(result) -> bool:
    return bool(getattr(result, "success", False))
