"""The one way the service-manager puts a vLLM pod to sleep (plan 2026-09-27 D1-D4).

Sleeping a vLLM engine cuts off every request it is still serving, which a
client sees as an abort / truncated stream / error. To keep sleep invisible to
clients, every sleep - SafeScale commit, controller scale-downs (fast and
slow loop), APA scale-downs, defrag, fleet repair, startup admission - runs
the same ordered protocol, and callers cannot skip a step:

1. **prepare** (writer lock held): reserve the bindings (per-binding
   reservation, :mod:`tre_sm.state.sleep_reservations`), write a journal entry
   per pod, and hide: one Pod patch sets ``tre.aibrix.io/routable=false``, the
   ``hidden`` state annotation and bumps ``tre.aibrix.io/route-gen``.
2. **drain** (no writer lock needed; the reservation fences the binding):
   a. gateway ack: at least ``gateway_min_instances`` plugin instances must be
      live (heartbeat score advancing, or a Ready plugin pod) and every live
      instance must report ``gen >= target`` / ``routable=false`` for the pod.
      An empty live set never converges (unless the opt-in
      ``fallback_no_plugin``). Timeout -> roll back.
   b. drain: poll the plugin in-flight counts of live instances
      (``tre:v2:gw:inflight:<pod>``; fields of non-live instances are ignored)
      AND vLLM's running + waiting gauges - always both: a plugin instance that
      shuts down zeroes its counts and leaves the live set while Envoy may
      still stream to the pod. Metrics unavailable or a read error
      = drain state UNKNOWN (never "drained"). ``non_continuable`` is sticky: a
      read error keeps the last known value.
      * drained (state known, nothing in flight) -> /sleep ``mode=wait``;
      * before the soft budget -> keep waiting;
      * state unknown -> keep waiting; still unknown at the hard cap -> ROLL
        BACK (never abort blind);
      * non-continuable requests in flight -> keep waiting; still there at the
        hard cap -> roll back;
      * state known, only continuable requests left past the soft budget ->
        /sleep ``mode=abort``, counted as a forced abort (the sidecar continues
        those requests).
      **No-drain paths** (``service_manager.sleep.no_drain_paths``; default the
      SafeScale commit, the fast-loop donors and APA - v1 / paper semantics,
      2026-09-29) skip the wait: right after the ack one round reads the load
      for the record, then nothing in flight -> ``mode=wait``, anything else
      (continuable, non-continuable or unknown) -> ``mode=abort`` at once. No
      rollback over in-flight requests; the outcome's ``aborted`` field counts
      what was cut off (``non_continuable`` = truncated or re-run from scratch
      by the sidecar, ``unclassified`` = engine-side requests the gateway did
      not count, ``state_known`` = False when the counts could not be read).
      Every poll renews the reservation and checks shutdown / the writer fence;
      shutdown or a lost writer fence rolls back. A LOST reservation is never
      re-acquired: another sleep may own the binding by now, so the drain
      stops and the caller resolves it under the writer lock (rollback unless
      another live reservation covers the binding).
      Each poll round reads the engine metrics of all targets in parallel and
      decides with the time measured after the reads, so the drain ends within
      the hard cap plus one round.
3. **commit** (writer lock held again): the reservation is renewed (lost ->
   roll back), then every target is committed IN PARALLEL: /sleep (``mode``
   only for vLLM versions that accept it, detected via ``GET /version``), then
   ``/is_sleeping`` is polled for all of them; the reservation keeps being
   renewed meanwhile (a target failing either step is re-probed once by its
   rollback, also in parallel). Confirmed -> ``sleeping`` annotation, journal entry and
   reservation released. /sleep returned but the physical state stays unknown
   -> the pod stays hidden, the journal entry stays (audit
   ``sleep_unconfirmed``); routing is never re-opened on a pod that may be
   asleep. The duration of a call is bounded by
   ``ServiceManagerConfig.worst_case_sleep_call_s`` whatever the target count.

Rollback restores the pod's previous routing state under a new route-gen
(SafeScale probe pods stay hidden). Results are per target: a multi-target
sleep that partly fails raises :class:`SleepIncomplete` carrying one outcome
per target (``status`` slept / rolled_back / unconfirmed / rollback_failed /
reservation_lost), so callers account for exactly the pods that slept.
"""

from __future__ import annotations

from collections import deque
from contextlib import nullcontext
from concurrent.futures import ThreadPoolExecutor, wait
import contextvars
from dataclasses import dataclass, field
import json
import logging
import re
import threading
import time
from typing import Callable, Mapping

from tre_common import rediskeys
from tre_common.registry import SleepPolicy
from tre_common.vllm_metrics import vllm_candidates
from tre_sm.allocator.slots import Binding
from tre_sm.state.operations import current_operation
from tre_sm.state.reconcile import POD_STATE_AWAKE, POD_STATE_HIDDEN, POD_STATE_SLEEPING
from tre_sm.state.sleep_reservations import ReservationConflict, SleepReservations

LOG = logging.getLogger("tre_sm.sleep")

VLLM_PORT = 8000
HIDDEN_SLEEP_HEADER = "X-TRE-Hidden"
RECENT_OUTCOMES = 256

#: First vLLM release whose ``POST /sleep`` accepts ``mode=wait|abort|keep``
#: (upstream PR #34528, "Cleanup engine pause/sleep logic", 2026-02-24; the
#: v0.18.0 release branch is the earliest verified to contain it; 0.10.1 reads
#: only ``level`` and ignores ``mode``). A version that does not parse (e.g. a
#: ``0.1.devN`` source build without tags) is treated as not supporting it: the
#: primitive then drains fully and sends a plain /sleep.
SLEEP_MODE_MIN_VERSION = (0, 18, 0)

STATUS_SLEPT = "slept"
STATUS_ROLLED_BACK = "rolled_back"
STATUS_UNCONFIRMED = "unconfirmed"
STATUS_ROLLBACK_FAILED = "rollback_failed"
STATUS_RESERVATION_LOST = "reservation_lost"

#: "not given" marker of :meth:`SleepPrimitive._load`'s ``engine_load``.
_UNSET = object()

#: Engine load = running + waiting. Neither family was renamed in vLLM 0.30; the names
#: come from tre_common.vllm_metrics like every other vLLM read. Only the first candidate
#: of each is summed (an engine exporting an old and a new name would double count).
_RUNNING_METRIC = vllm_candidates("num_requests_running")[0]
_WAITING_METRIC = vllm_candidates("num_requests_waiting")[0]
_SAMPLE = re.compile(
    r"^(?P<name>[A-Za-z_:][A-Za-z0-9_:]*)(?:\{(?P<labels>[^}]*)\})?\s+(?P<value>\S+)"
)
_VERSION = re.compile(r"^\s*v?(\d+)\.(\d+)(?:\.(\d+))?")


class Clock:
    """Time seams (tests substitute a virtual clock)."""

    def monotonic(self) -> float:
        return time.monotonic()

    def sleep(self, seconds: float) -> None:
        if seconds > 0:
            time.sleep(seconds)


#: Used when a SleepPrimitive gets no clock; read at call time so tests can patch it.
DEFAULT_CLOCK: Clock = Clock()


class SleepFailed(RuntimeError):
    """A sleep did not complete; ``outcomes`` has one entry per target handled."""

    def __init__(self, message: str, *, outcomes: list[dict] | None = None) -> None:
        super().__init__(message)
        self.outcomes: list[dict] = list(outcomes or [])


class GatewayAckTimeout(SleepFailed):
    pass


class SleepCancelled(SleepFailed):
    """The service-manager is shutting down: in-progress drains roll back."""


class ReservationLost(SleepFailed):
    pass


class SleepIncomplete(SleepFailed):
    """Some targets did not sleep; see ``outcomes`` (per-target ``status``)."""


class ServiceShuttingDown(RuntimeError):
    """No new sleep is accepted while the service-manager shuts down (HTTP 503)."""


@dataclass(frozen=True)
class SleepTarget:
    binding: Binding
    pod_ip: str
    #: The Pod object's UID: a same-name recreated Pod is a different engine (the
    #: vLLM version cache is keyed by it).
    pod_uid: str | None = None


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


def parse_vllm_version(text: str | None) -> tuple[int, int, int] | None:
    """``"0.30.0"`` -> (0, 30, 0); None when it does not look like a release."""
    if not text:
        return None
    match = _VERSION.match(str(text))
    if match is None:
        return None
    return (int(match.group(1)), int(match.group(2)), int(match.group(3) or 0))


def sleep_mode_supported(version: str | None) -> bool:
    parsed = parse_vllm_version(version)
    return parsed is not None and parsed >= SLEEP_MODE_MIN_VERSION


def _text(value: object) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


class GatewayState:
    """Read side of the gateway-plugin contract (keys in tre_common.rediskeys).

    Liveness is clock-independent: an instance is live while its heartbeat score
    in ``tre:v2:gw:instances`` keeps CHANGING between reads of this SM, within
    ``staleness_s`` of the SM's own monotonic clock. The score is never compared
    with a wall clock (a plugin may write local time or Redis TIME). A score
    that is first seen, or that stopped changing, is not live. Ready plugin pods
    from k8s (``plugin_pods``) count as live too: an instance with Redis trouble
    may keep routing while its heartbeat stalls, so it must still ack. Errors
    propagate; the primitive treats them as "not converged" (fail closed).
    """

    def __init__(
        self,
        redis_client,
        *,
        plugin_pods: Callable[[], set[str]] | None = None,
        monotonic: Callable[[], float] | None = None,
        future_warn_ms: int = 1000,
    ) -> None:
        self._redis = redis_client
        self._plugin_pods = plugin_pods
        self._monotonic = monotonic or time.monotonic
        self._future_warn_ms = int(future_warn_ms)
        #: instance -> [last score, SM monotonic of the last change (None = never),
        #:              SM monotonic of the first sighting]
        self._scores: dict[str, list] = {}
        self._warned_future: set[str] = set()
        self._lock = threading.Lock()

    def now_ms(self) -> int | None:
        """Redis server time (only used to warn about clocks ahead of it)."""
        redis_time = getattr(self._redis, "time", None)
        if not callable(redis_time):
            return None
        try:
            seconds, micros = redis_time()
        except Exception:
            return None
        return int(seconds) * 1000 + int(micros) // 1000

    def _observe(self, staleness_s: float) -> tuple[list[str], list[str]]:
        raw = self._redis.zrange(rediskeys.GW_INSTANCES_KEY, 0, -1, withscores=True) or []
        now = self._monotonic()
        redis_now = self.now_ms()
        live: set[str] = set()
        pending: set[str] = set()
        with self._lock:
            present: set[str] = set()
            for member, score in raw:
                instance = _text(member)
                score = float(score)
                present.add(instance)
                state = self._scores.get(instance)
                if state is None:
                    self._scores[instance] = [score, None, now]
                elif score != state[0]:
                    # A changed score is a new heartbeat write (even if the writer's
                    # clock stepped back).
                    state[0] = score
                    state[1] = now
                state = self._scores[instance]
                if state[1] is not None and now - state[1] <= staleness_s:
                    live.add(instance)
                elif state[1] is None and now - state[2] <= staleness_s:
                    pending.add(instance)
                if (
                    redis_now is not None
                    and score - redis_now > self._future_warn_ms
                    and instance not in self._warned_future
                ):
                    self._warned_future.add(instance)
                    LOG.warning(
                        "gateway plugin %s heartbeat is %.1fs ahead of Redis TIME: its "
                        "clock is skewed (liveness does not depend on it; fix NTP/chrony)",
                        instance,
                        (score - redis_now) / 1000.0,
                    )
            for gone in set(self._scores) - present:
                self._scores.pop(gone, None)
        return sorted(live), sorted(pending)

    def live_instances(self, staleness_s: float) -> list[str]:
        live, _pending = self.instances(staleness_s)
        return live

    def instances(self, staleness_s: float) -> tuple[list[str], list[str]]:
        """(live, pending): pending = first seen, not yet seen advancing."""
        live, pending = self._observe(staleness_s)
        result = set(live)
        if self._plugin_pods is not None:
            result |= {str(name) for name in self._plugin_pods()}
        return sorted(result), sorted(set(pending) - result)

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
        self._lock = threading.Lock()

    def begin(self, pod: str, record: dict) -> None:
        self._write(pod, dict(record))

    def update(self, pod: str, **fields) -> None:
        # This process is the only writer of its own entries: update from memory.
        with self._lock:
            cached = self._ops.get(pod)
        record = dict(cached or self.get(pod) or {})
        record.update(fields)
        self._write(pod, record)

    def end(self, pod: str) -> None:
        with self._lock:
            self._ops.pop(pod, None)
        if self._redis is not None:
            self._redis.hdel(rediskeys.SM_SLEEP_OPS_KEY, pod)

    def get(self, pod: str) -> dict | None:
        if self._redis is None:
            return self.cached(pod)
        raw = self._redis.hget(rediskeys.SM_SLEEP_OPS_KEY, pod)
        if raw is None:
            return None
        try:
            return json.loads(_text(raw))
        except (TypeError, ValueError):
            return {"corrupt": True}

    def cached(self, pod: str) -> dict | None:
        """This process's last write of the pod's entry (no Redis read)."""
        with self._lock:
            record = self._ops.get(pod)
        return None if record is None else dict(record)

    def entries(self) -> dict[str, dict]:
        if self._redis is None:
            with self._lock:
                return {pod: dict(record) for pod, record in self._ops.items()}
        result: dict[str, dict] = {}
        for field_name, raw in (self._redis.hgetall(rediskeys.SM_SLEEP_OPS_KEY) or {}).items():
            try:
                result[_text(field_name)] = json.loads(_text(raw))
            except (TypeError, ValueError):
                result[_text(field_name)] = {"corrupt": True}
        return result

    def incr(self, name: str, amount: int = 1) -> None:
        with self._lock:
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
            with self._lock:
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
        with self._lock:
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
    reserved: bool = False
    #: "wait" (drained) | "abort" (forced) once the drain decided; None = pending.
    decision: str | None = None
    sleep_called: bool = False
    done: bool = False
    #: Sticky non-continuable count: last value from a SUCCESSFUL gateway read.
    non_continuable: int | None = None
    last_load: dict = field(default_factory=dict)
    waited_s: float = 0.0
    outcome: dict | None = None
    #: Set once /sleep was sent and the pod awaits physical confirmation:
    #: {"mode", "forced", "forced_count"}.
    commit: dict | None = None

    @property
    def pod(self) -> str:
        return self.target.binding.serve_id

    @property
    def binding_id(self) -> str:
        return self.target.binding.binding_id


@dataclass
class SleepBatch:
    """One sleep call's targets across the prepare / drain / commit phases."""

    pods: list[_PodSleep]
    path: str
    soft_s: float
    hard_s: float
    started: float
    token: str | None = None
    ack: dict | None = None
    operation_id: str | None = None
    #: Extra fields of every target's journal entry.
    journal_extra: dict = field(default_factory=dict)
    #: No-drain path: abort everything in flight right after the ack.
    no_drain: bool = False
    #: The drain found the reservation lost (resolve under the writer lock).
    reservation_lost: bool = False
    #: Serializes reservation renewals with per-target releases (commit threads).
    lock: threading.Lock = field(default_factory=threading.Lock, repr=False, compare=False)

    def reserved_ids(self) -> list[str]:
        return [pod.binding_id for pod in self.pods if pod.reserved]

    def outcomes(self) -> list[dict]:
        return [pod.outcome for pod in self.pods if pod.outcome is not None]


class SleepPrimitive:
    def __init__(
        self,
        *,
        runtime_ops,
        vllm_ops,
        policy: SleepPolicy,
        gateway: GatewayState | None = None,
        journal: SleepJournal | None = None,
        reservations: SleepReservations | None = None,
        clock: Clock | None = None,
        owner: str = "service-manager",
        operation_id: Callable[[], str | None] | None = None,
        floor_guard: Callable[[list[SleepTarget], str], None] | None = None,
        floor_lock=None,
    ) -> None:
        self._runtime = runtime_ops
        # Replica floor (2026-09-29): ``floor_guard(targets, path)`` raises when the
        # sleep would take a model below its floor; it runs right before the targets
        # are reserved and hidden, under ``floor_lock`` together with the hide so no
        # other hide / sleep of this process passes the check in between.
        self._floor_guard = floor_guard
        self._floor_lock = floor_lock
        self._vllm = vllm_ops
        self._policy = policy
        self._gateway = gateway
        self._journal = journal or SleepJournal()
        self._reservations = reservations or SleepReservations()
        self._clock = clock
        self._owner = owner
        self._operation_id = operation_id or (lambda: None)
        self._recent: deque[dict] = deque(maxlen=RECENT_OUTCOMES)
        self._shutdown = threading.Event()
        self._active: set[int] = set()
        self._active_lock = threading.Condition()
        self._mode_cache: dict[tuple[str, str, str | None], bool] = {}

    @property
    def policy(self) -> SleepPolicy:
        return self._policy

    @property
    def journal(self) -> SleepJournal:
        return self._journal

    @property
    def reservations(self) -> SleepReservations:
        return self._reservations

    def recent(self) -> list[dict]:
        return list(self._recent)

    # ---------------------------------------------------------------- shutdown
    def begin_shutdown(self) -> None:
        """Refuse new sleeps; drains in progress roll back at their next poll."""
        self._shutdown.set()

    @property
    def shutting_down(self) -> bool:
        return self._shutdown.is_set()

    def active_count(self) -> int:
        with self._active_lock:
            return len(self._active)

    def wait_idle(self, timeout_s: float) -> bool:
        deadline = time.monotonic() + timeout_s
        with self._active_lock:
            while self._active:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return False
                self._active_lock.wait(remaining)
        return True

    # ------------------------------------------------------------------ public
    def sleep(
        self,
        targets: list[SleepTarget],
        *,
        path: str,
        drain_budget_s: float | None = None,
        journal_extra: Mapping | None = None,
    ) -> list[dict]:
        """prepare -> drain -> commit in one call (caller holds the writer lock).

        Several targets are hidden together and drain concurrently (one soft
        budget for the call). Returns one outcome per target, or raises a
        :class:`SleepFailed` whose ``outcomes`` says what happened to each.
        """
        if not targets:
            return []
        batch = self.prepare(
            targets, path=path, drain_budget_s=drain_budget_s, journal_extra=journal_extra
        )
        try:
            self.drain(batch)
        except ReservationLost as exc:
            # The caller holds the writer lock: resolve the lost reservation now.
            exc.outcomes = self.resolve_lost(batch)
            raise
        return self.commit(batch)

    def prepare(
        self,
        targets: list[SleepTarget],
        *,
        path: str,
        drain_budget_s: float | None = None,
        journal_extra: Mapping | None = None,
    ) -> SleepBatch:
        """Reserve the bindings, journal and hide every target (writer lock held).

        ``journal_extra`` is recorded in each target's journal entry (e.g. the
        desired power to record once the pod is confirmed asleep, which crash
        recovery applies when it finds the pod asleep)."""
        if self._shutdown.is_set():
            raise ServiceShuttingDown("service-manager is shutting down; no new sleep")
        with self._floor_lock if self._floor_lock is not None else nullcontext():
            if self._floor_guard is not None:
                self._floor_guard(list(targets), path)
            return self._prepare_unguarded(
                targets, path=path, drain_budget_s=drain_budget_s, journal_extra=journal_extra
            )

    def _prepare_unguarded(
        self,
        targets: list[SleepTarget],
        *,
        path: str,
        drain_budget_s: float | None,
        journal_extra: Mapping | None,
    ) -> SleepBatch:
        clock = self._clock or DEFAULT_CLOCK
        soft_s = self._policy.soft_budget_s(path, drain_budget_s)
        batch = SleepBatch(
            pods=[_PodSleep(target, _previous_state(target.binding)) for target in targets],
            path=path,
            soft_s=soft_s,
            hard_s=self._policy.hard_cap_s,
            started=clock.monotonic(),
            operation_id=self._operation_id(),
            journal_extra=dict(journal_extra or {}),
            no_drain=self._policy.no_drain(path),
        )
        batch.token = self._reservations.acquire(
            [target.binding for target in targets],
            owner=self._owner,
            operation_id=batch.operation_id,
            ttl_s=self._policy.reservation_ttl_s,
        )
        for pod in batch.pods:
            pod.reserved = True
        self._register(batch)
        try:
            self._hide(batch)
        except BaseException as exc:
            self._rollback_all(batch, reason=f"hide failed: {type(exc).__name__}: {exc}")
            self._unregister(batch)
            if isinstance(exc, SleepFailed):
                exc.outcomes = batch.outcomes()
            raise
        return batch

    def drain(self, batch: SleepBatch) -> None:
        """Gateway ack + drain; decides wait / abort / rollback per pod.

        Needs no writer lock (the reservation fences the bindings). Raises after
        rolling back every target on a batch-wide failure (ack timeout, shutdown,
        writer fence). A LOST reservation is different (review 2 P2-1): another
        sleep may own the binding by now, so the drain touches nothing and raises
        :class:`ReservationLost`; the caller resolves it under the writer lock
        (:meth:`resolve_lost`), else crash recovery does.

        Every round reads the engine metrics of all pending targets in parallel
        and decides with the time measured AFTER the reads, so a target still
        pending at the hard cap is decided in the round that crosses it: the
        drain ends within the hard cap plus one round.
        """
        clock = self._clock or DEFAULT_CLOCK
        try:
            batch.ack = self._await_ack(batch, clock)
            soft_deadline = batch.started + batch.soft_s
            hard_deadline = batch.started + max(batch.hard_s, batch.soft_s)
            pending = [
                pod
                for pod in batch.pods
                if pod.decision is None and not pod.done and pod.outcome is None
            ]
            while pending:
                self._check_alive(batch)
                loads = self._loads(pending, batch.ack)
                now = clock.monotonic()
                for pod in list(pending):
                    load = loads[id(pod)]
                    pod.last_load = load
                    verdict = _decide(
                        pod, load, now, soft_deadline, hard_deadline, no_drain=batch.no_drain
                    )
                    if verdict is None:
                        continue
                    pending.remove(pod)
                    pod.waited_s = now - batch.started
                    if verdict in ("wait", "abort"):
                        pod.decision = verdict
                        self._journal.update(
                            pod.pod,
                            phase="drained"
                            if verdict == "wait"
                            else ("no_drain_abort" if batch.no_drain else "drain_budget_spent"),
                            in_flight=load.get("in_flight"),
                        )
                    else:
                        self._journal.incr(verdict)
                        self._rollback_pod(batch, pod, reason=_ROLLBACK_REASONS[verdict])
                if pending:
                    now = clock.monotonic()
                    next_deadline = soft_deadline if now < soft_deadline else hard_deadline
                    clock.sleep(
                        max(0.0, min(self._policy.poll_interval_s, next_deadline - now))
                        or min(self._policy.poll_interval_s, 0.01)
                    )
        except ReservationLost:
            batch.reservation_lost = True
            raise
        except BaseException as exc:
            self._rollback_all(batch, reason=f"{type(exc).__name__}: {exc}")
            self._unregister(batch)
            if isinstance(exc, SleepFailed):
                exc.outcomes = batch.outcomes()
            raise

    def resolve_lost(self, batch: SleepBatch) -> list[dict]:
        """Resolve a batch whose reservation was lost (caller holds the writer
        lock, so no other sleep can start meanwhile). Never re-acquires: a
        binding now reserved by another sleep is left to it (outcome
        ``reservation_lost``). Every other unfinished target is probed first
        (review 3 P3): once the reservation expired another sleep may have taken
        the binding, slept it and finished - a pod found ASLEEP is recorded
        asleep (routing is never re-opened on it), one whose state cannot be
        read stays hidden (``unconfirmed``), only an awake one is rolled back."""
        try:
            live = self._reservations.active()
            unresolved: list[_PodSleep] = []
            for pod in batch.pods:
                if pod.done or pod.outcome is not None:
                    continue
                other = live.get(pod.binding_id)
                if other is not None and other.token != batch.token:
                    pod.reserved = False
                    self._end_journal_if_ours(pod, batch)
                    pod.outcome = self._outcome(
                        batch,
                        pod,
                        STATUS_RESERVATION_LOST,
                        reason=f"reservation lost; binding now reserved by {other.owner}",
                    )
                    continue
                unresolved.append(pod)
            physical = _parallel(
                lambda pod: self._physical(pod.target) if pod.hidden and not pod.sleep_called else None,
                unresolved,
            )
            for pod in unresolved:
                reason = "sleep reservation lost (expired)"
                if pod.hidden and not pod.sleep_called:
                    state = physical[id(pod)]
                    if state is True:
                        self._finalize_slept(
                            batch, pod, mode=None, forced=False, forced_count=0,
                            reason=f"{reason}; found asleep (slept by another owner)",
                            count=False,
                        )
                        continue
                    if state is None:
                        self._mark_unconfirmed(
                            batch, pod, reason=f"{reason}; physical state unknown, left hidden"
                        )
                        self._release(batch, [pod])
                        continue
                self._rollback_pod(batch, pod, reason=reason)
        finally:
            self._unregister(batch)
        return batch.outcomes()

    def commit(self, batch: SleepBatch) -> list[dict]:
        """/sleep + physical confirmation for every drained pod (writer lock held).

        The reservation is renewed first - a lost one rolls the batch back
        (never re-acquired, review 2 P2-1) - and then on every wait round while
        the targets are committed IN PARALLEL (review 2 P2-2): one thread per
        target sends /sleep (and its fallbacks), then one loop confirms
        ``/is_sleeping`` for all of them."""
        clock = self._clock or DEFAULT_CLOCK
        try:
            try:
                owned = self._renew(batch)
            except Exception as exc:  # Redis error: ownership unknown, nothing slept yet
                self._journal.incr("reservation_lost_total")
                self._rollback_all(
                    batch, reason=f"reservation renewal failed before commit: {type(exc).__name__}: {exc}"
                )
                owned = True
            if not owned:
                self._journal.incr("reservation_lost_total")
                self.resolve_lost(batch)
            ready = [
                pod
                for pod in batch.pods
                if not pod.done and pod.outcome is None and pod.decision is not None
            ]
            if self._shutdown.is_set():
                _parallel(
                    lambda pod: self._rollback_pod(batch, pod, reason="service-manager is shutting down"),
                    ready,
                )
                ready = []
            if ready:
                self._send_all(batch, ready, clock)
                self._confirm_all(
                    batch, [pod for pod in ready if pod.outcome is None and not pod.done], clock
                )
        finally:
            self._unregister(batch)
        outcomes = batch.outcomes()
        failed = [item for item in outcomes if item.get("status") != STATUS_SLEPT]
        if failed:
            raise SleepIncomplete(
                "sleep incomplete for "
                + ", ".join(
                    f"{item['serve_id']} ({item['status']}: {item.get('reason')})"
                    for item in failed
                ),
                outcomes=outcomes,
            )
        return outcomes

    def abandon(self, batch: SleepBatch, *, reason: str) -> list[dict]:
        """Roll back every target not yet slept (e.g. the commit phase could not
        take the writer lock)."""
        try:
            self._rollback_all(batch, reason=reason)
        finally:
            self._unregister(batch)
        return batch.outcomes()

    def release_unresolved(self, batch: SleepBatch) -> None:
        """Give up a batch without touching its pods (its reservation was lost and
        the writer lock is unavailable): crash recovery resolves the journal."""
        self._unregister(batch)

    # ------------------------------------------------------------------- steps
    def _hide(self, batch: SleepBatch) -> None:
        for pod in batch.pods:
            binding = pod.target.binding
            record = {
                "binding_id": binding.binding_id,
                "serve_id": binding.serve_id,
                "model": binding.model,
                "node": binding.slot.node,
                "gpu_ids": list(binding.slot.gpu_ids),
                "pod_ip": pod.target.pod_ip,
                "path": batch.path,
                "soft_budget_s": batch.soft_s,
                "hard_cap_s": batch.hard_s,
                "drain_policy": "no_drain" if batch.no_drain else "drain",
                "phase": "hiding",
                "previous_state": pod.previous_state,
                "owner": self._owner,
                "operation_id": batch.operation_id,
                "reservation_token": batch.token,
                "started_at_ms": int(time.time() * 1000),
            }
            record.update(batch.journal_extra)
            self._journal.begin(pod.pod, record)
            gen = self._runtime.write_binding_annotations(binding, state=POD_STATE_HIDDEN)
            pod.hidden = True
            pod.gen = gen if isinstance(gen, int) and not isinstance(gen, bool) else None
            self._journal.update(pod.pod, phase="awaiting_ack", target_gen=pod.gen)

    def _check_alive(self, batch: SleepBatch) -> None:
        """Every poll: shutdown, reservation ownership, writer fence (if held)."""
        if self._shutdown.is_set():
            raise SleepCancelled("service-manager is shutting down: rolling the drain back")
        if not self._renew(batch):
            self._journal.incr("reservation_lost_total")
            raise ReservationLost(
                f"sleep reservation of {[pod.binding_id for pod in batch.pods if pod.reserved]} was lost"
            )
        operation = current_operation()
        if operation is not None:
            operation.assert_active()

    def _renew(self, batch: SleepBatch) -> bool:
        """Renew the batch's reservation (False = lost). Redis errors propagate."""
        with batch.lock:  # a commit thread may be releasing one of them right now
            reserved = [
                pod.binding_id for pod in batch.pods if pod.reserved and pod.outcome is None
            ]
            if batch.token is None or not reserved:
                return True
            return self._reservations.renew(
                reserved, batch.token, ttl_s=self._policy.reservation_ttl_s
            )

    def _await_ack(self, batch: SleepBatch, clock: Clock) -> dict:
        policy = self._policy
        pods = [pod for pod in batch.pods if pod.outcome is None]
        started = clock.monotonic()
        deadline = started + policy.ack_timeout_s
        while True:
            self._check_alive(batch)
            error: str | None = None
            unacked: list[tuple[str, str]] = []
            live: list[str] = []
            pending: list[str] = []
            try:
                if self._gateway is not None:
                    live, pending = self._gateway.instances(policy.instance_staleness_s)
                if len(live) < policy.gateway_min_instances:
                    unacked = [
                        (
                            "*",
                            f"{len(live)} live plugin instance(s) < gateway_min_instances "
                            f"{policy.gateway_min_instances}",
                        )
                    ]
                else:
                    unacked = [
                        (pod.pod, instance)
                        for pod in pods
                        for instance in live
                        if not _acked(self._gateway.seen(pod.pod).get(instance), pod.gen)
                    ]
            except Exception as exc:  # Redis / k8s read error: not converged
                error = f"{type(exc).__name__}: {exc}"
                unacked = [(pod.pod, "<read error>") for pod in pods]
            if self._gateway is None:
                # This primitive is not wired to any gateway state at all (embedded /
                # unit-test use; the server always wires one): nothing can ack.
                return self._fallback_ack(batch, pods, clock, wired=False)
            if error is None and not live and not pending and policy.fallback_no_plugin:
                return self._fallback_ack(batch, pods, clock)
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
            clock.sleep(min(policy.poll_interval_s, max(0.0, deadline - now)) or 0.01)

    def _fallback_ack(
        self, batch: SleepBatch, pods: list[_PodSleep], clock: Clock, *, wired: bool = True
    ) -> dict:
        policy = self._policy
        LOG.warning(
            "%s: hiding %s on the k8s label and a %.1fs grace delay (no in-flight view "
            "from the gateway)",
            "no live gateway plugin instance and fallback_no_plugin is enabled"
            if wired
            else "no gateway state wired into this service-manager",
            [pod.pod for pod in pods],
            policy.no_plugin_grace_s,
        )
        self._journal.incr("ack_fallback_total" if wired else "ack_no_gateway_total")
        wait_unroutable = getattr(self._runtime, "wait_pod_unroutable", None)
        if callable(wait_unroutable):
            for pod in pods:
                wait_unroutable(pod.target.binding, timeout_s=policy.ack_timeout_s)
        clock.sleep(policy.no_plugin_grace_s)
        for pod in pods:
            self._journal.update(pod.pod, phase="draining", ack="fallback_no_plugin")
        return {"mode": "fallback_no_plugin", "instances": [], "latency_ms": None}

    def _loads(self, pods: list[_PodSleep], ack: dict | None) -> dict[int, dict]:
        """One drain round: engine metrics of every pod read in parallel (one
        probe timeout per round, whatever the number of targets), then the
        gateway side per pod."""
        engine = _parallel(self._engine_load, pods)
        return {id(pod): self._load(pod, ack, engine_load=engine[id(pod)]) for pod in pods}

    def _engine_load(self, pod: _PodSleep) -> int | None:
        metrics = getattr(self._vllm, "metrics", None)
        if not callable(metrics):
            return None
        try:
            return parse_vllm_load(metrics(pod.target.pod_ip, port=VLLM_PORT))
        except Exception:  # an unreachable engine reports nothing
            return None

    def _load(self, pod: _PodSleep, ack: dict | None, *, engine_load=_UNSET) -> dict:
        """One read of the drain state. ``known`` is False on any read error or
        unavailable engine metrics; ``non_continuable`` is the sticky value."""
        gateway_known = True
        gateway_total = 0
        if ack is not None and ack.get("mode") == "plugin" and self._gateway is not None:
            # Read only after the ack: the plugin writes a pod's inflight before its
            # seen field in one pipeline, so this value is at least as new as the ack.
            # A live instance without a field has nothing in flight (0); fields of
            # instances that are not live are ignored entirely (a crashed instance's
            # fields never expire while other instances refresh the hash TTL).
            try:
                live = set(self._gateway.live_instances(self._policy.instance_staleness_s))
                entries = self._gateway.inflight(pod.pod)
            except Exception:
                gateway_known = False
                entries = {}
                live = set()
            non_continuable = 0
            for instance, entry in entries.items():
                if instance not in live:
                    continue
                if entry is None:
                    gateway_known = False  # unreadable entry of a live instance
                    continue
                try:
                    gateway_total += max(0, int(entry.get("total", 0) or 0))
                    non_continuable += max(0, int(entry.get("non_continuable", 0) or 0))
                except (TypeError, ValueError):
                    gateway_known = False
            if gateway_known:
                pod.non_continuable = non_continuable
        if engine_load is _UNSET:
            engine_load = self._engine_load(pod)
        known = gateway_known and engine_load is not None
        in_flight = max(gateway_total, engine_load or 0)
        return {
            "known": known,
            "gateway_read_error": not gateway_known,
            "gateway_inflight": gateway_total if gateway_known else None,
            "non_continuable": pod.non_continuable,
            "engine_load": engine_load,
            "in_flight": in_flight,
            "drained": known and in_flight == 0,
        }

    def _mode_supported(self, target: SleepTarget) -> bool:
        setting = str(self._policy.vllm_sleep_mode_param).lower()
        if setting == "true":
            return True
        if setting == "false":
            return False
        key = (target.binding.serve_id, target.pod_ip, target.pod_uid)
        cached = self._mode_cache.get(key)
        if cached is not None:
            return cached
        version_fn = getattr(self._vllm, "version", None)
        if not callable(version_fn):
            return False
        try:
            version = version_fn(target.pod_ip, port=VLLM_PORT)
        except Exception:
            version = None
        if parse_vllm_version(version) is None:
            # Unknown / unparseable: no mode parameter this time (the drain makes a
            # plain /sleep safe); not cached, so a later sleep asks again.
            return False
        supported = sleep_mode_supported(version)
        self._mode_cache[key] = supported
        return supported

    def _send_all(self, batch: SleepBatch, pods: list[_PodSleep], clock: Clock) -> None:
        """Phase 1 of the commit: /sleep (with its fallbacks) for every pod in
        parallel; the reservation is renewed while the calls run."""
        context = contextvars.copy_context()
        with ThreadPoolExecutor(max_workers=max(1, len(pods))) as pool:
            futures = [
                pool.submit(context.copy().run, self._send_guarded, batch, pod) for pod in pods
            ]
            while True:
                done, not_done = wait(futures, timeout=self._policy.poll_interval_s)
                if not not_done:
                    break
                self._renew_during_commit(batch)
        for future in futures:
            future.result()  # _send_guarded never raises; surfaces programming errors

    def _send_guarded(self, batch: SleepBatch, pod: _PodSleep) -> None:
        try:
            self._send_one(batch, pod)
        except Exception as exc:
            self._rollback_pod(batch, pod, reason=f"{type(exc).__name__}: {exc}")

    def _send_one(self, batch: SleepBatch, pod: _PodSleep) -> None:
        """/sleep one drained pod. Leaves ``pod.commit`` set when the pod must be
        confirmed physically; an outcome when it was decided here."""
        target = pod.target
        policy = self._policy
        load = pod.last_load
        drained = pod.decision == "wait"
        supports_mode = self._mode_supported(target)
        mode: str | None = ("wait" if drained else "abort") if supports_mode else None
        forced = not drained
        forced_count = int(load.get("in_flight") or 0) if forced else 0
        aborted = _abort_breakdown(load) if forced else None
        self._journal.update(pod.pod, phase="sleeping", drained=drained, sleep_mode=mode)
        pod.sleep_called = True
        result = self._vllm.sleep(
            target.pod_ip,
            port=VLLM_PORT,
            mode=mode,
            timeout_s=policy.sleep_call_timeout_s,
            hidden=True,
        )
        if not _success(result) and self._physical(target) is not True:
            if mode != "wait":
                message = getattr(result, "message", "") or "operation failed"
                raise SleepFailed(f"vLLM sleep failed for {target.binding.serve_id}: {message}")
            # A straggler kept mode=wait from finishing within the call timeout.
            # Abort only when the drain state is KNOWN and nothing non-continuable
            # is in flight (a no-drain path aborts regardless); otherwise leave the
            # pod hidden for the audit/recovery.
            again = self._load(pod, batch.ack)
            pod.last_load = again
            if not batch.no_drain and (not again["known"] or (pod.non_continuable or 0) > 0):
                self._mark_unconfirmed(
                    batch,
                    pod,
                    reason="mode=wait did not finish and the drain state is unknown "
                    "or non-continuable",
                )
                return
            forced = True
            forced_count = max(1, int(again.get("in_flight") or 0))
            aborted = _abort_breakdown(again)
            mode = "abort"
            result = self._vllm.sleep(
                target.pod_ip,
                port=VLLM_PORT,
                mode="abort",
                timeout_s=policy.sleep_call_timeout_s,
                hidden=True,
            )
            if not _success(result) and self._physical(target) is not True:
                message = getattr(result, "message", "") or "operation failed"
                raise SleepFailed(f"vLLM sleep failed for {target.binding.serve_id}: {message}")
        pod.commit = {
            "mode": mode,
            "forced": forced,
            "forced_count": forced_count,
            "aborted": aborted,
        }

    def _confirm_all(self, batch: SleepBatch, pods: list[_PodSleep], clock: Clock) -> None:
        """Phase 2 of the commit: poll ``/is_sleeping`` of every sent pod (in
        parallel) until each is asleep or ``physical_confirm_timeout_s`` passed.
        Asleep -> slept; still unknown -> unconfirmed (stays hidden); awake ->
        rolled back. (vLLM ops without the probe: the /sleep result is trusted.)"""
        pending = [pod for pod in pods if pod.commit is not None]
        if not pending:
            return
        probe_available = callable(getattr(self._vllm, "is_sleeping", None))
        deadline = clock.monotonic() + self._policy.physical_confirm_timeout_s
        last: dict[int, bool | None] = {}
        while pending:
            if probe_available:
                answers = _parallel(lambda pod: self._physical(pod.target), pending)
            else:
                answers = {id(pod): True for pod in pending}
            confirmed = []
            for pod in list(pending):
                last[id(pod)] = answers[id(pod)]
                if answers[id(pod)] is True:
                    pending.remove(pod)
                    confirmed.append(pod)
            # One k8s annotation write per confirmed pod, in parallel (review 4
            # P3): the phase stays bounded whatever the number of targets.
            _parallel(lambda pod: self._finalize_slept(batch, pod, **pod.commit), confirmed)
            if not pending:
                return
            if clock.monotonic() >= deadline:
                break
            self._renew_during_commit(batch)
            clock.sleep(self._policy.poll_interval_s)
        # The rollbacks re-probe each pod (_rollback_pod): run them in parallel so
        # the phase stays bounded whatever the number of targets.
        def resolve(pod: _PodSleep) -> None:
            if last.get(id(pod)) is None:
                self._mark_unconfirmed(
                    batch, pod, reason="/sleep returned but /is_sleeping stayed unknown"
                )
            else:
                self._rollback_pod(
                    batch,
                    pod,
                    reason=f"vLLM sleep did not physically converge for {pod.pod}",
                )

        _parallel(resolve, pending)

    def _renew_during_commit(self, batch: SleepBatch) -> None:
        """Keep the reservation alive during a long commit (the writer lock fences
        new sleeps meanwhile; /sleep calls already sent cannot be undone, so a
        lost renewal is only counted and logged)."""
        try:
            owned = self._renew(batch)
        except Exception as exc:
            LOG.warning("renewing the sleep reservation during commit failed: %s", exc)
            return
        if not owned:
            self._journal.incr("reservation_lost_during_commit_total")
            LOG.error(
                "sleep reservation of %s lost during the commit phase (writer lock held)",
                [pod.binding_id for pod in batch.pods if pod.reserved],
            )

    def _finalize_slept(
        self,
        batch: SleepBatch,
        pod: _PodSleep,
        *,
        mode: str | None,
        forced: bool,
        forced_count: int,
        reason: str | None = None,
        count: bool = True,
        aborted: dict | None = None,
    ) -> None:
        self._runtime.write_binding_annotations(pod.target.binding, state=POD_STATE_SLEEPING)
        pod.done = True
        self._end_journal_if_ours(pod, batch)
        self._release(batch, [pod])
        if count:  # not a sleep of this batch (found asleep): no sleep counters
            self._journal.incr("sleeps_total")
            self._journal.incr(f"sleeps_path_{batch.path}")
        if forced:
            self._journal.incr("forced_abort_total")
            self._journal.incr("forced_abort_requests_total", forced_count)
        if count and batch.no_drain:
            self._journal.incr("no_drain_sleeps_total")
            if forced and aborted is not None:
                truncated = int(aborted.get("non_continuable") or 0)
                if truncated:
                    self._journal.incr("no_drain_non_continuable_aborted_total", truncated)
                unclassified = int(aborted.get("unclassified") or 0)
                if unclassified:
                    self._journal.incr("no_drain_unclassified_aborted_total", unclassified)
                if not aborted.get("state_known", True):
                    self._journal.incr("no_drain_unknown_state_abort_total")
        pod.outcome = self._outcome(
            batch,
            pod,
            STATUS_SLEPT,
            reason=reason,
            sleep_mode=mode,
            forced_abort=forced,
            forced_abort_requests=forced_count,
            aborted=aborted if forced else None,
        )
        log = LOG.warning if forced else LOG.info
        log("sleep %s", json.dumps(pod.outcome, sort_keys=True))

    def _mark_unconfirmed(self, batch: SleepBatch, pod: _PodSleep, *, reason: str) -> None:
        """/sleep was sent but the pod is not confirmed asleep: keep it hidden and
        keep the journal entry (audit ``sleep_unconfirmed``; recovery resolves it).
        Another sleep's journal entry of the pod is never overwritten."""
        if self._journal_is_ours(pod, batch):
            self._journal.update(pod.pod, phase="sleep_unconfirmed", reason=reason)
        self._journal.incr("sleep_unconfirmed_total")
        pod.outcome = self._outcome(batch, pod, STATUS_UNCONFIRMED, reason=reason)
        LOG.error("sleep of %s unconfirmed, pod stays hidden: %s", pod.pod, reason)

    def _physical(self, target: SleepTarget) -> bool | None:
        probe = getattr(self._vllm, "is_sleeping", None)
        if not callable(probe):
            return None
        try:
            return probe(target.pod_ip, port=VLLM_PORT)
        except Exception:
            return None

    def _rollback_all(self, batch: SleepBatch, *, reason: str) -> None:
        # In parallel (review 4 P3): each rollback may re-probe the pod and write
        # its annotations; _rollback_pod never raises.
        _parallel(
            lambda pod: self._rollback_pod(batch, pod, reason=reason),
            [pod for pod in batch.pods if pod.outcome is None],
        )

    def _rollback_pod(self, batch: SleepBatch, pod: _PodSleep, *, reason: str) -> None:
        if pod.done or pod.outcome is not None:
            return
        try:
            if not pod.hidden:
                self._end_journal_if_ours(pod, batch)
                self._release(batch, [pod])
                pod.outcome = self._outcome(batch, pod, STATUS_ROLLED_BACK, reason=reason)
                return
            if pod.sleep_called:
                physical = self._physical(pod.target)
                if physical is True:
                    # It did go to sleep; record the truth instead of re-routing.
                    forced = pod.decision != "wait"
                    self._finalize_slept(
                        batch,
                        pod,
                        mode=None,
                        forced=forced,
                        forced_count=int(pod.last_load.get("in_flight") or 0) if forced else 0,
                        reason=reason,
                        aborted=_abort_breakdown(pod.last_load) if forced else None,
                    )
                    return
                if physical is None:
                    self._mark_unconfirmed(batch, pod, reason=f"{reason}; physical state unknown")
                    return
            self._runtime.write_binding_annotations(
                pod.target.binding, state=pod.previous_state
            )
            self._end_journal_if_ours(pod, batch)
            self._release(batch, [pod])
            self._journal.incr("rollback_total")
            pod.outcome = self._outcome(batch, pod, STATUS_ROLLED_BACK, reason=reason)
            LOG.warning("sleep of %s rolled back to %s: %s", pod.pod, pod.previous_state, reason)
        except Exception as exc:  # keep the original failure; the journal stays as evidence
            LOG.exception("rollback of %s failed; journal entry kept", pod.pod)
            try:
                self._journal.update(pod.pod, phase="rollback_failed", reason=f"{reason}; {exc}")
            except Exception:
                pass
            pod.outcome = self._outcome(
                batch, pod, STATUS_ROLLBACK_FAILED, reason=f"{reason}; rollback failed: {exc}"
            )

    def _release(self, batch: SleepBatch, pods: list[_PodSleep]) -> None:
        with batch.lock:
            ids = [pod.binding_id for pod in pods if pod.reserved]
            for pod in pods:
                pod.reserved = False
            if ids and batch.token is not None:
                try:
                    self._reservations.release(ids, batch.token)
                except Exception:  # expires by itself
                    LOG.exception("releasing sleep reservation of %s failed", ids)

    def _journal_is_ours(self, pod: _PodSleep, batch: SleepBatch) -> bool:
        """The pod's journal entry is this batch's (or there is none): another
        sleep may have written its own since (the journal is keyed by pod)."""
        try:
            entry = self._journal.get(pod.pod) or {}
        except Exception:  # Redis read error: trust this process's own last write
            entry = self._journal.cached(pod.pod) or {}
        return entry.get("reservation_token") in (None, batch.token)

    def _end_journal_if_ours(self, pod: _PodSleep, batch: SleepBatch) -> None:
        if self._journal_is_ours(pod, batch):
            self._journal.end(pod.pod)

    def _outcome(
        self, batch: SleepBatch, pod: _PodSleep, status: str, *, reason=None, **extra
    ) -> dict:
        ack = batch.ack or {}
        outcome = {
            "serve_id": pod.pod,
            "binding_id": pod.binding_id,
            "path": batch.path,
            "drain_policy": "no_drain" if batch.no_drain else "drain",
            "status": status,
            "reason": reason,
            "previous_state": pod.previous_state,
            "ack_mode": ack.get("mode"),
            "ack_latency_ms": ack.get("latency_ms"),
            "drained": pod.decision == "wait",
            "sleep_mode": None,
            "forced_abort": False,
            "forced_abort_requests": 0,
            "aborted": None,
            "non_continuable_at_sleep": pod.last_load.get("non_continuable") or 0,
            "waited_s": round(pod.waited_s, 3),
        }
        outcome.update(extra)
        self._recent.append(outcome)
        return outcome

    def _register(self, batch: SleepBatch) -> None:
        with self._active_lock:
            self._active.add(id(batch))

    def _unregister(self, batch: SleepBatch) -> None:
        with self._active_lock:
            self._active.discard(id(batch))
            self._active_lock.notify_all()


_ROLLBACK_REASONS = {
    "drain_unknown_rollback_total": (
        "drain state still unknown at the hard cap (read errors / metrics unavailable): "
        "rolled back, never aborted blind"
    ),
    "non_continuable_rollback_total": (
        "non-continuable requests still in flight at the hard cap: rolled back, never aborted"
    ),
}


def _parallel(fn, items: list) -> dict[int, object]:
    """``fn`` over ``items`` (by ``id``): inline for one item, else one thread
    per item, so a round costs one call duration whatever the target count."""
    if len(items) <= 1:
        return {id(item): fn(item) for item in items}
    context = contextvars.copy_context()
    with ThreadPoolExecutor(max_workers=len(items)) as pool:
        futures = {id(item): pool.submit(context.copy().run, fn, item) for item in items}
        return {key: future.result() for key, future in futures.items()}


def _decide(
    pod: _PodSleep,
    load: dict,
    now: float,
    soft_deadline: float,
    hard_deadline: float,
    *,
    no_drain: bool = False,
) -> str | None:
    """None = keep waiting; "wait" / "abort" = go to /sleep; else a rollback counter.

    ``no_drain``: never wait - anything in flight (or an unknown state) aborts."""
    if load["known"] and load["in_flight"] == 0:
        return "wait"
    if no_drain:
        return "abort"
    if now < soft_deadline:
        return None
    if not load["known"]:
        return None if now < hard_deadline else "drain_unknown_rollback_total"
    if (pod.non_continuable or 0) > 0:
        return None if now < hard_deadline else "non_continuable_rollback_total"
    return "abort"


def _abort_breakdown(load: Mapping) -> dict:
    """What a forced /sleep mode=abort cuts off, from one drain read: the
    continuable requests (the reissue sidecar continues them), the gateway's
    non-continuable ones (truncated, or re-run from scratch by the sidecar when
    nothing was streamed yet) and engine-side requests the gateway did not count
    (``unclassified``). ``state_known`` False: the counts are a lower bound.
    The read happens BEFORE ``/sleep mode=abort`` is sent: requests that finish
    normally in between are counted here but never aborted (the engine sends them
    no abort output, so the sidecar has nothing to continue). These counts are
    therefore an upper bound of what the abort cut off; e.g. the 2026-09-30 smoke
    counted 135 forced-abort requests against 133 sidecar continuations, the two
    missing ones having completed with 200 about 30 ms before the abort."""
    gateway = load.get("gateway_inflight")
    non_continuable = max(0, int(load.get("non_continuable") or 0))
    in_flight = max(0, int(load.get("in_flight") or 0))
    gateway_total = max(0, int(gateway or 0))
    return {
        "state_known": bool(load.get("known")),
        "in_flight": in_flight,
        "continuable": max(0, gateway_total - non_continuable) if gateway is not None else None,
        "non_continuable": non_continuable,
        "unclassified": max(0, in_flight - gateway_total),
    }


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
