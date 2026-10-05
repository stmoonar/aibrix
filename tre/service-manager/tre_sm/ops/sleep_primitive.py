"""The one way the service-manager puts a vLLM pod to sleep (plan 2026-09-27 D1-D4).

Every sleep - SafeScale commit, controller scale-downs and transfers, APA
scale-downs, defrag, fleet repair, startup admission, a failed wake's
compensating sleep - runs the same ordered protocol, and callers cannot skip a
step. Since 2026-10-02 (design ``docs/design/20261002-sm-wholelock.md``) the
WHOLE protocol runs under the service-manager writer lock, which the caller
holds from before the hide until the outcome is in the books; the targets of one
call go through every step in parallel. There is no sleep reservation, no
phase without the lock and no hand-over between phases: the writer lock is the
only fence, the sleep journal the only crash evidence.

1. **floor**: ``floor_guard(targets, path)`` (the replica floor) raises before
   anything is hidden.
2. **hide**: a journal entry per pod, then one Pod patch per pod
   (``tre.aibrix.io/routable=false``, the ``hidden`` state annotation, a new
   ``tre.aibrix.io/route-gen``).
3. **gateway ack** (``ack_timeout_s``): at least ``gateway_min_instances`` live
   plugin instances (heartbeat score advancing, or a Ready plugin pod) and every
   live instance reports ``gen >= target`` / ``routable=false`` for the pod. An
   empty live set never converges (unless the opt-in ``fallback_no_plugin``).
   Timeout, shutdown or a lost writer fence -> every target is rolled back.
4. **one load read** for the record (no drain - draining is the controller's
   job): the plugin in-flight counts of live instances and vLLM's running +
   waiting gauges, together with the pod's ``GET /version`` (cached per pod),
   one probe round for every target in parallel.
5. **one /sleep** per pod, in parallel (``sleep_call_timeout_s``): ``mode=abort``
   (``sleep_mode_when_idle: wait`` opts into mode=wait when nothing is in
   flight). Requests in flight are cut off and continued by the reissue
   sidecar; the outcome's ``aborted`` field counts them. An engine without the
   mode parameter gets no plain /sleep while requests are in flight (or the load
   is unknown): that target is rolled back (``vllm_sleep_mode_param: false`` is
   the operator's explicit opt-in to a plain /sleep).
6. **physical confirmation** (``physical_confirm_timeout_s``): ``/is_sleeping``
   of every sent pod, in parallel, until each reads asleep.

Outcome per target (``status``):

* ``slept``: confirmed asleep -> ``sleeping`` annotation; the caller releases
  the GPU lease, updates the store / desired state and then ends the journal
  entry (:meth:`SleepPrimitive.end_journal`) - a crash in between leaves the
  entry, and ``recover_sleep_journal`` records the sleep from the physical
  state;
* ``unconfirmed``: the /sleep call timed out (no HTTP answer), or the pod was
  not confirmed asleep within ``physical_confirm_timeout_s``, or its state
  cannot be read -> the pod STAYS HIDDEN, the journal entry is marked
  ``sleep_unconfirmed`` and the call returns at once (the writer lock is
  released); the journal recovery settles it from the physical state later
  (routing is only re-opened on a pod read awake twice, a
  ``sleep_call_timeout_s`` apart, and confirmed un-paused);
* ``rolled_back``: nothing was sent (ack timeout, shutdown, no mode parameter),
  or /sleep answered with an error and the pod reads awake and un-paused
  (``/is_paused``, ``/resume``) -> its previous routing state is restored under
  a new route-gen (a SafeScale probe pod stays hidden);
* ``rollback_failed``: the rollback itself failed; journal entry kept.

A call that does not sleep every target raises :class:`SleepIncomplete` with
one outcome per target, so callers account for exactly the pods that slept.
The duration of a call is bounded by
``ServiceManagerConfig.worst_case_sleep_lock_s`` whatever the target count.
"""

from __future__ import annotations

from collections import deque
from concurrent.futures import ThreadPoolExecutor
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

LOG = logging.getLogger("tre_sm.sleep")

VLLM_PORT = 8000
HIDDEN_SLEEP_HEADER = "X-TRE-Hidden"
RECENT_OUTCOMES = 256

#: First vLLM release whose ``POST /sleep`` accepts ``mode=wait|abort|keep``
#: (upstream PR #34528, "Cleanup engine pause/sleep logic", 2026-02-24; the
#: v0.18.0 release branch is the earliest verified to contain it; 0.10.1 reads
#: only ``level`` and ignores ``mode``). A version that does not parse (e.g. a
#: ``0.1.devN`` source build without tags) is treated as not supporting it.
SLEEP_MODE_MIN_VERSION = (0, 18, 0)

STATUS_SLEPT = "slept"
STATUS_ROLLED_BACK = "rolled_back"
STATUS_UNCONFIRMED = "unconfirmed"
STATUS_ROLLBACK_FAILED = "rollback_failed"

#: Journal phase of a pod confirmed asleep whose books the caller has not
#: written yet (the entry is ended by :meth:`SleepPrimitive.end_journal`).
PHASE_SLEPT = "slept"

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
    """The service-manager is shutting down: a sleep waiting for its ack rolls back."""


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


def log_ignored_drain_settings(policy: SleepPolicy, logger: logging.Logger | None = None) -> str:
    """One deprecation line at service-manager start (2026-10-02): the drain and
    reservation settings still parse (live registries and the controller still
    carry them) but the service-manager ignores them. Returns the message."""
    message = (
        "deprecated and IGNORED: service_manager.sleep.budgets_s=%s, "
        "service_manager.sleep.no_drain_paths=%s, service_manager.sleep.hard_cap_s=%g, "
        "service_manager.sleep.reservation_ttl_s=%g, service_manager.commit_lock_wait_s and "
        "the drain_budget_s request field - the service-manager never drains and holds its "
        "writer lock through every sleep (hide, gateway ack, one load read, one /sleep, "
        "physical confirmation)"
    ) % (
        json.dumps(dict(policy.budgets_s), sort_keys=True),
        list(policy.no_drain_paths),
        float(policy.hard_cap_s),
        float(policy.reservation_ttl_s),
    )
    (logger or LOG).warning(message)
    return message


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

    def now_ms(self) -> int:
        """Redis server time (ms) - one clock for every service-manager replica -
        or the local clock without Redis (or when it cannot tell)."""
        reader = getattr(self._redis, "time", None)
        if callable(reader):
            try:
                seconds, micros = reader()
                return int(seconds) * 1000 + int(micros) // 1000
            except Exception:  # noqa: BLE001 - fall back to the local clock
                pass
        return int(time.time() * 1000)

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
    #: "wait" (nothing in flight, state known) | "abort" (anything else) once the
    #: load read decided; None = not decided.
    decision: str | None = None
    #: Whether vLLM's /sleep takes the mode parameter (read with the load).
    supports_mode: bool | None = None
    sleep_called: bool = False
    #: The /sleep call got no HTTP answer (timeout / transport error).
    sleep_uncertain: bool = False
    done: bool = False
    #: Non-continuable requests in flight at the load read (gateway view).
    non_continuable: int | None = None
    last_load: dict = field(default_factory=dict)
    waited_s: float = 0.0
    outcome: dict | None = None
    #: Set once /sleep was answered with success and the pod awaits physical
    #: confirmation: {"mode", "forced", "forced_count", "aborted"}.
    commit: dict | None = None

    @property
    def pod(self) -> str:
        return self.target.binding.serve_id

    @property
    def binding_id(self) -> str:
        return self.target.binding.binding_id


@dataclass
class SleepBatch:
    """One sleep call's targets."""

    pods: list[_PodSleep]
    path: str
    started: float
    ack: dict | None = None
    operation_id: str | None = None
    #: Extra fields of every target's journal entry.
    journal_extra: dict = field(default_factory=dict)
    #: The caller ends the journal entry of a slept pod itself once its books
    #: are written (:meth:`SleepPrimitive.end_journal`).
    keep_journal: bool = False

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
        clock: Clock | None = None,
        owner: str = "service-manager",
        operation_id: Callable[[], str | None] | None = None,
        floor_guard: Callable[[list[SleepTarget], str], None] | None = None,
    ) -> None:
        self._runtime = runtime_ops
        # Replica floor (2026-09-29): ``floor_guard(targets, path)`` raises when the
        # sleep would take a model below its floor; it runs right before the
        # targets are hidden, under the caller's writer lock.
        self._floor_guard = floor_guard
        self._vllm = vllm_ops
        self._policy = policy
        self._gateway = gateway
        self._journal = journal or SleepJournal()
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

    def recent(self) -> list[dict]:
        return list(self._recent)

    # ---------------------------------------------------------------- shutdown
    def begin_shutdown(self) -> None:
        """Refuse new sleeps; a sleep waiting for its gateway ack rolls back at
        its next poll, a sleep past /sleep finishes."""
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
        keep_journal: bool = False,
    ) -> list[dict]:
        """The whole protocol (see the module docstring) for ``targets``, in
        parallel; the caller holds the writer lock throughout. Returns one
        outcome per target, or raises a :class:`SleepFailed` whose ``outcomes``
        says what happened to each. ``drain_budget_s`` is accepted and ignored
        (deprecated: the service-manager never drains). ``keep_journal``: the
        journal entry of a slept pod stays (phase ``slept``) until the caller
        ends it with :meth:`end_journal` after writing its books."""
        del drain_budget_s  # deprecated and ignored
        if not targets:
            return []
        if self._shutdown.is_set():
            raise ServiceShuttingDown("service-manager is shutting down; no new sleep")
        if self._floor_guard is not None:
            self._floor_guard(list(targets), path)
        clock = self._clock or DEFAULT_CLOCK
        batch = SleepBatch(
            pods=[_PodSleep(target, _previous_state(target.binding)) for target in targets],
            path=path,
            started=clock.monotonic(),
            operation_id=self._operation_id(),
            journal_extra=dict(journal_extra or {}),
            keep_journal=bool(keep_journal),
        )
        self._register(batch)
        try:
            try:
                self._hide(batch)
                batch.ack = self._await_ack(batch, clock)
                self._read_loads(batch, clock)
                self._check_alive()
            except BaseException as exc:
                # Nothing was sent: every target goes back to its routing state.
                self._rollback_all(batch, reason=f"{type(exc).__name__}: {exc}")
                if isinstance(exc, SleepFailed):
                    exc.outcomes = batch.outcomes()
                raise
            ready = [pod for pod in batch.pods if pod.outcome is None and pod.decision is not None]
            _parallel(lambda pod: self._send_guarded(batch, pod), ready)
            self._confirm_all(batch, [pod for pod in ready if pod.outcome is None and pod.commit is not None], clock)
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

    def end_journal(self, outcomes: list[dict]) -> None:
        """End the journal entry of every slept pod of ``outcomes`` (a call made
        with ``keep_journal``), once the caller's books record the sleep."""
        for item in outcomes or ():
            if item.get("status") != STATUS_SLEPT:
                continue
            pod = str(item.get("serve_id") or "")
            entry = self._journal.get(pod)
            if entry is not None and entry.get("phase") == PHASE_SLEPT:
                self._journal.end(pod)

    def confirm_unpaused(self, target: SleepTarget) -> bool:
        """True once the engine is confirmed NOT paused: ``/is_paused`` reads
        false, or ``/resume`` succeeded and ``/is_paused`` does not read true
        (an engine without /is_paused: resume's success alone). Never raises.
        vLLM ops without either call (embedded / unit-test wiring): True."""
        probe = getattr(self._vllm, "is_paused", None)
        resume = getattr(self._vllm, "resume", None)
        if not callable(probe) and not callable(resume):
            return True

        def read() -> bool | None:
            if not callable(probe):
                return None
            try:
                return probe(target.pod_ip, port=VLLM_PORT)
            except Exception:
                return None

        if read() is False:
            return True
        if not callable(resume):
            return False
        try:
            resumed = _success(resume(target.pod_ip, port=VLLM_PORT))
        except Exception:
            resumed = False
        if not resumed:
            return False
        self._journal.incr("resume_before_reopen_total")
        return read() is not True

    # ------------------------------------------------------------------- steps
    def _hide(self, batch: SleepBatch) -> None:
        """Journal + hide every target, in parallel (a k8s read + patch each:
        the hold does not grow with the number of targets, review 2026-10-06).
        Every target is attempted; the first error is raised afterwards and the
        caller rolls back what was hidden."""
        _parallel(lambda pod: self._hide_pod(batch, pod), list(batch.pods))

    def _hide_pod(self, batch: SleepBatch, pod: "_PodSleep") -> None:
        binding = pod.target.binding
        record = {
            "binding_id": binding.binding_id,
            "serve_id": binding.serve_id,
            "model": binding.model,
            "node": binding.slot.node,
            "gpu_ids": list(binding.slot.gpu_ids),
            "pod_ip": pod.target.pod_ip,
            "path": batch.path,
            "drain_policy": "no_drain",
            "phase": "hiding",
            "previous_state": pod.previous_state,
            "owner": self._owner,
            "operation_id": batch.operation_id,
            "started_at_ms": int(time.time() * 1000),
        }
        record.update(batch.journal_extra)
        self._journal.begin(pod.pod, record)
        gen = self._runtime.write_binding_annotations(binding, state=POD_STATE_HIDDEN)
        pod.hidden = True
        pod.gen = gen if isinstance(gen, int) and not isinstance(gen, bool) else None
        self._journal.update(pod.pod, phase="awaiting_ack", target_gen=pod.gen)

    def _check_alive(self) -> None:
        """Every ack poll and right before /sleep: shutdown, the writer fence."""
        if self._shutdown.is_set():
            raise SleepCancelled("service-manager is shutting down: rolling the sleep back")
        operation = current_operation()
        if operation is not None:
            operation.assert_active()

    def _await_ack(self, batch: SleepBatch, clock: Clock) -> dict:
        policy = self._policy
        pods = [pod for pod in batch.pods if pod.outcome is None]
        started = clock.monotonic()
        deadline = started + policy.ack_timeout_s
        while True:
            self._check_alive()
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
                    self._journal.update(pod.pod, phase="acked", ack="plugin")
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
            # In parallel: the whole wait stays within one ack_timeout_s, whatever
            # the target count.
            _parallel(
                lambda pod: wait_unroutable(pod.target.binding, timeout_s=policy.ack_timeout_s),
                pods,
            )
        clock.sleep(policy.no_plugin_grace_s)
        self._check_alive()
        for pod in pods:
            self._journal.update(pod.pod, phase="acked", ack="fallback_no_plugin")
        return {"mode": "fallback_no_plugin", "instances": [], "latency_ms": None}

    def _read_loads(self, batch: SleepBatch, clock: Clock) -> None:
        """ONE probe round for every target in parallel: the load (for the record
        and the wait / abort decision) and whether /sleep takes ``mode``."""
        pods = [pod for pod in batch.pods if pod.outcome is None]
        tasks = [(pod, "load") for pod in pods] + [(pod, "mode") for pod in pods]

        def read(task: tuple[_PodSleep, str]):
            pod, kind = task
            return self._engine_load(pod) if kind == "load" else self._mode_supported(pod.target)

        answers = _parallel(read, tasks)
        by_pod = {(id(pod), kind): answers[id(task)] for task in tasks for pod, kind in [task]}
        now = clock.monotonic()
        for pod in pods:
            engine_load, supports_mode = by_pod[(id(pod), "load")], by_pod[(id(pod), "mode")]
            load = self._load(pod, batch.ack, engine_load=engine_load)
            pod.last_load = load
            pod.supports_mode = supports_mode
            pod.waited_s = now - batch.started
            pod.decision = "wait" if load["known"] and load["in_flight"] == 0 else "abort"
            if pod.decision == "abort" and not load.get("known"):
                # Counted at the decision, whatever the sleep's outcome.
                self._journal.incr("no_drain_unknown_state_abort_total")
            self._journal.update(
                pod.pod,
                phase="idle" if pod.decision == "wait" else "in_flight_abort",
                in_flight=load.get("in_flight"),
            )

    def _engine_load(self, pod: _PodSleep) -> int | None:
        metrics = getattr(self._vllm, "metrics", None)
        if not callable(metrics):
            return None
        try:
            return parse_vllm_load(metrics(pod.target.pod_ip, port=VLLM_PORT))
        except Exception:  # an unreachable engine reports nothing
            return None

    def _load(self, pod: _PodSleep, ack: dict | None, *, engine_load: int | None) -> dict:
        """The load of one pod. ``known`` is False on any gateway read error or
        unavailable engine metrics."""
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
            # Unknown / unparseable: no mode parameter this time; not cached, so
            # the next sleep asks again.
            return False
        supported = sleep_mode_supported(version)
        self._mode_cache[key] = supported
        return supported

    def _send_guarded(self, batch: SleepBatch, pod: _PodSleep) -> None:
        try:
            self._send_one(batch, pod)
        except Exception as exc:
            self._rollback_pod(batch, pod, reason=f"{type(exc).__name__}: {exc}")

    def _send_one(self, batch: SleepBatch, pod: _PodSleep) -> None:
        """ONE /sleep for one pod. Sets ``pod.commit`` when the call succeeded
        (the pod is then confirmed physically); marks the pod unconfirmed when
        the call got no answer; raises (-> rolled back after a re-probe) when it
        failed or must not be sent."""
        target = pod.target
        policy = self._policy
        load = pod.last_load
        drained = pod.decision == "wait"
        supports_mode = bool(pod.supports_mode)
        if not supports_mode and not drained:
            # Requests in flight (or an unknown load) and no mode parameter: a
            # plain /sleep is only sent when the operator chose it explicitly
            # (vllm_sleep_mode_param: false); with auto-detection the target is
            # rolled back - nothing was sent.
            if str(policy.vllm_sleep_mode_param).lower() != "false":
                self._journal.incr("sleep_rolled_back_no_mode_total")
                message = (
                    f"{target.binding.serve_id}: requests in flight (or load unknown) and the "
                    "vLLM /sleep mode parameter is not available (/version unreadable, "
                    f"unparseable or < {'.'.join(str(v) for v in SLEEP_MODE_MIN_VERSION)}): "
                    "no plain /sleep, rolled back. Upgrade vLLM, or set "
                    "service_manager.sleep.vllm_sleep_mode_param: false to send a plain "
                    "/sleep (cuts those requests off)"
                )
                LOG.error(json.dumps(
                    {"event": "sleep_rolled_back_no_mode", "serve_id": target.binding.serve_id,
                     "detail": message},
                    sort_keys=True,
                ))
                raise SleepFailed(message)
        idle_mode = "wait" if str(policy.sleep_mode_when_idle).lower() == "wait" else "abort"
        mode: str | None = (idle_mode if drained else "abort") if supports_mode else None
        forced = not drained
        # An unknown load still cut off at least one request.
        forced_count = max(1, int(load.get("in_flight") or 0)) if forced else 0
        aborted = _abort_breakdown(load) if forced else None
        if mode is None and not drained:
            self._journal.incr("plain_sleep_with_inflight_total")  # actually sent
        self._journal.update(pod.pod, phase="sleeping", drained=drained, sleep_mode=mode)
        pod.sleep_called = True
        result = self._vllm.sleep(
            target.pod_ip,
            port=VLLM_PORT,
            mode=mode,
            timeout_s=policy.sleep_call_timeout_s,
            hidden=True,
        )
        if _unanswered(result):
            # No HTTP answer within sleep_call_timeout_s (a timeout or a hung
            # engine): it may still be going to sleep. No further probe: the pod
            # stays hidden, the journal entry says so, the lock is released and
            # the journal recovery settles it from the physical state.
            pod.sleep_uncertain = True
            self._journal.incr("sleep_call_timeout_total")
            self._mark_unconfirmed(
                batch, pod, reason="/sleep got no answer within sleep_call_timeout_s: left hidden "
                "for the journal recovery",
            )
            return
        if not _success(result):
            message = getattr(result, "message", "") or "operation failed"
            raise SleepFailed(f"vLLM sleep failed for {target.binding.serve_id}: {message}")
        pod.commit = {
            "mode": mode,
            "forced": forced,
            "forced_count": forced_count,
            "aborted": aborted,
        }

    def _confirm_all(self, batch: SleepBatch, pods: list[_PodSleep], clock: Clock) -> None:
        """Poll ``/is_sleeping`` of every sent pod (in parallel) until each reads
        asleep or ``physical_confirm_timeout_s`` passed; one that did not is left
        hidden and ``unconfirmed`` (the journal recovery settles it). (vLLM ops
        without the probe: the /sleep result is trusted.)"""
        pending = list(pods)
        if not pending:
            return
        probe_available = callable(getattr(self._vllm, "is_sleeping", None))
        deadline = clock.monotonic() + self._policy.physical_confirm_timeout_s
        while pending:
            if probe_available:
                answers = _parallel(lambda pod: self._physical(pod.target), pending)
            else:
                answers = {id(pod): True for pod in pending}
            confirmed = [pod for pod in pending if answers[id(pod)] is True]
            pending = [pod for pod in pending if answers[id(pod)] is not True]
            # One k8s annotation write per confirmed pod, in parallel (review 4
            # P3): the step stays bounded whatever the number of targets.
            _parallel(lambda pod: self._finalize_slept(batch, pod, **pod.commit), confirmed)
            if not pending or clock.monotonic() >= deadline:
                break
            clock.sleep(min(self._policy.poll_interval_s, max(0.0, deadline - clock.monotonic())) or 0.01)
        for pod in pending:
            self._mark_unconfirmed(
                batch, pod,
                reason=f"not confirmed asleep within physical_confirm_timeout_s "
                f"({self._policy.physical_confirm_timeout_s:g}s): left hidden for the journal recovery",
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
        aborted: dict | None = None,
    ) -> None:
        self._runtime.write_binding_annotations(pod.target.binding, state=POD_STATE_SLEEPING)
        pod.done = True
        if batch.keep_journal:
            self._journal.update(pod.pod, phase=PHASE_SLEPT)
        else:
            self._journal.end(pod.pod)
        self._journal.incr("sleeps_total")
        self._journal.incr(f"sleeps_path_{batch.path}")
        self._journal.incr("no_drain_sleeps_total")
        if forced:
            self._journal.incr("forced_abort_total")
            self._journal.incr("forced_abort_requests_total", forced_count)
            if aborted is not None:
                truncated = int(aborted.get("non_continuable") or 0)
                if truncated:
                    self._journal.incr("no_drain_non_continuable_aborted_total", truncated)
                unclassified = int(aborted.get("unclassified") or 0)
                if unclassified:
                    self._journal.incr("no_drain_unclassified_aborted_total", unclassified)
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
        keep the journal entry (audit ``sleep_unconfirmed``; recovery resolves it)."""
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
                self._journal.end(pod.pod)
                pod.outcome = self._outcome(batch, pod, STATUS_ROLLED_BACK, reason=reason)
                return
            if pod.sleep_called:
                # /sleep answered with an error: re-probe before re-opening.
                physical = self._physical(pod.target)
                if physical is True:
                    # It did go to sleep; record the truth instead of re-routing.
                    forced = pod.decision != "wait"
                    self._finalize_slept(
                        batch,
                        pod,
                        mode=None,
                        forced=forced,
                        forced_count=max(1, int(pod.last_load.get("in_flight") or 0)) if forced else 0,
                        reason=reason,
                        aborted=_abort_breakdown(pod.last_load) if forced else None,
                    )
                    return
                if physical is None:
                    self._mark_unconfirmed(batch, pod, reason=f"{reason}; physical state unknown")
                    return
                if not self.confirm_unpaused(pod.target):
                    # Awake but possibly PAUSED (the /sleep paused the scheduler,
                    # then the weight offload failed): re-opening routing would
                    # send traffic to a frozen engine (review 2026-10-02 P2).
                    self._mark_unconfirmed(
                        batch,
                        pod,
                        reason=f"{reason}; awake but the engine could not be confirmed "
                        "un-paused (/is_paused, /resume): left hidden for the recovery",
                    )
                    return
            self._runtime.write_binding_annotations(
                pod.target.binding, state=pod.previous_state
            )
            self._journal.end(pod.pod)
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

    def _outcome(
        self, batch: SleepBatch, pod: _PodSleep, status: str, *, reason=None, **extra
    ) -> dict:
        ack = batch.ack or {}
        outcome = {
            "serve_id": pod.pod,
            "binding_id": pod.binding_id,
            "path": batch.path,
            "drain_policy": "no_drain",
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


def _parallel(fn, items: list) -> dict[int, object]:
    """``fn`` over ``items`` (by ``id``): inline for one item, else one thread
    per item, so a round costs one call duration whatever the target count."""
    if len(items) <= 1:
        return {id(item): fn(item) for item in items}
    context = contextvars.copy_context()
    with ThreadPoolExecutor(max_workers=len(items)) as pool:
        futures = {id(item): pool.submit(context.copy().run, fn, item) for item in items}
        return {key: future.result() for key, future in futures.items()}


def _abort_breakdown(load: Mapping) -> dict:
    """What a forced /sleep mode=abort cuts off, from the load read: the
    continuable requests (the reissue sidecar continues them), the gateway's
    non-continuable ones (truncated, or re-run from scratch by the sidecar when
    nothing was streamed yet) and engine-side requests the gateway did not count
    (``unclassified``). ``state_known`` False: the counts are a lower bound.
    The read happens BEFORE ``/sleep mode=abort`` is sent: requests that finish
    normally in between are counted here but never aborted, so these counts are
    an upper bound of what the abort cut off."""
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


def _unanswered(result) -> bool:
    """A failed vLLM call that got no HTTP answer at all (timeout / transport
    error: ``VllmOpResult.status_code`` None) - the engine may still act on it.
    Results without a ``status_code`` attribute (test doubles) never count."""
    return not _success(result) and hasattr(result, "status_code") and result.status_code is None
