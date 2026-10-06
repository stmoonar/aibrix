"""The tick loop: gather -> decide -> clamp -> scale-downs before scale-ups -> dispatch.

* Dry-run (the default) logs every decision and never calls the SM. A shell configured
  to actuate falls back to dry-run for any tick in which it does not hold the owner lock
  ``tre:v2:bl:owner`` (``SET NX PX``, renewed every tick by a compare-and-pexpire script),
  so two shells never actuate.
* Controller guard: a shell configured to actuate reads the TRE controller's run mode
  (``tre:v2:controller:mode``, missing = observe, as ``tools/arm.py``) every tick; unless it
  is ``observe`` (or the read fails) the tick is dry-run and every would-be SM call is
  logged as ``guard_controller_active`` (metric ``tre_bl_controller_guard``), so the TRE
  controller and a baseline never scale the same models.
* Both checks are repeated by the SM worker immediately before each call
  (:meth:`BaselineShell.actuation_guard`: the owner lock still holds our token, the mode is
  still observe). A gather or decide that outlived the lock TTL, or a mode switch in the
  meantime, drops the call (``sm_result.error = "dropped"``, no backoff). The window left
  is the time between that check and the SM handling the request; closing it needs the
  SM to reject a stale owner generation.
* Unknown is not idle (backstop): a scale-down is held at ``awake`` (reason
  ``incomplete``, the policy's own reason in ``inputs.policy_reason``) whenever
  :func:`~tre_baselines.snapshot.evidence_gaps` lists a gap for the model (with the
  policy's ``needs_events`` / ``event_history_s``). The policies apply the same gate
  themselves; this one covers any policy that does not.
* The dispatcher is asynchronous: a model whose previous SM call is still running gets
  ``inflight_skip`` (not queued); the tick never waits for the SM.
* Donor before dependent (no arbiter): scale-downs are submitted first. A scale-up that
  needs a donor - ``/v2/state`` ``gpus[]`` shows fewer GPU sets the model can wake on
  than replicas it adds (``ModelSnapshot.wakeable_slots``; unknown counts as none) - is
  not sent while any scale-down call is in flight (action ``wait_donor``): the SM answers
  a scale-down only after the sleep is committed and the GPU released, so it goes out on
  the first tick after that answer (state, not a timer; thread scheduling and model names
  no longer decide who gets a shared GPU). A scale-up with free GPUs goes out at once.
* SM refusals are logged (``sm_result`` on the model's next decision line).
* A refused scale-up (an SM answer) is not a model-wide pause: the SM's ``retry_after_s``
  means "avoid THOSE GPUs", and the shell cannot steer the SM to other GPUs, so it waits
  for evidence that the state changed (:class:`~tre_baselines.sm_client.RefusalGate`). The
  refusal records the refused ``node/gpu``, the blocking binding and the state view the
  call was decided on (the submitting tick's, so a change between the refusal and the
  tick that collects it is not lost). The next scale-up goes out on the first tick where
  the blocking binding is no longer awake, a GPU of a sleeping binding of the model became
  free, or the store ``version`` / wakeable GPU set of ``/v2/state`` differs (a lease, a
  journaled wake or a gpu-truth sample can block a wake without bumping the version);
  its line carries ``retry_evidence``. While nothing changed the line says
  ``wait_refusal`` and no call is sent (no busy loop). Only a cause ``/v2/state`` cannot
  show (writer lock, a failed wake, a text-only 409) is also retried once the SM's
  ``retry_after_s`` has passed (none given: the next tick).
* A failed scale-down, or a call the SM never answered (transport error, timeout), backs
  off (:class:`~tre_baselines.sm_client.Backoff`: ``max(retry_after_s, tick_s)``, doubling,
  capped at ``TRE_BL_BACKOFF_MAX_S``, default 10 s; action ``backoff``); a refused
  scale-down is retried at once when the SM state changes.
* A success or the desired count back at awake clears both.
* Scale-ups are stepped (:class:`~tre_baselines.sm_client.Dispatcher`): ``awake+1``,
  ``awake+2``, ... until the target or the first refusal, because the SM refuses a whole
  multi-replica target when one wake is blocked. Some replicas granted, then refused =
  ``partial_fill`` (``sm_result.partial_fill``, ``reached``; metric
  ``tre_bl_partial_fills_total``). Our own wakes changed the state, so the next tick asks
  again from the new awake count; a refusal of that call waits for evidence. An SM that
  fills partially itself (200 ``unfilled``, or a 409 that lists wakes done) is handled the
  same way: the wakes it reports are counted as done.
* One JSONL line per model per tick (``policy`` = the ``TRE_BL_POLICY`` key, ``arm`` = the
  adapted arm's name, e.g. ``PreServe-oracle``) to
  ``$TRE_BL_LOG_DIR/decisions-<policy>-<YYYYMMDD>.jsonl``
  (date of the Redis clock, UTC), mirrored to ``tre:v2:bl:decision:<model>`` (TTL 1 h)
  unless ``TRE_BL_WRITE_REDIS=false``, and appended to the stream ``tre:v2:bl:decisions``
  (MAXLEN ~ 100000) unless ``TRE_BL_DECISION_STREAM=false``.
* A tick that raises is logged and skipped; ``max_tick_failures`` consecutive failures
  turn ``/healthz`` (readiness) into 503. ``/livez`` (liveness) only says whether the loop
  is running and has ticked within ``TRE_BL_LIVENESS_STALL_S``, so an SM / Redis outage
  never restarts the pod.
"""
from __future__ import annotations

import datetime as _dt
import json
import logging
import os
import socket
import threading
import time
import uuid
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Mapping, Optional, Protocol

from tre_baselines.keys import (
    CONTROLLER_MODE_KEY,
    DECISION_TTL_S,
    DECISIONS_STREAM,
    DECISIONS_STREAM_MAXLEN,
    OWNER_KEY,
    decision_key,
)
from tre_baselines.policies.base import INCOMPLETE, Decision
from tre_baselines.sm_client import (
    DEFAULT_BACKOFF_MAX_S,
    DROPPED,
    Backoff,
    Completed,
    Dispatcher,
    RefusalGate,
    StateView,
)
from tre_baselines.snapshot import ClusterSnapshot, evidence_gaps

LOG = logging.getLogger(__name__)

REVERSAL_WINDOW_MS = 60_000
ACTIONS = ("none", "up", "down", "inflight_skip", "dry_run", "backoff", "wait_refusal", "guard_controller_active",
           "wait_donor")
#: Controller modes under which a baseline may actuate (missing key = observe).
_CONTROLLER_OK = (None, "", "observe")


class Source(Protocol):
    def gather(self, tick: int = 0) -> ClusterSnapshot: ...


def clamp(desired: int, lo: int, hi: int) -> int:
    return max(int(lo), min(int(hi), int(desired)))


#: KEYS[1] = lock, ARGV[1] = token, ARGV[2] = ttl ms: renew only while the value is ours
#: (atomic: a GET-then-PEXPIRE could extend a lock another shell took in between).
RENEW_LUA = (
    "if redis.call('get', KEYS[1]) == ARGV[1] then "
    "return redis.call('pexpire', KEYS[1], ARGV[2]) else return 0 end"
)
#: KEYS[1] = lock, ARGV[1] = token: delete only while the value is ours.
RELEASE_LUA = (
    "if redis.call('get', KEYS[1]) == ARGV[1] then "
    "return redis.call('del', KEYS[1]) else return 0 end"
)


class OwnerLock:
    """``SET key token NX PX ttl``; renewed (compare-and-pexpire) and released
    (compare-and-delete) by Lua scripts, so neither can touch another shell's lock."""

    def __init__(self, redis: Any, ttl_s: float, *, key: str = OWNER_KEY, token: Optional[str] = None) -> None:
        self._redis = redis
        self._ttl_ms = max(1000, int(ttl_s * 1000))
        self.key = key
        self.token = token or f"{socket.gethostname()}:{os.getpid()}:{uuid.uuid4().hex[:8]}"

    def ensure(self) -> bool:
        try:
            if int(self._redis.eval(RENEW_LUA, 1, self.key, self.token, self._ttl_ms) or 0) == 1:
                return True
            return bool(self._redis.set(self.key, self.token, nx=True, px=self._ttl_ms))
        except Exception as exc:
            LOG.warning("owner lock check failed: %s", exc)
            return False

    def held(self) -> bool:
        """Read-only: the lock still holds our token (no renewal; fails closed)."""
        try:
            raw = self._redis.get(self.key)
        except Exception as exc:
            LOG.warning("owner lock read failed: %s", exc)
            return False
        raw = raw.decode("utf-8", "replace") if isinstance(raw, bytes) else raw
        return raw == self.token

    def release(self) -> None:
        try:
            self._redis.eval(RELEASE_LUA, 1, self.key, self.token)
        except Exception as exc:
            LOG.warning("owner lock release failed: %s", exc)


@dataclass
class ShellStats:
    ticks: int = 0
    tick_failures: int = 0
    consecutive_failures: int = 0
    decisions: int = 0
    sm_calls: int = 0
    sm_failures: int = 0
    sm_dropped: int = 0
    redis_write_failures: int = 0
    effective_dry_run: bool = True
    owner: bool = False
    last_tick_ms: Optional[int] = None
    last_error: Optional[str] = None
    actions: dict[tuple[str, str], int] = field(default_factory=dict)
    reversals: dict[str, int] = field(default_factory=dict)
    event_lag_s: dict[str, float] = field(default_factory=dict)
    scrape_failures: int = 0
    controller_mode: Optional[str] = None
    controller_guard: bool = False
    guard_ticks: int = 0
    backoff_skips: int = 0
    #: Ticks a refused scale-up waited for evidence of a state change.
    refusal_waits: int = 0
    #: Stepped scale-ups that got some replicas, then a refusal (per model).
    partial_fills: dict[str, int] = field(default_factory=dict)
    #: arr events per model whose request does not stream (expected 0: PreServe's map
    #: needs ft, which a non-streaming request only gets at the end).
    nonstream_arrivals: dict[str, int] = field(default_factory=dict)
    #: arr events per model.
    arrivals: dict[str, int] = field(default_factory=dict)
    #: arr events per model whose in_tokens did not come from the client header
    #: (``in_src != "header"``): must be 0 when the clients send x-tre-bl-in-tokens.
    nonheader_in_tokens: dict[str, int] = field(default_factory=dict)


class DecisionLog:
    def __init__(self, log_dir: str, policy: str) -> None:
        self._dir = Path(log_dir)
        self._policy = policy
        self._dir.mkdir(parents=True, exist_ok=True)

    def path_for(self, ts_ms: int) -> Path:
        day = _dt.datetime.fromtimestamp(ts_ms / 1000.0, tz=_dt.timezone.utc).strftime("%Y%m%d")
        return self._dir / f"decisions-{self._policy}-{day}.jsonl"

    def write(self, lines: list[dict]) -> None:
        if not lines:
            return
        path = self.path_for(int(lines[0]["ts_ms"]))
        with path.open("a", encoding="utf-8") as fh:
            for line in lines:
                fh.write(json.dumps(line, sort_keys=True, default=str) + "\n")


class BaselineShell:
    def __init__(
        self,
        config: Any,
        source: Source,
        policy: Any,
        dispatcher: Dispatcher,
        redis: Any = None,
        *,
        lock: Optional[OwnerLock] = None,
        decision_log: Optional[DecisionLog] = None,
    ) -> None:
        self.config = config
        self.source = source
        self.policy = policy
        self.dispatcher = dispatcher
        self.redis = redis
        self.lock = lock
        self.log = decision_log or DecisionLog(config.log_dir, config.policy)
        #: Arm name in decision records (adaptations named: Chiron-global,
        #: TokenScale-colocated, PreServe-oracle).
        self.arm = str(getattr(policy, "label", None) or config.policy)
        self.stats = ShellStats()
        self.backoff = Backoff(config.tick_s, getattr(config, "backoff_max_s", DEFAULT_BACKOFF_MAX_S))
        #: Scale-up refusals, retried on evidence of a state change.
        self.refusals = RefusalGate()
        #: State view of the tick that submitted each model's scale-up in flight.
        self._submit_view: dict[str, StateView] = {}
        #: Models whose refused scale-down backs off (cleared when the SM state changes).
        self._refused: set[str] = set()
        self._sm_version: Any = None
        self._tick = 0
        self._running = False
        self._beat = 0.0
        self._last_dispatch: dict[str, tuple[str, int]] = {}
        self._stop = threading.Event()
        self._stats_lock = threading.Lock()
        dispatcher.guard = self.actuation_guard

    def actuation_guard(self) -> Optional[str]:
        """Checked by the SM worker right before each call: None = the call may go."""
        if self.config.dry_run:
            return "dry_run"
        if self.lock is not None and not self.lock.held():
            return "owner_lost"
        mode, guard = self._controller_mode()
        if guard:
            return f"controller_mode={mode}"
        return None

    # -- one tick ---------------------------------------------------------------------

    def _collect_results(self) -> dict[str, Completed]:
        latest: dict[str, Completed] = {}
        for done in self.dispatcher.drain_results():
            dropped = done.result.error == DROPPED
            with self._stats_lock:
                if dropped:
                    self.stats.sm_dropped += 1
                elif not done.result.ok:
                    self.stats.sm_failures += 1
            if not done.result.ok and not dropped:
                LOG.warning("SM refused %s %s->%s: %s", done.model, done.direction, done.target,
                            done.result.as_dict())
            latest[done.model] = done
        return latest

    def _owner(self) -> bool:
        if self.config.dry_run:
            return False
        if self.lock is None:
            return True
        return self.lock.ensure()

    def _controller_mode(self) -> tuple[Optional[str], bool]:
        """(raw mode, guard). Guard = the controller may be acting (not observe) or its
        mode cannot be read (fail closed)."""
        if self.redis is None:
            return "<no redis>", True
        try:
            raw = self.redis.get(CONTROLLER_MODE_KEY)
        except Exception as exc:
            LOG.warning("controller mode read failed (guard on): %s", exc)
            return "<read failed>", True
        mode = raw.decode("utf-8", "replace") if isinstance(raw, bytes) else raw
        mode = None if mode is None else str(mode).strip().lower()
        return mode, mode not in _CONTROLLER_OK

    def tick_once(self) -> list[dict]:
        owner = self._owner()
        mode, guard = (None, False) if self.config.dry_run else self._controller_mode()
        if guard and not self.stats.controller_guard:
            LOG.warning("TRE controller mode is %r (%s), not observe: actuation suspended", mode,
                        CONTROLLER_MODE_KEY)
        effective_dry = bool(self.config.dry_run or not owner or guard)
        snap = self.source.gather(self._tick)
        # After the gather: a tick whose gather raises leaves the results for the next one.
        results = self._collect_results()
        view = self._state_view(snap.extra or {})
        version = view.key
        if version is not None and version != self._sm_version:
            if self._sm_version is not None:  # the SM state changed: retry refused models now
                for model in self._refused:
                    self.backoff.reset(model)
                self._refused.clear()
            self._sm_version = version
        for model, done in results.items():
            if done.partial_fill:
                with self._stats_lock:
                    self.stats.partial_fills[model] = self.stats.partial_fills.get(model, 0) + 1
            submitted_view = self._submit_view.pop(model, None)
            if done.result.ok or (done.result.error == DROPPED and done.progressed):
                self.backoff.reset(model)
                self._refused.discard(model)
                self.refusals.clear(model)
            elif done.result.error == DROPPED:  # a dropped call never reached the SM
                pass
            elif done.direction == "up" and done.result.code is not None:
                # An SM answer: wait for evidence on the view the call was decided on.
                self.backoff.reset(model)
                self.refusals.refused(model, done.result, submitted_view or view, snap.now_ms)
            else:
                self.backoff.failed(model, snap.now_ms, done.result.retry_after_s)
                if done.result.code is not None:  # an SM answer, not a transport error
                    self._refused.add(model)
        with self._stats_lock:
            for model, ms in snap.models.items():
                for ev in ms.events:
                    if ev.kind == "arr":
                        self.stats.arrivals[model] = self.stats.arrivals.get(model, 0) + 1
                        if ev.stream is False:
                            self.stats.nonstream_arrivals[model] = self.stats.nonstream_arrivals.get(model, 0) + 1
                        if ev.in_src != "header":
                            self.stats.nonheader_in_tokens[model] = self.stats.nonheader_in_tokens.get(model, 0) + 1
        decisions: Mapping[str, Decision] = self.policy.decide(snap) or {}

        needs_events = bool(getattr(self.policy, "needs_events", False))
        history_s = float(getattr(self.policy, "event_history_s", 0.0) or 0.0)
        planned: list[dict] = []
        for model, ms in snap.models.items():
            decision = decisions.get(model)
            raw = None if decision is None else int(decision.desired)
            clamped = ms.awake if raw is None else clamp(raw, ms.min_replicas, ms.max_replicas)
            reason = "no_decision" if decision is None else decision.reason
            inputs = {} if decision is None else dict(decision.inputs)
            if clamped < ms.awake:
                gaps = evidence_gaps(ms, snap.now_ms, snap.tick_s, events=needs_events, history_s=history_s)
                if gaps:  # backstop: never scale down on incomplete evidence
                    inputs.update(gaps=list(gaps), policy_reason=reason, policy_clamped=clamped)
                    clamped, reason = ms.awake, INCOMPLETE
            direction = "up" if clamped > ms.awake else "down" if clamped < ms.awake else "none"
            planned.append({
                "model": model, "awake": ms.awake, "raw_desired": raw, "clamped": clamped,
                "direction": direction, "reason": reason, "inputs": inputs,
            })
        order = {"down": 0, "up": 1, "none": 2}
        planned.sort(key=lambda p: (order[p["direction"]], p["model"]))

        lines: list[dict] = []
        for item in planned:
            model, direction = item["model"], item["direction"]
            ms = snap.models[model]
            wait_s = None
            evidence = None
            donors: list[str] = []
            if direction == "none":
                action = "none"
                self.backoff.reset(model)  # the desired count is back at awake
                self._refused.discard(model)
                self.refusals.clear(model)
            elif guard:
                action = "guard_controller_active"
            elif effective_dry:
                action = "dry_run"
            elif (direction == "up" and self._needs_donor(snap.models[model], item["clamped"])
                  and (donors := self.dispatcher.inflight_downs())):
                action = "wait_donor"  # the GPU it needs is released when that call returns
            elif (wait_s := self.backoff.remaining_s(model, snap.now_ms)) is not None:
                action = "backoff"
            elif direction == "up" and (evidence := self.refusals.evidence(model, view, snap.now_ms)) is None:
                action = "wait_refusal"  # nothing changed since the refusal: no call
            elif self.dispatcher.submit(model, direction, item["clamped"],
                                        start=ms.awake if direction == "up" else None):
                action = direction
                if direction == "up":
                    self._submit_view[model] = view
                self._note_dispatch(model, direction, snap.now_ms)
            else:
                action = "inflight_skip"
            line = {
                "ts_ms": snap.now_ms,
                "tick": snap.tick,
                "policy": self.config.policy,
                "arm": self.arm,
                **item,
                "action": action,
                "dry_run": effective_dry,
                "owner": owner,
                "inflight": self.dispatcher.inflight(model),
            }
            if mode is not None or guard:
                line["controller_mode"] = mode
            if donors:
                line["wait_donor"] = donors
            if wait_s is not None:
                line["backoff_s"] = round(wait_s, 3)
                line["backoff_delay_s"] = self.backoff.delay_s(model)
            refusal = self.refusals.get(model) if direction == "up" else None
            if refusal is not None:
                line["refusal"] = refusal.as_dict(snap.now_ms)
                if action == "up":
                    line["retry_evidence"] = evidence
            done = results.get(model)
            if done is not None:
                line["sm_result"] = done.as_dict()
            lines.append(line)
            with self._stats_lock:
                key = (model, action)
                self.stats.actions[key] = self.stats.actions.get(key, 0) + 1
                self.stats.decisions += 1
                if action == "backoff":
                    self.stats.backoff_skips += 1
                elif action == "wait_refusal":
                    self.stats.refusal_waits += 1

        self.log.write(lines)
        self._write_redis(lines)
        extra = snap.extra or {}
        with self._stats_lock:
            self.stats.ticks += 1
            self.stats.consecutive_failures = 0
            self.stats.effective_dry_run = effective_dry
            self.stats.owner = owner
            self.stats.controller_mode = mode
            self.stats.controller_guard = guard
            self.stats.guard_ticks += int(guard)
            self.stats.last_tick_ms = snap.now_ms
            self.stats.event_lag_s = dict(extra.get("event_lag_s") or {})
            self.stats.scrape_failures += int(extra.get("scrape_failed") or 0)
        self._tick += 1
        return lines

    @staticmethod
    def _state_view(extra: Mapping[str, Any]) -> StateView:
        """The SM state a refusal is checked against: key = store version plus, when the
        source reports it, the wakeable GPU set (leases / journal / gpu-truth do not bump
        the version); the awake binding ids; the free GPUs of each model."""
        version = extra.get("sm_state_version")
        gpus = extra.get("sm_wakeable_gpus")
        key = version if gpus is None else (version, tuple(gpus))
        awake = extra.get("sm_awake_bindings")
        free = extra.get("sm_free_gpus")
        return StateView(
            key=key,
            awake_bindings=None if awake is None else frozenset(awake),
            free_gpus=None if not isinstance(free, Mapping) else {
                m: frozenset(v) for m, v in free.items() if v is not None},
        )

    @staticmethod
    def _needs_donor(ms: Any, target: int) -> bool:
        """The scale-up cannot be served from GPUs the model can wake on now."""
        free = getattr(ms, "wakeable_slots", None)
        return free is None or free < int(target) - int(ms.awake)

    def _note_dispatch(self, model: str, direction: str, now_ms: int) -> None:
        with self._stats_lock:
            self.stats.sm_calls += 1
            prev = self._last_dispatch.get(model)
            if prev is not None and prev[0] != direction and now_ms - prev[1] <= REVERSAL_WINDOW_MS:
                self.stats.reversals[model] = self.stats.reversals.get(model, 0) + 1
            self._last_dispatch[model] = (direction, now_ms)

    def _write_redis(self, lines: list[dict]) -> None:
        if self.redis is None:
            return
        to_key = bool(self.config.write_redis)
        to_stream = bool(getattr(self.config, "decision_stream", False))
        for line in lines:
            text = json.dumps(line, sort_keys=True, default=str)
            if to_key:
                try:
                    self.redis.set(decision_key(line["model"]), text, ex=DECISION_TTL_S)
                except Exception as exc:
                    with self._stats_lock:
                        self.stats.redis_write_failures += 1
                    LOG.warning("decision key write failed for %s: %s", line["model"], exc)
            if to_stream:
                try:
                    self.redis.xadd(DECISIONS_STREAM, {"line": text}, maxlen=DECISIONS_STREAM_MAXLEN,
                                    approximate=True)
                except Exception as exc:
                    with self._stats_lock:
                        self.stats.redis_write_failures += 1
                    LOG.warning("decision stream write failed for %s: %s", line["model"], exc)

    def safe_tick(self) -> Optional[list[dict]]:
        try:
            return self.tick_once()
        except Exception as exc:
            LOG.exception("tick %d failed; skipped", self._tick)
            with self._stats_lock:
                self.stats.tick_failures += 1
                self.stats.consecutive_failures += 1
                self.stats.last_error = repr(exc)[:500]
            self._tick += 1
            return None

    # -- run --------------------------------------------------------------------------

    def run(self, max_ticks: Optional[int] = None) -> None:
        period = float(self.config.tick_s)
        next_at = time.monotonic()
        done = 0
        self._beat = time.monotonic()
        self._running = True
        try:
            while not self._stop.is_set():
                self.safe_tick()
                self._beat = time.monotonic()
                done += 1
                if max_ticks is not None and done >= max_ticks:
                    break
                next_at += period
                delay = next_at - time.monotonic()
                if delay < 0:  # overran: skip the missed slots, never burst
                    next_at = time.monotonic()
                    delay = 0.0
                self._stop.wait(delay)
        finally:
            self._running = False

    def stop(self) -> None:
        self._stop.set()

    # -- health / metrics -------------------------------------------------------------

    def alive(self) -> bool:
        """Liveness: the loop is running and finished a tick (failed or not) recently."""
        stall = max(float(getattr(self.config, "liveness_stall_s", 120.0)), 2.0 * float(self.config.tick_s))
        return self._running and (time.monotonic() - self._beat) <= stall

    def healthy(self) -> bool:
        with self._stats_lock:
            return self.stats.consecutive_failures < int(self.config.max_tick_failures)

    def policy_counters(self) -> Mapping[str, Mapping[str, int]]:
        """The policy's cumulative per-model counters (anomalies, tier2_below_t1,
        empty_window_busy, ...); empty when the policy has none."""
        fn = getattr(self.policy, "counters", None)
        try:
            return dict(fn()) if callable(fn) else {}
        except Exception as exc:  # reporting must never break the loop
            LOG.warning("policy counters failed: %s", exc)
            return {}

    def health_doc(self) -> dict:
        with self._stats_lock:
            s = self.stats
            return {
                "ok": s.consecutive_failures < int(self.config.max_tick_failures),
                "policy": self.config.policy,
                "arm": self.arm,
                "ticks": s.ticks,
                "consecutive_failures": s.consecutive_failures,
                "last_error": s.last_error,
                "dry_run_configured": bool(self.config.dry_run),
                "dry_run_effective": s.effective_dry_run,
                "owner": s.owner,
                "last_tick_ms": s.last_tick_ms,
                "controller_mode": s.controller_mode,
                "controller_guard": s.controller_guard,
            }

    def metrics_text(self) -> str:
        with self._stats_lock:
            s = self.stats
            policy = self.config.policy
            lines = [
                "# TYPE tre_bl_ticks_total counter",
                f'tre_bl_ticks_total{{policy="{policy}"}} {s.ticks}',
                "# TYPE tre_bl_tick_failures_total counter",
                f'tre_bl_tick_failures_total{{policy="{policy}"}} {s.tick_failures}',
                "# TYPE tre_bl_consecutive_tick_failures gauge",
                f'tre_bl_consecutive_tick_failures{{policy="{policy}"}} {s.consecutive_failures}',
                "# TYPE tre_bl_decisions_total counter",
                f'tre_bl_decisions_total{{policy="{policy}"}} {s.decisions}',
                "# TYPE tre_bl_sm_calls_total counter",
                f'tre_bl_sm_calls_total{{policy="{policy}"}} {s.sm_calls}',
                "# TYPE tre_bl_sm_failures_total counter",
                f'tre_bl_sm_failures_total{{policy="{policy}"}} {s.sm_failures}',
                "# TYPE tre_bl_sm_dropped_total counter",
                f'tre_bl_sm_dropped_total{{policy="{policy}"}} {s.sm_dropped}',
                "# TYPE tre_bl_scrape_failures_total counter",
                f'tre_bl_scrape_failures_total{{policy="{policy}"}} {s.scrape_failures}',
                "# TYPE tre_bl_redis_write_failures_total counter",
                f'tre_bl_redis_write_failures_total{{policy="{policy}"}} {s.redis_write_failures}',
                "# TYPE tre_bl_inflight gauge",
                f'tre_bl_inflight{{policy="{policy}"}} {self.dispatcher.inflight_count()}',
                "# TYPE tre_bl_dry_run gauge",
                f'tre_bl_dry_run{{policy="{policy}"}} {int(s.effective_dry_run)}',
                "# TYPE tre_bl_owner gauge",
                f'tre_bl_owner{{policy="{policy}"}} {int(s.owner)}',
                "# TYPE tre_bl_controller_guard gauge",
                f'tre_bl_controller_guard{{policy="{policy}"}} {int(s.controller_guard)}',
                "# TYPE tre_bl_controller_guard_ticks_total counter",
                f'tre_bl_controller_guard_ticks_total{{policy="{policy}"}} {s.guard_ticks}',
                "# TYPE tre_bl_backoff_skips_total counter",
                f'tre_bl_backoff_skips_total{{policy="{policy}"}} {s.backoff_skips}',
                "# TYPE tre_bl_refusal_waits_total counter",
                f'tre_bl_refusal_waits_total{{policy="{policy}"}} {s.refusal_waits}',
                "# TYPE tre_bl_actions_total counter",
            ]
            for (model, action), count in sorted(s.actions.items()):
                lines.append(f'tre_bl_actions_total{{policy="{policy}",model="{model}",action="{action}"}} {count}')
            lines.append("# TYPE tre_bl_direction_reversals_60s_total counter")
            for model, count in sorted(s.reversals.items()):
                lines.append(f'tre_bl_direction_reversals_60s_total{{policy="{policy}",model="{model}"}} {count}')
            lines.append("# TYPE tre_bl_partial_fills_total counter")
            for model, count in sorted(s.partial_fills.items()):
                lines.append(f'tre_bl_partial_fills_total{{policy="{policy}",model="{model}"}} {count}')
            lines.append("# TYPE tre_bl_arrivals_total counter")
            for model, count in sorted(s.arrivals.items()):
                lines.append(f'tre_bl_arrivals_total{{policy="{policy}",model="{model}"}} {count}')
            lines.append("# TYPE tre_bl_nonstream_arrivals_total counter")
            for model, count in sorted(s.nonstream_arrivals.items()):
                lines.append(f'tre_bl_nonstream_arrivals_total{{policy="{policy}",model="{model}"}} {count}')
            lines.append("# TYPE tre_bl_nonheader_in_tokens_total counter")
            for model, count in sorted(s.nonheader_in_tokens.items()):
                lines.append(f'tre_bl_nonheader_in_tokens_total{{policy="{policy}",model="{model}"}} {count}')
            lines.append("# TYPE tre_bl_policy_events_total counter")
            for model, counts in sorted(self.policy_counters().items()):
                for name, count in sorted(counts.items()):
                    lines.append(f'tre_bl_policy_events_total{{policy="{policy}",model="{model}",name="{name}"}} {count}')
            lines.append("# TYPE tre_bl_event_lag_seconds gauge")
            for model, lag in sorted(s.event_lag_s.items()):
                lines.append(f'tre_bl_event_lag_seconds{{policy="{policy}",model="{model}"}} {lag:.3f}')
        return "\n".join(lines) + "\n"


def make_http_server(shell: BaselineShell, port: int, host: str = "0.0.0.0") -> ThreadingHTTPServer:
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802
            if self.path.startswith("/livez"):
                ok = shell.alive()
                self._send(200 if ok else 503, b"ok\n" if ok else b"loop not running\n", "text/plain")
            elif self.path.startswith("/healthz"):
                doc = shell.health_doc()
                body = json.dumps(doc, sort_keys=True).encode("utf-8")
                self._send(200 if doc["ok"] else 503, body, "application/json")
            elif self.path.startswith("/metrics"):
                self._send(200, shell.metrics_text().encode("utf-8"), "text/plain; version=0.0.4")
            else:
                self._send(404, b"not found\n", "text/plain")

        def _send(self, code: int, body: bytes, ctype: str) -> None:
            self.send_response(code)
            self.send_header("content-type", ctype)
            self.send_header("content-length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, fmt: str, *args: Any) -> None:
            LOG.debug("http: " + fmt, *args)

    server = ThreadingHTTPServer((host, port), Handler)
    server.daemon_threads = True
    return server
