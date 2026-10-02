from __future__ import annotations

import asyncio
import json
import logging
import time
from collections import deque
from dataclasses import dataclass, field, replace
from typing import Awaitable, Callable, Iterable, Mapping, Protocol

from tre_controller.sm_client import sm_actor
from tre_controller.planning.planner import (
    Action,
    DefragAction,
    HideAction,
    ReceiverTarget,
    SafeScaleCommitAction,
    ScaleAction,
    SourceLoop,
    TransferIntent,
    UnhideAction,
)

if False:  # annotations are strings (from __future__); avoids an import cycle
    from tre_controller.profiling import TickProfiler

CLUSTER_MODEL = "__cluster__"
#: Resource key of a cluster-wide action (defrag): it conflicts with EVERY other
#: action - a defrag migrates pods of any model across any GPU, so nothing that
#: touches a pod or a model runs next to it (review 3 P3).
CLUSTER_RESOURCE = "cluster:*"

#: Source loops whose actions are one-shot: the source never re-emits them (the
#: SafeScale state machine deletes a probe when it resolves), so a transient SM
#: failure must not drop them (review 2 P1-2 / P2-5). Actions of the other loops
#: are re-planned every tick and may be dropped.
ONE_SHOT_LOOPS = frozenset({"safescale"})

LOG = logging.getLogger(__name__)

QueueAction = Action | SafeScaleCommitAction

#: Signal states in which a model needs capacity (rescue / fairness receivers),
#: and those in which it clearly does not. Anything else (unknown, a receiver
#: whose band is not yet confirmed) is neither: the planned commit proceeds.
NEEDS_CAPACITY_STATES = frozenset({"critical", "low"})
NO_NEED_STATES = frozenset({"healthy", "high", "idle"})

#: None = the action is still wanted; else the reason it no longer is.
Revalidate = Callable[[QueueAction], "str | None"]


@dataclass(frozen=True)
class CommitVerdict:
    """Revalidation of a SafeScale commit before a (re)try (review 3 P2-1..P2-3):
    ``abandon_reason`` set = unhide the donor pods instead of sleeping them;
    otherwise run ``action`` (upscales the receivers no longer need removed,
    listed in ``dropped`` as (model, reason))."""

    action: SafeScaleCommitAction
    abandon_reason: str | None = None
    dropped: tuple[tuple[str, str], ...] = ()


CommitRevalidate = Callable[[SafeScaleCommitAction], CommitVerdict]


class ServiceManagerClient(Protocol):
    async def scale_model(self, model: str, delta: int) -> dict: ...

    async def scale_model_to(self, model: str, target: int) -> dict: ...

    async def model_awake(self, model: str) -> dict: ...

    async def set_routable(self, model: str, hidden_pods: tuple[str, ...]) -> dict: ...

    async def set_binding_power(self, serve_id: str, *, awake: bool) -> dict: ...

    async def defrag(self, migrations: tuple) -> dict: ...

    async def transfer(self, donor_model: str, receiver_model: str, count: int, *, sleep_path: str) -> dict: ...


@dataclass(frozen=True)
class RetryPolicy:
    """Bounded exponential backoff for one-shot actions that failed retriably
    (409 conflict / busy, 503 shutting down, timeouts, connection errors)."""

    max_attempts: int = 6
    base_backoff_s: float = 2.0
    max_backoff_s: float = 30.0

    def backoff_s(self, failures: int) -> float:
        return min(self.max_backoff_s, self.base_backoff_s * (2 ** max(0, failures - 1)))


@dataclass(frozen=True)
class QueuedAction:
    action: QueueAction
    model: str
    source_loop: SourceLoop
    #: Every model the action changes (inflight accounting); default (model,).
    models: tuple[str, ...] = ()
    #: Serialization keys: two queued actions sharing a key never run concurrently
    #: and keep their submit order (models and pods; the GPU keys were removed on
    #: 2026-10-02 - the service-manager serializes GPU use under its writer lock).
    resources: frozenset[str] = frozenset()
    #: Failed retriable attempts so far (one-shot actions).
    failures: int = 0

    def __post_init__(self) -> None:
        if not self.models:
            object.__setattr__(self, "models", (self.model,))
        if not self.resources:
            object.__setattr__(self, "resources", frozenset(f"model:{m}" for m in self.models))


@dataclass(frozen=True)
class SubmitResult:
    accepted: int
    #: One-shot (SafeScale) actions accepted while in observe mode: they run
    #: only in their observe-safe form (the unhide of the probe pods).
    held: int = 0
    dropped: tuple[tuple[str, str], ...] = ()
    replaced: tuple[tuple[str, SourceLoop], ...] = ()


@dataclass(frozen=True)
class DispatchResult:
    model: str
    action_kind: str
    ok: bool
    error: str | None = None
    retriable: bool = False
    attempts: int = 1
    #: Pods the SM reported ``unconfirmed`` (/sleep sent, never confirmed asleep)
    #: or whose rollback failed (review 4 P2-2): they must stay hidden.
    unconfirmed: tuple[str, ...] = ()
    #: The SM refused the call with 409 ``floor_violation`` (a binding-level sleep or a
    #: SafeScale hide): recorded as an event only since 2026-10-02 (no hold).
    floor_violation: bool = False
    #: S3: the located wake refusal / failure of a structured 409 (node, gpu_ids,
    #: scope, error), or None. An event only since 2026-10-02 (no GPU cooldown).
    wake_conflict: dict | None = None
    #: S5: where the SM woke the replicas (``picked`` of the response).
    picked: tuple = ()
    #: 2026-10-02: replicas a scale-down really put to sleep (``taken`` of a model-level
    #: ``/target`` shrink, a transfer's donor side); None = not reported. 0 = nothing
    #: changed (the floor clamped it): not a scale-down for any hold or stamp.
    taken: int | None = None
    #: 2026-10-02: receiver replicas a transfer woke (``done``); None for other actions.
    done: int | None = None
    #: 2026-10-02: the transfer summary of a TransferIntent's result (transfer_id,
    #: done, taken, unfilled, clamped_by_floor, pairs with pod names, refusals).
    transfer: dict | None = None
    #: 2026-10-02: the SM refused the call before changing anything (``routable_unknown``,
    #: ``writer_busy``, ``floor_violation``, a located wake refusal, a plain RetryLater, 400):
    #: accounted as not executed - no view-pending / O1 stamp, no last action.
    not_executed: bool = False


@dataclass
class RescueTargetRecord:
    """C1: the last fast-loop rescue target of one model (planner ``RescuePlan``).

    ``covered`` = replicas the target's scale-ups really added on top of what the plan
    counted already: the planner computes the next desired from ``base`` and plans only
    what exceeds it while the model's decision window does not reflect the scale-up
    (``done_ms`` None = still running; the window starts before ``done_ms``)."""

    target: int
    desired: int
    base: int
    covered_before: int
    issued_ms: int
    outstanding: int = 0
    gained: int = 0
    failures: int = 0
    done_ms: int | None = None

    @property
    def covered(self) -> int:
        return self.covered_before + self.gained


@dataclass
class _Backoff:
    """A one-shot action of a running dispatch task waiting out its retry backoff.
    A rescue action may preempt it (review 3 P2-3): ``queued`` is replaced, the
    ``results`` of what the preemption removed are handed to the task, and
    ``event`` wakes the task at once."""

    queued: QueuedAction
    event: asyncio.Event = field(default_factory=asyncio.Event)
    results: list[DispatchResult] = field(default_factory=list)


class ActionQueue:
    """Dispatches controller actions to the service-manager.

    Actions run concurrently unless they share a resource (a model, a pod; a
    defrag conflicts with everything): those run one at a time in submit order
    (review P1-3 / review 2 P1-1). A donor -> receiver relay (TransferIntent,
    2026-10-02) is ONE service-manager call (``POST /v2/transfers``) holding both
    models; the SM picks the pods / GPUs, sleeps the donors and wakes the
    receivers under its global writer lock, so when the call returns the relay
    has completed or failed. A SafeScale commit is an ordered unit for its hidden
    donor pods and its follow-up receivers (review 3 P2-2). One-shot actions (SafeScale commit /
    rollback) that fail retriably are retried with bounded backoff and
    re-validated before every retry; a retry only ever re-sends idempotent
    requests (named bindings, absolute targets - review 3 P2-1). A rescue action
    for a model whose one-shot commit waits out a backoff preempts that retry
    (review 3 P2-3). Re-plannable actions are dropped on failure.

    Observe mode (user decision 2026-09-28, record only): re-plannable actions
    are dropped; a SafeScale one-shot action runs only as the unhide of its
    probe pods; the mode is re-read (uncached) right before every
    capacity-changing step, so a commit already running stops there. A relay is
    checked right before its one ``POST /v2/transfers``; once sent, the SM
    completes it (design 20261002-controller-transfer).

    Refusals, ``clamped_by_floor`` and ``taken: 0`` are observation events only
    (``wake_refused``, ``transfer_clamped_by_floor``, ``scale_clamped_by_floor``):
    no GPU cooldown, no floor hold - the next tick re-plans from a new view.
    """

    def __init__(
        self,
        client: ServiceManagerClient,
        *,
        is_observe: Callable[[], bool] | None = None,
        prof: "TickProfiler | None" = None,
        now_ms: Callable[[], int] | None = None,
        retry: RetryPolicy | None = None,
        revalidate: Revalidate | None = None,
        revalidate_commit: CommitRevalidate | None = None,
        sleep: Callable[[float], Awaitable[None]] | None = None,
        fresh_view: Callable[[], object] | None = None,
        on_oneshot_done: Callable[[str, str, str], None] | None = None,
        commit_max_age_ms: float | None = None,
        is_observe_fresh: Callable[[], bool] | None = None,
        on_hide_failed: Callable[[str, tuple[str, ...], str], None] | None = None,
        on_hide_done: Callable[[str, tuple[str, ...]], None] | None = None,
        scale_memory: object | None = None,
        scale_memory_max_age_ms: float | None = 50_000.0,
    ) -> None:
        self._client = client
        #: Observation events for the next planner tick (transfer outcomes,
        #: wake_refused, *_clamped_by_floor, placement_substituted), drained by
        #: :meth:`drain_events`. No timer follows any of them (2026-10-02).
        self._events: list[str] = []
        #: P3: (model, pods, error) of a hide that did not take effect (failed, or
        #: not sent because of observe mode): its SafeScale probe is rolled back
        #: instead of being judged as if the pods were hidden.
        self._on_hide_failed = on_hide_failed
        #: 2026-09-29: (model, pods) right after the SM confirmed a hide - the
        #: SafeScale probe anchors its post-hide evidence window at this moment.
        self._on_hide_done = on_hide_done
        #: B8: a SafeScale commit whose decision (``decided_ms``) is older than this
        #: at its FIRST dispatch becomes the donor unhide (None / <= 0 = off).
        self._commit_max_age_ms = (
            float(commit_max_age_ms) if commit_max_age_ms is not None and commit_max_age_ms > 0 else None
        )
        self._pending: deque[QueuedAction] = deque()
        self._inflight: set[str] = set()
        # When this returns True the controller is in observe mode (record only):
        # re-plannable actions are drained (inflight cleared, so the next tick can
        # re-plan) but NEVER dispatched; SafeScale one-shot actions only run in
        # their observe-safe form (the unhide of the probe pods, see _execute).
        self._is_observe = is_observe or (lambda: False)
        #: Uncached mode read, right before every capacity-changing step (a hide,
        #: a transfer's donor / receiver, a commit's donor / each receiver).
        self._is_observe_fresh = is_observe_fresh or self._is_observe
        self._prof = prof
        # Review F4: model -> (epoch ms the last successful dispatch completed, "up"/"down").
        self._now_ms = now_ms or (lambda: int(time.time() * 1000))
        self._last_done: dict[str, tuple[int, str]] = {}
        #: O1: model -> epoch ms the last SM call that can change its routable count
        #: (scale / wake / sleep / hide / unhide / receiver target) returned - ok or not.
        #: A change the SM made during that call happened before this stamp, so it
        #: never dates a breakpoint early. In memory only (a restart starts empty).
        self._routable_change: dict[str, tuple[int, int]] = {}
        #: Timer cleanup review P2-1: model -> (when the last SM call that can change its
        #: routable count returned, ok or not, "up" / "down" hold rule) for the O1
        #: view-pending gate. An unhide (probe rollback) holds like "down": a CRITICAL
        #: receiver passes, LOW receivers and donors wait for a view showing it.
        self._view_change: dict[str, tuple[int, str]] = {}
        #: C1: model -> its last rescue target (see :meth:`rescue_targets`).
        self._rescue: dict[str, RescueTargetRecord] = {}
        #: C1 review P3: id(queued action) -> the record that part belongs to (a part of
        #: a superseded target updates that record, never its successor).
        self._rescue_parts: dict[int, RescueTargetRecord] = {}
        #: C1 review P2-2: durable copy of ``_last_done`` / ``_rescue`` (controller
        #: state store: ``load_scale_memory()`` / ``save_scale_memory(model, record)``),
        #: so a restarted controller neither repeats an unreflected scale-up nor drops
        #: a cooldown. None = in memory only.
        self._scale_memory = scale_memory
        #: P3-1: a persisted target issued longer ago than this is not restored.
        self._scale_memory_max_age_ms = scale_memory_max_age_ms
        #: P3-1: restored targets not yet checked against the live routable count.
        self._restored_unchecked: set[str] = set()
        self._load_scale_memory()
        self._retry = retry or RetryPolicy()
        self._revalidate = revalidate
        self._revalidate_commit = revalidate_commit
        self._sleep = sleep or asyncio.sleep
        #: The cluster view only while fresh (review 4 P2-1), else None.
        self._fresh_view = fresh_view
        #: (request_id, "commit" | "rollback", reason) once a SafeScale one-shot
        #: action is finished for good (review 4 P2-4): the probe is resolved then.
        self._on_oneshot_done = on_oneshot_done
        #: running dispatch task -> its queued action (updated as a commit progresses)
        self._running: dict[asyncio.Future, QueuedAction] = {}
        #: running dispatch task -> its one-shot action waiting out a retry backoff
        self._backoff: dict[asyncio.Future, _Backoff] = {}
        self._closed = False
        self._stats: dict[str, int] = {
            "oneshot_retries_total": 0,
            "oneshot_abandoned_total": 0,
            "oneshot_not_wanted_total": 0,
            "commit_abandoned_total": 0,
            "commit_upscale_dropped_total": 0,
            "commit_receiver_dropped_total": 0,
            "oneshot_preempted_total": 0,
            "dispatch_exceptions_total": 0,
            "sleep_path_refused_total": 0,
            "commit_abandoned_after_gate_total": 0,
            "commit_upscale_preempted_total": 0,
            "commit_failed_unhide_total": 0,
            "unconfirmed_kept_hidden_total": 0,
            "commit_evidence_stale_total": 0,
            "oneshot_cancelled_total": 0,
            # Observe mode (2026-09-28): hides not sent, commits stopped between
            # steps, commits turned into their donor unhide; a relay the SM was
            # already running when observe began is completed by the SM.
            "observe_hide_skipped_total": 0,
            "observe_commit_stopped_total": 0,
            "observe_transfer_completed_total": 0,
            "floor_violation_total": 0,
            # 2026-10-02 relays (POST /v2/transfers) and floor clamps.
            "transfer_total": 0,
            "transfer_pairs_done_total": 0,
            "transfer_partial_total": 0,
            "transfer_failed_total": 0,
            "transfer_clamped_by_floor_total": 0,
            "transfer_unsupported_total": 0,
            "scale_clamped_by_floor_total": 0,
        }

    # ------------------------------------------------------------------ submit
    def submit(self, actions: tuple[Action, ...] | list[Action]) -> SubmitResult:
        queued_actions = tuple(self._queued(action) for action in actions)
        if len(queued_actions) > 1 and all(
            queued.source_loop == "safescale" for queued in queued_actions
        ):
            models = [queued.model for queued in queued_actions]
            has_conflict = len(set(models)) != len(models) or any(
                model in self._inflight or self._has_pending_model(model)
                for model in models
            )
            if has_conflict:
                return SubmitResult(
                    accepted=0,
                    dropped=tuple((model, "atomic_batch_conflict") for model in models),
                )

        accepted = 0
        held = 0
        dropped: list[tuple[str, str]] = []
        replaced: list[tuple[str, SourceLoop]] = []
        observe = self._is_observe()
        #: C1: models whose rescue target this submit (re)started.
        rescue_started: set[str] = set()
        #: C1 review P2-1: pods a commit preemption gave back, per receiver model, not
        #: yet deducted from one of its scale-up parts (one preemption, all parts).
        restore_credit: dict[str, int] = {}

        for queued in queued_actions:
            if queued.source_loop == "rescue":
                for model in queued.models:
                    replaced.extend(self._remove_pending_fairness_for_model(model))
                part = _rescue_part(queued.action)
                preempted, taken = self._preempt_for_rescue(queued, restore_credit)
                if part is not None and taken > 0:
                    # The restored pods count toward the target without a dispatch.
                    self._rescue_record_for(part, rescue_started).gained += taken
                if preempted is None:
                    dropped.append((queued.model, "covered_by_preempted_commit"))
                    continue
                queued = preempted
            elif any(
                model in self._inflight or self._has_pending_model(model)
                for model in queued.models
            ):
                dropped.append((queued.model, "inflight"))
                continue

            self._pending.append(queued)
            self._inflight.update(queued.models)
            accepted += 1
            self._register_rescue(queued.action, rescue_started)
            if observe and queued.source_loop == "safescale":
                held += 1

        for model in rescue_started:
            record = self._rescue[model]
            if record.outstanding == 0 and record.done_ms is None:
                record.done_ms = int(self._now_ms())  # covered by restored pods only
            self._persist_scale_memory(model)

        return SubmitResult(
            accepted=accepted,
            held=held,
            dropped=tuple(dropped),
            replaced=tuple(replaced),
        )

    def pending_actions(self) -> tuple[QueuedAction, ...]:
        return tuple(self._pending)

    def inflight_models(self) -> set[str]:
        return set(self._inflight)

    def preemptible_models(self) -> set[str]:
        """Models whose ONLY in-flight work is a SafeScale commit waiting out a
        retry backoff (review 3 P2-3): a rescue scale-up of such a model is still
        planned, and its submit preempts that retry."""
        candidates: set[str] = set()
        for slot in self._backoff.values():
            if isinstance(slot.queued.action, SafeScaleCommitAction):
                candidates.update(slot.queued.action.touched_models)
        if not candidates:
            return set()
        busy = {
            model
            for task, queued in self._running.items()
            if task not in self._backoff
            for model in queued.models
        }
        busy.update(model for item in self._pending for model in item.models)
        return candidates - busy

    def last_actions(self) -> dict[str, tuple[int, str]]:
        return dict(self._last_done)

    def routable_changes(self) -> dict[str, tuple[int, int]]:
        """O1: model -> (when the last SM call that can change its routable count
        returned, the direction it could move it: +1 / -1) - see ``_routable_change``;
        the only breakpoint dating hint."""
        return dict(self._routable_change)

    def view_changes(self) -> dict[str, tuple[int, str]]:
        """Timer cleanup review P2-1: model -> (when its last routable-changing SM call
        returned - also a failed or partial one, also an unhide -, hold direction). A
        relay records both sides (donor "down", receiver "up"); a call that changed
        nothing (``taken: 0``, a relay side with no pair) records nothing (2026-10-02)."""
        return dict(self._view_change)

    def rescue_targets(self) -> dict[str, RescueTargetRecord]:
        """C1: model -> its last rescue target (copies). The planner tick keeps the
        ones whose effect the decision window does not reflect yet."""
        return {model: replace(record) for model, record in self._rescue.items()}

    def _rescue_record_for(self, part: ScaleAction, started: set[str]) -> RescueTargetRecord:
        """The record of ``part``'s rescue target in this submit: the first part of a
        model starts a new one (it supersedes the previous target of the model)."""
        record = self._rescue.get(part.model)
        if part.model not in started or record is None:
            started.add(part.model)
            plan = part.rescue
            record = RescueTargetRecord(
                target=int(plan.target), desired=int(plan.desired), base=int(plan.base),
                covered_before=int(plan.covered), issued_ms=int(self._now_ms()),
            )
            self._rescue[part.model] = record
            self._restored_unchecked.discard(part.model)
        return record

    def _register_rescue(self, action, started: set[str]) -> None:
        part = _rescue_part(action)
        if part is None:
            return
        record = self._rescue_record_for(part, started)
        record.outstanding += 1
        record.done_ms = None
        self._rescue_parts[id(action)] = record

    def record_rescue_covered(self, model: str, plan) -> None:
        """C1 review P2-1: a rescue target the planner tick found fully covered by the
        pods a SafeScale probe preemption gives back (nothing submitted): recorded as
        done now, so the next ticks hold until the window reflects those pods."""
        self._rescue[model] = RescueTargetRecord(
            target=int(plan.target), desired=int(plan.desired), base=int(plan.base),
            covered_before=int(plan.covered), issued_ms=int(self._now_ms()),
            done_ms=int(self._now_ms()),
        )
        self._persist_scale_memory(model)

    def _note_rescue_result(self, action, result: DispatchResult) -> None:
        """C1: count what a rescue scale-up really added (a partial hinted wake
        counts the replicas the SM picked)."""
        part = _rescue_part(action)
        record = self._rescue_parts.get(id(action)) if part is not None else None
        if record is None:
            return
        if isinstance(part, TransferIntent):
            # A relay adds the receivers the SM really woke (``done``), never ``count``.
            done = int(result.done or 0)
            record.gained += min(int(part.pairs), done)
            if done < int(part.pairs):
                record.failures += 1
            return
        if result.ok:
            record.gained += int(part.delta)
        else:
            record.failures += 1
            record.gained += min(int(part.delta), len(result.picked or ()))

    def _finish_rescue(self, action) -> None:
        """C1: one rescue scale-up ended (done, failed, dropped): the target is done
        once all of its parts ended."""
        record = self._rescue_parts.pop(id(action), None)
        if record is None or record.outstanding <= 0:
            return
        part = _rescue_part(action)
        record.outstanding -= 1
        if record.outstanding == 0:
            record.done_ms = int(self._now_ms())
            if self._rescue.get(part.model) is record:
                self._persist_scale_memory(part.model)
            LOG.info(
                json.dumps(
                    {"event": "rescue_target_done", "model": part.model, "target": record.target,
                     "desired": record.desired, "base": record.base, "covered": record.covered,
                     "failures": record.failures, "issued_ms": record.issued_ms,
                     "done_ms": record.done_ms},
                    sort_keys=True,
                )
            )

    def drain_events(self) -> list[str]:
        events, self._events = self._events, []
        return events

    def _note_wake_refused(self, action, result: DispatchResult) -> None:
        """A located wake refusal (structured 409): an observation event only - the next
        tick re-plans from a new view (the per-GPU / per-node cooldown was removed on
        2026-10-02)."""
        conflict = result.wake_conflict
        if conflict is None or _action_direction(action) != "up":
            return
        self._wake_refused_event(result.model, conflict)

    def _wake_refused_event(self, model: str, conflict: dict) -> None:
        if not conflict.get("node"):
            return
        gpus = [str(gpu) for gpu in conflict.get("gpu_ids") or ()]
        where = f"{conflict['node']}/{','.join(gpus)}"
        self._events.append(f"wake_refused:{model}:{where}:{conflict.get('error')}")
        LOG.info(
            json.dumps(
                {"event": "wake_refused", "model": model, "node": conflict.get("node"),
                 "gpu_ids": conflict.get("gpu_ids") or [], "scope": conflict.get("scope"),
                 "error_code": conflict.get("error"), "reason": conflict.get("reason"),
                 "blocking_binding_id": conflict.get("blocking_binding_id")},
                sort_keys=True,
            )
        )

    def _note_floor_violation(self, result: DispatchResult) -> None:
        """A 409 ``floor_violation`` (binding-level sleep, SafeScale hide): counted and
        logged; no hold (removed 2026-10-02) - the donor's ``floor_headroom`` in the next
        view bounds the planner."""
        if not result.floor_violation or result.model == CLUSTER_MODEL:
            return
        self._stats["floor_violation_total"] += 1
        self._events.append(f"floor_violation:{result.model}")
        LOG.warning(
            json.dumps({"event": "floor_violation", "model": result.model, "error": result.error}, sort_keys=True)
        )

    def has_request(self, request_id: str) -> bool:
        """A one-shot action of SafeScale probe ``request_id`` is queued, running
        or backing off (review 4 P2-4: a ``committing`` probe without one is
        recovered by the SafeScale loop)."""
        items = list(self._pending) + list(self._running.values()) + [
            slot.queued for slot in self._backoff.values()
        ]
        return any(_request_id(item.action) == request_id for item in items)

    def cancel_request(self, request_id: str) -> bool:
        """Drop the QUEUED (not yet started, e.g. blocked behind a running action) one-shot
        actions of SafeScale probe ``request_id`` without dispatching them - B8:
        the probe was resolved without an SM call (its pods are gone). Returns
        False when one of its actions is already running (or backing off): that
        one finishes on its own and the caller retries later."""
        running = list(self._running.values()) + [slot.queued for slot in self._backoff.values()]
        if any(_request_id(item.action) == request_id for item in running):
            return False
        kept: deque[QueuedAction] = deque()
        for item in self._pending:
            if item.source_loop in ONE_SHOT_LOOPS and _request_id(item.action) == request_id:
                self._stats["oneshot_cancelled_total"] += 1
                LOG.warning(
                    "one-shot %s of %s (%s) cancelled before dispatch: its probe was resolved",
                    _action_kind(item.action), item.model, request_id,
                )
                continue
            kept.append(item)
        self._pending = kept
        self._release_idle_models()
        return True

    def cluster_action_active(self) -> bool:
        """A defrag (cluster-wide action) is queued or running: every other
        action - rescue included - waits behind it (review 4 P3)."""
        items = list(self._pending) + list(self._running.values())
        return any(CLUSTER_RESOURCE in item.resources for item in items)

    def stats(self) -> dict[str, int]:
        """Counters: one-shot retries / abandons / preemptions, SafeScale commits
        abandoned or trimmed by revalidation, receiver wakes dropped after a failed
        donor sleep, dispatch exceptions; gauges ``defrag_active`` and
        ``pending_behind_defrag`` (actions waiting for a defrag, review 4 P3)."""
        stats = dict(self._stats)
        defrag = self.cluster_action_active()
        stats["defrag_active"] = int(defrag)
        stats["pending_behind_defrag"] = (
            sum(1 for item in self._pending if CLUSTER_RESOURCE not in item.resources) if defrag else 0
        )
        return stats

    # ------------------------------------------------------------------ running
    async def run(
        self,
        *,
        poll_interval_s: float = 0.1,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        """Start dispatch tasks without waiting for them: a slow SM call (a
        scale-down that drains for minutes) never delays an action on unrelated
        resources. Cancelling ``run`` cancels every dispatch in progress."""
        try:
            while True:
                self._dispatch_pending([])
                await sleep(poll_interval_s)
        except asyncio.CancelledError:
            await self.shutdown()
            raise

    async def shutdown(self) -> None:
        """Stop dispatching: cancel running dispatches and wait for them to end
        (controller shutdown, review 2 P3)."""
        self._closed = True
        tasks = [task for task in self._running if not task.done()]
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        self._running.clear()
        self._backoff.clear()

    async def drain_once(self) -> tuple[DispatchResult, ...]:
        """Dispatch everything pending and wait until it is done - concurrently
        where resources allow, in order where they overlap (tests / offline
        integration)."""
        results: list[DispatchResult] = []
        self._dispatch_pending(results)
        while self._running:
            tasks = list(self._running)
            await asyncio.gather(*tasks, return_exceptions=True)
        return tuple(results)

    def _dispatch_pending(self, results: list[DispatchResult]) -> None:
        if self._closed:
            return
        busy: set[str] = set()
        for queued in self._running.values():
            busy |= queued.resources
        # Observe mode (user decision 2026-09-28: record only): re-plannable
        # actions are dropped. SafeScale one-shot actions are never re-emitted (the
        # state machine deletes a probe on resolve), so they are neither dropped
        # nor held: they are dispatched and _execute runs only their observe-safe
        # form - the unhide of the probe pods (the controller undoing its own
        # half-action); a commit's donor sleep / receiver wakes are dropped.
        observe = self._is_observe()
        retained: deque[QueuedAction] = deque()
        for queued in self._pending:
            if _conflicts(queued.resources, busy):
                # Blocked behind a running or an earlier queued action on a shared
                # resource: keep the order.
                busy |= queued.resources
                retained.append(queued)
                continue
            if observe and not _runs_in_observe(queued):
                results.extend(_observe_skipped(queued))
                self._notify_hide_failed(queued.action, "observe_skipped")
                self._finish_rescue(queued.action)
                continue
            busy |= queued.resources
            task = asyncio.ensure_future(self._run_item(queued, results))
            self._running[task] = queued
        self._pending = retained
        if observe:
            self._release_idle_models()

    async def _run_item(self, queued: QueuedAction, results: list[DispatchResult]) -> None:
        task = asyncio.current_task()
        request_id = _request_id(queued.action) if queued.source_loop in ONE_SHOT_LOOPS else None
        done: tuple[str, str] | None = None
        try:
            done = await self._execute(queued, results)
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 - surfaced as failed results (review 3 P3)
            self._stats["dispatch_exceptions_total"] += 1
            current = self._running.get(task, queued)
            LOG.exception("action dispatch for %s failed", current.model)
            error = f"dispatch_exception: {type(exc).__name__}: {exc}"
            results.extend(
                DispatchResult(model=model, action_kind=_action_kind(current.action), ok=False, error=error)
                for model in current.models
            )
        finally:
            self._running.pop(task, None)
            self._backoff.pop(task, None)
            self._finish_rescue(queued.action)
            self._release_idle_models()
            if request_id is not None and done is not None:
                self._notify_done(request_id, *done)
            if not self._closed:
                # Start whatever this action was blocking right away.
                self._dispatch_pending(results)

    def _notify_hide_failed(self, action, error: str | None) -> None:
        if not isinstance(action, HideAction) or self._on_hide_failed is None:
            return
        try:
            self._on_hide_failed(action.model, tuple(action.pods), f"hide_failed: {error or 'unknown'}")
        except Exception:  # noqa: BLE001 - the probe is still bounded by its window
            LOG.exception("marking the SafeScale probe of %s after a failed hide failed", action.model)

    def _notify_hide_done(self, action) -> None:
        if not isinstance(action, HideAction) or self._on_hide_done is None:
            return
        try:
            self._on_hide_done(action.model, tuple(action.pods))
        except Exception:  # noqa: BLE001 - the probe then extends / rolls back (hide_unconfirmed)
            LOG.exception("anchoring the SafeScale probe of %s after its hide failed", action.model)

    def _notify_done(self, request_id: str, status: str, reason: str) -> None:
        if self._on_oneshot_done is None:
            return
        try:
            self._on_oneshot_done(request_id, status, reason)
        except Exception:  # noqa: BLE001 - the probe is recovered on the next tick
            LOG.exception("resolving SafeScale probe %s failed", request_id)

    async def _execute(
        self, queued: QueuedAction, results: list[DispatchResult]
    ) -> tuple[str, str] | None:
        """Run one queued action to its end. Returns how a one-shot action
        finished - ("commit" | "rollback", reason) - or None when it is not
        finished (dropped in observe mode, controller shutting down)."""
        if not _runs_in_observe(queued) and self._is_observe_fresh():
            results.extend(_observe_skipped(queued))
            self._notify_hide_failed(queued.action, "observe_skipped")
            return None
        if isinstance(queued.action, TransferIntent):
            results.extend(await self._execute_transfer(queued.action))
            return ("done", "transfer")
        if isinstance(queued.action, SafeScaleCommitAction):
            converted, finished = self._stale_commit(queued, results)
            if finished is not None:
                return finished
            if converted is not None:
                queued = converted
                self._set_running(queued)
        while True:
            if isinstance(queued.action, SafeScaleCommitAction):
                # Before every (re)try, i.e. right before the donor sleep: in
                # observe the commit becomes the unhide of its donor pods.
                if self._is_observe_fresh():
                    converted, finished = self._observe_commit(queued, results)
                    if finished is not None:
                        return finished
                    queued = converted
                    self._set_running(queued)
                    continue
                revalidated, finished = self._revalidated_commit(queued, results)
                if revalidated is None:
                    return finished
                queued = revalidated
                self._set_running(queued)
            failure, queued = await self._attempt(queued, results)
            if failure is None:
                return _resolution(queued.action)
            attempts = queued.failures + 1
            terminal = (
                queued.source_loop not in ONE_SHOT_LOOPS
                or not failure.retriable
                or not _retry_safe(queued.action)
            )
            if terminal or attempts >= self._retry.max_attempts:
                prefix = ""
                if not terminal:
                    prefix = f"abandoned after {attempts} attempts: "
                    self._stats["oneshot_abandoned_total"] += 1
                    LOG.error(
                        "one-shot action abandoned after %d attempts: %s",
                        attempts,
                        json.dumps(
                            {"model": queued.model, "action": _action_kind(queued.action),
                             "reason": getattr(queued.action, "reason", None), "error": failure.error},
                            sort_keys=True,
                        ),
                    )
                results.extend(self._final_failures(queued, failure, attempts=attempts, prefix=prefix))
                self._notify_hide_failed(queued.action, failure.error)
                # A commit whose donor sleep failed for good leaves the donor's
                # probe pods hidden but awake: give them their routing back
                # (review 4 P2-4), never on an unconfirmed pod (P2-2).
                unhide = self._unhide_after_failed_commit(queued, failure)
                if unhide is not None:
                    queued = unhide
                    self._set_running(queued)
                    continue
                return _resolution(queued.action, failure)
            backoff = self._retry.backoff_s(attempts)
            self._stats["oneshot_retries_total"] += 1
            LOG.warning(
                "one-shot %s of %s failed retriably (%s); retry %d/%d in %.1fs",
                _action_kind(queued.action), queued.model, failure.error,
                attempts + 1, self._retry.max_attempts, backoff,
            )
            queued = replace(queued, failures=attempts)
            self._set_running(queued)
            queued = await self._wait_backoff(queued, backoff, results)
            if self._closed:
                return None
            if isinstance(queued.action, SafeScaleCommitAction):
                # Revalidated at the top of the loop (observe: its donor unhide).
                continue
            if not _runs_in_observe(queued) and self._is_observe_fresh():
                # Observe entered while backing off: no retry (record only).
                results.extend(_observe_skipped(queued))
                return None
            reason = self._still_wanted(queued.action)
            if reason is not None:
                self._stats["oneshot_not_wanted_total"] += 1
                LOG.warning(
                    "one-shot %s of %s not retried: no longer wanted (%s)",
                    _action_kind(queued.action), queued.model, reason,
                )
                results.append(
                    DispatchResult(
                        model=queued.model,
                        action_kind=_action_kind(queued.action),
                        ok=False,
                        error=f"not_retried: {reason}",
                        attempts=queued.failures,
                    )
                )
                return _resolution(queued.action, reason=f"not_retried: {reason}")

    def _unhide_after_failed_commit(
        self, queued: QueuedAction, failure: DispatchResult
    ) -> QueuedAction | None:
        action = queued.action
        if not isinstance(action, SafeScaleCommitAction) or action.donor_done or not action.pods:
            return None
        unhide = self._donor_unhide(action, "safescale_commit_failed", filter_by_view=True)
        if unhide is None:
            return None
        self._stats["commit_failed_unhide_total"] += 1
        LOG.warning(
            "SafeScale commit of %s (%s) failed for good (%s): unhiding %s, keeping %s hidden",
            action.donor, action.request_id, failure.error, list(unhide.pods), list(unhide.keep_hidden),
        )
        return self._queued(unhide)

    def _stale_commit(
        self, queued: QueuedAction, results: list[DispatchResult]
    ) -> tuple[QueuedAction | None, tuple[str, str] | None]:
        """B8: a commit about to be dispatched for the FIRST time (no attempt yet)
        whose decision is older than ``commit_max_age_ms`` - it was held in
        observe mode, or re-submitted after a restart - is not run on that stale
        evidence. The donor's hidden pods get their routing back instead (only
        those a fresh cluster view shows awake and hidden; never an unconfirmed
        one) and the probe resolves as a rollback ``commit_evidence_stale``.
        Retries of a commit that already made an attempt are not aged (their
        evidence was acted on; the revalidation before every retry still runs).
        Returns (the action to run instead, None), (None, how the commit
        finished) or (None, None) = run the commit as planned."""
        action: SafeScaleCommitAction = queued.action
        if self._commit_max_age_ms is None or queued.failures > 0 or action.decided_ms is None:
            return None, None
        age_ms = float(self._now_ms()) - float(action.decided_ms)
        if age_ms <= self._commit_max_age_ms:
            return None, None
        reason = "commit_evidence_stale"
        self._stats["commit_evidence_stale_total"] += 1
        LOG.warning(
            json.dumps(
                {"event": "safescale_commit_evidence_stale", "donor": action.donor,
                 "request_id": action.request_id, "pods": list(action.pods),
                 "age_s": round(age_ms / 1000.0, 1), "max_age_s": round(self._commit_max_age_ms / 1000.0, 1)},
                sort_keys=True,
            )
        )
        detail = f"{reason}: decided {age_ms / 1000.0:.0f}s ago > {self._commit_max_age_ms / 1000.0:.0f}s"
        results.extend(self._receivers_dropped(action, detail))
        if action.donor_done or not action.pods:
            return None, ("rollback", f"{detail}; follow-up upscales dropped")
        unhide = self._donor_unhide(action, reason, filter_by_view=True, only_view_hidden=True)
        if unhide is None:
            return None, ("rollback", f"{detail}; no donor pod to unhide")
        return self._queued(unhide), None

    def _observe_commit(
        self, queued: QueuedAction, results: list[DispatchResult]
    ) -> tuple[QueuedAction | None, tuple[str, str] | None]:
        """Observe mode (2026-09-28): a SafeScale commit is not carried out. The
        donor not slept yet: its hidden probe pods get their routing back (the
        only action allowed in observe; never an unconfirmed pod, never one a
        fresh view shows asleep) and the probe resolves as a rollback
        ``observe_entered``. The donor already slept: the pending receiver wakes
        are dropped (recorded) and the probe resolves as the commit it was.
        Returns (the unhide to run, None) or (None, how the commit finished)."""
        action: SafeScaleCommitAction = queued.action
        self._stats["observe_commit_stopped_total"] += 1
        LOG.warning(
            json.dumps(
                {"event": "observe_entered_commit_stopped", "donor": action.donor,
                 "request_id": action.request_id, "pods": list(action.pods),
                 "donor_done": action.donor_done,
                 "receivers_not_woken": [item.model for item in action.upscales]},
                sort_keys=True,
            )
        )
        results.extend(self._receivers_dropped(action, "observe_entered: receiver wake not issued"))
        if action.donor_done or not action.pods:
            return None, ("commit", f"{action.reason}; observe_entered: follow-up upscales not issued")
        unhide = self._donor_unhide(action, "observe_entered", filter_by_view=True)
        if unhide is None:
            return None, ("rollback", "observe_entered; no donor pod to unhide")
        return self._queued(unhide), None

    def _donor_unhide(
        self,
        commit: SafeScaleCommitAction,
        reason: str,
        *,
        filter_by_view: bool = False,
        only_view_hidden: bool = False,
    ) -> UnhideAction | None:
        """The unhide giving a commit's donor pods their routing back, keeping
        every pod whose sleep is unconfirmed hidden (review 4 P2-2). With
        ``filter_by_view`` and a fresh cluster view, pods it shows asleep are
        left out (they need nothing); with ``only_view_hidden`` too, so are pods
        it does not list or shows already routable (B8: only pods the view shows
        awake AND hidden). None = nothing to unhide."""
        unconfirmed = set(commit.unconfirmed_pods)
        keep = tuple(pod for pod in commit.pods if pod in unconfirmed)
        candidates = [pod for pod in commit.pods if pod not in unconfirmed]
        if filter_by_view:
            view = self._view()
            if view is not None:
                bindings = {binding.serve_id: binding for binding in getattr(view, "bindings", ())}
                if only_view_hidden:
                    candidates = [
                        pod for pod in candidates
                        if pod in bindings and bindings[pod].awake and bindings[pod].hidden
                    ]
                else:
                    candidates = [
                        pod for pod in candidates if pod not in bindings or bindings[pod].awake
                    ]
        if keep:
            self._stats["unconfirmed_kept_hidden_total"] += len(keep)
            LOG.warning(
                "SafeScale donor pods %s of %s stay hidden: their sleep is unconfirmed "
                "(the service-manager's crash recovery resolves them)",
                list(keep), commit.donor,
            )
        if not candidates:
            return None
        return UnhideAction(
            commit.donor, tuple(candidates), reason, commit.source_loop,
            keep_hidden=keep, request_id=commit.request_id,
        )

    def _view(self):
        if self._fresh_view is None:
            return None
        try:
            return self._fresh_view()
        except Exception:  # noqa: BLE001 - no view = conservative behaviour
            return None

    async def _attempt(
        self, queued: QueuedAction, results: list[DispatchResult]
    ) -> tuple[DispatchResult | None, QueuedAction]:
        """One try. Returns (the failure to retry or report, None = done; the
        queued action with the progress made)."""
        action = queued.action
        attempts = queued.failures + 1
        if isinstance(action, SafeScaleCommitAction):
            return await self._attempt_commit(queued, results)
        result = await self._timed_dispatch(action, queued.model)
        self._note_rescue_result(action, result)
        if not result.ok:
            return result, queued
        self._record_done(queued.model, action, result)
        results.append(replace(result, attempts=attempts))
        self._notify_hide_done(action)
        return None, queued

    async def _attempt_commit(
        self, queued: QueuedAction, results: list[DispatchResult]
    ) -> tuple[DispatchResult | None, QueuedAction]:
        """Sleep the donor's hidden pods, then (only then) bring every receiver
        up to its absolute target. Progress is kept on the action: a retry never
        re-sleeps a donor that slept and re-sends only the pending upscales."""
        action: SafeScaleCommitAction = queued.action
        attempts = queued.failures + 1
        if not action.donor_done:
            donor = action.donor_sleep()
            result = await self._timed_dispatch(donor, action.donor)
            if not result.ok:
                if result.unconfirmed:
                    action = replace(
                        action,
                        unconfirmed_pods=tuple(dict.fromkeys(action.unconfirmed_pods + result.unconfirmed)),
                    )
                    queued = self._commit_queued(action, failures=queued.failures)
                    self._set_running(queued)
                return result, queued
            self._record_done(action.donor, donor, result)
            results.append(replace(result, attempts=attempts))
            action = replace(action, donor_done=True)
            queued = self._commit_queued(action, failures=queued.failures)
            self._set_running(queued)
        remaining: list[ReceiverTarget] = []
        failure: DispatchResult | None = None
        for index, upscale in enumerate(action.upscales):
            if upscale.target is None:
                upscale, resolve_failure = await self._resolve_target(upscale)
                if resolve_failure is not None:
                    if resolve_failure.retriable:
                        remaining.append(upscale)
                        failure = failure or resolve_failure
                    else:
                        results.append(replace(resolve_failure, attempts=attempts))
                    continue
            if self._is_observe_fresh():
                # Observe entered mid-commit (2026-09-28): the donor slept, the
                # receivers left are NOT woken - recorded, the probe resolves.
                stopped = replace(
                    action,
                    upscales=tuple(remaining) + (upscale,) + tuple(action.upscales[index + 1:]),
                    reason=f"{action.reason}; observe_entered: follow-up upscales not issued",
                )
                results.extend(self._receivers_dropped(stopped, "observe_entered: receiver wake not issued"))
                self._stats["observe_commit_stopped_total"] += 1
                LOG.warning(
                    json.dumps(
                        {"event": "observe_entered_mid_commit", "donor": action.donor,
                         "request_id": action.request_id, "donor_slept": list(action.pods),
                         "receivers_not_woken": [item.model for item in stopped.upscales]},
                        sort_keys=True,
                    )
                )
                queued = self._commit_queued(replace(stopped, upscales=()), failures=queued.failures)
                self._set_running(queued)
                return None, queued
            result = await self._timed_dispatch(upscale, upscale.model)
            if result.ok:
                self._record_done(upscale.model, upscale, result)
                results.append(replace(result, attempts=attempts))
            elif result.retriable:
                remaining.append(upscale)
                failure = failure or result
            else:
                results.append(replace(result, attempts=attempts))
        action = replace(action, upscales=tuple(remaining))
        queued = self._commit_queued(action, failures=queued.failures)
        self._set_running(queued)
        return failure, queued

    async def _resolve_target(
        self, upscale: ReceiverTarget
    ) -> tuple[ReceiverTarget, DispatchResult | None]:
        """No absolute target at planning time: resolve it ONCE from the SM's
        awake count; the retry then re-sends that same target."""
        try:
            response = await self._client.model_awake(upscale.model)
        except Exception as exc:  # noqa: BLE001
            return upscale, DispatchResult(
                model=upscale.model, action_kind="scale", ok=False,
                error=f"dispatch_exception: {type(exc).__name__}: {exc}",
            )
        if not bool(response.get("ok", False)):
            return upscale, _dispatch_result(model=upscale.model, action_kind="scale", response=response)
        awake = int(response["awake"])
        target = awake + max(0, int(upscale.delta))
        if upscale.cap is not None:
            target = min(target, max(int(upscale.cap), awake))
        return replace(upscale, target=target), None

    def _revalidated_commit(
        self, queued: QueuedAction, results: list[DispatchResult]
    ) -> tuple[QueuedAction | None, tuple[str, str] | None]:
        """Before every (re)try of a commit: the donor needing capacity again
        abandons the commit (its hidden pods are unhidden instead, never an
        unconfirmed one); receivers that no longer need capacity lose their
        upscale. Returns (the action to run, None) or (None, how the commit
        finished)."""
        action: SafeScaleCommitAction = queued.action
        verdict = CommitVerdict(action)
        if self._revalidate_commit is not None:
            try:
                verdict = self._revalidate_commit(action)
            except Exception as exc:  # noqa: BLE001 - keep the commit on a lookup error
                LOG.warning("revalidating the SafeScale commit of %s failed (kept): %r", action.donor, exc)
        if verdict.abandon_reason is not None:
            self._stats["commit_abandoned_total"] += 1
            if queued.failures == 0:
                # Abandoned on its first dispatch, i.e. right after the probe
                # passed its commit gate: the gate (probe-window tail of the
                # donor's serving pods) and this revalidation (the planner's
                # latest whole-model state) disagreed (review 4 P3).
                self._stats["commit_abandoned_after_gate_total"] += 1
                LOG.warning(
                    json.dumps(
                        {"event": "safescale_commit_abandoned_after_gate", "donor": action.donor,
                         "request_id": action.request_id, "reason": verdict.abandon_reason},
                        sort_keys=True,
                    )
                )
            LOG.warning(
                "SafeScale commit of %s (%s) abandoned, unhiding %s: %s",
                action.donor, action.request_id, list(action.pods), verdict.abandon_reason,
            )
            results.extend(self._receivers_dropped(action, f"commit_abandoned: {verdict.abandon_reason}"))
            unhide = self._donor_unhide(action, "safescale_commit_abandoned")
            if unhide is None:
                return None, ("rollback", f"commit_abandoned: {verdict.abandon_reason}; donor pods left hidden")
            return self._queued(unhide), None
        for model, reason in verdict.dropped:
            self._stats["commit_upscale_dropped_total"] += 1
            LOG.warning("SafeScale follow-up upscale of %s dropped: %s", model, reason)
            results.append(
                DispatchResult(model=model, action_kind="scale", ok=False, error=f"not_wanted: {reason}",
                               attempts=queued.failures)
            )
        action = verdict.action
        if action.donor_done and not action.upscales:
            return None, ("commit", action.reason)
        return self._commit_queued(action, failures=queued.failures), None

    async def _wait_backoff(
        self, queued: QueuedAction, backoff_s: float, results: list[DispatchResult]
    ) -> QueuedAction:
        """Wait out a retry backoff; a rescue preemption ends it early and may
        replace the action. Returns the action to continue with."""
        task = asyncio.current_task()
        slot = _Backoff(queued)
        self._backoff[task] = slot
        sleeper = asyncio.ensure_future(self._sleep(backoff_s))
        waker = asyncio.ensure_future(slot.event.wait())
        try:
            await asyncio.wait({sleeper, waker}, return_when=asyncio.FIRST_COMPLETED)
        finally:
            for pending in (sleeper, waker):
                if not pending.done():
                    pending.cancel()
            await asyncio.gather(sleeper, waker, return_exceptions=True)
            self._backoff.pop(task, None)
        results.extend(slot.results)
        return slot.queued

    # -------------------------------------------------------------- preemption
    def _load_scale_memory(self) -> None:
        """C1 review P2-2: restore ``_last_done`` / ``_rescue`` of a previous controller.
        A target that was still running then is taken as done now (its wake may have
        completed just before the restart): the next windows hold until they reflect
        it."""
        loader = getattr(self._scale_memory, "load_scale_memory", None)
        if not callable(loader):
            return
        try:
            memory = loader() or {}
        except Exception as exc:  # noqa: BLE001 - durable memory is best effort
            LOG.warning("scale memory not loaded: %r", exc)
            return
        now = int(self._now_ms())
        for model, entry in memory.items():
            if not isinstance(entry, dict):
                continue
            last = entry.get("last_done")
            try:
                if last:
                    self._last_done[str(model)] = (int(last[0]), str(last[1]))
                rescue = entry.get("rescue")
                max_age = self._scale_memory_max_age_ms
                if rescue and max_age is not None and max_age > 0 and now - int(rescue["issued_ms"]) > max_age:
                    LOG.info(json.dumps({"event": "scale_memory_rescue_expired", "model": str(model),
                                         "issued_ms": int(rescue["issued_ms"]), "now_ms": now},
                                        sort_keys=True))
                    rescue = None
                if rescue:
                    done = rescue.get("done_ms")
                    self._rescue[str(model)] = RescueTargetRecord(
                        target=int(rescue["target"]), desired=int(rescue["desired"]),
                        base=int(rescue["base"]), covered_before=int(rescue["covered_before"]),
                        issued_ms=int(rescue["issued_ms"]), gained=int(rescue.get("gained", 0)),
                        failures=int(rescue.get("failures", 0)),
                        done_ms=now if done is None or int(rescue.get("outstanding", 0)) > 0 else int(done),
                    )
                    self._restored_unchecked.add(str(model))
            except (KeyError, TypeError, ValueError, IndexError) as exc:
                LOG.warning("scale memory of %s ignored: %r", model, exc)
        if self._rescue or self._last_done:
            LOG.info(json.dumps({"event": "scale_memory_restored", "rescue": sorted(self._rescue),
                                 "last_done": sorted(self._last_done)}, sort_keys=True))

    def check_restored_targets(self, routable: Mapping[str, int]) -> list[str]:
        """C1 review P3-1: once, at the first tick after a restart, drop every restored
        rescue target the fleet contradicts - fewer routable replicas now than the
        target had counted before its scale-up (a pod went away meanwhile): holding
        on it would starve the model for a window. Returns the dropped models."""
        dropped: list[str] = []
        for model in sorted(self._restored_unchecked):
            if model not in routable:
                continue
            self._restored_unchecked.discard(model)
            record = self._rescue.get(model)
            if record is not None and int(routable[model]) < record.covered_before:
                del self._rescue[model]
                dropped.append(model)
                LOG.info(json.dumps({"event": "scale_memory_rescue_dropped", "model": model,
                                     "routable": int(routable[model]),
                                     "covered_before": record.covered_before}, sort_keys=True))
                self._persist_scale_memory(model)
        return dropped

    def _persist_scale_memory(self, model: str) -> None:
        saver = getattr(self._scale_memory, "save_scale_memory", None)
        if not callable(saver):
            return
        last = self._last_done.get(model)
        record = self._rescue.get(model)
        entry = {
            "last_done": list(last) if last else None,
            "rescue": None if record is None else {
                "target": record.target, "desired": record.desired, "base": record.base,
                "covered_before": record.covered_before, "issued_ms": record.issued_ms,
                "outstanding": record.outstanding, "gained": record.gained,
                "failures": record.failures, "done_ms": record.done_ms,
            },
        }
        try:
            saver(model, entry)
        except Exception as exc:  # noqa: BLE001 - durable memory is best effort
            LOG.warning("scale memory of %s not saved: %r", model, exc)

    def _preempt_for_rescue(
        self, queued: QueuedAction, credit: dict[str, int] | None = None
    ) -> tuple[QueuedAction | None, int]:
        """A rescue action scaling up a model whose SafeScale commit waits out a
        retry backoff preempts that commit (review 3 P2-3). The donor itself
        needing capacity: the commit becomes an unhide of its hidden pods; the
        rescue scale-up shrinks by the pods that restores - those a fresh cluster
        view shows awake and hidden (review 4 P2-3; v1 ``up_needed = delta -
        probe_hidden``); a receiver: its pending upscale is cancelled (the rescue
        action replaces it). None = the rescue action is fully covered by the
        restored pods. Also returns how many restored pods this action used.

        ``credit`` (C1 review P2-1, per submit): pods restored by an earlier part's
        preemption that it did not use - deducted from the model's later parts, so
        several scale-up parts of one receiver never each count the same pods."""
        credit = {} if credit is None else credit
        action = queued.action
        if isinstance(action, TransferIntent):
            model, delta = action.receiver_model, action.pairs
        elif isinstance(action, ScaleAction) and action.delta > 0:
            model, delta = action.model, action.delta
        else:
            return queued, 0
        restored = 0
        for task, slot in list(self._backoff.items()):
            commit = slot.queued.action
            if isinstance(commit, SafeScaleCommitAction) and model in commit.touched_models:
                restored += self._preempt_commit(task, slot, model)
        available = credit.get(model, 0) + restored
        if available <= 0:
            return queued, 0
        if isinstance(action, TransferIntent):
            # A relay is not split: covered entirely, or kept whole (credit kept).
            if available >= delta:
                credit[model] = available - delta
                return None, delta
            credit[model] = available
            return queued, 0
        taken = min(available, delta)
        credit[model] = available - taken
        up_needed = delta - taken
        if up_needed <= 0:
            return None, taken
        return self._queued(
            replace(action, delta=up_needed, pods=tuple(action.pods[:up_needed]) if action.pods else ())
        ), taken

    def _preempt_commit(self, task: asyncio.Future, slot: _Backoff, model: str) -> int:
        """Returns how many awake pods the preemption gives back to ``model``."""
        commit: SafeScaleCommitAction = slot.queued.action
        if model == commit.donor and not commit.donor_done:
            unhide = self._donor_unhide(commit, "safescale_commit_preempted")
            if unhide is None:
                return 0  # every donor pod unconfirmed: nothing to give back
            self._stats["oneshot_preempted_total"] += 1
            LOG.warning(
                "SafeScale commit of %s (%s) preempted by a rescue scale-up of %s: unhiding %s",
                commit.donor, commit.request_id, model, list(unhide.pods),
            )
            restored = self._restored_capacity(unhide.pods)
            slot.results.extend(self._receivers_dropped(commit, "commit_preempted_by_rescue"))
            slot.queued = self._queued(unhide)
            self._running[task] = slot.queued
            slot.event.set()
            self._release_idle_models()
            return restored
        if not any(item.model == model for item in commit.upscales):
            return 0  # e.g. the donor of a commit that already slept: nothing to preempt
        # A receiver: its pending upscale is cancelled (the rescue action replaces
        # it). The commit's backoff continues for its other receivers - it is not
        # cut short and not counted as a preemption (review 4 P3).
        self._stats["commit_upscale_preempted_total"] += 1
        LOG.warning("SafeScale follow-up upscale of %s preempted by a rescue action", model)
        slot.results.append(
            DispatchResult(model=model, action_kind="scale", ok=False, error="preempted_by_rescue")
        )
        trimmed = replace(
            commit, upscales=tuple(item for item in commit.upscales if item.model != model)
        )
        slot.queued = self._commit_queued(trimmed, failures=slot.queued.failures)
        self._running[task] = slot.queued
        if trimmed.donor_done and not trimmed.upscales:
            slot.event.set()  # nothing left to wait for
        self._release_idle_models()
        return 0

    def _restored_capacity(self, pods: tuple[str, ...]) -> int:
        """How many of ``pods`` the unhide gives back as serving capacity (review
        4 P2-3): those a FRESH cluster view shows awake and hidden. Without one
        nothing is assumed restored (the rescue scale-up is not shrunk)."""
        view = self._view()
        if view is None:
            return 0
        bindings = {binding.serve_id: binding for binding in getattr(view, "bindings", ())}
        return sum(
            1 for pod in pods if pod in bindings and bindings[pod].awake and bindings[pod].hidden
        )

    # ------------------------------------------------------------------ helpers
    def _final_failures(
        self, queued: QueuedAction, failure: DispatchResult, *, attempts: int, prefix: str = ""
    ) -> list[DispatchResult]:
        """The results of an action that is given up: its failure, plus one per
        commit receiver that will not be woken."""
        action = queued.action
        error = f"{prefix}{failure.error}"
        if isinstance(action, SafeScaleCommitAction):
            if not action.donor_done:
                out = [replace(failure, error=error, attempts=attempts)]
                out.extend(self._receivers_dropped(action, f"donor_sleep_failed: {failure.error}"))
                return out
            return [
                DispatchResult(model=item.model, action_kind="scale", ok=False, error=error,
                               retriable=failure.retriable, attempts=attempts)
                for item in action.upscales
            ]
        return [replace(failure, error=error, attempts=attempts)]

    def _receivers_dropped(self, action: SafeScaleCommitAction, reason: str) -> list[DispatchResult]:
        if action.upscales:
            self._stats["commit_receiver_dropped_total"] += len(action.upscales)
        return [
            DispatchResult(model=item.model, action_kind="scale", ok=False, error=reason)
            for item in action.upscales
        ]

    def _set_running(self, queued: QueuedAction) -> None:
        task = asyncio.current_task()
        if task in self._running:
            self._running[task] = queued
            self._release_idle_models()

    async def _execute_transfer(self, action: TransferIntent) -> list[DispatchResult]:
        """One relay = one ``POST /v2/transfers`` (2026-10-02). Accounting follows the
        response, never ``count``: the donor side by ``taken`` (donor replicas slept),
        the receiver side by ``done`` (receivers woken); a 200 may be partial. Pairs the
        SM left to its recovery keep both models held here until ``GET /v2/transfers``
        no longer lists the transfer."""
        result = await self._timed_dispatch(action, action.receiver_model)
        summary = result.transfer or {}
        donor_result = DispatchResult(
            model=action.donor_model, action_kind="transfer", ok=result.ok, error=result.error,
            retriable=False, taken=result.taken, transfer=result.transfer, not_executed=result.not_executed,
        )
        self._note_rescue_result(action, result)
        self._record_transfer_done(action, result)
        if result.ok and self._is_observe_fresh():
            # Observe began while the SM ran the relay: it was sent before (the mode is
            # read right before the call) and the SM completes it - recorded only.
            self._stats["observe_transfer_completed_total"] += 1
            LOG.warning(
                json.dumps(
                    {"event": "observe_entered_during_transfer", "donor": action.donor_model,
                     "receiver": action.receiver_model, "transfer_id": summary.get("transfer_id"),
                     "done": result.done, "taken": result.taken},
                    sort_keys=True,
                )
            )
        return [donor_result, result]

    async def _timed_dispatch(self, action, model: str) -> DispatchResult:
        """One SM call. An exception (a client bug, a malformed answer) becomes a
        failed, non-retriable result instead of killing the dispatch task."""
        started_ns = time.perf_counter_ns()
        refused = False
        try:
            with sm_actor(_actor_for(action)):
                result = await self._dispatch(action, model)
        except asyncio.CancelledError:
            raise
        except SleepPathRefused as exc:
            # Nothing reached the SM: own counter, one error line, no traceback, and
            # no routable-change stamp below (O1 must not see a change that never was).
            refused = True
            self._stats["sleep_path_refused_total"] += 1
            LOG.error(
                json.dumps(
                    {"event": "scale_down_without_sleep_path_refused", "model": model,
                     "reason": str(getattr(action, "reason", "")),
                     "delta": getattr(action, "delta", None)},
                    sort_keys=True,
                )
            )
            result = DispatchResult(
                model=model, action_kind=_action_kind(action), ok=False,
                error=f"sleep_path_refused: {exc}", retriable=False,
            )
        except Exception as exc:  # noqa: BLE001 - review 3 P3
            self._stats["dispatch_exceptions_total"] += 1
            LOG.exception("SM call for %s (%s) raised", model, _action_kind(action))
            result = DispatchResult(
                model=model,
                action_kind=_action_kind(action),
                ok=False,
                error=f"dispatch_exception: {type(exc).__name__}: {exc}",
            )
        self._note_floor_violation(result)
        self._note_wake_refused(action, result)
        if not refused:
            self._stamp_changes(action, model, result)
        if self._prof is not None:
            self._prof.record(
                {
                    "kind": "dispatch",
                    "ts_ms": self._prof.now_ms(),
                    "n_actions": 1,
                    "http_ns": time.perf_counter_ns() - started_ns,
                }
            )
        return result

    def _stamp_changes(self, action, model: str, result: DispatchResult) -> None:
        """O1 / view-pending stamps, after the SM answered: a failed or partial call may
        still have changed the routable set, so it is stamped too - except a call that
        certainly changed nothing (2026-10-02): a model-level shrink with ``taken: 0``,
        a relay side without any pair, a refused / unsupported relay."""
        now = int(self._now_ms())

        def stamp(target: str, direction: int, hold: str) -> None:
            self._routable_change[target] = (now, direction)
            self._view_change[target] = (now, hold)

        if isinstance(action, TransferIntent):
            donor_changed, receiver_changed = _transfer_changed(result)
            if donor_changed:
                stamp(action.donor_model, -1, "down")
            if receiver_changed:
                stamp(action.receiver_model, 1, "up")
            return
        if not isinstance(action, (ScaleAction, ReceiverTarget, HideAction, UnhideAction)):
            return
        if result.not_executed:
            return  # refused before any change (writer_busy, routable_unknown, ...)
        if isinstance(action, ScaleAction) and action.delta < 0 and result.ok and result.taken == 0:
            return  # the replica floor clamped the shrink to nothing: no change
        stamp(getattr(action, "model", model), _routable_direction(action), _view_hold_direction(action))

    def _still_wanted(self, action: QueueAction) -> str | None:
        if self._revalidate is None:
            return None
        try:
            return self._revalidate(action)
        except Exception as exc:  # noqa: BLE001 - keep the action on a lookup error
            LOG.warning("revalidating %s failed (kept): %r", _action_kind(action), exc)
            return None

    def _release_idle_models(self) -> None:
        busy = {model for queued in self._running.values() for model in queued.models}
        for model in list(self._inflight):
            if model not in busy and not self._has_pending_model(model):
                self._inflight.discard(model)

    async def _dispatch(self, action, model: str) -> DispatchResult:
        if isinstance(action, ReceiverTarget):
            # Absolute, grow-only target: re-sending it is a no-op (review 3 P2-1).
            response = await self._client.scale_model_to(action.model, int(action.target))
            return _dispatch_result(model=action.model, action_kind="scale", response=response)
        if isinstance(action, TransferIntent):
            transfer = getattr(self._client, "transfer", None)
            if not callable(transfer):
                response = {"ok": False, "error": "service-manager client has no transfer()",
                            "status": 404, "retriable": False, "unsupported": True}
            else:
                response = await transfer(
                    action.donor_model, action.receiver_model, int(action.count), sleep_path=action.sleep_path
                )
            return self._transfer_result(action, response)
        if isinstance(action, ScaleAction):
            if action.delta > 0 and action.pods and getattr(action, "hint", False):
                hinted = getattr(self._client, "scale_model_hinted", None)
                if callable(hinted):
                    # The SM may place the wake elsewhere (it serializes GPU use itself).
                    return self._hinted_result(
                        action, await hinted(action.model, action.delta, hints=tuple(action.pods))
                    )
            if action.delta != 0 and action.pods:
                return await self._dispatch_binding_power(action)
            response = await self._client.scale_model(
                action.model, action.delta, **_sleep_kwargs(action)
            )
            return self._scale_result(action, response)
        if isinstance(action, HideAction):
            if self._is_observe_fresh():
                # B8 gap: a SafeScale probe start's hide queued just before the
                # switch to observe is never sent (re-checked right before the SM
                # call); the SafeScale loop rolls the probe back.
                self._stats["observe_hide_skipped_total"] += 1
                LOG.warning(
                    json.dumps(
                        {"event": "observe_hide_skipped_at_dispatch", "model": action.model,
                         "pods": list(action.pods), "reason": action.reason},
                        sort_keys=True,
                    )
                )
                # P3: ok=False - the pods were NOT hidden; the queue's failure
                # path marks the probe for rollback (on_hide_failed).
                return DispatchResult(model=action.model, action_kind="hide", ok=False, error="observe_skipped")
            response = await self._client.set_routable(action.model, action.pods)
            return _dispatch_result(model=action.model, action_kind="hide", response=response)
        if isinstance(action, UnhideAction):
            # keep_hidden: unconfirmed donor pods stay hidden (review 4 P2-2).
            response = await self._client.set_routable(action.model, tuple(action.keep_hidden))
            return _dispatch_result(model=action.model, action_kind="unhide", response=response)
        if isinstance(action, DefragAction):
            response = await self._client.defrag(tuple(action.migrations))
            return _dispatch_result(model=CLUSTER_MODEL, action_kind="defrag", response=response)
        return DispatchResult(model=model, action_kind="unknown", ok=False, error="unsupported_action")

    def _scale_result(self, action: ScaleAction, response: dict) -> DispatchResult:
        """A model-level ``/target`` call. A shrink reports ``taken`` (replicas really
        put to sleep) and ``clamped_by_floor`` (2026-10-02: the SM clamps every
        model-level shrink at the replica floor and answers 200): accounted by
        ``taken`` - 0 changed nothing; a clamp is an event (``scale_clamped_by_floor``),
        never a hold."""
        result = _dispatch_result(model=action.model, action_kind="scale", response=response)
        if action.delta >= 0 or not result.ok:
            return result
        body = response.get("response") or {}
        if not isinstance(body, dict) or "taken" not in body:
            return result  # no shrink planned by the SM (already at / below target)
        try:
            taken = max(0, int(body.get("taken") or 0))
        except (TypeError, ValueError):
            return result
        asked = -int(action.delta)
        if taken < asked:
            if body.get("clamped_by_floor"):
                self._stats["scale_clamped_by_floor_total"] += 1
                self._events.append(f"scale_clamped_by_floor:{action.model}:taken={taken}:asked={asked}")
            else:
                self._events.append(f"scale_down_short:{action.model}:taken={taken}:asked={asked}")
            LOG.info(
                json.dumps(
                    {"event": "scale_down_taken", "model": action.model, "asked": asked, "taken": taken,
                     "clamped_by_floor": bool(body.get("clamped_by_floor")), "floor": body.get("floor")},
                    sort_keys=True, default=str,
                )
            )
        return replace(result, taken=taken)

    def _transfer_result(self, action: TransferIntent, response: dict) -> DispatchResult:
        """``POST /v2/transfers`` -> the receiver-side result with the transfer summary.
        ``ok`` = at least one pair done; ``done`` / ``taken`` say what really happened
        (a 200 may be partial; 409 ``partial`` = nothing completed). Refusals, a floor
        clamp and ``taken: 0`` are observation events only."""
        donor, receiver = action.donor_model, action.receiver_model
        relay = f"{donor}->{receiver}"
        self._stats["transfer_total"] += 1
        if response.get("unsupported"):
            # Deployed against a service-manager without the endpoint: not executed.
            self._stats["transfer_unsupported_total"] += 1
            self._events.append(f"transfer_unsupported:{relay}")
            LOG.error(
                json.dumps(
                    {"event": "transfer_endpoint_missing", "donor": donor, "receiver": receiver,
                     "count": action.count, "detail": "POST /v2/transfers returned 404: the controller "
                     "must be deployed together with a service-manager that implements it",
                     "error": response.get("error")},
                    sort_keys=True,
                )
            )
            return DispatchResult(
                model=receiver, action_kind="transfer", ok=False,
                error="transfer_unsupported: the service-manager has no POST /v2/transfers",
                taken=0, done=0, transfer={"outcome": "unsupported"},
            )
        body = response.get("response") if (response.get("ok") or response.get("partial")) else None
        if not isinstance(body, dict):
            status = response.get("status")
            # Refused before any change (400, 409 routable_unknown / writer_busy, ...):
            # nothing happened (writer_busy: the SM writer lock wait timed out). Anything else (timeout, transport,
            # 5xx, another 409): whether anything happened is unknown.
            outcome = "refused" if response.get("not_executed") else "unknown"
            self._stats["transfer_failed_total"] += 1
            error = str(response.get("error") or "transfer_failed")
            self._events.append(f"transfer_failed:{relay}:{outcome}:{error[:160]}")
            return DispatchResult(
                model=receiver, action_kind="transfer", ok=False, error=error, retriable=False,
                taken=0 if outcome == "refused" else None, done=0 if outcome == "refused" else None,
                transfer={"outcome": outcome, "status": status}, not_executed=outcome == "refused",
            )
        summary = _transfer_summary(body)
        done, taken, unfilled = summary["done"], summary["taken"], summary["unfilled"]
        self._stats["transfer_pairs_done_total"] += done
        if done < action.pairs:
            self._stats["transfer_partial_total"] += 1
        self._events.append(
            f"transfer_done:{relay}:count={action.count}:done={done}:taken={taken}:unfilled={unfilled}"
            f":id={summary['transfer_id']}"
        )
        for pair in summary["pairs"]:
            self._events.append(
                f"transfer_pair:{relay}:{'+'.join(pair['donors'])}->{pair['receiver']}"
                f"@{pair['node']}/{','.join(str(gpu) for gpu in pair['gpu_ids'])}:{pair['status']}"
            )
        for refusal in body.get("refusals") or ():
            if isinstance(refusal, dict):
                self._wake_refused_event(receiver, _refusal_conflict(refusal))
        if summary["clamped_by_floor"]:
            self._stats["transfer_clamped_by_floor_total"] += 1
            self._events.append(f"transfer_clamped_by_floor:{relay}:taken={taken}:count={action.count}")
        if unfilled > 0:
            skipped = ",".join(f"{key}={value}" for key, value in sorted(summary["skipped"].items()))
            self._events.append(f"transfer_unfilled:{relay}:{unfilled}:{skipped}")
        LOG.info(
            json.dumps(
                {"event": "transfer_result", "donor": donor, "receiver": receiver, "count": action.count,
                 "reason": action.reason, **summary},
                sort_keys=True, default=str,
            )
        )
        error = None
        if done <= 0:
            error = (
                "clamped_by_floor: no donor replica taken" if summary["clamped_by_floor"] and taken == 0
                else str(response.get("error") or "partial: no pair completed")
            )
        elif done < action.pairs:
            error = f"partial: {done} of {action.pairs} pairs done"
        return DispatchResult(
            model=receiver, action_kind="transfer", ok=done > 0, error=error, retriable=False,
            picked=tuple(body.get("picked") or ()), taken=taken, done=done, transfer=summary,
        )

    def _hinted_result(self, action: ScaleAction, response: dict) -> DispatchResult:
        """A hinted wake's result (S5): ``placement_substituted`` for every replica the
        SM woke somewhere else than its hint (paired by the hint's binding id, never by
        position); the SM's refusals are ``wake_refused`` events (no cooldown since
        2026-10-02); an unfilled growth is not a success - the planner re-plans
        (review P2-7)."""
        result = _dispatch_result(model=action.model, action_kind="scale", response=response)
        body = (response.get("response") or {}) if response.get("ok") else {}
        picked = tuple(body.get("picked") or ())
        for entry in picked:
            if not isinstance(entry, dict) or entry.get("hinted"):
                continue
            source = _binding_gpus(entry.get("hint_binding_id")) or "?"
            target = f"{entry.get('node')}/{','.join(str(g) for g in entry.get('gpu_ids') or ())}"
            self._events.append(f"placement_substituted:{action.model}:{source}->{target}")
        for refusal in body.get("refusals") or ():
            if isinstance(refusal, dict):
                self._wake_refused_event(action.model, _refusal_conflict(refusal))
        unfilled = int(body.get("unfilled") or 0)
        if unfilled > 0:
            return replace(
                result, ok=False, error=f"partial: {unfilled} of {action.delta} wakes unfilled",
                retriable=False, picked=picked,
            )
        return replace(result, picked=picked)

    async def _dispatch_binding_power(self, action: ScaleAction) -> DispatchResult:
        # Sleep (delta < 0) or wake (delta > 0) exactly the named bindings: the
        # SafeScale commit of the hidden probe pods.
        # Stops at the first failure (a retry re-sends every pod: the SM answers a
        # binding already in the wanted power state with a no-op).
        for pod in action.pods:
            response = await self._client.set_binding_power(
                pod, awake=action.delta > 0, **_sleep_kwargs(action)
            )
            if not bool(response.get("ok", False)):
                return _dispatch_result(model=action.model, action_kind="scale", response=response)
        return DispatchResult(model=action.model, action_kind="scale", ok=True)

    def _record_done(self, model: str, action, result: DispatchResult) -> None:
        direction = _action_direction(action)
        if not result.ok or direction is None:
            return
        if direction == "down" and result.taken == 0:
            return  # clamped to nothing (2026-10-02): no scale-down happened
        self._last_done[model] = (int(self._now_ms()), direction)
        self._persist_scale_memory(model)

    def _record_transfer_done(self, action: TransferIntent, result: DispatchResult) -> None:
        """F4 / scale memory of a relay, by what really happened: the donor "down" when
        it gave a replica (``taken``), the receiver "up" when one woke (``done``)."""
        now = int(self._now_ms())
        if (result.taken or 0) > 0:
            self._last_done[action.donor_model] = (now, "down")
            self._persist_scale_memory(action.donor_model)
        if (result.done or 0) > 0:
            self._last_done[action.receiver_model] = (now, "up")
            self._persist_scale_memory(action.receiver_model)

    def _has_pending_model(self, model: str) -> bool:
        return any(model in item.models for item in self._pending)

    def _remove_pending_fairness_for_model(self, model: str) -> tuple[tuple[str, SourceLoop], ...]:
        removed: list[tuple[str, SourceLoop]] = []
        retained: deque[QueuedAction] = deque()
        for item in self._pending:
            if model in item.models and item.source_loop == "fairness":
                removed.append((model, item.source_loop))
                continue
            retained.append(item)
        self._pending = retained
        return tuple(removed)

    def _queued(self, action: QueueAction) -> QueuedAction:
        if isinstance(action, DefragAction):
            return QueuedAction(
                action=action,
                model=CLUSTER_MODEL,
                source_loop=action.source_loop,
                resources=frozenset({CLUSTER_RESOURCE}),
            )
        if isinstance(action, SafeScaleCommitAction):
            return self._commit_queued(action)
        if isinstance(action, TransferIntent):
            # Both models; no pod / GPU - the SM picks them under its writer lock.
            models = tuple(dict.fromkeys((action.donor_model, action.receiver_model)))
            pods = ()
        else:
            models = (action.model,)
            pods = tuple(getattr(action, "pods", ()) or ())
        return QueuedAction(
            action=action,
            model=action.model,
            source_loop=action.source_loop,
            models=models,
            resources=frozenset(self._resources(models, pods)),
        )

    def _commit_queued(self, action: SafeScaleCommitAction, *, failures: int = 0) -> QueuedAction:
        """Queued form of a commit: it holds the donor (and its pods) only until the
        donor slept, then just the receivers still pending."""
        models = action.touched_models or (action.donor,)
        pods = () if action.donor_done else action.pods
        return QueuedAction(
            action=action,
            model=action.donor,
            source_loop=action.source_loop,
            models=models,
            resources=frozenset(self._resources(models, pods)),
            failures=failures,
        )

    def _resources(self, models: Iterable[str], pods: Iterable[str]) -> set[str]:
        """Models and pods. GPU keys were removed on 2026-10-02: the service-manager
        serializes GPU use (sleep reservations, GPU leases, writer lock)."""
        keys = {f"model:{model}" for model in models}
        keys.update(f"pod:{pod}" for pod in pods)
        return keys


def _conflicts(resources: frozenset[str] | set[str], busy: set[str]) -> bool:
    """Two resource sets conflict when they share a key; a cluster-wide action
    (defrag) conflicts with every other action."""
    if not resources or not busy:
        return False
    if CLUSTER_RESOURCE in resources or CLUSTER_RESOURCE in busy:
        return True
    return bool(resources & busy)


def _runs_in_observe(queued: QueuedAction) -> bool:
    """Observe mode runs only SafeScale one-shot actions, in their observe-safe
    form: an unhide (the controller undoing its own hide) or a commit (turned
    into the unhide of its donor pods by _observe_commit). Everything else is
    dropped."""
    return queued.source_loop in ONE_SHOT_LOOPS and isinstance(
        queued.action, (UnhideAction, SafeScaleCommitAction)
    )


def _rescue_part(action):
    """The C1-tagged rescue scale-up of an action (a ScaleAction or a relay intent's
    receiver side), or None."""
    if isinstance(action, TransferIntent):
        return action if action.rescue is not None else None
    if isinstance(action, ScaleAction) and action.delta > 0 and getattr(action, "rescue", None) is not None:
        return action
    return None


def _transfer_summary(body: dict) -> dict:
    """The accounting view of a ``POST /v2/transfers`` answer (200, or the body of a
    409 ``partial``): what was done, with the pod names of every pair."""

    def num(key: str) -> int:
        try:
            return max(0, int(body.get(key) or 0))
        except (TypeError, ValueError):
            return 0

    pairs = []
    for item in body.get("pairs") or ():
        if not isinstance(item, dict):
            continue
        donors = item.get("donors") or ([item["donor"]] if item.get("donor") else [])
        pair = {
            "donors": [str(pod) for pod in donors],
            "receiver": str(item.get("receiver") or ""),
            "node": str(item.get("node") or ""),
            "gpu_ids": list(item.get("gpu_ids") or ()),
            "status": str(item.get("status") or "unknown"),
        }
        if item.get("compensating_sleep"):
            pair["compensating_sleep"] = item.get("compensating_sleep")
        error = item.get("error")
        if isinstance(error, dict) and error.get("error"):
            pair["error"] = str(error.get("error"))
        pairs.append(pair)
    done = num("done") if "done" in body else sum(1 for pair in pairs if pair["status"] == "done")
    taken = num("taken") if "taken" in body else num("donors_slept")
    skipped = body.get("skipped") if isinstance(body.get("skipped"), dict) else {}
    return {
        "transfer_id": body.get("transfer_id"),
        "outcome": "ok" if done > 0 else "partial",
        "done": done,
        "taken": taken,
        "unfilled": num("unfilled"),
        "clamped_by_floor": bool(body.get("clamped_by_floor")),
        "pairs": pairs,
        "refusals": len(body.get("refusals") or ()),
        "skipped": {str(key): value for key, value in skipped.items()},
        "picked": [
            entry.get("serve_id") for entry in body.get("picked") or () if isinstance(entry, dict)
        ],
        "phases_ms": body.get("phases_ms") if isinstance(body.get("phases_ms"), dict) else {},
    }


def _transfer_changed(result: DispatchResult) -> tuple[bool, bool]:
    """(donor side changed, receiver side changed) of a relay result - for the O1 /
    view-pending stamps. Unknown outcome (timeout, transport, 5xx, exception): both."""
    summary = result.transfer
    if not summary or summary.get("outcome") == "unknown":
        return True, True
    if summary.get("outcome") in ("unsupported", "refused"):
        return False, False
    pairs = summary.get("pairs") or ()
    donor = bool(summary.get("taken")) or bool(pairs)
    receiver = bool(summary.get("done")) or any(pair.get("status") == "receiver_wake_failed" for pair in pairs)
    return donor, receiver


def _refusal_conflict(refusal: dict) -> dict:
    """A structured refusal body (``refusals[]`` of /target or /v2/transfers) in the
    shape of ``ServiceManagerError.wake_conflict``."""
    return {
        "node": refusal.get("node"), "gpu_ids": refusal.get("gpu_ids") or refusal.get("gpu") or (),
        "scope": refusal.get("scope"), "error": refusal.get("error"), "reason": refusal.get("reason"),
        "blocking_binding_id": refusal.get("blocking_binding_id"),
    }


def _retry_safe(action) -> bool:
    """Only idempotent requests are retried (review 3 P2-1): named bindings,
    hide / unhide, absolute targets. A pod-less relative scale (current + delta
    at dispatch) could be applied twice after a timed-out success: never retried."""
    if isinstance(action, ScaleAction):
        return bool(action.pods)
    return True


def _observe_skipped(queued: QueuedAction) -> list[DispatchResult]:
    action = queued.action
    if isinstance(action, SafeScaleCommitAction):
        return [
            DispatchResult(model=model, action_kind="scale", ok=True, error="observe_skipped")
            for model in queued.models
        ]
    if isinstance(action, TransferIntent):
        return [
            DispatchResult(model=model, action_kind="transfer", ok=True, error="observe_skipped")
            for model in (action.donor_model, action.receiver_model)
        ]
    return [
        DispatchResult(
            model=queued.model,
            action_kind=_action_kind(action),
            # A hide not sent did not take effect (P3): never reported as done.
            ok=not isinstance(action, HideAction),
            error="observe_skipped",
        )
    ]


class SleepPathRefused(ValueError):
    """A scale-down that names no sleep path: refused before any SM call."""


def _sleep_kwargs(action: ScaleAction) -> dict:
    """SM sleep path + drain budget of a scale-down (plan 2026-09-27 D1).

    Only non-default values are sent. A negative delta must name its path (explicitly
    or via an "*_immediate" reason): there is no implicit "scale_down" any more.
    "*_immediate" planner reasons are the fast-loop donor paths ("urgent").
    """
    if action.delta >= 0:
        return {}
    path = action.sleep_path or ("urgent" if str(action.reason).endswith("_immediate") else None)
    if path is None:
        # Every legitimate shrink names its path ("urgent" donors, "safescale_commit").
        # A path-less one would silently take the SM default `scale_down` (SM-side
        # drain, up to 150 s), bypassing SafeScale: refuse. ``_timed_dispatch`` turns
        # it into one failed, non-retriable result (single error log, own counter).
        raise SleepPathRefused(
            f"scale-down of {action.model} ({action.reason}) has no sleep_path: "
            "refusing the implicit SM scale_down default"
        )
    kwargs: dict = {}
    if path != "scale_down":
        kwargs["sleep_path"] = path
    if action.drain_budget_s is not None:
        kwargs["drain_budget_s"] = float(action.drain_budget_s)
    return kwargs


def _action_kind(action) -> str:
    if isinstance(action, SafeScaleCommitAction):
        return "safescale_commit"
    if isinstance(action, TransferIntent):
        return "transfer"
    if isinstance(action, (ScaleAction, ReceiverTarget)):
        return "scale"
    if isinstance(action, HideAction):
        return "hide"
    if isinstance(action, UnhideAction):
        return "unhide"
    if isinstance(action, DefragAction):
        return "defrag"
    return "unknown"


def _routable_direction(action) -> int:
    """+1 / -1: the direction an SM call can move a model's routable count."""
    if isinstance(action, (ReceiverTarget, UnhideAction)):
        return 1
    if isinstance(action, HideAction):
        return -1
    if isinstance(action, ScaleAction):
        return 1 if action.delta > 0 else -1
    return 0


def _view_hold_direction(action) -> str:
    """O1 view-pending hold rule of a routable-changing SM call: a wake / scale-up holds
    like "up" (no scale-down, no scale-up); a sleep, hide or unhide like "down" (a
    CRITICAL receiver still scales up, nothing else)."""
    if isinstance(action, ReceiverTarget):
        return "up"
    if isinstance(action, ScaleAction) and action.delta > 0:
        return "up"
    return "down"


def _action_direction(action) -> str | None:
    if isinstance(action, ReceiverTarget):
        return "up"
    if isinstance(action, ScaleAction):
        return "up" if action.delta > 0 else "down" if action.delta < 0 else None
    if isinstance(action, HideAction):
        return "down"
    # UnhideAction (safescale rollback) restores capacity; it is not a scaling decision
    # and must not hold a CRITICAL model in cooldown (review P2-4).
    return None


def _queued_action(action: QueueAction) -> QueuedAction:
    """Queued form of one action without GPU resources (kept for callers/tests)."""
    if isinstance(action, DefragAction):
        return QueuedAction(action=action, model=CLUSTER_MODEL, source_loop=action.source_loop)
    return QueuedAction(action=action, model=action.model, source_loop=action.source_loop)


#: SM sleep outcome statuses of a pod that may be asleep (review 4 P2-2).
UNCONFIRMED_OUTCOMES = frozenset({"unconfirmed", "rollback_failed"})


def _dispatch_result(*, model: str, action_kind: str, response: dict) -> DispatchResult:
    ok = bool(response.get("ok", False))
    error = None if ok else str(response.get("error") or "dispatch_failed")
    retriable = (not ok) and bool(response.get("retriable", False))
    unconfirmed = tuple(
        str(item.get("serve_id"))
        for item in (response.get("outcomes") or ())
        if isinstance(item, dict) and item.get("status") in UNCONFIRMED_OUTCOMES and item.get("serve_id")
    )
    conflict = response.get("wake_conflict") if not ok else None
    return DispatchResult(
        model=model, action_kind=action_kind, ok=ok, error=error, retriable=retriable, unconfirmed=unconfirmed,
        not_executed=(not ok) and bool(response.get("not_executed")),
        # sm_client ServiceManagerError.result() sets the key only for a floor refusal.
        floor_violation=(not ok) and "floor_violation" in response,
        wake_conflict=conflict if isinstance(conflict, dict) else None,
    )


def _binding_gpus(binding_id) -> str | None:
    """``node/gpus`` of a binding id ``model/node/gpus``."""
    parts = str(binding_id or "").split("/")
    return f"{parts[1]}/{parts[2]}" if len(parts) == 3 else None


def _actor_for(action) -> str:
    """``X-TRE-Actor`` of an SM call: controller/<loop>/<reason>."""
    loop = getattr(action, "source_loop", None) or "unknown"
    reason = getattr(action, "reason", None) or type(action).__name__
    return f"controller/{loop}/{reason}"


def _request_id(action) -> str | None:
    """The SafeScale probe a one-shot action belongs to (commit or unhide)."""
    if isinstance(action, (SafeScaleCommitAction, UnhideAction)):
        return getattr(action, "request_id", None)
    return None


def _resolution(action, failure: DispatchResult | None = None, *, reason: str | None = None) -> tuple[str, str]:
    """How a finished one-shot action resolves its SafeScale probe."""
    if isinstance(action, SafeScaleCommitAction):
        if failure is None:
            return ("commit", action.reason)
        if action.donor_done:
            return ("commit", f"upscale_failed: {failure.error}")
        return ("rollback", f"commit_failed: {failure.error}; donor pods left hidden")
    if isinstance(action, UnhideAction):
        if reason is not None:
            return ("rollback", f"{action.reason}; {reason}")
        if failure is not None:
            return ("rollback", f"{action.reason}; unhide_failed: {failure.error}")
        return ("rollback", action.reason)
    return ("done", reason or (failure.error if failure is not None else "ok") or "")


def revalidate_from_cluster_view(get_view: Callable[[], object]) -> Revalidate:
    """Retry guard for one-shot actions: still wanted while the latest cluster view
    shows at least one named pod not yet in the target state (no view = keep)."""

    def still_wanted(action: QueueAction) -> str | None:
        view = get_view()
        if view is None:
            return None
        bindings = {binding.serve_id: binding for binding in getattr(view, "bindings", ())}
        if isinstance(action, ScaleAction) and action.pods and action.delta != 0:
            wanted_awake = action.delta > 0
            if any(
                pod not in bindings or bindings[pod].awake != wanted_awake for pod in action.pods
            ):
                return None
            return f"pods {list(action.pods)} already {'awake' if wanted_awake else 'asleep'}"
        if isinstance(action, UnhideAction) and action.pods:
            keep = set(action.keep_hidden)
            pods = [pod for pod in action.pods if pod not in keep]
            if not pods or any(pod not in bindings or bindings[pod].hidden for pod in pods):
                return None
            return f"pods {pods} already routable"
        if isinstance(action, HideAction) and action.pods:
            if any(pod not in bindings or not bindings[pod].hidden for pod in action.pods):
                return None
            return f"pods {list(action.pods)} already hidden"
        return None

    return still_wanted


def revalidate_commit_from_signals(
    get_states: Callable[[], "Mapping[str, str] | None"],
    get_view: Callable[[], object] | None = None,
) -> CommitRevalidate:
    """Retry guard of a SafeScale commit (review 3 P2-1..P2-3), evaluated before
    every (re)try on the models' CURRENT signal state (``get_states``: model ->
    ModelState value of the latest planner tick; missing / stale = unknown):

    * donor pods already asleep in the cluster view (a timed-out first call
      that succeeded) -> the donor part is done;
    * else the donor needing capacity (critical / low) -> abandon: unhide;
    * a receiver that clearly no longer needs capacity (healthy / high / idle)
      -> its upscale is dropped. Unknown states keep the plan.

    ``get_view`` must return the cluster view only while FRESH (review 4 P2-1).

    Two different criteria decide a commit (review 4 P3): the SafeScale commit
    GATE judges the probe window's tail of the donor's SERVING pods (latency
    SLO, Z_m >= tau_low, KV cache, donor health) with the probe pods hidden;
    this REVALIDATION judges the planner's latest whole-model ModelState (its
    own window, dwell and warm-up rules). They can disagree right after the
    gate passed - the queue then counts ``commit_abandoned_after_gate_total``
    and logs ``safescale_commit_abandoned_after_gate``.
    """

    def check(action: SafeScaleCommitAction) -> CommitVerdict:
        states = dict(get_states() or {})
        view = get_view() if get_view is not None else None
        if not action.donor_done:
            if view is not None and action.pods:
                bindings = {binding.serve_id: binding for binding in getattr(view, "bindings", ())}
                if all(pod in bindings and not bindings[pod].awake for pod in action.pods):
                    action = replace(action, donor_done=True)
        if not action.donor_done:
            donor_state = states.get(action.donor)
            if donor_state in NEEDS_CAPACITY_STATES:
                return CommitVerdict(
                    action, abandon_reason=f"donor {action.donor} is {donor_state} (needs capacity)"
                )
        kept: list[ReceiverTarget] = []
        dropped: list[tuple[str, str]] = []
        for upscale in action.upscales:
            state = states.get(upscale.model)
            if state in NO_NEED_STATES:
                dropped.append((upscale.model, f"receiver {upscale.model} is {state} (no longer needs capacity)"))
                continue
            kept.append(upscale)
        return CommitVerdict(replace(action, upscales=tuple(kept)), dropped=tuple(dropped))

    return check
