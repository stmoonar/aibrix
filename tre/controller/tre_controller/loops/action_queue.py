from __future__ import annotations

import asyncio
import json
import logging
import time
from collections import deque
from dataclasses import dataclass, field, replace
from typing import Awaitable, Callable, Iterable, Mapping, Protocol

from tre_controller.loops import plan_items
from tre_controller.loops.plan_items import SleepPathRefused
from tre_controller.sm_client import sm_actor
from tre_controller.planning.planner import (
    Action,
    BlockSupplyAction,
    DefragAction,
    HideAction,
    ReceiverTarget,
    IMMEDIATE_DONOR_SLEEP_PATH,
    named_transfer_hold_reason,
    RelayHold,
    SafeScaleCommitAction,
    ScaleAction,
    SourceLoop,
    UnhideAction,
    upscale_of,
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
    async def get_state_result(self) -> dict: ...

    async def post_plan(self, request: dict) -> dict: ...

    async def set_routable(self, model: str, hidden_pods: tuple[str, ...]) -> dict: ...

    async def defrag(self, migrations: tuple) -> dict: ...


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
    #: 2026-10-02: the SM refused the call before changing anything (``routable_unknown``,
    #: ``writer_busy``, ``floor_violation``, a located wake refusal, 400 / 404 / 503):
    #: accounted as not executed - no view-pending stamp, no last action.
    not_executed: bool = False
    #: Review P3 (2026-10-06): pods a multi-pod binding-power call changed before a
    #: later pod failed. Such a failed result still changed the fleet: it is stamped
    #: (view-pending), recorded as the model's last action and counted for its
    #: rescue target.
    changed: tuple[str, ...] = ()
    #: Review P2-2 (2026-10-06): a growth the SM filled in part (409 ``partial`` /
    #: 200 ``unfilled``) that woke nothing (no ``wake`` action, nothing ``picked``):
    #: the routable set did not change - no view-pending stamp.
    woke_none: bool = False


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
    #: Process-unique id of a target started by a submit (> 0); 0 = restored from the
    #: durable memory or recorded as covered by a probe preemption (no dispatch).
    seq: int = 0

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
    (review P1-3 / review 2 P1-1). Every power change is ONE ``POST /v2/plans``
    (design 20261027-sm-plans): sleep and wake items executed by the SM under ONE
    writer-lock hold, each action exactly one item. A donor -> receiver transfer is one
    plan group of named items (the donor pods' sleeps, the receiver binding's wake -
    sm-plans section 10), so when the call returns it has completed or failed. A SafeScale commit is one plan per attempt: the sleep of its
    hidden donor pods plus its receivers' absolute targets (review 3 P2-2). One-shot actions (SafeScale commit /
    rollback) that fail retriably are retried with bounded backoff and
    re-validated before every retry; a retry only ever re-sends idempotent
    requests (named bindings, absolute targets - review 3 P2-1). A rescue action
    for a model whose one-shot commit waits out a backoff preempts that retry
    (review 3 P2-3). Re-plannable actions are dropped on failure.

    Observe mode (user decision 2026-09-28, record only): re-plannable actions
    are dropped; a SafeScale one-shot action runs only as the unhide of its
    probe pods; the mode is re-read (uncached) right before every
    capacity-changing step (a plan's POST, a commit attempt, a hide). Once a plan
    is posted, the SM completes it (also a commit's receivers).

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
        on_fleet_change: Callable[[], None] | None = None,
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
        #: a plan's POST, a commit attempt).
        self._is_observe_fresh = is_observe_fresh or self._is_observe
        self._prof = prof
        # Review F4: model -> (epoch ms the last successful dispatch completed, "up"/"down").
        self._now_ms = now_ms or (lambda: int(time.time() * 1000))
        self._last_done: dict[str, tuple[int, str]] = {}
        #: Timer cleanup review P2-1: model -> (when the last SM call that can change its
        #: routable count returned, ok or not, its direction) for the view-pending
        #: guard: the model takes no new action until a fleet view newer than it.
        self._view_change: dict[str, tuple[int, str]] = {}
        #: C1: model -> its last rescue target (see :meth:`rescue_targets`).
        self._rescue: dict[str, RescueTargetRecord] = {}
        #: Review P2-1 (2026-10-06): (donor, receiver) -> the hold of a relay the SM
        #: answered with nothing done (see :meth:`relay_holds`).
        self._relay_holds: dict[tuple[str, str], RelayHold] = {}
        #: C1 review P3: id(queued action) -> the record that part belongs to (a part of
        #: a superseded target updates that record, never its successor).
        self._rescue_parts: dict[int, RescueTargetRecord] = {}
        #: Last ``RescueTargetRecord.seq`` handed out.
        self._rescue_seq = 0
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
        # F2 (2026-10-07): called after an SM call of ours changed the fleet (every
        # view-pending stamp below) - the app asks for a cluster-view refresh.
        self._on_fleet_change = on_fleet_change
        #: running dispatch task -> its queued action (updated as a commit progresses)
        self._running: dict[asyncio.Future, QueuedAction] = {}
        #: running dispatch task -> its one-shot action waiting out a retry backoff
        self._backoff: dict[asyncio.Future, _Backoff] = {}
        #: Set when work may be dispatchable (an accepted submit, a cancelled pending
        #: action): wakes :meth:`run` before its poll (created by ``run``, on its loop).
        self._wakeup: asyncio.Event | None = None
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
            # steps, commits turned into their donor unhide.
            "observe_hide_skipped_total": 0,
            "observe_commit_stopped_total": 0,
            "floor_violation_total": 0,
            # Floor clamps.
            "scale_clamped_by_floor_total": 0,
            # 2026-10-06: a SafeScale receiver target the SM filled in part.
            "receiver_short_total": 0,
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
        #: This submit's accepted items (fused into plans below).
        accepted_items: list[QueuedAction] = []
        dropped: list[tuple[str, str]] = []
        replaced: list[tuple[str, SourceLoop]] = []
        observe = self._is_observe()
        #: C1: models whose rescue target this submit (re)started.
        rescue_started: set[str] = set()
        #: C1 review P2-1: pods a commit preemption gave back, per receiver model, not
        #: yet deducted from one of its scale-up parts (one preemption, all parts).
        restore_credit: dict[str, int] = {}
        #: 2026-10-28 (sm-plans section 10): models in flight or queued BEFORE this
        #: submit, and what this submit's accepted items claim. Named items of one model
        #: (an IDLE donor giving two receivers, each its own pod) are the planner's one
        #: choice and stay together; inside the submit the SM's request rule applies (a
        #: pod named twice, a count item beside any other item of its model).
        prior_models = set(self._inflight) | {model for item in self._pending for model in item.models}
        submit_keys = _PlanKeys()

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
            elif any(model in prior_models for model in queued.models) or submit_keys.clashes(
                _PlanKeys.of(queued.action)
            ):
                dropped.append((queued.model, "inflight"))
                continue
            submit_keys.add(_PlanKeys.of(queued.action))

            self._pending.append(queued)
            accepted_items.append(queued)
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

        self._fuse_into_plans(accepted_items)
        if accepted:
            self._wake()
        return SubmitResult(
            accepted=accepted,
            held=held,
            dropped=tuple(dropped),
            replaced=tuple(replaced),
        )

    def _fuse_into_plans(self, accepted: list[QueuedAction]) -> None:
        """One decision -> one plan (design 20261027-sm-plans §5.2 and section 10): the
        accepted re-plannable power actions of ONE submit (``ScaleAction``, transfer
        groups) become ONE :class:`PlanBatch`, in submit
        order, each action exactly one plan item. The split rule is the SM's request
        rule (section 10): a pod named twice, or a model with a count item and any
        other item of the plan, starts the next batch - several named items of one
        model (an IDLE donor giving a TP1 relay and a TP2 block, each its own pod) stay
        in ONE plan. The batches serialize through their shared models. A batch takes
        the queue position of its first member; a member never jumps over another
        accepted action of one of its models. No timer: nothing waits for more actions.

        Only members whose resources are FREE at submit time are fused (review
        2026-10-28): a rescue member whose model a running commit / plan (or an
        earlier queued action) still holds would make the whole batch wait for
        it; it stays its own queued action, as before fusion.

        A TP block (a :class:`PlanBatch` of its own: donor sleeps + receiver
        wake, design 20261028-single-loop L4) joins as ONE unit - both members in the
        same plan, never split - so one decision is one plan whenever the members touch
        different models (the SM's plan runs the receiver's wake as wave 2 after the
        donor's sleep, ``tre/docs/design/20261027-sm-plans.md`` section 4.1)."""
        if sum(1 for queued in accepted if _fusable(queued)) < 2:
            return
        mine = {id(queued) for queued in accepted}
        start = len(self._pending)
        while start > 0 and id(self._pending[start - 1]) in mine:
            start -= 1
        tail = list(self._pending)[start:]
        if any(id(queued) not in mine for queued in tail):
            return  # not this submit's contiguous tail (never expected): leave it unfused
        held: set[str] = set()
        for queued in [*self._running.values(), *list(self._pending)[:start]]:
            held |= set(queued.resources)
        out: list = []
        batch: list[QueuedAction] | None = None
        used = _PlanKeys()
        #: Models of unfused actions after the batch began: a later member may not jump
        #: over one of them on a shared model.
        barrier: set[str] = set()
        for queued in tail:
            keys = _PlanKeys.of(queued.action)
            if not _fusable(queued) or _conflicts(queued.resources, held):
                out.append(queued)
                barrier |= set(queued.models)
                continue
            if batch is not None and not used.clashes(keys) and not (set(queued.models) & barrier):
                batch.append(queued)
                used.add(keys)
                continue
            batch = [queued]
            out.append(batch)
            used = _PlanKeys()
            used.add(keys)
            barrier = set()
        fused: deque[QueuedAction] = deque(list(self._pending)[:start])
        for entry in out:
            if isinstance(entry, list) and len(entry) > 1:
                fused.append(self._queued(PlanBatch(tuple(
                    member for item in entry for member in _members(item.action)
                ))))
            else:
                fused.append(entry[0] if isinstance(entry, list) else entry)
        self._pending = fused

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

    def view_changes(self) -> dict[str, tuple[int, str]]:
        """Timer cleanup review P2-1: model -> (when its last routable-changing SM call
        returned - also a failed or partial one, also an unhide -, hold direction). A
        relay records both sides (donor "down", receiver "up"); a call that changed
        nothing (``taken: 0``, a relay side with no pair) records nothing (2026-10-02)."""
        return dict(self._view_change)

    def relay_holds(self) -> dict[tuple[str, str], RelayHold]:
        """Review P2-1: (donor, receiver) -> :class:`RelayHold` of its last relay when
        the SM answered it with nothing done (refused, vetoed, floor-clamped, unfilled,
        no pair completed) - the planner does not send the same pair again while the
        fleet view it was planned from is current. A relay with a pair done, or with an
        unknown outcome (timeout, transport error), leaves no hold."""
        return dict(self._relay_holds)

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
            self._rescue_seq += 1
            record = RescueTargetRecord(
                target=int(plan.target), desired=int(plan.desired), base=int(plan.base),
                covered_before=int(plan.covered), issued_ms=int(self._now_ms()), seq=self._rescue_seq,
            )
            self._rescue[part.model] = record
            self._restored_unchecked.discard(part.model)
        return record

    def _register_rescue(self, action, started: set[str]) -> None:
        if isinstance(action, PlanBatch):
            # A TP block (L4 rev 2): its receiver wake is a part of the target.
            for member in action.members:
                self._register_rescue(member, started)
            return
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
        if result.ok:
            record.gained += int(part.delta)
        else:
            record.failures += 1
            record.gained += min(int(part.delta), max(len(result.picked or ()), len(result.changed)))

    def _finish_rescue(self, action) -> None:
        """C1: one rescue scale-up ended (done, failed, dropped): the target is done
        once all of its parts ended (a fused plan: each member)."""
        if isinstance(action, PlanBatch):
            for member in action.members:
                self._finish_rescue(member)
            return
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
        if len(kept) != len(self._pending):
            self._wake()  # an action behind the cancelled one may be free now
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
        resources. Cancelling ``run`` cancels every dispatch in progress.

        Dispatches on every accepted :meth:`submit` (2026-10-28: the submit wakes it,
        no 0-100 ms poll delay) and, as a fallback, every ``poll_interval_s`` (mode
        changes, anything else that frees a queued action)."""
        wakeup = self._wakeup = asyncio.Event()
        try:
            while True:
                wakeup.clear()
                self._dispatch_pending([])
                await _wait_either(wakeup, sleep(poll_interval_s))
        except asyncio.CancelledError:
            await self.shutdown()
            raise

    def _wake(self) -> None:
        if self._wakeup is not None:
            self._wakeup.set()

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
            if isinstance(current.action, PlanBatch):
                results.extend(
                    DispatchResult(model=_member_model(member), action_kind=_action_kind(member), ok=False,
                                   error=error)
                    for member in current.action.members
                )
            else:
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
        if _replannable_power(queued):
            # One POST /v2/plans for the action (or the fused actions of one submit);
            # re-plannable: never retried, the next tick re-plans from a new view.
            results.extend(await self._execute_plan(_members(queued.action)))
            return ("done", "plan")
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
        if isinstance(action, ScaleAction):
            # A one-shot (SafeScale) named scale: a one-item plan, retried as is
            # (named pods are idempotent).
            result = (await self._run_plan((action,))).results[0]
        else:
            result = await self._timed_dispatch(action, queued.model)
        self._note_rescue_result(action, result)
        if not result.ok:
            if result.changed:
                self._record_done(queued.model, action, result)
            return result, queued
        self._record_done(queued.model, action, result)
        results.append(replace(result, attempts=attempts))
        self._notify_hide_done(action)
        return None, queued

    async def _attempt_commit(
        self, queued: QueuedAction, results: list[DispatchResult]
    ) -> tuple[DispatchResult | None, QueuedAction]:
        """One attempt = ONE plan (design 20261027-sm-plans §5.4): the sleep of the
        donor's hidden pods (unless it already slept) plus one wake item per pending
        receiver with its absolute target (resolved once by the plan's GET, then
        frozen). The SM wakes a receiver on a GPU the donor frees only after the donor
        is confirmed asleep; a receiver on a free GPU may wake even when the donor
        sleep fails. Once posted, the SM completes the plan (an observe switch no
        longer stops the receivers). Progress is kept on the action: a retry never
        re-sleeps a donor that slept and re-sends only the pending receivers."""
        action: SafeScaleCommitAction = queued.action
        attempts = queued.failures + 1
        donor: ScaleAction | None = None
        members: list = []
        if not action.donor_done and action.pods:
            donor = action.donor_sleep()
            members.append(donor)
        members.extend(action.upscales)
        if not members:
            return None, queued
        run = await self._run_plan(tuple(members))
        failure: DispatchResult | None = None
        offset = 0
        if donor is not None:
            offset = 1
            result = run.results[0]
            if not result.ok:
                if result.changed:
                    self._record_done(action.donor, donor, result)
                if result.unconfirmed:
                    action = replace(
                        action,
                        unconfirmed_pods=tuple(dict.fromkeys(action.unconfirmed_pods + result.unconfirmed)),
                    )
                failure = result
            else:
                self._record_done(action.donor, donor, result)
                results.append(replace(result, attempts=attempts))
                action = replace(action, donor_done=True)
        remaining: list[ReceiverTarget] = []
        for upscale, result in zip(run.members[offset:], run.results[offset:]):
            if run.whole_plan_failed and not result.ok:
                # The plan failed as a whole (refused, or its outcome unknown): every
                # receiver stays pending with its target, as before the plan was sent.
                remaining.append(upscale)
                failure = failure or result
            elif result.ok:
                self._record_done(upscale.model, upscale, result)
                results.append(replace(result, attempts=attempts))
            elif result.retriable:
                # Kept with its frozen target: the retry re-sends the same target.
                remaining.append(upscale)
                failure = failure or result
            else:
                results.append(replace(result, attempts=attempts))
        action = replace(action, upscales=tuple(remaining))
        queued = self._commit_queued(action, failures=queued.failures)
        self._set_running(queued)
        return failure, queued

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
        batch_ups = receiver_upscales(action) if isinstance(action, PlanBatch) else ()
        if len(batch_ups) == 1:
            # Review P2-2: a TP block (donor sleeps + receiver wake).
            model, delta = batch_ups[0]
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
        if isinstance(action, PlanBatch):
            # A transfer (a TP block, a relay pair) is not split: covered entirely, or kept
            # whole (credit kept).
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

    async def _execute_plan(self, members: tuple) -> list[DispatchResult]:
        """Re-plannable power actions = ONE ``POST /v2/plans`` (design 20261027-sm-plans
        §5.2), each action exactly one item. Each member keeps its own result and
        bookkeeping (rescue credit, F4 / scale memory, relay hold, events)."""
        run = await self._run_plan(members, observe_check=True)
        if run.observe_skipped:
            return [result for member in members for result in _member_observe_skipped(member)]
        out: list[DispatchResult] = []
        for member, result in zip(members, run.results):
            self._note_rescue_result(member, result)
            if result.ok or result.changed:
                self._record_done(member.model, member, result)
            self._note_transfer_hold(member, result)
            out.append(result)
        return out

    def _note_transfer_hold(self, member, result: DispatchResult) -> None:
        """Review P2-1 for a named transfer (sm-plans section 10): the receiver wake of
        a transfer group refused with nothing done (``not_executed``: a group refusal,
        a whole-plan refusal) holds the (donor, receiver) pair while the fleet view it
        was planned from is current (:class:`RelayHold`); a wake that woke, or an
        outcome that changed something / is unknown, clears it.
        :func:`named_transfer_hold_reason` says why and whether only a fleet change
        releases it."""
        if not isinstance(member, ScaleAction) or member.group is None or member.delta <= 0 or member.donor is None:
            return
        pair = (member.donor, member.model)
        if result.ok or result.changed or not result.not_executed or member.basis is None:
            self._relay_holds.pop(pair, None)
            return
        reason, fleet = named_transfer_hold_reason(result.error, result.wake_conflict)
        self._relay_holds[pair] = RelayHold(basis=member.basis, reason=reason, fleet=fleet)

    async def _run_plan(self, members: tuple, *, observe_check: bool = False) -> "_PlanRun":
        """ONE plan for ``members`` (each exactly one item): ONE ``GET /v2/state`` resolves
        the relative / absolute targets (only when an item needs it), items whose target
        is already met are answered locally (``woke_none`` / ``taken: 0``, no SM call), and
        ONE ``POST /v2/plans`` carries the rest - none when nothing is left. Returns one
        result per member (stamps, floor and refusal events done here, as for every SM
        call) and the members with their targets resolved (a SafeScale receiver keeps
        it). ``observe_check``: the mode is re-read right before the POST."""
        started_ns = time.perf_counter_ns()
        members = tuple(members)
        count = len(members)
        results: list[DispatchResult | None] = [None] * count
        resolved_members = list(members)
        specs: list[tuple[str, dict] | None] = [None] * count
        for index, member in enumerate(members):
            try:
                spec = plan_items.plan_item(member)
            except SleepPathRefused as exc:
                results[index] = self._sleep_path_refused(member, exc)
                continue
            if spec is None:
                results[index] = DispatchResult(
                    model=_member_model(member), action_kind=_action_kind(member), ok=True, woke_none=True
                )
                continue
            specs[index] = spec
        counts: dict[str, tuple[int, int]] = {}
        if any(spec is not None and plan_items.needs_state(*spec) for spec in specs):
            state = await self._client.get_state_result()
            if not bool(state.get("ok", False)):
                # Nothing was sent: no member changed anything.
                for index, member in enumerate(members):
                    if results[index] is None:
                        failed = _dispatch_result(model=_member_model(member), action_kind=_action_kind(member),
                                                  response=state)
                        results[index] = replace(failed, error=f"state_unreadable: {failed.error}", not_executed=True)
                return self._plan_done(members, resolved_members, results, (), started_ns, whole_plan_failed=True)
            counts = plan_items.state_counts(
                state.get("response"),
                {spec[1]["model"] for spec in specs if spec is not None and plan_items.needs_state(*spec)},
            )
        request: dict = {"sleeps": [], "wakes": []}
        slots: dict[int, tuple[str, int]] = {}
        for index, spec in enumerate(specs):
            if spec is None or results[index] is not None:
                continue
            kind, item = spec
            member = members[index]
            try:
                body = plan_items.resolve_item(kind, item, counts)
            except plan_items.StateUnreadable as exc:
                results[index] = DispatchResult(
                    model=_member_model(member), action_kind=_action_kind(member), ok=False,
                    error=str(exc), not_executed=True,
                )
                continue
            if isinstance(member, ReceiverTarget) and member.target is None and body is not None:
                if "at_least" in body:  # a named receiver (its bindings) has no target
                    resolved_members[index] = replace(member, target=int(body["at_least"]))
            elif isinstance(member, ReceiverTarget) and member.target is None:
                awake = counts[member.model][0]
                resolved_members[index] = replace(member, target=awake)
            if body is None:
                results[index] = self._met_locally(member)
                continue
            slots[index] = (kind, len(request[kind + "s"]))
            request[kind + "s"].append(body)
        if not slots:
            return self._plan_done(members, resolved_members, results, (), started_ns)
        if observe_check and self._is_observe_fresh():
            return _PlanRun(members=tuple(resolved_members), results=(), observe_skipped=True)
        try:
            with sm_actor(_plan_actor(members)):
                response = await self._client.post_plan(request)
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 - review 3 P3
            self._stats["dispatch_exceptions_total"] += 1
            LOG.exception("POST /v2/plans for %s raised", sorted({_member_model(m) for m in members}))
            response = {"ok": False, "error": f"dispatch_exception: {type(exc).__name__}: {exc}"}
        body = response.get("response") if (response.get("ok") or response.get("partial")) else None
        if not isinstance(body, dict):
            body = None
        if body is not None:
            self._log_plan(request, body)
        elif not response.get("not_executed"):
            # Timeout, transport error, 5xx: whether anything happened is unknown.
            LOG.warning(json.dumps({"event": "plan_outcome_unknown", "error": response.get("error"),
                                    "status": response.get("status")}, sort_keys=True, default=str))
        for index, (kind, position) in slots.items():
            member = resolved_members[index]
            if body is None:
                results[index] = self._plan_failed_member(member, response)
                continue
            item = _response_item(body, kind, position)
            if item is None:
                results[index] = DispatchResult(
                    model=_member_model(member), action_kind=_action_kind(member), ok=False,
                    error="plan_item_missing: the service-manager answered without this item",
                )
                continue
            results[index] = self._item_result(member, item, body.get("plan_id"))
        return self._plan_done(members, resolved_members, results, tuple(slots), started_ns,
                               whole_plan_failed=body is None)

    def _plan_done(
        self, members, resolved_members, results, sent: tuple[int, ...], started_ns: int, *,
        whole_plan_failed: bool = False,
    ) -> "_PlanRun":
        for index in sent:
            member, result = resolved_members[index], results[index]
            self._note_floor_violation(result)
            self._stamp_changes(member, _member_model(member), result)
        if sent and self._prof is not None:
            self._prof.record(
                {
                    "kind": "dispatch",
                    "ts_ms": self._prof.now_ms(),
                    "n_actions": len(sent),
                    "http_ns": time.perf_counter_ns() - started_ns,
                }
            )
        return _PlanRun(members=tuple(resolved_members), results=tuple(results), whole_plan_failed=whole_plan_failed)

    def _sleep_path_refused(self, member, exc: Exception) -> DispatchResult:
        # Nothing reached the SM: own counter, one error line, no traceback, and no
        # view-pending stamp (no change happened).
        self._stats["sleep_path_refused_total"] += 1
        LOG.error(
            json.dumps(
                {"event": "scale_down_without_sleep_path_refused", "model": _member_model(member),
                 "reason": str(getattr(member, "reason", "")),
                 "delta": getattr(member, "delta", None)},
                sort_keys=True,
            )
        )
        return DispatchResult(
            model=_member_model(member), action_kind=_action_kind(member), ok=False,
            error=f"sleep_path_refused: {exc}", retriable=False,
        )

    def _met_locally(self, member) -> DispatchResult:
        """No-op fix (design §5.5): a target already met makes no SM call and changes
        nothing - ``woke_none`` (a growth) / ``taken: 0`` (a shrink): no stamp, no last
        action, no rescue gain beyond what the call would have reported. A relay never
        gets here: its ``at_least`` is ``awake + pairs`` with ``pairs >= 1`` and no cap."""
        model = _member_model(member)
        if _action_direction(member) == "down":
            return DispatchResult(model=model, action_kind="scale", ok=True, taken=0)
        return DispatchResult(model=model, action_kind="scale", ok=True, woke_none=True)

    def _plan_failed_member(self, member, response: dict) -> DispatchResult:
        """A whole-plan failure seen by one member: a refusal before any change
        (``writer_busy``, ``routable_unknown``, 400 / 404 / 503) is ``not_executed``; a
        timeout / transport error / 5xx leaves the outcome unknown (stamped both ways)."""
        return _dispatch_result(model=_member_model(member), action_kind=_action_kind(member), response=response)

    def _item_result(self, member, item: dict, plan_id) -> DispatchResult:
        if isinstance(member, ReceiverTarget):
            return self._receiver_item_result(member, item)
        if member.delta < 0:
            if member.pods:
                return self._named_sleep_result(member, item)
            return self._shrink_result(member, item)
        if member.pods:
            return self._named_wake_result(member, item)
        return self._growth_result(member, item)

    def _named_sleep_result(self, action: ScaleAction, item: dict) -> DispatchResult:
        """A sleep of exactly the named bindings (SafeScale commit donor, a TP block's donor pods).
        A failure after some pods slept is reported with them (``changed``), never as
        "nothing happened"; unconfirmed pods (``outcomes``) must stay hidden."""
        if item.get("status") in plan_items.SLEEP_OK_STATUSES:
            return DispatchResult(model=action.model, action_kind="scale", ok=True)
        result = _dispatch_result(model=action.model, action_kind="scale", response=plan_items.item_failure(item))
        slept = plan_items.slept_ids(item)
        if slept:
            result = replace(result, changed=slept, not_executed=False)
        return result

    def _named_wake_result(self, action: ScaleAction, item: dict) -> DispatchResult:
        woke = plan_items.woke_ids(item)
        if item.get("status") in plan_items.WAKE_OK_STATUSES:
            return DispatchResult(model=action.model, action_kind="scale", ok=True,
                                  woke_none=item.get("status") == "noop" and not woke)
        self._refusal_events(action.model, item)
        result = _dispatch_result(model=action.model, action_kind="scale", response=plan_items.item_failure(item))
        if woke:
            result = replace(result, changed=woke, not_executed=False)
        return result

    def _shrink_result(self, action: ScaleAction, item: dict) -> DispatchResult:
        """A model-level shrink (``at_most``). Accounted by ``taken`` (replicas really put
        to sleep): 0 changed nothing; the SM clamps at the replica floor and reports
        ``clamped_by_floor`` - an event (``scale_clamped_by_floor``), never a hold."""
        if item.get("status") not in plan_items.SLEEP_OK_STATUSES:
            result = _dispatch_result(model=action.model, action_kind="scale", response=plan_items.item_failure(item))
            slept = plan_items.slept_ids(item)
            return replace(result, changed=slept, not_executed=False) if slept else result
        taken = plan_items.num(item, "taken")
        asked = -int(action.delta)
        if taken < asked:
            if item.get("clamped_by_floor"):
                self._stats["scale_clamped_by_floor_total"] += 1
                self._events.append(f"scale_clamped_by_floor:{action.model}:taken={taken}:asked={asked}")
            else:
                self._events.append(f"scale_down_short:{action.model}:taken={taken}:asked={asked}")
            LOG.info(
                json.dumps(
                    {"event": "scale_down_taken", "model": action.model, "asked": asked, "taken": taken,
                     "clamped_by_floor": bool(item.get("clamped_by_floor")), "floor": item.get("floor")},
                    sort_keys=True, default=str,
                )
            )
        return DispatchResult(model=action.model, action_kind="scale", ok=True, taken=taken)

    def _growth_result(self, action: ScaleAction, item: dict) -> DispatchResult:
        """A model-level growth (``at_least``, hinted or not). ``placement_substituted``
        for every replica the SM woke somewhere else than a hint; every refusal is a
        ``wake_refused`` event (no cooldown since 2026-10-02). A growth filled in part is
        not a success (relative: not retried, the planner re-plans) but what did wake is
        reported (``changed`` / ``picked``) and stamped."""
        entries = plan_items.woke_entries(item)
        woke = plan_items.woke_ids(item)
        picked = entries
        if action.hint:
            for entry in entries:
                if entry.get("hinted", True) or entry.get("source") == "hint":
                    continue
                source = _binding_gpus(entry.get("hint_binding_id")) or "?"
                target = f"{entry.get('node')}/{','.join(str(g) for g in entry.get('gpu_ids') or ())}"
                self._events.append(f"placement_substituted:{action.model}:{source}->{target}")
        self._refusal_events(action.model, item)
        status = item.get("status")
        unfilled = plan_items.num(item, "unfilled")
        if status in plan_items.WAKE_OK_STATUSES and unfilled <= 0:
            return DispatchResult(model=action.model, action_kind="scale", ok=True, picked=picked,
                                  woke_none=status == "noop" and not woke)
        if woke:
            return DispatchResult(
                model=action.model, action_kind="scale", ok=False,
                error=f"partial: {unfilled} of {action.delta} wakes unfilled", retriable=False,
                picked=picked, changed=woke,
            )
        result = _dispatch_result(model=action.model, action_kind="scale", response=plan_items.item_failure(item))
        # Nothing woke: no view-pending stamp. ``wake_conflict`` (the first refusal)
        # is information only - every refusal is an event already.
        return replace(result, retriable=False, woke_none=True)

    def _receiver_item_result(self, action: ReceiverTarget, item: dict) -> DispatchResult:
        """A SafeScale receiver's absolute grow-only target. Filled in part
        (``unfilled``) stays ok: the woken replicas are a real "up" and a re-send could
        not place more now - the next tick re-plans from the view (``receiver_short``
        event). Nothing woken (a no-op included) is ``woke_none`` (design §5.5): no
        stamp, no last action. A refused / failed item keeps the item's ``retriable``."""
        status = item.get("status")
        woke = plan_items.woke_ids(item)
        picked = plan_items.woke_entries(item)
        self._refusal_events(action.model, item)
        if status not in ("done", "noop", "partial") and not woke:
            result = _dispatch_result(model=action.model, action_kind="scale", response=plan_items.item_failure(item))
            return replace(result, woke_none=True)
        unfilled = plan_items.num(item, "unfilled")
        if unfilled > 0:
            self._stats["receiver_short_total"] += 1
            self._events.append(f"receiver_short:{action.model}:unfilled={unfilled}:target={action.target}")
            LOG.info(
                json.dumps(
                    {"event": "receiver_short", "model": action.model, "target": action.target,
                     "unfilled": unfilled, "woke": list(woke)},
                    sort_keys=True,
                )
            )
        return DispatchResult(model=action.model, action_kind="scale", ok=True, picked=picked,
                              woke_none=not woke and (status == "noop" or unfilled > 0))


    def _log_plan(self, request: dict, body: dict) -> None:
        LOG.info(
            json.dumps(
                {"event": "plan_result", "plan_id": body.get("plan_id"), "status": body.get("status"),
                 "sleeps": [[item.get("model"), item.get("status"), item.get("taken")]
                            for item in body.get("sleeps") or () if isinstance(item, dict)],
                 "wakes": [[item.get("model"), item.get("status"), len(item.get("woke") or ())]
                           for item in body.get("wakes") or () if isinstance(item, dict)],
                 "asked": {"sleeps": len(request["sleeps"]), "wakes": len(request["wakes"])},
                 "phases_ms": body.get("phases_ms") if isinstance(body.get("phases_ms"), dict) else {}},
                sort_keys=True, default=str,
            )
        )

    async def _timed_dispatch(self, action, model: str) -> DispatchResult:
        """One non-power SM call (hide / unhide / defrag). An exception (a client bug, a malformed answer) becomes a
        failed, non-retriable result instead of killing the dispatch task."""
        started_ns = time.perf_counter_ns()
        try:
            with sm_actor(_actor_for(action)):
                result = await self._dispatch(action, model)
        except asyncio.CancelledError:
            raise
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
        """View-pending stamps, after the SM answered: a failed or partial call may
        still have changed the routable set, so it is stamped too - except a call that
        certainly changed nothing (2026-10-02): a model-level shrink with ``taken: 0``,
        a relay side without any pair, a refused relay, a partial /
        unfilled growth that woke nothing (review P2-2)."""
        now = int(self._now_ms())

        def stamp(target: str, hold: str) -> None:
            self._view_change[target] = (now, hold)
            if self._on_fleet_change is not None:
                try:
                    self._on_fleet_change()
                except Exception:  # noqa: BLE001 - the periodic refresh still runs
                    LOG.exception("fleet-change callback failed")

        if not isinstance(action, (ScaleAction, ReceiverTarget, HideAction, UnhideAction)):
            return
        if result.not_executed:
            return  # refused before any change (writer_busy, routable_unknown, ...)
        if result.woke_none:
            return  # a partial / unfilled growth that woke nothing (review P2-2)
        if isinstance(action, ScaleAction) and action.delta < 0 and result.ok and result.taken == 0:
            return  # the replica floor clamped the shrink to nothing: no change
        stamp(getattr(action, "model", model), _view_hold_direction(action))

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
        """One non-power SM call (hide / unhide / defrag). Power actions go through
        :meth:`_run_plan`."""
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

    def _refusal_events(self, model: str, body: dict) -> None:
        """``refusals[]`` of a plan wake item: one ``wake_refused`` event each."""
        for refusal in body.get("refusals") or ():
            if isinstance(refusal, dict):
                self._wake_refused_event(model, plan_items.refusal_conflict(refusal))

    def _record_done(self, model: str, action, result: DispatchResult) -> None:
        direction = _action_direction(action)
        if not (result.ok or result.changed) or direction is None:
            return
        if result.woke_none and not result.changed:
            return  # no-op fix (design §5.5): a growth that woke nothing is no "up"
        if direction == "down" and result.taken == 0:
            return  # clamped to nothing (2026-10-02): no scale-down happened
        self._last_done[model] = (int(self._now_ms()), direction)
        self._persist_scale_memory(model)

    def _has_pending_model(self, model: str) -> bool:
        return any(model in item.models for item in self._pending)

    def _remove_pending_fairness_for_model(self, model: str) -> tuple[tuple[str, SourceLoop], ...]:
        removed: list[tuple[str, SourceLoop]] = []
        retained: deque[QueuedAction] = deque()
        for item in self._pending:
            if isinstance(item.action, PlanBatch):
                # A fused plan: only its fairness members of ``model`` go.
                kept = tuple(
                    member for member in item.action.members
                    if not (member.source_loop == "fairness" and model in _queued_models(member))
                )
                removed.extend((model, "fairness") for _ in range(len(item.action.members) - len(kept)))
                if len(kept) == len(item.action.members):
                    retained.append(item)
                elif kept:
                    retained.append(self._queued(kept[0] if len(kept) == 1 else PlanBatch(kept)))
                continue
            if model in item.models and item.source_loop == "fairness":
                removed.append((model, item.source_loop))
                continue
            retained.append(item)
        self._pending = retained
        return tuple(removed)

    def _queued(self, action: QueueAction) -> QueuedAction:
        if isinstance(action, BlockSupplyAction) and action.probe is None:
            action = transfer_plan(action)  # a transfer released at once: one plan group
        if isinstance(action, DefragAction):
            return QueuedAction(
                action=action,
                model=CLUSTER_MODEL,
                source_loop=action.source_loop,
                resources=frozenset({CLUSTER_RESOURCE}),
            )
        if isinstance(action, SafeScaleCommitAction):
            return self._commit_queued(action)
        if isinstance(action, PlanBatch):
            # Every member's models and named pods.
            models = tuple(dict.fromkeys(m for member in action.members for m in _queued_models(member)))
            pods = tuple(pod for member in action.members for pod in (getattr(member, "pods", ()) or ()))
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
        serializes GPU use (its global writer lock and GPU leases)."""
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
    """The C1-tagged rescue scale-up of an action (a ScaleAction, a PlanBatch's
    receiver member), or None."""
    if isinstance(action, PlanBatch):
        return next((part for member in action.members if (part := _rescue_part(member)) is not None), None)
    if isinstance(action, ScaleAction) and action.delta > 0 and getattr(action, "rescue", None) is not None:
        return action
    return None


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
    if isinstance(action, PlanBatch):
        return [result for member in _members(action) for result in _member_observe_skipped(member)]
    return [
        DispatchResult(
            model=queued.model,
            action_kind=_action_kind(action),
            # A hide not sent did not take effect (P3): never reported as done.
            ok=not isinstance(action, HideAction),
            error="observe_skipped",
        )
    ]


def _action_kind(action) -> str:
    if isinstance(action, PlanBatch):
        return "plan"
    if isinstance(action, SafeScaleCommitAction):
        return "safescale_commit"
    if isinstance(action, (ScaleAction, ReceiverTarget)):
        return "scale"
    if isinstance(action, HideAction):
        return "hide"
    if isinstance(action, UnhideAction):
        return "unhide"
    if isinstance(action, DefragAction):
        return "defrag"
    return "unknown"


def _view_hold_direction(action) -> str:
    """Direction of a routable-changing SM call for the view-pending guard (logged with
    the hold): a wake / scale-up is "up"; a sleep, hide or unhide "down"."""
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


@dataclass(frozen=True)
class PlanBatch:
    """Re-plannable power actions (``ScaleAction``; a transfer's named items) of ONE
    ``submit()`` fused into ONE ``POST /v2/plans`` (design 20261027-sm-plans §5.2, as
    narrowed by the architect): each member is exactly one plan item and no two members
    touch the same model (a relay's donor counts as used by its item)."""

    members: tuple[Action, ...]

    @property
    def model(self) -> str:
        return _member_model(self.members[0])

    @property
    def source_loop(self) -> SourceLoop:
        return self.members[0].source_loop

    @property
    def reason(self) -> str:
        return "+".join(dict.fromkeys(str(getattr(member, "reason", "")) for member in self.members))


@dataclass(frozen=True)
class _PlanRun:
    """One plan dispatch: the members (SafeScale receivers with their resolved target)
    and one result per member; ``observe_skipped`` = nothing sent (observe mode)."""

    members: tuple
    results: tuple
    observe_skipped: bool = False
    #: No item results: the GET or the POST failed as a whole (refused, unknown).
    whole_plan_failed: bool = False


def receiver_upscales(action) -> tuple[tuple[str, int], ...]:
    """(model, receiver replicas) every receiver part of an action adds: a scale-up or a
    relay (:func:`upscale_of`), the members of a :class:`PlanBatch`, the receiver of a TP
    block (BlockSupplyAction: its wake comes with the donors' release)."""
    if isinstance(action, PlanBatch):
        return tuple(up for member in action.members for up in receiver_upscales(member))
    if isinstance(action, BlockSupplyAction):
        return ((action.receiver, 1),)
    up = upscale_of(action)
    return (up,) if up is not None else ()


def _members(action) -> tuple:
    return tuple(action.members) if isinstance(action, PlanBatch) else (action,)


def _member_model(action) -> str:
    """The model a member's result is reported for (a relay: its receiver)."""
    return str(getattr(action, "model"))


def block_sleeps(action: BlockSupplyAction, reason: str, group: str | None = None) -> tuple[ScaleAction, ...]:
    """The urgent sleeps of exactly the transfer's immediate donor pods (``group``: the
    transfer's plan group when its receiver wakes in the same plan)."""
    return tuple(
        ScaleAction(
            donor, -1, reason, action.source_loop, donor=donor, receiver=action.receiver, pods=(pod,),
            sleep_path=IMMEDIATE_DONOR_SLEEP_PATH, group=group,
        )
        for donor, pod in action.sleeps
    )


def transfer_group(action: BlockSupplyAction) -> str:
    """The plan group of one transfer (sm-plans section 10): unique in a plan, as a
    receiver binding is woken by one transfer only."""
    return f"t:{action.receiver}:{action.receiver_pods[0] if action.receiver_pods else action.block}"


def transfer_plan(action: BlockSupplyAction, reason: str | None = None) -> PlanBatch:
    """One transfer released at once (sm-plans section 10, design 20261028-single-loop
    L4 rev 2) -> ONE plan group: the urgent sleeps of exactly its donor pods and the
    receiver's NAMED wake of its binding (it keeps its rescue target). The SM selects
    the group atomically and wakes the receiver once every donor slept; no other
    receiver can take the GPUs between. ``action.probe`` must be None (a probe pod
    sleeps through the SafeScale commit)."""
    reason = action.reason if reason is None else reason
    group = transfer_group(action)
    return PlanBatch((
        *block_sleeps(action, reason, group),
        ScaleAction(
            action.receiver, 1, reason, action.source_loop, receiver=action.receiver,
            donor=action.donor_models[0] if action.donor_models else None,
            pods=action.receiver_pods, rescue=action.rescue, group=group, basis=action.basis,
        ),
    ))


def _queued_models(action) -> tuple[str, ...]:
    """Every model an action changes."""
    if isinstance(action, DefragAction):
        return (CLUSTER_MODEL,)
    if isinstance(action, SafeScaleCommitAction):
        return tuple(action.touched_models) or (action.donor,)
    return (action.model,)


class _PlanKeys:
    """What one plan's items claim (the SM's request rules, sm-plans section 10): named
    pods (each at most once), models of count items (at most one item per model, no
    named item beside it) and models of named items."""

    def __init__(self) -> None:
        self.pods: set[str] = set()
        self.count_models: set[str] = set()
        self.named_models: set[str] = set()

    @classmethod
    def of(cls, action) -> "_PlanKeys":
        keys = cls()
        for member in _members(action):
            if isinstance(member, ScaleAction) and member.pods:
                keys.pods |= set(member.pods)
                keys.named_models.add(member.model)
            elif isinstance(member, BlockSupplyAction):
                keys.pods |= {pod for _, pod in member.sleeps} | set(member.receiver_pods)
                keys.named_models |= {*member.donor_models, member.receiver}
            elif isinstance(member, ScaleAction):
                keys.count_models.add(member.model)
            else:  # a commit, a defrag: every model it touches
                keys.count_models |= set(_queued_models(member))
        return keys

    def clashes(self, other: "_PlanKeys") -> bool:
        return bool(
            self.pods & other.pods
            or self.count_models & (other.count_models | other.named_models)
            or self.named_models & other.count_models
        )

    def add(self, other: "_PlanKeys") -> None:
        self.pods |= other.pods
        self.count_models |= other.count_models
        self.named_models |= other.named_models


def _fusable(queued: QueuedAction) -> bool:
    """A re-plannable power action that may share one plan with the other actions of
    its submit (a SafeScale one-shot action keeps its own plan and retries)."""
    action = queued.action
    if queued.source_loop in ONE_SHOT_LOOPS:
        return False
    if isinstance(action, PlanBatch):  # a TP block: one unit
        return True
    return isinstance(action, ScaleAction) and action.delta != 0


def _replannable_power(queued: QueuedAction) -> bool:
    """Dispatched as one plan without retry: a fused batch, a re-plannable scale."""
    action = queued.action
    if isinstance(action, PlanBatch):
        return True
    return isinstance(action, ScaleAction) and queued.source_loop not in ONE_SHOT_LOOPS


def _member_observe_skipped(member) -> list[DispatchResult]:
    return [DispatchResult(model=_member_model(member), action_kind=_action_kind(member), ok=True,
                           error="observe_skipped")]


def _response_item(body: dict, kind: str, position: int) -> dict | None:
    """The item of a plan response for request position ``position`` (by ``index``,
    else by position)."""
    items = [item for item in body.get(kind + "s") or () if isinstance(item, dict)]
    for item in items:
        if item.get("index") == position:
            return item
    if position < len(items) and "index" not in items[position]:
        return items[position]
    return None


def _plan_actor(members: tuple) -> str:
    """``X-TRE-Actor`` of a plan: the member's own actor, or the loop and the members'
    reasons for a fused plan."""
    if len(members) == 1:
        return _actor_for(members[0])
    loop = getattr(members[0], "source_loop", None) or "unknown"
    reasons = "+".join(dict.fromkeys(str(getattr(member, "reason", None) or type(member).__name__)
                                     for member in members))
    return f"controller/{loop}/plan:{reasons}"


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


async def _wait_either(event: asyncio.Event, poll: Awaitable[None]) -> None:
    """Return when ``event`` is set or ``poll`` (the poll sleep) finished, whichever
    comes first; an exception of ``poll`` propagates."""
    waiter = asyncio.ensure_future(event.wait())
    sleeper = asyncio.ensure_future(poll)
    try:
        done, _pending = await asyncio.wait((waiter, sleeper), return_when=asyncio.FIRST_COMPLETED)
    finally:
        for task in (waiter, sleeper):
            if not task.done():
                task.cancel()
    if sleeper in done:
        sleeper.result()
