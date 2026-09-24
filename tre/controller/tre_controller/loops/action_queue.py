from __future__ import annotations

import asyncio
import json
import logging
import time
from collections import deque
from dataclasses import dataclass, field, replace
from typing import Awaitable, Callable, Protocol

from tre_controller.planning.planner import Action, DefragAction, HideAction, ScaleAction, SourceLoop, UnhideAction

if False:  # annotations are strings (from __future__); avoids an import cycle
    from tre_controller.profiling import TickProfiler

CLUSTER_MODEL = "__cluster__"
LOG = logging.getLogger("tre_controller.action_queue")
_TERMINAL = frozenset({"succeeded", "failed", "superseded"})
_OK_TERMINAL = frozenset({"succeeded", "superseded"})


class ServiceManagerClient(Protocol):
    async def scale_model(self, model: str, delta: int) -> dict: ...

    async def set_routable(self, model: str, hidden_pods: tuple[str, ...]) -> dict: ...

    async def set_binding_power(self, serve_id: str, *, awake: bool) -> dict: ...

    async def defrag(self, migrations: tuple) -> dict: ...


@dataclass(frozen=True)
class QueuedAction:
    action: Action
    model: str
    source_loop: SourceLoop
    # TRE_SM_ASYNC only (0 / None otherwise): submit order and, for a receiver wake
    # that needs the GPU a donor sleep of the same batch frees, that sleep's seq.
    seq: int = 0
    depends_on: int | None = None
    # TRE_SM_ASYNC review M3: a rescue scale-up that supersedes the model's active
    # scale-down operation (the SM reclaims the draining binding at once).
    supersedes: bool = False
    # Review H2: bounded retry of one-shot actions (hide / unhide / safescale).
    attempt: int = 0
    first_try_ms: int = 0
    not_before_ms: int = 0


@dataclass(frozen=True)
class PendingOpView:
    """An SM operation the controller dispatched and is still waiting for."""

    model: str
    delta: int
    pods: tuple[str, ...]
    source_loop: SourceLoop
    reason: str
    age_ms: int


@dataclass
class _TrackedOp:
    queued: QueuedAction
    action_kind: str
    dispatched_ms: int
    deadline_ms: int
    next_poll_ms: int
    op_ids: list[str] = field(default_factory=list)
    status: dict[str, str] = field(default_factory=dict)
    records: dict[str, dict] = field(default_factory=dict)
    not_found: dict[str, int] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)
    polls: int = 0


@dataclass(frozen=True)
class SubmitResult:
    accepted: int
    held: int = 0
    dropped: tuple[tuple[str, str], ...] = ()
    replaced: tuple[tuple[str, SourceLoop], ...] = ()


@dataclass(frozen=True)
class DispatchResult:
    model: str
    action_kind: str
    ok: bool
    error: str | None = None


class ActionQueue:
    def __init__(
        self,
        client: ServiceManagerClient,
        *,
        is_observe: Callable[[], bool] | None = None,
        prof: "TickProfiler | None" = None,
        now_ms: Callable[[], int] | None = None,
        async_ops: bool = False,
        call_drain: bool = False,
        poll_interval_s: float = 1.0,
        op_timeout_s: float = 600.0,
        max_polls: int = 8,
        audit_on_failure: bool = True,
        oneshot_retry_s: float = 120.0,
        routable_timeout_s: float = 45.0,
    ) -> None:
        self._client = client
        self._pending: deque[QueuedAction] = deque()
        self._inflight: set[str] = set()
        # When this returns True the controller is paused: queued actions are drained
        # (inflight cleared, so the next tick can re-plan) but NEVER dispatched.
        self._is_observe = is_observe or (lambda: False)
        self._prof = prof
        # Review F4: model -> (epoch ms the last successful dispatch completed, "up"/"down").
        self._now_ms = now_ms or (lambda: int(time.time() * 1000))
        self._last_done: dict[str, tuple[int, str]] = {}
        # TRE_SM_ASYNC: dispatch returns an SM operation id; the model stays inflight
        # until the operation is terminal (polled, bounded). TRE_SM_CALL_DRAIN: every
        # sleep carries drain_s (SafeScale commit: its budget; everything else: 0).
        # Both off -> the _legacy paths below, identical to main.
        self._async_ops = bool(async_ops)
        self._call_drain = bool(call_drain)
        self._poll_ms = max(1, int(float(poll_interval_s) * 1000))
        self._op_timeout_ms = max(1, int(float(op_timeout_s) * 1000))
        self._max_polls = max(1, int(max_polls))
        self._audit_on_failure = bool(audit_on_failure)
        self._ops: dict[str, _TrackedOp] = {}
        self._seq = 0
        self._seq_outcome: dict[int, bool] = {}
        self._op_stats: dict[str, dict[str, float]] = {}
        self._last_audit_ms: int | None = None
        self._oneshot_retry_ms = max(0, int(float(oneshot_retry_s) * 1000))
        self._routable_timeout_s = float(routable_timeout_s)
        # Review M4: take over the SM operations a previous controller left running.
        self._adopted = False
        self._next_adopt_ms = 0
        self._action_ids = 0

    def submit(self, actions: tuple[Action, ...] | list[Action]) -> SubmitResult:
        if self._async_ops:
            return self._submit_async(actions)
        return self._submit_legacy(actions)

    def _submit_legacy(self, actions: tuple[Action, ...] | list[Action]) -> SubmitResult:
        queued_actions = tuple(_queued_action(action) for action in actions)
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

        for queued in queued_actions:
            if queued.source_loop == "rescue":
                removed = self._remove_pending_fairness_for_model(queued.model)
                replaced.extend(removed)
            elif queued.model in self._inflight or self._has_pending_model(queued.model):
                dropped.append((queued.model, "inflight"))
                continue

            self._pending.append(queued)
            self._inflight.add(queued.model)
            accepted += 1
            if observe and queued.source_loop == "safescale":
                held += 1

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

    def last_actions(self) -> dict[str, tuple[int, str]]:
        return dict(self._last_done)

    async def run(
        self,
        *,
        poll_interval_s: float = 0.1,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        while True:
            await self.drain_once()
            await sleep(poll_interval_s)

    async def drain_once(self) -> tuple[DispatchResult, ...]:
        if self._async_ops or self._call_drain:
            return await self._drain_once_v2()
        return await self._drain_once_legacy()

    async def _drain_once_legacy(self) -> tuple[DispatchResult, ...]:
        observe = self._is_observe()
        results: list[DispatchResult] = []
        # Safescale resolution commands are one-shot (they are never re-emitted by the
        # SafeScaleStateMachine, which deletes the probe on resolve). If we are paused in
        # observe mode we must NOT drop them like idempotent planner actions -- hold them
        # in _pending (keeping the model inflight so no conflicting action is queued) so
        # they dispatch for real once mode returns to non-observe.
        held: deque[QueuedAction] = deque()
        _prof_on = self._prof is not None
        _dispatched = 0
        _http_ns = 0
        while self._pending:
            queued = self._pending.popleft()
            if observe and queued.source_loop == "safescale":
                held.append(queued)
                continue
            if observe:
                results.append(DispatchResult(model=queued.model, action_kind=_action_kind(queued.action),
                                              ok=True, error="observe_skipped"))
            else:
                if _prof_on:
                    _d0 = time.perf_counter_ns()
                    results.append(await self._dispatch(queued.action, queued.model))
                    _http_ns += time.perf_counter_ns() - _d0
                    _dispatched += 1
                else:
                    results.append(await self._dispatch(queued.action, queued.model))
                self._record_done(queued, results[-1])
            self._inflight.discard(queued.model)
        self._pending = held
        if _prof_on and _dispatched:
            self._prof.record(
                {
                    "kind": "dispatch",
                    "ts_ms": self._prof.now_ms(),
                    "n_actions": _dispatched,
                    "http_ns": _http_ns,
                }
            )
        return tuple(results)

    async def _dispatch(self, action: Action, model: str) -> DispatchResult:
        if isinstance(action, ScaleAction):
            if action.delta != 0 and action.pods:
                return await self._dispatch_binding_power(action)
            response = await self._client.scale_model(action.model, action.delta)
            return _dispatch_result(model=action.model, action_kind="scale", response=response)
        if isinstance(action, HideAction):
            response = await self._client.set_routable(action.model, action.pods)
            return _dispatch_result(model=action.model, action_kind="hide", response=response)
        if isinstance(action, UnhideAction):
            response = await self._client.set_routable(action.model, ())
            return _dispatch_result(model=action.model, action_kind="unhide", response=response)
        if isinstance(action, DefragAction):
            response = await self._client.defrag(tuple(action.migrations))
            return _dispatch_result(model=CLUSTER_MODEL, action_kind="defrag", response=response)
        return DispatchResult(model=model, action_kind="unknown", ok=False, error="unsupported_action")

    async def _dispatch_binding_power(self, action: ScaleAction) -> DispatchResult:
        # Sleep (delta < 0) or wake (delta > 0) exactly the named bindings: safescale
        # commit of the hidden pod, slot-targeted donor, or a planned slot-aware wake.
        # Stops at the first failure.
        for pod in action.pods:
            response = await self._client.set_binding_power(pod, awake=action.delta > 0)
            if not bool(response.get("ok", False)):
                return _dispatch_result(model=action.model, action_kind="scale", response=response)
        return DispatchResult(model=action.model, action_kind="scale", ok=True)

    # ------------------------------------------------------------------
    # TRE_SM_ASYNC / TRE_SM_CALL_DRAIN paths (default off)
    # ------------------------------------------------------------------

    def pending_ops_view(self) -> tuple[PendingOpView, ...]:
        """Dispatched scale operations not yet terminal (planner capacity accounting:
        a pending wake is incoming capacity, a pending sleep is leaving capacity)."""
        now = int(self._now_ms())
        views: list[PendingOpView] = []
        for tracked in self._ops.values():
            action = tracked.queued.action
            if not isinstance(action, ScaleAction):
                continue
            views.append(
                PendingOpView(
                    model=action.model,
                    delta=int(action.delta),
                    pods=tuple(action.pods),
                    source_loop=tracked.queued.source_loop,
                    reason=action.reason,
                    age_ms=max(0, now - tracked.dispatched_ms),
                )
            )
        return tuple(views)

    def supersedable_models(self) -> set[str]:
        """Models whose active SM operation is a scale-down: a CRITICAL rescue scale-up
        of the same model may supersede it (review M3); nothing else may."""
        return {
            model
            for model, tracked in self._ops.items()
            if isinstance(tracked.queued.action, ScaleAction) and tracked.queued.action.delta < 0
        }

    def op_stats(self) -> dict[str, dict[str, float]]:
        return {key: dict(value) for key, value in self._op_stats.items()}

    def _submit_async(self, actions: tuple[Action, ...] | list[Action]) -> SubmitResult:
        # Same accept/drop rules as main (a model with an active SM operation is still
        # in _inflight, so it is dropped as "inflight" - no duplicate actions); then
        # the freshly accepted tail is numbered and dependencies are recorded.
        # main lets a rescue action through for an inflight model (it replaces a
        # pending fairness one); with async dispatch that model may have an SM
        # operation running, so such a rescue action is dropped ("active_op") - the
        # planner re-plans once the operation completed. Safescale batches are not
        # touched here: their all-or-nothing check below already rejects them.
        # Review M3: a rescue scale-UP of a model whose active operation is a
        # scale-down supersedes it (latest wins at the SM: the draining, still awake
        # binding is reclaimed as immediate capacity); every other action for a
        # model with an active operation is still blocked.
        blocked: list[tuple[str, str]] = []
        kept: list[Action] = []
        superseding: set[str] = set()
        supersedable = self.supersedable_models()
        for action in actions:
            queued = _queued_action(action)
            if queued.source_loop == "rescue" and queued.model in self._ops:
                if isinstance(action, ScaleAction) and action.delta > 0 and queued.model in supersedable:
                    superseding.add(queued.model)
                    kept.append(action)
                    continue
                blocked.append((queued.model, "active_op"))
                continue
            kept.append(action)
        before = list(self._pending)
        result = self._submit_legacy(kept)
        # Review H1: main's rescue path removes pending fairness actions of the
        # model; one of them may be a donor sleep a receiver wake depends on.
        remaining = {id(item) for item in self._pending}
        for item in before:
            if id(item) not in remaining:
                self._settle(item, ok=False)
        if result.accepted:
            items = list(self._pending)
            head, fresh = items[: -result.accepted], items[-result.accepted :]
            annotated = self._annotate_batch(fresh)
            for position, item in enumerate(annotated):
                action = item.action
                if (
                    item.model in superseding
                    and item.source_loop == "rescue"
                    and isinstance(action, ScaleAction)
                    and action.delta > 0
                ):
                    # model-level, so the SM target path reclaims the draining binding
                    annotated[position] = replace(item, action=replace(action, pods=()), supersedes=True)
            self._pending = deque(head + annotated)
        if blocked:
            result = replace(result, dropped=tuple(blocked) + result.dropped)
        return result

    def _annotate_batch(self, fresh: list[QueuedAction]) -> list[QueuedAction]:
        annotated: list[QueuedAction] = []
        sleep_seq: dict[str, int] = {}
        safescale_down_seq: int | None = None
        for queued in fresh:
            self._seq += 1
            seq = self._seq
            action = queued.action
            depends_on = None
            if isinstance(action, ScaleAction) and action.delta > 0:
                if action.donor and action.donor != action.model and action.donor in sleep_seq:
                    # critical/low donor-immediate pair: the receiver wakes into the
                    # slot the donor sleep frees -> only after that sleep succeeded.
                    depends_on = sleep_seq[action.donor]
                elif queued.source_loop == "safescale" and safescale_down_seq is not None:
                    # SafeScale commit follow-up upscales: only once the commit's
                    # sleep of the hidden pods succeeded.
                    depends_on = safescale_down_seq
            if isinstance(action, ScaleAction) and action.delta < 0:
                sleep_seq[queued.model] = seq
                if queued.source_loop == "safescale":
                    safescale_down_seq = seq
            annotated.append(replace(queued, seq=seq, depends_on=depends_on))
        return annotated

    async def _drain_once_v2(self) -> tuple[DispatchResult, ...]:
        observe = self._is_observe()
        results: list[DispatchResult] = []
        if self._async_ops:
            if not self._adopted:
                await self._adopt_active_ops()
            # Already-dispatched operations are tracked to completion in every mode,
            # observe included: nothing the SM is executing is silently forgotten.
            results.extend(await self._poll_ops())
        held: deque[QueuedAction] = deque()
        now = int(self._now_ms())
        while self._pending:
            queued = self._pending.popleft()
            kind = _action_kind(queued.action)
            if observe and queued.source_loop == "safescale":
                # One-shot (see _drain_once_legacy): hold, never drop.
                held.append(queued)
                continue
            if observe and not (queued.attempt and _is_oneshot(queued)):
                results.append(DispatchResult(model=queued.model, action_kind=kind, ok=True, error="observe_skipped"))
                self._settle(queued, ok=False)
                self._inflight.discard(queued.model)
                continue
            if observe or queued.not_before_ms > now:
                held.append(queued)  # a one-shot retry: back off, never drop
                continue
            if queued.depends_on is not None:
                outcome = self._seq_outcome.get(queued.depends_on)
                if outcome is None and not self._seq_alive(queued.depends_on, held):
                    # Review H1: the donor action vanished without an outcome
                    # (replaced, pruned): never wait for it forever.
                    outcome = False
                if outcome is None:
                    held.append(queued)  # its donor sleep is still running
                    continue
                if not outcome:
                    results.append(
                        DispatchResult(model=queued.model, action_kind=kind, ok=False, error="dependency_failed")
                    )
                    self._settle(queued, ok=False)
                    self._inflight.discard(queued.model)
                    continue
            dispatched = await self._dispatch_v2(queued)
            if dispatched is None:
                continue  # async SM operation accepted: the model stays inflight
            if not dispatched.ok and self._retry_oneshot(queued, dispatched, held):
                continue
            results.append(dispatched)
            self._record_done(queued, dispatched)
            self._settle(queued, ok=dispatched.ok)
            self._inflight.discard(queued.model)
        self._pending = held
        return tuple(results)

    def _seq_alive(self, seq: int, held: deque[QueuedAction]) -> bool:
        if any(item.seq == seq for item in self._pending) or any(item.seq == seq for item in held):
            return True
        return any(tracked.queued.seq == seq for tracked in self._ops.values())

    def _retry_oneshot(self, queued: QueuedAction, result: DispatchResult, held: deque[QueuedAction]) -> bool:
        """Review H2: a one-shot action (hide / unhide / SafeScale) that hit a busy
        writer lock (409), a 5xx or a timeout is retried with backoff for at most
        oneshot_retry_s - never dropped silently. True = rescheduled."""
        if not _is_oneshot(queued) or not _retryable(result.error):
            if _is_oneshot(queued):
                self._alert_oneshot(queued, result, retried=False)
            return False
        now = int(self._now_ms())
        first = queued.first_try_ms or now
        if now - first >= self._oneshot_retry_ms:
            self._alert_oneshot(queued, result, retried=True)
            return False
        backoff = min(10_000, 1_000 * (2 ** min(queued.attempt, 4)))
        held.append(replace(queued, attempt=queued.attempt + 1, first_try_ms=first, not_before_ms=now + backoff))
        LOG.warning(
            json.dumps(
                {
                    "event": "sm_oneshot_retry",
                    "model": queued.model,
                    "action": _action_kind(queued.action),
                    "source": queued.source_loop,
                    "attempt": queued.attempt + 1,
                    "backoff_ms": backoff,
                    "error": result.error,
                },
                separators=(",", ":"),
            )
        )
        return True

    def _alert_oneshot(self, queued: QueuedAction, result: DispatchResult, *, retried: bool) -> None:
        LOG.error(
            json.dumps(
                {
                    "event": "sm_oneshot_action_failed",
                    "model": queued.model,
                    "action": _action_kind(queued.action),
                    "source": queued.source_loop,
                    "pods": list(getattr(queued.action, "pods", ()) or ()),
                    "attempts": queued.attempt + 1,
                    "retried": retried,
                    "error": result.error,
                    # a pod left hidden is picked up by the SM reconcile / the
                    # controller HiddenOrphanDetector (TRE_ORPHAN_GRACE_S)
                    "handoff": "reconcile/hidden_orphan_detector",
                },
                separators=(",", ":"),
            )
        )

    async def _dispatch_v2(self, queued: QueuedAction) -> DispatchResult | None:
        action = queued.action
        if isinstance(action, (HideAction, UnhideAction)) and callable(
            getattr(self._client, "set_routable_v2", None)
        ):
            # The SM may wait (bounded) for its writer lock: allow for that.
            pods = action.pods if isinstance(action, HideAction) else ()
            response = await self._client.set_routable_v2(
                action.model, pods, timeout_s=self._routable_timeout_s
            )
            kind = "hide" if isinstance(action, HideAction) else "unhide"
            return _dispatch_result(model=action.model, action_kind=kind, response=response)
        if not isinstance(action, ScaleAction):
            # defrag stays synchronous, exactly as in main.
            return await self._dispatch(action, queued.model)
        drain_s = self._call_drain_s(action)
        meta = self._op_meta(queued) if self._async_ops else None
        extra = {"meta": meta} if meta is not None else {}
        responses: list[dict] = []
        if action.delta != 0 and action.pods:
            if not self._async_ops and drain_s and len(action.pods) > 1:
                # Review L4: synchronous commit of several pods - drain them in
                # parallel, one shared deadline, instead of N x the budget.
                responses = list(
                    await asyncio.gather(
                        *(
                            self._client.set_binding_power_v2(
                                pod, awake=action.delta > 0, drain_s=drain_s, async_op=False
                            )
                            for pod in action.pods
                        )
                    )
                )
            else:
                for pod in action.pods:
                    response = await self._client.set_binding_power_v2(
                        pod, awake=action.delta > 0, drain_s=drain_s, async_op=self._async_ops, **extra
                    )
                    responses.append(response)
                    if not bool(response.get("ok", False)):
                        break
        else:
            responses.append(
                await self._client.scale_model_v2(
                    action.model, action.delta, drain_s=drain_s, async_op=self._async_ops, **extra
                )
            )
        failure = next((item for item in responses if not bool(item.get("ok", False))), None)
        op_ids = [
            str(item["response"]["operation_id"])
            for item in responses
            if bool(item.get("ok", False))
            and isinstance(item.get("response"), dict)
            and item["response"].get("async_operation")
        ]
        if not op_ids:
            # Synchronous answer (flag off at the SM, or TRE_SM_ASYNC off here).
            if failure is not None:
                return _dispatch_result(model=action.model, action_kind="scale", response=failure)
            return DispatchResult(model=action.model, action_kind="scale", ok=True)
        now = int(self._now_ms())
        tracked = _TrackedOp(
            queued=queued,
            action_kind="scale",
            dispatched_ms=now,
            deadline_ms=now + self._op_timeout_ms,
            next_poll_ms=now + self._poll_ms,
            op_ids=op_ids,
        )
        if failure is not None:
            tracked.errors.append(str(failure.get("error") or "dispatch_failed"))
        previous = self._ops.pop(queued.model, None)
        if previous is not None:
            self._detach_superseded(previous, by=op_ids)
        self._ops[queued.model] = tracked
        return None

    def _op_meta(self, queued: QueuedAction) -> dict:
        """Stored with the SM operation so a restarted controller can take it over."""
        action = queued.action
        self._action_ids += 1
        meta: dict = {
            "controller": True,
            "action_id": f"{int(self._now_ms())}-{self._action_ids}",
            "model": queued.model,
            "source_loop": queued.source_loop,
        }
        if isinstance(action, ScaleAction):
            meta.update({"delta": int(action.delta), "pods": list(action.pods), "reason": action.reason})
            if queued.source_loop == "safescale" and action.delta < 0 and action.pods:
                meta["rollback_unhide"] = list(action.pods)
        return meta

    def _detach_superseded(self, tracked: _TrackedOp, *, by: list[str]) -> None:
        """Review M3: the controller replaced this scale-down with a scale-up of the
        same model. Not a success (no cooldown, no follow-ups) and no rollback: the
        SM reclaims its draining binding for the newer operation."""
        self._settle(tracked.queued, ok=False)
        result = DispatchResult(
            model=tracked.queued.model,
            action_kind=tracked.action_kind,
            ok=False,
            error="superseded_by_controller",
        )
        self._account(tracked, result, timed_out=False)
        LOG.info(
            json.dumps(
                {
                    "event": "sm_async_op_superseded",
                    "model": tracked.queued.model,
                    "operation_ids": list(tracked.op_ids),
                    "superseded_by": list(by),
                },
                separators=(",", ":"),
            )
        )

    async def _adopt_active_ops(self) -> None:
        """Review M4: after a controller restart, track the SM operations that are
        still active to their end (model inflight meanwhile). The action context is
        rebuilt from the meta stored with the operation; a failed SafeScale commit
        is still rolled back (meta.rollback_unhide)."""
        now = int(self._now_ms())
        if now < self._next_adopt_ms:
            return
        list_active = getattr(self._client, "list_active_operations", None)
        if not callable(list_active):
            self._adopted = True
            return
        response = await list_active()
        if not bool(response.get("ok", False)):
            self._next_adopt_ms = now + 5_000  # SM unreachable: try again shortly
            return
        self._adopted = True
        groups: dict[str, list[dict]] = {}
        for record in (response.get("response") or {}).get("operations") or []:
            if record.get("status") not in ("pending", "running"):
                continue
            meta = record.get("meta") or {}
            key = str(meta.get("action_id") or record.get("operation_id"))
            groups.setdefault(key, []).append(record)
        for records in groups.values():
            first = records[0]
            meta = first.get("meta") or {}
            model = str(meta.get("model") or first.get("model"))
            if model in self._ops:
                continue
            action = _adopted_action(first, meta)
            self._seq += 1
            queued = QueuedAction(
                action=action,
                model=model,
                source_loop=str(meta.get("source_loop") or "rescue"),
                seq=self._seq,
            )
            self._ops[model] = _TrackedOp(
                queued=queued,
                action_kind="scale",
                dispatched_ms=now,
                deadline_ms=now + self._op_timeout_ms,
                next_poll_ms=now,
                op_ids=[str(record["operation_id"]) for record in records],
            )
            self._inflight.add(model)
            LOG.warning(
                json.dumps(
                    {
                        "event": "sm_async_op_adopted",
                        "model": model,
                        "operation_ids": [str(record["operation_id"]) for record in records],
                        "meta": bool(meta),
                    },
                    separators=(",", ":"),
                )
            )

    def _call_drain_s(self, action: ScaleAction) -> float | None:
        if not self._call_drain or action.delta >= 0:
            return None
        return float(action.drain_s) if action.drain_s is not None else 0.0

    async def _poll_ops(self) -> list[DispatchResult]:
        if not self._ops:
            return []
        now = int(self._now_ms())
        due = [tracked for tracked in self._ops.values() if now >= tracked.next_poll_ms][: self._max_polls]

        async def poll(tracked: _TrackedOp) -> None:
            for operation_id in tracked.op_ids:
                if tracked.status.get(operation_id) in _TERMINAL:
                    continue
                response = await self._client.get_operation(operation_id)
                tracked.polls += 1
                if bool(response.get("ok", False)):
                    record = response.get("response") or {}
                    status = str(record.get("status") or "")
                    tracked.status[operation_id] = status
                    tracked.not_found.pop(operation_id, None)
                    if status in _TERMINAL:
                        tracked.records[operation_id] = record
                elif response.get("not_found"):
                    misses = tracked.not_found.get(operation_id, 0) + 1
                    tracked.not_found[operation_id] = misses
                    if misses >= 3:
                        tracked.status[operation_id] = "failed"
                        tracked.records[operation_id] = {"status": "failed", "error": "operation_unknown"}
                # transport errors: poll again next time, until the op timeout.
            tracked.next_poll_ms = int(self._now_ms()) + self._poll_ms

        if due:
            await asyncio.gather(*(poll(tracked) for tracked in due))
        results: list[DispatchResult] = []
        now = int(self._now_ms())
        for tracked in list(self._ops.values()):
            done = all(tracked.status.get(operation_id) in _TERMINAL for operation_id in tracked.op_ids)
            timed_out = not done and now >= tracked.deadline_ms
            if done or timed_out:
                results.append(await self._complete(tracked, timed_out=timed_out))
        return results

    async def _complete(self, tracked: _TrackedOp, *, timed_out: bool) -> DispatchResult:
        queued = tracked.queued
        self._ops.pop(queued.model, None)
        ok, error, rollback = _judge(tracked, timed_out=timed_out)
        result = DispatchResult(model=queued.model, action_kind=tracked.action_kind, ok=ok, error=error)
        # Review F4: the cooldown starts when the operation COMPLETED, not at dispatch.
        self._record_done(queued, result)
        self._settle(queued, ok=ok)
        self._inflight.discard(queued.model)
        self._account(tracked, result, timed_out=timed_out)
        if not ok:
            if rollback:
                self._rollback_failed_commit(queued)
            if self._audit_on_failure:
                await self._audit(queued, error)
        return result

    def _settle(self, queued: QueuedAction, *, ok: bool) -> None:
        if queued.seq:
            self._seq_outcome[queued.seq] = ok
            if len(self._seq_outcome) > 4096:
                for key in sorted(self._seq_outcome)[:2048]:
                    self._seq_outcome.pop(key, None)

    def _rollback_failed_commit(self, queued: QueuedAction) -> None:
        """A SafeScale commit whose sleep failed or timed out is rolled back like a
        probe rollback: unhide its pods (one-shot, safescale-sourced -> held, not
        dropped, in observe mode). The SM already made a known-awake pod routable
        again; the unhide is idempotent and skips bindings still draining."""
        action = queued.action
        if (
            queued.source_loop != "safescale"
            or not isinstance(action, ScaleAction)
            or action.delta >= 0
            or not action.pods
        ):
            return
        unhide = UnhideAction(action.model, tuple(action.pods), "safescale_commit_failed", "safescale")
        self._pending.append(QueuedAction(action=unhide, model=action.model, source_loop="safescale"))
        self._inflight.add(action.model)

    async def _audit(self, queued: QueuedAction, error: str | None) -> None:
        now = int(self._now_ms())
        if self._last_audit_ms is not None and now - self._last_audit_ms < 60_000:
            return
        get_audit = getattr(self._client, "get_audit", None)
        if not callable(get_audit):
            return
        self._last_audit_ms = now
        response = await get_audit()
        audit = response.get("response") if bool(response.get("ok", False)) else None
        LOG.warning(
            json.dumps(
                {
                    "event": "sm_async_op_failed_audit",
                    "model": queued.model,
                    "error": error,
                    "audit_ok": bool(response.get("ok", False)),
                    "healthy": (audit or {}).get("healthy"),
                    "issues": (audit or {}).get("issues"),
                },
                default=str,
                separators=(",", ":"),
            )
        )

    def _account(self, tracked: _TrackedOp, result: DispatchResult, *, timed_out: bool) -> None:
        action = tracked.queued.action
        direction = _action_direction(action) or "none"
        key = f"{tracked.action_kind}:{direction}:{tracked.queued.source_loop}"
        latency_ms = max(0, int(self._now_ms()) - tracked.dispatched_ms)
        drained_s = 0.0
        interrupted = 0
        interrupted_known = False
        for record in tracked.records.values():
            summary = record.get("summary") or {}
            for value in (summary.get("drained_s") or {}).values():
                if value is not None:
                    drained_s += float(value)
            if summary.get("interrupted") is not None:
                interrupted += int(summary["interrupted"])
                interrupted_known = True
        stats = self._op_stats.setdefault(
            key,
            {
                "count": 0,
                "ok": 0,
                "failed": 0,
                "timeout": 0,
                "latency_ms_sum": 0,
                "latency_ms_max": 0,
                "drained_s_sum": 0.0,
                "interrupted_sum": 0,
            },
        )
        stats["count"] += 1
        stats["ok" if result.ok else ("timeout" if timed_out else "failed")] += 1
        stats["latency_ms_sum"] += latency_ms
        stats["latency_ms_max"] = max(stats["latency_ms_max"], latency_ms)
        stats["drained_s_sum"] += drained_s
        stats["interrupted_sum"] += interrupted
        LOG.info(
            json.dumps(
                {
                    "event": "sm_async_op_done",
                    "model": tracked.queued.model,
                    "action_type": key,
                    "reason": getattr(action, "reason", None),
                    "operation_ids": list(tracked.op_ids),
                    "statuses": dict(tracked.status),
                    "ok": result.ok,
                    "error": result.error,
                    "timed_out": timed_out,
                    "latency_ms": latency_ms,
                    "drained_s": round(drained_s, 3),
                    "interrupted": interrupted if interrupted_known else None,
                    "polls": tracked.polls,
                },
                default=str,
                separators=(",", ":"),
            )
        )

    def _record_done(self, queued: QueuedAction, result: DispatchResult) -> None:
        direction = _action_direction(queued.action)
        if result.ok and direction is not None:
            self._last_done[queued.model] = (int(self._now_ms()), direction)

    def _has_pending_model(self, model: str) -> bool:
        return any(item.model == model for item in self._pending)

    def _remove_pending_fairness_for_model(self, model: str) -> tuple[tuple[str, SourceLoop], ...]:
        removed: list[tuple[str, SourceLoop]] = []
        retained: deque[QueuedAction] = deque()
        for item in self._pending:
            if item.model == model and item.source_loop == "fairness":
                removed.append((model, item.source_loop))
                continue
            retained.append(item)
        self._pending = retained
        return tuple(removed)


_RELEASED = frozenset({"slept", "already_sleeping"})


def _judge(tracked: _TrackedOp, *, timed_out: bool) -> tuple[bool, str | None, bool]:
    """(ok, error, roll back?) of a finished SM operation - per binding (review M2).

    A scale-DOWN releases capacity (donor sleep, SafeScale commit): it only succeeds
    when every binding it targeted is asleep. ``abandoned_*`` (the target changed or
    the binding was reclaimed by a newer operation) is not a success either - no
    cooldown, no follow-up - but also no rollback: someone else owns it now. A
    scale-UP succeeds when the SM reports succeeded / superseded."""
    if timed_out:
        return False, "op_timeout", True
    failed = [record for record in tracked.records.values() if record.get("status") not in _OK_TERMINAL]
    if tracked.errors or failed:
        reasons = list(tracked.errors) + [
            str(record.get("error") or record.get("status") or "failed") for record in failed
        ]
        return False, "; ".join(reasons) or "failed", True
    action = tracked.queued.action
    if not isinstance(action, ScaleAction) or action.delta >= 0:
        return True, None, False
    entries = [
        entry
        for record in tracked.records.values()
        for entry in (record.get("bindings") or [])
        if entry.get("action") == "sleep"
    ]
    not_released = sorted({str(entry.get("outcome")) for entry in entries if entry.get("outcome") not in _RELEASED})
    if not_released:
        abandoned_only = all(outcome.startswith("abandoned") for outcome in not_released)
        return False, "not_released:" + ",".join(not_released), not abandoned_only
    if not action.pods:
        planned = [
            record for record in tracked.records.values() if (record.get("plan") or {}).get("sleep")
        ]
        if planned and not any(entry.get("outcome") in _RELEASED for entry in entries):
            return False, "not_released:nothing_slept", False
    return True, None, False


def _is_oneshot(queued: QueuedAction) -> bool:
    return isinstance(queued.action, (HideAction, UnhideAction)) or queued.source_loop == "safescale"


def _retryable(error: str | None) -> bool:
    text = str(error or "")
    if text.startswith("HTTP "):
        return text[5:8] in ("409", "502", "503", "504")
    return bool(text)  # transport errors: timeouts, refused connections, ...


def _adopted_action(record: dict, meta: dict) -> ScaleAction:
    model = str(meta.get("model") or record.get("model"))
    if "delta" in meta:
        return ScaleAction(
            model,
            int(meta["delta"]),
            str(meta.get("reason") or "adopted"),
            str(meta.get("source_loop") or "rescue"),  # type: ignore[arg-type]
            pods=tuple(str(pod) for pod in meta.get("pods") or ()),
        )
    request = record.get("request") or {}
    if record.get("kind") == "binding_power":
        delta = 1 if request.get("awake") else -1
        return ScaleAction(model, delta, "adopted", "rescue", pods=(str(request.get("serve_id")),))
    direction = (record.get("plan") or {}).get("direction")
    delta = 1 if direction == "up" else -1 if direction == "down" else 0
    return ScaleAction(model, delta, "adopted", "rescue")


def _action_kind(action: Action) -> str:
    if isinstance(action, ScaleAction):
        return "scale"
    if isinstance(action, HideAction):
        return "hide"
    if isinstance(action, UnhideAction):
        return "unhide"
    if isinstance(action, DefragAction):
        return "defrag"
    return "unknown"


def _action_direction(action: Action) -> str | None:
    if isinstance(action, ScaleAction):
        return "up" if action.delta > 0 else "down" if action.delta < 0 else None
    if isinstance(action, HideAction):
        return "down"
    # UnhideAction (safescale rollback) restores capacity; it is not a scaling decision
    # and must not hold a CRITICAL model in cooldown (review P2-4).
    return None


def _queued_action(action: Action) -> QueuedAction:
    if isinstance(action, DefragAction):
        return QueuedAction(action=action, model=CLUSTER_MODEL, source_loop=action.source_loop)
    return QueuedAction(action=action, model=action.model, source_loop=action.source_loop)


def _dispatch_result(*, model: str, action_kind: str, response: dict) -> DispatchResult:
    ok = bool(response.get("ok", False))
    error = None if ok else str(response.get("error") or "dispatch_failed")
    return DispatchResult(model=model, action_kind=action_kind, ok=ok, error=error)
