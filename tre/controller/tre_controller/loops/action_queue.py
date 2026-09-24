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
        blocked: list[tuple[str, str]] = []
        kept: list[Action] = []
        for action in actions:
            queued = _queued_action(action)
            if queued.source_loop == "rescue" and queued.model in self._ops:
                blocked.append((queued.model, "active_op"))
                continue
            kept.append(action)
        result = self._submit_legacy(kept)
        if result.accepted:
            items = list(self._pending)
            head, fresh = items[: -result.accepted], items[-result.accepted :]
            self._pending = deque(head + self._annotate_batch(fresh))
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
            # Already-dispatched operations are tracked to completion in every mode,
            # observe included: nothing the SM is executing is silently forgotten.
            results.extend(await self._poll_ops())
        held: deque[QueuedAction] = deque()
        while self._pending:
            queued = self._pending.popleft()
            kind = _action_kind(queued.action)
            if observe and queued.source_loop == "safescale":
                # One-shot (see _drain_once_legacy): hold, never drop.
                held.append(queued)
                continue
            if observe:
                results.append(DispatchResult(model=queued.model, action_kind=kind, ok=True, error="observe_skipped"))
                self._settle(queued, ok=False)
                self._inflight.discard(queued.model)
                continue
            if queued.depends_on is not None:
                outcome = self._seq_outcome.get(queued.depends_on)
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
            results.append(dispatched)
            self._record_done(queued, dispatched)
            self._settle(queued, ok=dispatched.ok)
            self._inflight.discard(queued.model)
        self._pending = held
        return tuple(results)

    async def _dispatch_v2(self, queued: QueuedAction) -> DispatchResult | None:
        action = queued.action
        if not isinstance(action, ScaleAction):
            # hide / unhide (fast) and defrag stay synchronous, exactly as in main.
            return await self._dispatch(action, queued.model)
        drain_s = self._call_drain_s(action)
        responses: list[dict] = []
        if action.delta != 0 and action.pods:
            for pod in action.pods:
                response = await self._client.set_binding_power_v2(
                    pod, awake=action.delta > 0, drain_s=drain_s, async_op=self._async_ops
                )
                responses.append(response)
                if not bool(response.get("ok", False)):
                    break
        else:
            responses.append(
                await self._client.scale_model_v2(
                    action.model, action.delta, drain_s=drain_s, async_op=self._async_ops
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
        self._ops[queued.model] = tracked
        return None

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
        failed = [record for record in tracked.records.values() if record.get("status") not in _OK_TERMINAL]
        ok = not timed_out and not tracked.errors and not failed
        error = None
        if not ok:
            reasons = list(tracked.errors) + [
                str(record.get("error") or record.get("status") or "failed") for record in failed
            ]
            error = "op_timeout" if timed_out else "; ".join(reasons) or "failed"
        result = DispatchResult(model=queued.model, action_kind=tracked.action_kind, ok=ok, error=error)
        # Review F4: the cooldown starts when the operation COMPLETED, not at dispatch.
        self._record_done(queued, result)
        self._settle(queued, ok=ok)
        self._inflight.discard(queued.model)
        self._account(tracked, result, timed_out=timed_out)
        if not ok:
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
