from __future__ import annotations

import asyncio
import json
import logging
import time
from collections import deque
from dataclasses import dataclass, field, replace
from typing import Awaitable, Callable, Iterable, Protocol

from tre_controller.planning.planner import (
    Action,
    DefragAction,
    HideAction,
    ScaleAction,
    SourceLoop,
    TransferAction,
    UnhideAction,
    fuse_transfers,
)

if False:  # annotations are strings (from __future__); avoids an import cycle
    from tre_controller.profiling import TickProfiler

CLUSTER_MODEL = "__cluster__"

#: Source loops whose actions are one-shot: the source never re-emits them (the
#: SafeScale state machine deletes a probe when it resolves), so a transient SM
#: failure must not drop them (review 2 P1-2 / P2-5). Actions of the other loops
#: are re-planned every tick and may be dropped.
ONE_SHOT_LOOPS = frozenset({"safescale"})

LOG = logging.getLogger(__name__)

QueueAction = Action | TransferAction

#: (node, gpu ids) of a binding by serve_id, or None when unknown.
SlotLookup = Callable[[str], "tuple[str, tuple[int, ...]] | None"]
#: None = the action is still wanted; else the reason it no longer is.
Revalidate = Callable[[QueueAction], "str | None"]


class ServiceManagerClient(Protocol):
    async def scale_model(self, model: str, delta: int) -> dict: ...

    async def set_routable(self, model: str, hidden_pods: tuple[str, ...]) -> dict: ...

    async def set_binding_power(self, serve_id: str, *, awake: bool) -> dict: ...

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
    #: and keep their submit order (models, pods, and GPUs when known).
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


class ActionQueue:
    """Dispatches controller actions to the service-manager.

    Actions run concurrently unless they share a resource (a model, a pod, a GPU):
    those run one at a time in submit order (review P1-3 / review 2 P1-1). A
    donor -> receiver transfer is one compound action touching both models: the
    receiver is woken only after the donor slept. One-shot actions (SafeScale
    commit / rollback) that fail retriably are retried with bounded backoff and
    re-validated before every retry; re-plannable actions are dropped on failure.
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
        slot_of: SlotLookup | None = None,
        sleep: Callable[[float], Awaitable[None]] | None = None,
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
        self._retry = retry or RetryPolicy()
        self._revalidate = revalidate
        self._slot_of = slot_of
        self._sleep = sleep or asyncio.sleep
        #: running dispatch task -> its queued action
        self._running: dict[asyncio.Future, QueuedAction] = {}
        self._closed = False
        self._stats: dict[str, int] = {
            "oneshot_retries_total": 0,
            "oneshot_abandoned_total": 0,
            "oneshot_not_wanted_total": 0,
            "transfer_receiver_dropped_total": 0,
        }

    # ------------------------------------------------------------------ submit
    def submit(self, actions: tuple[Action, ...] | list[Action]) -> SubmitResult:
        queued_actions = tuple(self._queued(action) for action in fuse_transfers(actions))
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
                for model in queued.models:
                    replaced.extend(self._remove_pending_fairness_for_model(model))
            elif any(
                model in self._inflight or self._has_pending_model(model)
                for model in queued.models
            ):
                dropped.append((queued.model, "inflight"))
                continue

            self._pending.append(queued)
            self._inflight.update(queued.models)
            accepted += 2 if isinstance(queued.action, TransferAction) else 1
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

    def stats(self) -> dict[str, int]:
        """Counters: one-shot retries / abandons, receiver wakes dropped after a
        failed donor sleep."""
        return dict(self._stats)

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
                self._reap()
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

    async def drain_once(self) -> tuple[DispatchResult, ...]:
        """Dispatch everything pending and wait until it is done - concurrently
        where resources allow, in order where they overlap (tests / offline
        integration)."""
        results: list[DispatchResult] = []
        self._dispatch_pending(results)
        while self._running:
            tasks = list(self._running)
            outcomes = await asyncio.gather(*tasks, return_exceptions=True)
            for outcome in outcomes:
                if isinstance(outcome, BaseException):
                    raise outcome
        return tuple(results)

    def _dispatch_pending(self, results: list[DispatchResult]) -> None:
        if self._closed:
            return
        busy: set[str] = set()
        for queued in self._running.values():
            busy |= queued.resources
        if self._is_observe():
            # Safescale resolution commands are one-shot (they are never re-emitted by
            # the SafeScaleStateMachine, which deletes the probe on resolve). If we are
            # paused in observe mode we must NOT drop them like idempotent planner
            # actions -- hold them in _pending (keeping the model inflight so no
            # conflicting action is queued) so they dispatch for real once mode
            # returns to non-observe.
            retained: deque[QueuedAction] = deque()
            for queued in self._pending:
                if queued.source_loop in ONE_SHOT_LOOPS or queued.resources & busy:
                    retained.append(queued)
                    continue
                results.extend(_observe_skipped(queued))
            self._pending = retained
            self._release_idle_models()
            return
        retained = deque()
        for queued in self._pending:
            if queued.resources & busy:
                # Blocked behind a running or an earlier queued action on a shared
                # resource: keep the order.
                busy |= queued.resources
                retained.append(queued)
                continue
            busy |= queued.resources
            task = asyncio.ensure_future(self._run_item(queued, results))
            self._running[task] = queued
        self._pending = retained

    async def _run_item(self, queued: QueuedAction, results: list[DispatchResult]) -> None:
        task = asyncio.current_task()
        try:
            await self._execute(queued, results)
        finally:
            self._running.pop(task, None)
            self._release_idle_models()
            if not self._closed:
                # Start whatever this action was blocking right away.
                self._dispatch_pending(results)

    async def _execute(self, queued: QueuedAction, results: list[DispatchResult]) -> None:
        if self._is_observe():
            if queued.source_loop in ONE_SHOT_LOOPS:
                self._pending.appendleft(queued)  # held until non-observe
            else:
                results.extend(_observe_skipped(queued))
            return
        action = queued.action
        if isinstance(action, TransferAction):
            for result in await self._execute_transfer(action):
                results.append(result)
            return
        while True:
            result = await self._timed_dispatch(action, queued.model)
            attempts = queued.failures + 1
            if (
                result.ok
                or queued.source_loop not in ONE_SHOT_LOOPS
                or not result.retriable
            ):
                self._record_done(queued.model, action, result)
                results.append(replace(result, attempts=attempts))
                return
            if attempts >= self._retry.max_attempts:
                self._stats["oneshot_abandoned_total"] += 1
                LOG.error(
                    "one-shot action abandoned after %d attempts: %s",
                    attempts,
                    json.dumps(
                        {"model": queued.model, "action": _action_kind(action),
                         "reason": getattr(action, "reason", None), "error": result.error},
                        sort_keys=True,
                    ),
                )
                results.append(
                    replace(
                        result,
                        error=f"abandoned after {attempts} attempts: {result.error}",
                        attempts=attempts,
                    )
                )
                return
            backoff = self._retry.backoff_s(attempts)
            self._stats["oneshot_retries_total"] += 1
            LOG.warning(
                "one-shot %s of %s failed retriably (%s); retry %d/%d in %.1fs",
                _action_kind(action), queued.model, result.error,
                attempts + 1, self._retry.max_attempts, backoff,
            )
            queued = replace(queued, failures=attempts)
            await self._sleep(backoff)
            if self._closed:
                return
            if self._is_observe():
                self._pending.appendleft(queued)  # paused while backing off: hold it
                return
            reason = self._still_wanted(action)
            if reason is not None:
                self._stats["oneshot_not_wanted_total"] += 1
                LOG.warning(
                    "one-shot %s of %s not retried: no longer wanted (%s)",
                    _action_kind(action), queued.model, reason,
                )
                results.append(
                    DispatchResult(
                        model=queued.model,
                        action_kind=_action_kind(action),
                        ok=False,
                        error=f"not_retried: {reason}",
                        attempts=attempts,
                    )
                )
                return

    async def _execute_transfer(self, action: TransferAction) -> list[DispatchResult]:
        donor, receiver = action.donor, action.receiver
        donor_result = await self._timed_dispatch(donor, donor.model)
        self._record_done(donor.model, donor, donor_result)
        if not donor_result.ok:
            # The receiver's wake needs what the donor's sleep frees: never run it.
            self._stats["transfer_receiver_dropped_total"] += 1
            reason = f"donor_sleep_failed: {donor_result.error}"
            LOG.warning(
                "transfer %s -> %s: donor sleep failed, receiver wake dropped (%s)",
                donor.model, receiver.model, donor_result.error,
            )
            return [
                donor_result,
                DispatchResult(model=receiver.model, action_kind="scale", ok=False, error=reason),
            ]
        receiver_result = await self._timed_dispatch(receiver, receiver.model)
        self._record_done(receiver.model, receiver, receiver_result)
        return [donor_result, receiver_result]

    async def _timed_dispatch(self, action: Action, model: str) -> DispatchResult:
        started_ns = time.perf_counter_ns()
        result = await self._dispatch(action, model)
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

    def _reap(self) -> None:
        """Log (and drop) dispatch tasks that died with an exception."""
        for task, queued in list(self._running.items()):
            if task.done():
                self._running.pop(task, None)
                if not task.cancelled() and task.exception() is not None:
                    LOG.error("action dispatch for %s failed: %r", queued.model, task.exception())
        self._release_idle_models()

    async def _dispatch(self, action: Action, model: str) -> DispatchResult:
        if isinstance(action, ScaleAction):
            if action.delta != 0 and action.pods:
                return await self._dispatch_binding_power(action)
            response = await self._client.scale_model(
                action.model, action.delta, **_sleep_kwargs(action)
            )
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
        # Stops at the first failure (a retry re-sends every pod: the SM answers a
        # binding already in the wanted power state with a no-op).
        for pod in action.pods:
            response = await self._client.set_binding_power(
                pod, awake=action.delta > 0, **_sleep_kwargs(action)
            )
            if not bool(response.get("ok", False)):
                return _dispatch_result(model=action.model, action_kind="scale", response=response)
        return DispatchResult(model=action.model, action_kind="scale", ok=True)

    def _record_done(self, model: str, action: Action, result: DispatchResult) -> None:
        direction = _action_direction(action)
        if result.ok and direction is not None:
            self._last_done[model] = (int(self._now_ms()), direction)

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
            return QueuedAction(action=action, model=CLUSTER_MODEL, source_loop=action.source_loop)
        if isinstance(action, TransferAction):
            models = tuple(dict.fromkeys((action.donor.model, action.receiver.model)))
            pods = action.donor.pods + action.receiver.pods
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

    def _resources(self, models: Iterable[str], pods: Iterable[str]) -> set[str]:
        """Models, pods and - when the cluster view knows the pod - its GPUs: two
        actions on one GPU (e.g. a donor sleep and another model's wake there) are
        serialized even when they come from different ticks."""
        keys = {f"model:{model}" for model in models}
        for pod in pods:
            keys.add(f"pod:{pod}")
            slot = None
            if self._slot_of is not None:
                try:
                    slot = self._slot_of(pod)
                except Exception:  # noqa: BLE001 - serialization hint only
                    slot = None
            if slot is not None:
                node, gpus = slot
                keys.update(f"gpu:{node}/{int(gpu)}" for gpu in gpus)
        return keys


def _observe_skipped(queued: QueuedAction) -> list[DispatchResult]:
    action = queued.action
    if isinstance(action, TransferAction):
        return [
            DispatchResult(model=part.model, action_kind="scale", ok=True, error="observe_skipped")
            for part in (action.donor, action.receiver)
        ]
    return [
        DispatchResult(
            model=queued.model,
            action_kind=_action_kind(action),
            ok=True,
            error="observe_skipped",
        )
    ]


def _sleep_kwargs(action: ScaleAction) -> dict:
    """SM sleep path + drain budget of a scale-down (plan 2026-09-27 D1).

    Only non-default values are sent: the SM default path is "scale_down".
    "*_immediate" planner reasons are the fast-loop donor paths ("urgent").
    """
    if action.delta >= 0:
        return {}
    path = action.sleep_path or (
        "urgent" if str(action.reason).endswith("_immediate") else "scale_down"
    )
    kwargs: dict = {}
    if path != "scale_down":
        kwargs["sleep_path"] = path
    if action.drain_budget_s is not None:
        kwargs["drain_budget_s"] = float(action.drain_budget_s)
    return kwargs


def _action_kind(action: QueueAction) -> str:
    if isinstance(action, (ScaleAction, TransferAction)):
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


def _queued_action(action: QueueAction) -> QueuedAction:
    """Queued form of one action without GPU resources (kept for callers/tests)."""
    if isinstance(action, DefragAction):
        return QueuedAction(action=action, model=CLUSTER_MODEL, source_loop=action.source_loop)
    return QueuedAction(action=action, model=action.model, source_loop=action.source_loop)


def _dispatch_result(*, model: str, action_kind: str, response: dict) -> DispatchResult:
    ok = bool(response.get("ok", False))
    error = None if ok else str(response.get("error") or "dispatch_failed")
    retriable = (not ok) and bool(response.get("retriable", False))
    return DispatchResult(model=model, action_kind=action_kind, ok=ok, error=error, retriable=retriable)


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
            if any(pod not in bindings or bindings[pod].hidden for pod in action.pods):
                return None
            return f"pods {list(action.pods)} already routable"
        if isinstance(action, HideAction) and action.pods:
            if any(pod not in bindings or not bindings[pod].hidden for pod in action.pods):
                return None
            return f"pods {list(action.pods)} already hidden"
        return None

    return still_wanted


def slot_lookup_from_cluster_view(get_view: Callable[[], object]) -> SlotLookup:
    def slot_of(serve_id: str):
        view = get_view()
        if view is None:
            return None
        for binding in getattr(view, "bindings", ()):
            if binding.serve_id == serve_id:
                return binding.slot.node, tuple(binding.slot.gpu_ids)
        return None

    return slot_of
