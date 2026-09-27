from __future__ import annotations

import asyncio
import logging
import time
from collections import deque
from dataclasses import dataclass
from typing import Awaitable, Callable, Protocol

from tre_controller.planning.planner import Action, DefragAction, HideAction, ScaleAction, SourceLoop, UnhideAction

if False:  # annotations are strings (from __future__); avoids an import cycle
    from tre_controller.profiling import TickProfiler

CLUSTER_MODEL = "__cluster__"

LOG = logging.getLogger(__name__)


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
        # One dispatch worker per model (CLUSTER_MODEL for defrag): models proceed
        # concurrently, a model's own actions stay serialized (review P1-3).
        self._workers: dict[str, asyncio.Future] = {}

    def submit(self, actions: tuple[Action, ...] | list[Action]) -> SubmitResult:
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
        """Start per-model dispatch workers without waiting for them: a slow SM call
        for one model (a scale-down that drains for minutes) never delays another
        model's actions. Within a model, actions stay serialized and in order."""
        while True:
            self._dispatch_pending([])
            self._reap_workers()
            await sleep(poll_interval_s)

    async def drain_once(self) -> tuple[DispatchResult, ...]:
        """Dispatch everything pending - concurrently across models, in order within
        a model - and wait until it is done (tests / offline integration)."""
        results: list[DispatchResult] = []
        self._dispatch_pending(results)
        tasks = list(self._workers.values())
        if tasks:
            outcomes = await asyncio.gather(*tasks, return_exceptions=True)
            self._reap_workers()
            for outcome in outcomes:
                if isinstance(outcome, BaseException):
                    raise outcome
        return tuple(results)

    def _dispatch_pending(self, results: list[DispatchResult]) -> None:
        observe = self._is_observe()
        if observe:
            # Safescale resolution commands are one-shot (they are never re-emitted by
            # the SafeScaleStateMachine, which deletes the probe on resolve). If we are
            # paused in observe mode we must NOT drop them like idempotent planner
            # actions -- hold them in _pending (keeping the model inflight so no
            # conflicting action is queued) so they dispatch for real once mode
            # returns to non-observe.
            retained: deque[QueuedAction] = deque()
            for queued in self._pending:
                if queued.source_loop == "safescale" or queued.model in self._workers:
                    retained.append(queued)
                    continue
                results.append(
                    DispatchResult(
                        model=queued.model,
                        action_kind=_action_kind(queued.action),
                        ok=True,
                        error="observe_skipped",
                    )
                )
            self._pending = retained
            self._release_idle_models()
            return
        for model in dict.fromkeys(item.model for item in self._pending):
            if model in self._workers:
                continue
            self._workers[model] = asyncio.ensure_future(self._model_worker(model, results))

    async def _model_worker(self, model: str, results: list[DispatchResult]) -> None:
        try:
            while True:
                queued = self._next_pending(model)
                if queued is None:
                    return
                if self._is_observe():
                    if queued.source_loop == "safescale":
                        self._pending.appendleft(queued)  # held until non-observe
                        return
                    results.append(
                        DispatchResult(
                            model=queued.model,
                            action_kind=_action_kind(queued.action),
                            ok=True,
                            error="observe_skipped",
                        )
                    )
                    continue
                started_ns = time.perf_counter_ns()
                result = await self._dispatch(queued.action, queued.model)
                if self._prof is not None:
                    self._prof.record(
                        {
                            "kind": "dispatch",
                            "ts_ms": self._prof.now_ms(),
                            "n_actions": 1,
                            "http_ns": time.perf_counter_ns() - started_ns,
                        }
                    )
                self._record_done(queued, result)
                results.append(result)
        finally:
            self._workers.pop(model, None)
            if not self._has_pending_model(model):
                self._inflight.discard(model)

    def _next_pending(self, model: str) -> QueuedAction | None:
        for index, item in enumerate(self._pending):
            if item.model == model:
                del self._pending[index]
                return item
        return None

    def _release_idle_models(self) -> None:
        for model in list(self._inflight):
            if model not in self._workers and not self._has_pending_model(model):
                self._inflight.discard(model)

    def _reap_workers(self) -> None:
        """Log (and drop) workers that died with an exception."""
        for model, task in list(self._workers.items()):
            if task.done():
                self._workers.pop(model, None)
                if not task.cancelled() and task.exception() is not None:
                    LOG.error("action dispatch for %s failed: %r", model, task.exception())

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
        # Stops at the first failure.
        for pod in action.pods:
            response = await self._client.set_binding_power(
                pod, awake=action.delta > 0, **_sleep_kwargs(action)
            )
            if not bool(response.get("ok", False)):
                return _dispatch_result(model=action.model, action_kind="scale", response=response)
        return DispatchResult(model=action.model, action_kind="scale", ok=True)

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
