from __future__ import annotations

import asyncio
from collections import deque
from dataclasses import dataclass
from typing import Awaitable, Callable, Protocol

from tre_controller.planning.planner import Action, DefragAction, HideAction, ScaleAction, SourceLoop, UnhideAction

CLUSTER_MODEL = "__cluster__"


class ServiceManagerClient(Protocol):
    async def scale_model(self, model: str, delta: int) -> dict: ...

    async def set_routable(self, model: str, hidden_pods: tuple[str, ...]) -> dict: ...

    async def defrag(self, migrations: tuple) -> dict: ...


@dataclass(frozen=True)
class QueuedAction:
    action: Action
    model: str
    source_loop: SourceLoop


@dataclass(frozen=True)
class SubmitResult:
    accepted: int
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
    ) -> None:
        self._client = client
        self._pending: deque[QueuedAction] = deque()
        self._inflight: set[str] = set()
        # When this returns True the controller is paused: queued actions are drained
        # (inflight cleared, so the next tick can re-plan) but NEVER dispatched.
        self._is_observe = is_observe or (lambda: False)

    def submit(self, actions: tuple[Action, ...] | list[Action]) -> SubmitResult:
        accepted = 0
        dropped: list[tuple[str, str]] = []
        replaced: list[tuple[str, SourceLoop]] = []

        for action in actions:
            queued = _queued_action(action)
            if queued.source_loop == "rescue":
                removed = self._remove_pending_fairness_for_model(queued.model)
                replaced.extend(removed)
            elif queued.model in self._inflight or self._has_pending_model(queued.model):
                dropped.append((queued.model, "inflight"))
                continue

            self._pending.append(queued)
            self._inflight.add(queued.model)
            accepted += 1

        return SubmitResult(accepted=accepted, dropped=tuple(dropped), replaced=tuple(replaced))

    def pending_actions(self) -> tuple[QueuedAction, ...]:
        return tuple(self._pending)

    def inflight_models(self) -> set[str]:
        return set(self._inflight)

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
        observe = self._is_observe()
        results: list[DispatchResult] = []
        # Safescale resolution commands are one-shot (they are never re-emitted by the
        # SafeScaleStateMachine, which deletes the probe on resolve). If we are paused in
        # observe mode we must NOT drop them like idempotent planner actions -- hold them
        # in _pending (keeping the model inflight so no conflicting action is queued) so
        # they dispatch for real once mode returns to non-observe.
        held: deque[QueuedAction] = deque()
        while self._pending:
            queued = self._pending.popleft()
            if observe and queued.source_loop == "safescale":
                held.append(queued)
                continue
            if observe:
                results.append(DispatchResult(model=queued.model, action_kind=_action_kind(queued.action),
                                              ok=True, error="observe_skipped"))
            else:
                results.append(await self._dispatch(queued.action, queued.model))
            self._inflight.discard(queued.model)
        self._pending = held
        return tuple(results)

    async def _dispatch(self, action: Action, model: str) -> DispatchResult:
        if isinstance(action, ScaleAction):
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


def _queued_action(action: Action) -> QueuedAction:
    if isinstance(action, DefragAction):
        return QueuedAction(action=action, model=CLUSTER_MODEL, source_loop=action.source_loop)
    return QueuedAction(action=action, model=action.model, source_loop=action.source_loop)


def _dispatch_result(*, model: str, action_kind: str, response: dict) -> DispatchResult:
    ok = bool(response.get("ok", False))
    error = None if ok else str(response.get("error") or "dispatch_failed")
    return DispatchResult(model=model, action_kind=action_kind, ok=ok, error=error)
