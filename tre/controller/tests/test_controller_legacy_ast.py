"""Flags off == main (TRE_SM_ASYNC / TRE_SM_CALL_DRAIN): the ActionQueue and SM client
code paths the flag-off controller runs keep main's (merge base 2caa0514) AST. main's
source is embedded so the check needs neither git nor a fixed Python version."""
from __future__ import annotations

import ast
import inspect
import textwrap

import pytest

from tre_controller import sm_client
from tre_controller.loops import action_queue

MAIN_SUBMIT = '''
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
'''

MAIN_DRAIN_ONCE = '''
async def drain_once(self) -> tuple[DispatchResult, ...]:
    observe = self._is_observe()
    results: list[DispatchResult] = []
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
'''

MAIN_DISPATCH = '''
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
'''

MAIN_DISPATCH_BINDING_POWER = '''
async def _dispatch_binding_power(self, action: ScaleAction) -> DispatchResult:
    for pod in action.pods:
        response = await self._client.set_binding_power(pod, awake=action.delta > 0)
        if not bool(response.get("ok", False)):
            return _dispatch_result(model=action.model, action_kind="scale", response=response)
    return DispatchResult(model=action.model, action_kind="scale", ok=True)
'''

MAIN_RECORD_DONE = '''
def _record_done(self, queued: QueuedAction, result: DispatchResult) -> None:
    direction = _action_direction(queued.action)
    if result.ok and direction is not None:
        self._last_done[queued.model] = (int(self._now_ms()), direction)
'''

MAIN_SCALE_MODEL = '''
async def scale_model(self, model: str, delta: int) -> dict:
    try:
        state = await self.get_state()
        counts = state.get("models", {}).get(model, {})
        current = int(counts.get("awake", 0))
        bound = int(counts.get("bound", 0))
        serving_floor = 1 if bound > 0 and current > 0 and int(delta) < 0 else 0
        target = max(serving_floor, current + int(delta))
        response = await self._request(
            "PUT", f"/v2/models/{model}/target", json={"wake_replicas": target}, timeout_s=self._slow_timeout_s
        )
        return {"ok": True, "response": response}
    except ServiceManagerError as exc:
        return {"ok": False, "error": str(exc)}
'''

MAIN_SET_BINDING_POWER = '''
async def set_binding_power(self, serve_id: str, *, awake: bool) -> dict:
    try:
        response = await self._request(
            "PUT",
            f"/v2/bindings/{serve_id}/power",
            json={"awake": bool(awake)},
            timeout_s=self._slow_timeout_s,
        )
        return {"ok": True, "response": response}
    except ServiceManagerError as exc:
        return {"ok": False, "error": str(exc)}
'''


def _node(source: str) -> ast.AST:
    (node,) = ast.parse(textwrap.dedent(source)).body
    return node


def _shape(node) -> str:
    return ast.dump(ast.Module(body=list(node.decorator_list) + list(node.body), type_ignores=[]))


def _branch_node(cls, name: str):
    tree = ast.parse(textwrap.dedent(inspect.getsource(cls)))
    (class_node,) = tree.body
    (node,) = [item for item in class_node.body if getattr(item, "name", None) == name]
    return node


@pytest.mark.parametrize(
    "cls, branch_name, main_source",
    [
        (action_queue.ActionQueue, "_submit_legacy", MAIN_SUBMIT),
        (action_queue.ActionQueue, "_drain_once_legacy", MAIN_DRAIN_ONCE),
        (action_queue.ActionQueue, "_dispatch", MAIN_DISPATCH),
        (action_queue.ActionQueue, "_dispatch_binding_power", MAIN_DISPATCH_BINDING_POWER),
        (action_queue.ActionQueue, "_record_done", MAIN_RECORD_DONE),
        (sm_client.ServiceManagerClient, "scale_model", MAIN_SCALE_MODEL),
        (sm_client.ServiceManagerClient, "set_binding_power", MAIN_SET_BINDING_POWER),
    ],
)
def test_flag_off_paths_are_ast_equal_to_main(cls, branch_name, main_source):
    branch = _branch_node(cls, branch_name)
    main = _node(main_source)
    assert _shape(branch) == _shape(main), f"{cls.__name__}.{branch_name} diverged from main"
    assert ast.dump(branch.args) == ast.dump(main.args)


def test_flag_off_entry_points_dispatch_to_the_legacy_twins():
    submit = inspect.getsource(action_queue.ActionQueue.submit)
    assert "if self._async_ops:" in submit and "return self._submit_legacy(actions)" in submit
    drain = inspect.getsource(action_queue.ActionQueue.drain_once)
    assert "if self._async_ops or self._call_drain:" in drain
    assert "return await self._drain_once_legacy()" in drain
