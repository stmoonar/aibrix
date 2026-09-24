"""Flags off == main: the legacy entry points the flag-off dispatch lands on must keep
main's (merge base 2caa0514) decorators + body, AST-equal (names may differ: main's
public method became the ``_legacy`` twin). The main source is embedded verbatim so the
check needs neither git nor a fixed Python version for ast.dump."""
from __future__ import annotations

import ast
import inspect
import textwrap

import pytest

from tre_sm.api.v2 import ServiceManagerV2

MAIN_PUT_MODEL_TARGET = '''
@serialized_operation("put_model_target")
def put_model_target(self, model: str, *, wake_replicas: int) -> dict:
    spec = self._registry.model(model)
    if wake_replicas < 0:
        raise ValueError("wake_replicas must be non-negative")

    snapshot = self._store.load()
    self._ensure_target_within_cap(model, spec, wake_replicas, snapshot.bindings)
    model_bindings = [binding for binding in snapshot.bindings if binding.model == model]
    if self._runtime_ops is not None and wake_replicas > len(model_bindings) and not self._has_deployment_ops():
        raise ValueError("runtime create is not implemented for target growth beyond existing bindings")
    plan = self._plan_model_target(
        model=model,
        wake_replicas=wake_replicas,
        bindings=snapshot.bindings,
        tp_size=spec.tp_size,
    )
    self._set_model_desired_target(
        model=model,
        target_bindings=plan["target_bindings"],
        reason="model_target_request",
    )
    actions: list[dict] = []
    updated_by_serve = {binding.serve_id: binding for binding in snapshot.bindings}

    for binding in plan["sleep"]:
        self._apply_runtime_power_action(binding, action="sleep")
        updated_by_serve[binding.serve_id] = replace(
            binding, awake=False, hidden=False
        )
        actions.append({"action": "sleep", "serve_id": binding.serve_id})

    for binding in plan["wake"]:
        self._apply_runtime_power_action(binding, action="wake")
        updated_by_serve[binding.serve_id] = replace(
            binding, awake=True, hidden=False
        )
        actions.append({"action": "wake", "serve_id": binding.serve_id})

    for planned in plan["create"]:
        binding = planned
        if self._has_deployment_ops():
            binding = self._create_and_wake_runtime_binding(
                model, planned.slot
            )
        updated_by_serve[binding.serve_id] = binding
        actions.append(
            {
                "action": "create",
                "serve_id": binding.serve_id,
                "node": binding.slot.node,
                "gpu_ids": list(binding.slot.gpu_ids),
            }
        )

    version = snapshot.version
    if actions:
        updated = list(updated_by_serve.values())
        try:
            version = self._store.save(updated, expected_version=snapshot.version)
        except StateConflict:
            current = self._store.load()
            current_counts = self._model_counts(current.bindings).get(model, {"awake": 0})
            if current_counts["awake"] != wake_replicas:
                raise
            version = current.version

    return {
        "model": model,
        "wake_replicas": wake_replicas,
        "version": version,
        "actions": actions,
    }
'''

MAIN_PUT_BINDING_POWER = '''
@serialized_operation("put_binding_power")
def put_binding_power(self, serve_id: str, *, awake: bool) -> dict:
    if awake:
        snapshot = self._store.load()
        binding = next((item for item in snapshot.bindings if item.serve_id == serve_id), None)
        if binding is not None and not binding.awake:
            self._ensure_wake_within_cap(binding, snapshot.bindings)
    return self._put_binding_power_unlocked(serve_id, awake=awake)
'''

MAIN_PUT_BINDING_POWER_UNLOCKED = '''
def _put_binding_power_unlocked(self, serve_id: str, *, awake: bool) -> dict:
    snapshot = self._store.load()
    binding = next(
        (item for item in snapshot.bindings if item.serve_id == serve_id),
        None,
    )
    if binding is None:
        raise ValueError(f"unknown binding: {serve_id}")

    intent: dict[str, object] = {"power": "awake" if awake else "sleeping"}
    if not awake:
        intent["hidden"] = False
    self._update_desired(
        {binding.binding_id: intent},
        updated_by="service-manager-api",
        reason="binding_power_request",
    )

    actions: list[dict] = []
    version = snapshot.version
    updated_binding = binding
    if binding.awake != awake:
        action = "wake" if awake else "sleep"
        if awake:
            self._ensure_feasible_wake(binding, snapshot.bindings)
        self._apply_runtime_power_action(binding, action=action)
        updated_binding = replace(binding, awake=awake, hidden=False)
        updated = [
            updated_binding if item.serve_id == serve_id else item
            for item in snapshot.bindings
        ]
        version = self._store.save(updated, expected_version=snapshot.version)
        actions.append({"action": action, "serve_id": serve_id})

    return {
        "serve_id": serve_id,
        "awake": awake,
        "version": version,
        "actions": actions,
        "binding": self._binding_dict(updated_binding),
    }
'''


def _shape(func_source: str) -> str:
    """decorators + body (comments/line numbers/function name ignored)."""
    tree = ast.parse(textwrap.dedent(func_source))
    (node,) = tree.body
    assert isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    return ast.dump(ast.Module(body=list(node.decorator_list) + list(node.body), type_ignores=[]))


def _args(func_source: str) -> str:
    (node,) = ast.parse(textwrap.dedent(func_source)).body
    return ast.dump(node.args)


@pytest.mark.parametrize(
    "branch_method, main_source",
    [
        (ServiceManagerV2._put_model_target_legacy, MAIN_PUT_MODEL_TARGET),
        (ServiceManagerV2._put_binding_power_legacy, MAIN_PUT_BINDING_POWER),
        (ServiceManagerV2._put_binding_power_unlocked, MAIN_PUT_BINDING_POWER_UNLOCKED),
    ],
)
def test_flag_off_legacy_methods_are_ast_equal_to_main(branch_method, main_source):
    source = inspect.getsource(inspect.unwrap(branch_method))
    # getsource of the unwrapped function starts at the def; re-attach the decorator
    # text from the class source so decorators are compared too.
    full = inspect.getsource(ServiceManagerV2)
    name = branch_method.__name__
    tree = ast.parse(textwrap.dedent(full))
    (cls,) = tree.body
    (node,) = [item for item in cls.body if getattr(item, "name", None) == name]
    branch_shape = ast.dump(ast.Module(body=list(node.decorator_list) + list(node.body), type_ignores=[]))
    assert branch_shape == _shape(main_source), f"{name} diverged from main"
    assert ast.dump(node.args) == _args(main_source)
    assert source  # the method exists and is introspectable


def test_flag_off_dispatch_goes_to_the_legacy_twins():
    """With TRE_SM_HIDE_BEFORE_SLEEP off (and no drain_s), the public entry points only
    normalise the optional drain_s and call the legacy twin."""
    for public, legacy in (
        (ServiceManagerV2.put_model_target, "_put_model_target_legacy"),
        (ServiceManagerV2.put_binding_power, "_put_binding_power_legacy"),
    ):
        text = inspect.getsource(public)
        assert f"if not self._hide_enabled:\n            return self.{legacy}(" in text
