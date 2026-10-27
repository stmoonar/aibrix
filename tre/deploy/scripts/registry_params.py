"""Registry-parameter edit surface: whitelist, bounds, cross-field checks, safe render.

Used by deploy/scripts/release/params_set.py (moved here from the removed console,
tre/ui/tre_ui/params.py, 2026-10-20). Only per-model TRS/SLO/alt-threshold/replica params
(the surface R3 recalibration writes) are editable. Everything else -- names, weights,
images, tp_size, and the window-coupled ema_tau_ms -- is locked. Every candidate edit is
re-parsed with the controller's OWN load_registry before it is allowed to reach the
ConfigMap, so a bad edit is rejected here rather than crashlooping the controller on
restart.
"""
from __future__ import annotations

import difflib
import hashlib
import math
import tempfile
from pathlib import Path
from typing import Any, Callable

import yaml

from tre_common.alt_signals import ALT_SIGNAL_DEFINITIONS, Z_MAX
from tre_common.registry import EXPECTED_SIGNAL_DIRECTIONS, load_registry
from tre_common.tss import NUMERATOR_COMPLETION_HISTOGRAMS, NUMERATOR_STEP_COUNTERS

# field -> (min, max, kind, min_exclusive, max_exclusive)
_TRS_BOUNDS: dict[str, tuple] = {
    "theta_m": (0, 20000, "float", True, False),
    "tau_crit": (0, 2, "float", True, False),
    "tau_low": (0, 4, "float", True, False),
    "tau_high": (0, 8, "float", True, False),
    "w_p": (0, 10, "float", False, False),
    "w_d": (0, 10, "float", False, False),
    "lambda_wait": (0, 20, "float", False, False),
    "qmin": (0, 16, "float", False, False),
    "qsat": (0, 64, "float", True, False),
    "ema_alpha": (0, 1, "float", True, True),
    "epsat": (0, 0.5, "float", True, True),
    "hsat": (1, 20, "int", False, False),
    # 2026-10-28: what theta was fitted on (tre_common.tss.theta_basis_errors). Published
    # with theta in the same --apply; the controller refuses to start on a mismatch.
    "theta_window_ms": (1000, 120000, "int", False, False),
    "numerator": ((NUMERATOR_STEP_COUNTERS, NUMERATOR_COMPLETION_HISTOGRAMS), None, "enum", False, False),
}
_ALT_THRESHOLD_BOUNDS = (0, 1_000_000_000, "float", True, False)
_ALT_DELTA_BOUNDS = {
    "delta_crit": (0.0, 1.0, "float", False, True),
    # z is capped at Z_MAX (tre_common.alt_signals): a band above Z_MAX - 1 never fires.
    # The alt fit searches the same range (tre_calibration.alt_signals ALT_DELTA_HIGH_GRID).
    "delta_high": (0.0, Z_MAX - 1.0, "float", False, False),
}
#: alt_thresholds.<signal>.basis.<field> (tre_common.alt_signals.alt_basis_errors): what
#: theta was fitted on; published with theta, checked by the controller at start.
_ALT_BASIS_WINDOW_BOUNDS = (1000, 120000, "int", False, False)
#: basis.ema_tau_ms: the EMA tau the alt fit smoothed the value with (0 = raw; review L5).
_ALT_BASIS_TAU_BOUNDS = (0.0, 600000.0, "float", False, False)

_SLO_BOUNDS: dict[str, tuple] = {
    "ttft_p95_ms": (50, 5000, "float", False, False),
    "tpot_p95_ms": (10, 500, "float", False, False),
    "e2e_p95_ms": (1000, 120000, "float", False, False),
    # Idle-TTFT intercept / slope (c, b): live with safescale.slo_mode labels (TTFT threshold
    # max(500, k*(c + b*L))), so they are published together with theta in one --apply.
    "ttft_idle_c_ms": (0, 2000, "float", True, False),
    "ttft_idle_b_ms_per_token": (0, 1, "float", False, False),
}
_TOP_BOUNDS: dict[str, tuple] = {
    "min_replicas": (0, 4, "int", False, False),
    # Scaling cap (v1/paper alignment A1); max_replicas is the GPU layout size (locked).
    "max_awake_replicas": (1, 8, "int", False, False),
}
_LOCKED = {
    "name": "identity; change via git redeploy",
    "weights_path": "placement/structural; git redeploy",
    "vllm_image": "placement/structural; git redeploy",
    "tp_size": "placement/structural; git redeploy",
    "max_replicas": "GPU layout (binding count) = structural; release registry + "
                    "merge_live_registry.py (or release/registry_set.py) and the rendered manifests",
    "trs.ema_tau_ms": "window-coupled; frozen with W",
}


class ParamValidationError(Exception):
    def __init__(self, errors: list[dict]) -> None:
        super().__init__("param validation failed")
        self.errors = errors


def load_models(registry_yaml: str) -> list[dict]:
    data = yaml.safe_load(registry_yaml) or {}
    return list(data.get("models") or [])


def build_view(registry_yaml: str) -> dict[str, Any]:
    """Present each model's editable fields (value + bounds) and locked fields (value + reason)."""
    out: dict[str, Any] = {}
    for model in load_models(registry_yaml):
        name = model.get("name")
        trs = model.get("trs") or {}
        slo = model.get("slo") or {}
        alt_thresholds = model.get("alt_thresholds") or {}
        editable = {}
        for field, spec in _TRS_BOUNDS.items():
            editable[f"trs.{field}"] = _field_view(trs.get(field), spec)
        for signal, threshold in sorted(alt_thresholds.items()):
            editable[f"alt_thresholds.{signal}.theta"] = _field_view(
                threshold.get("theta"), _ALT_THRESHOLD_BOUNDS
            )
            editable[f"alt_thresholds.{signal}.direction"] = {
                "value": threshold.get("direction"),
                "type": "enum",
                "choices": [EXPECTED_SIGNAL_DIRECTIONS.get(signal)],
            }
            for field, spec in _ALT_DELTA_BOUNDS.items():
                editable[f"alt_thresholds.{signal}.{field}"] = _field_view(threshold.get(field), spec)
            basis = threshold.get("basis") or {}
            editable[f"alt_thresholds.{signal}.basis.definition"] = {
                "value": basis.get("definition"),
                "type": "enum",
                "choices": [ALT_SIGNAL_DEFINITIONS.get(signal)],
            }
            editable[f"alt_thresholds.{signal}.basis.window_ms"] = _field_view(
                basis.get("window_ms"), _ALT_BASIS_WINDOW_BOUNDS
            )
            editable[f"alt_thresholds.{signal}.basis.ema_tau_ms"] = _field_view(
                basis.get("ema_tau_ms"), _ALT_BASIS_TAU_BOUNDS
            )
        for field, spec in _SLO_BOUNDS.items():
            editable[f"slo.{field}"] = _field_view(slo.get(field), spec)
        for field, spec in _TOP_BOUNDS.items():
            editable[field] = _field_view(model.get(field), spec)
        locked = {
            "name": {"value": name, "reason": _LOCKED["name"]},
            "weights_path": {"value": model.get("weights_path"), "reason": _LOCKED["weights_path"]},
            "vllm_image": {"value": model.get("vllm_image"), "reason": _LOCKED["vllm_image"]},
            "tp_size": {"value": model.get("tp_size"), "reason": _LOCKED["tp_size"]},
            "max_replicas": {"value": model.get("max_replicas"), "reason": _LOCKED["max_replicas"]},
            "trs.ema_tau_ms": {"value": trs.get("ema_tau_ms"), "reason": _LOCKED["trs.ema_tau_ms"]},
        }
        out[name] = {
            "editable": editable,
            "locked": locked,
            "constraints": ["tau_crit <= tau_low < tau_high", "qmin <= qsat", "min_replicas <= max_replicas", "min_replicas <= max_awake_replicas <= max_replicas", "w_p and w_d not both 0"],
        }
    return out


def _field_view(value: Any, spec: tuple) -> dict[str, Any]:
    lo, hi, kind, lo_ex, hi_ex = spec
    if kind == "enum":
        return {"value": value, "type": "enum", "choices": list(lo)}
    return {"value": value, "min": lo, "max": hi, "type": kind, "min_exclusive": lo_ex, "max_exclusive": hi_ex}


def apply_and_validate(
    registry_yaml: str,
    edits: dict[str, Any],
    *,
    loader: Callable[[str], Any] = load_registry,
) -> str:
    """Return a new registry.yaml with `edits` applied, or raise ParamValidationError.

    edits: {model_name: {"trs": {f: v}, "slo": {f: v}, "alt_thresholds": {signal: {f: v}},
            "min_replicas": v, "max_awake_replicas": v}}
    """
    data = yaml.safe_load(registry_yaml) or {}
    models = {m.get("name"): m for m in (data.get("models") or [])}
    errors: list[dict] = []

    for model_name, changes in edits.items():
        model = models.get(model_name)
        if model is None:
            errors.append({"model": model_name, "field": "", "error": "unknown_model"})
            continue
        _apply_section(model.setdefault("trs", {}), changes.get("trs", {}), _TRS_BOUNDS, model_name, "trs.", errors)
        _apply_alt_thresholds(
            model.setdefault("alt_thresholds", {}),
            changes.get("alt_thresholds", {}),
            model_name,
            errors,
        )
        _apply_section(model.setdefault("slo", {}), changes.get("slo", {}), _SLO_BOUNDS, model_name, "slo.", errors)
        top = {k: v for k, v in changes.items() if k in _TOP_BOUNDS}
        _apply_section(model, top, _TOP_BOUNDS, model_name, "", errors)
        _reject_locked(changes, model_name, errors)

    if errors:
        raise ParamValidationError(errors)

    for model_name in edits:
        _cross_field(models[model_name], model_name, errors)
    if errors:
        raise ParamValidationError(errors)

    rendered = render(data)
    try:
        with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False, encoding="utf-8") as handle:
            handle.write(rendered)
            tmp = handle.name
        loader(tmp)  # controller's own parser is the final gate
    except Exception as exc:  # noqa: BLE001
        raise ParamValidationError([{"model": "", "field": "", "error": f"parse_failed: {exc}"}]) from exc
    finally:
        try:
            Path(tmp).unlink()
        except (OSError, NameError):
            pass
    return rendered


def _apply_section(target: dict, changes: dict, bounds: dict, model: str, prefix: str, errors: list) -> None:
    for field, raw in changes.items():
        key = f"{prefix}{field}"
        if key in _LOCKED:
            errors.append({"model": model, "field": key, "error": "locked", "reason": _LOCKED[key]})
            continue
        spec = bounds.get(field)
        if spec is None:
            errors.append({"model": model, "field": key, "error": "unknown_field"})
            continue
        coerced = _coerce(raw, spec, model, key, errors)
        if coerced is not None:
            target[field] = coerced


def _apply_alt_thresholds(target: dict, changes: dict, model: str, errors: list) -> None:
    if not isinstance(changes, dict):
        errors.append({"model": model, "field": "alt_thresholds", "error": "not_an_object"})
        return
    for signal, raw in changes.items():
        key = f"alt_thresholds.{signal}"
        expected_direction = EXPECTED_SIGNAL_DIRECTIONS.get(signal)
        if expected_direction is None:
            errors.append({"model": model, "field": key, "error": "unknown_signal"})
            continue
        if not isinstance(raw, dict):
            errors.append({"model": model, "field": key, "error": "not_an_object"})
            continue
        unknown = sorted(set(raw) - {"theta", "direction", "basis", *_ALT_DELTA_BOUNDS})
        for field in unknown:
            errors.append({"model": model, "field": f"{key}.{field}", "error": "unknown_field"})
        threshold = target.setdefault(signal, {})
        if "theta" in raw:
            theta = _coerce(raw["theta"], _ALT_THRESHOLD_BOUNDS, model, f"{key}.theta", errors)
            if theta is not None:
                threshold["theta"] = theta
        for field, spec in _ALT_DELTA_BOUNDS.items():
            if field in raw:
                value = _coerce(raw[field], spec, model, f"{key}.{field}", errors)
                if value is not None:
                    threshold[field] = value
        if "basis" in raw:
            _apply_alt_basis(threshold, raw["basis"], signal, model, errors)
        if "direction" in raw:
            direction = str(raw["direction"])
            if direction != expected_direction:
                errors.append(
                    {
                        "model": model,
                        "field": f"{key}.direction",
                        "error": "invalid_direction",
                        "expected": expected_direction,
                    }
                )
            else:
                threshold["direction"] = direction


def _apply_alt_basis(threshold: dict, raw: Any, signal: str, model: str, errors: list) -> None:
    key = f"alt_thresholds.{signal}.basis"
    if not isinstance(raw, dict):
        errors.append({"model": model, "field": key, "error": "not_an_object"})
        return
    for field in sorted(set(raw) - {"definition", "window_ms", "ema_tau_ms"}):
        errors.append({"model": model, "field": f"{key}.{field}", "error": "unknown_field"})
    basis = threshold.setdefault("basis", {})
    if "definition" in raw:
        expected = ALT_SIGNAL_DEFINITIONS.get(signal)
        if raw["definition"] != expected:
            errors.append({"model": model, "field": f"{key}.definition", "error": "invalid_definition",
                           "expected": expected})
        else:
            basis["definition"] = expected
    if "window_ms" in raw:
        value = _coerce(raw["window_ms"], _ALT_BASIS_WINDOW_BOUNDS, model, f"{key}.window_ms", errors)
        if value is not None:
            basis["window_ms"] = value
    if "ema_tau_ms" in raw:
        value = _coerce(raw["ema_tau_ms"], _ALT_BASIS_TAU_BOUNDS, model, f"{key}.ema_tau_ms", errors)
        if value is not None:
            basis["ema_tau_ms"] = value


def _coerce(raw: Any, spec: tuple, model: str, key: str, errors: list) -> Any:
    lo, hi, kind, lo_ex, hi_ex = spec
    if kind == "enum":
        if raw in lo:
            return raw
        errors.append({"model": model, "field": key, "error": "not_a_choice", "choices": list(lo)})
        return None
    if isinstance(raw, bool):  # YAML true/false would coerce to 1/0
        errors.append({"model": model, "field": key, "error": f"not_a_{kind}"})
        return None
    if kind == "int" and isinstance(raw, float) and not raw.is_integer():
        errors.append({"model": model, "field": key, "error": "not_a_int"})  # int() would truncate
        return None
    try:
        value = int(raw) if kind == "int" else float(raw)
    except (TypeError, ValueError):
        errors.append({"model": model, "field": key, "error": f"not_a_{kind}"})
        return None
    if not math.isfinite(value):
        errors.append({"model": model, "field": key, "error": "not_finite"})
        return None
    if (value <= lo if lo_ex else value < lo) or (value >= hi if hi_ex else value > hi):
        errors.append({"model": model, "field": key, "error": "out_of_bounds", "min": lo, "max": hi})
        return None
    return value


def _reject_locked(changes: dict, model: str, errors: list) -> None:
    for locked_key in ("name", "weights_path", "vllm_image", "tp_size", "max_replicas"):
        if locked_key in changes:
            errors.append({"model": model, "field": locked_key, "error": "locked", "reason": _LOCKED[locked_key]})
    if "ema_tau_ms" in changes.get("trs", {}):
        errors.append({"model": model, "field": "trs.ema_tau_ms", "error": "locked", "reason": _LOCKED["trs.ema_tau_ms"]})


def _cross_field(model: dict, name: str, errors: list) -> None:
    trs = model.get("trs", {})

    def add(field, msg):
        errors.append({"model": name, "field": field, "error": "constraint", "detail": msg})

    tc, tl, th = trs.get("tau_crit"), trs.get("tau_low"), trs.get("tau_high")
    if None not in (tc, tl) and tc > tl:
        add("trs.tau_crit", "tau_crit must be <= tau_low")
    if None not in (tl, th) and tl >= th:
        add("trs.tau_low", "tau_low must be < tau_high")
    qmin, qsat = trs.get("qmin"), trs.get("qsat")
    if None not in (qmin, qsat) and qmin > qsat:
        add("trs.qmin", "qmin must be <= qsat")
    mn, mx = model.get("min_replicas"), model.get("max_replicas")
    if None not in (mn, mx) and mn > mx:
        add("min_replicas", "min_replicas must be <= max_replicas")
    awake_cap = model.get("max_awake_replicas")
    if awake_cap is not None and None not in (mn, mx) and not (mn <= awake_cap <= mx):
        add("max_awake_replicas", "max_awake_replicas must be within [min_replicas, max_replicas]")
    if trs.get("w_p") == 0 and trs.get("w_d") == 0:
        add("trs.w_p", "w_p and w_d must not both be 0")


def render(data: dict) -> str:
    return yaml.safe_dump(data, sort_keys=False, default_flow_style=False, allow_unicode=True)


# ---- command-line surface (params_set.py) ----

_SECTIONS = ("trs", "slo")


def parse_sets(sets: list[str], model_names: list[str]) -> dict[str, Any]:
    """``["<model>.<path>=<value>", ...]`` -> the ``edits`` mapping of apply_and_validate.

    <path> is ``trs.<f>``, ``slo.<f>``, ``alt_thresholds.<signal>.<f>``,
    ``alt_thresholds.<signal>.basis.{definition,window_ms,ema_tau_ms}`` or a top-level
    replica bound. The model is matched by the longest registry name that prefixes the
    key (names may contain dots). VALUE is parsed as YAML. Raises ParamValidationError
    for a malformed item, an unknown model or a path outside the editable shapes (locked
    keys pass through so apply_and_validate reports them as locked)."""
    edits: dict[str, Any] = {}
    errors: list[dict] = []
    for item in sets:
        key, sep, raw = item.partition("=")
        if not sep:
            errors.append({"model": "", "field": item, "error": "expected_MODEL.PATH=VALUE"})
            continue
        model = max((m for m in model_names if key.startswith(f"{m}.")), key=len, default=None)
        if model is None:
            errors.append({"model": key.split(".", 1)[0], "field": key, "error": "unknown_model"})
            continue
        path = key[len(model) + 1:].split(".")
        value = yaml.safe_load(raw)
        changes = edits.setdefault(model, {})
        if len(path) == 2 and path[0] in _SECTIONS:
            changes.setdefault(path[0], {})[path[1]] = value
        elif len(path) == 3 and path[0] == "alt_thresholds":
            changes.setdefault("alt_thresholds", {}).setdefault(path[1], {})[path[2]] = value
        elif len(path) == 4 and path[0] == "alt_thresholds" and path[2] == "basis":
            (changes.setdefault("alt_thresholds", {}).setdefault(path[1], {})
             .setdefault("basis", {}))[path[3]] = value
        elif len(path) == 1 and (path[0] in _TOP_BOUNDS or path[0] in _LOCKED):
            changes[path[0]] = value
        else:
            errors.append({"model": model, "field": ".".join(path), "error": "unknown_field"})
    if errors:
        raise ParamValidationError(errors)
    return edits


def params_hash(registry_yaml: str) -> str:
    """The ``tre.dev/params-hash`` annotation of a restart-to-apply (same as the console's)."""
    return hashlib.sha256((registry_yaml or "").encode("utf-8")).hexdigest()[:16]


def changed_fields(old_yaml: str, new_yaml: str) -> list[tuple[str, str, Any, Any]]:
    """(model, dotted field, old, new) for every per-model leaf that differs."""
    old = {m.get("name"): m for m in load_models(old_yaml)}
    out: list[tuple[str, str, Any, Any]] = []
    for model in load_models(new_yaml):
        name = model.get("name")
        before, after = _leaves(old.get(name) or {}), _leaves(model)
        for field in sorted(set(before) | set(after)):
            if before.get(field) != after.get(field):
                out.append((name, field, before.get(field), after.get(field)))
    return out


def _leaves(node: Any, prefix: str = "") -> dict[str, Any]:
    if not isinstance(node, dict):
        return {prefix: node}
    out: dict[str, Any] = {}
    for key, value in node.items():
        out.update(_leaves(value, f"{prefix}{key}." if isinstance(value, dict) else f"{prefix}{key}"))
    return out


def unified_diff(old_yaml: str, new_yaml: str) -> str:
    """Diff of the two registries rendered the same way (comments / key quoting of the
    live text do not show up; only value changes do)."""
    a = render(yaml.safe_load(old_yaml) or {}).splitlines(keepends=True)
    b = render(yaml.safe_load(new_yaml) or {}).splitlines(keepends=True)
    return "".join(difflib.unified_diff(a, b, "live", "edited"))


def new_validation_errors(old_yaml: str, new_yaml: str, *, loader: Callable[[str], Any] = load_registry) -> list[str]:
    """Registry.validate() errors of the edited registry that the live one does not have
    (a pre-existing error must not block an unrelated edit)."""
    before = set(_validate(old_yaml, loader))
    return [e for e in _validate(new_yaml, loader) if e not in before]


def _validate(text: str, loader: Callable[[str], Any]) -> list[str]:
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False, encoding="utf-8") as handle:
        handle.write(text)
        tmp = handle.name
    try:
        return [str(e) for e in loader(tmp).validate()]
    finally:
        Path(tmp).unlink()
