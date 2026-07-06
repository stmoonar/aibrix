"""Registry-parameter edit surface: whitelist, bounds, cross-field checks, safe render.

The console may edit only per-model TRS/SLO/replica params (the surface R3 recalibration
writes). Everything else -- names, weights, images, tp_size, and the window-coupled
ema_tau_ms -- is locked. Every candidate edit is re-parsed with the controller's OWN
load_registry before it is allowed to reach the ConfigMap, so a bad edit is rejected here
rather than crashlooping the controller on restart.
"""
from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any, Callable

import yaml

from tre_common.registry import load_registry

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
}
_SLO_BOUNDS: dict[str, tuple] = {
    "ttft_p95_ms": (50, 5000, "float", False, False),
    "tpot_p95_ms": (10, 500, "float", False, False),
    "e2e_p95_ms": (1000, 120000, "float", False, False),
}
_TOP_BOUNDS: dict[str, tuple] = {
    "min_replicas": (0, 4, "int", False, False),
    "max_replicas": (1, 4, "int", False, False),
}
_LOCKED = {
    "name": "identity; change via git redeploy",
    "weights_path": "placement/structural; git redeploy",
    "vllm_image": "placement/structural; git redeploy",
    "tp_size": "placement/structural; git redeploy",
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
        editable = {}
        for field, spec in _TRS_BOUNDS.items():
            editable[f"trs.{field}"] = _field_view(trs.get(field), spec)
        for field, spec in _SLO_BOUNDS.items():
            editable[f"slo.{field}"] = _field_view(slo.get(field), spec)
        for field, spec in _TOP_BOUNDS.items():
            editable[field] = _field_view(model.get(field), spec)
        locked = {
            "name": {"value": name, "reason": _LOCKED["name"]},
            "weights_path": {"value": model.get("weights_path"), "reason": _LOCKED["weights_path"]},
            "vllm_image": {"value": model.get("vllm_image"), "reason": _LOCKED["vllm_image"]},
            "tp_size": {"value": model.get("tp_size"), "reason": _LOCKED["tp_size"]},
            "trs.ema_tau_ms": {"value": trs.get("ema_tau_ms"), "reason": _LOCKED["trs.ema_tau_ms"]},
        }
        out[name] = {
            "editable": editable,
            "locked": locked,
            "constraints": ["tau_crit <= tau_low < tau_high", "qmin <= qsat", "min_replicas <= max_replicas", "w_p and w_d not both 0"],
        }
    return out


def _field_view(value: Any, spec: tuple) -> dict[str, Any]:
    lo, hi, kind, lo_ex, hi_ex = spec
    return {"value": value, "min": lo, "max": hi, "type": kind, "min_exclusive": lo_ex, "max_exclusive": hi_ex}


def apply_and_validate(
    registry_yaml: str,
    edits: dict[str, Any],
    *,
    loader: Callable[[str], Any] = load_registry,
) -> str:
    """Return a new registry.yaml with `edits` applied, or raise ParamValidationError.

    edits: {model_name: {"trs": {f: v}, "slo": {f: v}, "min_replicas": v, "max_replicas": v}}
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


def _coerce(raw: Any, spec: tuple, model: str, key: str, errors: list) -> Any:
    lo, hi, kind, lo_ex, hi_ex = spec
    try:
        value = int(raw) if kind == "int" else float(raw)
    except (TypeError, ValueError):
        errors.append({"model": model, "field": key, "error": f"not_a_{kind}"})
        return None
    if (value <= lo if lo_ex else value < lo) or (value >= hi if hi_ex else value > hi):
        errors.append({"model": model, "field": key, "error": "out_of_bounds", "min": lo, "max": hi})
        return None
    return value


def _reject_locked(changes: dict, model: str, errors: list) -> None:
    for locked_key in ("name", "weights_path", "vllm_image", "tp_size"):
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
    if trs.get("w_p") == 0 and trs.get("w_d") == 0:
        add("trs.w_p", "w_p and w_d must not both be 0")


def render(data: dict) -> str:
    return yaml.safe_dump(data, sort_keys=False, default_flow_style=False, allow_unicode=True)
