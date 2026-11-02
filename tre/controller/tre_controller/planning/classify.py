from __future__ import annotations

import logging
import math
import re
from dataclasses import dataclass
from enum import Enum
from typing import Any

LOG = logging.getLogger(__name__)


class ModelState(str, Enum):
    CRITICAL = "critical"
    LOW = "low"
    HEALTHY = "healthy"
    HIGH = "high"
    IDLE = "idle"
    UNKNOWN = "unknown"


class ModelRole(str, Enum):
    RECEIVER = "receiver"
    DONOR = "donor"
    NEUTRAL = "neutral"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class TauThresholds:
    tau_low: float
    tau_crit: float
    tau_high: float

    @staticmethod
    def from_control(delta_crit: float = 0.2, delta_high: float = 0.25) -> "TauThresholds":
        return TauThresholds(tau_low=1.0, tau_crit=1.0 - delta_crit, tau_high=1.0 + delta_high)


@dataclass(frozen=True)
class ModelClassification:
    model_name: str
    state: ModelState
    role: ModelRole
    Z_m: float | None
    trs: float
    theta_m: float | None
    tau: TauThresholds
    # 2026-11-02: eta_m, eta_crit / eta_low and donor_tier (idle / waste / surplus) removed;
    # the donor order is donor_sort_key (IDLE first by state).
    #: TSS undefined because nothing was in flight (running + waiting == 0) while tokens
    #: still completed. Classified HEALTHY/NEUTRAL (surplus side, never a receiver) and
    #: exempt from the "Z missing -> incomplete" drop, see plan 6.4 idle rule.
    signal_idle: bool = False


def classify_model(
    *,
    model_name: str,
    trs: float,
    Z_m: float | None,
    theta_m: float | None,
    tau: TauThresholds,
    request_rate_rps: float | None = None,
    idle_rps_eps: float = 0.05,
) -> ModelClassification:
    request_rate = _as_nonneg_float(request_rate_rps)
    # Z = 0 is a TSS of 0 with a queue - nothing processed (a stalled engine, 2026-10-29),
    # not light traffic: the low-request-rate shortcut does not hide it.
    stalled = Z_m is not None and Z_m <= 0.0
    if request_rate is not None and request_rate < max(0.0, idle_rps_eps) and not stalled:
        return ModelClassification(
            model_name=model_name,
            state=ModelState.HEALTHY,
            role=ModelRole.NEUTRAL,
            Z_m=Z_m,
            trs=trs,
            theta_m=theta_m,
            tau=tau,
        )

    if Z_m is None:
        return ModelClassification(
            model_name=model_name,
            state=ModelState.UNKNOWN,
            role=ModelRole.UNKNOWN,
            Z_m=Z_m,
            trs=trs,
            theta_m=theta_m,
            tau=tau,
        )

    if Z_m < tau.tau_crit:
        state = ModelState.CRITICAL
        role = ModelRole.RECEIVER
    elif Z_m < tau.tau_low:
        state = ModelState.LOW
        role = ModelRole.RECEIVER
    elif Z_m <= tau.tau_high:
        state = ModelState.HEALTHY
        role = ModelRole.NEUTRAL
    else:
        state = ModelState.HIGH
        role = ModelRole.DONOR

    return ModelClassification(
        model_name=model_name,
        state=state,
        role=role,
        Z_m=Z_m,
        trs=trs,
        theta_m=theta_m,
        tau=tau,
    )


def model_control_configs_from_registry(
    registry: Any, signal_source: str = "zm"
) -> dict[str, dict[str, Any]]:
    """Per-model control knobs for :func:`classify_all_models`, read from the registry.

    For TSS (``signal_source == "zm"``) ``registry.yaml`` stores the fitted band edges as
    ``trs.tau_crit`` / ``trs.tau_high`` around ``trs.tau_low``; :class:`TauThresholds` is
    built from the *margins*, so they are converted back here.

    For an alternative signal (the ablation arms) the bands are that signal's own fitted
    ``alt_thresholds.<signal>.delta_crit`` / ``delta_high`` (plan §6.9 item 3) - running
    queue length on the TSS bands would compare the signals under TSS's margins. The
    controller refuses to start an alt arm without them (``config._validate_signal_thresholds``),
    so a missing margin here is an error, never a default band (the plan defaults 0.2 /
    0.25 fallback was removed, review 2026-10-28). An unfitted source (``latency_p95``,
    no ``alt_thresholds`` entry) keeps the TSS bands. Models missing from the registry
    simply get no entry, and ``classify_all_models`` then applies its defaults.
    """
    configs: dict[str, dict[str, Any]] = {}
    for spec in registry.models():
        if signal_source != "zm" and signal_source in getattr(spec, "alt_thresholds", {}):
            threshold = spec.alt_thresholds[signal_source]
            if threshold.delta_crit is None or threshold.delta_high is None:
                raise ValueError(
                    f"model {spec.name}: alt_thresholds.{signal_source} has no fitted delta_crit/delta_high "
                    f"(no default bands)"
                )
            configs[spec.name] = {"delta_crit": float(threshold.delta_crit), "delta_high": float(threshold.delta_high)}
            continue
        trs = spec.trs
        tau_low = float(trs.tau_low)
        configs[spec.name] = {
            "delta_crit": tau_low - float(trs.tau_crit),
            "delta_high": float(trs.tau_high) - tau_low,
        }
    return configs


def classify_all_models(
    model_contexts: dict[str, dict[str, Any]],
    *,
    delta_crit: float = 0.2,
    delta_high: float = 0.25,
    model_control_configs: dict[str, dict[str, Any]] | None = None,
    signal_idle_rps_eps: float = 0.05,
) -> list[ModelClassification]:
    results: list[ModelClassification] = []

    for model_name, ctx in model_contexts.items():
        control: dict[str, Any] = {}
        if model_control_configs:
            control = model_control_configs.get(model_name, {})

        d_crit = _float_or(control.get("delta_crit"), delta_crit)
        d_high = _float_or(control.get("delta_high"), delta_high)
        tau = TauThresholds.from_control(d_crit, d_high)

        if _ctx_is_signal_idle(ctx):
            # Plan 6.4 idle rule: A + W == 0 -> Z undefined. Not CRITICAL (a light model
            # would otherwise read as starved), not UNKNOWN (it would be dropped from the
            # plan); HEALTHY keeps it off the receiver list while util scale-down and the
            # middle-zone donor path can still take surplus replicas from it.
            results.append(
                ModelClassification(
                    model_name=model_name,
                    state=ModelState.HEALTHY,
                    role=ModelRole.NEUTRAL,
                    Z_m=None,
                    trs=ctx.get("trs", 0.0),
                    theta_m=ctx.get("theta_m"),
                    tau=tau,
                    signal_idle=True,
                )
            )
            continue

        if _ctx_is_zero_load(ctx):
            results.append(
                ModelClassification(
                    model_name=model_name,
                    state=ModelState.IDLE,
                    role=ModelRole.DONOR,
                    Z_m=ctx.get("z_m"),
                    trs=ctx.get("trs", 0.0),
                    theta_m=ctx.get("theta_m"),
                    tau=tau,
                )
            )
            continue

        results.append(
            classify_model(
                model_name=model_name,
                trs=ctx.get("trs", 0.0),
                Z_m=ctx.get("z_m"),
                theta_m=ctx.get("theta_m"),
                tau=tau,
                request_rate_rps=ctx.get("request_rate_rps"),
                idle_rps_eps=signal_idle_rps_eps,
            )
        )

    return results


def _natural_text_key(value: str) -> tuple[tuple[int, Any], ...]:
    parts = re.split(r"(\d+)", value.casefold())
    return tuple(
        (0, int(part)) if part.isdigit() else (1, part)
        for part in parts
        if part
    )


def receiver_sort_key(classification: Any) -> tuple[int, float]:
    """Receiver order: CRITICAL before LOW, then the most starved first (raw Z
    ascending; no Z sorts last)."""
    prio = 1 if classification.state == ModelState.CRITICAL else 2
    return (prio, classification.Z_m if classification.Z_m is not None else float("inf"))


def donor_sort_key(classification: Any) -> tuple[int, float, tuple[tuple[int, Any], ...]]:
    """Donor order, the same on every arm: IDLE first, then Z descending (no Z last),
    ties by natural model name.

    2026-11-02 (user decision): eta (TSS / routable pods) and its "waste" tier are gone.
    TSS is already a fleet ratio, so eta fell as 1/n at equal per-replica load and its
    v1 thresholds (200 / 300) were never calibrated; the order no longer depends on the
    replica count. An alt-signal arm sorts by its own Z (TRE_DISABLE_ETA_GATE removed)."""
    prio = 0 if classification.state == ModelState.IDLE else 1
    z_m = _float_or(classification.Z_m, None)
    return (
        prio,
        -z_m if z_m is not None else float("inf"),
        _natural_text_key(str(classification.model_name)),
    )


def _ctx_is_signal_idle(ctx: dict[str, Any] | None) -> bool:
    """TSS was undefined (idle rule) while tokens still flowed, on the zm signal source."""
    if not isinstance(ctx, dict) or ctx.get("tss_defined", True) is not False:
        return False
    if ctx.get("signal_source", "zm") != "zm":
        return False
    return not _ctx_is_zero_load(ctx)


def _ctx_is_zero_load(ctx: dict[str, Any] | None) -> bool:
    if not isinstance(ctx, dict):
        return False
    y_total = _as_nonneg_float(ctx.get("Y_m"))
    queue_total = _as_nonneg_float(ctx.get("Q"))
    if y_total is None or queue_total is None:
        return False
    return y_total <= 1e-9 and queue_total <= 1e-9


def _as_nonneg_float(value: Any) -> float | None:
    try:
        parsed = float(value)
    except Exception:
        return None
    if math.isnan(parsed) or math.isinf(parsed):
        return None
    if parsed < 0:
        return None
    return parsed


def _float_or(value: Any, default: Any) -> Any:
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default
