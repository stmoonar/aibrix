from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Any


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
    eta_m: float | None
    trs: float
    theta_m: float | None
    tau: TauThresholds
    donor_tier: str | None = None
    eta_crit: float | None = None
    eta_low: float | None = None


def classify_model(
    *,
    model_name: str,
    trs: float,
    Z_m: float | None,
    eta_m: float | None,
    theta_m: float | None,
    tau: TauThresholds,
    eta_crit: float | None = None,
    eta_low: float | None = None,
) -> ModelClassification:
    if Z_m is None:
        return ModelClassification(
            model_name=model_name,
            state=ModelState.UNKNOWN,
            role=ModelRole.UNKNOWN,
            Z_m=Z_m,
            eta_m=eta_m,
            trs=trs,
            theta_m=theta_m,
            tau=tau,
            eta_crit=eta_crit,
            eta_low=eta_low,
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

    donor_tier: str | None = None
    if role == ModelRole.DONOR:
        donor_tier = "waste" if eta_low is not None and eta_m is not None and eta_m < eta_low else "surplus"

    return ModelClassification(
        model_name=model_name,
        state=state,
        role=role,
        Z_m=Z_m,
        eta_m=eta_m,
        trs=trs,
        theta_m=theta_m,
        tau=tau,
        donor_tier=donor_tier,
        eta_crit=eta_crit,
        eta_low=eta_low,
    )


def classify_all_models(
    model_contexts: dict[str, dict[str, Any]],
    *,
    delta_crit: float = 0.2,
    delta_high: float = 0.25,
    model_control_configs: dict[str, dict[str, Any]] | None = None,
) -> list[ModelClassification]:
    results: list[ModelClassification] = []

    for model_name, ctx in model_contexts.items():
        control: dict[str, Any] = {}
        if model_control_configs:
            control = model_control_configs.get(model_name, {})

        d_crit = _float_or(control.get("delta_crit"), delta_crit)
        d_high = _float_or(control.get("delta_high"), delta_high)
        eta_crit = _float_or(control.get("receiver_thrashing_eff"), 200.0)
        eta_low = _float_or(control.get("donor_waste_eff"), 300.0)
        tau = TauThresholds.from_control(d_crit, d_high)

        if _ctx_is_zero_load(ctx):
            results.append(
                ModelClassification(
                    model_name=model_name,
                    state=ModelState.IDLE,
                    role=ModelRole.DONOR,
                    Z_m=ctx.get("z_m"),
                    eta_m=ctx.get("eta_m"),
                    trs=ctx.get("trs", 0.0),
                    theta_m=ctx.get("theta_m"),
                    tau=tau,
                    donor_tier="idle",
                    eta_crit=eta_crit,
                    eta_low=eta_low,
                )
            )
            continue

        results.append(
            classify_model(
                model_name=model_name,
                trs=ctx.get("trs", 0.0),
                Z_m=ctx.get("z_m"),
                eta_m=ctx.get("eta_m"),
                theta_m=ctx.get("theta_m"),
                tau=tau,
                eta_crit=eta_crit,
                eta_low=eta_low,
            )
        )

    return results


def filter_donors_by_eta(
    donors: list[ModelClassification],
) -> tuple[list[ModelClassification], list[ModelClassification]]:
    eligible: list[ModelClassification] = []
    filtered_out: list[ModelClassification] = []
    for donor in donors:
        if donor.state == ModelState.IDLE:
            eligible.append(donor)
            continue
        if donor.eta_m is not None and donor.eta_crit is not None and donor.eta_m < donor.eta_crit:
            filtered_out.append(donor)
        else:
            eligible.append(donor)
    return eligible, filtered_out


def split_receivers_donors(
    classifications: list[ModelClassification],
    *,
    apply_eta_gate: bool = True,
) -> tuple[list[ModelClassification], list[ModelClassification]]:
    receivers = [item for item in classifications if item.role == ModelRole.RECEIVER]
    donors = [item for item in classifications if item.role == ModelRole.DONOR]
    receivers.sort(key=_receiver_sort_key)
    donors.sort(key=donor_mock_cost_key)
    if apply_eta_gate:
        donors, _filtered = filter_donors_by_eta(donors)
    return receivers, donors


def build_comparison_log(
    *,
    model_name: str,
    legacy_type: str,
    paper_cls: ModelClassification,
) -> dict[str, Any]:
    eta_gate_pass: bool | None = None
    if paper_cls.role == ModelRole.DONOR and paper_cls.eta_m is not None and paper_cls.eta_crit is not None:
        eta_gate_pass = paper_cls.eta_m >= paper_cls.eta_crit
    return {
        "event": "state_comparison",
        "model": model_name,
        "legacy_type": legacy_type,
        "legacy_role": (
            "receiver"
            if legacy_type in ("THRASHING", "CONGESTED")
            else ("donor" if legacy_type in ("WASTE", "SURPLUS") else "neutral")
        ),
        "paper_state": paper_cls.state.value,
        "paper_role": paper_cls.role.value,
        "Z_m": paper_cls.Z_m,
        "eta_m": paper_cls.eta_m,
        "eta_crit": paper_cls.eta_crit,
        "eta_gate_pass": eta_gate_pass,
        "donor_tier": paper_cls.donor_tier,
        "theta_m": paper_cls.theta_m,
        "trs": paper_cls.trs,
    }


def donor_mock_cost_key(classification: Any) -> tuple[int, float, float]:
    donor_tier = getattr(classification, "donor_tier", None)
    if donor_tier == "idle":
        donor_prio = -1
    elif donor_tier == "waste":
        donor_prio = 0
    else:
        donor_prio = 1

    z_m = _float_or(getattr(classification, "Z_m", None), 0.0)
    eta_m_raw = _float_or(getattr(classification, "eta_m", None), None)
    eta_m = eta_m_raw if eta_m_raw is not None else float("inf")
    return (donor_prio, -z_m, eta_m)


def _receiver_sort_key(classification: ModelClassification) -> tuple[int, float]:
    prio = 1 if classification.state == ModelState.CRITICAL else 2
    return (prio, classification.Z_m if classification.Z_m is not None else float("inf"))


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
