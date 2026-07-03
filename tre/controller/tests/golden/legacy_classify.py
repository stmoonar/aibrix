from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional


def legacy_as_nonneg_float(value: Any) -> Optional[float]:
    try:
        parsed = float(value)
    except Exception:
        return None
    if math.isnan(parsed) or math.isinf(parsed):
        return None
    if parsed < 0:
        return None
    return parsed


def legacy_float_or(value: Any, default: float) -> float:
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def legacy_ctx_is_zero_load(ctx: Optional[dict[str, Any]]) -> bool:
    if not isinstance(ctx, dict):
        return False
    y_total = legacy_as_nonneg_float(ctx.get("Y_m"))
    queue_total = legacy_as_nonneg_float(ctx.get("Q"))
    if y_total is None or queue_total is None:
        return False
    return y_total <= 1e-9 and queue_total <= 1e-9


class LegacyModelState(str, Enum):
    CRITICAL = "critical"
    LOW = "low"
    HEALTHY = "healthy"
    HIGH = "high"
    IDLE = "idle"
    UNKNOWN = "unknown"


class LegacyModelRole(str, Enum):
    RECEIVER = "receiver"
    DONOR = "donor"
    NEUTRAL = "neutral"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class LegacyTauThresholds:
    tau_low: float
    tau_crit: float
    tau_high: float

    @staticmethod
    def from_control(delta_crit: float = 0.2, delta_high: float = 0.25) -> "LegacyTauThresholds":
        return LegacyTauThresholds(tau_low=1.0, tau_crit=1.0 - delta_crit, tau_high=1.0 + delta_high)


@dataclass(frozen=True)
class LegacyModelClassification:
    model_name: str
    state: LegacyModelState
    role: LegacyModelRole
    Z_m: Optional[float]
    eta_m: Optional[float]
    trs: float
    theta_m: Optional[float]
    tau: LegacyTauThresholds
    donor_tier: Optional[str] = None
    eta_crit: Optional[float] = None
    eta_low: Optional[float] = None


def legacy_classify_model(
    *,
    model_name: str,
    trs: float,
    Z_m: Optional[float],
    eta_m: Optional[float],
    theta_m: Optional[float],
    tau: LegacyTauThresholds,
    eta_crit: Optional[float] = None,
    eta_low: Optional[float] = None,
) -> LegacyModelClassification:
    if Z_m is None:
        return LegacyModelClassification(
            model_name=model_name,
            state=LegacyModelState.UNKNOWN,
            role=LegacyModelRole.UNKNOWN,
            Z_m=Z_m,
            eta_m=eta_m,
            trs=trs,
            theta_m=theta_m,
            tau=tau,
            eta_crit=eta_crit,
            eta_low=eta_low,
        )
    if Z_m < tau.tau_crit:
        state = LegacyModelState.CRITICAL
        role = LegacyModelRole.RECEIVER
    elif Z_m < tau.tau_low:
        state = LegacyModelState.LOW
        role = LegacyModelRole.RECEIVER
    elif Z_m <= tau.tau_high:
        state = LegacyModelState.HEALTHY
        role = LegacyModelRole.NEUTRAL
    else:
        state = LegacyModelState.HIGH
        role = LegacyModelRole.DONOR
    donor_tier: Optional[str] = None
    if role == LegacyModelRole.DONOR:
        donor_tier = "waste" if eta_low is not None and eta_m is not None and eta_m < eta_low else "surplus"
    return LegacyModelClassification(
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


def legacy_classify_all_models(
    model_contexts: dict[str, dict[str, Any]],
    *,
    delta_crit: float = 0.2,
    delta_high: float = 0.25,
    model_control_configs: Optional[dict[str, dict[str, Any]]] = None,
) -> list[LegacyModelClassification]:
    results: list[LegacyModelClassification] = []
    for model_name, ctx in model_contexts.items():
        control = model_control_configs.get(model_name, {}) if model_control_configs else {}
        d_crit = legacy_float_or(control.get("delta_crit"), delta_crit)
        d_high = legacy_float_or(control.get("delta_high"), delta_high)
        eta_crit = legacy_float_or(control.get("receiver_thrashing_eff"), 200.0)
        eta_low = legacy_float_or(control.get("donor_waste_eff"), 300.0)
        tau = LegacyTauThresholds.from_control(d_crit, d_high)
        if legacy_ctx_is_zero_load(ctx):
            results.append(
                LegacyModelClassification(
                    model_name=model_name,
                    state=LegacyModelState.IDLE,
                    role=LegacyModelRole.DONOR,
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
            legacy_classify_model(
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


def legacy_donor_mock_cost_key(classification: Any) -> tuple[int, float, float]:
    donor_tier = getattr(classification, "donor_tier", None)
    if donor_tier == "idle":
        donor_prio = -1
    elif donor_tier == "waste":
        donor_prio = 0
    else:
        donor_prio = 1
    z_m = legacy_float_or(getattr(classification, "Z_m", None), 0.0)
    eta_m_raw = legacy_float_or(getattr(classification, "eta_m", None), None)
    eta_m = eta_m_raw if eta_m_raw is not None else float("inf")
    return (donor_prio, -z_m, eta_m)


def legacy_filter_donors_by_eta(
    donors: list[LegacyModelClassification],
) -> tuple[list[LegacyModelClassification], list[LegacyModelClassification]]:
    eligible: list[LegacyModelClassification] = []
    filtered_out: list[LegacyModelClassification] = []
    for donor in donors:
        if donor.state == LegacyModelState.IDLE:
            eligible.append(donor)
            continue
        if donor.eta_m is not None and donor.eta_crit is not None and donor.eta_m < donor.eta_crit:
            filtered_out.append(donor)
        else:
            eligible.append(donor)
    return eligible, filtered_out


def legacy_split_receivers_donors(
    classifications: list[LegacyModelClassification],
    *,
    apply_eta_gate: bool = True,
) -> tuple[list[LegacyModelClassification], list[LegacyModelClassification]]:
    receivers = [item for item in classifications if item.role == LegacyModelRole.RECEIVER]
    donors = [item for item in classifications if item.role == LegacyModelRole.DONOR]
    receivers.sort(key=lambda item: (1 if item.state == LegacyModelState.CRITICAL else 2, item.Z_m if item.Z_m is not None else float("inf")))
    donors.sort(key=legacy_donor_mock_cost_key)
    if apply_eta_gate:
        donors, _filtered = legacy_filter_donors_by_eta(donors)
    return receivers, donors


def legacy_build_comparison_log(
    *,
    model_name: str,
    legacy_type: str,
    paper_cls: LegacyModelClassification,
) -> dict[str, Any]:
    eta_gate_pass: Optional[bool] = None
    if paper_cls.role == LegacyModelRole.DONOR and paper_cls.eta_m is not None and paper_cls.eta_crit is not None:
        eta_gate_pass = paper_cls.eta_m >= paper_cls.eta_crit
    return {
        "event": "state_comparison",
        "model": model_name,
        "legacy_type": legacy_type,
        "legacy_role": "receiver" if legacy_type in ("THRASHING", "CONGESTED") else ("donor" if legacy_type in ("WASTE", "SURPLUS") else "neutral"),
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
