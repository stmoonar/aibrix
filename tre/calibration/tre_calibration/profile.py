"""Build the machine-readable calibration artifact for one model.

The artifact records *how* a parameter was produced, not just its value: the method
string, every knob the fit was given (including ``trim_ramp_windows``, which moves
theta by a few percent), and -- when the caller supplies it -- the provenance of the
input CSV. A registry edit that cannot be traced back to one of these artifacts is a
number nobody can reproduce.
"""
from __future__ import annotations

from dataclasses import asdict
from typing import Any

from tre_calibration.fit import (
    THETA_METHOD_BALANCED_ACCURACY,
    THETA_METHOD_RELIABILITY,
    DELTA_METHOD,
    BalancedAccuracyThetaFit,
    DeltaMarginsFit,
    ReliabilityThetaFit,
)
from tre_calibration.signals import ParameterCandidateScore


def theta_method_of(theta_fit: ReliabilityThetaFit | BalancedAccuracyThetaFit) -> str:
    if isinstance(theta_fit, BalancedAccuracyThetaFit):
        return THETA_METHOD_BALANCED_ACCURACY
    if isinstance(theta_fit, ReliabilityThetaFit):
        return THETA_METHOD_RELIABILITY
    raise TypeError(f"unsupported theta fit type: {type(theta_fit)!r}")


def build_profile_patch(
    model_name: str,
    *,
    theta_fit: ReliabilityThetaFit | BalancedAccuracyThetaFit,
    parameter_score: ParameterCandidateScore,
    generated_at: str,
    fit_config: dict[str, Any] | None = None,
    delta_fit: DeltaMarginsFit | None = None,
    inputs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    theta_method = theta_method_of(theta_fit)
    patch: dict[str, Any] = {
        "generated_at": generated_at,
        "model_name": model_name,
        "publish": theta_fit.publish,
        "method": {
            "theta_m_method": theta_method,
            "delta_method": DELTA_METHOD if delta_fit is not None else None,
        },
        "fit_config": dict(sorted((fit_config or {}).items())),
        "fit": _theta_fit_block(theta_fit),
        "metrics": {
            "auroc": parameter_score.auroc,
            "objective": parameter_score.objective,
            "spearman_health": parameter_score.spearman_health,
        },
        "trs": {
            "lambda_wait": parameter_score.lambda_wait,
            "qmin": parameter_score.qmin,
            "theta_m": theta_fit.theta,
            "w_p": parameter_score.w_p,
        },
    }
    if delta_fit is not None:
        patch["trs"].update(
            {
                "tau_low": delta_fit.tau_low,
                "tau_crit": delta_fit.crit.tau,
                "tau_high": delta_fit.high.tau,
                "delta_crit": delta_fit.crit.delta,
                "delta_high": delta_fit.high.delta,
            }
        )
        patch["delta_fit"] = {
            "crit": asdict(delta_fit.crit),
            "high": asdict(delta_fit.high),
            "labels": asdict(delta_fit.labels),
            "theta": delta_fit.theta,
            "tau_low": delta_fit.tau_low,
        }
    if inputs is not None:
        patch["inputs"] = dict(sorted(inputs.items()))
    return patch


def _theta_fit_block(
    theta_fit: ReliabilityThetaFit | BalancedAccuracyThetaFit,
) -> dict[str, Any]:
    if isinstance(theta_fit, BalancedAccuracyThetaFit):
        return {
            "balanced_accuracy": theta_fit.balanced_accuracy,
            "candidate_count": theta_fit.candidate_count,
            "coverage_pass": theta_fit.coverage_pass,
            "family_counts": dict(sorted(theta_fit.family_counts.items())),
            "healthy_quantile": theta_fit.healthy_quantile,
            "healthy_window_count": theta_fit.healthy_window_count,
            "min_healthy_recall": theta_fit.min_healthy_recall,
            "precision_good": theta_fit.precision_good,
            "recall_good": theta_fit.recall_good,
            "reject_reason": theta_fit.reject_reason,
            "specificity_bad": theta_fit.specificity_bad,
            "violating_window_count": theta_fit.violating_window_count,
        }
    return {
        "attainment": theta_fit.attainment,
        "candidate_count": theta_fit.candidate_count,
        "confidence": theta_fit.confidence,
        "coverage_pass": theta_fit.coverage_pass,
        "family_counts": dict(sorted(theta_fit.family_counts.items())),
        "reject_reason": theta_fit.reject_reason,
        "support": theta_fit.support,
    }
