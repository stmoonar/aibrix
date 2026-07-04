from __future__ import annotations

from typing import Any

from tre_calibration.fit import ReliabilityThetaFit
from tre_calibration.signals import ParameterCandidateScore


def build_profile_patch(
    model_name: str,
    *,
    theta_fit: ReliabilityThetaFit,
    parameter_score: ParameterCandidateScore,
    generated_at: str,
) -> dict[str, Any]:
    return {
        "generated_at": generated_at,
        "model_name": model_name,
        "publish": theta_fit.publish,
        "fit": {
            "attainment": theta_fit.attainment,
            "candidate_count": theta_fit.candidate_count,
            "confidence": theta_fit.confidence,
            "coverage_pass": theta_fit.coverage_pass,
            "family_counts": dict(sorted(theta_fit.family_counts.items())),
            "reject_reason": theta_fit.reject_reason,
            "support": theta_fit.support,
        },
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
