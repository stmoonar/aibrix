from __future__ import annotations

from tre_calibration.fit import ReliabilityThetaFit
from tre_calibration.profile import build_profile_patch
from tre_calibration.signals import ParameterCandidateScore


def test_build_profile_patch_is_deterministic_and_publishable() -> None:
    theta_fit = ReliabilityThetaFit(
        publish=True,
        theta=105.0,
        support=3,
        attainment=1.0,
        confidence=1.0,
        coverage_pass=True,
        family_counts={"burst": 1, "steady": 2},
        reject_reason=None,
        candidate_count=5,
    )
    parameter_score = ParameterCandidateScore(
        w_p=3.0,
        lambda_wait=1.0,
        qmin=1.0,
        objective=1.0,
        spearman_health=1.0,
        auroc=1.0,
        scored_windows=[],
    )

    patch = build_profile_patch(
        "dsqwen-7b",
        theta_fit=theta_fit,
        parameter_score=parameter_score,
        generated_at="2026-07-04T00:00:00+00:00",
    )

    assert patch == {
        "generated_at": "2026-07-04T00:00:00+00:00",
        "model_name": "dsqwen-7b",
        "publish": True,
        "fit": {
            "attainment": 1.0,
            "candidate_count": 5,
            "confidence": 1.0,
            "coverage_pass": True,
            "family_counts": {"burst": 1, "steady": 2},
            "reject_reason": None,
            "support": 3,
        },
        "metrics": {
            "auroc": 1.0,
            "objective": 1.0,
            "spearman_health": 1.0,
        },
        "trs": {
            "lambda_wait": 1.0,
            "qmin": 1.0,
            "theta_m": 105.0,
            "w_p": 3.0,
        },
    }
