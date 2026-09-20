"""Calibration helpers for TRE signal fitting."""

from tre_calibration.capacity import CapacityPoint, CapacitySample, CapacitySurface, fit_capacity_surface
from tre_calibration.dataset import CalibrationWindow, load_windows_from_csv, split_by_scenario
from tre_calibration.evaluate import SignalDirectionEvaluation, ThresholdEvaluation, evaluate_signal_direction, evaluate_threshold
from tre_calibration.fit import (
    DELTA_METHOD,
    THETA_CRITERIA,
    THETA_METHOD_BALANCED_ACCURACY,
    THETA_METHOD_RELIABILITY,
    BalancedAccuracyThetaFit,
    DeltaLabelSummary,
    DeltaMarginFit,
    DeltaMarginsFit,
    FittedTheta,
    ReliabilityThetaFit,
    fit_delta_margins,
    fit_theta_by_balanced_accuracy,
    fit_theta_by_reliability,
    fit_theta_from_health,
)
from tre_calibration.profile import build_profile_patch, theta_method_of
from tre_calibration.signals import ParameterCandidateScore, ParameterSearchResult, SignalInputs, TrsBreakdown, compute_trs, grid_search_parameters, score_parameter_candidate

__all__ = [
    "BalancedAccuracyThetaFit",
    "CalibrationWindow",
    "DELTA_METHOD",
    "DeltaLabelSummary",
    "DeltaMarginFit",
    "DeltaMarginsFit",
    "THETA_CRITERIA",
    "THETA_METHOD_BALANCED_ACCURACY",
    "THETA_METHOD_RELIABILITY",
    "CapacityPoint",
    "CapacitySample",
    "CapacitySurface",
    "FittedTheta",
    "ReliabilityThetaFit",
    "ParameterCandidateScore",
    "ParameterSearchResult",
    "SignalDirectionEvaluation",
    "SignalInputs",
    "ThresholdEvaluation",
    "TrsBreakdown",
    "build_profile_patch",
    "compute_trs",
    "evaluate_signal_direction",
    "evaluate_threshold",
    "fit_capacity_surface",
    "fit_delta_margins",
    "fit_theta_by_balanced_accuracy",
    "fit_theta_by_reliability",
    "fit_theta_from_health",
    "grid_search_parameters",
    "load_windows_from_csv",
    "score_parameter_candidate",
    "split_by_scenario",
    "theta_method_of",
]
