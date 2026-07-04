"""Calibration helpers for TRE signal fitting."""

from tre_calibration.dataset import CalibrationWindow, load_windows_from_csv, split_by_scenario
from tre_calibration.evaluate import SignalDirectionEvaluation, ThresholdEvaluation, evaluate_signal_direction, evaluate_threshold
from tre_calibration.fit import FittedTheta, ReliabilityThetaFit, fit_theta_by_reliability, fit_theta_from_health
from tre_calibration.signals import ParameterCandidateScore, SignalInputs, TrsBreakdown, compute_trs, score_parameter_candidate

__all__ = [
    "CalibrationWindow",
    "FittedTheta",
    "ReliabilityThetaFit",
    "ParameterCandidateScore",
    "SignalDirectionEvaluation",
    "SignalInputs",
    "ThresholdEvaluation",
    "TrsBreakdown",
    "compute_trs",
    "evaluate_signal_direction",
    "evaluate_threshold",
    "fit_theta_by_reliability",
    "fit_theta_from_health",
    "load_windows_from_csv",
    "score_parameter_candidate",
    "split_by_scenario",
]
