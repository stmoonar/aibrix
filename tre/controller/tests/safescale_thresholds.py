"""Shared static SafeScale thresholds for tests (the machine's ``thresholds`` is required).
SafeScale judges TPOT only (2026-11-04): the one value is TPOT_slo (ms)."""
from tre_controller.planning.safescale_evidence import StaticThresholds

STATIC_TPOT_75 = StaticThresholds(75.0)
STATIC_TPOT_100 = StaticThresholds(100.0)
