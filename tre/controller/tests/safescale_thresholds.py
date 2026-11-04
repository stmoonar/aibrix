"""Shared static SafeScale thresholds for tests (the machine's ``thresholds`` is required)."""
from tre_controller.planning.safescale_evidence import StaticThresholds

STATIC_500_75 = StaticThresholds(500.0, 75.0)
STATIC_1000_100 = StaticThresholds(1000.0, 100.0)
