"""One planner section alone on one snapshot (tests only; the controller runs both
sections through :class:`tre_controller.loops.decision_task.DecisionLoop`)."""
from __future__ import annotations

from tre_common.metrics_schema import MetricsSnapshot
from tre_controller.loops.tick import LoopTickResult, run_planner_tick


def run_rescue_tick(snapshot: MetricsSnapshot, **kwargs) -> LoopTickResult:
    """The rescue section alone on one snapshot."""
    return run_planner_tick(snapshot, **{"rescue_due": True, "fairness_due": False, "loop": "rescue", **kwargs})


def run_fairness_tick(snapshot: MetricsSnapshot, **kwargs) -> LoopTickResult:
    """The rebalance section alone on one snapshot."""
    return run_planner_tick(snapshot, **{"rescue_due": False, "fairness_due": True, "loop": "fairness", **kwargs})
