"""tre_eval: offline evaluation report for TRE v2 arm directories.

Read-only over the run directories written by the pilot / E1 runners (one directory per
arm: ``client/performance_metrics.json``, ``layout.jsonl``, ``pod_gauges.jsonl``, ...).
The metric definitions are in ``docs/eval-metrics-spec-20261007.md`` of the local
workspace and in :mod:`tre_eval.metrics`; ``python3 -m tre_eval.report --help`` for usage.
"""

__all__ = ["load", "metrics", "timeseries", "plots", "report", "style"]
