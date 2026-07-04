from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


@dataclass(frozen=True)
class BehaviorTableRow:
    step_id: str
    old_shell_behavior: str
    python_status: str
    notes: str


def discover_config_traces(config_dir: str | Path, *, only_prefixes: Sequence[str] = ()) -> list[str]:
    root = Path(config_dir)
    if not root.is_dir():
        return []
    if (root / "config.yaml").exists():
        names = [root.name]
    else:
        names = sorted(path.name for path in root.iterdir() if path.is_dir() and (path / "config.yaml").exists())

    if not only_prefixes:
        return names

    matched: list[str] = []
    seen: set[str] = set()
    for prefix in only_prefixes:
        for name in names:
            if name.startswith(prefix) and name not in seen:
                matched.append(name)
                seen.add(name)
    return matched


def build_behavior_table() -> list[BehaviorTableRow]:
    return [
        BehaviorTableRow(
            "discover_traces",
            "Scan config root for child directories containing config.yaml; support --only prefix filters.",
            "implemented",
            "Implemented as deterministic offline discovery only.",
        ),
        BehaviorTableRow(
            "switch_mechanism",
            "Call toggle_tre_apa_hot_switch.sh to switch APA/TRE and wait for controller restart.",
            "not_executed_offline",
            "Cluster mutation is outside P7 offline verification and remains an explicit future command.",
        ),
        BehaviorTableRow(
            "reset_replicas",
            "Call run_all.sh scale to reset all model replicas to 0 then 1 and wait for readiness.",
            "not_executed_offline",
            "Would modify live service-manager/model state, so it is documented but not executed.",
        ),
        BehaviorTableRow(
            "dispatch_trace",
            "Run CustomTraceGenerator main.py dispatch stage for each trace/mechanism output directory.",
            "planned",
            "Future Python path will bind trace loading, scheduling, and dispatcher primitives.",
        ),
        BehaviorTableRow(
            "fetch_metrics",
            "Run fetch_and_plot.sh to collect TRE logs and plots after a trace run.",
            "planned",
            "Requires completed live or smoke run artifacts.",
        ),
        BehaviorTableRow(
            "compare_plots",
            "Run plot_compare_performance_cdf_latex.py over TRE and APA result directories.",
            "planned",
            "Comparison is retained as a post-processing step after real experiments.",
        ),
    ]


def behavior_table_markdown(rows: Sequence[BehaviorTableRow] | None = None) -> str:
    rows = list(rows) if rows is not None else build_behavior_table()
    lines = ["| Step | Old shell behavior | Python status | Notes |", "| --- | --- | --- | --- |"]
    for row in rows:
        lines.append(f"| `{row.step_id}` | {row.old_shell_behavior} | `{row.python_status}` | {row.notes} |")
    return "\n".join(lines)
