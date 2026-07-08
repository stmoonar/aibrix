"""Experiment-3 comparison driver: for each arm (tre, apa) x each trace, switch the decision
source, reset state, replay the trace, and score it into its own result directory.

This is the real driver the offline behavior table in orchestrate.py describes (that module
stays a documentation/behaviour table with its own tests; this one actuates). Cluster-mutating
steps -- the arm switch (deploy/scripts/toggle_tre_apa.sh) and the per-trace reset
(deploy/scripts/reset_between_traces.sh) -- are emitted into a plan.json and only executed
when execute_cluster_ops=True; the replay + scoring step (run_trace) runs offline in --dry-run
so the whole pipeline is exercisable in CI without a cluster.

Result layout: <out_root>/<arm>/<trace_name>/{requests.jsonl, summary.json}, plus
<out_root>/plan.json describing every command for the live run.
"""
from __future__ import annotations

import argparse
import json
import subprocess
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Sequence

from tre_replayer.run_trace import run_trace
from tre_replayer.traces.loader import discover_trace_set


@dataclass(frozen=True)
class ReplaySpec:
    trace_path: str
    out_jsonl: str
    summary_json: str


@dataclass(frozen=True)
class ComparisonStep:
    arm: str
    trace_name: str
    result_dir: str
    # Set only on the first trace of an arm: switch the whole cluster to this arm once.
    arm_switch_command: list[str] | None
    # Run before every trace: reset serving floor / redis / controller state.
    reset_command: list[str]
    replay: ReplaySpec


def build_comparison_plan(
    trace_root: str | Path,
    out_root: str | Path,
    *,
    arms: Sequence[str] = ("tre", "apa"),
    toggle_script: str = "deploy/scripts/toggle_tre_apa.sh",
    reset_script: str = "deploy/scripts/reset_between_traces.sh",
) -> list[ComparisonStep]:
    cases = discover_trace_set(trace_root).cases
    out_root = Path(out_root)
    steps: list[ComparisonStep] = []
    for arm in arms:
        for i, case in enumerate(cases):
            result_dir = out_root / arm / case.name
            steps.append(
                ComparisonStep(
                    arm=arm,
                    trace_name=case.name,
                    result_dir=str(result_dir),
                    arm_switch_command=[toggle_script, arm] if i == 0 else None,
                    reset_command=[reset_script, str(result_dir / "reset_archive")],
                    replay=ReplaySpec(
                        trace_path=str(case.path),
                        out_jsonl=str(result_dir / "requests.jsonl"),
                        summary_json=str(result_dir / "summary.json"),
                    ),
                )
            )
    return steps


def _run_cluster_op(command: list[str]) -> None:
    subprocess.run(command, check=True)


def run_comparison(
    steps: Sequence[ComparisonStep],
    out_root: str | Path,
    *,
    dry_run: bool = True,
    execute_cluster_ops: bool = False,
    gateway_url: str = "http://192.168.223.76:31592/v1/completions",
    registry_path: str | None = None,
    seed: int = 0,
) -> dict:
    out_root = Path(out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    plan = {
        "arms": sorted({s.arm for s in steps}),
        "dry_run": dry_run,
        "execute_cluster_ops": execute_cluster_ops,
        "steps": [asdict(s) for s in steps],
    }
    (out_root / "plan.json").write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")

    results: list[dict] = []
    for step in steps:
        Path(step.result_dir).mkdir(parents=True, exist_ok=True)
        if execute_cluster_ops:
            if step.arm_switch_command:
                _run_cluster_op(step.arm_switch_command)
            _run_cluster_op(step.reset_command)
        summary = run_trace(
            step.replay.trace_path,
            gateway_url=gateway_url,
            out_path=step.replay.out_jsonl,
            registry_path=registry_path,
            seed=seed,
            dry_run=dry_run,
        )
        summary["arm"] = step.arm
        Path(step.replay.summary_json).write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        results.append(summary)
    return {"plan": plan, "results": results}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Drive the experiment-3 TRE-vs-APA comparison.")
    ap.add_argument("--trace-root", required=True, help="trace set dir (INDEX.json + <name>/trace.json)")
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--arm", action="append", dest="arms", help="repeat; default tre then apa")
    ap.add_argument("--gateway-url", default="http://192.168.223.76:31592/v1/completions")
    ap.add_argument("--registry", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true", help="fake sender; no network")
    ap.add_argument(
        "--execute-cluster-ops", action="store_true",
        help="actually run toggle + reset (DANGER: mutates the cluster). Off by default.",
    )
    args = ap.parse_args(argv)

    arms = tuple(args.arms) if args.arms else ("tre", "apa")
    steps = build_comparison_plan(args.trace_root, args.out_root, arms=arms)
    out = run_comparison(
        steps, args.out_root, dry_run=args.dry_run, execute_cluster_ops=args.execute_cluster_ops,
        gateway_url=args.gateway_url, registry_path=args.registry, seed=args.seed,
    )
    print(json.dumps({"arms": out["plan"]["arms"], "results": len(out["results"])}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
