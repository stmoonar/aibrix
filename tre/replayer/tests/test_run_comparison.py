from __future__ import annotations

import json
from pathlib import Path

from tre_replayer.run_comparison import build_comparison_plan, run_comparison


def _write_trace(path: Path, rps: float = 2.0) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    trace = {
        "dsqwen-7b": [
            {"start_time": 0, "end_time": 3, "rps": rps, "input_tokens": 256, "max_tokens": 8},
        ]
    }
    path.write_text(json.dumps(trace), encoding="utf-8")


def _trace_root(tmp_path: Path) -> Path:
    root = tmp_path / "traceset"
    _write_trace(root / "t1_alpha" / "trace.json")
    _write_trace(root / "t2_beta" / "trace.json")
    return root


def test_plan_switches_arm_once_and_resets_every_trace(tmp_path: Path) -> None:
    steps = build_comparison_plan(_trace_root(tmp_path), tmp_path / "out", arms=("tre", "apa"))

    assert [(s.arm, s.trace_name) for s in steps] == [
        ("tre", "t1_alpha"), ("tre", "t2_beta"),
        ("apa", "t1_alpha"), ("apa", "t2_beta"),
    ]
    # arm switch happens once, on the first trace of each arm
    assert steps[0].arm_switch_command == ["deploy/scripts/toggle_tre_apa.sh", "tre"]
    assert steps[1].arm_switch_command is None
    assert steps[2].arm_switch_command == ["deploy/scripts/toggle_tre_apa.sh", "apa"]
    assert steps[3].arm_switch_command is None
    # reset runs before every trace
    for s in steps:
        assert s.reset_command[0] == "deploy/scripts/reset_between_traces.sh"
        assert s.result_dir.endswith(str(Path(s.arm) / s.trace_name))


def test_dry_run_replays_and_scores_each_arm_trace(tmp_path: Path) -> None:
    out_root = tmp_path / "out"
    steps = build_comparison_plan(_trace_root(tmp_path), out_root, arms=("tre", "apa"))
    result = run_comparison(steps, out_root, dry_run=True, execute_cluster_ops=False)

    # plan.json documents the (unexecuted) cluster ops for the live run
    plan = json.loads((out_root / "plan.json").read_text())
    assert plan["arms"] == ["apa", "tre"]
    assert plan["execute_cluster_ops"] is False
    assert plan["trim_ramp_windows"] == 1

    assert len(result["results"]) == 4
    for arm in ("tre", "apa"):
        for name in ("t1_alpha", "t2_beta"):
            summary_path = out_root / arm / name / "summary.json"
            summary = json.loads(summary_path.read_text())
            assert summary["arm"] == arm
            assert "dsqwen-7b" in summary["per_model"]
            assert (out_root / arm / name / "requests.jsonl").exists()
