from __future__ import annotations

from tre_replayer.orchestrate import build_behavior_table, discover_config_traces


def test_discover_config_traces_matches_old_shell_directory_scan(tmp_path) -> None:
    for name in ["B_trace", "A_trace"]:
        trace_dir = tmp_path / name
        trace_dir.mkdir()
        (trace_dir / "config.yaml").write_text("custom_load_test: {}\n", encoding="utf-8")
    (tmp_path / "ignored").mkdir()

    assert discover_config_traces(tmp_path) == ["A_trace", "B_trace"]
    assert discover_config_traces(tmp_path, only_prefixes=["B"]) == ["B_trace"]


def test_behavior_table_marks_live_cluster_steps_not_executed() -> None:
    table = build_behavior_table()

    by_id = {row.step_id: row for row in table}
    assert by_id["discover_traces"].python_status == "implemented"
    assert by_id["switch_mechanism"].python_status == "not_executed_offline"
    assert by_id["reset_replicas"].python_status == "not_executed_offline"
    assert by_id["dispatch_trace"].python_status == "planned"
    assert by_id["compare_plots"].python_status == "planned"
