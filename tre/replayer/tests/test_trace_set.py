from __future__ import annotations

import json

from tre_replayer.traces.loader import discover_trace_set


def test_discover_trace_set_reads_index_and_unindexed_trace_dirs(tmp_path) -> None:
    (tmp_path / "INDEX.json").write_text(json.dumps({"version": "trace_test", "workloads": ["A"]}), encoding="utf-8")
    for name in ["A", "B"]:
        trace_dir = tmp_path / name
        trace_dir.mkdir()
        (trace_dir / "trace.json").write_text(json.dumps({
            "dsqwen-7b": [{"start_time": 0, "end_time": 1, "rps": 1.0}],
        }), encoding="utf-8")

    trace_set = discover_trace_set(tmp_path)

    assert trace_set.version == "trace_test"
    assert [(case.name, case.indexed, len(case.segments)) for case in trace_set.cases] == [
        ("A", True, 1),
        ("B", False, 1),
    ]
