from pathlib import Path

from deploy.scripts.analysis.audit_node_placement import (
    PowerEvent,
    RunBound,
    audit_placement,
    load_power_events,
)


def test_audit_node_placement_replays_power_events_and_flags_asymmetry(tmp_path: Path):
    events = [
        PowerEvent(0.0, "m-node9", "dsqwen-7b", "nscc-ds-4a100-node9", (0,), True),
        PowerEvent(0.0, "m-node10", "dsqwen-7b", "nscc-ds-4a100-node10", (0,), False),
        PowerEvent(15.0, "m-node10", "dsqwen-7b", "nscc-ds-4a100-node10", (0,), True),
    ]
    timelines = {
        "tre": [{"ts": "1970-01-01T00:00:10Z", "awake": "dsqwen-7b=1"}],
        "apa": [{"ts": "1970-01-01T00:00:20Z", "awake": "dsqwen-7b=2"}],
    }
    bounds = [
        RunBound(f"t{index}", arm, 0.0, 30.0)
        for index in range(1, 8)
        for arm in ("tre", "apa")
    ]

    placement, summary, verdicts = audit_placement(
        timelines=timelines,
        bounds=bounds,
        power_events=events,
        output_dir=tmp_path,
        max_share_diff=0.10,
    )

    assert len(placement) == 7 * 2 * 3
    assert len(summary) == 14
    assert all(row["verdict"] == "FLAG" for row in verdicts)
    assert all(row["reason"] == "node10_share_diff;node10_coresidency_asymmetry" for row in verdicts)
    assert (tmp_path / "placement_audit.csv").is_file()


def test_load_power_events_accepts_nanoseconds_and_hyphenated_gpu_ids(tmp_path: Path):
    path = tmp_path / "events.csv"
    path.write_text(
        "timestamp,serve_id,model,node,gpu_ids,state,source\n"
        "2026-07-07T18:40:32.354230324Z,pod-a,dsqwen-14b,"
        "nscc-ds-4a100-node10,0-1,awake,vllm_wake_complete\n"
    )

    events = load_power_events(path)

    assert len(events) == 1
    assert events[0].gpu_ids == (0, 1)
    assert events[0].awake is True