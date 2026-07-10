from datetime import timedelta, timezone

import scripts.switch_microbench as microbench
from scripts.switch_microbench import (
    GpuSample,
    binding_layout,
    parse_engine_marker,
    parse_gpu_samples,
    percentile_nearest_rank,
    resolve_targets,
    summarize_inflight,
    transition_memory,
)


def test_parse_engine_marker_normalizes_node_clock_offset() -> None:
    text = (
        "2026-07-10T08:02:31.250000000Z "
        "\x1b[1;36m(EngineCore_0 pid=301)\x1b[0;0m INFO "
        "It took 0.601928 seconds to wake up tags {'weights'}.\n"
    )

    marker = parse_engine_marker(
        text,
        "wake",
        node_clock_offset_s=150.0,
        not_before_epoch=1783670400.0,
    )

    assert marker is not None
    epoch, duration, raw = marker
    assert epoch == 1783670401.25
    assert duration == 0.601928
    assert "wake up" in raw


def test_parse_engine_marker_rejects_old_or_wrong_direction() -> None:
    text = (
        "2026-07-10T08:02:31.250000000Z INFO "
        "It took 1.25 seconds to fall asleep.\n"
    )

    assert parse_engine_marker(
        text, "wake", node_clock_offset_s=0, not_before_epoch=0
    ) is None
    assert parse_engine_marker(
        text, "sleep", node_clock_offset_s=0, not_before_epoch=1783670600
    ) is None


def test_gpu_samples_and_transition_memory_sum_tp_gpus() -> None:
    text = """2026/07/10 16:02:30.000, 0, 100
2026/07/10 16:02:30.000, 1, 200
2026/07/10 16:02:34.000, 0, 10
2026/07/10 16:02:34.000, 1, 20
"""
    samples = parse_gpu_samples(
        text,
        utc_offset=timezone(timedelta(hours=8)),
        node_clock_offset_s=150.0,
    )

    before, after = transition_memory(
        samples,
        (0, 1),
        start_epoch=1783670401.0,
        ready_epoch=1783670403.0,
    )

    assert before == 300
    assert after == 30


def test_transition_memory_requires_complete_selected_gpu_sample() -> None:
    samples = [
        GpuSample(1.0, 0, 100),
        GpuSample(2.0, 0, 10),
        GpuSample(2.0, 1, 20),
    ]

    assert transition_memory(
        samples, (0, 1), start_epoch=1.5, ready_epoch=2.0
    ) == (None, 30)


def test_nearest_rank_and_inflight_summary() -> None:
    assert percentile_nearest_rank([1, 2, 3, 100], 0.99) == 100
    assert percentile_nearest_rank([], 0.99) is None
    rows = [
        {"ok": True, "latency_ms": 10},
        {"ok": False, "latency_ms": 50},
        {"ok": True, "latency_ms": 20},
    ]

    assert summarize_inflight(rows) == {
        "inflight_total": 3,
        "inflight_errors": 1,
        "inflight_p99_ms": 20.0,
    }


def test_resolve_targets_requires_awake_node9_binding(monkeypatch) -> None:
    state = {
        "bindings": [
            {
                "model": "m1",
                "serve_id": "serve-a",
                "node": "node9",
                "gpu_ids": [2, 3],
                "awake": True,
                "hidden": False,
            }
        ]
    }
    monkeypatch.setattr(
        microbench,
        "run_json",
        lambda _command: {"status": {"podIP": "10.0.0.1"}},
    )

    targets = resolve_targets(
        state, {"m1": "serve-a"}, namespace="default", expected_node="node9"
    )

    assert targets == [
        microbench.Target("m1", "serve-a", "node9", (2, 3), "10.0.0.1")
    ]

def test_binding_layout_ignores_version_and_order() -> None:
    first = {
        "version": 1,
        "bindings": [
            {"serve_id": "b", "model": "m", "node": "n", "gpu_ids": [1], "awake": False, "hidden": False},
            {"serve_id": "a", "model": "m", "node": "n", "gpu_ids": [0], "awake": True, "hidden": False},
        ],
    }
    second = {"version": 9, "bindings": list(reversed(first["bindings"]))}

    assert binding_layout(first) == binding_layout(second)