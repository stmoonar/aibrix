from __future__ import annotations

import json
from pathlib import Path

from tre_replayer.engine.schedule import build_deterministic_schedule
from tre_replayer.run_trace import run_trace
from tre_replayer.traces.loader import discover_trace_set

TRACE_SET_DIR = Path(__file__).resolve().parents[1] / "traces_v1paper"

EXPECTED_WORKLOADS = (
    "Alternating_hot_model_periodic_A",
    "Decode_heavy_burst",
    "Prefill_mixed_corner_decode_mix",
    "Simultaneous_spike_ramp_twice_tps1o2",
    "Sinusoidal_demand",
    "Real_code_2024_slice_a_tok70",
    "Real_conv_2023_slice_a_tok70",
)

EXPECTED_MODELS = {"dsllama-8b", "dsqwen-7b", "dsqwen-14b"}

#: request_count from conversion_manifest.json, i.e. the exact v1 traces.json count
#: (== the deterministic-schedule request count; see docs/v1-paper-trace-conversion.md).
EXPECTED_REQUEST_COUNTS = {
    "Alternating_hot_model_periodic_A": 12330,
    "Decode_heavy_burst": 21742,
    "Prefill_mixed_corner_decode_mix": 10430,
    "Simultaneous_spike_ramp_twice_tps1o2": 5120,
    "Sinusoidal_demand": 21632,
    "Real_code_2024_slice_a_tok70": 43083,
    "Real_conv_2023_slice_a_tok70": 45244,
}


async def _instant_sleep(_seconds: float) -> None:
    return None


def test_index_lists_all_seven_v1_paper_traces() -> None:
    trace_set = discover_trace_set(TRACE_SET_DIR)
    assert trace_set.version == "traceset-v1paper"
    names = {case.name for case in trace_set.cases}
    assert names == set(EXPECTED_WORKLOADS)


def test_each_trace_loads_all_three_fleet_models_with_valid_segments() -> None:
    trace_set = discover_trace_set(TRACE_SET_DIR)
    for case in trace_set.cases:
        models = {segment.model for segment in case.segments}
        assert models == EXPECTED_MODELS, case.name
        for segment in case.segments:
            assert segment.end_s > segment.start_s
            assert segment.rps > 0
            has_fixed = segment.input_tokens is not None and segment.max_output_tokens is not None
            has_dist = segment.input_tokens_range is not None and segment.max_output_tokens_range is not None
            assert has_fixed or has_dist, (case.name, segment)


def test_deterministic_schedule_reproduces_exact_v1_request_count() -> None:
    """Deterministic (non-random) replay of the 1s-binned rps segments reconstructs
    exactly the number of requests v1's traces.json planned for each trace -- the
    conversion's core lossless property (see docs/v1-paper-trace-conversion.md)."""
    trace_set = discover_trace_set(TRACE_SET_DIR)
    for case in trace_set.cases:
        schedule = build_deterministic_schedule(case.segments)
        assert len(schedule) == EXPECTED_REQUEST_COUNTS[case.name], case.name


def test_conversion_manifest_matches_trace_set() -> None:
    manifest = json.loads((TRACE_SET_DIR / "conversion_manifest.json").read_text(encoding="utf-8"))
    names = {entry["trace_name"] for entry in manifest}
    assert names == set(EXPECTED_WORKLOADS)
    for entry in manifest:
        assert entry["request_count"] == EXPECTED_REQUEST_COUNTS[entry["trace_name"]]


def test_run_trace_dry_run_replays_a_v1_paper_trace_without_network() -> None:
    """run_trace's whole pipeline (schedule -> dispatch -> score) exercises a converted
    v1-paper trace end to end with a fake sender and an instant fake clock -- no network,
    no real-time sleeping (item 5 of the conversion task: confirm the replayer can load
    and run the new trace set)."""
    trace_path = TRACE_SET_DIR / "Simultaneous_spike_ramp_twice_tps1o2" / "trace.json"

    summary = run_trace(
        str(trace_path),
        gateway_url="http://x",
        seed=20260924,
        dry_run=True,
        window_ms=30_000,
        step_ms=5_000,
        trim_ramp_windows=0,
        sleep=_instant_sleep,
    )

    # run_trace draws a *Poisson* schedule (build_poisson_schedule), not the
    # deterministic one used above, so the count is close to but not exactly the v1
    # request count (see docs/v1-paper-trace-conversion.md: <1% aggregate deviation
    # observed across all 7 traces).
    expected = EXPECTED_REQUEST_COUNTS["Simultaneous_spike_ramp_twice_tps1o2"]
    assert abs(summary["requests"] - expected) / expected < 0.05
    assert set(summary["per_model"]) == EXPECTED_MODELS
