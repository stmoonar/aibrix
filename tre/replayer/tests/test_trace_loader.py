from __future__ import annotations

import json

from tre_replayer.traces.loader import load_trace_segments


def test_load_trace_segments_reads_existing_model_keyed_json(tmp_path) -> None:
    src = tmp_path / "trace.json"
    src.write_text(json.dumps({
        "dsqwen-7b": [
            {"start_time": 0, "end_time": 2, "rps": 2.0, "input_tokens": 1000, "max_tokens": 600},
        ],
        "dsllama-8b": [
            {"start_time": 1, "end_time": 3, "rps": 1.5, "input_tokens": 500, "max_tokens": 400},
        ],
    }), encoding="utf-8")

    segments = load_trace_segments(src)

    assert [(segment.model, segment.start_s, segment.end_s, segment.rps) for segment in segments] == [
        ("dsqwen-7b", 0.0, 2.0, 2.0),
        ("dsllama-8b", 1.0, 3.0, 1.5),
    ]
    assert segments[0].input_tokens == 1000
    assert segments[0].max_output_tokens == 600
    assert segments[1].input_tokens == 500
    assert segments[1].max_output_tokens == 400
