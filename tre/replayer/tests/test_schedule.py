from __future__ import annotations

from tre_replayer.engine.schedule import RpsSegment, build_deterministic_schedule


def test_build_deterministic_schedule_uses_absolute_offsets_and_half_open_segments() -> None:
    schedule = build_deterministic_schedule([
        RpsSegment(model="dsqwen-7b", start_s=0.0, end_s=2.0, rps=2.0),
        RpsSegment(model="dsllama-8b", start_s=1.0, end_s=2.0, rps=1.0),
    ])

    assert [(event.request_id, event.model, event.scheduled_offset_s) for event in schedule] == [
        ("dsqwen-7b-000000", "dsqwen-7b", 0.0),
        ("dsqwen-7b-000001", "dsqwen-7b", 0.5),
        ("dsqwen-7b-000002", "dsqwen-7b", 1.0),
        ("dsllama-8b-000000", "dsllama-8b", 1.0),
        ("dsqwen-7b-000003", "dsqwen-7b", 1.5),
    ]
