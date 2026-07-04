from __future__ import annotations

from tre_replayer.engine.schedule import RpsSegment, build_poisson_schedule


def test_build_poisson_schedule_is_seed_stable_and_bounded() -> None:
    segments = [RpsSegment(model="dsqwen-7b", start_s=10.0, end_s=12.0, rps=20.0, input_tokens=100, max_output_tokens=50)]

    first = build_poisson_schedule(segments, seed=1234)
    second = build_poisson_schedule(segments, seed=1234)
    third = build_poisson_schedule(segments, seed=4321)

    assert first == second
    assert first != third
    assert first
    assert all(10.0 <= event.scheduled_offset_s < 12.0 for event in first)
    assert [event.scheduled_offset_s for event in first] == sorted(event.scheduled_offset_s for event in first)
    assert all(event.prompt_tokens == 100 for event in first)
    assert all(event.max_output_tokens == 50 for event in first)
