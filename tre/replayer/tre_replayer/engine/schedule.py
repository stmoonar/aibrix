from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class RpsSegment:
    model: str
    start_s: float
    end_s: float
    rps: float
    input_tokens: int | None = None
    max_output_tokens: int | None = None


@dataclass(frozen=True)
class ScheduledRequest:
    request_id: str
    model: str
    scheduled_offset_s: float
    prompt: str = ""
    prompt_tokens: int | None = None
    max_output_tokens: int | None = None


def build_deterministic_schedule(segments: Iterable[RpsSegment]) -> list[ScheduledRequest]:
    counters: dict[str, int] = {}
    events: list[ScheduledRequest] = []
    for segment in segments:
        if segment.rps <= 0.0 or segment.end_s <= segment.start_s:
            continue
        interval_s = 1.0 / segment.rps
        offset_s = segment.start_s
        while offset_s < segment.end_s - 1e-12:
            events.append(_event_for_segment(segment, counters, offset_s))
            offset_s += interval_s
    return sorted(events, key=lambda event: event.scheduled_offset_s)


def build_poisson_schedule(segments: Iterable[RpsSegment], *, seed: int | None = None) -> list[ScheduledRequest]:
    rng = random.Random(seed)
    counters: dict[str, int] = {}
    events: list[ScheduledRequest] = []
    for segment in segments:
        if segment.rps <= 0.0 or segment.end_s <= segment.start_s:
            continue
        offset_s = segment.start_s
        while True:
            offset_s += rng.expovariate(segment.rps)
            if offset_s >= segment.end_s:
                break
            events.append(_event_for_segment(segment, counters, offset_s))
    return sorted(events, key=lambda event: event.scheduled_offset_s)


def _event_for_segment(
    segment: RpsSegment,
    counters: dict[str, int],
    offset_s: float,
) -> ScheduledRequest:
    idx = counters.get(segment.model, 0)
    counters[segment.model] = idx + 1
    return ScheduledRequest(
        request_id=f"{segment.model}-{idx:06d}",
        model=segment.model,
        scheduled_offset_s=round(offset_s, 9),
        prompt_tokens=segment.input_tokens,
        max_output_tokens=segment.max_output_tokens,
    )
