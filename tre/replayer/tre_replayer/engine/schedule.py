from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RpsSegment:
    model: str
    start_s: float
    end_s: float
    rps: float


@dataclass(frozen=True)
class ScheduledRequest:
    request_id: str
    model: str
    scheduled_offset_s: float
    prompt: str = ""
    max_output_tokens: int | None = None


def build_deterministic_schedule(segments: list[RpsSegment]) -> list[ScheduledRequest]:
    counters: dict[str, int] = {}
    events: list[ScheduledRequest] = []
    for segment in segments:
        if segment.rps <= 0.0 or segment.end_s <= segment.start_s:
            continue
        interval_s = 1.0 / segment.rps
        offset_s = segment.start_s
        while offset_s < segment.end_s - 1e-12:
            idx = counters.get(segment.model, 0)
            counters[segment.model] = idx + 1
            events.append(
                ScheduledRequest(
                    request_id=f"{segment.model}-{idx:06d}",
                    model=segment.model,
                    scheduled_offset_s=round(offset_s, 9),
                )
            )
            offset_s += interval_s
    return sorted(events, key=lambda event: event.scheduled_offset_s)
