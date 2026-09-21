"""Arrival schedules, and the token shape each arrival carries.

A segment usually fixes one token shape for every request it produces. It may instead
carry a :class:`TokenRange`, and then each request draws its own length - which is what
a shape with *variance* in its output length needs, because a fixed ``max_tokens``
produces no decode tail at all: every request finishes after the same number of
steps, so the engine never has a few long generations holding batch slots while short
ones churn. The draws come from a per-segment RNG seeded from the schedule's own seed,
so a replay reproduces the same lengths request for request.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Any, Iterable, Mapping

#: The only sampled distribution implemented. Log-uniform rather than uniform because
#: request lengths in real traces span an order of magnitude and are roughly flat in log
#: space; a uniform draw over the same interval would put most of its mass at the long
#: end and quietly turn a "mixed" shape into a heavy one.
DIST_LOG_UNIFORM = "log_uniform"
DISTRIBUTIONS = (DIST_LOG_UNIFORM,)


@dataclass(frozen=True)
class TokenRange:
    """A per-request token length drawn from ``kind`` over ``[low, high]``."""

    low: int
    high: int
    kind: str = DIST_LOG_UNIFORM

    def __post_init__(self) -> None:
        if self.kind not in DISTRIBUTIONS:
            raise ValueError(f"unknown token distribution {self.kind!r} (expected {DISTRIBUTIONS})")
        if self.low < 1:
            raise ValueError("token range low must be >= 1")
        if self.high < self.low:
            raise ValueError("token range high must be >= low")

    def sample(self, rng: random.Random) -> int:
        """One draw, rounded to a whole token and clamped back into the range."""
        if self.low == self.high:
            return int(self.low)
        drawn = math.exp(rng.uniform(math.log(self.low), math.log(self.high)))
        return int(min(self.high, max(self.low, round(drawn))))

    @property
    def mean(self) -> float:
        """E[X]. For log-uniform that is ``(high - low) / ln(high / low)``.

        This - not the median - is what capacity sizing needs: the campaign's capacity
        model is ``1/C = i/P + o/D``, linear in both lengths, so the shape saturates the
        pod at the rate set by the *mean* length.
        """
        if self.low == self.high:
            return float(self.low)
        return (self.high - self.low) / math.log(self.high / self.low)

    @property
    def median(self) -> int:
        """Geometric midpoint. Used only to give the shape a stable, parseable cell id."""
        return int(round(math.sqrt(self.low * self.high)))

    def as_dict(self) -> dict:
        return {"kind": self.kind, "low": int(self.low), "high": int(self.high)}

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "TokenRange":
        return cls(
            low=int(raw["low"]),
            high=int(raw["high"]),
            kind=str(raw.get("kind", DIST_LOG_UNIFORM)),
        )


@dataclass(frozen=True)
class RpsSegment:
    model: str
    start_s: float
    end_s: float
    rps: float
    input_tokens: int | None = None
    max_output_tokens: int | None = None
    #: When set, every request of this segment draws its own prompt length from here and
    #: ``input_tokens`` is ignored (the two are mutually exclusive at load time).
    input_tokens_range: TokenRange | None = None
    #: Same, for the generation length.
    max_output_tokens_range: TokenRange | None = None

    @property
    def sampled(self) -> bool:
        return self.input_tokens_range is not None or self.max_output_tokens_range is not None


@dataclass(frozen=True)
class ScheduledRequest:
    request_id: str
    model: str
    scheduled_offset_s: float
    prompt: str = ""
    prompt_tokens: int | None = None
    max_output_tokens: int | None = None


def build_deterministic_schedule(
    segments: Iterable[RpsSegment], *, seed: int = 0
) -> list[ScheduledRequest]:
    counters: dict[str, int] = {}
    events: list[ScheduledRequest] = []
    for index, segment in enumerate(segments):
        if segment.rps <= 0.0 or segment.end_s <= segment.start_s:
            continue
        token_rng = _token_rng(seed, index)
        interval_s = 1.0 / segment.rps
        offset_s = segment.start_s
        while offset_s < segment.end_s - 1e-12:
            events.append(_event_for_segment(segment, counters, offset_s, token_rng))
            offset_s += interval_s
    return sorted(events, key=lambda event: event.scheduled_offset_s)


def build_poisson_schedule(segments: Iterable[RpsSegment], *, seed: int | None = None) -> list[ScheduledRequest]:
    rng = random.Random(seed)
    counters: dict[str, int] = {}
    events: list[ScheduledRequest] = []
    for index, segment in enumerate(segments):
        if segment.rps <= 0.0 or segment.end_s <= segment.start_s:
            continue
        # Token lengths are drawn from a SEPARATE per-segment stream, never from `rng`.
        # Sharing one stream would make every arrival time downstream of a sampled
        # segment depend on how many token draws happened, so adding variance to one
        # shape would silently move the arrival pattern of all the others.
        token_rng = _token_rng(seed, index)
        offset_s = segment.start_s
        while True:
            offset_s += rng.expovariate(segment.rps)
            if offset_s >= segment.end_s:
                break
            events.append(_event_for_segment(segment, counters, offset_s, token_rng))
    return sorted(events, key=lambda event: event.scheduled_offset_s)


#: Stride between per-segment token streams. A large prime, so two (seed, index) pairs
#: only collide when they are the same pair.
_TOKEN_SEED_STRIDE = 1_000_003


def _token_rng(seed: int | None, segment_index: int) -> random.Random:
    """Per-segment token stream. ``seed`` None keeps the caller's "unseeded" intent.

    The derived seed is an int rather than a ``(seed, index)`` tuple: seeding a
    :class:`random.Random` from a tuple goes through ``hash``, which Python deprecated in
    3.9 and which is salted per process for some types - the opposite of reproducible.
    """
    if seed is None:
        return random.Random(None)
    return random.Random(int(seed) * _TOKEN_SEED_STRIDE + int(segment_index))


def _event_for_segment(
    segment: RpsSegment,
    counters: dict[str, int],
    offset_s: float,
    token_rng: random.Random,
) -> ScheduledRequest:
    idx = counters.get(segment.model, 0)
    counters[segment.model] = idx + 1
    prompt_tokens = segment.input_tokens
    if segment.input_tokens_range is not None:
        prompt_tokens = segment.input_tokens_range.sample(token_rng)
    max_output_tokens = segment.max_output_tokens
    if segment.max_output_tokens_range is not None:
        max_output_tokens = segment.max_output_tokens_range.sample(token_rng)
    return ScheduledRequest(
        request_id=f"{segment.model}-{idx:06d}",
        model=segment.model,
        scheduled_offset_s=round(offset_s, 9),
        prompt_tokens=prompt_tokens,
        max_output_tokens=max_output_tokens,
    )
