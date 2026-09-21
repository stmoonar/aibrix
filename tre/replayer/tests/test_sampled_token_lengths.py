"""Per-request token-length sampling (the T9 shape) and what it must not break.

A sampled shape is only worth having if it keeps the three properties the calibration
depends on - exact realised length, determinism, and a distinct prompt per request - while
adding the one it exists for: variance in the generation length, which is what produces a
decode tail. Each of those is a test here.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from tre_replayer.engine.prompts import MODE_NATURAL, build_prompt
from tre_replayer.engine.schedule import (
    RpsSegment,
    TokenRange,
    build_deterministic_schedule,
    build_poisson_schedule,
)
from tre_replayer.traces.loader import load_trace_segments


class FakeTokenizer:
    """One token per whitespace word plus a BOS, which is enough to exercise the fit."""

    overhead = 1
    path = "fake"
    filler = " the"

    def encode_plain(self, text: str) -> list[int]:
        return [hash(word) % 1000 for word in text.split()]

    def decode_plain(self, ids) -> str:
        return " ".join("w%d" % i for i in ids)

    def count(self, text: str) -> int:
        return len(self.encode_plain(text)) + self.overhead


IN_RANGE = TokenRange(300, 2200)
OUT_RANGE = TokenRange(100, 580)


def _segment(**over) -> RpsSegment:
    body = dict(
        model="dsqwen-7b", start_s=0.0, end_s=60.0, rps=8.0,
        input_tokens_range=IN_RANGE, max_output_tokens_range=OUT_RANGE,
    )
    body.update(over)
    return RpsSegment(**body)


# ------------------------------------------------------------------ the distribution


def test_log_uniform_draws_stay_inside_the_range_and_spread_over_it() -> None:
    import random

    rng = random.Random(7)
    draws = [IN_RANGE.sample(rng) for _ in range(2000)]
    assert all(IN_RANGE.low <= d <= IN_RANGE.high for d in draws)
    # log-uniform, not uniform: the lower half of the log span holds about half the mass,
    # whereas a uniform draw over the same interval would put ~30 % there.
    midpoint = math.sqrt(IN_RANGE.low * IN_RANGE.high)
    below = sum(1 for d in draws if d < midpoint) / len(draws)
    assert 0.45 < below < 0.55


def test_the_mean_is_the_distribution_mean_not_the_midpoint() -> None:
    # Capacity and KV footprint are linear in length, so the mean is what sizes them.
    assert IN_RANGE.mean == pytest.approx((2200 - 300) / math.log(2200 / 300))
    assert IN_RANGE.mean != pytest.approx((2200 + 300) / 2)
    assert IN_RANGE.median == round(math.sqrt(300 * 2200))


def test_a_degenerate_range_is_just_a_fixed_length() -> None:
    import random

    point = TokenRange(512, 512)
    assert point.sample(random.Random(1)) == 512
    assert point.mean == 512.0


def test_an_impossible_range_is_a_loud_failure() -> None:
    with pytest.raises(ValueError, match="high must be >= low"):
        TokenRange(500, 100)
    with pytest.raises(ValueError, match="unknown token distribution"):
        TokenRange(100, 500, kind="pareto")


# ------------------------------------------------------------------ the schedule


def test_each_request_draws_its_own_length() -> None:
    events = build_poisson_schedule([_segment()], seed=11)
    assert len(events) > 50
    assert len({e.prompt_tokens for e in events}) > 20
    # the point of the shape: output lengths VARY, so the batch does not turn over in
    # lockstep and a decode tail exists at all
    assert len({e.max_output_tokens for e in events}) > 20
    assert all(IN_RANGE.low <= e.prompt_tokens <= IN_RANGE.high for e in events)
    assert all(OUT_RANGE.low <= e.max_output_tokens <= OUT_RANGE.high for e in events)


def test_the_same_seed_reproduces_the_same_lengths_request_for_request() -> None:
    a = build_poisson_schedule([_segment()], seed=1234)
    b = build_poisson_schedule([_segment()], seed=1234)
    assert [(e.request_id, e.prompt_tokens, e.max_output_tokens) for e in a] == [
        (e.request_id, e.prompt_tokens, e.max_output_tokens) for e in b
    ]
    c = build_poisson_schedule([_segment()], seed=5678)
    assert [e.prompt_tokens for e in c] != [e.prompt_tokens for e in a]


def test_sampling_one_segment_does_not_move_another_segments_arrivals() -> None:
    # Token draws come from a separate per-segment stream. Sharing one would make every
    # arrival downstream of a sampled segment depend on how many draws happened, so
    # adding variance to one shape would silently reshape all the others.
    fixed = RpsSegment("dsqwen-7b", 0.0, 60.0, 5.0, input_tokens=256, max_output_tokens=128)
    alone = build_poisson_schedule([fixed], seed=99)
    with_sampled = build_poisson_schedule([fixed, _segment(start_s=60.0, end_s=120.0)], seed=99)
    first = [e for e in with_sampled if e.scheduled_offset_s < 60.0]
    assert [e.scheduled_offset_s for e in first] == [e.scheduled_offset_s for e in alone]


def test_a_fixed_segment_is_untouched_by_the_sampling_path() -> None:
    fixed = RpsSegment("dsqwen-7b", 0.0, 10.0, 5.0, input_tokens=256, max_output_tokens=128)
    events = build_deterministic_schedule([fixed], seed=3)
    assert events
    assert {e.prompt_tokens for e in events} == {256}
    assert {e.max_output_tokens for e in events} == {128}
    assert not fixed.sampled


# ------------------------------------------------------------------ the trace schema


def test_the_loader_round_trips_a_distribution(tmp_path: Path) -> None:
    path = tmp_path / "trace.json"
    path.write_text(json.dumps({"dsqwen-7b": [{
        "start_time": 0, "end_time": 60, "rps": 4.0,
        "input_tokens_dist": IN_RANGE.as_dict(),
        "max_tokens_dist": OUT_RANGE.as_dict(),
    }]}), encoding="utf-8")
    segments = load_trace_segments(path)
    assert len(segments) == 1
    assert segments[0].input_tokens_range == IN_RANGE
    assert segments[0].max_output_tokens_range == OUT_RANGE
    assert segments[0].input_tokens is None


def test_a_segment_may_not_state_both_a_fixed_length_and_a_distribution(
    tmp_path: Path,
) -> None:
    # With both present nothing can tell a reader which one the run actually sent, and
    # the two would disagree in the index, the cell id and the raw log.
    path = tmp_path / "trace.json"
    path.write_text(json.dumps({"dsqwen-7b": [{
        "start_time": 0, "end_time": 60, "rps": 4.0,
        "input_tokens": 512, "input_tokens_dist": IN_RANGE.as_dict(),
    }]}), encoding="utf-8")
    with pytest.raises(ValueError, match="either fixed or sampled"):
        load_trace_segments(path)


# ------------------------------------------------------------------ the prompts


def test_a_sampled_request_still_gets_an_exactly_fitted_natural_prompt() -> None:
    # The sender keys the prompt on the request id and fits it to whatever length that
    # request drew, so a sampled shape keeps all three properties the fixed ones have.
    tokenizer = FakeTokenizer()
    events = build_poisson_schedule([_segment()], seed=17)[:12]
    prompts = []
    for event in events:
        text = build_prompt(
            event.prompt_tokens,
            f"{event.model}|{event.request_id}",
            mode=MODE_NATURAL,
            tokenizer=tokenizer,
        )
        # exact realised length, at whatever length this request drew
        assert tokenizer.count(text) == event.prompt_tokens
        prompts.append(text)
    # unique per request, and distinct inside the first few tokens (no shared prefix)
    assert len(set(prompts)) == len(prompts)
    heads = {p.split()[0] for p in prompts}
    assert len(heads) > 1


def test_a_sampled_requests_prompt_is_reproducible() -> None:
    tokenizer = FakeTokenizer()
    event = build_poisson_schedule([_segment()], seed=17)[0]
    key = f"{event.model}|{event.request_id}"
    first = build_prompt(event.prompt_tokens, key, mode=MODE_NATURAL, tokenizer=tokenizer)
    again = build_prompt(event.prompt_tokens, key, mode=MODE_NATURAL, tokenizer=tokenizer)
    assert first == again
