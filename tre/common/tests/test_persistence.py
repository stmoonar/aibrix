"""Round 4 invariants of the observation-time rules (docs/design/20261025-instant-sampling.md
"Evidence, not ticks"): a persistence verdict depends on observation times, not on how
many observations there are; a pod-set observation does not depend on the pod order.
Seeded, so deterministic."""
from __future__ import annotations

import random

from tre_common.metrics_schema import Observed
from tre_common.persistence import SPAN_TOLERANCE_MS, Persistence

SPAN = 10_000


def _timeline(rng: random.Random, *, step_ms: int = 500, length_ms: int = 120_000) -> list[tuple[int, bool]]:
    """A random on/off condition observed every ``step_ms`` (runs of 0.5 - 30 s)."""
    out: list[tuple[int, bool]] = []
    t, holds = 0, rng.random() < 0.5
    while t < length_ms:
        run_end = t + rng.randrange(500, 30_000, step_ms)
        while t < min(run_end, length_ms):
            out.append((t, holds))
            t += step_ms
        holds = not holds
    return out


def _verdicts(observations: list[tuple[int, bool]]) -> dict[int, bool]:
    rule = Persistence(SPAN)
    return {t: rule.observe(t, holds) for t, holds in observations}


def test_a_verdict_does_not_change_when_observations_are_thinned():
    # Dropping observations inside a true run (a longer evidence period, a missed round)
    # changes nothing at the observations that remain, as long as what the condition
    # did is still seen: every false observation and the first true one of each run.
    for seed in range(200):
        rng = random.Random(seed)
        full = _timeline(rng)
        kept = [
            (t, holds) for i, (t, holds) in enumerate(full)
            if not holds or i == 0 or not full[i - 1][1] or rng.random() < 0.3
        ]
        dense, thin = _verdicts(full), _verdicts(kept)
        assert {t: dense[t] for t, _ in kept} == thin, seed


def test_the_fire_time_moves_by_at_most_the_sampling_gap_not_by_the_count():
    # A condition that starts at t0 and keeps holding: whatever the period (0.5 - 10 s)
    # and phase, it fires between t0 + span and t0 + span + 2 periods (first sample up
    # to one period late, the firing sample up to one period after the span).
    for seed in range(100):
        rng = random.Random(seed)
        t0 = rng.randrange(0, 20_000)
        for period in (500, 2_000, 5_000, 10_000):
            first = t0 + rng.randrange(0, period)
            rule = Persistence(SPAN)
            fired = next(t for t in range(first, first + 10 * SPAN, period) if rule.observe(t, True))
            assert t0 + SPAN - SPAN_TOLERANCE_MS <= fired <= t0 + SPAN + 2 * period, (seed, period)


def test_a_re_read_of_the_same_observation_is_not_new_evidence():
    rule = Persistence(SPAN)
    assert rule.observe(0, True) is False
    for _ in range(50):  # the same snapshot read on every tick
        assert rule.observe(0, True) is False
    assert rule.held_ms == 0
    assert rule.observe(SPAN, True) is True


def test_a_false_verdict_resets_even_at_an_observation_time_already_seen():
    # Round 5 (L3): the same snapshot re-read once its verdict turned false (another
    # gate closed) must not keep the earlier span alive.
    rule = Persistence(SPAN)
    assert rule.observe(0, True) is False
    assert rule.observe(SPAN - 2_000, True) is False
    assert rule.observe(SPAN - 2_000, False) is False
    assert rule.held_ms == 0
    assert rule.observe(SPAN, True) is False  # a new span starts here


def test_a_pod_set_observation_does_not_depend_on_the_pod_order():
    for seed in range(200):
        rng = random.Random(seed)
        items = [
            None if rng.random() < 0.1 else Observed(
                t_first_ms=None if rng.random() < 0.1 else rng.randrange(0, 50_000),
                t_last_ms=None if rng.random() < 0.1 else rng.randrange(50_000, 100_000),
            )
            for _ in range(rng.randrange(1, 8))
        ]
        merged = Observed.merge(items)
        for _ in range(5):
            rng.shuffle(items)
            assert Observed.merge(items) == merged, seed
        # Merging in steps gives the same as merging at once.
        if len(items) > 2:
            assert Observed.merge([Observed.merge(items[:2]), *items[2:]]) == merged, seed
