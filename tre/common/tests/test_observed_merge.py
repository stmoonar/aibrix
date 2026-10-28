"""Round 4 invariant of the observation-time rules (docs/design/20261025-instant-sampling.md
"Evidence, not ticks"): a pod-set observation does not depend on the pod order. Seeded,
so deterministic. (The Persistence span-rule tests went with the rule, 2026-10-29.)"""
from __future__ import annotations

import random

from tre_common.metrics_schema import Observed


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
