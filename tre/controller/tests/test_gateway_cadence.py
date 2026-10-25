"""Declared gateway cadence (review round 1, M4/L6): the gateway declares its TRE doc
periods in ``tre:v2:gateway:cadence``; the metrics store compares it with the
controller's on every snapshot. Missing or different = the snapshot is stale (fail mode)
- never a crash, and never an inference from doc spacing."""
from __future__ import annotations

import logging

import pytest

from test_signal_validity_20261004 import _Redis, _registry
from tre_controller.store.metrics_store import MetricsStore

KEY = "tre:v2:gateway:cadence"


@pytest.mark.parametrize(
    "declared, enforce, stale",
    [
        ({"instant_ms": "500", "evidence_ms": "10000", "written_ms": "1", "version": "abc"}, True, False),
        ({"instant_ms": "10000", "evidence_ms": "10000", "version": "old"}, True, True),
        (None, True, True),
        ({"instant_ms": "500", "evidence_ms": "2000"}, False, False),
    ],
    ids=["match", "mismatch", "missing", "mismatch_warn_mode"],
)
def test_a_cadence_mismatch_marks_the_snapshot_stale(declared, enforce, stale, caplog) -> None:
    redis = _Redis()
    if declared is not None:
        redis.hashes[KEY] = declared
    store = MetricsStore(
        redis, _registry("m"), instant_sample_interval_ms=500, evidence_period_ms=10_000,
        expected_cadence=(500, 10_000), cadence_enforce=enforce,
    )
    with caplog.at_level(logging.ERROR, logger="tre_controller.store.metrics_store"):
        snapshot = store.read_snapshot(1_000_000, 1_030_000, use_cache=False, start_exclusive=True)
    assert snapshot.stale is stale
    assert (snapshot.stale_reason is None) is (declared is not None and declared.get("instant_ms") == "500"
                                               and declared.get("evidence_ms") == "10000")
    assert ("GATEWAY CADENCE" in caplog.text) is (snapshot.stale_reason is not None)
