"""D8 startup assertion: the gateway's instant / histogram doc periods equal the
controller's TRE_INSTANT_SAMPLE_INTERVAL_MS / TRE_EVIDENCE_PERIOD_MS (2026-10-25)."""
from __future__ import annotations

import logging

import pytest

from tre_controller.gateway_cadence import GatewayCadenceMismatch, check_gateway_cadence

BASE = 1_790_000_000_000


class _Redis:
    def __init__(self) -> None:
        self.sets: dict = {}
        self.zsets: dict = {}

    def smembers(self, key):
        return set(self.sets.get(key, set()))

    def zrange(self, key, start, end, withscores=False):
        items = self.zsets.get(key, [])
        sliced = items[start:] if end == -1 else items[start:end + 1]
        return [(m, s) for s, m in sliced]


def _with_ticks(redis: _Redis, inst_ms: int, hist_ms: int) -> None:
    redis.sets["tre:v2:pods:m"] = {b"default/p"}
    redis.zsets["tre:v2:inst:default/p"] = [(float(BASE + i * inst_ms), f"i{i}") for i in range(10)]
    redis.zsets["tre:v2:hist:default/p"] = [(float(BASE + i * hist_ms), f"h{i}") for i in range(10)]


def test_matching_periods_pass(caplog) -> None:
    redis = _Redis()
    _with_ticks(redis, 500, 10_000)
    with caplog.at_level(logging.INFO, logger="tre_controller.gateway_cadence"):
        evidence = check_gateway_cadence(redis, ["m"], expected_ms=500, expected_hist_ms=10_000)
    assert evidence.inst_spacing_ms == {"default/p": 500.0}
    assert evidence.hist_spacing_ms == {"default/p": 10_000.0}
    assert "gateway_cadence_ok" in caplog.text


def test_old_gateway_instant_cadence_is_fatal_and_warn_mode_only_logs(caplog) -> None:
    redis = _Redis()
    _with_ticks(redis, 10_000, 10_000)  # a gateway image before 2026-10-25
    with pytest.raises(GatewayCadenceMismatch, match="instant ticks of default/p: median spacing 10000"):
        check_gateway_cadence(redis, ["m"], expected_ms=500, expected_hist_ms=10_000)
    with caplog.at_level(logging.ERROR, logger="tre_controller.gateway_cadence"):
        check_gateway_cadence(redis, ["m"], mode="warn", expected_ms=500, expected_hist_ms=10_000)
    assert "GATEWAY CADENCE MISMATCH" in caplog.text


def test_histogram_period_mismatch_is_fatal() -> None:
    redis = _Redis()
    _with_ticks(redis, 500, 2_000)
    with pytest.raises(GatewayCadenceMismatch, match="TRE_EVIDENCE_PERIOD_MS=10000"):
        check_gateway_cadence(redis, ["m"], expected_ms=500, expected_hist_ms=10_000)


def test_no_evidence_warns_loudly_but_does_not_crash(caplog) -> None:
    with caplog.at_level(logging.WARNING, logger="tre_controller.gateway_cadence"):
        check_gateway_cadence(_Redis(), ["m"])
    assert "GATEWAY CADENCE UNVERIFIED" in caplog.text


def test_redis_errors_are_not_fatal(caplog) -> None:
    class Broken:
        def smembers(self, key):
            raise ConnectionError("down")

    with caplog.at_level(logging.WARNING, logger="tre_controller.gateway_cadence"):
        evidence = check_gateway_cadence(Broken(), ["m"])
    assert evidence.errors and "UNVERIFIED" in caplog.text


def test_off_mode_skips() -> None:
    assert check_gateway_cadence(_Redis(), ["m"], mode="off") is None
