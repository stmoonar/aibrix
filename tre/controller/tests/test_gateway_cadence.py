"""D8 startup assertion: gateway write period == SCRAPE_INTERVAL_MS."""
from __future__ import annotations

import json
import logging

import pytest

from tre_controller.gateway_cadence import GatewayCadenceMismatch, check_gateway_cadence

BASE = 1_790_000_000_000


class _Redis:
    def __init__(self) -> None:
        self.strings: dict = {}
        self.sets: dict = {}
        self.zsets: dict = {}

    def scan_iter(self, match, count=None):
        prefix = match[:-1]
        return iter([k for k in sorted(self.strings) if k.startswith(prefix)])

    def get(self, key):
        return self.strings.get(key)

    def smembers(self, key):
        return set(self.sets.get(key, set()))

    def zrange(self, key, start, end, withscores=False):
        items = self.zsets.get(key, [])
        sliced = items[start:] if end == -1 else items[start:end + 1]
        return [(m, s) for s, m in sliced]


def _with_ticks(redis: _Redis, spacing_ms: int) -> None:
    redis.sets["tre:v2:pods:m"] = {b"default/p"}
    redis.zsets["tre:v2:inst:default/p"] = [(float(BASE + i * spacing_ms), f"doc{i}") for i in range(10)]


def _with_trace(redis: _Redis, interval_s: int, *, key: str = "meta_interval_sec") -> None:
    redis.strings[f"aibrix:m_request_trace_{BASE // 1000}"] = json.dumps({key: interval_s, "meta_v": 4})
    redis.strings[f"aibrix:m_request_trace_{BASE // 1000 - 10}"] = json.dumps({key: 99})


def test_matching_trace_metadata_and_ticks_pass(caplog) -> None:
    redis = _Redis()
    _with_trace(redis, 10)
    _with_ticks(redis, 10_000)
    with caplog.at_level(logging.INFO, logger="tre_controller.gateway_cadence"):
        evidence = check_gateway_cadence(redis, ["m"])
    assert evidence.trace_interval_s == {"m": 10.0}  # the newest trace wins
    assert evidence.inst_spacing_ms == {"default/p": 10_000.0}
    assert "gateway_cadence_ok" in caplog.text


def test_trace_interval_alias_is_accepted() -> None:
    redis = _Redis()
    _with_trace(redis, 10, key="interval_in_seconds")
    assert check_gateway_cadence(redis, ["m"]).trace_interval_s == {"m": 10.0}


def test_trace_metadata_mismatch_is_fatal() -> None:
    redis = _Redis()
    _with_trace(redis, 5)  # a gateway rebuilt from /root/aibrix-main (trace.go: 5 s)
    with pytest.raises(GatewayCadenceMismatch, match="meta_interval_sec=5"):
        check_gateway_cadence(redis, ["m"])


def test_tick_spacing_mismatch_is_fatal_and_warn_mode_only_logs(caplog) -> None:
    redis = _Redis()
    _with_ticks(redis, 5_000)
    with pytest.raises(GatewayCadenceMismatch, match="median spacing 5000"):
        check_gateway_cadence(redis, ["m"])
    with caplog.at_level(logging.ERROR, logger="tre_controller.gateway_cadence"):
        check_gateway_cadence(redis, ["m"], mode="warn")
    assert "GATEWAY CADENCE MISMATCH" in caplog.text


def test_no_evidence_warns_loudly_but_does_not_crash(caplog) -> None:
    with caplog.at_level(logging.WARNING, logger="tre_controller.gateway_cadence"):
        check_gateway_cadence(_Redis(), ["m"])
    assert "GATEWAY CADENCE UNVERIFIED" in caplog.text


def test_ticks_without_trace_metadata_warn(caplog) -> None:
    redis = _Redis()
    _with_ticks(redis, 10_000)
    with caplog.at_level(logging.WARNING, logger="tre_controller.gateway_cadence"):
        check_gateway_cadence(redis, ["m"])
    assert "TRACE METADATA UNAVAILABLE" in caplog.text


def test_redis_errors_are_not_fatal(caplog) -> None:
    class Broken:
        def scan_iter(self, *a, **k):
            raise ConnectionError("down")

        def smembers(self, key):
            raise ConnectionError("down")

    with caplog.at_level(logging.WARNING, logger="tre_controller.gateway_cadence"):
        evidence = check_gateway_cadence(Broken(), ["m"])
    assert evidence.errors and "UNVERIFIED" in caplog.text


def test_off_mode_skips() -> None:
    assert check_gateway_cadence(_Redis(), ["m"], mode="off") is None
