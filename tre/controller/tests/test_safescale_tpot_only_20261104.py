"""SafeScale judges TPOT only (user 2026-11-04, a design change).

Invariant: a probe's latency verdict is the window's TPOT p95 against the model's TPOT_slo.
A window whose TTFT is far over any TTFT_slo but whose TPOT is under TPOT_slo commits; the
same window with TPOT over TPOT_slo rolls back. Checked on both evidence paths (direct
scrapes, the live default; the Redis evidence window), with real histograms.
"""
from __future__ import annotations

import pytest

from test_safescale_direct_20260929 import HIDE, W, Harness
from test_safescale_evidence_20260929 import DocRedis, _obs, _real_machine

MODEL = "m"
#: 5 s first-token time: over every TTFT_slo of every mode (floors 450 / 500 ms, k * idle).
SLOW_TTFT_S = 5.0


@pytest.mark.parametrize("itl_s, want", [(0.02, "commit"), (2.0, "rollback")])
def test_direct_path_judges_tpot_only(itl_s, want) -> None:
    h = Harness()
    h.start(pre_hide_itl=0.02)
    at, decision = h.run_until(HIDE + W, serve=5, itl_s=itl_s, ttft_s=SLOW_TTFT_S)[-1]
    assert decision.status == want, decision
    if want == "commit":
        assert at == HIDE + W and decision.reason == "formal_commit_gate_passed"
        assert decision.details["latency_violations"] == []
    else:
        reason = decision.details["rollback_reason"]
        assert (reason["code"], reason["metrics"]) == ("slo_violation_direct", ["tpot"])
    assert not any("ttft" in key for key in decision.details if key != "ttft_count")


def _post_hide_docs(redis: DocRedis, *, slow_tpot: bool) -> None:
    """m-0 (the remaining pod) serves 15 requests per 10 s after the 110 s boundary, each
    with a 5 s TTFT; their TPOT is 10 ms (fast) or 500 ms (slow). m-1 is the probe pod."""
    for ts in range(70_000, 130_001, 10_000):
        done = 15 * max(0, (ts - 110_000) // 10_000)
        fast, slow = (0, done) if slow_tpot else (done, 0)
        redis.add("m-0", ts, fast=fast, slow=slow, prompt=0.0, ttft_s=SLOW_TTFT_S)
        redis.add("m-1", ts, fast=0, slow=0, prompt=0.0)


@pytest.mark.parametrize("slow_tpot, want", [(False, "commit"), (True, "rollback")])
def test_redis_evidence_path_judges_tpot_only(slow_tpot, want) -> None:
    redis = DocRedis()
    _post_hide_docs(redis, slow_tpot=slow_tpot)
    machine = _real_machine(redis)
    machine.observe(MODEL, _obs(110_000), now_ms=110_000)
    machine.observe(MODEL, _obs(120_000), now_ms=120_000)
    decision = machine.observe(MODEL, _obs(130_000), now_ms=130_000)
    assert decision.status == want, decision
    assert decision.details["latency_samples"] == 30
    assert decision.details["latency_violations"] == ([] if want == "commit" else ["tpot"])
