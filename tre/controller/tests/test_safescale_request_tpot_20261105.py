"""SafeScale's TPOT evidence and threshold (user 2026-11-05).

The probe judges the per-request TPOT histogram (``vllm:request_time_per_output_token_seconds``,
one sample per finished request; its count is n) with a ``bucket_upper`` p95 - whatever
TRE_PERCENTILE_MODE says - against E = the smallest bucket edge >= TPOT_slo, derived from
the edges the pods export. Invariants, on real /metrics text:

(a) a true p95 just under E (reported = E) is no violation; just over E is one;
(b) TPOT_slo 42 ms (14b) with a p95 in (25, 50] ms is no violation (E = 50);
(c) a remaining pod without the per-request series never lets the probe commit - also
    when it still exports the per-token ITL; the Redis path (whose gateway docs carry no
    per-request series) never commits;
(d) no edge >= TPOT_slo (or no edges): no threshold, no commit.
"""
from __future__ import annotations

import dataclasses

import pytest

from tre_common.slo import tpot_bucket_threshold_ms
from test_safescale_direct_20260929 import HIDE, ITL_LES, Harness, _cfg
from tre_controller.config import SafeScaleConfig
from tre_controller.planning.safescale import SafeScaleStateMachine
from tre_controller.planning.safescale_evidence import StaticThresholds

MODEL = "m"
DIRECT_CAP = HIDE + 60_000  # the W ceiling (confirmed hide + 60 s)


class _Harness(Harness):
    """The direct harness with a chosen TPOT_slo."""

    def __init__(self, tpot_slo_ms: float, **cfg) -> None:
        self._thresholds = StaticThresholds(tpot_slo_ms)
        super().__init__(**cfg)

    def _machine(self, cfg):
        return SafeScaleStateMachine(thresholds=self._thresholds, config=_cfg(**cfg), store=self.store,
                                     evidence=self.evidence, wall_clock_ms=self.clock)


def _run(h: Harness, *, tpot_s: float):
    h.start(pre_hide=20, pre_hide_itl=2.0)  # slow before the hide: never in the evidence
    return [d for _, d in h.run_until(DIRECT_CAP, serve=5, itl_s=tpot_s)]


# ============================================================ (a) exact at E
@pytest.mark.parametrize("tpot_s, want, reported", [(0.074, "commit", 75.0), (0.076, "rollback", 100.0)])
def test_a_p95_at_e_is_no_violation_and_just_above_e_is_one(tpot_s, want, reported) -> None:
    # TPOT_slo 75 -> E 75 ms. The p95 is the bucket's upper edge (an interpolated p95
    # would be 73.75 / 98.75 ms here): TRE_PERCENTILE_MODE never reaches the probe.
    assert "percentile_mode" not in {f.name for f in dataclasses.fields(SafeScaleConfig)}
    decisions = _run(_Harness(75.0), tpot_s=tpot_s)
    final = decisions[-1]
    assert final.status == want, final
    assert all(d.status != "commit" for d in decisions[:-1])
    if want == "commit":
        assert final.details["evidence_tpot_p95_ms"] == reported
        assert final.details["tpot_threshold_ms"] == 75.0 and final.details["latency_violations"] == []
    else:
        reason = final.details["rollback_reason"]
        assert (reason["code"], reason["tpot_p95_ms"], reason["tpot_threshold_ms"]) == (
            "slo_violation_direct", reported, 75.0)


# ============================================================ (b) the 14b quantization
@pytest.mark.parametrize("tpot_s", [0.026, 0.045, 0.05])
def test_a_42_ms_slo_passes_every_p95_in_the_25_to_50_bucket(tpot_s) -> None:
    final = _run(_Harness(42.0), tpot_s=tpot_s)[-1]
    assert final.status == "commit", final
    assert (final.details["tpot_slo_ms"], final.details["tpot_threshold_ms"]) == (42.0, 50.0)
    assert final.details["evidence_tpot_p95_ms"] == 50.0


def test_a_42_ms_slo_rolls_back_a_p95_above_50() -> None:
    final = _run(_Harness(42.0), tpot_s=0.051)[-1]
    assert final.status == "rollback" and final.details["rollback_reason"]["code"] == "slo_violation_direct"


# ============================================================ (c) the series is required
def test_a_pod_without_the_per_request_tpot_never_lets_the_probe_commit() -> None:
    # m-2 still exports TTFT and the per-token ITL, but not the per-request TPOT.
    h = _Harness(75.0)
    h.sims["m-2"].request_tpot = False
    decisions = _run(h, tpot_s=0.02)
    assert all(d.status != "commit" for d in decisions)
    assert decisions[-1].status == "rollback"
    assert h.machine.active_probe(MODEL).direct.last.missing.get("m-2") == "pending_baseline"


def test_the_series_vanishing_after_the_baseline_never_lets_the_probe_commit() -> None:
    h = _Harness(75.0)
    h.start()
    h.tick(HIDE + 2_000, serve=5)
    h.sims["m-2"].request_tpot = False
    decisions = [d for _, d in h.run_until(DIRECT_CAP, serve=5, first=HIDE + 4_000)]
    assert h.machine.active_probe(MODEL).direct.last.unanswered == {"m-2": "no_tpot_histogram"}
    assert all(d.status != "commit" for d in decisions)
    assert decisions[-1].status == "rollback"
    # (The Redis path: test_safescale_evidence_20260929::test_real_reader_never_commits_...)


# ============================================================ (d) E fail-closed
def test_e_is_the_smallest_edge_at_or_above_the_slo_and_none_without_one() -> None:
    assert [tpot_bucket_threshold_ms(slo, ITL_LES) for slo in (42.0, 50.0, 61.0, 63.0, 75.0)] == [
        50.0, 50.0, 75.0, 75.0, 75.0]
    assert tpot_bucket_threshold_ms(80_001.0, ITL_LES) is None  # above the largest finite edge
    assert tpot_bucket_threshold_ms(42.0, ()) is None and tpot_bucket_threshold_ms(42.0, None) is None
    assert tpot_bucket_threshold_ms(42.0, (float("inf"),)) is None  # +Inf is no threshold


def test_no_edge_at_or_above_the_slo_never_commits() -> None:
    decisions = _run(_Harness(100_000.0), tpot_s=0.02)  # healthy, but no threshold E
    assert all(d.status != "commit" for d in decisions)
    final = decisions[-1]
    assert final.status == "rollback"
    assert final.details["rollback_reason"]["code"] == "insufficient_evidence:tpot_threshold_unresolved"
