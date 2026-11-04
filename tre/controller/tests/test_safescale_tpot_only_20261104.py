"""SafeScale judges TPOT only (user 2026-11-04, a design change).

Invariants, with real histograms:

1. the latency verdict is the window's TPOT p95 (direct scrapes - the live default -:
   the per-request TPOT, user 2026-11-05) against the threshold: a window whose TTFT is
   far over any TTFT_slo but whose TPOT is under it commits; the same window with TPOT
   over it rolls back;
2. fail closed on the judged latency (review of 9ad3ec32): the tail-snapshot fallback
   with traffic but no TPOT p95 never commits and rolls back
   ``insufficient_evidence:p95_unavailable`` at the ceiling. (A missing per-request TPOT
   family and the Redis path: ``test_safescale_request_tpot_20261105``.)
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

from safescale_thresholds import STATIC_TPOT_75
from test_safescale_direct_20260929 import HIDE, W, Harness
from tre_controller.config import SafeScaleConfig
from tre_controller.planning.safescale import ProbeObservation, SafeScaleStateMachine

MODEL = "m"
#: 5 s first-token time: over every TTFT_slo of every mode (floors 450 / 500 ms, k * idle).
SLOW_TTFT_S = 5.0
P95_UNAVAILABLE = "insufficient_evidence:p95_unavailable"


# ============================================================ 1. TPOT is the verdict
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
    assert not any("ttft" in key for key in decision.details)


def _tail_snapshot_run():
    """No evidence source: the gate reads the tail snapshots - traffic, no TPOT p95."""
    machine = SafeScaleStateMachine(thresholds=STATIC_TPOT_75, config=SafeScaleConfig(
        min_window_ms=20_000.0, window_ceiling_ms=60_000.0, hq=0.25, tau_low=1.0))
    machine.start_probe(model=MODEL, pods=("m-1",), now_ms=0)
    out = []
    for ts in range(10_000, 70_001, 10_000):
        observation = ProbeObservation(ts_ms=ts, tpot_p95_ms=None, z_m=2.0, has_traffic=True,
                                       window_start_ms=ts - 10_000, window_end_ms=ts)
        decision = machine.observe(MODEL, observation, now_ms=ts)
        out.append(decision)
        if decision.status in ("commit", "rollback"):
            break
    return out


def test_tail_snapshots_with_traffic_but_no_tpot_never_commit_and_roll_back_at_the_ceiling() -> None:
    decisions = _tail_snapshot_run()
    assert all(d.status != "commit" for d in decisions)
    assert decisions[-1].status == "rollback"
    assert decisions[-1].details["rollback_reason"]["code"] == P95_UNAVAILABLE


# ============================================================ structure
def test_the_safescale_code_has_no_ttft_threshold_or_ttft_p95() -> None:
    # No TTFT enters SafeScale's latency verdict (no TTFT threshold, no TTFT p95).
    tre = Path(__file__).resolve().parents[1] / "tre_controller"
    files = sorted((tre / "planning").glob("safescale*.py")) + [tre / "loops" / "safescale_task.py"]
    hits = {f.name: re.findall(r"\w*ttft_(?:ms|p95|threshold|slo)\w*", f.read_text(encoding="utf-8"))
            for f in files}
    assert not {name: found for name, found in hits.items() if found}, hits
