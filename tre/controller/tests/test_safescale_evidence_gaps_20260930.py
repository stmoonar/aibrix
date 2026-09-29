"""SafeScale commit evidence, review 3 (2026-09-30): no commit on incomplete evidence.

Decision 1: in direct mode the Redis evidence never decides a commit - every gap in the
direct evidence extends by one poll, and rolls back (``evidence_incomplete:<gap>``)
when it cannot heal or the ceiling is reached. Decision 2: at the ceiling with too few
samples the latency gate is no longer skipped - every sample there is is judged
(``evaluated_low_samples``), a stalled model (traffic, no completion) rolls back.
Redis mode (the rollback switch): the pods a commit needs come from a fresh cluster
view, each must have docs up to the evidence end, and the end must reach the deadline.
Plus: a pod that joins after the baseline is late; coverage ends at the request start
of the deciding poll; ``process_start_time_seconds`` detects restarts; the immediate
rollback judges max(per pod, pooled).

Every test here fails on 8d252108.
"""
from __future__ import annotations

import asyncio
import json
import math

import pytest

from tre_common.metrics_schema import ModelWindowMetrics, PodWindowMetrics
from tre_controller.planning.safescale import ProbeObservation, SafeScaleStateMachine
from tre_controller.planning.safescale_direct import (
    DirectEvidenceCollector,
    DirectState,
    parse_vllm_metrics,
)
from tre_controller.planning.safescale_evidence import HideAnchor

from test_safescale import FakeProbeStore
from test_safescale_direct_20260929 import (
    HIDE,
    LIVE_030,
    MODEL,
    START,
    W,
    FullEvidence,
    Harness,
    PodSim,
    SimScraper,
    _anchor,
    _cfg,
    _obs,
    _window,
)
from test_safescale_evidence_20260929 import FakeEvidence
from test_safescale_evidence_20260929 import _cfg as redis_cfg
from test_safescale_evidence_20260929 import _machine as redis_machine
from test_safescale_evidence_20260929 import _obs as redis_obs
from test_safescale_evidence_20260929 import _remaining, _started
from test_safescale_evidence_20260929 import _window as redis_window

CAP = HIDE + 60_000


def _healthy_redis() -> FullEvidence:
    """Gateway docs of every remaining pod, fast and plentiful: the evidence 8d252108
    committed on after a fallback."""
    return FullEvidence([_window(50, ttft=100.0)], anchor=_anchor())


# ============================================================ the review's reproductions
def test_repro_1_fifteen_ten_second_samples_at_the_ceiling_roll_back_direct() -> None:
    h = Harness(pods=("m-0",))
    h.start()
    h.tick(HIDE + 2_000, serve=15, ttft_s=10.0)  # 15 < min_commit_samples: no immediate rollback
    at, decision = h.run_until(CAP, first=HIDE + 4_000)[-1]
    assert at == CAP and decision.status == "rollback"
    assert decision.details["rollback_reason"]["gates"] == ["latency"]
    assert decision.details["latency_gate"] == "evaluated_low_samples"
    assert decision.details["low_sample_ttft_p95_ms"] == pytest.approx(10_000.0)
    assert decision.details["latency_samples"] == 15


def test_repro_1_fifteen_ten_second_samples_at_the_ceiling_roll_back_redis() -> None:
    machine = redis_machine(FakeEvidence([redis_window(15, ttft=10_000.0)]))
    _started(machine)
    for ts in range(110_000, 170_000, 10_000):
        decision = machine.observe(MODEL, redis_obs(ts), now_ms=ts)
    assert (decision.status, decision.reason) == ("rollback", "formal_commit_gate_failed")
    assert decision.details["latency_gate"] == "evaluated_low_samples"
    assert decision.details["rollback_reason"]["gates"] == ["latency"]


def test_repro_2_a_late_baseline_never_commits_on_the_redis_evidence() -> None:
    # 8d252108: late_baseline -> fallback -> the Redis window [110000, 120000] (before
    # the deadline 123000, docs the gateway may have re-written from its cache) commits.
    h = Harness(evidence=_healthy_redis())
    h.failing["m-2"] = "timeout"
    h.start()
    del h.failing["m-2"]
    at, decision = h.run_until(HIDE + W, serve=5)[-1]
    assert (at, decision.status) == (HIDE + W, "rollback")
    assert decision.details["rollback_reason"]["code"] == "evidence_incomplete:late_baseline"
    assert decision.details["rollback_reason"]["detail"]["m-2"]["cause"] == "pending_baseline"
    assert h.evidence.reads == []


def test_repro_3_pods_unanswered_at_the_ceiling_roll_back_even_with_healthy_redis_docs() -> None:
    h = Harness(evidence=_healthy_redis())
    h.start()
    h.failing["m-2"] = "timeout"
    at, decision = h.run_until(CAP, serve=5)[-1]
    assert (at, decision.status) == (CAP, "rollback")
    assert decision.details["rollback_reason"]["code"] == "evidence_incomplete:pods_missing"
    assert h.evidence.reads == []


# ============================================================ decision 1: other gaps
def test_a_legacy_record_that_had_fallen_back_is_rolled_back() -> None:
    store = FakeProbeStore()
    h = Harness(store=store, evidence=_healthy_redis())
    probe = h.start()
    record = json.loads(json.dumps(store.records[probe.request_id]))
    record["direct_evidence"]["fallback"] = {"reason": "pods_missing", "ts_ms": HIDE + 1_000}
    restored = SafeScaleStateMachine(config=_cfg(), store=FakeProbeStore(unresolved=[record]),
                                     evidence=h.evidence, wall_clock_ms=h.clock)
    assert restored.restore() == 1
    h.machine, h.collector = restored, h._collector(restored)
    decision = h.tick(HIDE + 2_000, serve=5)
    assert decision.status == "rollback"
    assert decision.details["rollback_reason"]["code"] == "evidence_incomplete:legacy_redis_fallback"


def test_a_direct_probe_whose_hide_is_never_confirmed_never_commits_without_a_reader() -> None:
    # 8d252108 judged it on the tail snapshots (no evidence reader): commit.
    machine = SafeScaleStateMachine(config=_cfg(), evidence=None, wall_clock_ms=lambda: 0)
    machine.start_probe(model=MODEL, pods=("m-1",), now_ms=START)
    for ts in range(110_000, 180_000, 10_000):
        decision = machine.observe(MODEL, _obs(ts), now_ms=ts)
        if decision.status != "probing":
            break
    assert (decision.status, decision.reason) == ("rollback", "hide_unconfirmed")


# ============================================================ P2-c: the remaining-pod set
def test_a_pod_joining_after_the_baseline_is_late_and_the_probe_rolls_back() -> None:
    h = Harness()
    h.start()
    h.tick(HIDE + 2_000, serve=5)
    h.sims["m-3"] = PodSim()  # a new replica: awake, not hidden, in the fresh view
    h.tick(HIDE + 4_000, serve=5)
    state = h.machine.active_probe(MODEL).direct
    assert state.late["m-3"]["cause"] == "joined_after_baseline" and "m-3" in state.pending
    h.tick(HIDE + 6_000, serve=5)
    assert "m-3" in h.machine.active_probe(MODEL).direct.baseline  # polled from now on
    at, decision = h.run_until(HIDE + W, serve=5, first=HIDE + 8_000)[-1]
    assert (at, decision.status) == (HIDE + W, "rollback")
    assert decision.details["rollback_reason"]["code"] == "evidence_incomplete:late_baseline"
    assert decision.details["rollback_reason"]["detail"]["m-3"]["cause"] == "joined_after_baseline"


def test_a_joined_pod_still_drives_the_immediate_rollback() -> None:
    h = Harness()
    h.start()
    h.sims["m-3"] = PodSim()
    h.tick(HIDE + 2_000, serve=5)  # m-3 joins
    h.tick(HIDE + 4_000, serve=5)  # its baseline
    h.sims["m-3"].serve(30, ttft_s=5.0)
    decision = h.tick(HIDE + 6_000, serve=5)
    assert decision.status == "rollback" and decision.details["rollback_reason"]["code"] == "slo_violation_direct"


def test_no_fresh_view_at_the_deciding_poll_defers_and_rolls_back_at_the_ceiling() -> None:
    h = Harness()
    view = {"fresh": True}
    h.collector = DirectEvidenceCollector(
        h.machine, h.scraper,
        lambda model, exclude: {p: f"http://{p}" for p in h.sims if p not in exclude} if view["fresh"] else {},
        urls=lambda model, pods: {p: f"http://{p}" for p in pods},
        poll_ms=2_000.0, clock_ms=h.clock,
    )
    h.start()
    h.run_until(HIDE + W - 2_000, serve=5)
    view["fresh"] = False
    decision = h.tick(HIDE + W, serve=5)
    assert decision.reason == "evidence_extended" and decision.details["extend_reason"] == "no_fresh_view"
    view["fresh"] = True
    assert h.tick(HIDE + W + 2_000, serve=5).status == "commit"
    view["fresh"] = False
    stale = Harness(window_ceiling_ms=float(W))
    stale.collector = DirectEvidenceCollector(
        stale.machine, stale.scraper,
        lambda model, exclude: {p: f"http://{p}" for p in stale.sims if p not in exclude} if view["fresh"] else {},
        urls=lambda model, pods: {p: f"http://{p}" for p in pods}, poll_ms=2_000.0, clock_ms=stale.clock,
    )
    view["fresh"] = True
    stale.start()
    stale.run_until(HIDE + W - 2_000, serve=5)
    view["fresh"] = False
    decision = stale.tick(HIDE + W, serve=5)
    assert decision.status == "rollback"
    assert decision.details["rollback_reason"]["code"] == "evidence_incomplete:no_fresh_view"


# ============================================================ decision 2: low samples
def test_a_low_sample_commit_is_evaluated_and_counted_apart() -> None:
    from scripts.analysis.safescale_summary import summarize

    h = Harness(pods=("m-0",))
    h.start()
    h.tick(HIDE + 2_000, serve=15, ttft_s=0.05)
    at, decision = h.run_until(CAP, first=HIDE + 4_000)[-1]
    assert (at, decision.status) == (CAP, "commit")
    assert decision.details["latency_gate"] == "evaluated_low_samples"
    assert decision.details["low_sample_commit"] is True and decision.details["latency_samples"] == 15
    assert decision.details["low_sample_ttft_p95_ms"] == pytest.approx(60.0)
    probe = h.machine.active_probe(MODEL)
    record = {"status": "resolved", "resolution": "commit", "window_terms": dict(probe.window_terms)}
    assert summarize([record])["low_sample_commits"] == 1


def test_requests_in_flight_on_the_pods_are_traffic_a_stalled_model_rolls_back() -> None:
    # The snapshot tail says idle, but the pods report requests running and none completes.
    h = Harness(pods=("m-0",))
    h.obs_kwargs = {"traffic": False}
    original = h.sims["m-0"].text
    h.sims["m-0"].text = lambda **kw: original(**kw) + 'vllm:num_requests_running{engine="0",model_name="m"} 3.0\n'
    h.start()
    at, decision = h.run_until(CAP)[-1]
    assert (at, decision.status) == (CAP, "rollback")
    assert decision.details["rollback_reason"]["code"] == "insufficient_evidence:stalled"
    assert decision.details["direct_in_flight"] == 3.0


# ============================================================ Redis mode: required pods
def _redis_until_decided(machine, *, end: int = 190_000):
    decisions = []
    for ts in range(110_000, end, 10_000):
        decision = machine.observe(MODEL, redis_obs(ts), now_ms=ts)
        decisions.append((ts, decision))
        if decision.status != "probing":
            break
    return decisions


def test_redis_mode_never_commits_without_evidence_of_a_remaining_pod_of_the_fresh_view() -> None:
    machine = redis_machine(FakeEvidence([redis_window(50)]), remaining=("m-0", "m-2"))  # docs of m-0 only
    _started(machine)
    decisions = _redis_until_decided(machine)
    assert all(d.status == "probing" for _, d in decisions[:-1])
    ts, decision = decisions[-1]
    assert (ts, decision.status) == (170_000, "rollback")  # the ceiling + one gateway period
    assert decision.details["rollback_reason"]["code"] == "evidence_incomplete:pods_incomplete"
    assert decision.details["rollback_reason"]["detail"] == {"m-2": "no_docs"}
    assert decision.details["required_pods"] == ["m-0", "m-2"]


def test_redis_mode_without_a_fresh_view_or_remaining_pods_never_commits() -> None:
    cases = (
        (None, "no_fresh_view"),  # not wired
        ((), "no_remaining_pods"),  # the view lists no remaining pod
    )
    for remaining, gap in cases:
        machine = redis_machine(FakeEvidence([redis_window(50)]), remaining=remaining)
        _started(machine)
        ts, decision = _redis_until_decided(machine)[-1]
        assert (ts, decision.status) == (170_000, "rollback"), gap
        assert decision.details["rollback_reason"]["code"] == f"evidence_incomplete:{gap}"
    stale = SafeScaleStateMachine(config=redis_cfg(), evidence=FakeEvidence([redis_window(50)]), wall_clock_ms=lambda: 0,
                                  remaining_pods=lambda model, exclude: None)  # a stale view
    _started(stale)
    assert _redis_until_decided(stale)[-1][1].details["rollback_reason"]["code"] == \
        "evidence_incomplete:no_fresh_view"


def test_redis_mode_docs_of_the_end_landing_late_wait_one_period_past_the_ceiling() -> None:
    class LateDocs(FakeEvidence):
        def read(self, model, *, start_ms, end_ms, exclude_pods):
            window = super().read(model, start_ms=start_ms, end_ms=end_ms, exclude_pods=exclude_pods)
            if end_ms < 170_000:  # m-0's doc of E is not written yet
                from dataclasses import replace
                window = replace(window, last_doc_ts_ms={"m-0": end_ms - 10_000})
            return window

    machine = redis_machine(LateDocs([redis_window(50)]))
    _started(machine)
    decisions = _redis_until_decided(machine)
    reasons = [d.reason for _, d in decisions]
    assert reasons[-2] == "evidence_pending" and decisions[-2][0] == 160_000  # the ceiling: wait
    ts, decision = decisions[-1]
    assert (ts, decision.status) == (170_000, "commit")


def test_redis_mode_waits_until_the_evidence_reaches_the_deadline() -> None:
    evidence = FakeEvidence([redis_window(50)])
    machine = redis_machine(evidence)
    _started(machine, p95_e2e=15_000.0)  # W = 30 s: deadline 130 s
    machine.observe(MODEL, redis_obs(110_000), now_ms=110_000)
    machine.observe(MODEL, redis_obs(120_000), now_ms=120_000)
    # A tick at 130 s still holding the 120 s snapshot only: E < deadline, not judged.
    decision = machine.observe(MODEL, redis_obs(120_000), now_ms=130_000)
    assert (decision.status, decision.reason) == ("probing", "evidence_pending")
    assert decision.details["evidence_gap"] == "before_deadline"
    assert machine.observe(MODEL, redis_obs(130_000), now_ms=130_000).status == "commit"


# ============================================================ P3
def test_the_deciding_poll_must_start_at_or_after_the_deadline() -> None:
    class EarlyStart(SimScraper):
        async def scrape(self, targets, *, model_name=None):
            out = await super().scrape(targets, model_name=model_name)
            return {pod: (value if isinstance(value, str) else
                          parse_vllm_metrics(self.h.sims[pod].text(), ts_ms=self.h.clock(), model_name=model_name,
                                             started_ms=self.h.clock() - 300))
                    for pod, value in out.items()}

    h = Harness()
    h.scraper = EarlyStart(h)
    h.collector = h._collector(h.machine)
    h.start()
    h.run_until(HIDE + W - 2_000, serve=5)
    decision = h.tick(HIDE + W, serve=5)  # the reads were sent at deadline - 300 ms
    assert (decision.status, decision.reason) == ("probing", "evidence_pending")
    assert decision.details["evidence_coverage_end_ms"] == HIDE + W - 300
    decision = h.tick(HIDE + W + 2_000, serve=5)
    assert decision.status == "commit"
    assert decision.details["evidence_coverage_end_ms"] == HIDE + W + 1_700
    assert decision.details["evidence_coverage_start_ms"] == HIDE


def test_a_direct_commit_records_its_coverage_and_pooled_p95() -> None:
    h = Harness()
    h.start()
    at, decision = h.run_until(HIDE + W, serve=5)[-1]
    assert decision.status == "commit"
    assert (decision.details["evidence_coverage_start_ms"], decision.details["evidence_coverage_end_ms"]) == (
        HIDE, HIDE + W)
    assert decision.details["evidence_pooled_ttft_p95_ms"] == pytest.approx(60.0)
    assert "evidence_pooled_tpot_p95_ms" in decision.details


def test_process_start_time_detects_a_restart_the_counters_hide() -> None:
    scrape = parse_vllm_metrics(LIVE_030.read_text(encoding="utf-8"), ts_ms=1)
    assert scrape.process_start_s == pytest.approx(1.79066792223e9) and scrape.in_flight == 0.0
    h = Harness()
    start = {"t": 1_790_000_000.0}
    original = h.sims["m-2"].text
    h.sims["m-2"].text = lambda **kw: original(**kw) + f"process_start_time_seconds {start['t']}\n"
    h.sims["m-2"].serve(40)  # pre-hide requests
    h.start()
    h.tick(HIDE + 2_000, serve=5)
    # Restarted, and already past its old counts: no counter goes backwards.
    h.sims["m-2"].restart()
    h.sims["m-2"].serve(60)
    start["t"] += 120.0
    h.tick(HIDE + 4_000, serve=5)
    state = h.machine.active_probe(MODEL).direct
    assert state.late["m-2"]["cause"] == "process_restarted"
    assert state.baseline["m-2"].process_start_s == start["t"]
    at, decision = h.run_until(HIDE + W, serve=5, first=HIDE + 6_000)[-1]
    assert (at, decision.status) == (HIDE + W, "rollback")
    assert decision.details["rollback_reason"]["detail"]["m-2"]["cause"] == "process_restarted"
    # The baseline's process start survives a controller restart.
    again = DirectState.from_record(json.loads(json.dumps(state.as_record())))
    assert again.baseline["m-2"].process_start_s == start["t"]


def test_the_immediate_rollback_judges_the_pooled_p95_too() -> None:
    from test_safescale_evidence_20260929 import _slo_registry
    from tre_controller.loops.safescale_task import _observation_from_metrics

    fast = ((0.1, 50.0), (5.0, 50.0), (math.inf, 50.0))
    slow = ((0.1, 0.0), (5.0, 9.0), (math.inf, 9.0))

    def pod(name, hist, count):
        return PodWindowMetrics(pod=name, prompt_tokens=0.0, generation_tokens=0.0, avg_waiting=0.0,
                                avg_running=1.0, avg_swapping=0.0, kv_cache_hit_rate=0.0,
                                ttft_p95_ms=100.0 if count >= 10 else None, tpot_p95_ms=None, e2e_p95_ms=None,
                                ttft_hist=hist, ttft_hist_count=count)

    metrics = ModelWindowMetrics(
        model=MODEL, window_start_ms=110_000, window_end_ms=140_000, prompt_tokens=0.0, generation_tokens=0.0,
        avg_waiting=0.0, avg_running=2.0, avg_swapping=0.0, kv_cache_hit_rate=0.0, ttft_p95_ms=100.0,
        tpot_p95_ms=None, e2e_p95_ms=None, routable_pods=2, assigned_replicas=2,
        per_pod={"m-0": pod("m-0", fast, 50.0), "m-3": pod("m-3", slow, 9.0), "m-1": pod("m-1", slow, 9.0)},
        p95_rule=("bucket_upper", 10),
    )
    registry = _slo_registry()
    observation = _observation_from_metrics(140_000, metrics, registry.model(MODEL), "zm", hidden_pods=("m-1",))
    assert observation.ttft_p95_ms == 100.0 and observation.pooled_ttft_p95_ms == 5_000.0
    machine = redis_machine(FakeEvidence([redis_window(50)]))
    _started(machine, p95_e2e=30_000.0)
    decision = machine.observe(MODEL, observation, now_ms=140_000)
    assert (decision.status, decision.reason) == ("rollback", "slo_violation")
    assert decision.details["rollback_reason"]["ttft_p95_ms"] == 5_000.0


# ============================================================ review of 84936742
def test_min_commit_samples_zero_never_makes_the_latency_gate_vacuous() -> None:
    # Review P2-1: with min_commit_samples 0 a stalled model (requests running, none
    # completed) committed with no p95 at all.
    h = Harness(pods=("m-0",), min_commit_samples=0)
    h.obs_kwargs = {"traffic": False}
    original = h.sims["m-0"].text
    h.sims["m-0"].text = lambda **kw: original(**kw) + 'vllm:num_requests_running{engine="0",model_name="m"} 3.0\n'
    h.start()
    at, decision = h.run_until(CAP)[-1]
    assert (at, decision.status) == (CAP, "rollback")
    assert decision.details["rollback_reason"]["code"] == "insufficient_evidence:stalled"
    machine = redis_machine(FakeEvidence([redis_window(0, ttft=None, tpot=None)]), min_commit_samples=0)
    _started(machine)
    for ts in range(110_000, 170_000, 10_000):
        decision = machine.observe(MODEL, redis_obs(ts, traffic=True), now_ms=ts)
    assert (decision.status, decision.reason) == ("rollback", "insufficient_evidence:stalled")


def test_a_pod_still_pending_at_the_deadline_rolls_back_without_extending() -> None:
    # Review P3-2: it can never heal (its baseline, once taken, is late).
    h = Harness()
    h.failing["m-2"] = "timeout"
    h.start()
    at, decision = h.run_until(CAP, serve=5)[-1]
    assert (at, decision.status) == (HIDE + W, "rollback")
    assert decision.details["rollback_reason"]["code"] == "evidence_incomplete:pending_baseline"
