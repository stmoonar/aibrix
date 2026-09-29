"""SafeScale direct evidence: evidence gaps never commit (2026-09-29 review, P0 / P1).

Invariant: a direct commit needs evidence covering [baseline, deadline] of EVERY live
remaining pod. A pod whose baseline came late (pending at the baseline scrape, the
whole probe's baseline late, a counter reset) has a hole in [hide, its baseline]: its
data still drives the immediate rollback, but the commit is judged on the Redis
evidence. A pod not scraped successfully by the deadline poll defers the commit.

The three ``test_repro_*`` cases reproduce the review's "SLO violation, still commits"
scenarios; they fail on a9e7fa77.
"""
from __future__ import annotations

import asyncio
import json
import logging
import time

import pytest
import yaml
from dataclasses import replace

from tre_common.registry import SafeScaleRegistryConfig, parse_safescale_config
from tre_controller.config import ControllerConfig
from tre_controller.planning.safescale import SafeScaleStateMachine
from tre_controller.planning.safescale_direct import (
    DirectEvidenceCollector,
    PodMetricsScraper,
    _unusable,
    parse_vllm_metrics,
)

from test_safescale import FakeProbeStore
from test_safescale_direct_20260929 import (
    HIDE,
    LIVE_030,
    MODEL,
    START,
    TRE_DIR,
    W,
    FakeEvidence,
    FullEvidence,
    Harness,
    PodSim,
    _anchor,
    _cfg,
    _obs,
    _window,
)


def _violating_redis() -> FakeEvidence:
    """Redis evidence (every remaining pod) that sees the violation the direct window lost."""
    return FullEvidence([_window(50, ttft=2_000.0)], anchor=_anchor())


# ============================================================ reproductions
def test_repro_p0_pending_pod_violating_before_its_late_baseline_never_commits_on_direct() -> None:
    # m-2 (overloaded) misses the baseline scrape and serves 5 s TTFTs before its first
    # successful scrape: those requests are inside its late baseline, i.e. lost.
    h = Harness(evidence=_violating_redis())
    h.failing["m-2"] = "timeout"
    h.start()
    del h.failing["m-2"]
    h.sims["m-2"].serve(30, ttft_s=5.0)
    decisions = h.run_until(HIDE + W, serve=5)
    at, decision = decisions[-1]
    assert decision.status != "commit" or decision.details["evidence_source_used"] != "direct"
    assert at == HIDE + W and decision.status == "rollback"
    assert decision.details["evidence_source_used"] == "redis_fallback"
    assert decision.details["direct_fallback"]["reason"] == "late_baseline"
    assert decision.details["latency_source"] == "evidence"  # the gateway docs judged it
    probe_terms = decision.details
    assert probe_terms["direct_late_pods"]["m-2"]["cause"] == "pending_baseline"
    assert probe_terms["direct_late_pods"]["m-2"]["lag_ms"] >= 2_000


def test_repro_p1a_violation_after_a_restart_rolls_back() -> None:
    # m-2 restarts after its baseline and then serves 2 s TTFTs: the pod must stay in
    # the evidence (zero baseline), not be dropped so m-0 alone commits.
    h = Harness()
    h.start(pre_hide=10, pre_hide_ttft=0.05)
    h.tick(HIDE + 2_000, serve=5)
    h.sims["m-2"].restart()
    h.sims["m-2"].serve(20, ttft_s=2.0)
    decision = h.tick(HIDE + 4_000, serve=5)
    assert decision.status == "rollback"
    assert decision.details["rollback_reason"]["code"] == "slo_violation_direct"
    assert decision.details["rollback_reason"]["ttft_p95_ms"] == pytest.approx(2_500.0)


def test_repro_p1b_a_pod_failing_the_deadline_poll_defers_the_commit() -> None:
    # m-2 answers until just before the deadline; its deadline scrape times out while it
    # serves 2 s TTFTs. Its older delta is still "fresh" (5 s), but not good enough to commit.
    h = Harness()
    h.start()
    h.run_until(HIDE + W - 2_000, serve=5)
    h.failing["m-2"] = "timeout"
    h.sims["m-2"].serve(40, ttft_s=2.0)
    decision = h.tick(HIDE + W, serve=5)
    assert decision.status == "probing" and decision.reason == "evidence_extended"
    assert decision.details["extend_reason"] == "pods_unanswered"
    assert decision.details["direct_unanswered_pods"] == {"m-2": "timeout"}
    del h.failing["m-2"]
    decision = h.tick(HIDE + W + 2_000, serve=5)
    assert decision.status == "rollback"
    assert decision.details["rollback_reason"]["code"] == "slo_violation_direct"


# ============================================================ P0: other late-baseline paths
def test_a_controller_restart_between_hide_and_baseline_makes_the_whole_probe_late() -> None:
    store = FakeProbeStore()
    h = Harness(store=store, evidence=_violating_redis())
    h.machine.start_probe(model=MODEL, pods=("m-1",), now_ms=START)
    h.clock.now = HIDE
    assert h.collector.on_hide_done(MODEL, ("m-1",))  # no loop: no baseline yet
    probe = h.machine.active_probe(MODEL)
    record = json.loads(json.dumps(store.records[probe.request_id]))
    # The controller restarts; meanwhile the remaining pods serve slow requests.
    for sim in h.sims.values():
        sim.serve(30, ttft_s=5.0)
    restored = SafeScaleStateMachine(config=_cfg(), store=FakeProbeStore(unresolved=[record]),
                                     evidence=h.evidence, wall_clock_ms=h.clock)
    assert restored.restore() == 1
    h.machine, h.collector = restored, h._collector(restored)
    h.tick(HIDE + 6_000, serve=5)  # the first tick after the restart takes the baseline
    state = h.machine.active_probe(MODEL).direct
    assert state.baseline_lag_ms == {"m-0": 6_000, "m-2": 6_000}
    assert {pod: value["cause"] for pod, value in state.late.items()} == {"m-0": "probe_baseline",
                                                                        "m-2": "probe_baseline"}
    at, decision = h.run_until(HIDE + W, serve=5, first=HIDE + 8_000)[-1]
    assert at == HIDE + W and decision.status == "rollback"
    assert decision.details["direct_fallback"]["reason"] == "late_baseline"
    assert decision.details["direct_baseline_lag_ms"] == {"m-0": 6_000, "m-2": 6_000}


def test_a_first_baseline_failing_everywhere_then_succeeding_is_late() -> None:
    h = Harness(evidence=_violating_redis())
    h.failing.update({"m-0": "timeout", "m-2": "timeout"})
    probe = h.start()
    assert probe.direct.baseline is None and probe.direct.failed_polls == 1
    h.failing.clear()
    h.tick(HIDE + 2_000, serve=5)  # the retry succeeds 2 s after the confirmation
    state = h.machine.active_probe(MODEL).direct
    assert state.fallback is None and set(state.late) == {"m-0", "m-2"}
    at, decision = h.run_until(HIDE + W, serve=5, first=HIDE + 4_000)[-1]
    assert at == HIDE + W and decision.status == "rollback"
    assert decision.details["evidence_source_used"] == "redis_fallback"
    assert decision.details["direct_fallback"]["reason"] == "late_baseline"


def test_a_late_pod_still_drives_the_immediate_rollback() -> None:
    h = Harness()
    h.failing["m-2"] = "timeout"
    h.start()
    del h.failing["m-2"]
    h.tick(HIDE + 2_000, serve=5)  # m-2's (late) baseline
    decision = h.tick(HIDE + 4_000, serve=25, ttft_s=2.0)
    assert decision.status == "rollback"
    assert decision.details["rollback_reason"]["code"] == "slo_violation_direct"


def test_an_on_time_baseline_is_not_late_and_commits_on_direct() -> None:
    h = Harness()
    h.start()
    at, decision = h.run_until(HIDE + W, serve=5)[-1]
    assert decision.status == "commit" and decision.details["evidence_source_used"] == "direct"
    assert decision.details["direct_late_pods"] == {}
    assert decision.details["direct_baseline_lag_ms"] == {"m-0": 0, "m-2": 0}
    assert decision.details["direct_late_after_ms"] == 1_000.0


# ============================================================ P1-b: the deciding poll
def test_a_pod_failing_the_deadline_poll_at_the_ceiling_hands_the_commit_to_redis() -> None:
    h = Harness(window_ceiling_ms=float(W))  # cap = deadline: nothing to extend
    h.start()
    h.run_until(HIDE + W - 2_000, serve=5)
    h.failing["m-2"] = "timeout"
    decision = h.tick(HIDE + W, serve=5)
    assert decision.status == "commit" and decision.details["evidence_source_used"] == "redis_fallback"
    assert decision.details["direct_fallback"]["reason"] == "pods_unanswered"
    assert decision.details["direct_fallback"]["detail"] == {"m-2": "timeout"}
    assert decision.details["latency_source"] == "evidence"


def test_no_poll_at_the_deadline_tick_defers_the_commit() -> None:
    h = Harness()
    h.start()
    h.run_until(HIDE + W - 2_000, serve=5)
    h.clock.now = HIDE + W
    decision = h.machine.observe(MODEL, _obs(120_000), now_ms=120_000, direct_poll=None)
    assert decision.reason == "evidence_extended" and decision.details["extend_reason"] == "no_poll_this_tick"
    assert h.tick(HIDE + W + 2_000, serve=5).status == "commit"


# ============================================================ P2: baseline delay
def test_the_baseline_waits_for_the_delay_and_the_deadline_stays_confirmation_plus_w() -> None:
    h = Harness(baseline_delay_ms=1_000.0)
    h.machine.start_probe(model=MODEL, pods=("m-1",), now_ms=START)
    h.clock.now = HIDE
    assert h.collector.on_hide_done(MODEL, ("m-1",))
    assert asyncio.run(h.collector.take_baselines()) == 0  # not before confirmation + 1 s
    assert h.machine.direct_baseline_due(now_ms=HIDE + 999) == ()
    assert h.machine.direct_baseline_planned_ms(MODEL) == HIDE + 1_000
    for sim in h.sims.values():
        sim.serve(30, ttft_s=5.0)  # still routed while the gateway applies the hide
    h.clock.now = HIDE + 1_000
    assert asyncio.run(h.collector.take_baselines()) == 1
    probe = h.machine.active_probe(MODEL)
    assert probe.deadline_ms == HIDE + W and probe.direct.baseline_ts_ms == HIDE + 1_000
    assert probe.direct.late == {}  # lag 1 s = the planned delay: on time
    # Ticks on the loop's own phase (confirmation + 3 s, + 5 s, ...): the first one at or
    # after confirmation + W decides.
    at, decision = h.run_until(HIDE + W + 1_000, serve=5, first=HIDE + 3_000)[-1]
    assert at == HIDE + W + 1_000 and decision.status == "commit"
    assert decision.details["evidence_source_used"] == "direct"
    assert decision.details["latency_samples"] == 10 * 2 * 5  # 10 ticks x 2 pods x 5; the 30 slow ones are not in it
    assert decision.details["baseline_delay_ms"] == 1_000
    assert decision.details["direct_baseline_planned_ms"] == HIDE + 1_000
    assert decision.details["direct_baseline_ts_ms"] == HIDE + 1_000
    assert decision.details["direct_baseline_lag_ms"] == {"m-0": 1_000, "m-2": 1_000}
    assert decision.details["direct_late_after_ms"] == 2_000.0


def test_the_hide_callback_schedules_the_baseline_after_the_delay() -> None:
    started = time.monotonic()

    def real_clock() -> int:
        return HIDE + int((time.monotonic() - started) * 1000)

    h = Harness(baseline_delay_ms=80.0)
    h.clock = real_clock  # type: ignore[assignment]
    h.machine = SafeScaleStateMachine(config=_cfg(baseline_delay_ms=80.0), evidence=h.evidence,
                                      wall_clock_ms=real_clock)
    h.collector = h._collector(h.machine)
    h.machine.start_probe(model=MODEL, pods=("m-1",), now_ms=START)

    async def scenario():
        assert h.collector.on_hide_done(MODEL, ("m-1",))
        await asyncio.sleep(0.02)
        assert h.machine.active_probe(MODEL).direct.baseline is None  # still waiting
        await asyncio.gather(*list(h.collector._tasks))

    asyncio.run(scenario())
    state = h.machine.active_probe(MODEL).direct
    assert state.baseline_ts_ms is not None and state.baseline_ts_ms >= HIDE + 80
    assert state.late == {}


def test_registry_baseline_delay_key() -> None:
    assert SafeScaleRegistryConfig().baseline_delay_ms == 1000.0
    assert parse_safescale_config({"baseline_delay_ms": 0}).baseline_delay_ms == 0.0
    assert parse_safescale_config({"baseline_delay_ms": 2500}).baseline_delay_ms == 2500.0
    for bad in ({"baseline_delay_ms": -1}, {"baseline_delay_ms": float("nan")}, {"baseline_delay_ms": True},
                {"baseline_delay_ms": 60_000}, {"baseline_delay_ms": "x"}):
        with pytest.raises(ValueError):
            parse_safescale_config(bad)


def test_config_maps_the_baseline_delay(tmp_path) -> None:
    raw = yaml.safe_load((TRE_DIR / "deploy" / "registry.yaml").read_text(encoding="utf-8"))
    assert raw["safescale"]["baseline_delay_ms"] == 1000
    raw["safescale"]["baseline_delay_ms"] = 1500
    path = tmp_path / "registry.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    cfg = ControllerConfig.from_env({"TRE_REGISTRY_PATH": str(path)}).safescale
    assert cfg.baseline_delay_ms == 1500.0


# ============================================================ P3: model_name filter
def test_samples_of_another_model_name_are_skipped() -> None:
    mine, other = PodSim(), PodSim()
    mine.serve(3)
    other.serve(7)
    text = mine.text() + other.text().replace('model_name="m"', 'model_name="other"')
    assert parse_vllm_metrics(text, ts_ms=1).ttft.count == 10  # no filter: summed
    scrape = parse_vllm_metrics(text, ts_ms=1, model_name="m")
    assert scrape.ttft.count == 3 and scrape.other_models == ("other",)
    foreign = parse_vllm_metrics(other.text().replace('model_name="m"', 'model_name="other"'), ts_ms=1,
                                 model_name="m")
    assert foreign.ttft is None and _unusable(foreign) == "model_mismatch:other"


def test_a_pod_ip_reused_by_another_model_is_no_evidence() -> None:
    h = Harness()
    h.start()
    h.run_until(HIDE + W - 2_000, serve=5)
    assert set(h.scraper.model_names) == {MODEL}
    original = h.sims["m-2"].text
    h.sims["m-2"].text = lambda **kw: original(**kw).replace('model_name="m"', 'model_name="x"')
    decision = h.tick(HIDE + W, serve=5)
    assert decision.reason == "evidence_extended" and decision.details["extend_reason"] == "pods_unanswered"
    assert h.machine.active_probe(MODEL).direct.last.unanswered == {"m-2": "model_mismatch:x"}


# ============================================================ P2: scrape pool timing
def test_the_scrape_timeout_counts_from_the_thread_start_not_the_queue() -> None:
    def fetch(url: str, timeout_s: float) -> str:
        time.sleep(0.2)
        return LIVE_030.read_text(encoding="utf-8")

    # 4 reads of 0.2 s on 2 threads: the last two wait 0.2 s for a thread; a timeout
    # counted from the submission (0.3 s) would fail them although each read took 0.2 s.
    scraper = PodMetricsScraper(timeout_s=0.3, fetch=fetch, max_workers=2)
    try:
        results = asyncio.run(scraper.scrape({f"p{i}": f"http://p{i}" for i in range(4)}))
    finally:
        scraper.close()
    assert all(not isinstance(results[f"p{i}"], str) for i in range(4)), results


def test_a_read_without_a_thread_is_pool_saturated_and_a_busy_pool_is_logged(caplog) -> None:
    def fetch(url: str, timeout_s: float) -> str:
        time.sleep(0.8 if "dead" in url else 0.01)
        return LIVE_030.read_text(encoding="utf-8")

    scraper = PodMetricsScraper(timeout_s=0.2, queue_timeout_s=0.15, fetch=fetch, max_workers=1)
    with caplog.at_level(logging.WARNING, logger="tre_controller.safescale"):
        started = time.monotonic()
        results = asyncio.run(scraper.scrape({"a": "http://dead", "b": "http://ok"}))
        elapsed = time.monotonic() - started
    scraper.close()
    assert results == {"a": "timeout", "b": "pool_saturated"}
    assert elapsed < 0.6  # bounded by queue_timeout + timeout, not by the dead read
    assert any("safescale_scrape_pool_busy" in record.getMessage() for record in caplog.records)


def test_occupancy_returns_to_zero_and_close_stops_the_pool() -> None:
    scraper = PodMetricsScraper(timeout_s=0.5, fetch=lambda url, t: LIVE_030.read_text(encoding="utf-8"))
    results = asyncio.run(scraper.scrape({"a": "http://a", "b": None}, model_name="dsqwen-7b"))
    assert results["a"].ttft.count == 6.0 and results["b"] == "no_endpoint"
    assert scraper.occupancy() == 0
    h = Harness()
    h.collector = DirectEvidenceCollector(h.machine, scraper, lambda model, exclude: {}, poll_ms=2_000.0)
    h.collector.close()
    assert asyncio.run(scraper.scrape({"a": "http://a"})) == {"a": "scraper_closed"}


# ============================================================ P3: fresh view, shutdown
def test_the_app_targets_only_a_fresh_cluster_view_and_closes_the_collector(monkeypatch) -> None:
    from test_controller_app import EmptyRedis
    from test_controller_app import REGISTRY_PATH as APP_REGISTRY
    from tre_common.registry import ClusterTopology, NodeSpec
    from tre_controller import app as app_module
    from tre_controller.app import create_controller_dependencies
    from tre_controller.loops.cluster_view_task import cluster_view_from_state

    cfg = ControllerConfig.from_env({"TRE_REGISTRY_PATH": str(APP_REGISTRY)})
    deps = create_controller_dependencies(cfg, redis_client=EmptyRedis())
    state = {"bindings": [{"serve_id": "m-0", "model": "m", "node": "n", "gpu_ids": [0], "awake": True,
                           "hidden": False}],
             "fleet": {"observed": [{"pod_name": "m-0", "pod_ip": "10.0.0.5"}]}}
    view = cluster_view_from_state(state, ClusterTopology(nodes=(NodeSpec(name="n", gpus=4,
                                                                          two_gpu_slots=((0, 1), (2, 3))),)))
    box = deps.cluster_view_box
    now = {"t": 1000.0}
    box._monotonic = lambda: now["t"]
    box.set(view)
    targets = deps.direct_evidence._targets
    assert list(targets("m", ())) == ["m-0"]
    now["t"] += box.max_age_s + 1  # stale view: no remaining-pod set to trust
    assert box.get() is view and targets("m", ()) == {}

    closed = []
    deps.direct_evidence.close = lambda: closed.append(True)  # type: ignore[method-assign]

    async def no_shutdown():
        return None

    deps.queue.shutdown = no_shutdown  # type: ignore[method-assign]
    monkeypatch.setattr(app_module, "build_controller_task_specs", lambda d, c: ())
    asyncio.run(app_module.run_controller(deps, cfg))
    assert closed == [True]


def test_a_pending_pod_that_falls_asleep_is_late_not_silently_dropped() -> None:
    # It served (slowly) between the hide and its sleep: out of the polls, but no direct
    # commit on the others.
    h = Harness(evidence=_violating_redis())
    h.failing["m-2"] = "timeout"
    h.start()
    del h.failing["m-2"]
    h.sims["m-2"].serve(30, ttft_s=5.0)
    h.sims["m-2"].awake = False
    h.tick(HIDE + 2_000, serve=5)
    state = h.machine.active_probe(MODEL).direct
    assert state.dropped["m-2"]["reason"] == "asleep"
    assert state.late["m-2"]["cause"] == "asleep_before_baseline"
    assert state.live_pods() == ("m-0",)
    at, decision = h.run_until(HIDE + W, serve=5, first=HIDE + 4_000)[-1]
    assert at == HIDE + W and decision.status == "rollback"
    assert decision.details["direct_fallback"]["reason"] == "late_baseline"
    assert decision.details["direct_fallback"]["detail"]["m-2"]["cause"] == "asleep_before_baseline"


# ============================================================ review 2, F1: pooled p95
def test_an_overloaded_pod_below_the_per_pod_minimum_still_weighs_in() -> None:
    # m-0 serves 50 fast; overloaded m-2 completes only 9 (< TRE_MIN_LATENCY_SAMPLES 10)
    # at 10 s: its own p95 is undefined, the per-pod maximum sees m-0 only.
    h = Harness()
    h.start()
    h.sims["m-0"].serve(50, ttft_s=0.05)
    h.sims["m-2"].serve(9, ttft_s=10.0)
    decision = h.tick(HIDE + 2_000)
    assert decision.status == "rollback"
    assert decision.details["rollback_reason"]["code"] == "slo_violation_direct"
    assert decision.details["rollback_reason"]["ttft_p95_ms"] == pytest.approx(10_000.0)
    assert decision.details["evidence_pooled_ttft_p95_ms"] == pytest.approx(10_000.0)
    assert decision.details["latency_samples_judged"] == 59


def test_the_redis_reader_pools_the_pods_too() -> None:
    from test_safescale_evidence_20260929 import DocRedis
    from test_safescale_evidence_20260929 import _reader as evidence_reader

    redis = DocRedis()
    for ts, fast, slow in ((110_000, 0, 0), (120_000, 40, 5), (130_000, 80, 9)):
        redis.add("m-0", ts, fast=fast, slow=0, prompt=float(fast))
        redis.add("m-3", ts, fast=0, slow=slow, prompt=float(slow))
    window = evidence_reader(redis).read(MODEL, start_ms=110_000, end_ms=130_000, exclude_pods=("m-1",))
    assert window.pods == ("m-0", "m-3") and window.ttft_count == 89
    assert window.pooled_ttft_p95_ms == 5_000.0 and window.ttft_p95_ms == 5_000.0  # m-3 alone has none
    assert window.judged == 89
    assert window.last_doc_ts_ms == {"m-0": 130_000, "m-3": 130_000}


def test_pooled_p95_rule() -> None:
    from tre_common.window_pods import pooled_p95_ms

    fast = ((0.1, 50.0), (5.0, 50.0), (float("inf"), 50.0))
    slow = ((0.1, 0.0), (5.0, 9.0), (float("inf"), 9.0))
    assert pooled_p95_ms([(fast, 50.0), (slow, 9.0)], ("bucket_upper", 10)) == 5_000.0
    assert pooled_p95_ms([(slow, 9.0)], ("bucket_upper", 10)) is None  # gate on the pooled count
    assert pooled_p95_ms([(None, 0.0)], ("bucket_upper", 0)) is None
    beyond = ((0.1, 0.0), (5.0, 0.0), (float("inf"), 20.0))
    assert pooled_p95_ms([(beyond, 20.0)], ("bucket_upper", 0)) == 5_000.0  # +Inf -> largest finite


# ============================================================ review 2, F2: Redis completeness
def test_a_ceiling_fallback_never_commits_on_redis_evidence_missing_a_remaining_pod() -> None:
    partial = FullEvidence([_window(50)], anchor=_anchor(), pods=("m-0",))  # gateway docs of m-0 only
    h = Harness(evidence=partial, window_ceiling_ms=float(W))
    h.start()
    h.run_until(HIDE + W - 2_000, serve=5)
    h.failing["m-2"] = "timeout"
    decision = h.tick(HIDE + W, serve=5)
    assert decision.status == "rollback"
    assert decision.details["rollback_reason"]["code"] == "evidence_incomplete"
    assert decision.details["rollback_reason"]["pods"] == {"m-2": "no_docs"}
    assert decision.details["direct_fallback"]["required_pods"] == ["m-0", "m-2"]


def test_redis_docs_of_a_remaining_pod_ending_early_do_not_commit() -> None:
    class EarlyEnd(FullEvidence):
        def read(self, model, *, start_ms, end_ms, exclude_pods):
            window = super().read(model, start_ms=start_ms, end_ms=end_ms, exclude_pods=exclude_pods)
            return replace(window, last_doc_ts_ms={**window.last_doc_ts_ms, "m-2": end_ms - 10_000})

    h = Harness(evidence=EarlyEnd([_window(50)], anchor=_anchor()), window_ceiling_ms=float(W))
    h.start()
    h.run_until(HIDE + W - 2_000, serve=5)
    h.failing["m-2"] = "timeout"
    decision = h.tick(HIDE + W, serve=5)
    assert decision.status == "rollback"
    assert decision.details["rollback_reason"]["pods"] == {"m-2": "docs_end:110000"}


def test_a_late_fallback_below_the_ceiling_extends_while_the_redis_evidence_is_incomplete() -> None:
    partial = FullEvidence([_window(50)], anchor=_anchor(), pods=("m-0",))
    h = Harness(evidence=partial)
    h.failing["m-2"] = "timeout"
    h.start()
    del h.failing["m-2"]
    decisions = h.run_until(HIDE + W, serve=5)
    at, decision = decisions[-1]
    assert at == HIDE + W and decision.status == "probing"
    assert decision.reason == "evidence_extended" and decision.details["extend_reason"] == "evidence_incomplete"
    assert decision.details["evidence_incomplete_pods"] == {"m-2": "no_docs"}
