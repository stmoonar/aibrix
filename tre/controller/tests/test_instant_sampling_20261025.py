"""Instant sampling (2026-10-25, design docs/design/20261025-instant-sampling.md).

Instant docs every 0.5 s, histogram docs every evidence period. A pod's queue level is
the mean over all its instant docs in the window; its tokens are the counter delta over
the real ``scraped_ms`` span of the histogram docs, scaled to the window. Coverage
(distinct ``scraped_ms``) and the span only flag a pod: a donor-gate flag, the pod stays
in the window. Only frozen (I3) pods and pods without docs are left out, from queue and
tokens alike. Count-based rules keep their span at any evidence period.
"""
from __future__ import annotations

import pytest

from relay_view import expand_relays
from test_signal_validity_20261004 import _Queue, _Redis
from tre_common.registry import (
    ClusterTopology, ModelSpec, NodeSpec, Registry, SloSpec, TrsParams,
)
from tre_controller.loops.tick import run_planner_tick
from tre_controller.planning.classify import ModelState
from tre_controller.planning.planner import ClusterView, ScaleAction
from tre_controller.store.metrics_store import MetricsStore
from tre_sm.allocator.slots import Binding, Slot

INSTANT = 500
EVIDENCE = 10_000
END = 1_000_000_000_000
WINDOW = 30_000
ORIGIN = END - 40_000


def _registry(*names: str) -> Registry:
    specs = [
        ModelSpec(
            name=name, weights_path="/w", tp_size=1, min_replicas=1, max_replicas=4, vllm_image="img",
            slo=SloSpec(ttft_p95_ms=1200.0, tpot_p95_ms=100.0, e2e_p95_ms=10_000.0),
            trs=TrsParams(w_p=0.04, w_d=1.0, lambda_wait=2.625, qmin=1.0, ema_alpha=1.0, theta_m=100.0,
                          tau_crit=0.8, tau_low=1.0, tau_high=1.25, qsat=4.0, epsat=0.05, hsat=1),
        )
        for name in names
    ]
    nodes = tuple(NodeSpec(name=node, gpus=4, two_gpu_slots=((0, 1), (2, 3))) for node in ("n", "n2"))
    return Registry(ClusterTopology(nodes=nodes), specs)


def _fresh(ts):
    return ts - 50


def _slow(ts):
    """A scrape that lands only every 1 s: two docs in a row repeat one scraped_ms
    (coverage 30 of 60)."""
    return ts - ts % 1_000 - 50


def _write_pod(redis, model, pod, *, running, waiting=0.0, gen_per_s=100.0, missing=(), scraped=_fresh,
               evidence=EVIDENCE, reset_at=None, hist_missing=()):
    """Gateway docs of one pod over [END - 40 s, END]: instant docs every 0.5 s (except
    ticks in ``missing``), histogram docs every ``evidence`` (except ``hist_missing``);
    ``scraped_ms = scraped(ts)``, and the histogram counters as of that observation
    (``gen_per_s`` tokens per second; restarted from 0 at ``reset_at``: an engine
    restart)."""
    redis.sadd(f"tre:v2:pods:{model}", f"default/{pod}")
    for ts in range(ORIGIN, END + 1, INSTANT):
        seen = scraped(ts)
        base = {"timestamp": ts, "written_ms": ts + 5, "pod_name": pod, "scraped_ms": seen}
        if ts % evidence == 0 and ts not in hist_missing:
            n = (seen - ORIGIN) / 1000 + 1.0 if reset_at is None or ts < reset_at else (seen - reset_at) / 1000
            redis.zadd(f"tre:v2:hist:default/{pod}", {**base, "model_counter_metrics": {
                f"{model}/prompt_tokens_total": 50.0 * n, f"{model}/generation_tokens_total": gen_per_s * n,
            }, "model_histogram_metrics": {
                f"{model}/request_prompt_tokens": {"sum": 50.0 * n, "count": n, "buckets": {"+Inf": n}},
                f"{model}/request_generation_tokens": {"sum": gen_per_s * n, "count": n, "buckets": {"+Inf": n}},
            }}, ts)
        if ts in missing:
            continue
        redis.zadd(f"tre:v2:inst:default/{pod}", {**base, "model_metrics": {
            f"{model}/num_requests_running": running, f"{model}/num_requests_waiting": waiting,
        }}, ts)


def _store(redis, registry, min_sample_ratio=0.9, evidence=EVIDENCE, **kw):
    return MetricsStore(
        redis, registry, instant_sample_interval_ms=INSTANT, evidence_period_ms=evidence,
        min_sample_ratio=min_sample_ratio, **kw,
    )


def _read(store, model="m"):
    return store.read_model_window(model, END - WINDOW, END, start_exclusive=True)


def _snapshot(redis, registry):
    return _store(redis, registry).read_snapshot(END - WINDOW, END, start_exclusive=True)


def _binding(pod, model, gpu, awake=True, node="n"):
    return Binding(serve_id=pod, model=model, slot=Slot(node, (gpu,)), awake=awake)


def _released(result, model):
    return any(
        isinstance(a, ScaleAction) and a.model == model and a.delta < 0 for a in expand_relays(result.actions)
    ) or any(e.startswith(f"safescale_probe_skipped:{model}:") for e in result.events)


def _tick(snapshot, registry, *bindings):
    view = ClusterView(topology=registry.topology(), bindings=tuple(bindings))
    return run_planner_tick(snapshot, queue=_Queue(), registry=registry, rescue_due=True, fairness_due=True,
                            cluster_view=view)


#: 10 of the window's 60 instant ticks (1 in 6: coverage 0.83 < 0.9).
LOST = frozenset(range(END - WINDOW + INSTANT, END + 1, 6 * INSTANT))
DONOR_VIEW = (
    _binding("m-0", "m", 0), _binding("m-1", "m", 1), _binding("r-0", "r", 2), _binding("r-1", "r", 3),
    _binding("r-2", "r", 0, awake=False), _binding("r-3", "r", 1, awake=False),
)


@pytest.mark.parametrize("case", ["complete", "pod_lost_ticks", "all_pods_slow_scrapes"])
def test_a_flagged_serving_pod_blocks_the_donor_and_stays_in_the_window(case):
    """m-1 loses 1 in 6 of its instant docs (m-0 has every tick: another pod covering the
    ticks does not help), or every pod's scrape lands only every 1 s (H2). The flagged
    pods stay in the window - queue and tokens - and m, HIGH, is no donor for the
    CRITICAL r (only the donor is blocked)."""
    redis = _Redis()
    scraped = _slow if case == "all_pods_slow_scrapes" else _fresh
    _write_pod(redis, "m", "m-0", running=2, gen_per_s=2_000, scraped=scraped)
    _write_pod(redis, "m", "m-1", running=2, gen_per_s=2_000, scraped=scraped,
               missing=LOST if case == "pod_lost_ticks" else ())
    for pod in ("r-0", "r-1"):
        _write_pod(redis, "r", pod, running=20, waiting=10, gen_per_s=10, scraped=scraped)
    registry = _registry("m", "r")
    snapshot = _snapshot(redis, registry)
    m = snapshot.models["m"]
    flagged = {"complete": (), "pod_lost_ticks": ("m-1",), "all_pods_slow_scrapes": ("m-0", "m-1")}[case]
    assert m.sample_incomplete_pods == flagged
    assert set(m.per_pod) == {"m-0", "m-1"}
    assert m.avg_running == pytest.approx(4.0)
    assert m.generation_tokens == pytest.approx(2 * 2_000 * 30, rel=0.02)

    result = _tick(snapshot, registry, *DONOR_VIEW)
    assert result.classifications["m"].state == ModelState.HIGH
    assert result.classifications["r"].state == ModelState.CRITICAL
    assert _released(result, "m") is (case == "complete")
    if case != "complete":
        assert result.model_contexts["m"]["sample_quality_hold"] == "samples_incomplete"
        assert f"samples_incomplete:m:{','.join(flagged)}" in result.events


def test_h2_receivers_with_slow_scrapes_scale_up_by_the_normal_step():
    """H2: every pod of the CRITICAL r scrapes only every 1 s (all flagged). The TSS is
    computed, the rescue takes its one step on a free GPU, and the saturation rescue still
    sees the pods' newest gauges."""
    redis = _Redis()
    for pod in ("r-0", "r-1"):
        _write_pod(redis, "r", pod, running=20, waiting=10, gen_per_s=10, scraped=_slow)
    registry = _registry("r")
    snapshot = _snapshot(redis, registry)
    assert snapshot.models["r"].sample_incomplete_pods == ("r-0", "r-1")
    result = _tick(
        snapshot, registry,
        _binding("r-0", "r", 0), _binding("r-1", "r", 1),
        _binding("r-2", "r", 2, awake=False), _binding("r-3", "r", 3, awake=False),
    )
    context = result.model_contexts["r"]
    assert result.classifications["r"].state == ModelState.CRITICAL
    assert context["z_m"] is not None
    up = sum(a.delta for a in expand_relays(result.actions) if isinstance(a, ScaleAction) and a.model == "r"
             and a.delta > 0)
    assert up == 1
    # The saturation rescue's input: each pod's newest gauges, observed (scraped_ms)
    # just before the window end (round 4: the observation time, not the doc stamp).
    for pod in snapshot.models["r"].per_pod.values():
        assert (pod.latest_instant_ms, pod.latest_waiting) == (_slow(END), 10.0)


def test_m1_a_serving_pod_without_any_doc_blocks_the_donor():
    """M1: m-1 serves (fleet view) but the gateway wrote no doc of it. m is HIGH on m-0
    alone - no donor for the CRITICAL r."""
    redis = _Redis()
    _write_pod(redis, "m", "m-0", running=2, gen_per_s=4_000)
    for pod in ("r-0", "r-1"):
        _write_pod(redis, "r", pod, running=20, waiting=10, gen_per_s=10)
    registry = _registry("m", "r")
    snapshot = _snapshot(redis, registry)
    assert snapshot.models["m"].sample_incomplete_pods == ()
    result = _tick(snapshot, registry, *DONOR_VIEW)
    assert result.classifications["m"].state == ModelState.HIGH
    assert result.model_contexts["m"]["sample_quality_hold"] == "pod_missing"
    assert not _released(result, "m")


@pytest.mark.parametrize("min_sample_ratio", [0.0, 0.9])
def test_queue_and_tokens_come_from_one_pod_set(min_sample_ratio):
    """m-1's scrape froze before the window (I3): its docs repeat a frozen queue (50
    running) and frozen counters - left out of queue AND tokens. m-2 scrapes slowly
    (flagged): in queue AND tokens. At any min ratio."""
    frozen = lambda ts: min(ts, END - 35_000) - 50  # noqa: E731
    redis = _Redis()
    _write_pod(redis, "m", "m-0", running=2, gen_per_s=1_000)
    _write_pod(redis, "m", "m-1", running=50, gen_per_s=1_000, scraped=frozen)
    _write_pod(redis, "m", "m-2", running=4, gen_per_s=1_000, scraped=_slow)
    window = _read(_store(redis, _registry("m"), min_sample_ratio))
    assert set(window.per_pod) == {"m-0", "m-2"}
    assert window.avg_running == pytest.approx(2.0 + 4.0)
    assert window.generation_tokens == pytest.approx(2 * 1_000 * 30, rel=0.02)
    assert window.scrape_stale_pods == ("m-1",)
    assert window.sample_incomplete_pods == (("m-2",) if min_sample_ratio else ())


def test_queue_level_is_the_mean_of_all_docs_not_biased_low():
    """3 of 60 instant docs lost (coverage 0.95, not flagged): the queue level is the
    pod's true level, not 57/60 of it (the v1 failure: Q too low -> Z too high)."""
    lost = {END - 10_000, END - 5_000, END - 1_000}
    redis = _Redis()
    _write_pod(redis, "m", "m-0", running=6, waiting=2, missing=lost)
    window = _read(_store(redis, _registry("m")))
    assert window.sample_incomplete_pods == ()
    assert window.avg_running == pytest.approx(6.0)
    assert window.avg_waiting == pytest.approx(2.0)
    assert window.generation_tokens == pytest.approx(100.0 * 30)


@pytest.mark.parametrize("case", ["start_doc_lost", "no_baseline_in_lookback"])
def test_a_lost_baseline_still_gives_the_true_token_rate(case):
    """H1: the histogram doc at the window start is lost - the delta runs from the doc
    10 s earlier (40 s of tokens, formerly 1.33x). H1b: no doc at or before the start at
    all - the baseline is the first doc inside the window (20 s of tokens, formerly
    0.67x). Divided by the real scraped_ms span, the rate is the true one for H1.
    H1b's 20 s span is under 0.67 windows: since review L3 (10-28) the pod is unknown
    (out of queue and tokens, flagged), not scaled up."""
    redis = _Redis()
    _write_pod(redis, "m", "m-0", running=4)
    key = "tre:v2:hist:default/m-0"
    keep = (lambda score: score != END - WINDOW) if case == "start_doc_lost" else (lambda score: score > END - WINDOW)
    redis.zsets[key] = [(score, member) for score, member in redis.zsets[key] if keep(score)]
    window = _read(_store(redis, _registry("m")))
    if case == "start_doc_lost":
        assert window.generation_tokens == pytest.approx(100.0 * 30)
        assert window.prompt_tokens == pytest.approx(50.0 * 30)
        assert window.sample_incomplete_pods == ()
    else:
        assert window.generation_tokens is None and window.prompt_tokens is None
        assert window.avg_running == 0.0 and window.per_pod == {}
        assert window.sample_incomplete_pods == ("m-0",)


# --------------------------------------------------- review round 3, default 2 s period

EVIDENCE_2S = 2_000


@pytest.mark.parametrize(
    "case", ["reset_early_in_window", "reset_mid_window", "reset_at_the_last_doc", "no_hist_doc_in_window"]
)
def test_queue_and_tokens_stay_on_one_pod_set_after_a_reset_or_a_lost_hist(case):
    """M2 at the 2 s default. m-1's engine restarted 25 s before the end: the delta
    restarts at the first doc after the reset (span 24 s >= 0.67 windows, scaled: the
    true rate), flagged, in queue AND tokens. Restarted 15 s before the end (span 14 s,
    below 0.67 windows; review L3 10-28), a reset with no doc after it, or a pod whose
    histogram docs are all missing from the window: out of queue AND tokens, flagged."""
    redis = _Redis()
    _write_pod(redis, "m", "m-0", running=2, gen_per_s=1_000, evidence=EVIDENCE_2S)
    kw = {
        "reset_early_in_window": {"reset_at": END - 25_000},
        "reset_mid_window": {"reset_at": END - 15_000},
        "reset_at_the_last_doc": {"reset_at": END - 1_000},
        "no_hist_doc_in_window": {"hist_missing": frozenset(range(END - WINDOW + 1, END + 1))},
    }[case]
    _write_pod(redis, "m", "m-1", running=4, gen_per_s=1_000, evidence=EVIDENCE_2S, **kw)
    window = _read(_store(redis, _registry("m"), evidence=EVIDENCE_2S))
    assert window.sample_incomplete_pods == ("m-1",)
    kept = case == "reset_early_in_window"
    assert set(window.per_pod) == ({"m-0", "m-1"} if kept else {"m-0"})
    assert window.avg_running == pytest.approx(6.0 if kept else 2.0)
    assert window.generation_tokens == pytest.approx((2 if kept else 1) * 1_000 * 30, rel=0.02)
    assert window.avg_running == pytest.approx(sum(pod.avg_running for pod in window.per_pod.values()))


def test_a_round_300_ms_late_gives_no_stale():
    """Round 4: the end round came 300 ms late - its docs carry the next instant stamp,
    so no instant doc sits at the window end (rounds 1-3 waited for that doc: the
    window stayed stale for good). That round still wrote the END histogram boundary;
    the gateway's last_hist_ms (END, in that round's MULTI) proves the window: the
    sampler publishes it on its first read, no stale window, no retry. Round 5: an older
    last_hist_ms (only instant rounds after it) does not."""
    import asyncio

    from tre_controller.loops.metrics_task import PhaseAlignedSampler, SnapshotBox, window_freshness

    redis = _Redis()
    for pod in ("m-0", "m-1"):
        _write_pod(redis, "m", pod, running=2, evidence=EVIDENCE_2S, missing={END})
    redis.hashes["tre:v2:gateway:cadence"] = {
        "instant_ms": str(INSTANT), "evidence_ms": str(EVIDENCE_2S), "written_ms": str(END + 805),
        "last_hist_ms": str(END), "version": "t",
    }
    store = _store(redis, _registry("m"), evidence=EVIDENCE_2S, expected_cadence=(INSTANT, EVIDENCE_2S))
    snapshot = store.read_snapshot(END - WINDOW, END, start_exclusive=True)
    assert END not in snapshot.models["m"].instant_ticks_ms and not snapshot.stale
    assert window_freshness(snapshot, window_end_ms=END, window_ms=WINDOW, period_ms=EVIDENCE_2S).fresh

    clock = [END + 1_000]

    async def sleep(seconds):
        clock[0] += int(seconds * 1000)

    box = SnapshotBox()
    sampler = PhaseAlignedSampler(store, box, window_ms=WINDOW, period_ms=EVIDENCE_2S, clock_ms=lambda: clock[0],
                                  sleep=sleep)
    sampler.last_end_ms = END - EVIDENCE_2S
    outcome = asyncio.run(sampler.run_once())
    assert (outcome.kind, outcome.window_end_ms, outcome.attempts) == ("published", END, 1)
    assert sampler.consecutive_stale == 0 and box.get().ts_ms == END and not box.get().stale

    redis.hashes["tre:v2:gateway:cadence"]["last_hist_ms"] = str(END - EVIDENCE_2S)
    snapshot = store.read_snapshot(END - WINDOW, END, start_exclusive=True)
    freshness = window_freshness(snapshot, window_end_ms=END, window_ms=WINDOW, period_ms=EVIDENCE_2S)
    assert not freshness.fresh and freshness.reason.startswith(f"hist_not_written:{END}")


def test_a_suffix_flag_survives_the_serving_restriction_and_holds_the_donor():
    """L1 at the 2 s default: m-1 sleeps but its gateway docs are in the window; m-0
    serves and lost 4 of the 16 instant docs of the O1 suffix (the last 8 s; the whole
    window is fine). The decision window is that suffix; restricting it to the serving
    pod keeps its flag, so m holds its donor evidence (no shrink / release / donor)."""
    from tre_controller.loops.tick import _model_contexts
    from tre_controller.signals.trs import BreakpointWindowConfig, SignalState

    bp = END - 8_000
    redis = _Redis()
    lost = frozenset(range(bp + INSTANT, END + 1, 4 * INSTANT))
    _write_pod(redis, "m", "m-0", running=1, gen_per_s=100, evidence=EVIDENCE_2S, missing=lost)
    _write_pod(redis, "m", "m-1", running=0, gen_per_s=0, evidence=EVIDENCE_2S)
    registry = _registry("m")
    store = _store(redis, registry, evidence=EVIDENCE_2S, suffix_period_ms=EVIDENCE_2S,
                   suffix_breakpoint=lambda model: bp)
    snapshot = store.read_snapshot(END - WINDOW, END, start_exclusive=True)
    assert snapshot.models["m"].sample_incomplete_pods == ()
    assert [w.sample_incomplete_pods for w in snapshot.models["m"].suffix_windows] == [("m-0",)]

    state = SignalState(breakpoint=BreakpointWindowConfig(
        enabled=True, onset_guard=False, min_evidence_ms=4_000))
    state.observe_traffic("m", has_traffic=True, window_start_ms=bp - WINDOW, window_end_ms=bp)
    view = ClusterView(topology=registry.topology(), bindings=(
        _binding("m-0", "m", 0), _binding("m-1", "m", 1, awake=False)))
    contexts, _events = _model_contexts(snapshot, registry, cluster_view=view, signal_state=state)
    assert contexts["m"]["sample_quality_hold"] == "samples_incomplete"
    assert contexts["m"]["window_idle"] is False
