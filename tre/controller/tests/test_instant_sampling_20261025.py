"""Instant sampling (2026-10-25, design docs/design/20261025-instant-sampling.md).

Instant docs every 0.5 s, histogram docs every evidence period. A doc counts only if its
scrape was at most ``max_scrape_age_ms`` old at its tick; a pod is complete for a window
only on its OWN valid docs (another pod covering the ticks does not help); an incomplete
serving pod is left out and the model gives no scale-down / release evidence (the I4
treatment of a scrape-stale pod); a pod's queue level is the mean over its valid samples.
Count-based rules keep their span at any evidence period (saturation confirm: >= 20 s).
"""
from __future__ import annotations

import pytest

from relay_view import expand_relays
from test_signal_validity_20261004 import _Queue, _Redis, _registry
from tre_controller.loops.tick import run_planner_tick
from tre_controller.planning.classify import ModelState
from tre_controller.planning.planner import ClusterView, ScaleAction
from tre_controller.signals.saturation import SaturationRescueConfig, SaturationSample, SaturationTracker
from tre_controller.store.metrics_store import MetricsStore
from tre_sm.allocator.slots import Binding, Slot

INSTANT = 500
EVIDENCE = 10_000
END = 1_000_000_000_000
WINDOW = 30_000


def _write_pod(redis, model, pod, *, running, waiting=0.0, gen_per_s=100.0, missing=(), scrape_age_ms=lambda ts: 50):
    """Gateway docs of one pod over (END - 40 s, END]: instant docs every 0.5 s (except
    ticks in ``missing``), histogram docs every 10 s; ``scraped_ms = ts - scrape_age_ms(ts)``."""
    redis.sadd(f"tre:v2:pods:{model}", f"default/{pod}")
    for ts in range(END - 40_000, END + 1, INSTANT):
        base = {"timestamp": ts, "written_ms": ts + 5, "pod_name": pod, "scraped_ms": ts - scrape_age_ms(ts)}
        if ts % EVIDENCE == 0:
            n = (ts - (END - 40_000)) // 1000
            redis.zadd(f"tre:v2:hist:default/{pod}", {**base, "model_histogram_metrics": {
                f"{model}/request_prompt_tokens": {"sum": 50.0 * n, "count": n, "buckets": {"+Inf": n}},
                f"{model}/request_generation_tokens": {"sum": gen_per_s * n, "count": n, "buckets": {"+Inf": n}},
            }}, ts)
        if ts in missing:
            continue
        redis.zadd(f"tre:v2:inst:default/{pod}", {**base, "model_metrics": {
            f"{model}/num_requests_running": running, f"{model}/num_requests_waiting": waiting,
        }}, ts)


def _store(redis, registry, **kw):
    return MetricsStore(
        redis, registry, instant_sample_interval_ms=INSTANT, evidence_period_ms=EVIDENCE,
        max_scrape_age_ms=1_000, min_sample_ratio=0.9, **kw,
    )


def _read(store, model="m"):
    return store.read_model_window(model, END - WINDOW, END, use_cache=False, start_exclusive=True)


#: 10 of the window's 60 instant ticks (1 in 6: ratio 0.83 < 0.9).
LOST = frozenset(range(END - WINDOW + INSTANT, END + 1, 6 * INSTANT))


@pytest.mark.parametrize("incomplete", [False, True], ids=["complete", "pod_lost_ticks"])
def test_a_pod_missing_ticks_is_incomplete_although_another_pod_covers_them(incomplete):
    """m-1 loses 1 in 6 of its instant docs; m-0 has every tick, so the gateway-wide tick
    union is complete (the former check passed). m-1 is incomplete on its own docs: left
    out, and m - HIGH on m-0 alone - is not donor evidence for the CRITICAL r."""
    redis = _Redis()
    _write_pod(redis, "m", "m-0", running=2, gen_per_s=2_000)
    _write_pod(redis, "m", "m-1", running=2, gen_per_s=2_000, missing=LOST if incomplete else ())
    for pod in ("r-0", "r-1"):
        _write_pod(redis, "r", pod, running=20, waiting=10, gen_per_s=10)
    registry = _registry("m", "r")
    snapshot = _store(redis, registry).read_snapshot(END - WINDOW, END, use_cache=False, start_exclusive=True)
    m = snapshot.models["m"]
    assert END in m.instant_ticks_ms  # the sampler's end-tick check passes either way
    assert m.sample_incomplete_pods == (("m-1",) if incomplete else ())
    assert set(m.per_pod) == ({"m-0"} if incomplete else {"m-0", "m-1"})

    def binding(pod, model, gpu, awake=True):
        return Binding(serve_id=pod, model=model, slot=Slot("n", (gpu,)), awake=awake)

    view = ClusterView(topology=registry.topology(), bindings=(
        binding("m-0", "m", 0), binding("m-1", "m", 1), binding("r-0", "r", 2), binding("r-1", "r", 3),
        binding("r-2", "r", 0, awake=False), binding("r-3", "r", 1, awake=False),
    ))
    result = run_planner_tick(snapshot, queue=_Queue(), registry=registry, rescue_due=True, fairness_due=True,
                              cluster_view=view)
    assert result.classifications["m"].state == ModelState.HIGH
    assert result.classifications["r"].state == ModelState.CRITICAL
    released = any(isinstance(a, ScaleAction) and a.model == "m" and a.delta < 0
                   for a in expand_relays(result.actions)) or any(
        e.startswith("safescale_probe_skipped:m:") for e in result.events)
    assert released is not incomplete
    if incomplete:
        assert result.model_contexts["m"]["signal_hold_reason"] == "samples_incomplete"
        assert "samples_incomplete:m:m-1" in result.events


def test_docs_with_an_old_scrape_count_as_missing():
    """The pod's scrape stalls for the window's last 6 s: the gateway still writes its
    docs (frozen values, old scraped_ms). The 11 docs over 1 s old are missing (49/60 < 0.9): the pod
    is incomplete. With the scrape stalled only 0.8 s (under max_scrape_age_ms) it is not."""
    stall_from = END - 6_000

    def stalled(ts):
        return ts - stall_from + 50 if ts > stall_from else 50

    redis = _Redis()
    _write_pod(redis, "m", "m-0", running=4, scrape_age_ms=stalled)
    assert _read(_store(redis, _registry("m"))).sample_incomplete_pods == ("m-0",)

    redis = _Redis()
    _write_pod(redis, "m", "m-0", running=4, scrape_age_ms=lambda ts: 800)
    assert _read(_store(redis, _registry("m"))).sample_incomplete_pods == ()


def test_queue_level_is_the_mean_of_the_valid_samples_not_biased_low():
    """3 of 60 instant docs lost (complete at 0.95): the queue level is the pods' true
    level, not 57/60 of it (the v1 failure: Q too low -> Z too high -> false HIGH)."""
    lost = {END - 10_000, END - 5_000, END - 1_000}
    redis = _Redis()
    _write_pod(redis, "m", "m-0", running=6, waiting=2, missing=lost)
    window = _read(_store(redis, _registry("m")))
    assert window.sample_incomplete_pods == ()
    assert window.avg_running == pytest.approx(6.0)
    assert window.avg_waiting == pytest.approx(2.0)
    assert window.generation_tokens == pytest.approx(100.0 * 30)


def test_a_pod_whose_histogram_docs_stop_early_is_incomplete():
    """The window's last evidence boundary has no histogram doc of the pod: its token
    delta would cover 20 of 30 s. Incomplete, left out (not a 1/3 lower TSS)."""
    redis = _Redis()
    _write_pod(redis, "m", "m-0", running=4)
    key = "tre:v2:hist:default/m-0"
    redis.zsets[key] = [(score, member) for score, member in redis.zsets[key] if score < END]
    assert _read(_store(redis, _registry("m"))).sample_incomplete_pods == ("m-0",)


@pytest.mark.parametrize("grid_ms, windows_to_fire", [(10_000, 2), (2_000, 10)])
def test_saturation_confirm_needs_20s_of_evidence_at_any_cadence(grid_ms, windows_to_fire):
    """The onset saturation confirm is a span (20 s), not "2 windows": at the 10 s grid
    the 2nd full window fires (as before); at a 2 s grid the 10th, not the 2nd (two 30 s
    windows 2 s apart are nearly the same evidence)."""
    tracker = SaturationTracker(SaturationRescueConfig(grid_ms=grid_ms))
    sample = SaturationSample(waiting=5.0, kv=0.5, pods=1, sample_ms=0, running=4.0)
    fired_at = None
    for k in range(1, 20):
        verdict = tracker.observe(
            "m", window_end_ms=END + k * grid_ms, routable=1, reason="numerator_zero", sample=sample
        )
        if verdict.fire:
            fired_at = k
            break
    assert fired_at == windows_to_fire
    assert (fired_at - 1) * grid_ms + grid_ms >= 20_000
