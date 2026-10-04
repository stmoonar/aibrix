"""Signal validity (2026-10-04, I3 / I4).

I3: every observation carries its own validity; data older than the window is
UNKNOWN - not 0 and not the last value. I4: a model is a donor (scale-down or
release) only on a level computed from tokens observed in the current window."""

from __future__ import annotations

import json

import pytest

from relay_view import expand_relays
from tre_common.registry import ClusterTopology, ModelSpec, NodeSpec, Registry, SloSpec, TrsParams
from tre_controller.loops.tick import PaperStateCache, run_planner_tick
from tre_controller.planning.classify import ModelState, classify_all_models
from tre_controller.planning.planner import ClusterView, PlanConfig, ScaleAction, build_plan
from tre_controller.store.metrics_store import MetricsStore
from tre_sm.allocator.slots import Binding, Slot


def _ctx(z_m: float | None, y_m: float, q: float) -> dict:
    return {
        "trs": 100.0 * (z_m or 0.0), "z_m": z_m, "trs_z_m": z_m, "theta_m": 100.0, "eta_m": 500.0,
        "tss_defined": True, "Y_m": y_m, "y_m": y_m, "Q": q, "Q_ctl": max(q, 1.0),
        "signal_source": "zm", "signal_unavailable_reason": None, "request_rate_rps": 5.0 if y_m else 0.0,
        "routable_pods": 3, "assigned_replicas": 3, "awake_replicas": 3,
        "signal_warm": True, "signal_full_window": True, "signal_hold_reason": None,
    }


def _missing_ctx() -> dict:
    """The tokens-missing context of a window without token data (tick._model_contexts)."""
    return {
        "trs": 0.0, "z_m": None, "signal_source": "zm", "signal_unavailable_reason": "tokens_missing",
        "Y_m": None, "y_m": None, "Q": 0.0, "Q_ctl": 1.0, "theta_m": 100.0,
        "routable_pods": 3, "assigned_replicas": 3, "awake_replicas": 3, "request_rate_rps": None,
    }


def _plan(donor_ctx: dict):
    receiver = _ctx(0.1, 500.0, 50.0)  # CRITICAL, warm, whole window
    contexts = {"r": receiver, "d": donor_ctx}
    return build_plan(
        model_contexts=contexts,
        classifications=classify_all_models(contexts),
        model_replicas={"r": 1, "d": 3},
        idle_gpus=0,
        cfg=PlanConfig(min_replicas_per_model=1, max_replicas_per_model=8),
    )


def _gives(plan, model: str) -> bool:
    released = any(
        isinstance(a, ScaleAction) and a.model == model and a.delta < 0 for a in expand_relays(plan.actions)
    )
    return released or model in plan.probe_upscale_plans


@pytest.mark.parametrize(
    "fresh",
    [
        _ctx(3.0, 900.0, 3.0),  # HIGH: immediate donor release
        _ctx(1.1, 600.0, 5.0),  # HEALTHY: middle-zone SafeScale scale-down
        _ctx(0.0, 0.0, 0.0),  # IDLE: immediate donor release
    ],
    ids=["high", "middle_zone", "idle"],
)
def test_held_context_is_never_donor_evidence(fresh):
    """D: a window without tokens holds the model's last context (PaperStateCache); the
    held level is not current-window evidence, so the model gives nothing up."""
    assert _gives(_plan(dict(fresh)), "d")  # control: the same level from a fresh window donates

    cache = PaperStateCache(max_stale_windows=3)
    cache.apply("d", dict(fresh), tokens_available=True)
    held, events = cache.apply("d", _missing_ctx(), tokens_available=False)
    assert events == ("paper_state_stale_hold:d",)

    plan = _plan(held)
    assert not _gives(plan, "d")
    assert "donor_suppressed_breakpoint_window:d" in plan.events



def test_held_receiver_rescue_is_capped_like_thin_evidence():
    """I4: the held context does not carry the old window's evidence count, so the rescue
    of a held CRITICAL receiver takes one capped step (control: the same level from a
    whole window asks for more)."""
    fresh = dict(_ctx(0.1, 500.0, 50.0), signal_evidence_requests=50.0)
    donor = _ctx(1.1, 600.0, 5.0)

    def plan(receiver):
        contexts = {"r": receiver, "d": donor}
        return build_plan(
            model_contexts=contexts,
            classifications=classify_all_models(contexts),
            model_replicas={"r": 3, "d": 3},
            idle_gpus=4,
            cfg=PlanConfig(min_replicas_per_model=1, max_replicas_per_model=8,
                           partial_window_max_step=1, partial_window_lowevidence_requests=10),
        )

    def added(result):
        return sum(a.delta for a in expand_relays(result.actions) if isinstance(a, ScaleAction) and a.model == "r" and a.delta > 0)

    assert added(plan(dict(fresh))) > 1
    cache = PaperStateCache(max_stale_windows=3)
    cache.apply("r", dict(fresh), tokens_available=True)
    held, _ = cache.apply("r", _missing_ctx(), tokens_available=False)
    assert added(plan(held)) == 1
# ------------------------------------------------- E: frozen gateway scrape


class _Redis:
    def __init__(self):
        self.sets: dict = {}
        self.zsets: dict = {}

    def sadd(self, key, *values):
        self.sets.setdefault(key, set()).update(values)

    def smembers(self, key):
        return set(self.sets.get(key, set()))

    def zadd(self, key, doc, score):
        self.zsets.setdefault(key, []).append((float(score), json.dumps(doc)))

    def zrangebyscore(self, key, lo, hi):
        return [m for score, m in sorted(self.zsets.get(key, [])) if float(lo) <= score <= float(hi)]


class _Queue:
    def __init__(self):
        self.submitted: list = []

    def inflight_models(self) -> set:
        return set()

    def submit(self, actions):
        self.submitted.extend(actions)


def _registry() -> Registry:
    spec = ModelSpec(
        name="m", weights_path="/w", tp_size=1, min_replicas=1, max_replicas=4, vllm_image="img",
        slo=SloSpec(ttft_p95_ms=1200.0, tpot_p95_ms=100.0, e2e_p95_ms=10_000.0),
        trs=TrsParams(w_p=0.04, w_d=1.0, lambda_wait=2.625, qmin=1.0, ema_alpha=1.0, theta_m=100.0,
                      tau_crit=0.8, tau_low=1.0, tau_high=1.25, qsat=4.0, epsat=0.05, hsat=1),
    )
    return Registry(ClusterTopology(nodes=(NodeSpec(name="n", gpus=4, two_gpu_slots=((0, 1), (2, 3))),)), [spec])


@pytest.mark.parametrize("gateway_writes_scraped_ms", [True, False], ids=["scraped_ms", "old_gateway"])
def test_frozen_scrape_is_unknown_not_a_zero_token_window(gateway_writes_scraped_ms):
    """E: the gateway's /metrics fetch of the only pod fails before the window; it keeps
    writing the pod's last values every round (same counters, same queue). The window is
    UNKNOWN - no scale-up, no donor - not a zero-token window. The validity is judged
    in the gateway's clock (scraped_ms vs the doc timestamps), so the controller's clock
    plays no part. An old gateway (no scraped_ms anywhere) keeps the former behaviour."""
    end = 1_000_000_000_000
    redis = _Redis()
    pod = "default/m-0"
    redis.sadd("tre:v2:pods:m", pod)
    for ts in range(end - 40_000, end + 1, 10_000):
        base = {"timestamp": ts, "written_ms": ts + 300, "pod_name": "m-0"}
        if gateway_writes_scraped_ms:
            base["scraped_ms"] = end - 35_000  # last successful fetch: before the window
        redis.zadd("tre:v2:hist:" + pod, {**base, "model_histogram_metrics": {
            "m/request_prompt_tokens": {"sum": 5_000, "count": 40, "buckets": {"+Inf": 40}},
            "m/request_generation_tokens": {"sum": 20_000, "count": 40, "buckets": {"+Inf": 40}},
        }}, ts)
        redis.zadd("tre:v2:inst:" + pod, {**base, "model_metrics": {
            "m/num_requests_running": 20, "m/num_requests_waiting": 8, "m/gpu_cache_usage_perc": 0.95,
        }}, ts)
    registry = _registry()
    store = MetricsStore(redis, registry, instant_sample_interval_ms=10_000)
    snapshot = store.read_snapshot(end - 30_000, end, use_cache=False, start_exclusive=True)
    view = ClusterView(topology=registry.topology(),
                       bindings=(Binding(serve_id="m-0", model="m", slot=Slot("n", (0,)), awake=True),
                                 Binding(serve_id="m-1", model="m", slot=Slot("n", (1,)), awake=False)))
    queue = _Queue()
    result = run_planner_tick(snapshot, queue=queue, registry=registry, rescue_due=True, fairness_due=True,
                              cluster_view=view, paper_state_cache=PaperStateCache())
    state = result.classifications["m"].state
    if gateway_writes_scraped_ms:
        assert snapshot.models["m"].generation_tokens is None  # unknown, not 0
        assert state == ModelState.UNKNOWN
        assert not queue.submitted
        assert "scrape_stale:m:m-0" in result.events
    else:
        assert snapshot.models["m"].generation_tokens == 0.0  # former behaviour
        assert state != ModelState.UNKNOWN
