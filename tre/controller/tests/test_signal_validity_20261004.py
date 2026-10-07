"""Signal validity (2026-10-04, I3 / I4).

I3: every observation carries its own validity; data older than the window is
UNKNOWN - not 0 and not the last value. I4: a model is a donor (scale-down or
release) only on a level computed from tokens observed in the current window."""

from __future__ import annotations

import json

import pytest

from relay_view import expand_relays
from tre_common import rediskeys
from tre_common.registry import ClusterTopology, ModelSpec, NodeSpec, Registry, SloSpec, TrsParams
from tre_controller.loops.tick import PaperStateCache, run_planner_tick
from tre_controller.planning.classify import ModelState, classify_all_models
from tre_controller.planning.planner import ClusterView, PlanConfig, ScaleAction, build_plan
from tre_controller.signals.saturation import SaturationRescueConfig, SaturationTracker
from tre_controller.signals.trs import SignalState
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
    def __init__(self, now_ms: int = 7_000_000):
        self.sets: dict = {}
        self.zsets: dict = {}
        self.hashes: dict = {}
        self.instances: dict = {}  # tre:v2:gw:instances member -> heartbeat (Redis TIME ms)
        self.now_ms = now_ms  # Redis TIME: its own clock, unrelated to the doc stamps

    def time(self):
        return self.now_ms // 1000, (self.now_ms % 1000) * 1000

    def zrange(self, key, start, stop, withscores=False):
        assert key == rediskeys.GW_INSTANCES_KEY
        return sorted(self.instances.items(), key=lambda item: item[1])

    def hgetall(self, key):
        return dict(self.hashes.get(key, {}))

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


def _registry(*names: str) -> Registry:
    specs = [
        ModelSpec(
            name=name, weights_path="/w", tp_size=1, min_replicas=1, max_replicas=4, vllm_image="img",
            slo=SloSpec(ttft_p95_ms=1200.0, tpot_p95_ms=100.0, e2e_p95_ms=10_000.0),
            trs=TrsParams(w_p=0.04, w_d=1.0, lambda_wait=2.625, qmin=1.0, ema_alpha=1.0, theta_m=100.0,
                          tau_crit=0.8, tau_low=1.0, tau_high=1.25, qsat=4.0, epsat=0.05, hsat=1),
        )
        for name in names or ("m",)
    ]
    return Registry(ClusterTopology(nodes=(NodeSpec(name="n", gpus=4, two_gpu_slots=((0, 1), (2, 3))),)), specs)


END = 1_000_000_000_000


def _write_pod(redis, model: str, pod: str, *, gen_per_grid: float, running: float, waiting: float,
               scraped_ms: int | None, prompt_per_grid: float = 500.0) -> None:
    """Gateway docs of one pod on the 10 s grid over (END - 40 s, END]; the counters grow
    by ``gen_per_grid`` per grid until ``scraped_ms`` (the last successful fetch), then
    repeat (a frozen scrape). ``scraped_ms`` None: written by an old gateway, fresh."""
    redis.sadd(f"tre:v2:pods:{model}", f"default/{pod}")
    for i, ts in enumerate(range(END - 40_000, END + 1, 10_000)):
        frozen_at = i if scraped_ms is None or scraped_ms >= ts else (scraped_ms - (END - 40_000)) // 10_000
        base = {"timestamp": ts, "written_ms": ts + 300, "pod_name": pod}
        if scraped_ms is not None:
            base["scraped_ms"] = min(scraped_ms, ts)
        n = 100 + 10 * frozen_at
        redis.zadd(f"tre:v2:hist:default/{pod}", {**base, "model_histogram_metrics": {
            f"{model}/request_prompt_tokens": {"sum": 5_000.0 + prompt_per_grid * frozen_at, "count": n, "buckets": {"+Inf": n}},
            f"{model}/request_generation_tokens": {"sum": gen_per_grid * frozen_at, "count": n, "buckets": {"+Inf": n}},
        }}, ts)
        redis.zadd(f"tre:v2:inst:default/{pod}", {**base, "model_metrics": {
            f"{model}/num_requests_running": running, f"{model}/num_requests_waiting": waiting,
        }}, ts)


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


@pytest.mark.parametrize("stale", [False, True], ids=["both_fresh", "one_stale"])
def test_a_stale_serving_pod_makes_no_donor(stale):
    """E review P2-1: m has 2 serving pods; m-1 (the overloaded one) times out on
    /metrics. Its docs are left out, so the level comes from the light m-0 alone (HIGH)
    - not evidence that m can give a GPU to the CRITICAL r. With both pods fresh the
    same HIGH level is released (control)."""
    redis = _Redis()
    fresh = END
    _write_pod(redis, "m", "m-0", gen_per_grid=20_000, running=2, waiting=0, scraped_ms=fresh)
    _write_pod(redis, "m", "m-1", gen_per_grid=20_000, running=2, waiting=0,
               scraped_ms=END - 35_000 if stale else fresh)
    for pod in ("r-0", "r-1"):
        _write_pod(redis, "r", pod, gen_per_grid=100, running=20, waiting=10, scraped_ms=fresh)
    registry = _registry("m", "r")
    snapshot = MetricsStore(redis, registry, instant_sample_interval_ms=10_000).read_snapshot(
        END - 30_000, END, use_cache=False, start_exclusive=True
    )

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
    # F1-B (2026-10-07): the HIGH donor is taken through a SafeScale probe; without a
    # SafeScale controller here the planned probe is dropped with an event.
    released = any(isinstance(a, ScaleAction) and a.model == "m" and a.delta < 0
                   for a in expand_relays(result.actions)) or any(
        e.startswith("safescale_probe_skipped:m:") for e in result.events)
    assert released is not stale
    if stale:
        assert result.model_contexts["m"]["signal_hold_reason"] == "scrape_stale"


def test_scrape_validity_is_judged_per_o1_suffix():
    """E: each O1 suffix window is judged against its own start - a pod whose last
    successful fetch lies inside the window but before the last grid is valid for the
    whole window and the 20 s suffix, not for the last 10 s."""
    redis = _Redis()
    _write_pod(redis, "m", "m-0", gen_per_grid=1_000, running=4, waiting=0, scraped_ms=END - 15_000)
    store = MetricsStore(redis, _registry("m"), instant_sample_interval_ms=10_000, suffix_period_ms=10_000)
    window = store.read_model_window("m", END - 30_000, END, use_cache=False, start_exclusive=True)
    assert window.generation_tokens is not None
    last_20s, last_10s = window.suffix_windows
    assert last_20s.generation_tokens is not None
    assert last_10s.generation_tokens is None and not last_10s.per_pod


# ---------------------------------------- frozen-scrape rescue (2026-10-04)


@pytest.mark.parametrize(
    "case",
    ["rescue", "dead_gateway_instance", "scraper_dead", "no_inflight", "unreadable_inflight"],
)
def test_frozen_scrape_with_inflight_demand_is_a_one_step_rescue(case):
    """No serving pod of m answered /metrics in the whole window, while the gateway
    scraper works (model o was scraped): vLLM's API-server loop is saturated. With
    requests in flight on m-0 (live gateway instance) m gets +1 on this first window and
    is never a donor. No rescue when the in-flight count comes only from a dead gateway
    instance, when every pod of every model is stale (the scraper is dead), when
    nothing is in flight, or when the in-flight count is unknown."""
    redis = _Redis()
    stale = END - 35_000
    _write_pod(redis, "m", "m-0", gen_per_grid=0, running=0, waiting=0, scraped_ms=stale)
    _write_pod(redis, "o", "o-0", gen_per_grid=2_000, running=4, waiting=0,
               scraped_ms=stale if case == "scraper_dead" else END)
    live_hb, dead_hb = redis.now_ms - 2_000, redis.now_ms - 120_000
    redis.instances = {"gw-live": live_hb, "gw-dead": dead_hb}
    inflight = {"rescue": {"gw-live": 6}, "dead_gateway_instance": {"gw-dead": 6},
                "scraper_dead": {"gw-live": 6}, "no_inflight": {"gw-live": 0},
                "unreadable_inflight": {"gw-live": None}}[case]
    redis.hashes[rediskeys.gw_inflight_key("m-0")] = {
        # P3-1: an unreadable live field is unknown - it never invents rescue demand.
        instance: "not-json" if total is None else json.dumps({"total": total, "non_continuable": 0, "ts": 1})
        for instance, total in inflight.items()
    }
    registry = _registry("m", "o")
    snapshot = MetricsStore(redis, registry, instant_sample_interval_ms=10_000).read_snapshot(
        END - 30_000, END, use_cache=False, start_exclusive=True
    )

    def binding(pod, model, gpu, awake=True):
        return Binding(serve_id=pod, model=model, slot=Slot("n", (gpu,)), awake=awake)

    view = ClusterView(topology=registry.topology(), bindings=(
        binding("m-0", "m", 0), binding("m-1", "m", 1, awake=False), binding("o-0", "o", 2),
    ))
    state = SignalState(saturation=SaturationTracker(SaturationRescueConfig()))
    result = run_planner_tick(snapshot, queue=_Queue(), registry=registry, rescue_due=True, fairness_due=True,
                              cluster_view=view, signal_state=state, paper_state_cache=PaperStateCache())
    deltas = [a.delta for a in expand_relays(result.actions) if isinstance(a, ScaleAction) and a.model == "m"]
    assert all(delta >= 0 for delta in deltas)  # never a donor
    if case == "rescue":
        assert sum(deltas) == 1
        assert result.model_contexts["m"]["saturation_reason"] == "scrape_stale_inflight"
    else:
        assert sum(deltas) == 0


# ------------------------------------------ IDLE window evidence (2026-10-06 P2-2)


@pytest.mark.parametrize("case", ["fresh_drained", "scrape_froze_early", "inflight", "no_live_gateway"])
def test_an_idle_window_needs_a_last_grid_scrape_and_nothing_in_flight(case):
    """P2-2: IDLE (whole-surplus release, O1 donor-hold exemption) needs every serving
    pod scraped in the last grid of the window and zero gateway in-flight on it. A pod
    whose scrape froze early in the window repeats zero-token docs while traffic
    arrives; an in-flight request, or no live gateway instance (unknown), is not idle."""
    redis = _Redis()
    scraped = END - 25_000 if case == "scrape_froze_early" else END
    _write_pod(redis, "m", "m-0", gen_per_grid=0, prompt_per_grid=0, running=0, waiting=0, scraped_ms=scraped)
    if case != "no_live_gateway":
        redis.instances = {"gw-live": redis.now_ms - 2_000}
    redis.hashes[rediskeys.gw_inflight_key("m-0")] = {
        "gw-live": json.dumps({"total": 3 if case in ("scrape_froze_early", "inflight") else 0, "ts": 1})
    }
    registry = _registry("m")
    snapshot = MetricsStore(redis, registry, instant_sample_interval_ms=10_000).read_snapshot(
        END - 30_000, END, use_cache=False, start_exclusive=True
    )
    assert (snapshot.models["m"].prompt_tokens, snapshot.models["m"].generation_tokens) == (0.0, 0.0)
    view = ClusterView(topology=registry.topology(),
                       bindings=(Binding(serve_id="m-0", model="m", slot=Slot("n", (0,)), awake=True),))
    result = run_planner_tick(snapshot, queue=_Queue(), registry=registry, rescue_due=True, fairness_due=True,
                              cluster_view=view)
    assert result.model_contexts["m"]["window_idle"] is (case == "fresh_drained")


# ------------------------------- one gateway liveness bound (2026-10-06 P3-2)


@pytest.mark.parametrize("heartbeat_age_ms, counted", [(2_000, True), (6_000, False)], ids=["live", "stale"])
def test_every_inflight_reader_uses_the_registry_instance_staleness(tmp_path, heartbeat_age_ms, counted):
    """P3-2: the frozen-scrape demand (metrics store) and the early commit
    (GatewayInflightReader, built as the app builds it) judge a gateway plugin instance
    live by the same bound - the registry's service_manager.sleep.instance_staleness_s
    (4 s here): a heartbeat 6 s old is ignored by both, one 2 s old counts in both."""
    from pathlib import Path

    from tre_common.gateway_inflight import instance_max_age_ms
    from tre_common.registry import load_registry
    from tre_controller.planning.safescale_direct import GatewayInflightReader

    text = (Path(__file__).resolve().parents[2] / "deploy" / "registry.yaml").read_text()
    assert text.count("\n    instance_staleness_s: 10\n") == 1
    path = tmp_path / "registry.yaml"
    path.write_text(text.replace("\n    instance_staleness_s: 10\n", "\n    instance_staleness_s: 4\n"))
    registry = load_registry(str(path))
    redis = _Redis()
    _write_pod(redis, "m", "m-0", gen_per_grid=0, running=0, waiting=0, scraped_ms=END - 35_000)  # frozen
    redis.instances = {"gw": redis.now_ms - heartbeat_age_ms}
    redis.hashes[rediskeys.gw_inflight_key("m-0")] = {"gw": json.dumps({"total": 3, "non_continuable": 0, "ts": 1})}
    window = MetricsStore(redis, registry, instant_sample_interval_ms=10_000).read_model_window(
        "m", END - 30_000, END, use_cache=False, start_exclusive=True
    )
    assert window.scrape_stale_pods == ("m-0",)
    assert window.scrape_stale_inflight.get("m-0", 0) == (3 if counted else 0)
    reader = GatewayInflightReader(redis, max_age_ms=instance_max_age_ms(registry))
    assert reader(("m-0",)) == (3.0 if counted else None)
