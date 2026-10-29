"""PreServe policy: event feed, iteration clock, Tier-2 thresholds, faithful composition (v1),
mLSTM Tier-1, load errors."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import pytest

from tre_baselines import trace_oracle as to
from tre_baselines.config import Config, ModelLimits
from tre_baselines.policies import POLICIES, Policy, build_policy
from tre_baselines.policies import preserve_tier1 as t1
from tre_baselines.policies.preserve import PreServePolicy, predictor_set_for, trace_name_of
from tre_baselines.snapshot import ClusterSnapshot, ModelSnapshot, PodSnapshot, ReplayInfo, RequestEvent

FIX = Path(__file__).resolve().parent / "fixtures"
MU = {"p": 100.0, "d": 100.0, "t": 150.0}
# window 0: P=120000 D=60000 -> N=2; window 1: P=60000 D=30000 -> N=1;
# window 2 (partial, 300 s): P=60000 -> N=2. Trace spans 3 windows of 600 s.
ORACLE = to.build_oracle(
    [to.TraceRequest(10.0, "m", 60_000, 30_000), to.TraceRequest(20.0, "m", 60_000, 30_000),
     to.TraceRequest(700.0, "m", 60_000, 30_000), to.TraceRequest(1500.0, "m", 60_000, 0)],
    path="repo/traces/case/seed1/trace.json", fmt=to.FORMAT_REQUESTS, window_s=600.0, duration_s=1500.0,
)
REPLAY = ReplayInfo(t0_ms=0, trace_path="/mnt/x/traces/case/seed1/trace.json")


def cfg(params: dict, models=("m",), seed: int = 0) -> Config:
    lim = {m: ModelLimits(name=m, min_replicas=1, max_replicas=8, gpus_per_replica=1,
                          ttft_slo_ms=500.0, tpot_slo_ms=75.0, max_num_seqs=256) for m in models}
    return Config(sm_url="http://sm.test", redis_url="redis://r.test", policy="preserve",
                  policy_params=params, models=lim, seed=seed)


def make(**params) -> PreServePolicy:
    """The feed / Tier-2 tests: oracle Tier-1 on 600 s windows, no load errors."""
    p = {"mu": {"m": dict(MU)}, "max_output_len": 4096, "tier1": "oracle", "load_error_mode": "none",
         "window_s": 600.0}
    p.update(params)
    return PreServePolicy(cfg(p), oracle=ORACLE)


def pod(name: str, t: int, itl=(0.0, 0.0), M: Optional[int] = 10_000, *, running=1.0,
        kv: Optional[float] = None, cover: bool = True) -> PodSnapshot:
    """Default: a busy engine (running 1, so it never reads as idle), events cover it."""
    blocks, bs = (None, None) if M is None else (M // 8, 8)
    return PodSnapshot(pod=name, model="m", node="n0", gpu_ids=(0,), running=running, waiting=0.0,
                       kv_usage=kv, counters={"itl_sum": itl[0], "itl_count": itl[1]},
                       num_gpu_blocks=blocks, block_size=bs, scraped_at_ms=t, events_cover=cover)


def ev(kind: str, req: str, p: Optional[str], ts: int, inn=None, mx=None, reissue="none",
       cid: Optional[str] = "auto") -> RequestEvent:
    return RequestEvent(kind=kind, model="m", pod=p, req_id=req, ts_ms=ts, in_tokens=inn,
                        in_src="header", max_tokens=mx, out_tokens=None, status=None, reissue=reissue,
                        client_id=f"c-{req}" if cid == "auto" else cid)


def snap(now: int, pods, events=(), awake=None, unscraped=(), replay=None, since=0, tick_s=2.0) -> ClusterSnapshot:
    ms = ModelSnapshot(model="m", awake=len(pods) if awake is None else awake, min_replicas=1,
                       max_replicas=8, gpus_per_replica=1, ttft_slo_ms=500.0, tpot_slo_ms=75.0,
                       max_num_seqs=256, pods=tuple(pods), events=tuple(events), unscraped=tuple(unscraped),
                       events_since_ms=since)
    return ClusterSnapshot(now_ms=now, tick_s=tick_s, models={"m": ms}, replay=replay)


def amap(policy: PreServePolicy, name: str = "p0"):
    return policy._models["m"].pods[name].amap


def load(req: str, p: str, ts: int, P: int, D: int = 1):
    return [ev("arr", req, p, ts, P, D), ev("ft", req, p, ts)]


# ------------------------------------------------------------------ registry / factory


def test_registered_and_factory_fails_closed() -> None:
    assert "preserve" in POLICIES
    trace = str(FIX / "preserve_trace_requests.json")
    models = ("dsqwen-7b", "dsllama-8b")
    mu = {m: dict(MU) for m in models}
    base = {"mu": mu, "trace_path": trace, "tier1": "oracle"}
    policy = build_policy("preserve", cfg(base, models))
    assert isinstance(policy, Policy) and policy.oracle.n_requests == 5
    assert policy.map_length("dsqwen-7b") == 100  # trace max max_tokens 70 x 1.2 = 84 < l
    assert policy.map_length("unknown") == 4916  # 4096 x 1.2
    assert policy.map_length("unknown", 0.01, 5.0) == 4916 and policy.map_length("dsqwen-7b", 0.01, 5.0) == 600
    with pytest.raises(ValueError, match="managed models"):
        build_policy("preserve", cfg({**base, "mu": {"dsqwen-7b": dict(MU)}}, models))
    with pytest.raises(ValueError):
        build_policy("preserve", cfg({**base, "mu": None}, models))
    with pytest.raises(ValueError):
        build_policy("preserve", cfg({**base, "mu": {**mu, "dsqwen-7b": {"p": 1, "d": 1, "t": 0}}}, models))
    with pytest.raises(ValueError, match="trace_path"):
        build_policy("preserve", cfg({"mu": mu, "tier1": "oracle"}, models))
    with pytest.raises(ValueError, match="calibration pending"):
        build_policy("preserve", cfg({**base, "calibration": "pending"}, models))
    with pytest.raises(ValueError, match="upper bound"):  # D_hat with errors is a prediction
        build_policy("preserve", cfg({**base, "out_len_is_upper_bound": True}, models))
    # mlstm (the default) needs the weights with their sha256 and a history file next to the trace
    with pytest.raises(ValueError, match="predictors"):
        build_policy("preserve", cfg({"mu": mu, "trace_path": trace}, models))
    with pytest.raises(ValueError, match="lead_s"):
        build_policy("preserve", cfg({"mu": mu, "trace_path": trace, "lead_s": 5}, models))


def test_trace_name_and_predictor_set() -> None:
    assert trace_name_of("/etc/x/camp/plain/Real_code/seed1/traces_tre.effective.json") == "Real_code"
    rules = [["Real_code", "code"], ["Real_conv", "conv"]]
    assert predictor_set_for("Real_code", rules) == "code" and predictor_set_for("Real_conv", rules) == "conv"
    assert predictor_set_for("Tight_pool", rules) == "pooled"


# ------------------------------------------------------------- iteration clock


def test_iterations_follow_measured_tpot_and_events_land_at_their_time() -> None:
    p = make()
    p.decide(snap(1000, [pod("p0", 1000)], load("r1", "p0", 1000, 1000, 512)))
    m = amap(p)
    assert m.it == 0 and m.at(0) == pytest.approx(0.1) and m.at(511) == pytest.approx(0.1511)
    # itl counters: 64 tokens in 1.0 s -> TPOT 1/64 s -> 64 iterations per second
    d = p.decide(snap(3000, [pod("p0", 3000, (1.0, 64.0))],
                      [ev("arr", "r2", "p0", 2000, 0, 100), ev("ft", "r2", "p0", 2500)]))
    assert d["m"].inputs["tier2"]["p0"]["tpot_src"] == "itl"
    assert m.it == 128
    assert m.requests[("r2", "p0")].start == 96  # placed at ft=2500, not arr=2000
    assert m.at(0) == pytest.approx((1128 + 32) / 10_000)
    assert m.at(35) == pytest.approx((1163 + 67) / 10_000)
    assert m.at(68) == pytest.approx(1196 / 10_000)


def test_tpot_falls_back_to_the_slo_and_carries_fractions() -> None:
    p = make()
    p.decide(snap(1000, [pod("p0", 1000, (1.0, 64.0))]))
    d = p.decide(snap(3000, [pod("p0", 3000, (1.0, 64.0))]))  # no new ITL samples
    assert d["m"].inputs["tier2"]["p0"]["tpot_src"] == "slo"
    assert amap(p).it == 26  # 2 s / 75 ms = 26.67
    p.decide(snap(5000, [pod("p0", 5000, (1.0, 64.0))]))
    assert amap(p).it == 53  # 26.67 + 0.67 carried
    # a counter reset (engine restart) is not a sample either
    d = p.decide(snap(7000, [pod("p0", 7000, (0.1, 5.0))]))
    assert d["m"].inputs["tier2"]["p0"]["tpot_src"] == "slo"
    assert p.anomalies["m"]["counter_reset"] == 1


# ----------------------------------------------------------- event robustness


def test_early_done_and_reissue_migration() -> None:
    p = make()
    pods = [pod("p0", 1000), pod("p1", 1000)]
    p.decide(snap(1000, pods, load("r", "p0", 1000, 1000, 100)))
    assert amap(p, "p0").at(0) == pytest.approx(0.1)
    # the sidecar re-sends r to p1 (prompt + generated so far, the remaining max_tokens)
    p.decide(snap(1000, pods, [ev("arr", "r", "p1", 1000, 1050, 50, "continued"),
                               ev("ft", "r", "p1", 1000, reissue="continued")]))
    assert amap(p, "p0").at(0) == 0.0
    assert amap(p, "p1").at(0) == pytest.approx(0.105) and amap(p, "p1").at(49) == pytest.approx(0.1099)
    p.decide(snap(1000, pods, [ev("done", "r", "p0", 1000)]))  # the original is released
    assert all(u == 0.0 for u in amap(p, "p0").window(200))
    assert amap(p, "p1").at(0) == pytest.approx(0.105)
    p.decide(snap(1000, pods, [ev("done", "r", "p1", 1000, reissue="continued")]))
    assert all(u == 0.0 for u in amap(p, "p1").window(200))
    assert p.anomalies["m"]["attempt_superseded"] == 1


def test_done_before_first_token_does_not_create_a_decode_request():
    p = make()
    events = [ev("arr", "q", "p0", 1000, 500, 20), ev("done", "q", "p0", 1100),
              ev("ft", "q", "p0", 1200), ev("ft", "unknown", "p0", 1200)]
    d = p.decide(snap(1200, [pod("p0", 1200)], events))["m"]
    assert d.inputs["tier2"]["p0"]["reqs"] == 0
    assert d.inputs["tier2"]["p0"]["pending"] == 0
    assert d.inputs["anom"]["ft_without_arr"] == 1


def test_pod_without_kv_capacity_is_skipped_unless_configured() -> None:
    p = make()
    d = p.decide(snap(1000, [pod("p0", 1000, M=None)]))
    assert d["m"].inputs["skipped"] == {"p0": "no_kv_capacity"}
    assert d["m"].inputs["frozen"] is True and "no_kv_capacity" in d["m"].inputs["t2_hold"]
    q = make(kv_capacity_tokens={"m": 1000})
    q.decide(snap(1000, [pod("p0", 1000, M=None)], load("r", "p0", 1000, 500, 10)))
    assert amap(q).at(0) == pytest.approx(0.5)


def test_bookkeeping_expires_but_a_request_in_the_map_does_not() -> None:
    p = make(req_ttl_s=10)
    p.decide(snap(1000, [pod("p0", 1000)], load("r", "p0", 1000, 100, 10) + [ev("arr", "s", "pZ", 1000, 5, 5)]))
    p.decide(snap(20_000, [pod("p0", 20_000)]))
    assert p._models["m"].pending == {} and p.anomalies["m"]["expired_buffered_arr"] == 1
    assert list(p._models["m"].active) == [("r", "p0")]  # leaves the map only on done


def test_map_clears_only_on_done() -> None:
    events = [e for i in range(10) for e in load(f"r{i}", "p0", 1000, 100, 20)]
    p = make()
    p.decide(snap(1000, [pod("p0", 1000)], events))
    full = amap(p).total()
    # 2 s at the 75 ms SLO TPOT = 26 iterations > D=20: the estimate ran ahead of the
    # requests, which are still in flight (no done): each keeps its full size (P + D) / M
    d = p.decide(snap(3000, [pod("p0", 3000)]))["m"]
    assert d.inputs["anom"]["overdue"] == 10 and len(p._models["m"].active) == 10
    assert amap(p).at(0) == pytest.approx(10 * (100 + 20) / 10_000) and amap(p).total() >= full
    p.decide(snap(3000, [pod("p0", 3000)], [ev("done", "r0", "p0", 3000)]))
    assert len(p._models["m"].active) == 9 and amap(p).at(0) == pytest.approx(9 * 120 / 10_000)
    # an engine reporting nothing in flight is a done for everything that started before
    d = p.decide(snap(5000, [pod("p0", 5000, running=0.0)]))["m"]
    assert p._models["m"].active == {} and amap(p).total() == 0.0
    assert d.inputs["anom"]["engine_idle_done"] == 9
    # the paper's mode keeps extending them instead
    q = make(out_len_is_upper_bound=False)
    q.decide(snap(1000, [pod("p0", 1000)], events))
    q.decide(snap(3000, [pod("p0", 3000)]))
    assert len(amap(q).requests) == 10 and amap(q).requests[("r0", "p0")].extensions >= 1


def test_event_after_the_scrape_is_not_out_of_order() -> None:
    p = make()
    p.decide(snap(1000, [pod("p0", 1000)]))
    # the event was written 5 ms after the pod was scraped (before the stream read)
    d = p.decide(snap(3000, [pod("p0", 2990)], load("r", "p0", 2995, 100, 50)))
    assert p.anomalies["m"]["out_of_order"] == 0 and "anom" not in d["m"].inputs
    assert amap(p).requests[("r", "p0")].start == amap(p).it
    # the next scrape walks on from the event time
    p.decide(snap(5000, [pod("p0", 4995)]))
    assert p.anomalies["m"]["out_of_order"] == 0 and amap(p).it == 53  # 26 + 27 (carry)


def test_first_tokens_before_the_first_awake_snapshot_are_retained():
    p = make()
    p.decide(snap(1000, [pod("p0", 1000)], load("r", "p1", 1000, 500, 100) + load("s", "p1", 1000, 300, 100)))
    p.decide(snap(1500, [pod("p0", 1500)], [ev("done", "s", "p1", 1400)]))
    d = p.decide(snap(3000, [pod("p0", 3000), pod("p1", 3000)]))["m"]
    assert d.inputs["tier2"]["p1"]["reqs"] == 1
    assert d.inputs["tier2"]["p1"]["pending"] == 0
    assert d.inputs["anom"]["ft_buffered_applied"] == 1


def test_waiting_does_not_advance_decode_and_blocks_underload_shrink():
    p = make()
    p.decide(snap(1000, [pod("p0", 1000)], [ev("arr", "q", "p0", 1000, 2000, 50)]))
    d = p.decide(snap(5000, [pod("p0", 5000)]))["m"]
    assert d.inputs["tier2"]["p0"]["reqs"] == 0
    assert d.inputs["tier2"]["p0"]["U0"] == 0
    assert d.inputs["down"] == "pending_prefill"
    d = p.decide(snap(6000, [pod("p0", 6000)], [ev("ft", "q", "p0", 6000)]))["m"]
    assert d.inputs["tier2"]["p0"]["reqs"] == 1
    assert d.inputs["tier2"]["p0"]["U0"] == pytest.approx(0.2)


def test_wake_keeps_new_attempts_and_rejects_old_attempt_events():
    p = make()
    p.decide(snap(1000, [pod("p0", 1000), pod("p1", 1000)], load("old", "p0", 1000, 500, 100)))
    p.decide(snap(3000, [pod("p1", 3000)], awake=1))
    # A wake races the next state snapshot: fresh arr/ft are earlier than that snapshot.
    p.decide(snap(8000, [pod("p1", 8000)], load("new", "p0", 7500, 200, 100), awake=1))
    d = p.decide(snap(9000, [pod("p0", 9000), pod("p1", 9000)],
                      [ev("ft", "old", "p0", 8500), ev("done", "old", "p0", 8800)]))["m"]
    assert d.inputs["tier2"]["p0"]["reqs"] == 1
    assert set(amap(p).requests) == {("new", "p0")}


def test_a_request_without_done_expires_and_holds_tier2_for_one_window() -> None:
    """v1 parity: a lost done must not keep a request in the map; Tier-2 holds meanwhile."""
    p = make(done_margin_s=10)  # window_s 600
    p.decide(snap(1000, [pod("p0", 1000)], load("r", "p0", 1000, 100, 10)))
    d = p.decide(snap(12_000, [pod("p0", 12_000)]))["m"]  # 11 s < 10 x 0.2 + 10 = 12 s
    assert len(amap(p).requests) == 1 and "done_lost" not in d.inputs.get("gaps", [])
    d = p.decide(snap(14_000, [pod("p0", 14_000)]))["m"]
    assert amap(p).requests == {} and p.anomalies["m"]["expired_routed"] == 1
    assert d.inputs["frozen"] is True and "done_lost" in d.inputs["t2_hold"]
    d = p.decide(snap(614_500, [pod("p0", 614_500)]))["m"]  # one window (600 s) later
    assert "done_lost" not in d.inputs.get("gaps", [])


# ------------------------------------------------------------------- Tier-2 (faithful)


def test_overload_threshold_and_one_instance_per_overloaded_pod() -> None:
    p = make()
    pods = [pod("p0", 1000, M=1000), pod("p1", 1000, M=1000)]
    # p0: U_i = (861+i)/1000 > 0.95 for i = 90..99 -> exactly 10 % at every start offset -> not overloaded
    d = p.decide(snap(1000, pods, load("a", "p0", 1000, 861, 100)))
    assert d["m"].reason == "hold" and d["m"].desired == 2 and d["m"].inputs["n_t2"] == 2  # starts at awake
    assert d["m"].inputs["tier2"]["p0"]["overload_frac"] == pytest.approx(0.10)
    # p1: 11 % above 0.95 -> overloaded -> one more instance on the current count
    d = p.decide(snap(1000, pods, load("b", "p1", 1000, 862, 100)))
    assert d["m"].reason == "tier2_overload" and d["m"].desired == 3
    # still overloaded next tick: already credited, the count is held (awake lags)
    d = p.decide(snap(1000, pods))
    assert d["m"].reason == "hold" and d["m"].desired == 3
    # the episode ends, then p1 overloads again -> one more
    p.decide(snap(1000, pods, [ev("done", "b", "p1", 1000)]))
    d = p.decide(snap(1000, pods, load("c", "p1", 1000, 990, 200)))
    assert d["m"].reason == "tier2_overload" and d["m"].desired == 4
    # two fresh overloaded pods in one tick -> +2
    q = make()
    d = q.decide(snap(1000, pods, load("a", "p0", 1000, 990, 200) + load("b", "p1", 1000, 990, 200)))
    assert d["m"].desired == 4


def test_overload_ahead_of_now_before_the_next_tick_is_caught() -> None:
    """v1 D55: the map's load crosses kv_high only past U[0:l) but before the next tick
    (start offsets up to H = ceil(tick / TPOT)): overloaded now, not one tick late."""
    pods = [pod("p0", 1000, M=1000)]
    # U_i = (840 + i)/1000 > 0.95 from i = 111: U[0:100) is clean
    d = make().decide(snap(1000, pods, load("a", "p0", 1000, 840, 300), tick_s=2.0))["m"]
    row = d.inputs["tier2"]["p0"]
    assert row["overload_frac"] == 0.0 and row["H"] == 27 and row["t_star"] == 22
    assert (d.reason, d.desired) == ("tier2_overload", 2)
    # a shorter tick (H = 7) does not reach it
    d = make().decide(snap(1000, pods, load("a", "p0", 1000, 840, 300), tick_s=0.5))["m"]
    assert d.inputs["tier2"]["p0"]["t_star"] is None and d.desired == 1


def test_scale_down_formula_once_per_window() -> None:
    p = make(done_margin_s=3600)  # the requests outlive the 600 s jump: not a lost done

    def pods_at(t):
        return [pod(f"p{i}", t, M=1000, kv=kv) for i, kv in enumerate((0.25, 0.2, 0.2, 0.0))]

    events = load("a", "p0", 1000, 250) + load("b", "p1", 1000, 200) + load("c", "p2", 1000, 200)
    # all max U < 0.3; keep ceil((0.25 + 0.2 + 0.2 + 0) / 0.3) = ceil(2.17) = 3 of 4
    d = p.decide(snap(1000, pods_at(1000), events))
    assert d["m"].reason == "tier2_underload" and d["m"].desired == 3
    d = p.decide(snap(3000, pods_at(3000)))  # same window: no second change
    assert d["m"].reason == "hold" and d["m"].desired == 3 and d["m"].inputs["down"] == "done_this_window"
    d = p.decide(snap(600_000, pods_at(600_000)))  # next wall-clock window (Tier-1 inactive)
    assert d["m"].reason == "hold" and d["m"].inputs["down"] == "no_gain"  # 3 is not below N_T2 = 3


def test_scale_down_blockers_and_floor() -> None:
    def pods3(*kv):
        return [pod(f"p{i}", 1000, M=1000, kv=k) for i, k in enumerate(kv)]

    ev3 = load("a", "p0", 1000, 250) + load("b", "p1", 1000, 200) + load("c", "p2", 1000, 200)
    d = make().decide(snap(1000, pods3(0.25, 0.2, 0.2), ev3))  # keep 3 of 3
    assert d["m"].reason == "hold" and d["m"].inputs["down"] == "no_gain"
    d = make().decide(snap(1000, pods3(0.3, 0, 0), load("a", "p0", 1000, 300)))  # 0.30 is not below T_f
    assert d["m"].inputs["down"] == "above_t_f"
    d = make().decide(snap(1000, [], awake=2))
    assert d["m"].inputs["down"] == "no_pods" and d["m"].desired == 2
    d = make().decide(snap(1000, pods3(0, 0, 0)))  # never below one instance
    assert d["m"].reason == "tier2_underload" and d["m"].desired == 1


def test_incomplete_evidence_freezes_tier2_in_both_directions() -> None:
    """v1 D56 / D64: an evidence gap holds Tier-2 (no overload credit, no underload), logged as
    ``frozen``; once the evidence is complete Tier-2 acts again."""
    p = make()
    for t in (1000, 63_000):
        busy = [pod("p0", t, M=1000, running=20.0, kv=0.9, cover=False), pod("p1", t, M=1000, kv=0.0)]
        d = p.decide(snap(t, busy, since=None))["m"]
        assert (d.desired, d.inputs["down"], d.inputs["frozen"]) == (2, "tier2_hold", True)
        assert "tier2_frozen" in d.reason and set(d.inputs["t2_hold"]) == {"event_gap", "no_event_history"}
    # an overload during a gap is not credited either
    d = p.decide(snap(64_000, [pod("p0", 64_000, M=1000, kv=0.99), pod("p1", 64_000, M=1000)],
                      load("a", "p0", 64_000, 990, 200), unscraped=("p9",), awake=3))["m"]
    assert d.inputs["frozen"] is True and d.desired == 2 and d.inputs["t2_hold"] == ["unscraped"]
    # complete: the pending overload is acted on
    d = p.decide(snap(65_000, [pod("p0", 65_000, M=1000, kv=0.99), pod("p1", 65_000, M=1000)]))["m"]
    assert d.inputs["frozen"] is False and (d.reason, d.desired) == ("tier2_overload", 3)
    assert p.counters()["m"]["t2_hold_ticks"] == 3 and p.counters()["m"]["ticks"] == 4


def test_kv_disagreement_is_logged_not_a_gate() -> None:
    d = make().decide(snap(1000, [pod("p0", 1000, M=1000, kv=0.9), pod("p1", 1000, M=1000, kv=None)]))["m"]
    assert d.inputs["kv_disagree"] == {"p0": [0.0, 0.9], "p1": [0.0, None]}
    assert (d.reason, d.desired, d.inputs["frozen"]) == ("tier2_underload", 1, False)


# ------------------------------------------------------------ Tier-1 + composition


def test_desired_is_max_of_the_two_tiers_and_a_window_start_resets_tier2() -> None:
    p = make(tier1="oracle", done_margin_s=3600)  # the request outlives the 600 s jump: not a lost done
    pods = [pod("p0", 1000, M=1000)]
    d = p.decide(snap(1000, pods, replay=REPLAY))["m"]
    assert d.reason == "tier1_window" and (d.desired, d.inputs["n_t1"], d.inputs["n_t2"]) == (2, 2, 1)
    assert d.inputs["tier1"] == {"window": 0, "mode": "oracle", "P_hat": 120000.0, "D_hat": 60000.0, "W": 600.0, "N": 2}
    d = p.decide(snap(3000, pods, replay=REPLAY))["m"]  # same window, empty map: Tier-2 cannot isolate below N_T1
    assert d.desired == 2
    # an overload adds on max(N_T1, N_T2)
    d = p.decide(snap(5000, [pod("p0", 5000, M=1000, kv=0.99)], load("a", "p0", 5000, 990, 200), replay=REPLAY))["m"]
    assert (d.reason, d.desired, d.inputs["n_t2"]) == ("tier2_overload", 3, 3)
    # window 1: N_T1 = 1 and N_T2 back to the minimum; the still-overloaded pod counts again on the new base
    d = p.decide(snap(600_000, [pod("p0", 600_000, M=1000, kv=0.99)], replay=REPLAY))["m"]
    assert d.reason == "tier1_window+tier2_overload" and (d.inputs["n_t1"], d.inputs["n_t2"], d.desired) == (1, 2, 2)


def test_tier1_is_never_held_by_an_evidence_gap() -> None:
    """v1 D56: Tier-1 reads no Tier-2 signal; a window whose forecast is lower scales down
    even while Tier-2 is frozen."""
    p = make(tier1="oracle")
    p.decide(snap(1000, [pod("p0", 1000)], replay=REPLAY))
    d = p.decide(snap(600_000, [pod("p0", 600_000, cover=False), pod("p1", 600_000)], awake=2, since=None,
                      replay=REPLAY))["m"]
    assert d.inputs["frozen"] is True and d.inputs["n_t1"] == 1
    assert d.desired == 1 and d.inputs["n_t2"] == 1  # the window start resets N_T2; frozen does not hold Tier-1


def test_tier1_after_the_trace_keeps_its_last_value_before_t0_none() -> None:
    p = make(tier1="oracle")
    pods = [pod("p0", 1000)]
    p.decide(snap(1000, pods, replay=REPLAY))
    d = p.decide(snap(1_200_000, pods, replay=REPLAY))["m"]  # partial last window (300 s)
    assert d.inputs["n_t1"] == 2 and d.inputs["tier1"]["W"] == 300.0
    d = p.decide(snap(1_800_000, pods, replay=REPLAY))["m"]
    assert d.inputs["tier1"] == {"inactive": "tier1_after_trace"} and d.inputs["n_t1"] == 2
    late = ReplayInfo(10_000, REPLAY.trace_path)
    d = make().decide(snap(5000, pods, replay=late))["m"]
    assert d.inputs["tier1"] == {"inactive": "tier1_before_t0"} and d.inputs["n_t1"] is None
    other = ReplayInfo(0, "/mnt/x/traces/case/seed2/trace.json")  # trace_match_parts 2
    assert make().decide(snap(1000, pods, replay=other))["m"].inputs["tier1"] == {"inactive": "tier1_trace_mismatch"}


def test_oracle_noisy_is_seeded_and_deterministic() -> None:
    def run(seed: int):
        p = PreServePolicy(cfg({"mu": {"m": dict(MU)}, "max_output_len": 4096, "tier1": "oracle_noisy",
                                "load_error_mode": "none", "window_s": 600.0}, seed=seed), oracle=ORACLE)
        return p.decide(snap(1000, [pod("p0", 1000)], replay=REPLAY))["m"].inputs["tier1"]

    a = run(4)
    assert a == run(4) and a["mode"] == "oracle_noisy"
    assert a["P_hat"] == round(120000 * t1.noise_factor(4, "m", 0, "P", t1.DEFAULT_NOISE_SIGMA), 1)
    assert run(5)["P_hat"] != a["P_hat"]


def test_model_without_mu_is_held() -> None:
    p = PreServePolicy(cfg({"mu": {"m": dict(MU)}, "tier1": "oracle", "load_error_mode": "none",
                            "window_s": 600.0}, models=()), oracle=ORACLE)
    ms = ModelSnapshot(model="z", awake=2, min_replicas=1, max_replicas=4, gpus_per_replica=1,
                       ttft_slo_ms=500.0, tpot_slo_ms=75.0, max_num_seqs=None, pods=(), events=())
    d = p.decide(ClusterSnapshot(now_ms=1, tick_s=2.0, models={"z": ms}))
    assert d["z"].reason == "no_mu" and d["z"].desired == 2 and d["z"].inputs["frozen"] is False


def test_replay_seed_of_the_marker_wins_over_trace_seed_param() -> None:
    trace = str(FIX / "preserve_trace_segments.json")
    params = {"mu": {"m": dict(MU)}, "trace_path": trace, "trace_seed": 0, "window_s": 60.0, "tier1": "oracle",
              "load_error_mode": "none"}
    p = PreServePolicy(cfg(params))
    seed0 = to.load_oracle(trace, window_s=60.0, seed=0)
    seed7 = to.load_oracle(trace, window_s=60.0, seed=7)
    assert seed0.totals != seed7.totals and p.oracle.totals == seed0.totals
    p.decide(snap(2000, [pod("p0", 2000)], replay=ReplayInfo(t0_ms=0, trace_path=trace, seed=7)))
    assert p.oracle.totals == seed7.totals and p._oracle_seed == 7


# ------------------------------------------------------------ mLSTM Tier-1


class _Mean:
    """A stand-in forecaster: the mean of its input (records every call)."""

    seq_len = 3

    def __init__(self) -> None:
        self.calls: list[list[float]] = []

    def predict(self, history):
        self.calls.append(list(history))
        return sum(history) / len(history)


def mlstm(history=None, **params):
    preds = {"prompt": _Mean(), "response": _Mean()}
    p = {"mu": {"m": dict(MU)}, "tier1": "mlstm", "seq_len": 3, "load_error_mode": "none", "window_s": 30.0}
    p.update(params)
    oracle = to.build_oracle([to.TraceRequest(1.0, "m", 1, 1), to.TraceRequest(89.0, "m", 1, 1)],
                             path=ORACLE.path, fmt=to.FORMAT_REQUESTS, window_s=30.0, duration_s=89.0)
    pol = PreServePolicy(cfg(p), oracle=oracle, predictors=preds,
                         history={"m": history or [[3000.0, 1500.0]] * 3})
    return pol, preds


def test_mlstm_history_holds_the_past_then_observed_windows_never_a_prediction() -> None:
    p, preds = mlstm(history=[[900.0, 300.0], [3000.0, 1500.0], [6000.0, 3000.0]])
    pods = [pod("p0", 1000)]
    d = p.decide(snap(1000, pods, replay=REPLAY))["m"]
    # window 0: mean of the pre-t0 history (3300, 1600) -> N = ceil(max(3300/3000, 1600/3000, 4900/4500)) = 2
    assert d.inputs["tier1"]["P_hat"] == 3300.0 and d.inputs["tier1"]["N"] == 2 and d.inputs["n_t1"] == 2
    # arrivals of window 0: new demand only (a reissue is not), with their max_tokens
    evs = [ev("arr", "a", "p0", 5000, 1200, 600), ev("arr", "b", "p0", 6000, 1800, 300),
           ev("arr", "a", "p0", 7000, 1300, 500, reissue="continued")]
    p.decide(snap(8000, pods, evs, replay=REPLAY))
    d = p.decide(snap(30_500, pods, replay=REPLAY))["m"]
    assert preds["prompt"].calls[-1] == [3000.0, 6000.0, 3000.0]  # observed P of window 0 = 3000
    assert preds["response"].calls[-1] == [1500.0, 3000.0, 900.0]
    assert d.inputs["tier1"]["last_closed"] == {"window": 0, "P": 3000.0, "D": 900.0, "requests": 2, "flags": []}
    # a window with no arrival closes as zero, never as the forecast
    d = p.decide(snap(60_500, pods, replay=REPLAY))["m"]
    assert preds["prompt"].calls[-1] == [6000.0, 3000.0, 0.0]


def test_mlstm_flags_a_window_observed_through_an_event_gap_and_restarts_per_replay() -> None:
    p, preds = mlstm()
    p.decide(snap(1000, [pod("p0", 1000)], replay=REPLAY))
    p.decide(snap(2000, [pod("p0", 2000, cover=False)], replay=REPLAY))
    d = p.decide(snap(30_500, [pod("p0", 30_500)], replay=REPLAY))["m"]
    assert d.inputs["tier1"]["last_closed"]["flags"] == ["event_gap"]
    second = ReplayInfo(t0_ms=5_000_000, trace_path=REPLAY.trace_path)
    d = p.decide(snap(5_000_100, [pod("p0", 5_000_100)], replay=second))["m"]
    assert d.inputs["tier1"]["window"] == 0 and preds["prompt"].calls[-1] == [3000.0] * 3  # fresh history


def test_mlstm_refuses_a_short_history() -> None:
    with pytest.raises(ValueError, match="history"):
        mlstm(history=[[1.0, 1.0]] * 2)


def test_shipped_weights_load_with_their_sha256_and_predict(tmp_path) -> None:
    np = pytest.importorskip("numpy")
    from tre_baselines.policies import preserve_history as ph
    from tre_baselines.tools import freeze_params as fp

    sha = fp.weights_sha256()
    trace = tmp_path / "Real_code" / "seed1" / "trace.json"
    trace.parent.mkdir(parents=True)
    trace.write_text('[{"timestamp": 1.0, "model_name": "m", "prompt_length": 100, "max_output_tokens": 50}]')
    hist = {"format": ph.FORMAT, "trace": "Real_code", "seed": 1, "window_s": 30,
            "models": {"m": [[30000.0 + 100 * i, 15000.0] for i in range(25)]}}
    (trace.parent / ph.DEFAULT_NAME).write_text(__import__("json").dumps(hist))
    params = {"mu": {"m": dict(MU)}, "trace_path": str(trace), "load_error_mode": "none",
              "predictors": {k: {"path": f"{fp.WEIGHTS}/{k}-paper.npz", "sha256": v} for k, v in sha.items()}}
    p = build_policy("preserve", cfg(params))
    assert p.predictor_set == "code" and p.history_info["trace"] == "Real_code"
    est = p._predictors["prompt"].predict([r[0] for r in p._history["m"]])
    assert np.isfinite(est) and 0 < est < 1e6
    # golden values: v1 tre_preserve.predictor (dev ceb88b4) on these weights, numpy 1.26.4
    from tre_baselines.policies import preserve_predictor as pp

    hp = [30000.0 + 100 * i + (3000 if i % 5 == 0 else 0) for i in range(25)]
    hd = [15000.0 + 50 * i for i in range(25)]
    gold = {"code": (32385.288988438784, 16027.245117373537), "conv": (32027.71266924556, 15797.959411714452),
            "pooled": (32344.189325879677, 15803.711096508441)}
    for name, (gp, gd) in gold.items():
        prs = pp.load_predictors(str(Path(fp.TRE_ROOT) / "baselines" / fp.WEIGHTS / f"{name}-paper.npz"), sha[name])
        assert prs["prompt"].predict(hp) == pytest.approx(gp, rel=1e-6)
        assert prs["response"].predict(hd) == pytest.approx(gd, rel=1e-6)
    bad = {**params, "predictors": {**params["predictors"], "code": {"path": f"{fp.WEIGHTS}/code-paper.npz",
                                                                     "sha256": "0" * 64}}}
    with pytest.raises(ValueError, match="sha256"):
        build_policy("preserve", cfg(bad))
    (trace.parent / ph.DEFAULT_NAME).unlink()
    with pytest.raises(ValueError, match="history"):
        build_policy("preserve", cfg(params))


# ------------------------------------------------------------ load errors


def test_load_errors_are_a_function_of_the_client_id_and_enter_the_map() -> None:
    def run(order):
        p = make(load_error_mode="paper_tab2", out_len_is_upper_bound=False)
        evs = [e for r in order for e in (ev("arr", r, "p0", 1000, 100, 400), ev("ft", r, "p0", 1000))]
        d = p.decide(snap(1000, [pod("p0", 1000)], evs))["m"]
        return {k[0]: req.D_pred for k, req in amap(p).requests.items()}, d.inputs["load_err"]

    ids = [f"r{i}" for i in range(400)]
    fwd, info = run(ids)
    assert run(list(reversed(ids)))[0] == fwd  # not the arrival order
    assert any(v != 400 for v in fwd.values()) and all(v >= 1 for v in fwd.values())
    assert info["cum"]["n"] == 400 and 40 < info["cum"]["mae"] < 120  # Tab. 2 MAE 78.25
    assert sorted(r[0] for r in info["new"]) == sorted(f"c-{r}" for r in ids)
    # a reissue repeats its original's error and is not counted again
    p = make(load_error_mode="paper_tab2", out_len_is_upper_bound=False)
    d = p.decide(snap(1000, [pod("p0", 1000), pod("p1", 1000)],
                      [ev("arr", "x", "p0", 1000, 100, 400), ev("ft", "x", "p0", 1000),
                       ev("arr", "x", "p1", 1000, 100, 400, "retried"), ev("ft", "x", "p1", 1000, reissue="retried")]))
    assert not amap(p, "p0").requests
    assert amap(p, "p1").requests[("x", "p1")].D_pred == p.errors.predict("m", "c-x", 400, 100, p.max_model_len)
    assert d["m"].inputs["load_err"]["cum"]["n"] == 1


def test_history_file_of_the_trace_generator_and_its_checks(tmp_path) -> None:
    """``pre_t0.json`` (tre-pre-t0-v1) next to a synthetic trace is the default history; it must be
    contiguous windows ending at t0 and name the replayed trace and seed."""
    import json

    from tre_baselines.policies import preserve_history as ph

    d = tmp_path / "Tight_pool" / "seed2"
    d.mkdir(parents=True)
    trace = d / "traces_tre.effective.json"
    trace.write_text("[]")
    assert ph.default_path(str(trace)) == str(d / ph.DEFAULT_NAME)  # Real_*: no pre_t0.json
    wins = [{"start_s": -90.0 + 30 * i, "end_s": -60.0 + 30 * i, "requests": 1, "prompt_tokens": 100 * (i + 1),
             "output_tokens": 10 * (i + 1)} for i in range(3)]
    doc = {"format": ph.PRE_T0_FORMAT, "trace": "Tight_pool", "seed": 2, "t0_s": 0.0, "window_s": 30.0,
           "windows": {"m": wins}, "requests": []}
    (d / ph.PRE_T0_NAME).write_text(json.dumps(doc))
    assert ph.default_path(str(trace)) == str(d / ph.PRE_T0_NAME)
    kw = dict(window_s=30.0, models=["m"], seq_len=3, trace="Tight_pool", seed=ph.seed_of(str(trace)))
    hist, info = ph.load(ph.default_path(str(trace)), **kw)
    assert hist == {"m": [(100.0, 10.0), (200.0, 20.0), (300.0, 30.0)]} and info["format"] == ph.PRE_T0_FORMAT
    for bad, match in (({"seed": 1}, "seed"), ({"seed": None}, "seed"), ({"trace": "Simul_spike"}, "trace"),
                       ({"trace": None}, "trace"), ({"t0_s": None}, "t0_s"), ({"window_s": 60.0}, "window_s"),
                       ({"windows": {"m": wins[:1] + wins[2:]}}, "contiguous"),
                       ({"windows": {"m": wins[1:]}}, "seq_len")):
        with pytest.raises(ph.HistoryError, match=match):
            ph.parse({**doc, **bad}, **kw)

def test_late_replay_marker_backfills_arrivals_already_consumed():
    p, _ = mlstm()
    p.decide(snap(5000, [pod("p0", 5000)], [ev("arr", "r", "p0", 1000, 1200, 600)]))
    marker = ReplayInfo(t0_ms=1000, trace_path=REPLAY.trace_path, seed=7)
    p.decide(snap(8000, [pod("p0", 8000)], replay=marker))
    d = p.decide(snap(32000, [pod("p0", 32000)], replay=marker))["m"]
    closed = d.inputs["tier1"]["last_closed"]
    assert closed["requests"] == 1 and closed["P"] == 1200 and closed["D"] == 600


def test_retry_uses_distinct_gateway_attempt_ids_and_late_parent_ft_is_ignored():
    p = make()
    p.decide(snap(1000, [pod("p0", 1000), pod("p1", 1000)],
                  [ev("arr", "attempt-a", "p0", 1000, 500, 50, cid="logical")]))
    d = p.decide(snap(2000, [pod("p0", 2000), pod("p1", 2000)],
        [ev("arr", "attempt-b", "p1", 1500, 500, 50, "retried", cid="logical"),
         ev("ft", "attempt-b", "p1", 1800, reissue="retried", cid="logical"),
         ev("ft", "attempt-a", "p0", 1900, cid="logical"),
         ev("done", "attempt-a", "p0", 1950, cid="logical")]))["m"]
    assert d.inputs["tier2"]["p0"]["reqs"] == 0
    assert d.inputs["tier2"]["p1"]["reqs"] == 1


def test_lost_bootstrap_arrivals_keep_refusing_the_same_marker():
    p, _ = mlstm(req_ttl_s=1)
    p.decide(snap(5000, [pod("p0", 5000)], [ev("arr", "r1", "p0", 1000, 1200, 600),
                                          ev("arr", "r2", "p0", 3000, 1200, 600)]))
    marker = ReplayInfo(t0_ms=1000, trace_path=REPLAY.trace_path, seed=7)
    for ts in (8000, 9000):
        with pytest.raises(ValueError, match="arrival history was lost"):
            p.decide(snap(ts, [pod("p0", ts)], replay=marker))


def test_wake_buffer_preserves_decode_progress_when_prefill_finishes_out_of_order():
    events = [ev("arr", "A", "p0", 1000, 1000, 100),
              ev("arr", "B", "p0", 1100, 1000, 100),
              ev("ft", "B", "p0", 1200), ev("ft", "A", "p0", 2200)]
    visible = make()
    direct = visible.decide(snap(2300, [pod("p0", 2300, running=2)], events))["m"]
    waking = make()
    waking.decide(snap(2300, [], events, awake=1))
    buffered = waking.decide(snap(2300, [pod("p0", 2300, running=2)]))["m"]
    # The same arrivals/first tokens must produce the same load when the pod becomes visible.
    for field in ("U0", "maxU"):
        assert buffered.inputs["tier2"]["p0"][field] == pytest.approx(direct.inputs["tier2"]["p0"][field])
