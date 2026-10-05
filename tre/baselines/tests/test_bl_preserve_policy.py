"""PreServe policy: event feed, iteration clock, Tier-2 thresholds, Tier-1 composition."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import pytest

from tre_baselines import trace_oracle as to
from tre_baselines.config import Config, ModelLimits
from tre_baselines.policies import POLICIES, Policy, build_policy
from tre_baselines.policies import preserve_tier1 as t1
from tre_baselines.policies.preserve import PreServePolicy
from tre_baselines.snapshot import ClusterSnapshot, ModelSnapshot, PodSnapshot, ReplayInfo, RequestEvent

FIX = Path(__file__).resolve().parent / "fixtures"
MU = {"p": 100.0, "d": 100.0, "t": 150.0}
# window 0: P=120000 D=60000 -> N=2; window 1: P=60000 D=30000 -> N=1;
# window 2 (partial, 300 s): P=60000 -> N=2. Trace spans 3 windows of 600 s.
ORACLE = to.build_oracle(
    [to.TraceRequest(10.0, "m", 60_000, 30_000), to.TraceRequest(20.0, "m", 60_000, 30_000),
     to.TraceRequest(700.0, "m", 60_000, 30_000), to.TraceRequest(1500.0, "m", 60_000, 0)],
    path="repo/traces/case/trace.json", fmt=to.FORMAT_REQUESTS, window_s=600.0, duration_s=1500.0,
)
REPLAY = ReplayInfo(t0_ms=0, trace_path="/mnt/x/traces/case/trace.json")


def cfg(params: dict, models=("m",), seed: int = 0) -> Config:
    lim = {m: ModelLimits(name=m, min_replicas=1, max_replicas=8, gpus_per_replica=1,
                          ttft_slo_ms=500.0, tpot_slo_ms=75.0, max_num_seqs=256) for m in models}
    return Config(sm_url="http://sm.test", redis_url="redis://r.test", policy="preserve",
                  policy_params=params, models=lim, seed=seed)


def make(**params) -> PreServePolicy:
    p = {"mu": {"m": dict(MU)}, "max_output_len": 4096}
    p.update(params)
    return PreServePolicy(cfg(p), oracle=ORACLE)


def pod(name: str, t: int, itl=(0.0, 0.0), M: Optional[int] = 10_000, *, running=1.0,
        kv: Optional[float] = None, cover: bool = True) -> PodSnapshot:
    """Default: a busy engine (running 1, so it never reads as idle), events cover it."""
    blocks, bs = (None, None) if M is None else (M // 8, 8)
    return PodSnapshot(pod=name, model="m", node="n0", gpu_ids=(0,), running=running, waiting=0.0,
                       kv_usage=kv, counters={"itl_sum": itl[0], "itl_count": itl[1]},
                       num_gpu_blocks=blocks, block_size=bs, scraped_at_ms=t, events_cover=cover)


def ev(kind: str, req: str, p: Optional[str], ts: int, inn=None, mx=None, reissue="none") -> RequestEvent:
    return RequestEvent(kind=kind, model="m", pod=p, req_id=req, ts_ms=ts, in_tokens=inn,
                        in_src="header", max_tokens=mx, out_tokens=None, status=None, reissue=reissue)


def snap(now: int, pods, events=(), awake=None, unscraped=(), replay=None, since=0) -> ClusterSnapshot:
    ms = ModelSnapshot(model="m", awake=len(pods) if awake is None else awake, min_replicas=1,
                       max_replicas=8, gpus_per_replica=1, ttft_slo_ms=500.0, tpot_slo_ms=75.0,
                       max_num_seqs=256, pods=tuple(pods), events=tuple(events), unscraped=tuple(unscraped),
                       events_since_ms=since)
    return ClusterSnapshot(now_ms=now, tick_s=2.0, models={"m": ms}, replay=replay)


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
    policy = build_policy("preserve", cfg({"mu": mu, "trace_path": trace}, models))
    assert isinstance(policy, Policy) and policy.oracle.n_requests == 5
    assert policy.map_length("dsqwen-7b") == 84  # trace max max_tokens 70 x 1.2
    assert policy.map_length("unknown") == 4916  # 4096 x 1.2
    per_model = build_policy("preserve", cfg({"mu": mu, "trace_path": trace,
                                              "max_output_len": {"dsqwen-7b": 1000}}, models))
    assert per_model.map_length("dsqwen-7b") == 1200 and per_model.map_length("dsllama-8b") == 600
    with pytest.raises(ValueError, match="managed models"):
        build_policy("preserve", cfg({"mu": {"dsqwen-7b": dict(MU)}, "trace_path": trace}, models))
    with pytest.raises(ValueError):
        build_policy("preserve", cfg({"trace_path": trace}, models))
    with pytest.raises(ValueError):
        build_policy("preserve", cfg({"mu": {**mu, "dsqwen-7b": {"p": 1, "d": 1, "t": 0}},
                                      "trace_path": trace}, models))
    with pytest.raises(ValueError, match="trace_path"):
        build_policy("preserve", cfg({"mu": mu}, models))
    with pytest.raises(ValueError):
        build_policy("preserve", cfg({"mu": mu, "trace_path": trace, "tier1": "mlstm"}, models))
    with pytest.raises(ValueError):
        build_policy("preserve", cfg({"mu": mu, "trace_path": trace, "hold_mode": "x"}, models))


# ------------------------------------------------------------- iteration clock


def test_iterations_follow_measured_tpot_and_events_land_at_their_time() -> None:
    p = make()
    p.decide(snap(1000, [pod("p0", 1000)], [ev("arr", "r1", "p0", 900, 1000, 512), ev("ft", "r1", "p0", 1000)]))
    m = amap(p)
    assert m.it == 0 and m.at(0) == pytest.approx(0.1) and m.at(511) == pytest.approx(0.1511)
    # itl counters: 64 tokens in 1.0 s -> TPOT 1/64 s -> 64 iterations per second
    d = p.decide(snap(3000, [pod("p0", 3000, (1.0, 64.0))],
                      [ev("arr", "r2", "p0", 1500, 0, 100), ev("ft", "r2", "p0", 2000)]))
    assert d["m"].inputs["tier2"]["p0"]["tpot"] == "itl"
    assert m.it == 128
    assert m.requests[("r2", "p0")].start == 64  # placed at its ft time, not at the tick
    assert m.at(0) == pytest.approx((1128 + 64) / 10_000)
    assert m.at(35) == pytest.approx((1163 + 99) / 10_000)
    assert m.at(36) == pytest.approx(1164 / 10_000)


def test_tpot_falls_back_to_the_slo_and_carries_fractions() -> None:
    p = make()
    p.decide(snap(1000, [pod("p0", 1000, (1.0, 64.0))]))
    d = p.decide(snap(3000, [pod("p0", 3000, (1.0, 64.0))]))  # no new ITL samples
    assert d["m"].inputs["tier2"]["p0"]["tpot"] == "slo"
    assert amap(p).it == 26  # 2 s / 75 ms = 26.67
    p.decide(snap(5000, [pod("p0", 5000, (1.0, 64.0))]))
    assert amap(p).it == 53  # 26.67 + 0.67 carried
    # a counter reset (engine restart) is not a sample either
    d = p.decide(snap(7000, [pod("p0", 7000, (0.1, 5.0))]))
    assert d["m"].inputs["tier2"]["p0"]["tpot"] == "slo"
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
    assert amap(p, "p0").at(0) == pytest.approx(0.1)
    assert amap(p, "p1").at(0) == pytest.approx(0.105) and amap(p, "p1").at(49) == pytest.approx(0.1099)
    p.decide(snap(1000, pods, [ev("done", "r", "p0", 1000)]))  # the original is released
    assert all(u == 0.0 for u in amap(p, "p0").window(200))
    assert amap(p, "p1").at(0) == pytest.approx(0.105)
    p.decide(snap(1000, pods, [ev("done", "r", "p1", 1000, reissue="continued")]))
    assert all(u == 0.0 for u in amap(p, "p1").window(200))
    assert sum(p.anomalies["m"].values()) == 0


def test_out_of_order_unknown_and_missing_events_do_not_crash() -> None:
    p = make()
    pods = [pod("p0", 5000)]
    events = [
        ev("done", "zz", "p0", 5000),                                  # unknown request
        ev("arr", "q", "p0", 5000, 10, 5), ev("done", "q", "p0", 5000),
        ev("ft", "q", "p0", 5000),                                     # ft after its done
        ev("ft", "w", "p0", 5000, 100, 10),                            # ft without arr
        ev("arr", "x", "pZ", 5000, 10, 5), ev("ft", "x", "pZ", 5000),  # pod not in snapshot
        ev("arr", "y", "p0", 5000, 10, None), ev("ft", "y", "p0", 5000),  # no max_tokens
        ev("arr", "v", None, 5000, None, 5), ev("ft", "v", "p0", 5000),   # no in_tokens, arr without pod
        ev("ft", "o", "p0", 4000, 1, 1),                               # older than the pod's clock
    ]
    d = p.decide(snap(5000, pods, events))
    anom = p.anomalies["m"]
    assert anom["unknown_req"] == 1 and anom["done_before_ft"] == 1 and anom["ft_after_done"] == 1
    assert anom["ft_without_arr"] == 2 and anom["unknown_pod"] == 1
    assert anom["missing_max_tokens"] == 1 and anom["missing_in_tokens"] == 1
    assert anom["out_of_order"] == 1
    assert d["m"].inputs["anom"]["unknown_req"] == 1
    m = amap(p)
    assert set(m.requests) == {("w", "p0"), ("y", "p0"), ("v", "p0"), ("o", "p0")}
    assert m.requests[("y", "p0")].D == 4096  # max_output_len stands in for max_tokens
    assert m.at(4095) == pytest.approx((10 + 4095) / 10_000) and m.at(4096) == 0.0
    # the pod disappears: its map goes, a later done is just unknown
    p.decide(snap(7000, [], [ev("done", "w", "p0", 6000)], awake=0))
    assert "p0" not in p._models["m"].pods and p._models["m"].active == {}
    assert anom["pod_dropped"] == 1 and anom["unknown_req"] == 2


def test_pod_without_kv_capacity_is_skipped_unless_configured() -> None:
    p = make()
    d = p.decide(snap(1000, [pod("p0", 1000, M=None)]))
    assert d["m"].inputs["skipped"] == {"p0": "no_kv_capacity"}
    q = make(kv_capacity_tokens={"m": 1000})
    q.decide(snap(1000, [pod("p0", 1000, M=None)], load("r", "p0", 1000, 500, 10)))
    assert amap(q).at(0) == pytest.approx(0.5)


def test_bookkeeping_expires_but_a_request_in_the_map_does_not() -> None:
    p = make(req_ttl_s=10)
    p.decide(snap(1000, [pod("p0", 1000)], load("r", "p0", 1000, 100, 10) + [ev("arr", "s", "p0", 1000, 5, 5)]))
    p.decide(snap(20_000, [pod("p0", 20_000)]))
    assert p._models["m"].arr == {} and p.anomalies["m"]["expired_arr"] == 1
    assert list(p._models["m"].active) == [("r", "p0")]  # leaves the map only on done


# ------------------------------------------------------------------- Tier-2


def test_overload_threshold_and_one_instance_per_overloaded_pod() -> None:
    p = make()
    pods = [pod("p0", 1000, M=1000), pod("p1", 1000, M=1000)]
    # p0: U_i = (861+i)/1000 > 0.95 for i = 90..99 -> exactly 10 % -> not overloaded
    d = p.decide(snap(1000, pods, load("a", "p0", 1000, 861, 200)))
    assert d["m"].reason == "hold" and d["m"].desired == 2
    assert d["m"].inputs["tier2"]["p0"]["overload_frac"] == pytest.approx(0.10)
    # p1: 11 % above 0.95 -> overloaded -> one more instance
    d = p.decide(snap(1000, pods, load("b", "p1", 1000, 862, 200)))
    assert d["m"].reason == "tier2_overload" and d["m"].desired == 3
    # still overloaded next tick: already credited, the target is held (awake lags)
    d = p.decide(snap(1000, pods))
    assert d["m"].reason == "hold" and d["m"].desired == 3
    # the episode ends, then p1 overloads again -> one more
    p.decide(snap(1000, pods, [ev("done", "b", "p1", 1000)]))
    d = p.decide(snap(1000, pods, load("c", "p1", 1000, 990, 200)))
    assert d["m"].reason == "tier2_overload" and d["m"].desired == 4
    # two fresh overloaded pods in one tick -> +2 (clamped to the model max 8)
    q = make()
    d = q.decide(snap(1000, pods, load("a", "p0", 1000, 990, 200) + load("b", "p1", 1000, 990, 200)))
    assert d["m"].desired == 4


def test_hold_mode_awake_releases_the_target() -> None:
    p = make(hold_mode="awake")
    pods = [pod("p0", 1000, M=1000)]
    assert p.decide(snap(1000, pods, load("a", "p0", 1000, 990, 200)))["m"].desired == 2
    assert p.decide(snap(1000, pods))["m"].desired == 1


def test_scale_down_formula_once_per_window_and_guards() -> None:
    p = make()
    def pods_at(t):
        return [pod(f"p{i}", t, M=1000, kv=kv) for i, kv in enumerate((0.25, 0.2, 0.2, 0.0))]

    pods = pods_at(1000)
    events = load("a", "p0", 1000, 250) + load("b", "p1", 1000, 200) + load("c", "p2", 1000, 200)
    # all max U < 0.3; keep ceil((0.25 + 0.2 + 0.2 + 0) / 0.3) = ceil(2.17) = 3 of 4
    d = p.decide(snap(1000, pods, events))
    assert d["m"].reason == "tier2_underload" and d["m"].desired == 3
    # same window: no second scale-down
    d = p.decide(snap(3000, pods_at(3000)))
    assert d["m"].reason == "hold" and d["m"].desired == 3 and d["m"].inputs["down"] == "done_this_window"
    # next window (wall-clock window of window_s while Tier-1 is inactive): allowed again
    d = p.decide(snap(600_000, pods_at(600_000)))
    assert d["m"].reason == "tier2_underload" and d["m"].desired == 3


def test_scale_down_blockers() -> None:
    def pods3(*kv, cover=True):
        return [pod(f"p{i}", 1000, M=1000, kv=k, cover=cover) for i, k in enumerate(kv)]

    ev3 = load("a", "p0", 1000, 250) + load("b", "p1", 1000, 200) + load("c", "p2", 1000, 200)
    d = make().decide(snap(1000, pods3(0.25, 0.2, 0.2), ev3))  # keep 3 of 3
    assert d["m"].reason == "hold" and d["m"].inputs["down"] == "no_gain"
    d = make().decide(snap(1000, pods3(0.3, 0, 0), load("a", "p0", 1000, 300)))  # 0.30 is not below T_f
    assert d["m"].inputs["down"] == "above_t_f"
    d = make().decide(snap(1000, pods3(0, 0, 0), unscraped=("p9",), awake=4))
    assert d["m"].inputs["down"] == "incomplete" and d["m"].desired == 4
    d = make().decide(snap(1000, [], awake=2))
    assert d["m"].inputs["down"] == "no_pods" and d["m"].desired == 2
    # never below one instance
    d = make().decide(snap(1000, pods3(0, 0, 0)))
    assert d["m"].reason == "tier2_underload" and d["m"].desired == 1


def test_unknown_is_not_idle_no_scale_down() -> None:
    """Review repro: running 20, KV 0.9, empty event history -> was tier2_underload after
    the 60 s grace. Now the map must be complete and agree with the engine's KV."""
    p = make()
    for t in (1000, 63_000, 600_000):
        busy = [pod("p0", t, M=1000, running=20.0, kv=0.9, cover=False), pod("p1", t, M=1000, kv=0.0)]
        d = p.decide(snap(t, busy, since=None))["m"]
        assert (d.desired, d.inputs["down"]) == (2, "incomplete")
    # complete events but the map (empty) disagrees with the engine's KV by > 0.15
    d = make().decide(snap(1000, [pod("p0", 1000, M=1000, kv=0.9), pod("p1", 1000, M=1000, kv=0.0)]))["m"]
    assert (d.desired, d.inputs["down"]) == (2, "incomplete") and d.inputs["kv_disagree"] == {"p0": [0.0, 0.9]}
    # a KV gauge that is missing is unknown, not agreement
    d = make().decide(snap(1000, [pod("p0", 1000, M=1000, kv=None), pod("p1", 1000, M=1000, kv=0.0)]))["m"]
    assert d.inputs["down"] == "incomplete"
    # the window's one scale-down was not used up: it happens once the evidence is complete
    d = p.decide(snap(600_002, [pod("p0", 600_002, M=1000, kv=0.0), pod("p1", 600_002, M=1000, kv=0.0)]))["m"]
    assert (d.desired, d.reason) == (1, "tier2_underload")


# ------------------------------------------------------------ Tier-1 + composition


def test_tier1_oracle_windows_and_hold() -> None:
    p = make(tier1="oracle")
    pods = [pod("p0", 1000)]
    d = p.decide(snap(1000, pods, replay=REPLAY))
    assert d["m"].reason == "tier1_window" and d["m"].desired == 2
    assert d["m"].inputs["tier1"] == {"window": 0, "mode": "oracle", "P_hat": 120000.0,
                                      "D_hat": 60000.0, "W": 600.0, "N": 2}
    d = p.decide(snap(3000, pods, replay=REPLAY))  # same window; awake still 1
    assert d["m"].reason == "hold" and d["m"].desired == 2
    d = p.decide(snap(600_000, pods, replay=REPLAY))
    assert d["m"].reason == "tier1_window" and d["m"].desired == 1
    d = p.decide(snap(1_200_000, pods, awake=1, replay=REPLAY))  # partial last window (300 s)
    assert d["m"].desired == 2 and d["m"].inputs["tier1"]["W"] == 300.0
    d = p.decide(snap(1_800_000, pods, replay=REPLAY))
    assert d["m"].reason == "hold" and d["m"].inputs["tier1"] == {"inactive": "tier1_after_trace"}


def test_tier1_inactive_reasons() -> None:
    pods = [pod("p0", 1000)]
    assert make().decide(snap(1000, pods))["m"].inputs["tier1"] == {"inactive": "tier1_no_replay"}
    other = ReplayInfo(0, "/mnt/x/traces/other/trace.json")
    assert make().decide(snap(1000, pods, replay=other))["m"].inputs["tier1"] == {"inactive": "tier1_trace_mismatch"}
    late = ReplayInfo(10_000, REPLAY.trace_path)
    assert make().decide(snap(5000, pods, replay=late))["m"].inputs["tier1"] == {"inactive": "tier1_before_t0"}


def test_tier1_plus_overload_in_the_same_tick() -> None:
    p = make(tier1="oracle")
    pods = [pod("p0", 1000, M=1000)]
    d = p.decide(snap(1000, pods, load("a", "p0", 1000, 990, 200), replay=REPLAY))
    assert d["m"].reason == "tier1_window+tier2_overload" and d["m"].desired == 3
    d = p.decide(snap(1000, pods, replay=REPLAY))  # credited at the window start
    assert d["m"].reason == "hold" and d["m"].desired == 3


def test_last_window_mode() -> None:
    p = make(tier1="last_window")
    pods = [pod("p0", 1000)]
    d = p.decide(snap(1000, pods, replay=REPLAY))
    assert d["m"].reason == "tier1_no_history" and d["m"].desired == 1
    d = p.decide(snap(600_000, pods, replay=REPLAY))
    assert d["m"].reason == "tier1_window" and d["m"].desired == 2  # window 0's actual load
    assert d["m"].inputs["tier1"]["P_hat"] == 120000.0


def test_oracle_noisy_is_seeded_and_deterministic() -> None:
    def run(seed: int):
        p = PreServePolicy(cfg({"mu": {"m": dict(MU)}, "max_output_len": 4096}, seed=seed), oracle=ORACLE)
        seq = [snap(1000, [pod("p0", 1000)], load("a", "p0", 1000, 500, 50), replay=REPLAY),
               snap(3000, [pod("p0", 3000, (1.0, 64.0))], replay=REPLAY),
               snap(600_000, [pod("p0", 600_000, (2.0, 128.0))], replay=REPLAY)]
        return [p.decide(s) for s in seq]

    a, b = run(4), run(4)
    assert a == b
    inputs = a[0]["m"].inputs["tier1"]
    assert inputs["mode"] == "oracle_noisy"
    assert inputs["P_hat"] == round(120000 * t1.noise_factor(4, "m", 0, "P", t1.DEFAULT_NOISE_SIGMA), 1)
    assert inputs["D_hat"] == round(60000 * t1.noise_factor(4, "m", 0, "D", t1.DEFAULT_NOISE_SIGMA), 1)
    assert run(5)[0]["m"].inputs["tier1"]["P_hat"] != inputs["P_hat"]


def test_model_without_mu_is_held() -> None:
    p = PreServePolicy(cfg({"mu": {"m": dict(MU)}}, models=()), oracle=ORACLE)
    ms = ModelSnapshot(model="z", awake=2, min_replicas=1, max_replicas=4, gpus_per_replica=1,
                       ttft_slo_ms=500.0, tpot_slo_ms=75.0, max_num_seqs=None, pods=(), events=())
    d = p.decide(ClusterSnapshot(now_ms=1, tick_s=2.0, models={"z": ms}))
    assert d["z"].reason == "no_mu" and d["z"].desired == 2


# ------------------------------------------------------------ replay seed from the marker


def test_replay_seed_of_the_marker_wins_over_trace_seed_param() -> None:
    trace = str(FIX / "preserve_trace_segments.json")
    params = {"mu": {"m": dict(MU)}, "trace_path": trace, "trace_seed": 0, "window_s": 60.0}
    p = PreServePolicy(cfg(params))
    seed0 = to.load_oracle(trace, window_s=60.0, seed=0)
    seed7 = to.load_oracle(trace, window_s=60.0, seed=7)
    assert seed0.totals != seed7.totals  # the seed really moves the arrivals
    assert p.oracle.totals == seed0.totals
    # no seed in the marker: the param stays in force
    p.decide(snap(1000, [pod("p0", 1000)], replay=ReplayInfo(t0_ms=0, trace_path=trace)))
    assert p.oracle.totals == seed0.totals
    # marker seed: the oracle is rebuilt once with it
    p.decide(snap(2000, [pod("p0", 2000)], replay=ReplayInfo(t0_ms=0, trace_path=trace, seed=7)))
    assert p.oracle.totals == seed7.totals and p._oracle_seed == 7
    oracle = p.oracle
    p.decide(snap(3000, [pod("p0", 3000)], replay=ReplayInfo(t0_ms=0, trace_path=trace, seed=7)))
    assert p.oracle is oracle  # not reloaded again
    # a marker without seed after that keeps the last one (the marker is per run)
    p.decide(snap(4000, [pod("p0", 4000)], replay=ReplayInfo(t0_ms=0, trace_path=trace)))
    assert p.oracle is oracle


def test_injected_oracle_is_never_replaced_by_the_marker_seed() -> None:
    p = make()
    p.decide(snap(1000, [pod("p0", 1000)], replay=ReplayInfo(t0_ms=0, trace_path="x/trace.json", seed=3)))
    assert p.oracle is ORACLE


# ------------------------------------------------------------ review fixes (P1 / P2)


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


def test_second_replay_restarts_tier1_at_window_0() -> None:
    p = make(tier1="oracle")
    pods = [pod("p0", 1000)]
    assert p.decide(snap(1000, pods, replay=REPLAY))["m"].reason == "tier1_window"
    d = p.decide(snap(600_000, pods, replay=REPLAY))
    assert d["m"].inputs["tier1"]["window"] == 1 and d["m"].desired == 1
    second = ReplayInfo(t0_ms=5_000_000, trace_path=REPLAY.trace_path)
    d = p.decide(snap(5_001_000, pods, replay=second))  # a new run, same shell
    assert d["m"].reason == "tier1_window" and d["m"].inputs["tier1"]["window"] == 0
    assert d["m"].desired == 2
    d = p.decide(snap(5_003_000, pods, replay=second))
    assert d["m"].reason == "hold"


def test_tier2_never_isolates_below_the_windows_tier1_n() -> None:
    """Review P1-3: at a window start Tier-1 provisions N=2 before the burst fills the
    look-ahead maps; Tier-2 used to isolate back to 1 on the next tick (empty maps)."""
    p = make(tier1="oracle")
    two = lambda t, kv=(0.0, 0.0): [pod(f"p{i}", t, M=1000, kv=k) for i, k in enumerate(kv)]  # noqa: E731
    d = p.decide(snap(1000, two(1000), replay=REPLAY))["m"]
    assert (d.reason, d.desired, d.inputs["tier1_n"]) == ("tier1_window", 2, 2)
    d = p.decide(snap(3000, two(3000), replay=REPLAY))["m"]  # empty maps: Tier-2 wants 1
    assert d.desired == 2 and d.reason == "hold" and d.inputs["tier2_below_t1"] == 1
    # Tier-2 may add above N ...
    d = p.decide(snap(5000, two(5000, (0.99, 0.0)), load("a", "p0", 5000, 990, 200), replay=REPLAY))["m"]
    assert (d.reason, d.desired) == ("tier2_overload", 3)
    # ... and isolate back down to N, not below
    p.decide(snap(7000, two(7000, (0.99, 0.0)), [ev("done", "a", "p0", 7000)], replay=REPLAY))
    three = [pod(f"p{i}", 9000, M=1000, kv=0.0) for i in range(3)]
    d = p.decide(snap(9000, three, replay=REPLAY))["m"]
    assert (d.reason, d.desired) == ("tier2_underload", 2) and d.inputs["tier2_below_t1"] == 2
    # next window N=1: the floor follows the window
    d = p.decide(snap(600_000, two(600_000), awake=2, replay=REPLAY))["m"]
    assert d.desired == 1 and d.inputs["tier1_n"] == 1
