import math
from types import SimpleNamespace

import pytest

from tre_baselines.policies import POLICIES, build_policy
from tre_baselines.policies.chiron import ChironPolicy
from tre_baselines.snapshot import ClusterSnapshot, ModelSnapshot, PodSnapshot

M = "m"


def pod(name, t_ms, gen, isum, icnt, running=0.0, waiting=0.0):
    return PodSnapshot(
        pod=name, model=M, node=None, gpu_ids=(0,), running=running, waiting=waiting,
        kv_usage=None,
        counters={"gen_tokens": gen, "itl_sum": isum, "itl_count": icnt},
        num_gpu_blocks=None, block_size=None, scraped_at_ms=t_ms,
    )


def snap(pods, awake=None, t_ms=0, unscraped=(), max_num_seqs=100, tick=0):
    awake = len(pods) if awake is None else awake
    ms = ModelSnapshot(
        model=M, awake=awake, min_replicas=1, max_replicas=8, gpus_per_replica=1,
        ttft_slo_ms=1000.0, tpot_slo_ms=100.0, max_num_seqs=max_num_seqs,
        pods=tuple(pods), events=(), unscraped=tuple(unscraped),
    )
    return ClusterSnapshot(now_ms=t_ms, tick_s=2.0, models={M: ms}, tick=tick)


def policy(**params):
    params.setdefault("theta", {"*": 1 / 3})
    return ChironPolicy(SimpleNamespace(policy_params=params))


def one(pol, s):
    return pol.decide(s)[M]


def info(d, name="p0"):
    return d.inputs["pods"][name]


P = dict(b_init=10, b_max=100, alpha=0.5)


def test_alg1_hand_trajectory():
    pol = policy(**P)
    d = one(pol, snap([pod("p0", 0, 0, 0, 0, running=4)]))
    assert info(d)["skip"] == "baseline" and info(d)["B"] == 10
    # window 1: thr=100, ITL=.02 -> LBP=.2, gate off (running 4 < 10 last tick): B=.5*10/.2+5=30
    d = one(pol, snap([pod("p0", 2000, 200, 2, 100, running=12)], t_ms=2000))
    assert info(d)["LBP"] == pytest.approx(0.2) and info(d)["TBP"] is None
    assert info(d)["B"] == pytest.approx(30.0)
    # window 2: thr=200, LBP=.4, cap was binding (12>=10): TBP=100/200=.5 -> bp=.5: B=30+15=45
    d = one(pol, snap([pod("p0", 4000, 600, 6, 200, running=50)], t_ms=4000))
    assert info(d)["TBP"] == pytest.approx(0.5)
    assert info(d)["B"] == pytest.approx(45.0)
    # window 3: thr=300, LBP=.6, TBP=200/300 -> bp=2/3: B=.5*45*1.5+22.5=56.25
    d = one(pol, snap([pod("p0", 6000, 1200, 12, 300, running=60)], t_ms=6000))
    assert info(d)["B"] == pytest.approx(56.25)
    # window 4: thr=150, ITL=.12 -> LBP=1.2, TBP=300/150=2 -> bp=2 >= 1: halve
    d = one(pol, snap([pod("p0", 8000, 1500, 24, 400, running=60)], t_ms=8000))
    assert info(d)["TBP"] == pytest.approx(2.0)
    assert info(d)["B"] == pytest.approx(28.125, abs=0.01)


def test_growth_capped_at_b_max():
    pol = policy(b_init=90, b_max=100)
    one(pol, snap([pod("p0", 0, 0, 0, 0)]))
    d = one(pol, snap([pod("p0", 2000, 200, 1, 100)], t_ms=2000))  # LBP=.1 -> 495 -> 100
    assert info(d)["B"] == 100.0


def test_default_bounds_from_max_num_seqs_then_256():
    pol = policy()
    d = one(pol, snap([pod("p0", 0, 0, 0, 0)], max_num_seqs=64))
    assert info(d)["B"] == 64
    d = one(policy(), snap([pod("p0", 0, 0, 0, 0)], max_num_seqs=None))
    assert info(d)["B"] == 256


def _drop_run(running_at_t1):
    pol = policy(**P)
    one(pol, snap([pod("p0", 0, 0, 0, 0)]))
    d1 = one(pol, snap([pod("p0", 2000, 200, 2, 100, running=running_at_t1)], t_ms=2000))
    b1 = info(d1)["B"]
    # throughput halves (thr 50), latency still low (LBP .2)
    d2 = one(pol, snap([pod("p0", 4000, 300, 4, 200, running=1)], t_ms=4000))
    return b1, info(d2)


def test_tbp_gate_not_binding_does_not_halve():
    b1, i2 = _drop_run(running_at_t1=5)  # 5 < B_in=10 -> cap not binding
    assert i2["TBP"] is None
    assert i2["B"] > b1


def test_tbp_gate_binding_halves():
    b1, i2 = _drop_run(running_at_t1=12)  # 12 >= 10 -> cap binding
    assert i2["TBP"] == pytest.approx(2.0)
    assert i2["B"] == pytest.approx(b1 / 2)


def test_empty_window_and_zero_thr_do_not_update():
    pol = policy(**P)
    one(pol, snap([pod("p0", 0, 100, 1, 10)]))
    d = one(pol, snap([pod("p0", 2000, 100, 1, 10)], t_ms=2000))
    assert info(d)["skip"] == "empty_window" and info(d)["B"] == 10
    # itl_count moved but no tokens
    d = one(pol, snap([pod("p0", 4000, 100, 2, 20)], t_ms=4000))
    assert info(d)["skip"] == "empty_window" and info(d)["B"] == 10


def test_counter_reset_skips_and_rebaselines():
    pol = policy(**P)
    one(pol, snap([pod("p0", 0, 1000, 10, 100)]))
    d = one(pol, snap([pod("p0", 2000, 50, 1, 10)], t_ms=2000))  # decrease -> reset
    assert info(d)["skip"] == "counter_reset" and info(d)["B"] == 10
    # next window is differenced against the post-reset baseline: 200 tok/2s, ITL .02
    d = one(pol, snap([pod("p0", 4000, 250, 3, 110)], t_ms=4000))
    assert info(d)["thr"] == pytest.approx(100.0)
    assert info(d)["B"] == pytest.approx(30.0)


def test_pod_add_remove_state():
    pol = policy(**P)
    one(pol, snap([pod("p0", 0, 0, 0, 0)]))
    d = one(pol, snap([pod("p0", 2000, 200, 2, 100), pod("p1", 2000, 999, 9, 9)], t_ms=2000))
    assert info(d, "p0")["B"] == 30.0
    assert info(d, "p1")["skip"] == "baseline" and info(d, "p1")["B"] == 10  # new pod: b_init
    # p1 unscraped keeps state; p0 disappears and loses it
    one(pol, snap([], awake=2, unscraped=("p1",), t_ms=4000))
    assert set(pol._state[M]) == {"p1"}
    one(pol, snap([pod("p0", 6000, 5, 1, 1)], awake=2, t_ms=6000))
    assert set(pol._state[M]) == {"p0"}
    d = one(pol, snap([pod("p0", 8000, 205, 3, 101)], t_ms=8000))
    assert info(d)["B"] == 10 or info(d)["B"] == pytest.approx(30.0)  # fresh B=10, LBP .02 -> grows


def test_no_pods():
    d = one(policy(), snap([], awake=3))
    assert (d.desired, d.reason) == (3, "no_pods")


def _busy_snap(specs, awake):
    return snap([pod(f"p{i}", 0, 0, 0, 0, running=r, waiting=w) for i, (r, w) in enumerate(specs)],
                awake=awake)


def test_busy_def_at_cap():
    pol = policy(busy_def="at_cap", b_init=10)
    # p0 at cap via running, p1 via running+waiting, p2 idle -> busy 2 -> ceil(2 / (1/3)) = 6
    d = one(pol, _busy_snap([(10, 0), (5, 5), (3, 0)], awake=3))
    assert d.inputs["busy"] == 2 and d.reason == "ibp_target" and d.desired == 6
    assert d.inputs["IBP"] == pytest.approx(0.6667) and d.inputs["target"] == 6


def test_busy_def_nonidle():
    pol = policy(busy_def="nonidle", b_init=10)
    d = one(pol, _busy_snap([(1, 0), (0, 0), (0, 0)], awake=3))  # busy 1 at theta 1/3 -> 3
    assert d.inputs["busy"] == 1 and d.reason == "ibp_target" and d.desired == 3
    d = one(policy(busy_def="nonidle", b_init=10), _busy_snap([(0, 0)] * 3, awake=3))
    assert d.reason == "ibp_target" and d.desired == 1  # nothing busy: floor of one instance


def test_theta_per_model_and_default():
    pol = policy(busy_def="nonidle", theta={M: 0.5, "*": 0.9})
    d = one(pol, _busy_snap([(1, 0), (0, 0)], awake=2))  # busy 1 / .5 = 2
    assert d.desired == 2 and d.inputs["theta"] == 0.5
    with pytest.raises(ValueError):  # no silent 1/3 default for a model without theta
        one(policy(busy_def="nonidle", theta={"other": 0.1}), _busy_snap([(1, 0)] * 2, awake=2))


def test_theta_and_busy_def_defaults_come_from_config():
    with pytest.raises(ValueError):
        ChironPolicy(SimpleNamespace(policy_params={}))
    with pytest.raises(ValueError):  # a managed model without theta (and no "*")
        ChironPolicy(SimpleNamespace(policy_params={"theta": {"a": 0.5}}, models={"a": 1, "b": 1}))
    assert policy().busy_def == "effective"  # packed busy count; nonidle / at_cap for sensitivity


def test_unknown_is_not_idle_no_scale_down():
    # Review repro: awake 3, the busy pod is unscraped, the two scraped ones idle -> was 1.
    pol = dict(busy_def="nonidle", theta={"*": 1 / 3})
    d = one(policy(**pol), snap([pod("p0", 0, 0, 0, 0), pod("p1", 0, 0, 0, 0)], awake=3, unscraped=("p2",)))
    assert (d.desired, d.reason) == (3, "incomplete") and d.inputs["policy_desired"] == 1
    # a missing running/waiting gauge is unknown, not 0
    d = one(policy(**pol), snap([pod("p0", 0, 0, 0, 0, running=None), pod("p1", 0, 0, 0, 0)], awake=2))
    assert (d.desired, d.reason) == (2, "incomplete") and "missing_gauges" in d.inputs["gaps"]
    # scale-up still uses the evidence there is: 2 known busy pods at 1/3 -> 6
    d = one(policy(**pol), snap([pod("p0", 0, 0, 0, 0, running=1), pod("p1", 0, 0, 0, 0, running=1)],
                           awake=3, unscraped=("p2",)))
    assert d.desired == 6 and d.reason == "ibp_target"


def test_exact_multiples_do_not_round_up():
    for theta in (1 / 3, 0.3333333333, 0.25, 0.2, 0.5):
        pol = policy(busy_def="nonidle", theta={"*": theta})
        for busy in range(1, 9):
            d = one(pol, _busy_snap([(1, 0)] * busy, awake=busy))
            assert d.desired == busy * round(1 / theta), (theta, busy, d.desired)  # theta = 1/k


def _closed_loop(theta, busy, awake0, ticks=40):
    """Constant ``busy`` pods, the rest idle; the model follows the decision each tick."""
    pol = policy(busy_def="nonidle", theta={"*": theta})
    awake, seen = awake0, []
    for t in range(ticks):
        specs = [(1, 0)] * busy + [(0, 0)] * max(0, awake - busy)
        d = pol.decide(snap([pod(f"p{i}", t * 2000, 0, 0, 0, running=r, waiting=w)
                             for i, (r, w) in enumerate(specs)], awake=awake, t_ms=t * 2000))[M]
        seen.append(d.desired)
        awake = max(1, min(8, d.desired))
    return seen


@pytest.mark.parametrize("theta", [0.3333, 0.3333333333, 0.37, 0.45])
@pytest.mark.parametrize("busy", [1, 2])
def test_constant_load_never_flip_flops(theta, busy):
    for awake0 in (1, 2, 3, 5, 8):
        seen = _closed_loop(theta, busy, awake0)
        assert len(set(seen[1:])) == 1, (theta, busy, awake0, seen[:10])  # settled after one tick
        target = seen[-1]
        assert target * theta >= busy - 1e-6  # IBP = busy / target <= theta
    # the old +-1 rule flip-flopped here: theta .37, busy 1 -> N 2 (IBP .5) up, N 3 (.33) down
    assert set(_closed_loop(0.37, 1, 2)) == {3}


def test_bad_params():
    with pytest.raises(ValueError):
        policy(busy_def="bogus")
    with pytest.raises(ValueError):
        policy(alpha=0)
    with pytest.raises(ValueError):
        policy(theta={"*": 0})


def test_deterministic_and_json_able():
    import json

    seq = [
        snap([pod("p0", 0, 0, 0, 0, running=4)]),
        snap([pod("p0", 2000, 200, 2, 100, running=12, waiting=3)], t_ms=2000),
        snap([pod("p0", 4000, 600, 6, 200, running=50)], t_ms=4000),
    ]
    runs = []
    for _ in range(2):
        pol = policy(**P)
        runs.append([pol.decide(s)[M] for s in seq])
    assert runs[0] == runs[1]
    json.dumps([r.inputs for r in runs[0]])


def test_registered():
    assert POLICIES["chiron"] is ChironPolicy
    assert build_policy("chiron", SimpleNamespace(policy_params={"theta": 0.5})).name == "chiron"


def test_effective_busy_spread_load_gives_a_constant_target_no_ratchet():
    """Review P1-1: the gateway spreads 0.5x of one pod's B over every awake pod. nonidle
    counts every pod busy and ratchets to the cap; the packed count stays at
    ceil(1 / theta) instances whatever N is."""
    theta, B, cap = 0.45, 100.0, 8

    def run(busy_def, awake):
        pol = policy(busy_def=busy_def, theta={"*": theta}, b_init=B, b_max=B)
        seen = []
        for t in range(12):
            per_pod = 0.5 * B / awake  # total load 0.5 B, spread evenly
            d = one(pol, snap([pod(f"p{i}", t * 2000, 0, 0, 0, running=per_pod) for i in range(awake)],
                              awake=awake, t_ms=t * 2000, max_num_seqs=None))
            seen.append(d.desired)
            awake = max(1, min(cap, d.desired))
        return seen

    want = math.ceil(1 / theta)  # busy_eff = ceil(0.5 B / B) = 1
    for awake0 in (1, 3, 8):
        assert set(run("effective", awake0)) == {want}, awake0
    assert min(cap, run("nonidle", 1)[-1]) == cap  # the old definition ratchets up to the cap


def test_b_max_is_what_the_kv_cache_holds_for_the_trace_shape():
    """Review P2-C: B is capped by min(max_num_seqs, KV blocks x block size / (in + out))."""
    def kv_pod(blocks):
        return PodSnapshot(pod="p0", model=M, node=None, gpu_ids=(0,), running=1.0, waiting=0.0, kv_usage=None,
                           counters={}, num_gpu_blocks=blocks, block_size=16, scraped_at_ms=0)

    d = one(policy(kv_request_tokens=892), snap([kv_pod(20381)], max_num_seqs=1024))
    assert d.inputs["b_max"] == 365 and d.inputs["b_max_src"] == "kv_cache"   # 20381 * 16 // 892
    assert info(d)["B"] == 365                                                  # b_init capped too
    d = one(policy(kv_request_tokens=892), snap([kv_pod(20381)], max_num_seqs=256))
    assert (d.inputs["b_max"], d.inputs["b_max_src"]) == (256, "max_num_seqs")
    d = one(policy(), snap([kv_pod(20381)], max_num_seqs=1024))                 # shape not configured
    assert (d.inputs["b_max"], d.inputs["b_max_src"]) == (1024, "max_num_seqs")
    d = one(policy(b_max=50, kv_request_tokens=892), snap([kv_pod(20381)]))
    assert (d.inputs["b_max"], d.inputs["b_max_src"]) == (50, "param")
