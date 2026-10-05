from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest
import yaml

from tre_baselines.config import Config, ModelLimits
from tre_baselines.policies import POLICIES, build_policy
from tre_baselines.snapshot import ClusterSnapshot, ModelSnapshot, PodSnapshot, RequestEvent

M = "m"
BIG = 1000.0
_COUNTER = [0]


def _grid(**over):
    g = [[BIG] * 3 for _ in range(3)]
    for k, v in over.items():
        g[int(k[1])][int(k[2])] = v
    return g


def _params(**kw):
    p = {
        "models": [M],
        "bucket_edges": {"*": {"in": [100, 200], "out": [50, 100]}},
        "velocity": {M: {"buckets": _grid(b00=7, b11=11, b22=25), "v_prefill": BIG}},
        "misbucket_rate": 0.0,
        "window_s": 10,
    }
    p.update(kw)
    return p


def _cfg(params, **kw):
    return Config(sm_url="x", redis_url="y", policy="tokenscale", policy_params=params, **kw)


def _policy(**kw):
    return build_policy("tokenscale", _cfg(_params(**kw)))


def _ev(ts, tin=50, mt=20, src="header", reissue="none", kind="arr"):
    _COUNTER[0] += 1
    return RequestEvent(kind=kind, model=M, pod=None, req_id=f"r{_COUNTER[0]}", ts_ms=ts, in_tokens=tin,
                        in_src=src, max_tokens=mt, out_tokens=None, status=None, reissue=reissue)


COMPLETE = object()


def _model(now, events=(), awake=2, busy=0.0, name=M, cover=True, since=COMPLETE):
    """Default: complete evidence (the stream covers the pod, 60 s of gap-free events)."""
    pod = PodSnapshot(pod="p", model=name, node=None, gpu_ids=(0,), running=busy, waiting=0.0, kv_usage=0.1,
                      counters={}, num_gpu_blocks=None, block_size=None, scraped_at_ms=now,
                      tracked_inflight=int(busy), events_cover=cover)
    return ModelSnapshot(model=name, awake=awake, min_replicas=1, max_replicas=8, gpus_per_replica=1,
                         ttft_slo_ms=500, tpot_slo_ms=75, max_num_seqs=None, pods=(pod,), events=tuple(events),
                         events_since_ms=now - 60_000 if since is COMPLETE else since)


def _snap(now, events=(), awake=2, busy=0.0, **kw):
    return ClusterSnapshot(now_ms=now, tick_s=2.0, models={M: _model(now, events, awake, busy, **kw)})


def _hand_events(t=95_000):
    # b00: 2 x (50+20)=140 ; b11: 150+70=220 ; b22: 300+200=500
    return [_ev(t, 50, 20), _ev(t, 50, 20), _ev(t, 150, 70), _ev(t, 300, 200)]


def test_registered():
    assert "tokenscale" in POLICIES


def test_hand_computed_3x3():
    d = _policy().decide(_snap(100_000, _hand_events()))[M]
    # lambda: 14, 22, 50 tok/s ; 14/7 + 22/11 + 50/25 = 6 ; lambda_in = 550/10 = 55 -> 0.055
    assert d.reason == "velocity" and d.desired == 6
    lam = d.inputs["lambda_b"]
    assert (lam["00"], lam["11"], lam["22"]) == (14.0, 22.0, 50.0)
    assert d.inputs["term_buckets"] == 6.0 and d.inputs["lambda_in"] == 55.0


def test_ceil_rounds_up():
    p = _policy(velocity={M: {"buckets": _grid(b00=6, b11=11, b22=25), "v_prefill": BIG}})
    d = p.decide(_snap(100_000, _hand_events()))[M]
    assert d.inputs["term_buckets"] > 6.3 and d.desired == 7  # 14/6 + 2 + 2 = 6.33


def test_prefill_guard_dominates():
    p = _policy(velocity={M: {"buckets": _grid(b00=7, b11=11, b22=25), "v_prefill": 5.0}})
    d = p.decide(_snap(100_000, _hand_events()))[M]
    assert d.inputs["term_prefill"] == 11.0 and d.desired == 11  # 55/5 > 6


def test_window_eviction():
    p = _policy()
    p.decide(_snap(100_000, [_ev(95_000, 50, 20)]))
    assert p.decide(_snap(104_000))[M].inputs["events"] == 1
    d = p.decide(_snap(105_000))[M]  # ts <= now - window is evicted
    assert d.inputs["events"] == 0 and d.reason == "idle" and d.desired == 0
    p2 = _policy()
    p2.decide(_snap(100_000, [_ev(90_001, 50, 20)]))
    assert p2.decide(_snap(100_000))[M].inputs["events"] == 1


def test_reissue_skipped_and_other_kinds_ignored():
    evs = [_ev(99_000, reissue="continued"), _ev(99_000, reissue="retried"),
           _ev(99_000, kind="done"), _ev(99_000, kind="ft"), _ev(99_000)]
    d = _policy().decide(_snap(100_000, evs))[M]
    assert d.inputs["events"] == 1 and d.inputs["skipped_reissue"] == 2
    d2 = _policy(skip_reissue=False).decide(_snap(100_000, [_ev(99_000, reissue="continued")]))[M]
    assert d2.inputs["events"] == 1


def test_missing_max_tokens_uses_default():
    d = _policy(default_out_tokens=80).decide(_snap(100_000, [_ev(99_000, 50, None)]))[M]
    assert d.inputs["missing_out"] == 1
    assert d.inputs["lambda_b"]["01"] == (50 + 80) / 10  # out 80 -> output bucket 1


def _run_mis(seed):
    p = _policy(misbucket_rate=0.15, seed=seed)
    return p.decide(_snap(100_000, [_ev(99_000, 50, 20) for _ in range(4000)]))[M]


def test_misbucket_rate_and_determinism():
    d = _run_mis(7)
    assert 0.13 < d.inputs["misbucketed"] / 4000 < 0.17
    # a misbucketed event never stays in its true bucket, so bucket 00 keeps ~85 %
    assert 0.83 < d.inputs["lambda_b"]["00"] * 10 / (4000 * 70) < 0.87
    assert _run_mis(7) == d
    assert _run_mis(8).inputs["lambda_b"] != d.inputs["lambda_b"]


def test_rate_zero_never_misbuckets():
    d = _policy(misbucket_rate=0.0).decide(_snap(100_000, [_ev(99_000) for _ in range(200)]))[M]
    assert d.inputs["misbucketed"] == 0


def test_unknown_is_not_idle_no_scale_down():
    # Review repro: first start, no event history, the pod runs 20 requests -> was idle/0.
    d = _policy().decide(_snap(100_000, awake=3, busy=20.0, cover=False, since=None))[M]
    assert (d.desired, d.reason) == (3, "incomplete")
    assert set(d.inputs["gaps"]) == {"no_event_history", "event_gap"}
    # after a stream gap the pod still runs requests from before it (cohort not drained)
    p = _policy()
    p.decide(_snap(100_000, [_ev(99_000)]))
    d = p.decide(_snap(140_000, awake=3, busy=4.0, cover=False))[M]
    assert (d.desired, d.reason) == (3, "incomplete") and d.inputs["gaps"] == ["event_gap"]
    # a window not yet covered by gap-free history (5 s of a 10 s window): no scale-down ...
    d = _policy().decide(_snap(100_000, [_ev(99_000)], awake=3, since=95_000))[M]
    assert (d.desired, d.reason) == (3, "incomplete") and d.inputs["policy_desired"] == 1
    # ... but a scale-up still acts on the arrivals seen so far
    d = _policy().decide(_snap(100_000, _hand_events(99_000), awake=3, since=95_000))[M]
    assert (d.desired, d.reason) == (6, "velocity")


def test_empty_window_is_idle_only_when_engines_are_idle():
    p = _policy()
    p.decide(_snap(100_000, [_ev(99_000)]))
    d = p.decide(_snap(140_000, awake=3, busy=0.0))[M]
    assert d.reason == "idle" and d.desired == 0
    d = p.decide(_snap(142_000, awake=3, busy=2.0))[M]      # tracked long requests still decoding
    assert (d.desired, d.reason) == (3, "empty_window_busy")
    assert p.counters()[M]["empty_window_busy"] == 1            # reported per run


def test_arrivals_without_an_input_count_count_like_estimates():
    no_in = [_ev(99_000, tin=None) for _ in range(2)]
    d = _policy().decide(_snap(100_000, no_in + [_ev(99_000) for _ in range(8)], awake=4))[M]
    assert d.reason == "degraded_estimate_frac" and d.inputs["estimate_frac"] == 0.2
    assert d.inputs["events"] == 8 and d.inputs["skipped_missing_in"] == 2


def test_degraded_estimate_fraction():
    evs = [_ev(99_000, src="estimate")] + [_ev(99_000) for _ in range(9)]
    d = _policy().decide(_snap(100_000, evs, awake=4))[M]
    assert d.reason == "degraded_estimate_frac" and d.desired == 4 and d.inputs["estimate_frac"] == 0.1
    ok = [_ev(99_000, src="estimate")] + [_ev(99_000) for _ in range(99)]  # 1 % <= 5 %
    assert _policy().decide(_snap(100_000, ok, awake=4))[M].reason == "velocity"


def test_unmanaged_model_gets_no_decision():
    snap = _snap(100_000)
    other = _model(100_000, name="other")
    out = _policy().decide(replace(snap, models={M: snap.models[M], "other": other}))
    assert set(out) == {M}


@pytest.mark.parametrize("bad", [None, 0, -3, float("nan"), "x"])
def test_factory_fails_closed_on_bad_velocity(bad):
    g = _grid()
    g[1][2] = bad
    with pytest.raises(ValueError, match=r"buckets\[1\]\[2\]"):
        _policy(velocity={M: {"buckets": g, "v_prefill": 5.0}})
    with pytest.raises(ValueError, match="v_prefill"):
        _policy(velocity={M: {"buckets": _grid(), "v_prefill": bad}})


def test_factory_fails_closed_on_missing_model_or_edges():
    with pytest.raises(ValueError, match=r"velocity\[m\] missing"):
        _policy(velocity={})
    with pytest.raises(ValueError, match="bucket_edges"):
        _policy(bucket_edges={})
    with pytest.raises(ValueError, match="out_len_source"):
        _policy(out_len_source="predictor")


def test_registry_models_must_all_have_velocity():
    lim = {n: ModelLimits(n, 1, 4, 1, 500, 75, None) for n in (M, "n2")}
    params = _params()
    del params["models"]
    with pytest.raises(ValueError, match=r"velocity\[n2\]"):
        build_policy("tokenscale", _cfg(params, models=lim))


def test_example_params_load():
    path = Path(__file__).resolve().parents[1] / "examples" / "tokenscale.yaml"
    build_policy("tokenscale", _cfg(yaml.safe_load(path.read_text(encoding="utf-8"))))
