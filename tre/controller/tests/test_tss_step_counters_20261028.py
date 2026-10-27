"""TSS numerator from vLLM's step counters, 20 s window (2026-10-28, user decision; design
docs/design/20261028-tss-step-counters.md).

Invariants: the numerator is the tokens the engine PROCESSED in the window (counter
delta over the observed span, scaled to the window), not the tokens of the requests
that completed; a counter reset restarts the delta and flags the pod; a missing counter
is unknown, never zero; flat counters (a pod asleep) are no reset; the window length
comes from config and the registry's thetas must be stamped for it."""
from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from tre_common.registry import load_registry
from tre_common.tss import NUMERATOR_STEP_COUNTERS, theta_basis_errors
from tre_controller.config import ControllerConfig
from tre_controller.store.metrics_store import MetricsStore
from test_metrics_store import FakeRedis

REGISTRY_PATH = Path(__file__).resolve().parents[2] / "deploy" / "registry.yaml"
MODEL = "dsqwen-7b"
END = 1_000_000
WINDOW = 20_000
P = 2_000


def _docs(redis, pod, counter, *, completed=lambda ts: 0.0, drop_counters=(), start=END - 40_000):
    """Histogram + instant docs of ``pod`` every 2 s up to END; ``counter(ts)`` -> (prompt,
    generation) totals or None (no counter in that doc)."""
    key = f"default/{pod}"
    redis.sadd(f"tre:v2:pods:{MODEL}", key)
    for ts in range(start, END + 1, P):
        base = {"pod_name": pod, "timestamp": ts, "scraped_ms": ts - 50, "written_ms": ts + 5}
        doc = {**base, "model_histogram_metrics": {
            f"{MODEL}/request_prompt_tokens": {"sum": completed(ts), "count": completed(ts) / 100.0, "buckets": {}},
        }}
        values = counter(ts)
        if values is not None and ts not in drop_counters:
            doc["model_counter_metrics"] = {
                f"{MODEL}/prompt_tokens_total": values[0], f"{MODEL}/generation_tokens_total": values[1],
            }
        redis.zadd(f"tre:v2:hist:{key}", {json.dumps(doc): ts})
        redis.zadd(f"tre:v2:inst:{key}", {json.dumps({**base, "model_metrics": {
            f"{MODEL}/num_requests_running": 4.0, f"{MODEL}/num_requests_waiting": 0.0,
        }}): ts})


def _read(redis, window=WINDOW):
    store = MetricsStore(redis, load_registry(str(REGISTRY_PATH)), instant_sample_interval_ms=P, schema="v2")
    return store.read_model_window(MODEL, END - window, END, use_cache=False, start_exclusive=True)


def test_the_numerator_is_processed_tokens_not_completed_ones():
    # 50 prompt + 100 generated tokens per second, nothing completes in the window.
    redis = FakeRedis()
    _docs(redis, "a", lambda ts: (50.0 * ts / 1000, 100.0 * ts / 1000))
    w = _read(redis)
    assert w.generation_tokens == pytest.approx(100.0 * WINDOW / 1000)
    assert w.prompt_tokens == pytest.approx(50.0 * WINDOW / 1000)
    assert w.completed_prompt_tokens == 0.0 and w.request_count_observed == 0.0
    assert not w.token_counter_reset and w.sample_incomplete_pods == ()


def test_the_numerator_is_a_window_total_of_the_configured_window():
    redis = FakeRedis()
    _docs(redis, "a", lambda ts: (0.0, 100.0 * ts / 1000))
    assert _read(redis, 20_000).generation_tokens == pytest.approx(2_000.0)
    assert _read(redis, 30_000).generation_tokens == pytest.approx(3_000.0)


def test_a_counter_reset_restarts_the_delta_and_flags_the_pod():
    reset = END - 16_000  # the engine restarted: counters back near zero, then growing
    redis = FakeRedis()
    _docs(redis, "a", lambda ts: (0.0, 1e6 + 100.0 * ts / 1000) if ts < reset else (0.0, 100.0 * (ts - reset) / 1000))
    w = _read(redis)
    assert w.token_counter_reset is True
    assert "a" in w.sample_incomplete_pods
    # rate after the reset (100 tok/s) over the post-reset span (14 s >= 0.67 W), as a window total
    assert w.generation_tokens == pytest.approx(100.0 * WINDOW / 1000)


def test_a_short_token_span_is_unknown_not_scaled_up():
    # L3 (10-28): a pod woken (or restarted) late in the window has no baseline at the
    # window start; its span (here 8 s < 0.67 x 20 s) would scale its tokens up x2.5.
    # It leaves queue AND tokens; with no pod left the model's tokens are unknown.
    wake = END - 8_000
    redis = FakeRedis()
    _docs(redis, "a", lambda ts: (0.0, 100.0 * (ts - wake) / 1000), start=wake)
    w = _read(redis)
    assert w.prompt_tokens is None and w.generation_tokens is None
    assert "a" not in w.per_pod and "a" in w.sample_incomplete_pods
    assert w.avg_running == 0.0


def test_a_pod_without_a_queue_sample_is_unknown():
    # L2 (10-28): histogram docs but no instant doc - the queue is unknown, not 0 (0 would
    # raise Z): the pod leaves queue AND tokens, like a missing counter.
    redis = FakeRedis()
    _docs(redis, "a", lambda ts: (0.0, 100.0 * ts / 1000))
    _docs(redis, "b", lambda ts: (0.0, 100.0 * ts / 1000))
    redis.zsets.pop("tre:v2:inst:default/b")
    w = _read(redis)
    assert set(w.per_pod) == {"a"} and "b" in w.sample_incomplete_pods
    assert w.generation_tokens == pytest.approx(100.0 * WINDOW / 1000)
    assert w.avg_running == pytest.approx(4.0)


def test_a_missing_counter_is_unknown_never_zero():
    redis = FakeRedis()
    _docs(redis, "a", lambda ts: (0.0, 100.0 * ts / 1000), drop_counters=(END,))
    w = _read(redis)
    assert w.prompt_tokens is None and w.generation_tokens is None
    assert "a" not in w.per_pod and "a" in w.sample_incomplete_pods
    # a second pod with its counters keeps the model known - from that pod alone
    redis2 = FakeRedis()
    _docs(redis2, "a", lambda ts: (0.0, 100.0 * ts / 1000), drop_counters=(END,))
    _docs(redis2, "b", lambda ts: (0.0, 40.0 * ts / 1000))
    w2 = _read(redis2)
    assert set(w2.per_pod) == {"b"} and w2.generation_tokens == pytest.approx(40.0 * WINDOW / 1000)
    assert w2.avg_running == pytest.approx(4.0)  # queue from the same pods as the tokens


def test_flat_counters_while_asleep_are_no_reset():
    wake = END - 8_000
    redis = FakeRedis()
    _docs(redis, "a", lambda ts: (0.0, 5_000.0 + max(0.0, 100.0 * (ts - wake) / 1000)))
    w = _read(redis)
    assert not w.token_counter_reset and w.sample_incomplete_pods == ()
    assert w.generation_tokens == pytest.approx(100.0 * 8.0)


def test_the_window_is_20s_from_config_and_thetas_must_be_stamped_for_it(tmp_path):
    assert ControllerConfig.from_env({}).metrics_window_ms == 20_000
    registry = load_registry(str(REGISTRY_PATH))
    assert theta_basis_errors(registry.models(), window_ms=20_000) == []
    assert len(theta_basis_errors(registry.models(), window_ms=30_000)) == len(registry.models())
    # an unstamped (pre-2026-10-28) registry: the controller refuses to start
    raw = yaml.safe_load(REGISTRY_PATH.read_text(encoding="utf-8"))
    for model in raw["models"]:
        model["trs"].pop("numerator")
        model["trs"].pop("theta_window_ms")
    path = tmp_path / "registry.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    old = load_registry(str(path))
    assert all("trs.numerator=None" in line for line in theta_basis_errors(old.models(), window_ms=20_000))
    from tre_controller.app import create_controller_dependencies
    from test_controller_app import EmptyRedis

    with pytest.raises(ValueError, match="thetas were not fitted"):
        create_controller_dependencies(
            ControllerConfig.from_env({"TRE_REGISTRY_PATH": str(path)}), redis_client=EmptyRedis()
        )
    assert NUMERATOR_STEP_COUNTERS == "step_counters"


def test_the_v1_metrics_schema_is_refused_at_start():
    # L4 (10-28): v1 has no step counters / span scaling - every token total unknown.
    from tre_controller.app import create_controller_dependencies
    from test_controller_app import EmptyRedis

    cfg = ControllerConfig.from_env({"TRE_REGISTRY_PATH": str(REGISTRY_PATH), "TRE_METRICS_SCHEMA": "v1"})
    with pytest.raises(ValueError, match="TRE_METRICS_SCHEMA"):
        create_controller_dependencies(cfg, redis_client=EmptyRedis())


def test_online_and_offline_numerators_agree_on_clean_windows():
    # M2 (10-28): one synthetic counter series into the controller (MetricsStore) and into
    # the calibration replay (l3 CounterTokenSource, the fit numerator): equal window
    # totals on clean windows. A reset window that offline voids but online uses is
    # counted as such (selection-bias visibility), and both sides agree it is usable.
    from scripts import l3_numerator as l3
    from tre_controller.store import metrics_store as ms

    assert l3.ONLINE_MIN_SPAN_FACTOR == ms.MIN_TOKEN_SPAN_FACTOR
    assert l3.ONLINE_LOOKBACK_MS == ControllerConfig.from_env({}).histogram_lookback_ms

    def f_a(t):  # bursty: decode 300 tok/s with a 0-rate pause, prefill in steps
        s = (t - (END - 80_000)) / 1000
        return 4096.0 * (s // 7), 300.0 * s - 300.0 * max(0.0, min(s, 40.0) - 30.0)

    reset = END - 35_000

    def f_b(t):  # 120 tok/s, the engine restarts at ``reset`` (counters from 0)
        s = (t - (END - 80_000 if t < reset else reset)) / 1000
        return 50.0 * s, 120.0 * s

    redis = FakeRedis()
    offline = []
    for pod, f in (("a", f_a), ("b", f_b)):
        _docs(redis, pod, lambda ts, f=f: f(ts - 50), start=END - 80_000)  # scraped_ms = ts - 50
        offline.append(l3.PodCounters(f"default/{pod}", [(ts - 50, *f(ts - 50)) for ts in range(END - 80_000, END + 1, P)]))
    store = MetricsStore(redis, load_registry(str(REGISTRY_PATH)), instant_sample_interval_ms=P, schema="v2")
    src = l3.CounterTokenSource(offline)
    clean = online_only = 0
    for end in range(END - 50_000, END + 1, 10_000):
        start = end - WINDOW
        on = store.read_model_window(MODEL, start, end, use_cache=False, start_exclusive=True)
        off = src(start, end)
        if off is None:
            assert src.status[(start, end)] == l3.STATUS_RESET
            usable = (start, end) in src.online_used
            assert usable == ("b" in on.per_pod)
            online_only += usable
            continue
        clean += 1
        assert on.sample_incomplete_pods == ()
        assert (on.prompt_tokens, on.generation_tokens) == (pytest.approx(off[0]), pytest.approx(off[1]))
    assert clean >= 3 and online_only >= 1
