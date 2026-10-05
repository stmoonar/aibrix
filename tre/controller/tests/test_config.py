from __future__ import annotations

from pathlib import Path

import logging

import pytest
import yaml

from tre_common.rediskeys import SCRAPE_INTERVAL_MS
from tre_controller.config import ControllerConfig


def test_config_defaults_are_plan_aligned() -> None:
    config = ControllerConfig.from_env({})

    assert config.redis_url == "redis://aibrix-redis-master:6379/0"
    assert config.metrics_redis_url == "redis://aibrix-redis-master:6379/0"
    assert config.metrics_schema == "v2"
    assert config.service_manager_url == "http://aibrix-tre-service-manager:8000"
    assert config.registry_path.endswith("tre/deploy/registry.yaml")
    assert config.monitor_interval_s == 20.0
    assert config.metrics_refresh_interval_s == 5.0
    assert config.rescue_interval_s == 5.0
    assert config.fairness_interval_s == 10.0
    assert config.metrics_window_ms == 30_000
    assert config.metrics_window_mode == "sliding"
    # Must default to the gateway scrape cadence so expected_samples matches the real 10s
    # write cadence (r3 SMOKE_FINDINGS defect 2): a mismatch halves the controller's queue.
    assert config.instant_sample_interval_ms == SCRAPE_INTERVAL_MS == 10_000
    assert config.histogram_lookback_ms == 90_000
    assert config.min_latency_samples == 10
    assert config.percentile_mode == "bucket_upper"
    assert config.signal_source == "zm"
    assert config.signal_idle_rps_eps == 0.05
    assert config.signal_warmup_ms == -1
    assert config.sm_slow_timeout_s is None  # -> registry service_manager.api_call_timeout_s
    assert config.paper_stale_max_windows == 3
    assert config.incomplete_policy == "drop_model"
    assert config.enable_tre_scaling is True
    assert config.ablation_disable_fast_loop is False
    assert config.ablation_disable_safescale is False
    assert config.disable_eta_gate is False
    # v1/paper alignment A2: the receiver-less HIGH proactive SafeScale shrink is live.
    assert config.safescale_suppress_hot_proactive is False
    # Timer cleanup (2026-10-02): the band dwell, the fixed rollback backoff and the
    # floor-violation donor hold were removed. Their old variables are ignored without
    # validation: an old overlay starts.
    for key, value in (("TRE_DWELL_WINDOWS", "2"), ("TRE_DWELL_STATES", "bogus"),
                       ("TRE_SAFESCALE_ROLLBACK_BACKOFF_MS", "-1"),
                       ("TRE_FLOOR_VIOLATION_COOLDOWN_TICKS", "-1")):
        ControllerConfig.from_env({key: value})  # no ValueError


def test_config_donor_health_and_backoff_defaults_and_env() -> None:
    cfg = ControllerConfig.from_env({})
    assert cfg.safescale.donor_error_rate_max == 0.01
    assert cfg.safescale.donor_min_requests == 20.0
    assert cfg.safescale.kv_cache_max == 0.8
    assert cfg.gateway_stats_urls == ()  # guard source off unless configured
    assert cfg.gateway_route_namespace == "tre-v2"
    cfg = ControllerConfig.from_env(
        {
            "TRE_SAFESCALE_DONOR_ERROR_RATE_MAX": "0.05",
            "TRE_SAFESCALE_DONOR_MIN_REQUESTS": "50",
            "SAFE_SCALE_KV_CACHE_MAX": "0.9",
            "TRE_GATEWAY_STATS_URL": "http://a:19001/stats/prometheus, http://b:19001/stats/prometheus",
            "TRE_GATEWAY_ROUTE_NAMESPACE": "other",
        }
    )
    assert (cfg.safescale.donor_error_rate_max, cfg.safescale.donor_min_requests) == (0.05, 50.0)
    assert cfg.safescale.kv_cache_max == 0.9
    assert cfg.gateway_stats_urls == ("http://a:19001/stats/prometheus", "http://b:19001/stats/prometheus")
    assert cfg.gateway_route_namespace == "other"


def test_config_hot_proactive_guard_is_opt_in() -> None:
    assert ControllerConfig.from_env({"TRE_SAFESCALE_SUPPRESS_HOT_PROACTIVE": "1"}).safescale_suppress_hot_proactive is True
    assert ControllerConfig.from_env({"TRE_SAFESCALE_SUPPRESS_HOT_PROACTIVE": "0"}).safescale_suppress_hot_proactive is False


def test_config_reads_centralized_environment_values() -> None:
    config = ControllerConfig.from_env(
        {
            "TRE_REDIS_URL": "redis://redis.example:6379/2",
            "TRE_METRICS_REDIS_URL": "redis://metrics.example:6379/0",
            "TRE_METRICS_SCHEMA": "v1",
            "TRE_SERVICE_MANAGER_URL": "http://service-manager.example:9000",
            "TRE_REGISTRY_PATH": "/etc/aibrix/registry.yaml",
            "TRE_MONITOR_INTERVAL_SECONDS": "30",
            "TRE_RESCUE_INTERVAL_SECONDS": "1.5",
            "TRE_FAIRNESS_INTERVAL_SECONDS": "7.25",
            "TRE_METRICS_WINDOW_MS": "45000",
            "TRE_INSTANT_SAMPLE_INTERVAL_MS": "2500",
            "SAFE_SCALE_MIN_WINDOW_MS": "70000",
            "TRE_HIST_BASELINE_LOOKBACK_MS": "120000",
            "TRE_PERCENTILE_MODE": "interpolated",
            "TRE_SIGNAL_SOURCE": "latency_p95",
            "TRE_PAPER_STALE_MAX_WINDOWS": "5",
            "TRE_INCOMPLETE_POLICY": "drop_all",
            "ENABLE_TRE_SCALING": "false",
            "TRE_ABLATION_DISABLE_FAST_LOOP": "1",
            "TRE_ABLATION_DISABLE_SAFESCALE": "yes",
            "TRE_DISABLE_ETA_GATE": "true",
        }
    )

    assert config.redis_url == "redis://redis.example:6379/2"
    assert config.metrics_redis_url == "redis://metrics.example:6379/0"
    assert config.metrics_schema == "v1"
    assert config.service_manager_url == "http://service-manager.example:9000"
    assert config.registry_path == "/etc/aibrix/registry.yaml"
    assert config.monitor_interval_s == 30.0
    assert config.rescue_interval_s == 1.5
    assert config.fairness_interval_s == 7.25
    assert config.metrics_window_ms == 45_000
    assert config.instant_sample_interval_ms == 2_500
    assert config.histogram_lookback_ms == 120_000
    assert config.percentile_mode == "interpolated"
    assert config.signal_source == "latency_p95"
    assert config.paper_stale_max_windows == 5
    assert config.incomplete_policy == "drop_all"
    assert config.enable_tre_scaling is False
    assert config.ablation_disable_fast_loop is True
    assert config.ablation_disable_safescale is True
    assert config.disable_eta_gate is True


@pytest.mark.parametrize(
    "source",
    ["zm", "latency_p95", "queue_len", "decode_tps", "prefill_tps", "kv_cache"],
)
def test_config_accepts_plan_signal_sources(source: str) -> None:
    assert ControllerConfig.from_env({"TRE_SIGNAL_SOURCE": source}).signal_source == source


def test_queue_signal_source_rejects_missing_model_threshold(tmp_path) -> None:
    source = Path(__file__).parents[2] / "deploy" / "registry.yaml"
    registry = yaml.safe_load(source.read_text(encoding="utf-8"))
    registry["models"][0].pop("alt_thresholds")
    path = tmp_path / "registry.yaml"
    path.write_text(yaml.safe_dump(registry, sort_keys=False), encoding="utf-8")

    with pytest.raises(ValueError, match="missing=.*dsqwen-7b"):
        ControllerConfig.from_env(
            {"TRE_SIGNAL_SOURCE": "queue_len", "TRE_REGISTRY_PATH": str(path)}
        )


@pytest.mark.parametrize("source", ["decode_tps", "prefill_tps"])
def test_tps_signal_source_rejects_missing_model_threshold(tmp_path, source) -> None:
    registry_path = Path(__file__).parents[2] / "deploy" / "registry.yaml"
    registry = yaml.safe_load(registry_path.read_text(encoding="utf-8"))
    registry["models"][0]["alt_thresholds"].pop(source)
    path = tmp_path / "registry.yaml"
    path.write_text(yaml.safe_dump(registry, sort_keys=False), encoding="utf-8")

    with pytest.raises(ValueError, match="missing=.*dsqwen-7b"):
        ControllerConfig.from_env(
            {"TRE_SIGNAL_SOURCE": source, "TRE_REGISTRY_PATH": str(path)}
        )


@pytest.mark.parametrize("key", ["TRE_MONITOR_INTERVAL_SECONDS", "TRE_RESCUE_INTERVAL_SECONDS", "TRE_FAIRNESS_INTERVAL_SECONDS"])
def test_config_rejects_non_positive_loop_intervals(key: str) -> None:
    with pytest.raises(ValueError, match=key):
        ControllerConfig.from_env({key: "0"})


def test_config_centralizes_legacy_safescale_and_state_values() -> None:
    config = ControllerConfig.from_env(
        {
            "TRE_RUNTIME_STATE_DIR": "/var/lib/aibrix/tre",
            "PROACTIVE_RELEASE_MIN_TRS": "3000",
            "SAFE_SCALE_TTFT_P95_SLO_MS": "1300",
            "SAFE_SCALE_TPOT_P95_SLO_MS": "120",
            "SAFE_SCALE_WINDOW_FLOOR_MS": "90000",
            "SAFE_SCALE_E2E_MULTIPLIER": "3.5",
            "SAFE_SCALE_HQ": "0.5",
            "SAFE_SCALE_TAU_LOW": "1.25",
            "SAFE_SCALE_PROBE_POLL_SECONDS": "3",
        }
    )

    assert config.runtime_state_dir == "/var/lib/aibrix/tre"
    assert config.proactive_release_min_trs == 3000.0
    assert config.safescale.ttft_p95_slo_ms == 1300.0
    assert config.safescale.tpot_p95_slo_ms == 120.0
    assert config.safescale.min_window_ms == 90_000.0
    assert config.safescale.e2e_multiplier == 3.5
    assert config.safescale.hq == 0.5
    assert config.safescale.tau_low == 1.25
    assert config.safescale.probe_poll_seconds == 3.0


def test_config_rejects_invalid_percentile_mode() -> None:
    with pytest.raises(ValueError, match="TRE_PERCENTILE_MODE"):
        ControllerConfig.from_env({"TRE_PERCENTILE_MODE": "nearest"})


def test_metrics_window_mode_can_be_overridden_and_validated() -> None:
    assert ControllerConfig.from_env({"TRE_METRICS_WINDOW_MODE": "tumbling"}).metrics_window_mode == "tumbling"
    with pytest.raises(ValueError):
        ControllerConfig.from_env({"TRE_METRICS_WINDOW_MODE": "rolling"})


def test_safescale_short_floor_warns_instead_of_refusing_to_start(caplog) -> None:
    # N2 (tail observations fully post-hide) needs W_floor*(1-hq) >= metrics window +
    # refresh + read offset = 42000 by default; the 20 s floor cannot satisfy it, so the
    # guard only warns (v1 ran a 15 s floor the same way).
    with caplog.at_level(logging.WARNING, logger="tre_controller.config"):
        config = ControllerConfig.from_env({})
    assert config.safescale.min_window_ms == 20_000.0
    assert any("SAFE_SCALE_MIN_WINDOW_MS" in r.getMessage() and "min_window_ms=20000.0" in r.getMessage()
               for r in caplog.records)
    # A floor that satisfies the invariant (56000*0.75 = 42000) loads without the warning.
    caplog.clear()
    with caplog.at_level(logging.WARNING, logger="tre_controller.config"):
        ControllerConfig.from_env({"SAFE_SCALE_WINDOW_FLOOR_MS": "56000"})
    assert not [r for r in caplog.records if "SAFE_SCALE_MIN_WINDOW_MS" in r.getMessage()]


def test_safescale_window_defaults_and_env_parsing() -> None:
    # A6: W = max(2 * p95_e2e, 20 s), no ceiling, no default/fallback/decode terms.
    safescale = ControllerConfig.from_env({}).safescale
    assert (safescale.min_window_ms, safescale.e2e_multiplier, safescale.hq) == (20_000.0, 2.0, 0.25)
    tuned = ControllerConfig.from_env(
        {"SAFE_SCALE_WINDOW_FLOOR_MS": "30000", "SAFE_SCALE_E2E_MULTIPLIER": "1.5"}
    ).safescale
    assert (tuned.min_window_ms, tuned.e2e_multiplier) == (30_000.0, 1.5)
    for bad in ("0", "-1", "abc"):
        with pytest.raises(ValueError, match="SAFE_SCALE_E2E_MULTIPLIER"):
            ControllerConfig.from_env({"SAFE_SCALE_E2E_MULTIPLIER": bad})


def test_config_reads_and_validates_signal_idle_rps_epsilon() -> None:
    assert ControllerConfig.from_env(
        {"TRE_SIGNAL_IDLE_RPS_EPS": "0"}
    ).signal_idle_rps_eps == 0.0
    with pytest.raises(ValueError, match="TRE_SIGNAL_IDLE_RPS_EPS"):
        ControllerConfig.from_env({"TRE_SIGNAL_IDLE_RPS_EPS": "-0.1"})


def test_config_rejects_invalid_signal_source() -> None:
    with pytest.raises(ValueError, match="TRE_SIGNAL_SOURCE"):
        ControllerConfig.from_env({"TRE_SIGNAL_SOURCE": "legacy"})


def test_config_rejects_invalid_bool() -> None:
    with pytest.raises(ValueError, match="ENABLE_TRE_SCALING"):
        ControllerConfig.from_env({"ENABLE_TRE_SCALING": "maybe"})


def test_config_rejects_invalid_metrics_schema() -> None:
    with pytest.raises(ValueError, match="TRE_METRICS_SCHEMA"):
        ControllerConfig.from_env({"TRE_METRICS_SCHEMA": "legacy"})


def test_config_rejects_invalid_paper_stale_window_limit() -> None:
    with pytest.raises(ValueError, match="TRE_PAPER_STALE_MAX_WINDOWS"):
        ControllerConfig.from_env({"TRE_PAPER_STALE_MAX_WINDOWS": "0"})


def test_config_rejects_invalid_incomplete_policy() -> None:
    with pytest.raises(ValueError, match="TRE_INCOMPLETE_POLICY"):
        ControllerConfig.from_env({"TRE_INCOMPLETE_POLICY": "drop_cluster"})



def test_sm_call_timeout_defaults_to_the_registry_and_must_outlast_a_sleep() -> None:
    from types import SimpleNamespace

    import pytest

    from tre_common.registry import ClusterTopology, Registry, parse_service_manager_config
    from tre_controller.app import resolve_sm_call_timeout_s

    registry = Registry(ClusterTopology(nodes=()), [], service_manager=parse_service_manager_config(None))
    assert resolve_sm_call_timeout_s(SimpleNamespace(sm_slow_timeout_s=None), registry) == 360.0
    assert resolve_sm_call_timeout_s(SimpleNamespace(sm_slow_timeout_s=400.0), registry) == 400.0
    # The timeout must outlast the SM's worst-case call (common formula: lock wait +
    # longest writer-lock hold incl. bounded Kubernetes calls + I/O margin).
    worst = registry.service_manager().worst_case_sleep_call_s()
    assert worst < 360.0
    assert resolve_sm_call_timeout_s(SimpleNamespace(sm_slow_timeout_s=worst + 1.0), registry) == worst + 1.0
    with pytest.raises(ValueError, match="TRE_SM_SLOW_TIMEOUT_SECONDS"):
        resolve_sm_call_timeout_s(SimpleNamespace(sm_slow_timeout_s=worst), registry)
    slow = Registry(
        ClusterTopology(nodes=()),
        [],
        # 150 s: one sleep, a compensating sleep and the lock wait exceed 360 s.
        service_manager=parse_service_manager_config({"sleep": {"sleep_call_timeout_s": 150}}),
    )
    with pytest.raises(ValueError, match="api_call_timeout_s = 360s"):
        resolve_sm_call_timeout_s(SimpleNamespace(sm_slow_timeout_s=None), slow)
