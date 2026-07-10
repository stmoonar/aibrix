from __future__ import annotations

from pathlib import Path

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
    assert config.sm_slow_timeout_s == 300.0
    assert config.paper_stale_max_windows == 3
    assert config.incomplete_policy == "drop_model"
    assert config.enable_tre_scaling is True
    assert config.ablation_disable_fast_loop is False
    assert config.ablation_disable_safescale is False
    assert config.disable_eta_gate is False


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
            "SAFE_SCALE_DEFAULT_WINDOW_MS": "70000",
            "SAFE_SCALE_MIN_WINDOW_MS": "20000",
            "SAFE_SCALE_MAX_WINDOW_MS": "320000",
            "SAFE_SCALE_CW2_FALLBACK_MS": "310000",
            "SAFE_SCALE_CDEC": "3",
            "SAFE_SCALE_HQ": "0.5",
            "SAFE_SCALE_TAU_LOW": "1.25",
            "SAFE_SCALE_EPSILON_MU": "0.000001",
            "SAFE_SCALE_PROBE_POLL_SECONDS": "3",
        }
    )

    assert config.runtime_state_dir == "/var/lib/aibrix/tre"
    assert config.proactive_release_min_trs == 3000.0
    assert config.safescale.ttft_p95_slo_ms == 1300.0
    assert config.safescale.tpot_p95_slo_ms == 120.0
    assert config.safescale.default_window_ms == 70_000.0
    assert config.safescale.min_window_ms == 20_000.0
    assert config.safescale.max_window_ms == 320_000.0
    assert config.safescale.cw2_fallback_ms == 310_000.0
    assert config.safescale.cdec == 3.0
    assert config.safescale.hq == 0.5
    assert config.safescale.tau_low == 1.25
    assert config.safescale.epsilon_mu == 0.000001
    assert config.safescale.probe_poll_seconds == 3.0


def test_config_rejects_invalid_percentile_mode() -> None:
    with pytest.raises(ValueError, match="TRE_PERCENTILE_MODE"):
        ControllerConfig.from_env({"TRE_PERCENTILE_MODE": "nearest"})


def test_metrics_window_mode_can_be_overridden_and_validated() -> None:
    assert ControllerConfig.from_env({"TRE_METRICS_WINDOW_MODE": "tumbling"}).metrics_window_mode == "tumbling"
    with pytest.raises(ValueError):
        ControllerConfig.from_env({"TRE_METRICS_WINDOW_MODE": "rolling"})


def test_safescale_window_must_cover_metrics_window_post_hide_tail() -> None:
    # N2 invariant: default_window_ms*(1-hq) >= metrics_window_ms so the commit-gate
    # tail observations are fully post-hide. hq default 0.25.
    # 15000*0.75 = 11250 < 30000 -> reject (the exact case the guard exists for).
    with pytest.raises(ValueError):
        ControllerConfig.from_env(
            {"SAFE_SCALE_DEFAULT_WINDOW_MS": "15000", "TRE_METRICS_WINDOW_MS": "30000"}
        )
    # Just below the boundary: 39999*0.75 = 29999.25 < 30000 -> reject.
    with pytest.raises(ValueError):
        ControllerConfig.from_env(
            {"SAFE_SCALE_DEFAULT_WINDOW_MS": "39999", "TRE_METRICS_WINDOW_MS": "30000"}
        )
    # Exact boundary: 40000*0.75 = 30000 >= 30000 -> loads.
    cfg = ControllerConfig.from_env(
        {"SAFE_SCALE_DEFAULT_WINDOW_MS": "40000", "TRE_METRICS_WINDOW_MS": "30000"}
    )
    assert cfg.metrics_window_ms == 30_000
    # Defaults (60000 / hq 0.25 / 30000) load fine.
    ControllerConfig.from_env({})


def test_config_rejects_inverted_safescale_window_bounds() -> None:
    with pytest.raises(ValueError, match="SAFE_SCALE_MIN_WINDOW_MS"):
        ControllerConfig.from_env(
            {
                "SAFE_SCALE_MIN_WINDOW_MS": "300000",
                "SAFE_SCALE_MAX_WINDOW_MS": "15000",
            }
        )


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
