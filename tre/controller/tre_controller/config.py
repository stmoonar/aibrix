from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

SIGNAL_SOURCES = {"zm", "latency_p95", "queue_len", "kv_cache"}
PERCENTILE_MODES = {"bucket_upper", "interpolated"}
WINDOW_MODES = {"tumbling", "sliding"}
METRICS_SCHEMAS = {"v1", "v2"}
INCOMPLETE_POLICIES = {"drop_model", "drop_all"}
_TRUE_VALUES = {"1", "true", "yes", "y", "on"}
_FALSE_VALUES = {"0", "false", "no", "n", "off"}


@dataclass(frozen=True)
class SafeScaleConfig:
    ttft_p95_slo_ms: float = 1200.0
    tpot_p95_slo_ms: float = 100.0
    default_window_ms: float = 60_000.0
    min_window_ms: float = 15_000.0
    max_window_ms: float = 300_000.0
    cw2_fallback_ms: float = 300_000.0
    cdec: float = 2.0
    hq: float = 0.25
    tau_low: float = 1.0
    epsilon_mu: float = 1e-6
    probe_poll_seconds: float = 2.0


@dataclass(frozen=True)
class ControllerConfig:
    redis_url: str
    metrics_redis_url: str
    metrics_schema: str
    service_manager_url: str
    registry_path: str
    runtime_state_dir: str
    monitor_interval_s: float
    metrics_refresh_interval_s: float
    rescue_interval_s: float
    fairness_interval_s: float
    metrics_window_ms: int
    metrics_window_mode: str
    instant_sample_interval_ms: int
    histogram_lookback_ms: int
    min_latency_samples: int
    percentile_mode: str
    signal_source: str
    signal_warmup_ms: int
    paper_stale_max_windows: int
    incomplete_policy: str
    enable_tre_scaling: bool
    ablation_disable_fast_loop: bool
    ablation_disable_safescale: bool
    proactive_release_min_trs: float
    safescale: SafeScaleConfig

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> "ControllerConfig":
        values = os.environ if env is None else env
        repo_tre_dir = Path(__file__).resolve().parents[2]
        default_registry = repo_tre_dir / "deploy" / "registry.yaml"
        default_state_dir = repo_tre_dir / ".runtime"

        percentile_mode = _get_str(values, "TRE_PERCENTILE_MODE", "bucket_upper")
        if percentile_mode not in PERCENTILE_MODES:
            raise ValueError(f"TRE_PERCENTILE_MODE must be one of {sorted(PERCENTILE_MODES)}")

        signal_source = _get_str(values, "TRE_SIGNAL_SOURCE", "zm")
        if signal_source not in SIGNAL_SOURCES:
            raise ValueError(f"TRE_SIGNAL_SOURCE must be one of {sorted(SIGNAL_SOURCES)}")

        metrics_schema = _get_str(values, "TRE_METRICS_SCHEMA", "v2")
        if metrics_schema not in METRICS_SCHEMAS:
            raise ValueError(f"TRE_METRICS_SCHEMA must be one of {sorted(METRICS_SCHEMAS)}")

        incomplete_policy = _get_str(values, "TRE_INCOMPLETE_POLICY", "drop_model")
        if incomplete_policy not in INCOMPLETE_POLICIES:
            raise ValueError(f"TRE_INCOMPLETE_POLICY must be one of {sorted(INCOMPLETE_POLICIES)}")

        metrics_window_mode = _get_str(values, "TRE_METRICS_WINDOW_MODE", "sliding")
        if metrics_window_mode not in WINDOW_MODES:
            raise ValueError(f"TRE_METRICS_WINDOW_MODE must be one of {sorted(WINDOW_MODES)}")

        redis_url = _get_str(values, "TRE_REDIS_URL", "redis://aibrix-redis-master:6379/0")

        safescale = SafeScaleConfig(
            ttft_p95_slo_ms=_get_positive_float(values, "SAFE_SCALE_TTFT_P95_SLO_MS", 1200.0),
            tpot_p95_slo_ms=_get_positive_float(values, "SAFE_SCALE_TPOT_P95_SLO_MS", 100.0),
            default_window_ms=_get_positive_float(values, "SAFE_SCALE_DEFAULT_WINDOW_MS", 60_000.0),
            min_window_ms=_get_positive_float(values, "SAFE_SCALE_MIN_WINDOW_MS", 15_000.0),
            max_window_ms=_get_positive_float(values, "SAFE_SCALE_MAX_WINDOW_MS", 300_000.0),
            cw2_fallback_ms=_get_positive_float(values, "SAFE_SCALE_CW2_FALLBACK_MS", 300_000.0),
            cdec=_get_positive_float(values, "SAFE_SCALE_CDEC", 2.0),
            hq=_get_positive_float(values, "SAFE_SCALE_HQ", 0.25),
            tau_low=_get_positive_float(values, "SAFE_SCALE_TAU_LOW", 1.0),
            epsilon_mu=_get_positive_float(values, "SAFE_SCALE_EPSILON_MU", 1e-6),
            probe_poll_seconds=_get_positive_float(values, "SAFE_SCALE_PROBE_POLL_SECONDS", 2.0),
        )
        if safescale.min_window_ms > safescale.max_window_ms:
            raise ValueError("SAFE_SCALE_MIN_WINDOW_MS must be <= SAFE_SCALE_MAX_WINDOW_MS")

        metrics_window_ms = _get_positive_int(values, "TRE_METRICS_WINDOW_MS", 30_000)
        # N2 invariant (plan 15 §6 N2, architect-ruled): the SafeScale commit gate only
        # inspects the tail (hq fraction) of probe observations. Those tail observations'
        # metrics windows must be fully post-hide, i.e. the probe must run at least one
        # metrics window past the tail start: default_window_ms - tail_span >= metrics_window_ms.
        # Guards a future SAFE_SCALE_DEFAULT_WINDOW_MS being set too short for the metrics
        # window (e.g. 15000 < 30000) from silently diluting the commit gate with pre-hide
        # traffic. (SafeScaleConfig.min_window_ms=15000 is currently DEAD config — never wired
        # to a probe deadline — so it is not guarded here; see 05_paper_vs_impl.md.)
        if safescale.hq < 1.0:
            tail_span_ms = safescale.hq * safescale.default_window_ms
        else:
            tail_span_ms = safescale.hq * safescale.probe_poll_seconds * 1000.0
        if safescale.default_window_ms - tail_span_ms < metrics_window_ms:
            raise ValueError(
                "SAFE_SCALE_DEFAULT_WINDOW_MS minus the commit-gate tail span must be >= "
                "TRE_METRICS_WINDOW_MS so SafeScale probe tail observations are fully post-hide "
                f"(default_window_ms={safescale.default_window_ms}, hq={safescale.hq}, "
                f"metrics_window_ms={metrics_window_ms})"
            )

        return cls(
            redis_url=redis_url,
            metrics_redis_url=_get_str(values, "TRE_METRICS_REDIS_URL", redis_url),
            metrics_schema=metrics_schema,
            service_manager_url=_get_str(
                values,
                "TRE_SERVICE_MANAGER_URL",
                "http://aibrix-tre-service-manager:8000",
            ).rstrip("/"),
            registry_path=_get_str(values, "TRE_REGISTRY_PATH", str(default_registry)),
            runtime_state_dir=_get_str(values, "TRE_RUNTIME_STATE_DIR", str(default_state_dir)),
            monitor_interval_s=_get_positive_float(values, "TRE_MONITOR_INTERVAL_SECONDS", 20.0),
            metrics_refresh_interval_s=_get_positive_float(
                values, "TRE_METRICS_REFRESH_INTERVAL_SECONDS", 5.0
            ),
            rescue_interval_s=_get_positive_float(values, "TRE_RESCUE_INTERVAL_SECONDS", 5.0),
            fairness_interval_s=_get_positive_float(values, "TRE_FAIRNESS_INTERVAL_SECONDS", 10.0),
            metrics_window_ms=metrics_window_ms,
            metrics_window_mode=metrics_window_mode,
            instant_sample_interval_ms=_get_positive_int(values, "TRE_INSTANT_SAMPLE_INTERVAL_MS", 5_000),
            histogram_lookback_ms=_get_nonneg_int(values, "TRE_HIST_BASELINE_LOOKBACK_MS", 90_000),
            min_latency_samples=_get_nonneg_int(values, "TRE_MIN_LATENCY_SAMPLES", 10),
            percentile_mode=percentile_mode,
            signal_source=signal_source,
            # F-onset warmup guard: -1 auto (window fully inside traffic period),
            # 0 disabled (A/B ablation), >0 explicit span-since-onset in ms.
            signal_warmup_ms=int(str(values.get("TRE_SIGNAL_WARMUP_MS", "-1")).strip()),
            paper_stale_max_windows=_get_positive_int(values, "TRE_PAPER_STALE_MAX_WINDOWS", 3),
            incomplete_policy=incomplete_policy,
            enable_tre_scaling=_get_bool(values, "ENABLE_TRE_SCALING", True),
            ablation_disable_fast_loop=_get_bool(values, "TRE_ABLATION_DISABLE_FAST_LOOP", False),
            ablation_disable_safescale=_get_bool(values, "TRE_ABLATION_DISABLE_SAFESCALE", False),
            proactive_release_min_trs=_get_positive_float(values, "PROACTIVE_RELEASE_MIN_TRS", 2000.0),
            safescale=safescale,
        )


def _get_str(env: Mapping[str, str], key: str, default: str) -> str:
    value = env.get(key, default)
    text = str(value).strip()
    if not text:
        raise ValueError(f"{key} must not be empty")
    return text


def _get_bool(env: Mapping[str, str], key: str, default: bool) -> bool:
    raw = env.get(key)
    if raw is None:
        return default
    normalized = str(raw).strip().lower()
    if normalized in _TRUE_VALUES:
        return True
    if normalized in _FALSE_VALUES:
        return False
    raise ValueError(f"{key} must be a boolean value")


def _get_positive_float(env: Mapping[str, str], key: str, default: float) -> float:
    raw = env.get(key, str(default))
    try:
        value = float(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{key} must be a number") from exc
    if value <= 0.0:
        raise ValueError(f"{key} must be positive")
    return value


def _get_positive_int(env: Mapping[str, str], key: str, default: int) -> int:
    raw = env.get(key, str(default))
    try:
        value = int(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{key} must be an integer") from exc
    if value <= 0:
        raise ValueError(f"{key} must be positive")
    return value


def _get_nonneg_int(env: Mapping[str, str], key: str, default: int) -> int:
    raw = env.get(key, str(default))
    try:
        value = int(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{key} must be an integer") from exc
    if value < 0:
        raise ValueError(f"{key} must be non-negative")
    return value
