from __future__ import annotations

import logging
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from tre_common.rediskeys import SCRAPE_INTERVAL_MS
from tre_common.registry import EXPECTED_SIGNAL_DIRECTIONS, load_registry
from tre_controller.loops.metrics_task import REFRESH_MODES
from tre_controller.signals.trs import DWELL_STATES

LOG = logging.getLogger(__name__)

SIGNAL_SOURCES = {
    "zm",
    "latency_p95",
    "queue_len",
    "decode_tps",
    "prefill_tps",
    "kv_cache",
}
PERCENTILE_MODES = {"bucket_upper", "interpolated"}
WINDOW_MODES = {"tumbling", "sliding"}
METRICS_SCHEMAS = {"v1", "v2"}
INCOMPLETE_POLICIES = {"drop_model", "drop_all"}
GATEWAY_INTERVAL_CHECKS = {"fail", "warn", "off"}
_TRUE_VALUES = {"1", "true", "yes", "y", "on"}
_FALSE_VALUES = {"0", "false", "no", "n", "off"}


@dataclass(frozen=True)
class SafeScaleConfig:
    ttft_p95_slo_ms: float = 500.0
    tpot_p95_slo_ms: float = 75.0
    # A6: probe window W = min(max(e2e_multiplier * p95_e2e, min_window_ms), window_ceiling_ms).
    # When p95_e2e is missing W = min_window_ms (no avg_ttft fallback). The 20 s floor is
    # short of the N2 invariant (see from_env): tail observations of a short probe read
    # metrics windows that partly precede the hide, as in v1 (floor 15 s); every probe
    # records how much (tail_pre_hide_fraction_*). Env: SAFE_SCALE_WINDOW_FLOOR_MS (the
    # legacy SAFE_SCALE_MIN_WINDOW_MS is read only when the new name is absent, see
    # _safescale_window_floor_ms).
    min_window_ms: float = 20_000.0
    e2e_multiplier: float = 2.0
    # P3-11: W ceiling = 2 x registry gateway.route_timeout_s (ms; no request - so no
    # p95_e2e - outlives the route timeout, so a longer window only delays the commit).
    # Set by from_env from the registry (the registry's own default route timeout, 150 s,
    # applies when the key is absent -> 300 s, v1's cap). None = no ceiling: only when
    # the registry could not be read (a warning is logged) or in direct constructions.
    # Never below min_window_ms (calc_probe_window_details raises it to the floor).
    window_ceiling_ms: float | None = None
    hq: float = 0.25
    tau_low: float = 1.0
    probe_poll_seconds: float = 2.0
    # A12 (v1 _tail_summary_allows_commit): the commit gate rejects when the tail's max
    # avg KV-cache fill of the donor's remaining serving pods exceeds this (v1: 0.8).
    kv_cache_max: float = 0.8
    # A13 donor-health guard: roll a probe back as soon as the donor model's gateway error
    # ratio since the probe started (Envoy 5xx + circuit-breaker overflow + no-healthy-
    # upstream over all its requests) exceeds donor_error_rate_max, once at least
    # donor_min_requests requests were seen. Needs TRE_GATEWAY_STATS_URL (else fail-open).
    donor_error_rate_max: float = 0.01
    donor_min_requests: float = 20.0
    # A13 rollback backoff: after a probe of a model rolls back, no receiver-less HIGH
    # proactive probe of that model for this long (v1 had no cooldown for demand-driven
    # donor releases, so those are not held). 0 disables.
    rollback_backoff_ms: float = 60_000.0
    # B8: max age (ms) of a probe's commit evidence at the commit's FIRST dispatch. A commit
    # decided (probe marked ``committing``) longer ago than this - held in observe mode,
    # queued behind a long action, or re-submitted after a controller restart - is not run
    # on that stale evidence: the ActionQueue turns it into the donor unhide (rollback,
    # reason ``commit_evidence_stale``). Retries of a commit that already started are
    # exempt. 120 s is independent of the probe window W (the age counts from the
    # decision, after the window): it bounds how stale the evidence a commit runs on may be. TRE_SAFESCALE_COMMIT_MAX_AGE_MS
    # (0 disables).
    commit_max_age_ms: float = 120_000.0


@dataclass(frozen=True)
class ControllerConfig:
    redis_url: str
    metrics_redis_url: str
    metrics_schema: str
    service_manager_url: str
    #: Timeout of slow SM calls; None = registry service_manager.api_call_timeout_s.
    #: Either way it must exceed the worst-case sleeping SM call (checked at start).
    sm_slow_timeout_s: float | None
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
    signal_idle_rps_eps: float
    signal_warmup_ms: int
    paper_stale_max_windows: int
    incomplete_policy: str
    enable_tre_scaling: bool
    ablation_disable_fast_loop: bool
    ablation_disable_safescale: bool
    disable_eta_gate: bool
    orphan_scan_enabled: bool
    orphan_grace_s: float
    # t1: suppress the receiver-less proactive scale-down probe on hot (HIGH) donors
    # (planner high_proactive_safescale). Default False (v1/paper alignment A2: the v1
    # paper_high_proactive_shrink path is live, HIGH models shrink through SafeScale);
    # TRE_SAFESCALE_SUPPRESS_HOT_PROACTIVE=1 re-enables the t1 guard.
    safescale_suppress_hot_proactive: bool
    proactive_release_min_trs: float
    # Review F4: per-model action cooldown (hold a model's next action until a metrics
    # window starting after its last executed action). TRE_ACTION_COOLDOWN=0 disables.
    action_cooldown: bool
    # Opt-in control-loop profiling (research toggle, off by default). When
    # profile_enabled is False the profiler object is None everywhere (zero overhead).
    profile_enabled: bool
    profile_stream_maxlen: int
    profile_proc_sample_interval_s: float
    profile_flush_interval_s: float
    safescale: SafeScaleConfig
    # --- D8 (plan §6.9i): phase-aligned sampler, band dwell, gateway cadence check ---
    # Defaults keep direct constructions (tests) working; from_env sets them all.
    # TRE_METRICS_REFRESH_MODE: phase_aligned (default) | free_running (old loop).
    metrics_refresh_mode: str = "phase_aligned"
    # TRE_METRICS_PHASE_OFFSET_MS: read at boundary + offset (a floor, see adapt).
    metrics_phase_offset_ms: int = 2_000
    # TRE_METRICS_PHASE_ADAPT: raise the offset to when the boundary tick actually
    # appears (the gateway ticker phase), re-learned every 60 cycles.
    metrics_phase_adapt: bool = True
    # TRE_METRICS_PHASE_RETRY_MS: re-read period while the boundary tick is missing.
    metrics_phase_retry_ms: int = 500
    # TRE_METRICS_STALE_HOLD_WINDOWS: stale windows during which the previous snapshot
    # keeps being served before it is marked stale (decision loops then hold).
    metrics_stale_hold_windows: int = 2
    # TRE_DWELL_WINDOWS (1 = off) / TRE_DWELL_STATES (subset of critical,low,high).
    # Default 1 = off (v1/paper alignment A5: neither the paper nor v1 has a band dwell;
    # a band acts on the first window that shows it). >= 2 re-enables the D8 dwell.
    dwell_windows: int = 1
    dwell_states: tuple[str, ...] = ("critical", "low", "high")
    # TRE_GATEWAY_INTERVAL_CHECK: fail (default) | warn | off.
    gateway_interval_check: str = "fail"
    # A13 donor-health guard source: Envoy /stats/prometheus URL(s) of the tre-v2 gateway
    # proxy (TRE_GATEWAY_STATS_URL, comma-separated; empty = guard off / fail-open),
    # the HTTPRoute namespace naming its per-model clusters (TRE_GATEWAY_ROUTE_NAMESPACE)
    # and the scrape timeout (TRE_GATEWAY_STATS_TIMEOUT_SECONDS).
    gateway_stats_urls: tuple[str, ...] = ()
    gateway_route_namespace: str = "tre-v2"
    gateway_stats_timeout_s: float = 1.0
    # Review 2 P1-2 / P2-5: a one-shot action (SafeScale commit / rollback) that fails
    # retriably (SM 409 / 503 / timeout) is retried with exponential backoff
    # TRE_ONESHOT_RETRY_BASE_SECONDS * 2^n capped at TRE_ONESHOT_RETRY_MAX_SECONDS, at
    # most TRE_ONESHOT_RETRY_MAX_ATTEMPTS attempts in total, then abandoned (logged).
    oneshot_retry_max_attempts: int = 6
    oneshot_retry_base_s: float = 2.0
    oneshot_retry_max_s: float = 30.0
    # P2-6: after the SM refused a hide / sleep of a donor with 409 floor_violation, the
    # planner does not pick that donor for any scale-down for this many fast-loop ticks
    # (TRE_FLOOR_VIOLATION_COOLDOWN_TICKS; held for ticks * rescue_interval_s on the
    # ActionQueue clock; 6 x 5 s = 30 s by default; 0 = off). Without it the fast loop
    # re-plans the same urgent donor every tick against a stale view (livelock).
    floor_violation_cooldown_ticks: int = 6

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> "ControllerConfig":
        values = os.environ if env is None else env
        repo_tre_dir = Path(__file__).resolve().parents[2]
        default_registry = repo_tre_dir / "deploy" / "registry.yaml"
        default_state_dir = repo_tre_dir / ".runtime"
        registry_path = _get_str(values, "TRE_REGISTRY_PATH", str(default_registry))

        percentile_mode = _get_str(values, "TRE_PERCENTILE_MODE", "bucket_upper")
        if percentile_mode not in PERCENTILE_MODES:
            raise ValueError(f"TRE_PERCENTILE_MODE must be one of {sorted(PERCENTILE_MODES)}")

        signal_source = _get_str(values, "TRE_SIGNAL_SOURCE", "zm")
        if signal_source not in SIGNAL_SOURCES:
            raise ValueError(f"TRE_SIGNAL_SOURCE must be one of {sorted(SIGNAL_SOURCES)}")
        _validate_signal_thresholds(registry_path, signal_source)

        metrics_schema = _get_str(values, "TRE_METRICS_SCHEMA", "v2")
        if metrics_schema not in METRICS_SCHEMAS:
            raise ValueError(f"TRE_METRICS_SCHEMA must be one of {sorted(METRICS_SCHEMAS)}")

        incomplete_policy = _get_str(values, "TRE_INCOMPLETE_POLICY", "drop_model")
        if incomplete_policy not in INCOMPLETE_POLICIES:
            raise ValueError(f"TRE_INCOMPLETE_POLICY must be one of {sorted(INCOMPLETE_POLICIES)}")

        metrics_window_mode = _get_str(values, "TRE_METRICS_WINDOW_MODE", "sliding")
        if metrics_window_mode not in WINDOW_MODES:
            raise ValueError(f"TRE_METRICS_WINDOW_MODE must be one of {sorted(WINDOW_MODES)}")

        try:
            signal_warmup_ms = int(str(values.get("TRE_SIGNAL_WARMUP_MS", "-1")).strip())
        except (TypeError, ValueError) as exc:
            raise ValueError("TRE_SIGNAL_WARMUP_MS must be an integer (-1 auto, 0 off, >0 ms)") from exc

        redis_url = _get_str(values, "TRE_REDIS_URL", "redis://aibrix-redis-master:6379/0")

        metrics_refresh_mode = _get_str(values, "TRE_METRICS_REFRESH_MODE", "phase_aligned")
        if metrics_refresh_mode not in REFRESH_MODES:
            raise ValueError(f"TRE_METRICS_REFRESH_MODE must be one of {sorted(REFRESH_MODES)}")
        gateway_interval_check = _get_str(values, "TRE_GATEWAY_INTERVAL_CHECK", "fail")
        if gateway_interval_check not in GATEWAY_INTERVAL_CHECKS:
            raise ValueError(
                f"TRE_GATEWAY_INTERVAL_CHECK must be one of {sorted(GATEWAY_INTERVAL_CHECKS)}"
            )
        dwell_states = tuple(
            state.strip().lower()
            for state in _get_str(values, "TRE_DWELL_STATES", "critical,low,high").split(",")
            if state.strip()
        )
        unknown_states = set(dwell_states) - set(DWELL_STATES)
        if unknown_states:
            raise ValueError(f"TRE_DWELL_STATES must be a subset of {list(DWELL_STATES)}")
        instant_sample_interval_ms = _get_positive_int(
            values, "TRE_INSTANT_SAMPLE_INTERVAL_MS", SCRAPE_INTERVAL_MS
        )
        metrics_phase_offset_ms = _get_nonneg_int(values, "TRE_METRICS_PHASE_OFFSET_MS", 2_000)
        if metrics_phase_offset_ms >= instant_sample_interval_ms:
            raise ValueError("TRE_METRICS_PHASE_OFFSET_MS must be below the gateway period")

        safescale = SafeScaleConfig(
            ttft_p95_slo_ms=_get_positive_float(values, "SAFE_SCALE_TTFT_P95_SLO_MS", 500.0),
            tpot_p95_slo_ms=_get_positive_float(values, "SAFE_SCALE_TPOT_P95_SLO_MS", 75.0),
            min_window_ms=_safescale_window_floor_ms(values),
            e2e_multiplier=_get_positive_float(values, "SAFE_SCALE_E2E_MULTIPLIER", 2.0),
            window_ceiling_ms=_safescale_window_ceiling_ms(registry_path),
            hq=_get_positive_float(values, "SAFE_SCALE_HQ", 0.25),
            tau_low=_get_positive_float(values, "SAFE_SCALE_TAU_LOW", 1.0),
            probe_poll_seconds=_get_positive_float(values, "SAFE_SCALE_PROBE_POLL_SECONDS", 2.0),
            kv_cache_max=_get_positive_float(values, "SAFE_SCALE_KV_CACHE_MAX", 0.8),
            donor_error_rate_max=_get_positive_float(values, "TRE_SAFESCALE_DONOR_ERROR_RATE_MAX", 0.01),
            donor_min_requests=_get_positive_float(values, "TRE_SAFESCALE_DONOR_MIN_REQUESTS", 20.0),
            rollback_backoff_ms=_get_nonneg_float(values, "TRE_SAFESCALE_ROLLBACK_BACKOFF_MS", 60_000.0),
            commit_max_age_ms=_get_nonneg_float(
                values, "TRE_SAFESCALE_COMMIT_MAX_AGE_MS", SafeScaleConfig.commit_max_age_ms
            ),
        )

        metrics_window_ms = _get_positive_int(values, "TRE_METRICS_WINDOW_MS", 30_000)
        # phase_aligned needs metrics_window_ms to be a multiple of the gateway period;
        # metrics_task falls back to free_running (with an error log) when it is not.
        # N2 invariant (plan 15 §6 N2): the SafeScale commit gate only inspects the tail (hq
        # fraction) of the probe observations; a tail observation starts at W*(1-hq) after
        # the hide and reads a metrics window ending up to one refresh period + the read
        # offset earlier and spanning metrics_window_ms. Fully post-hide tails would need
        # W_floor*(1-hq) >= metrics_window + refresh + offset (30 + 10 + 2 s today). With
        # the 20 s floor this does not hold for short probes, so it is only a warning
        # (v1, floor 15 s, ran the same way): the tail observations of a short probe read
        # windows that partly precede the hide. Checked on the FLOOR min_window_ms.
        if safescale.hq < 1.0:
            tail_span_ms = safescale.hq * safescale.min_window_ms
        else:
            tail_span_ms = safescale.hq * safescale.probe_poll_seconds * 1000.0
        if metrics_refresh_mode == "phase_aligned":
            refresh_ms = float(instant_sample_interval_ms)
            read_offset_ms = float(metrics_phase_offset_ms)
        else:
            refresh_ms = _get_positive_float(values, "TRE_METRICS_REFRESH_INTERVAL_SECONDS", 5.0) * 1000.0
            read_offset_ms = 0.0
        required_ms = metrics_window_ms + refresh_ms + read_offset_ms
        if safescale.min_window_ms - tail_span_ms < required_ms:
            LOG.warning(
                "SAFE_SCALE_MIN_WINDOW_MS minus the commit-gate tail span is below "
                "TRE_METRICS_WINDOW_MS + metrics refresh + read offset: tail observations of "
                "a probe as short as the floor read metrics windows that partly precede the "
                "hide (as in v1, floor 15 s); longer probes (2 x p95_e2e above the floor) are "
                f"unaffected (min_window_ms={safescale.min_window_ms}, hq={safescale.hq}, "
                f"metrics_window_ms={metrics_window_ms}, refresh_ms={refresh_ms}, "
                f"read_offset_ms={read_offset_ms})"
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
            # B1: wake/create + defrag run for minutes inside the SM handler.
            sm_slow_timeout_s=(
                _get_positive_float(values, "TRE_SM_SLOW_TIMEOUT_SECONDS", 300.0)
                if values.get("TRE_SM_SLOW_TIMEOUT_SECONDS") not in (None, "")
                else None
            ),
            registry_path=registry_path,
            runtime_state_dir=_get_str(values, "TRE_RUNTIME_STATE_DIR", str(default_state_dir)),
            monitor_interval_s=_get_positive_float(values, "TRE_MONITOR_INTERVAL_SECONDS", 20.0),
            metrics_refresh_interval_s=_get_positive_float(
                values, "TRE_METRICS_REFRESH_INTERVAL_SECONDS", 5.0
            ),
            rescue_interval_s=_get_positive_float(values, "TRE_RESCUE_INTERVAL_SECONDS", 5.0),
            fairness_interval_s=_get_positive_float(values, "TRE_FAIRNESS_INTERVAL_SECONDS", 10.0),
            metrics_window_ms=metrics_window_ms,
            metrics_window_mode=metrics_window_mode,
            # Must equal the gateway scrape cadence (SCRAPE_INTERVAL_MS): _instant_avg
            # divides the summed in-window instant buckets by expected_samples =
            # window_ms / this. A smaller value inflates expected_samples and HALVES the
            # queue average the controller sees (r3 SMOKE_FINDINGS defect 2). Aligned to
            # the real 10s write cadence; do not re-introduce a 5s magic number.
            instant_sample_interval_ms=instant_sample_interval_ms,
            histogram_lookback_ms=_get_nonneg_int(values, "TRE_HIST_BASELINE_LOOKBACK_MS", 90_000),
            min_latency_samples=_get_nonneg_int(values, "TRE_MIN_LATENCY_SAMPLES", 10),
            percentile_mode=percentile_mode,
            signal_source=signal_source,
            signal_idle_rps_eps=_get_nonneg_float(
                values, "TRE_SIGNAL_IDLE_RPS_EPS", 0.05
            ),
            # F-onset warmup guard: -1 auto (window fully inside traffic period),
            # 0 disabled (A/B ablation), >0 explicit span-since-onset in ms.
            signal_warmup_ms=signal_warmup_ms,
            paper_stale_max_windows=_get_positive_int(values, "TRE_PAPER_STALE_MAX_WINDOWS", 3),
            incomplete_policy=incomplete_policy,
            enable_tre_scaling=_get_bool(values, "ENABLE_TRE_SCALING", True),
            ablation_disable_fast_loop=_get_bool(values, "TRE_ABLATION_DISABLE_FAST_LOOP", False),
            ablation_disable_safescale=_get_bool(values, "TRE_ABLATION_DISABLE_SAFESCALE", False),
            disable_eta_gate=_get_bool(values, "TRE_DISABLE_ETA_GATE", False),
            orphan_scan_enabled=_get_bool(values, "TRE_ORPHAN_SCAN_ENABLED", True),
            orphan_grace_s=_get_positive_float(values, "TRE_ORPHAN_GRACE_S", 600.0),
            safescale_suppress_hot_proactive=_get_bool(
                values, "TRE_SAFESCALE_SUPPRESS_HOT_PROACTIVE", False
            ),
            proactive_release_min_trs=_get_positive_float(values, "PROACTIVE_RELEASE_MIN_TRS", 2000.0),
            action_cooldown=_get_bool(values, "TRE_ACTION_COOLDOWN", True),
            profile_enabled=_get_bool(values, "TRE_PROFILE", False),
            profile_stream_maxlen=_get_positive_int(values, "TRE_PROFILE_STREAM_MAXLEN", 200_000),
            profile_proc_sample_interval_s=_get_positive_float(
                values, "TRE_PROFILE_PROC_SAMPLE_INTERVAL_SECONDS", 5.0
            ),
            profile_flush_interval_s=_get_positive_float(
                values, "TRE_PROFILE_FLUSH_INTERVAL_SECONDS", 1.0
            ),
            safescale=safescale,
            metrics_refresh_mode=metrics_refresh_mode,
            metrics_phase_offset_ms=metrics_phase_offset_ms,
            metrics_phase_adapt=_get_bool(values, "TRE_METRICS_PHASE_ADAPT", True),
            metrics_phase_retry_ms=_get_positive_int(values, "TRE_METRICS_PHASE_RETRY_MS", 500),
            metrics_stale_hold_windows=_get_nonneg_int(values, "TRE_METRICS_STALE_HOLD_WINDOWS", 2),
            dwell_windows=_get_positive_int(values, "TRE_DWELL_WINDOWS", 1),
            dwell_states=dwell_states,
            gateway_interval_check=gateway_interval_check,
            gateway_stats_urls=tuple(
                url.strip() for url in str(values.get("TRE_GATEWAY_STATS_URL", "")).split(",") if url.strip()
            ),
            gateway_route_namespace=_get_str(values, "TRE_GATEWAY_ROUTE_NAMESPACE", "tre-v2"),
            gateway_stats_timeout_s=_get_positive_float(values, "TRE_GATEWAY_STATS_TIMEOUT_SECONDS", 1.0),
            oneshot_retry_max_attempts=_get_positive_int(values, "TRE_ONESHOT_RETRY_MAX_ATTEMPTS", 6),
            oneshot_retry_base_s=_get_positive_float(values, "TRE_ONESHOT_RETRY_BASE_SECONDS", 2.0),
            oneshot_retry_max_s=_get_positive_float(values, "TRE_ONESHOT_RETRY_MAX_SECONDS", 30.0),
            floor_violation_cooldown_ticks=_get_nonneg_int(values, "TRE_FLOOR_VIOLATION_COOLDOWN_TICKS", 6),
        )


#: P2-7 (rollback safety): env names of the SafeScale probe-window floor. The new name
#: is read by this image; the legacy name is what controller images before 2026-09-29
#: read, and their N2 startup guard REJECTS a legacy value below 60 000 (min*(1-hq) >=
#: metrics window + refresh + offset). The overlay therefore sets the new name to the
#: new floor and leaves the legacy name at 60000, so an image rollback alone still
#: starts (with its own 60 s band).
WINDOW_FLOOR_ENV = "SAFE_SCALE_WINDOW_FLOOR_MS"
LEGACY_WINDOW_FLOOR_ENV = "SAFE_SCALE_MIN_WINDOW_MS"
#: Env keys of the pre-2026-09-29 window band that this image no longer reads.
RETIRED_WINDOW_ENVS = (
    "SAFE_SCALE_MAX_WINDOW_MS",
    "SAFE_SCALE_DEFAULT_WINDOW_MS",
    "SAFE_SCALE_CW2_FALLBACK_MS",
    "SAFE_SCALE_CDEC",
    "SAFE_SCALE_EPSILON_MU",
)


def _safescale_window_floor_ms(values: Mapping[str, str]) -> float:
    """The probe-window floor (ms). Precedence:

    1. ``SAFE_SCALE_WINDOW_FLOOR_MS`` set -> it wins; a legacy ``SAFE_SCALE_MIN_WINDOW_MS``
       next to it is ignored (it is kept in the overlay only for older images) - logged
       at INFO when the two differ;
    2. only the legacy name set -> it is used, with a WARNING (an overlay predating the
       rename: the floor it pins, e.g. 60 s, stays in force);
    3. neither -> the default 20 s.
    """
    for key in RETIRED_WINDOW_ENVS:
        if values.get(key) not in (None, ""):
            LOG.warning("%s is set but no longer read by this controller (probe window A6)", key)
    legacy_set = values.get(LEGACY_WINDOW_FLOOR_ENV) not in (None, "")
    if values.get(WINDOW_FLOOR_ENV) not in (None, ""):
        floor = _get_positive_float(values, WINDOW_FLOOR_ENV, SafeScaleConfig.min_window_ms)
        if legacy_set:
            legacy = _get_positive_float(values, LEGACY_WINDOW_FLOOR_ENV, floor)
            if legacy != floor:
                LOG.info(
                    "%s=%s ignored: %s=%s takes precedence (the legacy name is kept for "
                    "controller images before 2026-09-29 only)",
                    LEGACY_WINDOW_FLOOR_ENV, legacy, WINDOW_FLOOR_ENV, floor,
                )
        return floor
    if legacy_set:
        floor = _get_positive_float(values, LEGACY_WINDOW_FLOOR_ENV, SafeScaleConfig.min_window_ms)
        LOG.warning(
            "%s is not set; using the legacy %s=%s as the SafeScale probe-window floor "
            "(rename it to %s)",
            WINDOW_FLOOR_ENV, LEGACY_WINDOW_FLOOR_ENV, floor, WINDOW_FLOOR_ENV,
        )
        return floor
    return SafeScaleConfig.min_window_ms


def _safescale_window_ceiling_ms(registry_path: str) -> float | None:
    """P3-11: W ceiling = 2 x registry ``gateway.route_timeout_s`` (ms). The registry
    parser supplies its default route timeout (150 s) when the key is absent, so the
    ceiling is then 300 s. None (no ceiling, WARNING) only when the registry cannot be
    loaded here or has no usable route timeout - the controller then runs as the
    2026-09-29 commit did (W unbounded above)."""
    try:
        gateway = load_registry(registry_path).gateway()
        timeout_s = float(gateway.route_timeout_s)
    except Exception as exc:  # noqa: BLE001 - the ceiling is a safety bound, not a gate
        LOG.warning("SafeScale probe window has no ceiling: registry gateway.route_timeout_s unreadable (%r)", exc)
        return None
    if not math.isfinite(timeout_s) or timeout_s <= 0:
        LOG.warning("SafeScale probe window has no ceiling: gateway.route_timeout_s=%r is not positive", timeout_s)
        return None
    return 2.0 * timeout_s * 1000.0


def _validate_signal_thresholds(registry_path: str, signal_source: str) -> None:
    expected_direction = EXPECTED_SIGNAL_DIRECTIONS.get(signal_source)
    if expected_direction is None:
        return
    registry = load_registry(registry_path)
    missing = []
    invalid_direction = []
    for model in registry.models():
        threshold = model.alt_thresholds.get(signal_source)
        if threshold is None or not math.isfinite(threshold.theta) or threshold.theta <= 0.0:
            missing.append(model.name)
        elif threshold.direction != expected_direction:
            invalid_direction.append(model.name)
    if missing:
        raise ValueError(
            f"TRE_SIGNAL_SOURCE={signal_source} requires fitted alt_thresholds for every model; "
            f"missing={sorted(missing)}"
        )
    if invalid_direction:
        raise ValueError(
            f"TRE_SIGNAL_SOURCE={signal_source} requires direction={expected_direction}; "
            f"invalid={sorted(invalid_direction)}"
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


def _get_nonneg_float(env: Mapping[str, str], key: str, default: float) -> float:
    raw = env.get(key, str(default))
    try:
        value = float(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{key} must be a number") from exc
    if not math.isfinite(value) or value < 0.0:
        raise ValueError(f"{key} must be non-negative and finite")
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
