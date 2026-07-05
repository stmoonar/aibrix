from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PodWindowMetrics:
    pod: str
    prompt_tokens: float | None
    generation_tokens: float | None
    avg_waiting: float
    avg_running: float
    avg_swapping: float
    kv_cache_hit_rate: float
    ttft_p95_ms: float | None
    tpot_p95_ms: float | None
    e2e_p95_ms: float | None


@dataclass(frozen=True)
class ModelWindowMetrics:
    model: str
    window_start_ms: int
    window_end_ms: int
    prompt_tokens: float | None
    generation_tokens: float | None
    avg_waiting: float
    avg_running: float
    avg_swapping: float
    kv_cache_hit_rate: float
    ttft_p95_ms: float | None
    tpot_p95_ms: float | None
    e2e_p95_ms: float | None
    routable_pods: int
    assigned_replicas: int
    per_pod: dict[str, PodWindowMetrics]


@dataclass(frozen=True)
class MetricsSnapshot:
    ts_ms: int
    models: dict[str, ModelWindowMetrics]
    stale: bool
