"""The vLLM metric names TRE reads, in one table per name space.

vLLM renamed several Prometheus families (deprecated in 0.10.x, removed in 0.11+ and the
0.30 fork): ``gpu_cache_usage_perc`` -> ``kv_cache_usage_perc``,
``time_per_output_token_seconds`` -> ``inter_token_latency_seconds`` (the same per-token
histogram) and ``gpu_prefix_cache_{queries,hits}`` -> ``prefix_cache_{queries,hits}``.
Every reader resolves a quantity through the candidates below, newest name first, so one
code path serves 0.10.1 pods (both names of the gauges, only the old TPOT name) and 0.30
pods (new names only). See ``docs/design/20260928-vllm-030-metric-names.md``.

Two name spaces:

* :data:`VLLM_METRICS` - raw engine names in a pod's ``/metrics`` text (``vllm:...``).
  The Go gateway keeps the same table in ``pkg/metrics/engine_fetcher.go``
  (``engineMetricEquivalents``).
* :data:`GATEWAY_DOC_KEYS` - the metric part of the ``<model>/<metric>`` keys of the
  gateway's Redis docs (``tre:v2:hist:*`` / ``tre:v2:inst:*`` and the legacy
  ``aibrix:pod_*_metrics_*``). These are AIBrix metric identifiers, not engine names:
  the gateway resolves the engine name itself and still writes
  ``time_per_output_token_seconds`` / ``gpu_cache_usage_perc``. Readers also accept the
  new identifiers, so a gateway that switches keys keeps working.
"""
from __future__ import annotations

from typing import Iterable, Mapping, Optional

#: canonical quantity -> raw vLLM family names, newest first. Every entry is required:
#: the guard test (deploy/tests/test_vllm_metric_names.py) resolves each one against a live
#: 0.30 sample and a 0.10.1 sample.
VLLM_METRICS: Mapping[str, tuple[str, ...]] = {
    "num_requests_running": ("vllm:num_requests_running",),
    "num_requests_waiting": ("vllm:num_requests_waiting",),
    "kv_cache_usage_perc": ("vllm:kv_cache_usage_perc", "vllm:gpu_cache_usage_perc"),
    "prefix_cache_queries": (
        "vllm:prefix_cache_queries_total",
        "vllm:gpu_prefix_cache_queries_total",
    ),
    "prefix_cache_hits": ("vllm:prefix_cache_hits_total", "vllm:gpu_prefix_cache_hits_total"),
    "time_to_first_token_seconds": ("vllm:time_to_first_token_seconds",),
    "inter_token_latency_seconds": (
        "vllm:inter_token_latency_seconds",
        "vllm:time_per_output_token_seconds",
    ),
    "e2e_request_latency_seconds": ("vllm:e2e_request_latency_seconds",),
    "request_prompt_tokens": ("vllm:request_prompt_tokens",),
    "request_generation_tokens": ("vllm:request_generation_tokens",),
}

#: Read but optional: the V0-only swap gauge is absent on the V1 engine (genuinely zero).
OPTIONAL_VLLM_METRICS: Mapping[str, tuple[str, ...]] = {
    "num_requests_swapped": ("vllm:num_requests_swapped",),
}

#: Gateway Redis doc metric keys: identifier the gateway writes -> keys a reader accepts,
#: newest first. Identifiers without an entry are read as themselves.
GATEWAY_DOC_KEYS: Mapping[str, tuple[str, ...]] = {
    "time_per_output_token_seconds": (
        "inter_token_latency_seconds",
        "time_per_output_token_seconds",
    ),
    "gpu_cache_usage_perc": ("kv_cache_usage_perc", "gpu_cache_usage_perc"),
}

_HISTOGRAM_SUFFIXES = ("_bucket", "_sum", "_count")


def vllm_candidates(canonical: str) -> tuple[str, ...]:
    """Raw names of ``canonical`` (a :data:`VLLM_METRICS` or optional key), newest first."""
    if canonical in VLLM_METRICS:
        return VLLM_METRICS[canonical]
    if canonical in OPTIONAL_VLLM_METRICS:
        return OPTIONAL_VLLM_METRICS[canonical]
    raise KeyError(f"unknown vLLM metric {canonical!r}")


def doc_key_candidates(metric: str) -> tuple[str, ...]:
    """Doc keys a reader tries for the gateway identifier ``metric``, newest first."""
    return GATEWAY_DOC_KEYS.get(metric, (metric,))


def doc_lookup(metrics: Mapping[str, object], model: str, metric: str) -> Optional[object]:
    """``metrics[f"{model}/{key}"]`` for the first candidate key present, else None."""
    for key in doc_key_candidates(metric):
        full = f"{model}/{key}"
        if full in metrics:
            return metrics[full]
    return None


def sample_name(line: str) -> Optional[str]:
    """Metric name of one Prometheus text sample line (None for comments / blanks)."""
    line = line.strip()
    if not line or line.startswith("#"):
        return None
    return line.split("{", 1)[0].split(" ", 1)[0].strip() or None


def sample_names(text: str) -> set[str]:
    """Every sample name in a Prometheus text body."""
    return {name for name in (sample_name(line) for line in text.splitlines()) if name}


def _present(raw: str, names: set[str]) -> bool:
    return raw in names or any(raw + suffix in names for suffix in _HISTOGRAM_SUFFIXES)


def resolve_vllm_name(canonical: str, names: Iterable[str]) -> Optional[str]:
    """The first raw name of ``canonical`` present in ``names`` (sample names), as a plain
    sample or as a histogram (``_bucket`` / ``_sum`` / ``_count``); None if none is."""
    present = names if isinstance(names, set) else set(names)
    for raw in vllm_candidates(canonical):
        if _present(raw, present):
            return raw
    return None
