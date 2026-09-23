"""Per-model gateway error counters for the SafeScale donor-health guard (A13).

The isolated tre-v2 Envoy proxy routes each model through its own cluster,
``httproute/<route namespace>/<model>-router/rule/<n>`` (one HTTPRoute per model, see
deploy/gen_model_manifests.py), and exposes Prometheus stats on its admin/metrics port
(19001, ``/stats/prometheus``). Per cluster we read the cumulative counters

* ``envoy_cluster_upstream_rq_xx{envoy_response_code_class=...}`` - responses by class
  (upstream replies and the router's own upstream-failure replies, e.g. 503 reset / 504
  timeout);
* ``envoy_cluster_upstream_rq_pending_overflow`` - requests shed by the circuit breaker
  (maxConnections / maxPendingRequests; Envoy answers 503 locally);
* ``envoy_cluster_upstream_cx_none_healthy`` - requests with no healthy endpoint.

``errors`` = 5xx + overflow + none-healthy, ``requests`` = all classes + overflow +
none-healthy. The guard compares the deltas since the probe started.

Counting rule and blind spots (review P2-c):

* Requests WITHOUT a response code are in neither the numerator nor the denominator:
  ``upstream_rq_total - sum(upstream_rq_xx)`` (about 1 % on the live proxy) is mostly
  requests still in flight - long streaming generations - plus streams reset before any
  header. Counting that gap as errors would turn in-flight load into false errors, so
  it is left out; the cost is that a stream reset before headers is invisible here.
* Rejections the gateway plugin (ext_proc) answers itself never reach a cluster and are
  not counted (Envoy only has them per listener, not per model).
* A 200 whose stream later breaks (mid-body reset) counts as a success.
"""
from __future__ import annotations

import logging
import re
import time
import urllib.request
from dataclasses import dataclass
from typing import Callable, Iterable, Mapping

LOG = logging.getLogger("tre_controller.gateway_health")

_LINE = re.compile(r"^(?P<name>[a-zA-Z_:][a-zA-Z0-9_:]*)\{(?P<labels>[^}]*)\}\s+(?P<value>\S+)")
_LABEL = re.compile(r'(\w+)="((?:[^"\\]|\\.)*)"')
_RQ_XX = "envoy_cluster_upstream_rq_xx"
_OVERFLOW = "envoy_cluster_upstream_rq_pending_overflow"
_NONE_HEALTHY = "envoy_cluster_upstream_cx_none_healthy"


@dataclass(frozen=True)
class GatewayCounters:
    """Cumulative per-model gateway counters (monotonic until an Envoy restart)."""

    requests: float
    errors: float


def model_cluster_prefix(model: str, *, route_namespace: str) -> str:
    """Envoy cluster-name prefix of a model's HTTPRoute (every rule of the route)."""
    dns = re.sub(r"[^a-z0-9-]+", "-", model.lower()).strip("-")
    return f"httproute/{route_namespace}/{dns}-router/rule/"


def parse_envoy_cluster_counters(
    text: str,
    models: Iterable[str],
    *,
    route_namespace: str,
) -> dict[str, GatewayCounters]:
    prefixes = {model: model_cluster_prefix(model, route_namespace=route_namespace) for model in models}
    requests = {model: 0.0 for model in prefixes}
    errors = {model: 0.0 for model in prefixes}
    seen: set[str] = set()
    for line in text.splitlines():
        if not line.startswith("envoy_cluster_"):
            continue
        match = _LINE.match(line)
        if match is None:
            continue
        name = match.group("name")
        if name not in (_RQ_XX, _OVERFLOW, _NONE_HEALTHY):
            continue
        labels = dict(_LABEL.findall(match.group("labels")))
        cluster = labels.get("envoy_cluster_name", "")
        model = next((m for m, prefix in prefixes.items() if cluster.startswith(prefix)), None)
        if model is None:
            continue
        try:
            value = float(match.group("value"))
        except ValueError:
            continue
        seen.add(model)
        requests[model] += value
        if name != _RQ_XX or labels.get("envoy_response_code_class") == "5":
            errors[model] += value
    return {model: GatewayCounters(requests=requests[model], errors=errors[model]) for model in seen}


def _http_get(url: str, timeout_s: float) -> str:
    with urllib.request.urlopen(url, timeout=timeout_s) as response:  # noqa: S310 - cluster-internal
        return response.read().decode("utf-8", errors="replace")


class EnvoyStatsSource:
    """Reads per-model counters from one or more Envoy stats endpoints (summed).

    ``read()`` returns None when any endpoint fails - a partial sum would look like a
    counter drop - so the guard then fails open (no donor-health verdict, as in v1 which
    had no such guard) and the probe is judged by the other checks.
    """

    def __init__(
        self,
        urls: Iterable[str],
        models: Iterable[str],
        *,
        route_namespace: str = "tre-v2",
        timeout_s: float = 1.0,
        fetch: Callable[[str, float], str] | None = None,
        clock: Callable[[], float] | None = None,
        warn_interval_s: float = 60.0,
    ) -> None:
        self.urls = tuple(url for url in urls if url)
        self.models = tuple(models)
        self.route_namespace = route_namespace
        self.timeout_s = timeout_s
        self._fetch = fetch or _http_get
        self._clock = clock or time.monotonic
        # P3: one "unavailable" warning per endpoint per warn_interval_s (the source is
        # polled every 2 s while probes run; an outage must not flood the log).
        self.warn_interval_s = warn_interval_s
        self._last_warn: dict[str, float] = {}
        self.suppressed_warnings = 0

    def read(self) -> dict[str, GatewayCounters] | None:
        totals: dict[str, list[float]] = {}
        for url in self.urls:
            try:
                text = self._fetch(url, self.timeout_s)
            except Exception as exc:  # noqa: BLE001 - any transport error fails open
                self._warn_unavailable(url, exc)
                return None
            for model, counters in parse_envoy_cluster_counters(
                text, self.models, route_namespace=self.route_namespace
            ).items():
                bucket = totals.setdefault(model, [0.0, 0.0])
                bucket[0] += counters.requests
                bucket[1] += counters.errors
        return {model: GatewayCounters(requests=value[0], errors=value[1]) for model, value in totals.items()}

    def _warn_unavailable(self, url: str, exc: Exception) -> None:
        now = self._clock()
        last = self._last_warn.get(url)
        if last is not None and now - last < self.warn_interval_s:
            self.suppressed_warnings += 1
            return
        self._last_warn[url] = now
        LOG.warning(
            "gateway_stats_unavailable (donor-health guard fails open): %s: %s (%d repeats suppressed)",
            url,
            exc,
            self.suppressed_warnings,
        )
        self.suppressed_warnings = 0


def counters_for(counters: Mapping[str, GatewayCounters] | None, model: str) -> GatewayCounters | None:
    if counters is None:
        return None
    return counters.get(model)
