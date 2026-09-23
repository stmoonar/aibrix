"""A13 donor-health source: per-model Envoy cluster counters of the tre-v2 gateway."""
from __future__ import annotations

from tre_controller.gateway_health import (
    EnvoyStatsSource,
    GatewayCounters,
    model_cluster_prefix,
    parse_envoy_cluster_counters,
)

# Shape of the live tre-v2 proxy's /stats/prometheus (2026-09-24), trimmed.
SAMPLE = """
# TYPE envoy_cluster_upstream_rq_xx counter
envoy_cluster_upstream_rq_xx{envoy_response_code_class="2",envoy_cluster_name="httproute/tre-v2/dsqwen-7b-router/rule/0"} 631366
envoy_cluster_upstream_rq_xx{envoy_response_code_class="5",envoy_cluster_name="httproute/tre-v2/dsqwen-7b-router/rule/0"} 30
envoy_cluster_upstream_rq_xx{envoy_response_code_class="4",envoy_cluster_name="httproute/tre-v2/dsqwen-7b-router/rule/0"} 4
envoy_cluster_upstream_rq_xx{envoy_response_code_class="2",envoy_cluster_name="httproute/tre-v2/dsllama-8b-router/rule/0"} 450142
envoy_cluster_upstream_rq_xx{envoy_response_code_class="5",envoy_cluster_name="httproute/tre-v2/dsllama-8b-router/rule/0"} 28
envoy_cluster_upstream_rq_xx{envoy_response_code_class="5",envoy_cluster_name="xds_cluster"} 2
envoy_cluster_upstream_rq{envoy_response_code="503",envoy_cluster_name="httproute/tre-v2/dsqwen-7b-router/rule/0"} 30
envoy_cluster_upstream_rq_pending_overflow{envoy_cluster_name="httproute/tre-v2/dsqwen-7b-router/rule/0"} 80
envoy_cluster_upstream_rq_pending_overflow{envoy_cluster_name="httproute/tre-v2/dsllama-8b-router/rule/0"} 0
envoy_cluster_upstream_cx_none_healthy{envoy_cluster_name="httproute/tre-v2/dsqwen-7b-router/rule/0"} 1
envoy_cluster_upstream_rq_total{envoy_cluster_name="httproute/tre-v2/dsqwen-7b-router/rule/0"} 637510
envoy_cluster_upstream_rq_xx{envoy_response_code_class="5",envoy_cluster_name="httproute/aibrix-system/dsqwen-7b-router/rule/0"} 999
"""


def test_cluster_prefix_follows_the_generated_httproute_name() -> None:
    assert model_cluster_prefix("dsqwen-7b", route_namespace="tre-v2") == "httproute/tre-v2/dsqwen-7b-router/rule/"


def test_parse_counts_5xx_overflow_and_no_healthy_upstream_per_model() -> None:
    counters = parse_envoy_cluster_counters(SAMPLE, ["dsqwen-7b", "dsllama-8b", "dsqwen-14b"], route_namespace="tre-v2")

    # requests = every response class + circuit-breaker overflow + no-healthy-upstream;
    # errors = 5xx + overflow + no-healthy. Other namespaces' routes and xds are ignored,
    # and the per-code upstream_rq / upstream_rq_total series are not double counted.
    assert counters["dsqwen-7b"] == GatewayCounters(requests=631366 + 30 + 4 + 80 + 1, errors=30 + 80 + 1)
    assert counters["dsllama-8b"] == GatewayCounters(requests=450142 + 28, errors=28)
    assert "dsqwen-14b" not in counters  # no series -> unknown, not zero


def test_source_sums_endpoints_and_fails_open_on_any_error() -> None:
    pages = {"http://a/stats": SAMPLE, "http://b/stats": SAMPLE}
    source = EnvoyStatsSource(pages, ["dsqwen-7b"], fetch=lambda url, timeout: pages[url])
    assert source.read() == {"dsqwen-7b": GatewayCounters(requests=2 * 631481, errors=2 * 111)}

    def broken(url, timeout):
        if url.endswith("b/stats"):
            raise OSError("connection refused")
        return SAMPLE

    assert EnvoyStatsSource(pages, ["dsqwen-7b"], fetch=broken).read() is None
