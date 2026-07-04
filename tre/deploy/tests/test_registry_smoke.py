from __future__ import annotations

from tre_common.registry import ClusterTopology, ModelSpec, NodeSpec, Registry, SloSpec, TrsParams
from registry_smoke import registry_warnings


def _model(name: str, *, theta: float, ttft: float = 1200.0) -> ModelSpec:
    return ModelSpec(
        name=name,
        weights_path="/weights",
        tp_size=1,
        min_replicas=0,
        max_replicas=2,
        vllm_image="image",
        slo=SloSpec(ttft_p95_ms=ttft, tpot_p95_ms=100.0, e2e_p95_ms=10000.0),
        trs=TrsParams(
            w_p=0.04,
            w_d=1.0,
            lambda_wait=2.625,
            qmin=1.0,
            ema_alpha=0.5,
            theta_m=theta,
            tau_crit=0.8,
            tau_low=1.0,
            tau_high=1.25,
            qsat=4.0,
            epsat=0.05,
            hsat=3,
        ),
    )


def test_registry_warnings_report_zero_theta_and_profile_slo_drift() -> None:
    registry = Registry(
        ClusterTopology(nodes=(NodeSpec(name="node-a", gpus=4, two_gpu_slots=((0, 1), (2, 3))),)),
        [_model("m1", theta=0.0, ttft=1200.0)],
    )
    profiles = {"models": {"m1": {"latency_slo_ms": {"ttft_p95": 500.0, "tpot_p95": 75.0, "e2e_p95": 12000.0}}}}

    warnings = registry_warnings(registry, profiles=profiles)

    assert "WARNING m1.trs.theta_m is 0.0" in warnings
    assert "WARNING m1.slo.ttft_p95_ms differs from profile: 1200.0 != 500.0" in warnings
    assert "WARNING m1.slo.tpot_p95_ms differs from profile: 100.0 != 75.0" in warnings
    assert "WARNING m1.slo.e2e_p95_ms differs from profile: 10000.0 != 12000.0" in warnings


def test_registry_warnings_is_empty_when_parameters_match_profiles() -> None:
    registry = Registry(
        ClusterTopology(nodes=(NodeSpec(name="node-a", gpus=4, two_gpu_slots=((0, 1), (2, 3))),)),
        [_model("m1", theta=100.0, ttft=500.0)],
    )
    profiles = {"models": {"m1": {"latency_slo_ms": {"ttft_p95": 500.0, "tpot_p95": 100.0, "e2e_p95": 10000.0}}}}

    assert registry_warnings(registry, profiles=profiles) == []
