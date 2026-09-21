"""The two experiment arms must admit traffic under identical rules.

An A/B result only means something if both arms shed at the same point. The
APA arm (NodePort 31592, aibrix-system) and the TRE arm (NodePort 31094,
tre-v2) each get their circuit breakers from their own file, so the two can
drift apart silently -- and did: the TRE arm was raised to 4096/1024 while the
APA arm sat at 256/64, which would have made "TRE served more" indistinguishable
from "APA was shed sooner".
"""
from __future__ import annotations

from pathlib import Path

import yaml

HARDENING = Path(__file__).resolve().parents[1] / "gateway-hardening"
APA_ARM = HARDENING / "backendtrafficpolicy-aibrix-system.yaml"
TRE_ARM = HARDENING / "backendtrafficpolicy-tre-v2.yaml"

MODELS = {"dsqwen-7b", "dsllama-8b", "dsqwen-14b"}


def _circuit_breakers(path: Path) -> dict[str, dict[str, int]]:
    docs = [d for d in yaml.safe_load_all(path.read_text(encoding="utf-8")) if d]
    return {
        doc["metadata"]["name"]: doc["spec"]["circuitBreaker"]
        for doc in docs
        if doc.get("kind") == "BackendTrafficPolicy"
    }


def test_both_arms_declare_a_breaker_for_every_model() -> None:
    for path in (APA_ARM, TRE_ARM):
        breakers = _circuit_breakers(path)
        covered = {name.removesuffix("-circuitbreaker") for name in breakers}
        assert covered == MODELS, f"{path.name} covers {covered}, expected {MODELS}"


def test_the_two_arms_admit_traffic_under_identical_limits() -> None:
    apa = _circuit_breakers(APA_ARM)
    tre = _circuit_breakers(TRE_ARM)

    assert apa.keys() == tre.keys()
    for name in sorted(apa):
        assert apa[name] == tre[name], (
            f"{name} differs between arms: "
            f"aibrix-system={apa[name]} tre-v2={tre[name]}. "
            "A/B comparisons are invalid unless both arms shed at the same point."
        )


def test_every_model_carries_the_same_limits_within_an_arm() -> None:
    for path in (APA_ARM, TRE_ARM):
        values = {tuple(sorted(cb.items())) for cb in _circuit_breakers(path).values()}
        assert len(values) == 1, f"{path.name} has per-model differences: {values}"


def test_limits_are_above_the_per_replica_engine_cap() -> None:
    """The gateway must not be the admission controller.

    vLLM runs with --max-num-seqs 256 per replica, which grows with replica
    count; an Envoy cluster cap does not. Keeping the gateway ceiling well
    above a single replica's cap is what makes scaling out actually raise
    capacity -- the bug that made the previous theta_m fit meaningless.
    """
    engine_cap_per_replica = 256
    for path in (APA_ARM, TRE_ARM):
        for name, cb in _circuit_breakers(path).items():
            assert cb["maxParallelRequests"] > engine_cap_per_replica, (
                f"{path.name}:{name} caps parallel requests at "
                f"{cb['maxParallelRequests']}, at or below one replica's "
                f"engine cap ({engine_cap_per_replica})"
            )
