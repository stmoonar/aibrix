from __future__ import annotations

import math

import pytest

from scripts.admission_cap import (
    DEFAULT_CAP_NAME,
    ENGINE_CAPPED,
    GATEWAY_CAPPED,
    VLLM_DEFAULT_MAX_NUM_SEQS,
    AdmissionCap,
    cap_for_registry,
    get_cap,
    max_num_seqs_from_args,
    max_num_seqs_from_registry,
    ramp_backlog_coefficient,
)

#: Measured from each model's vLLM startup log on 2026-09-20.
KV_CACHE_TOKENS = {"dsqwen-7b": 349232, "dsllama-8b": 147536, "dsqwen-14b": 202336}

#: The campaign ramp: linear 0.4 -> 1.2, then a quarter-length hold.
RAMP = (0.4, 1.2, 0.25)


def test_the_superseded_policy_matches_what_the_pre_check_measured() -> None:
    # A pre-check against the live gateway on 2026-09-20 opened requests until they were
    # refused: every request up to in-flight 320 was served and the first 503 arrived at
    # 321. That was 256 parallel + 64 pending, shared across replicas. SUPERSEDED on
    # 2026-09-21 - kept so a capture taken before then can be re-read under the rules it
    # was taken under, and because it is still the evidence that the Envoy limits are per
    # cluster and that a shed carries no header the classifier can key on.
    assert GATEWAY_CAPPED.shed_ceiling == 320
    assert GATEWAY_CAPPED.max_num_seqs is None
    assert GATEWAY_CAPPED.sequence_limit == VLLM_DEFAULT_MAX_NUM_SEQS
    assert GATEWAY_CAPPED.name != DEFAULT_CAP_NAME


def test_the_superseded_policy_put_the_gateway_in_charge_of_admission() -> None:
    # That was the whole calibration problem: the gateway shed below the level at which
    # the engine would have queued, so num_requests_waiting could not be driven by
    # concurrency at all - only by KV exhaustion.
    assert GATEWAY_CAPPED.admission_controller == "gateway"
    assert ENGINE_CAPPED.admission_controller == "engine"


def test_ramp_backlog_coefficient_is_derived_from_the_ramp_shape() -> None:
    # 0.025 accumulated over the part of the ramp above rho=1, plus 0.05 over the hold.
    assert ramp_backlog_coefficient(*RAMP) == pytest.approx(0.075)
    # A ramp that never exceeds rho=1 accumulates no backlog at all.
    assert ramp_backlog_coefficient(0.4, 0.9, 0.25) == 0.0


def test_ramp_seconds_spends_exactly_the_admission_headroom() -> None:
    coefficient = ramp_backlog_coefficient(*RAMP)
    # 256 parallel minus ~110 typical running, rounded down to a whole 10 -> 140.
    assert GATEWAY_CAPPED.ramp_excess_budget == 140
    # Slow enough that the backlog fits the headroom: 0.075 * C_s * T_r == 140.
    assert GATEWAY_CAPPED.ramp_seconds(20.0, coefficient) == pytest.approx(140 / (0.075 * 20.0))
    # ... and clamped, so one cell stays bounded however small C_s is.
    assert GATEWAY_CAPPED.ramp_seconds(0.5, coefficient) == GATEWAY_CAPPED.ramp_max_s


def test_ramp_seconds_refuses_a_ramp_that_never_overloads() -> None:
    with pytest.raises(ValueError, match="no backlog budget"):
        GATEWAY_CAPPED.ramp_seconds(5.0, 0.0)


def test_kv_cache_binds_the_engine_below_its_sequence_limit_today() -> None:
    # dsqwen-7b at shape S1 (256 in + 128 out): 349232 / 384 = 909 concurrent requests
    # fit in the KV cache, which is below the 1024 default sequence limit.
    sizing = GATEWAY_CAPPED.burst_sizing(KV_CACHE_TOKENS["dsqwen-7b"], 384)
    assert sizing.kv_request_limit == 909
    assert sizing.engine_running_limit == 909
    assert sizing.binding_limit == "kv_cache"


def test_a_burst_the_gateway_would_shed_is_declared_unreachable() -> None:
    sizing = GATEWAY_CAPPED.burst_sizing(KV_CACHE_TOKENS["dsqwen-7b"], 384)
    assert sizing.requests_needed == 1364  # ceil(1.5 * 909)
    assert sizing.request_cap == 240
    assert not sizing.reachable
    assert "shed by the gateway instead of queued by the engine" in sizing.reason


def test_a_burst_that_fits_the_budget_is_sized_to_overshoot_the_engine() -> None:
    # dsllama-8b at S3 (2048 in + 96 out): 147536 / 2144 = 68 concurrent, so 102 requests
    # overshoot it and sit far below the 240-request admission budget.
    sizing = GATEWAY_CAPPED.burst_sizing(KV_CACHE_TOKENS["dsllama-8b"], 2144)
    assert sizing.engine_running_limit == 68
    assert sizing.requests == 102
    assert sizing.reachable


def test_raising_the_cap_makes_every_shape_reachable() -> None:
    # With the circuit breaker far above the offered load and an explicit per-pod
    # sequence limit, the engine queues instead of the gateway shedding, so a burst only
    # has to overshoot 256 sequences - which every shape can now afford. This is the
    # deployed policy, not a proposal.
    for tokens in KV_CACHE_TOKENS.values():
        for tokens_per_request in (384, 960, 2144, 704, 1152):
            sizing = ENGINE_CAPPED.burst_sizing(tokens, tokens_per_request)
            assert sizing.reachable, (tokens, tokens_per_request, sizing.reason)
            assert sizing.requests <= 384


def test_the_shed_ceiling_does_not_scale_with_replicas_but_the_engine_limit_does() -> None:
    # The Envoy limits are per cluster, so four replicas hit exactly the same ceiling as
    # one - which is why an autoscaling experiment run under them measures the circuit
    # breaker. The engine's limit is per pod and does scale.
    four = ENGINE_CAPPED.with_replicas(4)
    assert four.shed_ceiling == ENGINE_CAPPED.shed_ceiling
    assert four.fleet_sequence_limit == 4 * ENGINE_CAPPED.sequence_limit


def test_a_custom_policy_derives_its_own_budgets() -> None:
    cap = AdmissionCap(name="probe", max_parallel_requests=1000,
                       max_pending_requests=100, max_num_seqs=64, typical_running=200)
    assert cap.shed_ceiling == 1100
    assert cap.burst_request_cap == 825  # 75 % of the shed ceiling
    assert cap.ramp_excess_budget == 800
    assert cap.burst_sizing(1_000_000, 500).engine_running_limit == 64  # seq-bound


def test_an_unknown_policy_name_is_a_loud_failure() -> None:
    with pytest.raises(SystemExit, match="unknown admission cap"):
        get_cap("whatever")


# ======================================================== the deployed policy (2026-09-21)


def test_the_deployed_policy_is_the_engine_capped_one() -> None:
    # The 256 + 64 ceiling is superseded. Anything generated against it measures a proxy
    # the cluster no longer runs, which is how theta 1718 / 1494 / 1414 came to be
    # invalid by construction.
    assert get_cap(DEFAULT_CAP_NAME) is ENGINE_CAPPED
    assert ENGINE_CAPPED.max_parallel_requests == 4096
    assert ENGINE_CAPPED.max_pending_requests == 1024
    assert ENGINE_CAPPED.shed_ceiling == 5120
    assert ENGINE_CAPPED.sequence_limit == 256


def test_the_admission_ceiling_is_the_engines_and_grows_with_replicas() -> None:
    # The whole reason the policy moved: the Envoy limits are per cluster and do not
    # grow when TRE scales out, so an autoscaling experiment run under them measures the
    # circuit breaker. max_num_seqs * replicas does grow.
    assert ENGINE_CAPPED.admission_controller == "engine"
    for replicas in (1, 2, 4):
        cap = ENGINE_CAPPED.with_replicas(replicas)
        assert cap.fleet_sequence_limit == 256 * replicas
        assert cap.shed_ceiling == ENGINE_CAPPED.shed_ceiling


def test_burst_reachability_is_judged_against_the_engine_not_a_fixed_in_flight() -> None:
    # Old rule: compare the overshoot against 320 admitted requests. New rule: against
    # min(max_num_seqs, KV capacity) * replicas, which is what actually queues.
    for model, tokens in KV_CACHE_TOKENS.items():
        sizing = ENGINE_CAPPED.burst_sizing(tokens, 384)
        assert sizing.reachable, (model, sizing.reason)
        assert sizing.engine_running_limit == min(256, sizing.kv_request_limit)
        assert sizing.requests_needed == math.ceil(1.5 * sizing.engine_running_limit)
        assert sizing.requests_needed <= ENGINE_CAPPED.burst_request_cap == 3840


def test_the_burst_a_shape_needs_scales_with_the_replica_count() -> None:
    two = ENGINE_CAPPED.with_replicas(2)
    one = ENGINE_CAPPED.burst_sizing(KV_CACHE_TOKENS["dsllama-8b"], 2144)
    both = two.burst_sizing(KV_CACHE_TOKENS["dsllama-8b"], 2144)
    assert both.engine_running_limit == 2 * one.engine_running_limit
    assert both.requests_needed > one.requests_needed


def test_every_shape_the_old_ceiling_excluded_is_reachable_now() -> None:
    # Directly answers "did raising the cap make the skipped shapes viable": yes, all of
    # them, which is what makes num_requests_waiting - and lambda_wait - identifiable.
    tokens_per_request = (384, 960, 2144, 704, 1152, 1712, 1226.7, 953.6)
    for tokens in KV_CACHE_TOKENS.values():
        for tpr in tokens_per_request:
            old = GATEWAY_CAPPED.burst_sizing(tokens, tpr)
            new = ENGINE_CAPPED.burst_sizing(tokens, tpr)
            assert new.reachable, (tokens, tpr, new.reason)
            if not old.reachable:
                assert new.requests_needed <= new.request_cap


# ---------------------------------------------------------------- registry sourcing


REGISTRY = {
    "models": [
        {"name": "dsqwen-7b", "vllm_extra_args": ["--no-enable-prefix-caching",
                                                  "--max-num-seqs", "256"]},
        {"name": "dsllama-8b", "vllm_extra_args": ["--max-num-seqs=256"]},
    ]
}


def test_the_sequence_limit_is_read_from_the_registry_not_assumed() -> None:
    assert max_num_seqs_from_args(["--max-num-seqs", "256"]) == 256
    assert max_num_seqs_from_args(["--max-num-seqs=128"]) == 128
    assert max_num_seqs_from_args(["--swap-space", "16"]) is None
    assert max_num_seqs_from_registry(REGISTRY) == 256


def test_a_model_with_no_sequence_limit_is_a_loud_failure() -> None:
    # Silently falling back to vLLM's 1024 default would put the admission ceiling four
    # times higher than the pods actually run at, and every burst would be mis-sized.
    with pytest.raises(ValueError, match="does not pass --max-num-seqs"):
        max_num_seqs_from_registry({"models": [{"name": "x", "vllm_extra_args": []}]})


def test_models_that_disagree_on_the_sequence_limit_are_a_loud_failure() -> None:
    doc = {"models": [
        {"name": "a", "vllm_extra_args": ["--max-num-seqs", "256"]},
        {"name": "b", "vllm_extra_args": ["--max-num-seqs", "64"]},
    ]}
    with pytest.raises(ValueError, match="disagree"):
        max_num_seqs_from_registry(doc)


def test_cap_for_registry_builds_the_deployed_policy_from_what_is_running() -> None:
    cap = cap_for_registry(REGISTRY, replicas=3)
    assert cap.name == ENGINE_CAPPED.name
    assert cap.sequence_limit == 256
    assert cap.fleet_sequence_limit == 768
    smaller = cap_for_registry(
        {"models": [{"name": "a", "vllm_extra_args": ["--max-num-seqs", "32"]}]}
    )
    assert smaller.sequence_limit == 32
