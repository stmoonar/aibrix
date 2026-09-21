#!/usr/bin/env python3
"""The serving path's admission limits, as one configurable object.

Why this module exists
----------------------
Two independent limits decide how many requests can be in flight against one model,
and which of the two binds changes what a load primitive can possibly observe:

``maxParallelRequests`` + ``maxPendingRequests``
    Envoy circuit-breaker limits from ``deploy/gateway-hardening``. They are enforced
    **per Envoy cluster, shared across every replica of the model**, so adding replicas
    does not raise this ceiling. Beyond it Envoy *sheds*: it answers 503 with a
    plain-text body and the request never reaches vLLM.

``--max-num-seqs`` (and the KV cache)
    vLLM's own running-set limit, **per pod**. Beyond it the engine *queues*: the
    surplus lands in the waiting queue and shows up in ``vllm:num_requests_waiting``.
    The engine never sheds.

That difference is the whole calibration problem. ``num_requests_waiting`` - the
regressor that makes ``lambda_wait`` identifiable - can only become non-zero when the
*engine* is the binding limit. When the gateway ceiling sits below the engine's running
limit, offered load is shed before a queue can form, and the only remaining route to a
non-zero waiting count is KV-cache exhaustion.

Where the ceiling is today (2026-09-21)
---------------------------------------
``deploy/gateway-hardening`` now sets, identically on both experiment arms::

    maxConnections: 4096   maxParallelRequests: 4096
    maxPendingRequests: 1024   maxParallelRetries: 16

and all three models pass ``--max-num-seqs 256`` in the registry's ``vllm_extra_args``.
The shed ceiling is therefore 5120 per cluster - far above anything the campaign offers -
and the real admission ceiling is the engine's::

    admission ceiling = max_num_seqs * awake replicas

which, unlike the Envoy cluster limits, **grows when TRE scales out**. That is the
property an autoscaling experiment has to be measured under, and it is why
:data:`ENGINE_CAPPED` is the policy the campaign now generates against. The sequence
limit must be read from the registry rather than assumed - see
:func:`max_num_seqs_from_registry` and :func:`cap_for_registry`.

Superseded observation (2026-09-20) - kept, not deleted
-------------------------------------------------------
The diagnosis that produced this module measured the *previous* policy: with
``maxParallelRequests: 256`` + ``maxPendingRequests: 64`` and no ``--max-num-seqs``
override, a pre-check against the live gateway opened requests until they were refused
and the first 503 arrived at in-flight 321, carrying Envoy's plain-text overflow body
and no ``x-envoy-*`` header. That measurement is **no longer the deployed ceiling**, and
every number derived from it is invalid by construction - including the theta values
1718 / 1494 / 1414, which fitted the circuit breaker and mistook it for capacity. It is
recorded here because it is still the evidence for two things that remain true: that the
Envoy limits are per *cluster* and so do not scale with replicas, and that a shed carries
no header the classifier can key on (see :func:`scripts.openloop.classify_failure`).

Everything downstream (burst sizing, the ramp's backlog budget, the skip list) is
expressed against an :class:`AdmissionCap` rather than against a hard-coded number, so
the campaign can be regenerated for a different policy without editing formulas. Two
policies are named below: :data:`ENGINE_CAPPED` (what is deployed) and
:data:`GATEWAY_CAPPED` (the superseded one, kept so an older capture can be re-read
against the rules it was actually taken under).

This module describes limits; it never applies them. Changing the deployed policy is a
shared-resource edit to the BackendTrafficPolicy and the model manifests.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Optional, Sequence

#: vLLM's default running-set limit when ``--max-num-seqs`` is not passed. The deployed
#: manifests DO override it now (registry ``vllm_extra_args``), so this is only the
#: fallback for a policy that deliberately describes an engine with no override.
VLLM_DEFAULT_MAX_NUM_SEQS = 1024

#: The vLLM flag that sets the per-pod running-set limit, as it appears in the
#: registry's ``vllm_extra_args`` list.
MAX_NUM_SEQS_FLAG = "--max-num-seqs"

#: The ramp's excess-request headroom is rounded down to a multiple of this, so a change
#: in the ``typical_running`` estimate does not jitter every generated schedule.
RAMP_BUDGET_QUANTUM = 10


def ramp_backlog_coefficient(rho_start: float, rho_end: float, hold_fraction: float) -> float:
    """Backlog accumulated by one ramp+hold, as a multiple of ``C_s * T_r``.

    The ramp is linear from ``rho_start`` to ``rho_end`` over ``T_r``, then holds
    ``rho_end`` for ``hold_fraction * T_r``. Backlog only accrues while rho > 1:

    * ramp part - rho exceeds 1 over the last ``(rho_end - 1) / (rho_end - rho_start)``
      of the ramp, and the excess grows linearly from 0 to ``(rho_end - 1) * C_s``, so it
      integrates to ``0.5 * above * (rho_end - 1) * C_s * T_r``.
    * hold part - a constant excess ``(rho_end - 1) * C_s`` for ``hold_fraction * T_r``.

    For the campaign's 0.4 -> 1.2 ramp with a quarter-length hold this is 0.075, i.e. a
    cell accumulates ``0.075 * C_s * T_r`` excess requests. Inverting that against the
    admission headroom is what fixes ``T_r``; see :meth:`AdmissionCap.ramp_seconds`.
    """
    if rho_end <= 1.0:
        return 0.0
    span = rho_end - rho_start
    if span <= 0.0:
        raise ValueError("ramp must increase: rho_end must exceed rho_start")
    above = (rho_end - 1.0) / span
    return 0.5 * above * (rho_end - 1.0) + hold_fraction * (rho_end - 1.0)


@dataclass(frozen=True)
class BurstSizing:
    """How many requests one burst spike carries, and whether it can work at all."""

    requests: int
    requests_needed: int
    request_cap: int
    engine_running_limit: int
    sequence_limit: int
    kv_request_limit: int
    reachable: bool
    binding_limit: str
    reason: str

    def as_dict(self) -> dict:
        return {
            "burst_requests": self.requests,
            "burst_requests_needed": self.requests_needed,
            "burst_request_cap": self.request_cap,
            "engine_running_limit": self.engine_running_limit,
            "sequence_limit": self.sequence_limit,
            "kv_request_limit": self.kv_request_limit,
            "reachable": self.reachable,
            "binding_limit": self.binding_limit,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class AdmissionCap:
    """One admission policy, plus the campaign constants derived from it."""

    name: str
    #: Envoy ``maxParallelRequests``: in-flight requests allowed per cluster, shared
    #: across replicas.
    max_parallel_requests: int
    #: Envoy ``maxPendingRequests``: the queue in front of that, also per cluster.
    max_pending_requests: int
    #: vLLM ``--max-num-seqs`` per pod; ``None`` means the engine default.
    max_num_seqs: Optional[int] = None
    #: Awake replicas during the calibration cell. Calibration drives a single awake
    #: replica, matching how the capacity priors were measured.
    replicas: int = 1
    #: Steady-state running set a loaded model shows under the campaign's shapes. Only
    #: used to leave headroom in the ramp's backlog budget.
    typical_running: int = 110
    #: A burst may claim this fraction of the shed ceiling, leaving the rest for the
    #: primitive's base stream and for ordinary jitter.
    burst_admission_fraction: float = 0.75
    #: A burst must overshoot the engine's running limit by this factor for the surplus
    #: to be unambiguously a queue rather than scheduling noise.
    burst_overshoot_factor: float = 1.5
    #: Ceiling on the ramp's rise time regardless of headroom, so one cell stays bounded.
    ramp_max_s: float = 300.0

    # ---------------------------------------------------------------- derived limits

    @property
    def shed_ceiling(self) -> int:
        """In-flight level beyond which Envoy answers 503 without reaching vLLM."""
        return self.max_parallel_requests + self.max_pending_requests

    @property
    def sequence_limit(self) -> int:
        """Per-pod vLLM running-set limit actually in force."""
        return VLLM_DEFAULT_MAX_NUM_SEQS if self.max_num_seqs is None else int(self.max_num_seqs)

    @property
    def fleet_sequence_limit(self) -> int:
        """Engine running capacity across the awake replicas. Unlike
        :attr:`shed_ceiling` this one scales with scaling, which is the property the
        autoscaler is supposed to be measured on."""
        return self.sequence_limit * max(1, self.replicas)

    @property
    def admission_controller(self) -> str:
        """Which limit binds first: ``gateway`` (requests are shed) or ``engine``
        (requests are queued, so ``num_requests_waiting`` can move)."""
        return "gateway" if self.shed_ceiling <= self.fleet_sequence_limit else "engine"

    @property
    def burst_request_cap(self) -> int:
        """Largest burst that can be admitted without tripping the circuit breaker."""
        return max(1, int(math.floor(self.burst_admission_fraction * self.shed_ceiling)))

    @property
    def ramp_excess_budget(self) -> int:
        """Excess requests a ramp may accumulate before it risks being shed.

        Measured against ``maxParallelRequests`` alone, not the shed ceiling: the pending
        slots are a shock absorber for jitter, and a ramp that plans to consume them has
        no margin left for the Poisson arrivals it is built from. The headroom is then
        rounded down to a whole :data:`RAMP_BUDGET_QUANTUM`, because ``typical_running``
        is an estimate and spending its last few slots buys nothing.

        Under the deployed policy this is 3980 slots, which no campaign ramp comes near:
        :meth:`ramp_seconds` is then pinned at :attr:`ramp_max_s` for every shape, i.e.
        the ramp's length is bounded by "one cell must stay bounded" rather than by the
        proxy. Under the superseded 256+64 policy it was 140 and it was the binding
        constraint. Both are the same formula; only which term wins has changed.
        """
        headroom = self.max_parallel_requests - self.typical_running
        quantised = (headroom // RAMP_BUDGET_QUANTUM) * RAMP_BUDGET_QUANTUM
        return max(1, quantised)

    # ------------------------------------------------------------------- primitives

    def kv_request_limit(self, kv_cache_tokens: int, tokens_per_request: float) -> int:
        """Concurrent requests of this shape whose KV footprint fills the cache."""
        if tokens_per_request <= 0:
            raise ValueError("tokens_per_request must be positive")
        return max(1, int(math.floor(kv_cache_tokens / tokens_per_request)))

    def engine_running_limit(self, kv_cache_tokens: int, tokens_per_request: float) -> int:
        """Requests of this shape the *fleet's engines* can run at once: the tighter of
        the sequence limit and what the KV cache holds, times the awake replicas. Beyond
        this the engine queues.

        Both terms are per pod and both scale with :attr:`replicas`, which is the whole
        reason this ceiling is the right one to size a burst against: it is the limit
        that moves when the autoscaler acts, whereas the Envoy cluster limits do not.
        Calibration drives one awake replica, so there ``replicas == 1``.
        """
        per_pod = min(self.sequence_limit,
                      self.kv_request_limit(kv_cache_tokens, tokens_per_request))
        return max(1, per_pod * max(1, self.replicas))

    def burst_sizing(self, kv_cache_tokens: int, tokens_per_request: float) -> BurstSizing:
        """Size one burst spike for a shape, and say whether it can reach the queue.

        A burst has to push the engine past its running limit, because that is the only
        way ``num_requests_waiting`` moves. If the requests needed to do that exceed what
        the gateway will admit, the spike is shed instead of queued and the primitive
        cannot observe anything - the cell must be skipped rather than run to produce a
        flat zero.

        The comparison is against the *engine* ceiling ``max_num_seqs * replicas`` (or
        the KV cache, whichever binds first), never against a fixed in-flight number. Under
        the superseded 256+64 policy the gateway budget was 240 requests and almost every
        shape was declared unreachable; under the deployed one the budget is 3840 and the
        overshoot needed is a few hundred, so the skip list is empty and
        ``num_requests_waiting`` becomes identifiable for every shape.
        """
        kv_limit = self.kv_request_limit(kv_cache_tokens, tokens_per_request)
        running_limit = self.engine_running_limit(kv_cache_tokens, tokens_per_request)
        needed = int(math.ceil(self.burst_overshoot_factor * running_limit))
        cap = self.burst_request_cap
        reachable = needed <= cap
        binding = "kv_cache" if kv_limit <= self.sequence_limit else "max_num_seqs"
        if reachable:
            reason = (
                f"{needed} concurrent requests overshoot the engine running limit of "
                f"{running_limit} ({binding}-bound) and fit under the {cap}-request "
                f"admission budget of the {self.name} policy"
            )
        else:
            reason = (
                f"reaching waiting > 0 needs {needed} concurrent requests "
                f"({self.burst_overshoot_factor}x the {running_limit}-request engine "
                f"running limit, {binding}-bound), but the {self.name} policy admits at "
                f"most {cap} ({self.burst_admission_fraction:.0%} of the "
                f"{self.shed_ceiling}-request shed ceiling); the spike would be shed by "
                f"the gateway instead of queued by the engine"
            )
        return BurstSizing(
            requests=min(cap, needed),
            requests_needed=needed,
            request_cap=cap,
            engine_running_limit=running_limit,
            sequence_limit=self.sequence_limit,
            kv_request_limit=kv_limit,
            reachable=reachable,
            binding_limit=binding,
            reason=reason,
        )

    def ramp_seconds(self, capacity_rps: float, backlog_coefficient: float) -> float:
        """Rise time for a ramp whose accumulated backlog fits the admission headroom.

        ``backlog_coefficient`` comes from :func:`ramp_backlog_coefficient` and depends
        only on the ramp's shape. Solving ``coefficient * C_s * T_r <= budget`` for
        ``T_r`` and clamping at :attr:`ramp_max_s` gives a ramp that crosses the
        violation boundary without the backlog being truncated by gateway shedding.
        """
        if capacity_rps <= 0.0:
            raise ValueError("capacity_rps must be positive")
        if backlog_coefficient <= 0.0:
            raise ValueError("a ramp that never exceeds rho=1 has no backlog budget")
        return min(self.ramp_max_s, self.ramp_excess_budget / (backlog_coefficient * capacity_rps))

    # ------------------------------------------------------------------- bookkeeping

    def with_replicas(self, replicas: int) -> "AdmissionCap":
        return replace(self, replicas=int(replicas))

    def as_dict(self) -> dict:
        return {
            "name": self.name,
            "max_parallel_requests": self.max_parallel_requests,
            "max_pending_requests": self.max_pending_requests,
            "max_num_seqs": self.max_num_seqs,
            "sequence_limit": self.sequence_limit,
            "replicas": self.replicas,
            "fleet_sequence_limit": self.fleet_sequence_limit,
            "shed_ceiling": self.shed_ceiling,
            "admission_controller": self.admission_controller,
            "typical_running": self.typical_running,
            "burst_request_cap": self.burst_request_cap,
            "burst_overshoot_factor": self.burst_overshoot_factor,
            "ramp_excess_budget": self.ramp_excess_budget,
            "ramp_max_s": self.ramp_max_s,
        }


#: SUPERSEDED on 2026-09-21, kept so a capture taken before that date can be re-read
#: against the rules it was actually taken under. Envoy admitted 256 + 64 per model
#: cluster shared across replicas and vLLM's sequence limit was the unreachable 1024
#: default; the pre-check measured the first shed at in-flight 321, confirming 320.
#: Nothing new should be generated against this policy.
GATEWAY_CAPPED = AdmissionCap(
    name="gateway-capped",
    max_parallel_requests=256,
    max_pending_requests=64,
    max_num_seqs=None,
)

#: APPLIED 2026-09-21 and live on both experiment arms: the circuit breaker sits far
#: above any load the campaign offers and vLLM's sequence limit is set explicitly, so the
#: binding limit is the engine's. Two consequences matter. Requests past the limit queue
#: instead of being shed, so ``num_requests_waiting`` becomes observable for every shape;
#: and the ceiling is per pod, so it scales with replicas - which is what an autoscaler
#: experiment needs to be measuring. ``max_num_seqs`` here is the value the registry
#: carries today; :func:`cap_for_registry` re-reads it rather than trusting this copy.
ENGINE_CAPPED = AdmissionCap(
    name="engine-capped",
    max_parallel_requests=4096,
    max_pending_requests=1024,
    max_num_seqs=256,
)

CAPS = {cap.name: cap for cap in (GATEWAY_CAPPED, ENGINE_CAPPED)}

#: The policy the campaign generates against unless told otherwise. It tracks what is
#: deployed, so a campaign run from the committed tree matches the cluster it runs on.
DEFAULT_CAP_NAME = ENGINE_CAPPED.name


def get_cap(name: str) -> AdmissionCap:
    try:
        return CAPS[name]
    except KeyError:
        raise SystemExit(
            f"unknown admission cap {name!r}; known: {', '.join(sorted(CAPS))}"
        ) from None


# ------------------------------------------------------------------ registry sourcing


def max_num_seqs_from_args(extra_args: Sequence[str]) -> Optional[int]:
    """``--max-num-seqs`` out of one model's ``vllm_extra_args``, or None.

    Accepts both spellings the registry may carry: the flag and its value as two list
    entries, and ``--max-num-seqs=256`` as one.
    """
    args = [str(a) for a in extra_args or ()]
    for index, arg in enumerate(args):
        if arg == MAX_NUM_SEQS_FLAG:
            if index + 1 >= len(args):
                raise ValueError(f"{MAX_NUM_SEQS_FLAG} is the last entry and has no value")
            return int(args[index + 1])
        if arg.startswith(MAX_NUM_SEQS_FLAG + "="):
            return int(arg.split("=", 1)[1])
    return None


def max_num_seqs_from_registry(registry_doc: dict, models: Optional[Sequence[str]] = None) -> int:
    """The per-pod sequence limit the fleet actually runs, read from the registry.

    The admission ceiling is ``max_num_seqs * replicas``, so hard-coding 256 would make
    every burst-reachability verdict silently wrong the moment the manifests changed -
    which is exactly how the superseded 320 became load-bearing. The value is therefore
    read from the same ``vllm_extra_args`` the pods are launched with.

    Every named model must agree: a fleet whose models run different sequence limits has
    no single admission ceiling, and quietly picking one of them would size bursts for a
    model that is not the one being driven.
    """
    entries = [e for e in (registry_doc.get("models") or []) if isinstance(e, dict)]
    if models is not None:
        wanted = set(models)
        entries = [e for e in entries if e.get("name") in wanted]
    if not entries:
        raise ValueError("registry names no models to read a sequence limit from")
    found: dict[str, int] = {}
    for entry in entries:
        value = max_num_seqs_from_args(entry.get("vllm_extra_args") or ())
        if value is None:
            raise ValueError(
                f"model {entry.get('name')!r} does not pass {MAX_NUM_SEQS_FLAG}; the "
                "admission ceiling is then vLLM's 1024 default, which is not what the "
                "deployed manifests carry - fix the registry rather than assuming"
            )
        found[str(entry.get("name"))] = value
    distinct = set(found.values())
    if len(distinct) != 1:
        raise ValueError(
            f"models disagree on {MAX_NUM_SEQS_FLAG}: {found}. The admission ceiling is "
            "per model, so there is no single cap to size a campaign against"
        )
    return distinct.pop()


def cap_for_registry(
    registry_doc: dict,
    *,
    base: Optional[AdmissionCap] = None,
    models: Optional[Sequence[str]] = None,
    replicas: int = 1,
) -> AdmissionCap:
    """``base`` (default the deployed policy) with its sequence limit taken from the
    registry and its replica count set explicitly."""
    cap = base or get_cap(DEFAULT_CAP_NAME)
    return replace(
        cap,
        max_num_seqs=max_num_seqs_from_registry(registry_doc, models),
        replicas=int(replicas),
    )
