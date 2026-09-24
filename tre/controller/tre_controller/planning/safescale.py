from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import Any, Literal, Protocol

from tre_controller.config import SafeScaleConfig

ProbeStatus = Literal["none", "probing", "commit", "rollback"]
CommandKind = Literal["hide", "unhide", "scale_down", "scale_up"]


class ProbeStore(Protocol):
    def save_probe(self, request_id: str, record: dict[str, Any]) -> None: ...

    def delete_probe(self, request_id: str) -> None: ...

    def list_unresolved_probes(self) -> list[dict[str, Any]]: ...

    def append_probe_journal(self, request_id: str, record: dict[str, Any]) -> None: ...

    def load_probe_journal(self, request_id: str) -> list[dict[str, Any]]: ...


@dataclass(frozen=True)
class ProbeObservation:
    ts_ms: int
    ttft_p95_ms: float | None = None
    tpot_p95_ms: float | None = None
    z_m: float | None = None
    q_ctl: float | None = None
    has_traffic: bool = False
    avg_gpu_cache_norm: float | None = None
    #: Cumulative gateway counters of the donor model (A13 donor-health), None = unknown.
    gateway_requests: float | None = None
    gateway_errors: float | None = None


@dataclass(frozen=True)
class ProbeWindowInputs:
    """The donor's metrics at probe start, for the adaptive probe window (v1
    ``start_hidden_probe`` -> ``_calc_probe_window_details``). Units as in v1:

    * ``p95_e2e_ms`` / ``p95_tpot_ms``: window p95 latencies (ms) of the serving pods;
    * ``q``: Q_ctl (in-flight requests, the TSS queue term);
    * ``y_total``: Y_m, the TSS numerator = weighted tokens in the metrics window;
      ``y_per_pod``: y_m = Y_m / routable pods (used only when Y_m is missing);
    * ``z_m``: the donor's Z (spare-capacity multiplier, v1 ``max(1, z_m)``);
    * ``routable_pods``: serving (awake, not hidden) pods BEFORE the hide;
    * ``interval_s``: metrics window length (s) that Y_m was summed over.
    """

    p95_e2e_ms: float | None = None
    p95_tpot_ms: float | None = None
    #: v1 fallbacks when a p95 is missing: overall window mean TTFT for the e2e term
    #: (sic - v1 uses avg TTFT, not avg e2e) and mean TPOT for the decode term.
    avg_ttft_ms: float | None = None
    avg_tpot_ms: float | None = None
    q: float | None = None
    y_total: float | None = None
    y_per_pod: float | None = None
    z_m: float | None = None
    routable_pods: int | None = None
    interval_s: float | None = None


@dataclass(frozen=True)
class CommitDrainPolicy:
    """SM drain budget for a SafeScale commit (TRE_SM_CALL_DRAIN): the probe pod has
    been hidden for the whole probe window, so at commit the SM only waits for its
    residual in-flight requests: clamp(factor * p95_e2e, min_s, max_s), default_s
    when the donor has no e2e p95. Every other sleep is a direct one (drain 0)."""

    factor: float = 2.0
    min_s: float = 10.0
    max_s: float = 120.0
    default_s: float = 30.0

    def __post_init__(self) -> None:
        for name in ("factor", "min_s", "max_s", "default_s"):
            if not float(getattr(self, name)) > 0:
                raise ValueError(f"CommitDrainPolicy.{name} must be positive")
        if self.min_s > self.max_s:
            raise ValueError("TRE_SAFESCALE_COMMIT_DRAIN_MIN_S must not exceed _MAX_S")

    def budget_s(self, p95_e2e_ms: float | None) -> float:
        p95 = _positive(p95_e2e_ms)
        raw = self.default_s if p95 is None else self.factor * p95 / 1000.0
        return min(max(raw, self.min_s), self.max_s)

    @classmethod
    def from_config(cls, cfg: Any) -> "CommitDrainPolicy | None":
        """None unless TRE_SM_CALL_DRAIN is on (then nothing changes vs main)."""
        if not bool(getattr(cfg, "sm_call_drain", False)):
            return None
        return cls(
            factor=float(getattr(cfg, "safescale_commit_drain_factor", 2.0)),
            min_s=float(getattr(cfg, "safescale_commit_drain_min_s", 10.0)),
            max_s=float(getattr(cfg, "safescale_commit_drain_max_s", 120.0)),
            default_s=float(getattr(cfg, "safescale_commit_drain_default_s", 30.0)),
        )


@dataclass(frozen=True)
class SafeScaleCommand:
    kind: CommandKind
    model: str
    pods: tuple[str, ...] = ()
    delta: int = 0
    reason: str = ""


@dataclass(frozen=True)
class SafeScaleDecision:
    status: ProbeStatus
    reason: str
    commands: tuple[SafeScaleCommand, ...] = ()
    #: probe_started: the adaptive window breakdown (calc_probe_window_details).
    details: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SafeScaleProbe:
    model: str
    pods: tuple[str, ...]
    start_ms: int
    deadline_ms: int
    request_id: str
    status: Literal["probing"] = "probing"
    pending_upscales: dict[str, int] = field(default_factory=dict)
    observations: tuple[ProbeObservation, ...] = ()
    #: Adaptive window W (ms) and its breakdown (calc_probe_window_details), for reports.
    window_ms: float | None = None
    window_terms: dict[str, Any] = field(default_factory=dict)
    #: Why the probe ended (gate failures, tail summary), persisted with the resolution.
    terminal_details: dict[str, Any] = field(default_factory=dict)
    #: A13: the donor's gateway (requests, errors) counters at the first observation.
    gateway_baseline: tuple[float, float] | None = None
    #: v1 receiver_need_upscale: the model itself now needs capacity; the next
    #: observation rolls the probe back with this reason.
    preempt_reason: str | None = None


@dataclass(frozen=True)
class ProbeTailSummary:
    latency_ok: bool
    z_min: float | None
    has_traffic: bool
    sample_count: int
    tail_count: int
    gpu_cache_max: float | None = None


class SafeScaleStateMachine:
    def __init__(self, *, config: SafeScaleConfig, store: ProbeStore | None = None) -> None:
        self._config = config
        self._store = store
        self._probes: dict[str, SafeScaleProbe] = {}
        # A13 rollback backoff: model -> time (ms, snapshot clock) of its last rollback.
        # Deliberately in-memory only: a controller restart forgets it, i.e. at most one
        # extra HIGH probe per model right after a restart (the probe itself is still
        # guarded by SLO / donor-health / commit gate). Not worth a persisted schema.
        self._last_rollback_ms: dict[str, int] = {}

    def active_probe(self, model: str) -> SafeScaleProbe | None:
        return self._probes.get(model)

    def active_probes(self) -> tuple[SafeScaleProbe, ...]:
        return tuple(self._probes.values())

    def request_preemption(self, model: str, *, reason: str = "receiver_need_upscale") -> int:
        """v1 apply_safescale_to_deltas: a scale-up of a model that is itself probing rolls
        its probe back first. Marks the probe; its next observation returns the rollback
        (unhide, reason ``reason``). Returns how many pods that restores (0 = no probe)."""
        probe = self._probes.get(model)
        if probe is None:
            return 0
        if probe.preempt_reason is None:
            probe = replace(probe, preempt_reason=reason)
            self._probes[model] = probe
            self._persist_probe(probe)
        return len(probe.pods)

    def rollback_backoff_models(self, now_ms: int) -> set[str]:
        """Models whose last probe rolled back less than rollback_backoff_ms ago (A13):
        the planner holds their receiver-less HIGH proactive probe meanwhile."""
        backoff = float(getattr(self._config, "rollback_backoff_ms", 0.0) or 0.0)
        if backoff <= 0:
            return set()
        return {model for model, ts in self._last_rollback_ms.items() if 0 <= now_ms - ts < backoff}

    def start_probe(
        self,
        *,
        model: str,
        pods: tuple[str, ...],
        now_ms: int,
        pending_upscales: dict[str, int] | None = None,
        window_inputs: ProbeWindowInputs | None = None,
    ) -> SafeScaleDecision:
        if model in self._probes:
            return SafeScaleDecision(status="probing", reason="probe_already_active")
        if not pods:
            return SafeScaleDecision(status="none", reason="no_pods_to_probe")

        normalized_pending = _normalize_pending_upscales(pending_upscales)
        # A6 (v1 safescale.py start_hidden_probe): adaptive window
        # W = clamp(W_lo, W_hi, max(2*p95_e2e, cdec*p95_tpot, Q/rate_gap)).
        terms = calc_probe_window_details(
            window_inputs or ProbeWindowInputs(), hidden_count=len(pods), config=self._config
        )
        window_ms = float(terms["W"])
        probe = SafeScaleProbe(
            model=model,
            pods=tuple(pods),
            start_ms=int(now_ms),
            deadline_ms=int(now_ms + window_ms),
            request_id=f"{model}-{int(now_ms)}",
            pending_upscales=normalized_pending,
            window_ms=window_ms,
            window_terms=terms,
        )
        self._probes[model] = probe
        self._persist_probe(probe)
        return SafeScaleDecision(
            status="probing",
            reason="probe_started",
            commands=(SafeScaleCommand(kind="hide", model=model, pods=probe.pods, reason="probe_started"),),
            details=dict(terms),
        )

    def observe(self, model: str, observation: ProbeObservation, *, now_ms: int) -> SafeScaleDecision:
        probe = self._probes.get(model)
        if probe is None:
            return SafeScaleDecision(status="none", reason="probe_not_found")

        updated = _with_gateway_baseline(
            _replace_observations(probe, probe.observations + (observation,)), observation
        )
        self._probes[model] = updated
        self._persist_observation(updated, observation)

        health = donor_health(updated, observation)
        if updated.preempt_reason is not None:
            self._probes[model] = replace(updated, terminal_details={"preempted": updated.preempt_reason})
            return self._rollback(updated, reason=updated.preempt_reason)
        if self._violates_slo(observation):
            self._probes[model] = replace(
                updated,
                terminal_details={
                    "slo": {"ttft_p95_ms": observation.ttft_p95_ms, "tpot_p95_ms": observation.tpot_p95_ms}
                },
            )
            return self._rollback(updated, reason="slo_violation")

        if (
            health is not None
            and health["requests"] >= self._config.donor_min_requests
            and health["error_rate"] > self._config.donor_error_rate_max
        ):
            self._probes[model] = replace(updated, terminal_details={"donor_health": health})
            return self._rollback(updated, reason="donor_health")

        if now_ms < updated.deadline_ms:
            self._persist_probe(updated)
            return SafeScaleDecision(status="probing", reason="probe_pending")

        summary = _summarize_tail(
            updated,
            hq=self._config.hq,
            ttft_p95_slo_ms=self._config.ttft_p95_slo_ms,
            tpot_p95_slo_ms=self._config.tpot_p95_slo_ms,
        )
        failures = tail_gate_failures(
            summary, tau_low=self._config.tau_low, kv_cache_max=self._config.kv_cache_max
        )
        details: dict[str, Any] = {
            "gate_failures": list(failures),
            "tail": _tail_record(summary),
            # A12/P2-a: no KV-cache sample in the tail (no pod reported the gauge) - the
            # gate passes this check like v1, but the record says so explicitly.
            "kv_cache": "unavailable" if summary.gpu_cache_max is None else summary.gpu_cache_max,
        }
        if health is not None:
            details["donor_health"] = health
        self._probes[model] = replace(updated, terminal_details=details)
        if not failures:
            return self._commit(updated, reason="formal_commit_gate_passed")
        return self._rollback(updated, reason="formal_commit_gate_failed")

    def restore(self) -> int:
        if self._store is None:
            return 0
        restored = 0
        for row in self._store.list_unresolved_probes():
            probe = _probe_from_record(row, self._store)
            if probe is None or probe.model in self._probes:
                continue
            self._probes[probe.model] = probe
            restored += 1
        return restored

    def _commit(self, probe: SafeScaleProbe, *, reason: str) -> SafeScaleDecision:
        # Review F2/F3: commit sleeps exactly the hidden probe pods (binding-level power).
        # A model-level scale_down(-n) let the SM pick the tail of `awake`, which could
        # sleep a serving pod and leave the hidden one awake as an orphan.
        commands: list[SafeScaleCommand] = [
            SafeScaleCommand(
                kind="scale_down",
                model=probe.model,
                pods=probe.pods,
                delta=-len(probe.pods),
                reason=reason,
            )
        ]
        for model, delta in sorted(probe.pending_upscales.items()):
            commands.append(
                SafeScaleCommand(kind="scale_up", model=model, delta=delta, reason="safescale_followup_upscale")
            )
        return SafeScaleDecision(status="commit", reason=reason, commands=tuple(commands))

    def _rollback(self, probe: SafeScaleProbe, *, reason: str) -> SafeScaleDecision:
        return SafeScaleDecision(
            status="rollback",
            reason=reason,
            commands=(SafeScaleCommand(kind="unhide", model=probe.model, pods=probe.pods, reason=reason),),
        )

    def resolve(
        self,
        model: str,
        *,
        status: Literal["commit", "rollback"],
        reason: str,
        now_ms: int,
    ) -> bool:
        probe = self._probes.get(model)
        if probe is None:
            return False
        self._finish_probe(
            probe,
            status=status,
            reason=reason,
            resolved_ts=float(now_ms) / 1000.0,
        )
        self._probes.pop(model, None)
        # A preemption for the model's own scale-up is not a failed probe: no backoff.
        if status == "rollback" and probe.preempt_reason is None:
            self._last_rollback_ms[model] = int(now_ms)
        return True

    def _violates_slo(self, observation: ProbeObservation) -> bool:
        return (
            observation.ttft_p95_ms is not None
            and observation.ttft_p95_ms > self._config.ttft_p95_slo_ms
        ) or (
            observation.tpot_p95_ms is not None
            and observation.tpot_p95_ms > self._config.tpot_p95_slo_ms
        )

    def _persist_probe(self, probe: SafeScaleProbe, *, terminal_reason: str | None = None) -> None:
        if self._store is None:
            return
        self._store.save_probe(probe.request_id, _probe_record(probe, terminal_reason=terminal_reason))

    def _persist_observation(self, probe: SafeScaleProbe, observation: ProbeObservation) -> None:
        if self._store is None:
            return
        record = _probe_record(probe, terminal_reason=None)
        record["last_observation"] = _observation_record(observation)
        self._store.append_probe_journal(probe.request_id, record)

    def _finish_probe(
        self,
        probe: SafeScaleProbe,
        *,
        status: Literal["commit", "rollback"],
        reason: str,
        resolved_ts: float,
    ) -> None:
        if self._store is None:
            return
        self._store.save_probe(
            probe.request_id,
            _probe_record(
                probe,
                terminal_reason=reason,
                status="resolved",
                resolution=status,
                resolved_ts=resolved_ts,
            ),
        )


def _replace_observations(probe: SafeScaleProbe, observations: tuple[ProbeObservation, ...]) -> SafeScaleProbe:
    return replace(probe, pending_upscales=dict(probe.pending_upscales), observations=observations)


def _with_gateway_baseline(probe: SafeScaleProbe, observation: ProbeObservation) -> SafeScaleProbe:
    """Set the donor-health baseline at the first observation carrying gateway counters
    (the hide is dispatched asynchronously, so nothing it causes precedes it); re-baseline
    after a counter drop (Envoy restart)."""
    if observation.gateway_requests is None or observation.gateway_errors is None:
        return probe
    current = (float(observation.gateway_requests), float(observation.gateway_errors))
    baseline = probe.gateway_baseline
    if baseline is None or current[0] < baseline[0] or current[1] < baseline[1]:
        return replace(probe, gateway_baseline=current)
    return probe


def donor_health(probe: SafeScaleProbe, observation: ProbeObservation) -> dict[str, float] | None:
    """Gateway requests / errors / error ratio of the donor model since the probe's
    baseline (A13), or None without counters (guard fails open)."""
    baseline = probe.gateway_baseline
    if baseline is None or observation.gateway_requests is None or observation.gateway_errors is None:
        return None
    requests = max(0.0, float(observation.gateway_requests) - baseline[0])
    errors = max(0.0, float(observation.gateway_errors) - baseline[1])
    return {"requests": requests, "errors": errors, "error_rate": (errors / requests) if requests > 0 else 0.0}


def _estimate_post_drain_gap_per_second(
    *,
    y_total: float | None,
    y_per_pod: float | None,
    z_m: float | None,
    routable_pods: int | None,
    hidden_count: int,
    interval_s: float | None,
) -> float | None:
    """v1 ``_estimate_post_drain_gap_per_second`` verbatim: the spare service rate left
    after hiding ``hidden_count`` pods, in Y units (weighted tokens) per second.

    arrival = Y_m / interval; capacity = arrival * max(1, Z_m) (Z_m = TSS / theta_m, used
    by v1 as the spare-capacity multiplier); gap = per-pod capacity * remaining pods -
    arrival, floored at 0.
    """
    interval = _positive(interval_s)
    current_pods = max(1, int(routable_pods) if routable_pods is not None else 1)
    remaining_pods = max(0, current_pods - max(0, int(hidden_count)))
    if interval is None or remaining_pods <= 0:
        return None
    y_all = _nonneg(y_total)
    y_pod = _nonneg(y_per_pod)
    if y_all is None and y_pod is None:
        return None
    arrival_rate = (y_all / interval) if y_all is not None else (y_pod * current_pods) / interval
    if arrival_rate <= 0:
        return None
    spare_multiplier = _positive(z_m) or 1.0
    current_capacity = arrival_rate * max(1.0, spare_multiplier)
    per_pod_capacity = current_capacity / current_pods
    mu_post = per_pod_capacity * remaining_pods
    return max(0.0, mu_post - arrival_rate)


def calc_probe_window_details(
    inputs: ProbeWindowInputs,
    *,
    hidden_count: int,
    config: SafeScaleConfig,
) -> dict[str, Any]:
    """Port of v1 ``safescale._calc_probe_window_details`` (v1 safescale.py:302-359).
    The FORMULA is v1's; the band is not (see SafeScaleConfig: v1 ran 15 s / 300 s with a
    20 s cW2 fallback).

    * W1 = 2 * p95_e2e (``default_window_ms`` when unavailable);
    * queue term cW2 = Q / rate_gap (s -> ms) when Q > 0; ``cw2_fallback_ms`` when the
      post-hide rate gap is unknown or <= epsilon_mu;
    * decode term = cdec * p95_tpot;
    * W2 = max(queue term, decode term) (``default_window_ms`` when neither exists);
    * W = clamp(min_window_ms, max_window_ms, max(W1, W2)).

    Returns every term plus ``dominant`` (which term set the unclamped max: e2e / queue /
    decode / default) and ``clamped`` (lo / hi / None) for the probe record and events.
    """
    default_ms = float(config.default_window_ms)
    lo_ms = float(config.min_window_ms)
    hi_ms = float(config.max_window_ms)
    # v1 start_hidden_probe (safescale.py:504-509): p95_e2e or avg_ttft, p95_tpot or avg_tpot.
    p95_e2e = _positive(inputs.p95_e2e_ms)
    latency_source = "p95_e2e" if p95_e2e is not None else None
    if p95_e2e is None:
        p95_e2e = _positive(inputs.avg_ttft_ms)
        latency_source = "avg_ttft" if p95_e2e is not None else None
    p95_tpot = _positive(inputs.p95_tpot_ms)
    decode_source = "p95_tpot" if p95_tpot is not None else None
    if p95_tpot is None:
        p95_tpot = _positive(inputs.avg_tpot_ms)
        decode_source = "avg_tpot" if p95_tpot is not None else None
    w1 = 2.0 * p95_e2e if p95_e2e is not None else default_ms

    gap = _estimate_post_drain_gap_per_second(
        y_total=inputs.y_total,
        y_per_pod=inputs.y_per_pod,
        z_m=inputs.z_m,
        routable_pods=inputs.routable_pods,
        hidden_count=hidden_count,
        interval_s=inputs.interval_s,
    )
    cw2_fallback = min(hi_ms, _positive(config.cw2_fallback_ms) or hi_ms)
    q = _nonneg(inputs.q)
    queue_term: float | None = None
    queue_fallback = False
    if q is not None and q > 0:
        if gap is None or gap <= config.epsilon_mu:
            queue_term = cw2_fallback
            queue_fallback = True
        else:
            queue_term = (q / max(gap, config.epsilon_mu)) * 1000.0
    decode_term = max(0.0, config.cdec) * p95_tpot if p95_tpot is not None else None

    w2_candidates = [value for value in (queue_term, decode_term) if value is not None]
    w2 = max(w2_candidates) if w2_candidates else default_ms
    raw = max(w1, w2)
    window = max(lo_ms, min(hi_ms, raw))

    if raw == w1 and p95_e2e is not None:
        dominant = "e2e"
    elif w2_candidates and raw == w2:
        dominant = "queue" if queue_term is not None and w2 == queue_term else "decode"
    else:
        dominant = "default"
    clamped = "lo" if raw < lo_ms else ("hi" if raw > hi_ms else None)
    return {
        "W": window,
        "W_raw": raw,
        "W1": w1,
        "W2": w2,
        "cW2": queue_term,
        "cW2_fallback": queue_fallback,
        "decode_term_ms": decode_term,
        "rate_gap_per_second": gap,
        "dominant": dominant,
        "clamped": clamped,
        "W_lo": lo_ms,
        "W_hi": hi_ms,
        "inputs": {
            "p95_e2e_ms": p95_e2e,
            "p95_tpot_ms": p95_tpot,
            "latency_source": latency_source,
            "decode_source": decode_source,
            "q": q,
            "y_total": _nonneg(inputs.y_total),
            "y_per_pod": _nonneg(inputs.y_per_pod),
            "z_m": _positive(inputs.z_m),
            "routable_pods": inputs.routable_pods,
            "hidden_count": int(hidden_count),
            "interval_s": _positive(inputs.interval_s),
        },
    }


def format_window_event(model: str, terms: dict[str, Any]) -> str:
    """One-line decision event for a probe's adaptive window (reports grep these)."""

    def fmt(value: Any) -> str:
        if value is None:
            return "na"
        if isinstance(value, bool):
            return "1" if value else "0"
        number = float(value)
        return f"{number:.0f}" if abs(number) >= 1 else f"{number:.3g}"

    return (
        f"safescale_probe_window:{model}:W={fmt(terms.get('W'))}"
        f":dominant={terms.get('dominant')}:clamped={terms.get('clamped') or 'none'}"
        f":e2e={fmt(terms.get('W1'))}:queue={fmt(terms.get('cW2'))}"
        f":decode={fmt(terms.get('decode_term_ms'))}:gap={fmt(terms.get('rate_gap_per_second'))}"
        f":fallback={fmt(terms.get('cW2_fallback'))}"
    )


def _positive(value: Any) -> float | None:
    parsed = _optional_float(value)
    return parsed if parsed is not None and parsed > 0 else None


def _nonneg(value: Any) -> float | None:
    parsed = _optional_float(value)
    return parsed if parsed is not None and parsed >= 0 else None


def _summarize_tail(
    probe: SafeScaleProbe,
    *,
    hq: float,
    ttft_p95_slo_ms: float,
    tpot_p95_slo_ms: float,
) -> ProbeTailSummary:
    observations = probe.observations
    if not observations:
        return ProbeTailSummary(latency_ok=True, z_min=None, has_traffic=False, sample_count=0, tail_count=0)

    sample_count = len(observations)
    hq_value = hq if hq > 0 else 0.25
    if hq_value < 1.0:
        desired_tail = max(2, int(math.ceil(sample_count * hq_value)))
    else:
        desired_tail = max(2, int(hq_value))
    tail = observations[-min(sample_count, desired_tail) :]

    latency_ok = True
    has_traffic = False
    z_values: list[float] = []
    gpu_cache_values: list[float] = []
    for observation in tail:
        if observation.has_traffic:
            has_traffic = True
        if observation.ttft_p95_ms is not None and observation.ttft_p95_ms > ttft_p95_slo_ms:
            latency_ok = False
        if observation.tpot_p95_ms is not None and observation.tpot_p95_ms > tpot_p95_slo_ms:
            latency_ok = False
        if observation.z_m is not None:
            z_values.append(float(observation.z_m))
        if observation.avg_gpu_cache_norm is not None:
            gpu_cache_values.append(float(observation.avg_gpu_cache_norm))

    z_min = min(z_values) if z_values else None
    if z_min is None and not has_traffic:
        z_min = float("inf")
    return ProbeTailSummary(
        latency_ok=latency_ok,
        z_min=z_min,
        has_traffic=has_traffic,
        sample_count=sample_count,
        tail_count=len(tail),
        gpu_cache_max=max(gpu_cache_values) if gpu_cache_values else None,
    )


def _tail_allows_commit(summary: ProbeTailSummary, *, tau_low: float, kv_cache_max: float = 0.8) -> bool:
    return not tail_gate_failures(summary, tau_low=tau_low, kv_cache_max=kv_cache_max)


def tail_gate_failures(
    summary: ProbeTailSummary, *, tau_low: float, kv_cache_max: float = 0.8
) -> tuple[str, ...]:
    """The formal commit gate (v1 _tail_summary_allows_commit), returning which checks
    failed instead of a bool (empty = commit). v1 short-circuits in this order: latency,
    missing Z under traffic, Z < tau_low, KV-cache > 0.8; the same order decides here, and
    every failing check is listed for the rollback-reason report."""
    failures: list[str] = []
    if not summary.latency_ok:
        failures.append("latency")
    if summary.z_min is None:
        if summary.has_traffic:
            failures.append("z_missing")
        # v1: no Z and no traffic -> commit, whatever the KV cache says.
        return tuple(failures)
    if summary.z_min < tau_low:
        failures.append("z_below_tau_low")
    if summary.gpu_cache_max is not None and summary.gpu_cache_max > kv_cache_max:
        failures.append("kv_cache")
    return tuple(failures)


def _tail_record(summary: ProbeTailSummary) -> dict[str, Any]:
    return {
        "latency_ok": summary.latency_ok,
        "z_min": summary.z_min if summary.z_min is None or math.isfinite(summary.z_min) else "inf",
        "has_traffic": summary.has_traffic,
        "sample_count": summary.sample_count,
        "tail_count": summary.tail_count,
        "gpu_cache_max": summary.gpu_cache_max,
    }


def _probe_record(
    probe: SafeScaleProbe,
    *,
    terminal_reason: str | None = None,
    status: str = "probing",
    resolution: str | None = None,
    resolved_ts: float | None = None,
) -> dict[str, Any]:
    record = {
        "model": probe.model,
        "request_id": probe.request_id,
        "pods": list(probe.pods),
        "hidden_count": len(probe.pods),
        "start_ms": probe.start_ms,
        "deadline_ms": probe.deadline_ms,
        "status": status,
        "pending_upscales": dict(probe.pending_upscales),
        "terminal_reason": terminal_reason,
        "window_ms": probe.window_ms,
        "window_terms": dict(probe.window_terms),
        "terminal_details": dict(probe.terminal_details),
        "gateway_baseline": list(probe.gateway_baseline) if probe.gateway_baseline is not None else None,
        "preempt_reason": probe.preempt_reason,
    }
    if resolution is not None:
        record["resolution"] = resolution
    if resolved_ts is not None:
        record["resolved_ts"] = resolved_ts
    return record


def _observation_record(observation: ProbeObservation) -> dict[str, Any]:
    return {
        "ts_ms": observation.ts_ms,
        "ttft_p95_ms": observation.ttft_p95_ms,
        "tpot_p95_ms": observation.tpot_p95_ms,
        "z_m": observation.z_m,
        "q_ctl": observation.q_ctl,
        "has_traffic": observation.has_traffic,
        "avg_gpu_cache_norm": observation.avg_gpu_cache_norm,
        "gateway_requests": observation.gateway_requests,
        "gateway_errors": observation.gateway_errors,
    }


def _probe_from_record(row: dict[str, Any], store: ProbeStore) -> SafeScaleProbe | None:
    model = str(row.get("model", "")).strip()
    request_id = str(row.get("request_id") or row.get("probe_id") or "").strip()
    pods = _normalize_pods(row)
    if not model or not request_id or not pods:
        return None
    try:
        start_ms = _time_ms(row, "start_ms", "start_ts")
        deadline_ms = _time_ms(row, "deadline_ms", "deadline_ts")
    except (TypeError, ValueError):
        return None

    observations: list[ProbeObservation] = []
    for entry in store.load_probe_journal(request_id):
        if not isinstance(entry, dict):
            continue
        raw = entry.get("last_observation")
        if isinstance(raw, dict):
            observation = _observation_from_record(raw)
            if observation is not None:
                observations.append(observation)

    return SafeScaleProbe(
        model=model,
        pods=pods,
        start_ms=start_ms,
        deadline_ms=deadline_ms,
        request_id=request_id,
        pending_upscales=_normalize_pending_upscales(row.get("pending_upscales")),
        observations=tuple(observations),
        window_ms=_optional_float(row.get("window_ms")),
        window_terms=dict(row["window_terms"]) if isinstance(row.get("window_terms"), dict) else {},
        gateway_baseline=_baseline_from_record(row.get("gateway_baseline")),
        preempt_reason=str(row["preempt_reason"]) if row.get("preempt_reason") else None,
    )


def _baseline_from_record(raw: Any) -> tuple[float, float] | None:
    if not isinstance(raw, (list, tuple)) or len(raw) != 2:
        return None
    requests, errors = _optional_float(raw[0]), _optional_float(raw[1])
    if requests is None or errors is None:
        return None
    return (requests, errors)


def _normalize_pods(row: dict[str, Any]) -> tuple[str, ...]:
    raw_pods = row.get("pods")
    if isinstance(raw_pods, (list, tuple)):
        pods = tuple(str(item).strip() for item in raw_pods if str(item).strip())
        if pods:
            return pods
    targets = row.get("target_instances")
    if isinstance(targets, list):
        pods = []
        for item in targets:
            if not isinstance(item, dict):
                continue
            pod = str(item.get("pod_name") or item.get("serve_name") or "").strip()
            if pod:
                pods.append(pod)
        if pods:
            return tuple(pods)
    hidden_count = row.get("hidden_count")
    try:
        count = int(hidden_count)
    except (TypeError, ValueError):
        count = 0
    if count > 0:
        return tuple(f"hidden-{idx}" for idx in range(count))
    return ()


def _time_ms(row: dict[str, Any], ms_key: str, seconds_key: str) -> int:
    if row.get(ms_key) is not None:
        return int(float(row[ms_key]))
    return int(float(row[seconds_key]) * 1000)


def _observation_from_record(raw: dict[str, Any]) -> ProbeObservation | None:
    try:
        ts_ms = int(float(raw.get("ts_ms", raw.get("ts", 0))))
    except (TypeError, ValueError):
        return None
    return ProbeObservation(
        ts_ms=ts_ms,
        ttft_p95_ms=_optional_float(raw.get("ttft_p95_ms")),
        tpot_p95_ms=_optional_float(raw.get("tpot_p95_ms")),
        z_m=_optional_float(raw.get("z_m")),
        q_ctl=_optional_float(raw.get("q_ctl", raw.get("Q_ctl"))),
        has_traffic=bool(raw.get("has_traffic", False)),
        avg_gpu_cache_norm=_optional_float(raw.get("avg_gpu_cache_norm")),
        gateway_requests=_optional_float(raw.get("gateway_requests")),
        gateway_errors=_optional_float(raw.get("gateway_errors")),
    )


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(parsed):
        return None
    return parsed


def _normalize_pending_upscales(raw: dict[str, int] | None | Any) -> dict[str, int]:
    if not isinstance(raw, dict):
        return {}
    normalized: dict[str, int] = {}
    for model, value in raw.items():
        try:
            delta = int(value)
        except (TypeError, ValueError):
            continue
        if model and delta > 0:
            normalized[str(model)] = delta
    return normalized
