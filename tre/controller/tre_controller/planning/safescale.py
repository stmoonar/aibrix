from __future__ import annotations

import json
import logging
import math
import time
from dataclasses import dataclass, field, replace
from typing import Any, Callable, Literal, Protocol

from tre_controller.config import SafeScaleConfig
from tre_controller.planning.safescale_evidence import (
    EvidenceSource,
    EvidenceWindow,
    HideAnchor,
    ThresholdResolver,
    anchor_clock_offsets,
    anchor_reference_ms,
    config_thresholds,
    evidence_start,
    log_clock_skew_alert,
)

LOG = logging.getLogger("tre_controller.safescale")

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
    #: P1-2 audit: the metrics window this observation read (``ModelWindowMetrics``
    #: ``window_start_ms`` / ``window_end_ms``, epoch ms on the snapshot clock - the
    #: same clock as ``ts_ms`` and the probe's ``start_ms``). None = unknown (an
    #: observation journalled by an older controller).
    window_start_ms: int | None = None
    window_end_ms: int | None = None
    #: Mean prompt length (tokens) of the window: ``request_prompt_tokens`` sum / count
    #: delta (the L of the labels-mode TTFT threshold). None = no requests / unknown.
    mean_prompt_tokens: float | None = None


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
class SafeScaleCommand:
    kind: CommandKind
    model: str
    pods: tuple[str, ...] = ()
    delta: int = 0
    reason: str = ""
    #: scale_down (commit) only: the SM drain budget (s) for the hidden probe pods.
    drain_budget_s: float | None = None


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
    #: "probing" (observed every tick) or "committing" (review 4 P2-4): its
    #: commit / rollback was handed to the action queue; resolved only when the
    #: queue finished it. A committing probe still owns its hidden pods.
    status: Literal["probing", "committing"] = "probing"
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
    #: While committing: the decision handed to the queue ("commit" / "rollback").
    resolution: str | None = None
    resolution_reason: str | None = None
    #: While committing: when the decision was made (ms, snapshot clock = epoch ms;
    #: persisted as ``committing_ts`` in s). B8: the queue refuses to act on a commit
    #: whose decision is older than ``commit_max_age_ms`` at its first dispatch.
    committing_ms: int | None = None
    #: P3: the probe's hide never took effect (failed, or skipped by the observe
    #: re-check right before the SM call): the probe is rolled back with this
    #: reason instead of being judged as if its pods were hidden.
    abort_reason: str | None = None
    #: When the hide took effect (Redis TIME right after the SM confirmed it, see
    #: :meth:`SafeScaleStateMachine.mark_hidden`). None = not confirmed yet.
    hide_anchor: HideAnchor | None = None
    #: Wall clock (ms) at probe start, for ``probe_wall_clock_ms`` in the audit.
    start_wall_ms: int | None = None
    #: Deadline extensions (one gateway period each) while the evidence was short.
    extensions: int = 0
    #: Where the window W counts from: the last gateway boundary before the confirmed
    #: hide (= start_ms unless the hide was confirmed late). None = start_ms.
    window_base_ms: int | None = None


@dataclass(frozen=True)
class ProbeTailSummary:
    latency_ok: bool
    z_min: float | None
    has_traffic: bool
    sample_count: int
    tail_count: int
    gpu_cache_max: float | None = None
    #: P1-2 audit (see :func:`tail_pre_hide_stats`): mean / max over the tail
    #: observations of the share of their metrics window that precedes the hide.
    pre_hide_fraction_mean: float | None = None
    pre_hide_fraction_max: float | None = None


class SafeScaleStateMachine:
    def __init__(
        self,
        *,
        config: SafeScaleConfig,
        store: ProbeStore | None = None,
        evidence: EvidenceSource | None = None,
        thresholds: ThresholdResolver | None = None,
        wall_clock_ms: Callable[[], int] | None = None,
    ) -> None:
        """``evidence`` (the live controller: ``MetricsEvidenceReader``) switches the
        latency part of the commit gate to the post-hide evidence window (2026-09-29);
        without it (direct constructions / offline replays) the gate reads the tail of
        the snapshot observations as before. ``thresholds`` resolves the per-model
        latency thresholds from the registry (``RegistryThresholds``); without it the
        config values (or 500 / 75 ms) apply."""
        self._config = config
        self._store = store
        self._evidence = evidence
        self._thresholds = thresholds
        self._wall_clock_ms = wall_clock_ms or (lambda: int(time.time() * 1000))
        self._probes: dict[str, SafeScaleProbe] = {}
        # A13 rollback backoff: model -> time (ms, snapshot clock) of its last rollback.
        # Deliberately in-memory only: a controller restart forgets it, i.e. at most one
        # extra HIGH probe per model right after a restart (the probe itself is still
        # guarded by SLO / donor-health / commit gate). Not worth a persisted schema.
        self._last_rollback_ms: dict[str, int] = {}
        #: request_id -> committing-probe recoveries submitted by this process.
        self._recoveries: dict[str, int] = {}

    def active_probe(self, model: str) -> SafeScaleProbe | None:
        return self._probes.get(model)

    def active_probes(self) -> tuple[SafeScaleProbe, ...]:
        """Probes still observed (status ``probing``)."""
        return tuple(probe for probe in self._probes.values() if probe.status == "probing")

    def committing_probes(self) -> tuple[SafeScaleProbe, ...]:
        """Probes whose resolution the action queue has not finished yet."""
        return tuple(probe for probe in self._probes.values() if probe.status == "committing")

    def all_probes(self) -> tuple[SafeScaleProbe, ...]:
        """Every unresolved probe, probing or committing."""
        return tuple(self._probes.values())

    def busy_models(self) -> set[str]:
        """Models with a probe in any state: their hidden pods are the probe's
        (never planner donors, no new probe) until it is resolved."""
        return set(self._probes)

    def mark_committing(
        self,
        model: str,
        *,
        status: Literal["commit", "rollback"],
        reason: str,
        now_ms: int,
    ) -> bool:
        """The probe's commit / rollback was accepted by the action queue (review
        4 P2-4): persist it as ``committing`` (with the decision) and stop
        observing it; :meth:`resolve_request` resolves it once the queue is done.
        A controller restart finds it in the store and re-submits the decision."""
        probe = self._probes.get(model)
        if probe is None:
            return False
        probe = replace(
            probe, status="committing", resolution=status, resolution_reason=reason, committing_ms=int(now_ms)
        )
        self._probes[model] = probe
        if self._store is not None:
            # committing_ts is written by _probe_record (from committing_ms).
            self._store.save_probe(probe.request_id, _probe_record(probe, terminal_reason=reason, status="committing"))
        return True

    def resolve_request(
        self,
        request_id: str,
        *,
        status: Literal["commit", "rollback"],
        reason: str,
        now_ms: int,
    ) -> bool:
        """Resolve the probe ``request_id`` (the action queue finished it)."""
        for model, probe in list(self._probes.items()):
            if probe.request_id == request_id:
                self._recoveries.pop(request_id, None)
                return self.resolve(model, status=status, reason=reason, now_ms=now_ms)
        return False

    def note_recovery(self, request_id: str, *, max_attempts: int = 5) -> bool:
        """Count one re-submission of a committing probe; False once
        ``max_attempts`` were made (the caller then resolves it)."""
        attempts = self._recoveries.get(request_id, 0) + 1
        self._recoveries[request_id] = attempts
        return attempts <= max_attempts

    def request_preemption(self, model: str, *, reason: str = "receiver_need_upscale") -> int:
        """v1 apply_safescale_to_deltas: a scale-up of a model that is itself probing rolls
        its probe back first. Marks the probe; its next observation returns the rollback
        (unhide, reason ``reason``). Returns how many pods that restores (0 = no probe)."""
        probe = self._probes.get(model)
        if probe is None or probe.status != "probing":
            # A committing probe is no longer observed: its rollback would never
            # be issued. The action queue preempts a committing commit itself.
            return 0
        if probe.preempt_reason is None:
            probe = replace(probe, preempt_reason=reason)
            self._probes[model] = probe
            self._persist_probe(probe)
        return len(probe.pods)

    def abort_probe(self, model: str, *, pods: tuple[str, ...], reason: str) -> bool:
        """P3: the hide of ``pods`` did not take effect - mark the ``probing`` probe
        owning them for rollback (the SafeScale loop rolls it back, unhiding its
        pods idempotently). False when no probing probe of ``model`` owns them."""
        probe = self._probes.get(model)
        if probe is None or probe.status != "probing" or not set(pods) & set(probe.pods):
            return False
        if probe.abort_reason is None:
            probe = replace(probe, abort_reason=reason)
            self._probes[model] = probe
            self._persist_probe(probe)
        return True

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
        # A6: adaptive window W = min(max(e2e_multiplier * p95_e2e, floor), W_max).
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
            start_wall_ms=self._wall_ms(),
        )
        self._probes[model] = probe
        self._persist_probe(probe)
        return SafeScaleDecision(
            status="probing",
            reason="probe_started",
            commands=(SafeScaleCommand(kind="hide", model=model, pods=probe.pods, reason="probe_started"),),
            details=dict(terms),
        )

    def mark_hidden(self, model: str, *, pods: tuple[str, ...], anchor: HideAnchor | None = None) -> bool:
        """The SM confirmed the hide of ``pods`` (ActionQueue ``on_hide_done``): anchor
        the probe's evidence at this moment - Redis TIME and the model's newest gateway
        doc stamp via the evidence source (see ``safescale_evidence``), else the
        controller clock. Only the first confirmation counts (a retried hide does not
        move the anchor). False when no probing probe of ``model`` owns the pods."""
        probe = self._probes.get(model)
        if probe is None or probe.status != "probing" or not set(pods) & set(probe.pods):
            return False
        if probe.hide_anchor is not None:
            return True
        if anchor is None:
            if self._evidence is not None:
                try:
                    anchor = self._evidence.hide_anchor(model)
                except Exception:  # noqa: BLE001 - fall back to the local clock (recorded)
                    LOG.warning("safescale hide anchor of %s failed; using the controller clock", model, exc_info=True)
            if anchor is None:
                now = int(self._wall_ms() or 0)
                # With an evidence source, a clock-only anchor cannot be verified
                # against the gateway stamps: the probe rolls back (anchor_unverified).
                anchor = HideAnchor(
                    ts_ms=now, source="controller_clock", controller_ts_ms=now,
                    newest_doc_error=self._evidence is not None,
                )
        step = _evidence_step_ms(self._config)
        offsets = anchor_clock_offsets(anchor, period_ms=step, tolerance_ms=_clock_tolerance_ms(self._config))
        if offsets["clock_skew_alert"]:
            log_clock_skew_alert(model, probe.request_id, anchor, offsets)
        # W counts from the hide: a hide confirmed late (queued behind a long action)
        # moves the window (and its ceiling) with it; the normal case (confirmed within
        # the planning period) is unchanged. On the controller clock, like the deadline
        # it is compared with (snapshot boundaries), never on the gateway stamps.
        confirmed = anchor.controller_ts_ms if anchor.controller_ts_ms is not None else anchor.ts_ms
        base = max(int(probe.start_ms), int(confirmed) // step * step)
        deadline = max(int(probe.deadline_ms), base + int(probe.window_ms or 0))
        probe = replace(
            probe,
            hide_anchor=anchor,
            window_base_ms=base,
            deadline_ms=deadline,
            window_terms={**probe.window_terms, **_anchor_terms(anchor, step), **offsets, "window_base_ms": base},
        )
        self._probes[model] = probe
        self._persist_probe(probe)
        return True

    def observe(self, model: str, observation: ProbeObservation, *, now_ms: int) -> SafeScaleDecision:
        """One SafeScale tick (every ``probe_poll_seconds``) for ``model``'s probe.

        Snapshots are published once per gateway period and re-read every tick, so an
        observation is appended (journalled, counted in the hq tail) only once per
        snapshot, keyed by its ``window_end_ms``. Preemption / abort and the donor-health
        guard run on every tick (the gateway counters are fresh each time). The
        immediate SLO rollback judges each snapshot once, and only a snapshot whose
        whole window follows the hide (``window_start_ms >= hide``); earlier ones are
        recorded, not judged. At the deadline the commit gate runs (:meth:`_judge`)."""
        probe = self._probes.get(model)
        if probe is None:
            return SafeScaleDecision(status="none", reason="probe_not_found")
        if probe.status != "probing":
            return SafeScaleDecision(status="none", reason="probe_committing")

        key = _observation_key(observation)
        is_new = all(_observation_key(seen) != key for seen in probe.observations)
        updated = _replace_observations(probe, probe.observations + (observation,)) if is_new else probe
        updated = _with_gateway_baseline(updated, observation)
        self._probes[model] = updated
        if is_new:
            self._persist_observation(updated, observation)

        health = donor_health(updated, observation)
        if updated.preempt_reason is not None:
            return self._rollback_now(
                updated,
                reason=updated.preempt_reason,
                details={"preempted": updated.preempt_reason},
                rollback_reason={"code": "preempted", "detail": updated.preempt_reason},
            )
        if updated.abort_reason is not None:
            return self._rollback_now(
                updated,
                reason=updated.abort_reason,
                details={"aborted": updated.abort_reason},
                rollback_reason={"code": "hide_failed", "detail": updated.abort_reason},
            )
        if self._evidence is not None and updated.hide_anchor is not None and updated.hide_anchor.newest_doc_error:
            # The evidence start cannot be anchored on the gateway stamps: fail closed
            # now instead of keeping the pods hidden until the deadline.
            return self._rollback_now(
                updated,
                reason="evidence_clock_skew",
                details={"anchor": updated.hide_anchor.as_record()},
                rollback_reason={
                    "code": "evidence_clock_skew", "check": "anchor_unverified",
                    "hide_ts_ms": int(updated.hide_anchor.ts_ms),
                },
            )
        violation = self._instant_violation(updated, observation) if is_new else None
        if violation is not None:
            return self._rollback_now(
                updated,
                reason="slo_violation",
                details={"slo": {"ttft_p95_ms": observation.ttft_p95_ms, "tpot_p95_ms": observation.tpot_p95_ms}},
                rollback_reason=violation,
            )

        if (
            health is not None
            and health["requests"] >= self._config.donor_min_requests
            and health["error_rate"] > self._config.donor_error_rate_max
        ):
            return self._rollback_now(
                updated,
                reason="donor_health",
                details={"donor_health": health},
                rollback_reason={"code": "donor_health", **health},
            )

        if now_ms < updated.deadline_ms:
            if updated is not probe:
                self._persist_probe(updated)
            return SafeScaleDecision(status="probing", reason="probe_pending")
        return self._judge(updated, health, now_ms=now_ms)

    def _judge(self, probe: SafeScaleProbe, health: dict[str, float] | None, *, now_ms: int) -> SafeScaleDecision:
        """The formal commit gate (v1 _tail_summary_allows_commit) at the deadline.

        Z (tail min) and the KV-cache fill come from the hq tail of the snapshot
        observations, as before. The latency check reads the post-hide evidence window
        (:meth:`_evidence_outcome`) when an evidence source is wired; without one (direct
        constructions) it reads the tail snapshots, as before 2026-09-29."""
        model = probe.model
        evidence_mode = self._evidence is not None
        summary = _summarize_tail(
            probe,
            hq=self._config.hq,
            thresholds=lambda observation: self._thresholds_for(model, observation.mean_prompt_tokens),
            judge_latency=not evidence_mode,
            hide_ts_ms=self._post_hide_start(probe),
        )
        tail_audit = {
            "tail_pre_hide_fraction_mean": summary.pre_hide_fraction_mean,
            "tail_pre_hide_fraction_max": summary.pre_hide_fraction_max,
            "tail_observation_count": summary.tail_count,
        }
        idle = False
        if evidence_mode:
            outcome = self._evidence_outcome(probe, summary, now_ms=now_ms)
            if outcome.kind == "extend":
                return self._extend(probe, outcome.audit)
            if outcome.kind == "rollback":
                return self._rollback_now(
                    probe,
                    reason=str(outcome.rollback_reason.get("code")),
                    details={"evidence": outcome.audit},
                    rollback_reason=outcome.rollback_reason,
                    audit={**tail_audit, **outcome.audit},
                )
            summary = replace(summary, latency_ok=outcome.latency_ok)
            latency_audit = outcome.audit
            idle = outcome.idle
        else:
            thresholds = self._thresholds_for(model, _last_mean_prompt(probe))
            latency_audit = {
                "latency_source": "tail_snapshots",
                "latency_gate": "evaluated",
                "threshold_mode": thresholds["mode"],
                "ttft_threshold_ms": thresholds["ttft_ms"],
                "tpot_threshold_ms": thresholds["tpot_ms"],
                # The tail snapshots WERE the latency evidence.
                "tail_pre_hide_fraction": summary.pre_hide_fraction_max,
                "extensions": probe.extensions,
                "clamped": bool(probe.window_terms.get("clamped")),
            }
        failures = () if idle else tail_gate_failures(
            summary, tau_low=self._config.tau_low, kv_cache_max=self._config.kv_cache_max
        )
        audit = {**tail_audit, **latency_audit}
        details: dict[str, Any] = {
            "gate_failures": list(failures),
            "tail": _tail_record(summary),
            # P1-2 audit, also at the top level and in window_terms (reports read either).
            **audit,
            # A12/P2-a: no KV-cache sample in the tail (no pod reported the gauge) - the
            # gate passes this check like v1, but the record says so explicitly.
            "kv_cache": "unavailable" if summary.gpu_cache_max is None else summary.gpu_cache_max,
        }
        if idle:
            details["idle_commit"] = True
        if health is not None:
            details["donor_health"] = health
        if failures:
            rollback_reason: dict[str, Any] = {
                "code": "formal_commit_gate_failed",
                "gates": list(failures),
                "z_min": _tail_record(summary)["z_min"],
                "kv_cache_max": summary.gpu_cache_max,
                "latency": {
                    key: audit.get(key)
                    for key in (
                        "latency_gate", "latency_samples", "evidence_ttft_p95_ms", "evidence_tpot_p95_ms",
                        "ttft_threshold_ms", "tpot_threshold_ms",
                    )
                    if key in audit
                },
            }
            audit["rollback_reason"] = rollback_reason
            details["rollback_reason"] = rollback_reason
        audit["probe_wall_clock_ms"] = details["probe_wall_clock_ms"] = self._probe_wall_clock_ms(probe)
        updated = replace(probe, terminal_details=details, window_terms={**probe.window_terms, **audit})
        self._probes[model] = updated
        if not failures:
            decision = self._commit(updated, reason="formal_commit_gate_passed")
        else:
            decision = self._rollback(updated, reason="formal_commit_gate_failed")
        # The commit / rollback decision record carries the audit too.
        return replace(decision, details={**decision.details, **audit})

    def _evidence_outcome(
        self, probe: SafeScaleProbe, summary: "ProbeTailSummary", *, now_ms: int
    ) -> "_EvidenceOutcome":
        """The latency verdict of the post-hide evidence window (module docstring of
        ``safescale_evidence``), or "extend" (deadline + one gateway period, up to the
        W ceiling) while the evidence is short, or "rollback" (fail-closed)."""
        cfg = self._config
        model = probe.model
        step = _evidence_step_ms(cfg)
        cap = _deadline_cap(probe, cfg)
        can_extend = probe.deadline_ms < cap
        window_clamped = bool(probe.window_terms.get("window_clamped", probe.window_terms.get("clamped")))
        audit: dict[str, Any] = {
            "latency_source": "evidence",
            "extensions": probe.extensions,
            "window_clamped": window_clamped,
            "clamped": window_clamped,
            "deadline_cap_ms": cap,
        }
        anchor = probe.hide_anchor
        if anchor is None:
            if can_extend:
                return _EvidenceOutcome("extend", audit={**audit, "extend_reason": "hide_unconfirmed"})
            return _EvidenceOutcome(
                "rollback", audit=audit, rollback_reason={"code": "hide_unconfirmed", "deadline_cap_ms": cap}
            )
        hide_ts = int(anchor.ts_ms)
        start = evidence_start(anchor, step)
        end = _latest_window_end(probe)
        if end is None:
            end = int(now_ms)
        audit.update(
            evidence_start_ms=start,
            evidence_end_ms=end,
            hide_ts_ms=hide_ts,
            hide_anchor_source=anchor.source,
            evidence_anchor="gateway_doc" if anchor.newest_doc_ts_ms is not None else anchor.source,
        )
        if anchor.newest_doc_error:
            # The newest gateway doc could not be read at the hide: S cannot be anchored
            # on the gateway stamps and a clock-based S is unverified -> fail closed.
            return _EvidenceOutcome(
                "rollback", audit=audit,
                rollback_reason={"code": "evidence_clock_skew", "check": "anchor_unverified", "hide_ts_ms": hide_ts},
            )
        if end <= start:
            if can_extend:
                return _EvidenceOutcome("extend", audit={**audit, "extend_reason": "evidence_empty"})
            # A snapshot grid that never reaches the gateway's post-hide boundary within
            # W_max: the controller clock lags the gateway stamps (or snapshots stalled).
            LOG.error(
                json.dumps(
                    {"event": "safescale_evidence_empty", "model": model, "request_id": probe.request_id,
                     "evidence_start_ms": start, "evidence_end_ms": end, **anchor.as_record()},
                    sort_keys=True,
                )
            )
            return _EvidenceOutcome(
                "rollback", audit=audit,
                rollback_reason={"code": "evidence_empty", "evidence_start_ms": start, "evidence_end_ms": end},
            )
        try:
            evidence = self._evidence.read(model, start_ms=start, end_ms=end, exclude_pods=tuple(probe.pods))
        except Exception as exc:  # noqa: BLE001 - fail closed
            LOG.error("safescale evidence of %s unreadable: %r", model, exc)
            return _EvidenceOutcome(
                "rollback", audit=audit, rollback_reason={"code": "evidence_unavailable", "error": repr(exc)}
            )
        tolerance = _clock_tolerance_ms(cfg)
        clock = evidence_clock_failure(anchor, evidence, tolerance_ms=tolerance, period_ms=step)
        first_doc = min(evidence.first_doc_ts_ms.values()) if evidence.first_doc_ts_ms else None
        audit.update(
            evidence_first_doc_ts_ms=first_doc,
            evidence_pods=list(evidence.pods),
            evidence_excluded_pods=list(evidence.excluded_pods),
            # Share of the latency evidence that precedes the first post-hide boundary.
            tail_pre_hide_fraction=_pre_hide_fraction(start, first_doc if first_doc is not None else start, end),
        )
        if clock is not None:
            LOG.error(
                json.dumps(
                    {"event": "safescale_evidence_clock_skew", "model": model, "request_id": probe.request_id,
                     **clock},
                    sort_keys=True,
                )
            )
            return _EvidenceOutcome(
                "rollback", audit=audit, rollback_reason={"code": "evidence_clock_skew", **clock}
            )
        samples = float(evidence.ttft_count)
        judged = float(evidence.judged)
        mean_prompt = evidence.mean_prompt_tokens
        thresholds = self._thresholds_for(model, mean_prompt)
        min_samples = int(getattr(cfg, "min_commit_samples", 20))
        audit.update(
            latency_samples=samples,
            latency_samples_judged=judged,
            min_commit_samples=min_samples,
            mean_prompt_tokens=mean_prompt,
            evidence_ttft_p95_ms=evidence.ttft_p95_ms,
            evidence_tpot_p95_ms=evidence.tpot_p95_ms,
            threshold_mode=thresholds["mode"],
            threshold_source=thresholds.get("source"),
            ttft_threshold_ms=thresholds["ttft_ms"],
            tpot_threshold_ms=thresholds["tpot_ms"],
        )
        if "fallback" in thresholds:
            audit["threshold_fallback"] = thresholds["fallback"]
        p95_available = evidence.ttft_p95_ms is not None or evidence.tpot_p95_ms is not None
        # n counts only the pods whose p95 is judged (per-pod minimum samples, the
        # snapshot rule): a pod below it neither decides the p95 nor fills the quota.
        if min_samples <= 0 or (judged >= min_samples and p95_available):
            violations = []
            if evidence.ttft_p95_ms is not None and evidence.ttft_p95_ms > thresholds["ttft_ms"]:
                violations.append("ttft")
            if evidence.tpot_p95_ms is not None and evidence.tpot_p95_ms > thresholds["tpot_ms"]:
                violations.append("tpot")
            audit.update(latency_gate="evaluated", latency_violations=violations)
            return _EvidenceOutcome("judge", audit=audit, latency_ok=not violations)
        # Too few requests at all, or enough but spread over pods below the per-pod
        # p95 minimum (no p95 to judge them by).
        short = "insufficient_samples" if samples < min_samples else "p95_unavailable"
        if not evidence.first_doc_ts_ms and not can_extend:
            # No doc of any remaining pod in [S, E] by W_max: the gateway wrote nothing
            # for them (it writes cumulative docs for every pod every period, idle ones
            # included) - no evidence is not evidence of health.
            LOG.error(
                json.dumps(
                    {"event": "safescale_evidence_empty", "model": model, "request_id": probe.request_id,
                     "evidence_start_ms": start, "evidence_end_ms": end, "pods": list(evidence.pods),
                     **anchor.as_record()},
                    sort_keys=True,
                )
            )
            return _EvidenceOutcome(
                "rollback", audit=audit,
                rollback_reason={"code": "evidence_empty", "evidence_start_ms": start, "evidence_end_ms": end,
                                 "check": "no_docs"},
            )
        if can_extend:
            return _EvidenceOutcome("extend", audit={**audit, "extend_reason": short})
        # At the W ceiling with too little evidence.
        audit["clamped"] = True
        if samples <= 0 and not summary.has_traffic:
            # v1 idle rule: nothing served and nothing in flight -> commit.
            audit.update(latency_gate="skipped", latency_skip_reason="idle")
            return _EvidenceOutcome("judge", audit=audit, latency_ok=True, idle=True)
        audit.update(latency_gate="skipped", latency_skip_reason=short)
        return _EvidenceOutcome("judge", audit=audit, latency_ok=True)

    def _extend(self, probe: SafeScaleProbe, audit: dict[str, Any]) -> SafeScaleDecision:
        cap = _deadline_cap(probe, self._config)
        # One period past the newest evidence, so a jump of the snapshot clock (after a
        # stale hold) needs a NEW snapshot before the next evaluation - no extension
        # is burnt on the same data.
        newest = max(int(probe.deadline_ms), int(audit.get("evidence_end_ms") or _latest_window_end(probe) or 0))
        deadline = min(newest + _evidence_step_ms(self._config), cap)
        extensions = probe.extensions + 1
        audit = {**audit, "extensions": extensions, "deadline_ms": deadline}
        updated = replace(
            probe, deadline_ms=deadline, extensions=extensions, window_terms={**probe.window_terms, **audit}
        )
        self._probes[probe.model] = updated
        self._persist_probe(updated)
        return SafeScaleDecision(status="probing", reason="evidence_extended", details=audit)

    def _rollback_now(
        self,
        probe: SafeScaleProbe,
        *,
        reason: str,
        details: dict[str, Any],
        rollback_reason: dict[str, Any],
        audit: dict[str, Any] | None = None,
    ) -> SafeScaleDecision:
        """A rollback decided before / instead of the formal gate, with its structured
        reason in terminal_details, window_terms and the decision details."""
        record = {
            **(audit or {}),
            "rollback_reason": rollback_reason,
            "probe_wall_clock_ms": self._probe_wall_clock_ms(probe),
        }
        updated = replace(
            probe,
            terminal_details={**details, **record},
            window_terms={**probe.window_terms, **record},
        )
        self._probes[probe.model] = updated
        decision = self._rollback(updated, reason=reason)
        return replace(decision, details={**decision.details, **record})

    def _instant_violation(self, probe: SafeScaleProbe, observation: ProbeObservation) -> dict[str, Any] | None:
        """The immediate SLO rollback of one snapshot, judged only when its whole
        window follows the hide: its baseline doc is stamped at or after the first
        gateway boundary after the hide (window_start_ms >= S). None = no violation /
        not judged (pre-hide or overlapping window, unknown window, hide unconfirmed)."""
        hide_ts = self._post_hide_start(probe)
        start = observation.window_start_ms
        if hide_ts is None or start is None or int(start) < int(hide_ts):
            return None
        thresholds = self._thresholds_for(probe.model, observation.mean_prompt_tokens)
        metrics = []
        if observation.ttft_p95_ms is not None and observation.ttft_p95_ms > thresholds["ttft_ms"]:
            metrics.append("ttft")
        if observation.tpot_p95_ms is not None and observation.tpot_p95_ms > thresholds["tpot_ms"]:
            metrics.append("tpot")
        if not metrics:
            return None
        return {
            "code": "slo_violation",
            "metrics": metrics,
            "ttft_p95_ms": observation.ttft_p95_ms,
            "tpot_p95_ms": observation.tpot_p95_ms,
            "ttft_threshold_ms": thresholds["ttft_ms"],
            "tpot_threshold_ms": thresholds["tpot_ms"],
            "threshold_mode": thresholds["mode"],
            "window_start_ms": observation.window_start_ms,
            "window_end_ms": observation.window_end_ms,
            "evidence_start_ms": hide_ts,
        }

    def _post_hide_start(self, probe: SafeScaleProbe) -> int | None:
        """The first gateway boundary after the confirmed hide (doc-stamp domain, the
        grid snapshot windows are read on). Without an evidence source nothing confirms
        the hide: the planned start. None = hide not confirmed yet."""
        if probe.hide_anchor is not None:
            return evidence_start(probe.hide_anchor, _evidence_step_ms(self._config))
        return None if self._evidence is not None else int(probe.start_ms)

    def _thresholds_for(self, model: str, mean_prompt_tokens: float | None) -> dict[str, Any]:
        if self._thresholds is not None:
            try:
                return self._thresholds.resolve(model, mean_prompt_tokens)
            except Exception:  # noqa: BLE001 - config values are the documented fallback
                LOG.warning("safescale thresholds of %s failed; using the config values", model, exc_info=True)
        return config_thresholds(self._config, mean_prompt_tokens)

    def _wall_ms(self) -> int | None:
        try:
            return int(self._wall_clock_ms())
        except Exception:  # noqa: BLE001 - audit only
            return None

    def _probe_wall_clock_ms(self, probe: SafeScaleProbe) -> int | None:
        now = self._wall_ms()
        if now is None or probe.start_wall_ms is None:
            return None
        return max(0, now - int(probe.start_wall_ms))

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
                # Plan D1: drain_budget_s = W is still sent to the SM, but it is ignored
                # while safescale_commit is in service_manager.sleep.no_drain_paths (the
                # default): SleepPolicy.soft_budget_s (tre_common/registry.py) returns 0
                # for no-drain paths whatever budget the caller passes. The probe window,
                # with the pods already hidden, was the drain (v1 / paper, 2026-09-29).
                drain_budget_s=(
                    float(probe.window_ms) / 1000.0 if probe.window_ms else None
                ),
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

    def _persist_probe(self, probe: SafeScaleProbe, *, terminal_reason: str | None = None) -> None:
        if self._store is None:
            return
        self._store.save_probe(
            probe.request_id,
            _probe_record(probe, terminal_reason=terminal_reason, status=probe.status),
        )

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


@dataclass(frozen=True)
class _EvidenceOutcome:
    kind: Literal["judge", "extend", "rollback"]
    audit: dict[str, Any] = field(default_factory=dict)
    latency_ok: bool = True
    idle: bool = False
    rollback_reason: dict[str, Any] = field(default_factory=dict)


def _observation_key(observation: ProbeObservation) -> int:
    """Observations are deduplicated per metrics snapshot: its window end, else its ts."""
    if observation.window_end_ms is not None:
        return int(observation.window_end_ms)
    return int(observation.ts_ms)


def _anchor_terms(anchor: HideAnchor, period_ms: int) -> dict[str, Any]:
    return {
        "hide_ts_ms": int(anchor.ts_ms),
        "hide_anchor_source": anchor.source,
        "hide_newest_doc_ts_ms": anchor.newest_doc_ts_ms,
        "evidence_start_ms": evidence_start(anchor, period_ms),
    }


def _clock_tolerance_ms(config: SafeScaleConfig) -> float:
    return float(getattr(config, "evidence_clock_tolerance_ms", 20_000.0) or 20_000.0)


def _evidence_step_ms(config: SafeScaleConfig) -> int:
    return max(1, int(getattr(config, "evidence_step_ms", 10_000.0) or 10_000))


def _deadline_cap(probe: SafeScaleProbe, config: SafeScaleConfig) -> int:
    """The latest deadline: probe start + W_max (the W ceiling, never below the floor).
    Without a ceiling the deadline is never extended."""
    ceiling = _positive(getattr(config, "window_ceiling_ms", None))
    if ceiling is None:
        return int(probe.deadline_ms)
    cap_ms = max(float(ceiling), float(config.min_window_ms))
    base = probe.window_base_ms if probe.window_base_ms is not None else probe.start_ms
    return max(int(probe.deadline_ms), int(base + cap_ms))


def _latest_window_end(probe: SafeScaleProbe) -> int | None:
    ends = [int(obs.window_end_ms) for obs in probe.observations if obs.window_end_ms is not None]
    return max(ends) if ends else None


def _last_mean_prompt(probe: SafeScaleProbe) -> float | None:
    for observation in reversed(probe.observations):
        if observation.mean_prompt_tokens is not None:
            return observation.mean_prompt_tokens
    return None


def _pre_hide_fraction(hide_ts_ms: int, start_ms: int, end_ms: int) -> float:
    """Share of the latency evidence window ``[start, end]`` (start = its first doc)
    that precedes ``hide_ts_ms`` (the first post-hide boundary). 0 by construction:
    a regression assertion in every probe record."""
    span = float(end_ms) - float(start_ms)
    if span <= 0:
        return 0.0
    return min(1.0, max(0.0, float(hide_ts_ms) - float(start_ms)) / span)


def evidence_clock_failure(
    anchor: HideAnchor, evidence: EvidenceWindow, *, tolerance_ms: float, period_ms: int = 10_000
) -> dict[str, Any] | None:
    """Fail-closed continuity / clock check of the evidence window (None = passed).

    Every remaining pod's delta must start at a doc stamped within
    ``[S, hide + tolerance]`` - ``S`` = the first boundary after the hide, ``hide`` =
    the newest doc stamp at the hide (doc-stamp domain; hide_ts when there was none).
    A later first doc means missing gateway ticks (the pod's post-hide evidence starts
    too late: more than one tick missed with the default 20 s), an earlier one a doc
    that predates the hide (``first_doc_outside``). Pods without a TTFT histogram in the
    window are not checked."""
    lower = evidence_start(anchor, period_ms)
    upper = int(anchor_reference_ms(anchor) + float(tolerance_ms))
    outside = {pod: int(ts) for pod, ts in sorted(evidence.first_doc_ts_ms.items()) if not lower <= int(ts) <= upper}
    if outside:
        return {
            "check": "first_doc_outside",
            "hide_ts_ms": int(anchor.ts_ms),
            "evidence_start_ms": lower,
            "latest_allowed_ms": upper,
            "tolerance_ms": float(tolerance_ms),
            "first_doc_ts_ms": outside,
        }
    return None


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


def calc_probe_window_details(
    inputs: ProbeWindowInputs,
    *,
    hidden_count: int,
    config: SafeScaleConfig,
) -> dict[str, Any]:
    """Probe window W = min(max(e2e_multiplier * p95_e2e_ms, min_window_ms), W_max).

    W_max = ``config.window_ceiling_ms`` (registry safescale.window_ceiling_s, 60 s by
    default; deadline extensions never pass it either; None = no ceiling), never below
    the floor. A
    missing / non-positive p95_e2e gives W = min_window_ms (no avg_ttft fallback).
    Returns W, W1 (multiplier * p95_e2e or None), W_floor, W_max, ``clamped`` (True when
    W_max cut W1), e2e_multiplier, ``dominant`` (e2e / floor / ceiling) and the inputs,
    for the probe record and events.
    """
    floor_ms = float(config.min_window_ms)
    multiplier = float(config.e2e_multiplier)
    ceiling = _positive(getattr(config, "window_ceiling_ms", None))
    ceiling_ms = max(ceiling, floor_ms) if ceiling is not None else None
    p95_e2e = _positive(inputs.p95_e2e_ms)
    w1 = multiplier * p95_e2e if p95_e2e is not None else None
    window = max(w1, floor_ms) if w1 is not None else floor_ms
    clamped = ceiling_ms is not None and window > ceiling_ms
    if clamped:
        window = ceiling_ms
    dominant = "ceiling" if clamped else ("e2e" if w1 is not None and w1 > floor_ms else "floor")
    return {
        "W": window,
        "W1": w1,
        "W_floor": floor_ms,
        "W_max": ceiling_ms,
        "clamped": clamped,
        "window_clamped": clamped,
        "e2e_multiplier": multiplier,
        "dominant": dominant,
        "inputs": {
            "p95_e2e_ms": p95_e2e,
            "hidden_count": int(hidden_count),
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
        f":dominant={terms.get('dominant')}:e2e={fmt(terms.get('W1'))}"
        f":floor={fmt(terms.get('W_floor'))}:max={fmt(terms.get('W_max'))}"
        f":clamped={fmt(bool(terms.get('clamped')))}"
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
    ttft_p95_slo_ms: float | None = None,
    tpot_p95_slo_ms: float | None = None,
    thresholds: Callable[[ProbeObservation], dict[str, Any]] | None = None,
    judge_latency: bool = True,
    hide_ts_ms: int | None = None,
) -> ProbeTailSummary:
    """The hq tail of the (one-per-snapshot) observations: Z min, traffic, KV max and -
    with ``judge_latency`` (no evidence source) - the tail latency check against
    ``thresholds(observation)`` or the fixed ``*_slo_ms`` values."""
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
    pre_hide = tail_pre_hide_stats(tail, hide_ts_ms=int(probe.start_ms if hide_ts_ms is None else hide_ts_ms))

    latency_ok = True
    has_traffic = False
    z_values: list[float] = []
    gpu_cache_values: list[float] = []
    for observation in tail:
        if observation.has_traffic:
            has_traffic = True
        if judge_latency:
            if thresholds is not None:
                limits = thresholds(observation)
                ttft_limit, tpot_limit = limits["ttft_ms"], limits["tpot_ms"]
            else:
                ttft_limit = 500.0 if ttft_p95_slo_ms is None else ttft_p95_slo_ms
                tpot_limit = 75.0 if tpot_p95_slo_ms is None else tpot_p95_slo_ms
            if observation.ttft_p95_ms is not None and observation.ttft_p95_ms > ttft_limit:
                latency_ok = False
            if observation.tpot_p95_ms is not None and observation.tpot_p95_ms > tpot_limit:
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
        pre_hide_fraction_mean=pre_hide["tail_pre_hide_fraction_mean"],
        pre_hide_fraction_max=pre_hide["tail_pre_hide_fraction_max"],
    )


def tail_pre_hide_stats(
    tail: tuple[ProbeObservation, ...] | list[ProbeObservation], *, hide_ts_ms: int
) -> dict[str, Any]:
    """P1-2 audit (the commit criterion is NOT changed by it): how much of the evidence
    the commit gate judged predates the hide.

    For each tail observation (the ``hq`` tail :func:`_summarize_tail` hands to the
    commit gate) its pre-hide share is ``max(0, hide_ts - window_start) / window_len``,
    capped at 1, where ``[window_start, window_end]`` is the metrics window that
    observation read (``ModelWindowMetrics``; sliding window of TRE_METRICS_WINDOW_MS
    ending at the snapshot's read boundary - the refresh period and read offset are
    therefore already in ``window_start``) and ``hide_ts`` is the probe's ``start_ms``:
    the snapshot time the probe (and its hide) was planned at. The hide reaches the
    gateway only after that, so the share is a LOWER bound of the real one.

    * ``tail_pre_hide_fraction_mean`` / ``_max``: mean / max of the shares over the
      tail observations that carry window timestamps (None when none does);
    * ``tail_observation_count``: how many observations the gate judged (the tail size,
      with or without timestamps).
    """
    fractions: list[float] = []
    for observation in tail:
        start, end = observation.window_start_ms, observation.window_end_ms
        if start is None or end is None or end <= start:
            continue
        share = max(0.0, float(hide_ts_ms) - float(start)) / float(end - start)
        fractions.append(min(1.0, share))
    return {
        "tail_pre_hide_fraction_mean": (sum(fractions) / len(fractions)) if fractions else None,
        "tail_pre_hide_fraction_max": max(fractions) if fractions else None,
        "tail_observation_count": len(tail),
    }


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
        "tail_pre_hide_fraction_mean": summary.pre_hide_fraction_mean,
        "tail_pre_hide_fraction_max": summary.pre_hide_fraction_max,
        "tail_observation_count": summary.tail_count,
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
    if probe.abort_reason is not None:
        record["abort_reason"] = probe.abort_reason
    if probe.hide_anchor is not None:
        record["hide_anchor"] = probe.hide_anchor.as_record()
    if probe.start_wall_ms is not None:
        record["start_wall_ms"] = probe.start_wall_ms
    if probe.extensions:
        record["extensions"] = probe.extensions
    if probe.window_base_ms is not None:
        record["window_base_ms"] = probe.window_base_ms
    if probe.resolution is not None and status == "committing":
        record["resolution"] = probe.resolution
        record["resolution_reason"] = probe.resolution_reason
        if probe.committing_ms is not None:
            record["committing_ts"] = float(probe.committing_ms) / 1000.0
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
        "window_start_ms": observation.window_start_ms,
        "window_end_ms": observation.window_end_ms,
        "mean_prompt_tokens": observation.mean_prompt_tokens,
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
            # One observation per snapshot (journals of older controllers hold one per tick).
            if observation is not None and all(
                _observation_key(seen) != _observation_key(observation) for seen in observations
            ):
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
        abort_reason=str(row["abort_reason"]) if row.get("abort_reason") else None,
        hide_anchor=HideAnchor.from_record(row.get("hide_anchor")),
        start_wall_ms=_optional_int(row.get("start_wall_ms")),
        extensions=int(_optional_int(row.get("extensions")) or 0),
        window_base_ms=_optional_int(row.get("window_base_ms")),
        **_committing_fields(row),
    )


def _committing_fields(row: dict[str, Any]) -> dict[str, Any]:
    """A probe persisted as ``committing`` (review 4 P2-4) is restored as such,
    with the decision that was handed to the action queue."""
    if str(row.get("status", "probing")) != "committing" or row.get("resolution") not in ("commit", "rollback"):
        return {}
    committing_ts = _optional_float(row.get("committing_ts"))
    return {
        "status": "committing",
        "resolution": str(row["resolution"]),
        "resolution_reason": str(row.get("resolution_reason") or row.get("terminal_reason") or ""),
        # B8: the decision time survives a restart, so a recovered commit is aged.
        "committing_ms": int(committing_ts * 1000.0) if committing_ts is not None else None,
    }


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
        window_start_ms=_optional_int(raw.get("window_start_ms")),
        window_end_ms=_optional_int(raw.get("window_end_ms")),
        mean_prompt_tokens=_optional_float(raw.get("mean_prompt_tokens")),
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


def _optional_int(value: Any) -> int | None:
    parsed = _optional_float(value)
    return int(parsed) if parsed is not None else None


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
