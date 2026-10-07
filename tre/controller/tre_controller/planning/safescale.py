from __future__ import annotations

import json
import logging
import math
import time
from dataclasses import dataclass, field, replace
from typing import Any, Callable, Collection, Literal, Mapping, Protocol

from tre_controller.config import SafeScaleConfig
from tre_controller.planning.safescale_direct import (
    SOURCE_DIRECT,
    SOURCE_REDIS,
    DirectPoll,
    DirectState,
    DirectWindow,
    PodScrape,
    evaluate_poll,
    take_baseline,
)
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
    #: p95 of the remaining pods' window histograms pooled first (minimum samples on the
    #: pooled count): the immediate rollback judges max(per pod, pooled), like the gate.
    pooled_ttft_p95_ms: float | None = None
    pooled_tpot_p95_ms: float | None = None


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
    #: Direct evidence: the confirmation itself (controller clock, not grid-aligned).
    window_base_ms: int | None = None
    #: Direct evidence path (``safescale.evidence_source: direct``): baseline, drops,
    #: scrape log, fallback (planning.safescale_direct). None = not started / redis.
    direct: DirectState | None = None
    #: Timer cleanup (2026-10-02): the decision state that started the probe - the
    #: donor's Z and routable count of the planner tick (``ProbeWindowInputs``). A
    #: capacity rollback keeps them as the evidence a retry must beat. None = unknown.
    start_z_m: float | None = None
    start_routable: int | None = None


#: Rollback codes that say the model could not spare the probe pods (the retry needs
#: new evidence: another routable count or a clearly higher Z). Every other rollback
#: (evidence gaps, hide failures, maintenance, observe mode, pods gone) says nothing
#: about capacity and only needs a metrics window after it.
CAPACITY_ROLLBACK_CODES = frozenset(
    {
        "slo_violation", "slo_violation_direct", "formal_commit_gate_failed", "donor_health",
        # Review P2-4: traffic in flight and not one request completed within W - the
        # remaining pods are saturated.
        "insufficient_evidence:stalled",
    }
)


@dataclass(frozen=True)
class RollbackEvidence:
    """Timer cleanup (2026-10-02): what a rolled-back probe of a model showed, kept
    until the model's next probe resolves. Replaces the fixed A13 rollback backoff."""

    rolled_back_ms: int
    reason: str
    capacity: bool
    #: Z / routable count of the decision that started the failed probe (None = unknown;
    #: an unknown Z is taken from the first window after the rollback).
    z_m: float | None
    routable: int | None


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
        remaining_pods: Callable[[str, tuple[str, ...]], Collection[str] | None] | None = None,
    ) -> None:
        """``evidence`` (the live controller: ``MetricsEvidenceReader``) switches the
        latency part of the commit gate to the post-hide evidence window (2026-09-29);
        without it (direct constructions / offline replays) the gate reads the tail of
        the snapshot observations as before. ``thresholds`` resolves the per-model
        latency thresholds from the registry (``RegistryThresholds``); without it the
        config values (or 500 / 75 ms) apply. ``remaining_pods(model, probe_pods)`` = the
        model's awake, not hidden pods minus the probe pods of a FRESH cluster view (None
        without one): the pods a Redis-mode commit needs evidence of; not wired = no
        Redis-mode commit.

        ``config.evidence_source == "direct"`` (registry default; 2026-09-29 B+D): the
        latency / KV evidence comes from the controller's own scrapes of the remaining
        pods (``safescale_direct``; baseline at the hide confirmation, one poll per
        tick, immediate rollback as soon as ``min_commit_samples`` are judged, deadline
        = confirmation + W on the controller clock, extended by one poll period while
        short). The Redis evidence never decides a commit in this mode (2026-09-29
        review P1-A): any gap in the direct evidence extends the deadline by one poll,
        and rolls the probe back (``evidence_incomplete:<gap>``) when it cannot heal or
        the ceiling is reached. ``evidence`` then only anchors the hide (audit)."""
        self._config = config
        self._store = store
        self._evidence = evidence
        self._thresholds = thresholds
        self._remaining_pods = remaining_pods
        self._direct_mode = str(getattr(config, "evidence_source", SOURCE_REDIS) or SOURCE_REDIS) == SOURCE_DIRECT
        self._wall_clock_ms = wall_clock_ms or (lambda: int(time.time() * 1000))
        self._probes: dict[str, SafeScaleProbe] = {}
        # Timer cleanup (2026-10-02, replaces the A13 60 s rollback backoff): model -> the
        # evidence of its last rolled-back probe; a receiver-less HIGH probe of the model
        # waits until the planner's signal beats it (:meth:`rollback_retry_holds`).
        # In-memory only, like the backoff was: a controller restart forgets it, i.e. at
        # most one extra HIGH probe per model right after a restart (the probe itself is
        # still guarded by SLO / donor-health / commit gate).
        self._rollback_evidence: dict[str, RollbackEvidence] = {}
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

    def rollback_evidence(self) -> dict[str, RollbackEvidence]:
        """Model -> the evidence of its last rolled-back probe (until its next probe
        resolves)."""
        return dict(self._rollback_evidence)

    def rollback_retry_holds(
        self, signals: Mapping[str, tuple[float | None, int | None, int | None]]
    ) -> dict[str, str]:
        """Timer cleanup (2026-10-02): models whose receiver-less HIGH probe stays held
        after a rollback -> why. ``signals`` = model -> (Z, routable count, window end
        ms) of the planner tick's decision window. A model is free again once

        * a metrics window ends after the rollback (new evidence; else ``no_new_window``),
        * and, after a capacity rollback (:data:`CAPACITY_ROLLBACK_CODES`), its routable
          count differs from the one the failed probe started from, or its Z is at least
          ``rollback_retry_z_margin`` above the Z that started it (else ``same_evidence``).

        A model without a signal this tick is held (``no_signal``). A capacity rollback
        whose starting Z is unknown takes the Z of the first window after it."""
        margin = float(getattr(self._config, "rollback_retry_z_margin", 0.25) or 0.0)
        holds: dict[str, str] = {}
        for model, evidence in list(self._rollback_evidence.items()):
            z, routable, window_end = signals.get(model, (None, None, None))
            if window_end is None:
                holds[model] = "no_signal"
                continue
            if int(window_end) <= int(evidence.rolled_back_ms):
                holds[model] = "no_new_window"
                continue
            if not evidence.capacity:
                continue
            if routable is not None and evidence.routable is not None and int(routable) != int(evidence.routable):
                continue
            if z is None:
                holds[model] = "same_evidence"
                continue
            if evidence.z_m is None:
                # The starting Z is unknown (e.g. a probe restored after a restart): the
                # first window after the rollback becomes the reference.
                self._rollback_evidence[model] = replace(evidence, z_m=float(z))
                holds[model] = "same_evidence"
                continue
            if float(z) >= float(evidence.z_m) + margin:
                continue
            holds[model] = "same_evidence"
        return holds

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
        inputs = window_inputs or ProbeWindowInputs()
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
            start_z_m=_optional_float(inputs.z_m),
            start_routable=_optional_int(inputs.routable_pods),
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
        direct: DirectState | None = None
        if self._direct_mode:
            # Direct evidence: W counts from the confirmation itself (controller clock,
            # the clock the scrapes are stamped with), not from a gateway boundary.
            base = int(confirmed)
            deadline = base + int(probe.window_ms or 0)
            direct = probe.direct or DirectState()
        else:
            base = max(int(probe.start_ms), int(confirmed) // step * step)
            deadline = max(int(probe.deadline_ms), base + int(probe.window_ms or 0))
        probe = replace(
            probe,
            hide_anchor=anchor,
            window_base_ms=base,
            deadline_ms=deadline,
            direct=direct,
            window_terms={
                **probe.window_terms, **_anchor_terms(anchor, step), **offsets, "window_base_ms": base,
                "hide_confirm_ms": int(confirmed), "deadline_ms": deadline,
            },
        )
        self._probes[model] = probe
        self._persist_probe(probe)
        return True

    # ------------------------------------------------------------ direct evidence
    def direct_mode(self) -> bool:
        return self._direct_mode

    def direct_baseline_delay_ms(self) -> int:
        """Registry ``safescale.baseline_delay_ms``: the baseline is scraped this long
        after the hide confirmation (the gateway applies the hide through its pod watch
        meanwhile); the deadline still counts from the confirmation."""
        return max(0, int(getattr(self._config, "baseline_delay_ms", 0.0) or 0))

    def direct_baseline_planned_ms(self, model: str) -> int | None:
        """When ``model``'s probe baseline is due (confirmation + delay); None = no
        probe waiting for one."""
        probe = self._probes.get(model)
        if probe is None or all(due.request_id != probe.request_id for due in self.direct_baseline_due()):
            return None
        return self._baseline_planned_ms(probe)

    def _baseline_planned_ms(self, probe: SafeScaleProbe) -> int | None:
        confirm = _hide_confirm_ms(probe)
        return None if confirm is None else confirm + self.direct_baseline_delay_ms()

    def _late_after_ms(self) -> float:
        """A baseline scraped later than confirmation + delay + one scrape timeout was
        not the planned first attempt (retry, restarted controller): late."""
        return float(self.direct_baseline_delay_ms()) + 1000.0 * float(
            getattr(self._config, "scrape_timeout_s", 1.0) or 1.0
        )

    def direct_baseline_due(self, now_ms: int | None = None) -> tuple[SafeScaleProbe, ...]:
        """Probing probes on the direct path whose hide is confirmed and whose baseline
        is not taken yet (normally taken ``baseline_delay_ms`` after the confirmation;
        after a restart or a failed first attempt, at a later tick - still post-hide, but
        then late: see ``take_baseline``). With ``now_ms``, only probes whose planned
        baseline time is reached."""
        if not self._direct_mode:
            return ()
        due = tuple(
            probe for probe in self._probes.values()
            if probe.status == "probing" and probe.hide_anchor is not None
            and probe.abort_reason is None and probe.preempt_reason is None
            and (probe.direct is None or (probe.direct.baseline is None and probe.direct.fallback is None))
        )
        if now_ms is None:
            return due
        return tuple(
            probe for probe in due
            if (self._baseline_planned_ms(probe) or 0) <= int(now_ms)
        )

    def direct_poll_due(self) -> tuple[SafeScaleProbe, ...]:
        """Probing probes on the direct path with a baseline to difference against."""
        if not self._direct_mode:
            return ()
        return tuple(
            probe for probe in self._probes.values()
            if probe.status == "probing" and probe.direct is not None and probe.direct.baseline
            and probe.direct.fallback is None and probe.direct.live_pods()
        )

    def set_direct_baseline(self, model: str, *, request_id: str, results, ts_ms: int) -> bool:
        """Store the baseline scrape of ``request_id``'s remaining pods (only once; pods
        that did not answer stay pending and get their baseline at their first later
        successful scrape). No pod answered: retried at the next tick (a later baseline
        is late: no commit; none by the deadline: rolled back ``no_baseline``). Empty
        ``results`` (no targets known yet) are not an attempt."""
        probe = self._probes.get(model)
        if (
            not self._direct_mode or probe is None or probe.request_id != request_id
            or probe.status != "probing" or probe.hide_anchor is None
        ):
            return False
        if probe.direct is not None and (probe.direct.baseline is not None or probe.direct.fallback is not None):
            return False
        if not results:
            return False
        state = take_baseline(
            results, ts_ms=int(ts_ms), previous=probe.direct,
            hide_confirm_ms=_hide_confirm_ms(probe), late_after_ms=self._late_after_ms(),
        )
        audit = {
            **self._baseline_audit(probe, state),
            "direct_baseline_pods": sorted(state.baseline or {}),
            "direct_pending_pods": list(state.pending),
            "direct_excluded_pods": {pod: value.get("reason") for pod, value in sorted(state.dropped.items())},
        }
        probe = replace(probe, direct=state, window_terms={**probe.window_terms, **audit})
        self._probes[model] = probe
        self._persist_probe(probe)
        return True

    def _direct_live(self, probe: SafeScaleProbe) -> bool:
        """The probe is judged on the direct scrape: direct mode, hide confirmed. (There
        is no Redis fallback any more: in direct mode nothing else decides a commit.)"""
        return self._direct_mode and probe.hide_anchor is not None

    def _gap_rollback(
        self, probe: SafeScaleProbe, gap: str, *, detail: Any = None, audit: dict[str, Any] | None = None
    ) -> SafeScaleDecision:
        """Roll back a direct-mode probe whose evidence has a gap that cannot heal (or
        hit the ceiling): ``rollback_reason.code = evidence_incomplete:<gap>``."""
        code = f"evidence_incomplete:{gap}"
        return self._rollback_now(
            probe, reason=code, details={"evidence_gap": gap},
            rollback_reason={"code": code, "gap": gap, "detail": detail},
            audit={**self._direct_audit(probe), **(audit or {}), "evidence_gap": gap},
        )

    def _direct_tick(
        self, probe: SafeScaleProbe, poll: DirectPoll | None
    ) -> tuple[SafeScaleProbe, int, SafeScaleDecision | None]:
        """One tick of a direct-evidence probe: difference the poll against the baseline
        and roll back at once on a judged SLO violation (``slo_violation_direct``).
        Returns (probe, the tick's wall clock, an immediate decision or None)."""
        wall_now = int(poll.ts_ms) if poll is not None else int(self._wall_ms() or 0)
        state = probe.direct or DirectState()
        if state.fallback is not None:
            # A record of the previous release that had switched to the Redis evidence:
            # the Redis evidence never decides a direct-mode commit (review P1-A).
            return probe, wall_now, self._gap_rollback(
                probe, "legacy_redis_fallback", detail=dict(state.fallback))
        if state.baseline is None:
            if wall_now >= int(probe.deadline_ms):
                # No baseline by the deadline: any later one would be late anyway.
                return probe, wall_now, self._gap_rollback(
                    probe, "no_baseline",
                    detail={"pending": list(state.pending), "failed_polls": state.failed_polls})
            if probe.direct is None:
                probe = replace(probe, direct=state)
                self._probes[probe.model] = probe
            return probe, wall_now, None
        if poll is None or poll.request_id != probe.request_id:
            return probe, wall_now, None
        state, window = evaluate_poll(
            state, poll,
            percentile_mode=str(getattr(self._config, "percentile_mode", "bucket_upper")),
            min_latency_samples=int(getattr(self._config, "min_latency_samples", 0) or 0),
            fresh_ms=_direct_fresh_ms(self._config),
            hq=float(getattr(self._config, "hq", 0.25)),
            hide_confirm_ms=_hide_confirm_ms(probe),
        )
        probe = replace(probe, direct=state)
        self._probes[probe.model] = probe
        # Every remaining pod failing is not a switch to another evidence source any
        # more: the missing / unanswered pods defer the commit, up to the ceiling.
        if window is None:
            return probe, wall_now, None
        violation = self._direct_violation(probe, window)
        if violation is not None:
            return probe, wall_now, self._rollback_now(
                probe,
                reason="slo_violation_direct",
                details={"slo": {"ttft_p95_ms": window.ttft_p95_ms, "tpot_p95_ms": window.tpot_p95_ms}},
                rollback_reason=violation,
                audit={
                    **self._direct_audit(probe), **window.latency_audit(), "latency_gate": "evaluated",
                    "latency_violations": violation["metrics"], "threshold_mode": violation["threshold_mode"],
                    "ttft_threshold_ms": violation["ttft_threshold_ms"],
                    "tpot_threshold_ms": violation["tpot_threshold_ms"],
                },
            )
        return probe, wall_now, None

    def _direct_violation(self, probe: SafeScaleProbe, window: DirectWindow) -> dict[str, Any] | None:
        """Immediate rollback on the direct window once ``min_commit_samples`` requests
        of pods with a p95 are in it (the commit gate's own sample rule)."""
        min_samples = int(getattr(self._config, "min_commit_samples", 20))
        if not window.p95_available or (min_samples > 0 and window.judged_count < min_samples):
            return None
        thresholds = self._thresholds_for(probe.model, window.mean_prompt_tokens)
        metrics = _latency_violations(window.ttft_p95_ms, window.tpot_p95_ms, thresholds)
        if not metrics:
            return None
        return {
            "code": "slo_violation_direct",
            "metrics": metrics,
            "ttft_p95_ms": window.ttft_p95_ms,
            "tpot_p95_ms": window.tpot_p95_ms,
            "pooled_ttft_p95_ms": window.pooled_ttft_p95_ms,
            "pooled_tpot_p95_ms": window.pooled_tpot_p95_ms,
            "ttft_threshold_ms": thresholds["ttft_ms"],
            "tpot_threshold_ms": thresholds["tpot_ms"],
            "threshold_mode": thresholds["mode"],
            "latency_samples": window.ttft_count,
            "latency_samples_judged": window.judged_count,
            "mean_prompt_tokens": window.mean_prompt_tokens,
            "evidence_start_ms": window.start_ms,
            "evidence_end_ms": window.end_ms,
        }

    def _baseline_audit(self, probe: SafeScaleProbe, state: DirectState) -> dict[str, Any]:
        """Baseline timing: planned vs actual, per-pod lag after the hide confirmation,
        and the pods whose evidence has a hole (late)."""
        return {
            "baseline_delay_ms": self.direct_baseline_delay_ms(),
            "direct_baseline_planned_ms": self._baseline_planned_ms(probe),
            "direct_baseline_ts_ms": state.baseline_ts_ms,
            "direct_baseline_lag_ms": dict(sorted(state.baseline_lag_ms.items())),
            "direct_late_after_ms": self._late_after_ms(),
            "direct_late_pods": {pod: dict(value) for pod, value in sorted(state.late.items())},
        }

    def _direct_audit(self, probe: SafeScaleProbe) -> dict[str, Any]:
        state = probe.direct or DirectState()
        audit: dict[str, Any] = {
            "evidence_source_used": SOURCE_DIRECT,
            "latency_source": "direct",
            **self._baseline_audit(probe, state),
            "direct_scrape_ts_ms": list(state.scrapes),
            "direct_excluded_pods": {pod: value.get("reason") for pod, value in sorted(state.dropped.items())},
        }
        if state.last is not None:
            audit.update(state.last.audit())
            audit["direct_excluded_pods"] = dict(sorted(state.last.excluded.items()))
            audit["kv_source"] = "direct" if state.last.kv_tail_max is not None else "redis_snapshot_tail"
            audit["kv_direct"] = state.last.kv_cache
            audit["kv_direct_tail_max"] = state.last.kv_tail_max
            audit["kv_ts_ms"] = state.kv_history[-1][0] if state.kv_history else None
        return audit

    def _source_audit(self, probe: SafeScaleProbe) -> dict[str, Any]:
        if not self._direct_mode:
            return {"evidence_source_used": SOURCE_REDIS}
        if probe.direct is not None and probe.direct.fallback is not None:
            # Legacy record (previous release): rolled back, never judged on Redis.
            return {"evidence_source_used": SOURCE_DIRECT, "direct_fallback_legacy": dict(probe.direct.fallback)}
        return {"evidence_source_used": SOURCE_DIRECT}

    def observe(
        self,
        model: str,
        observation: ProbeObservation,
        *,
        now_ms: int,
        direct_poll: DirectPoll | None = None,
        critical_receivers: Collection[str] = (),
    ) -> SafeScaleDecision:
        """One SafeScale tick (every ``probe_poll_seconds``) for ``model``'s probe.

        Snapshots are published once per gateway period and re-read every tick, so an
        observation is appended (journalled, counted in the hq tail) only once per
        snapshot, keyed by its ``window_end_ms``. Preemption / abort and the donor-health
        guard run on every tick (the gateway counters are fresh each time). The
        immediate SLO rollback judges each snapshot once, and only a snapshot whose
        whole window follows the hide (``window_start_ms >= hide``); earlier ones are
        recorded, not judged. At the deadline the commit gate runs (:meth:`_judge`).

        Direct evidence (``direct_poll`` = this tick's scrape of the remaining pods):
        the poll is differenced against the baseline and judged at once; the snapshot
        immediate rollback is not used; the deadline is compared with the poll's wall
        clock (the controller's own). The Redis evidence never decides a direct-mode
        probe; before its hide is confirmed it is only extended / rolled back
        (``hide_unconfirmed``).

        ``critical_receivers`` (F4, design donor-evidence-20261007): models the latest
        planner tick classified CRITICAL - an early-commit trigger (:meth:`_try_early_commit`)."""
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
        wall_now: int | None = None
        if self._direct_live(updated):
            updated, wall_now, immediate = self._direct_tick(updated, direct_poll)
            if immediate is not None:
                return immediate
        direct_live = self._direct_live(updated)
        # Redis mode only: the gateway-stamp anchor and the snapshot immediate rollback.
        # A direct-mode probe whose hide is not confirmed yet is never judged on Redis.
        redis_mode = not self._direct_mode
        if (
            redis_mode and self._evidence is not None and updated.hide_anchor is not None
            and updated.hide_anchor.newest_doc_error
        ):
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
        violation = self._instant_violation(updated, observation) if is_new and redis_mode else None
        if violation is not None:
            return self._rollback_now(
                updated,
                reason="slo_violation",
                details={"slo": {"ttft_p95_ms": violation["ttft_p95_ms"], "tpot_p95_ms": violation["tpot_p95_ms"]}},
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

        if direct_live:
            # Deadline on the controller clock (hide confirmation + W), checked every tick.
            clock = int(wall_now if wall_now is not None else (self._wall_ms() or 0))
            if clock < updated.deadline_ms:
                # Timer cleanup (2026-10-02): commit before the deadline when the evidence
                # is already complete (every rollback check above ran first, unchanged).
                early = self._try_early_commit(updated, health, now_ms=now_ms, wall_now_ms=clock,
                                               poll=direct_poll, critical_receivers=critical_receivers)
                if early is not None:
                    return early
                if updated is not probe:
                    self._persist_probe(updated)
                return SafeScaleDecision(status="probing", reason="probe_pending")
            return self._judge(updated, health, now_ms=now_ms, wall_now_ms=clock)
        if now_ms < updated.deadline_ms:
            if updated is not probe:
                self._persist_probe(updated)
            return SafeScaleDecision(status="probing", reason="probe_pending")
        return self._judge(updated, health, now_ms=now_ms)

    def _judge(
        self,
        probe: SafeScaleProbe,
        health: dict[str, float] | None,
        *,
        now_ms: int,
        wall_now_ms: int | None = None,
        early: dict[str, Any] | None = None,
    ) -> SafeScaleDecision | None:
        """The formal commit gate (v1 _tail_summary_allows_commit) at the deadline.

        Z (tail min) comes from the hq tail of the snapshot observations, as before. The
        latency check (and, on the direct path, the KV-cache fill) reads the direct
        scrape window (:meth:`_direct_outcome`), else the post-hide Redis evidence window
        (:meth:`_evidence_outcome`) when an evidence source is wired; without either
        (direct constructions) it reads the tail snapshots, as before 2026-09-29.

        ``early`` (timer cleanup 2026-10-02, direct path only): the same gates before
        the deadline, on the evidence covered so far (``early["deadline_ms"]``). Returns
        None - and changes nothing - unless every gate passes (no extension, wait,
        rollback or gate failure is acted on early: the deadline decides those)."""
        model = probe.model
        direct_live = self._direct_live(probe)
        if early is not None and not (self._direct_mode and direct_live):
            return None
        evidence_mode = self._evidence is not None
        summary = _summarize_tail(
            probe,
            hq=self._config.hq,
            thresholds=lambda observation: self._thresholds_for(model, observation.mean_prompt_tokens),
            judge_latency=not (evidence_mode or direct_live),
            hide_ts_ms=self._post_hide_start(probe),
        )
        tail_audit = {
            **self._source_audit(probe),
            "tail_pre_hide_fraction_mean": summary.pre_hide_fraction_mean,
            "tail_pre_hide_fraction_max": summary.pre_hide_fraction_max,
            "tail_observation_count": summary.tail_count,
            "z_source": "redis_snapshot_tail",
            "z_ts_ms": _latest_window_end(probe),
            # Both KV sources are recorded; the gate uses kv_source.
            "kv_redis_tail_max": summary.gpu_cache_max,
            "kv_source": "redis_snapshot_tail",
        }
        idle = False
        low_samples = False
        if self._direct_mode:
            # Direct mode: only the direct evidence decides a commit (review P1-A). A
            # hide never confirmed extends to the ceiling, then rolls back.
            if direct_live:
                outcome = self._direct_outcome(
                    probe, summary, wall_now_ms=int(wall_now_ms or now_ms),
                    deadline_ms=early["deadline_ms"] if early is not None else None,
                )
            else:
                outcome = self._unconfirmed_outcome(probe)
            if early is not None and (
                outcome.kind != "judge" or not outcome.latency_ok or outcome.idle or outcome.low_samples
            ):
                return None
            if outcome.kind == "extend":
                return self._extend(probe, outcome.audit,
                                    direct_now_ms=int(wall_now_ms or now_ms) if direct_live else None)
            if outcome.kind == "wait":
                return self._wait(probe, outcome.audit)
            if outcome.kind == "rollback":
                return self._rollback_now(
                    probe,
                    reason=str(outcome.rollback_reason.get("code")),
                    details={"evidence": outcome.audit},
                    rollback_reason=outcome.rollback_reason,
                    audit={**tail_audit, **outcome.audit},
                )
            kv = probe.direct.last.kv_tail_max if probe.direct is not None and probe.direct.last is not None else None
            summary = replace(summary, latency_ok=outcome.latency_ok,
                              gpu_cache_max=kv if kv is not None else summary.gpu_cache_max)
            latency_audit = outcome.audit
            idle = outcome.idle
            low_samples = outcome.low_samples
        elif evidence_mode:
            outcome = self._evidence_outcome(probe, summary, now_ms=now_ms)
            if outcome.kind == "extend":
                return self._extend(probe, outcome.audit)
            if outcome.kind == "wait":
                return self._wait(probe, outcome.audit)
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
            low_samples = outcome.low_samples
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
        if early is not None and failures:
            return None
        audit = {**tail_audit, **latency_audit}
        if early is not None:
            audit["early_commit"] = {key: value for key, value in early.items() if key != "deadline_ms"}
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
        if early is not None:
            LOG.info(json.dumps({
                "event": "safescale_early_commit", "model": model, "request_id": probe.request_id,
                **audit["early_commit"],
            }, sort_keys=True))
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
        if low_samples and not failures:
            # Counted apart (audit + summary): a commit whose latency was judged on fewer
            # than min_commit_samples requests at the ceiling.
            audit["low_sample_commit"] = details["low_sample_commit"] = True
            LOG.warning(json.dumps({
                "event": "safescale_low_sample_commit", "model": model, "request_id": probe.request_id,
                "latency_samples": audit.get("latency_samples"),
                "low_sample_ttft_p95_ms": audit.get("low_sample_ttft_p95_ms"),
                "low_sample_tpot_p95_ms": audit.get("low_sample_tpot_p95_ms"),
            }, sort_keys=True))
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
        # 2026-09-29 review: the pods a commit needs evidence of come from a FRESH
        # cluster view (awake, not hidden, not the probe's), not from whatever docs the
        # window holds; each must have docs up to E, and E must reach the deadline.
        required = self._required_pods(probe)
        gap: str | None = None
        gap_detail: Any = None
        if required is None:
            gap, gap_detail = "no_fresh_view", None
        elif not required:
            gap, gap_detail = "no_remaining_pods", []
        elif int(end) < int(probe.deadline_ms):
            gap, gap_detail = "before_deadline", {"evidence_end_ms": int(end), "deadline_ms": int(probe.deadline_ms)}
        else:
            incomplete = _evidence_incomplete(required, evidence, end_ms=int(end))
            if incomplete:
                gap, gap_detail = "pods_incomplete", incomplete
                audit["evidence_incomplete_pods"] = incomplete
        audit["required_pods"] = list(required) if required is not None else None
        last_docs = [int(evidence.last_doc_ts_ms[pod]) for pod in (required or ()) if pod in evidence.last_doc_ts_ms]
        audit.update(
            evidence_coverage_start_ms=max(evidence.first_doc_ts_ms.values()) if evidence.first_doc_ts_ms else None,
            evidence_coverage_end_ms=min(last_docs) if required and len(last_docs) == len(required) else None,
        )
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
        audit.update(evidence_pooled_ttft_p95_ms=evidence.pooled_ttft_p95_ms,
                     evidence_pooled_tpot_p95_ms=evidence.pooled_tpot_p95_ms)
        p95_available = evidence.ttft_p95_ms is not None or evidence.tpot_p95_ms is not None
        # n counts the requests the p95 judges: the pods whose own p95 is defined
        # (per-pod minimum samples), or all of them once the pooled p95 exists.
        # At least one judged request whatever min_commit_samples says (0 must not make
        # the gate vacuous: no sample is not evidence of health).
        enough = judged >= max(1, min_samples) and p95_available
        violations = _latency_violations(evidence.ttft_p95_ms, evidence.tpot_p95_ms, thresholds) if enough else []
        if violations:
            # A violation is one whatever pods are missing.
            audit.update(latency_gate="evaluated", latency_violations=violations)
            return _EvidenceOutcome("judge", audit=audit, latency_ok=False)
        if not evidence.first_doc_ts_ms and not can_extend and gap != "no_fresh_view":
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
        if gap is not None:
            # No commit on this evidence: extend while possible, then wait at most one
            # more gateway period past the ceiling (the snapshot boundary >= deadline /
            # the docs of E may land a period late), then roll back.
            audit.update(evidence_gap=gap)
            if can_extend and gap != "before_deadline":
                return _EvidenceOutcome("extend", audit={**audit, "extend_reason": f"evidence_incomplete:{gap}"})
            if int(now_ms) < int(cap) + step:
                return _EvidenceOutcome("wait", audit={**audit, "wait_reason": f"evidence_incomplete:{gap}"})
            return _EvidenceOutcome("rollback", audit=audit, rollback_reason={
                "code": f"evidence_incomplete:{gap}", "gap": gap, "detail": gap_detail,
                "evidence_end_ms": int(end), "deadline_cap_ms": cap})
        if enough:
            audit.update(latency_gate="evaluated", latency_violations=[])
            return _EvidenceOutcome("judge", audit=audit, latency_ok=True)
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
        # At the W ceiling with too little evidence: judged on what there is.
        return _ceiling_outcome(
            audit, samples=samples, has_traffic=summary.has_traffic, short=short, thresholds=thresholds,
            ttft_ms=_max_present(evidence.low_ttft_p95_ms, evidence.ttft_p95_ms),
            tpot_ms=_max_present(evidence.low_tpot_p95_ms, evidence.tpot_p95_ms),
        )

    def _direct_outcome(
        self, probe: SafeScaleProbe, summary: "ProbeTailSummary", *, wall_now_ms: int,
        deadline_ms: int | None = None,
    ) -> "_EvidenceOutcome":
        """The latency verdict of the direct window at the deadline.

        Invariant (2026-09-29 reviews): a direct commit needs evidence of EVERY remaining
        pod covering [the planned baseline (+ at most one scrape timeout), the deadline]:
        no late pod (late / retried baseline, restart, a pod that joined), no missing or
        unanswered pod in the deciding poll (this tick's), a fresh cluster view listing
        no pod outside the evidence, and every pod's evidence reaching the deadline. A
        gap that can heal extends the deadline by one poll period (up to the ceiling);
        one that cannot, or any gap at the ceiling, rolls back
        ``evidence_incomplete:<gap>``. The Redis evidence is never consulted. A judged
        violation rolls back whatever is missing.

        ``deadline_ms`` (early commit): the evidence must cover up to this moment
        instead of the probe's deadline."""
        cfg = self._config
        deadline = int(probe.deadline_ms) if deadline_ms is None else int(deadline_ms)
        cap = _deadline_cap(probe, cfg)
        can_extend = probe.deadline_ms < cap
        window_clamped = bool(probe.window_terms.get("window_clamped", probe.window_terms.get("clamped")))
        audit: dict[str, Any] = {
            **self._direct_audit(probe),
            "extensions": probe.extensions,
            "window_clamped": window_clamped,
            "clamped": window_clamped,
            "deadline_cap_ms": cap,
        }
        anchor = probe.hide_anchor
        if anchor is not None:
            audit.update(hide_ts_ms=int(anchor.ts_ms), hide_anchor_source=anchor.source)

        def gap(name: str, detail: Any, *, heals: bool = True) -> "_EvidenceOutcome":
            marked = {**audit, "evidence_gap": name}
            if heals and can_extend:
                return _EvidenceOutcome("extend", audit={**marked, "extend_reason": name})
            return _EvidenceOutcome("rollback", audit=marked, rollback_reason={
                "code": f"evidence_incomplete:{name}", "gap": name, "detail": detail, "deadline_cap_ms": cap})

        state = probe.direct or DirectState()
        if not state.live_pods():
            # No remaining pod left in the evidence (all asleep at / before their
            # baseline): nothing proves the model can serve without the probe pods.
            return gap("no_live_pods", {pod: dict(value) for pod, value in sorted(state.dropped.items())},
                       heals=False)
        window = state.last
        if window is None or int(wall_now_ms) - int(window.end_ms) > _direct_fresh_ms(cfg):
            # No poll differenced yet (restart / first tick) or the latest is stale.
            why = "no_direct_window" if window is None else "direct_window_stale"
            return gap(why, {"window_end_ms": window.end_ms if window is not None else None})
        confirm = probe.window_base_ms if probe.window_base_ms is not None else probe.start_ms
        samples, judged = float(window.ttft_count), float(window.judged_count)
        thresholds = self._thresholds_for(probe.model, window.mean_prompt_tokens)
        min_samples = int(getattr(cfg, "min_commit_samples", 20))
        audit.update(
            **window.latency_audit(),
            evidence_pods=list(window.pods),
            # Share of the evidence before the hide confirmation (the baseline follows it).
            tail_pre_hide_fraction=_pre_hide_fraction(int(confirm), window.start_ms, window.end_ms),
            min_commit_samples=min_samples,
            threshold_mode=thresholds["mode"],
            threshold_source=thresholds.get("source"),
            ttft_threshold_ms=thresholds["ttft_ms"],
            tpot_threshold_ms=thresholds["tpot_ms"],
            deadline_ms=deadline,
        )
        if "fallback" in thresholds:
            audit["threshold_fallback"] = thresholds["fallback"]
        # At least one judged request whatever min_commit_samples says (0 must not make
        # the gate vacuous: no sample is not evidence of health).
        enough = judged >= max(1, min_samples) and window.p95_available
        violations = _latency_violations(window.ttft_p95_ms, window.tpot_p95_ms, thresholds) if enough else []
        if violations:
            # A violation is one whatever pods are missing.
            audit.update(latency_gate="evaluated", latency_violations=violations)
            return _EvidenceOutcome("judge", audit=audit, latency_ok=False)
        if window.late:
            # A pod's evidence has a hole after the hide (live or dropped since): waiting
            # cannot fill it - roll back at the deadline (its data kept driving the
            # immediate rollback until now).
            return gap("late_baseline", {pod: dict(value) for pod, value in sorted(window.late.items())},
                       heals=False)
        if window.missing:
            # A live pod has no fresh evidence (pending baseline, failing scrapes). A pod
            # still waiting for its baseline cannot heal: once it answers it is late.
            missing = dict(sorted(window.missing.items()))
            pending = any(reason == "pending_baseline" for reason in missing.values())
            return gap("pending_baseline" if pending else "pods_missing", missing, heals=not pending)
        if window.unanswered or int(window.end_ms) != int(wall_now_ms):
            # The deciding poll must have read every live pod: a pod whose latest scrape
            # failed (a timeout is often the overload itself) may hide the violation in
            # the seconds its older delta does not cover. The freshness tolerance is for
            # the intermediate ticks only.
            if window.unanswered:
                return gap("pods_unanswered", dict(sorted(window.unanswered.items())))
            return gap("no_poll_this_tick", {"poll_ts_ms": window.end_ms})
        if window.view_pods is None:
            # The remaining-pod set cannot be checked against the cluster (review P2-c).
            return gap("no_fresh_view", {"poll_ts_ms": window.end_ms})
        outside = sorted(set(window.view_pods) - set(window.pods))
        if outside:
            # Defensive: evaluate_poll already makes such a pod late.
            return gap("pods_outside_evidence", outside, heals=False)
        if window.coverage_end_ms is None or window.coverage_end_ms < deadline:
            # The deciding poll's reads were sent before the deadline: the evidence ends
            # short of it. The next poll covers it; the deadline stays.
            if int(wall_now_ms) - deadline <= _direct_fresh_ms(cfg):
                return _EvidenceOutcome("wait", audit={**audit, "wait_reason": "coverage_before_deadline"})
            return gap("coverage_short", {"coverage_end_ms": window.coverage_end_ms,
                                          "deadline_ms": deadline}, heals=False)
        if enough:
            audit.update(latency_gate="evaluated", latency_violations=[])
            return _EvidenceOutcome("judge", audit=audit, latency_ok=True)
        short = "insufficient_samples" if samples < min_samples else "p95_unavailable"
        if can_extend:
            return _EvidenceOutcome("extend", audit={**audit, "extend_reason": short})
        in_flight = window.in_flight is not None and window.in_flight > 0
        return _ceiling_outcome(
            audit, samples=samples, has_traffic=summary.has_traffic or in_flight, short=short,
            thresholds=thresholds, ttft_ms=window.low_ttft_p95_ms, tpot_ms=window.low_tpot_p95_ms,
        )

    def _try_early_commit(
        self,
        probe: SafeScaleProbe,
        health: dict[str, float] | None,
        *,
        now_ms: int,
        wall_now_ms: int,
        poll: DirectPoll | None,
        critical_receivers: Collection[str] = (),
    ) -> SafeScaleDecision | None:
        """Timer cleanup (2026-10-02, simplified 2026-10-06 Q4): commit a direct-evidence
        probe before its deadline when (a) ``min_commit_samples`` requests of the
        remaining pods are judged, (b) the formal commit gates pass on the evidence so far
        (:meth:`_judge` early mode: SLO, KV-cache, Z tail, evidence completeness), (c) the
        hidden pods have nothing in flight - the gateway's in-flight count of the live
        plugin instances and vLLM running + waiting both known and 0 - and (d) one
        evidence floor: the newest snapshot window holds ``early_commit_min_grids``
        (= ``scaling.min_evidence_grids``, the O1 warm rule) complete post-hide gateway
        grids AND at least the donor's p95 end-to-end latency passed since the hide
        confirmation (the remaining pods' concurrency needs about one e2e to settle).
        No fixed share of W and no separate minimum observation time.

        F4 (2026-10-07, design donor-evidence-20261007): while another model is CRITICAL
        (``critical_receivers``), (c) is waived: the hidden pods' in-flight requests are
        a cost, not evidence (the sleep aborts them and the sidecar re-issues them, as on
        an immediate release). (a), (b) and all of (d) stay: the p95 e2e term is evidence
        completeness - SLO samples count at completion, so less than one e2e after the
        hide they are biased towards short requests. The freed GPU reaches the waiting
        model through the event-driven view refresh (F2).
        None = keep probing (the deadline decides as before)."""
        cfg = self._config
        if not bool(getattr(cfg, "early_commit", False)) or poll is None:
            return None
        if poll.request_id != probe.request_id or probe.window_base_ms is None:
            return None
        state = probe.direct
        window = state.last if state is not None else None
        if window is None or window.coverage_end_ms is None or int(window.end_ms) != int(wall_now_ms):
            return None
        elapsed = int(wall_now_ms) - int(probe.window_base_ms)
        # Review P2-3 / Q4: at least one p95 end-to-end latency since the hide
        # confirmation (the remaining pods' concurrency needs about one e2e to reach its
        # new steady state); the post-hide grids below are the other half of the floor.
        inputs = (probe.window_terms or {}).get("inputs") or {}
        p95_e2e = _optional_float(inputs.get("p95_e2e_ms")) if isinstance(inputs, dict) else None
        grids = max(1, int(getattr(cfg, "early_commit_min_grids", 2) or 1))
        min_elapsed = float(p95_e2e or 0.0)
        if elapsed < min_elapsed:
            return None
        min_samples = int(getattr(cfg, "min_commit_samples", 20))
        if not window.p95_available or window.judged_count < max(1, min_samples):
            return None
        # O1-style warm tail: the newest snapshot window holds ``grids`` complete gateway
        # grids after the first boundary following the hide.
        post_hide = self._post_hide_start(probe)
        latest = _latest_window_end(probe)
        if post_hide is None or latest is None or int(latest) < int(post_hide) + grids * _evidence_step_ms(cfg):
            return None
        drained = _hidden_drained(probe, poll)
        urgent = sorted({str(name) for name in critical_receivers or ()} - {probe.model})
        if drained is None:
            if not urgent:
                return None
            # F4: a CRITICAL model waits - the hidden pods' requests are re-issued.
            drained = {"hidden_drained": False}
        if urgent:
            drained = {**drained, "critical_receivers": urgent}
        return self._judge(
            probe, health, now_ms=now_ms, wall_now_ms=wall_now_ms,
            early={
                "deadline_ms": int(window.coverage_end_ms),
                "elapsed_ms": elapsed,
                "min_elapsed_ms": int(min_elapsed),
                "post_hide_grids": grids,
                "samples": float(window.judged_count),
                "planned_deadline_ms": int(probe.deadline_ms),
                **drained,
            },
        )

    def _unconfirmed_outcome(self, probe: SafeScaleProbe) -> "_EvidenceOutcome":
        """A direct-mode probe at its deadline whose hide was never confirmed: extend
        (one gateway period) up to the ceiling, then roll back ``hide_unconfirmed``."""
        cap = _deadline_cap(probe, self._config)
        audit = {"latency_source": "direct", "extensions": probe.extensions, "deadline_cap_ms": cap}
        if probe.deadline_ms < cap:
            return _EvidenceOutcome("extend", audit={**audit, "extend_reason": "hide_unconfirmed"})
        return _EvidenceOutcome("rollback", audit=audit,
                                rollback_reason={"code": "hide_unconfirmed", "deadline_cap_ms": cap})

    def _required_pods(self, probe: SafeScaleProbe) -> tuple[str, ...] | None:
        """Redis mode: the remaining pods of a fresh cluster view (None = no fresh view
        or not wired: no commit)."""
        if self._remaining_pods is None:
            return None
        try:
            pods = self._remaining_pods(probe.model, tuple(probe.pods))
        except Exception:  # noqa: BLE001 - no view: no commit
            LOG.warning("safescale remaining pods of %s unavailable", probe.model, exc_info=True)
            return None
        return None if pods is None else tuple(sorted(str(pod) for pod in pods if pod not in probe.pods))

    def _wait(self, probe: SafeScaleProbe, audit: dict[str, Any]) -> SafeScaleDecision:
        """Not judged this tick, the deadline unchanged: the next poll / snapshot decides."""
        updated = replace(probe, window_terms={**probe.window_terms, **audit})
        self._probes[probe.model] = updated
        self._persist_probe(updated)
        return SafeScaleDecision(status="probing", reason="evidence_pending", details=audit)

    def _extend(
        self, probe: SafeScaleProbe, audit: dict[str, Any], *, direct_now_ms: int | None = None
    ) -> SafeScaleDecision:
        cap = _deadline_cap(probe, self._config)
        if direct_now_ms is not None:
            # Direct path: one poll period past now (the next tick brings a new scrape).
            deadline = min(max(int(probe.deadline_ms), int(direct_now_ms)) + _direct_poll_ms(self._config), cap)
            extensions = probe.extensions + 1
            audit = {**audit, "extensions": extensions, "deadline_ms": deadline}
            updated = replace(
                probe, deadline_ms=deadline, extensions=extensions, window_terms={**probe.window_terms, **audit}
            )
            self._probes[probe.model] = updated
            self._persist_probe(updated)
            return SafeScaleDecision(status="probing", reason="evidence_extended", details=audit)
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
            **self._source_audit(probe),
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
        # max(per pod, pooled), as the deadline gate judges (2026-09-29 review P3).
        ttft = _max_present(observation.ttft_p95_ms, observation.pooled_ttft_p95_ms)
        tpot = _max_present(observation.tpot_p95_ms, observation.pooled_tpot_p95_ms)
        metrics = _latency_violations(ttft, tpot, thresholds)
        if not metrics:
            return None
        return {
            "code": "slo_violation",
            "metrics": metrics,
            "ttft_p95_ms": ttft,
            "tpot_p95_ms": tpot,
            "pooled_ttft_p95_ms": observation.pooled_ttft_p95_ms,
            "pooled_tpot_p95_ms": observation.pooled_tpot_p95_ms,
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
                # Plan D1: drain_budget_s = W is still sent to the SM, which ignores it:
                # the SM never drains on any sleep path (2026-10-02; the registry key
                # service_manager.sleep.no_drain_paths is deprecated). The probe window,
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
        # A preemption for the model's own scale-up is not a failed probe: no evidence
        # hold. A commit clears the previous rollback's evidence.
        if status == "rollback" and probe.preempt_reason is None:
            code = _rollback_code(probe, reason)
            self._rollback_evidence[model] = RollbackEvidence(
                rolled_back_ms=int(now_ms),
                reason=code,
                capacity=code in CAPACITY_ROLLBACK_CODES,
                z_m=probe.start_z_m,
                routable=probe.start_routable,
            )
        elif status == "commit":
            self._rollback_evidence.pop(model, None)
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
    #: judge = run the formal gate with ``latency_ok``; extend = deadline + one period;
    #: wait = not judged this tick, deadline unchanged; rollback = fail closed.
    kind: Literal["judge", "extend", "wait", "rollback"]
    audit: dict[str, Any] = field(default_factory=dict)
    latency_ok: bool = True
    idle: bool = False
    #: The latency was judged on fewer than min_commit_samples requests (ceiling).
    low_samples: bool = False
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


def _direct_poll_ms(config: SafeScaleConfig) -> int:
    return max(1, int(getattr(config, "evidence_poll_ms", 2_000.0) or 2_000))


def _direct_fresh_ms(config: SafeScaleConfig) -> float:
    """A pod's latest delta counts as fresh within two poll periods plus one scrape
    timeout (one failed scrape in between is tolerated at the intermediate ticks: its
    delta is cumulative). Not at the deadline: a commit needs every live pod answered in
    the deciding poll (``_direct_outcome``)."""
    return 2.0 * _direct_poll_ms(config) + 1000.0 * float(getattr(config, "scrape_timeout_s", 1.0) or 1.0)


def _evidence_incomplete(required, evidence: EvidenceWindow, *, end_ms: int) -> dict[str, str]:
    """Redis evidence: every required remaining pod (a fresh cluster view's) must have
    a TTFT delta in the window that ends at its end (a doc stamped >= ``end_ms``).
    pod -> why not; {} = complete."""
    missing: dict[str, str] = {}
    present = set(evidence.pods)
    for pod in required:
        if pod not in present:
            missing[pod] = "no_docs"
        elif pod not in evidence.first_doc_ts_ms:
            missing[pod] = "no_ttft_histogram"
        else:
            last = evidence.last_doc_ts_ms.get(pod)
            if last is None or int(last) < int(end_ms):
                missing[pod] = f"docs_end:{last}"
    return missing


def _max_present(*values: float | None) -> float | None:
    present = [float(value) for value in values if value is not None]
    return max(present) if present else None


def _ceiling_outcome(
    audit: dict[str, Any],
    *,
    samples: float,
    has_traffic: bool,
    short: str,
    thresholds: dict[str, Any],
    ttft_ms: float | None,
    tpot_ms: float | None,
) -> "_EvidenceOutcome":
    """At the W ceiling with fewer than ``min_commit_samples`` judged requests (both
    evidence paths; 2026-09-29 review: the latency gate is never skipped any more):

    * no request completed and nothing in flight (idle) -> commit (v1 idle rule);
    * no request completed but traffic -> rollback ``insufficient_evidence:stalled``;
    * otherwise every sample there is is judged: ``ttft_ms`` / ``tpot_ms`` = max(per-pod,
      pooled) p95 without any minimum-samples rule; above a threshold -> the latency
      gate fails (rollback), else it passes (``latency_gate = evaluated_low_samples``;
      a commit is counted apart as ``low_sample_commit``). No p95 at all despite
      completed requests -> rollback ``insufficient_evidence:p95_unavailable``."""
    audit = {**audit, "clamped": True, "low_sample_reason": short}
    if samples <= 0:
        if not has_traffic:
            audit.update(latency_gate="skipped", latency_skip_reason="idle")
            return _EvidenceOutcome("judge", audit=audit, latency_ok=True, idle=True)
        audit.update(latency_gate="insufficient_evidence")
        return _EvidenceOutcome("rollback", audit=audit, rollback_reason={
            "code": "insufficient_evidence:stalled", "latency_samples": samples, "has_traffic": True})
    audit.update(low_sample_ttft_p95_ms=ttft_ms, low_sample_tpot_p95_ms=tpot_ms)
    if ttft_ms is None and tpot_ms is None:
        audit.update(latency_gate="insufficient_evidence")
        return _EvidenceOutcome("rollback", audit=audit, rollback_reason={
            "code": "insufficient_evidence:p95_unavailable", "latency_samples": samples})
    violations = _latency_violations(ttft_ms, tpot_ms, thresholds)
    audit.update(latency_gate="evaluated_low_samples", latency_violations=violations)
    return _EvidenceOutcome("judge", audit=audit, latency_ok=not violations, low_samples=True)


def _hide_confirm_ms(probe: SafeScaleProbe) -> int | None:
    """The hide confirmation (controller clock) of a direct-path probe: its window base
    (set by ``mark_hidden``). None = hide not confirmed."""
    if probe.hide_anchor is None:
        return None
    if probe.window_base_ms is not None:
        return int(probe.window_base_ms)
    anchor = probe.hide_anchor
    return int(anchor.controller_ts_ms if anchor.controller_ts_ms is not None else anchor.ts_ms)


def _latency_violations(ttft_p95_ms: float | None, tpot_p95_ms: float | None, thresholds: dict[str, Any]) -> list[str]:
    violations = []
    if ttft_p95_ms is not None and ttft_p95_ms > thresholds["ttft_ms"]:
        violations.append("ttft")
    if tpot_p95_ms is not None and tpot_p95_ms > thresholds["tpot_ms"]:
        violations.append("tpot")
    return violations


def _hidden_drained(probe: SafeScaleProbe, poll: DirectPoll) -> dict[str, Any] | None:
    """Early-commit condition (c): every hidden probe pod was scraped in this poll with
    vLLM running + waiting known and 0, and the gateway's in-flight count of the hidden
    pods is known and 0. None = not drained or unknown."""
    hidden = getattr(poll, "hidden", None)
    gateway = getattr(poll, "gateway_inflight", None)
    if hidden is None or gateway is None or float(gateway) != 0.0:
        return None
    for pod in probe.pods:
        scrape = hidden.get(pod)
        if not isinstance(scrape, PodScrape) or scrape.in_flight is None or float(scrape.in_flight) != 0.0:
            return None
    return {"hidden_in_flight": 0.0, "gateway_in_flight": 0.0}


def _rollback_code(probe: SafeScaleProbe, reason: str) -> str:
    """The structured rollback code of a resolved probe (its terminal details), else the
    decision reason recorded when it was handed to the queue, else ``reason``."""
    details = probe.terminal_details or {}
    rollback = details.get("rollback_reason") if isinstance(details, dict) else None
    if isinstance(rollback, dict) and rollback.get("code"):
        return str(rollback["code"])
    return str(probe.resolution_reason or reason or "")


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
    if probe.start_z_m is not None:
        record["start_z_m"] = probe.start_z_m
    if probe.start_routable is not None:
        record["start_routable"] = probe.start_routable
    if probe.direct is not None:
        record["direct_evidence"] = probe.direct.as_record()
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
        "pooled_ttft_p95_ms": observation.pooled_ttft_p95_ms,
        "pooled_tpot_p95_ms": observation.pooled_tpot_p95_ms,
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
        direct=DirectState.from_record(row.get("direct_evidence")),
        start_z_m=_optional_float(row.get("start_z_m")),
        start_routable=_optional_int(row.get("start_routable")),
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
        pooled_ttft_p95_ms=_optional_float(raw.get("pooled_ttft_p95_ms")),
        pooled_tpot_p95_ms=_optional_float(raw.get("pooled_tpot_p95_ms")),
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
