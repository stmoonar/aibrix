from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Any, Iterable

from tre_common.dwell import DwellCounter
from tre_common.metrics_schema import ModelWindowMetrics
from tre_common.registry import TrsParams
from tre_common.tss import TssEma, replica_factor, signal_ema, tss_terms, window_is_idle

#: Raw classification states the controller's band dwell can gate (TRE_DWELL_STATES).
DWELL_STATES = ("critical", "low", "high")


@dataclass
class TRSInput:
    """Inputs of one window's TSS (the unified definition, ``tre_common.tss``).

    ``prompt_tokens_total`` / ``generation_tokens_total`` are window totals and the TSS
    numerator is that total (tokens per window, window = TRE_METRICS_WINDOW_MS).
    ``window_ms`` is the window duration; it only drives the EMA idle-gap reset (``None``
    disables it). ``avg_swapping`` and ``w_d`` are carried for schema compatibility and are
    ignored by the formula (a non-zero swapping / a w_d != 1 is logged once).
    """

    prompt_tokens_total: float
    generation_tokens_total: float
    avg_waiting: float
    avg_running: float
    avg_swapping: float
    routable_pods: int
    assigned_replicas: int
    w_p: float = 0.04
    w_d: float = 1.0
    lambda_wait: float = 2.625
    qmin: float = 1.0
    kv_cache_hit_rate: float = 0.0
    window_ms: float | None = None

    @classmethod
    def from_metrics(cls, metrics: ModelWindowMetrics, params: TrsParams) -> "TRSInput":
        return cls(
            prompt_tokens_total=metrics.prompt_tokens,
            generation_tokens_total=metrics.generation_tokens,
            avg_waiting=metrics.avg_waiting,
            avg_running=metrics.avg_running,
            avg_swapping=metrics.avg_swapping,
            routable_pods=metrics.routable_pods,
            assigned_replicas=metrics.assigned_replicas,
            w_p=params.w_p,
            w_d=params.w_d,
            lambda_wait=params.lambda_wait,
            qmin=params.qmin,
            kv_cache_hit_rate=metrics.kv_cache_hit_rate,
            window_ms=float(metrics.window_end_ms - metrics.window_start_ms),
        )


@dataclass
class TRSResult:
    Y_m: float
    y_m: float
    Q: float
    Q_ctl: float
    TRS_raw: float
    TRS: float
    eta_m: float | None
    Z_m: float | None
    ema_alpha: float
    prev_Y: float | None = None
    prev_Q_ctl: float | None = None
    #: False when TSS is undefined for this window (idle: running + waiting == 0). TRS /
    #: TRS_raw are then 0.0 and Z_m is None - never a small Z that would read CRITICAL.
    defined: bool = True


class TRSComputer:
    """Raw TSS of one window plus the per-model EMA.

    With ``ema_tau_ms`` set (every deployed registry entry) the EMA *is* a
    :class:`tre_common.tss.TssEma` - the same object and rules the offline paths use
    (idle-gap reset after one metrics window, None/0 passthrough, per-window dedup), so
    online and offline agree bitwise. ``ema_alpha`` feeds only the DEPRECATED fixed-alpha
    branch (``ema_tau_ms`` unset), which is kept for the golden parity tests.
    """

    def __init__(self, ema_alpha: float = 0.5, ema_tau_ms: float | None = None) -> None:
        self.ema_alpha = ema_alpha
        self.ema_tau_ms = ema_tau_ms
        self._ema: TssEma | None = (
            TssEma(ema_tau_ms) if ema_tau_ms is not None and ema_tau_ms > 0 else None
        )
        # DEPRECATED fixed-alpha branch state (only used when _ema is None).
        self._legacy_ema: float | None = None
        self._legacy_last_ms: int | None = None
        self._prev_Y: float | None = None
        self._prev_Q_ctl: float | None = None

    @property
    def tss_ema(self) -> TssEma | None:
        """The shared-definition EMA (None in the deprecated fixed-alpha mode)."""
        return self._ema

    @property
    def current_ema(self) -> float | None:
        return self._ema.value if self._ema is not None else self._legacy_ema

    def reset_ema(self) -> None:
        """Forget the EMA (idle tick). Equivalent to a fresh computer / a restart."""
        if self._ema is not None:
            self._ema.reset()
        self._legacy_ema = None
        self._legacy_last_ms = None

    def restore(
        self,
        *,
        ema: float | None = None,
        prev_Y: float | None = None,
        prev_Q_ctl: float | None = None,
        last_update_ms: int | None = None,
    ) -> None:
        if ema is not None:
            if self._ema is not None:
                self._ema.value = ema
            else:
                self._legacy_ema = ema
        if prev_Y is not None:
            self._prev_Y = prev_Y
        if prev_Q_ctl is not None:
            self._prev_Q_ctl = prev_Q_ctl
        if last_update_ms is not None:
            if self._ema is not None:
                self._ema.last_ms = float(last_update_ms)
            else:
                self._legacy_last_ms = last_update_ms

    def snapshot(self) -> dict[str, Any]:
        return {"ema": self.current_ema, "prev_Y": self._prev_Y, "prev_Q_ctl": self._prev_Q_ctl}

    def compute(
        self, inp: TRSInput, theta_m: float | None = None, *, window_end_ms: int | None = None
    ) -> TRSResult:
        effective_pods = max(1, inp.routable_pods)
        terms = tss_terms(
            prompt_tokens=inp.prompt_tokens_total,
            generation_tokens=inp.generation_tokens_total,
            avg_running=inp.avg_running,
            avg_waiting=inp.avg_waiting,
            w_p=inp.w_p,
            lambda_wait=inp.lambda_wait,
            qmin=inp.qmin,
            kv_cache_hit_rate=inp.kv_cache_hit_rate,
            avg_swapping=inp.avg_swapping,
            w_d=inp.w_d,
            factor=replica_factor(inp.assigned_replicas, effective_pods),
        )
        y_total = terms.numerator
        y_per_pod = y_total / effective_pods
        q = terms.queue
        q_ctl = terms.queue_ctl
        # Idle rule (plan 6.4): A + W == 0 -> TSS undefined. Reported as 0.0, which the
        # EMA passes through without advancing and compute_z_m maps to None.
        trs_raw = terms.raw if terms.raw is not None else 0.0
        idle = window_is_idle(inp.prompt_tokens_total, inp.generation_tokens_total)
        trs = self._update_ema(trs_raw, window_end_ms=window_end_ms, window_ms=inp.window_ms, idle=idle)
        eta = compute_eta_m(trs, effective_pods)
        z_m = compute_z_m(trs, theta_m)
        saved_prev_y = self._prev_Y
        saved_prev_q_ctl = self._prev_Q_ctl
        self._prev_Y = y_total
        self._prev_Q_ctl = q_ctl
        return TRSResult(
            Y_m=y_total,
            y_m=y_per_pod,
            Q=q,
            Q_ctl=q_ctl,
            TRS_raw=trs_raw,
            TRS=trs,
            eta_m=eta,
            Z_m=z_m,
            ema_alpha=self.ema_alpha,
            prev_Y=saved_prev_y,
            prev_Q_ctl=saved_prev_q_ctl,
            defined=terms.defined,
        )

    def _update_ema(
        self,
        raw: float,
        window_end_ms: int | None = None,
        window_ms: float | None = None,
        idle: bool = False,
    ) -> float:
        if self._ema is not None:
            # Time-constant EMA (S1.3 / ADR-0011): delegated to tre_common.tss.TssEma, the
            # one implementation the offline paths use as well (idle-gap reset after
            # window_ms, None/0 passthrough that still takes part in the gap check,
            # per-window dedup for the rescue/fairness/safescale re-reads, dt in data time).
            if window_end_ms is None:
                # No time reference -> cannot advance a wall-clock EMA. Passthrough.
                return raw
            value = self._ema.update(raw, window_end_ms, window_ms, idle)
            return raw if value is None else value
        # DEPRECATED legacy fixed-alpha branch (ema_tau_ms unset). Every deployed registry
        # entry sets ema_tau_ms; this path is kept only for the golden parity tests
        # (controller/tests/golden/legacy_trs.py, test_trs_signals.py) and has no idle-gap
        # reset. Byte-identical to pre-S1.3 behaviour when window_end_ms is None (golden).
        if not _is_finite_positive(raw):
            return raw
        if (
            window_end_ms is not None
            and window_end_ms == self._legacy_last_ms
            and self._legacy_ema is not None
        ):
            return self._legacy_ema
        if self.ema_alpha <= 0:
            self._legacy_ema = raw
            if window_end_ms is not None:
                self._legacy_last_ms = window_end_ms
            return raw
        if self._legacy_ema is None:
            self._legacy_ema = raw
        else:
            self._legacy_ema = self.ema_alpha * self._legacy_ema + (1 - self.ema_alpha) * raw
        if window_end_ms is not None:
            self._legacy_last_ms = window_end_ms
        return self._legacy_ema


# ADR-0014: the SaturationResult / SaturationGuard classes (qsat/epsat/hsat -> is_saturated)
# were removed. Scaling and fairness receiver eligibility are decided solely by z_m
# threshold bands (tau_crit/tau_low/tau_high). See docs/refactor/DECISIONS.md ADR-0014.


def _is_finite_positive(value: float) -> bool:
    if value != value:
        return False
    if value == float("inf") or value == float("-inf"):
        return False
    if value == 0:
        return False
    return True


def compute_eta_m(trs: float, routable_pods: int | float) -> float | None:
    if not _is_finite_positive(trs):
        return None
    try:
        effective_pods = max(1.0, float(routable_pods))
    except (TypeError, ValueError):
        effective_pods = 1.0
    return trs / effective_pods


def compute_z_m(trs: float, theta_m: float | None) -> float | None:
    if theta_m is None or theta_m <= 0:
        return None
    if not _is_finite_positive(trs):
        return None
    return trs / theta_m


_compute_z_m = compute_z_m


class SignalState:
    """Per-model registry of stateful ``TRSComputer`` instances shared across the
    rescue / fairness / safescale loops (S1.3 / ADR-0011).

    The live control path previously constructed a fresh ``TRSComputer`` every
    tick, so the EMA never persisted (``TRS == TRS_raw`` always). Holding one
    computer per model here lets the wall-clock time-constant EMA carry across
    ticks. One shared computer per model means one EMA per model (rescue and
    fairness share it), per the "one window, one theta, one EMA" contract.

    In-process only: a controller restart starts every EMA from scratch, which is exactly
    the state an idle reset leaves behind (restart duality): after a restart the next
    defined sample seeds the EMA, as it would after an idle gap.

    **Idle reset.** One predicate, :func:`tre_common.tss.window_is_idle` (no token in the
    window), defines an idle window everywhere: every EMA update receives it
    (``TssEma.update(idle=...)``: clear and pass through), the tick loop derives
    ``has_traffic`` from it for ``observe_traffic`` (which clears the warmup onset and,
    idempotently, the model's EMAs), and the offline smoothing computes it from each window
    row. Independently, ``TssEma``'s idle-gap rule clears an EMA when a sample arrives more
    than one metrics window after the last advancing sample.

    Also tracks a per-model **traffic-onset** cursor for the F-onset warmup guard
    (see ``observe_traffic``): at load onset the sliding window is still filling with
    traffic, so TRS is structurally low (window-fill fraction) and z_m dips falsely
    CRITICAL. The guard suppresses receiver (CRITICAL/LOW) scale-ups until the window
    lies fully inside the traffic period (ADR-0014: the saturation bypass was removed, so
    a genuine flash crowd in the warmup window is delayed at most one window). In-process
    only, like the EMA: after a restart mid-traffic, one window of receiver-suppression.
    """

    def __init__(
        self,
        warmup_ms: int = -1,
        *,
        dwell_windows: int = 1,
        dwell_states: Iterable[str] = DWELL_STATES,
    ) -> None:
        self._by_model: dict[str, TRSComputer] = {}
        # Band dwell (plan §6.9i / D8): CRITICAL / LOW / HIGH only act after holding for
        # dwell_windows consecutive NEW metrics windows (tre_common.dwell). 1 = off.
        self.dwell_windows = max(1, int(dwell_windows))
        states = {str(state).strip().lower() for state in dwell_states if str(state).strip()}
        unknown = states - set(DWELL_STATES)
        if unknown:
            raise ValueError(f"dwell_states must be a subset of {DWELL_STATES}, got {sorted(unknown)}")
        self.dwell_states = frozenset(states)
        # model -> band -> counter; bands: "critical" (Z < tau_crit), "receiver"
        # (Z < tau_low, CRITICAL or LOW), "high" (Z > tau_high).
        self._dwell: dict[str, dict[str, DwellCounter]] = {}
        # warmup_ms: -1 = auto (window fully inside traffic period), 0 = disabled
        # (pre-fix behaviour, for A/B ablation), >0 = explicit span since onset.
        self._warmup_ms = warmup_ms
        self._onset_ms: dict[str, int | None] = {}
        # One EMA per (model, alternative signal), same tau/alpha as the TSS EMA
        # (plan §6.9 item 4); see tre_controller.signals.sources._thresholded_signal.
        self._signal_ema: dict[tuple[str, str], TssEma] = {}

    def computer_for(self, model: str, *, ema_alpha: float, ema_tau_ms: float | None) -> TRSComputer:
        computer = self._by_model.get(model)
        if computer is None:
            computer = TRSComputer(ema_alpha=ema_alpha, ema_tau_ms=ema_tau_ms)
            self._by_model[model] = computer
        return computer

    def smooth_signal(
        self,
        model: str,
        source: str,
        raw: float | None,
        *,
        window_end_ms: int,
        tau_ms: float,
        window_ms: float | None = None,
        idle: bool = False,
    ) -> float | None:
        """EMA'd value of an alternative signal (``tre_common.tss.signal_ema``).

        Advances at most once per distinct ``window_end_ms`` (``TssEma`` keeps its value
        on a repeated window), so the rescue/fairness/safescale re-reads of one snapshot
        do not over-smooth - the same dedup rule as the TSS EMA.
        """
        key = (model, source)
        ema = self._signal_ema.get(key)
        if ema is None or ema.tau_ms != float(tau_ms):
            ema = signal_ema(tau_ms)
            self._signal_ema[key] = ema
        return ema.update(raw, window_end_ms, window_ms, idle)

    def reset_ema(self, model: str) -> None:
        """Clear every EMA of ``model`` (TSS and alternative signals) - the idle reset."""
        computer = self._by_model.get(model)
        if computer is not None:
            computer.reset_ema()
        for (owner, _source), ema in self._signal_ema.items():
            if owner == model:
                ema.reset()

    def observe_traffic(
        self, model: str, *, has_traffic: bool, window_start_ms: int, window_end_ms: int
    ) -> bool:
        """Return whether ``model``'s signal is 'warm' (trustworthy on the low side).

        Records the traffic-onset window_end on the first traffic-bearing tick; resets
        on an idle (no-traffic) tick. Warm iff the current sliding window no longer
        straddles the onset. Idempotent under the duplicate-window_end re-reads the
        rescue/fairness/safescale loops do (mirrors the EMA per-window dedup).

        ``has_traffic`` must be ``not tre_common.tss.window_is_idle(...)`` of the window -
        the predicate every EMA update receives - so the onset and the EMAs restart on the
        same condition. The idle tick also clears the model's EMAs (:meth:`reset_ema`); that
        is idempotent with the idle-window reset the EMA updates already applied, and it
        happens even when the warmup guard is disabled."""
        if not has_traffic:
            self.reset_ema(model)
            self.reset_dwell(model)
            self._onset_ms[model] = None
            return True  # idle -> UNKNOWN, nothing to warm up for
        if self._warmup_ms == 0:
            return True  # disabled
        onset = self._onset_ms.get(model)
        if onset is None:
            onset = window_end_ms
            self._onset_ms[model] = onset
        if self._warmup_ms < 0:
            # auto: warm once the whole window lies inside the traffic period.
            return window_start_ms >= onset
        return (window_end_ms - onset) >= self._warmup_ms

    # ------------------------------------------------------------------ band dwell

    def reset_dwell(self, model: str) -> None:
        for counter in self._dwell.get(model, {}).values():
            counter.reset()

    def dwell_run(self, model: str, band: str) -> int:
        counter = self._dwell.get(model, {}).get(band)
        return counter.run if counter is not None else 0

    def apply_dwell(
        self,
        classifications: list,
        contexts: dict[str, dict],
        windows: dict[str, ModelWindowMetrics],
    ) -> tuple[list, tuple[str, ...]]:
        """Gate band changes on ``dwell_windows`` consecutive new windows.

        Called once per planner tick, after classification. Counters advance at most once
        per distinct ``window_end_ms`` of the model's window (rescue/fairness re-reads of
        one snapshot do not count twice) and only for windows whose tokens are present
        (a scrape gap, whose context the paper-state cache may be holding, neither counts
        nor resets). Composition with the other guards:

        * warmup: a window whose signal is not yet warm (``signal_warm`` False) resets the
          receiver runs - the onset windows are exactly the structurally-low ones the
          warmup guard distrusts, so they must not pre-confirm a CRITICAL;
        * idle: an idle tick (``observe_traffic(has_traffic=False)``) resets every run,
          like the EMA; a gap > one metrics window restarts the run (TssEma's rule);
        * cooldown: independent - counting continues, the planner's cooldown still holds
          the action.

        Verdicts (``dwell_states`` picks which raw states are gated):

        * CRITICAL not confirmed -> LOW if the receiver band (Z < tau_low) is confirmed,
          otherwise the receiver is suppressed for this tick (``dwell_confirmed=False`` in
          its context; ``build_plan`` drops it like a warming-up receiver, so it is
          neither a receiver nor a donor);
        * LOW not confirmed -> suppressed the same way;
        * HIGH not confirmed -> HEALTHY/NEUTRAL (it is at least healthy).
        """
        if self.dwell_windows <= 1 or not self.dwell_states:
            return classifications, ()
        from tre_controller.planning.classify import ModelRole, ModelState

        events: list[str] = []
        out: list = []
        for item in classifications:
            model = item.model_name
            ctx = contexts.get(model)
            metrics = windows.get(model)
            if ctx is None or metrics is None:
                out.append(item)
                continue
            counters = self._dwell.setdefault(model, {})
            window_ms = float(metrics.window_end_ms - metrics.window_start_ms)
            for band in ("critical", "receiver", "high"):
                if band not in counters:
                    counters[band] = DwellCounter(required=self.dwell_windows, max_gap_ms=window_ms)
            tokens_present = metrics.prompt_tokens is not None and metrics.generation_tokens is not None
            if tokens_present:
                warm = bool(ctx.get("signal_warm", True))
                state = item.state
                end = int(metrics.window_end_ms)
                counters["critical"].update(end, state == ModelState.CRITICAL, eligible=warm)
                counters["receiver"].update(
                    end, state in (ModelState.CRITICAL, ModelState.LOW), eligible=warm
                )
                counters["high"].update(end, state == ModelState.HIGH)
            crit_ok = "critical" not in self.dwell_states or counters["critical"].confirmed
            recv_ok = "low" not in self.dwell_states or counters["receiver"].confirmed
            high_ok = "high" not in self.dwell_states or counters["high"].confirmed
            n = self.dwell_windows
            if item.state == ModelState.CRITICAL and not crit_ok:
                events.append(f"dwell_hold:{model}:critical:{counters['critical'].run}/{n}")
                if recv_ok:
                    item = replace(item, state=ModelState.LOW, role=ModelRole.RECEIVER)
                else:
                    ctx["dwell_confirmed"] = False
            elif item.state == ModelState.LOW and not recv_ok:
                events.append(f"dwell_hold:{model}:low:{counters['receiver'].run}/{n}")
                ctx["dwell_confirmed"] = False
            elif item.state == ModelState.HIGH and not high_ok:
                events.append(f"dwell_hold:{model}:high:{counters['high'].run}/{n}")
                item = replace(item, state=ModelState.HEALTHY, role=ModelRole.NEUTRAL, donor_tier=None)
            ctx["dwell_runs"] = {band: counter.run for band, counter in counters.items()}
            out.append(item)
        return out, tuple(events)
