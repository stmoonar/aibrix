from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from tre_common.metrics_schema import ModelWindowMetrics
from tre_common.registry import TrsParams
from tre_common.tss import TssEma, replica_factor, signal_ema, tss_terms, window_is_idle


@dataclass
class TRSInput:
    """Inputs of one window's TSS (the unified definition, ``tre_common.tss``).

    ``prompt_tokens_total`` / ``generation_tokens_total`` are window totals and the TSS
    numerator is that total (tokens per window, the window's own length; 2026-10-28: from
    vLLM's step counters, ``tre_common.tss``).
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
    #: False when the window's zero tokens may be missing data, not a stall: some pod of
    #: it has a stale scrape or incomplete samples (:func:`zero_tokens_unconfirmed`).
    #: Then a zero-token window is treated as before 2026-10-29: Z unknown, EMA reset.
    zero_tokens_confirmed: bool = True

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
            zero_tokens_confirmed=not zero_tokens_unconfirmed(metrics),
        )


def zero_tokens_unconfirmed(metrics: ModelWindowMetrics) -> bool:
    """The window has zero tokens, but some pod of it has a stale scrape or incomplete
    samples (frozen counters, a short token span, a pod left out): the zero may be missing
    data. Only a zero-token window of complete, fresh samples is a stall (TSS = 0, Z = 0)."""
    if metrics.prompt_tokens is None or metrics.generation_tokens is None:
        return False
    if float(metrics.prompt_tokens) + float(metrics.generation_tokens) > 0.0:
        return False
    return bool(getattr(metrics, "scrape_stale_pods", ()) or getattr(metrics, "sample_incomplete_pods", ()))


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
    #: A defined TSS of 0 (zero tokens with a queue: a stalled engine) is Z = 0.
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
        # ema_tau_ms <= 0 is tau = 0, i.e. alpha = 1: no smoothing, TSS = raw - what the
        # offline fit scores (theta_verdict.build_signal_spec maps tau <= 0 to no EMA). It
        # used to fall through to the fixed-alpha branch below, whose ema_alpha weighs the
        # OLD value: with the ema_alpha: 1.0 that alpha_fit.registry_fields writes for
        # tau = 0 the TSS would freeze at its first sample. Only an unset tau (None) is legacy.
        self._raw_passthrough = ema_tau_ms is not None and ema_tau_ms <= 0
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
        self,
        inp: TRSInput,
        theta_m: float | None = None,
        *,
        window_end_ms: int | None = None,
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
        # Idle rule (plan 6.4; the only idle rule since 2026-10-29): A + W == 0 -> TSS
        # undefined: None to the EMA (idle reset), reported as 0.0, Z None. Zero tokens
        # with a queue is a defined TSS of 0 -> Z = 0, unless the zero may be missing data
        # (``zero_tokens_confirmed`` False): then as before - EMA reset, Z None.
        unconfirmed_zero = terms.defined and terms.numerator <= 0.0 and not inp.zero_tokens_confirmed
        defined = terms.defined and not unconfirmed_zero
        idle = window_is_idle(inp.avg_running, inp.avg_waiting) or unconfirmed_zero
        smoothed = self._update_ema(
            terms.raw if defined else None, window_end_ms=window_end_ms, window_ms=inp.window_ms, idle=idle
        )
        trs_raw = terms.raw if terms.raw is not None else 0.0
        trs = smoothed if defined and smoothed is not None else 0.0
        eta = compute_eta_m(trs, effective_pods)
        z_m = compute_z_m(trs, theta_m) if defined else None
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
        raw: float | None,
        window_end_ms: int | None = None,
        window_ms: float | None = None,
        idle: bool = False,
    ) -> float | None:
        if self._raw_passthrough:
            return raw
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
        if raw is None or not _is_finite_positive(raw):
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
    """Z = TSS / theta; 0 for a TSS of 0 (a stall). The caller passes only a defined TSS
    (an undefined one is reported as 0.0 and must not become Z = 0)."""
    if theta_m is None or theta_m <= 0:
        return None
    if trs is None or trs != trs or trs in (float("inf"), float("-inf")) or trs < 0:
        return None
    return trs / theta_m


_compute_z_m = compute_z_m


class SignalState:
    """Per-model registry of stateful ``TRSComputer`` instances shared across the
    decision loop and the SafeScale loop (S1.3 / ADR-0011; one decision loop since
    2026-10-28): one EMA per model (rescue and fairness share it), per the "one window,
    one theta, one EMA" contract.

    In-process only: a controller restart starts every EMA from scratch, which is exactly
    the state an idle reset leaves behind (restart duality).

    **Idle reset.** One predicate, :func:`tre_common.tss.window_is_idle` (nothing in flight:
    running + waiting == 0), defines an idle window everywhere: every EMA update receives it
    (``TssEma.update(idle=...)``) and the tick clears the model's EMAs on it
    (:meth:`reset_ema`, also the alternative signals'). Independently, ``TssEma``'s
    idle-gap rule clears an EMA when a sample arrives more than one metrics window after
    the last advancing sample.

    TSS only (2026-10-29, design docs/design/20261029-tss-only.md): Z is always computed on
    the whole sliding window plus this EMA - no breakpoint window, no onset guard, no
    saturation rescue.
    """

    def __init__(self) -> None:
        self._by_model: dict[str, TRSComputer] = {}
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

    def held_signal(
        self,
        model: str,
        source: str,
        *,
        window_end_ms: int,
        tau_ms: float,
        window_ms: float | None = None,
        idle: bool = False,
    ) -> float | None:
        """The last smoothed value of an alternative signal for a window that has no
        sample of it (ttft_mean: no first token in the window), or None.

        v1 held the last EMA in such a window. Here the window goes through the same
        ``TssEma`` rules as a sample (``update(None, ...)``): an idle window or a gap of
        more than one metrics window since the last sample clears the EMA, so a value is
        held for at most one window and is then unknown, never a stale healthy value.
        """
        ema = self._signal_ema.get((model, source))
        if ema is None or ema.tau_ms != float(tau_ms):
            return None
        ema.update(None, window_end_ms, window_ms, idle)
        return ema.value

    def reset_ema(self, model: str) -> None:
        """Clear every EMA of ``model`` (TSS and alternative signals) - the idle reset."""
        computer = self._by_model.get(model)
        if computer is not None:
            computer.reset_ema()
        for (owner, _source), ema in self._signal_ema.items():
            if owner == model:
                ema.reset()
