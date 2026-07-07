from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from tre_common.metrics_schema import ModelWindowMetrics
from tre_common.registry import TrsParams


@dataclass
class TRSInput:
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


class TRSComputer:
    def __init__(self, ema_alpha: float = 0.5, ema_tau_ms: float | None = None) -> None:
        self.ema_alpha = ema_alpha
        self.ema_tau_ms = ema_tau_ms
        self._trs_ema: float | None = None
        self._prev_Y: float | None = None
        self._prev_Q_ctl: float | None = None
        self._last_update_ms: int | None = None

    @property
    def current_ema(self) -> float | None:
        return self._trs_ema

    def restore(
        self,
        *,
        ema: float | None = None,
        prev_Y: float | None = None,
        prev_Q_ctl: float | None = None,
        last_update_ms: int | None = None,
    ) -> None:
        if ema is not None:
            self._trs_ema = ema
        if prev_Y is not None:
            self._prev_Y = prev_Y
        if prev_Q_ctl is not None:
            self._prev_Q_ctl = prev_Q_ctl
        if last_update_ms is not None:
            self._last_update_ms = last_update_ms

    def snapshot(self) -> dict[str, Any]:
        return {"ema": self._trs_ema, "prev_Y": self._prev_Y, "prev_Q_ctl": self._prev_Q_ctl}

    def compute(
        self, inp: TRSInput, theta_m: float | None = None, *, window_end_ms: int | None = None
    ) -> TRSResult:
        y_total = inp.prompt_tokens_total * (1 - inp.kv_cache_hit_rate) * inp.w_p + inp.generation_tokens_total * inp.w_d
        effective_pods = max(1, inp.routable_pods)
        y_per_pod = y_total / effective_pods
        q = inp.avg_waiting * inp.lambda_wait + inp.avg_running + inp.avg_swapping
        q_ctl = max(q, inp.qmin)
        if q_ctl > 0:
            trs_raw = y_total / q_ctl
        else:
            trs_raw = float("inf") if y_total > 0 else 0.0
        effective_assigned = inp.assigned_replicas
        if effective_assigned <= 0:
            effective_assigned = effective_pods
        if effective_pods > 0:
            trs_raw = trs_raw * effective_assigned / effective_pods
        trs = self._update_ema(trs_raw, window_end_ms=window_end_ms)
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
        )

    def _update_ema(self, raw: float, window_end_ms: int | None = None) -> float:
        if not _is_finite_positive(raw):
            # Non-finite/zero raw is a passthrough: never advance EMA or timestamp
            # (mirrors legacy behaviour; keeps the window_end_ms cursor clean).
            return raw
        # Per-window dedup (both modes): a shared computer is re-read by
        # rescue(5s)/fairness(10s)/safescale between metrics refreshes. The EMA
        # must advance at most once per distinct window_end_ms, else it over-
        # smooths on duplicate snapshots. When window_end_ms is None (offline /
        # golden path) this guard is inert and behaviour is byte-identical.
        if window_end_ms is not None and window_end_ms == self._last_update_ms and self._trs_ema is not None:
            return self._trs_ema
        tau = self.ema_tau_ms
        if tau is not None and tau > 0:
            # Wall-clock time-constant EMA (S1.3 / ADR-0011): smoothing strength is
            # set by tau alone, decoupled from refresh frequency. dt is measured in
            # data time (window_end_ms deltas), not scheduler wall-clock, so it only
            # advances when the underlying window actually advances.
            if window_end_ms is None:
                # No time reference -> cannot advance a wall-clock EMA. Passthrough.
                return raw
            if self._trs_ema is None or self._last_update_ms is None:
                self._trs_ema = raw
                self._last_update_ms = window_end_ms
                return raw
            dt_ms = window_end_ms - self._last_update_ms
            if dt_ms <= 0:
                # Window regressed (clock/window rewind): keep EMA, don't advance.
                return self._trs_ema
            decay = math.exp(-dt_ms / tau)
            self._trs_ema = decay * self._trs_ema + (1.0 - decay) * raw
            self._last_update_ms = window_end_ms
            return self._trs_ema
        # Legacy fixed-alpha branch: byte-identical to pre-S1.3 behaviour when
        # window_end_ms is None (golden). With a shared computer + window_end_ms it
        # advances once per window (the dedup above) using the fixed alpha.
        if self.ema_alpha <= 0:
            self._trs_ema = raw
            if window_end_ms is not None:
                self._last_update_ms = window_end_ms
            return raw
        if self._trs_ema is None:
            self._trs_ema = raw
        else:
            self._trs_ema = self.ema_alpha * self._trs_ema + (1 - self.ema_alpha) * raw
        if window_end_ms is not None:
            self._last_update_ms = window_end_ms
        return self._trs_ema


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

    In-process only: on controller restart the EMA re-seeds from raw within ~tau.

    Also tracks a per-model **traffic-onset** cursor for the F-onset warmup guard
    (see ``observe_traffic``): at load onset the sliding window is still filling with
    traffic, so TRS is structurally low (window-fill fraction) and z_m dips falsely
    CRITICAL. The guard suppresses receiver (CRITICAL/LOW) scale-ups until the window
    lies fully inside the traffic period (ADR-0014: the saturation bypass was removed, so
    a genuine flash crowd in the warmup window is delayed at most one window). In-process
    only, like the EMA: after a restart mid-traffic, one window of receiver-suppression.
    """

    def __init__(self, warmup_ms: int = -1) -> None:
        self._by_model: dict[str, TRSComputer] = {}
        # warmup_ms: -1 = auto (window fully inside traffic period), 0 = disabled
        # (pre-fix behaviour, for A/B ablation), >0 = explicit span since onset.
        self._warmup_ms = warmup_ms
        self._onset_ms: dict[str, int | None] = {}

    def computer_for(self, model: str, *, ema_alpha: float, ema_tau_ms: float | None) -> TRSComputer:
        computer = self._by_model.get(model)
        if computer is None:
            computer = TRSComputer(ema_alpha=ema_alpha, ema_tau_ms=ema_tau_ms)
            self._by_model[model] = computer
        return computer

    def observe_traffic(
        self, model: str, *, has_traffic: bool, window_start_ms: int, window_end_ms: int
    ) -> bool:
        """Return whether ``model``'s signal is 'warm' (trustworthy on the low side).

        Records the traffic-onset window_end on the first traffic-bearing tick; resets
        on an idle (no-traffic) tick. Warm iff the current sliding window no longer
        straddles the onset. Idempotent under the duplicate-window_end re-reads the
        rescue/fairness/safescale loops do (mirrors the EMA per-window dedup)."""
        if self._warmup_ms == 0:
            return True  # disabled
        if not has_traffic:
            self._onset_ms[model] = None
            return True  # idle -> UNKNOWN, nothing to warm up for
        onset = self._onset_ms.get(model)
        if onset is None:
            onset = window_end_ms
            self._onset_ms[model] = onset
        if self._warmup_ms < 0:
            # auto: warm once the whole window lies inside the traffic period.
            return window_start_ms >= onset
        return (window_end_ms - onset) >= self._warmup_ms
