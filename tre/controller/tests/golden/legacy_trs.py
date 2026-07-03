from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional


@dataclass
class LegacyTRSInput:
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


@dataclass
class LegacyTRSResult:
    Y_m: float
    y_m: float
    Q: float
    Q_ctl: float
    TRS_raw: float
    TRS: float
    eta_m: Optional[float]
    Z_m: Optional[float]
    ema_alpha: float
    prev_Y: Optional[float] = None
    prev_Q_ctl: Optional[float] = None


class LegacyTRSComputer:
    def __init__(self, ema_alpha: float = 0.5) -> None:
        self.ema_alpha = ema_alpha
        self._trs_ema: Optional[float] = None
        self._prev_Y: Optional[float] = None
        self._prev_Q_ctl: Optional[float] = None

    @property
    def current_ema(self) -> Optional[float]:
        return self._trs_ema

    def restore(
        self,
        *,
        ema: Optional[float] = None,
        prev_Y: Optional[float] = None,
        prev_Q_ctl: Optional[float] = None,
    ) -> None:
        if ema is not None:
            self._trs_ema = ema
        if prev_Y is not None:
            self._prev_Y = prev_Y
        if prev_Q_ctl is not None:
            self._prev_Q_ctl = prev_Q_ctl

    def snapshot(self) -> dict[str, Any]:
        return {"ema": self._trs_ema, "prev_Y": self._prev_Y, "prev_Q_ctl": self._prev_Q_ctl}

    def compute(self, inp: LegacyTRSInput, theta_m: Optional[float] = None) -> LegacyTRSResult:
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
        trs = self._update_ema(trs_raw)
        eta = legacy_compute_eta_m(trs, effective_pods)
        z_m = legacy_compute_z_m(trs, theta_m)
        saved_prev_y = self._prev_Y
        saved_prev_q_ctl = self._prev_Q_ctl
        self._prev_Y = y_total
        self._prev_Q_ctl = q_ctl
        return LegacyTRSResult(
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

    def _update_ema(self, raw: float) -> float:
        if not legacy_is_finite_positive(raw):
            return raw
        if self.ema_alpha <= 0:
            self._trs_ema = raw
            return raw
        if self._trs_ema is None:
            self._trs_ema = raw
        else:
            self._trs_ema = self.ema_alpha * self._trs_ema + (1 - self.ema_alpha) * raw
        return self._trs_ema


@dataclass
class LegacySaturationResult:
    gamma: Optional[float]
    sat_windows: int
    is_saturated: bool
    last_q_ctl: float
    last_y: float


class LegacySaturationGuard:
    def __init__(self, qsat: float = 4.0, epsat: float = 0.05, Hsat: int = 3) -> None:
        self.qsat = qsat
        self.epsat = epsat
        self.Hsat = Hsat
        self._sat_windows = 0
        self._last_gamma: Optional[float] = None

    @property
    def current_sat_windows(self) -> int:
        return self._sat_windows

    @property
    def last_gamma(self) -> Optional[float]:
        return self._last_gamma

    def restore(self, sat_windows: int = 0, gamma: Optional[float] = None) -> None:
        self._sat_windows = max(0, sat_windows)
        self._last_gamma = gamma

    def snapshot(self) -> dict[str, Any]:
        return {"sat_windows": self._sat_windows, "gamma": self._last_gamma}

    def evaluate(self, trs_result: LegacyTRSResult) -> LegacySaturationResult:
        gamma: Optional[float] = None
        if trs_result.prev_Y is not None and trs_result.prev_Q_ctl is not None:
            dq = trs_result.Q_ctl - trs_result.prev_Q_ctl
            if abs(dq) > 1e-12:
                gamma = (trs_result.Y_m - trs_result.prev_Y) / dq
        sat_this_window = False
        if trs_result.Q_ctl >= self.qsat and gamma is not None and abs(gamma) <= self.epsat:
            sat_this_window = True
        if sat_this_window:
            self._sat_windows += 1
        else:
            self._sat_windows = 0
        is_saturated = self._sat_windows >= self.Hsat
        self._last_gamma = gamma
        return LegacySaturationResult(
            gamma=gamma,
            sat_windows=self._sat_windows,
            is_saturated=is_saturated,
            last_q_ctl=trs_result.Q_ctl,
            last_y=trs_result.Y_m,
        )


def legacy_is_finite_positive(value: float) -> bool:
    if value != value:
        return False
    if value == float("inf") or value == float("-inf"):
        return False
    if value == 0:
        return False
    return True


def legacy_compute_eta_m(trs: float, routable_pods: int | float) -> Optional[float]:
    if not legacy_is_finite_positive(trs):
        return None
    try:
        effective_pods = max(1.0, float(routable_pods))
    except (TypeError, ValueError):
        effective_pods = 1.0
    return trs / effective_pods


def legacy_compute_z_m(trs: float, theta_m: Optional[float]) -> Optional[float]:
    if theta_m is None or theta_m <= 0:
        return None
    if not legacy_is_finite_positive(trs):
        return None
    return trs / theta_m
