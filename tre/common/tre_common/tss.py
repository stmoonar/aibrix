"""The one TSS (TRS) definition shared by the online controller and every offline path.

Plan 2026-09-21 §6.4 (unified definition)::

    TSS(t) = [ w_p * r_p(t) + r_d(t) ] / max( A(t) + w_q * W(t), qmin )

* ``r_p`` - prefill tokens per window that MISSED the prefix cache
  (``prompt_tokens * (1 - kv_cache_hit_rate)``);
* ``r_d`` - generated tokens per window (``generation_tokens``);
* **numerator source** (2026-10-28, user decision; design
  ``docs/design/20261028-tss-step-counters.md``): vLLM's per-step counters
  ``vllm:prompt_tokens_total`` / ``vllm:generation_tokens_total`` - the tokens the engine
  PROCESSED in the window (generated tokens every step, prompt tokens at the first token),
  their delta over the observed span scaled to the window. Before: the
  ``request_*_tokens`` histogram sums, i.e. the tokens of the requests that COMPLETED in
  the window (:data:`NUMERATOR_COMPLETION_HISTOGRAMS`; offline only, to reproduce old
  freezes);
* window = the metrics window ``TRE_METRICS_WINDOW_MS`` (20 s since 2026-10-28, 30 s
  before), so the numerator is a **window token total**, the convention main and v1
  always used - theta is therefore tied to the window length and to the numerator
  source: the registry stamps both per model (``trs.numerator``, ``trs.theta_window_ms``)
  and the controller refuses to start on a mismatch (:func:`theta_basis_errors`);
* ``A`` / ``W`` - running / waiting requests, the window mean of the instant samples;
* ``w_q`` is the registry's ``lambda_wait``; ``qmin`` (1.0) is a numerical guard only;
* the assigned/routable replica correction stays a multiplicative factor applied by the
  signal source (:func:`replica_factor`).

Changes against the v1/v2 code formula, all deliberate:

* the numerator stays a **window total** (tokens per window, window =
  ``TRE_METRICS_WINDOW_MS``), not a rate. Online and in the fit the window is the same,
  so Z is identical under either convention; only theta's magnitude differs (x window
  seconds).
  A rate-valued numerator was tried on 2026-09-22 (commit 22b59f96) and reverted by user
  decision; :func:`convert_window_total_theta` remains to express a theta in rate units;
* ``swapping`` is gone from the denominator (vLLM v1 never swaps; a non-zero value is
  logged and ignored);
* ``w_d`` is gone (it is 1 by definition; a registry value != 1 is logged and ignored);
* **idle rule**: ``A + W == 0`` means nothing is in flight, so there is no per-request
  service rate to speak of - TSS and Z are *undefined* (``raw is None``), never a small
  number that would read as CRITICAL. It is the only idle rule (2026-10-29,
  :func:`window_is_idle`): zero tokens with a queue (a stuck engine) is TSS = 0, Z = 0.

The EMA is a wall-clock time-constant EMA, ``alpha_k = 1 - exp(-dt_k / tau)``
(:func:`ema_alpha` / :func:`ema_step`), and is the same function online
(``tre_controller.signals.trs.TRSComputer``) and offline
(``tre_calibration.dataset`` recompute, ``tre_calibration.signals``).

**Idle-gap reset** (:class:`TssEma`; its threshold follows the window: 20 s since
2026-10-28; tau itself is a separate registry value, ``trs.ema_tau_ms``, 10 s, unchanged):
when a sample arrives more than one metrics window
after the last sample that advanced the EMA (``window_end_ms - last_ms > window_ms``), the
EMA is cleared first, so a traffic period never inherits the previous period's value
(it would otherwise survive with weight ``exp(-gap/tau)``). ``window_ms`` is the sample's
own window duration, i.e. the configured metrics window (``TRE_METRICS_WINDOW_MS``), not
a separate constant. Offline, every cell additionally starts from a fresh EMA.

Nothing here may import from the controller or calibration packages: both import it.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Iterable, Optional, Sequence

LOG = logging.getLogger(__name__)

#: Units the published theta is expressed in.
TSS_UNITS = "decode-equivalent tokens per metrics window (TRE_METRICS_WINDOW_MS) per in-flight request"

#: EMA time constant the plan fixes (§6.5: tau is designed, not fitted).
DEFAULT_EMA_TAU_MS = 20_000.0

#: Numerator sources a theta can be fitted on (registry ``trs.numerator``).
NUMERATOR_STEP_COUNTERS = "step_counters"
NUMERATOR_COMPLETION_HISTOGRAMS = "completion_histograms"
#: The numerator the online controller computes (``MetricsStore``).
TSS_NUMERATOR = NUMERATOR_STEP_COUNTERS


def theta_basis_errors(models: Iterable[object], *, window_ms: float, numerator: str = TSS_NUMERATOR) -> list[str]:
    """Why the registry's thetas do not fit this controller, one line per model (empty:
    they fit). A theta is a window token total of one numerator source, so it is only
    valid on the window and the numerator it was fitted on: each model's ``trs`` must
    stamp ``numerator`` == ``numerator`` and ``theta_window_ms`` == ``window_ms``. A
    missing stamp is a mismatch (a registry from before 2026-10-28: completion
    histograms, 30 s). Restarting a new controller on old thetas would shift every Z by
    the window ratio; the atomic release (thetas + stamps + controller) is the fix."""
    errors: list[str] = []
    for spec in models:
        trs = getattr(spec, "trs", None)
        name = getattr(spec, "name", "?")
        stamp_numerator = getattr(trs, "numerator", None)
        stamp_window = getattr(trs, "theta_window_ms", None)
        if stamp_numerator != numerator or stamp_window is None or float(stamp_window) != float(window_ms):
            errors.append(
                f"{name}: trs.numerator={stamp_numerator!r} trs.theta_window_ms={stamp_window!r}, "
                f"controller computes numerator={numerator!r} on TRE_METRICS_WINDOW_MS={int(window_ms)}"
            )
    return errors

_warned: set[str] = set()


def _warn_once(key: str, message: str) -> None:
    if key in _warned:
        return
    _warned.add(key)
    LOG.warning(message)


@dataclass(frozen=True)
class TssTerms:
    """Every intermediate of one window's raw TSS, so callers never recompute a piece."""

    #: ``w_p * r_p + r_d`` in tokens per window (fleet total, window = TRE_METRICS_WINDOW_MS).
    numerator: float
    #: ``A + w_q * W`` (unfloored).
    queue: float
    #: ``max(queue, qmin)``.
    queue_ctl: float
    #: ``A + W`` - the idle test.
    in_flight: float
    #: Raw TSS after the replica factor; ``None`` when undefined (idle: ``A + W == 0``).
    raw: Optional[float]

    @property
    def defined(self) -> bool:
        return self.raw is not None


def replica_factor(assigned_replicas: float, routable_pods: float) -> float:
    """assigned / routable correction, with the controller's historical guards."""
    effective_pods = max(1.0, float(routable_pods))
    assigned = float(assigned_replicas)
    if assigned <= 0:
        assigned = effective_pods
    return assigned / effective_pods


def tss_queue(avg_running: float, avg_waiting: float, lambda_wait: float) -> float:
    """The TSS denominator before the qmin guard: ``A + w_q * W`` (no swapping term)."""
    return float(avg_running) + float(lambda_wait) * float(avg_waiting)


def tss_terms(
    *,
    prompt_tokens: float,
    generation_tokens: float,
    avg_running: float,
    avg_waiting: float,
    w_p: float,
    lambda_wait: float,
    qmin: float = 1.0,
    kv_cache_hit_rate: float = 0.0,
    avg_swapping: float = 0.0,
    w_d: float = 1.0,
    factor: float = 1.0,
) -> TssTerms:
    """Raw (un-smoothed) TSS of one window under the unified definition.

    ``prompt_tokens`` / ``generation_tokens`` are the totals over one metrics window
    (``TRE_METRICS_WINDOW_MS``); the numerator is that window total, not a rate.
    ``avg_swapping`` and ``w_d`` are accepted for schema compatibility only.
    """
    if avg_swapping:
        _warn_once(
            "swapping",
            f"avg_swapping={avg_swapping!r} is non-zero; the unified TSS ignores swapping "
            "(vLLM v1 never swaps) - check the metrics source",
        )
    if w_d != 1.0:
        _warn_once(
            f"w_d:{w_d}",
            f"w_d={w_d!r} != 1 in the registry; the unified TSS fixes w_d = 1 and ignores it",
        )
    tokens_p = float(prompt_tokens) * (1.0 - float(kv_cache_hit_rate))
    tokens_d = float(generation_tokens)
    numerator = float(w_p) * tokens_p + tokens_d
    running = float(avg_running)
    waiting = float(avg_waiting)
    queue = tss_queue(running, waiting, lambda_wait)
    queue_ctl = max(queue, float(qmin))
    in_flight = running + waiting
    if in_flight <= 0.0:
        raw: Optional[float] = None
    elif queue_ctl > 0.0:
        raw = numerator / queue_ctl * float(factor)
    else:  # qmin <= 0 and an empty queue cannot happen with in_flight > 0; kept total.
        raw = math.inf if numerator > 0 else 0.0
    return TssTerms(
        numerator=numerator,
        queue=queue,
        queue_ctl=queue_ctl,
        in_flight=in_flight,
        raw=raw,
    )


def window_is_idle(avg_running: Optional[float], avg_waiting: Optional[float]) -> bool:
    """THE idle-window predicate = the TSS idle rule: nothing was in flight over the
    window (running + waiting == 0), so TSS and Z are undefined. Used by the online idle
    reset (``SignalState.reset_ema``) and by every EMA - online and offline - through
    ``TssEma.update(idle=...)``. Unknown gauges (``None``) are not idle: nothing is known
    about the window.

    2026-10-29 (design docs/design/20261029-tss-only.md): was "no token in the window". A
    window with zero tokens and a queue is not idle - TSS = 0, Z = 0 (a stuck engine).
    """
    if avg_running is None or avg_waiting is None:
        return False
    return not (float(avg_running) + float(avg_waiting) > 0.0)


def ema_alpha(dt_ms: float, tau_ms: float) -> float:
    """Weight of the NEW sample after ``dt_ms`` of data time: ``1 - exp(-dt/tau)``."""
    if tau_ms <= 0:
        raise ValueError("tau_ms must be positive")
    if dt_ms <= 0:
        return 0.0
    return 1.0 - math.exp(-float(dt_ms) / float(tau_ms))


def ema_step(prev: float, raw: float, dt_ms: float, tau_ms: float) -> float:
    """One wall-clock EMA update. Written as ``decay*prev + (1-decay)*raw`` - the exact
    float expression the controller has always used - so online and offline agree bitwise."""
    decay = math.exp(-float(dt_ms) / float(tau_ms))
    return decay * prev + (1.0 - decay) * raw


class TssEma:
    """Streaming form of :func:`smooth_series` for one cell (same rules, same floats).

    The one wall-clock EMA every signal is smoothed with (plan §6.9 item 4): TSS inside
    ``TRSComputer`` / the offline recompute, and every alternative signal of the ablation
    through :func:`signal_ema`. A zero raw is an ordinary sample (2026-10-29: for TSS too -
    an undefined TSS is ``None``, never a 0 placeholder).

    Rules of :meth:`update`, in order:

    0. **idle-window reset** - ``idle=True`` (:func:`window_is_idle`: nothing in flight)
       clears ``value`` / ``last_ms`` and passes the raw through without seeding.
       This is the same condition as the controller's idle tick, so with sliding windows a
       traffic gap shorter than ``window + step`` still resets, online and offline alike.
    1. **idle-gap reset** - if ``window_ms`` is given and ``window_end_ms - last_ms >
       window_ms``, clear ``value`` / ``last_ms``. Checked for *every* sample, including a
       None / non-finite raw that is then passed through. Strictly greater:
       back-to-back tumbling windows sit exactly one window apart and must not reset;
       one fully idle window in between (or any longer gap) does.
    2. a None / non-finite raw is passed through without advancing
       ``value`` or ``last_ms``;
    3. the first sample after construction or a reset seeds ``value = raw``;
    4. a repeated or regressed ``window_end_ms`` keeps the current value;
    5. otherwise one :func:`ema_step` with ``dt = window_end_ms - last_ms``.

    A fresh instance and a reset instance are the same state, so a controller restart is
    equivalent to an idle-gap reset (restart duality).
    """

    def __init__(self, tau_ms: float) -> None:
        if tau_ms is None or tau_ms <= 0:
            raise ValueError("tau_ms must be positive")
        self.tau_ms = float(tau_ms)
        self.value: Optional[float] = None
        self.last_ms: Optional[float] = None

    def reset(self) -> None:
        """Forget the EMA: the next advancing sample seeds it (same as a fresh instance)."""
        self.value = None
        self.last_ms = None

    def update(
        self,
        raw: Optional[float],
        window_end_ms: float,
        window_ms: Optional[float] = None,
        idle: bool = False,
    ) -> Optional[float]:
        if idle:
            self.reset()
            return raw
        if (
            window_ms is not None
            and window_ms > 0
            and self.last_ms is not None
            and float(window_end_ms) - self.last_ms > float(window_ms)
        ):
            self.reset()
        if raw is None or not math.isfinite(raw):
            return raw
        if self.value is None or self.last_ms is None:
            self.value, self.last_ms = raw, float(window_end_ms)
            return self.value
        dt = float(window_end_ms) - self.last_ms
        if dt <= 0:
            return self.value
        self.value = ema_step(self.value, raw, dt, self.tau_ms)
        self.last_ms = float(window_end_ms)
        return self.value


def signal_ema(tau_ms: float) -> TssEma:
    """The EMA an alternative (ablation) signal is smoothed with: the same class, the same
    tau and the same alpha as TSS."""
    return TssEma(tau_ms)


def smooth_series(
    raws: Sequence[Optional[float]],
    window_end_ms: Sequence[float],
    *,
    tau_ms: float,
    window_ms: Optional[float | Sequence[Optional[float]]] = None,
    idle: Optional[Sequence[bool]] = None,
) -> list[Optional[float]]:
    """EMA a single cell's raw series in data time, exactly as the controller would.

    Mirrors ``TRSComputer._update_ema`` in its time-constant mode (:class:`TssEma` rules):
    an undefined (None) raw is passed through without advancing the EMA or its timestamp;
    a repeated or regressed ``window_end_ms`` keeps the current EMA; the first defined
    sample seeds it; a gap of more than ``window_ms`` (a scalar or one value per sample)
    since the last advancing sample resets it. ``window_ms=None`` disables the gap rule.
    ``idle`` (one :func:`window_is_idle` flag per sample) applies the idle-window reset.
    """
    ema = TssEma(tau_ms)
    if window_ms is None or isinstance(window_ms, (int, float)):
        windows: Sequence[Optional[float]] = [window_ms] * len(raws)
    else:
        windows = window_ms
    idles: Sequence[bool] = idle if idle is not None else [False] * len(raws)
    return [
        ema.update(raw, end, win, flag)
        for raw, end, win, flag in zip(raws, window_end_ms, windows, idles)
    ]


def convert_window_total_theta(theta_total: float, window_ms: float) -> float:
    """theta in window-total units (the published convention) -> the same threshold in
    rate units (theta / window_s). Z is identical under both conventions; this is only for
    reporting / comparing against a rate-valued definition (e.g. the paper's r = tokens/s)."""
    return float(theta_total) / (float(window_ms) / 1000.0)


__all__: Iterable[str] = (
    "DEFAULT_EMA_TAU_MS",
    "NUMERATOR_COMPLETION_HISTOGRAMS",
    "NUMERATOR_STEP_COUNTERS",
    "TSS_NUMERATOR",
    "TSS_UNITS",
    "TssEma",
    "TssTerms",
    "theta_basis_errors",
    "convert_window_total_theta",
    "ema_alpha",
    "ema_step",
    "replica_factor",
    "signal_ema",
    "smooth_series",
    "tss_queue",
    "tss_terms",
    "window_is_idle",
)
