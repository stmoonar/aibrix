"""PreServe Tier-1 (§4.1, Alg.2): window token forecast -> required instances.

``mlstm`` (default, decision 2026-10-26, v1 D15 / D57)
    the paper's forecaster: one mLSTM per token series (prompt P, response D) predicts the
    next window one step ahead from the last ``seq_len`` (25) windows (:class:`MlstmTier1`,
    weights :mod:`preserve_predictor`). The history holds the windows before ``t0`` (the
    history file, :mod:`preserve_history`) and, as the replay goes on, the **observed**
    windows: per model the sum of ``in_tokens`` and ``max_tokens`` of the gateway ``arr``
    events with ``reissue == "none"`` whose Redis time falls in the window (a reissue is not new
    demand). A prediction never enters the history.

Oracle modes (D8, sensitivity only; the trace itself is the forecast):

``oracle_noisy`` (default)
    the trace's true ``P_w`` / ``D_w`` times independent lognormal factors
    ``exp(sigma * z)``, ``z ~ N(0, 1)``. For that factor the mean absolute percentage error
    is exactly ``E|e^{sigma z} - 1| = e^{sigma^2/2} (2 Phi(sigma) - 1)``; the default
    ``sigma`` solves it for 6.17 %, the paper's overall mean APE (Table 1: 8.10 % code,
    4.23 % chat, 6.17 % overall), giving ``sigma ~= 0.0772``. The paper's *max* APE (24 %)
    is not matched: with a handful of windows per run the lognormal rarely strays that far.
    Each draw is seeded from ``(seed, model, window, "P"|"D")`` through SHA-256, so it does
    not depend on tick timing or on the order models are visited.
``oracle``
    the true values (an upper bound on any forecaster).
``last_window``
    the previous window's true values (a naive forecaster); window 0 has no history and
    Tier-1 holds (``tier1_no_history``).

Instances (Alg.2 line 7, units ours - the paper writes ``P / mu_p`` without units):
``N = ceil(max(P / (mu_p W), D / (mu_d W), (P + D) / (mu_t W)))`` where ``P``, ``D`` are
token totals of the window, ``W`` its effective length in seconds and ``mu_*`` the
per-replica token rates (tok/s) that ``tools/preserve_mu.py`` profiles as in Alg.1
(max per-replica rate over SLO-violation-free windows). Totals / (rate x seconds) is the
dimensionless replica count.
"""
from __future__ import annotations

import hashlib
import math
import random
from dataclasses import dataclass
from dataclasses import field
from typing import Any, Mapping, Optional, Sequence

TIER1_MODES = ("mlstm", "oracle_noisy", "oracle", "last_window")
ORACLE_MODES = ("oracle_noisy", "oracle", "last_window")
#: Paper Table 1, overall mean APE of PreServe's workload predictor.
PAPER_MEAN_APE = 0.0617


def mean_ape_of_sigma(sigma: float) -> float:
    """Exact mean APE of a lognormal factor ``exp(sigma z)``."""
    s = abs(float(sigma))
    return math.exp(s * s / 2.0) * math.erf(s / math.sqrt(2.0))


def sigma_for_mean_ape(ape: float) -> float:
    """Inverse of :func:`mean_ape_of_sigma` (bisection; monotone in sigma)."""
    if ape <= 0:
        return 0.0
    lo, hi = 0.0, 5.0
    for _ in range(200):
        mid = (lo + hi) / 2.0
        if mean_ape_of_sigma(mid) < ape:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


DEFAULT_NOISE_SIGMA = sigma_for_mean_ape(PAPER_MEAN_APE)


def _stable_seed(*parts: Any) -> int:
    digest = hashlib.sha256("|".join(str(p) for p in parts).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


def noise_factor(seed: int, model: str, window: int, which: str, sigma: float) -> float:
    if sigma <= 0:
        return 1.0
    z = random.Random(_stable_seed(seed, model, window, which)).gauss(0.0, 1.0)
    return math.exp(sigma * z)


@dataclass(frozen=True)
class Mu:
    p: float
    d: float
    t: float


def parse_mu(raw: Any, models: Any) -> dict[str, Mu]:
    """``mu: {model: {p, d, t}}`` -> Mu per model; every name in ``models`` must have all
    three positive (fail closed: a missing capacity would make Tier-1 silently wrong)."""
    if not isinstance(raw, Mapping):
        raise ValueError("preserve: params.mu must be a mapping {model: {p, d, t}} (tok/s per replica)")
    out: dict[str, Mu] = {}
    for model, spec in raw.items():
        if not isinstance(spec, Mapping):
            raise ValueError(f"preserve: mu[{model!r}] must be a mapping with p, d, t")
        vals = {}
        for key in ("p", "d", "t"):
            try:
                v = float(spec[key])
            except (KeyError, TypeError, ValueError):
                raise ValueError(f"preserve: mu[{model!r}].{key} is missing or not a number") from None
            if not math.isfinite(v) or v <= 0:
                raise ValueError(f"preserve: mu[{model!r}].{key} must be > 0, got {v}")
            vals[key] = v
        out[str(model)] = Mu(**vals)
    missing = sorted(str(m) for m in models if str(m) not in out)
    if missing:
        raise ValueError(f"preserve: params.mu has no capacity for managed models {missing}")
    return out


def required_replicas(P: float, D: float, mu: Mu, window_len_s: float) -> int:
    if window_len_s <= 0:
        raise ValueError("window length must be > 0")
    need = max(P / (mu.p * window_len_s), D / (mu.d * window_len_s),
               (P + D) / (mu.t * window_len_s))
    # 1e-9 keeps an exact integer (e.g. 2.0000000001 from float division) from rounding up.
    return int(math.ceil(need - 1e-9)) if need > 0 else 0


def estimate(mode: str, *, true_P: int, true_D: int, prev: Optional[tuple[int, int]],
             seed: int, model: str, window: int, sigma: float) -> Optional[tuple[float, float]]:
    """(P_hat, D_hat) for ``window``, or None when the mode has nothing to go on."""
    if mode == "oracle":
        return float(true_P), float(true_D)
    if mode == "oracle_noisy":
        return (true_P * noise_factor(seed, model, window, "P", sigma),
                true_D * noise_factor(seed, model, window, "D", sigma))
    if mode == "last_window":
        if prev is None:
            return None
        return float(prev[0]), float(prev[1])
    raise ValueError(f"unknown tier1 mode {mode!r}")


@dataclass
class _Obs:
    P: float = 0.0
    D: float = 0.0
    n: int = 0
    incomplete: int = 0


@dataclass
class MlstmTier1:
    """One model's mLSTM Tier-1 for one replay (``t0``); port of v1 ``tre_preserve.tier1.Tier1Model``.

    Windows ``i = floor((t - t0) / W)``; at the first tick of window ``i`` :meth:`decide` closes
    every window before ``i`` (appends its observed ``(P, D)`` to the history, oldest out) and
    predicts window ``i``."""

    model: str
    t0_ms: int
    window_s: float
    n_windows: int
    predictors: Mapping[str, Any]
    mu: Mu
    seq_len: int
    history: Sequence[Sequence[float]]
    hist_p: list = field(default_factory=list)
    hist_d: list = field(default_factory=list)
    observed: int = 0
    closed: int = -1
    pre_t0_arrivals: int = 0
    last_closed: Optional[dict] = None
    _obs: dict = field(default_factory=dict)
    _flags: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if len(self.history) < self.seq_len:
            raise ValueError(f"{self.model}: history has {len(self.history)} windows < seq_len {self.seq_len}")
        pre = list(self.history)[-self.seq_len:]
        self.hist_p = [float(w[0]) for w in pre]
        self.hist_d = [float(w[1]) for w in pre]

    def index(self, t_ms: int) -> int:
        return int(math.floor((int(t_ms) - self.t0_ms) / (self.window_s * 1000.0)))

    def observe_arrival(self, t_ms: int, in_tokens: Optional[int], max_tokens: Optional[int]) -> None:
        i = self.index(t_ms)
        if i < 0:
            self.pre_t0_arrivals += 1
            return
        if i <= self.closed:  # not with Redis-time stream IDs read after the boundary
            self.flag(i, "late_event")
            return
        o = self._obs.setdefault(i, _Obs())
        if in_tokens is None or max_tokens is None:
            o.incomplete += 1
            return
        o.P += max(0, int(in_tokens))
        o.D += max(0, int(max_tokens))
        o.n += 1

    def flag(self, i: int, why: str) -> None:
        """Mark window ``i`` as observed with a gap (logged with the closed window)."""
        if i >= 0:
            self._flags.setdefault(i, set()).add(why)

    def _close_until(self, idx_now: int) -> None:
        for i in range(self.closed + 1, min(idx_now, self.n_windows)):
            o = self._obs.pop(i, _Obs())
            self.hist_p = (self.hist_p + [o.P])[-self.seq_len:]
            self.hist_d = (self.hist_d + [o.D])[-self.seq_len:]
            self.observed += 1
            flags = sorted(self._flags.pop(i, set()))
            if o.incomplete:
                flags.append("incomplete_events")
            self.last_closed = {"window": i, "P": o.P, "D": o.D, "requests": o.n, "flags": flags}
            self.closed = i

    def decide(self, idx_now: int) -> tuple[Optional[int], dict]:
        """(N or None on a predictor error, info) for window ``idx_now``."""
        self._close_until(idx_now)
        info: dict = {"window": idx_now, "mode": "mlstm", "hist_observed": min(self.observed, self.seq_len),
                      "last_closed": self.last_closed}
        try:
            p_hat = float(self.predictors["prompt"].predict(list(self.hist_p)))
            d_hat = float(self.predictors["response"].predict(list(self.hist_d)))
        except Exception as e:  # noqa: BLE001 - a bad forecast is logged, Tier-1 keeps its last N
            info["error"] = f"{type(e).__name__}: {e}"
            return None, info
        if not (math.isfinite(p_hat) and math.isfinite(d_hat)):
            info["error"] = f"non-finite forecast P={p_hat} D={d_hat}"
            return None, info
        p_hat, d_hat = max(0.0, p_hat), max(0.0, d_hat)
        n = required_replicas(p_hat, d_hat, self.mu, self.window_s)
        info.update(P_hat=round(p_hat, 1), D_hat=round(d_hat, 1), W=self.window_s, N=n,
                    terms=[round(p_hat / (self.mu.p * self.window_s), 3), round(d_hat / (self.mu.d * self.window_s), 3),
                           round((p_hat + d_hat) / (self.mu.t * self.window_s), 3)])
        return n, info
