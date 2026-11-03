"""PreServe Tier-1 (§4.1, Alg.2): window token forecast -> required instances.

``mlstm`` (default, decision 2026-10-26, v1 D15 / D57)
    the paper's forecaster: one mLSTM per token series (prompt P, response D) predicts one
    window ahead from the last ``seq_len`` (25) windows (:class:`MlstmTier1`,
    weights :mod:`preserve_predictor`). The history holds the windows before ``t0`` (the
    history file, :mod:`preserve_history`) and, as the replay goes on, the **observed**
    windows: per model the sum of ``in_tokens`` and ``max_tokens`` of the gateway ``arr``
    events with ``reissue == "none"`` whose Redis time falls in the window (a reissue is not new
    demand). A prediction never enters the history.

Observed ``D`` (``tier1_d_source``, 2026-11-04):

``done_tokens`` (default; paper Alg. 1 / Alg. 2)
    a window's ``P_j`` / ``D_j`` are the prompt / response tokens of the requests whose
    timestamp (arrival) falls in ``T_j`` (Alg. 1 "Aggregate the prefill and decode tokens
    within T_{i-k}, ..., T_i"; §4.1 "request timestamps, prompt tokens, and response tokens").
    ``P_j`` is known at arrival (``in_tokens``). A response length is known only when the
    request completes: ``D_j`` sums ``out_tokens`` (the usage ``completion_tokens``; for a
    stitched sidecar continuation the client stream's merged usage) of the ``done`` events with
    ``reissue == "none"`` of ``T_j``'s requests. Alg. 2 appends the ground truth "if T_i has
    concluded"; with windows of seconds a request often outlives its window, so ``D_j`` is
    written once known: a forecast at the start of ``T_i`` (instant ``t0 + i W``) counts
    exactly the ``done`` events with a Redis time before that instant, never an in-flight
    request; a later forecast sees the later ``done`` events of the same window. The info logs
    ``d_unknown`` (history requests whose ``done`` is not yet in). A ``done`` without usage
    flags the window ``done_no_usage`` and adds nothing.
``done_or_predicted`` (sensitivity only, not the paper)
    as ``done_tokens``, but a request of a concluded window still in flight at the forecast
    instant contributes its Tier-2 predicted length ``D_hat`` (the look-ahead map's prediction,
    with the configured load error, computed at arrival from ``max_tokens``), replaced by its
    ``out_tokens`` once its ``done`` precedes a later forecast instant. The info logs
    ``d_actual`` / ``d_predicted`` (history totals) and ``d_split`` ``[[window, actual,
    predicted]]`` for the windows with a predicted part.
``arrival_max_tokens`` (before 2026-11-04, kept to reproduce those runs)
    ``D_j`` = the ``max_tokens`` of ``T_j``'s arrivals, known at arrival.

Horizon (``tier1_horizon``, 2026-11-04), what the firing at the start of window ``i`` targets:

``paper_two_step`` (default; paper Alg. 2, §4.1 "Online prediction phase", §4.3.2)
    at the start of ``T_i`` the history holds the observed windows up to ``T_{i-1}``. Step 1
    predicts the current window, ``P_hat_i = M_P(P)``; step 2 appends that prediction to a
    *copy* of the history, ``P' = P + {P_hat_i}``, and predicts ``P_hat_{i+1} = M_P(P')``
    (same for ``D``). ``N_{i+1}`` from ``P_hat_{i+1}``, ``D_hat_{i+1}`` is the scaler's target
    from the start of ``T_i`` (§4.3.2: "At the beginning of each prediction time window T_i,
    if N_{i+1} exceeds N_c ... pre-initialized"; conversely isolated). The true ``P_i``, ``D_i``
    enter the history only when ``T_i`` has concluded. Oracle modes in this horizon:
    ``oracle`` / ``oracle_noisy`` give window ``i+1``'s (noisy) truth, ``last_window`` the
    latest observed window ``i-1`` (a last-value forecaster iterated twice).
``one_step_now`` (before 2026-11-04, kept to reproduce those runs)
    at the start of ``T_i`` one step predicts ``T_i`` itself, effective at once (v1 D18).

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
PAPER_TWO_STEP = "paper_two_step"
ONE_STEP_NOW = "one_step_now"
HORIZONS = (PAPER_TWO_STEP, ONE_STEP_NOW)
DONE_TOKENS = "done_tokens"
DONE_OR_PREDICTED = "done_or_predicted"
ARRIVAL_MAX_TOKENS = "arrival_max_tokens"
D_SOURCES = (DONE_TOKENS, DONE_OR_PREDICTED, ARRIVAL_MAX_TOKENS)
BY_DONE = (DONE_TOKENS, DONE_OR_PREDICTED)
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
    predicts window ``i + 1`` in two steps (``paper_two_step``, Alg. 2) or window ``i``
    (``one_step_now``)."""

    model: str
    t0_ms: int
    window_s: float
    n_windows: int
    predictors: Mapping[str, Any]
    mu: Mu
    seq_len: int
    history: Sequence[Sequence[float]]
    horizon: str = PAPER_TWO_STEP
    d_source: str = DONE_TOKENS
    hist_p: list = field(default_factory=list)
    hist_d: list = field(default_factory=list)
    #: per history slot, the observed window whose D is read from ``done`` events (None = fixed value)
    hist_dw: list = field(default_factory=list)
    observed: int = 0
    closed: int = -1
    pre_t0_arrivals: int = 0
    last_closed: Optional[dict] = None
    _obs: dict = field(default_factory=dict)
    _flags: dict = field(default_factory=dict)
    #: done_tokens: gateway req_id of a new-demand arrival -> its window, until its done
    _req_win: dict = field(default_factory=dict)
    #: done_tokens: window -> [(done Redis ms, out_tokens, req_id)]; requests per window
    _done: dict = field(default_factory=dict)
    #: done_or_predicted: window -> {req_id: D_hat}
    _pred: dict = field(default_factory=dict)
    _n_req: dict = field(default_factory=dict)
    unknown_done: int = 0

    def __post_init__(self) -> None:
        if self.horizon not in HORIZONS:
            raise ValueError(f"{self.model}: tier1_horizon must be one of {HORIZONS}, got {self.horizon!r}")
        if self.d_source not in D_SOURCES:
            raise ValueError(f"{self.model}: tier1_d_source must be one of {D_SOURCES}, got {self.d_source!r}")
        if len(self.history) < self.seq_len:
            raise ValueError(f"{self.model}: history has {len(self.history)} windows < seq_len {self.seq_len}")
        pre = list(self.history)[-self.seq_len:]
        self.hist_p = [float(w[0]) for w in pre]
        self.hist_d = [float(w[1]) for w in pre]
        self.hist_dw = [None] * len(pre)

    def index(self, t_ms: int) -> int:
        return int(math.floor((int(t_ms) - self.t0_ms) / (self.window_s * 1000.0)))

    def observe_arrival(self, t_ms: int, in_tokens: Optional[int], max_tokens: Optional[int],
                        req_id: Optional[str] = None, d_hat: Optional[int] = None) -> None:
        i = self.index(t_ms)
        if i < 0:
            self.pre_t0_arrivals += 1
            return
        if i <= self.closed:  # not with Redis-time stream IDs read after the boundary
            self.flag(i, "late_event")
            return
        o = self._obs.setdefault(i, _Obs())
        by_done = self.d_source in BY_DONE
        if in_tokens is None or (max_tokens is None and not by_done):
            o.incomplete += 1
            return
        o.P += max(0, int(in_tokens))
        o.n += 1
        if not by_done:
            o.D += max(0, int(max_tokens))
        elif req_id:
            self._req_win[req_id] = i
            self._n_req[i] = self._n_req.get(i, 0) + 1
            if self.d_source == DONE_OR_PREDICTED:
                self._pred.setdefault(i, {})[req_id] = max(0, int(d_hat or 0))
                if d_hat is None:
                    o.incomplete += 1  # no prediction (no max_tokens): counts 0 until done
        else:
            o.incomplete += 1  # its done cannot be matched

    def observe_done(self, t_ms: int, req_id: Optional[str], out_tokens: Optional[int]) -> None:
        """``done_tokens``: a new-demand request completed; its response tokens join its arrival window."""
        if self.d_source not in BY_DONE:
            return
        i = self._req_win.pop(req_id, None) if req_id else None
        if i is None:
            self.unknown_done += 1  # arrived before t0, or its arrival was not counted
            return
        if out_tokens is None:
            self.flag(i, "done_no_usage")
            return
        self._done.setdefault(i, []).append((int(t_ms), max(0, int(out_tokens)), req_id))

    def _d_known(self, i: int, cut_ms: int) -> tuple[float, float, int]:
        """Window ``i`` as known at ``cut_ms``: (D from its done events before then, D_hat of its
        other requests (``done_or_predicted``; else 0), requests whose done is not in yet)."""
        seen = [(out, rid) for ts, out, rid in self._done.get(i, ()) if ts < cut_ms]
        actual = float(sum(out for out, _ in seen))
        pred = 0.0
        if self.d_source == DONE_OR_PREDICTED:
            ids = {rid for _, rid in seen}
            pred = float(sum(d for rid, d in self._pred.get(i, {}).items() if rid not in ids))
        return actual, pred, self._n_req.get(i, 0) - len(seen)

    def _history_d(self, idx_now: int) -> tuple[list, dict]:
        """The D history as known at the start of window ``idx_now`` and what it is made of."""
        if self.d_source not in BY_DONE:
            return list(self.hist_d), {}
        cut = self.t0_ms + int(round(idx_now * self.window_s * 1000.0))
        out, unknown, tot_a, tot_p, split = [], 0, 0.0, 0.0, []
        for v, w in zip(self.hist_d, self.hist_dw):
            if w is None:
                out.append(v)
                continue
            a, p, u = self._d_known(w, cut)
            out.append(a + p)
            unknown += u
            tot_a, tot_p = tot_a + a, tot_p + p
            if p > 0:
                split.append([w, a, p])
        info: dict = {"d_unknown": unknown}
        if self.d_source == DONE_OR_PREDICTED:
            info.update(d_actual=tot_a, d_predicted=tot_p, d_split=split)
        return out, info

    def flag(self, i: int, why: str) -> None:
        """Mark window ``i`` as observed with a gap (logged with the closed window)."""
        if i >= 0:
            self._flags.setdefault(i, set()).add(why)

    def _close_until(self, idx_now: int) -> None:
        for i in range(self.closed + 1, min(idx_now, self.n_windows)):
            o = self._obs.pop(i, _Obs())
            self.hist_p = (self.hist_p + [o.P])[-self.seq_len:]
            by_done = self.d_source in BY_DONE
            self.hist_d = (self.hist_d + [0.0 if by_done else o.D])[-self.seq_len:]
            self.hist_dw = (self.hist_dw + [i if by_done else None])[-self.seq_len:]
            self.observed += 1
            flags = sorted(self._flags.pop(i, set()))
            if o.incomplete:
                flags.append("incomplete_events")
            self.last_closed = {"window": i, "P": o.P, "D": None if by_done else o.D, "requests": o.n,
                                "flags": flags}
            self.closed = i
        # windows that left the history: their done events are no longer read
        keep = {w for w in self.hist_dw if w is not None}
        for store in (self._done, self._n_req, self._pred):
            for w in [w for w in store if w <= self.closed and w not in keep]:
                store.pop(w)
        if self._req_win:
            self._req_win = {r: w for r, w in self._req_win.items() if w > self.closed or w in keep}

    def _step(self, hist_p: list, hist_d: list) -> tuple[float, float]:
        """One forecast step: (P_hat, D_hat) of the window after ``hist``; raises on a bad value."""
        p_hat = float(self.predictors["prompt"].predict(list(hist_p)))
        d_hat = float(self.predictors["response"].predict(list(hist_d)))
        if not (math.isfinite(p_hat) and math.isfinite(d_hat)):
            raise ValueError(f"non-finite forecast P={p_hat} D={d_hat}")
        return max(0.0, p_hat), max(0.0, d_hat)

    def decide(self, idx_now: int) -> tuple[Optional[int], dict]:
        """(N or None on a predictor error, info) for the target window: ``idx_now + 1``
        (``paper_two_step``) or ``idx_now`` (``one_step_now``)."""
        self._close_until(idx_now)
        two = self.horizon == PAPER_TWO_STEP
        info: dict = {"window": idx_now, "target": idx_now + 1 if two else idx_now, "mode": "mlstm",
                      "horizon": self.horizon, "hist_observed": min(self.observed, self.seq_len),
                      "last_closed": self.last_closed}
        hist_d, d_info = self._history_d(idx_now)
        info.update(d_info)
        try:
            p_hat, d_hat = self._step(self.hist_p, hist_d)
            if two:
                # Alg. 2: P' = P + {P_hat_i} is a copy; the history itself stays observed-only.
                info.update(P_hat_cur=round(p_hat, 1), D_hat_cur=round(d_hat, 1))
                k = self.seq_len
                p_hat, d_hat = self._step((self.hist_p + [p_hat])[-k:], (hist_d + [d_hat])[-k:])
        except Exception as e:  # noqa: BLE001 - a bad forecast is logged, Tier-1 keeps its last N
            info["error"] = f"{type(e).__name__}: {e}"
            return None, info
        n = required_replicas(p_hat, d_hat, self.mu, self.window_s)
        info.update(P_hat=round(p_hat, 1), D_hat=round(d_hat, 1), W=self.window_s, N=n,
                    terms=[round(p_hat / (self.mu.p * self.window_s), 3), round(d_hat / (self.mu.d * self.window_s), 3),
                           round((p_hat + d_hat) / (self.mu.t * self.window_s), 3)])
        return n, info
