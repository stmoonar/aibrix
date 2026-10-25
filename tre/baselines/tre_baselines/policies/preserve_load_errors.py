"""Response-length errors for "PreServe w/ load errors" (paper §5.3 RQ2, error model of Tab. 2).

Port of the TRE v1 file (``tre_code`` dev ``ceb88b4``, ``tre_preserve/load_errors.py``; decision
2026-10-26). v2 change: the draw key is ``(seed, model, client request id)`` (gateway field
``client_id`` = ``x-tre-request-id``, :attr:`tre_baselines.snapshot.RequestEvent.draw_key`), the
same id on every reissue; v1 keyed ``(seed, req_id)`` with its client's trace id.

The paper's RQ2 headline (0.250 s) is "PreServe w/ load errors": the ground-truth response
length plus "prediction errors based on the error proportion distribution from RQ1 Tab. 2".
Upstream ships no error-injection code, so the distribution is rebuilt from the table
(PreServe row, ShareGPT, lengths up to 4096):

    MAE 78.25 tokens;  |err| <= 25: 56.77 %,  <= 50: 68.79 %,  <= 100: 77.95 %

as a piecewise CDF of the **absolute** error: uniform on [0, 25], (25, 50], (50, 100] with those
masses (edges at b + 0.5, so the rounded integer error meets the shares exactly), and above
100 a shifted exponential ``100 + Exp(mean m)`` with ``m`` solved so that the
overall mean is the MAE (the exponential is the maximum-entropy tail on [100, inf) for a given
mean). The sign is symmetric (the table gives none). Ours: the tail shape and the symmetric sign.

Each request's error is a pure function of ``(seed, model, key)`` (sha256), so the same request gets
the same error in every arm and every re-run; a reissue (same client id) gets the same error as
its original. The predicted length is

    D_hat = clip(max_tokens + err, 1, max_model_len - P)

``max_tokens`` stands for the true output length D (v1 measured: output >= 99 % of ``max_tokens``,
``tre_v1/docs/preserve-port-design.md``); Tier-1 keeps using it as the observed ground truth. The clip at 1 shrinks large negative errors on
short requests, so the realised error is reported per run (scaler log ``inputs.load_err``).
"""
from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple

#: (|err| upper bound, cumulative share) of Tab. 2, PreServe row
TAB2_ACC: Tuple[Tuple[float, float], ...] = ((25.0, 0.5677), (50.0, 0.6879), (100.0, 0.7795))
TAB2_MAE = 78.25
MODES = ("none", "paper_tab2")


def _edges(acc: Sequence[Tuple[float, float]]) -> Tuple[Tuple[float, float], ...]:
    """Continuous bounds ``b + 0.5``: rounding the draw then gives integer errors whose
    shares ``|err| <= b`` are exactly the table's."""
    return tuple((b + 0.5, c) for b, c in acc)


def _tail_mean(acc: Sequence[Tuple[float, float]], mae: float) -> float:
    """Mean of the exponential part of the tail (above the last bound) that makes E|err| = mae."""
    body, lo, prev = 0.0, 0.0, 0.0
    for hi, cum in _edges(acc):
        body += (cum - prev) * (lo + hi) / 2.0
        lo, prev = hi, cum
    tail_mass = 1.0 - prev
    m = (mae - body) / tail_mass - lo
    if not m > 0:
        raise ValueError(f"MAE {mae} too small for the table {acc}")
    return m


@dataclass(frozen=True)
class Tab2Errors:
    seed: int
    acc: Tuple[Tuple[float, float], ...] = TAB2_ACC
    mae: float = TAB2_MAE

    @property
    def tail_mean(self) -> float:
        return _tail_mean(self.acc, self.mae)

    def abs_error(self, u: float) -> float:
        """Inverse CDF of |err| at ``u`` in [0, 1)."""
        lo, prev = 0.0, 0.0
        for hi, cum in _edges(self.acc):
            if u < cum:
                return lo + (hi - lo) * (u - prev) / (cum - prev)
            lo, prev = hi, cum
        rest = (u - prev) / (1.0 - prev)                      # in [0, 1)
        return lo - self.tail_mean * math.log(max(1e-300, 1.0 - rest))

    def _uniforms(self, key: str) -> Tuple[float, int]:
        h = hashlib.sha256(f"{self.seed}|{key}".encode("utf-8")).digest()
        u = int.from_bytes(h[:8], "big") / 2.0 ** 64
        return u, (1 if h[8] & 1 else -1)

    def error(self, model: str, key: str) -> int:
        """Signed token error of request ``key`` of ``model`` (deterministic in ``(seed, model, key)``)."""
        u, sign = self._uniforms(f"{model}|{key}")
        return sign * int(round(self.abs_error(u)))

    def predict(self, model: str, key: str, max_tokens: int, prompt_tokens: int, max_model_len: int) -> int:
        hi = max(1, int(max_model_len) - max(0, int(prompt_tokens)))
        return max(1, min(hi, int(max_tokens) + self.error(model, key)))


def make(mode: str, seed: int) -> Optional[Tab2Errors]:
    if mode == "none":
        return None
    if mode == "paper_tab2":
        return Tab2Errors(int(seed))
    raise ValueError(f"load_error_mode must be one of {MODES}, got {mode!r}")


class ErrorStats:
    """Realised length error of the requests entered into the map (first sends only)."""

    def __init__(self) -> None:
        self.n = 0
        self.abs_sum = 0.0
        self.ape_sum = 0.0
        self.within = [0, 0, 0]   # |err| <= 25 / 50 / 100

    def add(self, true_len: int, pred_len: int) -> None:
        e = abs(int(pred_len) - int(true_len))
        self.n += 1
        self.abs_sum += e
        self.ape_sum += e / max(1, int(true_len))
        for i, b in enumerate((25, 50, 100)):
            self.within[i] += e <= b

    def summary(self) -> Dict[str, float]:
        if not self.n:
            return {"n": 0}
        return {"n": self.n, "mae": round(self.abs_sum / self.n, 2), "mape": round(self.ape_sum / self.n, 4),
                "acc25": round(self.within[0] / self.n, 4), "acc50": round(self.within[1] / self.n, 4),
                "acc100": round(self.within[2] / self.n, 4)}

