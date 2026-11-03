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

Mode ``paper_tab2_relative`` (default since 2026-11-04). The table measures tokens on ShareGPT
responses (mean 363 tokens); added as absolute tokens to our much shorter outputs it is a far larger
relative error than the paper's predictor makes (56-73 % mean on the Real_* traces). This mode
makes the error relative to the true length, ``err = sign * round(|r| * D)``, with ``|r|``
independent of ``D`` and

    |r| ~ (1 - w) LogNormal(mu1, sigma) + w LogNormal(mu2, sigma)
    w = 0.36802, mu1 = -3.34518, mu2 = -0.83636, sigma = 0.58512

i.e. 63 % of requests within a median |r| of 3.5 %, 37 % with a median |r| of 43 %. The four
parameters are solved (``tools/preserve_load_error_fit``) so that this rule applied to the ShareGPT
responses (the paper's test data: first ``gpt`` reply per conversation, 1..4096 LLaMA-2 tokens,
92 751 replies of ``ShareGPT_V3_unfiltered_cleaned_split.json``, sha256 ``35f0e213...``) reproduces
Tab. 2 exactly: MAE 78.25, Acc-25/50/100 56.77 / 68.79 / 77.95 %. Ours: the relative form, the
mixture family and the symmetric sign. ``paper_tab2`` (the absolute rule above) is kept to
reproduce runs before 2026-11-04.

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
from statistics import NormalDist
from typing import Dict, Optional, Sequence, Tuple

#: (|err| upper bound, cumulative share) of Tab. 2, PreServe row
TAB2_ACC: Tuple[Tuple[float, float], ...] = ((25.0, 0.5677), (50.0, 0.6879), (100.0, 0.7795))
TAB2_MAE = 78.25
REL_MIXTURE = {"w": 0.36801530958020456, "mu1": -3.345178146309775, "mu2": -0.8363588453768755,
               "sigma": 0.5851190727377948}
#: the data REL_MIXTURE was fitted on (tools/preserve_load_error_fit)
REL_FIT_SOURCE = {"sharegpt_sha256": "35f0e213ce091ed9b9af2a1f0755e9d39f9ccec34ab281cd4ca60d70f6479ba4",
                  "tokenizer": "CodeLlama-13b-hf (LLaMA-2 vocabulary)", "replies": 92751}
MODES = ("none", "paper_tab2", "paper_tab2_relative")


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


@dataclass(frozen=True)
class Tab2RelativeErrors:
    """``paper_tab2_relative``: ``err = sign * round(|r| * D)``, ``|r|`` from :data:`REL_MIXTURE`.
    Same draw key and sign bit as :class:`Tab2Errors`; one more hash byte range picks the component."""

    seed: int
    w: float = REL_MIXTURE["w"]
    mu1: float = REL_MIXTURE["mu1"]
    mu2: float = REL_MIXTURE["mu2"]
    sigma: float = REL_MIXTURE["sigma"]

    def rel_error(self, model: str, key: str) -> float:
        """Signed relative error ``r`` of request ``key`` (deterministic in ``(seed, model, key)``)."""
        h = hashlib.sha256(f"{self.seed}|{model}|{key}".encode("utf-8")).digest()
        u = int.from_bytes(h[:8], "big") / 2.0 ** 64
        sign = 1 if h[8] & 1 else -1
        v = int.from_bytes(h[9:17], "big") / 2.0 ** 64
        z = NormalDist().inv_cdf(min(max(u, 1e-12), 1.0 - 1e-12))
        mu = self.mu2 if v < self.w else self.mu1
        return sign * math.exp(mu + self.sigma * z)

    def predict(self, model: str, key: str, max_tokens: int, prompt_tokens: int, max_model_len: int) -> int:
        hi = max(1, int(max_model_len) - max(0, int(prompt_tokens)))
        D = int(max_tokens)
        return max(1, min(hi, D + int(round(self.rel_error(model, key) * D))))


def make(mode: str, seed: int):
    if mode == "none":
        return None
    if mode == "paper_tab2":
        return Tab2Errors(int(seed))
    if mode == "paper_tab2_relative":
        return Tab2RelativeErrors(int(seed))
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

