"""PreServe Tier-1 workload predictor: numpy-only forward pass of the exported mLSTM.

Port of the TRE v1 file (``tre_code`` dev ``ceb88b4``, ``tre_preserve/predictor.py``, decision
2026-10-26: v2 reuses v1's ``paper_past`` weights, ``tre_v1/preserve-mlstm-20261024-paper``; v2's
Real_* traces are v1's slice a, so the train / test split in time is the same). The image has
no torch; the weights trained by v1 ``tre_preserve.train`` are an ``.npz`` file evaluated here
with numpy (v1: numpy vs torch max relative difference 1.4e-07). :func:`load_predictors`
checks the file's sha256 against the ConfigMap's (fail closed).

Model (faithful copy of upstream PreServe ``mLSTM_predict.py``): one mLSTMCell (hidden 64)
over a sequence of ``seq_len`` (25) scalar window values, then Linear(64 -> 1) on the last
hidden state. Upstream computes ``m = tanh(W_m x * U_m h)`` but never feeds it into the gates,
so the cell is effectively a plain LSTM; the exported file keeps W_m / U_m for completeness
and this forward pass (like upstream's) does not use them.

Normalization ``per_sequence_mean`` (our deliberate deviation from upstream's global min-max,
decision D15): inputs are divided by ``s = max(mean(history[-seq_len:]), eps)``, the model
predicts the next window as a ratio, and the prediction is ``max(0, ratio) * s``.

File layout (written by ``preserve_train.export.export_npz``, the copy of the v1
``tre_preserve.train`` exporter): scalar keys ``seq_len``, ``hidden``, ``eps``, ``normalization``,
``format_version``, ``kinds``; per model kind
(``prompt`` / ``response``) the tensors ``<kind>.W_ih.weight`` (4H x 1), ``<kind>.W_ih.bias``,
``<kind>.W_hh.weight`` (4H x H), ``<kind>.W_hh.bias``, ``<kind>.fc.weight`` (1 x H),
``<kind>.fc.bias`` (+ unused ``<kind>.W_m.*`` / ``<kind>.U_m.*``).

Window (2026-11-03): a model predicts the next window of the length it was trained on, so a file
declares it: scalar key ``window_s`` (written by ``preserve_train/export.py``). The
three files of ``weights/preserve-mlstm-20261024-paper`` predate the key; their window (30 s, their
REPORT.md) is known by sha256 (:data:`LEGACY_WINDOW_S`). Any other file without the key is refused,
and the policy refuses weights of another window than ``params.window_s``.
"""

from __future__ import annotations

import hashlib
from typing import Dict, Optional, Sequence

import numpy as np

FORMAT_VERSION = 1
KINDS = ("prompt", "response")
PER_SEQUENCE_MEAN = "per_sequence_mean"
#: sha256 -> training window (s) of exported files without a ``window_s`` key (v1 paper_past, 10-24).
LEGACY_WINDOW_S = {
    "3e70e4642829427ac5a528bfd6ff77e8e29335960adbab76832a6d5168251fb4": 30.0,  # code-paper.npz
    "64448df062646bbb0cbe0ce8936330509172046c4bd2ee568b595b4ad4dc4478": 30.0,  # conv-paper.npz
    "0dfd5649600b2c86208528e3f0de5b19ac3266fc27af77c2189d6851134c1585": 30.0,  # pooled-paper.npz
}


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 0.5 * (1.0 + np.tanh(0.5 * x))  # numerically safe logistic


class Predictor:
    """One trained mLSTM (prompt tokens or response tokens) for one model set."""

    def __init__(self, weights: Dict[str, np.ndarray], seq_len: int, hidden: int, eps: float,
                 normalization: str, kind: str, window_s: Optional[float] = None):
        if normalization != PER_SEQUENCE_MEAN:
            raise ValueError(f"unsupported normalization {normalization!r} (only {PER_SEQUENCE_MEAN!r} ships)")
        self.seq_len = int(seq_len)
        self.hidden = int(hidden)
        self.eps = float(eps)
        self.normalization = normalization
        self.kind = kind
        #: training window (s); None = the file does not say (see :func:`load_predictors`)
        self.window_s = None if window_s is None else float(window_s)
        h = self.hidden
        self._w_ih = np.asarray(weights["W_ih.weight"], dtype=np.float64).reshape(4 * h, -1)[:, 0]
        self._b = (np.asarray(weights["W_ih.bias"], dtype=np.float64)
                   + np.asarray(weights["W_hh.bias"], dtype=np.float64))
        self._w_hh = np.asarray(weights["W_hh.weight"], dtype=np.float64).reshape(4 * h, h)
        self._fc_w = np.asarray(weights["fc.weight"], dtype=np.float64).reshape(h)
        self._fc_b = float(np.asarray(weights["fc.bias"], dtype=np.float64).reshape(-1)[0])

    def forward(self, x: Sequence[float]) -> float:
        """Raw model output for an already-normalized input sequence (any length >= 1)."""
        h = np.zeros(self.hidden)
        c = np.zeros(self.hidden)
        H = self.hidden
        for v in np.asarray(x, dtype=np.float64).reshape(-1):
            gates = self._w_ih * v + self._w_hh @ h + self._b
            i = _sigmoid(gates[0:H])
            f = _sigmoid(gates[H:2 * H])
            g = np.tanh(gates[2 * H:3 * H])
            o = _sigmoid(gates[3 * H:4 * H])
            c = f * c + i * g
            h = o * np.tanh(c)
        return float(self._fc_w @ h + self._fc_b)

    def predict(self, history: Sequence[float]) -> float:
        """Predict the next window's token sum from the last ``seq_len`` window sums."""
        hist = np.asarray(history, dtype=np.float64).reshape(-1)
        if hist.size < self.seq_len:
            raise ValueError(f"history has {hist.size} windows, need at least seq_len={self.seq_len}")
        x = hist[-self.seq_len:]
        if not np.all(np.isfinite(x)):
            raise ValueError("history contains non-finite values")
        scale = max(float(x.mean()), self.eps)
        return max(0.0, self.forward(x / scale)) * scale


def load_predictor(path: str, kind: str) -> Predictor:
    """Load the ``kind`` ("prompt" or "response") model of one exported set (``<set>.npz``)."""
    if kind not in KINDS:
        raise ValueError(f"kind must be one of {KINDS}, got {kind!r}")
    with np.load(path, allow_pickle=False) as z:
        version = int(z["format_version"])
        if version != FORMAT_VERSION:
            raise ValueError(f"{path}: format_version {version}, expected {FORMAT_VERSION}")
        prefix = kind + "."
        weights = {k[len(prefix):]: z[k] for k in z.files if k.startswith(prefix)}
        if not weights:
            raise ValueError(f"{path}: no {kind!r} model in file")
        window_s = float(z["window_s"]) if "window_s" in z.files else None
        return Predictor(weights, seq_len=int(z["seq_len"]), hidden=int(z["hidden"]), eps=float(z["eps"]),
                         normalization=str(z["normalization"]), kind=kind, window_s=window_s)


def sha256_of(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_predictors(path: str, sha256: Optional[str] = None) -> Dict[str, Predictor]:
    """Both models of one exported set: {"prompt": Predictor, "response": Predictor}. With
    ``sha256`` the file must have exactly that digest. Each predictor's ``window_s`` is the file's
    key, else :data:`LEGACY_WINDOW_S` of its digest; neither -> ValueError (fail closed)."""
    got = sha256_of(path)
    if sha256 is not None and got != str(sha256).strip().lower():
        raise ValueError(f"{path}: sha256 {got} != expected {sha256}")
    out = {k: load_predictor(path, k) for k in KINDS}
    for pr in out.values():
        if pr.window_s is None:
            pr.window_s = LEGACY_WINDOW_S.get(got)
        if pr.window_s is None:
            raise ValueError(f"{path}: the weights do not declare their training window (no window_s key, "
                             "not a known legacy file)")
    return out
