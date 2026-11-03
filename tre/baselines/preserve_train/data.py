"""Azure LLM inference 2024 week traces -> fixed real-time windows -> training sequences.

Columns used: ``TIMESTAMP`` (UTC), ``ContextTokens``, ``GeneratedTokens``. Window i of a stream
covers ``[origin + i*W, origin + (i+1)*W)`` where ``origin`` is 00:00 UTC of the day of the
first request; P_i = sum ContextTokens, D_i = sum GeneratedTokens. No time compression.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np

DAY_S = 86400


def sha256_file(path: str, chunk: int = 1 << 22) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def window_csv(csv_path: str, window_s: int, chunksize: int = 2_000_000) -> Dict[str, object]:
    """Stream the CSV and return per-window sums (P, D, request count) plus metadata."""
    import pandas as pd

    W_ns = int(window_s) * 1_000_000_000
    origin_ns = None
    first_ns = last_ns = None
    P = np.zeros(0, dtype=np.int64)
    D = np.zeros(0, dtype=np.int64)
    N = np.zeros(0, dtype=np.int64)
    rows = 0
    for chunk in pd.read_csv(csv_path, usecols=["TIMESTAMP", "ContextTokens", "GeneratedTokens"],
                             chunksize=chunksize):
        ts = pd.to_datetime(chunk["TIMESTAMP"], format="ISO8601", utc=True)
        ns = ts.dt.tz_convert(None).to_numpy().astype("datetime64[ns]").astype(np.int64)
        if origin_ns is None:
            first_ns = int(ns.min())
            origin_ns = first_ns - first_ns % (DAY_S * 1_000_000_000)
        lo = int(ns.min())
        if lo < origin_ns:
            raise ValueError("timestamps before the first day's midnight (CSV not sorted by day?)")
        first_ns = min(first_ns, lo)
        last_ns = int(ns.max()) if last_ns is None else max(last_ns, int(ns.max()))
        idx = (ns - origin_ns) // W_ns
        n = int(idx.max()) + 1
        if n > len(P):
            P = np.pad(P, (0, n - len(P)))
            D = np.pad(D, (0, n - len(D)))
            N = np.pad(N, (0, n - len(N)))
        P += np.bincount(idx, weights=chunk["ContextTokens"].to_numpy(np.float64), minlength=len(P)).astype(np.int64)
        D += np.bincount(idx, weights=chunk["GeneratedTokens"].to_numpy(np.float64), minlength=len(D)).astype(np.int64)
        N += np.bincount(idx, minlength=len(N)).astype(np.int64)
        rows += len(chunk)
    if origin_ns is None:
        raise ValueError(f"{csv_path}: no rows")
    # Keep whole days only: the trace covers [origin, origin + k days); drop nothing inside.
    days = math.ceil((last_ns - origin_ns) / (DAY_S * 1e9))
    n_windows = days * DAY_S // int(window_s)
    if len(P) > n_windows:
        raise ValueError("internal: more windows than whole days")
    P = np.pad(P, (0, n_windows - len(P)))
    D = np.pad(D, (0, n_windows - len(D)))
    N = np.pad(N, (0, n_windows - len(N)))
    return {"P": P, "D": D, "N": N, "origin_ns": origin_ns, "first_ns": first_ns, "last_ns": last_ns,
            "rows": rows, "window_s": int(window_s)}


def iso_to_ns(s: str) -> int:
    """UTC ISO time (naive = UTC) -> ns since the epoch (standard library: the training host
    needs pandas only to window a CSV that has no cache yet)."""
    import datetime as _dt

    t = _dt.datetime.fromisoformat(s)
    t = t.replace(tzinfo=_dt.timezone.utc) if t.tzinfo is None else t.astimezone(_dt.timezone.utc)
    delta = t - _dt.datetime(1970, 1, 1, tzinfo=_dt.timezone.utc)
    return (delta.days * 86400 + delta.seconds) * 10**9 + delta.microseconds * 1000


def ns_to_iso(ns: int) -> str:
    import datetime as _dt

    s, rem = divmod(int(ns), 10**9)
    t = _dt.datetime(1970, 1, 1, tzinfo=_dt.timezone.utc) + _dt.timedelta(seconds=s, microseconds=rem // 1000)
    return t.isoformat()


def exclusion_windows(origin_ns: int, window_s: int, start_iso: str, end_iso: str, n_windows: int) -> Tuple[int, int]:
    """Window index range [a, b) overlapping the UTC interval [start, end)."""
    W_ns = int(window_s) * 1_000_000_000
    s, e = iso_to_ns(start_iso) - origin_ns, iso_to_ns(end_iso) - origin_ns
    if not e > s:
        raise ValueError(f"empty exclusion {start_iso}..{end_iso}")
    a = max(0, s // W_ns)
    b = min(n_windows, -(-e // W_ns))
    return int(a), int(b)


@dataclass
class Stream:
    """One dataset stream after windowing, exclusion and the chronological split."""
    name: str
    P: np.ndarray            # (n,) prompt-token sums per window
    D: np.ndarray            # (n,) generated-token sums per window
    valid: np.ndarray        # (n,) bool, False inside exclusions
    split: int               # windows [split, n) are held out

    @property
    def n(self) -> int:
        return len(self.P)

    def targets(self, seq_len: int, held_out: bool) -> np.ndarray:
        """Target indices t whose inputs t-seq_len..t-1 and t are all valid (never span a gap).

        Training targets lie before ``split``; held-out targets at or after it (their inputs may
        be the last training windows: past observations, as in upstream's evaluation)."""
        v = self.valid.astype(np.int64)
        c = np.concatenate([[0], np.cumsum(v)])
        t = np.arange(seq_len, self.n)
        ok = (c[t + 1] - c[t - seq_len]) == seq_len + 1
        t = t[ok]
        return t[t >= self.split] if held_out else t[t < self.split]


def build_xy(series: np.ndarray, targets: np.ndarray, seq_len: int) -> Tuple[np.ndarray, np.ndarray]:
    """Raw (unnormalized) inputs X (N, seq_len) and targets y (N,)."""
    idx = targets[:, None] + np.arange(-seq_len, 0)[None, :]
    return series[idx].astype(np.float64), series[targets].astype(np.float64)


def per_sequence_normalize(X: np.ndarray, y: np.ndarray, eps: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    scale = np.maximum(X.mean(axis=1), eps)
    return X / scale[:, None], y / scale, scale


def mape(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    """Mean absolute percentage error over windows with y_true > 0 (as upstream train_predictor.py)."""
    yt = np.asarray(y_true, dtype=np.float64)
    yp = np.asarray(y_pred, dtype=np.float64)
    m = yt > 0
    if not m.any():
        return float("nan")
    return float(np.mean(np.abs((yt[m] - yp[m]) / yt[m])) * 100.0)


def segments(valid: np.ndarray) -> List[Tuple[int, int]]:
    """Contiguous valid runs [a, b)."""
    out, a = [], None
    for i, v in enumerate(valid):
        if v and a is None:
            a = i
        elif not v and a is not None:
            out.append((a, i))
            a = None
    if a is not None:
        out.append((a, len(valid)))
    return out
