"""Azure LLM inference traces (AzurePublicDataset): reading, length fits, arrival dispersion.

Input: the public CSVs with columns ``TIMESTAMP,ContextTokens,GeneratedTokens``
(2023: ``AzureLLMInferenceTrace_{conv,code}.csv``; 2024: ``..._{conv,code}_1week.csv``).
The files are never committed; their sha256 goes into the fit record.

Length fit: a lognormal by maximum likelihood on ``ln(x)`` over the rows with ``x >= 1``
(``mu`` = mean, ``sigma`` = population std of ``ln x``). The generator uses ``sigma`` as the
*shape* and re-solves ``mu`` per target mean (see :mod:`.lengths`).

Arrival dispersion: 1 s request counts, detrended by a centred moving mean (31 s), Fano
factor ``var(n - trend) / mean(n)`` per 720 s slice. A Poisson process has Fano 1. Thinning
a process with Fano ``F`` by keep-probability ``p`` gives ``1 + p (F - 1)``; superposing
independent copies keeps ``F``. The audit uses this to put the Azure reference at the
trace's own rate (:func:`reference_cv`).
"""
from __future__ import annotations

import csv
import hashlib
import math
from datetime import date
from pathlib import Path
from typing import Iterator

TREND_WINDOW_S = 31
SLICE_S = 720
_EPOCH = date(1970, 1, 1).toordinal()


def parse_timestamp(value: str, cache: dict) -> float:
    """``YYYY-MM-DD HH:MM:SS.ffffff[f][+00:00]`` -> unix seconds (UTC; the 2023 files carry no zone)."""
    day = value[:10]
    base = cache.get(day)
    if base is None:
        base = (date.fromisoformat(day).toordinal() - _EPOCH) * 86400
        cache[day] = base
    sec = value[17:]
    for marker in ("+", "-"):
        at = sec.find(marker, 2)
        if at >= 0:
            sec = sec[:at]
            break
    return base + int(value[11:13]) * 3600 + int(value[14:16]) * 60 + float(sec)


def iter_rows(path: str | Path) -> Iterator[tuple[float, int, int]]:
    """(unix ts, context tokens, generated tokens) per row, in file order."""
    cache: dict = {}
    with open(path, newline="", encoding="utf-8") as fh:
        reader = csv.reader(fh)
        header = next(reader)
        idx = {name: header.index(name) for name in ("TIMESTAMP", "ContextTokens", "GeneratedTokens")}
        for row in reader:
            if not row:
                continue
            yield (parse_timestamp(row[idx["TIMESTAMP"]].strip(), cache),
                   int(row[idx["ContextTokens"]]), int(row[idx["GeneratedTokens"]]))


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


class _LenStats:
    """Streaming lognormal fit of integer lengths (exact quantiles via a value histogram)."""

    def __init__(self):
        self.hist: dict = {}
        self.n = self.zero = 0
        self.s1 = self.s2 = self.raw = 0.0

    def add(self, v: int) -> None:
        self.hist[v] = self.hist.get(v, 0) + 1
        if v < 1:
            self.zero += 1
            return
        x = math.log(v)
        self.n += 1; self.s1 += x; self.s2 += x * x; self.raw += v

    def quantile(self, q: float) -> int:
        target = q * (self.n + self.zero)
        acc = 0
        for v in sorted(self.hist):
            acc += self.hist[v]
            if acc > target:
                return v
        return max(self.hist)

    def record(self) -> dict:
        mu = self.s1 / self.n
        sigma = math.sqrt(max(0.0, self.s2 / self.n - mu * mu))
        return {"mu": mu, "sigma": sigma, "n": self.n, "zero_frac": self.zero / (self.n + self.zero),
                "mean": self.raw / self.n, "max": max(self.hist),
                **{f"p{k}": self.quantile(q) for k, q in (("50", 0.5), ("90", 0.9), ("99", 0.99), ("99_5", 0.995))}}


def dispersion(counts: list, slice_s: int = SLICE_S, window: int = TREND_WINDOW_S) -> dict:
    """Fano factor and CV of detrended 1 s counts, per ``slice_s`` slice (median/p10/p90)."""
    half = window // 2
    n = len(counts)
    prefix = [0]
    for c in counts:
        prefix.append(prefix[-1] + c)
    trend = []
    for i in range(n):
        lo, hi = max(0, i - half), min(n, i + half + 1)
        trend.append((prefix[hi] - prefix[lo]) / (hi - lo))
    fanos, cvs, rates = [], [], []
    for start in range(0, n - slice_s + 1, slice_s):
        seg = range(start, start + slice_s)
        mean = sum(counts[i] for i in seg) / slice_s
        if mean <= 0:
            continue
        resid = [counts[i] - trend[i] for i in seg]
        mr = sum(resid) / slice_s
        var = sum((r - mr) ** 2 for r in resid) / slice_s
        fanos.append(var / mean)
        cvs.append(math.sqrt(var) / mean)
        rates.append(mean)
    order = sorted(range(len(fanos)), key=lambda k: fanos[k])

    def pick(q):
        return order[min(len(order) - 1, int(q * len(order)))]
    return {"slices": len(fanos), "slice_s": slice_s, "trend_window_s": window,
            "fano_median": fanos[pick(0.5)], "fano_p10": fanos[pick(0.1)], "fano_p90": fanos[pick(0.9)],
            "rate_at_median": rates[pick(0.5)], "cv_resid_median": sorted(cvs)[len(cvs) // 2],
            "mean_rate": sum(counts) / max(1, n)}


def fit_csv(path: str | Path, name: str) -> dict:
    """The fit record of one dataset (lengths + dispersion + provenance); streams the file."""
    lin, lout = _LenStats(), _LenStats()
    counts: list = []
    t0 = None
    rows = 0
    sxy = sx = sy = sxx = syy = 0.0
    npair = 0
    for t, i, o in iter_rows(path):
        if t0 is None:
            t0 = t
        k = int(t - t0)
        if k >= 0:
            if k >= len(counts):
                counts.extend([0] * (k + 1 - len(counts)))
            counts[k] += 1
        rows += 1
        lin.add(i); lout.add(o)
        if i >= 1 and o >= 1:
            a, b = math.log(i), math.log(o)
            npair += 1; sx += a; sy += b; sxx += a * a; syy += b * b; sxy += a * b
    cov = sxy / npair - (sx / npair) * (sy / npair)
    vx = sxx / npair - (sx / npair) ** 2
    vy = syy / npair - (sy / npair) ** 2
    return {
        "name": name,
        "source": {"file": Path(path).name, "sha256": sha256_file(path), "rows": rows,
                   "start_unix": t0, "duration_s": len(counts)},
        "in": lin.record(),
        "out": lout.record(),
        "log_corr_in_out": cov / math.sqrt(vx * vy) if vx > 0 and vy > 0 else 0.0,
        "arrivals": dispersion(counts),
    }


def load_window(path: str | Path, start_s: float, end_s: float) -> list[tuple[float, int, int]]:
    """Rows with ``start_s <= t - t_first < end_s`` as (t - start_s, in, out); stops reading past ``end_s``."""
    out = []
    t0 = None
    for t, i, o in iter_rows(path):
        if t0 is None:
            t0 = t
        rel = t - t0
        if rel < start_s:
            continue
        if rel >= end_s:
            # files are time-ordered up to sub-second jitter; allow 5 s of disorder
            if rel >= end_s + 5:
                break
            continue
        out.append((rel - start_s, i, o))
    out.sort()
    return out


def reference_cv(fit: dict, rate: float) -> float:
    """Expected detrended 1 s count CV of the dataset's arrival process at mean ``rate`` req/s.

    Below the dataset's native rate the process is a thinning (Fano 1 + p (F - 1)); above,
    a superposition of independent copies (Fano F)."""
    arr = fit["arrivals"]
    f = arr["fano_median"]
    p = min(1.0, rate / arr["rate_at_median"])
    return math.sqrt((1.0 + p * (f - 1.0)) / rate)
