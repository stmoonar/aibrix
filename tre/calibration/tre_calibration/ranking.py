"""Ranking metrics of the normalised pressure signal: AUROC and Kendall tau-b, with
cell-bootstrap confidence intervals. Disclosure only - nothing here gates a decision.

Definitions (docs/design/20260930-ranking-metrics.md; the same text is in
:data:`DEFINITIONS`, which the JSON outputs carry):

* **pressure** = -Z, Z = ``tre_calibration.fit.signal_z`` (higher-is-healthier for both
  signal orientations), so a larger pressure is a more stressed window.
* **AUROC**: positives = violated windows (not ``slo_met``), score = pressure; the
  Mann-Whitney U with average ranks for ties (a tie between a positive and a negative
  counts one half). None when a class is empty.
* **Kendall tau-b (i), window level**: tau_b(pressure, severity) over all windows (per
  model, or pooled over every window of every model). severity = the label's ratio_max
  (max over the label's metrics of p95 / SLO; unserved windows >= 2.0). Windows without a
  finite severity or Z are dropped and counted. Positive = Z orders pressure correctly.
* **Kendall tau-b (ii), cross-model**: only pairs of windows of DIFFERENT models at the
  same instant; stratified by instant: sum over instants of (C - D) divided by
  sqrt(sum of pairs not tied in pressure * sum of pairs not tied in severity). None, with a
  reason, when no instant holds windows of two models.
* **CIs**: cell bootstrap (cells drawn with replacement; pooled metrics draw each model's
  cells from that model's own cells, i.e. stratified by model), ``n_resamples`` draws from
  ``random.Random(seed)``, 95 % percentile interval with the index rule of
  ``scripts.dline_refit._ci95``; a resample on which a metric is undefined is skipped and
  ``resamples_used`` counts the rest. With one stratum the draw sequence is exactly the one
  of ``dline_refit.acceptance_bootstrap`` (sorted cells, ``rng.choice`` per cell), so the
  per-model intervals come from the same resamples as the BA interval.

The bootstrap is exact and cheap: a resample with cell multiplicities w (a cell drawn k
times is k copies of its windows) has, over its expanded window list,

    sum over ordered pairs sgn(dx) sgn(dy)       = w' K w
    ordered pairs not tied in x (resp. y)        = w' A w   (resp. w' B w)
    2 x Mann-Whitney U (positive beats negative) = w' U w,  n_pos = w.P, n_neg = w.N

with the cell x cell matrices K, A, B, U precomputed once (a window paired with its own
copy is tied in both coordinates and adds nothing), so tau_b = w'Kw / sqrt(w'Aw w'Bw) and
AUROC = w'Uw / (2 n_pos n_neg). Every entry is an integer, so the numpy path (used when
numpy imports; ``use_numpy`` forces either) and the pure-Python path give bit-identical
numbers. The pure-Python matrices cost O(windows^2) and each resample O(cells^2): fine for
tests and small sets, slow for a whole campaign - numpy is the practical path there.
"""
from __future__ import annotations

import math
import random
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Callable, Hashable, Iterable, Iterator, Mapping, Optional, Sequence

#: The definitions every disclosure block carries (``definition`` fields of the JSON).
DEFINITIONS: dict[str, str] = {
    "pressure": "pressure = -Z; Z = tre_calibration.fit.signal_z(signal, theta, direction), higher is healthier",
    "severity": ("severity = label ratio_max = max over the label's metrics of p95 / SLO (unserved windows "
                 ">= UNSERVED_MIN_RATIO = 2.0); windows without a finite severity are dropped and counted"),
    "auroc": ("AUROC of pressure for violated windows (positives = not slo_met): Mann-Whitney U / "
              "(n_pos n_neg), average ranks for ties (a positive-negative tie counts 1/2); pooled = over "
              "the windows of every model together (Z makes models comparable)"),
    "kendall_tau_b_window": ("Kendall tau-b (i) of (pressure, severity) over windows: (C - D) / sqrt((n0 - n1) "
                             "(n0 - n2)), n1 / n2 the pairs tied in pressure / severity; per model and pooled "
                             "over all windows of all models; positive = Z ranks pressure correctly"),
    "kendall_tau_b_cross_model": ("Kendall tau-b (ii): only pairs of windows of different models at the same "
                                  "instant; stratified: sum over instants of (C - D) / sqrt(sum of pairs not "
                                  "tied in pressure * sum of pairs not tied in severity)"),
    "bootstrap": ("cell bootstrap: cells (scenario ids) drawn with replacement, stratified by model for pooled "
                  "metrics; 95 % percentile interval, sorted values v -> [v[int(.025 n)], v[int(.975 n) - 1]]; "
                  "resamples on which a metric is undefined are skipped (resamples_used)"),
    "gating": "disclosure only: no acceptance criterion reads these numbers",
}

#: Why tau-b (ii) is None when no instant holds two models' windows.
CROSS_MODEL_UNAVAILABLE = ("needs multi-model concurrent data (E1 or concurrent campaigns): no instant "
                           "holds windows of two models")


# ------------------------------------------------------------------ point metrics


def average_ranks(values: Sequence[float]) -> list[float]:
    """1-based ranks in input order, ties sharing their average rank."""
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i, n = 0, len(order)
    while i < n:
        j = i
        while j + 1 < n and values[order[j + 1]] == values[order[i]]:
            j += 1
        r = (i + 1 + j + 1) / 2.0
        for k in range(i, j + 1):
            ranks[order[k]] = r
        i = j + 1
    return ranks


def auroc(scores: Sequence[float], positives: Sequence[bool]) -> Optional[float]:
    """AUROC of ``scores`` for ``positives`` (higher score = more likely positive): the
    Mann-Whitney U with average ranks (a tie between a positive and a negative counts 1/2),
    over n_pos * n_neg. None when either class is empty. The rank sum is a sum of
    half-integers, so the value is exactly the pairwise-count AUROC."""
    if len(scores) != len(positives):
        raise ValueError("scores and positives must have the same length")
    n1 = sum(1 for p in positives if p)
    n0 = len(positives) - n1
    if not n1 or not n0:
        return None
    r = average_ranks(scores)
    u = sum(ri for ri, p in zip(r, positives) if p) - n1 * (n1 + 1) / 2.0
    return u / (n1 * n0)


def _tie_pairs(sorted_values: Sequence[Any]) -> int:
    out, i, n = 0, 0, len(sorted_values)
    while i < n:
        j = i
        while j + 1 < n and sorted_values[j + 1] == sorted_values[i]:
            j += 1
        t = j - i + 1
        out += t * (t - 1) // 2
        i = j + 1
    return out


def _count_inversions(values: list) -> int:
    """Pairs i < j with values[i] > values[j] (strict), bottom-up merge sort, O(n log n)."""
    n = len(values)
    src, inv, width = list(values), 0, 1
    while width < n:
        dst = []
        for lo in range(0, n, 2 * width):
            mid, hi = min(lo + width, n), min(lo + 2 * width, n)
            i, j = lo, mid
            while i < mid and j < hi:
                if src[i] <= src[j]:
                    dst.append(src[i])
                    i += 1
                else:
                    dst.append(src[j])
                    inv += mid - i
                    j += 1
            dst.extend(src[i:mid])
            dst.extend(src[j:hi])
        src, width = dst, width * 2
    return inv


def tau_b_counts(x: Sequence[float], y: Sequence[float]) -> tuple[int, int, int]:
    """``(C - D, n0 - n1, n0 - n2)`` over the unordered pairs of ``(x, y)`` (Knight 1966,
    O(n log n)): C / D concordant / discordant pairs, n0 = n(n-1)/2, n1 / n2 pairs tied in
    x / y."""
    if len(x) != len(y):
        raise ValueError("x and y must have the same length")
    n = len(x)
    order = sorted(range(n), key=lambda i: (x[i], y[i]))
    xs = [x[i] for i in order]
    ys = [y[i] for i in order]
    n0 = n * (n - 1) // 2
    n1 = _tie_pairs(xs)
    n3 = _tie_pairs(list(zip(xs, ys)))
    discordant = _count_inversions(ys)  # x-ties are sorted by y: no inversion inside them
    n2 = _tie_pairs(sorted(ys))
    return n0 - n1 - n2 + n3 - 2 * discordant, n0 - n1, n0 - n2


def kendall_tau_b(x: Sequence[float], y: Sequence[float]) -> Optional[float]:
    """Kendall's tau-b with the tie correction; None when fewer than two pairs are untied
    in x or in y (the denominator is zero)."""
    s, a, b = tau_b_counts(x, y)
    return s / math.sqrt(a * b) if a > 0 and b > 0 else None


def _sgn(v: float) -> int:
    return (v > 0) - (v < 0)


# -------------------------------------------------------------------- records


@dataclass(frozen=True)
class RankRecord:
    """One window as the ranking metrics see it. ``cell`` is the bootstrap unit within
    ``model`` (a scenario id); ``instant`` the window's end (ms) for the cross-model pairs,
    None when unknown; ``severity`` None when the window has none."""

    model: str
    cell: str
    instant: Optional[float]
    z: float
    severity: Optional[float]
    violated: bool

    @property
    def pressure(self) -> float:
        return -self.z


def _finite(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def records_from_windows(model: str, windows: Sequence[Any], *, theta: float, direction: str,
                         instants: Optional[Sequence[Optional[float]]] = None) -> list[RankRecord]:
    """``CalibrationWindow`` s of one model -> records (Z at ``theta``; severity =
    ``latency_ratio_p95``; ``instants`` parallel to ``windows``, default None)."""
    from tre_calibration.fit import signal_z

    if instants is not None and len(instants) != len(windows):
        raise ValueError("instants must be parallel to windows")
    out = []
    for i, w in enumerate(windows):
        sev = w.latency_ratio_p95
        out.append(RankRecord(
            model=model, cell=str(w.scenario_id),
            instant=instants[i] if instants is not None else None,
            z=signal_z(w.signal, theta, direction) if _finite(w.signal) else math.nan,
            severity=float(sev) if _finite(sev) else None, violated=not w.slo_met))
    return out


def bin_instant(instant: Optional[float], bin_ms: Optional[float]) -> Optional[float]:
    """``instant`` (exact) or its nearest multiple of ``bin_ms``."""
    if instant is None or bin_ms is None:
        return instant
    return math.floor(instant / bin_ms + 0.5) * bin_ms


def cross_model_pairs(records: Sequence[RankRecord], *, bin_ms: Optional[float] = None
                      ) -> tuple[list[tuple[int, int]], int]:
    """Index pairs (i < j) of usable records of different models at the same (binned)
    instant, and how many instants hold at least one such pair."""
    groups: dict[float, list[int]] = defaultdict(list)
    for i, r in enumerate(records):
        t = bin_instant(r.instant, bin_ms)
        if t is not None and _finite(r.z) and r.severity is not None:
            groups[t].append(i)
    pairs, instants = [], 0
    for t in sorted(groups):
        idx = groups[t]
        here = [(a, b) for k, a in enumerate(idx) for b in idx[k + 1:] if records[a].model != records[b].model]
        if here:
            instants += 1
            pairs += here
    return pairs, instants


def stratified_cross_model_tau_b(records: Sequence[RankRecord], *, bin_ms: Optional[float] = None) -> dict:
    """Kendall tau-b (ii): cross-model pairs at the same instant, stratified by instant."""
    pairs, instants = cross_model_pairs(records, bin_ms=bin_ms)
    s = a = b = 0
    for i, j in pairs:
        dx = _sgn(records[i].pressure - records[j].pressure)
        dy = _sgn(records[i].severity - records[j].severity)
        s += dx * dy
        a += dx != 0
        b += dy != 0
    out: dict[str, Any] = {"instants": instants, "pairs": len(pairs),
                           "instant_rule": ("window_end_ms, exact" if bin_ms is None
                                            else f"window_end_ms rounded to the nearest {bin_ms:g} ms")}
    if not pairs:
        return out | {"value": None, "reason": CROSS_MODEL_UNAVAILABLE}
    if not (a and b):
        return out | {"value": None, "reason": "every cross-model pair is tied in pressure or in severity"}
    return out | {"value": s / math.sqrt(a * b)}


# ------------------------------------------------------------------ bootstrap


def ci95(values: Sequence[float]) -> list[Optional[float]]:
    """95 % percentile interval, the index rule of ``scripts.dline_refit._ci95``."""
    v = sorted(values)
    if not v:
        return [None, None]
    return [v[int(0.025 * len(v))], v[int(0.975 * len(v)) - 1]]


def stratified_draws(strata: Mapping[str, Sequence[Hashable]], n_resamples: int, seed: int
                     ) -> Iterator[dict[Hashable, int]]:
    """Cell multiplicities of ``n_resamples`` resamples: per resample, for every stratum in
    sorted order, ``len(cells)`` draws ``rng.choice(cells)`` (cells sorted), one
    ``random.Random(seed)`` for the whole sequence."""
    rng = random.Random(seed)
    keys = sorted(strata)
    cells = {k: sorted(strata[k]) for k in keys}
    for _ in range(n_resamples):
        w: dict[Hashable, int] = defaultdict(int)
        for k in keys:
            cs = cells[k]
            for _ in cs:
                w[rng.choice(cs)] += 1
        yield w


def _have_numpy() -> bool:
    try:
        import numpy  # noqa: F401
    except ImportError:
        return False
    return True


class PairForms:
    """Integer cell x cell matrices of one metric; ``value(w)`` evaluates it on a resample
    with multiplicity vector ``w`` (a list over the cell index)."""

    def __init__(self, n_cells: int, kind: str, mats: dict[str, Any], vecs: dict[str, Any], numpy_: bool):
        self.n, self.kind, self.mats, self.vecs, self.numpy = n_cells, kind, mats, vecs, numpy_

    def _quad(self, m: Any, w: Any) -> int:
        if self.numpy:
            return int(w @ m @ w)
        tot = 0
        for c, wc in enumerate(w):
            if wc:
                row = m[c]
                tot += wc * sum(row[d] * wd for d, wd in enumerate(w) if wd)
        return tot

    def _dot(self, v: Any, w: Any) -> int:
        if self.numpy:
            return int(v @ w)
        return sum(a * b for a, b in zip(v, w))

    def value(self, w: Sequence[int]) -> Optional[float]:
        if self.numpy:
            import numpy as np

            w = np.asarray(w, dtype=np.int64)
        if self.kind == "auroc":
            npos, nneg = self._dot(self.vecs["P"], w), self._dot(self.vecs["N"], w)
            if not npos or not nneg:
                return None
            return self._quad(self.mats["U"], w) / (2.0 * npos * nneg)
        a, b = self._quad(self.mats["A"], w), self._quad(self.mats["B"], w)
        if not (a and b):
            return None
        return self._quad(self.mats["K"], w) / math.sqrt(a * b)


def _empty(n: int) -> list[list[int]]:
    return [[0] * n for _ in range(n)]


def auroc_forms(cell_of: Sequence[int], scores: Sequence[float], positives: Sequence[bool], n_cells: int, *,
                use_numpy: bool) -> PairForms:
    """U[c][d] = 2 * #(positive of c beats negative of d) + #(ties); P / N = positives /
    negatives per cell."""
    if use_numpy:
        import numpy as np

        ci = np.asarray(cell_of, dtype=np.int64)
        s = np.asarray(scores, dtype=float)
        pos = np.asarray(positives, dtype=bool)
        U = np.zeros((n_cells, n_cells), dtype=np.int64)
        neg_idx = np.nonzero(~pos)[0]
        s_neg, c_neg = s[neg_idx], ci[neg_idx]
        for c in range(n_cells):
            rows = np.nonzero(pos & (ci == c))[0]
            if not len(rows) or not len(neg_idx):
                continue
            d = s[rows][:, None] - s_neg[None, :]
            per_neg = 2 * (d > 0).sum(axis=0) + (d == 0).sum(axis=0)
            U[c] = np.bincount(c_neg, weights=per_neg, minlength=n_cells).round().astype(np.int64)
        P = np.bincount(ci[pos], minlength=n_cells).astype(np.int64)
        N = np.bincount(ci[~pos], minlength=n_cells).astype(np.int64)
        return PairForms(n_cells, "auroc", {"U": U}, {"P": P, "N": N}, True)
    U = _empty(n_cells)
    P, N = [0] * n_cells, [0] * n_cells
    pos_i = [i for i, p in enumerate(positives) if p]
    neg_i = [i for i, p in enumerate(positives) if not p]
    for i in pos_i:
        P[cell_of[i]] += 1
    for j in neg_i:
        N[cell_of[j]] += 1
    for i in pos_i:
        row, si = U[cell_of[i]], scores[i]
        for j in neg_i:
            sj = scores[j]
            row[cell_of[j]] += 2 if si > sj else (1 if si == sj else 0)
    return PairForms(n_cells, "auroc", {"U": U}, {"P": P, "N": N}, False)


def tau_forms(cell_of: Sequence[int], x: Sequence[float], y: Sequence[float], n_cells: int, *,
              use_numpy: bool, pairs: Optional[Sequence[tuple[int, int]]] = None) -> PairForms:
    """K / A / B over ordered pairs: every pair of distinct records, or only ``pairs``
    (unordered index pairs, each entered in both orders)."""
    if pairs is not None:
        K, A, B = _empty(n_cells), _empty(n_cells), _empty(n_cells)
        for i, j in pairs:
            dx, dy = _sgn(x[i] - x[j]), _sgn(y[i] - y[j])
            c, d = cell_of[i], cell_of[j]
            for p, q in ((c, d), (d, c)):
                K[p][q] += dx * dy
                A[p][q] += dx != 0
                B[p][q] += dy != 0
        if use_numpy:
            import numpy as np

            return PairForms(n_cells, "tau", {k: np.asarray(v, dtype=np.int64) for k, v in
                                              (("K", K), ("A", A), ("B", B))}, {}, True)
        return PairForms(n_cells, "tau", {"K": K, "A": A, "B": B}, {}, False)
    if use_numpy:
        import numpy as np

        ci = np.asarray(cell_of, dtype=np.int64)
        xa, ya = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
        mats = {k: np.zeros((n_cells, n_cells), dtype=np.int64) for k in ("K", "A", "B")}
        for c in range(n_cells):
            rows = np.nonzero(ci == c)[0]
            if not len(rows):
                continue
            sx = np.sign(xa[rows][:, None] - xa[None, :]).astype(np.int8)
            sy = np.sign(ya[rows][:, None] - ya[None, :]).astype(np.int8)
            for key, m in (("K", (sx * sy).sum(axis=0, dtype=np.int64)),
                           ("A", (sx != 0).sum(axis=0, dtype=np.int64)),
                           ("B", (sy != 0).sum(axis=0, dtype=np.int64))):
                mats[key][c] = np.bincount(ci, weights=m, minlength=n_cells).round().astype(np.int64)
        return PairForms(n_cells, "tau", mats, {}, True)
    K, A, B = _empty(n_cells), _empty(n_cells), _empty(n_cells)
    n = len(x)
    for i in range(n):
        c, xi, yi = cell_of[i], x[i], y[i]
        Kc, Ac, Bc = K[c], A[c], B[c]
        for j in range(n):
            if j == i:
                continue
            dx, dy = _sgn(xi - x[j]), _sgn(yi - y[j])
            d = cell_of[j]
            Kc[d] += dx * dy
            Ac[d] += dx != 0
            Bc[d] += dy != 0
    return PairForms(n_cells, "tau", {"K": K, "A": A, "B": B}, {}, False)


# ------------------------------------------------------------- the disclosure


def _metric(value: Optional[float], boot: list[float], **counts: Any) -> dict:
    return {"value": value, "ci95": ci95(boot), "resamples_used": len(boot), **counts}


def ranking_disclosure(records: Sequence[RankRecord], *, n_resamples: int, seed: int,
                       cross_model_bins: Sequence[Optional[float]] = (),
                       use_numpy: Optional[bool] = None) -> dict:
    """The disclosure block of ``records`` (one model, or several = pooled): AUROC and
    tau-b (i) with cell-bootstrap CIs (stratified by model), plus tau-b (ii) for every
    instant rule in ``cross_model_bins`` (None = exact window end; a number = nearest
    multiple of that many ms). The cell universe is every record's (model, cell), usable
    or not, so a single-model block draws the same cells as the BA bootstrap."""
    numpy_ = _have_numpy() if use_numpy is None else bool(use_numpy)
    strata: dict[str, set] = defaultdict(set)
    for r in records:
        strata[r.model].add((r.model, r.cell))
    index = {c: k for k, c in enumerate(sorted(c for cs in strata.values() for c in cs))}
    n_cells = len(index)
    cell_of_all = [index[(r.model, r.cell)] for r in records]

    finite_z = [i for i, r in enumerate(records) if _finite(r.z)]
    ranked = [i for i in finite_z if records[i].severity is not None]
    no_sev = sum(1 for i in finite_z if records[i].severity is None)
    n_bad_z = len(records) - len(finite_z)

    sub_a = [records[i] for i in finite_z]
    a_scores = [r.pressure for r in sub_a]
    a_pos = [r.violated for r in sub_a]
    auc_value = auroc(a_scores, a_pos)
    sub_t = [records[i] for i in ranked]
    t_x = [r.pressure for r in sub_t]
    t_y = [float(r.severity) for r in sub_t]
    tau_value = kendall_tau_b(t_x, t_y) if len(sub_t) >= 2 else None

    forms: dict[str, PairForms] = {
        "auroc": auroc_forms([cell_of_all[i] for i in finite_z], a_scores, a_pos, n_cells, use_numpy=numpy_),
        "kendall_tau_b_window": tau_forms([cell_of_all[i] for i in ranked], t_x, t_y, n_cells, use_numpy=numpy_),
    }
    cross: dict[str, dict] = {}
    for bin_ms in cross_model_bins:
        key = "exact" if bin_ms is None else f"bin_{bin_ms:g}ms"
        cross[key] = stratified_cross_model_tau_b(records, bin_ms=bin_ms)
        if cross[key]["value"] is not None:
            pairs, _ = cross_model_pairs(records, bin_ms=bin_ms)
            forms[f"cross:{key}"] = tau_forms(cell_of_all, [r.pressure for r in records],
                                              [r.severity if r.severity is not None else math.nan for r in records],
                                              n_cells, use_numpy=numpy_, pairs=pairs)
    boot: dict[str, list[float]] = {k: [] for k in forms}
    for draw in stratified_draws({m: list(cs) for m, cs in strata.items()}, n_resamples, seed):
        w = [0] * n_cells
        for c, k in draw.items():
            w[index[c]] = k
        for key, f in forms.items():
            v = f.value(w)
            if v is not None:
                boot[key].append(v)

    out: dict[str, Any] = {
        "gating": False,
        "definition": DEFINITIONS,
        "models": sorted(strata),
        "windows": len(records), "cells": n_cells,
        "bootstrap": {"n_resamples": n_resamples, "seed": seed,
                      "unit": ("cell (scenario id), drawn with replacement"
                               + (", stratified by model" if len(strata) > 1 else "")),
                      "interval": "95 % percentile"},
        "auroc": _metric(auc_value, boot["auroc"], windows=len(sub_a), violated=sum(a_pos),
                         healthy=len(sub_a) - sum(a_pos), dropped_nonfinite_z=n_bad_z),
        "kendall_tau_b_window": _metric(tau_value, boot["kendall_tau_b_window"], windows=len(sub_t),
                                        dropped_no_severity=no_sev, dropped_nonfinite_z=n_bad_z),
    }
    if cross_model_bins:
        out["kendall_tau_b_cross_model"] = {
            key: {**c, "ci95": ci95(boot.get(f"cross:{key}", [])),
                  "resamples_used": len(boot.get(f"cross:{key}", []))}
            for key, c in cross.items()}
    return out


def disclosure_table(blocks: Mapping[str, Mapping[str, Any]]) -> list[str]:
    """Markdown rows (header first) of disclosure blocks: label -> block."""

    def fmt(m: Optional[Mapping[str, Any]]) -> str:
        if not m or m.get("value") is None:
            return "n/a" + (f" ({m.get('reason')})" if m and m.get("reason") else "")
        lo, hi = m.get("ci95") or [None, None]
        ci = f" [{lo:.3f}, {hi:.3f}]" if lo is not None and hi is not None else ""
        return f"{m['value']:.3f}{ci}"

    rows = ["| set | windows | cells | AUROC [95% CI] | tau_b (i) [95% CI] | tau_b (ii) [95% CI] (instants / pairs) |",
            "|---|---|---|---|---|---|"]
    for label, b in blocks.items():
        cross = b.get("kendall_tau_b_cross_model") or {}
        cm = "; ".join(f"{k}: {fmt(v)} ({v.get('instants')} / {v.get('pairs')})" for k, v in cross.items()) or "-"
        rows.append(f"| {label} | {b['auroc']['windows']} | {b['cells']} | {fmt(b['auroc'])} | "
                    f"{fmt(b['kendall_tau_b_window'])} | {cm} |")
    return rows
