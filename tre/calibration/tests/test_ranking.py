"""tre_calibration.ranking: AUROC, Kendall tau-b (window level and stratified cross-model)
and the exact matrix-weighted cell bootstrap, on hand-computed examples."""
from __future__ import annotations

import itertools
import math
import random

import pytest

from tre_calibration import ranking as rk
from tre_calibration.evaluate import _auc


def _pairwise_auc(scores, positives):
    pos = [s for s, p in zip(scores, positives) if p]
    neg = [s for s, p in zip(scores, positives) if not p]
    if not pos or not neg:
        return None
    return sum(1.0 if a > b else 0.5 if a == b else 0.0 for a in pos for b in neg) / (len(pos) * len(neg))


def _naive_tau_b(x, y):
    s = a = b = 0
    for i, j in itertools.combinations(range(len(x)), 2):
        dx, dy = rk._sgn(x[i] - x[j]), rk._sgn(y[i] - y[j])
        s += dx * dy
        a += dx != 0
        b += dy != 0
    return s / math.sqrt(a * b) if a and b else None


# ----------------------------------------------------------------------- AUROC


def test_auroc_counts_a_positive_negative_tie_as_one_half() -> None:
    # positives {2, 3} vs negatives {1, 2}: 2>1, 2=2 (1/2), 3>1, 3>2 -> 3.5 / 4
    assert rk.auroc([1.0, 2.0, 2.0, 3.0], [False, True, False, True]) == 0.875
    assert rk.auroc([1.0, 2.0], [True, True]) is None and rk.auroc([], []) is None
    assert rk.auroc([5.0, 5.0, 5.0], [True, False, True]) == 0.5


def test_auroc_is_exactly_the_pairwise_count_on_tied_data() -> None:
    rng = random.Random(7)
    for _ in range(30):
        n = rng.randint(2, 40)
        s = [float(rng.randint(0, 5)) for _ in range(n)]
        p = [rng.random() < 0.4 for _ in range(n)]
        assert rk.auroc(s, p) == _pairwise_auc(s, p)  # exact, not approx: half-integer sums


def test_evaluate_auc_delegates_and_keeps_its_one_class_fallback() -> None:
    assert _auc([1.0, 2.0, 2.0, 3.0], [0, 1, 0, 1]) == 0.875
    assert _auc([1.0, 2.0], [1, 1]) == 0.5 and _auc([], []) == 0.5


# ---------------------------------------------------------------------- tau-b


def test_tau_b_with_ties_in_x_and_in_y() -> None:
    # pairs: (1,2) tied y; (2,3) tied x; the other four concordant -> 4 / sqrt(5 * 5)
    assert rk.kendall_tau_b([1, 2, 2, 3], [1, 1, 2, 3]) == pytest.approx(0.8)
    assert rk.tau_b_counts([1, 2, 2, 3], [1, 1, 2, 3]) == (4, 5, 5)


def test_tau_b_with_joint_ties() -> None:
    # x = 1 1 2 2, y = 1 1 2 1: (1,2) joint tie; (3,4) tied x; (1,4) (2,4) tied y;
    # (1,3) (2,3) concordant -> C - D = 2, n0 - n1 = 4, n0 - n2 = 3
    assert rk.tau_b_counts([1, 1, 2, 2], [1, 1, 2, 1]) == (2, 4, 3)
    assert rk.kendall_tau_b([1, 1, 2, 2], [1, 1, 2, 1]) == pytest.approx(2 / math.sqrt(12))


def test_tau_b_extremes_and_undefined() -> None:
    assert rk.kendall_tau_b([1, 2, 3], [3, 2, 1]) == -1.0
    assert rk.kendall_tau_b([1, 2, 3], [1, 2, 3]) == 1.0
    assert rk.kendall_tau_b([1, 1, 1], [1, 2, 3]) is None
    assert rk.kendall_tau_b([1], [1]) is None and rk.kendall_tau_b([], []) is None


def test_knight_tau_b_equals_the_naive_quadratic_one() -> None:
    rng = random.Random(3)
    for _ in range(40):
        n = rng.randint(2, 60)
        x = [float(rng.randint(0, 6)) for _ in range(n)]
        y = [float(rng.randint(0, 4)) for _ in range(n)]
        want = _naive_tau_b(x, y)
        got = rk.kendall_tau_b(x, y)
        assert (got is None and want is None) or got == pytest.approx(want, abs=1e-12)


# -------------------------------------------------------------- cross-model


def _r(model, cell, t, z, sev, violated=False):
    return rk.RankRecord(model=model, cell=cell, instant=t, z=z, severity=sev, violated=violated)


def _cross_world():
    return [
        # instant 1000: a vs b concordant (a less pressure, less severe)
        _r("a", "a1", 1000, 1.0, 0.5), _r("b", "b1", 1000, 0.5, 1.0),
        # instant 2000: ab C, ac C, bc D (b less pressure but more severe than c)
        _r("a", "a1", 2000, 2.0, 0.2), _r("b", "b1", 2000, 0.8, 0.9), _r("c", "c1", 2000, 0.6, 0.7),
        # instant 3000: two windows of one model only - no cross-model pair
        _r("a", "a1", 3000, 0.1, 3.0), _r("a", "a2", 3000, 3.0, 0.1),
        # instant 4000: one model; and a window without severity at 1000 (dropped)
        _r("c", "c1", 4000, 1.0, 1.0), _r("c", "c2", 1000, 0.2, None),
    ]


def test_stratified_cross_model_tau_b_pairs_only_different_models_at_one_instant() -> None:
    res = rk.stratified_cross_model_tau_b(_cross_world())
    # sum (C - D) = 1 + (1 + 1 - 1) = 2 over 4 pairs, none tied -> 2 / sqrt(4 * 4)
    assert res["value"] == pytest.approx(0.5) and res["instants"] == 2 and res["pairs"] == 4
    assert res["instant_rule"] == "window_end_ms, exact"


def test_cross_model_tau_b_is_null_without_concurrent_models() -> None:
    one = [_r("a", "a1", 1000, 1.0, 1.0), _r("a", "a1", 2000, 0.5, 2.0), _r("b", "b1", 3000, 0.4, 2.0)]
    res = rk.stratified_cross_model_tau_b(one)
    assert res["value"] is None and res["pairs"] == 0 and res["instants"] == 0
    assert res["reason"] == rk.CROSS_MODEL_UNAVAILABLE and "E1 or concurrent campaigns" in res["reason"]
    # rounding the instants to 10 s brings 2000 / 3000 no closer; 1000 / 1004 do meet
    near = [_r("a", "a1", 1000, 1.0, 1.0), _r("b", "b1", 1004, 0.5, 2.0)]
    assert rk.stratified_cross_model_tau_b(near)["value"] is None
    binned = rk.stratified_cross_model_tau_b(near, bin_ms=10)
    assert binned["value"] == 1.0 and binned["pairs"] == 1 and "nearest 10 ms" in binned["instant_rule"]


# ---------------------------------------------------------------- bootstrap


def test_ci95_index_rule() -> None:
    assert rk.ci95(list(range(100))) == [2, 96]
    assert rk.ci95([]) == [None, None] and rk.ci95([0.3]) == [0.3, 0.3]


def test_single_stratum_draws_are_the_acceptance_bootstrap_draws() -> None:
    cells = ["c3", "c1", "c2", "c5"]
    rng = random.Random(11)
    want = []
    for _ in range(25):
        pick = [rng.choice(sorted(cells)) for _ in cells]
        want.append({c: pick.count(c) for c in set(pick)})
    got = [dict(w) for w in rk.stratified_draws({"m": cells}, 25, 11)]
    assert got == want


def test_pooled_draws_are_stratified_by_model() -> None:
    strata = {"a": [("a", "1"), ("a", "2"), ("a", "3")], "b": [("b", "1"), ("b", "2")]}
    for w in rk.stratified_draws(strata, 200, 5):
        assert sum(k for c, k in w.items() if c[0] == "a") == 3
        assert sum(k for c, k in w.items() if c[0] == "b") == 2
    assert [dict(w) for w in rk.stratified_draws(strata, 20, 5)] == [dict(w) for w in rk.stratified_draws(strata, 20, 5)]


def _small_world(seed: int = 4) -> list:
    rng = random.Random(seed)
    recs = []
    for m in ("a", "b", "c"):
        for c in range(3):
            for t in range(4):
                z = rng.choice([0.5, 0.8, 1.0, 1.2, 1.2, 2.0])
                sev = rng.choice([0.3, 0.6, 0.6, 1.1, 2.0, None])
                recs.append(_r(m, f"{m}{c}", 1000 * (t + 3 * c if m != "c" else t), z, sev, violated=z < 1.0))
    return recs


def _expand(records, w):
    return [r for (m, c), k in sorted(w.items()) for _ in range(k) for r in records if (r.model, r.cell) == (m, c)]


@pytest.mark.parametrize("backend", ["python", "numpy"])
def test_matrix_bootstrap_equals_the_expanded_resample(backend) -> None:
    if backend == "numpy":
        pytest.importorskip("numpy")
    recs = _small_world()
    strata = {}
    for r in recs:
        strata.setdefault(r.model, set()).add((r.model, r.cell))
    index = {c: k for k, c in enumerate(sorted(c for cs in strata.values() for c in cs))}
    cell_of = [index[(r.model, r.cell)] for r in recs]
    n = len(index)
    use_np = backend == "numpy"
    ok = [i for i, r in enumerate(recs) if r.severity is not None]
    f_auc = rk.auroc_forms(cell_of, [r.pressure for r in recs], [r.violated for r in recs], n, use_numpy=use_np)
    f_tau = rk.tau_forms([cell_of[i] for i in ok], [recs[i].pressure for i in ok],
                         [recs[i].severity for i in ok], n, use_numpy=use_np)
    pairs, _ = rk.cross_model_pairs(recs)
    f_x = rk.tau_forms(cell_of, [r.pressure for r in recs], [r.severity or math.nan for r in recs], n,
                       use_numpy=use_np, pairs=pairs)
    checked = 0
    for draw in rk.stratified_draws({m: list(cs) for m, cs in strata.items()}, 40, 9):
        w = [0] * n
        for c, k in draw.items():
            w[index[c]] = k
        ex = _expand(recs, draw)
        want_auc = rk.auroc([r.pressure for r in ex], [r.violated for r in ex])
        exs = [r for r in ex if r.severity is not None]
        want_tau = rk.kendall_tau_b([r.pressure for r in exs], [r.severity for r in exs])
        want_x = rk.stratified_cross_model_tau_b(ex)["value"]
        for got, want in ((f_auc.value(w), want_auc), (f_tau.value(w), want_tau), (f_x.value(w), want_x)):
            assert (got is None and want is None) or got == pytest.approx(want, abs=1e-12)
            checked += got is not None
    assert checked > 60
    # all ones = the point estimate
    ones = [1] * n
    assert f_tau.value(ones) == pytest.approx(rk.kendall_tau_b([recs[i].pressure for i in ok],
                                                               [recs[i].severity for i in ok]))
    assert f_x.value(ones) == pytest.approx(rk.stratified_cross_model_tau_b(recs)["value"])


def test_disclosure_is_deterministic_and_backend_independent() -> None:
    recs = _small_world(8)
    a = rk.ranking_disclosure(recs, n_resamples=60, seed=3, cross_model_bins=(None, 1000.0), use_numpy=False)
    assert a == rk.ranking_disclosure(recs, n_resamples=60, seed=3, cross_model_bins=(None, 1000.0), use_numpy=False)
    assert a["gating"] is False and a["bootstrap"]["unit"].endswith("stratified by model")
    assert a["auroc"]["value"] == rk.auroc([r.pressure for r in recs], [r.violated for r in recs])
    n_sev = sum(1 for r in recs if r.severity is None)
    assert a["kendall_tau_b_window"]["dropped_no_severity"] == n_sev
    assert a["kendall_tau_b_window"]["windows"] == len(recs) - n_sev
    assert set(a["kendall_tau_b_cross_model"]) == {"exact", "bin_1000ms"}
    lo, hi = a["auroc"]["ci95"]
    assert lo <= hi and a["auroc"]["resamples_used"] <= 60
    try:
        import numpy  # noqa: F401
    except ImportError:
        return
    b = rk.ranking_disclosure(recs, n_resamples=60, seed=3, cross_model_bins=(None, 1000.0), use_numpy=True)
    assert a == b  # integer matrices: bit-identical


def test_disclosure_drops_nonfinite_z_and_counts_it() -> None:
    recs = [_r("a", "c1", 0, 1.0, 0.5), _r("a", "c1", 1, math.nan, 0.9, True), _r("a", "c2", 0, 0.5, 1.5, True),
            _r("a", "c2", 1, 2.0, 0.2)]
    d = rk.ranking_disclosure(recs, n_resamples=20, seed=1, use_numpy=False)
    assert d["auroc"]["dropped_nonfinite_z"] == 1 and d["auroc"]["windows"] == 3
    assert d["auroc"]["value"] == 1.0 and d["kendall_tau_b_window"]["value"] == 1.0
    assert "kendall_tau_b_cross_model" not in d and d["cells"] == 2
    assert d["definition"] == rk.DEFINITIONS


def test_records_from_windows_use_signal_z_and_the_label_ratio() -> None:
    from tre_calibration.dataset import CalibrationWindow

    ws = [CalibrationWindow("c", "f", 50.0, False, window_start_ms=0.0, latency_ratio_p95=1.4),
          CalibrationWindow("c", "f", 200.0, True, window_start_ms=10_000.0, latency_ratio_p95=None)]
    recs = rk.records_from_windows("m", ws, theta=100.0, direction="higher_is_healthier", instants=[30_000.0, None])
    assert [(r.z, r.pressure, r.severity, r.violated, r.instant) for r in recs] == [
        (0.5, -0.5, 1.4, True, 30_000.0), (2.0, -2.0, None, False, None)]
