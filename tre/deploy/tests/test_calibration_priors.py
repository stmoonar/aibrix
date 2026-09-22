"""Priors the second calibration run takes from the first (``scripts.analysis.calibration_priors``).

Two inputs are frozen before the run: the regime grouping LORO holds out, and the
per-shape boundary the stage-0 search starts from. These tests pin the rules that turn
first-run data into those numbers.
"""
from __future__ import annotations

import json
import math
import random

import pytest

from scripts.analysis import calibration_priors as cp


# ------------------------------------------------------------------ regime groups


def test_seven_shapes_cut_three_ways_with_two_each_has_exactly_three_cuts() -> None:
    assert sorted(cp.contiguous_partitions(7, 3, 2)) == [(2, 2, 3), (2, 3, 2), (3, 2, 2)]


def test_grouping_follows_the_gaps_in_log_ratio() -> None:
    ratios = {"S4": 0.007, "S5": 0.012, "S1": 0.024, "S2": 0.025, "T9": 0.026, "T8": 0.07, "S3": 0.12}
    g = cp.group_by_ratio(ratios)
    assert g["groups"] == {"decode": ["S4", "S5"], "mixed": ["S1", "S2", "T9"], "prefill": ["S3", "T8"]}
    assert g["order_low_to_high"][0] == "S4" and g["order_low_to_high"][-1] == "S3"


def test_grouping_needs_enough_shapes() -> None:
    with pytest.raises(ValueError):
        cp.group_by_ratio({"a": 1.0, "b": 2.0, "c": 3.0, "d": 4.0, "e": 5.0})


def _requests(model, shape, ratio, n=30, primitive="steps", start=0):
    rows = []
    for k in range(n):
        decode = 1000.0 + 10 * k
        rows.append({"model": model, "shape": shape, "primitive": primitive, "cell_id": f"c_{shape}",
                     "attempt": "1", "cell_status": "valid", "outcome": "ok", "output_tokens": "100",
                     "ttft_ms": str(ratio * decode), "e2e_ms": str(decode + ratio * decode),
                     "send_ts_ms": str(start + 1000 * k)})
    return rows


def test_time_ratio_is_the_median_of_ttft_over_decode_time_of_served_requests() -> None:
    rows = _requests("m", "S1", 0.02)
    rows.append({**rows[0], "outcome": "client_timeout", "ttft_ms": "9e9"})  # unserved: ignored
    rows.append({**rows[0], "output_tokens": "1", "ttft_ms": "9e9"})  # no decode phase: ignored
    rows.append({**rows[0], "cell_status": "void", "ttft_ms": "9e9"})  # void attempt: ignored
    out = cp.request_time_ratios(rows)
    assert math.isclose(out["m"]["S1"]["median_ratio"], 0.02) and out["m"]["S1"]["n"] == 30


def test_regime_groups_report_whether_the_models_agree() -> None:
    ratios = {"S4": 0.007, "S5": 0.012, "S1": 0.024, "S2": 0.025, "T9": 0.026, "T8": 0.07, "S3": 0.12}
    rows, cells = [], []
    for model in ("a", "b"):
        for shape, r in ratios.items():
            rows += _requests(model, shape, r)
            cells.append({"model": model, "shape": shape, "primitive": "steps", "cell_id": f"c_{shape}",
                          "attempt": "1", "start_ms": "0"})
    # model b sees T9 as prefill-heavy: its grouping differs, and the document must say so
    rows = [dict(r, ttft_ms=str(float(r["ttft_ms"]) * 4)) if r["model"] == "b" and r["shape"] == "T9" else r
            for r in rows]
    doc = cp.regime_groups(rows, cells)
    assert not doc["consistent_across_models"] and doc["groups"] is None
    assert doc["groups_by_model"]["a"] == {"decode": ["S4", "S5"], "mixed": ["S1", "S2", "T9"], "prefill": ["S3", "T8"]}
    assert doc["groups_by_model"]["a"] != doc["groups_by_model"]["b"]
    assert doc["models"]["a"]["light_load_agrees"] is True


# -------------------------------------------------------------------- boundary


def test_pava_returns_a_non_decreasing_fit() -> None:
    blocks = cp.pava([0, 1, 0, 0, 1, 1, 0, 1, 1])
    fitted = [b[2] for b in blocks]
    assert fitted == sorted(fitted)
    assert blocks[0][0] == 0 and blocks[-1][1] == 8


def test_the_boundary_is_the_jump_of_the_monotone_fit() -> None:
    pts = [(0.8 + 0.01 * i, 0) for i in range(20)] + [(1.2 + 0.01 * i, 1) for i in range(20)]
    iso = cp.isotonic_crossing(pts, min_violating_windows=7)
    assert iso["found"] and iso["healthy_side_rho"] == pytest.approx(0.99)
    assert iso["violating_side_rho"] == pytest.approx(1.2)
    assert iso["crossing"] == pytest.approx((0.99 + 1.2) / 2)


def test_all_healthy_means_no_boundary_and_the_highest_healthy_load() -> None:
    iso = cp.isotonic_crossing([(0.5 + 0.1 * i, 0) for i in range(15)])
    assert not iso["found"] and iso["reason"] == "never_crosses"
    assert iso["max_healthy_rho"] == pytest.approx(1.9)


def test_a_crossing_on_too_few_violating_windows_is_not_a_boundary() -> None:
    pts = [(1.0, 0)] * 30 + [(1.5, 1)] * 3
    iso = cp.isotonic_crossing(pts, min_violating_windows=7)
    assert not iso["found"] and iso["reason"] == "too_little_violating_evidence"
    assert iso["crossing"] is not None


def test_logistic_cross_check_recovers_the_midpoint() -> None:
    rng = random.Random(1)
    pts = []
    for _ in range(3000):
        x = rng.uniform(0.5, 2.0)
        p = 1 / (1 + math.exp(-8 * math.log(x / 1.2)))
        pts.append((x, 1 if rng.random() < p else 0))
    assert cp.logistic_fit(pts)["rho50"] == pytest.approx(1.2, rel=0.05)


def test_logistic_says_so_when_one_class_is_missing() -> None:
    assert cp.logistic_fit([(1.0, 0), (1.2, 0)])["rho50"] is None


def _entry(found, **kw):
    base = {"boundary_found": found, "boundary_rho": 1.2 if found else None,
            "healthy_side_rho": 1.1 if found else None, "violating_side_rho": 1.3 if found else None,
            "max_healthy_rho": 1.9, "max_driven_rho": 1.9, "tentative_boundary_rho": None,
            "extrapolation": {"rho_latency_ratio_reaches_1": None, "rho_running_reaches_max_num_seqs": None}}
    base.update(kw)
    return base


def test_a_found_boundary_brackets_the_search_around_it() -> None:
    s = cp.suggest_search(_entry(True))
    assert s["start_rho"] == 1.2 and s["lower_rho"] <= 1.1 and s["upper_rho"] >= 1.3


def test_a_missing_boundary_searches_above_the_highest_healthy_load() -> None:
    s = cp.suggest_search(_entry(False))
    assert s["start_rho"] > 1.9 and s["lower_rho"] == 1.9
    assert s["upper_rho"] >= 1.5 * 1.9


def test_the_batch_full_ceiling_bounds_the_search_but_never_below_one_and_a_half_times() -> None:
    ext = {"rho_latency_ratio_reaches_1": 3.0, "rho_running_reaches_max_num_seqs": 3.5}
    assert cp.suggest_search(_entry(False, extrapolation=ext))["upper_rho"] == 3.5
    ext = {"rho_latency_ratio_reaches_1": None, "rho_running_reaches_max_num_seqs": 2.0}
    assert cp.suggest_search(_entry(False, extrapolation=ext))["upper_rho"] == pytest.approx(2.85)


def test_the_search_is_capped() -> None:
    ext = {"rho_latency_ratio_reaches_1": None, "rho_running_reaches_max_num_seqs": 40.0}
    assert cp.suggest_search(_entry(False, extrapolation=ext), search_cap_rho=6.0)["upper_rho"] == 6.0


def test_a_tentative_crossing_is_where_the_search_starts() -> None:
    s = cp.suggest_search(_entry(False, tentative_boundary_rho=1.45, max_healthy_rho=1.44))
    assert s["start_rho"] == 1.45 and s["lower_rho"] == pytest.approx(0.85 * 1.45, abs=1e-3)


def test_capacity_must_match_what_the_run_scheduled_with(tmp_path) -> None:
    (tmp_path / "m" / "capacity").mkdir(parents=True)
    (tmp_path / "m" / "capacity" / "m_S1.json").write_text(json.dumps(
        {"model": "m", "shape": "S1", "capacity_used_rps": 5.0}))
    ok = cp.capacities(tmp_path, [{"model": "m", "shape": "S1", "primitive": "hold", "capacity_rps": "5.0"}])
    assert ok[("m", "S1")]["C_s_rps"] == 5.0
    with pytest.raises(ValueError, match="capacity"):
        cp.capacities(tmp_path, [{"model": "m", "shape": "S1", "primitive": "ramp", "capacity_rps": "4.0"}])


def test_the_tool_refuses_to_write_into_the_run_directory(tmp_path) -> None:
    with pytest.raises(SystemExit, match="read-only"):
        cp.main([str(tmp_path), "--out-dir", str(tmp_path / "priors")])
