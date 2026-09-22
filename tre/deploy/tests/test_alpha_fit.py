"""The D4' alpha rule (plan 2026-09-21 §6.11): scripts.alpha_fit."""
from __future__ import annotations

import math
from pathlib import Path

import pytest

from scripts import alpha_fit as af
from scripts import calibration_campaign as campaign
from scripts.theta_verdict import critical_dwell_flags
from tre_calibration.dataset import CalibrationWindow


def test_alpha_grid_is_the_plan_grid_at_dt_ref_10s() -> None:
    alphas = [round(af.alpha_of_tau(t), 2) for t in af.TAU_GRID_S]
    assert alphas == [1.0, 0.86, 0.63, 0.49, 0.39, 0.28, 0.22, 0.15]
    for t in af.TAU_GRID_S:
        assert af.tau_of_alpha(af.alpha_of_tau(t)) == pytest.approx(t)
    with pytest.raises(ValueError):
        af.tau_of_alpha(0.0)


def test_step_response_counts_refreshes_and_dwell() -> None:
    assert af.step90_s(0) == 10.0                       # alpha 1: one refresh
    assert af.step90_s(10) == 30.0                      # 1-(1-.632)^3 >= .9
    assert af.step90_s(10, dwell_windows=2) == 40.0     # + one confirming window
    assert af.step90_s(60) > af.step90_s(20) > af.step90_s(10)


def test_shape_and_steady_cell_from_scenario_id() -> None:
    table = af.shape_table()
    assert af.shape_of("i1600_o112_c1120", table) == "T8"
    assert af.shape_of("i812_o241_c95", table) == "T9"      # sampled shape, nominal lengths
    assert af.shape_of("i400_o320_c2100", table) == "G400x320"
    assert af.shape_of("i256_o128_c60#3", table) == "S1"    # bootstrap copy
    assert af.shape_of("i9_o9_c95", table) == "i9_o9"       # unknown: its own shape
    assert af.is_steady_cell("i256_o128_c1177") and af.is_steady_cell("i400_o160_c2085")
    assert not any(af.is_steady_cell(f"i256_o128_c{c}") for c in (60, 95, 120))


def test_cell_stats_spurious_episodes_and_lags() -> None:
    t = [i * 10_000.0 for i in range(12)]
    crit = [0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0]
    viol = [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0]
    st = af.cell_stats("i256_o128_c1100", t, [bool(c) for c in crit], [bool(v) for v in viol])
    # run at 10-20 s has no violation within +-30 s -> spurious; run at 80-90 s is not
    assert st.spurious == 1
    assert (st.tp, st.fn, st.fp, st.tn) == (1, 1, 3, 7)
    assert st.lags_s == [-10.0]           # confirmed at 80 s, episode starts at 90 s
    assert st.steady and not st.healthy
    assert st.hours == pytest.approx(12 * 10 / 3600)


def _p(tau, ba, se, fa, spur):
    return {"tau_s": tau, "alpha": af.alpha_of_tau(tau), "ba": ba, "se": se, "fa": fa, "spurious_per_h": spur}


def test_select_feasibility_then_1se_then_spurious_then_larger_alpha() -> None:
    pts = [_p(0, 0.90, 0.02, 0.08, 1.0),   # best BA but infeasible (FA > 5 %)
           _p(5, 0.85, 0.02, 0.04, 6.0),
           _p(10, 0.84, 0.02, 0.03, 3.0),
           _p(20, 0.80, 0.02, 0.01, 0.0)]  # fewest spurious but outside 1 SE
    r = af.select(pts)
    assert r["best_ba_tau_s"] == 5 and r["within_1se_tau_s"] == [5, 10]
    assert r["chosen_tau_s"] == 10 and "spurious" in r["reason"]
    tie = af.select([_p(5, 0.85, 0.02, 0.04, 3.0), _p(10, 0.84, 0.02, 0.03, 3.0)])
    assert tie["chosen_tau_s"] == 5 and "larger alpha" in tie["reason"]
    none = af.select([_p(0, 0.9, 0.02, 0.2, 0.0), _p(10, 0.8, 0.02, 0.1, 0.0)])
    assert none["no_feasible_alpha"] and none["chosen_tau_s"] == 0


def _w(cell, k, signal, cls=None):
    return CalibrationWindow(cell, "f", float(signal), cls is None, window_start_ms=k * 10_000.0,
                             violation_class=cls)


def _synthetic(tau):
    """Two dynamic cells per shape with a clean violation, plus steady healthy hold cells
    whose two-window dip is dwell-confirmed only without smoothing (tau 0)."""
    ws = []
    for shape in ("i256_o128", "i768_o192", "i2048_o96"):
        for code in (60, 95):
            cell = f"{shape}_c{code}"
            sig = [200] * 5 + [10] * 4 + [200] * 5
            ws += [_w(cell, k, s, "both" if s < 50 else None) for k, s in enumerate(sig)]
        hold = f"{shape}_c1100"
        dip = 10 if tau == 0 else 80
        sig = [200] * 5 + [dip] * 2 + [200] * 5
        ws += [_w(hold, k, s) for k, s in enumerate(sig)]
    return ws


def _crit(ws, theta, tau_crit):
    return critical_dwell_flags(ws, theta=theta, tau_crit=tau_crit, direction="higher_is_healthier")


def test_alpha_rule_prefers_fewer_spurious_episodes_and_writes_registry_fields() -> None:
    fits = []

    def fit(train):
        fits.append(len(train))
        return 100.0, 0.5

    rep = af.alpha_rule(_synthetic, fit, _crit, tau_grid_s=(0.0, 10.0), bootstrap=40, se_resamples=50)
    by = {p["tau_s"]: p for p in rep["curve"]}
    assert len(fits) == 2 * 3                       # one fit per LOSO fold per alpha
    assert by[0.0]["spurious_n"] == 3 and by[10.0]["spurious_n"] == 0
    assert by[0.0]["steady_healthy_cells"] == 3
    assert by[0.0]["fa"] > by[10.0]["fa"]
    assert rep["selection"]["chosen_tau_s"] == 10.0
    assert rep["registry_fields"]["trs"] == {"ema_tau_ms": 10_000.0, "ema_alpha": round(1 - math.exp(-1), 6)}
    freq = rep["bootstrap"]["selection_frequency_by_tau_s"]
    assert sum(freq.values()) == pytest.approx(1.0) and rep["bootstrap"]["used"] == 40
    assert by[10.0]["detect_lag_median_s"] == 10.0  # dwell 2: confirmed one window in
    assert by[10.0]["step90_with_dwell_s"] == 40.0


def test_bootstrap_refit_refits_every_fold_and_keeps_copies_apart() -> None:
    calls = []

    def fit(train):
        calls.append({w.scenario_id for w in train})
        return 100.0, 0.5

    rep = af.alpha_rule(_synthetic, fit, _crit, tau_grid_s=(0.0, 10.0), bootstrap=3, se_resamples=10,
                        bootstrap_refit=True)
    assert rep["bootstrap"]["refit"] is True and rep["bootstrap"]["used"] == 3
    assert len(calls) > 6                            # 6 full-sample fits + refits per resample
    assert any("#" in c for ids in calls[6:] for c in ids)


def test_tau_zero_deploys_as_identity_ema() -> None:
    assert af.registry_fields(0.0)["trs"] == {"ema_tau_ms": 0.0, "ema_alpha": 1.0}


class _Args:
    out_dir = Path("/out")
    window_ms = 30000
    fit_step_ms = 10000
    instant_sample_ms = 1000
    ttft_slo_ms = 500.0
    tpot_slo_ms = 75.0
    max_model_error_rate = 0.05
    envoy_stats_url = None


def test_fit_plan_runs_the_alpha_stage_before_theta() -> None:
    plan = campaign.fit_plan(["dsqwen-7b"], Path("/out"), Path("/raw"), _Args())
    order = plan["order"]
    assert order.index("rewindow") < order.index("alpha") < order.index("theta")
    [entry] = plan["alpha"]
    cmd = entry["command"]
    assert cmd[1:3] == ["-m", "scripts.alpha_fit"]
    assert cmd[cmd.index("--fitting-csv") + 1] == "/out/fit/dsqwen-7b_fitting.csv"
    assert cmd[cmd.index("--step-ms") + 1] == "10000"
    assert "--ttft-slo-mode" in cmd
    rule = plan["alpha_rule"]
    assert rule["alpha_grid"] == [1.0, 0.8647, 0.6321, 0.4866, 0.3935, 0.2835, 0.2212, 0.1535]
    assert rule["label_horizon"] == "same window" and rule["fa_max"] == 0.05
