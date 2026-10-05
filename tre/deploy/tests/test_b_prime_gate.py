"""B' as the gate of ``dline_refit accept`` (user 2026-10-03): the severity cut is sealed
from the training windows at freeze time, the gate is judged at the controller's dwell
(dwell 2 disclosed), and a missing cut refuses instead of falling back."""
from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path

import pytest

import test_dline_freeze_accept as base  # the D22 freeze / accept fixture world
from scripts import b_prime
from scripts import dline_refit as dl
from tre_calibration.dataset import CalibrationWindow

MODEL = base.MODEL
OVERLAY = Path(__file__).resolve().parents[1] / "overlays" / "tre-v2" / "controller.yaml"


def test_the_default_gate_dwell_is_the_controllers_deployed_dwell() -> None:
    # The controller has no band dwell since the timer cleanup (2026-10-02): every band
    # acts on its first window, i.e. dwell 1. An overlay that still pins it must say 1.
    text = OVERLAY.read_text(encoding="utf-8")
    m = re.search(r"name:\s*TRE_DWELL_WINDOWS\s*\n\s*value:\s*\"?(\d+)\"?", text)
    assert (int(m.group(1)) if m else 1) == dl.ONLINE_DWELL_WINDOWS == 1


def test_the_cut_is_sealed_from_training_windows_and_m_does_not_move_it(tmp_path) -> None:
    w = base._world(tmp_path)
    assert base._freeze(w) == 0
    doc = dl.verify_freeze(w["freeze"])
    rec = doc["models"][MODEL]["b_prime"]
    # the training violators are all TPOT 90 / 75 ms (fixed label): severity 1.2
    assert rec["severity_cut"] == pytest.approx(base.TRAIN_VIOLATING_TPOT / 75.0)
    assert rec["training_csv_sha256"] == base._sha(w["fit"] / f"{MODEL}_fitting.csv")
    assert doc["b_prime_gate"]["recall_severe_min"] == 0.80 and doc["b_prime_gate"]["online_dwell_windows"] == 1
    # M's own .65 quantile would be the TPOT cells' 100 / 75 - accept uses the sealed cut
    vh = doc["models"][MODEL]["verdict_for_holdout"]
    from scripts import theta_verdict as tv

    m_windows = tv.SignalSpec.from_dict(vh["signal_spec"]).load(
        w["mdir"] / "windows.csv", dl.slo_labels.LabelDefinition.from_dict(vh["label_def"]), 1)
    assert b_prime.severity_cut(m_windows) == pytest.approx(100.0 / 75.0)
    assert base._accept(w, base._seal(w)) == 0
    res = json.loads(dl.freeze_paths(w["freeze"])["result"].read_text())
    bp = res["models"][MODEL]["criteria"]["B_prime"]
    assert bp["severity_cut"] == rec["severity_cut"] and bp["cut_source"]["source"] == "freeze"
    assert res["thresholds"]["B_prime"]["cut_source"] == "freeze"


def _cells(n_cells: int, length: int, *, signal: float, healthy: bool, tag: str) -> list:
    return [CalibrationWindow(f"{tag}{c}", "f", signal, healthy, window_start_ms=10_000.0 * k,
                              latency_ratio_p95=None if healthy else 2.0,
                              health_score=0.8 if healthy else 1 / 3,
                              violation_class=None if healthy else "tpot_only")
            for c in range(n_cells) for k in range(length)]


def test_the_gate_reads_only_its_own_dwell_and_discloses_dwell_2() -> None:
    # violating cells two windows long: dwell 1 catches both, dwell 2 only the second
    ws = _cells(6, 2, signal=1.0, healthy=False, tag="v") + _cells(6, 4, signal=500.0, healthy=True, tag="h")
    cfg = {"severity_cut": 1.5, "gate": dict(b_prime.DEFAULT_GATE), "dwell_source": "test"}

    def ev(dwell):
        return dl.b_prime_evaluation(ws, theta=100.0, tau_crit=0.7, direction="higher_is_healthier",
                                     window_ms=30_000.0, cfg={**cfg, "dwell_windows": dwell},
                                     n_resamples=50, seed=1)

    one = ev(1)
    assert one["passed"] and one["criteria"][0]["value"] == 1.0
    assert set(one["by_dwell"]) == {"1", "2"} and one["by_dwell"]["2"]["recall_severe"] == 0.5
    two = ev(2)
    assert not two["passed"] and two["criteria"][0]["value"] == 0.5
    assert two["by_dwell"]["1"] == one["by_dwell"]["1"]
    # no cut: not evaluable, never passed
    none = dl.b_prime_evaluation(ws, theta=100.0, tau_crit=0.7, direction="higher_is_healthier",
                                 window_ms=30_000.0, cfg={**cfg, "dwell_windows": 1, "severity_cut": None},
                                 n_resamples=50, seed=1)
    assert none["evaluable"] is False and none["passed"] is False


def _as_old_freeze(w: dict) -> None:
    """Rewrite the freeze as a revision-2 one (no sealed cut), re-hashed."""
    ff = w["freeze"]
    doc = json.loads(ff.read_text())
    del doc["models"][MODEL]["b_prime"]
    del doc["b_prime_gate"]
    doc["format_revision"] = 2
    doc.pop("freeze_sha256")
    doc["freeze_sha256"] = dl.canonical_sha256(doc)
    data = dl._json_bytes(doc)
    os.chmod(ff, 0o644)
    ff.write_bytes(data)
    side = Path(f"{ff}.sha256")
    os.chmod(side, 0o644)
    side.write_text(f"{hashlib.sha256(data).hexdigest()}  {ff.name}\n")


def test_a_freeze_without_a_sealed_cut_refuses_unless_given_an_explicit_file(tmp_path, capsys) -> None:
    w = base._world(tmp_path)
    assert base._freeze(w) == 0
    man = base._seal(w)
    thresholds = tmp_path / "b_prime_thresholds.json"
    thresholds.write_text(json.dumps({"severity_cut_train": {MODEL: 1.2}, "gate": b_prime.DEFAULT_GATE}))
    # a revision-3 freeze carries its own cut: the file is refused, not mixed in
    capsys.readouterr()
    assert base._accept(w, man, "--b-prime-thresholds", str(thresholds)) == dl.EXIT_REFUSED
    assert "the freeze seals its own B' cuts" in capsys.readouterr().out
    _as_old_freeze(w)
    man = base._seal(w)
    base._refused(w, man, capsys, "seals no B' severity cut")
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"severity_cut_train": {"other-model": 1.2}, "gate": b_prime.DEFAULT_GATE}))
    capsys.readouterr()
    assert base._accept(w, man, "--b-prime-thresholds", str(bad)) == dl.EXIT_REFUSED
    assert "has no severity_cut_train for it" in capsys.readouterr().out
    assert base._accept(w, man, "--b-prime-thresholds", str(thresholds)) == 0
    res = json.loads(dl.freeze_paths(w["freeze"])["result"].read_text())
    src = res["models"][MODEL]["criteria"]["B_prime"]["cut_source"]
    assert src["source"] == "thresholds_file" and src["sha256"] == base._sha(thresholds)


def test_the_decision_reports_b_prime_from_accept_at_the_controllers_dwell(tmp_path) -> None:
    from scripts.analysis import calibration_decision as cd

    w = base._world(tmp_path)
    assert base._freeze(w) == 0
    assert base._accept(w, base._seal(w)) == 0
    result_path = dl.freeze_paths(w["freeze"])["result"]
    stored = json.loads(result_path.read_text())["models"][MODEL]["criteria"]["B_prime"]
    bp = cd.b_prime_from_accept(result_path)
    m = bp["models"][MODEL]
    assert bp["gate_dwell_windows"] == dl.ONLINE_DWELL_WINDOWS
    assert m["gate_point"] == stored["by_dwell"][str(dl.ONLINE_DWELL_WINDOWS)]
    assert m["passed"] == stored["passed"] and m["agrees_with_accept"] and bp["passed"] == m["passed"]
    assert "2" in m["disclosed_not_gating"] and str(dl.ONLINE_DWELL_WINDOWS) not in m["disclosed_not_gating"]
    # judged at dwell 2 instead: the stored dwell-2 point, and dwell 1 becomes the disclosure
    two = cd.b_prime_from_accept(result_path, online_dwell=2)["models"][MODEL]
    assert two["gate_point"] == stored["by_dwell"]["2"] and "1" in two["disclosed_not_gating"]
    # a dwell accept never stored is not evaluable, never passed
    five = cd.b_prime_from_accept(result_path, online_dwell=5)["models"][MODEL]
    assert five["evaluable"] is False and five["passed"] is False


# ------------------------------------------------ the onset gate (design 2026-10-05 item 3)


def _ew(cell: str, t_s: float, ratio: float):
    """One window of an episode timeline: ``ratio`` = p95 latency / SLO (<= 1 healthy)."""
    from types import SimpleNamespace

    return SimpleNamespace(scenario_id=cell, window_start_ms=t_s * 1000.0, slo_met=ratio <= 1.0, signal=1.0,
                           latency_ratio_p95=ratio, latency_ratio_avg=None, health_score=1.0)


def test_onset_episodes_score_onsets_not_drain_tails_and_count_misses_and_late_hits() -> None:
    cut = 5.0
    # cell -> (primitive, [(t_s, ratio)], CRITICAL instants)
    timeline = {
        # severe from 40 s, CRITICAL at 50 / 60 s (lag 10 s); the drain tail 70-110 s stays
        # violating without CRITICAL: one caught episode, not four tail misses
        "burst": ("bursts", [(0, .5), (10, .5), (20, .5), (30, 2), (40, 6), (50, 8), (60, 8), (70, 6), (80, 6),
                             (90, 6), (100, 2), (110, 2), (120, .5)], {50, 60}),
        "missed": ("steps", [(0, .5), (10, 2), (20, 6), (30, 6), (40, .5)], set()),
        "late": ("ramp", [(0, .5), (10, 6), (20, 6), (30, 6), (40, 6), (50, .5)], {40}),   # lag 30 s > budget
        "lead": ("bursts", [(0, .5), (10, 2), (20, 2), (30, 6), (40, .5)], {10}),         # lag -20 s
        "hold": ("hold", [(0, 6), (10, 6), (20, .5)], set()),                              # disclosed only
    }
    windows, crit, prim = [], [], {}
    for cell, (p, pts, hits) in timeline.items():
        prim[cell] = p
        for t, r in pts:
            windows.append(_ew(cell, t, r))
            crit.append(t in hits)
    got = b_prime.onset_detection(windows, crit, cut=cut, primitive_of=prim, gate=b_prime.ONSET_GATE,
                                  n_resamples=200)
    lags = {e["cell"]: e["lag_s"] for e in got["episodes"]}
    assert lags == {"burst": 10.0, "missed": None, "late": 30.0, "lead": -20.0, "hold": None}
    assert (got["onset"]["episodes"], got["onset"]["detected"], got["onset"]["within_budget"]) == (4, 3, 2)
    assert got["hold_disclosed"]["episodes"] == 1
    # a miss and a late hit: within the miss tolerance (2), but 2 of 4 fails the CP bound
    assert got["criteria"][0]["value"] == 2 and got["criteria"][0]["met"] and not got["passed"]
    # the same burst scored per window (B') counts the drain tail as misses
    burst = [i for i, w in enumerate(windows) if w.scenario_id == "burst"]
    point = b_prime.series_point([windows[i] for i in burst], theta=1.0, cut=cut, crit=[crit[i] for i in burst])
    assert point["recall_severe"] == pytest.approx(2 / 6)


def test_correlated_episodes_do_not_count_as_independent() -> None:
    # 3 cells x 2 episodes (60 s apart, beyond the 30 s look-back), all caught at lag 0: no miss,
    # but identical lags within a cell
    # leave the ICC undefined -> 1 (the conservative end) -> n_eff = 3 cells, not 6 episodes
    windows, crit, prim = [], [], {}
    for c in ("a", "b", "c"):
        prim[c] = "bursts"
        for t, r in [(0, 6), (10, .5), (20, .5), (30, .5), (40, .5), (50, .5), (60, 6), (70, .5)]:
            windows.append(_ew(c, t, r))
            crit.append(r > 1)
    got = b_prime.onset_detection(windows, crit, cut=5.0, primitive_of=prim, gate=b_prime.ONSET_GATE,
                                  n_resamples=200)
    assert got["onset"]["within_budget"] == 6 and got["icc_raw"] is None and got["icc"] == 1.0
    assert got["n_eff"] == pytest.approx(3.0)
    assert got["clopper_pearson_lower"] == pytest.approx(0.05 ** (1 / 3)) and not got["passed"]
    # the design's power figure: 0 misses on n_eff 14 -> one-sided 95 % lower bound .81
    assert b_prime.clopper_pearson_lower(14, 14, 0.05) == pytest.approx(0.807, abs=1e-3)
    assert b_prime.clopper_pearson_lower(13, 14, 0.05) == pytest.approx(0.70327, abs=1e-4)  # scipy beta.ppf


def _result(w: dict) -> dict:
    return json.loads(dl.freeze_paths(w["freeze"])["result"].read_text())


def test_the_onset_gate_has_three_verdicts_one_rule_for_every_model(tmp_path, monkeypatch) -> None:
    # pass: 16 steps cells, one onset episode each, caught at its first window (lag 0); one
    # episode per cell -> ICC undefined -> 1 -> n_eff 16 -> Clopper-Pearson low .05 ** (1/16) = .83
    w = base._world(tmp_path / "pass", steps_cells=16)
    assert base._freeze(w, "onset") == 0
    gate = dl.verify_freeze(w["freeze"])["accept_gate"]
    assert gate["rule"] == "onset" and gate["onset"]["lag_budget_s"] == 20.0 and gate["onset"]["miss_tolerance"] == 2
    assert base._accept(w, base._seal(w)) == 0
    res = _result(w)
    r, c = res["models"][MODEL], res["models"][MODEL]["criteria"]
    assert r["verdict"] == res["verdict"] == dl.VERDICT_PASS and res["passed"]
    assert (c["onset"]["onset"]["episodes"], c["onset"]["onset"]["within_budget"]) == (16, 16)
    assert c["onset"]["clopper_pearson_lower"] == pytest.approx(0.05 ** (1 / 16))
    assert c["window_fa"]["passed"] and c["B_prime"]["gating"] is False and c["B"]["gating"] is False
    # pass_a_disclosed: only A fails -> go-live allowed (exit 0), A disclosed as a limitation
    w2 = base._world(tmp_path / "a", steps_cells=16)
    assert base._freeze(w2, "onset") == 0
    a_min = dl.A_BA_MIN
    monkeypatch.setattr(dl, "A_BA_MIN", 1.01)   # BA is 1.0 on this M: A alone fails
    assert base._accept(w2, base._seal(w2)) == 0
    monkeypatch.setattr(dl, "A_BA_MIN", a_min)
    res2 = _result(w2)
    assert res2["verdict"] == dl.VERDICT_PASS_A_DISCLOSED and res2["passed"] and res2["failed"] == []
    assert not res2["models"][MODEL]["criteria"]["A"]["passed"]
    assert len(res2["disclosed_limitations"]) == 1 and "A failed" in res2["disclosed_limitations"][0]
    # fail: hold-only M has no onset episode -> the onset gate is not evaluable -> fail, exit 3,
    # though A, the window false alarm, D (and the disclosed B') pass
    w3 = base._world(tmp_path / "f")
    assert base._freeze(w3, "onset") == 0
    assert base._accept(w3, base._seal(w3)) == dl.EXIT_ACCEPT_FAILED
    res3 = _result(w3)
    c3 = res3["models"][MODEL]["criteria"]
    assert res3["verdict"] == dl.VERDICT_FAIL and not res3["passed"]
    assert c3["A"]["passed"] and c3["window_fa"]["passed"] and c3["D"]["passed"] and c3["B_prime"]["passed"]
    assert c3["onset"]["evaluable"] is False and res3["failed"] == [f"{MODEL}: onset failed - "
                                                                    "no severe-violation episode in a dynamic cell"]


def test_a_critical_of_an_earlier_episode_is_never_credited_to_the_next() -> None:
    # two episodes 20 s apart in one bursts cell; CRITICAL only during the first. The 30 s
    # look-back of episode 2 (from 40 s) reaches back to 10 s, but it is clipped at the end of
    # episode 1 (20 s, exclusive): episode 2 is missed
    pts = [(0, .5), (10, 6), (20, 6), (30, .5), (40, 6), (50, 6), (60, .5)]
    windows = [_ew("b", t, r) for t, r in pts]
    for hits, lag2 in (({10, 20}, None),        # CRITICAL ends with episode 1
                       ({10, 20, 30}, -10.0)):  # the run is still active after episode 1: it counts
        crit = [t in hits for t, _ in pts]
        got = b_prime.onset_detection(windows, crit, cut=5.0, primitive_of={"b": "bursts"}, gate=b_prime.ONSET_GATE,
                                      n_resamples=50)
        assert [e["lag_s"] for e in got["episodes"]] == [0.0, lag2]
        assert [e["t_end_prev"] for e in got["episodes"]] == [None, 20_000.0]


# ------------------------------------------- review 2026-10-05 P2-5 / P2-2 / P2-3 / P3-13


def test_the_verdict_fails_whenever_a_gate_other_than_a_fails() -> None:
    ok, ko = {"passed": True}, {"passed": False}
    crit = lambda **f: {g: f.get(g, ok) for g in dl.ONSET_GATING_CRITERIA}  # noqa: E731
    assert dl.verdict_of(crit()) == dl.VERDICT_PASS
    assert dl.verdict_of(crit(A=ko)) == dl.VERDICT_PASS_A_DISCLOSED
    assert dl.verdict_of(crit(A=ko, D=ko)) == dl.VERDICT_FAIL      # D is not waived with A
    assert dl.verdict_of(crit(D=ko)) == dl.VERDICT_FAIL
    assert dl.verdict_of(crit(window_fa=ko)) == dl.VERDICT_FAIL
    assert dl.verdict_of(crit(onset=ko, A=ko)) == dl.VERDICT_FAIL


def test_accept_refuses_a_dataset_of_another_label_attribution(tmp_path, capsys) -> None:
    w = base._world(tmp_path, steps_cells=16)
    assert base._freeze(w, "onset") == 0          # completion (label v1) freeze
    man = w["mdir"] / "manifest.json"
    man.write_text(json.dumps({**json.loads(man.read_text()), "attribution": {"value": "hybrid"}}))
    assert base._accept(w, base._seal(w)) == dl.EXIT_REFUSED
    assert "label attribution" in capsys.readouterr().out
    assert base._written(w) == ["params_freeze.json", "params_freeze.json.sha256"]


def test_freeze_refuses_a_fitted_label_whose_attribution_is_not_the_trainsets(tmp_path, capsys) -> None:
    w = base._world(tmp_path)
    d = w["out"] / MODEL / base.ARM
    for name in ("final.json", "verdict_final.json"):     # a fit labelled hybrid on completion data
        doc = json.loads((d / name).read_text())
        doc["label_def"] = {**doc["label_def"], "attribution": "hybrid"}
        (d / name).write_text(json.dumps(doc))
    assert base._freeze(w, "onset") == dl.EXIT_REFUSED
    assert "attribution 'hybrid' != the training datasets' 'completion'" in capsys.readouterr().out
    assert not w["freeze"].exists()


def test_accept_refuses_a_sealed_onset_gate_that_is_not_this_codes(tmp_path, capsys) -> None:
    w = base._world(tmp_path, steps_cells=16)
    assert base._freeze(w, "onset") == 0
    ff = Path(w["freeze"])
    doc = json.loads(ff.read_text())
    doc["accept_gate"]["onset"]["lookback_clip"] = "none"
    doc.pop("freeze_sha256")
    doc["freeze_sha256"] = dl.canonical_sha256(doc)
    data = dl._json_bytes(doc)
    side = Path(f"{ff}.sha256")
    for p in (ff, side):
        os.chmod(p, 0o644)
    ff.write_bytes(data)
    side.write_text(f"{hashlib.sha256(data).hexdigest()}  {ff.name}\n")
    assert base._accept(w, base._seal(w)) == dl.EXIT_REFUSED
    assert "accept_gate.onset.lookback_clip" in capsys.readouterr().out


def test_a_new_freeze_has_one_gate_and_refuses_another_tau(tmp_path, monkeypatch, capsys) -> None:
    w = base._world(tmp_path)
    with pytest.raises(SystemExit):   # no --accept-gate any more: one gate, no fork
        dl.main(["freeze", "--model", MODEL, "--arm", base.ARM, "--fit-dir", str(w["fit"]), "--out-dir",
                 str(w["out"]), "--freeze-file", str(w["freeze"]), "--accept-gate", "b_prime"])
    monkeypatch.setattr(dl, "ONSET_TAU_S", 5.0)   # the lag budget's base tau != the fit's 10 s
    assert base._freeze(w, "onset") == dl.EXIT_REFUSED
    assert "tau_s 10.0 != 5" in capsys.readouterr().out and not w["freeze"].exists()


def test_the_onset_gate_tolerates_two_misses_and_the_cp_bound_decides() -> None:
    """User 2026-10-05: misses <= 2 AND the one-sided 95 % CP lower bound on (x_eff, n_eff) >= .80."""
    def run(n_cells: int, misses: int) -> dict:
        windows, crit, prim = [], [], {}
        for k in range(n_cells):
            c = f"c{k}"
            prim[c] = "bursts"
            for t, r in [(0, .5), (10, 6), (20, 6), (30, .5)]:
                windows.append(_ew(c, t, r))
                crit.append(r > 1 and k >= misses)
        return b_prime.onset_detection(windows, crit, cut=5.0, primitive_of=prim, gate=b_prime.ONSET_GATE,
                                       n_resamples=100)

    two = run(40, 2)          # one episode per cell -> n_eff 40, x_eff 38: CP low ~.85
    assert two["passed"] and two["clopper_pearson_lower"] >= 0.80
    assert not run(40, 3)["passed"]                      # a third miss fails, whatever the bound
    assert not run(10, 2)["passed"]                      # 2 misses of 10: the CP bound fails
