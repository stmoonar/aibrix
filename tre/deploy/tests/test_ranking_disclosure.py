"""The ranking disclosure of ``dline_refit accept`` and ``scripts.ranking_report``, and the
windowing parameters (window / step / dt_ref / horizon / dwell) of the D-line stages."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
from pathlib import Path

import pytest

import test_dline_freeze_accept as base  # the D22 freeze / accept fixture world
from scripts import alpha_fit as af
from scripts import dline_refit as dl
from tre_calibration import ranking as rk
from tre_calibration.dataset import CalibrationWindow

MODEL = base.MODEL


# ------------------------------------------------------------- shared helpers


def test_the_auroc_callers_delegate_with_their_semantics() -> None:
    from scripts import calibration_resplit
    from scripts.analysis import e5_timeline_auroc as e5
    from tre_calibration.evaluate import _auc

    assert calibration_resplit.auroc([1.0, 2.0, 2.0, 3.0], [False, True, False, True]) == 0.875
    assert calibration_resplit.auroc([1.0], [True]) is None
    assert e5.auroc([(1.0, False), (2.0, True), (2.0, False), (3.0, True)]) == 0.875
    assert e5.auroc([(1.0, True)]) is None
    assert _auc([1.0], [1]) == 0.5  # evaluate keeps its one-class fallback


def test_ranking_ci95_is_dline_refit_ci95() -> None:
    vals = [random.Random(1).random() for _ in range(137)]
    assert rk.ci95(vals) == dl._ci95(vals)


# ---------------------------------------------------------------- accept


def _accepted(tmp_path: Path, **world) -> tuple[dict, dict]:
    w = base._world(tmp_path, **world)
    assert base._freeze(w) == 0
    assert base._accept(w, base._seal(w)) == 0
    return w, json.loads(dl.freeze_paths(w["freeze"])["result"].read_text())


def test_accept_discloses_ranking_per_model_and_pooled_never_gating(tmp_path, capsys) -> None:
    w, res = _accepted(tmp_path)
    out = capsys.readouterr().out
    assert "ranking disclosure (pressure = -Z; not gating)" in out and "| pooled |" in out
    assert res["format_revision"] == dl.ACCEPT_FORMAT_REVISION == 2
    r = res["models"][MODEL]
    assert r["windowing"] == dl.DEFAULT_WINDOWING and res["thresholds"]["dwell_windows"] == 2
    d = r["ranking_disclosure"]
    assert d["gating"] is False and d["definition"] == rk.DEFINITIONS
    # healthy windows sit far above theta, violating ones far below: AUROC 1
    assert d["auroc"]["value"] == 1.0 and d["auroc"]["ci95"] == [1.0, 1.0]
    assert (d["auroc"]["windows"], d["auroc"]["violated"], d["auroc"]["healthy"]) == (132, 66, 66)
    # severity (fixed label ratio_max): healthy .4, TPOT cells 100/75, TTFT-only 900/500; the
    # violating windows share one pressure: C = 66 * 66, x-untied 4356, y-untied 8646 - 3685
    assert d["kendall_tau_b_window"]["value"] == pytest.approx(math.sqrt(4356 / 4961))
    assert d["kendall_tau_b_window"]["dropped_no_severity"] == 0
    assert d["bootstrap"] == {"n_resamples": base.BOOT, "seed": dl.SEED,
                              "unit": "cell (scenario id), drawn with replacement", "interval": "95 % percentile"}
    pooled = res["ranking_disclosure"]
    assert pooled["gating"] is False and pooled["models"] == [MODEL]
    assert pooled["auroc"]["value"] == 1.0
    cm = pooled["kendall_tau_b_cross_model"]
    assert set(cm) == {"exact", "bin_10000ms"}
    assert cm["exact"]["value"] is None and cm["exact"]["reason"] == rk.CROSS_MODEL_UNAVAILABLE
    # the BA and the ranking CIs come from the same cell resamples
    assert r["bootstrap"]["cells"] == d["cells"] == 12


def _resign_marker(w: dict, stored: dict) -> None:
    fp = dl.freeze_paths(w["freeze"])
    data = dl._json_bytes(stored)
    for f in (fp["result"], fp["marker"]):
        os.chmod(f, 0o644)
    fp["result"].write_bytes(data)
    fp["marker"].write_bytes(dl._json_bytes({"result": str(fp["result"]),
                                             "result_sha256": hashlib.sha256(data).hexdigest(),
                                             "accepted_at_utc": "t"}))


def test_recheck_of_a_revision_1_result_ignores_only_the_new_keys(tmp_path, capsys) -> None:
    w, res = _accepted(tmp_path)
    old = dl.as_revision(res, 1)
    assert "ranking_disclosure" not in old and "windowing" not in old["models"][MODEL]
    assert old["format_revision"] == 1 and old["models"][MODEL]["criteria"] == res["models"][MODEL]["criteria"]
    _resign_marker(w, old)
    man = base._seal(w)
    capsys.readouterr()
    assert base._accept(w, man, "--recheck") == 0
    out = capsys.readouterr().out
    assert "format revision 1" in out and "identical" in out
    # a real difference is still one
    old["models"][MODEL]["M"]["violating"] = 65
    _resign_marker(w, old)
    assert base._accept(w, man, "--recheck") == dl.EXIT_RECHECK_DIFFERS
    assert "models.dsqwen-7b.M.violating: stored 65 != recomputed 66" in capsys.readouterr().out


def test_a_revision_1_freeze_verifies_and_accepts_at_the_default_windowing(tmp_path) -> None:
    w = base._world(tmp_path)
    assert base._freeze(w) == 0
    ff = w["freeze"]
    doc = json.loads(ff.read_text())
    assert doc["format_revision"] == 2
    assert doc["models"][MODEL]["windowing"] == {**dl.DEFAULT_WINDOWING,
                                                 "source": "defaults: final.json predates the windowing record"}
    # rewrite it as a revision-1 freeze (no windowing), re-hashed: the D22 shape
    del doc["models"][MODEL]["windowing"]
    doc["format_revision"] = 1
    doc.pop("freeze_sha256")
    doc["freeze_sha256"] = dl.canonical_sha256(doc)
    data = dl._json_bytes(doc)
    os.chmod(ff, 0o644)
    ff.write_bytes(data)
    side = Path(f"{ff}.sha256")
    os.chmod(side, 0o644)
    side.write_text(f"{hashlib.sha256(data).hexdigest()}  {ff.name}\n")
    assert dl.verify_freeze(ff)["format_revision"] == 1
    assert base._accept(w, base._seal(w)) == 0
    res = json.loads(dl.freeze_paths(ff)["result"].read_text())
    assert res["models"][MODEL]["windowing"] == dl.DEFAULT_WINDOWING
    assert res["models"][MODEL]["holdout_report"]["with_dwell"]["dwell_windows"] == 2
    # a revision the reader does not know is refused
    doc["format_revision"] = 3
    doc["freeze_sha256"] = dl.canonical_sha256({k: v for k, v in doc.items() if k != "freeze_sha256"})
    data = dl._json_bytes(doc)
    ff.write_bytes(data)
    side.write_text(f"{hashlib.sha256(data).hexdigest()}  {ff.name}\n")
    with pytest.raises(dl.FreezeError, match="not a format revision 1 / 2 freeze"):
        dl.verify_freeze(ff)


def test_accept_uses_the_dwell_the_freeze_recorded(tmp_path) -> None:
    w = base._world(tmp_path)
    final = w["out"] / MODEL / base.ARM / "final.json"
    doc = json.loads(final.read_text())
    doc["windowing"] = dl.windowing(dwell_windows=3)
    final.write_text(json.dumps(doc))
    assert base._freeze(w) == 0
    entry = dl.verify_freeze(w["freeze"])["models"][MODEL]
    assert entry["windowing"]["dwell_windows"] == 3 and entry["windowing"]["source"] == "final.json"
    assert base._accept(w, base._seal(w)) == dl.EXIT_ACCEPT_FAILED  # 45 / 55 < 0.85: B fails
    res = json.loads(dl.freeze_paths(w["freeze"])["result"].read_text())
    c = res["models"][MODEL]["criteria"]
    # dwell 3: the first two CRITICAL windows of each 11-window violating cell are unconfirmed
    assert c["B"]["criteria"][0]["value"] == pytest.approx(45 / 55) and c["B"]["dwell_windows"] == 3
    assert res["thresholds"]["dwell_windows"] == 3


def test_freeze_refuses_stage_outputs_with_different_windowing(tmp_path, capsys) -> None:
    w = base._world(tmp_path)
    d = w["out"] / MODEL / base.ARM
    for name, win in (("final", dl.windowing()), ("alpha", dl.windowing(dwell_windows=3))):
        doc = json.loads((d / f"{name}.json").read_text())
        doc["windowing"] = win
        (d / f"{name}.json").write_text(json.dumps(doc))
    assert base._freeze(w) == dl.EXIT_REFUSED
    assert "disagree on the windowing" in capsys.readouterr().out


# --------------------------------------------------------- windowing threading


def _cell(n: int, violated_from: int = 99, cell: str = "c"):
    return [CalibrationWindow(cell, "f", 1.0, k < violated_from, window_start_ms=10_000.0 * k) for k in range(n)]


def test_horizon_changes_future_pairs_and_detection_lags() -> None:
    ws = _cell(5, violated_from=4)
    crit = [False, True, False, False, False]
    assert len(dl.future_pairs(ws, crit)) == 2                    # 0 -> 30 s, 10 -> 40 s
    assert len(dl.future_pairs(ws, crit, horizon_ms=20_000)) == 3  # 0, 10, 20 -> 20, 30, 40 s
    # the violation episode starts at 40 s: a CRITICAL at 10 s is 30 s early - counted at
    # the default look-back, missed at 20 s
    assert dl.detection_lags(ws, crit) == [-30.0]
    assert dl.detection_lags(ws, crit, horizon_ms=20_000) == [None]


def test_dt_ref_changes_alpha_and_the_step_response() -> None:
    assert dl.alpha_of(10.0) == dl.alpha_of(10.0, dl.DT_REF_S) == pytest.approx(1 - math.exp(-1))
    assert dl.alpha_of(10.0, 5.0) == pytest.approx(1 - math.exp(-0.5))
    assert dl.step90_s(10.0, 5.0) == 5.0 * math.ceil(math.log(0.1) / math.log(math.exp(-0.5)))
    doc = dl.publish_alpha({"rule": "d4prime", "chosen_tau_s": 10.0, "alpha_fit": {"curve": []}}, 10.0, dt_ref_s=5.0)
    assert doc["published_registry_fields"]["ema_alpha"] == round(1 - math.exp(-0.5), 6)


def test_windowing_validates_and_old_documents_fall_back_to_defaults() -> None:
    assert dl.DEFAULT_WINDOWING == {"window_ms": 30_000.0, "step_ms": 10_000.0, "dt_ref_s": 10.0,
                                    "horizon_ms": 30_000, "dwell_windows": 2}
    assert dl.windowing_of({}) == dl.DEFAULT_WINDOWING
    assert dl.windowing_of({"windowing": {"dwell_windows": 3}})["dwell_windows"] == 3
    with pytest.raises(ValueError):
        dl.windowing(dwell_windows=0)
    with pytest.raises(ValueError):
        dl.windowing(window_ms=-1)


def test_stage_flags_reach_alpha_fit_and_are_recorded(tmp_path, monkeypatch) -> None:
    w = base._world(tmp_path)
    seen = {}

    def fake_run(ns):
        seen["ns"] = ns
        return {"selection": {"chosen_tau_s": 10.0, "chosen_alpha": 0.5}, "curve": []}

    monkeypatch.setattr(af, "run", fake_run)
    argv = ["alpha", "--model", MODEL, "--arm", base.ARM, "--fit-dir", str(w["fit"]), "--out-dir", str(w["out"]),
            "--alpha-w-p", "0.01", "--registry", str(base.REGISTRY), "--window-ms", "20000", "--step-ms", "5000",
            "--dt-ref-s", "5", "--horizon-ms", "20000", "--dwell-windows", "3"]
    assert dl.main(argv) == 0
    ns = seen["ns"]
    assert (ns.window_ms, ns.step_ms, ns.dt_ref_s, ns.dwell_windows) == (20_000.0, 5_000.0, 5.0, 3)
    doc = json.loads((w["out"] / MODEL / base.ARM / "alpha.json").read_text())
    assert doc["windowing"] == {"window_ms": 20_000.0, "step_ms": 5_000.0, "dt_ref_s": 5.0,
                                "horizon_ms": 20_000, "dwell_windows": 3}
    assert doc["published_alpha"] == pytest.approx(1 - math.exp(-0.5))
    # defaults: the module constants, recorded as such
    assert dl.main(argv[:13]) == 0
    assert (seen["ns"].window_ms, seen["ns"].step_ms, seen["ns"].dt_ref_s, seen["ns"].dwell_windows) == (
        30_000.0, 10_000.0, 10.0, 2)
    doc = json.loads((w["out"] / MODEL / base.ARM / "alpha.json").read_text())
    assert doc["windowing"] == dl.DEFAULT_WINDOWING
    assert doc["published_registry_fields"] == {"ema_tau_ms": 10_000.0, "ema_alpha": round(dl.alpha_of(10.0), 6)}


def test_alpha_fit_threads_window_and_episode_margin(monkeypatch) -> None:
    from scripts import theta_verdict as tv
    from tre_common import slo_labels

    got = {}

    def fake_rule(load, fit, crit_fn, **kw):
        crit_fn([], 1.0, 0.5)
        got["rule"] = kw
        return {"selection": {}, "curve": [], "bootstrap": {"chosen_frequency": None}}

    def fake_flags(ws, **kw):
        got["flags"] = kw
        return []

    monkeypatch.setattr(af, "alpha_rule", fake_rule)
    monkeypatch.setattr(tv, "critical_dwell_flags", fake_flags)
    monkeypatch.setattr(slo_labels, "label_def_from_args", lambda a, m: dl.label_for(m, "fixed", str(base.REGISTRY)))
    ns = af._parse(["--model", MODEL, "--fitting-csv", "x.csv", "--w-p", "0", "--lambda-wait", "1", "--output", "o",
                    "--window-ms", "20000", "--episode-margin-ms", "15000", "--dwell-windows", "3"])
    rep = af.run(ns)
    assert got["flags"]["window_ms"] == 20_000.0 and got["flags"]["dwell_windows"] == 3
    assert got["rule"]["margin_ms"] == 15_000.0 and rep["window_ms"] == 20_000.0
    ns = af._parse(["--model", MODEL, "--fitting-csv", "x.csv", "--w-p", "0", "--lambda-wait", "1", "--output", "o"])
    af.run(ns)
    assert got["flags"]["window_ms"] == af.DEFAULT_WINDOW_MS and got["rule"]["margin_ms"] == af.EPISODE_MARGIN_MS


def test_episode_margin_changes_spurious_episodes() -> None:
    starts = [10_000.0 * k for k in range(8)]
    crit = [True, False, False, False, False, False, False, False]
    viol = [False, False, False, False, True, False, False, False]
    # CRITICAL at 0 s, violation at 40 s: non-spurious within +-40 s, spurious at +-30 s
    assert af.cell_stats("c", starts, crit, viol, steady=True).spurious == 1
    assert af.cell_stats("c", starts, crit, viol, steady=True, margin_ms=40_000).spurious == 0


# ------------------------------------------------------------- ranking_report


def test_ranking_report_cli_on_a_freeze(tmp_path, capsys) -> None:
    from scripts import ranking_report

    w = base._world(tmp_path)
    assert base._freeze(w) == 0
    out = tmp_path / "rep"
    rc = ranking_report.main(["--dataset", f"mrun={w['mdir']}", "--freeze-file", str(w["freeze"]),
                              "--split", "holdout", "--role", "ladder", "--resamples", "30", "--out-dir", str(out),
                              "--name", "m"])
    assert rc == 0
    doc = json.loads((out / "m.json").read_text())
    assert doc["gating"] is False and doc["filters"]["split"] == ["holdout"]
    # holdout + valid + ladder: the 12 M cells (the sealed probe is role boundary, the
    # training cell split train)
    blk = doc["models"][MODEL]
    assert blk["cells"] == 12 and doc["inputs"][0]["rows_kept"] == {MODEL: 12 * 12}
    assert blk["auroc"]["value"] == 1.0
    assert doc["pooled"]["kendall_tau_b_cross_model"]["exact"]["value"] is None
    md = (out / "m.md").read_text()
    assert "| dsqwen-7b | 132 | 12 |" in md and "Definitions" in md
    # explicit parameters give the same numbers
    vh = dl.verify_freeze(w["freeze"])["models"][MODEL]["verdict_for_holdout"]
    params = {MODEL: {"theta": vh["published"]["theta_m"], "w_p": 0.0, "lambda_wait": 1.0, "tau_s": 10.0,
                      "label_def": vh["label_def"]}}
    (tmp_path / "p.json").write_text(json.dumps(params))
    assert ranking_report.main(["--dataset", f"mrun={w['mdir']}", "--params-json", str(tmp_path / "p.json"),
                                "--split", "holdout", "--role", "ladder", "--resamples", "30", "--backend", "python",
                                "--out-dir", str(out), "--name", "p"]) == 0
    doc2 = json.loads((out / "p.json").read_text())
    assert doc2["models"] == doc["models"] and doc2["backend"] == "python"
