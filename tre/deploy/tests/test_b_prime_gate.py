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
    text = OVERLAY.read_text(encoding="utf-8")
    m = re.search(r"name:\s*TRE_DWELL_WINDOWS\s*\n\s*value:\s*\"?(\d+)\"?", text)
    assert m, "TRE_DWELL_WINDOWS not found in the controller overlay"
    assert int(m.group(1)) == dl.ONLINE_DWELL_WINDOWS


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
