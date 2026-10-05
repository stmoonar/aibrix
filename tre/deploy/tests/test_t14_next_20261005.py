"""T14 next round (2026-10-05): the scorer never mixes label attributions, and the sealed
cross-shape claim rule (D1 sample SD, D2 single-class shape) on a per-shape table."""
from __future__ import annotations

import json
import statistics
from pathlib import Path

from scripts.analysis import t14_score as t14


def test_the_scorer_refuses_mixed_label_attributions_and_writes_nothing(tmp_path, monkeypatch, capsys) -> None:
    # a dataset built under completion (manifest without the attribution key) ...
    ds = tmp_path / "run" / "dataset"
    ds.mkdir(parents=True)
    (ds / "manifest.json").write_text(json.dumps({"run_root": str(tmp_path / "run")}))
    (ds / "windows.csv").write_text("model\n")
    # ... scored against parameters frozen under the hybrid label
    monkeypatch.setattr(t14, "check_inputs", lambda *a: {
        "schema": "v1", "model": "dsqwen-14b", "freeze_attribution": t14.ATTRIBUTION_HYBRID, "prereg": {}})
    out = tmp_path / "score.json"
    rc = t14.main(["--prereg", "p.json", "--freeze-file", "f.json", "--dry-run-dataset", str(ds), "--out", str(out)])
    text = capsys.readouterr().out
    assert rc == 1 and "REFUSED" in text and "mixed label attributions" in text
    assert not out.exists() and not Path(f"{out}.d").exists()

    # the same dataset relabelled hybrid agrees with the freeze
    (ds / "manifest.json").write_text(json.dumps({"attribution": {"value": "hybrid", "rule": "r"}}))
    assert t14.dataset_attribution(ds) == t14.ATTRIBUTION_HYBRID
    one, problems = t14.check_attributions({"freeze": "hybrid", "dataset": t14.dataset_attribution(ds),
                                            "prereg": "hybrid"})
    assert (one, problems) == ("hybrid", [])
    # an unknown attribution is refused, not treated as completion
    assert t14.check_attributions({"freeze": "hybrid", "dataset": "first_token"})[1]
    # a real run needs a dry run under the same attribution (a v1 dry-run record = completion)
    rec = tmp_path / "dry.json"
    rec.write_text(json.dumps({"dry_run": True, "code": {"commit": "c"}, "scorer": {"sha256": "s"},
                               "prereg": {"sha256": "p"}}))
    code = {"commit": "c", "dirty": False}
    assert t14.check_dry_run_record(rec, "s", code, prereg_sha256="p", attribution="completion") == []
    assert any("mixed label attributions" in p
               for p in t14.check_dry_run_record(rec, "s", code, prereg_sha256="p", attribution="hybrid"))


def _shape(ba, half, single=False):
    if single:
        return {"single_class": True, "ba": None, "ba_ci95_half_width": None}
    return {"single_class": False, "ba": ba, "ba_ci95_half_width": half}


def test_the_cross_shape_claim_per_kind_on_a_per_shape_table() -> None:
    interpolation = {"G512x256": _shape(0.85, 0.05), "G1200x240": _shape(0.86, 0.06),
                     "G640x400": _shape(0.84, 0.04), "G1800x160": _shape(0.85, 0.05)}
    extrapolation = {"G3072x96": _shape(0.95, 0.03), "G4096x64": _shape(0.70, 0.03),
                     "G256x768": _shape(0.90, 0.03), "G512x1024": _shape(0.60, 0.03)}
    ci = t14.claim(interpolation)
    assert ci["one_theta_transfers"] is True                 # SD .008 <= median half width .05
    assert abs(ci["sd_sample"] - statistics.stdev([0.85, 0.86, 0.84, 0.85])) < 1e-12
    ce = t14.claim(extrapolation)                            # stated separately: extrapolation does not transfer
    assert ce["one_theta_transfers"] is False and ce["sd_sample"] > ce["median_ci95_half_width"]

    # D1: the sample SD (n - 1) decides; the population SD would claim, and is disclosure only
    two = {"A": _shape(0.80, 0.045), "B": _shape(0.90, 0.045)}   # pstdev .05 > .045; stdev .0707
    near = {"A": _shape(0.80, 0.06), "B": _shape(0.90, 0.06)}    # pstdev .05 <= .06 < stdev .0707
    assert t14.claim(two)["one_theta_transfers"] is False
    cn = t14.claim(near)
    assert cn["one_theta_transfers"] is False and cn["disclosure_population_sd"]["would_claim"] is True

    # D2: a single-class shape makes its kind not evaluable - never dropped to claim on the rest
    with_single = {**interpolation, "G1800x160": _shape(None, None, single=True)}
    cs = t14.claim(with_single)
    assert cs["one_theta_transfers"] == t14.CLAIM_NOT_EVALUABLE
    assert cs["single_class_shapes"] == ["G1800x160"] and cs["shapes"] == 4
    rest = cs["disclosure_only_over_remaining_shapes"]
    assert rest["shapes"] == 3 and rest["sd_sample"] is not None and "sd_sample" not in cs
    # the other kind is judged on its own table, unaffected
    assert t14.claim(extrapolation)["one_theta_transfers"] is False
