"""M2 forking-path disclosure (scripts.analysis.m2_disclosure): side-by-side variants, never
the verdict; a hold-only freeze younger than the M2 manifest is refused."""
from __future__ import annotations

import json
import os
import shutil

import test_dline_freeze_accept as base  # the D22 freeze / accept fixture world
from scripts import dline_refit as dl
from scripts.analysis import m2_disclosure as md

MODEL = base.MODEL


def test_disclosure_scores_the_variants_and_refuses_a_hold_only_freeze_made_after_m2(tmp_path) -> None:
    w = base._world(tmp_path, steps_cells=16)
    assert base._freeze(w, "onset") == 0
    hold = tmp_path / "hold" / "params_freeze.json"          # a second freeze of the same (hold-only) fit
    hold.parent.mkdir()
    assert dl.main(["freeze", "--model", MODEL, "--arm", base.ARM, "--fit-dir", str(w["fit"]), "--out-dir",
                    str(w["out"]), "--freeze-file", str(hold)]) == 0
    man = base._seal(w)
    ds = f"m={w['mdir']}"
    doc = md.disclose(w["freeze"], hold, [ds], [ds], [str(man)], tmp_path / "work", n_resamples=50, seed=dl.SEED)
    r = doc["models"][MODEL]
    assert doc["changes_the_verdict"] is False and "not acceptance" in doc["what"]
    assert set(r) == {"frozen_hybrid_reference", "completion_label", "holdonly_theta"}
    assert r["frozen_hybrid_reference"]["verdict"] == dl.VERDICT_PASS
    # completion data under a completion freeze: (a) is the reference itself
    assert r["completion_label"]["verdict"] == r["frozen_hybrid_reference"]["verdict"]
    assert not base._written(w)[2:]                           # accept's outputs untouched
    # an M2 manifest written BEFORE the hold-only freeze: (b) could have been tuned on M2
    m = json.loads(man.read_text())
    m["written_at_utc"] = "2000-01-01T00:00:00+00:00"
    os.chmod(man, 0o644)
    man.write_text(json.dumps(m))
    shutil.rmtree(tmp_path / "work")
    try:
        md.disclose(w["freeze"], hold, [ds], [ds], [str(man)], tmp_path / "work", n_resamples=50, seed=dl.SEED)
    except md.Refused as exc:
        assert any("not older than the M2 manifest" in p for p in exc.problems)
    else:
        raise AssertionError("a hold-only freeze younger than M2 was accepted")
