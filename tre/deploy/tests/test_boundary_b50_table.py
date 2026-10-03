"""``scripts.analysis.boundary_b50_table``: the D6' b50 table M and the T14 capacity prior
read. Pins: b50 sits at the label flip in rho*_run2 units, the warm-up / held-out rows are
not used, an arm without a transition has no b50, and the table never overwrites."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

pytest.importorskip("pandas")
pytest.importorskip("scipy")

from scripts.analysis import boundary_b50_table as b50  # noqa: E402

MODEL, SHAPE, ANCHOR = "dsqwen-14b", "S2", 2.0


def _inputs(tmp_path: Path) -> tuple[Path, Path]:
    base = tmp_path / "run2"
    (base / MODEL).mkdir(parents=True)
    (base / MODEL / "design_result.json").write_text(json.dumps(
        {"models": [{"model": MODEL, "anchors": {SHAPE: ANCHOR}}]}), encoding="utf-8")
    rows = []
    for i in range(40):
        rf = 0.7 + 0.02 * i                       # 0.70 .. 1.48
        p = "violated" if rf >= 1.2 else "healthy"
        rows.append({"model": MODEL, "shape": SHAPE, "rho": rf * ANCHOR, "role": "ladder",
                     "split": "train", "in_warmup": "False", "slo_label": p,
                     "slo_label_fixed": "violated" if rf < 0.9 else "healthy"})
    # neither a warm-up window nor a held-out row may move the boundary
    rows.append({**rows[0], "in_warmup": "True", "slo_label": "violated"})
    rows.append({**rows[1], "split": "holdout", "slo_label": "violated"})
    windows = tmp_path / "windows.csv"
    with open(windows, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    return windows, base


def test_b50_is_the_label_flip_in_rho_star_units(tmp_path) -> None:
    windows, base = _inputs(tmp_path)
    out = tmp_path / "table.csv"
    assert b50.main(["--windows", str(windows), "--base-run", str(base), "--out", str(out)]) == 0
    (row,) = list(csv.DictReader(open(out, newline="", encoding="utf-8")))
    assert (row["model"], row["shape"], float(row["rho_star_fixed"])) == (MODEL, SHAPE, ANCHOR)
    assert int(row["n"]) == 40
    assert float(row["P_b50_rf"]) == pytest.approx(1.19, abs=0.02)
    assert float(row["P_v[0,0.9)"]) == 0.0 and float(row["P_v[1.1,9)"]) > 0.5
    assert row["F_b50_rf"] == ""              # violated only below: no rising transition
    with pytest.raises(SystemExit):           # never overwrites
        b50.main(["--windows", str(windows), "--base-run", str(base), "--out", str(out)])
