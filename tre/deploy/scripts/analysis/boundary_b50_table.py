#!/usr/bin/env python3
"""The D6' b50 table of a second-round ladder: per (model, shape) the load factor at which
half of the labelled windows violate, from a logistic fit in log(rho / rho*_run2).

This is the ``--boundary-table`` the acceptance set M (``calibration_acceptance``) and the
T14 capacity prior (``calibration_t14 capacity-prior``) read: ``boundary_rates`` takes
``P_b50_rf x rho*_run2`` as a training shape's D6' boundary and checks ``rho_star_fixed``
against the base run's anchor.

Input: one or more window CSVs (the standard dataset's ``windows.csv`` of the second round,
or a training table built from it), and the second round's run root for the anchors
(``<base-run>/<model>/design_result.json``, ``models[0].anchors``). Rows kept: not in the
warm-up, role in ``--roles``, split in ``--splits`` (when the CSV has a ``split`` column),
``run`` equal to ``--run`` (only when given; the 2026-09-23 training table mixed two rounds
under that column). Per arm - ``P`` the primary (D6') label ``slo_label``, ``F`` the fixed
one ``slo_label_fixed`` - unlabelled windows are dropped and ``y = violated``.

Fit: maximum likelihood of ``p = 1 / (1 + exp(-(a + b log rf)))`` (Nelder-Mead from
``(0, 5)``); ``b50 = exp(-a / b)``, empty when the slope is not clearly positive
(``b <= 0.5``: no transition inside the data). Plus the violated fraction in the bands
rf < 0.9, 0.9 <= rf < 1.1, rf >= 1.1.

    python -m scripts.analysis.boundary_b50_table --windows <run2>/dataset/windows.csv \\
        --base-run <run2 root> --out <table.csv>

Columns: model, shape, rho_star_fixed, rf_max, n, then per arm ``{P,F}_b50_rf``,
``{P,F}_slope``, ``{P,F}_v[0,0.9)``, ``{P,F}_v[0.9,1.1)``, ``{P,F}_v[1.1,9)``.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional, Sequence

DEFAULT_ROLES = ("ladder", "adaptive", "sentinel", "boundary")
DEFAULT_SPLITS = ("train", "auxiliary")
ARMS = (("P", "slo_label"), ("F", "slo_label_fixed"))
BANDS = ((0, 0.9), (0.9, 1.1), (1.1, 9))
#: A fitted slope at or below this has no transition inside the data: no b50.
MIN_SLOPE = 0.5


def fit50(x, y) -> tuple[Optional[float], float]:
    """(b50 in rf units or None, slope) of the logistic in log rf."""
    import numpy as np
    from scipy.optimize import minimize

    lx = np.log(x)

    def nll(p):
        z = p[0] + p[1] * lx
        q = np.clip(1 / (1 + np.exp(-z)), 1e-6, 1 - 1e-6)
        return -(y * np.log(q) + (1 - y) * np.log(1 - q)).sum()

    a, b = minimize(nll, [0, 5], method="Nelder-Mead").x
    return (float(np.exp(-a / b)) if b > MIN_SLOPE else None), float(b)


def load_anchors(base_run: Path, models: Sequence[str]) -> dict:
    out = {}
    for model in models:
        path = Path(base_run) / model / "design_result.json"
        doc = json.loads(path.read_text(encoding="utf-8"))
        out[model] = doc["models"][0]["anchors"]
    return out


def read_windows(paths: Sequence[Path], *, roles: Sequence[str], splits: Sequence[str],
                 run: Optional[str]):
    import pandas as pd

    frames = [pd.read_csv(p, low_memory=False, dtype={"in_warmup": str}) for p in paths]
    t = pd.concat(frames, ignore_index=True)
    keep = (t.in_warmup.str.lower() != "true") & t.role.isin(list(roles))
    if run is not None:
        if "run" not in t.columns:
            raise ValueError("--run given but the windows have no 'run' column")
        keep &= t.run == run
    if splits and "split" in t.columns:
        keep &= t.split.isin(list(splits))
    return t[keep]


def b50_table(t, anchors: dict):
    import pandas as pd

    rows = []
    for (model, shape), g in t.groupby(["model", "shape"]):
        a = anchors[model][shape]
        g = g.copy()
        g["rf"] = g.rho / a
        out = {"model": model, "shape": shape, "rho_star_fixed": round(a, 3),
               "rf_max": round(g.rf.max(), 2), "n": len(g)}
        for arm, col in ARMS:
            gg = g[g[col] != "unlabeled"]
            y = (gg[col] == "violated").astype(float).values
            r50, b = fit50(gg.rf.values, y)
            out[f"{arm}_b50_rf"] = None if r50 is None else round(r50, 3)
            out[f"{arm}_slope"] = round(b, 1)
            for lo, hi in BANDS:
                h = gg[(gg.rf >= lo) & (gg.rf < hi)]
                out[f"{arm}_v[{lo},{hi})"] = None if len(h) == 0 else round((h[col] == "violated").mean(), 2)
        rows.append(out)
    return pd.DataFrame(rows)


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--windows", type=Path, action="append", required=True,
                    help="window CSV of the second round (repeatable)")
    ap.add_argument("--base-run", type=Path, required=True,
                    help="the second round's run root (<model>/design_result.json anchors)")
    ap.add_argument("--out", type=Path, required=True, help="the table (CSV); refuses to overwrite")
    ap.add_argument("--roles", default=",".join(DEFAULT_ROLES))
    ap.add_argument("--splits", default=",".join(DEFAULT_SPLITS),
                    help="kept splits when the CSV has a split column ('' keeps all)")
    ap.add_argument("--run", default=None, help="keep only rows whose 'run' column is this")
    args = ap.parse_args(argv)
    if args.out.exists():
        ap.error(f"{args.out} exists; write the table to a new file")
    t = read_windows(args.windows, roles=[r for r in args.roles.split(",") if r],
                     splits=[s for s in args.splits.split(",") if s], run=args.run)
    if t.empty:
        ap.error("no windows left after the filters")
    anchors = load_anchors(args.base_run, sorted(t.model.unique()))
    table = b50_table(t, anchors)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.out, index=False)
    print(table.to_string())
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
