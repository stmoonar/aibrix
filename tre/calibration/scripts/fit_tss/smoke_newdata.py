#!/usr/bin/env python3
"""smoke_newdata.py (was test_newdata.py) - smoke test of the NEW-source code paths of refit_supp.py (they are not exercised by the old roots):
--data NEW=ROOT with ROOT/<src>/<model>/dataset_hybrid (several sources, names NEW.A / NEW.B), no manifest, default
ladder sources (run2/run2b + added), a low-load 'A' ladder extension of S1 (fake: run2 S1 holds with rho x 0.5), a new
static shape 'G768x192' without rho_factor (fake: run2 S2 holds renamed), criterion 3 on the new shape.
The fake rows are COPIES of old training cells (numbers meaningless); written under --work and removed afterwards.
Usage: TRE_FIT_DATA_ROOT=<data root> nice -n 19 python3 -B smoke_newdata.py --work DIR"""
import argparse
import os
import shutil
import sys

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import pandas as pd  # noqa: E402
import supp_lib as L  # noqa: E402
import refit_supp as RS  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--work", required=True, help="scratch directory (the fake root and the fit output)")
args = ap.parse_args()
if 10000 not in L.OLD_ROOTS:
    raise SystemExit("set TRE_FIT_DATA_ROOT (the old 10 s training root is copied from)")
M = "dsqwen-7b"
T = os.path.abspath(args.work)
ROOT = T + "/fake_new_root"
src_dir = L.OLD_ROOTS[10000] + "/run2/" + M + "/dataset_hybrid"
w = pd.read_csv(src_dir + "/windows.csv", low_memory=False)
q = pd.read_csv(src_dir + "/requests.csv", low_memory=False)
tr = w[(w.split == "train") & (w.primitive == "hold") & w.rho_factor.notna()]
a_cells = tr[(tr["shape"] == "S1")].groupby("cell_id").rho_factor.first().sort_values().index[:6]
b_cells = tr[(tr["shape"] == "S2")].cell_id.unique()[:12]
for name, cells, fn in (
        ("A", a_cells, lambda d: d.assign(rho=d.rho * 0.5, rho_factor=d.rho_factor * 0.5)),
        ("B", b_cells, lambda d: d.assign(shape="G768x192", primitive="static", rho_factor=float("nan")))):
    d = f"{ROOT}/{name}/{M}/dataset_hybrid"
    os.makedirs(d, exist_ok=True)
    fn(w[w.cell_id.isin(cells)]).to_csv(d + "/windows.csv", index=False)
    fn(q[q.cell_id.isin(cells)]).to_csv(d + "/requests.csv", index=False)
out = T + "/newdata_smoketest"
res = RS.main(["--out", out, "--models", M, "--variant", "primary", "--nboot", "20", "--no-hash",
               "--data", "old", "--data", "NEW=" + ROOT])
x = res["primary"][M]
print("sources:", x["sources"], "| ladder sources:", x["ladder_sources"])
print("ladder notes:", x["ladder_notes"])
print("c3:", {k: v for k, v in x["criteria"]["c3_new_shapes_p_at_theta"].items() if k != "rule"})
print("c4:", {k: v for k, v in x["criteria"]["c4_tau_high_interpolated"].items() if k != "rule"})
for row in x["per_shape"]:
    if row["shape"] in ("S1", "G768x192"):
        print(row["shape"], {k: row.get(k) for k in ("class", "tau_high_n3", "in_ladder_n3", "ladder_n_rho_levels",
                                                    "ladder_rho_min", "ladder_rho_max", "ladder_cells_by_source")})
ok = ("NEW.A" in x["ladder_sources"] and "NEW.B" in x["ladder_sources"]
      and "G768x192" in x["criteria"]["c3_new_shapes_p_at_theta"]["shapes"]
      and any("NEW.B: no rho_factor" in n for n in x["ladder_notes"]))
shutil.rmtree(ROOT)
print("RESULT:", "OK" if ok else "FAIL")
