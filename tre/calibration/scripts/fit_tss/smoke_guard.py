#!/usr/bin/env python3
"""smoke_guard.py (was test_guard.py) - the sealed-set guards of the fit tools, WITHOUT opening M or E (string tests +
refusals before I/O).  Needs the real data (TRE_FIT_DATA_ROOT: the old roots, for sections 3 and 4).
 1. sealed_reason on path strings (no file-system access);
 2. refit_supp.py / validate_supp.py (dev) refuse M / E roots before reading;
 3. ModelData refuses an unknown shape (a validation mixture) and a held-out shape with a training split;
 4. final-mode code path + one-time ledger, exercised on TRAINING rows only (--final-splits train,auxiliary).
Usage: TRE_FIT_DATA_ROOT=<data root> nice -n 19 python3 -B smoke_guard.py --work DIR --candidate CAND
       (writes only under --work; CAND = a candidate.json of refit_supp.py, e.g. run_before.sh's before/primary/)"""
import argparse
import os
import shutil
import subprocess
import sys

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import pandas as pd  # noqa: E402
import supp_lib as L  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--work", required=True, help="scratch directory (fake sources, ledger, outputs)")
ap.add_argument("--candidate", required=True, help="a candidate.json of refit_supp.py")
args = ap.parse_args()
if not L.DATA_ROOT:
    raise SystemExit("set TRE_FIT_DATA_ROOT (the data root of the sealed / old roots)")
X = L.DATA_ROOT
SEALED = [X + "/calib_20261003/M", X + "/calib_20261003/M/dsqwen-7b/dataset_hybrid",
          X + "/calib_20261003/next-20261005/M2/dsqwen-7b", X + "/calib_supp_20261103/E",
          X + "/calib_supp_20261103/E/dsqwen-7b/dataset_hybrid", X + "/calib_supp_20261103/datasets/w10000/E",
          X + "/calib_supp_20261103/datasets/w10000/E/dsqwen-7b/dataset_hybrid", X + "/calib_supp_20261103/Emix",
          X + "/calib_supp_20261103/datasets/E_sealed", X + "/somewhere/M/dsqwen-7b/dataset_hybrid",
          X + "/somewhere/holdout_set", X + "/datasets-w10000/validation/dsqwen-7b", X + "/x/E-2026/dsqwen-7b"]
OPEN = [L.OLD_ROOTS[10000] + "/run2/dsqwen-7b/dataset_hybrid", L.OLD_ROOTS[20000] + "/run2b/dsqwen-14b/dataset_hybrid",
        X + "/calib_supp_20261103/A", X + "/calib_supp_20261103/B/dsqwen-7b",
        X + "/calib_supp_20261103/datasets/w10000/A/dsqwen-7b/dataset_hybrid",
        X + "/calib_supp_20261103/datasets-w20000/B", X + "/calib_20261003/run2/dsqwen-7b", os.path.abspath(args.work) + "/fake"]
fails = []
for p in SEALED:
    if L.sealed_reason(p) is None:
        fails.append(f"NOT refused (should be): {p}")
for p in OPEN:
    if L.sealed_reason(p) is not None:
        fails.append(f"refused (should not be): {p}: {L.sealed_reason(p)}")
print(f"1. string guard: {len(SEALED)} sealed, {len(OPEN)} open paths checked; failures {len(fails)}")

env = dict(os.environ, OMP_NUM_THREADS="1", PYTHONDONTWRITEBYTECODE="1")
cand = os.path.abspath(args.candidate)
T = os.path.abspath(args.work)
os.makedirs(T, exist_ok=True)


def run(args):
    p = subprocess.run([sys.executable, "-B"] + args, capture_output=True, text=True, env=env, cwd=HERE)
    return p.returncode, (p.stdout + p.stderr)


cases = [["refit_supp.py", "--out", T + "/x", "--models", "dsqwen-7b", "--data", "old", "--data", "M=" + X + "/calib_20261003/M"],
         ["refit_supp.py", "--out", T + "/x", "--models", "dsqwen-7b", "--data", "E=" + X + "/calib_supp_20261103/datasets/w10000/E"],
         ["validate_supp.py", "--candidate", cand, "--models", "dsqwen-7b", "--out", T + "/x", "--data", "M=" + X + "/calib_20261003/M"],
         ["validate_supp.py", "--candidate", cand, "--models", "dsqwen-7b", "--out", T + "/x", "--data", X + "/calib_supp_20261103/E"]]
for c in cases:
    rc, out = run(c)
    ok = rc != 0 and "REFUSED" in out
    print(f"2. {'ok ' if ok else 'FAIL'} refused before reading: {' '.join(c[:1] + c[-2:])} -> rc {rc}: "
          f"{[ln for ln in out.splitlines() if 'REFUSED' in ln][:1]}")
    if not ok:
        fails.append(f"not refused: {c}")

# 3. fake new sources built from a few old training rows (written under tests/, removed at the end)
src_dir = L.OLD_ROOTS[10000] + "/run2/dsqwen-7b/dataset_hybrid"
w = pd.read_csv(src_dir + "/windows.csv", low_memory=False)
w = w[(w.split == "train") & (w["shape"] == "S1")]
keys = w.cell_id.unique()[:2]
w = w[w.cell_id.isin(keys)]
q = pd.read_csv(src_dir + "/requests.csv", low_memory=False)
q = q[q.cell_id.isin(keys)]
for name, shape in (("fake_unknown", "EMIX1"), ("fake_ho", "G800x240")):
    d = f"{T}/{name}/dsqwen-7b/dataset_hybrid"
    os.makedirs(d, exist_ok=True)
    w.assign(shape=shape).to_csv(d + "/windows.csv", index=False)
    q.assign(shape=shape).to_csv(d + "/requests.csv", index=False)
    try:
        L.ModelData("dsqwen-7b", 10000, L.resolve_specs([f"{name}={T}/{name}"], "dsqwen-7b", 10000), L.TRAIN_SPLITS,
                    hash_files=False)
        fails.append(f"ModelData accepted shape {shape}")
        print(f"3. FAIL ModelData accepted shape {shape}")
    except SystemExit as e:
        print(f"3. ok  ModelData refused shape {shape}: {str(e)[:110]}")
    shutil.rmtree(f"{T}/{name}")

# 4. final-mode path + ledger on TRAINING rows only (no sealed data), then the ledger refuses a repeat
led = T + "/ledger_smoketest.jsonl"
if os.path.exists(led):
    os.remove(led)
fin = ["validate_supp.py", "--candidate", cand, "--models", "dsqwen-7b", "--data", "old", "--final-splits",
       "train,auxiliary", "--i-am-the-final-evaluation", "--set-name", "E", "--ledger", led, "--no-hash",
       "--out", T + "/final_mode_smoketest_on_training_rows"]
rc1, out1 = run(fin)
rc2, out2 = run(fin)
ok = rc1 == 0 and rc2 != 0 and "already evaluated" in out2
print(f"4. {'ok ' if ok else 'FAIL'} final-mode smoke test: first rc {rc1}, repeat rc {rc2} "
      f"({[ln for ln in out2.splitlines() if 'already' in ln][:1]})")
if not ok:
    fails.append("final-mode ledger")
    print(out1[-2000:])
print("RESULT:", "ALL OK" if not fails else fails)
