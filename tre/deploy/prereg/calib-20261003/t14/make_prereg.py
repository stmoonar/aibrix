# T14 preregistration for the 2026-10-03 round (written before any T14 / M data exists).
# Schema of calibration_v1lambda_20260924/preregistration.json; no amendment.
import glob, hashlib, json, os, stat, subprocess, sys
from datetime import datetime, timezone
from pathlib import Path

C = Path("/data/nfs_shared_data/xxy/calib_20261003")
OUT = C / "t14" / "preregistration.json"
if OUT.exists():
    sys.exit(f"{OUT} exists: a preregistration is written once")
WT = Path("/data/nfs_shared_data/xxy/aibrix-wt/calib-theta-20261003")
FREEZE = C / "freeze" / "params_freeze.json"
PRIOR = C / "t14" / "capacity_prior_dsqwen-14b.json"
B50 = C / "b50.csv"
DRY = Path("/tmp/calibG-t14-dryrun-20261004/dsqwen-14b/plan.json")
SEED = 20261004
GATEWAY = "http://192.168.223.76:31094/v1/chat/completions"
ENGINE_IMAGE = "vllm-openai-tre:0.30.0-ts-8dc0f2a7"


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


plan = json.load(open(DRY))
prior = json.load(open(PRIOR))
fz = json.load(open(FREEZE))
fm = fz["models"]["dsqwen-14b"]
cells = {c["cell_id"]: c for c in plan["static_cells"]["dsqwen-14b"]}
head = subprocess.check_output(["git", "-C", str(WT), "rev-parse", "HEAD"]).decode().strip()
dirty = subprocess.check_output(["git", "-C", str(WT), "status", "--short"]).decode().strip()
interp = ["G512x256", "G1200x240", "G640x400", "G1800x160"]
extrap = ["G3072x96", "G4096x64", "G256x768", "G512x1024"]
rows = []
for i, cid in enumerate(plan["order"], 1):
    c = cells[cid]
    cap = prior["predicted_rps"][c["shape"]]
    rows.append({"order": i, "cell_id": cid, "shape": c["shape"],
                 "kind": "interpolation" if c["shape"] in interp else "extrapolation",
                 "factor": c["rho"], "rps": round(c["rho"] * cap, 4), "hold_s": c["duration_s"],
                 "warmup_s": c["warmup_s"], "arrival_seed": c["arrival_seed"], "prompt_key": c["prompt_key"],
                 "split": c["split"], "role": c["role"]})

# seed / id disjointness against every ledger of this round and of the earlier calibration roots
seeds, own_ids, other_ids = set(), set(), set()
ledgers = glob.glob(str(C / "**" / "cells.jsonl"), recursive=True) + \
    glob.glob("/data/nfs_shared_data/xxy/calibration_*/**/cells.jsonl", recursive=True)
for f in ledgers:
    for line in open(f):
        try:
            d = json.loads(line)
        except ValueError:
            continue
        if d.get("arrival_seed") is not None:
            seeds.add(int(d["arrival_seed"]))
        if d.get("cell_id"):
            (own_ids if f.startswith(str(C)) else other_ids).add(d["cell_id"])
t14_seeds = {r["arrival_seed"] for r in rows}
t14_ids = {r["cell_id"] for r in rows}
roots = [str(C / r) for r in ("run1", "run2", "run2b", "supp", "p1", "p1r2", "fit", "freeze", "M", "cb_idle", "prereg")]

doc = {
    "what": ("T14 degraded collection (dsqwen-14b, 24 hold cells), round 2026-10-03 (vLLM 0.30) - preregistration: "
             "design, capacity prior, B-prime criterion, evaluation rule. Written before any T14 or M data of this round "
             "exists; the collection refuses to start unless this file matches its sha256 sidecar and the bound keys "
             "(calibration_t14.check_preregistration)."),
    "written_at_utc": datetime.now(timezone.utc).isoformat(),
    "written_by": ("main session (Claude) on the owner instruction to launch stage G (2026-10-04); schema and rules of "
                   "the 2026-09-24 T14 preregistration; owner review pending"),
    "code": {"branch": "calib/theta-20261003", "commit": head, "dirty": bool(dirty),
             "modules": ["deploy/scripts/calibration_t14.py", "deploy/scripts/dline_refit.py (accept, B-prime)",
                         "deploy/scripts/b_prime.py"]},
    "t14": {
        "model": "dsqwen-14b", "api": "chat",
        "shapes": {"interpolation": interp, "extrapolation": extrap},
        "factors": [0.9, 1.0, 1.1], "hold_s": 240.0, "warmup_dropped_s": 60.0, "cells_total": 24,
        "design_seed": SEED, "cell_serial_base": 80500, "prompt_key_prefix": f"p{SEED}.",
        "arrival_seed_rule": f"calibration_design.derived_seed({SEED}, 'dsqwen-14b', cell_id, 'arrivals')",
        "order_rule": f"shapes interleaved, derived_seed({SEED}, 'dsqwen-14b', 't14-order')",
        "cells": rows,
        "seed_disjointness": {
            "checked_against": f"every cells.jsonl under {C} and /data/nfs_shared_data/xxy/calibration_*/",
            "ledgers": len(ledgers), "existing_arrival_seeds": len(seeds),
            "t14_arrival_seed_overlap": len(t14_seeds & seeds),
            "t14_cell_id_overlap_this_round": len(t14_ids & own_ids),
            "t14_cell_id_overlap_earlier_roots": len(t14_ids & other_ids),
            "note": ("cell ids are the fixed T14 serials (base 80500), shared with the 2026-09-24 design by construction; "
                     "arrival seeds and prompt keys follow the design seed"),
        },
        "capacity_prior": {
            "path": str(PRIOR), "sha256": sha(PRIOR), "form": prior["form"], "coefficients": prior["coefficients"],
            "fit_points": ("the 7 training shapes only (D6' boundary rates: b50 x rho*_run2 x C_s, run2 roots per model "
                           "with run2b for 14b; S3 from the boundary supplement) - no M, no H2, no T14"),
            "boundary_table": {"path": str(B50), "sha256": sha(B50)},
            "leave_one_out": prior["leave_one_out"], "predicted_rps": prior["predicted_rps"],
        },
        "held_out_assertion": "calibration_t14 refuses unless gen.is_held_out(shape) and split == holdout for every cell (dry run included)",
        "max_model_len_check": plan.get("max_model_len_check"),
        "gateway": {"url": GATEWAY, "headers": {"model": "dsqwen-14b", "routing-strategy": "least-gpu-cache"},
                    "known_difference": ("training / M use the per-model HTTPRoute path; single replica, so pod choice is "
                                         "identical and only the ext_proc per-request overhead differs")},
        "engine_image": ENGINE_IMAGE,
        "conditions": {
            "controller_mode": "observe", "sm_mode": "observe", "routable_replicas": 1,
            "routable_pod_node": "nscc-ds-4a100-node9 gpu 2-3",
            "other_load": "M of dsqwen-7b / dsllama-8b may still run (other GPUs); no other 14b load",
            "label": ("D6' primary (slowdown TTFT k=5 floor 500 ms over the 2026-10-03 idle c/b, TPOT 75 ms, "
                      "min 20 completions, unserved = violated)"),
            "label_def_sha256": fm["label_def_sha256"],
            "prompt_corpus": fm.get("prompt_corpus"),
            "windows": "30 s windows on the 10 s grid, first 60 s of each cell dropped, ramp trim 1 (as the training fit)",
        },
        "no_adaptive_points": True,
        "void_rule": ("a void cell is re-driven once; a second void stops the run; a stopped run is never evaluated - "
                      "re-run all 24 cells into a new root"),
        "forbidden_roots": roots,
        "expected_wall_clock_h": {"expected": 2.15, "upper": 2.88},
    },
    "parameter_sets": {
        "freeze": {"path": str(FREEZE), "sha256": sha(FREEZE), "freeze_sha256": fz["freeze_sha256"],
                   "dsqwen-14b": fm["published"],
                   "how": ("gateway numerator; lambda and w_p from the v1 selection (dline_refit wp --lambda-method v1); "
                           "tau 10 s (D18); final --no-holdout; owner choice 2026-10-04")},
        "v1lambda": {"path": str(FREEZE), "sha256": sha(FREEZE), "freeze_sha256": fz["freeze_sha256"],
                     "note": ("ONE frozen set this round: the campaign needs two parameter files, so the freeze is named "
                              "twice (RUN plan G2). There is no second set and no replacement rule.")},
    },
    "evaluation": {
        "times": "once on T14, with the freeze parameters; nothing is tuned on T14",
        "data": "the sealed T14 dataset (T14_manifest.json cells only, the valid attempt), dsqwen-14b, gateway-numerator dataset/",
        "A": {"metric": "window balanced accuracy at the published theta (no dwell)",
              "ci": "cell bootstrap, 1000 resamples, seed 20260922, 95 % percentile",
              "gates_reported": {"ba_min": 0.80, "ba_ci95_low_min": 0.75, "max_drop_from_training_ba": 0.08}},
        "B_prime": {"cut": "the freeze's sealed b_prime.severity_cut (training .65 quantile)",
                    "dsqwen-14b_cut": fm["b_prime"]["severity_cut"], "gate": fz.get("b_prime_gate"),
                    "dwell": "judged at dwell 1 (the controller's TRE_DWELL_WINDOWS); dwell 2 disclosed"},
        "also_reported": ["AUROC of Z vs SLO label", "Kendall tau-b ranking disclosure", "C: TTFT-only recall (disclosure)",
                          "per-shape BA, interpolation and extrapolation separately"],
        "outcome_statements": {"pass": "A and B-prime gates all met", "fail": "reported as is"},
    },
    "known_risks": ["the capacity prior's leave-one-out of S1 is far off (the short-prompt S1 dominates the intercept); "
                    "the 8 T14 predictions are all positive"],
}
OUT.write_text(json.dumps(doc, indent=1) + "\n")
h = sha(OUT)
Path(str(OUT) + ".sha256").write_text(f"{h}  {OUT.name}\n")
os.chmod(OUT, stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
os.chmod(str(OUT) + ".sha256", stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
print(h, head, "dirty" if dirty else "clean", "ledgers", len(ledgers), "seed_overlap", len(t14_seeds & seeds),
      "id_overlap_round", len(t14_ids & own_ids), "id_overlap_earlier", len(t14_ids & other_ids))
