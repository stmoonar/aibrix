# T14 cross-shape / chapter-2 addendum for the 2026-10-03 round, sealed before any T14 data exists.
# Adapts chapter2_outputs of calibration_v1lambda_20260924/preregistration.json to ONE frozen set.
import hashlib, json, os, stat, sys
from datetime import datetime, timezone
from pathlib import Path

C = Path("/data/nfs_shared_data/xxy/calib_20261003")
OUT = C / "prereg" / "ADDENDUM-T14-crossshape.json"
if OUT.exists():
    sys.exit(f"{OUT} exists: an addendum is written once")
PREREG = C / "t14" / "preregistration.json"
FREEZE = C / "freeze" / "params_freeze.json"
SRC = Path("/data/nfs_shared_data/xxy/calibration_v1lambda_20260924/preregistration.json")
if (C / "T14").exists() and any((C / "T14").iterdir()):
    sys.exit("T14 data exists: too late to register")


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


fz = json.load(open(FREEZE))
doc = {
    "what": ("Addendum to the T14 preregistration of the 2026-10-03 round: the chapter-2 outputs (cross-shape claim rule, "
             "Fig 2.1, lambda disclosure), adapted from the 2026-09-24 preregistration's chapter2_outputs to the single "
             "frozen parameter set. Registered before any T14 data exists (user decision 2026-10-04)."),
    "written_at_utc": datetime.now(timezone.utc).isoformat(),
    "amends": {"path": str(PREREG), "sha256": sha(PREREG),
               "note": "adds keys only; changes no bound key of the preregistration and no collection parameter"},
    "source": {"path": str(SRC), "sha256": sha(SRC), "block": "chapter2_outputs"},
    "parameter_set": {"path": str(FREEZE), "sha256": sha(FREEZE), "freeze_sha256": fz["freeze_sha256"],
                      "dsqwen-14b": fz["models"]["dsqwen-14b"]["published"],
                      "note": ("the only parameter set of this round (gateway numerator + v1-lambda); every output below "
                               "uses it. There is no second set and no replacement rule.")},
    "does_not_change_acceptance": ("Acceptance (RUN plan H: dline_refit accept A, B' at dwell 1, D on M; the T14 "
                                   "evaluation A / B' of the preregistration) is unchanged. Everything here is a reported "
                                   "output: cross_shape states a claim or not; fig_2_1 and lambda_disclosure are "
                                   "disclosure only. No output here gates, tunes or selects anything."),
    "chapter2_outputs": {
        "cross_shape": {
            "what": ("fixed Z = 1 (the published theta of the freeze): BA and AUROC per T14 shape, interpolation and "
                     "extrapolation in separate tables, cell-bootstrap CI per shape (1000 resamples, seed 20260922)"),
            "claim_rule": ("'one theta transfers across shapes' iff the SD of the per-shape BA <= the median per-shape "
                           "BA CI95 half width; stated separately for interpolation and extrapolation"),
            "role": "claim rule for the text; not an acceptance gate",
        },
        "fig_2_1": {
            "what": ("T14 hold cells ordered by health (cell median health score); per cell running+waiting, KV-cache "
                     "usage (1 Hz instant sidecar kv_cache_usage, mean), prefill / decode tokens per second, Z"),
            "metrics": ["cell-level Spearman(median Z, median health)",
                        "number of non-monotone points of Z along the health order", "window-level AUROC"],
            "prefill_vs_decode_contrast": ["G3072x96 @ 1.1", "G256x768 @ 0.9"],
            "parameters": "the freeze (the single parameter set)",
            "role": "disclosure only",
        },
        "lambda_disclosure": {
            "what": ("Z-ranking invariance to lambda on T14: window-level Kendall tau of Z(lambda) vs Z(lambda=1) for "
                     "lambda in {0, 0.5, 1, 1.5, 2, 3, 4} at the freeze's w_p"),
            "role": "disclosure only",
        },
        "not_in_degraded_version": ["Fig 2.2 step cell (no step cell collected)", "deep-overload cells in T14",
                                    "T14 extra mixed shapes"],
        "dropped_from_2026_09_24": ["the replacement rule between two parameter sets (one set this round)",
                                    "'both shown in the appendix' (one set)",
                                    "cross_model_pooled_z_auroc (not requested for this round)"],
    },
}
OUT.write_text(json.dumps(doc, indent=1, ensure_ascii=False) + "\n")
h = sha(OUT)
Path(str(OUT) + ".sha256").write_text(f"{h}  {OUT.name}\n")
os.chmod(OUT, stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
os.chmod(str(OUT) + ".sha256", stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
print(h)
