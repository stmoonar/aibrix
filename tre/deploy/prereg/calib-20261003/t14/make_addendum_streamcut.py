# T14 stream-cut addendum for the 2026-10-03 round, sealed before any T14 data exists.
import hashlib, json, os, stat, sys
from datetime import datetime, timezone
from pathlib import Path

C = Path("/data/nfs_shared_data/xxy/calib_20261003")
OUT = C / "prereg" / "ADDENDUM-T14-streamcut.json"
if OUT.exists():
    sys.exit(f"{OUT} exists: an addendum is written once")
if (C / "T14").exists() and any((C / "T14").iterdir()):
    sys.exit("T14 data exists: too late to register")
PREREG = C / "t14" / "preregistration.json"
FREEZE = C / "freeze" / "params_freeze.json"
CROSS = C / "prereg" / "ADDENDUM-T14-crossshape.json"
ROUTE_TIMEOUT_MS = 150000


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


fz = json.load(open(FREEZE))
doc = {
    "what": ("Addendum to the T14 preregistration of the 2026-10-03 round: the model-error void limit and the "
             "route-timeout stream-cut rule. Registered before any T14 data exists (user decision 2026-10-04)."),
    "written_at_utc": datetime.now(timezone.utc).isoformat(),
    "amends": {"path": str(PREREG), "sha256": sha(PREREG),
               "note": "adds rules only; changes no bound key of the preregistration (design, seeds, cells, prior, parameter sets)"},
    "parameter_set": {"path": str(FREEZE), "sha256": sha(FREEZE), "freeze_sha256": fz["freeze_sha256"]},
    "sibling_addendum": {"path": str(CROSS), "sha256": sha(CROSS)},
    "why": ("In this round the tre-v2 HTTPRoute total timeout (150 s: HTTPRoute timeouts.request + EnvoyPatchPolicy "
            "tre-route-timeouts) cut streams mid-decode under overload; the client sees RemoteProtocolError with e2e "
            "about 151 s and records it as model_error. 7b P1 at 3 x rho* voided twice on it (5.23 % / 5.18 % > 5 %; "
            "ATTEMPTS 2026-10-04T13:58) and was rerun as p1r2 with the same treatment as here."),
    "runtime_limit": {
        "max_model_error_rate": 0.10,
        "default_was": 0.05,
        "how": ("calibration_campaign --max-model-error-rate 0.10 (passed to every r3_grid cell; openloop.check_cell voids "
                "a cell when model_errors / sent > the limit). T14 is launched by hand as "
                "`run_T14.sh dsqwen-14b -- --max-model-error-rate 0.10`, every other input as RUN plan G4 / chain/run_G.sh."),
        "unchanged": "every other guard (shed void, proxy transient budget, 50 ms lateness p99, reissue contamination, drain gate, void re-drive once then stop)",
    },
    "audit_rule": {
        "cut": (f"a request with outcome model_error and e2e_ms >= {ROUTE_TIMEOUT_MS} (the route timeout) is a "
                "route-timeout cut: censored, not a model_error"),
        "same_as": "the p1r2 audit rule (ATTEMPTS 2026-10-04T13:58)",
        "rule": ("every model_error in an accepted T14 cell must be a route-timeout cut; any other model_error is a real "
                 "engine error, reported per cell and judged against the original 0.05 limit (a cell above it is void "
                 "at audit)"),
        "label": ("the SLO label is unchanged: a cut request did not complete within the route timeout and stays an "
                  "unserved request (violated window), as in training and M; the rule only changes the void audit and "
                  "the error accounting"),
        "report": "per cell: model_error count, cuts, non-cut errors, cut share of sent",
    },
    "does_not_change_acceptance": "the T14 evaluation (A, B' at dwell 1) and RUN plan H are unchanged",
}
OUT.write_text(json.dumps(doc, indent=1, ensure_ascii=False) + "\n")
h = sha(OUT)
Path(str(OUT) + ".sha256").write_text(f"{h}  {OUT.name}\n")
os.chmod(OUT, stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
os.chmod(str(OUT) + ".sha256", stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
print(h)
