#!/usr/bin/env python3
"""supp_lib: shared code of the supplementary TSS calibration fit tools (2026-11-03).  CPU only, read-only on data.

Spec  : tre_local_workspace/docs/prereg-calib-supplement-20261103.md (prereg: primary analysis, S1-S5, acceptance §4)
Design: tre_local_workspace/docs/calib-supplement-design-20261103.md; background train-set-slo-review-20261102.md §8, §10.

Imported unchanged (never copied): refit_fixedslo.py (this directory; from <data root>/refit-fixedslo-w10-20261031/,
    only its paths made configurable)
    glm, local_root (local logistic on log TSS, Gaussian kernel h 0.15), local_curve, auroc, boot_ci, antitonic.
    Its tss_all / kept_windows / count are used only by selfcheck.py, to assert that the generalised code below gives
    identical windows, TSS and counts on the old roots.
Copied unchanged: the tau_high ladder / n = 3 rule of <data root>/campaign-icse-tune-20261031/fit/rstar_F.py (ladder, logtss_at,
    tau_high, n3 median, half-up rounding, floor_.01 cap at 0.85 C_m, C_m = C_m^D q99(TSS)/q99(TSS_D)).  Generalised ONLY
    in which cells form the ladder (sources / primitives are parameters; default = run2 / run2b holds = tune F, plus the
    hold cells of every added source).
Generalised here (old roots give the same numbers; asserted by selfcheck.py):
  * several dataset roots (the 10-03 training roots + the new campaign's), tolerant source naming / layout;
  * trim of the earliest kept window per (source, scenario_id) instead of per scenario_id (identical on the old roots:
    no scenario_id is in two sources; the new campaign may reuse cell serials);
  * attribution: first_token (prereg §2.1: send_ts + ttft in (end - 10 s, end]; a request without a first token, i.e. a
    failure, by its done / abort time, = analysis-train-slo-20261102/A/followup2/grid_ft.py) or completion (refit k1/n1);
  * TSS vectorised (= tre_common.tss.tss_terms as recompute_tss_rows calls it, EMA 0), asserted equal to
    recompute_tss_rows (this repo's; the freeze tree's until the 11-03 move) on every source for every basis used;
  * guard: sealed held-out data (M under calib_20261003/M, the coming E set) is refused BEFORE any file is opened.

Site configuration (2026-11-03, moved into the repo; nothing site-specific is written here):
  TRE_FIT_DATA_ROOT   the directory that holds the calibration data roots (calib_20261003/, calib_supp_20261103/,
                      analysis-tss-params-20261030/, refit-fixedslo-w10-20261031/, freeze-ctr20-20261028/); the
                      'old' training roots, the freeze parameters and the sealed-root prefixes are named relative to it.
                      Unset: '--data old' is refused and the freeze lambda cross-check is skipped (logged).  The
                      sealed guard does not depend on it (it matches the sealed roots' names anywhere in a path).
  TRE_EXCLUSIVE_WINDOW_FILE  the exclusive-window marker (default <data root>/TRE_EXCLUSIVE_WINDOW)
  TRE_EXCLUSIVE_WINDOW_DENY_HOSTS, TRE_FIT_CPUS  while the marker exists the tools run only when both are set, the
                      host is not denied (a comma list of hostname substrings; at our site the control-plane node) and
                      the process is pinned to exactly TRE_FIT_CPUS (fit_env.sh: nice + taskset).  Fail closed.
"""
import hashlib
import importlib.util
import json
import math
import os
import re
import sys
import time
import zlib

sys.dont_write_bytecode = True
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

DATA_ROOT = os.path.normpath(os.environ["TRE_FIT_DATA_ROOT"]) if os.environ.get("TRE_FIT_DATA_ROOT") else None


def _under_root(rel):
    return os.path.join(DATA_ROOT, rel) if DATA_ROOT else None


REF = os.path.join(os.path.dirname(os.path.abspath(__file__)), "refit_fixedslo.py")
FREEZE_PARAMS = _under_root("freeze-ctr20-20261028/freeze/params_freeze.json")
# old (10-03 calibration) training roots, built from calib_20261003/<stage>/<model> by calibration_dataset.py
# (main 9f2856a1, --attribution hybrid --queue-estimator raw_mean --numerator vllm_counter --step-ms 10000)
OLD_ROOTS = {w: p for w, p in ((10000, _under_root("analysis-tss-params-20261030/win10/alphafit-w10000-s10000/datasets")),
                               (20000, _under_root("refit-fixedslo-w10-20261031/datasets-main-w20000-s10000")))
             if p}  # no 30 s root exists (11-03)
OLD_SOURCES = {"dsqwen-7b": ("run1", "run2", "supp", "p1r2"),
               "dsllama-8b": ("run1", "run2", "supp", "p1"),
               "dsqwen-14b": ("run1", "run2b", "supp", "p1")}
MODELS = tuple(OLD_SOURCES)
DATASET_DIR = "dataset_hybrid"
OLD7 = ("S1", "S2", "S3", "S4", "S5", "T8", "T9")
S3_OLD = ("S1", "S2", "S4", "S5", "T9")  # prereg §3 S3: the old shapes with input <= 1100 (explicit list)
S3_MAX_INPUT = 1100
GSHAPE = re.compile(r"^G(\d+)x(\d+)$")
# held out by gen_calibration_schedules.is_held_out: M, ACCEPTANCE_SHAPES, T14_SHAPES, VALIDATION_SHAPES (never trained on)
HELD_OUT_SHAPES = frozenset({"M", "MP", "MD", "G800x240", "U512x512", "G512x256", "G1200x240", "G640x400",
                             "G1800x160", "G3072x96", "G4096x64", "G256x768", "G512x1024", "ES", "EL"})
TRAIN_SPLITS = ("train", "auxiliary")
TTFT_SLO_MS, TPOT_SLO_MS = 500.0, 75.0
K_LAM = TPOT_SLO_MS / TTFT_SLO_MS  # lambda = 0.15 L_out
K_WP = TTFT_SLO_MS / TPOT_SLO_MS   # w_p = 6.667 / L_in
STEP_MS = 10000.0
TRIM = 1
H_LOCAL = 0.15
R_CRIT = 0.17
CAP, MIN_W = 0.85, 0.20
CM_D_10S = {"dsqwen-7b": 700.0, "dsllama-8b": 690.0, "dsqwen-14b": 610.0}  # tune D: W / TPOT_min at W = 10 s
LAM_D = {"dsqwen-7b": 2.125, "dsllama-8b": 2.0, "dsqwen-14b": 2.0}  # tune D TSS basis (freeze lambda, w_p 0)
LIVE_F = {"dsqwen-7b": dict(theta_m=404.9, tau_crit=0.65, tau_high=1.39, lambda_wait=20.0, w_p=0.01),
          "dsllama-8b": dict(theta_m=363.6, tau_crit=0.52, tau_high=1.59, lambda_wait=20.0, w_p=0.01),
          "dsqwen-14b": dict(theta_m=439.4, tau_crit=0.48, tau_high=1.20, lambda_wait=20.0, w_p=0.01)}
LADDER_SRC_OLD = ("run2", "run2b")
LADDER_PRIMITIVES = ("hold", "static")
NBOOT = 200
SEED = 20261103
CONTRACT = {"numerator": "vllm_counter", "queue_estimator": "raw_mean", "attribution": "hybrid",
            "step_ms": 10000, "window_align": "grid"}
PRESETS = {
    "primary": dict(attribution="first_token", lamwp="formula", r_star=0.015, shapes="all"),
    "S1": dict(attribution="first_token", lamwp="formula", r_star=0.05, shapes="all"),
    "S2": dict(attribution="first_token", lamwp="fixed", lam=20.0, wp=0.01, r_star=0.015, shapes="all"),
    "S3": dict(attribution="first_token", lamwp="formula", r_star=0.015, shapes="le1100"),
    "S4": dict(attribution="completion", lamwp="formula", r_star=0.015, shapes="all"),
    "S5": dict(attribution="first_token", lamwp="formula", r_star=0.015, shapes="all"),  # run at --window-ms 20000/30000
}
half_up = lambda v: math.floor(v * 100 + 0.5 + 1e-9) / 100  # noqa: E731  (rstar_F / make_tuneF)
floor2 = lambda v: math.floor(v * 100 + 1e-9) / 100  # noqa: E731
ceil2 = lambda v: math.ceil(v * 100 - 1e-9) / 100  # noqa: E731
T0 = time.time()


def log(*a):
    print(f"[{time.time() - T0:7.1f}s]", *a, file=sys.stderr, flush=True)


def model_seed(model):
    return SEED + zlib.crc32(model.encode()) % 100000


# ------------------------------------------------------------------ refit_fixedslo (imported unchanged)
def _load_ref():
    spec = importlib.util.spec_from_file_location("refit_fixedslo", REF)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


r = _load_ref()


def configure(window_ms):
    """refit_fixedslo's globals at this window, raw TSS (EMA 0).  Its estimators read none of the changed globals;
    kept_windows / tss_all (selfcheck only) read WINDOW_MS / EMA_TAU_MS / TRAIN_SPLITS."""
    root = OLD_ROOTS.get(int(window_ms), OLD_ROOTS.get(10000, ""))  # read by selfcheck's reference path only
    r.configure(["--no-stage2", "--ds-root", root, "--window-ms", str(int(window_ms)), "--ema-tau-ms", "0",
                 "--out", os.devnull])
    r.TRAIN_SPLITS = TRAIN_SPLITS
    assert r.EMA_TAU_MS is None and r.STEP_MS == STEP_MS and r.TRIM == TRIM and r.H_LOCAL == H_LOCAL
    assert (r.TTFT_SLO, r.TPOT_SLO) == (TTFT_SLO_MS, TPOT_SLO_MS)
    if FREEZE_PARAMS is None:
        log("note: TRE_FIT_DATA_ROOT unset - the freeze lambda (LAM_D) cross-check is skipped")
        return
    fz = json.load(open(FREEZE_PARAMS))["models"]
    assert {m: fz[m]["published"]["lambda_wait"] for m in MODELS} == LAM_D, "freeze lambda != LAM_D"


# ------------------------------------------------------------------ sealed-set guard
# sealed roots, matched as consecutive path components anywhere in a path (no data root needed)
SEALED_ROOTS = ("calib_20261003/M", "calib_20261003/next-20261005/M2", "calib_20261003/next-20261005/T14",
                "calib_20261003/T14", "calib_20261003/t14")
SUPP_RAW_NAME = "calib_supp_20261103"  # the new campaign's root: a component starting with 'E' below it = the E set
_SEALED_COMPONENT = re.compile(r"^(M|M2|T14|E|E[_\-.+].*|.*sealed.*|.*holdout.*|.*held[_\-]?out.*|.*validation.*"
                               r"|.*acceptance.*)$", re.I)


def sealed_reason(path, extra=()):
    """Why a path names sealed held-out data (None = not sealed).  String test only: no file-system access."""
    p = os.path.normpath(os.path.abspath(str(path)))
    parts = p.split(os.sep)
    for rel in SEALED_ROOTS:
        rp = rel.split("/")
        if any(parts[i:i + len(rp)] == rp for i in range(len(parts) - len(rp) + 1)):
            return f"under the sealed root .../{rel}"
    for c in parts:
        if c and _SEALED_COMPONENT.match(c):
            return f"path component {c!r} names a sealed / validation set"
    if SUPP_RAW_NAME in parts:
        for c in parts[parts.index(SUPP_RAW_NAME) + 1:]:
            if c.startswith("E"):
                return f"component {c!r} under .../{SUPP_RAW_NAME} (the E set)"
    pats = tuple(extra) + tuple(x for x in os.environ.get("SUPP_SEALED_PATTERNS", "").split(",") if x)
    for pat in pats:
        if re.search(pat, p):
            return f"matches the sealed pattern {pat!r}"
    return None


def guard_path(path, final=False, extra=(), what="dataset"):
    """Refuse a sealed path before anything under it is opened; the symlink target is checked only after the plain
    string passed.  final=True (validate_supp.py --i-am-the-final-evaluation) lets it through and returns the reason."""
    reason = sealed_reason(path, extra)
    if reason is None:
        reason = sealed_reason(os.path.realpath(path), extra)
    if reason is not None and not final:
        raise SystemExit(f"REFUSED {what} {path}: {reason}.  Sealed held-out sets (M = calib_20261003/M, the E set) "
                         "never enter a fit, and are read only by the one-time final evaluation "
                         "(validate_supp.py --i-am-the-final-evaluation).")
    return reason


# ------------------------------------------------------------------ sources
def _discover(path, model):
    """[(subname | None, dataset_dir)] for one spec path.  Layouts: DIR/windows.csv; DIR/dataset_hybrid/;
    DIR/<model>/dataset_hybrid/; DIR/<src>/<model>/dataset_hybrid/ (several sources)."""
    if os.path.isfile(os.path.join(path, "windows.csv")):
        return [(None, path)]
    if os.path.isfile(os.path.join(path, DATASET_DIR, "windows.csv")):
        return [(None, os.path.join(path, DATASET_DIR))]
    if os.path.isfile(os.path.join(path, model, DATASET_DIR, "windows.csv")):
        return [(None, os.path.join(path, model, DATASET_DIR))]
    subs = []
    if os.path.isdir(path):
        for s in sorted(os.listdir(path)):
            d = os.path.join(path, s, model, DATASET_DIR)
            if os.path.isfile(os.path.join(d, "windows.csv")):
                subs.append((s, d))
    return subs


def _default_name(path, ddir, model):
    parts = os.path.normpath(ddir).split(os.sep)
    if parts[-1] == DATASET_DIR and len(parts) >= 3 and parts[-2] == model:
        return parts[-3]
    if parts[-1] == DATASET_DIR:
        return parts[-2]
    return os.path.basename(os.path.normpath(path))


def resolve_specs(specs, model, window_ms, final=False, extra=(), only_src=None):
    """specs: 'old' (the 10-03 roots at this window), 'old=ROOT', 'NAME=PATH' or 'PATH'.
    Returns [dict(name, ddir, origin)]; names are unique (key = name|cell_id|attempt)."""
    out = []
    for spec in specs:
        if spec == "old" or spec.startswith("old="):
            root = spec[4:] if spec.startswith("old=") else OLD_ROOTS.get(int(window_ms))
            if root is None:
                raise SystemExit(f"no old training root for window {window_ms} ms (known: {sorted(OLD_ROOTS)}; "
                                 "TRE_FIT_DATA_ROOT names them); build one (build_datasets.sh) and pass old=ROOT")
            guard_path(root, False, extra, "old root")
            for s in OLD_SOURCES[model]:
                out.append(dict(name=s, ddir=f"{root}/{s}/{model}/{DATASET_DIR}", origin="old"))
            continue
        name, path = spec.split("=", 1) if "=" in spec else (None, spec)
        guard_path(path, final, extra)
        path = os.path.normpath(os.path.abspath(path))
        found = _discover(path, model)
        if not found:
            log(f"note: {spec}: no {DATASET_DIR}/windows.csv for {model} - skipped for this model")
        for sub, ddir in found:
            guard_path(ddir, final, extra)
            nm = (f"{name}.{sub}" if name and sub else name) or sub or _default_name(path, ddir, model)
            out.append(dict(name=nm, ddir=ddir, origin="added"))
    if only_src:
        out = [s for s in out if s["name"] in only_src]
    names = [s["name"] for s in out]
    dup = sorted({n for n in names if names.count(n) > 1})
    if dup:
        raise SystemExit(f"{model}: source names not unique {dup}: name the added sources NAME=PATH")
    return out


def _sha256(path, chunk=1 << 22):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def provenance(src, window_ms, final=False, extra=(), hash_files=True, allow_contract=False):
    """manifest contract (window / step / numerator / queue estimator / attribution), builder commit, run root, file
    sizes and sha256.  A run root under a sealed set is refused (training / dev)."""
    ddir = src["ddir"]
    rec = {"name": src["name"], "dataset_dir": ddir, "origin": src["origin"], "files": {}, "contract_problems": []}
    for f in ("windows.csv", "requests.csv", "manifest.json", "cells.csv"):
        p = os.path.join(ddir, f)
        if os.path.isfile(p):
            st = os.stat(p)
            rec["files"][f] = {"bytes": st.st_size, "mtime": int(st.st_mtime),
                               "sha256": _sha256(p) if hash_files else None}
    mp = os.path.join(ddir, "manifest.json")
    if not os.path.isfile(mp):
        rec["contract_problems"].append("no manifest.json (contract not checked)")
        return rec
    m = json.load(open(mp))
    win = m.get("windowing") or {}
    rec.update(format_revision=m.get("format_revision"), built_at_utc=m.get("built_at_utc"),
               builder_commit=(m.get("builder") or {}).get("code_commit"),
               builder_dirty=(m.get("builder") or {}).get("code_dirty"), run_root=m.get("run_root"),
               window_ms=win.get("window_ms"), step_ms=win.get("step_ms"), window_align=win.get("window_align"),
               numerator=(m.get("numerator") or {}).get("source"),
               queue_estimator=(m.get("queue_estimator") or {}).get("name"),
               attribution=(m.get("attribution") or {}).get("value"))
    if rec["run_root"]:
        reason = sealed_reason(rec["run_root"], extra)
        if reason and not final:
            raise SystemExit(f"REFUSED {ddir}: its manifest run_root {rec['run_root']} is sealed ({reason})")
    hard = []
    if rec["window_ms"] is not None and int(rec["window_ms"]) != int(window_ms):
        hard.append(f"window_ms {rec['window_ms']} != {window_ms}")
    if rec["step_ms"] is not None and int(rec["step_ms"]) != CONTRACT["step_ms"]:
        hard.append(f"step_ms {rec['step_ms']} != {CONTRACT['step_ms']}")
    soft = []
    for k in ("numerator", "queue_estimator", "window_align"):
        if rec.get(k) is not None and rec[k] != CONTRACT[k]:
            soft.append(f"{k} {rec[k]!r} != {CONTRACT[k]!r}")
    if src.get("origin") != "old" and not rec.get("builder_commit"):  # provenance; the 10-03 old roots exempt
        soft.append("builder code_commit is null (git could not read the builder repo: build with "
                    "build_datasets.sh, which refuses then; GIT_CONFIG_GLOBAL with [safe] directory, fit_env.sh)")
    if rec.get("attribution") not in (None, CONTRACT["attribution"]):
        rec["contract_problems"].append(f"attribution {rec['attribution']!r} (label columns only; not used here)")
    rec["contract_problems"] += hard + soft
    if hard or (soft and not allow_contract):
        raise SystemExit(f"{ddir}: dataset contract violated: {hard + soft} (TSS inputs differ from the old roots; "
                         "rebuild with build_datasets.sh, or --allow-contract-mismatch for the soft items)")
    return rec


# ------------------------------------------------------------------ loading
WIN_NEED = ("model", "shape", "primitive", "rho", "cell_id", "attempt", "split", "cell_status", "rho_factor",
            "possibly_contaminated", "in_warmup", "scenario_id", "window_start_ms", "window_end_ms",
            "prompt_tokens_total", "generation_tokens_total", "avg_waiting", "avg_running")
REQ_NEED = ("model", "cell_id", "attempt", "send_ts_ms", "done_ts_ms", "ttft_ms", "e2e_ms", "input_tokens",
            "output_tokens", "http_status", "outcome", "request_timeout_s")


def per_request_frame(q, name):
    """refit_fixedslo.per_request (label: not ok | TTFT > 500 | TPOT > 75; done time) + first-token time + served
    lengths (outcome ok, http 200, both token counts present) for the lambda / w_p formula."""
    ok = (q.outcome == "ok") & (q.http_status == 200)
    tt = q.ttft_ms
    out = q.output_tokens
    tpot = np.where(out > 1, (q.e2e_ms - tt) / (out - 1), np.nan)
    v = (~ok) | (tt.fillna(0.0) > TTFT_SLO_MS) | (np.nan_to_num(tpot, nan=0.0) > TPOT_SLO_MS)
    done = q.done_ts_ms.fillna(q.send_ts_ms + q.e2e_ms.fillna(q.request_timeout_s * 1000))
    ttv = tt.values.astype(float)
    tft = np.where(np.isfinite(ttv), q.send_ts_ms.values.astype(float) + ttv, done.values.astype(float))
    served = ok & q.input_tokens.notna() & q.output_tokens.notna()
    return pd.DataFrame({
        "key": (name + "|" + q.cell_id.astype(str) + "|" + q.attempt.astype(str)).values,
        "v": v.values.astype(float), "done": done.values.astype(float), "tft": tft,
        "served": served.values.astype(float), "fail": (~ok).values.astype(float),
        "lin": np.where(served.values, q.input_tokens.values.astype(float), 0.0),
        "lout": np.where(served.values, q.output_tokens.values.astype(float), 0.0),
        "outcome": q.outcome.astype(str).values})


def shape_class(shape):
    """old7 | new (G<in>x<out>, not held out) | heldout | unknown"""
    if shape in HELD_OUT_SHAPES:
        return "heldout"
    if shape in OLD7:
        return "old7"
    if GSHAPE.match(str(shape)):
        return "new"
    return "unknown"


def nominal_input(shape):
    if shape in OLD7:
        return {"S1": 256, "S2": 768, "S3": 2048, "S4": 256, "S5": 768, "T8": 1600, "T9": 812}[shape]
    g = GSHAPE.match(str(shape))
    return int(g.group(1)) if g else None


class ModelData:
    """All windows (split-filtered) and requests (of those windows' cells) of one model over the given sources."""

    def __init__(self, model, window_ms, sources, splits, final=False, extra=(), allow_shapes=(),
                 hash_files=True, allow_contract=False, warmup_fallback_s=0.0):
        self.model, self.window_ms, self.splits, self.final = model, int(window_ms), splits, final
        self.sources, self.prov, self.notes = sources, [], []
        Ws, Qs = [], []
        for s in sources:
            guard_path(s["ddir"], final, extra)
            self.prov.append(provenance(s, window_ms, final, extra, hash_files, allow_contract))
            wpath, qpath = os.path.join(s["ddir"], "windows.csv"), os.path.join(s["ddir"], "requests.csv")
            w = pd.read_csv(wpath, low_memory=False)
            miss = [c for c in WIN_NEED if c not in w.columns]
            if miss:
                raise SystemExit(f"{wpath}: missing columns {miss}")
            w = w[w.model == model]
            n_rows = len(w)
            if splits is not None:
                w = w[w.split.isin(splits)]
            w = w.copy()
            w["src"] = s["name"]
            w["origin"] = s["origin"]
            w["key"] = s["name"] + "|" + w.cell_id.astype(str) + "|" + w.attempt.astype(str)
            if warmup_fallback_s > 0 and s["origin"] != "old":
                nf = w.groupby("key").in_warmup.transform(lambda c: c.isna().all())
                st = w.groupby("key").window_start_ms.transform("min")
                drop = nf.astype(bool) & (w.window_end_ms <= st + warmup_fallback_s * 1000.0)
                if drop.any():
                    self.notes.append(f"{s['name']}: warm-up fallback dropped {int(drop.sum())} windows "
                                      f"(cells without in_warmup, first {warmup_fallback_s:g} s)")
                w = w[~drop]
            if s["origin"] != "old":
                nf = w.groupby("key").in_warmup.apply(lambda c: c.isna().all())
                if nf.any():
                    self.notes.append(f"{s['name']}: {int(nf.sum())}/{len(nf)} cells carry no in_warmup flag "
                                      "(warm-up windows kept unless --warmup-fallback-s)")
            hdr = pd.read_csv(qpath, nrows=0).columns
            miss = [c for c in REQ_NEED if c not in hdr]
            if miss:
                raise SystemExit(f"{qpath}: missing columns {miss}")
            q = pd.read_csv(qpath, usecols=list(REQ_NEED), low_memory=False)
            q = q[q.model == model]
            q = per_request_frame(q, s["name"])
            q = q[q.key.isin(set(w.key.unique()))]  # requests of other splits / models never enter
            self.prov[-1].update(n_window_rows_model=int(n_rows), n_window_rows_used=int(len(w)),
                                 n_requests_used=int(len(q)))
            Ws.append(w)
            Qs.append(q)
        if not Ws:
            raise SystemExit(f"{model}: no dataset source")
        self.Wall = pd.concat(Ws, ignore_index=True)
        self.Wall["_row"] = np.arange(len(self.Wall))
        self.Q = pd.concat(Qs, ignore_index=True)
        if not final:  # training / dev: only training shapes
            cls = {sh: shape_class(sh) for sh in self.Wall["shape"].unique()}
            bad = sorted(sh for sh, c in cls.items() if c == "heldout")
            if bad:
                raise SystemExit(f"{model}: held-out shapes {bad} with split in {splits}: refusing (a sealed set "
                                 "must never enter a fit; check the source / split)")
            unk = sorted(sh for sh, c in cls.items() if c == "unknown" and sh not in allow_shapes)
            if unk:
                raise SystemExit(f"{model}: unknown shapes {unk} in the training data (not S1-S5/T8/T9, not "
                                 "G<in>x<out>): refusing - a validation mixture must never enter a fit.  Pass "
                                 "--allow-shapes only for a genuine training shape.")


def tss_vec(W, lam, wp, qmin=1.0):
    """Raw TSS per row = tre_common.tss.tss_terms as recompute_tss_rows calls it (EMA 0):
    (w_p P (1 - kv) + D) / max(R + lambda Wq, qmin) x factor; NaN if tokens missing or R + Wq <= 0."""
    def col(c, default):
        if c not in W:
            return np.full(len(W), default, float)
        v = pd.to_numeric(W[c], errors="coerce").to_numpy(float)
        v = np.where(np.isfinite(v), v, default)
        return np.where(v == 0, default, v) if default != 0 else v  # '_as_float(x, d) or d'
    P = pd.to_numeric(W.prompt_tokens_total, errors="coerce").to_numpy(float)
    D = pd.to_numeric(W.generation_tokens_total, errors="coerce").to_numpy(float)
    R, Q, kv = col("avg_running", 0.0), col("avg_waiting", 0.0), col("kv_cache_hit_rate", 0.0)
    ar, rp = col("assigned_replicas", 1.0), col("routable_pods", 1.0)
    eff = np.maximum(1.0, rp)
    factor = np.where(ar <= 0, eff, ar) / eff
    with np.errstate(invalid="ignore", divide="ignore"):
        raw = (wp * P * (1.0 - kv) + D) / np.maximum(R + lam * Q, qmin) * factor
    raw[~(np.isfinite(P) & np.isfinite(D))] = np.nan
    raw[(R + Q) <= 0.0] = np.nan
    return raw


def assert_tss(W, lam, wp, tss):
    """The vectorised TSS equals recompute_tss_rows (EMA None; this repo's tre_calibration.dataset) on every row."""
    vals = r.recompute_tss_rows(W.to_dict("records"), r.TssRecompute(w_p=wp, lambda_wait=lam, qmin=1.0,
                                                                      ema_tau_ms=None))
    ref = np.array([np.nan if v is None else v for v in vals], float)
    a, b = np.isnan(ref), np.isnan(tss)
    assert np.array_equal(a, b), ("TSS NaN pattern differs", int((a != b).sum()))
    assert np.allclose(ref[~a], tss[~b], rtol=1e-12, atol=1e-9), "TSS differs from recompute_tss_rows"


def kept(W, tss, window_ms, splits=TRAIN_SPLITS):
    """refit_fixedslo.kept_windows' filter: split, valid, not in_warmup (missing flag = kept), not possibly_contaminated,
    finite TSS > 0, tokens > 0; then the earliest TRIM windows per (source, scenario_id) dropped.  Row order kept."""
    w = W.copy()
    w["tss"] = tss
    m = ((w.cell_status == "valid") & (w.in_warmup != True) & (w.possibly_contaminated != True)  # noqa: E712
         & np.isfinite(w.tss) & (w.tss > 0)
         & ((w.prompt_tokens_total.fillna(0) + w.generation_tokens_total.fillna(0)) > 0))
    if splits is not None:
        m &= w.split.isin(splits)
    w = w[m]
    dur = np.unique((w.window_end_ms - w.window_start_ms).to_numpy(float))
    if len(dur) and not np.all(np.abs(dur - float(window_ms)) < 1e-6):
        raise SystemExit(f"windows are {dur[:5]} ms long, expected {window_ms} (dataset built at another window?)")
    w = w.sort_values(["src", "scenario_id", "window_start_ms"], kind="stable")
    first = w.groupby(["src", "scenario_id"]).cumcount() < TRIM
    return w[~first].sort_index().reset_index(drop=True)


def attr_counts(K, Q, tcol, cols):
    """Per kept window: [n, sum(cols)...] over requests of the same key with Q[tcol] in (end - STEP, end]
    (= refit_fixedslo.count / grid_ft.count_by).  Windows of one key end STEP apart: each request at most once."""
    res = np.zeros((len(K), len(cols) + 1))
    qidx = Q.groupby("key").indices
    lo_all = K.window_end_ms.to_numpy(float) - STEP_MS
    hi_all = K.window_end_ms.to_numpy(float)
    tv = Q[tcol].to_numpy(float)
    cv_all = Q[list(cols)].to_numpy(float)
    for key, idx in K.groupby("key").indices.items():
        qi = qidx.get(key)
        if qi is None:
            continue
        t = tv[qi]
        o = np.argsort(t, kind="stable")
        ts = t[o]
        cv = np.vstack([np.zeros(len(cols)), np.cumsum(cv_all[qi][o], axis=0)])
        lo = np.searchsorted(ts, lo_all[idx], side="right")
        hi = np.searchsorted(ts, hi_all[idx], side="right")
        res[idx, 0] = hi - lo
        res[idx, 1:] = cv[hi] - cv[lo]
    return res


TCOL = {"first_token": "tft", "completion": "done"}


def shape_subset(K, shapes, allow_new_unparsed=()):
    """'all' | 'le1100' (prereg S3: S1, S2, S4, S5, T9 + new shapes with nominal input <= 1100) | comma list."""
    present = sorted(K["shape"].unique())
    if shapes == "all":
        keep = present
    elif shapes == "le1100":
        keep = []
        for s in present:
            if s in OLD7:
                if s in S3_OLD:
                    keep.append(s)
                continue
            ni = nominal_input(s)
            if ni is None and s not in allow_new_unparsed:
                raise SystemExit(f"S3: nominal input of shape {s!r} unknown; give the subset with --shapes")
            if ni is not None and ni <= S3_MAX_INPUT:
                keep.append(s)
    else:
        keep = [s for s in shapes.split(",") if s]
        miss = sorted(set(keep) - set(present))
        if miss:
            raise SystemExit(f"--shapes: {miss} not in the data ({present})")
    return K[K["shape"].isin(keep)].reset_index(drop=True), keep


# ------------------------------------------------------------------ estimators
def p_at(x, k, n, x0, wt=None, h=H_LOCAL):
    """Local-logistic p-hat at log TSS x0 (= grid_ft.p_at; NaN below 200 kernel-weighted requests)."""
    base = np.ones_like(x) if wt is None else wt
    w = base * np.exp(-0.5 * ((x - x0) / h) ** 2)
    if (w * n).sum() < 200:
        return np.nan
    b = r.glm(x, k, n, w)
    return float(1 / (1 + np.exp(-(b[0] + b[1] * x0))))


def neff_at(x, n, x0, h=H_LOCAL):
    return float((np.exp(-0.5 * ((x - x0) / h) ** 2) * n).sum())


def cluster_weight_fn(keys):
    codes, uniq = pd.factorize(keys)
    nu = len(uniq)

    def draw(rng):
        pick = rng.integers(0, nu, size=nu)  # one draw per resample (= refit.cluster_weights)
        return np.bincount(pick, minlength=nu)[codes].astype(float)
    return draw, nu


def pct_ci(vals, nb, need=0.8):
    vals = np.asarray(vals, float)
    fin = vals[np.isfinite(vals)]
    ff = float(np.mean(~np.isfinite(vals))) if len(vals) else 1.0
    if len(fin) < need * nb:
        return [None, None], ff
    lo, hi = np.percentile(fin, [5, 95])
    return [float(lo), float(hi)], ff


def _f(v, nd=4):
    if v is None:
        return None
    try:
        if not np.isfinite(v):
            return None
    except TypeError:
        return v
    return round(float(v), nd)


# ------------------------------------------------------------------ tau_high: rstar_F ladder (copied unchanged)
NS = (2, 3, 4)


def ladder_cells(K, ladder_src, prims=LADDER_PRIMITIVES):
    """Hold cells of the ladder sources with rho: cell median TSS (>= 3 kept windows).  rho_factor rule per source:
    a source whose ladder-primitive cells carry rho_factor must have it (= tune F on run2 / run2b: excludes the
    boundary-search probes); a source without any rho_factor (static-grid cells) enters on rho alone."""
    h = K[K.src.isin(ladder_src) & K.primitive.isin(prims) & K.rho.notna() & (K.rho > 0)]
    parts, notes = [], []
    for s, g in h.groupby("src"):
        if g.rho_factor.notna().any():
            parts.append(g[g.rho_factor.notna()])
        else:
            parts.append(g)
            notes.append(f"{s}: no rho_factor on any ladder cell - cells enter on rho alone")
    if not parts:
        return pd.DataFrame(columns=["shape", "key", "rho", "tss", "nw", "src", "rho_factor"]), notes
    h = pd.concat(parts)
    cell = h.groupby(["shape", "key"]).agg(rho=("rho", "first"), tss=("tss", "median"), nw=("tss", "size"),
                                            src=("src", "first"), rho_factor=("rho_factor", "first")).reset_index()
    cell = cell[cell.nw >= 3]
    # unit check: rho / rho_factor must be one constant per shape across sources (rho in units of the shape's C_s)
    for sh, g in cell[cell.rho_factor.notna()].groupby("shape"):
        u = (g.rho / g.rho_factor).groupby(g.src).median()
        if len(u) > 1 and (u.max() / u.min() - 1) > 0.01:
            notes.append(f"WARNING {sh}: rho / rho_factor differs between ladder sources {u.round(4).to_dict()} "
                         "- rho units disagree, the ladder mixes scales")
    return cell, notes


def ladder(cell):
    out = {}
    for shape, g in cell.groupby("shape"):
        lv = g.groupby("rho").tss.apply(lambda s: np.exp(np.log(s).mean())).sort_index()
        lr = np.log(lv.index.values)
        _, fit = r.antitonic(lr, np.log(lv.values))
        out[shape] = (lr, fit)
    return out


def logtss_at(lr, fit, l, cap):
    if l < lr[0]:
        s = (fit[1] - fit[0]) / (lr[1] - lr[0]) if len(lr) > 1 else 0.0
        v = fit[0] + s * (l - lr[0])
    elif l > lr[-1]:
        s = (fit[-1] - fit[-2]) / (lr[-1] - lr[-2]) if len(lr) > 1 else 0.0
        v = fit[-1] + s * (l - lr[-1])
    else:
        v = np.interp(l, lr, fit)
    return min(v, np.log(cap))


def tau_high_rows(lad, theta, cap):
    rows = []
    lt = np.log(theta)
    for shape, (lr, fit) in lad.items():
        if len(lr) < 2:
            continue
        grid = np.linspace(lr[0] - 4, lr[-1] + 2, 4000)
        vals = np.array([logtss_at(lr, fit, g, cap) for g in grid])
        if not (vals.max() >= lt >= vals.min()):
            rows.append({"shape": shape, "bracketed": False, "note": "theta outside capped curve"})
            continue
        i = np.nonzero(vals >= lt)[0][-1]
        l1 = grid[i]
        row = {"shape": shape, "bracketed": bool(fit.max() >= lt >= fit.min()), "rho1": float(np.exp(l1))}
        for n in NS:
            ln = l1 + np.log((n - 1) / n)
            row[f"tau_high_n{n}"] = float(np.exp(logtss_at(lr, fit, ln, cap) - lt))
            row[f"in_ladder_n{n}"] = bool(lr[0] <= ln <= lr[-1])
        rows.append(row)
    return rows


def tau_high_rule(lad, theta_pub, q99, C_m, rounding="half_up"):
    """tau_high = min(round_.01(median over shapes of the n = 3 ladder value), floor_.01(0.85 C_m / theta)),
    width tau_high - 1 >= 0.2 (rstar_F; prereg §2.7).  rounding: half_up (= tune F code) | ceil."""
    rows = tau_high_rows(lad, theta_pub, q99)
    T = pd.DataFrame(rows)
    med = float(T.tau_high_n3.median()) if "tau_high_n3" in T and T.tau_high_n3.notna().any() else None
    cap = floor2(CAP * C_m / theta_pub)
    rnd = {"half_up": half_up, "ceil": ceil2}[rounding]
    lad_val = rnd(med) if med is not None else None
    val = min(lad_val, cap) if lad_val is not None else cap
    width_ok = bool(cap - 1.0 >= MIN_W - 1e-9 and val - 1.0 >= MIN_W - 1e-9)
    by = {row["shape"]: row for row in rows}
    interp = sorted(s for s, row in by.items() if row.get("in_ladder_n3") is True)
    extrap = sorted(s for s, row in by.items() if row.get("in_ladder_n3") is False)
    return {"value": val, "ladder_median_n3": med, "ladder_rounded": lad_val, "rounding": rounding, "cap": cap,
            "cap_exact": CAP * C_m / theta_pub, "bound_by": ("cap" if lad_val is None or cap < lad_val else "ladder"),
            "width_ok": width_ok, "n_shapes": len(rows), "interpolated_n3": interp, "extrapolated_n3": extrap,
            "not_bracketed_theta": sorted(s for s, row in by.items() if not row.get("bracketed", False)),
            "rows": rows}


def ladder_info(cell, lad):
    info = {}
    for sh, g in cell.groupby("shape"):
        lr = lad[sh][0]
        info[sh] = {"n_cells": int(len(g)), "n_rho_levels": int(len(lr)), "rho_min": float(np.exp(lr[0])),
                    "rho_max": float(np.exp(lr[-1])), "sources": sorted(g.src.unique()),
                    "cells_by_source": {k: int(v) for k, v in g.src.value_counts().items()}}
    return info


def lam_wp_from(C, served_col=2, lin_col=3, lout_col=4):
    ns = C[:, served_col].sum()
    L_in, L_out = C[:, lin_col].sum() / ns, C[:, lout_col].sum() / ns
    return float(L_in), float(L_out), int(ns), float(K_LAM * L_out), float(K_WP / L_in)


def json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(type(o))


def clean(o):
    """NaN / inf -> None recursively (strict JSON)."""
    if isinstance(o, dict):
        return {k: clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, float) and not math.isfinite(o):
        return None
    if isinstance(o, np.floating):
        return None if not np.isfinite(o) else float(o)
    return o


def tool_hashes():
    """sha256 of the code that produced a result (this directory's tools + the imported estimator library)."""
    here = os.path.dirname(os.path.abspath(__file__))
    out = {}
    for f in ("supp_lib.py", "refit_supp.py", "validate_supp.py"):
        p = os.path.join(here, f)
        if os.path.isfile(p):
            out[f] = _sha256(p)
    out["refit_fixedslo.py"] = _sha256(REF)
    return out


def cpu_list(spec):
    """'0,1' / '0-7' / '0-3,8' -> set of CPU numbers (taskset -c syntax, no strides)."""
    out = set()
    for part in str(spec).split(","):
        part = part.strip()
        if not part:
            continue
        lo, _, hi = part.partition("-")
        out.update(range(int(lo), int(hi or lo) + 1))
    return out


def marker_check(host_ok_75=True):
    """Exclusive window, fail closed: while the marker exists, run only when TRE_EXCLUSIVE_WINDOW_DENY_HOSTS and
    TRE_FIT_CPUS are both set (the site env file), this host is not denied, and the process is pinned to exactly
    TRE_FIT_CPUS (fit_env.sh starts the tools under nice TRE_FIT_NICE + taskset -c TRE_FIT_CPUS)."""
    mk = os.environ.get("TRE_EXCLUSIVE_WINDOW_FILE") or _under_root("TRE_EXCLUSIVE_WINDOW")
    if not (mk and os.path.exists(mk)):
        return
    host = os.uname().nodename
    deny = [h.strip() for h in os.environ.get("TRE_EXCLUSIVE_WINDOW_DENY_HOSTS", "").split(",") if h.strip()]
    cpus = os.environ.get("TRE_FIT_CPUS", "").strip()
    if not deny or not cpus:
        raise SystemExit(f"{mk} exists: exclusive window - refusing: TRE_EXCLUSIVE_WINDOW_DENY_HOSTS and TRE_FIT_CPUS "
                         "must both be set (the site env file) and the tool started by a run_*.sh wrapper")
    if any(h in host for h in deny):
        raise SystemExit(f"{mk} exists: exclusive window - do not run on {host}; use another host with "
                         "nice + taskset (TRE_FIT_NICE / TRE_FIT_CPUS; the run_*.sh wrappers do it)")
    if hasattr(os, "sched_getaffinity"):
        try:
            want = cpu_list(cpus)
        except ValueError:
            raise SystemExit(f"TRE_FIT_CPUS {cpus!r} is not a CPU list")
        have = os.sched_getaffinity(0)
        if have != want:
            raise SystemExit(f"{mk} exists: exclusive window - this process may run on {len(have)} CPUs, not "
                             f"TRE_FIT_CPUS {cpus}: start it under taskset -c {cpus} (fit_env.sh does it)")
    log(f"note: {mk} exists; running on {host}, CPUs {cpus} (window rules: nice, taskset; see fit_env.sh)")
