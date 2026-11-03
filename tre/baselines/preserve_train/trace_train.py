"""PreServe Tier-1 mLSTM trained on the evaluated service's own past (2026-11-04).

The paper (section 5.3, RQ2) trains each service's predictor on that service's first two days and replays the
next two. Our evaluation traces are 720 s synthetic / compressed-real services, so "the service's past" is made
here: each trace's own generator and spec, run with seeds that no evaluation uses, gives independent
realizations of the same service. 4 realizations x 72 windows of 10 s = 288 windows (the paper's 2 days of
10 min windows = 288). The evaluation realization (seed 1) is never used for training.

Config: ``preserve_train/configs/trace-train-20261104.json`` (traces, sets, generators, seeds, hyper-parameters).
Paths are arguments (``--data-root`` holds the trace sets; ``--out`` the generated data, outside git).

Subcommands (one command per step; nothing existing is overwritten)::

    realize  run each set's own generator (a copy of the set's ``_src/``, sha256-checked against the
             evaluation manifests) for the training seeds into ``<out>/<set>/``, and for the evaluation seed into
             ``<out>/_repro/<set>/``; then check that the seed-1 rerun reproduces the evaluation trace and its
             design inputs byte for byte. CPU, needs the v1 generator and the Azure CSVs (76).
    series   per (trace, seed, model) the window sums P (prompt_length) and D (max_output_tokens; null = the
             loadgen config's max_tokens) of the requests by arrival, windows of W from the first arrival
             (the policy's t0 = the first gateway arrival) -> ``<out>/series_w<W>.json`` + variability table.
    train    V1 per trace (one predictor per trace x model, training series = the 4 realizations in seed order)
             and V2 concat (one per model, training series = all traces in config order, each in seed order);
             the 11-03 hyper-parameters (``train_mlstm``: seq_len 25, hidden 64, 50 epochs, batch 32, Adam 1e-3,
             MSE, seed 42) with the paper's min-max normalization (Alg. 1) over the training series; numpy vs
             torch check. Needs torch (75).
    finish   weights dir (npz + manifest + REPORT.md), the history files (last seq_len windows of each trace's
             training series), one-step-ahead MAPE on the evaluation realization for V1, V2, the shipped
             weights with their current history file, and the naive last-window forecaster. numpy only.
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import datetime as dt
import hashlib
import json
import math
import multiprocessing as mp
import os
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

HERE = Path(__file__).resolve().parent
BASELINES = HERE.parent
DEFAULT_CONFIG = HERE / "configs" / "trace-train-20261104.json"
KINDS = {"prompt": "P", "response": "D"}
HISTORY_METHOD = "train-realization-tail-v1-20261104"


# ----------------------------------------------------------------------------------------------- helpers


def log(msg: str) -> None:
    print(f"[{dt.datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def load_config(path) -> dict:
    cfg = json.loads(Path(path).read_text())
    cfg["_path"] = str(path)
    cfg["_sha256"] = sha256_file(path)
    for t in cfg["traces"]:
        if t["set"] not in cfg["sets"]:
            raise SystemExit(f"config: trace {t['trace']} names unknown set {t['set']}")
    if cfg["eval_seed"] in cfg["train_seeds"]:
        raise SystemExit("config: the evaluation seed is a training seed")
    return cfg


def git_info(path: Path) -> dict:
    try:
        sha = subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()
        dirty = subprocess.check_output(["git", "-C", str(path), "status", "--porcelain", "--", "."],
                                        text=True).strip()
        return {"sha": sha, "dirty": bool(dirty)}
    except Exception as e:  # noqa: BLE001
        return {"sha": None, "error": str(e)}


def write_new(path: Path, text: str) -> None:
    """Write ``text`` to a new file; an identical existing file is kept, a different one refuses."""
    if path.exists():
        if path.read_text() == text:
            return
        raise SystemExit(f"{path} exists with other content (nothing is overwritten)")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text)
    os.replace(tmp, path)


def trace_dir(root: Path, set_name: str, trace: str, seed: int) -> Path:
    return root / set_name / trace / f"seed{seed}"


def train_keys(cfg: dict, t: dict) -> List[str]:
    """The training realizations of a trace, oldest first: the training seeds, or for ``train_source: real_past``
    (user 2026-11-04, Real_*) the real Azure intervals before the slice, ``past<n>`` .. ``past1`` (past1 ends where
    the evaluation slice's source interval starts)."""
    if t.get("train_source") == "real_past":
        return [f"past{k}" for k in range(int(cfg["real_past"]["n"]), 0, -1)]
    return [str(s) for s in cfg["train_seeds"]]


def train_file(cfg: dict, out: Path, t: dict, key: str) -> Path:
    if key.startswith("past"):
        return out / "real-past" / t["set"] / key / t["trace"] / f"seed{cfg['eval_seed']}" / "traces_tre.effective.json"
    return trace_dir(out, t["set"], t["trace"], int(key)) / "traces_tre.effective.json"


# ----------------------------------------------------------------------------------------------- realize


def _gen_cmd(profile: str, sset: dict, script: Path, out: Path, seeds: Sequence[int], traces: Sequence[str],
             a) -> List[str]:
    root, az, v1 = Path(a.data_root), Path(a.azure_root), Path(a.v1_gen)
    tre = Path(a.tre_root)
    common = ["--out", str(out), "--v1-gen", str(v1), "--configs-root", str(tre / "loadgen_v1/configs/traces_v14"),
              "--azure-root", str(az), "--v9", str(az / "v1-retrace/traces_v9"),
              "--seeds", *[str(s) for s in seeds], "--processes", str(a.processes)]
    for t in traces:
        common += ["--trace", t]
    if profile == "r3c":
        return [str(script), *common, "--counts-cache", str(root / sset["counts_cache"])]
    if profile == "r3e":
        return [str(script), *common, "--counts-cache", str(root / sset["counts_cache"]),
                "--selection", str(root / sset["selection"])]
    if profile == "ktrace":
        return [str(script), *common, "--k-map", str(root / sset["k_map"]), "--history", "azure",
                "--v1-history-dir", str(root / sset["v1_history_dir"])]
    raise SystemExit(f"unknown generator profile {profile!r}")


def cmd_realize(a) -> int:
    cfg = load_config(a.config)
    root, out = Path(a.data_root), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "_logs").mkdir(exist_ok=True)
    (out / "_run_cwd").mkdir(exist_ok=True)
    record = {"created_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
              "config": cfg["_path"], "config_sha256": cfg["_sha256"], "tool": git_info(HERE),
              "host": platform.node(), "cpus": a.cpus, "nice": a.nice, "sets": {}}
    prefix = ["nice", "-n", str(a.nice)] + (["taskset", "-c", a.cpus] if a.cpus else []) + [sys.executable]
    for set_name, sset in cfg["sets"].items():
        traces = [t["trace"] for t in cfg["traces"] if t["set"] == set_name]
        if not traces:
            continue
        src, gsrc = root / set_name / "_src", out / "_gen_src" / set_name
        if not gsrc.exists():
            shutil.copytree(src, gsrc, ignore=shutil.ignore_patterns("__pycache__"), symlinks=True)
        script = gsrc / sset["script"]
        files = {p.name: sha256_file(p) for p in sorted(gsrc.glob("*.py"))}
        for name, h in files.items():
            if sha256_file(src / name) != h:
                raise SystemExit(f"{set_name}: _src/{name} differs from its copy")
        for t in traces:  # the copy is the generator that wrote the evaluation trace
            man = json.loads((trace_dir(root, set_name, t, cfg["eval_seed"]) / "manifest.json").read_text())
            if man["generator"]["sha256"] != files[sset["script"]]:
                raise SystemExit(f"{set_name}/{t}: evaluation manifest generator sha256 {man['generator']['sha256']} "
                                 f"!= _src/{sset['script']} {files[sset['script']]}")
        rec = {"profile": sset["profile"], "traces": traces, "generator_copy": str(gsrc),
               "generator_sha256": files, "runs": {}}
        for tag, seeds, dst in (("train", cfg["train_seeds"], out / set_name),
                                ("repro", [cfg["eval_seed"]], out / "_repro" / set_name)):
            done = all((trace_dir(dst.parent, dst.name, t, s) / "manifest.json").exists()
                       for t in traces for s in seeds)
            cmd = prefix + _gen_cmd(sset["profile"], sset, script, dst, seeds, traces, a)
            rec["runs"][tag] = {"out": str(dst), "seeds": list(seeds), "argv": cmd}
            if done:
                log(f"{set_name} {tag}: outputs exist, not rerun")
                continue
            logf = out / "_logs" / f"{set_name}.{tag}.log"
            log(f"{set_name} {tag}: {' '.join(cmd)} > {logf}")
            t0 = time.time()
            with open(logf, "w") as fh:
                rc = subprocess.call(cmd, cwd=str(out / "_run_cwd"), stdout=fh, stderr=subprocess.STDOUT)
            rec["runs"][tag].update(rc=rc, seconds=round(time.time() - t0, 1))
            if rc != 0:
                raise SystemExit(f"{set_name} {tag}: generator exit {rc} (see {logf})")
        record["sets"][set_name] = rec
    record["checks"] = verify_realizations(cfg, root, out)
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    write_new(out / "_logs" / f"realize-{stamp}.json", json.dumps(record, indent=1) + "\n")
    bad = [c for c in record["checks"] if not c["equal"]]
    for c in record["checks"]:
        log(f"check {c['what']}: {'equal' if c['equal'] else 'DIFFERENT'}")
    return 2 if bad else 0


def verify_realizations(cfg: dict, root: Path, out: Path) -> List[dict]:
    """The seed-1 rerun equals the evaluation trace (and its pre-t0 history); the design inputs the generator
    rebuilt equal the evaluation set's."""
    checks = []
    for t in cfg["traces"]:
        s, tr, e = t["set"], t["trace"], cfg["eval_seed"]
        rep = trace_dir(out / "_repro", s, tr, e)
        ev = trace_dir(root, s, tr, e)
        for name in ("traces_tre.effective.json", "preserve_history.json"):
            a_, b_ = sha256_file(rep / name), sha256_file(ev / name)
            checks.append({"what": f"repro {s}/{tr}/seed{e}/{name}", "equal": a_ == b_, "rerun": a_, "eval": b_})
        if sha256_file(ev / "traces_tre.effective.json") != t["eval_sha256"]:
            checks.append({"what": f"eval {s}/{tr} sha256 = config", "equal": False})
        for base in (out, out / "_repro"):
            d_new = base / (s + ".design") / "config" / "icse_final" / tr
            d_old = root / (s + ".design") / "config" / "icse_final" / tr
            if not d_new.exists():
                continue
            for p in sorted(d_new.rglob("*")):
                if p.is_file() and not p.is_symlink():
                    rel = p.relative_to(d_new)
                    old = d_old / rel
                    eq = old.exists() and sha256_file(p) == sha256_file(old)
                    checks.append({"what": f"design {base.name}/{s}.design/{tr}/{rel}", "equal": eq})
    return checks


# ----------------------------------------------------------------------------------------------- real past


def used_intervals(root: Path, PHA) -> List[Tuple[str, Optional[str], dt.datetime, dt.datetime]]:
    """Azure intervals already used: the generator's exclusion zones (Real_* v1 slices + pre-slice history, other
    suites, with their guards), the r3c / r3d / r3e Real_* source intervals (no guard) and every history segment
    recorded by a ``preserve_history*.json`` of a trace set under ``root``. (name, family or None = both, a, b)."""
    out = [(n, None, a, b) for n, a, b in PHA.EXCLUDE_BASE]
    out += [(f"r3c {n}", f, PHA._z(st), PHA._z(st) + dt.timedelta(seconds=720)) for n, f, st in PHA.R3C_REAL]
    out += [(f"r3d {n}", f, PHA._z(st), PHA._z(st) + dt.timedelta(seconds=L)) for n, f, st, L in PHA.R3D_REAL]
    out += [(f"r3e {n}", f, PHA._z(st), PHA._z(st) + dt.timedelta(seconds=L)) for n, f, st, L in PHA.R3E_REAL]
    seen = set()
    for p in sorted(root.glob("traces-*/*/seed*/preserve_history*.json")):
        try:
            src = json.loads(p.read_text()).get("source") or {}
        except ValueError:
            continue
        for m, r in (src.get("per_model") or {}).items():
            if isinstance(r, dict) and r.get("segment_start_utc") and r.get("family"):
                key = (r["family"], r["segment_start_utc"], r["segment_end_utc_exclusive"])
                if key not in seen:
                    seen.add(key)
                    out.append((f"history {p.relative_to(root).parts[0]}/{p.parent.parent.name}/{m}", r["family"],
                                PHA._z(r["segment_start_utc"]), PHA._z(r["segment_end_utc_exclusive"])))
    return out


def _past_unit(args) -> dict:
    """One past interval of one trace through the r3e generator (forked; ``run_unit`` chdirs / edits sys.path)."""
    gsrc, trace, seed, a = args
    sys.path.insert(0, gsrc)
    import gen_r3e as G  # noqa: E402

    return G.run_unit((trace, seed, a))


def cmd_realize_real_past(a) -> int:
    """Real_* (user 2026-11-04): train on the real Azure past of the slice. Per model, the n intervals right before
    the slice's own source interval (same length, contiguous going back; an interval that overlaps a used one moves
    back on the 60 s grid to the nearest earlier free position), each through the r3e transform of the slice
    (``r3e_select.materialize``: x8 compression, 121 s envelope, plateau-clip min-max to the model's lo..hi target x
    the slice's beta, tok70 lengths, thinning / copies) and the r3e generator (``gen_r3e.run_unit``, the slice's
    seed), into ``<out>/real-past/<set>/past<k>/``."""
    cfg = load_config(a.config)
    root, out = Path(a.data_root), Path(a.out)
    rp = cfg["real_past"]
    set_name = rp["set"]
    sset = cfg["sets"][set_name]
    gsrc = out / "_gen_src" / set_name
    for name in ("gen_r3e.py", "r3e_select.py", "preserve_history_azure.py"):
        if sha256_file(gsrc / name) != sha256_file(root / set_name / "_src" / name):
            raise SystemExit(f"{gsrc / name} differs from the set's _src copy")
    sys.path.insert(0, str(gsrc))
    import gen_r3e as G  # noqa: E402
    import preserve_history_azure as PHA  # noqa: E402
    import r3e_select as RS  # noqa: E402

    sel = json.loads((root / sset["selection"]).read_text())
    traces = [t for t in cfg["traces"] if t.get("train_source") == "real_past"]
    fams = {G.REAL[t["trace"]] for t in traces}
    counts = {f: np.load(root / sset["counts_cache"] / f"{f}_counts.npy") for f in fams}
    used = used_intervals(root, PHA)
    L, n, grid = RS.C * RS.DUR, int(rp["n"]), int(RS.GRID_S)
    plan, intervals = {}, []
    for t in traces:
        tr = t["trace"]
        fam = G.REAL[tr]
        r = sel["selection"][fam]
        for m in G.MODELS:
            end = int(r["chosen"][m]["start_s"])
            for k in range(1, n + 1):
                st, skipped = end - L, []
                while True:
                    a0 = RS.START[fam] + dt.timedelta(seconds=st)
                    a1 = a0 + dt.timedelta(seconds=L)
                    hit = [nm for nm, f, u0, u1 in used if (f is None or f == fam) and a0 < u1 and u0 < a1]
                    if not hit:
                        break
                    skipped += [h for h in hit if h not in skipped]
                    st -= grid
                if st < 0:
                    raise SystemExit(f"{tr}/{m} past{k}: before the start of the week")
                u = RS.clip_u(RS.envelope(RS.compress(counts[fam], st)))
                hl, hs = RS.hot_block(u)
                x = {"start_s": st, "hot_len": hl, "hot_start": hs, "head": float(u[:60].mean())}
                plan[(tr, m, k)] = (fam, x, r["beta"])
                intervals.append({"trace": tr, "model": m, "key": f"past{k}", "family": fam, "start_s": st,
                                  "start_utc": a0.isoformat(), "end_utc": a1.isoformat(),
                                  "shifted_s": end - L - st, "collisions_skipped": skipped,
                                  "slice_start_utc": r["chosen"][m]["start_utc"]})
                used.append((f"real-past {tr}/{m}/past{k}", fam, a0, a1))  # the pasts of one family never overlap
                end = st
    v1 = Path(a.v1_gen)
    git = lambda *x: subprocess.run(["git", "-C", str(v1), *x], capture_output=True, text=True).stdout.strip()
    v1_prov = {"repo": str(v1), "git_sha": git("rev-parse", "HEAD"), "dirty_src": git("status", "--porcelain", "--", "src"),
               **{f"{nm}_sha256": sha256_file(v1 / "src" / nm) for nm in
                  ("trace_generator.py", "data_generator.py", "real_trace_adapter.py", "config_manager.py")}}
    units = []
    for k in range(1, n + 1):
        base = out / "real-past" / set_name
        design = base / f"past{k}.design"
        (design / "config" / "icse_final").mkdir(parents=True, exist_ok=True)
        if not (design / "src").exists():
            os.symlink(v1 / "src", design / "src")
        for t in traces:
            tr = t["trace"]
            dd = design / "config" / "icse_final" / tr
            (dd / "real_traces").mkdir(parents=True, exist_ok=True)
            if not (dd / "config.yaml").exists():
                os.symlink(v1 / "config" / "icse_final" / tr / "config.yaml", dd / "config.yaml")
            prov = {}
            for m in G.MODELS:
                fam, x, beta = plan[(tr, m, k)]
                rows, pv = RS.materialize(fam, m, x, beta, Path(a.azure_root), counts[fam])
                f = dd / "real_traces" / f"{m}.json"
                f.write_text(json.dumps(rows) + "\n")
                prov[m] = {**pv, "design_list_sha256": sha256_file(f)}
                log(f"{tr}/{m} past{k}: {pv['interval_start_utc']} {pv['requests']} requests")
            mk = json.loads((root / (set_name + ".design") / "config" / "icse_final" / tr / "r3e.json").read_text())
            mk.update(per_model=prov, real_past={"k": k, "rule": cmd_realize_real_past.__doc__})
            (dd / "r3e.json").write_text(json.dumps(mk, indent=1, default=float) + "\n")
            if not (base / f"past{k}" / tr / f"seed{cfg['eval_seed']}" / "manifest.json").exists():
                ua = {"design": str(design), "out": str(base / f"past{k}"), "v1_gen": str(v1),
                      "v9": str(Path(a.azure_root) / "v1-retrace/traces_v9"), "azure_root": str(a.azure_root),
                      "me": sha256_file(gsrc / "gen_r3e.py"), "v1_prov": v1_prov}
                units.append((str(gsrc), tr, int(cfg["eval_seed"]), ua))
    t0 = time.time()
    with mp.get_context("fork").Pool(a.processes, maxtasksperchild=1) as pool:
        for res in pool.imap_unordered(_past_unit, units):
            log(f"generated {res}")
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    rec = {"created_utc": stamp, "tool": git_info(HERE), "host": platform.node(), "set": set_name,
           "seed": int(cfg["eval_seed"]), "rule": cmd_realize_real_past.__doc__.strip(),
           "used_intervals": [[nm, f, u0.isoformat(), u1.isoformat()] for nm, f, u0, u1 in used
                              if not nm.startswith("real-past")],
           "intervals": intervals, "generator_sha256": {nm: sha256_file(gsrc / nm) for nm in
                                                        ("gen_r3e.py", "r3e_select.py", "preserve_history_azure.py")},
           "seconds": round(time.time() - t0, 1)}
    write_new(out / "_logs" / f"real-past-{stamp}.json", json.dumps(rec, indent=1) + "\n")
    for it in intervals:
        log(f"{it['trace']}/{it['model']} {it['key']}: {it['start_utc']} .. {it['end_utc']} shifted {it['shifted_s']} s "
            f"skipped {it['collisions_skipped']}")
    return 0


# ----------------------------------------------------------------------------------------------- series


def loadgen_caps(tre_root: Path, rel: str, trace: str) -> Dict[str, int]:
    import yaml

    c = yaml.safe_load((tre_root / rel / trace / "config.yaml").read_text())["custom_load_test"]
    return {m["name"]: int(m["max_tokens"]) for m in c["models"]}


def window_series(records: Sequence[dict], models: Sequence[str], window_s: float, n_windows: int,
                  caps: Dict[str, int]) -> Tuple[Dict[str, Dict[str, List[int]]], dict]:
    """{model: {"P": [...], "D": [...], "N": [...]}} by arrival, windows of ``window_s`` from the first arrival;
    requests at or after ``n_windows`` windows are counted (``after_last_window``), not binned."""
    t0 = min(float(r["timestamp"]) for r in records)
    out = {m: {"P": [0] * n_windows, "D": [0] * n_windows, "N": [0] * n_windows} for m in models}
    nulls, late = 0, 0
    for r in records:
        m = r["model_name"]
        i = int(math.floor((float(r["timestamp"]) - t0) / window_s))
        if i >= n_windows:
            late += 1
            continue
        mo = r.get("max_output_tokens")
        if mo is None:
            mo, nulls = caps[m], nulls + 1
        out[m]["P"][i] += int(r["prompt_length"])
        out[m]["D"][i] += int(mo)
        out[m]["N"][i] += 1
    # the synthetic generators emit a 721st second: its requests fall in a 1 s partial window n_windows, left out
    # (a 1 s window inside a 10 s series is not a window of the service)
    return out, {"t0_s": t0, "requests": len(records), "null_max_output_tokens": nulls,
                 "after_last_window": late}


def cmd_series(a) -> int:
    cfg = load_config(a.config)
    root, out, tre = Path(a.data_root), Path(a.out), Path(a.tre_root)
    W, n = float(cfg["window_s"]), int(round(cfg["duration_s"] / cfg["window_s"]))
    doc = {"format": "preserve-train-trace-series-v1", "window_s": W, "n_windows": n, "models": cfg["models"],
           "config_sha256": cfg["_sha256"], "tool": git_info(HERE),
           "rule": "per model: P = sum prompt_length, D = sum max_output_tokens (null -> loadgen config max_tokens) "
                   "of the requests whose arrival falls in window i = floor((timestamp - t0) / W), t0 = the "
                   "trace's first arrival (the policy's t0 = the first gateway arrival), i < duration / W; the "
                   "synthetic generators' 721st second (a 1 s partial window, after_last_window) is left out",
           "traces": {}}
    for t in cfg["traces"]:
        s, tr = t["set"], t["trace"]
        caps = loadgen_caps(tre, cfg["loadgen_configs"], tr)
        rows = {}
        keys = train_keys(cfg, t)
        for key in [str(cfg["eval_seed"]), *keys]:
            p = (trace_dir(root, s, tr, cfg["eval_seed"]) / "traces_tre.effective.json" if key == str(cfg["eval_seed"])
                 else train_file(cfg, out, t, key))
            recs = json.loads(p.read_text())
            ser, meta = window_series(recs, cfg["models"], W, n, caps)
            rows[key] = {"file": str(p), "sha256": sha256_file(p), **meta, "series": ser}
            log(f"series {tr} {key}: {meta['requests']} requests, nulls {meta['null_max_output_tokens']}")
        doc["traces"][tr] = {"set": s, "eval_seed": cfg["eval_seed"], "train_keys": keys,
                             "train_source": t.get("train_source", "seeds"), "loadgen_max_tokens": caps, "seeds": rows}
    doc["variability"] = variability(doc, cfg)
    write_new(out / f"series_w{W:g}.json", json.dumps(doc) + "\n")
    log(f"wrote {out / f'series_w{W:g}.json'}")
    write_sources(cfg, root, out, doc)
    return 0


def write_sources(cfg: dict, root: Path, out: Path, doc: dict) -> None:
    """``<out>/SOURCES.md`` + ``<out>/realizations.sha256`` (paths relative to ``out``)."""
    recs = sorted((out / "_logs").glob("realize-*.json"))
    if not recs:
        raise SystemExit("no realize record in _logs/")
    rz = json.loads(recs[-1].read_text())
    sums = []
    L = [f"# {out.name}: training realizations of the PreServe evaluation traces\n",
         "Made by `tre/baselines/preserve_train/trace_train.py` (aibrix " + str(rz["tool"].get("sha")) +
         f", config `{Path(cfg['_path']).name}` sha256 `{cfg['_sha256']}`). Not in git. Each trace's own generator "
         "(a copy of its evaluation set's `_src/`, in `_gen_src/<set>/`; the script's sha256 equals the one in the "
         "evaluation manifest) and spec, run for the training seeds "
         f"{cfg['train_seeds']} (no evaluation uses them; evaluation seed {cfg['eval_seed']}) into `<set>/<trace>/seed<s>/`, "
         "and for the evaluation seed into `_repro/` (reproduction check only, never trained on). Host "
         f"{rz['host']}, nice {rz['nice']}, taskset {rz['cpus'] or '-'}; record `_logs/{recs[-1].name}`.\n",
         "## Generators\n", "| set | profile | script | sha256 | argv (train) |", "|---|---|---|---|---|"]
    for sn, r in rz["sets"].items():
        sc = cfg["sets"][sn]["script"]
        L.append(f"| {sn} | {r['profile']} | `_gen_src/{sn}/{sc}` | `{r['generator_sha256'][sc]}` | "
                 f"`{' '.join(r['runs']['train']['argv'])}` |")
    L += ["", "## Checks (seed-1 rerun vs the evaluation files; rebuilt design inputs vs the evaluation set's)\n",
          "| check | equal |", "|---|---|"]
    L += [f"| {c['what']} | {c['equal']} |" for c in rz["checks"]]
    rp = sorted((out / "_logs").glob("real-past-*.json"))
    if rp:
        r = json.loads(rp[-1].read_text())
        L += ["", "## Real_* training source: the real Azure intervals before the slice (user 2026-11-04)\n",
              r["rule"], "", f"Record `_logs/{rp[-1].name}`. Generator: `_gen_src/{r['set']}/` "
              f"(`r3e_select.materialize` per interval, then `gen_r3e.run_unit` seed {r['seed']}).\n",
              "| trace | model | key | source interval (UTC) | shifted back (s) | overlaps |", "|---|---|---|---|---:|---|"]
        for it in r["intervals"]:
            L.append(f"| {it['trace']} | {it['model']} | {it['key']} | {it['start_utc']} .. {it['end_utc']} | "
                     f"{it['shifted_s']} | {', '.join(it['collisions_skipped']) or 'none'} |")
    L += ["", "## Realizations\n", "| trace | key | requests | traces_tre.effective.json sha256 |", "|---|---|---:|---|"]
    for t in cfg["traces"]:
        for sd in doc["traces"][t["trace"]]["train_keys"]:
            rec = doc["traces"][t["trace"]]["seeds"][sd]
            rel = Path(rec["file"]).relative_to(out)
            L.append(f"| {t['trace']} | {sd} | {rec['requests']} | `{rec['sha256']}` |")
            sums.append(f"{rec['sha256']}  {rel}\n")
            for extra in ("manifest.json", "preserve_history.json"):
                sums.append(f"{sha256_file(out / rel.parent / extra)}  {rel.parent / extra}\n")
    L += ["", f"Series (`series_w{doc['window_s']:g}.json`): {doc['rule']}.", "",
          "`realizations.sha256`: every training file (check with `sha256sum -c` from this directory). The "
          "generators' own `preserve_history.json` next to each realization is not used (the history before t0 of "
          "the evaluation is the tail of the training series, `history/`).", ""]
    write_new(out / "realizations.sha256", "".join(sums))
    (out / "SOURCES.md").write_text("\n".join(L) + "\n")


def variability(doc: dict, cfg: dict) -> dict:
    """How far each training realization is from the evaluation one: mean over windows of |x_s - x_eval| / x_eval
    (x_eval > 0), %, per (trace, model, P|D), averaged over the training seeds; and the max over seeds."""
    out = {}
    for tr, td in doc["traces"].items():
        ev = td["seeds"][str(cfg["eval_seed"])]["series"]
        for m in cfg["models"]:
            for k in ("P", "D"):
                y = np.asarray(ev[m][k], dtype=float)
                vals = []
                for s in td["train_keys"]:
                    x = np.asarray(td["seeds"][str(s)]["series"][m][k], dtype=float)
                    ok = y > 0
                    vals.append(float(np.mean(np.abs(x[ok] - y[ok]) / y[ok]) * 100) if ok.any() else float("nan"))
                out.setdefault(tr, {}).setdefault(m, {})[k] = {"mean_pct": round(float(np.mean(vals)), 3),
                                                               "max_pct": round(float(np.max(vals)), 3)}
    return out


def training_series(doc: dict, cfg: dict) -> Tuple[dict, dict]:
    """V1 {(trace, model): {"P","D"}} (4 realizations in seed order) and V2 {model: {"P","D"}} (traces in config
    order, each in seed order), plus the V2 segment record."""
    v1, v2, segs = {}, {m: {"P": [], "D": []} for m in cfg["models"]}, []
    for t in cfg["traces"]:
        tr = t["trace"]
        td = doc["traces"][tr]
        for m in cfg["models"]:
            cat = {k: sum((td["seeds"][s]["series"][m][k] for s in td["train_keys"]), []) for k in ("P", "D")}
            v1[(tr, m)] = cat
        start = len(v2[cfg["models"][0]]["P"])
        for m in cfg["models"]:
            for k in ("P", "D"):
                v2[m][k] += v1[(tr, m)][k]
        segs.append({"trace": tr, "seeds": td["train_keys"], "windows": [start, len(v2[cfg["models"][0]]["P"])]})
    return v1, {"series": v2, "segments": segs}


# ----------------------------------------------------------------------------------------------- train


def minmax_xy(series: Sequence[float], seq_len: int):
    x = np.asarray(series, dtype=np.float64)
    mn, mx = float(x.min()), float(x.max())
    if not mx > mn:
        raise SystemExit("constant training series: min-max is undefined")
    t = np.arange(seq_len, len(x))
    idx = t[:, None] + np.arange(-seq_len, 0)[None, :]
    X, y = x[idx], x[t]
    return (X - mn) / (mx - mn), (y - mn) / (mx - mn), mn, mx


def _train_job(job: dict) -> dict:
    import torch

    torch.set_num_threads(job["threads"])
    from preserve_train.model import train_mlstm

    losses: List[str] = []
    t0 = time.time()
    model = train_mlstm(job["X"], job["y"], hidden_dim=job["hidden"], num_epochs=job["epochs"],
                        batch_size=job["batch_size"], lr=job["lr"], seed=job["seed"], log=losses.append)
    sd = {k: v.detach().cpu().numpy() for k, v in model.state_dict().items()}
    return {"id": job["id"], "state_dict": sd, "seconds": time.time() - t0, "loss_log": losses}


def cmd_train(a) -> int:
    import torch

    from preserve_train.export import export_npz
    from preserve_train.model import forward_batch
    from preserve_train.train_mlstm import model_from_state
    from tre_baselines.policies import preserve_predictor as pp

    cfg = load_config(a.config)
    hp, L, W = cfg["hyper"], int(cfg["seq_len"]), float(cfg["window_s"])
    if hp["normalization"] != pp.MINMAX:
        raise SystemExit("this trainer ships min-max weights only")
    doc = json.loads(Path(a.series).read_text())
    if float(doc["window_s"]) != W:
        raise SystemExit("series window != config window")
    v1, v2 = training_series(doc, cfg)
    td = Path(a.train_dir)
    td.mkdir(parents=True, exist_ok=True)
    units = {}  # unit id -> (variant, trace or "*", model, {"P","D"} series)
    for (tr, m), ser in v1.items():
        units[f"v1/{tr}/{m}"] = ("v1", tr, m, ser)
    for m, ser in v2["series"].items():
        units[f"v2/concat/{m}"] = ("v2", "*", m, ser)
    jobs, scal = [], {}
    for uid, (_v, _tr, _m, ser) in units.items():
        for kind, key in KINDS.items():
            X, y, mn, mx = minmax_xy(ser[key], L)
            scal[f"{uid}/{kind}"] = (mn, mx)
            jobs.append({"id": f"{uid}/{kind}", "X": X, "y": y, "hidden": hp["hidden"], "epochs": hp["epochs"],
                         "batch_size": hp["batch_size"], "lr": hp["lr"], "seed": hp["seed"], "threads": a.threads})
    log(f"training {len(jobs)} models ({len(units)} predictors), {a.jobs} jobs x {a.threads} threads")
    torch.set_num_threads(max(1, min(8, a.jobs * a.threads)))
    res = {}
    with cf.ProcessPoolExecutor(max_workers=a.jobs, mp_context=mp.get_context("spawn")) as ex:
        for r in ex.map(_train_job, jobs):
            res[r["id"]] = r
            log(f"done {r['id']} in {r['seconds']:.0f}s; {r['loss_log'][-1] if r['loss_log'] else ''}")
    files, check = {}, {}
    for uid, (variant, tr, m, ser) in units.items():
        models = {k: model_from_state(res[f"{uid}/{k}"]["state_dict"], hp["hidden"]) for k in KINDS}
        name = f"{tr}.{m}.npz" if variant == "v1" else f"concat.{m}.npz"
        path = td / variant / name
        path.parent.mkdir(parents=True, exist_ok=True)
        export_npz(models, str(path), seq_len=L, hidden=hp["hidden"], eps=1.0, window_s=W,
                   normalization=pp.MINMAX, minmax={k: scal[f"{uid}/{k}"] for k in KINDS},
                   meta={"model": m, "trace": tr, "variant": variant})
        preds = pp.load_predictors(str(path))
        worst = 0.0
        for kind, key in KINDS.items():  # numpy (policy) vs torch on every training sequence
            X, _, mn, mx = minmax_xy(ser[key], L)
            p_t = forward_batch(models[kind], X) * (mx - mn) + mn
            raw = np.asarray(ser[key], dtype=float)
            for j in range(0, len(X), max(1, len(X) // 50)):
                p_n = preds[kind].predict(raw[j:j + L])
                worst = max(worst, abs(p_n - max(0.0, p_t[j])) / max(abs(p_t[j]), 1.0))
        files[uid] = {"variant": variant, "trace": tr, "model": m, "file": f"{variant}/{name}",
                      "sha256": sha256_file(path), "train_windows": len(ser["P"]),
                      "train_sequences": len(ser["P"]) - L,
                      "minmax": {k: {"min": scal[f"{uid}/{k}"][0], "max": scal[f"{uid}/{k}"][1]} for k in KINDS},
                      "numpy_vs_torch_max_rel_diff": worst}
        check[uid] = worst
        log(f"exported {path} (numpy vs torch {worst:.1e})")
    man = {"created_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
           "code": git_info(HERE), "argv": sys.argv, "host": platform.node(),
           "env": {"python": platform.python_version(), "torch": torch.__version__, "numpy": np.__version__,
                   "host_cpu_count": os.cpu_count()},
           "config": cfg["_path"], "config_sha256": cfg["_sha256"],
           "series": str(a.series), "series_sha256": sha256_file(a.series),
           "window_s": W, "seq_len": L, "hyper": hp, "v2_segments": v2["segments"],
           "training": {"seconds": {k: round(r["seconds"], 1) for k, r in res.items()},
                        "last_loss_log": {k: (r["loss_log"][-1] if r["loss_log"] else None) for k, r in res.items()}},
           "files": files}
    (td / "train_manifest.json").write_text(json.dumps(man, indent=1) + "\n")
    log(f"wrote {td / 'train_manifest.json'}; max numpy vs torch {max(check.values()):.1e}")
    return 0


# ----------------------------------------------------------------------------------------------- finish


def history_doc(cfg: dict, doc: dict, trace: str, variant: str) -> dict:
    """``tre-preserve-history-v1`` for the evaluation seed: the last seq_len windows of the trace's training
    series = the end of the last training realization (seed order)."""
    L, W = int(cfg["seq_len"]), float(cfg["window_s"])
    td = doc["traces"][trace]
    last = td["train_keys"][-1]
    rec = td["seeds"][last]
    n = int(doc["n_windows"])
    if n < L:
        raise SystemExit("a realization is shorter than seq_len")
    rows = {m: [[rec["series"][m]["P"][i], rec["series"][m]["D"][i]] for i in range(n - L, n)]
            for m in cfg["models"]}
    return {"format": "tre-preserve-history-v1", "trace": trace, "seed": int(cfg["eval_seed"]), "window_s": W,
            "models": rows,
            "source": {"method": HISTORY_METHOD, "variant": variant,
                       "rule": "the last seq_len windows of the trace's training series (its training realizations "
                               "concatenated oldest first) = windows [n - seq_len, n) of the last one (synthetic: the "
                               "last training seed; Real_*: past1, the real Azure interval right before the slice's "
                               "source interval, i.e. the true pre-context); the evaluation realization is the next one",
                       "realization": {"set": td["set"], "trace": trace, "key": last, "file": rec["file"],
                                       "sha256": rec["sha256"], "windows": [n - L, n]},
                       "series_rule": doc["rule"], "tool": "tre/baselines/preserve_train/trace_train.py",
                       "tool_git": git_info(HERE)}}


def mape(y, p) -> float:
    y, p = np.asarray(y, float), np.asarray(p, float)
    ok = y > 0
    return float(np.mean(np.abs(y[ok] - p[ok]) / y[ok]) * 100) if ok.any() else float("nan")


def one_step(pred, hist: Sequence[float], ev: Sequence[float], L: int) -> List[float]:
    """Prediction of every evaluation window i from the last L of history + ev[:i] (the policy's Tier-1:
    observed windows enter the history, predictions never do; clamped at 0 as the policy does)."""
    seq = list(map(float, hist)) + list(map(float, ev))
    h = len(hist)
    return [max(0.0, float(pred.predict(seq[h + i - L:h + i]))) for i in range(len(ev))]


def two_step(pred, hist: Sequence[float], ev: Sequence[float], L: int) -> List[Optional[float]]:
    """Two-step forecast (paper Alg. 2, what the policy uses since the Tier-1 two-step branch): at the start of
    window j - 1 the observed windows are those < j - 1; predict j - 1, append that prediction, predict j.
    Entry j is the forecast of evaluation window j; j = 0 has none (it would be made before t0)."""
    seq = list(map(float, hist)) + list(map(float, ev))
    h = len(hist)
    out: List[Optional[float]] = [None]
    for j in range(1, len(ev)):
        obs = seq[:h + j - 1]
        p1 = max(0.0, float(pred.predict(obs[-L:])))
        out.append(max(0.0, float(pred.predict((obs + [p1])[-L:]))))
    return out


def _scores(y: Dict[str, list], pP: list, pD: list) -> dict:
    keep = [j for j in range(len(pP)) if pP[j] is not None]
    yP, yD = [y["P"][j] for j in keep], [y["D"][j] for j in keep]
    qP, qD = [pP[j] for j in keep], [pD[j] for j in keep]
    return {"P": mape(yP, qP), "D": mape(yD, qD), "total": mape(np.add(yP, yD), np.add(qP, qD))}


def evaluate(cfg: dict, doc: dict, root: Path, weights: Path, hist_new: dict, current: dict, pp) -> dict:
    L = int(cfg["seq_len"])
    W = float(cfg["window_s"])
    from tre_baselines.policies import preserve_history as ph
    from tre_baselines.policies.preserve import predictor_set_for

    cur = cfg["current"]
    out = {}
    for t in cfg["traces"]:
        tr, s = t["trace"], t["set"]
        ev = doc["traces"][tr]["seeds"][str(cfg["eval_seed"])]["series"]
        cur_hist_path = trace_dir(root, s, tr, cfg["eval_seed"]) / ph.history_names(W)[-1]
        cur_hist, cur_info = ph.load(str(cur_hist_path), window_s=W, models=cfg["models"], seq_len=L, trace=tr,
                                     seed=cfg["eval_seed"])
        cur_set = predictor_set_for(tr, cur["rules"], cur["default"])
        new_hist = hist_new[tr]["models"]
        for m in cfg["models"]:
            y = {k: ev[m][k] for k in ("P", "D")}
            hn = {k: [r[i] for r in new_hist[m][-L:]] for i, k in enumerate(("P", "D"))}
            hc = {k: [r[i] for r in cur_hist[m]] for i, k in enumerate(("P", "D"))}
            res = {}
            for name, preds, hist in (("V1", weights[("v1", tr, m)], hn), ("V2", weights[("v2", "*", m)], hn),
                                      (f"current:{cur_set}", current[cur_set], hc)):
                key = name.split(":")[0]
                res[key] = _scores(y, one_step(preds["prompt"], hist["P"], y["P"], L),
                                   one_step(preds["response"], hist["D"], y["D"], L))
                res[key + "_2step"] = _scores(y, two_step(preds["prompt"], hist["P"], y["P"], L),
                                              two_step(preds["response"], hist["D"], y["D"], L))
            seqP, seqD = hn["P"] + y["P"], hn["D"] + y["D"]
            h = len(hn["P"])
            res["naive"] = _scores(y, seqP[h - 1:h - 1 + len(y["P"])], seqD[h - 1:h - 1 + len(y["D"])])
            # two-step naive: at the start of window j - 1 the last observed window is j - 2
            res["naive_2step"] = _scores(y, [None] + seqP[h - 1:h - 2 + len(y["P"])],
                                         [None] + seqD[h - 1:h - 2 + len(y["D"])])
            res["targets"] = int(sum(1 for v in np.add(y["P"], y["D"]) if v > 0))
            res["current_set"] = cur_set
            res["current_history"] = {"path": str(cur_hist_path), "sha256": cur_info["sha256"]}
            out.setdefault(tr, {})[m] = res
    return out


def cmd_finish(a) -> int:
    from tre_baselines.policies import preserve_history as ph
    from tre_baselines.policies import preserve_predictor as pp

    cfg = load_config(a.config)
    root, out, wout = Path(a.data_root), Path(a.out), Path(a.weights_out)
    L, W = int(cfg["seq_len"]), float(cfg["window_s"])
    doc = json.loads(Path(a.series).read_text())
    tman = json.loads((Path(a.train_dir) / "train_manifest.json").read_text())
    if tman["series_sha256"] != sha256_file(a.series) or tman["config_sha256"] != cfg["_sha256"]:
        raise SystemExit("the training run used another series / config")
    # weights: copy, check they load, declare W / min-max / their model
    weights = {}
    for uid, f in tman["files"].items():
        src, dst = Path(a.train_dir) / f["file"], wout / f["file"]
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists() and sha256_file(dst) != f["sha256"]:
            raise SystemExit(f"{dst} exists with another sha256")
        if not dst.exists():
            shutil.copyfile(src, dst)
        preds = pp.load_predictors(str(dst), f["sha256"])
        for pr in preds.values():
            if pr.window_s != W or pr.normalization != pp.MINMAX or pr.model != f["model"]:
                raise SystemExit(f"{dst}: declares window {pr.window_s}, {pr.normalization}, model {pr.model}")
        weights[(f["variant"], f["trace"], f["model"])] = preds
    cur = cfg["current"]
    current = {k: pp.load_predictors(str(BASELINES / cur["weights_dir"] / v)) for k, v in cur["files"].items()}
    # histories
    variant = cfg["history_variant"]
    name = ph.history_names(W, variant=variant)[-1]
    hist, hist_files = {}, []
    for t in cfg["traces"]:
        hd = history_doc(cfg, doc, t["trace"], variant)
        ph.parse(hd, window_s=W, models=cfg["models"], seq_len=L, trace=t["trace"], seed=cfg["eval_seed"])
        rel = Path(t["set"]) / t["trace"] / f"seed{cfg['eval_seed']}" / name
        p = out / "history" / rel
        if p.exists():  # keep an existing file (it records its own tool sha); it must hold the same rows
            old = json.loads(p.read_text())
            if old["models"] != hd["models"] or old["source"]["realization"] != hd["source"]["realization"]:
                raise SystemExit(f"{p} exists with other rows")
        else:
            write_new(p, json.dumps(hd, indent=1) + "\n")
        hist[t["trace"]] = json.loads(p.read_text())
        hist_files.append((sha256_file(p), str(Path("history") / rel)))
    ev = evaluate(cfg, doc, root, weights, hist, current, pp)
    man = {"created_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
           "finish": {"code": git_info(HERE), "argv": sys.argv, "host": platform.node()},
           "trainer": {k: tman[k] for k in ("created_utc", "code", "argv", "host", "env", "training")},
           "config": {"path": "tre/baselines/preserve_train/configs/" + Path(cfg["_path"]).name,
                      "sha256": cfg["_sha256"]},
           "data_root": str(root), "out": str(out),
           "series": {"path": str(a.series), "sha256": tman["series_sha256"]},
           "window_s": W, "seq_len": L, "hyper": cfg["hyper"], "eval_seed": cfg["eval_seed"],
           "train_seeds": cfg["train_seeds"], "v2_segments": tman["v2_segments"],
           "normalization": {"name": pp.MINMAX, "rule": "paper Alg. 1: x' = (x - min) / (max - min), min / max over "
                             "the predictor's training series, stored per kind in the npz (<kind>.norm_min / "
                             "norm_max); prediction = max(0, out x (max - min) + min)"},
           "files": tman["files"],
           "history": {"variant": variant, "name": name, "method": HISTORY_METHOD,
                       "files": [{"path": p, "sha256": h} for h, p in hist_files]},
           "eval": ev, "variability": doc["variability"],
           "real_past": _real_past_record(out)}
    (wout / "manifest.json").write_text(json.dumps(man, indent=1) + "\n")
    (wout / "REPORT.md").write_text(report(man, cfg))
    if a.history_sha256_out:
        Path(a.history_sha256_out).write_text("".join(f"{h}  {p}\n" for h, p in hist_files))
    log(f"wrote {wout / 'manifest.json'}, REPORT.md")
    return 0


def _real_past_record(out: Path) -> Optional[dict]:
    rp = sorted((out / "_logs").glob("real-past-*.json"))
    if not rp:
        return None
    r = json.loads(rp[-1].read_text())
    return {"record": str(rp[-1]), "rule": r["rule"], "seed": r["seed"], "intervals": r["intervals"],
            "generator_sha256": r["generator_sha256"]}


def _f(x) -> str:
    return "-" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.1f}"


def report(man: dict, cfg: dict) -> str:
    ev = man["eval"]
    L = [f"# PreServe mLSTM predictors trained on each trace's own past, W = {man['window_s']:g} s\n",
         f"Created {man['created_utc']} by `preserve_train/trace_train.py finish` (aibrix "
         f"{man['finish']['code'].get('sha')}, dirty {man['finish']['code'].get('dirty')}); trained "
         f"{man['trainer']['created_utc']} on {man['trainer']['host']} (aibrix {man['trainer']['code'].get('sha')}, "
         f"dirty {man['trainer']['code'].get('dirty')}), torch {man['trainer']['env']['torch']}, numpy "
         f"{man['trainer']['env']['numpy']}, CPU. Config `{man['config']['path']}` (sha256 `{man['config']['sha256']}`).\n",
         "## What\n",
         "The paper trains a service's predictor on that service's own past (section 5.3 RQ2: first two days -> "
         "next two). Here the past of an evaluation trace = independent realizations of the same service made by "
         "the trace's own generator and spec with seeds no evaluation uses "
         f"({', '.join(map(str, man['train_seeds']))}; evaluation seed {man['eval_seed']} is never trained on): "
         f"4 x 72 windows of {man['window_s']:g} s = 288 windows (paper: 2 days of 10 min windows = 288).\n",
         "- **V1 per trace** (`v1/<trace>.<model>.npz`, 21 files): training series = the trace's 4 realizations "
         "concatenated in seed order (288 windows, 263 sequences).",
         "- **Real_* (user 2026-11-04)**: V1 of Real_conv / Real_code is trained on the REAL past of the slice "
         "instead of seed realizations: per model the 4 Azure intervals right before the slice's own source "
         "interval, each through the slice's r3e transform and generator (table below); training order past4 .. "
         "past1, so the history before t0 (end of the training series) is the true pre-context of the slice."
         if man.get("real_past") else "",
         "- **V2 concat** (`v2/concat.<model>.npz`, 3 files): training series = all 7 traces concatenated "
         "(config order: " + ", ".join(s["trace"] for s in man["v2_segments"]) + "; each trace's realizations in "
         "seed order; 2016 windows; sequences run across trace boundaries).",
         f"- Hyper-parameters: the 11-03 protocol (seq_len {man['seq_len']}, hidden {man['hyper']['hidden']}, "
         f"{man['hyper']['epochs']} epochs, batch {man['hyper']['batch_size']}, Adam {man['hyper']['lr']}, MSE, "
         f"seed {man['hyper']['seed']}) except normalization = **min-max (paper Alg. 1)** over the training series; "
         "min / max are stored in the file (`<kind>.norm_min`, `<kind>.norm_max`, `normalization` = `minmax`) and "
         "applied at inference; the policy refuses a file whose normalization differs from "
         "`params.predictor_normalization` (default `per_sequence_mean`, the 11-03 weights).",
         f"- History before t0: `{man['history']['name']}` (policy `history_variant: {man['history']['variant']}`), "
         f"method `{man['history']['method']}`: the last {man['seq_len']} windows of the trace's training series "
         "(= the end of the last training realization; the evaluation realization is the next one). Files under "
         f"`{man['out']}/history/` (sha256 in manifest.json).\n",
         "## One-step-ahead MAPE (%) on the evaluation realization (seed 1)\n",
         "Every window i of the 72 is predicted from the last 25 of [history before t0, observed windows < i] (the "
         "policy's Tier-1: observed windows enter the history, predictions never do; prediction clamped at 0). "
         "MAPE = mean |y - yhat| / y over windows with y > 0. `current` = the shipped "
         "`preserve-mlstm-20261103-paper-w10` set the policy picks (Real_code -> code, Real_conv -> conv, else "
         "pooled; per_sequence_mean) with the trace's current `preserve_history_w10.json`; `naive` = the previous "
         "window (window 0: the last row of the new history). Series: P = prompt_length, D = max_output_tokens "
         "(null = loadgen config max_tokens) by arrival, windows from the first arrival.\n",
         "`*_2step` = the two-step forecast of window j made at the start of window j - 1 (observed windows "
         "< j - 1, predict j - 1, append it, predict j: paper Alg. 2, what the policy's Tier-1 now uses to set "
         "N_{i+1}); scored on windows 1..71 (window 0 would be forecast before t0). Naive two-step = window j - 2.\n",
         "D here is the request's max_output_tokens: under the runs' ignore_eos the actual output tokens equal "
         "max_tokens, so this is also the paper's D (completed output tokens attributed to the arrival window).\n",
         "### Total (P + D) per trace, mean over the 3 models\n",
         "| trace | V1 | V2 | current | naive | V1 2-step | V2 2-step | current 2-step | naive 2-step |",
         "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    cols = ("V1", "V2", "current", "naive", "V1_2step", "V2_2step", "current_2step", "naive_2step")
    for tr, per in ev.items():
        row = [float(np.nanmean([per[m][v]["total"] for m in per])) for v in cols]
        L.append(f"| {tr} | " + " | ".join(_f(x) for x in row) + " |")
    allrow = [float(np.nanmean([ev[tr][m][v]["total"] for tr in ev for m in ev[tr]])) for v in cols]
    L.append("| **all 21** | " + " | ".join(f"**{_f(x)}**" for x in allrow) + " |")
    L += ["", "### Per trace and model (P / D / total)\n",
          "| trace | model | current set | V1 | V2 | current | naive | V1 2-step | V2 2-step | current 2-step |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for tr, per in ev.items():
        for m, r in per.items():
            cells = [f"{_f(r[v]['P'])} / {_f(r[v]['D'])} / {_f(r[v]['total'])}" for v in
                     ("V1", "V2", "current", "naive", "V1_2step", "V2_2step", "current_2step")]
            L.append(f"| {tr} | {m} | {r['current_set']} | " + " | ".join(cells) + " |")
    if man.get("real_past"):
        L += ["", "## Real_* training intervals (real Azure past of each model's slice)\n", man["real_past"]["rule"], "",
              "| trace | model | key | source interval (UTC) | slice starts | shifted back (s) | collisions skipped |",
              "|---|---|---|---|---|---:|---|"]
        for it in man["real_past"]["intervals"]:
            L.append(f"| {it['trace']} | {it['model']} | {it['key']} | {it['start_utc']} .. {it['end_utc']} | "
                     f"{it['slice_start_utc']} | {it['shifted_s']} | {'; '.join(it['collisions_skipped']) or 'none'} |")
    L += ["", "## How different the training realizations are from the evaluation one\n",
          "Mean over windows of |x_seed - x_eval| / x_eval (%), mean (max) over the 4 training seeds. Near 0 means "
          "the generator fixes the window sums and the seed only moves arrivals inside a second / prompt text: "
          "the predictor then has seen (almost) this very series.\n",
          "| trace | model | P | D |", "|---|---|---:|---:|"]
    for tr, per in man["variability"].items():
        for m, r in per.items():
            L.append(f"| {tr} | {m} | {r['P']['mean_pct']:.2f} ({r['P']['max_pct']:.2f}) | "
                     f"{r['D']['mean_pct']:.2f} ({r['D']['max_pct']:.2f}) |")
    L += ["", "## Files\n", "| file | trace | model | sha256 |", "|---|---|---|---|"]
    for f in sorted(man["files"].values(), key=lambda f: f["file"]):
        L.append(f"| `{f['file']}` | {f['trace']} | {f['model']} | `{f['sha256']}` |")
    L += ["", "## Policy params (example)\n", "```yaml",
          "# V1 (one set per trace, a file per model); V2: one set 'concat' with predictor_rules [['', concat]]",
          "window_s: 10", "predictor_normalization: minmax", f"history_variant: {man['history']['variant']}",
          "predictors:"]
    first = cfg["traces"][0]["trace"]
    L.append(f"  v1-{first}:")
    L.append("    models:")
    for f in man["files"].values():
        if f["variant"] == "v1" and f["trace"] == first:
            L.append(f"      {f['model']}: {{path: weights/preserve-mlstm-20261104-trace-w10/{f['file']}, "
                     f"sha256: {f['sha256']}}}")
    L += ["  # ... one set per trace", "predictor_rules:", f"- [{first}, v1-{first}]", "```", ""]
    return "\n".join(L) + "\n"


# ----------------------------------------------------------------------------------------------- main


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p):
        p.add_argument("--config", default=str(DEFAULT_CONFIG))
        p.add_argument("--data-root", help="the directory with the evaluation trace sets")
        p.add_argument("--out", help="generated data (outside git), e.g. <datasets>/preserve-train-trace-20261104")
        p.add_argument("--tre-root", default=str(BASELINES.parent), help="the tre/ tree (loadgen configs)")
    p = sub.add_parser("realize")
    common(p)
    p.add_argument("--v1-gen", required=True, help="v1 CustomTraceGenerator (src/, config/icse_final/)")
    p.add_argument("--azure-root", required=True, help="datasets/azure-llm-inference")
    p.add_argument("--processes", type=int, default=4)
    p.add_argument("--cpus", default="", help="taskset CPU list for the generators")
    p.add_argument("--nice", type=int, default=19)
    p.set_defaults(fn=cmd_realize)
    p = sub.add_parser("realize-real-past")
    common(p)
    p.add_argument("--v1-gen", required=True)
    p.add_argument("--azure-root", required=True)
    p.add_argument("--processes", type=int, default=4)
    p.set_defaults(fn=cmd_realize_real_past)
    p = sub.add_parser("series")
    common(p)
    p.set_defaults(fn=cmd_series)
    p = sub.add_parser("train")
    common(p)
    p.add_argument("--series", required=True)
    p.add_argument("--train-dir", required=True)
    p.add_argument("--jobs", type=int, default=8)
    p.add_argument("--threads", type=int, default=1)
    p.set_defaults(fn=cmd_train)
    p = sub.add_parser("finish")
    common(p)
    p.add_argument("--series", required=True)
    p.add_argument("--train-dir", required=True)
    p.add_argument("--weights-out", required=True)
    p.add_argument("--history-sha256-out", help="write the history sha256 list (paths relative to --out)")
    p.set_defaults(fn=cmd_finish)
    a = ap.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
