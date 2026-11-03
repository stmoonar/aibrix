"""PreServe Tier-1 mLSTM weights from the Azure LLM inference 2024 week traces, end to end.

    Azure CSVs -> window caches (``windows``) -> train + export npz with ``window_s`` (``train_mlstm``)
    -> weights directory with REPORT.md (``finish``)

One command (CPU; torch, see requirements-train.txt)::

    PYTHONPATH=tre/baselines python3 -m preserve_train.pipeline all \\
        --azure-root <datasets/azure-llm-inference> --window-s 10 \\
        --train-dir <new dir> --weights-out tre/baselines/weights/<name> \\
        [--check-cache code=<v1 30 s cache> --check-cache conv=<...>] [--ref-train-dir <30 s train dir>] \\
        [--jobs 12 --threads 1]

Subcommands:

``windows``
    The trainer windows the CSV with pandas (``data.window_csv``) unless its cache
    ``windows_<stream>_<W>s.npz`` exists. This writes that cache with the standard library + numpy
    (same arrays and keys), so the training host needs no pandas. With ``--check-cache STREAM=PATH``
    it first windows the CSV at ``--check-window-s`` (30) and requires every array to equal that cache
    (e.g. the v1 30 s caches); any difference -> exit 2, nothing written.
``train``
    ``train_mlstm`` with the v1 protocol ``paper_past`` (the trainer's defaults) at ``--window-s``.
``finish``
    Copy each ``<set><suffix>.npz`` of the training directory into the weights directory (it must
    declare ``window_s`` = ``--window-s``), check it loads with the policy's loader, compute the naive
    persistence forecaster (prediction = the last window) MAPE on the same held-out targets (also at
    the reference window with ``--ref-train-dir``), and write ``REPORT.md`` + ``manifest.json``.
``compare``
    Every array of two weight files must be identical (``window_s`` and the zip container aside).
``all``
    windows, train, finish.

The training directory is the trainer's ``--out``; nothing in it is overwritten by ``windows`` /
``finish`` (both refuse existing outputs).
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict

import numpy as np

DAY_S = 86400
KEYS = ("P", "D", "N", "origin_ns", "first_ns", "last_ns", "rows", "window_s")
#: public dataset files (Azure LLM inference 2024, release dataset-llm-2024), under --azure-root
CSV_REL = {"code": "2024/AzureLLMInferenceTrace_code_1week.csv", "conv": "2024/AzureLLMInferenceTrace_conv_1week.csv"}
HERE = Path(__file__).resolve().parent


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


# ------------------------------------------------------------------ windows (no pandas)
_DAY: Dict[str, int] = {}


def ts_ns(s: str) -> int:
    """``YYYY-MM-DD HH:MM:SS[.fraction][+00:00|Z]`` (UTC) -> ns since the epoch, as
    ``pd.to_datetime(format="ISO8601", utc=True)`` gives for the Azure 2024 files."""
    day = s[:10]
    base = _DAY.get(day)
    if base is None:
        base = (dt.date.fromisoformat(day).toordinal() - dt.date(1970, 1, 1).toordinal()) * DAY_S * 10**9
        _DAY[day] = base
    if s[10] not in " T":
        raise ValueError(f"unexpected timestamp {s!r}")
    t = (int(s[11:13]) * 3600 + int(s[14:16]) * 60 + int(s[17:19])) * 10**9
    rest, frac = s[19:], 0
    if rest.startswith("."):
        k = 1
        while k < len(rest) and rest[k].isdigit():
            k += 1
        frac = int((rest[1:k] + "000000000")[:9])
        rest = rest[k:]
    if rest not in ("", "+00:00", "Z", "z"):
        raise ValueError(f"non-UTC timestamp {s!r}")
    return base + t + frac


def read_csv(path) -> tuple:
    """(ns, ContextTokens, GeneratedTokens) of every row, in file order."""
    ns, ctx, gen = [], [], []
    with open(path, "r", encoding="utf-8") as f:
        header = f.readline().strip().split(",")
        i_ts, i_c, i_g = header.index("TIMESTAMP"), header.index("ContextTokens"), header.index("GeneratedTokens")
        for line in f:
            parts = line.rstrip("\r\n").split(",")
            ns.append(ts_ns(parts[i_ts]))
            ctx.append(int(parts[i_c]))
            gen.append(int(parts[i_g]))
    return np.asarray(ns, dtype=np.int64), np.asarray(ctx, dtype=np.int64), np.asarray(gen, dtype=np.int64)


def window(ns, ctx, gen, window_s: int, first_chunk: int = 2_000_000) -> dict:
    """The arrays of ``data.window_csv`` (origin = 00:00 UTC of the day of the first chunk's earliest
    request, whole days, P / D / N per window)."""
    W_ns = int(window_s) * 10**9
    first_ns = int(ns[:first_chunk].min())
    origin_ns = first_ns - first_ns % (DAY_S * 10**9)
    if int(ns.min()) < origin_ns:
        raise ValueError("timestamps before the first day's midnight (CSV not sorted by day?)")
    first_ns, last_ns = int(ns.min()), int(ns.max())
    idx = (ns - origin_ns) // W_ns
    days = math.ceil((last_ns - origin_ns) / (DAY_S * 1e9))
    n_windows = days * DAY_S // int(window_s)
    if int(idx.max()) + 1 > n_windows:
        raise ValueError("internal: more windows than whole days")
    P = np.bincount(idx, weights=ctx.astype(np.float64), minlength=n_windows).astype(np.int64)
    D = np.bincount(idx, weights=gen.astype(np.float64), minlength=n_windows).astype(np.int64)
    N = np.bincount(idx, minlength=n_windows).astype(np.int64)
    return {"P": P, "D": D, "N": N, "origin_ns": origin_ns, "first_ns": first_ns, "last_ns": last_ns,
            "rows": int(len(ns)), "window_s": int(window_s)}


def cmd_windows(a) -> int:
    out = Path(a.train_dir)
    out.mkdir(parents=True, exist_ok=True)
    refs = dict(c.split("=", 1) for c in (a.check_cache or []))
    checks = {}
    for name, rel in CSV_REL.items():
        path = Path(a.azure_root) / rel
        dst = out / f"windows_{name}_{a.window_s}s.npz"
        if dst.exists():
            print(f"{dst} exists: not overwritten", file=sys.stderr)
            return 2
        print(f"{name}: sha256 + read {path}", flush=True)
        sha = sha256_file(path)
        ns, ctx, gen = read_csv(path)
        rec: dict = {}
        if name in refs:
            mine = window(ns, ctx, gen, a.check_window_s)
            with np.load(refs[name], allow_pickle=False) as z:
                ref = {k: z[k] for k in z.files}
            diff = [k for k in KEYS if not np.array_equal(np.asarray(mine[k]), ref[k])]
            if str(ref["sha256"]) != sha:
                diff.append("sha256")
            rec.update(reference=refs[name], reference_sha256=sha256_file(refs[name]), window_s=a.check_window_s,
                       equal=not diff, differs=diff)
            print(f"{name}: {a.check_window_s} s windows vs {refs[name]}: {'equal' if not diff else 'DIFFER ' + str(diff)}",
                  flush=True)
            if diff:
                return 2
        w = window(ns, ctx, gen, a.window_s)
        tmp = dst.with_name(dst.name + ".tmp.npz")
        np.savez(tmp, sha256=np.array(sha), **{k: np.asarray(v) for k, v in w.items()})
        os.replace(tmp, dst)
        rec.update(written=str(dst), written_sha256=sha256_file(dst), windows=int(len(w["P"])), csv_sha256=sha)
        checks[name] = rec
        print(f"{name}: wrote {dst} ({len(w['P'])} windows)", flush=True)
    (out / f"windows_check_{a.window_s}s.json").write_text(json.dumps(checks, indent=1))
    return 0


# ------------------------------------------------------------------ train
def cmd_train(a) -> int:
    from preserve_train import train_mlstm

    argv = [f"--csv=code={Path(a.azure_root) / CSV_REL['code']}", f"--csv=conv={Path(a.azure_root) / CSV_REL['conv']}",
            "--out", str(a.train_dir), "--window-s", str(a.window_s), "--protocol", "paper_past",
            f"--file-suffix={a.file_suffix}", "--jobs", str(a.jobs), "--threads", str(a.threads)]
    return train_mlstm.main(argv)


# ------------------------------------------------------------------ finish
def _naive_mape(cache_dir: Path, window_s: int, seq_len: int, protocol: str) -> dict:
    """Persistence forecast (y_hat[t] = y[t-1]) MAPE on the held-out targets of each set (the
    trainer's split and targets, the trainer's ``mape``)."""
    from preserve_train import data as D
    from preserve_train import train_mlstm as T

    streams, origins = {}, {}
    for name in ("code", "conv"):
        with np.load(cache_dir / f"windows_{name}_{window_s}s.npz", allow_pickle=False) as z:
            w = {k: z[k] for k in z.files}
        n = len(w["P"])
        valid = np.ones(n, dtype=bool)
        for start, end, _a, _b in T.DEFAULT_EXCLUSIONS.get(name, []):
            lo, hi = D.exclusion_windows(int(w["origin_ns"]), window_s, start, end, n)
            valid[lo:hi] = False
        streams[name] = D.Stream(name=name, P=w["P"], D=w["D"], valid=valid, split=n)
        origins[name] = int(w["origin_ns"])
    sets, _ = T.set_streams_for(protocol, streams, origins, window_s, seq_len)
    out: dict = {}
    for set_name, members in T.SETS.items():
        for m in members:
            st = sets[set_name][m]
            t = st.targets(seq_len, held_out=True)
            yP, yD = st.P[t].astype(np.float64), st.D[t].astype(np.float64)
            pP, pD = st.P[t - 1].astype(np.float64), st.D[t - 1].astype(np.float64)
            out.setdefault(set_name, {})[m] = {
                "prompt": round(D.mape(yP, pP), 2), "response": round(D.mape(yD, pD), 2),
                "total": round(D.mape(yP + yD, pP + pD), 2), "targets": int(len(t))}
    return out


def _git(path: Path) -> dict:
    try:
        sha = subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()
        dirty = bool(subprocess.check_output(["git", "-C", str(path), "status", "--porcelain", "--", "."],
                                             text=True).strip())
        return {"sha": sha, "dirty": dirty}
    except Exception as e:  # noqa: BLE001
        return {"sha": None, "error": str(e)}


def cmd_finish(a) -> int:
    from tre_baselines.policies import preserve_predictor as pp

    train_dir, out = Path(a.train_dir), Path(a.weights_out)
    man = json.loads((train_dir / "manifest.json").read_text())
    if int(man["window_s"]) != int(a.window_s):
        print(f"training window {man['window_s']} != --window-s {a.window_s}", file=sys.stderr)
        return 2
    out.mkdir(parents=True, exist_ok=True)
    files: dict = {}
    for set_name, rec in man["sets"].items():
        src, dst = train_dir / rec["file"], out / rec["file"]
        if sha256_file(src) != rec["sha256"]:
            print(f"{src}: sha256 differs from the trainer's manifest", file=sys.stderr)
            return 2
        if dst.exists():
            print(f"{dst} exists: not overwritten", file=sys.stderr)
            return 2
        prs = pp.load_predictors(str(src), rec["sha256"])
        if {p.window_s for p in prs.values()} != {float(a.window_s)}:
            print(f"{src}: declares window {[p.window_s for p in prs.values()]}, not {a.window_s}", file=sys.stderr)
            return 2
        shutil.copyfile(src, dst)
        files[set_name] = {"file": rec["file"], "sha256": sha256_file(dst), "streams": rec.get("streams")}
    seq_len = man["model"]["seq_len"]
    naive = {str(a.window_s): _naive_mape(train_dir, int(a.window_s), seq_len, man["protocol"])}
    mlstm = {str(a.window_s): man["held_out_mape"]}
    if a.ref_train_dir:
        rdir = Path(a.ref_train_dir)
        rman = json.loads((rdir / "manifest.json").read_text())
        naive[str(rman["window_s"])] = _naive_mape(rdir, int(rman["window_s"]), rman["model"]["seq_len"],
                                                   rman["protocol"])
        mlstm[str(rman["window_s"])] = rman["held_out_mape"]
    shutil.copyfile(train_dir / "REPORT.md", out / "REPORT.trainer.md")
    doc = {"created_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
           "pipeline": {"path": "tre/baselines/preserve_train/pipeline.py", **_git(HERE)},
           "trainer_dir": str(train_dir), "trainer_manifest_sha256": sha256_file(train_dir / "manifest.json"),
           "trainer_code": man["code"], "trainer_env": man["env"], "trainer_argv": man["argv"],
           "window_s": int(a.window_s), "files": files, "held_out_mape_mlstm": mlstm,
           "held_out_mape_naive_last_window": naive, "reference_dir": a.ref_train_dir,
           "run_notes": list(a.note or [])}
    (out / "manifest.json").write_text(json.dumps(doc, indent=1) + "\n")
    (out / "REPORT.md").write_text(_report(doc, man))
    print(json.dumps({"files": files, "naive": naive}, indent=1))
    return 0


def _report(doc: dict, man: dict) -> str:
    W = str(doc["window_s"])
    L = [f"# PreServe mLSTM predictors, W = {W} s", "",
         f"Created {doc['created_utc']} by `preserve_train/pipeline.py finish` (aibrix {doc['pipeline'].get('sha')}, "
         f"dirty {doc['pipeline'].get('dirty')}). Trained by `preserve_train/train_mlstm.py` (aibrix "
         f"{man['code'].get('sha')}, preserve_train dirty: {man['code'].get('dirty_tre_preserve_train')}; the v1 "
         "trainer tre_code 80571402 with imports adjusted and `window_s` exported, see preserve_train/README.md), "
         f"torch {man['env']['torch']}, numpy {man['env']['numpy']}, python {man['env']['python']}, CPU; "
         f"argv `{' '.join(man['argv'])}`. Trainer output: `{doc['trainer_dir']}`; its own report is "
         "`REPORT.trainer.md` here.", "",
         f"Protocol: the v1 `paper_past` protocol of `preserve-mlstm-20261024-paper` (seq_len, hidden, epochs, batch, "
         f"lr, seed, normalization, splits, exclusions, dataset files) at W = {W} s. Each file declares "
         f"`window_s` = {W}; the policy refuses to start when it differs from `params.window_s`.", ""]
    if doc.get("run_notes"):
        L += ["Run notes:", ""] + [f"- {n}" for n in doc["run_notes"]] + [""]
    L += ["| set | file | sha256 |", "|---|---|---|"]
    for s, r in doc["files"].items():
        L.append(f"| {s} | `{r['file']}` | `{r['sha256']}` |")
    L += ["", "## Held-out MAPE (%): mLSTM (shipped per_sequence_mean) vs naive last window", "",
          man["mape_definition"] + ". Naive: the prediction of window t is the value of window t-1 "
          "(same held-out targets).", "",
          "| W (s) | set | evaluated on | predictor | prompt | response | total | targets |",
          "|---|---|---|---|---|---|---|---|"]
    for w in sorted(doc["held_out_mape_naive_last_window"], key=float):
        ml = doc["held_out_mape_mlstm"].get(w) or {}
        for set_name, by_stream in doc["held_out_mape_naive_last_window"][w].items():
            for stream, nv in by_stream.items():
                mr = ((ml.get(set_name) or {}).get("per_sequence_mean") or {}).get(stream)
                if mr:
                    L.append(f"| {w} | {set_name} | {stream} | mLSTM | {mr['prompt']} | {mr['response']} | "
                             f"{mr['total']} | {mr['targets']} |")
                L.append(f"| {w} | {set_name} | {stream} | naive | {nv['prompt']} | {nv['response']} | "
                         f"{nv['total']} | {nv['targets']} |")
    L.append("")
    return "\n".join(L)


# ------------------------------------------------------------------ compare
def compare_weights(a_path, b_path) -> dict:
    """{key: max |a - b|} over the arrays both files hold (``window_s`` skipped), plus keys only one has."""
    with np.load(a_path, allow_pickle=False) as za, np.load(b_path, allow_pickle=False) as zb:
        ka, kb = set(za.files) - {"window_s"}, set(zb.files) - {"window_s"}
        out: dict = {"only_a": sorted(ka - kb), "only_b": sorted(kb - ka), "diff": {}}
        for k in sorted(ka & kb):
            x, y = za[k], zb[k]
            if x.dtype.kind in "fiu" and x.shape == y.shape:
                out["diff"][k] = float(np.max(np.abs(x.astype(np.float64) - y.astype(np.float64)))) if x.size else 0.0
            else:
                out["diff"][k] = 0.0 if np.array_equal(x, y) else float("inf")
    out["identical"] = not out["only_a"] and not out["only_b"] and all(v == 0.0 for v in out["diff"].values())
    return out


def cmd_compare(a) -> int:
    res = {Path(x).name: compare_weights(x, y) for x, y in zip(a.a, a.b)}
    print(json.dumps({k: {"identical": v["identical"], "max_diff": max(v["diff"].values() or [0.0]),
                          "only_a": v["only_a"], "only_b": v["only_b"]} for k, v in res.items()}, indent=1))
    return 0 if all(v["identical"] for v in res.values()) else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p, train=True):
        p.add_argument("--train-dir", required=True, help="the trainer's --out (a new directory)")
        p.add_argument("--window-s", type=int, required=True)
        if train:
            p.add_argument("--azure-root", required=True, help="holds 2024/AzureLLMInferenceTrace_{code,conv}_1week.csv")

    w = sub.add_parser("windows")
    common(w)
    w.add_argument("--check-cache", action="append", help="STREAM=PATH of a cache at --check-window-s to reproduce")
    w.add_argument("--check-window-s", type=int, default=30)
    t = sub.add_parser("train")
    common(t)
    f = sub.add_parser("finish")
    common(f, train=False)
    c = sub.add_parser("compare")
    c.add_argument("--a", action="append", required=True)
    c.add_argument("--b", action="append", required=True)
    al = sub.add_parser("all")
    common(al)
    al.add_argument("--check-cache", action="append")
    al.add_argument("--check-window-s", type=int, default=30)
    for p in (t, al):
        p.add_argument("--file-suffix", default="-paper")
        p.add_argument("--jobs", type=int, default=12)
        p.add_argument("--threads", type=int, default=1, help="torch threads per training job (v1: 1)")
    for p in (f, al):
        p.add_argument("--weights-out", required=True, help="weights directory (tre/baselines/weights/<name>)")
        p.add_argument("--ref-train-dir", help="a training dir of another window (naive / mLSTM MAPE side by side)")
        p.add_argument("--note", action="append", default=[], help="a run note for REPORT.md (repeatable)")
    a = ap.parse_args(argv)
    if a.cmd == "windows":
        return cmd_windows(a)
    if a.cmd == "train":
        return cmd_train(a)
    if a.cmd == "finish":
        return cmd_finish(a)
    if a.cmd == "compare":
        return cmd_compare(a)
    for step in (cmd_windows, cmd_train, cmd_finish):
        rc = step(a)
        if rc:
            return rc
    return 0


if __name__ == "__main__":
    sys.exit(main())
