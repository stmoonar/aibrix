"""Retrain the PreServe Tier-1 mLSTM at another window W with TRE v1's trainer, unchanged.

Decision 2026-11-03 (user: baseline windows 30 s -> 10 s): the weights are retrained by the v1
trainer ``tre_preserve.train.train_mlstm`` (tre_code ``80571402``, protocol ``paper_past``, the
trainer of ``weights/preserve-mlstm-20261024-paper``) with ``--window-s 10``; everything else is the
v1 protocol. This tool only adds what the v1 trainer lacks:

``windows``
    The v1 trainer windows the Azure CSV with pandas (``data.window_csv``) unless its cache
    ``windows_<stream>_<W>s.npz`` exists; the CPU training host has no pandas. This command writes
    that cache with the standard library + numpy (same arrays, same keys), and first proves the
    equivalence: it windows the CSV at ``--check-window-s`` (30) too and compares every array with
    the v1 cache of the 30 s run (``--check-cache``); any difference -> exit 2, nothing written.

``finish``
    After training: copy each ``<set><suffix>.npz`` of the training directory into the weights
    directory with one more scalar key ``window_s`` (the window the weights were trained on; read by
    :mod:`tre_baselines.policies.preserve_predictor`, the policy refuses another window), check that
    the copy predicts exactly as the trainer's file, compute the naive persistence forecaster
    (prediction = the last window) MAPE on the same held-out targets at the new window and at the
    reference window, and write ``REPORT.md`` + ``manifest.json`` in the weights directory.

The training directory (the v1 trainer's ``--out``) is never modified.
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
from typing import Any, Dict

import numpy as np

DAY_S = 86400
KEYS = ("P", "D", "N", "origin_ns", "first_ns", "last_ns", "rows", "window_s")


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


_DAY_CACHE: Dict[str, int] = {}


def ts_ns(s: str) -> int:
    """``YYYY-MM-DD HH:MM:SS[.fraction][+00:00|Z]`` (UTC) -> ns since the epoch, exactly as
    ``pd.to_datetime(format="ISO8601", utc=True)`` gives for the Azure 2024 files."""
    day = s[:10]
    base = _DAY_CACHE.get(day)
    if base is None:
        base = (dt.date.fromisoformat(day).toordinal() - dt.date(1970, 1, 1).toordinal()) * DAY_S * 10**9
        _DAY_CACHE[day] = base
    if s[10] not in " T":
        raise ValueError(f"unexpected timestamp {s!r}")
    t = (int(s[11:13]) * 3600 + int(s[14:16]) * 60 + int(s[17:19])) * 10**9
    rest = s[19:]
    frac = 0
    if rest.startswith("."):
        k = 1
        while k < len(rest) and rest[k].isdigit():
            k += 1
        digits = rest[1:k]
        frac = int((digits + "000000000")[:9])
        rest = rest[k:]
    if rest not in ("", "+00:00", "Z", "z"):
        raise ValueError(f"non-UTC timestamp {s!r}")
    return base + t + frac


def read_csv(path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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


def window(ns: np.ndarray, ctx: np.ndarray, gen: np.ndarray, window_s: int, first_chunk: int = 2_000_000) -> dict:
    """The arrays of v1 ``tre_preserve.train.data.window_csv`` (origin = 00:00 UTC of the day of the
    first chunk's earliest request, whole days, P / D / N per window)."""
    W_ns = int(window_s) * 10**9
    first_ns = int(ns[:first_chunk].min())
    origin_ns = first_ns - first_ns % (DAY_S * 10**9)
    if int(ns.min()) < origin_ns:
        raise ValueError("timestamps before the first day's midnight (CSV not sorted by day?)")
    first_ns = int(ns.min())
    last_ns = int(ns.max())
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
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    checks = {}
    for item in a.csv:
        name, path = item.split("=", 1)
        print(f"{name}: sha256 {path}", flush=True)
        sha = sha256_file(path)
        print(f"{name}: reading {path}", flush=True)
        ns, ctx, gen = read_csv(path)
        ref_path = None
        for c in a.check_cache:
            cn, cp = c.split("=", 1)
            if cn == name:
                ref_path = cp
        if ref_path is None:
            print(f"{name}: no --check-cache given", file=sys.stderr)
            return 2
        mine = window(ns, ctx, gen, a.check_window_s)
        with np.load(ref_path, allow_pickle=False) as z:
            ref = {k: z[k] for k in z.files}
        diff = [k for k in KEYS if not np.array_equal(np.asarray(mine[k]), ref[k])]
        if str(ref["sha256"]) != sha:
            diff.append("sha256")
        checks[name] = {"reference": ref_path, "reference_sha256": sha256_file(ref_path),
                        "window_s": a.check_window_s, "equal": not diff, "differs": diff}
        print(f"{name}: {a.check_window_s} s windows vs {ref_path}: {'equal' if not diff else 'DIFFER ' + str(diff)}",
              flush=True)
        if diff:
            return 2
        w = window(ns, ctx, gen, a.window_s)
        dst = out / f"windows_{name}_{a.window_s}s.npz"
        if dst.exists():
            print(f"{dst} exists: not overwritten", file=sys.stderr)
            return 2
        tmp = dst.with_name(dst.name + ".tmp.npz")
        np.savez(tmp, sha256=np.array(sha), **{k: np.asarray(v) for k, v in w.items()})
        os.replace(tmp, dst)
        checks[name].update(written=str(dst), written_sha256=sha256_file(dst), windows=int(len(w["P"])))
        print(f"{name}: wrote {dst} ({len(w['P'])} windows)", flush=True)
    (out / "windows_check.json").write_text(json.dumps(checks, indent=1))
    return 0


# ------------------------------------------------------------------ finish

def _naive_mape(train_mlstm, D, cache_dir: Path, window_s: int, seq_len: int, protocol: str) -> dict:
    """Persistence forecast (y_hat[t] = y[t-1]) MAPE on the held-out targets of each set (the
    trainer's ``set_streams_for`` split and ``Stream.targets``, the trainer's ``mape``)."""
    streams, origins = {}, {}
    for name in ("code", "conv"):
        with np.load(cache_dir / f"windows_{name}_{window_s}s.npz", allow_pickle=False) as z:
            w = {k: z[k] for k in z.files}
        n = len(w["P"])
        valid = np.ones(n, dtype=bool)
        for start, end, _a, _b in train_mlstm.DEFAULT_EXCLUSIONS.get(name, []):
            lo, hi = D.exclusion_windows(int(w["origin_ns"]), window_s, start, end, n)
            valid[lo:hi] = False
        streams[name] = D.Stream(name=name, P=w["P"], D=w["D"], valid=valid, split=n)
        origins[name] = int(w["origin_ns"])
    sets, _ = train_mlstm.set_streams_for(protocol, streams, origins, window_s, seq_len)
    out: dict = {}
    for set_name, members in train_mlstm.SETS.items():
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
    v1 = Path(a.v1_python).resolve()
    sys.path.insert(0, str(v1))
    from tre_preserve.train import data as D  # noqa: E402  (v1 trainer's data module)
    from tre_preserve.train import train_mlstm  # noqa: E402
    from tre_preserve.predictor import load_predictors as v1_load  # noqa: E402

    here = Path(__file__).resolve()
    sys.path.insert(0, str(here.parents[2]))
    from tre_baselines.policies import preserve_predictor as v2p  # noqa: E402

    train_dir, out = Path(a.train_dir), Path(a.out)
    man = json.loads((train_dir / "manifest.json").read_text())
    if int(man["window_s"]) != a.window_s:
        print(f"training window {man['window_s']} != --window-s {a.window_s}", file=sys.stderr)
        return 2
    out.mkdir(parents=True, exist_ok=True)
    files: dict = {}
    rng = np.random.default_rng(0)
    for set_name, rec in man["sets"].items():
        src = train_dir / rec["file"]
        if sha256_file(src) != rec["sha256"]:
            print(f"{src}: sha256 differs from the trainer's manifest", file=sys.stderr)
            return 2
        dst = out / rec["file"]
        if dst.exists():
            print(f"{dst} exists: not overwritten", file=sys.stderr)
            return 2
        with np.load(src, allow_pickle=False) as z:
            arrays = {k: z[k] for k in z.files}
        if "window_s" in arrays:
            print(f"{src} already has window_s", file=sys.stderr)
            return 2
        arrays["window_s"] = np.array(float(a.window_s))
        tmp = dst.with_name(dst.name + ".tmp.npz")
        np.savez(tmp, **arrays)
        os.replace(tmp, dst)
        sha = sha256_file(dst)
        # the stamped copy must predict exactly as the trainer's file, and declare its window
        ref, new = v1_load(str(src)), v2p.load_predictors(str(dst), sha)
        worst = 0.0
        for kind in v2p.KINDS:
            if new[kind].window_s != float(a.window_s):
                print(f"{dst}: window_s {new[kind].window_s} != {a.window_s}", file=sys.stderr)
                return 2
            for _ in range(200):
                x = rng.gamma(2.0, 1e5, size=ref[kind].seq_len)
                worst = max(worst, abs(ref[kind].predict(x) - new[kind].predict(x)))
        if worst != 0.0:
            print(f"{dst}: predictions differ from the trainer's file (max {worst})", file=sys.stderr)
            return 2
        files[set_name] = {"file": rec["file"], "sha256": sha, "trainer_file_sha256": rec["sha256"],
                           "streams": rec.get("streams")}
    naive = {str(a.window_s): _naive_mape(train_mlstm, D, train_dir, a.window_s, man["model"]["seq_len"],
                                          man["protocol"])}
    ref_man = None
    if a.ref_train_dir:
        rdir = Path(a.ref_train_dir)
        ref_man = json.loads((rdir / "manifest.json").read_text())
        rw = int(ref_man["window_s"])
        naive[str(rw)] = _naive_mape(train_mlstm, D, rdir, rw, ref_man["model"]["seq_len"], ref_man["protocol"])
    shutil.copyfile(train_dir / "REPORT.md", out / "REPORT.trainer.md")
    doc = {"created_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
           "tool": {"path": str(here), **_git(here.parent)}, "trainer_dir": str(train_dir),
           "trainer_manifest_sha256": sha256_file(train_dir / "manifest.json"), "trainer_code": man["code"],
           "trainer_env": man["env"], "trainer_argv": man["argv"], "window_s": a.window_s, "files": files,
           "held_out_mape_mlstm": {str(a.window_s): man["held_out_mape"]},
           "held_out_mape_naive_last_window": naive, "run_notes": list(a.note)}
    if ref_man is not None:
        doc["held_out_mape_mlstm"][str(ref_man["window_s"])] = ref_man["held_out_mape"]
        doc["reference_dir"] = str(a.ref_train_dir)
    (out / "manifest.json").write_text(json.dumps(doc, indent=1))
    (out / "REPORT.md").write_text(_report(doc, man))
    print(json.dumps({"files": files, "naive": naive}, indent=1))
    return 0


def _report(doc: dict, man: dict) -> str:
    W = str(doc["window_s"])
    L = ["# PreServe mLSTM predictors, W = %s s" % W, "",
         f"Created {doc['created_utc']} by `tre_baselines/tools/preserve_mlstm_retrain.py finish` "
         f"(aibrix {doc['tool'].get('sha')}, dirty {doc['tool'].get('dirty')}).", "",
         f"Trained by the TRE v1 trainer, unchanged: tre_code {man['code'].get('sha')} "
         f"(tre_preserve/train dirty: {man['code'].get('dirty_tre_preserve_train')}), "
         f"torch {man['env']['torch']}, numpy {man['env']['numpy']}, python {man['env']['python']}, CPU; "
         f"argv `{' '.join(man['argv'])}`. Trainer output (not modified): `{doc['trainer_dir']}`; its own report "
         "is `REPORT.trainer.md` here.", "",
         f"The only protocol change against `preserve-mlstm-20261024-paper` is the window: W = {W} s "
         "(was 30 s). seq_len, hidden, epochs, batch, lr, seed, normalization, paper_past splits, exclusions "
         "and dataset files are the trainer's defaults / the same arguments.", "",
         f"Each file below is the trainer's file plus one scalar key `window_s` = {W} "
         "(the policy refuses to start when it differs from `params.window_s`); predictions are bit-identical "
         "to the trainer's file.", ""]
    if doc.get("run_notes"):
        L += ["Run differences (not protocol):", ""] + [f"- {n}" for n in doc["run_notes"]] + [""]
    L += ["| set | file | sha256 | trainer file sha256 |", "|---|---|---|---|"]
    for s, r in doc["files"].items():
        L.append(f"| {s} | `{r['file']}` | `{r['sha256']}` | `{r['trainer_file_sha256']}` |")
    L += ["", "## Held-out MAPE (%): mLSTM (shipped per_sequence_mean) vs naive last window", "",
          man["mape_definition"] + ". Naive: the prediction of window t is the value of window t-1 "
          "(same held-out targets).", "",
          "| W (s) | set | evaluated on | predictor | prompt | response | total | targets |",
          "|---|---|---|---|---|---|---|---|"]
    for w in sorted(doc["held_out_mape_naive_last_window"], key=float):
        ml = (doc["held_out_mape_mlstm"].get(w) or {})
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


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    w = sub.add_parser("windows", help="write the v1 trainer's window caches without pandas")
    w.add_argument("--csv", action="append", required=True, help="STREAM=PATH (code, conv)")
    w.add_argument("--check-cache", action="append", required=True,
                   help="STREAM=PATH of the v1 cache at --check-window-s (must be reproduced exactly)")
    w.add_argument("--check-window-s", type=int, default=30)
    w.add_argument("--window-s", type=int, required=True)
    w.add_argument("--out", required=True, help="the trainer's --out (new directory)")
    f = sub.add_parser("finish", help="stamp window_s, naive MAPE, REPORT.md")
    f.add_argument("--train-dir", required=True, help="the trainer's --out")
    f.add_argument("--ref-train-dir", help="the trainer's --out of the reference window (30 s)")
    f.add_argument("--v1-python", required=True, help="tre_code/python of the trainer's checkout")
    f.add_argument("--window-s", type=int, required=True)
    f.add_argument("--out", required=True, help="weights directory (tre/baselines/weights/<name>)")
    f.add_argument("--note", action="append", default=[], help="a run difference for REPORT.md (repeatable)")
    a = ap.parse_args(argv)
    return cmd_windows(a) if a.cmd == "windows" else cmd_finish(a)


if __name__ == "__main__":
    sys.exit(main())
