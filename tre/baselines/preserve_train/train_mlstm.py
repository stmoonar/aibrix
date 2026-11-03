"""Train the PreServe Tier-1 mLSTM predictors on the Azure LLM inference 2024 week traces.

Three model sets (``code``, ``conv``, ``pooled`` = code + conv sequences), each a prompt-token
model and a response-token model. Shipped: per-sequence-mean normalization (D15). Also trained
on the same split, for comparison only (not exported): upstream's global min-max variant.

Usage (CPU; paths are arguments)::

    python -m preserve_train.train_mlstm \\
        --csv code=<dir>/AzureLLMInferenceTrace_code_1week.csv \\
        --csv conv=<dir>/AzureLLMInferenceTrace_conv_1week.csv \\
        --out <data dir> [--epochs 50] [--jobs 12] [--threads 1]

Outputs in ``--out``: ``<set><suffix>.npz`` (read by ``tre_baselines.policies.preserve_predictor``), ``manifest.json``,
``REPORT.md``, and small window caches ``windows_<stream>_<W>s.npz``.

Protocols (``--protocol``):

* ``heldout_frac`` (default; the 10-22 weights): every set trains on the first
  ``1 - held_out_frac`` of each stream (the Real_* slice intervals excluded) and is scored on
  the rest.
* ``paper_past`` (redesign 2026-10-24, paper section 5.3 RQ2 / 5.2 RQ1: train on the past,
  predict the future): ``code`` trains only on code windows before ``REAL_CUTOFFS['code']``
  (the earliest Real_code slice start), ``conv`` only on conv windows before
  ``REAL_CUTOFFS['conv']``; their held-out part is everything after the cutoff. ``pooled``
  (the synthetic traces) splits each stream 1:1 in time and trains on the first halves.
  Upstream's request-length filter and its sub-range selection are not applied (the test
  slices are unfiltered): a disclosed deviation.
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import datetime as dt
import json
import multiprocessing as mp
import os
import platform
import subprocess
import sys
import time
from typing import Dict, List, Tuple

import numpy as np

from preserve_train import data as D

# Dataset facts (public Azure LLM inference 2024 week traces, release dataset-llm-2024).
EXPECTED_SHA256 = {
    "code": "71de5c55cbc35f8f1ed0b6b7806b4cd1e9764b0058469725a6aac98023a1448f",
    "conv": "a0cc9b969a9bbf0fd811802cbf4323edd3a209ace791e3799ad4f9207f213941",
}
# Leakage exclusion: the v1 Real_* traces were cut from these intervals (the whole ReTrace
# artifact window, docs/trace-design-v2.md "Real-* 的 v1 方法"), given as week offsets from
# the first day's 00:00 UTC; the UTC form is checked against the offsets at run time.
DEFAULT_EXCLUSIONS = {
    "conv": [("2024-05-13T09:00:00+00:00", "2024-05-13T11:00:00+00:00", 118800, 126000)],
    "code": [("2024-05-11T00:00:00+00:00", "2024-05-11T00:30:00+00:00", 86400, 88200)],
}
SETS = {"code": ["code"], "conv": ["conv"], "pooled": ["code", "conv"]}
#: protocol paper_past: the set trained for Real_<stream> sees only windows before this instant
#: (the earliest of the three model slices: Real_code 8b 05-11 00:00, Real_conv 8b 05-13 09:00 UTC)
REAL_CUTOFFS = {"code": "2024-05-11T00:00:00+00:00", "conv": "2024-05-13T09:00:00+00:00"}
PROTOCOLS = ("heldout_frac", "paper_past")
KINDS = {"prompt": "P", "response": "D"}
NORMS = ("per_sequence_mean", "global_minmax")


def log(msg: str) -> None:
    print(f"[{dt.datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def git_info() -> Dict[str, object]:
    here = os.path.dirname(os.path.abspath(__file__))
    try:
        sha = subprocess.check_output(["git", "-C", here, "rev-parse", "HEAD"], text=True).strip()
        dirty = bool(subprocess.check_output(["git", "-C", here, "status", "--porcelain", "--", "."],
                                             text=True).strip())
        return {"sha": sha, "dirty_tre_preserve_train": dirty}
    except Exception as e:  # not a git checkout
        return {"sha": None, "error": str(e)}


def parse_kv(items: List[str], what: str) -> Dict[str, str]:
    out = {}
    for it in items:
        if "=" not in it:
            raise SystemExit(f"--{what} expects NAME=VALUE, got {it!r}")
        k, v = it.split("=", 1)
        out[k] = v
    return out


def load_stream(name: str, csv_path: str, args) -> Tuple[D.Stream, Dict[str, object]]:
    log(f"{name}: sha256 {csv_path}")
    sha = D.sha256_file(csv_path)
    if not args.no_verify_sha and sha != EXPECTED_SHA256.get(name):
        raise SystemExit(f"{name}: sha256 {sha} != expected {EXPECTED_SHA256.get(name)}")
    cache = os.path.join(args.out, f"windows_{name}_{args.window_s}s.npz")
    w = None
    if os.path.exists(cache):
        with np.load(cache, allow_pickle=False) as z:
            if str(z["sha256"]) == sha and int(z["window_s"]) == args.window_s:
                w = {k: z[k] for k in z.files}
                w = {**w, "origin_ns": int(w["origin_ns"]), "first_ns": int(w["first_ns"]),
                     "last_ns": int(w["last_ns"]), "rows": int(w["rows"])}
                log(f"{name}: windows from cache {cache}")
    if w is None:
        log(f"{name}: windowing CSV (W={args.window_s}s)")
        w = D.window_csv(csv_path, args.window_s)
        np.savez(cache, sha256=np.array(sha), **{k: np.asarray(v) for k, v in w.items()})
    n = len(w["P"])
    valid = np.ones(n, dtype=bool)
    excl_meta = []
    for start, end, off_a, off_b in args.exclusions.get(name, []):
        if off_a is not None:
            got = ((D.iso_to_ns(start) - w["origin_ns"]) // 10**9, (D.iso_to_ns(end) - w["origin_ns"]) // 10**9)
            if got != (off_a, off_b):
                raise SystemExit(f"{name}: exclusion {start}..{end} is week offset {got}, expected {(off_a, off_b)}")
        a, b = D.exclusion_windows(w["origin_ns"], args.window_s, start, end, n)
        valid[a:b] = False
        excl_meta.append({"start_utc": start, "end_utc": end, "week_offset_s": [off_a, off_b],
                          "window_range": [a, b], "windows": b - a})
    split = int(np.floor(n * (1.0 - args.held_out_frac)))
    st = D.Stream(name=name, P=w["P"], D=w["D"], valid=valid, split=split)
    meta = {
        "file": os.path.basename(csv_path), "sha256": sha, "rows": int(w["rows"]),
        "origin_utc": D.ns_to_iso(w["origin_ns"]), "first_request_utc": D.ns_to_iso(w["first_ns"]),
        "last_request_utc": D.ns_to_iso(w["last_ns"]), "windows": n,
        "exclusions": excl_meta,
        "split": {
            "held_out_frac": args.held_out_frac, "split_window": split,
            "held_out_starts_utc": D.ns_to_iso(w["origin_ns"] + split * args.window_s * 10**9),
            "train_windows": int(valid[:split].sum()), "held_out_windows": int(valid[split:].sum()),
            "excluded_windows": int((~valid).sum()),
            "train_segments": [list(s) for s in D.segments(valid[:split])],
            "train_sequences": int(len(st.targets(args.seq_len, held_out=False))),
            "held_out_targets": int(len(st.targets(args.seq_len, held_out=True))),
        },
        "mean_P_per_window": float(w["P"][valid].mean()), "mean_D_per_window": float(w["D"][valid].mean()),
    }
    return st, meta


def set_streams_for(protocol: str, streams: Dict[str, D.Stream], origins: Dict[str, int], window_s: int,
                     seq_len: int) -> Tuple[Dict[str, Dict[str, D.Stream]], Dict[str, dict]]:
    """{set: {stream: Stream with that set's split}} and the split record for the manifest."""
    out: Dict[str, Dict[str, D.Stream]] = {}
    rec: Dict[str, dict] = {}
    W_ns = int(window_s) * 10**9
    for set_name, members in SETS.items():
        out[set_name], rec[set_name] = {}, {}
        for m in members:
            base = streams[m]
            if protocol == "heldout_frac":
                split, rule = base.split, "first 1 - held_out_frac"
            elif set_name in REAL_CUTOFFS:
                cut = D.iso_to_ns(REAL_CUTOFFS[set_name]) - origins[m]
                if cut % W_ns:
                    raise SystemExit(f"{set_name}: cutoff {REAL_CUTOFFS[set_name]} is not on a window edge")
                split, rule = int(cut // W_ns), f"windows before {REAL_CUTOFFS[set_name]} (Real_{set_name} slices)"
            else:
                split, rule = base.n // 2, "chronological 1:1 (first half)"
            st = D.Stream(name=m, P=base.P, D=base.D, valid=base.valid, split=split)
            t = st.targets(seq_len, held_out=False)
            assert len(t) and int(t.max()) < split, (set_name, m)   # inputs and targets end before split
            out[set_name][m] = st
            rec[set_name][m] = {"rule": rule, "split_window": split,
                                "train_until_utc": D.ns_to_iso(origins[m] + split * W_ns),
                                "train_windows": int(st.valid[:split].sum()),
                                "train_sequences": int(len(t)),
                                "held_out_targets": int(len(st.targets(seq_len, held_out=True)))}
    return out, rec


def make_training_data(streams: List[D.Stream], key: str, norm: str, seq_len: int, eps: float):
    Xs, ys = [], []
    for st in streams:
        series = getattr(st, key)
        X, y = D.build_xy(series, st.targets(seq_len, held_out=False), seq_len)
        Xs.append(X)
        ys.append(y)
    X, y = np.concatenate(Xs), np.concatenate(ys)
    if norm == "per_sequence_mean":
        Xn, yn, _ = D.per_sequence_normalize(X, y, eps)
        return Xn, yn, {}
    vals = np.concatenate([getattr(st, key)[:st.split][st.valid[:st.split]] for st in streams]).astype(np.float64)
    mn, mx = float(vals.min()), float(vals.max())
    return (X - mn) / (mx - mn), (y - mn) / (mx - mn), {"min": mn, "max": mx}


def predict_held_out(model, st: D.Stream, key: str, norm: str, scaler: dict, seq_len: int, eps: float):
    from preserve_train.model import forward_batch

    t = st.targets(seq_len, held_out=True)
    X, y = D.build_xy(getattr(st, key), t, seq_len)
    if norm == "per_sequence_mean":
        Xn, _, scale = D.per_sequence_normalize(X, y, eps)
        pred = np.maximum(forward_batch(model, Xn), 0.0) * scale
    else:
        mn, mx = scaler["min"], scaler["max"]
        pred = forward_batch(model, (X - mn) / (mx - mn)) * (mx - mn) + mn
    return t, y, pred


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


def model_from_state(sd: dict, hidden: int):
    import torch
    from preserve_train.model import mLSTMModel

    m = mLSTMModel(input_dim=1, hidden_dim=hidden, output_dim=1)
    m.load_state_dict({k: torch.from_numpy(np.asarray(v)) for k, v in sd.items()})
    m.eval()
    return m


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", action="append", required=True, help="STREAM=PATH (streams: code, conv)")
    ap.add_argument("--out", required=True, help="output data directory")
    ap.add_argument("--window-s", type=int, default=30)
    ap.add_argument("--seq-len", type=int, default=25)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=50,
                    help="upstream mLSTM_predict.py default 50 (their train_predictor.py used 30)")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--eps", type=float, default=1.0, help="per-sequence mean floor (tokens)")
    ap.add_argument("--held-out-frac", type=float, default=0.2)
    ap.add_argument("--exclude", action="append", default=[],
                    help="STREAM=START/END (UTC ISO); replaces the built-in exclusions of that stream")
    ap.add_argument("--no-verify-sha", action="store_true")
    ap.add_argument("--protocol", choices=PROTOCOLS, default="heldout_frac")
    ap.add_argument("--norms", default=",".join(NORMS),
                    help="comma list of normalizations to train (per_sequence_mean is the shipped one)")
    ap.add_argument("--file-suffix", default="", help="exported files are <set><suffix>.npz")
    ap.add_argument("--jobs", type=int, default=12)
    ap.add_argument("--threads", type=int, default=1, help="torch threads per training job")
    args = ap.parse_args(argv)

    csvs = parse_kv(args.csv, "csv")
    unknown = set(csvs) - set(EXPECTED_SHA256)
    if unknown or set(csvs) != {"code", "conv"}:
        raise SystemExit(f"need --csv code=... and --csv conv=... (got {sorted(csvs)})")
    args.exclusions = {k: list(v) for k, v in DEFAULT_EXCLUSIONS.items()}
    overridden = set()
    for item in args.exclude:
        k, v = item.split("=", 1)
        s, e = v.split("/", 1)
        if k not in overridden:
            args.exclusions[k] = []
            overridden.add(k)
        args.exclusions[k].append((s, e, None, None))
    os.makedirs(args.out, exist_ok=True)

    import torch

    torch.set_num_threads(max(1, min(8, args.jobs * args.threads)))  # held-out forward passes
    streams: Dict[str, D.Stream] = {}
    meta_ds: Dict[str, dict] = {}
    for name in ("code", "conv"):
        streams[name], meta_ds[name] = load_stream(name, csvs[name], args)
        sp = meta_ds[name]["split"]
        log(f"{name}: windows={meta_ds[name]['windows']} excluded={sp['excluded_windows']} "
            f"train={sp['train_windows']} held_out={sp['held_out_windows']} "
            f"train_seq={sp['train_sequences']} held_out_targets={sp['held_out_targets']}")

    norms = [n for n in args.norms.split(",") if n]
    if "per_sequence_mean" not in norms or any(n not in NORMS for n in norms):
        raise SystemExit(f"--norms must include per_sequence_mean and name only {NORMS}")
    origins = {m: D.iso_to_ns(meta_ds[m]["origin_utc"]) for m in streams}
    sets, set_splits = set_streams_for(args.protocol, streams, origins, args.window_s, args.seq_len)
    for set_name, recs in set_splits.items():
        for m, r in recs.items():
            log(f"split {set_name}/{m}: {r}")
    jobs, scalers = [], {}
    for set_name, members in SETS.items():
        for kind, key in KINDS.items():
            for norm in norms:
                X, y, scaler = make_training_data([sets[set_name][m] for m in members], key, norm, args.seq_len,
                                                  args.eps)
                jid = f"{set_name}/{kind}/{norm}"
                scalers[jid] = scaler
                jobs.append({"id": jid, "X": X, "y": y, "hidden": args.hidden, "epochs": args.epochs,
                             "batch_size": args.batch_size, "lr": args.lr, "seed": args.seed,
                             "threads": args.threads})
    log(f"training {len(jobs)} models, {args.jobs} parallel jobs x {args.threads} threads, {args.epochs} epochs")
    results = {}
    with cf.ProcessPoolExecutor(max_workers=args.jobs, mp_context=mp.get_context("spawn")) as ex:
        for r in ex.map(_train_job, jobs):
            results[r["id"]] = r
            log(f"done {r['id']} in {r['seconds']:.0f}s; {r['loss_log'][-1] if r['loss_log'] else ''}")

    models = {jid: model_from_state(r["state_dict"], args.hidden) for jid, r in results.items()}

    # Held-out evaluation: every set on the held-out part of each member stream.
    mape_tab: Dict[str, dict] = {}
    for set_name, members in SETS.items():
        for norm in norms:
            for m in members:
                preds = {}
                for kind, key in KINDS.items():
                    jid = f"{set_name}/{kind}/{norm}"
                    preds[key] = predict_held_out(models[jid], sets[set_name][m], key, norm, scalers[jid],
                                                  args.seq_len, args.eps)
                (tP, yP, pP), (tD, yD, pD) = preds["P"], preds["D"]
                assert np.array_equal(tP, tD)
                mape_tab.setdefault(set_name, {}).setdefault(norm, {})[m] = {
                    "prompt": round(D.mape(yP, pP), 2), "response": round(D.mape(yD, pD), 2),
                    "total": round(D.mape(yP + yD, pP + pD), 2), "targets": int(len(tP)),
                }
                log(f"held-out {set_name:6s} {norm:17s} on {m}: {mape_tab[set_name][norm][m]}")

    # Export shipped models and check numpy == torch on held-out histories.
    from preserve_train.export import export_npz
    from tre_baselines.policies.preserve_predictor import load_predictors

    outputs, np_check = {}, {}
    rng = np.random.default_rng(0)
    for set_name, members in SETS.items():
        path = os.path.join(args.out, f"{set_name}{args.file_suffix}.npz")
        export_npz({k: models[f"{set_name}/{k}/per_sequence_mean"] for k in KINDS}, path,
                   seq_len=args.seq_len, hidden=args.hidden, eps=args.eps, window_s=args.window_s)
        outputs[set_name] = {"file": os.path.basename(path), "sha256": D.sha256_file(path)}
        preds = load_predictors(path)
        worst = 0.0
        for m in members:
            st = sets[set_name][m]
            for kind, key in KINDS.items():
                t, _, p_torch = predict_held_out(models[f"{set_name}/{kind}/per_sequence_mean"], st, key,
                                                 "per_sequence_mean", {}, args.seq_len, args.eps)
                pick = rng.choice(len(t), size=min(200, len(t)), replace=False)
                series = getattr(st, key)
                for j in pick:
                    p_np = preds[kind].predict(series[t[j] - args.seq_len:t[j]])
                    worst = max(worst, abs(p_np - p_torch[j]) / max(abs(p_torch[j]), 1.0))
        np_check[set_name] = worst
        log(f"exported {path}; numpy vs torch max rel diff {worst:.2e}")

    manifest = {
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "code": git_info(),
        "env": {"python": platform.python_version(), "torch": torch.__version__, "numpy": np.__version__,
                "host_cpu_count": os.cpu_count()},
        "window_s": args.window_s,
        "protocol": args.protocol,
        "set_splits": set_splits,
        "real_cutoffs_utc": REAL_CUTOFFS if args.protocol == "paper_past" else None,
        "deviations": ["upstream's request-length filter and sub-range selection are not applied: the "
                       "test slices are unfiltered"] if args.protocol == "paper_past" else [],
        "datasets": meta_ds,
        "model": {"arch": "upstream PreServe mLSTM (mLSTM_predict.py): 1 mLSTMCell + Linear(H->1); "
                          "m computed but unused in the gates (effectively LSTM), kept as a faithful copy",
                  "seq_len": args.seq_len, "hidden": args.hidden},
        "training": {"loss": "MSE", "optimizer": "Adam", "lr": args.lr, "batch_size": args.batch_size,
                     "shuffle": True, "seed": args.seed, "epochs": args.epochs,
                     "epochs_reason": "upstream mLSTM_predict.py default (50); upstream train_predictor.py "
                                      "used 30, we follow the RQ1 predictor script",
                     "sequences_span_no_gap": True,
                     "seconds": {jid: round(r["seconds"], 1) for jid, r in results.items()},
                     "last_loss_log": {jid: (r["loss_log"][-1] if r["loss_log"] else None)
                                       for jid, r in results.items()}},
        "normalization": {
            "shipped": "per_sequence_mean",
            "per_sequence_mean": f"x / max(mean(x[-{args.seq_len}:]), eps), eps={args.eps}; "
                                 f"prediction = max(0, ratio) * that mean (D15, deviation from upstream)",
            "global_minmax": "upstream: (x - min) / (max - min) over the set's training windows; not shipped",
            "global_minmax_scalers": {jid: s for jid, s in scalers.items() if s},
        },
        "sets": {k: {"streams": v, **outputs[k]} for k, v in SETS.items()},
        "held_out_mape": mape_tab,
        "mape_definition": "mean |y - yhat| / y * 100 over windows with y > 0 (upstream train_predictor.py); "
                           "total = (P + D) vs (Phat + Dhat)",
        "upstream_reference_total_mape": {"code": {"5s": 17.8, "600s": 9.2}, "conv": {"5s": 10.4, "600s": 4.6}},
        "numpy_vs_torch_max_rel_diff": np_check,
        "argv": sys.argv,
    }
    mpath = os.path.join(args.out, "manifest.json")
    old = {}
    if os.path.exists(mpath):
        with open(mpath) as f:
            old = json.load(f)
    if "trace_eval" in old:
        manifest["trace_eval_stale"] = True  # predictors changed; rerun eval_traces
    with open(mpath, "w") as f:
        json.dump(manifest, f, indent=2)
    from preserve_train.report import write_report

    write_report(args.out)
    log(f"wrote {mpath}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
