"""Idle TTFT(L) fit from the raw alone: in-flight concurrency reconstructed exactly from
send/done timestamps of the same cell (1 replica). A request is isolated when at most K
other requests are in flight at its send time and no other request is sent while it is
in prefill [send, send+ttft]. Held-out M cells excluded. Read-only on the campaign."""
import bisect, glob, json, os, sys
from collections import defaultdict
import numpy as np
from scipy import stats

C = "/data/nfs_shared_data/xxy/calibration_20260921"
K = int(sys.argv[1]) if len(sys.argv) > 1 else 0

def load(p):
    with open(p) as f:
        return [json.loads(l) for l in f if l.strip()]

def huber(x, y, k=1.345, it=100):
    X = np.c_[np.ones_like(x), x]; w = np.ones_like(y)
    for _ in range(it):
        W = np.sqrt(w)
        beta, *_ = np.linalg.lstsq(X * W[:, None], y * W, rcond=None)
        r = y - X @ beta
        s = 1.4826 * np.median(np.abs(r - np.median(r))) or 1.0
        u = np.abs(r) / (k * s); wn = np.where(u <= 1, 1.0, 1.0 / u)
        if np.allclose(wn, w): break
        w = wn
    return beta

out = {}
for m in ("dsqwen-7b", "dsllama-8b", "dsqwen-14b"):
    L, T, cells = [], [], defaultdict(int)
    for d in sorted(glob.glob(f"{C}/{m}/raw/*")):
        if os.path.basename(d).startswith(f"{m}_M_"):
            continue
        for rp in glob.glob(f"{d}/*.jsonl"):
            if rp.endswith(".instant.jsonl") or rp.endswith(".failures.jsonl"):
                continue
            recs = [r for r in load(rp) if r.get("send_ts_ms") is not None and r.get("done_ts_ms") is not None]
            recs.sort(key=lambda r: r["send_ts_ms"])
            sends = [r["send_ts_ms"] for r in recs]
            dones = sorted(r["done_ts_ms"] for r in recs)
            for j, r in enumerate(recs):
                if r.get("ttft_ms") is None or r.get("input_tokens") is None or r.get("http_status") != 200:
                    continue
                s = r["send_ts_ms"]
                started_before = j  # sends strictly earlier (sorted)
                done_before = bisect.bisect_right(dones, s)
                inflight = started_before - done_before
                if inflight > K:
                    continue
                if j + 1 < len(recs) and sends[j + 1] <= s + r["ttft_ms"]:
                    continue
                L.append(float(r["input_tokens"])); T.append(float(r["ttft_ms"]))
                cells[os.path.basename(d)] += 1
    x, y = np.array(L), np.array(T)
    c_h, b_h = huber(x, y); ols = stats.linregress(x, y)
    pred = c_h + b_h * x
    r2_h = 1 - np.sum((y - pred) ** 2) / np.sum((y - y.mean()) ** 2)
    per = {int(l): {"n": int((x == l).sum()), "median": float(np.median(y[x == l])), "p95": float(np.percentile(y[x == l], 95))}
           for l in sorted(set(x.tolist())) if (x == l).sum() >= 3}
    lx = np.array(list(per)); ly = np.array([v["median"] for v in per.values()])
    ts_ = stats.theilslopes(ly, lx)
    out[m] = {"k_inflight_max": K, "n": int(len(x)), "n_cells": len(cells),
              "huber": {"c_ms": float(c_h), "b_ms_per_token": float(b_h), "r2": float(r2_h)},
              "ols": {"c_ms": float(ols.intercept), "b_ms_per_token": float(ols.slope), "r2": float(ols.rvalue ** 2)},
              "theil_sen_on_length_medians": {"c_ms": float(ts_[1]), "b_ms_per_token": float(ts_[0])},
              "per_length": per}
print(json.dumps(out, indent=1))
