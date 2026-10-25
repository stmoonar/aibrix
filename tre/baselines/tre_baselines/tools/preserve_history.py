"""PreServe history before t0: build it for ``Real_*`` traces, check any history file.

Synthetic traces carry the generator's ``pre_t0.json`` (format ``tre-pre-t0-v1``); ``Real_*``
have none, their past is the real Azure windows right before the slice (v1 redesign #8,
v1 ``tools/real_history.py`` counts file ``history/Real_*_slice_*_tok70.json``):

    python3 -m tre_baselines.tools.preserve_history real \\
        --counts <v1 counts json> --trace <trace dir>/traces_tre.effective.json --out <file>
    python3 -m tre_baselines.tools.preserve_history check --trace <trace> [--history <file>]

``real`` writes ``tre-preserve-history-v1`` (:mod:`tre_baselines.policies.preserve_history`):
per model, the counts x scale x the trace's mean ``prompt_length`` / ``max_output_tokens``;
scale = the trace's requests/s of the model / the slice's ``slice_mean_count_s``, so the past
has the replayed load level (v1 applied its replay factor to the trace, not to the counts).
The campaign runner copies the file next to the trace copy as ``preserve_history.json``.

``check`` loads a history as the policy does (window, seq_len, trace / seed match) and prints,
per model, the last history window next to the trace's first window (same units), so a level
mismatch shows before a run.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any, Optional

from tre_baselines.policies import preserve_history as ph
from tre_baselines.policies.preserve import trace_name_of


def _requests(trace: str) -> list[tuple[float, str, float, float]]:
    rows = json.loads(Path(trace).read_text(encoding="utf-8"))
    if isinstance(rows, dict):
        rows = rows.get("requests") or []
    out = []
    for r in rows:
        out.append((float(r["timestamp"]), str(r["model_name"]), float(r["prompt_length"]),
                    float(r.get("max_output_tokens") or 0)))
    if not out:
        raise ValueError(f"{trace}: no requests")
    return out


def _duration(trace: str, reqs) -> float:
    man = Path(trace).parent / "manifest.json"
    if man.exists():
        d = json.loads(man.read_text(encoding="utf-8")).get("duration_s")
        if d:
            return float(d)
    return float(math.ceil(max(t for t, *_ in reqs)))


def build_real(counts: str, trace: str, *, window_s: float = 30.0, n: Optional[int] = None,
               duration_s: Optional[float] = None) -> dict[str, Any]:
    raw = Path(counts).read_bytes()
    cdoc = json.loads(raw.decode("utf-8"))
    reqs = _requests(trace)
    dur = float(duration_s) if duration_s else _duration(trace, reqs)
    rows, info = ph.real_counts(cdoc, reqs, window_s=window_s, duration_s=dur, n=n)
    return {"format": ph.FORMAT, "trace": trace_name_of(trace), "seed": ph.seed_of(trace), "window_s": window_s,
            "models": rows,
            "source": {"method": "real_counts (v1 redesign #8): counts of the Azure windows before the slice "
                                 "x (trace rps / slice_mean_count_s) x the trace's mean lengths",
                       "counts_file": Path(counts).name, "counts_sha256": hashlib.sha256(raw).hexdigest(),
                       "counts_trace": cdoc.get("trace"), "csv_sha256": cdoc.get("csv_sha256"),
                       "trace_file": Path(trace).name, "duration_s": dur, "per_model": info}}


def check(trace: str, history: Optional[str], *, window_s: float = 30.0, seq_len: int = 25) -> dict[str, Any]:
    path = history or ph.default_path(trace)
    doc = json.loads(Path(path).read_text(encoding="utf-8"))
    models = sorted(doc.get("models") or doc.get("windows") or {})
    hist, info = ph.load(path, window_s=window_s, models=models, seq_len=seq_len, trace=trace_name_of(trace),
                         seed=ph.seed_of(trace))
    reqs = _requests(trace)
    first: dict[str, list[float]] = {m: [0.0, 0.0] for m in models}
    for t, m, p, d in reqs:
        if 0.0 <= t < window_s and m in first:
            first[m][0] += p
            first[m][1] += d
    rep = {"history": path, "format": info["format"], "sha256": info["sha256"], "trace": info["trace"],
           "seed": info["seed"], "models": {}}
    for m in models:
        last = hist[m][-1]
        rep["models"][m] = {"last_P": last[0], "last_D": last[1], "trace_w0_P": first[m][0],
                            "trace_w0_D": first[m][1],
                            "ratio_P": round(first[m][0] / last[0], 3) if last[0] else None,
                            "mean_P": round(sum(r[0] for r in hist[m]) / len(hist[m]), 1)}
    return rep


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("real")
    r.add_argument("--counts", required=True)
    r.add_argument("--trace", required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--windows", type=int, default=None, help="last N counted windows (default: all)")
    r.add_argument("--duration-s", type=float, default=None, help="default: the manifest's, else the last arrival")
    c = sub.add_parser("check")
    c.add_argument("--trace", required=True)
    c.add_argument("--history", default=None)
    c.add_argument("--seq-len", type=int, default=25)
    for s in (r, c):
        s.add_argument("--window-s", type=float, default=30.0)
    a = ap.parse_args(argv)
    try:
        if a.cmd == "real":
            doc = build_real(a.counts, a.trace, window_s=a.window_s, n=a.windows, duration_s=a.duration_s)
            ph.parse(doc, window_s=a.window_s, models=sorted(doc["models"]), seq_len=1)
            Path(a.out).write_text(json.dumps(doc, indent=1) + "\n", encoding="utf-8")
            print(json.dumps({"wrote": a.out, **doc["source"]["per_model"]}, indent=1))
        else:
            print(json.dumps(check(a.trace, a.history, window_s=a.window_s, seq_len=a.seq_len), indent=1))
    except (OSError, KeyError, ValueError, TypeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
