"""Load envelope of the evaluation traces, per model, per 10 s window of arrivals (output = max_output_tokens,
or the trace's model-config ``max_tokens`` where a request has none), and the three envelope files for
``preserve_mu alg1 --envelope``:  a = P share [min, max],  b = P share [p5, p95],  c = mean input and mean output
length each [min, max].

    python3 -m tre_baselines.tools.preserve_eval_envelope --data-root R --out DIR [--window-s 10]

Traces: the 7 of campaign-bl-w10-preserve-20261103 (trace set dir / trace, ``seed1/traces_tre.effective.json``).
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
from pathlib import Path

import numpy as np
import yaml

TRACES = (("traces-icse-r3c-20261101", "Alternating_hot_model_periodic_A", None),
          ("traces-icse-r3c-20261101", "Simultaneous_spike_ramp_twice_tps1o2", 600),
          ("traces-icse-r3c-20261101", "Prefill_mixed_corner_decode_mix", 700),
          ("traces-icse-ktrace-20261031", "Decode_heavy_burst", None),
          ("traces-icse-ktrace-20261031", "Sinusoidal_demand", None),
          ("traces-icse-r3e-20261101", "Real_conv_2023_slice_a_tok70", None),
          ("traces-icse-r3e-20261101", "Real_code_2024_slice_a_tok70", None))   # (set, trace, default max_tokens)


def _q(x):
    return {"min": x.min(), "max": x.max(), "p1": np.percentile(x, 1), "p99": np.percentile(x, 99),
            "p5": np.percentile(x, 5), "p95": np.percentile(x, 95)}


def window_rows(root: Path, window_s: float):
    rows, shas = [], {}
    for sdir, trace, dflt in TRACES:
        f = root / sdir / trace / "seed1" / "traces_tre.effective.json"
        shas[trace] = hashlib.sha256(f.read_bytes()).hexdigest()
        acc = collections.defaultdict(lambda: [0, 0, 0])
        for r in json.loads(f.read_text()):
            out = r.get("max_output_tokens")
            a = acc[(r["model_name"], int(r["timestamp"] // window_s))]
            a[0] += 1
            a[1] += int(r["prompt_length"])
            a[2] += int(out if out is not None else dflt)
        rows += [(m, trace, w, *v) for (m, w), v in acc.items()]
    return rows, shas


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--window-s", type=float, default=10.0)
    a = ap.parse_args(argv)
    rows, shas = window_rows(Path(a.data_root), a.window_s)
    stats, bounds = {}, {"a": {}, "b": {}, "c": {}}
    for m in sorted({r[0] for r in rows}):
        rr = [r for r in rows if r[0] == m]
        n, P, D = (np.array([r[i] for r in rr], float) for i in (3, 4, 5))
        sh, li, lo = P / (P + D), P / n, D / n
        stats[m] = {"windows": len(rr), "requests": int(n.sum()), "share": _q(sh), "mean_in": _q(li),
                    "mean_out": _q(lo), "max_p_rate": P.max() / a.window_s, "max_d_rate": D.max() / a.window_s,
                    "max_t_rate": (P + D).max() / a.window_s}
        bounds["a"][m] = {"share": [float(sh.min()), float(sh.max())]}
        bounds["b"][m] = {"share": [float(np.percentile(sh, 5)), float(np.percentile(sh, 95))]}
        bounds["c"][m] = {"mean_in": [float(li.min()), float(li.max())], "mean_out": [float(lo.min()), float(lo.max())]}
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    names = {"a": "share-minmax", "b": "share-p5p95", "c": "lengths-minmax"}
    for k, b in bounds.items():
        (out / f"envelope_{k}.yaml").write_text(yaml.safe_dump(
            {"name": names[k], "source": {"window_s": a.window_s, "trace_sha256": shas}, "bounds": b}, sort_keys=True))
    (out / "eval_stats.json").write_text(json.dumps(stats, indent=1, default=float))
    print(json.dumps(stats, indent=1, default=float))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
