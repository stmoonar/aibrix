"""Load envelope of the evaluation traces, per model, per 10 s window of arrivals, and the three envelope files
for ``preserve_mu alg1 --envelope``:  a = P share [min, max],  b = P share [p5, p95],  c = mean input and mean
output length each [min, max] (c is the default of the PreServe mu calibration, user 2026-11-04).

    python3 -m tre_baselines.tools.preserve_eval_envelope --config tre_baselines/tools/preserve_mu_envelope.yaml \\
        --data-root R --out DIR

The config names the trace set (``traces``: set dir / trace / pinned sha256 of ``trace_file``; refused on a
mismatch) and ``window_s``. A request's output length = its ``max_output_tokens``, or, where it has none, the
``max_tokens`` of its model in the trace's loadgen config (``<tre>/<loadgen_config_root>/<trace>/config.yaml``,
the value the replayer sends). Deterministic: the same trace files give the same bytes. Writes
``envelope_{a,b,c}.yaml``, ``eval_stats.json`` and ``provenance.json`` (tool sha, config / trace / loadgen
config sha256).
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import yaml

TRE_DIR = Path(__file__).resolve().parents[3]
NAMES = {"a": "share-minmax", "b": "share-p5p95", "c": "lengths-minmax"}


def _sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _q(x):
    return {"min": x.min(), "max": x.max(), "p1": np.percentile(x, 1), "p99": np.percentile(x, 99),
            "p5": np.percentile(x, 5), "p95": np.percentile(x, 95)}


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh) or {}
    miss = [k for k in ("window_s", "default", "trace_file", "loadgen_config_root", "traces") if k not in cfg]
    if miss:
        raise SystemExit(f"preserve_eval_envelope: {path}: missing {miss}")
    if cfg["default"] not in NAMES:
        raise SystemExit(f"preserve_eval_envelope: {path}: default must be one of {sorted(NAMES)}")
    for t in cfg["traces"]:
        if not all(t.get(k) for k in ("set", "trace", "sha256")):
            raise SystemExit(f"preserve_eval_envelope: {path}: each trace needs set, trace, sha256")
    return cfg


def loadgen_max_tokens(path: Path) -> dict:
    """model -> max_tokens of a loadgen config (its ``models`` list, wherever it sits)."""
    out: dict = {}

    def walk(node: Any) -> None:
        if isinstance(node, Mapping):
            for k, v in node.items():
                if k == "models" and isinstance(v, list):
                    out.update({m["name"]: int(m["max_tokens"]) for m in v
                                if isinstance(m, Mapping) and "name" in m and m.get("max_tokens") is not None})
                else:
                    walk(v)
        elif isinstance(node, list):
            for v in node:
                walk(v)

    walk(yaml.safe_load(path.read_text(encoding="utf-8")))
    return out


def window_rows(cfg: Mapping, root: Path, tre_dir: Path = TRE_DIR):
    window_s = float(cfg["window_s"])
    rows, shas, lg_shas = [], {}, {}
    for t in cfg["traces"]:
        f = root / t["set"] / t["trace"] / cfg["trace_file"]
        raw = f.read_bytes()
        shas[t["trace"]] = _sha(raw)
        if shas[t["trace"]] != t["sha256"]:
            raise SystemExit(f"preserve_eval_envelope: {f}: sha256 {shas[t['trace']]} != pinned {t['sha256']}")
        lg = tre_dir / cfg["loadgen_config_root"] / t["trace"] / "config.yaml"
        dflt = loadgen_max_tokens(lg) if lg.is_file() else {}
        if lg.is_file():
            lg_shas[t["trace"]] = _sha(lg.read_bytes())
        acc = collections.defaultdict(lambda: [0, 0, 0])
        for r in json.loads(raw):
            out = r.get("max_output_tokens")
            if out is None:
                out = dflt.get(r["model_name"])
                if out is None:
                    raise SystemExit(f"preserve_eval_envelope: {t['trace']}: a {r['model_name']} request has no "
                                     f"max_output_tokens and {lg} gives no max_tokens")
            a = acc[(r["model_name"], int(r["timestamp"] // window_s))]
            a[0] += 1
            a[1] += int(r["prompt_length"])
            a[2] += int(out)
        rows += [(m, t["trace"], w, *v) for (m, w), v in acc.items()]
    return rows, shas, lg_shas


def envelopes(rows, window_s: float) -> tuple:
    stats, bounds = {}, {"a": {}, "b": {}, "c": {}}
    for m in sorted({r[0] for r in rows}):
        rr = [r for r in rows if r[0] == m]
        n, P, D = (np.array([r[i] for r in rr], float) for i in (3, 4, 5))
        sh, li, lo = P / (P + D), P / n, D / n
        stats[m] = {"windows": len(rr), "requests": int(n.sum()), "share": _q(sh), "mean_in": _q(li),
                    "mean_out": _q(lo), "max_p_rate": P.max() / window_s, "max_d_rate": D.max() / window_s,
                    "max_t_rate": (P + D).max() / window_s}
        bounds["a"][m] = {"share": [float(sh.min()), float(sh.max())]}
        bounds["b"][m] = {"share": [float(np.percentile(sh, 5)), float(np.percentile(sh, 95))]}
        bounds["c"][m] = {"mean_in": [float(li.min()), float(li.max())], "mean_out": [float(lo.min()), float(lo.max())]}
    return stats, bounds


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True, help="preserve_mu_envelope.yaml")
    ap.add_argument("--data-root", required=True, help="root of the trace set dirs")
    ap.add_argument("--out", required=True, help="output dir (must not hold envelope files)")
    a = ap.parse_args(argv)
    from tre_baselines.tools.preserve_mu import _git_sha

    cfg = load_config(a.config)
    out = Path(a.out)
    if any((out / f"envelope_{k}.yaml").exists() for k in NAMES):
        raise SystemExit(f"preserve_eval_envelope: {out} holds envelope files; use a new --out")
    window_s = float(cfg["window_s"])
    rows, shas, lg_shas = window_rows(cfg, Path(a.data_root))
    stats, bounds = envelopes(rows, window_s)
    out.mkdir(parents=True, exist_ok=True)
    for k, b in bounds.items():
        (out / f"envelope_{k}.yaml").write_text(yaml.safe_dump(
            {"name": NAMES[k], "source": {"window_s": window_s, "trace_sha256": shas}, "bounds": b}, sort_keys=True))
    (out / "eval_stats.json").write_text(json.dumps(stats, indent=1, default=float))
    prov = {"tool": "tre_baselines.tools.preserve_eval_envelope", "tool_sha": _git_sha(),
            "config": {"path": a.config, "sha256": _sha(Path(a.config).read_bytes())},
            "default": {"variant": cfg["default"], "name": NAMES[cfg["default"]],
                        "file": f"envelope_{cfg['default']}.yaml"},
            "trace_file": cfg["trace_file"], "trace_sha256": shas, "loadgen_config_sha256": lg_shas,
            "files_sha256": {f.name: _sha(f.read_bytes()) for f in sorted(out.glob("envelope_*.yaml"))}}
    (out / "provenance.json").write_text(json.dumps(prov, indent=1, sort_keys=True) + "\n")
    print(json.dumps(stats, indent=1, default=float))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
