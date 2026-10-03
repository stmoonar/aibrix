#!/usr/bin/env python3
"""Fit idle TTFT(L) = c_m + b_m * L per model from per-request raw logs.

``c_m`` (ms) / ``b_m`` (ms per prompt token) feed the D6' TTFT label
``max(500 ms, 5 * (c_m + b_m * L))`` (registry ``models[].slo.ttft_idle_c_ms`` /
``ttft_idle_b_ms_per_token``). TTFT is the client-side one of the raw log (``ttft_ms``,
first streamed token minus the instant the request went on the wire), the basis every
SLO label uses; ``L`` is the engine's ``usage.prompt_tokens`` (``input_tokens``).

Input: ``--root DIR`` with ``DIR/<model>/raw/**/<cell>.jsonl`` - the layout of both
:mod:`scripts.ttft_idle_capture` (``raw/idle/``; ``raw/warmup/`` is excluded) and the
calibration campaign (``raw/<cell dir>/``; held-out ``*_M_*`` cells are excluded).

A request is kept only when it is **isolated** within its raw file (one cell, one
replica): http 200 with a TTFT and a prompt length, no stream error, its prompt length
equal to the one it was built to (when recorded), at most ``--max-inflight`` other
requests in flight at its send instant (reconstructed from send/done stamps), no other
request sent during its prefill ``[send, send + ttft]``, and - with ``--min-gap-ms`` - the
previous completion at least that long before its send.

Estimator: Huber IRLS (k = 1.345, MAD scale) over all kept requests; OLS is reported
next to it. Output (``--out`` or stdout): per model ``c``, ``b``, ``n``, per-length
n / median / p95 / residual of the median, the Huber parameters and the sha256 of every
input file. ``--write-registry PATH`` rewrites only the two fields of each fitted model in
that registry file (everything else byte-for-byte); ``--registry-patch`` prints them.

This replaces docs/refactor/p11_evidence/ttft_idle_fit_20260922.py (same selection and
estimator; the 09-22 values are reproduced from the 2026-09-21 campaign raw with
``--root <that campaign>``).
"""
from __future__ import annotations

import argparse
import bisect
import fnmatch
import hashlib
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Optional, Sequence

import numpy as np

DEFAULT_MODELS = ("dsqwen-7b", "dsllama-8b", "dsqwen-14b")
#: Path components (below ``<model>/raw``) never fitted: held-out M cells, warmup cells.
DEFAULT_EXCLUDE = ("*_M_*", "warmup")
SIDE_SUFFIXES = (".instant.jsonl", ".failures.jsonl")
HUBER_K = 1.345
#: --write-registry refuses a fit with fewer kept requests or distinct lengths.
MIN_REQUESTS = 20
MIN_LENGTHS = 3
#: Published precision (the registry's 36.4 / 0.0527 style).
C_DECIMALS = 1
B_DECIMALS = 4
C_FIELD = "ttft_idle_c_ms"
B_FIELD = "ttft_idle_b_ms_per_token"


# ------------------------------------------------------------------ selection

def isolated_requests(records: Sequence[dict], *, max_inflight: int = 0,
                      min_gap_ms: float = 0.0) -> tuple[list[tuple[float, float]], dict]:
    """``(L, ttft_ms)`` of the isolated requests of one raw file, and why the others were
    dropped. In-flight is counted over every record with a send and a done stamp."""
    recs = sorted((r for r in records if r.get("send_ts_ms") is not None and r.get("done_ts_ms") is not None),
                  key=lambda r: r["send_ts_ms"])
    sends = [float(r["send_ts_ms"]) for r in recs]
    dones = sorted(float(r["done_ts_ms"]) for r in recs)
    kept: list[tuple[float, float]] = []
    dropped = {"not_served": 0, "prompt_mismatch": 0, "inflight": 0, "overlap_prefill": 0, "gap": 0}
    for j, r in enumerate(recs):
        if (r.get("ttft_ms") is None or r.get("input_tokens") is None or r.get("http_status") != 200
                or r.get("stream_error")):
            dropped["not_served"] += 1
            continue
        expected = r.get("expected_prompt_tokens")
        if expected is not None and int(expected) != int(r["input_tokens"]):
            dropped["prompt_mismatch"] += 1
            continue
        s = sends[j]
        done_before = bisect.bisect_right(dones, s)
        if j - done_before > max_inflight:
            dropped["inflight"] += 1
            continue
        if j + 1 < len(recs) and sends[j + 1] <= s + float(r["ttft_ms"]):
            dropped["overlap_prefill"] += 1
            continue
        if min_gap_ms > 0 and done_before > 0 and s - dones[done_before - 1] < min_gap_ms:
            dropped["gap"] += 1
            continue
        kept.append((float(r["input_tokens"]), float(r["ttft_ms"])))
    return kept, dropped


def raw_files(model_root: Path, exclude: Sequence[str] = DEFAULT_EXCLUDE) -> list[Path]:
    """Every per-request raw JSONL under ``<model_root>/raw`` not excluded by a path
    component matching one of ``exclude`` (fnmatch)."""
    raw = model_root / "raw"
    out = []
    for p in sorted(raw.rglob("*.jsonl")):
        if p.name.endswith(SIDE_SUFFIXES):
            continue
        parts = p.relative_to(raw).parts[:-1]
        if any(fnmatch.fnmatch(part, pat) for part in parts for pat in exclude):
            continue
        out.append(p)
    return out


def _load_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# ------------------------------------------------------------------ estimator

def huber(x: np.ndarray, y: np.ndarray, k: float = HUBER_K, max_iter: int = 100) -> dict:
    """Huber IRLS for ``y = c + b x`` with a MAD scale re-estimated every iteration."""
    X = np.c_[np.ones_like(x), x]
    w = np.ones_like(y)
    it = 0
    scale = 1.0
    beta = np.zeros(2)
    for it in range(1, max_iter + 1):
        sw = np.sqrt(w)
        beta, *_ = np.linalg.lstsq(X * sw[:, None], y * sw, rcond=None)
        r = y - X @ beta
        scale = 1.4826 * float(np.median(np.abs(r - np.median(r)))) or 1.0
        u = np.abs(r) / (k * scale)
        wn = np.where(u <= 1, 1.0, 1.0 / u)
        if np.allclose(wn, w):
            break
        w = wn
    return {"c_ms": float(beta[0]), "b_ms_per_token": float(beta[1]), "k": k,
            "scale_ms": scale, "iterations": it, "downweighted": int((w < 1).sum())}


def fit_pairs(pairs: Sequence[tuple[float, float]], *, k: float = HUBER_K) -> dict:
    x = np.array([p[0] for p in pairs], dtype=float)
    y = np.array([p[1] for p in pairs], dtype=float)
    if len(x) < 2 or len(set(x.tolist())) < 2:
        raise ValueError(f"need at least two distinct prompt lengths, have {sorted(set(x.tolist()))}")
    h = huber(x, y, k=k)
    pred = h["c_ms"] + h["b_ms_per_token"] * x
    ss_tot = float(np.sum((y - y.mean()) ** 2)) or 1.0
    b_ols, c_ols = np.polyfit(x, y, 1)
    pred_ols = c_ols + b_ols * x
    per = {}
    for length in sorted(set(x.tolist())):
        ys = y[x == length]
        med = float(np.median(ys))
        per[str(int(length))] = {
            "n": int(len(ys)), "median_ms": med, "p95_ms": float(np.percentile(ys, 95)),
            "residual_median_ms": med - (h["c_ms"] + h["b_ms_per_token"] * length),
        }
    return {
        "n": int(len(x)), "n_lengths": len(per),
        "c_ms": h["c_ms"], "b_ms_per_token": h["b_ms_per_token"],
        "r2": 1.0 - float(np.sum((y - pred) ** 2)) / ss_tot,
        "huber": h,
        "ols": {"c_ms": float(c_ols), "b_ms_per_token": float(b_ols),
                "r2": 1.0 - float(np.sum((y - pred_ols) ** 2)) / ss_tot},
        "per_length": per,
    }


def fit_model(model_root: Path, *, exclude: Sequence[str] = DEFAULT_EXCLUDE, max_inflight: int = 0,
              min_gap_ms: float = 0.0, k: float = HUBER_K) -> dict:
    pairs: list[tuple[float, float]] = []
    inputs = []
    dropped_total: dict[str, int] = {}
    for path in raw_files(model_root, exclude):
        records = _load_jsonl(path)
        kept, dropped = isolated_requests(records, max_inflight=max_inflight, min_gap_ms=min_gap_ms)
        pairs.extend(kept)
        for key, v in dropped.items():
            dropped_total[key] = dropped_total.get(key, 0) + v
        inputs.append({"path": str(path), "sha256": _sha256(path), "records": len(records), "kept": len(kept)})
    if not pairs:
        raise ValueError(f"no isolated requests under {model_root / 'raw'}")
    out = fit_pairs(pairs, k=k)
    out.update({"dropped": dropped_total, "n_files": len(inputs),
                "n_files_contributing": sum(1 for i in inputs if i["kept"]), "inputs": inputs})
    return out


# ------------------------------------------------------------------ registry

def published(fit: dict) -> dict:
    return {C_FIELD: round(fit["c_ms"], C_DECIMALS), B_FIELD: round(fit["b_ms_per_token"], B_DECIMALS)}


def _model_block(lines: list[str], model: str) -> tuple[int, int]:
    """[start, end) line span of ``model``'s entry in the top-level ``models:`` list."""
    top = next((i for i, ln in enumerate(lines) if re.match(r"^models:\s*(#.*)?$", ln)), None)
    if top is None:
        raise ValueError("registry has no top-level 'models:' key")
    start, end, indent = None, len(lines), None
    for i in range(top + 1, len(lines)):
        if re.match(r"^[A-Za-z_]", lines[i]):  # the next top-level key ends the list
            end = i
            break
        item = re.match(r"^(\s*)- ", lines[i])
        if item is None or (indent is not None and item.group(1) != indent):
            continue  # not an entry of the models list (a list nested inside one)
        indent = item.group(1)
        if start is not None:  # the next entry ends this one
            end = i
            break
        m = re.match(r"^\s*- name:\s*['\"]?([^'\"#\s]+)", lines[i])
        if m and m.group(1) == model:
            start = i
    if start is None:
        raise ValueError(f"registry has no model {model!r}")
    return start, end


def rewrite_registry_text(text: str, values: dict[str, dict]) -> str:
    """``text`` with only ``ttft_idle_c_ms`` / ``ttft_idle_b_ms_per_token`` of each model in
    ``values`` replaced (indent and trailing comments kept). Raises if a field is missing."""
    lines = text.splitlines(keepends=True)
    for model, fields in values.items():
        start, end = _model_block(lines, model)
        for field_name, value in fields.items():
            pat = re.compile(rf"^(\s+{re.escape(field_name)}:\s*)([^#\s]+)(.*)$", re.S)
            hits = [i for i in range(start, end) if pat.match(lines[i])]
            if len(hits) != 1:
                raise ValueError(f"model {model}: expected one {field_name!r} line, found {len(hits)}")
            i = hits[0]
            lines[i] = pat.sub(lambda m: f"{m.group(1)}{value}{m.group(3)}", lines[i])
    return "".join(lines)


def check_only_fields_changed(before: str, after: str, values: dict[str, dict]) -> None:
    """Parse both texts; everything but the rewritten fields must be equal."""
    import yaml

    a, b = yaml.safe_load(before), yaml.safe_load(after)
    for doc in (a, b):
        for entry in doc.get("models") or []:
            if entry.get("name") in values:
                for f in values[entry["name"]]:
                    (entry.get("slo") or {}).pop(f, None)
    if a != b:
        raise ValueError("registry rewrite changed more than the idle TTFT fields")
    got = {e["name"]: e.get("slo") or {} for e in yaml.safe_load(after).get("models") or []}
    for model, fields in values.items():
        for f, v in fields.items():
            if float(got[model][f]) != float(v):
                raise ValueError(f"registry rewrite: {model}.{f} reads {got[model][f]}, wanted {v}")


def refuse_thin(fit: dict, model: str) -> None:
    if fit["n"] < MIN_REQUESTS or fit["n_lengths"] < MIN_LENGTHS:
        raise SystemExit(f"refusing to publish {model}: {fit['n']} request(s) over {fit['n_lengths']} "
                         f"length(s) (need >= {MIN_REQUESTS} over >= {MIN_LENGTHS})")


# ------------------------------------------------------------------ CLI

def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, required=True, help="directory holding <model>/raw/")
    ap.add_argument("--models", default=",".join(DEFAULT_MODELS))
    ap.add_argument("--exclude", action="append", default=None,
                    help=f"path component pattern under raw/ to skip (repeatable; default {list(DEFAULT_EXCLUDE)})")
    ap.add_argument("--max-inflight", type=int, default=0, help="other requests allowed in flight at send")
    ap.add_argument("--min-gap-ms", type=float, default=0.0,
                    help="minimum time since the previous completion (0 = not required)")
    ap.add_argument("--huber-k", type=float, default=HUBER_K)
    ap.add_argument("--out", type=Path, default=None, help="JSON output (default stdout)")
    ap.add_argument("--registry-patch", action="store_true", help="print the registry values to set")
    ap.add_argument("--write-registry", type=Path, default=None,
                    help="rewrite only the two idle TTFT fields of each fitted model in this file")
    args = ap.parse_args(argv)
    models = [m for m in args.models.split(",") if m]
    exclude = tuple(args.exclude) if args.exclude is not None else DEFAULT_EXCLUDE

    fits = {m: fit_model(args.root / m, exclude=exclude, max_inflight=args.max_inflight,
                         min_gap_ms=args.min_gap_ms, k=args.huber_k) for m in models}
    doc = {
        "generated_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "script": "deploy/scripts/ttft_idle_fit.py",
        "root": str(args.root), "exclude": list(exclude), "max_inflight": args.max_inflight,
        "min_gap_ms": args.min_gap_ms,
        "ttft_basis": "client-side ttft_ms (raw log)", "length": "usage.prompt_tokens (input_tokens)",
        "models": {m: {"published": published(f), **f} for m, f in fits.items()},
    }
    text = json.dumps(doc, indent=1, sort_keys=True) + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8")
    else:
        sys.stdout.write(text)
    for m, f in fits.items():
        print(f"{m}: c={f['c_ms']:.2f} ms b={f['b_ms_per_token']:.5f} ms/token n={f['n']} "
              f"lengths={f['n_lengths']} r2={f['r2']:.3f}", file=sys.stderr)
    values = {m: published(f) for m, f in fits.items()}
    if args.registry_patch:
        for m, v in values.items():
            print(f"# models[name={m}].slo\n{C_FIELD}: {v[C_FIELD]}\n{B_FIELD}: {v[B_FIELD]}")
    if args.write_registry:
        for m, f in fits.items():
            refuse_thin(f, m)
        before = args.write_registry.read_text(encoding="utf-8")
        after = rewrite_registry_text(before, values)
        check_only_fields_changed(before, after, values)
        args.write_registry.write_text(after, encoding="utf-8")
        print(f"wrote {args.write_registry}: {json.dumps(values, sort_keys=True)}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
