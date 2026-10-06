"""Trace spec + seed -> per-request design plan (no prompt text) + manifest.

Spec (JSON)::

    {"trace": "Alternating", "duration_s": 1080, "isolates": "...",
     "defaults": {"lengths": {"in": {...}, "out": {...}}},          # see lengths.LengthModel
     "models": {"dsllama-8b": {"rate": {"unit": "rho", "fn": <function spec>},
                               "lengths": {"out": {...}}}, ...},     # per-model overrides
     "audit": {...}}                                                  # see audit.py

``rate.unit``: ``rho`` (replicas at the knee; req/s = rho / E[cost] at the *current* mean
lengths), ``rho_ref`` (req/s = rho / cost(reference shape): the arrival rate does not move
when the lengths do), ``rps``. Function specs: :mod:`.rates`.

Real slices instead of ``models``: ``"real": {"dataset": "conv2024", "offset_s": ... (or
"offsets_s_by_seed": {"<seed>": ...}: the seed picks the slice),
"rho": {"<model>": target mean rho, ...}, "in_min": 32}`` (needs ``--azure-csv``): the
dataset's own arrivals and lengths in ``[offset_s, offset_s + duration_s)``, lengths clamped
at the dataset's p99.5; ``ceil(sum lambda / native rate)`` consecutive windows are overlaid
when the window alone is too thin, then every request is assigned to one model (or dropped)
with probability ``lambda_m / rate`` - an independent thinning split of the real process.

Arrivals: non-homogeneous Poisson by thinning (Lewis-Shedler) with a bound taken on a
0.1 s grid (x 1.02); a candidate above the bound raises. Every random stream is its own
``random.Random`` seeded from sha256(seed, trace, model, stream), so streams are
independent and a change to one model or to lengths leaves the other draws unchanged.
"""
from __future__ import annotations

import hashlib
import json
import math
import random
import subprocess
from pathlib import Path
from typing import Any

from . import azure, rates
from .capacity import Capacity, load_capacity, load_fits
from .lengths import LengthModel

GENERATOR = "tre_replayer.tracegen"
FORMAT = "tre-trace-plan-v2"
PHASE_TYPE = "tracegen-v2"
DESIGN_FIELDS = ("request_id", "timestamp", "model_name", "prompt_length", "phase_type", "max_output_tokens")
DESIGN_FILE = "design.json"
MANIFEST_FILE = "manifest.json"


def rng_for(seed: int, trace: str, model: str, stream: str) -> random.Random:
    d = hashlib.sha256(f"{seed}|{trace}|{model}|{stream}".encode()).digest()
    return random.Random(int.from_bytes(d[:8], "big"))


def _merge(a: dict, b: dict) -> dict:
    out = dict(a)
    for k, v in (b or {}).items():
        out[k] = _merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) and "kind" not in v else v
    return out


class ModelPlan:
    """One model's design functions: lambda(t) req/s, rho(t), length models."""

    def __init__(self, name: str, mspec: dict, spec: dict, seed: int, cap: Capacity, fits: dict):
        self.name = name
        self.cap = cap.models[name]
        dur = float(spec["duration_s"])
        lspec = _merge(spec.get("defaults", {}).get("lengths", {}), mspec.get("lengths", {}))
        self.lengths = {}
        for side in ("in", "out"):
            s = lspec[side]
            mean_fn = rates.build(s["mean"], duration_s=dur, rng=rng_for(seed, spec["trace"], name, f"{side}-mean"))
            self.lengths[side] = LengthModel(s, fits, mean_fn)
        r = mspec["rate"]
        self.unit = r.get("unit", "rho")
        self.fn = rates.build(r["fn"], duration_s=dur, rng=rng_for(seed, spec["trace"], name, "rate"))
        self.ref_cost = self.cap.cost(cap.ref_in, cap.ref_out)

    def mean_cost(self, t: float) -> float:
        return self.cap.cost(self.lengths["in"].mean(t), self.lengths["out"].mean(t))

    def lam(self, t: float) -> float:
        v = max(0.0, self.fn(t))
        if self.unit == "rps":
            return v
        if self.unit == "rho_ref":
            return v / self.ref_cost
        if self.unit == "rho":
            return v / self.mean_cost(t)
        raise ValueError(f"unknown rate unit {self.unit!r}")

    def rho(self, t: float) -> float:
        return self.lam(t) * self.mean_cost(t)


def model_plans(spec: dict, seed: int, cap: Capacity, fits: dict) -> dict:
    return {m: ModelPlan(m, ms, spec, seed, cap, fits) for m, ms in spec.get("models", {}).items()}


def nhpp(lam, duration_s: float, rng: random.Random, grid_s: float = 0.1) -> list[float]:
    """Arrival times of a non-homogeneous Poisson process with intensity ``lam`` on [0, duration_s)."""
    n = int(math.ceil(duration_s / grid_s))
    bound = max(lam(min(duration_s, i * grid_s)) for i in range(n + 1)) * 1.02
    if bound <= 0:
        return []
    out, t = [], 0.0
    while True:
        t += rng.expovariate(bound)
        if t >= duration_s:
            return out
        v = lam(t)
        if v > bound:
            raise AssertionError(f"intensity {v} above the thinning bound {bound} at t={t}")
        if rng.random() * bound < v:
            out.append(t)


def _synthetic_rows(spec: dict, seed: int, cap: Capacity, fits: dict) -> list[tuple]:
    rows = []
    dur = float(spec["duration_s"])
    for name, plan in model_plans(spec, seed, cap, fits).items():
        times = nhpp(plan.lam, dur, rng_for(seed, spec["trace"], name, "arrivals"))
        lrng = rng_for(seed, spec["trace"], name, "lengths")
        for t in times:
            rows.append((t, name, plan.lengths["in"].draw(t, lrng), plan.lengths["out"].draw(t, lrng)))
    return rows


def _real_rows(spec: dict, seed: int, cap: Capacity, fits: dict, azure_csv: dict) -> tuple[list, dict]:
    real = spec["real"]
    ds = real["dataset"]
    if ds not in azure_csv:
        raise ValueError(f"real slice needs --azure-csv {ds}=<path>")
    fit = fits[ds]
    dur = float(spec["duration_s"])
    by_seed = real.get("offsets_s_by_seed")
    offset = float(by_seed[str(seed)] if by_seed else real["offset_s"])
    in_min = int(real.get("in_min", 32))
    in_max = int(real.get("in_max") or fit["in"]["p99_5"])
    out_max = int(real.get("out_max") or fit["out"]["p99_5"])

    def clamp(rows):
        return [(t, min(in_max, max(in_min, i)), min(out_max, max(1, o))) for t, i, o in rows]
    base = clamp(azure.load_window(azure_csv[ds], offset, offset + dur))
    rate0 = len(base) / dur
    lam = {}
    for m, rho in real["rho"].items():
        c = sum(cap.models[m].cost(i, o) for _, i, o in base) / len(base)
        lam[m] = float(rho) / c
    copies = max(1, math.ceil(sum(lam.values()) / rate0))
    rows_all = list(base)
    for j in range(1, copies):
        rows_all += clamp(azure.load_window(azure_csv[ds], offset + j * dur, offset + (j + 1) * dur))
    rows_all.sort()
    total = len(rows_all) / dur
    probs = [(m, lam[m] / total) for m in real["rho"]]
    if sum(p for _, p in probs) > 1 + 1e-9:
        raise AssertionError("split probabilities exceed 1")
    rng = rng_for(seed, spec["trace"], "*", "split")
    out = []
    for t, i, o in rows_all:
        u = rng.random()
        acc = 0.0
        for m, p in probs:
            acc += p
            if u < acc:
                out.append((t, m, i, o))
                break
    info = {"dataset": ds, "offset_s": offset, "windows": copies, "native_rate": rate0,
            "rows_used": len(rows_all), "target_lambda": lam, "split_p": dict(probs),
            "clamp": {"in": [in_min, in_max], "out": [1, out_max]}}
    return out, info


def design_rows(spec: dict, seed: int, cap: Capacity | None = None, fits: dict | None = None,
                azure_csv: dict | None = None) -> tuple[list[dict], dict]:
    """The design plan (sorted, ids assigned) and generation info."""
    cap = cap or load_capacity()
    fits = fits if fits is not None else load_fits()
    info: dict[str, Any] = {}
    if "real" in spec:
        raw, info["real"] = _real_rows(spec, seed, cap, fits, azure_csv or {})
    else:
        raw = _synthetic_rows(spec, seed, cap, fits)
    raw.sort(key=lambda r: (r[0], r[1]))
    rows = []
    for k, (t, m, i, o) in enumerate(raw):
        rows.append({"request_id": f"req_{k:06d}", "timestamp": round(t, 6), "model_name": m,
                     "prompt_length": int(i), "phase_type": PHASE_TYPE, "max_output_tokens": int(o)})
    check_design(rows, spec)
    return rows, info


def check_design(rows: list[dict], spec: dict) -> None:
    """Integrity of a design plan: field set, ordering, every request has its own int lengths."""
    dur = float(spec["duration_s"])
    last = -1.0
    ids = set()
    for r in rows:
        if tuple(r.keys()) != DESIGN_FIELDS:
            raise AssertionError(f"design row fields {tuple(r.keys())}")
        if not isinstance(r["max_output_tokens"], int) or r["max_output_tokens"] < 1:
            raise AssertionError(f"{r['request_id']}: max_output_tokens {r['max_output_tokens']!r}")
        if not isinstance(r["prompt_length"], int) or r["prompt_length"] < 1:
            raise AssertionError(f"{r['request_id']}: prompt_length {r['prompt_length']!r}")
        if not (0 <= r["timestamp"] < dur) or r["timestamp"] < last:
            raise AssertionError(f"{r['request_id']}: timestamp {r['timestamp']} out of order/range")
        last = r["timestamp"]
        if r["request_id"] in ids:
            raise AssertionError(f"duplicate request id {r['request_id']}")
        ids.add(r["request_id"])


def dumps_plan(rows: list[dict]) -> bytes:
    """Canonical bytes of a plan (one request per line inside a JSON list)."""
    return ("[\n" + ",\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n]\n").encode("utf-8")


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def code_provenance() -> dict:
    here = Path(__file__).resolve().parent
    def git(*args):
        try:
            return subprocess.run(["git", "-C", str(here), *args], capture_output=True, text=True,
                                  timeout=10).stdout.strip()
        except Exception:  # noqa: BLE001
            return ""
    replayer = here.parents[1]
    return {"module": GENERATOR, "git_sha": git("rev-parse", "HEAD") or None,
            "git_branch": git("rev-parse", "--abbrev-ref", "HEAD") or None,
            "dirty_files": len([l for l in git("status", "--porcelain", "--", str(replayer)).splitlines() if l])}


def summary(rows: list[dict], duration_s: float) -> dict:
    by: dict = {}
    for r in rows:
        s = by.setdefault(r["model_name"], {"requests": 0, "in_sum": 0, "out_sum": 0, "in_max": 0, "out_max": 0})
        s["requests"] += 1
        s["in_sum"] += r["prompt_length"]; s["out_sum"] += r["max_output_tokens"]
        s["in_max"] = max(s["in_max"], r["prompt_length"]); s["out_max"] = max(s["out_max"], r["max_output_tokens"])
    return {m: {"requests": s["requests"], "mean_rps": s["requests"] / duration_s,
                "mean_in": s["in_sum"] / s["requests"], "mean_out": s["out_sum"] / s["requests"],
                "max_in": s["in_max"], "max_out": s["out_max"]} for m, s in sorted(by.items())}


def generate(spec_path: str | Path, seed: int, out_dir: str | Path, *, capacity_path=None, fits_path=None,
             azure_csv: dict | None = None) -> dict:
    """Write ``design.json`` + ``manifest.json`` under ``out_dir``; returns the manifest."""
    spec_bytes = Path(spec_path).read_bytes()
    spec = json.loads(spec_bytes)
    cap = load_capacity(capacity_path)
    fits = load_fits(fits_path)
    rows, info = design_rows(spec, seed, cap, fits, azure_csv)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    plan = dumps_plan(rows)
    (out / DESIGN_FILE).write_bytes(plan)
    used_fits = sorted({k for k in json.dumps(spec).split('"') if k in fits})
    manifest = {
        "format": FORMAT,
        "trace": spec["trace"], "seed": seed, "duration_s": spec["duration_s"],
        "generator": code_provenance(),
        "spec_file": Path(spec_path).name, "spec_sha256": sha256_bytes(spec_bytes), "spec": spec,
        "capacity": cap.raw, "fits": {k: {kk: fits[k][kk] for kk in ("source", "in", "out", "arrivals")} for k in used_fits},
        "generation": info,
        "design": {"file": DESIGN_FILE, "sha256": sha256_bytes(plan), "requests": len(rows),
                   "per_model": summary(rows, float(spec["duration_s"]))},
        "client_contract": "replay the effective file with tre_loadgen_v1 --trace-file <effective> --ignore-eos: "
                           "every request carries its own max_output_tokens (never null), so the config's "
                           "models[].max_tokens is never used",
    }
    (out / MANIFEST_FILE).write_text(json.dumps(manifest, indent=1, ensure_ascii=False) + "\n")
    return manifest
