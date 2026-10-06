"""Real-* the v1 way: v1's per-second slices, scaled by one factor per trace.

v1 (ICSE) built ``Real_{conv_2023,code_2024}_slice_<x>_tok70`` in steps (reproduced from the
Azure CSVs, see docs/trace-design-v2.md "Real-* 的 v1 方法"):

1. ReTrace "realtime": per-second request counts of a dataset window, min-max mapped to a
   range (conv 2024 week [118800, 126000) s -> [10, 80]; code 2024 week [86400, 88200) s ->
   [50, 200]); every request gets lengths drawn from the dataset's marginal CDFs.
2. v8: each model takes its own 720 s window of that series and min-max maps the counts to
   8b [2, 11], 7b [3, 13], 14b [1, 7] (dropping requests at random); lengths min-max to 70 % of
   a target range.
3. v9: counts min-max again to [2, 10] / [3, 10] / [1, 7]; lengths to in [250, 700],
   out [80, 500] (conv) / [80, 400] (code).
4. scaling: each second's count x a factor (conv x4; code 8b, 7b x5, 14b x2.5), rounded; the
   added requests are copies of a random request of the same second whose input and output
   are both multiplied by one factor ~ U(0.85, 1.15) (inferred from the v1 files: the v1
   script of this step is not kept); request times uniform inside the second.

This module starts from the v9 files (step 3 output; v1's own requests, lengths included) and
applies step 4 with ``factor_m = k * v1_factor_m``. ``k = 1`` is v1. ``k`` is the one
documented deviation: v1 chose its factors after seeing results; here ``k`` is solved so that
the expected mean GPU demand over the trace's slices is ``target_mean_g`` (:func:`solve_k`).
Outputs are capped at the model's route-timeout ``out_max``.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

from . import azure

MODELS = ("dsllama-8b", "dsqwen-7b", "dsqwen-14b")


def _load(real: dict, slice_name: str, model: str, sources: dict) -> list:
    root = sources.get(real["source"])
    if root is None:
        raise ValueError(f"real slice needs --source {real['source']}=<dir of v1 traces_v9 files>")
    rel = f"{slice_name}/{model}.json"
    path = Path(root) / rel
    want = real["files_sha256"].get(rel)
    got = azure.sha256_file(path)
    if want != got:  # named by sha256, never by path
        raise ValueError(f"{path}: sha256 {got} != {want}")
    return json.loads(path.read_text())


def factors(real: dict, k: float) -> dict:
    return {m: k * float(f) for m, f in real["v1_factor"].items()}


def expected_mean_g(real: dict, slice_name: str, k: float, cap, sources: dict, duration_s: float) -> float:
    """Mean GPU demand of one slice at factor k (each second: round(n f) requests at the mean
    cost of its base requests; the jitter has mean 1)."""
    total = 0.0
    for m, f in factors(real, k).items():
        mc = cap.models[m]
        for b in _load(real, slice_name, m, sources):
            reqs = b["requests"]
            if not reqs:
                continue
            c = sum(mc.cost(q["input_tokens"], min(q["output_tokens"], mc.out_max or 10**9)) for q in reqs) / len(reqs)
            total += round(len(reqs) * f) * c * mc.gpus
    return total / duration_s


def solve_k(spec: dict, cap, sources: dict) -> dict:
    """k such that the mean over the spec's slices of the expected mean G is the target."""
    real = spec["real"]
    target = float(real["k_rule"]["target_mean_g"])
    slices = list(real["slice_by_seed"].values())
    dur = float(spec["duration_s"])

    def g(k):
        return sum(expected_mean_g(real, s, k, cap, sources, dur) for s in slices) / len(slices)
    lo, hi = 0.05, 20.0
    for _ in range(60):
        mid = (lo + hi) / 2
        if g(mid) < target:
            lo = mid
        else:
            hi = mid
    k = round((lo + hi) / 2, 4)
    return {"k": k, "mean_g_at_k": g(k), "mean_g_at_v1": g(1.0),
            "per_slice_at_k": {s: expected_mean_g(real, s, k, cap, sources, dur) for s in slices}}


def rows(spec: dict, seed: int, cap, sources: dict, rng_for) -> tuple[list, dict]:
    real = spec["real"]
    slice_name = real["slice_by_seed"][str(seed)]
    k = float(real["k"])
    lo_j, hi_j = (float(x) for x in real.get("dup_jitter", (0.85, 1.15)))
    out = []
    capped = {}
    for m, f in factors(real, k).items():
        mc = cap.models[m]
        rng = rng_for(seed, spec["trace"], m, "v1-scale")
        n_cap = 0
        for b in _load(real, slice_name, m, sources):
            base = [(int(q["input_tokens"]), int(q["output_tokens"])) for q in b["requests"]]
            if not base:
                continue
            want = int(round(len(base) * f))
            if want <= len(base):
                keep = sorted(rng.sample(range(len(base)), want))
                reqs = [base[i] for i in keep]
            else:
                reqs = list(base)
                for _ in range(want - len(base)):
                    i, o = base[rng.randrange(len(base))]
                    j = rng.uniform(lo_j, hi_j)
                    reqs.append((max(1, int(round(i * j))), max(1, int(round(o * j)))))
            t0, t1 = float(b["start_time"]), float(b["end_time"])
            for i, o in reqs:
                if mc.out_max is not None and o > mc.out_max:
                    o = mc.out_max
                    n_cap += 1
                out.append((t0 + rng.random() * (t1 - t0) * (1 - 1e-9), m, max(1, i), o))
        capped[m] = n_cap
    info = {"method": "v1", "slice": slice_name, "k": k, "factors": factors(real, k),
            "v1_factor": real["v1_factor"], "route_cap_clamped": capped}
    return out, info
