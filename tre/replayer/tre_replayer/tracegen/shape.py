"""Shape-aware per-replica capacity (audit only; the generator still sizes by the additive model).

The generator's cost model (:mod:`.capacity`) is linear in tokens and was fitted at 492/400.
Long outputs are KV-bound instead (docs: e1-envoy4 shape-analysis, 8b at out ~1200 serves
~0.48 V_b). Per window of requests with mean input ``i``, mean output ``o`` and mean
squared output ``o2``, the knee in (in+out) tok/s per replica is the smaller of

* compute knee ``V_cmp = (i + o) / (i / v_p + o / v_d)`` (the additive model), and
* KV knee ``V_kv = C * (i + o) / (ttft + o * tpot_kv)``, ``C = min(max_num_seqs, KV / ctx)``,
  ``ctx = i + o2 / (2 o)`` (Little's law on KV tokens; mean context of a request in flight),

the formula of ``shape_cf_ts.py`` (e1-envoy4, checked against the measured saturation need:
8b Decode_burst 2.67 vs measured 2.74, Prefill_mix decode-heavy 2.68 vs 2.78 / 3.22).
``rho_shape = lambda * (i + o) / (rho_factor * min(V_cmp, V_kv))`` is in the same unit as the
generator's rho (replicas at V_slo); it equals the additive rho when compute binds.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

DEFAULT_SHAPE = Path(__file__).resolve().parent / "data" / "shape_capacity_20261019.json"


@dataclass(frozen=True)
class ShapeCap:
    kv_tokens: float
    tpot_kv_s: float
    ttft_s: float
    max_num_seqs: int


def load_shape(path: str | Path | None = None) -> dict:
    raw = json.loads(Path(path or DEFAULT_SHAPE).read_text())
    return {m: ShapeCap(float(v["kv_tokens"]), float(v["tpot_kv_s"]), float(raw["ttft_s"]), int(raw["max_num_seqs"]))
            for m, v in raw["models"].items()}


def knee(mc, sc: ShapeCap, i: float, o: float, o2: float) -> tuple[float, str]:
    """(knee tok/s per replica, binding 'kv' | 'cmp') for a window's mean shape."""
    v_cmp = (i + o) / (i / mc.v_p + o / mc.v_d)
    ctx = i + (o2 / (2 * o) if o > 0 else 0.0)
    c = min(sc.max_num_seqs, sc.kv_tokens / ctx) if ctx > 0 else sc.max_num_seqs
    v_kv = c * (i + o) / (sc.ttft_s + o * sc.tpot_kv_s)
    return (v_kv, "kv") if v_kv < v_cmp else (v_cmp, "cmp")


def rho_series(rows: list[dict], model: str, mc, sc: ShapeCap, duration_s: float, width: int = 30) -> list[dict]:
    """Per ``width`` s window of one model's requests: additive rho, shape-aware rho, binding."""
    n = int(math.ceil(duration_s / width))
    acc = [[0, 0.0, 0.0, 0.0, 0.0] for _ in range(n)]   # count, sum in, sum out, sum out^2, additive cost
    for r in rows:
        if r["model_name"] != model:
            continue
        b = min(n - 1, int(r["timestamp"] // width))
        i, o = r["prompt_length"], r["max_output_tokens"]
        a = acc[b]
        a[0] += 1; a[1] += i; a[2] += o; a[3] += o * o; a[4] += mc.cost(i, o)
    out = []
    for k, (c, si, so, so2, cost) in enumerate(acc):
        if c == 0:
            out.append({"t0": k * width, "rho_add": 0.0, "rho_shape": 0.0, "bind": None})
            continue
        i, o, o2 = si / c, so / c, so2 / c
        v, bind = knee(mc, sc, i, o, o2)
        lam = c / width
        out.append({"t0": k * width, "rho_add": cost / width, "rho_shape": lam * (i + o) / (mc.rho_factor * v),
                    "bind": bind, "in": i, "out": o})
    return out
