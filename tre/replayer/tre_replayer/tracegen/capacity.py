"""Per-replica capacity and the additive per-request cost model.

A request's work in replica-seconds at the knee is ``in / v_p + out / v_d``; ``v_d`` is
fitted so that the reference shape (492/400) costs exactly ``(in + out) / v_b``. GPU demand
``G = sum_m rho_m * gpus_m``.

Calibration (``calibration.factor[m]`` in the capacity file, 2026-10-07 frozen rule in
docs/trace-design-v2.md): rho = 1 is one replica at ``factor * V_b`` (V_slo, the highest
open-loop rate meeting the SLO on the traces' own lengths), not at the knee. ``cost`` is the
knee cost divided by the factor, so ``rho = sum(cost) / window`` counts replicas at V_slo
and, for the reference shape, ``req/s = rho * factor * v_b / 892``. Without the field the
factor is 1 (rho = 1 at the knee, the 2026-10-06 files).
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent / "data"
DEFAULT_CAPACITY = DATA_DIR / "capacity_20261007_vslo.json"
DEFAULT_FITS = DATA_DIR / "azure_fits.json"


@dataclass(frozen=True)
class ModelCap:
    name: str
    v_b: float
    v_p: float
    v_d: float
    gpus: int
    max_awake: int
    floor: int
    knee_ttft_s: float
    knee_tpot_s: float
    knee_p95_ttft_s: float | None = None
    knee_p95_tpot_s: float | None = None
    out_max: int | None = None
    #: V_slo / V_b: the share of the knee rate one replica serves within the SLO (rho = 1).
    rho_factor: float = 1.0

    def cost(self, in_tokens: float, out_tokens: float) -> float:
        """Replica-seconds of one request at rho = 1 (the knee cost / ``rho_factor``)."""
        return (in_tokens / self.v_p + out_tokens / self.v_d) / self.rho_factor

    def service_s(self, out_tokens: float) -> float:
        """Mean time in system of one request at the knee (Little's law W)."""
        return self.knee_ttft_s + out_tokens * self.knee_tpot_s


@dataclass(frozen=True)
class Capacity:
    name: str
    pool_gpus: int
    u_target: float
    ref_in: int
    ref_out: int
    models: dict
    raw: dict

    def replicas_needed(self, model: str, rho: float) -> int:
        m = self.models[model]
        return min(m.max_awake, max(m.floor, math.ceil(rho / self.u_target - 1e-9)))


def route_out_max(timeout_s: float, frac: float, ttft_s: float, tpot_s: float) -> int:
    """Longest output that finishes within ``frac`` of the route timeout at the knee:
    ``floor((frac * timeout_s - ttft_s) / tpot_s)``."""
    return int(math.floor((frac * timeout_s - ttft_s) / tpot_s))


def load_capacity(path: str | Path | None = None) -> Capacity:
    raw = json.loads(Path(path or DEFAULT_CAPACITY).read_text())
    ref_in, ref_out = int(raw["ref_shape"]["in"]), int(raw["ref_shape"]["out"])
    models = {}
    rc = raw.get("route_cap")
    factor = (raw.get("calibration") or {}).get("factor") or {}
    for name, m in raw["models"].items():
        v_d = ref_out / ((ref_in + ref_out) / m["v_b"] - ref_in / m["v_p"])
        out_max = None
        if rc and m.get("knee_p95_tpot_s"):
            out_max = route_out_max(rc["timeout_s"], rc["frac"], m["knee_p95_ttft_s"], m["knee_p95_tpot_s"])
        models[name] = ModelCap(name, float(m["v_b"]), float(m["v_p"]), v_d, int(m["gpus"]),
                                int(m["max_awake"]), int(m.get("floor", 1)),
                                float(m["knee_ttft_s"]), float(m["knee_tpot_s"]),
                                m.get("knee_p95_ttft_s"), m.get("knee_p95_tpot_s"), out_max,
                                float(factor.get(name, 1.0)))
    return Capacity(raw["name"], int(raw["pool_gpus"]), float(raw["u_target"]), ref_in, ref_out, models, raw)


def load_fits(path: str | Path | None = None) -> dict:
    p = Path(path or DEFAULT_FITS)
    return json.loads(p.read_text()) if p.exists() else {}
