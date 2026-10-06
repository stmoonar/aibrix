"""Per-replica capacity and the additive per-request cost model.

A request's work in replica-seconds at the knee is ``s = in / v_p + out / v_d``; ``v_d`` is
fitted so that the reference shape (492/400) costs exactly ``(in + out) / v_b``. So
``rho = sum(s) / window`` is the number of replicas the load needs at the knee, and for the
reference shape ``req/s = rho * v_b / 892``. GPU demand ``G = sum_m rho_m * gpus_m``.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent / "data"
DEFAULT_CAPACITY = DATA_DIR / "capacity_20261006.json"
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

    def cost(self, in_tokens: float, out_tokens: float) -> float:
        """Replica-seconds of one request at the knee."""
        return in_tokens / self.v_p + out_tokens / self.v_d

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


def load_capacity(path: str | Path | None = None) -> Capacity:
    raw = json.loads(Path(path or DEFAULT_CAPACITY).read_text())
    ref_in, ref_out = int(raw["ref_shape"]["in"]), int(raw["ref_shape"]["out"])
    models = {}
    for name, m in raw["models"].items():
        v_d = ref_out / ((ref_in + ref_out) / m["v_b"] - ref_in / m["v_p"])
        models[name] = ModelCap(name, float(m["v_b"]), float(m["v_p"]), v_d, int(m["gpus"]),
                                int(m["max_awake"]), int(m.get("floor", 1)),
                                float(m["knee_ttft_s"]), float(m["knee_tpot_s"]))
    return Capacity(raw["name"], int(raw["pool_gpus"]), float(raw["u_target"]), ref_in, ref_out, models, raw)


def load_fits(path: str | Path | None = None) -> dict:
    p = Path(path or DEFAULT_FITS)
    return json.loads(p.read_text()) if p.exists() else {}
