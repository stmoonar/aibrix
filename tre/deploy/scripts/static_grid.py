"""Static steady-state grid cells for the calibration campaign (opt-in, ``--static-grid``).

Plan 2026-09-21 §6.9g: v1 calibrated on a wide static grid (prompt 300..800 x output
300..800 x RPS 4..7, 120 s per point), v2 on seven shapes. This module plans a small
subsample of v1's idea - a few static (input, output, rho) cells per family that fill the
lengths between the seven shapes - without touching the default campaign.

Where a cell's offered load comes from
--------------------------------------
Each grid shape is placed relative to its own SLO boundary, not to a capacity prior:
offered rps = ``(rho/rho*) x r*(i, o)`` with ``rho/rho*`` in
:data:`scripts.gen_calibration_schedules.STATIC_GRID_RHO_FRACTIONS`. ``r*`` is the
boundary load (``rho_star x capacity_used_rps``) measured by a previous campaign's
boundary search - by default the 2026-09-21 campaign - interpolated to (i, o) with the
same two-parameter surface ``1/r = i/P + o/D`` the schedule generator fits capacity with
(:func:`scripts.gen_calibration_schedules.fit_capacity_model`). A shape whose search
never saw a violation (``boundary_found`` false: its rho* is only a lower bound) does not
enter the boundary fit. The cell's rho is expressed against the same interpolated
capacity ``C(i, o)`` (``capacity_used_rps`` surface), so ``rho = (rho/rho*) x rho*`` with
``rho* = r*/C``, and the offered rps is ``rho x C`` exactly as for every other hold.

Every cell is the same constant-rate open-loop hold the boundary search drives
(``gen.hold_segments``, through ``r3_grid`` with the campaign's own client rules:
ignore_eos, unique natural prompts, no retries, failures recorded, shed -> void). The
cells are training cells: not in the schedule index's held-out list, never M, and their
family comes from the prompt/output ratio (``gen.static_grid_family``).
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Mapping, Optional, Sequence

from scripts import gen_calibration_schedules as gen

#: The campaign whose measured capacities and boundaries place the grid.
DEFAULT_SOURCE_CAMPAIGN = Path("/data/nfs_shared_data/xxy/calibration_20260921")

#: ``capacity_source`` recorded on every static cell's schedule metadata.
CAPACITY_SOURCE = "static_grid_boundary_fit"


@dataclass(frozen=True)
class BoundarySurface:
    """One model's interpolated capacity and SLO-boundary surfaces."""

    model: str
    source: str
    capacity: gen.CapacityModel
    boundary: gen.CapacityModel
    capacity_shapes: tuple[str, ...]
    boundary_shapes: tuple[str, ...]
    #: shape -> why it did not enter the boundary fit
    excluded: dict = field(default_factory=dict)

    def capacity_rps(self, i: float, o: float) -> float:
        return self.capacity.rps(i, o)

    def boundary_rps(self, i: float, o: float) -> float:
        return self.boundary.rps(i, o)

    def rho_star(self, i: float, o: float) -> float:
        return self.boundary_rps(i, o) / self.capacity_rps(i, o)

    def as_dict(self) -> dict:
        return {
            "model": self.model,
            "source": self.source,
            "capacity_fit": {
                "prefill_tokens_per_s": round(self.capacity.prefill_tps, 3),
                "decode_tokens_per_s": round(self.capacity.decode_tps, 3),
                "rms_relative_error": round(self.capacity.rms_rel_error, 4),
                "shapes": list(self.capacity_shapes),
            },
            "boundary_fit": {
                "prefill_tokens_per_s": round(self.boundary.prefill_tps, 3),
                "decode_tokens_per_s": round(self.boundary.decode_tps, 3),
                "rms_relative_error": round(self.boundary.rms_rel_error, 4),
                "shapes": list(self.boundary_shapes),
            },
            "excluded": dict(self.excluded),
        }


def _mean_lengths(shape: str) -> tuple[float, float]:
    (_w, i, o), = gen.shape_components(shape)
    return gen._length_mean(i), gen._length_mean(o)


def load_surface(source_dir: Path, model: str) -> BoundarySurface:
    """Fit ``C(i, o)`` and ``r*(i, o)`` from ``<source>/<model>/{capacity,boundary}/``."""
    root = Path(source_dir) / model
    capacity_points: list[tuple[float, float, float]] = []
    boundary_points: list[tuple[float, float, float]] = []
    capacity_shapes: list[str] = []
    boundary_shapes: list[str] = []
    excluded: dict = {}
    for shape in gen.TRAINING_SHAPES:
        cap_path = root / "capacity" / f"{model}_{shape}.json"
        bnd_path = root / "boundary" / f"{model}_{shape}.json"
        if not cap_path.exists():
            excluded[shape] = "no capacity file"
            continue
        i, o = _mean_lengths(shape)
        capacity = float(json.loads(cap_path.read_text(encoding="utf-8"))["capacity_used_rps"])
        capacity_points.append((i, o, capacity))
        capacity_shapes.append(shape)
        if not bnd_path.exists():
            excluded[shape] = "no boundary file"
            continue
        search = json.loads(bnd_path.read_text(encoding="utf-8"))
        if not search.get("boundary_found") or search.get("rho_star") is None:
            excluded[shape] = "boundary not found (rho* is only a lower bound)"
            continue
        if search.get("stopped_reason"):
            excluded[shape] = f"search stopped: {search['stopped_reason']}"
            continue
        boundary_points.append((i, o, float(search["rho_star"]) * capacity))
        boundary_shapes.append(shape)
    return BoundarySurface(
        model=model,
        source=str(source_dir),
        capacity=gen.fit_capacity_model(model, capacity_points),
        boundary=gen.fit_capacity_model(model, boundary_points),
        capacity_shapes=tuple(capacity_shapes),
        boundary_shapes=tuple(boundary_shapes),
        excluded=excluded,
    )


@dataclass(frozen=True)
class StaticCell:
    model: str
    shape: str
    family: Optional[str]
    input_tokens: int
    output_tokens: int
    rho_over_rho_star: float
    rho_star: float
    rho: float
    capacity_rps: float
    offered_rps: float
    duration_s: float
    cell_id: str
    gpus: int
    held_out: bool = False

    def as_dict(self) -> dict:
        return asdict(self)

    @property
    def stem(self) -> str:
        """Schedule / output file stem; the cell directory is ``<model>_<shape>_<stem>``."""
        return f"{self.shape}_{gen.STATIC_PRIMITIVE}{gen.static_load_code(self.rho_over_rho_star)}"


def plan_static_grid(
    models: Sequence[str],
    surfaces: Mapping[str, BoundarySurface],
    *,
    gpus: Mapping[str, int],
    hold_s: float = gen.STATIC_GRID_HOLD_S,
    fractions: Sequence[float] = gen.STATIC_GRID_RHO_FRACTIONS,
) -> list[StaticCell]:
    """The static cells, per model in (shape, rho/rho*) order."""
    cells: list[StaticCell] = []
    for model in models:
        surface = surfaces[model]
        for shape, (i, o) in gen.STATIC_GRID_SHAPES.items():
            capacity = surface.capacity_rps(i, o)
            rho_star = surface.rho_star(i, o)
            for fraction in fractions:
                rho = float(fraction) * rho_star
                cells.append(StaticCell(
                    model=model,
                    shape=shape,
                    family=gen.static_grid_family(i, o),
                    input_tokens=i,
                    output_tokens=o,
                    rho_over_rho_star=float(fraction),
                    rho_star=round(rho_star, 4),
                    rho=round(rho, 4),
                    capacity_rps=round(capacity, 4),
                    offered_rps=round(rho * capacity, 4),
                    duration_s=float(hold_s),
                    cell_id=f"i{i}_o{o}_c{gen.static_load_code(fraction)}",
                    gpus=int(gpus.get(model, 1)),
                    held_out=gen.is_held_out(shape),
                ))
    return cells


def estimate(cells: Sequence[StaticCell], cooldown_s: float) -> dict:
    """Per model and total: cells, offered-load minutes, wall minutes (with cooldowns) and
    GPU-minutes (wall minutes x the GPUs one replica of the model occupies)."""
    per_model: dict = {}
    for cell in cells:
        entry = per_model.setdefault(cell.model, {
            "cells": 0, "gpus": cell.gpus, "hold_min": 0.0, "wall_min": 0.0, "gpu_min": 0.0,
            "families": {},
        })
        entry["cells"] += 1
        entry["hold_min"] += cell.duration_s / 60.0
        wall = (cell.duration_s + cooldown_s) / 60.0
        entry["wall_min"] += wall
        entry["gpu_min"] += wall * cell.gpus
        key = cell.family or "none"
        entry["families"][key] = entry["families"].get(key, 0) + 1
    for entry in per_model.values():
        for key in ("hold_min", "wall_min", "gpu_min"):
            entry[key] = round(entry[key], 2)
    total = {
        key: round(sum(e[key] for e in per_model.values()), 2)
        for key in ("cells", "hold_min", "wall_min", "gpu_min")
    }
    return {"cooldown_s": cooldown_s, "per_model": per_model, "total": total}


def format_listing(cells: Sequence[StaticCell], cooldown_s: float) -> str:
    lines = [
        f"static grid: {len(cells)} cells, {gen.STATIC_GRID_HOLD_S:.0f} s default hold, "
        f"rho/rho* in {list(gen.STATIC_GRID_RHO_FRACTIONS)}",
        f"  {'model':12} {'shape':10} {'family':14} {'rho/rho*':>8} {'rho*':>7} "
        f"{'rho':>7} {'C_s':>8} {'rps':>8} {'hold':>6}  cell_id",
    ]
    for c in cells:
        lines.append(
            f"  {c.model:12} {c.shape:10} {(c.family or '-'):14} {c.rho_over_rho_star:8.2f} "
            f"{c.rho_star:7.3f} {c.rho:7.3f} {c.capacity_rps:8.3f} {c.offered_rps:8.3f} "
            f"{c.duration_s:5.0f}s  {c.cell_id}"
        )
    est = estimate(cells, cooldown_s)
    lines.append(f"estimate (cooldown {cooldown_s:.0f} s per cell):")
    for model, e in est["per_model"].items():
        lines.append(
            f"  {model:12} {e['cells']:3d} cells  hold {e['hold_min']:6.1f} min  "
            f"wall {e['wall_min']:6.1f} min  x{e['gpus']} GPU = {e['gpu_min']:6.1f} GPU-min  "
            f"families {e['families']}"
        )
    t = est["total"]
    lines.append(
        f"  {'total':12} {t['cells']:3d} cells  hold {t['hold_min']:6.1f} min  "
        f"wall {t['wall_min']:6.1f} min  {t['gpu_min']:6.1f} GPU-min"
    )
    return "\n".join(lines)


def model_gpus(models: Sequence[str], registry_path: Optional[str] = None) -> dict:
    """GPUs one replica of each model occupies (its registry ``tp_size``)."""
    from tre_common.registry import load_registry

    registry = load_registry(registry_path)
    return {m: int(registry.model(m).tp_size) for m in models}
