#!/usr/bin/env python3
"""Priors a calibration run takes from an earlier run's standard dataset.

The second calibration run is pre-registered (``docs/preregistration-20260923-
calibration-run2.md``). Two of its inputs are fixed *before* it starts and are computed
from the first run, so that they cannot be tuned on the data they will be judged on:

* **regime groups** (preregistration §5.1) - the unit the leave-one-regime-out CV
  holds out. The seven training shapes are ordered by their measured prefill/decode
  time ratio and cut into three contiguous groups of at least two shapes each; of the
  (few) admissible cuts the one with the smallest within-group spread of log ratio wins.
  Per request, prefill time is TTFT and decode time is E2E - TTFT; a shape's ratio is
  the median of the per-request ratios.
* **rho priors** (preregistration §7.3, stage 0) - where each (model, shape) turns from
  healthy to violating under the *client* label, as a starting point for the boundary
  search. The first run's recorded rho* is deliberately not used: its probe verdicts
  were made with a server-side label and read "inconclusive" as "healthy".

Boundary estimation. Every labelled window of a stationary cell (the boundary probes,
primitive ``hold``, and the capacity ``steps`` cell) is one point ``(rho, violated)``,
where ``rho`` is the load *actually offered* in that window (requests whose send instant
falls in it, per second) divided by ``C_s``. The first ``hold_warmup_s`` of every hold
cell is dropped (the queue-building phase; preregistration §7.1 discards the same span).
The violation fraction is fitted as a monotone (isotonic, pool-adjacent-violators)
function of rho; the boundary is where that step function crosses 0.5, i.e. midway
between the last block below 0.5 and the first block at or above it. It counts as found
only if the blocks at or above 0.5 hold at least ``min_violating_windows`` violating
windows. A logistic fit on log(rho) is reported alongside as a smooth cross-check.

``C_s`` is the capacity the first run normalised rho by: ``capacity_used_rps`` of the
run's ``<model>/capacity/<model>_<shape>.json``, which is also the ``capacity_rps`` the
hold probes and the regenerated ramp were scheduled with (checked against
``cells.csv``). When a shape stayed healthy at every load driven, the entry says
``boundary_found: false`` and proposes an upward search range bounded by two
extrapolations from the healthy windows: where the worst client latency ratio
(p95 / SLO) reaches 1, and where the running batch reaches ``--max-num-seqs`` (past
which the waiting queue must grow and TTFT must break).

Standard library only; reads ``<run>/dataset/{requests,windows,cells}.csv`` and the
capacity artifacts, writes the two JSON files. It never writes into the run directory.

Usage::

    python3 -m scripts.analysis.calibration_priors <run_dir> --out-dir <dir>
"""
from __future__ import annotations

import argparse
import bisect
import csv
import hashlib
import json
import math
import statistics
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

#: The seven shapes the fit trains on, and the held-out mixed shape (not grouped).
TRAINING_SHAPES = ("S1", "S2", "S3", "S4", "S5", "T8", "T9")
HELD_OUT_SHAPE = "M"
#: Group names, lowest prefill/decode ratio first.
GROUP_NAMES = ("decode", "mixed", "prefill")
#: Preregistration §5.1 reference grouping (first run's family split).
REFERENCE_GROUPS = {"decode": ["S4", "S5"], "mixed": ["S1", "S2", "T9"], "prefill": ["S3", "T8"]}

#: Primitives whose load is held steady long enough to read a boundary off.
STATIONARY_PRIMITIVES = ("hold", "steps")
DEFAULT_HOLD_WARMUP_S = 60.0
DEFAULT_MIN_VIOLATING_WINDOWS = 7  # 7 sliding 5 s steps span 60 s = two disjoint 30 s windows
DEFAULT_MAX_NUM_SEQS = 256  # registry vllm_extra_args, all three models
DEFAULT_LEVEL = 0.5
#: Hard cap on a proposed upward search, in multiples of C_s.
DEFAULT_SEARCH_CAP_RHO = 6.0

#: Steps-cell level boundaries (s since cell start); the first level is the light-load one.
STEPS_FIRST_LEVEL_END_S = 90.0


# --------------------------------------------------------------------------- io


def read_csv(path: Path) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _f(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _cell_key(row: Mapping[str, Any]) -> tuple[str, str, str, str, str]:
    return (row["model"], row["shape"], row["primitive"], row["cell_id"], str(row["attempt"]))


# --------------------------------------------------------------- regime groups


def request_time_ratios(
    requests: Iterable[Mapping[str, Any]],
    *,
    cell_start_ms: Mapping[tuple, float] | None = None,
    light_load_only: bool = False,
) -> dict[str, dict[str, dict[str, float]]]:
    """``{model: {shape: {median_ratio, n}}}`` of TTFT / (E2E - TTFT) per served request.

    Only requests that completed (``outcome == ok``) in a non-void attempt and produced
    at least two tokens have a decode phase. ``light_load_only`` keeps only the first
    (rho = 0.5) level of each ``steps`` cell, where queueing does not inflate TTFT.
    """
    ratios: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in requests:
        if row.get("outcome") != "ok" or row.get("cell_status") == "void":
            continue
        if light_load_only:
            if row.get("primitive") != "steps" or cell_start_ms is None:
                continue
            start = cell_start_ms.get(_cell_key(row))
            send = _f(row.get("send_ts_ms"))
            if start is None or send is None or send - start >= STEPS_FIRST_LEVEL_END_S * 1000:
                continue
        out_tokens = _f(row.get("output_tokens"))
        ttft = _f(row.get("ttft_ms"))
        e2e = _f(row.get("e2e_ms"))
        if out_tokens is None or out_tokens <= 1 or ttft is None or e2e is None or e2e <= ttft:
            continue
        ratios[row["model"]][row["shape"]].append(ttft / (e2e - ttft))
    return {
        model: {
            shape: {"median_ratio": statistics.median(vals), "n": len(vals)}
            for shape, vals in sorted(by_shape.items())
        }
        for model, by_shape in sorted(ratios.items())
    }


def contiguous_partitions(n: int, k: int, min_size: int) -> list[tuple[int, ...]]:
    """Every way to cut ``n`` ordered items into ``k`` contiguous runs of >= ``min_size``,
    as tuples of run sizes."""
    out: list[tuple[int, ...]] = []

    def rec(remaining: int, parts: int, prefix: tuple[int, ...]) -> None:
        if parts == 1:
            if remaining >= min_size:
                out.append(prefix + (remaining,))
            return
        for size in range(min_size, remaining - min_size * (parts - 1) + 1):
            rec(remaining - size, parts - 1, prefix + (size,))

    rec(n, k, ())
    return out


def group_by_ratio(
    ratios: Mapping[str, float], *, k: int = 3, min_size: int = 2
) -> dict[str, Any]:
    """Cut shapes, ordered by ratio, into ``k`` contiguous groups (>= ``min_size`` each)
    minimising the within-group sum of squared deviations of log(ratio)."""
    if k != len(GROUP_NAMES):
        raise ValueError(f"k must be {len(GROUP_NAMES)} (one name per group)")
    order = sorted(ratios, key=lambda s: (ratios[s], s))
    logs = [math.log(ratios[s]) for s in order]
    candidates = []
    for sizes in contiguous_partitions(len(order), k, min_size):
        groups, ss, i = [], 0.0, 0
        for size in sizes:
            block = logs[i:i + size]
            mean = sum(block) / len(block)
            ss += sum((x - mean) ** 2 for x in block)
            groups.append(order[i:i + size])
            i += size
        candidates.append({"sizes": list(sizes), "groups": groups, "within_ss_log": ss})
    if not candidates:
        raise ValueError(f"{len(order)} shapes cannot form {k} groups of >= {min_size}")
    best = min(candidates, key=lambda c: (c["within_ss_log"], c["sizes"]))
    return {
        "order_low_to_high": order,
        "groups": {name: sorted(g) for name, g in zip(GROUP_NAMES, best["groups"])},
        "within_ss_log": best["within_ss_log"],
        "candidates": candidates,
    }


def _canonical(groups: Mapping[str, Sequence[str]]) -> dict[str, tuple[str, ...]]:
    return {name: tuple(sorted(members)) for name, members in groups.items()}


def regime_groups(
    requests: Sequence[Mapping[str, Any]],
    cells: Sequence[Mapping[str, Any]],
    *,
    shapes: Sequence[str] = TRAINING_SHAPES,
) -> dict[str, Any]:
    """The ``regime_groups.json`` document (see module docstring)."""
    starts = {_cell_key(c): _f(c.get("start_ms")) for c in cells if _f(c.get("start_ms")) is not None}
    all_load = request_time_ratios(requests)
    light = request_time_ratios(requests, cell_start_ms=starts, light_load_only=True)
    per_model: dict[str, Any] = {}
    for model in sorted(all_load):
        ratios = {s: all_load[model][s]["median_ratio"] for s in shapes if s in all_load[model]}
        missing = sorted(set(shapes) - set(ratios))
        if missing:
            raise ValueError(f"{model}: no served requests for shapes {missing}")
        grouping = group_by_ratio(ratios)
        light_ratios = {s: light[model][s]["median_ratio"] for s in shapes if s in light.get(model, {})}
        light_grouping = group_by_ratio(light_ratios) if len(light_ratios) == len(shapes) else None
        per_model[model] = {
            "shape_ratio": {
                s: {
                    "median_ttft_over_decode": all_load[model][s]["median_ratio"],
                    "requests": all_load[model][s]["n"],
                    "light_load_median": light.get(model, {}).get(s, {}).get("median_ratio"),
                    "light_load_requests": light.get(model, {}).get(s, {}).get("n"),
                }
                for s in shapes
            },
            "held_out_shape_ratio": (
                all_load[model].get(HELD_OUT_SHAPE, {}).get("median_ratio")
            ),
            "order_low_to_high": grouping["order_low_to_high"],
            "groups": grouping["groups"],
            "within_ss_log": grouping["within_ss_log"],
            "candidate_cuts": grouping["candidates"],
            "light_load_groups": None if light_grouping is None else light_grouping["groups"],
            "light_load_agrees": (
                None if light_grouping is None
                else _canonical(light_grouping["groups"]) == _canonical(grouping["groups"])
            ),
        }
    canon = {m: _canonical(v["groups"]) for m, v in per_model.items()}
    consistent = len(set(tuple(sorted(c.items())) for c in canon.values())) == 1
    shared = next(iter(per_model.values()))["groups"] if consistent else None
    reference_matches = (
        None if shared is None else _canonical(shared) == _canonical(REFERENCE_GROUPS)
    )
    return {
        "purpose": "LORO hold-out unit for the second calibration run (preregistration §5.1)",
        "method": (
            "per request: prefill = ttft_ms, decode = e2e_ms - ttft_ms (served, non-void, "
            "output_tokens > 1); per (model, shape): median of ttft/decode; per model: sort "
            "the 7 training shapes by that median and choose, among all cuts into 3 "
            "contiguous groups of >= 2 shapes, the cut with the smallest within-group sum "
            "of squared deviations of log(ratio). Groups are named decode < mixed < prefill "
            "by ratio. light_load_* repeats the computation on the rho=0.5 level of each "
            "steps cell only (no queueing in TTFT) as a robustness check."
        ),
        "shapes": list(shapes),
        "held_out_shape": HELD_OUT_SHAPE,
        "consistent_across_models": consistent,
        "groups": shared,
        "groups_by_model": {m: v["groups"] for m, v in per_model.items()},
        "reference_groups": REFERENCE_GROUPS,
        "matches_reference": reference_matches,
        "models": per_model,
    }


# ------------------------------------------------------------------ rho priors


def pava(values: Sequence[float], weights: Sequence[float] | None = None) -> list[tuple[int, int, float, float]]:
    """Pool-adjacent-violators for a non-decreasing fit.

    Returns blocks ``(first_index, last_index, fitted_value, weight)`` over the input
    order (the caller sorts by the covariate first).
    """
    w = list(weights) if weights is not None else [1.0] * len(values)
    blocks: list[list[float]] = []  # [first, last, value, weight]
    for i, (v, wt) in enumerate(zip(values, w)):
        blocks.append([i, i, float(v), float(wt)])
        while len(blocks) >= 2 and blocks[-2][2] > blocks[-1][2]:
            b = blocks.pop()
            a = blocks[-1]
            total = a[3] + b[3]
            a[2] = (a[2] * a[3] + b[2] * b[3]) / total
            a[1] = b[1]
            a[3] = total
    return [(int(b[0]), int(b[1]), b[2], b[3]) for b in blocks]


def isotonic_crossing(
    points: Sequence[tuple[float, int]],
    *,
    level: float = DEFAULT_LEVEL,
    min_violating_windows: int = DEFAULT_MIN_VIOLATING_WINDOWS,
) -> dict[str, Any]:
    """Where a monotone fit of ``violated`` on ``rho`` crosses ``level``.

    ``points`` are ``(rho, violated)`` with violated in {0, 1}. The fit is a step function
    of rho; the crossing is the midpoint of the jump from the last block below ``level``
    to the first block at or above it.
    """
    pts = sorted(points)
    if not pts:
        return {"found": False, "reason": "no_points"}
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    blocks: list[list[float]] = []
    for first, last, value, weight in pava(ys):
        # adjacent blocks with the same fitted value are one step of the same fit
        if blocks and blocks[-1][2] == value:
            blocks[-1][1] = last
        else:
            blocks.append([first, last, value, weight])
    summary = []
    for first, last, value, _ in blocks:
        first, last = int(first), int(last)
        seg = xs[first:last + 1]
        summary.append({
            "rho_lo": seg[0], "rho_hi": seg[-1], "rho_mean": sum(seg) / len(seg),
            "fitted": value, "windows": len(seg),
            "violating": int(sum(ys[first:last + 1])),
        })
    above = [i for i, b in enumerate(summary) if b["fitted"] >= level]
    out: dict[str, Any] = {"blocks": summary, "max_rho": xs[-1], "min_rho": xs[0]}
    below = [b for b in summary if b["fitted"] < level]
    out["max_healthy_rho"] = max((b["rho_hi"] for b in below), default=None)
    if not above:
        out.update(found=False, reason="never_crosses", crossing=None)
        return out
    first = above[0]
    violating_above = sum(summary[i]["violating"] for i in above)
    if first == 0:
        crossing = summary[0]["rho_lo"]
        reason = "violating_from_lowest_load"
    else:
        # The monotone fit is a step function; it crosses ``level`` at the jump between
        # the last block below and the first block at/above it.
        crossing = 0.5 * (summary[first - 1]["rho_hi"] + summary[first]["rho_lo"])
        reason = ""
    found = violating_above >= min_violating_windows
    out.update(
        found=found,
        reason=reason if found else "too_little_violating_evidence",
        crossing=crossing,
        healthy_side_rho=summary[first - 1]["rho_hi"] if first > 0 else None,
        violating_side_rho=summary[first]["rho_lo"],
        violating_windows_above=violating_above,
    )
    return out


def logistic_fit(points: Sequence[tuple[float, int]], *, iters: int = 50) -> dict[str, Any]:
    """Maximum-likelihood ``P(violated) = 1 / (1 + exp(-(a + b ln rho)))`` by Newton steps.

    Returns ``rho50 = exp(-a / b)`` when the slope is positive; ``None`` when the data
    cannot identify it (one class only, or separation)."""
    pts = [(math.log(x), y) for x, y in points if x > 0]
    n1 = sum(y for _, y in pts)
    if not pts or n1 == 0 or n1 == len(pts):
        return {"rho50": None, "a": None, "b": None, "reason": "single_class"}
    a = math.log(n1 / (len(pts) - n1))
    b = 0.0
    for _ in range(iters):
        ga = gb = haa = hab = hbb = 0.0
        for x, y in pts:
            z = max(-40.0, min(40.0, a + b * x))
            p = 1.0 / (1.0 + math.exp(-z))
            r = y - p
            wgt = p * (1 - p)
            ga += r
            gb += r * x
            haa += wgt
            hab += wgt * x
            hbb += wgt * x * x
        det = haa * hbb - hab * hab
        if det <= 1e-12:
            break
        da = (hbb * ga - hab * gb) / det
        db = (haa * gb - hab * ga) / det
        a += da
        b += db
        if abs(da) < 1e-9 and abs(db) < 1e-9:
            break
        if abs(b) > 200:
            return {"rho50": None, "a": a, "b": b, "reason": "separated"}
    if b <= 0:
        return {"rho50": None, "a": a, "b": b, "reason": "non_positive_slope"}
    return {"rho50": math.exp(-a / b), "a": a, "b": b, "reason": ""}


def _linear_fit(xs: Sequence[float], ys: Sequence[float]) -> tuple[float, float] | None:
    if len(xs) < 3:
        return None
    mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx <= 0:
        return None
    slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx
    return my - slope * mx, slope


def _extrapolate_to(fit: tuple[float, float] | None, target: float) -> float | None:
    if fit is None or fit[1] <= 0:
        return None
    return (target - fit[0]) / fit[1]


def capacities(run_dir: Path, cells: Sequence[Mapping[str, Any]]) -> dict[tuple[str, str], dict[str, Any]]:
    """C_s per (model, shape) from the capacity artifacts, cross-checked with cells.csv."""
    out: dict[tuple[str, str], dict[str, Any]] = {}
    for path in sorted(Path(run_dir).glob("*/capacity/*.json")):
        doc = json.loads(path.read_text())
        key = (doc["model"], doc["shape"])
        out[key] = {
            "C_s_rps": float(doc["capacity_used_rps"]),
            "source": str(path.relative_to(run_dir)) + " -> capacity_used_rps",
            "capacity_source": doc.get("capacity_source"),
            "capacity_prior_rps": doc.get("capacity_prior_rps"),
            "capacity_measured_rps": doc.get("capacity_measured_rps"),
            "steps_saturated": doc.get("saturated"),
        }
    for cell in cells:
        if cell.get("primitive") not in ("hold", "ramp"):
            continue
        key = (cell["model"], cell["shape"])
        cap = _f(cell.get("capacity_rps"))
        if key in out and cap is not None and not math.isclose(cap, out[key]["C_s_rps"], rel_tol=1e-6):
            raise ValueError(
                f"{key}: cells.csv {cell['primitive']} capacity {cap} != capacity artifact "
                f"{out[key]['C_s_rps']} - rho would not mean what the first run meant"
            )
    return out


def stationary_points(
    windows: Sequence[Mapping[str, Any]],
    requests: Sequence[Mapping[str, Any]],
    cells: Sequence[Mapping[str, Any]],
    caps: Mapping[tuple[str, str], Mapping[str, Any]],
    *,
    primitives: Sequence[str] = STATIONARY_PRIMITIVES,
    hold_warmup_s: float = DEFAULT_HOLD_WARMUP_S,
) -> dict[tuple[str, str], list[dict[str, Any]]]:
    """Per (model, shape): labelled windows of ``primitives`` with their offered rho."""
    sends: dict[tuple, list[float]] = defaultdict(list)
    for r in requests:
        if r.get("cell_status") == "void" or r.get("primitive") not in primitives:
            continue
        t = _f(r.get("send_ts_ms"))
        if t is not None:
            sends[_cell_key(r)].append(t)
    for v in sends.values():
        v.sort()
    starts = {_cell_key(c): _f(c.get("start_ms")) for c in cells}
    out: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for w in windows:
        if w.get("primitive") not in primitives or w.get("slo_label") not in ("violated", "healthy"):
            continue
        key = _cell_key(w)
        ws, we = _f(w["window_start_ms"]), _f(w["window_end_ms"])
        start = starts.get(key)
        if w["primitive"] == "hold" and start is not None and ws - start < hold_warmup_s * 1000:
            continue
        s = sends.get(key, [])
        rps = (bisect.bisect_left(s, we) - bisect.bisect_left(s, ws)) / ((we - ws) / 1000.0)
        cap = caps[(w["model"], w["shape"])]["C_s_rps"]
        ttft = _f(w.get("p95_ttft_client_ms"))
        tpot = _f(w.get("p95_tpot_client_ms"))
        out[(w["model"], w["shape"])].append({
            "rho": rps / cap,
            "rps": rps,
            "violated": 1 if w["slo_label"] == "violated" else 0,
            "primitive": w["primitive"],
            "latency_ratio": None if ttft is None or tpot is None else max(ttft / 500.0, tpot / 75.0),
            "running": _f(w.get("avg_running")),
        })
    return out


def rho_prior(
    points: Sequence[Mapping[str, Any]],
    cap: Mapping[str, Any],
    *,
    ramp_points: Sequence[Mapping[str, Any]] = (),
    min_violating_windows: int = DEFAULT_MIN_VIOLATING_WINDOWS,
    max_num_seqs: int = DEFAULT_MAX_NUM_SEQS,
    search_cap_rho: float = DEFAULT_SEARCH_CAP_RHO,
) -> dict[str, Any]:
    """The rho_priors.json entry for one (model, shape)."""
    xy = [(p["rho"], p["violated"]) for p in points]
    iso = isotonic_crossing(xy, min_violating_windows=min_violating_windows)
    logit = logistic_fit(xy)
    healthy = [p for p in points if not p["violated"]]
    lat = [(p["rho"], p["latency_ratio"]) for p in healthy if p["latency_ratio"] is not None]
    run = [(p["rho"], p["running"]) for p in healthy if p["running"] is not None]
    # Fit only the upper half of the healthy load range: that is the part the extrapolation
    # continues from, and latency is flat (not linear) at light load.
    def upper_half(pairs):
        if not pairs:
            return pairs
        mid = statistics.median(x for x, _ in pairs)
        return [(x, y) for x, y in pairs if x >= mid]
    lat_fit = _linear_fit(*zip(*upper_half(lat))) if len(lat) >= 6 else None
    run_fit = _linear_fit(*zip(*upper_half(run))) if len(run) >= 6 else None
    rho_latency_one = _extrapolate_to(lat_fit, 1.0)
    rho_batch_full = _extrapolate_to(run_fit, float(max_num_seqs))
    entry: dict[str, Any] = {
        "C_s_rps": cap["C_s_rps"],
        "C_s_source": cap["source"],
        "C_s_capacity_source": cap.get("capacity_source"),
        "windows": len(points),
        "violating_windows": sum(p["violated"] for p in points),
        "max_driven_rho": iso.get("max_rho"),
        "boundary_found": bool(iso.get("found")),
        "boundary_method": "isotonic_crossing_0.5",
        "boundary_rho": iso.get("crossing") if iso.get("found") else None,
        "boundary_rps": iso["crossing"] * cap["C_s_rps"] if iso.get("found") else None,
        "healthy_side_rho": iso.get("healthy_side_rho"),
        "violating_side_rho": iso.get("violating_side_rho"),
        "max_healthy_rho": iso.get("max_healthy_rho"),
        # The monotone fit crossed 0.5 but on too little violating evidence to call it.
        "tentative_boundary_rho": (
            iso.get("crossing") if not iso.get("found") and iso.get("crossing") is not None else None
        ),
        "isotonic_note": iso.get("reason"),
        "isotonic_blocks": iso.get("blocks"),
        "logistic_rho50": logit["rho50"],
        "logistic_extrapolated": (
            None if logit["rho50"] is None
            else not (iso.get("min_rho", 0) <= logit["rho50"] <= iso.get("max_rho", 0))
        ),
        "logistic_note": logit["reason"],
        "extrapolation": {
            "rho_latency_ratio_reaches_1": rho_latency_one,
            "rho_running_reaches_max_num_seqs": rho_batch_full,
            "max_num_seqs": max_num_seqs,
            "fit_on": "healthy stationary windows, upper half of their rho range, linear",
        },
    }
    if ramp_points:
        ramp_iso = isotonic_crossing(
            [(p["rho"], p["violated"]) for p in ramp_points],
            min_violating_windows=min_violating_windows,
        )
        entry["ramp_crossing_rho"] = ramp_iso.get("crossing") if ramp_iso.get("found") else None
        entry["ramp_note"] = "non-stationary; reported for reference, not used for the boundary"
    entry["search"] = suggest_search(entry, search_cap_rho=search_cap_rho)
    return entry


def suggest_search(entry: Mapping[str, Any], *, search_cap_rho: float = DEFAULT_SEARCH_CAP_RHO) -> dict[str, Any]:
    """Stage-0 search range, in rho (x C_s rps)."""
    if entry["boundary_found"]:
        b = entry["boundary_rho"]
        lo = min(0.85 * b, entry.get("healthy_side_rho") or 0.85 * b)
        hi = max(1.15 * b, entry.get("violating_side_rho") or 1.15 * b)
        return {
            "start_rho": round(b, 3),
            "lower_rho": round(lo, 3),
            "upper_rho": round(hi, 3),
            "rule": "found: start at the empirical boundary; bracket [min(0.85 b, healthy side), max(1.15 b, violating side)]",
        }
    max_healthy = entry.get("max_healthy_rho") or entry.get("max_driven_rho") or 1.0
    tentative = entry.get("tentative_boundary_rho")
    ext = entry["extrapolation"]
    ceilings = [x for x in (ext["rho_running_reaches_max_num_seqs"],) if x is not None and x > max_healthy]
    guesses = [x for x in (ext["rho_latency_ratio_reaches_1"],) if x is not None and x > max_healthy]
    # Upper: the batch-full ceiling bounds the flip from above (beyond it waiting grows
    # without bound). Without one, fall back to twice the highest healthy load. Never
    # below 1.5x the highest healthy load (the latency extrapolation is linear and the
    # flip is usually sharper), never above the hard cap.
    upper = min(ceilings) if ceilings else 2.0 * max_healthy
    upper = min(max(upper, 1.5 * max_healthy), search_cap_rho)
    start = round(tentative if tentative is not None else 1.1 * max_healthy, 3)
    lower = 0.85 * tentative if tentative is not None else max_healthy
    return {
        "start_rho": start,
        "lower_rho": round(lower, 3),
        "upper_rho": round(upper, 3),
        "expected_flip_rho": None if not guesses else round(min(guesses), 3),
        "rule": (
            "not found: start at the tentative crossing (monotone fit crossed 0.5 on too "
            "little evidence; lower = 0.85x it) else 10% above the highest load seen healthy "
            "(lower = that load); upper = the load at "
            "which the running batch extrapolates to max-num-seqs (hard ceiling on the "
            "flip), else 2x the highest healthy load; at least 1.5x the highest healthy "
            f"load; capped at {search_cap_rho} x C_s. expected_flip_rho = linear "
            "extrapolation of the worst p95/SLO ratio to 1 (usually an over-estimate: "
            "latency turns up sharply near saturation)"
        ),
    }


def rho_priors(
    run_dir: Path,
    windows: Sequence[Mapping[str, Any]],
    requests: Sequence[Mapping[str, Any]],
    cells: Sequence[Mapping[str, Any]],
    *,
    hold_warmup_s: float = DEFAULT_HOLD_WARMUP_S,
    min_violating_windows: int = DEFAULT_MIN_VIOLATING_WINDOWS,
    max_num_seqs: int = DEFAULT_MAX_NUM_SEQS,
    search_cap_rho: float = DEFAULT_SEARCH_CAP_RHO,
) -> dict[str, Any]:
    caps = capacities(run_dir, cells)
    points = stationary_points(windows, requests, cells, caps, hold_warmup_s=hold_warmup_s)
    ramp = stationary_points(windows, requests, cells, caps, primitives=("ramp",), hold_warmup_s=hold_warmup_s)
    models: dict[str, dict[str, Any]] = defaultdict(dict)
    for (model, shape), cap in sorted(caps.items()):
        models[model][shape] = rho_prior(
            points.get((model, shape), []), cap,
            ramp_points=ramp.get((model, shape), []),
            min_violating_windows=min_violating_windows,
            max_num_seqs=max_num_seqs, search_cap_rho=search_cap_rho,
        )
    return {
        "purpose": "stage-0 boundary-search priors for the second calibration run (preregistration §7.3)",
        "label": "client per-request (windows.csv slo_label); first-run recorded rho* NOT used",
        "rho_definition": "offered load / C_s; offered load = requests whose send_ts_ms falls in the window, per second",
        "C_s_definition": (
            "first run's capacity_used_rps (<model>/capacity/<model>_<shape>.json), the value "
            "its hold probes and regenerated ramps were scheduled with (checked == cells.csv)"
        ),
        "method": {
            "evidence": f"labelled windows of stationary cells {list(STATIONARY_PRIMITIVES)}; "
                        f"first {hold_warmup_s:g} s of each hold cell dropped (queue build-up)",
            "boundary": "isotonic (PAVA) fit of violated ~ rho; boundary = where the step "
                        "function crosses 0.5 (midway across the jump); found only with >= "
                        f"{min_violating_windows} violating windows in the blocks at/above 0.5",
            "cross_check": "logistic regression violated ~ log(rho) (logistic_rho50)",
            "not_found": "boundary_found=false, max_healthy_rho = upper edge of the highest block "
                         "below 0.5; search range from suggest_search()",
        },
        "parameters": {
            "hold_warmup_s": hold_warmup_s,
            "min_violating_windows": min_violating_windows,
            "max_num_seqs": max_num_seqs,
            "search_cap_rho": search_cap_rho,
        },
        "models": dict(models),
    }


# ------------------------------------------------------------------------- cli


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("run_dir", type=Path, help="run directory holding dataset/ and <model>/capacity/")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--hold-warmup-s", type=float, default=DEFAULT_HOLD_WARMUP_S)
    ap.add_argument("--min-violating-windows", type=int, default=DEFAULT_MIN_VIOLATING_WINDOWS)
    ap.add_argument("--max-num-seqs", type=int, default=DEFAULT_MAX_NUM_SEQS)
    ap.add_argument("--search-cap-rho", type=float, default=DEFAULT_SEARCH_CAP_RHO)
    args = ap.parse_args(argv)
    run_dir = args.run_dir.resolve()
    out_dir = args.out_dir.resolve()
    if out_dir == run_dir or run_dir in out_dir.parents:
        raise SystemExit("refusing to write inside the run directory (it is read-only evidence)")
    ds = run_dir / "dataset"
    requests = read_csv(ds / "requests.csv")
    windows = read_csv(ds / "windows.csv")
    cells = read_csv(ds / "cells.csv")
    provenance = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "generator": "scripts.analysis.calibration_priors",
        "generator_sha256": _sha256(Path(__file__)),
        "parameters": {
            "hold_warmup_s": args.hold_warmup_s,
            "min_violating_windows": args.min_violating_windows,
            "max_num_seqs": args.max_num_seqs,
            "search_cap_rho": args.search_cap_rho,
        },
        "run_dir": str(run_dir),
        "inputs_sha256": {n: _sha256(ds / n) for n in ("requests.csv", "windows.csv", "cells.csv")},
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    groups = regime_groups(requests, cells)
    priors = rho_priors(
        run_dir, windows, requests, cells,
        hold_warmup_s=args.hold_warmup_s, min_violating_windows=args.min_violating_windows,
        max_num_seqs=args.max_num_seqs, search_cap_rho=args.search_cap_rho,
    )
    for name, doc in (("regime_groups.json", groups), ("rho_priors.json", priors)):
        (out_dir / name).write_text(json.dumps({"provenance": provenance, **doc}, indent=1) + "\n")
        print(f"wrote {out_dir / name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
