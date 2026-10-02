#!/usr/bin/env python3
"""Offline, READ-ONLY replay of the onset saturation rescue against recorded runs.

Design: docs/design/20261002-saturation-onset-rescue.md. For every evidence directory
(``pod_gauges.jsonl`` 5 s engine gauges, ``ctrl_ticks.jsonl`` decision snapshots,
``layout.jsonl`` fleet layout, ``load_start_epoch``) and every model it estimates when
the new rule would have fired, how far it would have scaled, and compares that with the
recorded first scale-up. It writes nothing but its report (stdout, or ``--json``).

The decision itself is the controller's code (``SaturationTracker``,
``eligibility_reason``-equivalent on the logged contexts, ``saturation_target``); the
inputs are reconstructed. Approximations (also printed with the report):

A1 "engine full" is read from the external 5 s gauge sampler (``pod_gauges.jsonl``),
   the sample nearest to each window end (within 3 s), not from the gateway's instant
   doc stamped at the window end that the controller reads;
A2 eligibility (numerator zero / receiver gate not warm) comes from the recorded tick of
   that window. Exact until the first simulated step; afterwards the recorded run had a
   different replica count, so a simulated routable change is also assumed to put the
   model under the O1 evidence gate for ``min_evidence_grids`` complete grids after it;
A3 after the first simulated step the gauges are still the recorded ones (recorded
   replica count): later steps are indicative only (they overstate saturation when the
   simulation has more replicas than the recording, understate it in the other case);
A4 a planned step lands ``--wake-s`` seconds after the tick (sleeping-replica wake) and
   capacity is always found (free GPU or a donor);
A5 one decision per window, on the first rescue tick of the window (the controller's
   rescue loop reads each window twice; the tracker counts it once);
A6 once the simulated model's TSS is warm again and the recorded tick says CRITICAL, the
   C1 hand-off is estimated with ``rescue_desired`` on the recorded Z rescaled to the
   simulated replica count (Z ~ n at fixed load, the C1 assumption).

Usage (on the control node, read-only)::

    PYTHONPATH=common:controller python3 controller/tools/saturation_onset_replay.py \\
        <evidence dir> [<evidence dir> ...] [--registry deploy/registry.yaml] [--json out.json]
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from tre_common.registry import load_registry
from tre_controller.planning.planner import rescue_desired
from tre_controller.signals.saturation import (
    REASON_NUMERATOR_ZERO,
    REASON_O1_HOLD,
    SaturationRescueConfig,
    SaturationSample,
    SaturationTracker,
    saturation_target,
)

APPROXIMATIONS = (
    "A1 engine gauges from the external 5 s sampler (nearest sample to the window end), not the gateway doc",
    "A2 eligibility from the recorded tick; after a simulated step also the O1 hold that step would cause",
    "A3 after the first simulated step the gauges are the recorded run's (different replica count): indicative",
    "A4 a step lands --wake-s after the tick; capacity always found",
    "A5 one decision per window (first rescue tick of the window)",
    "A6 after the TSS warms again: C1 hand-off estimated from the recorded Z rescaled to the simulated n",
)


def _jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def _epoch(path: Path) -> float | None:
    try:
        return float(path.read_text().strip())
    except (OSError, ValueError):
        return None


def _serve_prefix(serve_id: str) -> str:
    """``model/node/0,1`` -> the pod-name prefix ``model-node-gpu-0-1-``."""
    model, node, gpus = serve_id.split("/")
    return f"{model}-{node}-gpu-{gpus.replace(',', '-')}-"


@dataclass
class _Layout:
    rows: list[tuple[float, dict]]

    def first_reach(self, model: str, count: int, after: float) -> float | None:
        """First layout time at or after ``after`` with >= ``count`` routable bindings."""
        for stamp, models in self.rows:
            if stamp < after or model not in models:
                continue
            entry = models[model]
            hidden = set(entry.get("hidden") or ())
            if len([s for s in entry.get("awake") or () if s not in hidden]) >= count:
                return stamp
        return None

    def routable_prefixes(self, model: str, ts: float) -> tuple[list[str], int] | None:
        """Pod-name prefixes of ``model``'s awake, not hidden bindings at ``ts`` (latest
        layout row at or before it) and their count."""
        row = None
        for stamp, models in self.rows:
            if stamp > ts:
                break
            row = models
        if row is None or model not in row:
            return None
        entry = row[model]
        hidden = set(entry.get("hidden") or ())
        awake = [serve for serve in entry.get("awake") or () if serve not in hidden]
        return [_serve_prefix(serve) for serve in awake], len(awake)


def _gauge_near(gauges: list[dict], ts: float, tolerance_s: float = 3.0) -> dict | None:
    best = None
    for row in gauges:
        delta = abs(float(row["ts"]) - ts)
        if delta <= tolerance_s and (best is None or delta < best[0]):
            best = (delta, row)
    return best[1] if best else None


def _sample(row: dict | None, model: str, prefixes: list[str] | None) -> SaturationSample | None:
    if row is None:
        return None
    waiting = 0.0
    kv_values = []
    pods = 0
    for name, gauge in row.get("pods", {}).items():
        if gauge.get("model") != model:
            continue
        if prefixes is not None and not any(name.startswith(prefix) for prefix in prefixes):
            continue
        pods += 1
        waiting += float(gauge.get("num_requests_waiting") or 0.0)
        kv = gauge.get("kv_cache_usage_perc", gauge.get("gpu_cache_usage_perc"))
        if kv is not None:
            kv_values.append(float(kv))
    if pods == 0:
        return None
    return SaturationSample(
        waiting=waiting, kv=(sum(kv_values) / len(kv_values)) if kv_values else None,
        pods=pods, sample_ms=int(float(row["ts"]) * 1000),
    )


def _recorded_reason(state: dict) -> str | None:
    """The logged context's eligibility (same rule as ``eligibility_reason``)."""
    if state.get("signal_source", "zm") != "zm" or state.get("signal_unavailable_reason") == "tokens_missing":
        return None
    y_total = state.get("y_m")
    if y_total is None:
        return None
    if float(y_total) <= 1e-9:
        return REASON_NUMERATOR_ZERO
    if state.get("signal_warm") is False:
        return REASON_O1_HOLD
    return None


@dataclass
class ModelReport:
    model: str
    cap: int
    start_routable: int | None = None
    recorded_first_up: dict | None = None
    recorded_reach: dict = field(default_factory=dict)
    triggers: list[dict] = field(default_factory=list)
    steps: list[dict] = field(default_factory=list)
    pending: list[dict] = field(default_factory=list)
    #: Every window the rule looked at (eligible or not), for segment checks.
    windows: list[dict] = field(default_factory=list)


def replay_run(
    directory: Path,
    *,
    registry: Any,
    config: SaturationRescueConfig,
    wake_s: float = 3.0,
    min_evidence_grids: int = 2,
    margin_ms: int = 1000,
) -> dict:
    t0 = _epoch(directory / "load_start_epoch")
    gauges = sorted(_jsonl(directory / "pod_gauges.jsonl"), key=lambda row: float(row["ts"]))
    layout = _Layout(sorted(((float(r["ts"]), r.get("models", {})) for r in _jsonl(directory / "layout.jsonl")),
                            key=lambda item: item[0]))
    ticks = [row for row in _jsonl(directory / "ctrl_ticks.jsonl") if row.get("event") == "trs_calc_result"]
    rescue: dict[int, dict] = {}
    for row in sorted(ticks, key=lambda r: float(r.get("emit_ts") or 0.0)):
        if row.get("loop") != "rescue" or str(row.get("stale")) == "true":
            continue
        rescue.setdefault(int(float(row["ts_ms"])), row)  # first rescue tick of each window
    models = sorted({gauge["model"] for row in gauges for gauge in row.get("pods", {}).values()})
    if t0 is None:
        t0 = min(rescue) / 1000.0 if rescue else 0.0
    grid = int(config.grid_ms)
    reports = {}
    for model in models:
        try:
            cap = int(registry.model(model).scale_max_replicas)
        except Exception:  # noqa: BLE001 - a model the registry does not know
            continue
        report = ModelReport(model=model, cap=cap)
        tracker = SaturationTracker(config)
        n_sim: int | None = None
        landing: tuple[float, int] | None = None
        changes: list[int] = []
        simulated = False
        for end_ms in sorted(rescue):
            row = rescue[end_ms]
            state = (row.get("model_states") or {}).get(model)
            if not state:
                continue
            tick_ts = float(row.get("emit_ts") or end_ms / 1000.0)
            recorded_n = int(state.get("routable_pods") or 0)
            if report.start_routable is None:
                report.start_routable = recorded_n
            # Recorded scale-ups (first one, and the first tick each count is reached).
            for action in row.get("actions") or ():
                if action.get("model") == model and int(action.get("delta") or 0) > 0 and report.recorded_first_up is None:
                    report.recorded_first_up = {
                        "window_end_s": round(end_ms / 1000.0 - t0, 1),
                        "tick_s": round(tick_ts - t0, 1),
                        "delta": int(action["delta"]),
                        "reason": action.get("reason"),
                    }
            if n_sim is None:
                n_sim = recorded_n
            if landing is not None and tick_ts >= landing[0]:
                changes.append(int(landing[0] * 1000))
                n_sim = landing[1]
                landing = None
            if not simulated:
                n_view = recorded_n
            else:
                n_view = n_sim
            reason = _recorded_reason(state)
            if simulated and reason is None and changes:
                # A2: the O1 evidence gate after a simulated routable change.
                eff = int(math.ceil((changes[-1] + margin_ms) / grid) * grid)
                if end_ms > eff - grid and end_ms < eff + min_evidence_grids * grid:
                    reason = REASON_O1_HOLD
            view = layout.routable_prefixes(model, end_ms / 1000.0)
            sample = _sample(_gauge_near(gauges, end_ms / 1000.0), model, view[0] if view else None)
            verdict = tracker.observe(model, window_end_ms=end_ms, routable=n_view, reason=reason, sample=sample)
            info = {
                "window_end_s": round(end_ms / 1000.0 - t0, 1),
                "tick_s": round(tick_ts - t0, 1),
                "reason": verdict.reason,
                "waiting": None if sample is None else round(sample.waiting, 1),
                "kv": None if sample is None or sample.kv is None else round(sample.kv, 3),
                "ticks": verdict.ticks,
                "n": n_view,
            }
            report.windows.append({**info, "full": verdict.full, "fire": verdict.fire})
            if verdict.full and not verdict.fire:
                report.pending.append(info)
            if not verdict.fire:
                if (
                    simulated and reason is None and landing is None and n_sim < cap
                    and state.get("state") == "critical" and state.get("z_m") is not None
                ):
                    # A6: C1 hand-off on the warm TSS.
                    spec = registry.model(model)
                    scaling = registry.scaling()
                    z = float(state["z_m"]) * n_sim / max(1, recorded_n)
                    desired = min(cap, rescue_desired(
                        n_sim, z, float(spec.trs.tau_crit), float(scaling.rescue_max_step_ratio),
                        int(scaling.rescue_max_step_pods),
                    ))
                    if desired > n_sim:
                        report.steps.append({**info, "target": desired, "path": "c1"})
                        landing = (tick_ts + wake_s, desired)
                continue
            report.triggers.append(info)
            if n_view >= cap or landing is not None:
                continue
            target = saturation_target(n_view, config.max_step_factor, cap)
            report.steps.append({**info, "target": target, "path": "saturation"})
            tracker.note_step(model, window_end_ms=end_ms, routable=n_view)
            landing = (tick_ts + wake_s, target)
            if not simulated:
                simulated = True
                n_sim = n_view
        for target in (2, 4):
            reached = layout.first_reach(model, target, t0)
            if reached is not None and (report.start_routable or 0) < target:
                report.recorded_reach[str(target)] = round(reached - t0, 1)
        reports[model] = report
    return {"run": directory.name, "t0": t0, "models": {name: _as_dict(r) for name, r in reports.items()}}


def _as_dict(report: ModelReport) -> dict:
    reach = {}
    for step in report.steps:
        for target in (2, 4):
            if step["target"] >= target and str(target) not in reach:
                reach[str(target)] = step["tick_s"]  # decision tick (the wake lands --wake-s later)
    return {
        "cap": report.cap,
        "start_routable": report.start_routable,
        "recorded_first_up": report.recorded_first_up,
        "recorded_reach_s": report.recorded_reach,
        "new_first_trigger": report.triggers[0] if report.triggers else None,
        "new_steps": report.steps,
        "new_decided_reach_s": reach,
        "triggers": report.triggers,
        "pending": report.pending,
        "windows": report.windows,
    }


def segment_check(result: dict, model: str, start_s: float, end_s: float) -> dict:
    """Windows of ``model`` ending in [start_s, end_s] (s after load start): how many the
    rule was eligible on, how many of those were 'engine full', and the triggers."""
    rows = [w for w in (result["models"].get(model) or {}).get("windows", ())
            if start_s <= w["window_end_s"] <= end_s]
    eligible = [w for w in rows if w["reason"] is not None]
    return {
        "run": result["run"], "model": model, "segment_s": [start_s, end_s], "windows": len(rows),
        "eligible": len(eligible), "full": sum(1 for w in eligible if w["full"]),
        "fired": [w["window_end_s"] for w in rows if w["fire"]],
        "max_waiting": max((w["waiting"] or 0.0 for w in eligible), default=None),
        "max_kv": max((w["kv"] or 0.0 for w in eligible), default=None),
        "reasons": sorted({w["reason"] for w in eligible}),
    }


def _segment_triggers(result: dict, model: str, start_s: float, end_s: float) -> list[dict]:
    entry = result["models"].get(model) or {}
    return [t for t in entry.get("triggers", ()) if start_s <= t["window_end_s"] <= end_s]


def _print(result: dict) -> None:
    print(f"== {result['run']} (t0 = load start {result['t0']:.0f})")
    for model, entry in result["models"].items():
        first = entry["recorded_first_up"]
        trig = entry["new_first_trigger"]
        print(
            f"  {model:12s} cap={entry['cap']} n0={entry['start_routable']} | recorded first up: "
            + (f"window {first['window_end_s']}s tick {first['tick_s']}s +{first['delta']} ({first['reason']})" if first else "none")
            + f" reach={entry['recorded_reach_s']}"
        )
        print(
            "               new rule first trigger: "
            + (f"window {trig['window_end_s']}s tick {trig['tick_s']}s reason={trig['reason']} "
               f"waiting={trig['waiting']} kv={trig['kv']}" if trig else "none")
        )
        for step in entry["new_steps"]:
            print(f"               step[{step.get('path')}]: tick {step['tick_s']}s n={step['n']} -> {step['target']} "
                  f"(reason={step['reason']} waiting={step['waiting']} kv={step['kv']})")
        if entry["triggers"]:
            print("               all trigger windows (s): "
                  + ", ".join(f"{t['window_end_s']}[{t['reason']},w={t['waiting']},kv={t['kv']}]"
                              for t in entry["triggers"]))
        if entry["pending"]:
            first_pending = entry["pending"][0]
            print(f"               first single full window: {first_pending['window_end_s']}s "
                  f"(waiting={first_pending['waiting']} kv={first_pending['kv']} reason={first_pending['reason']})")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("runs", nargs="+", type=Path)
    parser.add_argument("--registry", default=None, help="registry.yaml (default: the shipped one)")
    parser.add_argument("--wake-s", type=float, default=3.0)
    parser.add_argument("--json", type=Path, default=None, help="also write the full report here")
    parser.add_argument("--check", action="append", default=[],
                        help="RUN_SUBSTRING:MODEL:START_S:END_S - false-trigger check of one segment")
    args = parser.parse_args(argv)
    registry = load_registry(args.registry) if args.registry else load_registry()
    config = SaturationRescueConfig.from_registry(registry, grid_ms=10_000)
    scaling = registry.scaling()
    results = []
    print("approximations:")
    for line in APPROXIMATIONS:
        print("  " + line)
    for run in args.runs:
        if not (run / "pod_gauges.jsonl").exists():
            print(f"== {run.name}: no pod_gauges.jsonl, skipped")
            continue
        result = replay_run(
            run, registry=registry, config=config, wake_s=args.wake_s,
            min_evidence_grids=int(getattr(scaling, "min_evidence_grids", 2)),
            margin_ms=int(getattr(scaling, "breakpoint_margin_ms", 1000)),
        )
        results.append(result)
        _print(result)
    for spec in args.check:
        run_part, model, start_s, end_s = spec.split(":")
        for result in results:
            if run_part in result["run"]:
                print("check", json.dumps(segment_check(result, model, float(start_s), float(end_s))))
    if args.json is not None:
        args.json.write_text(json.dumps(results, indent=1, sort_keys=True), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
