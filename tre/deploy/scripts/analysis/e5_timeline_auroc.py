#!/usr/bin/env python3
"""E5 (RQ1): align per-window SLO violations with timeline signals and score
each signal's early-warning discriminability via AUROC (Mann-Whitney U).

Reads only the canonical rerun evidence (read-only) and writes aligned per-run
CSVs plus a summary (JSON + CSV) to ``--out``. Standard library only.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional

# Scoring window span (ms) used by the replayer when emitting violation windows.
WINDOW_SPAN_MS = 5000
# Max lookback (ms) for the "nearest prior row" fallback when a window is empty.
FALLBACK_MAX_GAP_MS = 10000
# Signals scored for AUROC. Semantics are unified so that a *higher* score means
# *more likely to violate*; hence z_m (higher-is-healthier) is negated.
SIGNALS = ("z_m", "queue_len", "decode_tps", "prefill_tps")
# Signals whose aligned mean we also emit in the per-run CSV (not all are scored).
ALIGNED_SIGNAL_FIELDS = ("z_m", "queue_len", "decode_tps", "prefill_tps", "replicas_awake")
MODELS_HINT = ("dsqwen-7b", "dsllama-8b", "dsqwen-14b")


def parse_float(value: Optional[str]) -> Optional[float]:
    """Parse a CSV cell to a finite float, or ``None`` for empty/nan/inf."""
    if value is None:
        return None
    text = value.strip()
    if not text:
        return None
    try:
        number = float(text)
    except ValueError:
        return None
    if not math.isfinite(number):
        return None
    return number


def _mean(values: Iterable[float]) -> Optional[float]:
    collected = [v for v in values]
    if not collected:
        return None
    return sum(collected) / len(collected)


@dataclass
class SignalRow:
    ts_ms: int
    values: dict  # signal field -> Optional[float]


@dataclass
class ScoringWindow:
    window_end_ms: int
    violated: bool
    n_requests: int


@dataclass
class AlignedWindow:
    window_end_ms: int
    violated: bool
    n_requests: int
    signals: dict  # aligned signal field -> Optional[float]
    missing_signal: bool


def read_timeline(path: Path) -> dict:
    """Return ``{model: [SignalRow, ...]}`` sorted by ts_ms."""
    by_model: dict = {}
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            ts = parse_float(row.get("ts"))
            model = (row.get("model") or "").strip()
            if ts is None or not model:
                continue
            values = {fieldname: parse_float(row.get(fieldname)) for fieldname in ALIGNED_SIGNAL_FIELDS}
            by_model.setdefault(model, []).append(SignalRow(ts_ms=round(ts * 1000), values=values))
    for rows in by_model.values():
        rows.sort(key=lambda r: r.ts_ms)
    return by_model


def read_violations(path: Path) -> dict:
    """Return ``{model: [ScoringWindow, ...]}`` sorted by window_end_ms."""
    by_model: dict = {}
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            model = (row.get("model") or "").strip()
            end = parse_float(row.get("window_end_ms"))
            if not model or end is None:
                continue
            n_requests = parse_float(row.get("n_requests"))
            window = ScoringWindow(
                window_end_ms=round(end),
                violated=(row.get("violated") or "").strip().lower() == "true",
                n_requests=int(n_requests) if n_requests is not None else 0,
            )
            by_model.setdefault(model, []).append(window)
    for windows in by_model.values():
        windows.sort(key=lambda w: w.window_end_ms)
    return by_model


def align_window(window: ScoringWindow, rows: list) -> AlignedWindow:
    """Align one scoring window to signal rows (mean-in-window, else nearest prior)."""
    low = window.window_end_ms - WINDOW_SPAN_MS
    in_window = [r for r in rows if low < r.ts_ms <= window.window_end_ms]
    missing = False
    if in_window:
        signals = {
            fieldname: _mean(r.values[fieldname] for r in in_window if r.values[fieldname] is not None)
            for fieldname in ALIGNED_SIGNAL_FIELDS
        }
    else:
        prior = [r for r in rows if r.ts_ms <= window.window_end_ms]
        nearest = prior[-1] if prior else None  # rows are sorted by ts_ms
        if nearest is not None and window.window_end_ms - nearest.ts_ms <= FALLBACK_MAX_GAP_MS:
            signals = dict(nearest.values)
        else:
            signals = {fieldname: None for fieldname in ALIGNED_SIGNAL_FIELDS}
            missing = True
    return AlignedWindow(
        window_end_ms=window.window_end_ms,
        violated=window.violated,
        n_requests=window.n_requests,
        signals=signals,
        missing_signal=missing,
    )


def auroc(scores_labels: list) -> Optional[float]:
    """AUROC via Mann-Whitney U with average ranks for ties.

    ``scores_labels`` is a list of ``(score, label_bool)``. Convention: a higher
    score means more likely positive (violated). Returns ``None`` if either class
    is empty.
    """
    pos = [s for s, label in scores_labels if label]
    neg = [s for s, label in scores_labels if not label]
    if not pos or not neg:
        return None
    ordered = sorted(scores_labels, key=lambda sl: sl[0])
    ranks = _average_ranks([s for s, _ in ordered])
    rank_sum_pos = sum(rank for rank, (_, label) in zip(ranks, ordered) if label)
    n_pos = len(pos)
    n_neg = len(neg)
    u_pos = rank_sum_pos - n_pos * (n_pos + 1) / 2.0
    return u_pos / (n_pos * n_neg)


def _average_ranks(sorted_scores: list) -> list:
    """Fractional (average) ranks, 1-based, for an already-sorted score list."""
    ranks = [0.0] * len(sorted_scores)
    i = 0
    n = len(sorted_scores)
    while i < n:
        j = i
        while j + 1 < n and sorted_scores[j + 1] == sorted_scores[i]:
            j += 1
        average = (i + 1 + j + 1) / 2.0  # mean of 1-based ranks in [i, j]
        for k in range(i, j + 1):
            ranks[k] = average
        i = j + 1
    return ranks


def signal_score(fieldname: str, value: float) -> float:
    """Unify signals so higher == more likely to violate."""
    if fieldname == "z_m":
        return -value
    return value


def lead_labels(aligned: list, lead_windows: int) -> list:
    """Return ``[(index, label_or_None)]`` where label = any violation in the next
    ``lead_windows`` windows. Windows with no lookahead get ``None`` (skipped)."""
    labels = []
    n = len(aligned)
    for i in range(n):
        lookahead = aligned[i + 1 : i + 1 + lead_windows]
        if not lookahead:
            labels.append(None)
        else:
            labels.append(any(w.violated for w in lookahead))
    return labels


def compute_run_model(
    aligned: list, lead_windows: int
) -> dict:
    """Compute per-signal AUROC (and lead variant) for one (run, model)."""
    n_windows = len(aligned)
    n_violated = sum(1 for w in aligned if w.violated)
    n_missing = sum(1 for w in aligned if w.missing_signal)
    usable = [w for w in aligned if not w.missing_signal]

    lead_by_index = lead_labels(aligned, lead_windows) if lead_windows >= 1 else None

    results = {}
    for fieldname in SIGNALS:
        base_pairs = [
            (signal_score(fieldname, w.signals[fieldname]), w.violated)
            for w in usable
            if w.signals.get(fieldname) is not None
        ]
        entry = {"auroc": auroc(base_pairs)}
        if lead_windows >= 1:
            lead_pairs = []
            for i, w in enumerate(aligned):
                if w.missing_signal or w.signals.get(fieldname) is None:
                    continue
                label = lead_by_index[i]
                if label is None:
                    continue
                lead_pairs.append((signal_score(fieldname, w.signals[fieldname]), label))
            entry[f"auroc_lead{lead_windows}"] = auroc(lead_pairs)
        results[fieldname] = entry
    return {
        "n_windows": n_windows,
        "n_violated": n_violated,
        "n_missing_signal": n_missing,
        "signals": results,
    }


def _fmt(value: Optional[float]) -> str:
    return "" if value is None else repr(value)


def write_aligned_csv(path: Path, run_id: str, model_aligned: dict) -> None:
    header = [
        "run_id",
        "model",
        "window_end_ms",
        "violated",
        "n_requests",
        "z_m",
        "queue_len",
        "decode_tps",
        "prefill_tps",
        "replicas_awake",
        "missing_signal",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        for model in sorted(model_aligned):
            for w in model_aligned[model]:
                writer.writerow(
                    [
                        run_id,
                        model,
                        w.window_end_ms,
                        w.violated,
                        w.n_requests,
                        _fmt(w.signals.get("z_m")),
                        _fmt(w.signals.get("queue_len")),
                        _fmt(w.signals.get("decode_tps")),
                        _fmt(w.signals.get("prefill_tps")),
                        _fmt(w.signals.get("replicas_awake")),
                        w.missing_signal,
                    ]
                )


def analyze(
    evidence_root: Path,
    out_dir: Path,
    runs_glob: str = "t*_*_seed*",
    lead_windows: int = 0,
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    run_dirs = sorted(p for p in evidence_root.glob(runs_glob) if p.is_dir())
    lead_field = f"auroc_lead{lead_windows}" if lead_windows >= 1 else None

    rows: list = []  # summary rows (dicts)
    for run_dir in run_dirs:
        run_id = run_dir.name
        timeline_path = run_dir / "timeline_signals.csv"
        viol_path = run_dir / "violation_windows.csv"
        if not timeline_path.exists() or not viol_path.exists():
            continue
        timeline = read_timeline(timeline_path)
        violations = read_violations(viol_path)

        model_aligned: dict = {}
        for model, windows in violations.items():
            signal_rows = timeline.get(model, [])
            model_aligned[model] = [align_window(w, signal_rows) for w in windows]

        write_aligned_csv(out_dir / f"aligned_{run_id}.csv", run_id, model_aligned)

        for model in sorted(model_aligned):
            stats = compute_run_model(model_aligned[model], lead_windows)
            for fieldname in SIGNALS:
                entry = stats["signals"][fieldname]
                row = {
                    "run_id": run_id,
                    "model": model,
                    "signal": fieldname,
                    "n_windows": stats["n_windows"],
                    "n_violated": stats["n_violated"],
                    "n_missing_signal": stats["n_missing_signal"],
                    "auroc": entry["auroc"],
                }
                if lead_field is not None:
                    row[lead_field] = entry.get(lead_field)
                rows.append(row)

    macro_rows = _macro_rows(rows, lead_field)
    all_rows = rows + macro_rows

    _write_summary_csv(out_dir / "summary.csv", all_rows, lead_field)
    meta = {
        "evidence_root": str(evidence_root),
        "runs_glob": runs_glob,
        "runs_matched": len(run_dirs),
        "lead_windows": lead_windows,
    }
    summary = {"meta": meta, "rows": all_rows}
    with (out_dir / "summary.json").open("w") as handle:
        json.dump(summary, handle, indent=2)
    return summary


def _macro_rows(rows: list, lead_field: Optional[str]) -> list:
    macro = []
    for fieldname in SIGNALS:
        signal_rows = [r for r in rows if r["signal"] == fieldname]
        aurocs = [r["auroc"] for r in signal_rows if r["auroc"] is not None]
        row = {
            "run_id": "__macro__",
            "model": "__all__",
            "signal": fieldname,
            "n_windows": len(aurocs),  # count of contributing (run, model) AUROCs
            "n_violated": "",
            "n_missing_signal": "",
            "auroc": (sum(aurocs) / len(aurocs)) if aurocs else None,
        }
        if lead_field is not None:
            lead_vals = [r.get(lead_field) for r in signal_rows if r.get(lead_field) is not None]
            row[lead_field] = (sum(lead_vals) / len(lead_vals)) if lead_vals else None
        macro.append(row)
    return macro


def _write_summary_csv(path: Path, rows: list, lead_field: Optional[str]) -> None:
    header = [
        "run_id",
        "model",
        "signal",
        "n_windows",
        "n_violated",
        "n_missing_signal",
        "auroc",
    ]
    if lead_field is not None:
        header.append(lead_field)
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        for row in rows:
            line = [
                row["run_id"],
                row["model"],
                row["signal"],
                row["n_windows"],
                row["n_violated"],
                row["n_missing_signal"],
                _fmt(row["auroc"]),
            ]
            if lead_field is not None:
                line.append(_fmt(row.get(lead_field)))
            writer.writerow(line)


def main(argv: Optional[list] = None) -> None:
    parser = argparse.ArgumentParser(description="E5 timeline/AUROC analysis (RQ1).")
    parser.add_argument("--evidence-root", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--runs", default="t*_*_seed*", help="glob for run dirs")
    parser.add_argument("--lead-windows", type=int, default=0)
    args = parser.parse_args(argv)
    analyze(
        evidence_root=args.evidence_root,
        out_dir=args.out,
        runs_glob=args.runs,
        lead_windows=args.lead_windows,
    )


if __name__ == "__main__":
    main()
