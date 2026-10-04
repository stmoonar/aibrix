#!/usr/bin/env python3
"""Windows the zero-token rule drops: a read-only audit, and the conservative variant's
extra windows (stage H preparation, 2026-10-04). DRAFT - disclosure only, never a gate.

The rule
--------
``tre_calibration.dataset.calibration_window_from_row`` - the single row-to-window rule of
every fit, ``theta_verdict`` and ``dline_refit accept`` - drops a row whose
``prompt_tokens_total + generation_tokens_total <= 0`` BEFORE it is labelled. With the
gateway numerator the token totals count only completed requests, so a window in which the
engine completed nothing (queue > 0, requests timing out or cut by the 150 s route timeout)
carries 0 tokens and is dropped although it is a violation (its unserved requests make the
label ``violated``, class ``unserved``). theta is not biased by it (the dropped windows are
not near the boundary), but the accept metrics (A, B', all-violation recall) are computed
without the most stalled windows, i.e. they are optimistic. Nothing counted these drops.

Online, such a window is not CRITICAL either: TSS = 0 / queue = 0, the controller's idle
predicate (``tre_common.tss.window_is_idle``: no token) resets the EMA and the dwell runs,
and a zero TSS maps to Z = None (``compute_z_m``). So counting them as non-CRITICAL misses
is what the controller would have done, not a worst case invented here.

What this module provides
-------------------------
* :func:`classify_rows` - per CSV row, why the loader keeps or drops it (mirrors the order of
  ``calibration_window_from_row``; the signal is the frozen recompute
  ``tre_calibration.dataset.recompute_tss_rows``, the label the frozen
  ``LabelDefinition.classify`` - nothing is re-implemented);
* :func:`audit_csv` / the CLI - counts per dataset, model and set (training / H2 dynamic):
  zero-token drops split by backlog (``avg_running + avg_waiting > 0``), failure evidence
  (``model_errors`` / ``proxy_transient_errors`` / ``client_timeouts`` > 0,
  ``slo_labels.UNSERVED_COLUMNS``) and unserved-violated label; plus the low-n /
  missing-latency ``unlabeled`` rows that had a backlog;
* :func:`phantom_windows` - the dropped zero-token rows with evidence as
  ``CalibrationWindow`` s that are violating and never CRITICAL (the conservative variant
  of ``scripts.analysis.h_conservative_score``).

Excluded from every count: rows the loader filters anyway (warm-up ``in_warmup`` /
``is_warmup``, contaminated, ``filter_reason``, non-model scope), rows of a cell whose
``cell_status`` is not ``valid``, and rows of the ``holdout`` split (H2 sealed / M cells -
run1's three retained mixture cells are M cells and are not read).

    cd tre/deploy && PYTHONPATH=../common:.:../controller:../service-manager:../calibration:../replayer:../ui \\
      python3 -m scripts.analysis.h_dropped_windows --freeze-file <params_freeze.json> \\
        --dataset run1:dsqwen-7b=<run1/dsqwen-7b/dataset> ... --out <file.json>
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from tre_calibration.dataset import (
    CalibrationWindow,
    _as_float,
    _skip_row,
    recompute_tss_rows,
)
from tre_common import slo_labels

#: Row fates, in the order ``calibration_window_from_row`` decides them (tokens before the
#: signal here: a zero-token row without a queue has no signal either, and both rules drop
#: it - it is counted as zero-token, with ``backlog`` False).
FATE_FILTERED = "filtered"            # warm-up / contaminated / filter_reason / scope
FATE_INVALID_CELL = "invalid_cell"    # cell_status not valid
FATE_HOLDOUT = "holdout_not_read"     # split holdout (H2 sealed / M): skipped unread
FATE_ZERO_TOKEN = "zero_token"        # prompt + generation tokens <= 0 (the rule audited)
FATE_NO_SIGNAL = "no_signal"          # tokens > 0 but no TSS (tokens missing / A + W == 0)
FATE_UNLABELED = "unlabeled"          # label None (low_n / missing_latency)
FATE_KEPT = "kept"

#: Z the conservative variant gives a phantom window: finite (every finite filter keeps it)
#: and >= 1 (predicted healthy for BA, never CRITICAL, outside the LOW band). Online the
#: window has Z = None; this is its stand-in, not a measured value.
PHANTOM_Z = 1.0e6

TIER_FAILURE = "failure"          # label violated (unserved evidence in the window)
TIER_BACKLOG_ONLY = "backlog_only"  # queue > 0, no unserved request, label None


@dataclass(frozen=True)
class RowFate:
    index: int
    fate: str
    backlog: bool = False
    failure: bool = False
    label: Optional[slo_labels.WindowLabel] = None
    exclusion: Optional[str] = None   # low_n / missing_latency for an unlabeled row


def backlog_of(row: Mapping[str, Any]) -> bool:
    """``avg_running + avg_waiting > 0`` (missing columns read as 0)."""
    return ((_as_float(row.get("avg_running"), 0.0) or 0.0) + (_as_float(row.get("avg_waiting"), 0.0) or 0.0)) > 0.0


def failure_of(row: Mapping[str, Any]) -> bool:
    """An unserved request was sent in the window (the label's own evidence columns)."""
    return slo_labels.row_unserved(row)


def zero_tokens(row: Mapping[str, Any]) -> bool:
    """The loader's token test, verbatim (a missing total reads as 0)."""
    p = _as_float(row.get("prompt_tokens_total"), 0.0) or 0.0
    g = _as_float(row.get("generation_tokens_total"), 0.0) or 0.0
    return p + g <= 0.0


def classify_rows(rows: Sequence[Mapping[str, Any]], signals: Sequence[Optional[float]],
                  label: slo_labels.LabelDefinition, *, read_holdout: bool = False,
                  valid_cells_only: bool = True) -> list[RowFate]:
    """The fate of every row. ``signals`` parallel to ``rows`` (the frozen recompute).

    ``valid_cells_only`` False mirrors the loader, which does not read ``cell_status``: the
    training fitting CSVs hold the ``inconclusive`` boundary probes, and the fit used them.
    With True (the audit) such a row is ``invalid_cell``; its zero-token flag is kept in
    ``exclusion`` (``"zero_token"``) so the audit can still count it."""
    out: list[RowFate] = []
    for i, (row, sig) in enumerate(zip(rows, signals)):
        if _skip_row(row):
            out.append(RowFate(i, FATE_FILTERED))
            continue
        if valid_cells_only and (row.get("cell_status") or "valid").strip() != "valid":
            out.append(RowFate(i, FATE_INVALID_CELL, backlog_of(row), failure_of(row),
                               exclusion=FATE_ZERO_TOKEN if zero_tokens(row) else None))
            continue
        if not read_holdout and (row.get("split") or "").strip() == "holdout":
            out.append(RowFate(i, FATE_HOLDOUT))
            continue
        bl, fl = backlog_of(row), failure_of(row)
        if zero_tokens(row):
            lab, why = label.classify(row)
            out.append(RowFate(i, FATE_ZERO_TOKEN, bl, fl, lab, why))
            continue
        if sig is None:
            out.append(RowFate(i, FATE_NO_SIGNAL, bl, fl))
            continue
        lab, why = label.classify(row)
        out.append(RowFate(i, FATE_KEPT if lab is not None else FATE_UNLABELED, bl, fl, lab, why))
    return out


def frozen_spec_and_label(entry: Mapping[str, Any]):
    """(SignalSpec, LabelDefinition, trim) of one freeze entry (``verdict_for_holdout``)."""
    from scripts import theta_verdict as tv

    vh = entry["verdict_for_holdout"]
    return (tv.SignalSpec.from_dict(vh["signal_spec"]), slo_labels.LabelDefinition.from_dict(vh["label_def"]),
            int(vh["trim_ramp_windows"]))


def read_rows(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def row_signals(rows: Sequence[Mapping[str, Any]], spec) -> list[Optional[float]]:
    """The frozen signal of every row (EMA over every row of a cell, as the loader)."""
    if spec.tss is None:
        raise ValueError("only the TSS signal is audited")
    return recompute_tss_rows(rows, spec.tss)


def _set_of(row: Mapping[str, Any], sealed_to_h2: bool) -> str:
    from scripts import dline_refit as dl

    try:
        return dl.assign_set(row, sealed_to_h2=sealed_to_h2, sentinels=True)
    except dl.TrainingSetError as exc:
        return f"unassigned ({exc})"


def _blank() -> dict:
    return {"rows": 0, FATE_FILTERED: 0, FATE_INVALID_CELL: 0, "invalid_cell_zero_token": 0,
            "invalid_cell_zero_token_with_backlog_or_failure": 0, FATE_HOLDOUT: 0, FATE_KEPT: 0,
            FATE_NO_SIGNAL: 0,
            FATE_ZERO_TOKEN: {"total": 0, "with_backlog": 0, "with_failure_evidence": 0,
                              "backlog_or_failure": 0, "backlog_without_failure": 0,
                              "failure_without_backlog": 0, "unserved_violated": 0,
                              "label_unlabeled": 0, "label_healthy": 0},
            FATE_UNLABELED: {"total": 0, "low_n": 0, "missing_latency": 0, "with_backlog": 0,
                             "low_n_with_backlog": 0, "with_failure_evidence": 0}}


def _add(c: dict, f: RowFate) -> None:
    c["rows"] += 1
    if f.fate == FATE_ZERO_TOKEN:
        z = c[FATE_ZERO_TOKEN]
        z["total"] += 1
        z["with_backlog"] += f.backlog
        z["with_failure_evidence"] += f.failure
        z["backlog_or_failure"] += f.backlog or f.failure
        z["backlog_without_failure"] += f.backlog and not f.failure
        z["failure_without_backlog"] += f.failure and not f.backlog
        if f.label is None:
            z["label_unlabeled"] += 1
        elif f.label.slo_met:
            z["label_healthy"] += 1
        elif f.label.violation_class == "unserved":
            z["unserved_violated"] += 1
    elif f.fate == FATE_UNLABELED:
        u = c[FATE_UNLABELED]
        u["total"] += 1
        u[f.exclusion or "missing_latency"] += 1
        u["with_backlog"] += f.backlog
        u["low_n_with_backlog"] += f.backlog and f.exclusion == slo_labels.EXCLUDED_LOW_N
        u["with_failure_evidence"] += f.failure
    else:
        c[f.fate] += 1
        if f.fate == FATE_INVALID_CELL and f.exclusion == FATE_ZERO_TOKEN:
            c["invalid_cell_zero_token"] += 1
            c["invalid_cell_zero_token_with_backlog_or_failure"] += f.backlog or f.failure


def audit_csv(path: Path, entry: Mapping[str, Any], *, model: str, sealed_to_h2: bool,
              read_holdout: bool = False) -> dict:
    """Counts for one windows CSV (rows of ``model`` only), per set and in total, plus the
    cells holding zero-token windows with evidence. ``read_holdout`` True only for a CSV that
    IS the evaluated set (the T14 scorer's validation CSV, after its seal checks)."""
    spec, label, _trim = frozen_spec_and_label(entry)
    all_rows = read_rows(path)
    signals = row_signals(all_rows, spec)
    fates = classify_rows(all_rows, signals, label, read_holdout=read_holdout)
    by_set: dict[str, dict] = defaultdict(_blank)
    total = _blank()
    cells: dict[tuple, Counter] = defaultdict(Counter)
    other_models = 0
    for f in fates:
        row = all_rows[f.index]
        if row.get("model") != model:
            other_models += 1
            continue
        _add(total, f)
        if f.fate in (FATE_FILTERED, FATE_INVALID_CELL, FATE_HOLDOUT):
            continue
        _add(by_set[_set_of(row, sealed_to_h2)], f)
        if f.fate == FATE_ZERO_TOKEN and (f.backlog or f.failure):
            key = (row.get("cell_id"), row.get("attempt"), row.get("shape"), row.get("primitive"),
                   row.get("stage"), row.get("rho"))
            cells[key]["windows"] += 1
            cells[key]["with_failure_evidence"] += f.failure
    return {
        "windows_csv": str(path), "model": model, "rows_of_other_models": other_models,
        "total": total, "by_set": dict(sorted(by_set.items())),
        "cells_with_dropped_evidence_windows": [
            {"cell_id": k[0], "attempt": k[1], "shape": k[2], "primitive": k[3], "stage": k[4], "rho": k[5],
             **dict(v)} for k, v in sorted(cells.items(), key=lambda kv: (-kv[1]["windows"], str(kv[0])))],
    }


# ------------------------------------------------------------- conservative variant


def phantom_windows(rows: Sequence[Mapping[str, Any]], signals: Sequence[Optional[float]],
                    label: slo_labels.LabelDefinition, *, theta: float, trim: int,
                    read_holdout: bool = True) -> tuple[list[CalibrationWindow], list[str], dict]:
    """The dropped zero-token rows with evidence (backlog or failure), as violating windows
    the controller never flags CRITICAL: signal = ``PHANTOM_Z * theta`` (Z >= 1).

    Label: the frozen label's own verdict when it is ``violated`` (failure evidence: class
    ``unserved``, ratio ``UNSERVED_MIN_RATIO`` without latency samples) - tier ``failure``;
    a row with a backlog and no verdict (low_n, no unserved request) is counted violated with
    the label's no-sample convention (class ``unserved``, ratio ``UNSERVED_MIN_RATIO``) -
    tier ``backlog_only``. A zero-token row the label calls healthy is not added (counted).

    Ramp trim: the loader drops the first ``trim`` KEPT windows of each scenario; a phantom
    among the first ``trim`` non-filtered rows of its scenario (by window start) is not added,
    so the frozen windows stay exactly the frozen ones and no onset window is added.

    ``read_holdout`` True: the CSV is already the evaluated set (a fitting CSV or accept's
    validation CSV); every row of it is in scope."""
    fates = classify_rows(rows, signals, label, read_holdout=read_holdout, valid_cells_only=False)
    starts: dict[str, list[float]] = defaultdict(list)
    for f in fates:
        if f.fate not in (FATE_FILTERED, FATE_INVALID_CELL, FATE_HOLDOUT):
            r = rows[f.index]
            starts[_sid(r)].append(_as_float(r.get("window_start_ms"), float(f.index)) or 0.0)
    onset = {sid: set(sorted(v)[:max(0, trim)]) for sid, v in starts.items()}
    out: list[CalibrationWindow] = []
    tiers: list[str] = []
    counts = Counter()
    for f in fates:
        if f.fate != FATE_ZERO_TOKEN or not (f.backlog or f.failure):
            continue
        r = rows[f.index]
        sid = _sid(r)
        start = _as_float(r.get("window_start_ms"), float(f.index)) or 0.0
        if start in onset.get(sid, set()):
            counts["not_added_ramp_trim"] += 1
            continue
        lab = f.label
        if lab is not None and lab.slo_met:
            counts["not_added_labelled_healthy"] += 1
            continue
        if lab is not None:
            tier, ratio, cls = TIER_FAILURE, float(lab.ratio_max), lab.violation_class
        else:
            tier, ratio, cls = TIER_BACKLOG_ONLY, float(slo_labels.UNSERVED_MIN_RATIO), "unserved"
        counts[f"added_{tier}"] += 1
        out.append(CalibrationWindow(
            scenario_id=sid, scenario_family=(r.get("scenario_family") or "unknown").strip() or "unknown",
            signal=PHANTOM_Z * float(theta), slo_met=False, health_score=1.0 / (1.0 + ratio),
            window_start_ms=_as_float(r.get("window_start_ms")), latency_ratio_p95=ratio,
            latency_ratio_avg=None, queue_raw=None, violation_class=cls))
        tiers.append(tier)
    return out, tiers, {"phantom_z": PHANTOM_Z, **dict(sorted(counts.items()))}


def _sid(row: Mapping[str, Any]) -> str:
    return (row.get("scenario_id") or "unknown").strip() or "unknown"


# --------------------------------------------------------------------------- CLI


def parse_dataset_arg(text: str) -> tuple[str, str, Path]:
    """``RUN:MODEL=DIR`` -> (run, model, windows.csv)."""
    head, sep, path = text.partition("=")
    run, sep2, model = head.partition(":")
    if not sep or not sep2 or not run or not model:
        raise argparse.ArgumentTypeError(f"{text!r}: expected RUN:MODEL=DIR")
    p = Path(path)
    if p.is_dir():
        p = p / "windows.csv"
    return run, model, p


def main(argv: Optional[Sequence[str]] = None) -> int:
    from scripts import dline_refit as dl

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--freeze-file", type=Path, required=True,
                    help="the frozen parameters: each model's signal spec and label are read from it")
    ap.add_argument("--dataset", action="append", default=[], type=parse_dataset_arg, metavar="RUN:MODEL=DIR",
                    help="a TRAINING standard dataset (repeatable); never M or T14")
    ap.add_argument("--h2-run", action="append", default=["run1", "run2", "run2b"],
                    help="runs whose sealed split joins H2 (the RUN plan §E --h2-dataset runs)")
    ap.add_argument("--out", type=Path, required=True, help="JSON output (refuses to overwrite)")
    args = ap.parse_args(argv)
    if args.out.exists():
        ap.error(f"{args.out} exists (write once)")
    for run, _m, p in args.dataset:
        if any(part in ("M", "T14") for part in p.resolve().parts):
            ap.error(f"{p}: M / T14 data is never read here")
    doc = dl.verify_freeze(args.freeze_file)
    result: dict[str, Any] = {
        "what": "read-only audit of the windows the zero-token rule drops (DRAFT, disclosure only)",
        "freeze": {"path": str(args.freeze_file), "sha256": dl.sha256_file(args.freeze_file),
                   "freeze_sha256": doc["freeze_sha256"]},
        "code": dl.code_state(),
        "rule": "tre_calibration.dataset.calibration_window_from_row: prompt_tokens_total + "
                "generation_tokens_total <= 0 -> dropped before labelling",
        "excluded_from_counts": [FATE_FILTERED, FATE_INVALID_CELL, FATE_HOLDOUT],
        "note_invalid_cell": ("cell_status != valid is excluded here as asked, but the loader does not read "
                              "cell_status: the fitting CSVs hold 'inconclusive' boundary probes and the fit used "
                              "them; their zero-token rows are counted under invalid_cell_zero_token"),
        "backlog": "avg_running + avg_waiting > 0",
        "failure_evidence": f"any of {list(slo_labels.UNSERVED_COLUMNS)} > 0",
        "datasets": [],
    }
    totals: dict[str, dict] = defaultdict(_blank)
    totals_training: dict[str, dict] = defaultdict(_blank)
    for run, model, path in args.dataset:
        entry = doc["models"].get(model)
        if entry is None:
            ap.error(f"{model} is not in the freeze")
        a = audit_csv(path, entry, model=model, sealed_to_h2=run in args.h2_run)
        a.update({"run": run, "windows_csv_sha256": dl.sha256_file(path)})
        result["datasets"].append(a)
        _merge(totals[model], a["total"])
        if dl.SET_TRAINING in a["by_set"]:
            _merge(totals_training[model], a["by_set"][dl.SET_TRAINING])
    result["per_model_total"] = dict(sorted(totals.items()))
    result["per_model_training_set"] = dict(sorted(totals_training.items()))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "x", encoding="utf-8") as fh:
        json.dump(result, fh, indent=1, sort_keys=True)
        fh.write("\n")
    for model in sorted(totals):
        z, u = totals[model][FATE_ZERO_TOKEN], totals[model][FATE_UNLABELED]
        zt = totals_training[model][FATE_ZERO_TOKEN]
        print(f"[{model}] zero-token dropped {z['total']} (backlog {z['with_backlog']}, failure evidence "
              f"{z['with_failure_evidence']}, unserved-violated {z['unserved_violated']}, backlog or failure "
              f"{z['backlog_or_failure']}); training set {zt['total']} / {zt['with_backlog']} / "
              f"{zt['with_failure_evidence']} / {zt['unserved_violated']}; unlabeled {u['total']} "
              f"(low_n with backlog {u['low_n_with_backlog']}); kept {totals[model][FATE_KEPT]}")
    print(f"wrote {args.out}")
    return 0


def _merge(acc: dict, add: Mapping[str, Any]) -> None:
    for k, v in add.items():
        if isinstance(v, dict):
            _merge(acc.setdefault(k, {}), v)
        else:
            acc[k] = acc.get(k, 0) + v


if __name__ == "__main__":
    sys.exit(main())
