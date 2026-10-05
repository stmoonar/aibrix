#!/usr/bin/env python3
"""Ranking disclosure report: AUROC and Kendall tau-b of the normalised pressure -Z on any
standard dataset, with cell-bootstrap CIs. Disclosure only - never a gate.

The metrics, their definitions and the bootstrap are ``tre_calibration.ranking`` - the
same functions ``dline_refit accept`` discloses on M (docs/design/20260930-ranking-metrics.md):

* per model: AUROC (positives = violated windows, score = -Z) and Kendall tau-b (i) of
  (-Z, severity = the label's ratio_max);
* pooled over the models: the same two over every window of every model (Z is what makes
  models comparable), cells resampled stratified by model;
* cross-model tau-b (ii): pairs of windows of different models at the same instant
  (window_end_ms exact, plus every ``--instant-bin-ms`` rounding), stratified by instant.

Parameters come from a freeze (``--freeze-file``: each model's ``verdict_for_holdout`` -
signal spec, label, trim, theta, direction - and windowing) or from ``--params-json``: a
JSON object model -> {theta, w_p, lambda_wait, tau_s, [qmin], [trim_ramp_windows],
[direction], [window_ms], and label_def (a LabelDefinition dict) or arm (primary / fixed /
k3, with ``--registry``)}.

Rows are filtered on the dataset's own columns (``--split``, ``--cell-status``, ``--role``),
split per (dataset, model) into a temporary CSV in dataset order (whole cells, so the
TSS EMA runs over each cell as in the fit) and loaded under that model's frozen spec.

    python -m scripts.ranking_report --dataset run2=/data/.../dataset \\
        --freeze-file /data/.../params_freeze.json --split holdout --out-dir OUT --name run2_holdout

writes ``OUT/NAME.json`` and ``OUT/NAME.md``.
"""
from __future__ import annotations

import argparse
import csv
import dataclasses
import json
import sys
import tempfile
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from scripts import dline_refit as dl

CELL_STATUS_ANY = "any"


def params_from_freeze(path: Path, *, verify: bool = True) -> dict[str, dict]:
    """model -> {"vh": verdict_for_holdout, "windowing": ...} of a freeze file."""
    doc = dl.verify_freeze(path) if verify else json.loads(Path(path).read_text(encoding="utf-8"))
    return {m: {"vh": e["verdict_for_holdout"], "windowing": dl.windowing_of(e)} for m, e in doc["models"].items()}


def params_from_json(path: Path, *, registry: Optional[str] = None,
                     attribution: Optional[str] = None) -> dict[str, dict]:
    """model -> {"vh", "windowing"} from explicit per-model parameters (module docstring)."""
    from tre_common import slo_labels

    from scripts import theta_verdict as tv

    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    out: dict[str, dict] = {}
    for model, q in raw.items():
        tau_s = float(q["tau_s"])
        spec = tv.build_signal_spec("tss", w_p=float(q["w_p"]), lambda_wait=float(q["lambda_wait"]),
                                    qmin=float(q.get("qmin", 1.0)),
                                    ema_tau_ms=(tau_s * 1000.0 if tau_s > 0 else None))
        if "label_def" in q:
            label = slo_labels.LabelDefinition.from_dict(q["label_def"])
        else:
            label = dl.label_for(model, q.get("arm", "primary"), registry, attribution)
        vh = {"model": model, "signal": "tss", "signal_spec": spec.as_dict(), "label_def": label.as_dict(),
              "trim_ramp_windows": int(q.get("trim_ramp_windows", dl.TRIM_RAMP_WINDOWS)),
              "fit_config": {"direction": q.get("direction", spec.direction)},
              "published": {"theta_m": float(q["theta"])}}
        out[model] = {"vh": vh, "windowing": dl.windowing(window_ms=float(q.get("window_ms", dl.WINDOW_MS)),
                                                          step_ms=float(q.get("step_ms", dl.STEP_MS)))}
    return out


def keep_row(row: Mapping[str, str], *, splits: Sequence[str], cell_status: str, roles: Sequence[str]) -> bool:
    if splits and row.get("split") not in splits:
        return False
    if cell_status != CELL_STATUS_ANY and row.get("cell_status") != cell_status:
        return False
    if roles and row.get("role") not in roles:
        return False
    return True


def split_dataset(src: dl.DatasetSource, models: Sequence[str], work: Path, **filters: Any
                  ) -> dict[str, tuple[Path, int]]:
    """The filtered rows of ``src`` per model in ``work/<run>__<model>.csv`` (dataset
    header, dataset order): model -> (path, rows)."""
    rows: dict[str, list[list[str]]] = defaultdict(list)
    with open(src.windows, newline="", encoding="utf-8") as fh:
        reader = csv.reader(fh)
        header = next(reader, None) or []
        for values in reader:
            row = dict(zip(header, values))
            if row.get("model") in models and keep_row(row, **filters):
                rows[row["model"]].append(values)
    out = {}
    for model, vals in rows.items():
        path = work / f"{src.name}__{model}.csv"
        with open(path, "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh, lineterminator="\n")
            w.writerow(header)
            w.writerows(vals)
        out[model] = (path, len(vals))
    return out


def model_records(model: str, p: Mapping[str, Any], csv_path: Path, *, run: Optional[str]) -> list:
    """Ranking records of one (dataset, model) CSV under the model's parameters; the cell
    is prefixed with ``run`` when several datasets are pooled."""
    from tre_common import slo_labels

    from scripts import theta_verdict as tv

    vh = p["vh"]
    spec = tv.SignalSpec.from_dict(vh["signal_spec"])
    label = slo_labels.LabelDefinition.from_dict(vh["label_def"])
    windows = spec.load(csv_path, label, int(vh["trim_ramp_windows"]))
    recs = dl.ranking_records(model, windows, csv_path, theta=float(vh["published"]["theta_m"]),
                              direction=vh["fit_config"]["direction"], window_ms=float(p["windowing"]["window_ms"]))
    if run is not None:
        recs = [dataclasses.replace(r, cell=f"{run}/{r.cell}") for r in recs]
    return recs


def report(sources: Sequence[dl.DatasetSource], params: Mapping[str, Mapping[str, Any]], *,
           models: Sequence[str], splits: Sequence[str] = (), cell_status: str = dl.CELL_STATUS_VALID,
           roles: Sequence[str] = (), n_resamples: int = dl.ACCEPT_RESAMPLES, seed: int = dl.SEED,
           instant_bins_ms: Sequence[float] = (), use_numpy: Optional[bool] = None) -> dict:
    from tre_calibration import ranking

    t0 = time.monotonic()
    by_model: dict[str, list] = defaultdict(list)
    inputs = []
    with tempfile.TemporaryDirectory(prefix="ranking_report_") as tmp:
        for src in sources:
            parts = split_dataset(src, models, Path(tmp), splits=splits, cell_status=cell_status, roles=roles)
            inputs.append({"name": src.name, "windows_csv": str(src.windows),
                           "windows_csv_sha256": dl.sha256_file(src.windows),
                           "rows_kept": {m: n for m, (_, n) in sorted(parts.items())}})
            for model, (path, _) in sorted(parts.items()):
                by_model[model] += model_records(model, params[model], path,
                                                 run=src.name if len(sources) > 1 else None)
    timings: dict[str, float] = {"load_s": round(time.monotonic() - t0, 3)}
    blocks: dict[str, Any] = {}
    for model in sorted(by_model):
        t = time.monotonic()
        blocks[model] = ranking.ranking_disclosure(by_model[model], n_resamples=n_resamples, seed=seed,
                                                   use_numpy=use_numpy)
        timings[model] = round(time.monotonic() - t, 3)
    pooled = None
    if by_model:
        t = time.monotonic()
        pooled = ranking.ranking_disclosure([r for m in sorted(by_model) for r in by_model[m]],
                                            n_resamples=n_resamples, seed=seed,
                                            cross_model_bins=(None, *instant_bins_ms), use_numpy=use_numpy)
        timings["pooled"] = round(time.monotonic() - t, 3)
    return {
        "what": "ranking disclosure of pressure = -Z (AUROC, Kendall tau-b); not gating",
        "gating": False,
        "definition": ranking.DEFINITIONS,
        "filters": {"split": list(splits) or "any", "cell_status": cell_status, "role": list(roles) or "any"},
        "bootstrap": {"n_resamples": n_resamples, "seed": seed},
        "inputs": inputs,
        "parameters": {m: {"theta": params[m]["vh"]["published"]["theta_m"],
                           "signal_spec": params[m]["vh"]["signal_spec"],
                           "label_def": params[m]["vh"]["label_def"],
                           "trim_ramp_windows": params[m]["vh"]["trim_ramp_windows"],
                           "direction": params[m]["vh"]["fit_config"]["direction"],
                           "windowing": params[m]["windowing"]} for m in sorted(by_model)},
        "models": blocks,
        "pooled": pooled,
        "backend": "numpy" if (ranking._have_numpy() if use_numpy is None else use_numpy) else "python",
        "timings_s": timings | {"total_s": round(time.monotonic() - t0, 3)},
    }


def markdown(doc: Mapping[str, Any], title: str) -> str:
    from tre_calibration import ranking

    blocks = dict(doc["models"])
    if doc.get("pooled"):
        blocks["pooled"] = doc["pooled"]
    lines = [f"# {title}", "",
             f"Ranking disclosure of pressure = -Z (not gating). Filters: {json.dumps(doc['filters'])}; "
             f"cell bootstrap {doc['bootstrap']['n_resamples']} resamples, seed {doc['bootstrap']['seed']} "
             "(pooled: stratified by model).", ""]
    lines += ranking.disclosure_table(blocks)
    lines += ["", "Definitions:", ""] + [f"- **{k}**: {v}" for k, v in doc["definition"].items()]
    return "\n".join(lines) + "\n"


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", action="append", required=True, metavar="[RUN=]DIR",
                    help="a standard dataset (calibration_dataset) directory; repeatable")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--freeze-file", type=Path, help="a dline_refit freeze (verified unless --no-verify)")
    src.add_argument("--params-json", type=Path, help="explicit per-model parameters (see the docstring)")
    ap.add_argument("--no-verify", action="store_true", help="read the freeze without checking its hashes")
    ap.add_argument("--registry", default=None, help="--params-json arm labels: registry of the idle TTFT fit")
    ap.add_argument("--model", action="append", default=[], help="restrict to these models (default: all)")
    ap.add_argument("--split", action="append", default=[], help="keep rows of this split (repeatable; default any)")
    ap.add_argument("--cell-status", default=dl.CELL_STATUS_VALID,
                    help=f"keep rows of this cell_status (default {dl.CELL_STATUS_VALID}; '{CELL_STATUS_ANY}' = all)")
    ap.add_argument("--role", action="append", default=[], help="keep rows of this role (repeatable; default any)")
    ap.add_argument("--resamples", type=int, default=dl.ACCEPT_RESAMPLES)
    ap.add_argument("--seed", type=int, default=dl.SEED)
    ap.add_argument("--instant-bin-ms", type=float, action="append", default=None,
                    help="cross-model tau_b also with window ends rounded to this many ms (repeatable; "
                         "default: the parameters' re-window step); the exact rule is always reported")
    ap.add_argument("--backend", choices=("auto", "numpy", "python"), default="auto",
                    help="bootstrap matrices: numpy when importable (auto), or force one")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--name", required=True, help="output stem: OUT/NAME.json and OUT/NAME.md")
    args = ap.parse_args(argv)

    try:
        sources = [dl.DatasetSource.parse(t, sealed_to_h2=False) for t in args.dataset]
    except dl.TrainingSetError as exc:
        ap.error(str(exc))
    if len({s.name for s in sources}) != len(sources):
        ap.error(f"two datasets share a run name: {[s.name for s in sources]}")
    # the datasets' one label attribution (refused when they disagree): a parameter set
    # without a label record takes it, one with a label record must state the same
    ds_attr = dl.attribution_of_inputs(datasets=[s.directory for s in sources], what="ranking_report datasets")
    try:
        params = (params_from_freeze(args.freeze_file, verify=not args.no_verify) if args.freeze_file
                  else params_from_json(args.params_json, registry=args.registry, attribution=ds_attr))
    except dl.FreezeError as exc:
        print(f"ranking_report: the freeze does not verify: {exc}", file=sys.stderr)
        return dl.EXIT_REFUSED
    wrong = {m: dl.label_attribution(p["vh"]["label_def"]) for m, p in params.items()
             if dl.label_attribution(p["vh"]["label_def"]) != ds_attr}
    if wrong:
        print(f"ranking_report: the parameters' labels {wrong} != the datasets' attribution {ds_attr!r}",
              file=sys.stderr)
        return dl.EXIT_REFUSED
    models = args.model or sorted(params)
    missing = [m for m in models if m not in params]
    if missing:
        ap.error(f"no parameters for {missing} (have {sorted(params)})")
    bins = args.instant_bin_ms
    if bins is None:
        bins = sorted({float(params[m]["windowing"]["step_ms"]) for m in models})
    use_numpy = {"auto": None, "numpy": True, "python": False}[args.backend]
    doc = report(sources, params, models=models, splits=args.split, cell_status=args.cell_status,
                 roles=args.role, n_resamples=args.resamples, seed=args.seed, instant_bins_ms=bins,
                 use_numpy=use_numpy)
    doc["command"] = ["python", "-m", "scripts.ranking_report", *(sys.argv[1:] if argv is None else argv)]
    doc["code"] = dl.code_state()
    doc["params_source"] = ({"freeze_file": str(args.freeze_file), "sha256": dl.sha256_file(args.freeze_file)}
                            if args.freeze_file else
                            {"params_json": str(args.params_json), "sha256": dl.sha256_file(args.params_json)})
    args.out_dir.mkdir(parents=True, exist_ok=True)
    jpath, mpath = args.out_dir / f"{args.name}.json", args.out_dir / f"{args.name}.md"
    jpath.write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    mpath.write_text(markdown(doc, args.name), encoding="utf-8")
    from tre_calibration import ranking

    blocks = dict(doc["models"]) | ({"pooled": doc["pooled"]} if doc["pooled"] else {})
    print("\n".join(ranking.disclosure_table(blocks)))
    print(f"wrote {jpath} and {mpath} ({doc['backend']}, {doc['timings_s']['total_s']} s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
