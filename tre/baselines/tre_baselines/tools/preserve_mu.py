"""Profile PreServe's per-replica serving capability mu_p, mu_d, mu_t (Alg.1 lines 8-11).

PreServe keeps, over its history, the largest prefill / decode / total token rate of one
instance in any time window in which *all* requests completed without an SLO violation.
We read that from a TRE calibration capture (layout: ``tre/calibration/README.md``):

* ``<run>/<model>/cells.jsonl`` (the ladder ledger) names the driven attempts; we keep the
  ``hold`` cells (``--primitive``) that are not void, not truncated and, unless
  ``--include-contaminated``, not ``possibly_contaminated``;
* ``<run>/<model>/raw/<stem>/*.jsonl`` is each attempt's client per-request log, read with
  the re-windower's own loader (``scripts.rewindow_from_raw.load_cell_capture``, which
  attaches each request's outcome);
* tumbling ``--window-s`` windows (default 30 s, the calibration window) from the cell
  start + its warm-up to its end; a request belongs to the window its completion falls in
  (``[start, end)``, like the re-windower's token totals), an *unserved* request to the
  window it was sent in (``openloop.mark_unserved_request_windows``, like the labels);
* the SLO is the model's label definition (``tre_common.slo_labels`` via the same reader
  the baseline shell uses, ``tre_baselines.config``): ``--criterion request`` (default, the
  paper's "all requests complete without an SLO violation") requires every request sent
  in the window to be served and every served request completed in it to have
  ``ttft <= ttft_slo(L)`` and ``tpot <= tpot_slo``; ``--criterion label`` instead requires
  the window's calibration label (p95 over the window, min-n guard) to be ``healthy``;
* rates are per replica: tokens / window seconds / replicas (``--replicas``, default 1,
  the calibration runs a model at one replica; a ledger record's ``replicas`` wins).

Prints YAML in the policy's ``mu`` format on stdout, a summary on stderr. Exit 1 when a
model has no violation-free window (the policy refuses to start without its mu).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence

import yaml

LEDGER = "cells.jsonl"
DEFAULT_WINDOW_S = 30.0
CRITERIA = ("request", "label")


@dataclass
class ModelProfile:
    model: str
    windows: int = 0
    clean: int = 0
    cells: int = 0
    skipped_cells: dict = field(default_factory=dict)
    p: float = 0.0
    d: float = 0.0
    t: float = 0.0

    def as_mu(self) -> dict:
        return {"p": round(self.p, 3), "d": round(self.d, 3), "t": round(self.t, 3)}


def _read_jsonl(path: Path) -> list[dict]:
    out = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def model_dirs(paths: Iterable[str]) -> list[Path]:
    """Each path is a model dir (holds ``cells.jsonl``) or a run dir of model dirs."""
    out: list[Path] = []
    for raw in paths:
        p = Path(raw)
        if (p / LEDGER).is_file():
            out.append(p)
            continue
        found = sorted(d for d in p.iterdir() if d.is_dir() and (d / LEDGER).is_file()) if p.is_dir() else []
        if not found:
            raise SystemExit(f"preserve_mu: no {LEDGER} under {p}")
        out.extend(found)
    return out


def _served(record: Mapping) -> bool:
    from scripts.rewindow_from_raw import is_served

    return is_served(record)


def _window_unserved(records: Sequence[dict], ws: int, we: int) -> int:
    from scripts import openloop
    from tre_common import slo_labels

    row = openloop.mark_unserved_request_windows(
        [{"window_start_ms": ws, "window_end_ms": we}], records)[0]
    return slo_labels.unserved_requests(row)


def request_clean(done: Sequence[dict], unserved: int, label: Any, min_requests: int) -> Optional[bool]:
    """Paper criterion. None = not judgeable (no completions / unknown prompt length)."""
    if unserved > 0:
        return False
    served = [r for r in done if _served(r)]
    if len(served) < max(1, min_requests):
        return None
    for r in served:
        ttft = r.get("ttft_ms")
        if ttft is not None:
            L = r.get("input_tokens")
            if label.slowdown and L is None:
                return None
            if float(ttft) > label.ttft_slo_ms(L):
                return False
        tpot = r.get("tpot_ms")
        if tpot is not None and float(tpot) > float(label.tpot_p95_ms):
            return False
    return True


def label_clean(records: Sequence[dict], ws: int, we: int, label: Any,
                min_latency_samples: int) -> Optional[bool]:
    """Calibration-label criterion: the window's primary label is ``healthy``."""
    from scripts import openloop, rewindow_from_raw as rw
    from tre_common import slo_labels

    done = [r for r in records if rw.in_window(r.get("done_ts_ms"), ws, we) and rw.is_served(r)]
    mode = label.percentile_mode
    row = {
        "window_start_ms": ws,
        "window_end_ms": we,
        slo_labels.SLO_COLUMNS["ttft_p95"]: rw._guarded_p95(
            [r["ttft_ms"] for r in done if r.get("ttft_ms") is not None], mode, min_latency_samples),
        slo_labels.SLO_COLUMNS["tpot_p95"]: rw._guarded_p95(
            [r["tpot_ms"] for r in done if r.get("tpot_ms") is not None], mode, min_latency_samples),
        **rw.window_request_evidence(records, ws, we),
    }
    row = openloop.mark_unserved_request_windows([row], records)[0]
    verdict = label.window_label(row)
    if verdict == slo_labels.LABEL_UNLABELED:
        return None
    return verdict == slo_labels.LABEL_HEALTHY


def profile_model(model_dir: Path, label: Any, *, window_s: float = DEFAULT_WINDOW_S,
                  criterion: str = "request", primitive: str = "hold", replicas: int = 1,
                  include_contaminated: bool = False, min_requests: int = 1,
                  min_latency_samples: int = 10, model: Optional[str] = None) -> ModelProfile:
    from scripts import r3_grid, rewindow_from_raw as rw

    ledger = _read_jsonl(model_dir / LEDGER)
    name = model or next((str(r["model"]) for r in ledger if r.get("model")), model_dir.name)
    prof = ModelProfile(model=name)
    W = int(round(window_s * 1000))

    def skip(why: str) -> None:
        prof.skipped_cells[why] = prof.skipped_cells.get(why, 0) + 1

    for rec in ledger:
        if rec.get("model") not in (None, name):
            continue
        if str(rec.get("primitive") or rec.get("profile") or "") != primitive:
            skip("primitive")
            continue
        if rec.get("void_reasons"):
            skip("void")
            continue
        if rec.get("truncated"):
            skip("truncated")
            continue
        if rec.get("possibly_contaminated") and not include_contaminated:
            skip("contaminated")
            continue
        stem = str(rec.get("stem") or "")
        raw_dir = model_dir / "raw" / stem
        if not stem or not raw_dir.is_dir():
            skip("no_raw")
            continue
        files, _ = rw.discover_cell_files(raw_dir)
        if not files:
            skip("no_raw")
            continue
        n_rep = int(rec.get("replicas") or replicas)
        prof.cells += 1
        for path in files:
            records, _inst, guard, _unmatched = rw.load_cell_capture(path)
            if not records:
                continue
            start = rec.get("start_ms") or guard.get("start_ms")
            end = rec.get("end_ms") or guard.get("end_ms")
            if start is None:
                start = min(int(r["send_ts_ms"]) for r in records if r.get("send_ts_ms") is not None)
            if end is None:
                end = max(int(r["done_ts_ms"]) for r in records if r.get("done_ts_ms") is not None) + 1
            start = int(start) + int(round(float(rec.get("warmup_s") or 0.0) * 1000))
            windows = rw.enumerate_windows(start, int(end), W, W)
            rows, _dropped = r3_grid.censor_after(
                [{"window_start_ms": ws, "window_end_ms": we} for ws, we in windows],
                guard.get("truncated_at_ts_ms"))
            for row in rows:
                ws, we = int(row["window_start_ms"]), int(row["window_end_ms"])
                prof.windows += 1
                done = [r for r in records if rw.in_window(r.get("done_ts_ms"), ws, we)]
                if criterion == "request":
                    ok = request_clean(done, _window_unserved(records, ws, we), label, min_requests)
                else:
                    ok = label_clean(records, ws, we, label, min_latency_samples)
                if not ok:
                    continue
                prof.clean += 1
                P = sum(int(r["input_tokens"]) for r in done if r.get("input_tokens") is not None)
                D = sum(int(r["output_tokens"]) for r in done if r.get("output_tokens") is not None)
                denom = (we - ws) / 1000.0 * max(1, n_rep)
                prof.p = max(prof.p, P / denom)
                prof.d = max(prof.d, D / denom)
                prof.t = max(prof.t, (P + D) / denom)
    return prof


def label_for(model: str, registry: Any) -> Any:
    """The model's label definition, through the shell's own reader (registry SLO, the
    shared ``slo_labels`` definition). mu is exported for actuating runs, so a registry
    without the live idle-TTFT fit (c/b) is refused, as the shell does."""
    from tre_baselines.config import _slo_definition

    return _slo_definition(model, registry.model(model), registry, strict=True)


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("paths", nargs="+", help="calibration model dir(s) (with cells.jsonl) or run dir(s)")
    ap.add_argument("--registry", default=os.environ.get("TRE_REGISTRY_PATH") or "/etc/tre/registry.yaml")
    ap.add_argument("--window-s", type=float, default=DEFAULT_WINDOW_S)
    ap.add_argument("--criterion", choices=CRITERIA, default="request")
    ap.add_argument("--primitive", default="hold")
    ap.add_argument("--replicas", type=int, default=1)
    ap.add_argument("--include-contaminated", action="store_true")
    ap.add_argument("--min-requests", type=int, default=1,
                    help="request criterion: completed requests a window needs to be judged")
    ap.add_argument("--min-latency-samples", type=int, default=10,
                    help="label criterion: the re-windower's p95 guard")
    ap.add_argument("--models", default="", help="comma list of models to keep (default: all)")
    args = ap.parse_args(argv)

    from tre_common.registry import load_registry

    registry = load_registry(args.registry)
    only = {m.strip() for m in args.models.split(",") if m.strip()}
    mu: dict[str, dict] = {}
    status = 0
    for mdir in model_dirs(args.paths):
        ledger = _read_jsonl(mdir / LEDGER)
        model = next((str(r["model"]) for r in ledger if r.get("model")), mdir.name)
        if only and model not in only:
            continue
        prof = profile_model(
            mdir, label_for(model, registry), window_s=args.window_s, criterion=args.criterion,
            primitive=args.primitive, replicas=args.replicas,
            include_contaminated=args.include_contaminated, min_requests=args.min_requests,
            min_latency_samples=args.min_latency_samples, model=model,
        )
        print(f"# {model}: {prof.clean}/{prof.windows} violation-free windows in {prof.cells} "
              f"{args.primitive} cells (criterion={args.criterion}, window={args.window_s:g}s); "
              f"skipped cells {prof.skipped_cells}", file=sys.stderr)
        if prof.clean == 0:
            print(f"# {model}: no violation-free window, no mu", file=sys.stderr)
            status = 1
            continue
        mu[model] = prof.as_mu()
    sys.stdout.write(yaml.safe_dump({"mu": mu}, sort_keys=True, default_flow_style=None))
    return status


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
