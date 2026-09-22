#!/usr/bin/env python3
"""Turn a calibration run directory into the standard dataset (``<run>/dataset/``).

A campaign leaves its evidence spread over one directory per driven cell - per-request
JSONL, sidecars, guard verdicts, online CSVs, boundary-search JSON - in whatever layout
the code of the day wrote. Every analysis then starts by re-assembling it, and every
re-assembly is a chance to pool a voided attempt, drop the held-out split or read the
wrong latency column. This tool does the assembly once, in one place:

* ``manifest.json`` - code commit and registry hash of the capture (when the run
  recorded them) and of this conversion, the label definition, the windowing, every
  attempt of every cell (void and inconclusive ones included) with its files, every
  boundary-search probe with the verdict the search acted on *and* the verdict under the
  current rule, and every discrepancy found while converting.
* ``windows.csv`` - all models, all non-void cells, one row per window, on the fitting
  windows (the controller's 30 s / 5 s sliding window), labelled by
  :func:`tre_common.slo_labels.window_slo_label` through
  :func:`scripts.rewindow_from_raw.label_cell` - the path the boundary search and the fit
  use.
* ``requests.csv`` - every request of every attempt (void included): the raw evidence
  from which any other label definition can be recomputed.
* ``cells.csv`` - one row per attempt: status, verdicts, outcome counts, files.
* ``DATASET.md`` - what every column means (a copy of ``tre/docs/DATASET.md``).

A campaign of the preregistered ladder design (``plan.json`` says ``"design":
"ladder"``) is read from its own ledger, ``cells.jsonl``: one line per driven attempt
naming the cell's role, rho factor, replicate, seeds, warm-up and whether the engine had
drained before it. Its cells are identified by that ledger, never by parsing directory
names, and every window and request carries ``in_warmup``.

It **only reads** the run: nothing under the run directory is written, renamed or
removed except the ``dataset/`` directory it owns. A campaign calls :func:`build_dataset`
when it ends; any run - including those made before this tool existed - converts with::

    python -m scripts.calibration_dataset /data/nfs_shared_data/xxy/calibration_20260921

A run directory is either one campaign's ``--out-dir`` (it holds ``plan.json``) or a
parent of several (one per model); the latter yields one dataset across all of them.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import subprocess
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Optional, Sequence

from tre_common import slo_labels
from tre_common.rediskeys import SCRAPE_INTERVAL_MS

from scripts import adaptive_boundary as boundary
from scripts import gen_calibration_schedules as gen
from scripts import openloop, r3_grid, rewindow_from_raw

DATASET_DIR = "dataset"
MANIFEST = "manifest.json"
WINDOW_TABLE = "windows.csv"
REQUEST_TABLE = "requests.csv"
CELL_TABLE = "cells.csv"
README = "DATASET.md"
FORMAT_REVISION = 1

#: The repository's copy of the column reference, shipped into every dataset.
README_SOURCE = Path(__file__).resolve().parents[2] / "docs" / README

STATUS_VALID = "valid"
STATUS_VOID = "void"
STATUS_INCONCLUSIVE = "inconclusive"
STATUS_MISSING = "missing"

SPLIT_TRAIN = "train"
SPLIT_HOLDOUT = "holdout"

#: Columns every row of the window and request tables starts with.
IDENTITY_COLUMNS = [
    "model", "shape", "primitive", "stage", "rho", "cell_id", "attempt", "split",
    "cell_status", "role", "rho_factor", "replicate", "possibly_contaminated",
]
WINDOW_COLUMNS = IDENTITY_COLUMNS + ["in_warmup"] + list(r3_grid.CSV_COLUMNS)
REQUEST_COLUMNS = IDENTITY_COLUMNS + [
    "in_warmup", "request_id", "scheduled_send_ts_ms", "send_ts_ms", "first_token_ts_ms", "done_ts_ms",
    "on_wire_delay_ms", "ttft_ms", "tpot_ms", "e2e_ms", "input_tokens", "output_tokens",
    "http_status", "outcome", "proxy_reason", "in_flight_at_send", "request_timeout_s",
    "target_pod",
]
CELL_COLUMNS = [
    "model", "shape", "primitive", "stage", "rho", "cell_id", "attempt", "split",
    "status", "void_reasons", "probe_verdict", "probe_verdict_recorded",
    "windows", "labeled_windows", "independent_windows", "violating_windows",
    "start_ms", "end_ms", "planned_duration_s", "capacity_rps", "offered_rps",
    "requests", "requests_ok", "requests_shed", "requests_model_error",
    "requests_proxy_transient", "requests_client_timeout", "goodput",
    "raw_path", "guard_path", "online_csv_path", "schedule_path",
    "role", "rho_factor", "replicate", "warmup_s", "arrival_seed", "prompt_key",
    "possibly_contaminated", "drained_before", "drain_waited_s", "backlog_stopped",
]

#: The ladder design's per-attempt ledger (see ``scripts.calibration_ladder``).
LEDGER = "cells.jsonl"
LADDER_DESIGN = "ladder"


# ----------------------------------------------------------------------- discovery


def campaign_dirs(root: Path) -> list[Path]:
    """The campaign output directories a run directory holds (``plan.json`` marks one)."""
    root = Path(root)
    if (root / "plan.json").exists():
        return [root]
    return sorted(p for p in root.iterdir() if p.is_dir() and (p / "plan.json").exists())


@dataclass
class Attempt:
    """One driven (or planned but never driven) attempt of one cell."""

    campaign: Path
    model: str
    shape: str
    primitive: str
    attempt: int
    stem: str
    raw_dir: Optional[Path]
    cell_id: str = ""
    stage: str = ""
    rho: Optional[float] = None
    raw_path: Optional[Path] = None
    guard: dict = field(default_factory=dict)
    recorded_probe: Optional[dict] = None
    #: The ladder design's ledger line for this attempt (or its planned cell, when it was
    #: never driven); None for a campaign of the primitives design.
    ledger: Optional[dict] = None


def parse_stem(stem: str, model: str) -> Optional[tuple[str, str, int]]:
    """``<model>_<shape>_<primitive>[_a<n>]`` or ``<model>_<shape>_<shape>_hold<c>_a<n>``
    -> (shape, primitive, attempt); None for a directory that is neither."""
    prefix = f"{model}_"
    if not stem.startswith(prefix):
        return None
    parts = stem[len(prefix):].split("_")
    attempt = 1
    if len(parts) >= 2 and parts[-1].startswith("a") and parts[-1][1:].isdigit():
        attempt = int(parts[-1][1:])
        parts = parts[:-1]
    if len(parts) == 2:
        shape, primitive = parts
        if primitive not in gen.PRIMITIVES:
            return None
        return shape, primitive, attempt
    if len(parts) == 3 and parts[0] == parts[1] and parts[2].startswith(gen.HOLD_PRIMITIVE):
        return parts[0], gen.HOLD_PRIMITIVE, attempt
    return None


def _raw_root(campaign: Path) -> Path:
    """Where the campaign's per-cell raw directories are.

    ``<out-dir>/raw`` for every run launched by run_calibration.sh; otherwise whatever
    ``--raw-dir`` the fit plan names."""
    local = campaign / "raw"
    if local.is_dir():
        return local
    plan = _read_json(campaign / "fit_plan.json")
    for entry in plan.get("rewindow", []) or []:
        command = entry.get("command") or []
        if "--raw-dir" in command:
            return Path(command[command.index("--raw-dir") + 1])
    return local


def _read_json(path: Path) -> dict:
    try:
        doc = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return doc if isinstance(doc, dict) else {}


def _cell_capture(raw_dir: Path) -> tuple[Optional[Path], Optional[str]]:
    """(the per-request capture, its cell id) inside one attempt directory."""
    for path in sorted(raw_dir.iterdir()):
        name = path.name
        if name.endswith(rewindow_from_raw.SIDECAR_JSONL_SUFFIXES):
            continue
        if name.endswith(".jsonl") or name.endswith(".jsonl" + r3_grid.VOID_RAW_SUFFIX):
            return path, name.split(".", 1)[0]
    return None, None


def _read_jsonl(path: Path) -> list[dict]:
    rows = []
    try:
        with Path(path).open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
    except OSError:
        return []
    return rows


def discover_ledger_attempts(campaign: Path, plan: dict, discrepancies: list[str]) -> list[Attempt]:
    """Attempts of a ladder-design campaign, from its ledger.

    Every directory under the raw root that no ledger line names is reported and ignored;
    every planned cell that no ledger line drove is registered as missing.
    """
    models = plan.get("models") or []
    if len(models) != 1:
        discrepancies.append(
            f"{campaign.name}: plan.json names {len(models)} model(s); expected one per campaign"
        )
    raw_root = _raw_root(campaign)
    records = _read_jsonl(campaign / LEDGER)
    if not records:
        discrepancies.append(f"{campaign.name}: ladder design but no {LEDGER}")
    attempts: list[Attempt] = []
    stems: set[str] = set()
    for record in records:
        stem = str(record.get("stem", ""))
        stems.add(stem)
        raw_dir = raw_root / stem
        raw_path, captured_id = (_cell_capture(raw_dir) if raw_dir.is_dir() else (None, None))
        cell_id = str(record.get("cell_id", ""))
        if captured_id and captured_id != cell_id:
            discrepancies.append(
                f"{stem}: the ledger says {cell_id}, the capture is {captured_id}")
        attempt = Attempt(
            campaign=campaign, model=str(record.get("model", "")),
            shape=str(record.get("shape", "")), primitive=str(record.get("primitive", "")),
            attempt=int(record.get("attempt", 1)), stem=stem,
            raw_dir=raw_dir if raw_dir.is_dir() else None, cell_id=cell_id,
            stage=str(record.get("stage") or ""),
            rho=None if record.get("rho") is None else float(record["rho"]),
            raw_path=raw_path, ledger=record,
        )
        attempt.guard = _read_json(raw_dir / f"{cell_id}.guard.json") if raw_dir.is_dir() else {}
        if record.get("verdict") or record.get("void_reasons"):
            attempt.recorded_probe = {
                "verdict": (boundary.VERDICT_VOID if record.get("void_reasons")
                            else record.get("verdict")),
                "rho": record.get("rho"), "stage": record.get("stage"),
                "attempt": attempt.attempt, "cell_id": cell_id,
                "duration_s": record.get("duration_s"),
            }
        if raw_path is None:
            discrepancies.append(f"{stem}: no per-request capture")
        elif not attempt.guard:
            discrepancies.append(f"{stem}: no guard artifact")
        attempts.append(attempt)
    if raw_root.is_dir():
        for raw_dir in sorted(p for p in raw_root.iterdir() if p.is_dir()):
            if raw_dir.name not in stems:
                discrepancies.append(f"{raw_dir}: not in {LEDGER}; ignored")
    driven = {str(r.get("cell_id")) for r in records}
    for model, cells in (plan.get("static_cells") or {}).items():
        for cell in cells:
            if str(cell.get("cell_id")) in driven:
                continue
            attempts.append(Attempt(
                campaign=campaign, model=model, shape=str(cell.get("shape", "")),
                primitive=str(cell.get("primitive", "")), attempt=1, stem="",
                raw_dir=None, cell_id=str(cell.get("cell_id", "")),
                stage=str(cell.get("stage") or ""),
                rho=None if cell.get("rho") is None else float(cell["rho"]), ledger=cell,
            ))
            discrepancies.append(
                f"{model}/{cell.get('shape')}/{cell.get('role')} {cell.get('cell_id')}: "
                "planned, never driven")
    return attempts


def discover_attempts(campaign: Path, discrepancies: list[str]) -> list[Attempt]:
    plan = _read_json(campaign / "plan.json")
    if plan.get("design") == LADDER_DESIGN:
        return discover_ledger_attempts(campaign, plan, discrepancies)
    models = plan.get("models") or []
    if len(models) != 1:
        discrepancies.append(
            f"{campaign.name}: plan.json names {len(models)} model(s); expected one per campaign"
        )
    model = models[0] if models else campaign.name
    raw_root = _raw_root(campaign)
    attempts: list[Attempt] = []
    if raw_root.is_dir():
        for raw_dir in sorted(p for p in raw_root.iterdir() if p.is_dir()):
            parsed = parse_stem(raw_dir.name, model)
            if parsed is None:
                discrepancies.append(f"{raw_dir}: not a cell directory of {model}; ignored")
                continue
            shape, primitive, attempt_no = parsed
            raw_path, cell_id = _cell_capture(raw_dir)
            attempt = Attempt(
                campaign=campaign, model=model, shape=shape, primitive=primitive,
                attempt=attempt_no, stem=raw_dir.name, raw_dir=raw_dir,
                cell_id=cell_id or "", raw_path=raw_path,
            )
            if cell_id:
                attempt.guard = _read_json(raw_dir / f"{cell_id}.guard.json")
            if raw_path is None:
                discrepancies.append(f"{raw_dir}: no per-request capture")
            elif not attempt.guard:
                discrepancies.append(f"{raw_dir}: no guard artifact")
            attempts.append(attempt)
    else:
        discrepancies.append(f"{campaign.name}: raw directory {raw_root} not found")

    # Probe metadata comes from the boundary search that asked for the probe.
    by_key = {(a.shape, a.cell_id, a.attempt): a for a in attempts if a.primitive == gen.HOLD_PRIMITIVE}
    for path in sorted((campaign / "boundary").glob("*.json")):
        search = _read_json(path)
        shape = search.get("shape", "")
        for probe in search.get("probes", []) or []:
            key = (shape, probe.get("cell_id", ""), int(probe.get("attempt", 1)))
            attempt = by_key.get(key)
            if attempt is None:
                # Asked for, recorded, but no capture on disk: register it anyway.
                attempt = Attempt(
                    campaign=campaign, model=model, shape=shape,
                    primitive=gen.HOLD_PRIMITIVE, attempt=key[2], stem="", raw_dir=None,
                    cell_id=key[1],
                )
                attempts.append(attempt)
                discrepancies.append(
                    f"{model}/{shape}: boundary probe {key[1]} attempt {key[2]} has no raw capture"
                )
            attempt.stage = str(probe.get("stage", ""))
            attempt.rho = float(probe["rho"]) if probe.get("rho") is not None else None
            attempt.recorded_probe = probe
    for attempt in attempts:
        if attempt.primitive == gen.HOLD_PRIMITIVE and attempt.recorded_probe is None:
            discrepancies.append(
                f"{attempt.stem}: probe capture not referenced by any boundary search"
            )

    # Scheduled cells that were planned and never driven.
    driven = {(a.shape, a.primitive) for a in attempts}
    for cell in plan.get("cells", []) or []:
        if (cell.get("shape"), cell.get("primitive")) not in driven:
            attempts.append(Attempt(
                campaign=campaign, model=model, shape=cell.get("shape", ""),
                primitive=cell.get("primitive", ""), attempt=1, stem="", raw_dir=None,
                cell_id=cell.get("cell_id", ""),
            ))
            discrepancies.append(
                f"{model}/{cell.get('shape')}/{cell.get('primitive')}: planned, never driven"
            )
    return attempts


# ------------------------------------------------------------------------ helpers


def _rel(path: Optional[Path | str], root: Path) -> Optional[str]:
    if path is None or path == "":
        return None
    path = Path(path)
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path)


def _sha256(path: Path) -> Optional[str]:
    try:
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()
    except OSError:
        return None


def _git(*argv: str) -> Optional[str]:
    try:
        proc = subprocess.run(
            ["git", "-C", str(Path(__file__).resolve().parents[2]), *argv],
            capture_output=True, text=True, timeout=10, check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return proc.stdout.strip() if proc.returncode == 0 else None


def _split(shape: str) -> str:
    return SPLIT_HOLDOUT if gen.is_held_out(shape) else SPLIT_TRAIN


def _recorded_verdict(probe: Optional[dict]) -> Optional[str]:
    """The verdict the search acted on, in either artifact format."""
    if not probe:
        return None
    if probe.get("verdict"):
        return str(probe["verdict"])
    if not probe.get("valid", True):
        return boundary.VERDICT_VOID
    return boundary.VERDICT_VIOLATED if probe.get("violated") else boundary.VERDICT_HEALTHY


def _write_csv(path: Path, columns: Sequence[str], rows: Iterable[dict]) -> int:
    count = 0
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(columns), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
            count += 1
    return count


# ------------------------------------------------------------------------- build


@dataclass(frozen=True)
class Settings:
    window_ms: int
    step_ms: int
    ttft_slo_ms: float
    tpot_slo_ms: float
    min_latency_samples: int
    percentile_mode: str
    registry_path: Path

    @property
    def latency_slo_ms(self) -> dict:
        return slo_labels.slo_targets(ttft_slo_ms=self.ttft_slo_ms, tpot_slo_ms=self.tpot_slo_ms)


def _settings_for(campaigns: Sequence[Path], overrides: dict) -> tuple[Settings, list[dict]]:
    """Windowing and SLO the run declared, overridable; plus each campaign's provenance."""
    provenances = []
    window_ms = step_ms = None
    ttft = tpot = None
    registry = None
    for campaign in campaigns:
        plan = _read_json(campaign / "plan.json")
        fit = _read_json(campaign / "fit_plan.json")
        prov = plan.get("provenance") or {}
        entry = {
            "campaign": campaign.name,
            "code": prov.get("code"),
            "registry_path": prov.get("registry_path"),
            "registry_sha256": prov.get("registry_sha256"),
            "status": _read_json(campaign / "campaign_status.json") or None,
        }
        if plan.get("design") == LADDER_DESIGN:
            manifest_path = campaign / str(plan.get("run_manifest") or "run_manifest.json")
            actual = _sha256(manifest_path)
            recorded = plan.get("run_manifest_sha256")
            entry["design"] = LADDER_DESIGN
            entry["run_manifest"] = {
                "path": manifest_path.name,
                "sha256": actual,
                "sha256_at_start": recorded,
                "unchanged_since_start": None if recorded is None else actual == recorded,
            }
            entry["design_result"] = _read_json(campaign / "design_result.json") or None
        provenances.append(entry)
        window_ms = window_ms or prov.get("window_ms") or fit.get("window_ms")
        step_ms = step_ms or prov.get("step_ms") or fit.get("step_ms")
        slo = (prov.get("label") or {}).get("slo_ms") or {}
        ttft = ttft or slo.get(slo_labels.P95_TTFT_CLIENT)
        tpot = tpot or slo.get(slo_labels.P95_TPOT_CLIENT)
        if prov.get("registry_path") and Path(prov["registry_path"]).exists():
            registry = registry or Path(prov["registry_path"])
    if ttft is None or tpot is None:
        # Older runs recorded the pinned SLO on every cell's guard, not in the plan.
        for campaign in campaigns:
            for guard_path in sorted(_raw_root(campaign).glob("*/*.guard.json"))[:1]:
                guard = _read_json(guard_path)
                ttft = ttft or guard.get("ttft_slo_ms")
                tpot = tpot or guard.get("tpot_slo_ms")
    settings = Settings(
        window_ms=int(overrides.get("window_ms") or window_ms or 30000),
        step_ms=int(overrides.get("step_ms") or step_ms or 5000),
        ttft_slo_ms=float(overrides.get("ttft_slo_ms") or ttft or 500.0),
        tpot_slo_ms=float(overrides.get("tpot_slo_ms") or tpot or 75.0),
        min_latency_samples=int(overrides.get("min_latency_samples") or 10),
        percentile_mode=str(overrides.get("percentile_mode") or "bucket_upper"),
        registry_path=Path(
            overrides.get("registry")
            or registry
            or Path(__file__).resolve().parents[1] / "registry.yaml"
        ),
    )
    return settings, provenances


def _cell_metadata(attempt: Attempt, plan_cells: dict) -> dict:
    """Capacity, offered load and planned duration of one attempt, from what the run wrote."""
    campaign = attempt.campaign
    out = {"capacity_rps": None, "offered_rps": None, "planned_duration_s": None,
           "schedule_path": None}
    if attempt.ledger is not None:
        ledger = attempt.ledger
        out.update(
            capacity_rps=ledger.get("capacity_rps"),
            offered_rps=ledger.get("offered_rps"),
            planned_duration_s=ledger.get("duration_s"),
            schedule_path=ledger.get("schedule_path"),
        )
        return out
    if attempt.primitive == gen.HOLD_PRIMITIVE:
        code = attempt.cell_id.rsplit("_c", 1)[-1] if attempt.cell_id else ""
        stem = f"{attempt.shape}_{gen.HOLD_PRIMITIVE}{code}"
        names = [f"{stem}_a{attempt.attempt}", stem] if attempt.attempt > 1 else [stem]
        for name in names:
            meta_path = campaign / "schedules" / attempt.model / f"{name}.meta.json"
            if meta_path.exists():
                meta = _read_json(meta_path)
                out.update(
                    capacity_rps=meta.get("capacity_rps"),
                    offered_rps=meta.get("offered_rps"),
                    planned_duration_s=meta.get("duration_s"),
                    schedule_path=meta_path.with_name(f"{name}.json"),
                )
                break
        if out["planned_duration_s"] is None and attempt.recorded_probe:
            out["planned_duration_s"] = attempt.recorded_probe.get("duration_s")
    else:
        cell = plan_cells.get((attempt.shape, attempt.primitive)) or {}
        out["capacity_rps"] = cell.get("capacity_rps")
        out["planned_duration_s"] = cell.get("duration_s")
        if attempt.primitive == "ramp":
            regenerated = campaign / "schedules" / attempt.model / f"{attempt.shape}_ramp.json"
            if regenerated.exists():
                out["schedule_path"] = regenerated
                meta = _read_json(regenerated.with_name(f"{attempt.shape}_ramp.meta.json"))
                measured = meta.get("capacity") or {}
                out["capacity_rps"] = measured.get("capacity_used_rps", out["capacity_rps"])
    if attempt.guard.get("schedule") and out["schedule_path"] is None:
        out["schedule_path"] = attempt.guard.get("schedule")
    return out


def build_dataset(
    run_dir: Path,
    *,
    out_dir: Optional[Path] = None,
    overrides: Optional[dict] = None,
) -> Path:
    """Convert ``run_dir`` into the standard dataset; returns the dataset directory.

    Reads only. The dataset is assembled in a sibling temporary directory and moved into
    place at the end, so a failed conversion never leaves a half-written dataset where a
    complete one used to be.
    """
    run_dir = Path(run_dir).resolve()
    out_dir = Path(out_dir) if out_dir else run_dir / DATASET_DIR
    if out_dir.exists() and not (out_dir / MANIFEST).exists():
        raise SystemExit(f"{out_dir} exists and is not a dataset this tool wrote; refusing to replace it")
    campaigns = campaign_dirs(run_dir)
    if not campaigns:
        raise SystemExit(f"{run_dir}: no campaign directory (plan.json) found")

    from tre_common.registry import load_registry

    discrepancies: list[str] = []
    settings, provenances = _settings_for(campaigns, overrides or {})
    registry = load_registry(str(settings.registry_path))

    staging = out_dir.with_name(f".{out_dir.name}.building")
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)

    cell_rows: list[dict] = []
    manifest_cells: list[dict] = []
    window_count = request_count = 0
    window_fh = (staging / WINDOW_TABLE).open("w", newline="", encoding="utf-8")
    request_fh = (staging / REQUEST_TABLE).open("w", newline="", encoding="utf-8")
    try:
        window_writer = csv.DictWriter(window_fh, fieldnames=WINDOW_COLUMNS, extrasaction="ignore")
        request_writer = csv.DictWriter(request_fh, fieldnames=REQUEST_COLUMNS, extrasaction="ignore")
        window_writer.writeheader()
        request_writer.writeheader()
        boundary_docs: list[dict] = []
        for campaign in campaigns:
            plan = _read_json(campaign / "plan.json")
            plan_cells = {(c.get("shape"), c.get("primitive")): c for c in plan.get("cells", []) or []}
            attempts = discover_attempts(campaign, discrepancies)
            attempts.sort(key=lambda a: (a.model, a.shape, a.primitive, a.cell_id, a.attempt))
            for attempt in attempts:
                converted = _convert_attempt(
                    attempt, run_dir, settings, registry, plan_cells, discrepancies,
                )
                cell_rows.append(converted["cell"])
                manifest_cells.append(converted["manifest"])
                for row in converted["windows"]:
                    window_writer.writerow(row)
                    window_count += 1
                for row in converted["requests"]:
                    request_writer.writerow(row)
                    request_count += 1
            for path in sorted((campaign / "boundary").glob("*.json")):
                search = _read_json(path)
                boundary_docs.append({
                    "file": _rel(path, run_dir),
                    "model": search.get("model"),
                    "shape": search.get("shape"),
                    "recorded": {k: v for k, v in search.items() if k != "probes"},
                    "probes": [
                        _probe_summary(probe, manifest_cells, search)
                        for probe in search.get("probes", []) or []
                    ],
                })
    finally:
        window_fh.close()
        request_fh.close()

    legacy_online = sum(
        1 for c in manifest_cells if c.get("online_csv_parity") == PARITY_NOT_COMPARABLE
    )
    if legacy_online:
        discrepancies.append(
            f"{legacy_online} online CSV(s) predate the shared label (server-side p95 on "
            "30 s tumbling windows); they are kept as files but every window in this "
            "dataset is recomputed from raw"
        )
    cells_written = _write_csv(staging / CELL_TABLE, CELL_COLUMNS, cell_rows)
    if README_SOURCE.exists():
        shutil.copyfile(README_SOURCE, staging / README)
    else:
        discrepancies.append(f"{README_SOURCE} missing; DATASET.md not copied")

    manifest = {
        "format_revision": FORMAT_REVISION,
        "built_at_utc": datetime.now(timezone.utc).isoformat(),
        "builder": {
            "module": "scripts.calibration_dataset",
            "code_commit": _git("rev-parse", "HEAD"),
            "code_dirty": (lambda s: None if s is None else bool(s))(
                _git("status", "--porcelain", "--untracked-files=no")
            ),
        },
        "run_root": str(run_dir),
        "campaigns": provenances,
        "registry_used_for_signal_columns": {
            "path": str(settings.registry_path),
            "sha256": _sha256(settings.registry_path),
            "note": (
                "trs / queue_control depend on the registry's trs parameters and are "
                "recomputed with this file; the SLO label does not depend on it"
            ),
        },
        "label": slo_labels.label_definition(
            settings.latency_slo_ms, min_latency_samples=settings.min_latency_samples
        ),
        "windowing": {
            "window_ms": settings.window_ms,
            "step_ms": settings.step_ms,
            "kind": "sliding" if settings.step_ms < settings.window_ms else "tumbling",
            "bounds": "each attempt's own [start_ms, end_ms] from its guard artifact",
            "queue_source": f"sidecar samples on the live grid ({SCRAPE_INTERVAL_MS} ms)",
            "percentile_mode": settings.percentile_mode,
            "implementation": "scripts.rewindow_from_raw.label_cell",
        },
        "probe_rule": {
            "min_disjoint_windows": boundary.MIN_PROBE_WINDOWS,
            "violation_window_fraction": boundary.VIOLATION_WINDOW_FRACTION,
            "implementation": "scripts.adaptive_boundary.probe_verdict",
        },
        "tables": {
            WINDOW_TABLE: {"rows": window_count, "columns": WINDOW_COLUMNS,
                           "contains": "every window of every non-void attempt"},
            REQUEST_TABLE: {"rows": request_count, "columns": REQUEST_COLUMNS,
                            "contains": "every request of every attempt, void included"},
            CELL_TABLE: {"rows": cells_written, "columns": CELL_COLUMNS,
                         "contains": "one row per attempt, void and missing included"},
        },
        "cells": manifest_cells,
        "boundary_searches": boundary_docs,
        "discrepancies": discrepancies,
    }
    (staging / MANIFEST).write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    if out_dir.exists():
        shutil.rmtree(out_dir)
    staging.rename(out_dir)
    return out_dir


def _probe_summary(probe: dict, manifest_cells: Sequence[dict], search: dict) -> dict:
    match = next(
        (c for c in manifest_cells
         if c["model"] == search.get("model") and c["shape"] == search.get("shape")
         and c["cell_id"] == probe.get("cell_id") and c["attempt"] == int(probe.get("attempt", 1))),
        None,
    )
    return {
        "cell_id": probe.get("cell_id"),
        "attempt": int(probe.get("attempt", 1)),
        "stage": probe.get("stage"),
        "rho": probe.get("rho"),
        "duration_s": probe.get("duration_s"),
        "verdict_recorded": _recorded_verdict(probe),
        "verdict_current_rule": None if match is None else match.get("probe_verdict"),
        "recorded": probe,
    }


def _convert_attempt(attempt, run_dir, settings, registry, plan_cells, discrepancies) -> dict:
    guard = attempt.guard
    void_reasons = [str(r) for r in (guard.get("void_reasons") or [])]
    if attempt.raw_path is not None and attempt.raw_path.name.endswith(r3_grid.VOID_RAW_SUFFIX):
        void_reasons = void_reasons or ["raw capture quarantined as void"]
    if attempt.raw_path is not None and not guard:
        void_reasons = void_reasons or ["no guard artifact"]
    meta = _cell_metadata(attempt, plan_cells)
    ledger = attempt.ledger or {}
    identity = {
        "model": attempt.model,
        "shape": attempt.shape,
        "primitive": attempt.primitive,
        "stage": attempt.stage,
        "rho": attempt.rho,
        "cell_id": attempt.cell_id,
        "attempt": attempt.attempt,
        "split": ledger.get("split") or _split(attempt.shape),
        "role": ledger.get("role"),
        "rho_factor": ledger.get("rho_factor"),
        "replicate": ledger.get("replicate"),
        "possibly_contaminated": ledger.get("possibly_contaminated"),
    }
    warmup_s = ledger.get("warmup_s") if attempt.ledger is not None else None
    warmup_end_ms = None
    if warmup_s is not None and guard.get("start_ms") is not None:
        warmup_end_ms = int(guard["start_ms"]) + int(round(float(warmup_s) * 1000))

    def in_warmup(ts_ms) -> Optional[bool]:
        if warmup_end_ms is None or ts_ms is None:
            return None
        return int(ts_ms) < warmup_end_ms
    records: list[dict] = []
    instants: list[dict] = []
    windows: list[dict] = []
    if attempt.raw_path is not None:
        records, instants, _guard, unmatched = rewindow_from_raw.load_cell_capture(attempt.raw_path)
        if unmatched:
            discrepancies.append(
                f"{attempt.stem}: {unmatched} failure record(s) matched no raw request"
            )
        sent = guard.get("sent")
        if sent is not None and int(sent) != len(records):
            discrepancies.append(
                f"{attempt.stem}: guard says {sent} request(s) sent, raw capture has {len(records)}"
            )
    verdict = None
    if records and not void_reasons:
        try:
            cell = r3_grid.GridCell.from_scenario_id(attempt.cell_id)
        except ValueError:
            cell = None
            discrepancies.append(f"{attempt.stem}: cell id {attempt.cell_id!r} does not parse")
        if cell is not None:
            windows = rewindow_from_raw.label_cell(
                records, instants, cell, registry.model(attempt.model),
                latency_slo_ms=settings.latency_slo_ms,
                window_ms=settings.window_ms, step_ms=settings.step_ms,
                percentile_mode=settings.percentile_mode,
                min_latency_samples=settings.min_latency_samples,
                instant_sample_interval_ms=SCRAPE_INTERVAL_MS,
                instant_grid=rewindow_from_raw.INSTANT_GRID_LIVE,
                start_ms=rewindow_from_raw._as_int(guard.get("start_ms")),
                end_ms=rewindow_from_raw._as_int(guard.get("end_ms")),
                truncated_at_ts_ms=rewindow_from_raw._as_int(guard.get("truncated_at_ts_ms")),
            )
            if attempt.primitive == gen.HOLD_PRIMITIVE and attempt.ledger is None:
                verdict = boundary.probe_verdict(
                    windows, ttft_slo_ms=settings.ttft_slo_ms, tpot_slo_ms=settings.tpot_slo_ms,
                )
            elif attempt.primitive == gen.HOLD_PRIMITIVE:
                # The ladder design labels every hold cell on its post-warm-up windows,
                # and a backlog stop is a violation (scripts.calibration_design).
                kept = [w for w in windows if not in_warmup(w["window_start_ms"])]
                verdict = boundary.probe_verdict(
                    kept, ttft_slo_ms=settings.ttft_slo_ms, tpot_slo_ms=settings.tpot_slo_ms,
                )
                if (guard.get("truncated")
                        and guard.get("truncation_cause") == openloop.TRUNCATION_BACKLOG):
                    verdict = replace(verdict, verdict=boundary.VERDICT_VIOLATED)
    if attempt.raw_path is None:
        status = STATUS_MISSING
    elif void_reasons:
        status = STATUS_VOID
    elif (verdict is not None and verdict.verdict == boundary.VERDICT_INCONCLUSIVE
          and ledger.get("role", "boundary") == "boundary"):
        # A probe with too little evidence is inconclusive; a ladder, supplementary or
        # sentinel cell short of labelled windows is still a valid measurement.
        status = STATUS_INCONCLUSIVE
    else:
        status = STATUS_VALID
    identity["cell_status"] = status
    recorded = _recorded_verdict(attempt.recorded_probe)
    current = (
        boundary.VERDICT_VOID if (attempt.primitive == gen.HOLD_PRIMITIVE and status == STATUS_VOID)
        else (verdict.verdict if verdict is not None else None)
    )
    if attempt.primitive == gen.HOLD_PRIMITIVE and recorded and current and recorded != current:
        discrepancies.append(
            f"{attempt.model}/{attempt.shape} probe {attempt.cell_id} attempt {attempt.attempt} "
            f"(rho={attempt.rho}): the search recorded {recorded}, the current rule says {current}"
        )

    online_csv = attempt.campaign / f"{attempt.stem}.csv" if attempt.stem else None
    if online_csv is not None and not online_csv.exists():
        online_csv = None
    online_parity = None
    if online_csv is not None and windows and status != STATUS_VOID:
        online_parity = _check_online_parity(attempt, online_csv, windows, discrepancies)

    outcomes = {name: 0 for name in openloop.OUTCOME_NAMES.values()}
    request_rows = []
    for record in records:
        outcome = rewindow_from_raw.request_outcome(record)
        outcomes[outcome] = outcomes.get(outcome, 0) + 1
        request_rows.append({
            **identity,
            "in_warmup": in_warmup(record.get("send_ts_ms")),
            "request_id": record.get("request_id"),
            "scheduled_send_ts_ms": record.get("scheduled_send_ts_ms"),
            "send_ts_ms": record.get("send_ts_ms"),
            "first_token_ts_ms": record.get("recv_first_token_ts_ms"),
            "done_ts_ms": record.get("done_ts_ms"),
            "on_wire_delay_ms": record.get("on_wire_delay_ms"),
            "ttft_ms": record.get("ttft_ms"),
            "tpot_ms": record.get("tpot_ms"),
            "e2e_ms": record.get("e2e_ms"),
            "input_tokens": record.get("input_tokens"),
            "output_tokens": record.get("output_tokens"),
            "http_status": record.get("http_status"),
            "outcome": outcome,
            "proxy_reason": record.get("proxy_reason"),
            "in_flight_at_send": record.get("in_flight_at_send"),
            "request_timeout_s": record.get("request_timeout_s"),
            "target_pod": record.get("target_pod"),
        })
    window_rows = (
        [{**identity, "in_warmup": in_warmup(row["window_start_ms"]), **row} for row in windows]
        if status != STATUS_VOID else []
    )

    goodput = (guard.get("goodput") or {}).get("goodput") if isinstance(guard.get("goodput"), dict) else None
    guard_path = (
        attempt.raw_dir / f"{attempt.cell_id}.guard.json"
        if attempt.raw_dir is not None and attempt.cell_id else None
    )
    files = {}
    if attempt.raw_dir is not None:
        files = {
            p.name.split(".", 1)[1] if "." in p.name else p.name: _rel(p, run_dir)
            for p in sorted(attempt.raw_dir.iterdir()) if p.is_file()
        }
    if online_csv is not None:
        files["online_csv"] = _rel(online_csv, run_dir)
    if meta["schedule_path"]:
        files["schedule"] = _rel(meta["schedule_path"], run_dir)
    if guard.get("prompt_file"):
        files["prompts"] = _rel(guard["prompt_file"], run_dir)

    cell_row = {
        **{k: v for k, v in identity.items() if k != "cell_status"},
        "status": status,
        "void_reasons": "; ".join(void_reasons),
        "probe_verdict": current,
        "probe_verdict_recorded": recorded,
        "windows": len(windows),
        "labeled_windows": None if verdict is None else verdict.labeled_windows,
        "independent_windows": None if verdict is None else verdict.independent_windows,
        "violating_windows": sum(
            1 for w in windows if w.get(slo_labels.LABEL_COLUMN) == slo_labels.LABEL_VIOLATED
        ),
        "start_ms": guard.get("start_ms"),
        "end_ms": guard.get("end_ms"),
        "planned_duration_s": meta["planned_duration_s"],
        "capacity_rps": meta["capacity_rps"],
        "offered_rps": meta["offered_rps"],
        "requests": len(records),
        "requests_ok": outcomes.get("ok", 0),
        "requests_shed": outcomes.get("shed", 0),
        "requests_model_error": outcomes.get("model_error", 0),
        "requests_proxy_transient": outcomes.get("proxy_transient", 0),
        "requests_client_timeout": outcomes.get("client_timeout", 0),
        "goodput": goodput,
        "raw_path": _rel(attempt.raw_path, run_dir),
        "guard_path": _rel(guard_path, run_dir) if guard else None,
        "online_csv_path": _rel(online_csv, run_dir),
        "schedule_path": _rel(meta["schedule_path"], run_dir),
        "role": ledger.get("role"),
        "rho_factor": ledger.get("rho_factor"),
        "replicate": ledger.get("replicate"),
        "warmup_s": warmup_s,
        "arrival_seed": ledger.get("arrival_seed"),
        "prompt_key": ledger.get("prompt_key"),
        "possibly_contaminated": ledger.get("possibly_contaminated"),
        "drained_before": (ledger.get("drain_before") or {}).get("drained"),
        "drain_waited_s": (ledger.get("drain_before") or {}).get("waited_s"),
        "backlog_stopped": ledger.get("backlog_stopped"),
    }
    manifest_cell = {
        **{k: v for k, v in identity.items() if k != "cell_status"},
        "status": status,
        "void_reasons": void_reasons,
        "probe_verdict": current,
        "probe_verdict_recorded": recorded,
        "files": files,
    }
    manifest_cell["online_csv_parity"] = online_parity
    return {"cell": cell_row, "manifest": manifest_cell,
            "windows": window_rows, "requests": request_rows}


#: ``online_csv_parity`` values in the manifest.
PARITY_IDENTICAL = "identical"
PARITY_DIFFERENT = "different"
PARITY_NOT_COMPARABLE = "not comparable (online CSV predates the shared label)"


def _check_online_parity(attempt, online_csv: Path, windows: Sequence[dict], discrepancies) -> str:
    """For a capture made with the shared labelling path, its online CSV must carry the
    very windows and labels this conversion recomputes - a live check of the parity the
    tests pin. An older online CSV (server-side latency, 30 s tumbling) cannot be compared
    and says so in the manifest; it is summarised once per run, not listed per cell."""
    with online_csv.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        online = list(reader)
        columns = reader.fieldnames or []
    if slo_labels.LABEL_COLUMN not in columns:
        return PARITY_NOT_COMPARABLE
    ours = [(int(w["window_start_ms"]), int(w["window_end_ms"]), w[slo_labels.LABEL_COLUMN]) for w in windows]
    theirs = [(int(r["window_start_ms"]), int(r["window_end_ms"]), r[slo_labels.LABEL_COLUMN]) for r in online]
    if ours != theirs:
        discrepancies.append(
            f"{attempt.stem}: online CSV windows/labels differ from the re-labelled raw "
            f"({len(theirs)} vs {len(ours)} window(s))"
        )
        return PARITY_DIFFERENT
    return PARITY_IDENTICAL


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir", type=Path, help="a campaign --out-dir, or a directory of them")
    ap.add_argument("--out-dir", type=Path, default=None,
                    help=f"where to write (default: <run_dir>/{DATASET_DIR})")
    ap.add_argument("--registry", default=None,
                    help="registry whose trs parameters fill the signal columns (default: "
                         "the one the run recorded, else the repository's)")
    ap.add_argument("--window-ms", type=int, default=None)
    ap.add_argument("--step-ms", type=int, default=None)
    ap.add_argument("--ttft-slo-ms", type=float, default=None)
    ap.add_argument("--tpot-slo-ms", type=float, default=None)
    ap.add_argument("--min-latency-samples", type=int, default=None)
    args = ap.parse_args(argv)
    out = build_dataset(args.run_dir, out_dir=args.out_dir, overrides={
        "registry": args.registry, "window_ms": args.window_ms, "step_ms": args.step_ms,
        "ttft_slo_ms": args.ttft_slo_ms, "tpot_slo_ms": args.tpot_slo_ms,
        "min_latency_samples": args.min_latency_samples,
    })
    manifest = json.loads((out / MANIFEST).read_text(encoding="utf-8"))
    for name, table in manifest["tables"].items():
        print(f"{out / name}: {table['rows']} rows")
    print(f"{len(manifest['discrepancies'])} discrepancy note(s) in {out / MANIFEST}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
