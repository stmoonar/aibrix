"""The standard dataset: one layout for every run, built without touching the run."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import pytest

from tre_common import slo_labels
from tre_common.registry import load_registry
from scripts import calibration_campaign as campaign
from scripts import calibration_dataset as dataset
from scripts import openloop, r3_grid, rewindow_from_raw

TRE_ROOT = Path(__file__).resolve().parents[2]
REGISTRY_PATH = TRE_ROOT / "deploy" / "registry.yaml"
START = 1_790_000_000_000


def _sender(i: int, send_ms: int, *, tpot: float, failed: bool = False) -> dict:
    record = {
        "request_id": f"r-{i:06d}", "actual_send_ts_ms": send_ms, "on_wire_delay_ms": 2.0,
        "ttft_ms": 150.0, "e2e_ms": 150.0 + tpot * 127, "prompt_tokens": 256,
        "completion_tokens": 128, "http_status": 200, "error": None, "error_body": None,
        "error_headers": None, "target_pod": None, "client_timeout": False,
        "request_timeout_s": 32.0, "in_flight_at_send": 4,
    }
    if failed:
        record.update(
            http_status=503, e2e_ms=3.0, ttft_ms=None, prompt_tokens=None, completion_tokens=None,
            error_body="upstream connect error or disconnect/reset before headers. "
                       "reset reason: connection termination",
            error_headers={"content-type": "text/plain"},
        )
    return record


def _capture(raw_dir: Path, cell_id: str, *, start: int, seconds: int, tpot: float,
             void: bool = False, with_outcomes: bool = True, fail_at: int | None = None) -> None:
    """Lay down one attempt the way r3_grid does (raw, sidecar, failures, guard)."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    senders = [_sender(i, start + 300 + 250 * i, tpot=tpot) for i in range(seconds * 4)]
    if fail_at is not None:
        senders.append(_sender(99999, start + fail_at, tpot=tpot, failed=True))
    raw = [openloop._raw_from_sender_record(cell_id, r) for r in senders]
    if not with_outcomes:
        raw = [{k: v for k, v in r.items() if k in r3_grid.RAW_COLUMNS} for r in raw]
    raw_path = raw_dir / f"{cell_id}.jsonl"
    openloop._append_jsonl(raw_path, raw)
    openloop._append_jsonl(raw_dir / f"{cell_id}.instant.jsonl", openloop.mark_live_grid([
        {"ts_ms": start + 1000 * s, "waiting": 0.0, "running": 3.0, "swapping": 0.0}
        for s in range(seconds + 1)
    ]))
    failures = [openloop.failure_signature(r) for r in senders
                if openloop.classify_failure(r) != openloop.FAILURE_NONE]
    if failures:
        openloop._append_jsonl(raw_dir / f"{cell_id}.failures.jsonl", failures)
    guard = {
        "cell_id": cell_id, "start_ms": start, "end_ms": start + seconds * 1000 + 300,
        "truncated_at_ts_ms": None, "void_reasons": ["envoy pending overflow"] if void else [],
        "sent": len(senders), "ttft_slo_ms": 500.0, "tpot_slo_ms": 75.0,
        "goodput": {"goodput": 0.97},
    }
    (raw_dir / f"{cell_id}.guard.json").write_text(json.dumps(guard), encoding="utf-8")
    if void:
        raw_path.replace(Path(str(raw_path) + r3_grid.VOID_RAW_SUFFIX))


def _campaign(root: Path, model: str, *, provenance: bool = True) -> Path:
    out = root / model
    raw = out / "raw"
    plan = {
        "models": [model],
        "cells": [
            {"model": model, "shape": "S1", "primitive": "steps", "cell_id": "i256_o128_c95",
             "capacity_rps": 4.0, "duration_s": 120.0},
            {"model": model, "shape": "M", "primitive": "steps", "cell_id": "i0_o0_c95",
             "capacity_rps": 3.0, "duration_s": 120.0},
        ],
    }
    if provenance:
        plan["provenance"] = {
            "code": {"commit": "abc123", "dirty": False},
            "registry_path": str(REGISTRY_PATH),
            "registry_sha256": hashlib.sha256(REGISTRY_PATH.read_bytes()).hexdigest(),
            "window_ms": 30000, "step_ms": 5000,
            "label": slo_labels.label_definition(
                slo_labels.slo_targets(ttft_slo_ms=500.0, tpot_slo_ms=75.0),
                min_latency_samples=10,
            ),
        }
    out.mkdir(parents=True)
    (out / "plan.json").write_text(json.dumps(plan), encoding="utf-8")
    (out / "fit_plan.json").write_text(json.dumps({"window_ms": 30000, "step_ms": 5000}),
                                       encoding="utf-8")
    _capture(raw / f"{model}_S1_steps", "i256_o128_c95", start=START, seconds=120, tpot=20.0,
             with_outcomes=provenance, fail_at=70_000)
    _capture(raw / f"{model}_M_steps", "i0_o0_c95", start=START + 200_000, seconds=120, tpot=20.0,
             with_outcomes=provenance)
    # a probe that voided, and its re-drive, which violated
    _capture(raw / f"{model}_S1_S1_hold1090_a1", "i256_o128_c1090", start=START + 400_000,
             seconds=90, tpot=20.0, void=True, with_outcomes=provenance)
    _capture(raw / f"{model}_S1_S1_hold1090_a2", "i256_o128_c1090", start=START + 600_000,
             seconds=90, tpot=120.0, with_outcomes=provenance)
    (out / "boundary").mkdir()
    (out / "boundary" / f"{model}_S1.json").write_text(json.dumps({
        "model": model, "shape": "S1",
        "probes": [
            {"rho": 0.9, "duration_s": 90.0, "stage": "coarse", "attempt": 1,
             "verdict": "void", "void_reasons": ["envoy pending overflow"],
             "cell_id": "i256_o128_c1090"},
            {"rho": 0.9, "duration_s": 90.0, "stage": "coarse", "attempt": 2,
             "verdict": "violated", "void_reasons": [], "cell_id": "i256_o128_c1090"},
        ],
    }), encoding="utf-8")
    return out


def _snapshot(root: Path) -> dict:
    return {
        str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(root.rglob("*")) if p.is_file() and dataset.DATASET_DIR not in p.parts
    }


def _read(path: Path) -> tuple[list[str], list[dict]]:
    with path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        return list(reader.fieldnames or []), list(reader)


def test_a_run_converts_into_the_standard_layout_without_touching_it(tmp_path):
    root = tmp_path / "run"
    _campaign(root, "dsqwen-7b")
    _campaign(root, "dsllama-8b")
    before = _snapshot(root)

    out = dataset.build_dataset(root)

    assert out == root / "dataset"
    assert _snapshot(root) == before  # nothing outside dataset/ written, moved or removed
    assert not (root / ".dataset.building").exists()
    assert {p.name for p in out.iterdir()} == {
        "manifest.json", "windows.csv", "requests.csv", "cells.csv", "DATASET.md",
    }

    columns, windows = _read(out / "windows.csv")
    assert columns == dataset.WINDOW_COLUMNS
    assert {w["model"] for w in windows} == {"dsqwen-7b", "dsllama-8b"}  # merged
    assert all(w["cell_status"] != "void" for w in windows)
    assert {w["split"] for w in windows if w["shape"] == "M"} == {"holdout"}
    assert {w["split"] for w in windows if w["shape"] == "S1"} == {"train"}
    assert not any(c.startswith("p95_") and not c.endswith(("_client_ms", "_server_ms"))
                   for c in columns)

    columns, requests = _read(out / "requests.csv")
    assert columns == dataset.REQUEST_COLUMNS
    void_requests = [r for r in requests if r["cell_status"] == "void"]
    assert void_requests and {r["attempt"] for r in void_requests} == {"1"}
    assert {r["outcome"] for r in requests} == {"ok", "proxy_transient"}

    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["format_revision"] == 2
    assert manifest["label"]["latency_source"] == "client per-request"
    # revision 2: the D6' primary label per model, three arms, D8 grid windows
    assert set(manifest["label_by_model"]) == {"dsqwen-7b", "dsllama-8b"}
    assert manifest["label"]["mode"] == "slowdown" and manifest["label"]["min_n"] == 20
    assert set(manifest["label"]["columns"].values()) == {"slo_label", "slo_label_fixed", "slo_label_k3"}
    assert manifest["windowing"]["window_ms"] == 30000 and manifest["windowing"]["step_ms"] == 10000
    assert manifest["windowing"]["window_align"] == "grid"
    assert all(int(w["window_end_ms"]) % 10000 == 0 for w in windows)
    assert {"slo_label_fixed", "slo_label_k3", "ttft_len_samples"} <= set(dataset.WINDOW_COLUMNS)
    assert manifest["campaigns"][0]["code"] == {"commit": "abc123", "dirty": False}
    assert manifest["tables"]["windows.csv"]["rows"] == len(windows)
    probes = [c for c in manifest["cells"] if c["model"] == "dsqwen-7b" and c["primitive"] == "hold"]
    assert [(c["attempt"], c["status"]) for c in sorted(probes, key=lambda c: c["attempt"])] == [
        (1, "void"), (2, "valid"),
    ]
    assert probes[0]["files"]["jsonl.void"].endswith("i256_o128_c1090.jsonl.void")
    assert all(c["probe_verdict"] == c["probe_verdict_recorded"] for c in probes)


def test_the_window_table_is_the_labelling_path_s_output(tmp_path):
    root = tmp_path / "run"
    campaign_dir = _campaign(root, "dsqwen-7b")
    out = dataset.build_dataset(root)
    _, windows = _read(out / "windows.csv")
    ours = [w for w in windows if w["cell_id"] == "i256_o128_c95"]

    raw_path = campaign_dir / "raw" / "dsqwen-7b_S1_steps" / "i256_o128_c95.jsonl"
    records, instants, guard, _ = rewindow_from_raw.load_cell_capture(raw_path)
    theirs = rewindow_from_raw.label_cell(
        records, instants, r3_grid.GridCell.from_scenario_id("i256_o128_c95"),
        load_registry(str(REGISTRY_PATH)).model("dsqwen-7b"),
        label=slo_labels.label_def_for_model(
            "dsqwen-7b", ttft_p95_ms=500.0, tpot_p95_ms=75.0, registry=str(REGISTRY_PATH)),
        window_ms=30000, step_ms=10000, window_align="grid",
        percentile_mode="bucket_upper", min_latency_samples=10,
        instant_sample_interval_ms=10000, instant_grid="live",
        start_ms=guard["start_ms"], end_ms=guard["end_ms"],
    )
    arms = ("slo_label", "slo_label_fixed", "slo_label_k3")
    assert [(int(w["window_start_ms"]), *(w[a] for a in arms)) for w in ours] == [
        (t["window_start_ms"], *(t[a] for a in arms)) for t in theirs
    ]
    # the dropped connection at +70 s is in the table as a violation
    assert any(w["proxy_transient_errors"] == "1" and w["slo_label"] == "violated" for w in ours)


def _write_online_csv(campaign_dir: Path, stem: str, cell_id: str, model: str) -> Path:
    """The online CSV the 09-23 driver wrote for one attempt: 30 s / 5 s free-phase windows,
    the fixed 500 / 75 ms label with no min-n guard (what the run's provenance records)."""
    raw_path = campaign_dir / "raw" / stem / f"{cell_id}.jsonl"
    records, instants, guard, _ = rewindow_from_raw.load_cell_capture(raw_path)
    rows = rewindow_from_raw.label_cell(
        records, instants, r3_grid.GridCell.from_scenario_id(cell_id),
        load_registry(str(REGISTRY_PATH)).model(model),
        latency_slo_ms=slo_labels.slo_targets(ttft_slo_ms=500.0, tpot_slo_ms=75.0),
        window_ms=30000, step_ms=5000, percentile_mode="bucket_upper", min_latency_samples=10,
        instant_sample_interval_ms=10000, instant_grid="live",
        start_ms=guard["start_ms"], end_ms=guard["end_ms"],
    )
    out = campaign_dir / f"{stem}.csv"
    r3_grid.write_csv(rows, out)
    return out


def test_the_online_csvs_of_the_run_are_reproduced_window_for_window(tmp_path):
    """The parity check of the merge: every online window label of a 09-23 capture is
    reproduced by re-windowing its raw on the run's own grid with the label the run
    recorded - and a single changed label is caught."""
    root = tmp_path / "run"
    campaign_dir = _campaign(root, "dsqwen-7b")
    online = _write_online_csv(campaign_dir, "dsqwen-7b_S1_steps", "i256_o128_c95", "dsqwen-7b")
    _write_online_csv(campaign_dir, "dsqwen-7b_M_steps", "i0_o0_c95", "dsqwen-7b")
    out = dataset.build_dataset(root)
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    parity = {c["cell_id"]: c["online_csv_parity"] for c in manifest["cells"]
              if c["primitive"] == "steps"}
    assert parity == {"i256_o128_c95": "identical", "i0_o0_c95": "identical"}
    counts = manifest["online_parity"]["fixed_arm_min_n_differences"]
    assert counts["windows_compared"] > 0
    # the dataset's own fixed arm differs from the online label only where min-n bites
    assert counts["differ"] == counts["differ_all_low_n"]

    _, rows = _read(online)
    rows[3]["slo_label"] = "violated" if rows[3]["slo_label"] != "violated" else "healthy"
    with online.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    out = dataset.build_dataset(root)
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    parity = {c["cell_id"]: c["online_csv_parity"] for c in manifest["cells"]
              if c["primitive"] == "steps"}
    assert parity["i256_o128_c95"] == "different"
    assert any("differ from the re-labelled raw in 1 window" in d for d in manifest["discrepancies"])


def test_a_run_from_before_provenance_still_converts_and_says_what_it_lacks(tmp_path):
    root = tmp_path / "run"
    _campaign(root, "dsqwen-7b", provenance=False)
    out = dataset.build_dataset(root)
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["campaigns"][0]["code"] is None
    assert manifest["campaigns"][0]["registry_sha256"] is None
    # the SLO the cells were judged against comes from their guards
    assert manifest["label"]["slo_ms"] == {"p95_ttft_client_ms": 500.0, "p95_tpot_client_ms": 75.0}
    # outcomes of an older capture come from the failure sidecar
    _, requests = _read(out / "requests.csv")
    assert sum(r["outcome"] == "proxy_transient" for r in requests) == 1


def test_an_old_probe_booked_healthy_on_two_windows_is_flagged(tmp_path):
    root = tmp_path / "run"
    out_dir = _campaign(root, "dsqwen-7b")
    # A 60 s probe recorded "healthy" the old way (violated: false, valid: true).
    _capture(out_dir / "raw" / "dsqwen-7b_S1_S1_hold1060_a1", "i256_o128_c1060",
             start=START + 900_000, seconds=60, tpot=120.0)
    search = json.loads((out_dir / "boundary" / "dsqwen-7b_S1.json").read_text())
    search["probes"].insert(0, {"rho": 0.6, "duration_s": 60.0, "stage": "coarse", "attempt": 1,
                                "violated": False, "valid": True, "void_reasons": [],
                                "windows": 2, "violating_windows": 2,
                                "cell_id": "i256_o128_c1060"})
    (out_dir / "boundary" / "dsqwen-7b_S1.json").write_text(json.dumps(search))

    out = dataset.build_dataset(root)
    _, cells = _read(out / "cells.csv")
    row = next(c for c in cells if c["cell_id"] == "i256_o128_c1060")
    assert row["probe_verdict_recorded"] == "healthy"
    assert row["probe_verdict"] == "inconclusive" and row["status"] == "inconclusive"
    assert row["independent_windows"] == "2"
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert any("recorded healthy, the current rule says inconclusive" in d
               for d in manifest["discrepancies"])


def test_the_tool_never_replaces_a_directory_it_did_not_write(tmp_path):
    root = tmp_path / "run"
    _campaign(root, "dsqwen-7b")
    (root / "dataset").mkdir()
    (root / "dataset" / "precious.txt").write_text("keep")
    with pytest.raises(SystemExit):
        dataset.build_dataset(root)
    assert (root / "dataset" / "precious.txt").read_text() == "keep"


def test_a_campaign_builds_its_dataset_and_the_last_one_builds_the_merge(tmp_path):
    root = tmp_path / "run"
    first = _campaign(root, "dsqwen-7b")
    second = _campaign(root, "dsllama-8b")

    campaign.finalize_run(first, status="complete", exit_code=0)
    assert (first / "dataset" / "windows.csv").exists()
    assert json.loads((first / "campaign_status.json").read_text())["status"] == "complete"
    assert not (root / "dataset").exists()  # the sibling is still running

    campaign.finalize_run(second, status="stopped", exit_code=1)
    assert (second / "dataset" / "windows.csv").exists()
    _, windows = _read(root / "dataset" / "windows.csv")
    assert {w["model"] for w in windows} == {"dsqwen-7b", "dsllama-8b"}
