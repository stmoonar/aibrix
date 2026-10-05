"""M2 stream-cut rule (user 2026-10-05, T14's rule): a route-timeout cut is a censored request
whose window stays a violation (label), not a void; a cell with > 5 % non-cut model errors is
void at audit and excluded from accept (listed). One definition: scripts.stream_cut."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import test_dline_freeze_accept as base  # the D22 freeze / accept fixture world
from scripts import dline_refit as dl
from scripts import r3_grid, rewindow_from_raw, stream_cut
from tre_common import slo_labels
from tre_common.registry import load_registry

MODEL = base.MODEL
REGISTRY = Path(__file__).resolve().parents[1] / "registry.yaml"


def test_a_cut_is_a_violation_not_a_void_and_non_cut_errors_void_the_cell(tmp_path, capsys) -> None:
    # 1. the label: 25 served requests + 2 cut mid-stream by the 150 s route timeout, all sent in
    #    the first window -> that window is violated (unserved), under the hybrid label too
    ok = [{"send_ts_ms": 1000 + 100 * k, "recv_first_token_ts_ms": 1200 + 100 * k, "done_ts_ms": 2200 + 100 * k,
           "ttft_ms": 200.0, "tpot_ms": 10.0, "e2e_ms": 1200.0, "input_tokens": 512, "output_tokens": 100,
           "http_status": 200, "outcome": "ok"} for k in range(25)]
    cuts = [{"send_ts_ms": 5000 + k, "recv_first_token_ts_ms": 5300, "done_ts_ms": None, "ttft_ms": 300.0,
             "e2e_ms": 151_000.0, "input_tokens": 512, "http_status": 200, "outcome": "model_error"} for k in range(2)]
    assert [stream_cut.classify(r) for r in cuts] == ["cut", "cut"]
    label = slo_labels.LabelDefinition(ttft_p95_ms=500.0, tpot_p95_ms=75.0, min_completed_requests=20,
                                       attribution="hybrid")
    rows = rewindow_from_raw.label_cell(
        ok + cuts, [], r3_grid.GridCell.from_scenario_id("i512_o100_c9"), load_registry(str(REGISTRY)).model(MODEL),
        label=label, window_ms=30_000, step_ms=30_000, percentile_mode="bucket_upper", min_latency_samples=10,
        instant_sample_interval_ms=1_000, instant_grid="raw", start_ms=0, end_ms=30_000)
    assert rows[0]["model_errors"] == 2 and rows[0][slo_labels.LABEL_COLUMN] == slo_labels.LABEL_VIOLATED

    # 2. accept on an M2 manifest: the cut cell is evaluated, the 6 % non-cut cell is excluded
    w = base._world(tmp_path, steps_cells=16)
    assert base._freeze(w, "onset") == 0
    man_path = base._seal(w)
    man = json.loads(man_path.read_text())
    man.update({"composition_name": dl.M2_COMPOSITION_NAME, "stream_cut": stream_cut.record()})
    man_path.write_text(json.dumps(man))
    cells = [c["cell_id"] for c in man["cells"]]
    bad, cut_cell = cells[-1], cells[-2]
    with open(w["mdir"] / "requests.csv", "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=["cell_id", "attempt", "outcome", "e2e_ms", "in_warmup"])
        wr.writeheader()
        for cid in cells:
            for k in range(100):
                err = (cid == bad and k < 6) or (cid == cut_cell and k < 10)
                wr.writerow({"cell_id": cid, "attempt": 1, "outcome": "model_error" if err else "ok",
                             "e2e_ms": (151_000.0 if cid == cut_cell else 3_000.0) if err else 1_000.0,
                             "in_warmup": False})
    with open(w["mdir"] / "windows.csv", newline="") as fh:
        rows_m = [r for r in csv.DictReader(fh) if r["cell_id"] in cells]
    bad_windows = sum(1 for r in rows_m if r["cell_id"] == bad)
    code = base._accept(w, man_path)
    assert code in (0, dl.EXIT_ACCEPT_FAILED)
    res = json.loads(dl.freeze_paths(w["freeze"])["result"].read_text())["models"][MODEL]
    audit = res["stream_cut_audit"]
    assert audit["excluded_from_evaluation"] == [bad]
    by = {c["cell_id"]: c for c in audit["cells"]}
    assert (by[cut_cell]["cut"], by[cut_cell]["non_cut"], by[cut_cell]["void_at_audit"]) == (10, 0, False)
    assert (by[bad]["non_cut"], by[bad]["void_at_audit"]) == (6, True)
    assert bad_windows and res["validation_rows"] == len(rows_m) - bad_windows


def test_an_m2_manifest_with_another_runtime_limit_is_refused(tmp_path, capsys) -> None:
    w = base._world(tmp_path, steps_cells=16)
    assert base._freeze(w, "onset") == 0
    man_path = base._seal(w)
    man = json.loads(man_path.read_text())
    man.update({"composition_name": dl.M2_COMPOSITION_NAME,
                "stream_cut": {**stream_cut.record(), "max_model_error_rate": 0.05}})
    man_path.write_text(json.dumps(man))
    assert base._accept(w, man_path) == dl.EXIT_REFUSED
    assert "max_model_error_rate 0.05" in capsys.readouterr().out
