"""The boundary search and the fit use one ruler, window for window.

The 2026-09-21 campaign judged its probes on server-side p95 over 30 s tumbling windows
and fitted theta on client-side p95 over 30 s / 5 s sliding windows re-built from the raw
log: two definitions of "violated", which disagreed on one window in four. These tests
pin the repair by construction - the probe's online rows and an offline
``rewindow_from_raw`` of the very same capture must be the same file, byte for byte, and
must give the same probe verdict.
"""
from __future__ import annotations

import json
import sys
from argparse import Namespace
from pathlib import Path

import pytest

from tre_common import slo_labels
from tre_common.registry import load_registry
from scripts import adaptive_boundary as boundary
from scripts import calibration_campaign as campaign
from scripts import openloop, r3_grid, rewindow_from_raw

TRE_ROOT = Path(__file__).resolve().parents[2]
REGISTRY_PATH = TRE_ROOT / "deploy" / "registry.yaml"
MODEL = "dsqwen-7b"
CELL_ID = "i256_o128_c1090"
START_MS = 1_790_000_000_000
DURATION_S = 90
TTFT_SLO, TPOT_SLO = 500.0, 75.0
TERMINATION_OFFSET_MS = 70_250


def _sender_records() -> list[dict]:
    """A 90 s open-loop cell as the sender records it: 4 rps, a slow patch in the
    middle, one dropped connection, one engine error and one client timeout."""
    records: list[dict] = []
    for i in range(DURATION_S * 4):
        offset = 500 + i * 250
        slow = 40_000 <= offset < 55_000
        ttft = 120.0 + (i % 7) * 10.0
        tpot = 110.0 if slow else 20.0 + (i % 5)
        e2e = ttft + tpot * 127
        records.append({
            "request_id": f"{MODEL}-{i:06d}", "model": MODEL,
            "scheduled_offset_s": offset / 1000.0,
            "actual_send_ts_ms": START_MS + offset,
            "on_wire_delay_ms": 1.25,
            "ttft_ms": ttft, "e2e_ms": e2e,
            "input_tokens": 256, "output_tokens": 128,
            "prompt_tokens": 256, "completion_tokens": 128,
            "http_status": 200, "error": None, "error_body": None, "error_headers": None,
            "target_pod": None, "client_timeout": False, "request_timeout_s": 32.0,
            "in_flight_at_send": 3 + i % 4,
        })
    base = dict(records[0], ttft_ms=None, completion_tokens=None, prompt_tokens=None)
    records.append(dict(
        base, request_id=f"{MODEL}-900001", actual_send_ts_ms=START_MS + TERMINATION_OFFSET_MS,
        http_status=503, e2e_ms=2.9, error="HTTP 503",
        error_body="upstream connect error or disconnect/reset before headers. "
                   "reset reason: connection termination",
        error_headers={"content-type": "text/plain", "connection": "close"},
    ))
    records.append(dict(
        base, request_id=f"{MODEL}-900002", actual_send_ts_ms=START_MS + 75_100,
        http_status=500, e2e_ms=31.0, error="HTTP 500",
        error_body='{"error": "engine died"}', error_headers={"content-type": "application/json"},
    ))
    records.append(dict(
        base, request_id=f"{MODEL}-900003", actual_send_ts_ms=START_MS + 72_400,
        http_status=0, e2e_ms=32_050.0, error="TimeoutError", client_timeout=True,
    ))
    return records


def _sidecar() -> list[dict]:
    samples = [
        {"ts_ms": START_MS + 1000 * s + 3, "waiting": float(s % 3), "running": 4.0,
         "swapping": 0.0}
        for s in range(DURATION_S + 1)
    ]
    return openloop.mark_live_grid(samples)


def _args() -> Namespace:
    return Namespace(window_ms=30000, step_ms=5000, percentile_mode="bucket_upper",
                     min_latency_samples=10)


def _online_rows(records, sidecar, spec) -> list[dict]:
    return r3_grid.label_schedule_cell_windows(
        _args(), spec, r3_grid.GridCell.from_scenario_id(CELL_ID), CELL_ID,
        records, sidecar,
        start_ms=START_MS, end_ms=START_MS + DURATION_S * 1000 + 400,
        truncated_at_ts_ms=None, ttft_slo_ms=TTFT_SLO, tpot_slo_ms=TPOT_SLO,
    )


def _write_capture(raw_dir: Path, records, sidecar, *, with_outcomes: bool = True) -> None:
    """Lay the capture down the way ``drive_cell_schedule`` + ``r3_grid`` do."""
    raw = [openloop._raw_from_sender_record(CELL_ID, r) for r in records]
    if not with_outcomes:
        # A capture from before 2026-09-23: the raw log had no request fields, and the
        # verdict of each failure lived only in the failure sidecar.
        raw = [{k: v for k, v in r.items() if k in r3_grid.RAW_COLUMNS} for r in raw]
    openloop._append_jsonl(raw_dir / f"{CELL_ID}.jsonl", raw)
    openloop._append_jsonl(raw_dir / f"{CELL_ID}.instant.jsonl", sidecar)
    openloop._append_jsonl(raw_dir / f"{CELL_ID}.failures.jsonl", [
        openloop.failure_signature(r) for r in records
        if openloop.classify_failure(r) != openloop.FAILURE_NONE
    ])
    (raw_dir / f"{CELL_ID}.guard.json").write_text(json.dumps({
        "cell_id": CELL_ID, "start_ms": START_MS,
        "end_ms": START_MS + DURATION_S * 1000 + 400, "truncated_at_ts_ms": None,
        "void_reasons": [],
    }), encoding="utf-8")


def _rewindow(monkeypatch, raw_dir: Path, out: Path) -> None:
    monkeypatch.setattr(sys, "argv", [
        "rewindow_from_raw.py", "--model", MODEL, "--raw-dir", str(raw_dir),
        "--output", str(out), "--window-ms", "30000", "--step-ms", "5000",
        "--instant-grid", "live", "--instant-sample-ms", "10000",
        "--ttft-slo-ms", str(TTFT_SLO), "--tpot-slo-ms", str(TPOT_SLO),
        "--registry", str(REGISTRY_PATH),
    ])
    assert rewindow_from_raw.main() == 0


@pytest.fixture()
def spec():
    return load_registry(str(REGISTRY_PATH)).model(MODEL)


def test_the_probe_and_the_fit_label_the_same_capture_identically(tmp_path, monkeypatch, spec):
    records, sidecar = _sender_records(), _sidecar()
    online = _online_rows(records, sidecar, spec)
    online_csv = tmp_path / "online.csv"
    r3_grid.write_csv(online, online_csv)

    raw_dir = tmp_path / "raw" / "cell"
    raw_dir.mkdir(parents=True)
    _write_capture(raw_dir, records, sidecar)
    offline_csv = tmp_path / "fit.csv"
    _rewindow(monkeypatch, tmp_path / "raw", offline_csv)

    # Same windows, same latency, same unserved counts, same label - the same file.
    assert online_csv.read_text(encoding="utf-8") == offline_csv.read_text(encoding="utf-8")
    labels = [r["slo_label"] for r in online]
    assert {"violated", "healthy"} <= set(labels)

    # ... and therefore the same probe verdict, from the in-memory rows and from the CSV
    # the campaign reads back.
    kwargs = dict(ttft_slo_ms=TTFT_SLO, tpot_slo_ms=TPOT_SLO)
    assert (
        boundary.probe_verdict(online, **kwargs)
        == boundary.probe_verdict(campaign.read_window_rows(online_csv), **kwargs)
        == boundary.probe_verdict(campaign.read_window_rows(offline_csv), **kwargs)
    )


def test_a_dropped_connection_marks_its_re_windowed_window_violated(tmp_path, monkeypatch):
    # The online path always marked it; the re-window the fit is built from did not (its
    # slo_violated column was False on every row of refit_20260922). Now both call
    # openloop.mark_unserved_request_windows through rewindow_from_raw.label_cell.
    raw_dir = tmp_path / "raw" / "cell"
    raw_dir.mkdir(parents=True)
    _write_capture(raw_dir, _sender_records(), _sidecar())
    out = tmp_path / "fit.csv"
    _rewindow(monkeypatch, tmp_path / "raw", out)

    rows = campaign.read_window_rows(out)
    hit = START_MS + TERMINATION_OFFSET_MS
    holding = [r for r in rows if int(r["window_start_ms"]) <= hit < int(r["window_end_ms"])]
    assert holding
    for row in holding:
        assert row["proxy_transient_errors"] == 1
        assert row["slo_violated"] is True and row["slo_label"] == "violated"
    # a timed-out request and an engine error count too, each in its own column
    assert any(r["client_timeouts"] == 1 for r in rows)
    assert any(r["model_errors"] == 1 for r in rows)


def test_a_capture_without_request_fields_is_labelled_the_same(tmp_path, monkeypatch):
    # The converter re-labels 2026-09-21 captures, whose raw records carry no outcome; the
    # failure sidecar supplies it. The labels must not depend on which format was read.
    records, sidecar = _sender_records(), _sidecar()
    for name, with_outcomes in (("current", True), ("older", False)):
        raw_dir = tmp_path / name / "raw" / "cell"
        raw_dir.mkdir(parents=True)
        _write_capture(raw_dir, records, sidecar, with_outcomes=with_outcomes)
        _rewindow(monkeypatch, tmp_path / name / "raw", tmp_path / name / "fit.csv")
    current = campaign.read_window_rows(tmp_path / "current" / "fit.csv")
    older = campaign.read_window_rows(tmp_path / "older" / "fit.csv")
    assert current == older


def test_failed_requests_leave_no_latency_sample(spec):
    # A 503 in 3 ms or a 30 s client timeout is not a latency of the engine; it reaches
    # the label through the unserved counts, never through the p95.
    records = _sender_records()
    raw = [openloop._raw_from_sender_record(CELL_ID, r) for r in records]
    served_only = [r for r in raw if r["outcome"] == "ok"]
    ws, we = START_MS + 55_000, START_MS + 85_000
    kwargs = dict(percentile_mode="bucket_upper", min_latency_samples=10,
                  instant_sample_interval_ms=10000)
    assert (
        rewindow_from_raw.aggregate_window(raw, [], MODEL, ws, we, **kwargs).e2e_p95_ms
        == rewindow_from_raw.aggregate_window(served_only, [], MODEL, ws, we, **kwargs).e2e_p95_ms
    )


class _FakeStream:
    """The SSE seam, answering without a network: a fixed TTFT and end-to-end time."""

    def __call__(self, url, headers, body, timeout):
        from tre_replayer.engine.http_sender import StreamResult

        payload = json.loads(body)
        return StreamResult(200, 11.0, 40.0, 16, payload["max_tokens"])


class _FakeStore:
    """The redis MetricsStore, reduced to the one read the driver makes after a cell."""

    def read_model_window(self, model, start_ms, end_ms):
        from types import SimpleNamespace

        return SimpleNamespace(ttft_p95_ms=250.0, tpot_p95_ms=100.0, e2e_p95_ms=900.0)


def test_the_driver_itself_writes_what_the_re_window_reproduces(tmp_path, monkeypatch, spec):
    # End to end through r3_grid.run_schedule_cell - the code a probe really runs - rather
    # than through the helper it calls: drive a (tiny, fake-served) schedule, write the
    # online CSV, then re-window the capture it left on disk with the fit's parameters.
    schedule = tmp_path / "S1_hold1090.json"
    schedule.write_text(json.dumps({MODEL: [
        {"start_time": 0, "end_time": 1.2, "rps": 40.0, "input_tokens": 16, "max_tokens": 8},
    ]}), encoding="utf-8")
    real_drive = openloop.drive_cell_schedule

    def drive(*args, **kwargs):
        kwargs.update(prompt_dir=None, stream_call=_FakeStream())
        return real_drive(*args, **kwargs)

    monkeypatch.setattr(openloop, "drive_cell_schedule", drive)
    monkeypatch.setattr(openloop, "make_pod_metrics_sampler",
                        lambda endpoints: (lambda now: {"waiting": 1.0, "running": 2.0}))
    out = tmp_path / "online" / "dsqwen-7b_S1_S1_hold1090_a1.csv"
    args = r3_grid.parse_args([
        "--model", MODEL, "--gateway-url", "http://gw/v1/completions",
        "--schedule", str(schedule), "--cell-id", "i16_o8_c1090",
        "--output", str(out), "--raw-dir", str(tmp_path / "raw"),
        "--window-ms", "400", "--step-ms", "200", "--instant-sample-ms", "100",
        "--min-latency-samples", "1", "--ttft-slo-ms", "30", "--tpot-slo-ms", "75",
        "--pod-endpoint", "http://pod/metrics", "--prompt-mode", "token_ids",
        "--registry", str(REGISTRY_PATH), "--guard-mode", "warn",
    ])
    rows, guard = r3_grid.run_schedule_cell(args, _FakeStore(), spec)
    out.parent.mkdir(parents=True, exist_ok=True)
    r3_grid.write_csv(rows, out)
    assert rows and not guard.void_reasons
    assert all(r["p95_tpot_server_ms"] == 100.0 for r in rows)  # diagnostic, filled

    fit = tmp_path / "fit.csv"
    monkeypatch.setattr(sys, "argv", [
        "rewindow_from_raw.py", "--model", MODEL, "--raw-dir", str(tmp_path / "raw"),
        "--output", str(fit), "--window-ms", "400", "--step-ms", "200",
        "--instant-grid", "live", "--instant-sample-ms", "10000", "--min-latency-samples", "1",
        "--ttft-slo-ms", "30", "--tpot-slo-ms", "75", "--registry", str(REGISTRY_PATH),
    ])
    assert rewindow_from_raw.main() == 0

    server = {slo_labels.P95_TTFT_SERVER, slo_labels.P95_TPOT_SERVER, slo_labels.P95_E2E_SERVER}
    online_rows = campaign.read_window_rows(out)
    fit_rows = campaign.read_window_rows(fit)
    strip = lambda rs: [{k: v for k, v in r.items() if k not in server} for r in rs]  # noqa: E731
    assert strip(online_rows) == strip(fit_rows)
    # TTFT 11 ms and TPOT (40-11)/7 = 4.1 ms are under both SLOs
    assert {r["slo_label"] for r in online_rows} == {"healthy"}


def test_every_label_column_is_written_by_one_function(spec):
    rows = _online_rows(_sender_records(), _sidecar(), spec)
    for row in rows:
        expected = slo_labels.window_slo_label(row, slo_labels.slo_targets(
            ttft_slo_ms=TTFT_SLO, tpot_slo_ms=TPOT_SLO))
        assert row["slo_label"] == expected
        assert row["slo_violated"] == (
            None if expected == "unlabeled" else expected == "violated"
        )
