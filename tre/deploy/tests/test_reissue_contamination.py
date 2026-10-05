"""Reissue-sidecar contamination of calibration cells (2026-10-03): a request the sidecar
continued during a cell (tre_reissue_total{kind="continue"} up on any model pod)
contaminates the cell; a pod whose counter was not read is unmeasured, never 0. Both are
void in the dataset."""
from __future__ import annotations

import json

import test_calibration_dataset as base  # the standard-dataset fixture run
from test_probe_label_parity import spec  # noqa: F401 - the registry model fixture
from scripts import calibration_capture as cc
from scripts import calibration_dataset as dataset

#: The sidecar's body (live format, 2026-10-03): the family is declared, no series until
#: the first event.
EMPTY = ("# HELP tre_reissue_total Requests the sidecar retried, continued, failed to save, or passed an "
         "abort through for.\n# TYPE tre_reissue_total counter\n"
         "# TYPE tre_reissue_proxy_added_seconds histogram\n"
         'tre_reissue_proxy_added_seconds_count{model="m"} 0\n')


def _body(**kinds: int) -> str:
    return EMPTY + "".join(f'tre_reissue_total{{model="m",kind="{k}",reason="r{i}"}} {v}\n'
                           for i, (k, v) in enumerate(kinds.items()))


def _read(bodies: dict):
    def fetch(url, timeout):
        b = bodies[url]
        if isinstance(b, Exception):
            raise b
        return b
    return cc.scrape_reissue({p: p for p in bodies}, fetch=fetch, now_ms=lambda: 1)


def test_counters_parse_and_an_absent_family_is_not_zero() -> None:
    assert cc.parse_reissue_counters(EMPTY) == {}
    assert cc.parse_reissue_counters(_body(retry=3, continue_=0) + _body(retry=2)) == {"retry": 5.0, "continue_": 0.0}
    try:
        cc.parse_reissue_counters('vllm:num_requests_running{model_name="m"} 1\n')  # vLLM's own /metrics
    except ValueError:
        pass
    else:
        raise AssertionError("a body without the tre_reissue_total family must not read as 0")
    assert cc.reissue_metrics_url("http://10.0.0.1:8000/metrics") == "http://10.0.0.1:8000/tre-reissue/metrics"


def test_a_continue_increase_contaminates_and_a_pod_not_read_is_unmeasured() -> None:
    clean = cc.reissue_check(_read({"a": EMPTY, "b": _body(retry=1)}), _read({"a": _body(retry=4), "b": _body(retry=1)}))
    assert clean["status"] == cc.REISSUE_CLEAN and clean["continue_delta"] == 0
    assert clean["delta_by_kind"] == {"retry": 4} and cc.reissue_void_reason(clean) is None

    hit = cc.reissue_check(_read({"a": EMPTY, "b": EMPTY}),
                           _read({"a": _body(**{"continue": 2}), "b": OSError("refused")}))
    assert hit["status"] == cc.REISSUE_CONTAMINATED and hit["continue_by_pod"] == {"a": 2}
    assert cc.reissue_void_reason(hit).startswith(cc.REISSUE_VOID_PREFIX + "contaminated")

    lost = cc.reissue_check(_read({"a": EMPTY, "b": EMPTY}), _read({"a": EMPTY, "b": OSError("timed out")}))
    assert lost["status"] == cc.REISSUE_UNMEASURED and lost["continue_delta"] is None
    assert lost["unmeasured_pods"] == ["b"] and cc.reissue_void_reason(lost).startswith("reissue_unmeasured")
    # a counter that went down (sidecar restart) and no pod at all are unmeasured too
    reset = cc.reissue_check(_read({"a": _body(**{"continue": 5})}), _read({"a": _body(**{"continue": 1})}))
    assert reset["status"] == cc.REISSUE_UNMEASURED
    assert cc.reissue_check(_read({}), _read({}))["status"] == cc.REISSUE_UNMEASURED
    assert cc.reissue_check(_read({}), _read({}), discovery_error="kubectl: boom")["status"] == cc.REISSUE_UNMEASURED


def _set_reissue(raw_dir, cell_id: str, rec: dict) -> None:
    path = raw_dir / f"{cell_id}.guard.json"
    guard = json.loads(path.read_text())
    guard["reissue"] = rec
    path.write_text(json.dumps(guard))


def test_the_dataset_excludes_contaminated_and_unmeasured_cells_and_counts_them(tmp_path) -> None:
    root = tmp_path / "run"
    model = "dsqwen-7b"
    camp = base._campaign(root, model)
    raw = camp / "raw"
    contaminated = cc.reissue_check(_read({"a": EMPTY}), _read({"a": _body(**{"continue": 1})}), policy="record")
    unmeasured = cc.reissue_check(_read({"a": EMPTY}), _read({"a": OSError("refused")}), policy="record")
    clean = cc.reissue_check(_read({"a": EMPTY}), _read({"a": EMPTY}))
    _set_reissue(raw / f"{model}_S1_steps", "i256_o128_c95", contaminated)
    _set_reissue(raw / f"{model}_M_steps", "i0_o0_c95", unmeasured)
    _set_reissue(raw / f"{model}_S1_S1_hold1090_a2", "i256_o128_c1090", clean)

    out = dataset.build_dataset(root)
    _, windows = base._read(out / "windows.csv")
    assert {w["cell_id"] for w in windows} == {"i256_o128_c1090"}       # only the clean cell
    _, cells = base._read(out / "cells.csv")
    by = {(c["cell_id"], c["attempt"]): c for c in cells}
    assert by[("i256_o128_c95", "1")]["status"] == "void"
    assert by[("i256_o128_c95", "1")]["reissue_status"] == "contaminated"
    assert by[("i256_o128_c95", "1")]["reissue_continue_delta"] == "1"
    assert by[("i0_o0_c95", "1")]["status"] == "void" and by[("i0_o0_c95", "1")]["reissue_continue_delta"] == ""
    assert by[("i256_o128_c1090", "2")]["status"] == "valid"
    assert by[("i256_o128_c1090", "1")]["reissue_status"] == cc.REISSUE_NOT_RECORDED  # captured before the check
    man = json.loads((out / "manifest.json").read_text())
    rc = man["reissue_check"]
    assert rc["counts"] == {"clean": 1, "contaminated": 1, "not_recorded": 1, "unmeasured": 1}
    assert {(e["cell_id"], e["reissue_status"]) for e in rc["excluded"]} == {
        ("i256_o128_c95", "contaminated"), ("i0_o0_c95", "unmeasured")}
    assert any(d.startswith("reissue check: 2 attempt(s) excluded") for d in man["discrepancies"])


def test_the_driver_voids_a_cell_with_a_continued_request(tmp_path, monkeypatch, spec) -> None:
    """r3_grid.run_schedule_cell under --capture-dir (the campaign's command): the counters
    are read before and after the load, a continue increase voids the cell and the record
    lands in the guard artifact the campaign and the dataset read."""
    from scripts import openloop, r3_grid

    import test_probe_label_parity as e2e

    schedule = tmp_path / "S1_hold1090.json"
    schedule.write_text(json.dumps({e2e.MODEL: [
        {"start_time": 0, "end_time": 0.6, "rps": 20.0, "input_tokens": 16, "max_tokens": 8}]}))
    real_drive = openloop.drive_cell_schedule
    monkeypatch.setattr(openloop, "drive_cell_schedule",
                        lambda *a, **kw: real_drive(*a, **{**kw, "prompt_dir": None, "stream_call": e2e._FakeStream()}))
    monkeypatch.setattr(openloop, "make_pod_metrics_sampler",
                        lambda endpoints, **kw: (lambda now: {"waiting": 1.0, "running": 2.0}))
    reads = iter([{"at_ms": 1, "pods": {"p": {"url": "u", "counters": {}}}},
                  {"at_ms": 2, "pods": {"p": {"url": "u", "counters": {"continue": 1.0}}}}])
    seen = []
    monkeypatch.setattr(cc, "scrape_reissue", lambda targets, **kw: seen.append(dict(targets)) or next(reads))

    def run(*extra):
        args = r3_grid.parse_args([
            "--model", e2e.MODEL, "--gateway-url", "http://gw/v1/completions", "--api", "completions", "--schedule", str(schedule),
            "--cell-id", "i16_o8_c1090", "--output", str(tmp_path / "o" / "m_S1_hold_a1.csv"),
            "--raw-dir", str(tmp_path / "raw"), "--capture-dir", str(tmp_path / "o" / "cells"),
            "--window-ms", "400", "--step-ms", "200", "--instant-sample-ms", "100", "--min-latency-samples", "1",
            "--ttft-slo-ms", "30", "--tpot-slo-ms", "75", "--min-completed-requests", "1",
            "--pod-endpoint", "http://10.0.0.9:8000/metrics", "--prompt-mode", "token_ids",
            "--registry", str(e2e.REGISTRY_PATH), "--guard-mode", "warn", "--no-vllm-metrics-capture",
            "--no-gateway-dump", "--no-controller-ticks", *extra])
        return r3_grid.run_schedule_cell(args, e2e._FakeStore(), spec)

    _, guard = run()
    assert seen == [{"http://10.0.0.9:8000/metrics": "http://10.0.0.9:8000/tre-reissue/metrics"}] * 2
    assert guard.voided and guard.void_reasons[-1].startswith("reissue_contaminated")
    artifact = json.loads(next((tmp_path / "raw").rglob("*.guard.json")).read_text())
    assert artifact["reissue"]["status"] == cc.REISSUE_CONTAMINATED and artifact["void_reasons"]
