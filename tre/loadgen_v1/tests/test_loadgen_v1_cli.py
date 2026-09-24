"""CLI tests: replay a v1-format recorded traces.json end to end (fake server, localhost only)."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import uuid
from pathlib import Path

import pytest

from loadgen_v1_testlib import AUDIT_FIELDS, LOADGEN_ROOT, V1_RESPONSE_FIELDS, trace_rec, write_config
from tre_loadgen_v1.cli import normalize_gateway_endpoint
from tre_loadgen_v1.config_manager import ConfigManager

V14_DIR = LOADGEN_ROOT / "configs" / "traces_v14"
V14_TRACES = [
    "Alternating_hot_model_periodic_A", "Decode_heavy_burst", "Prefill_mixed_corner_decode_mix",
    "Real_code_2024_slice_a_tok70", "Real_conv_2023_slice_a_tok70", "Simultaneous_spike_ramp_twice_tps1o2",
    "Sinusoidal_demand",
]


def _run_cli(args, cwd):
    env = dict(os.environ)
    env["PYTHONPATH"] = str(LOADGEN_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    return subprocess.run([sys.executable, "-m", "tre_loadgen_v1", *args], cwd=cwd, env=env,
                          capture_output=True, text=True, timeout=240)


def _recorded_dir(tmp_path: Path, tag: str) -> Path:
    rec = tmp_path / "recorded" / "tre" / "Some_trace"
    rec.mkdir(parents=True)
    traces = [
        trace_rec("req_000001", 0.2, "ok", f"{tag} a", max_out=3),
        trace_rec("req_000002", 0.4, "ok", f"{tag} b", max_out=None),
        trace_rec("req_000003", 0.6, "ok-stop", f"{tag} c", max_out=2, phase="transition"),
        trace_rec("req_000004", 0.8, "fail503x1", f"{tag} d", max_out=2),
    ]
    (rec / "traces.json").write_text(json.dumps(traces, indent=2, ensure_ascii=False), encoding="utf-8")
    timeline = [{"timestamp": 0.0, "total_load": 4.0, "phase_type": "stable", "model_loads": {"ok": 4.0}}]
    (rec / "load_timeline_rps.json").write_text(json.dumps(timeline), encoding="utf-8")
    (rec / "load_timeline_token_rate.json").write_text(json.dumps(timeline), encoding="utf-8")
    return rec


def test_replay_recorded_trace_end_to_end(fake_server, tmp_path):
    tag = uuid.uuid4().hex[:8]
    rec = _recorded_dir(tmp_path, tag)
    cfg = write_config(tmp_path / "config.yaml", tmp_path / "base")
    out = tmp_path / "out" / "tre" / "Some_trace"
    proc = _run_cli(["--config", str(cfg), "--trace-file", str(rec / "traces.json"),
                     "--base-url", fake_server.url + "/v1", "--output", str(out),
                     "--max-retries", "0", "--routing-strategy", ""], cwd=tmp_path)
    assert proc.returncode == 0, proc.stdout[-3000:] + proc.stderr[-3000:]

    # v1 output layout, inputs copied byte-for-byte
    for name in ("traces.json", "load_timeline_rps.json", "load_timeline_token_rate.json"):
        assert (out / name).read_bytes() == (rec / name).read_bytes()
    lines = [json.loads(line) for line in (out / "performance_metrics.json").read_text().splitlines() if line]
    assert len(lines) == 4
    for line in lines:
        assert list(line.keys()) == V1_RESPONSE_FIELDS + AUDIT_FIELDS
    by_id = {line["request_id"]: line for line in lines}
    assert by_id["req_000003"]["phase_type"] == "transition"
    assert by_id["req_000001"]["output_tokens"] == 3 and by_id["req_000002"]["output_tokens"] == 6
    # --max-retries 0: a single 503 is final
    d = by_id["req_000004"]
    assert d["success"] is False and d["attempts"] == 1 and d["http_status"] == 503
    # v1 dispatcher side outputs + stage-3 analysis still produced
    # (process_load_over_time.png is skipped by v1 itself when no worker stats were sampled: short run)
    assert (out / "actual_send_rps_by_model.png").exists()
    assert (out / "plot_analysis" / "all_requests_cdf.png").exists()
    meta = json.loads((out / "loadgen_run_meta.json").read_text())
    assert meta["max_retries"] == 0 and meta["routing_strategy_header"] is None
    assert meta["gateway_endpoint"] == fake_server.url
    assert meta["summary"]["collected_records"] == 4 and meta["summary"]["failed"] == 1

    # --routing-strategy "" -> header not sent; base_url normalised to <gw>/v1/chat/completions
    reqs = fake_server.records_for({f"{tag} a", f"{tag} b", f"{tag} c", f"{tag} d"})
    assert len(reqs) == 4
    for r in reqs:
        assert r["path"] == "/v1/chat/completions"
        assert "routing-strategy" not in r["headers"]


def test_dispatch_requires_explicit_base_url(tmp_path):
    rec = _recorded_dir(tmp_path, "nobase")
    cfg = write_config(tmp_path / "config.yaml", tmp_path / "base")
    proc = _run_cli(["--config", str(cfg), "--trace-file", str(rec / "traces.json"),
                     "--stage", "dispatch", "--output", str(tmp_path / "o")], cwd=tmp_path)
    assert proc.returncode == 2
    assert "--base-url" in proc.stdout
    assert not (tmp_path / "o" / "performance_metrics.json").exists()


@pytest.mark.parametrize("raw,expected", [
    ("http://gw:80", "http://gw:80"),
    ("http://gw:80/", "http://gw:80"),
    ("http://gw:80/v1", "http://gw:80"),
    ("http://gw:80/v1/", "http://gw:80"),
])
def test_base_url_normalisation(raw, expected):
    assert normalize_gateway_endpoint(raw) == expected


@pytest.mark.parametrize("trace", V14_TRACES)
def test_vendored_v14_configs_load_with_v1_client_settings(trace, tmp_path):
    cm = ConfigManager(str(V14_DIR / trace / "config.yaml"))
    cfg = cm.load_config(str(tmp_path / trace))  # sub_dir override (absolute) keeps the repo clean
    c = cfg.client
    assert c.max_retries == 2  # not in v1 configs -> v1's hard-coded default
    assert c.routing_algorithm == "least-gpu-cache"
    assert (c.process_count, c.max_coroutines_per_process, c.task_batch_window) == (8, 100, 5.0)
    assert c.timeout == 300.0 and c.enable_streaming is True
    assert [m.name for m in cfg.models] == ["dsllama-8b", "dsqwen-7b", "dsqwen-14b"]
    assert all(m.temperature is None and m.max_tokens for m in cfg.models)
