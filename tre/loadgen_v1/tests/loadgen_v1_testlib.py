"""Shared helpers for tre_loadgen_v1 tests (fake server fixture, v1 field lists, config writer)."""

from __future__ import annotations

import json
import subprocess
import sys
import urllib.request
from pathlib import Path

import pytest
import yaml

HERE = Path(__file__).resolve().parent
LOADGEN_ROOT = HERE.parent

# v1 performance_metrics.json field order, taken from a real v1 run
# (output_traces_v14/tre/Decode_heavy_burst/performance_metrics.json).
V1_RESPONSE_FIELDS = [
    "request_id", "model_name", "timestamp", "start_time", "end_time", "e2e_latency", "ttft", "tpot",
    "input_tokens", "output_tokens", "total_tokens", "success", "error_message", "http_status",
    "phase_type", "target_pod", "process_id",
]
AUDIT_FIELDS = ["attempts", "stream_interrupted", "stream_error", "finish_reason", "attempt_log"]
# v1 traces.json record fields (RequestTrace)
V1_TRACE_FIELDS = ["request_id", "timestamp", "model_name", "prompt", "prompt_length", "phase_type",
                   "max_output_tokens"]
# v1 config.yaml output.files block (identical in every traces_v14 config)
V1_OUTPUT_FILES = {
    "load_timeline_rps": "load_timeline_rps.json",
    "load_timeline_token_rate": "load_timeline_token_rate.json",
    "trace_data": "traces.json",
    "trace_plots": "traces.png",
    "performance_metrics": "performance_metrics.json",
    "performance_plots": "plot_analysis/",
}
FAKE_MODELS = ["ok", "ok-stop", "fail503x1", "fail503x2", "fail503x5", "always500", "drop", "stall",
               "hang", "sseerror"]


class FakeServer:
    def __init__(self, proc: subprocess.Popen, port: int):
        self.proc = proc
        self.url = f"http://127.0.0.1:{port}"

    def records(self) -> list[dict]:
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        with opener.open(self.url + "/_records", timeout=10) as resp:
            return json.loads(resp.read())["records"]

    def records_for(self, prompts: set[str]) -> list[dict]:
        return [r for r in self.records() if r["body"].get("messages", [{}])[0].get("content") in prompts]


@pytest.fixture(scope="session")
def fake_server():
    proc = subprocess.Popen([sys.executable, str(HERE / "loadgen_v1_fake_server.py")],
                            stdout=subprocess.PIPE, text=True)
    line = proc.stdout.readline()
    assert line.startswith("PORT "), line
    srv = FakeServer(proc, int(line.split()[1]))
    yield srv
    proc.kill()
    proc.wait(timeout=10)


def write_config(path: Path, out_base: Path, *, model_max_tokens: int = 6, **client) -> Path:
    """A v1-shaped config.yaml (same keys/defaults as config/traces_v14/*/config.yaml)."""
    client_cfg = {
        "api_key": "dummy", "timeout": 300.0, "enable_streaming": True, "log_level": "INFO",
        "routing_algorithm": "least-gpu-cache", "process_count": 2, "max_coroutines_per_process": 100,
        "task_batch_window": 5.0, "load_monitor_interval": 1.0,
    }
    client_cfg.update(client)
    cfg = {"custom_load_test": {
        "duration_seconds": 10,
        "gateway_endpoint": "http://localhost:8888",
        "generate_mode": "custom",
        "output": {"base_dir": str(out_base), "sub_dir_name": "run", "files": dict(V1_OUTPUT_FILES)},
        "random_seed": 1,
        "client": client_cfg,
        "models": [{"name": m, "modelscope_url": f"fake/{m}", "max_tokens": model_max_tokens}
                   for m in FAKE_MODELS],
        "load_mode": "rps", "total_load": 4,
        "stable_period": {"min_duration": 30, "max_duration": 60},
        "transition_period": {"min_duration": 2, "max_duration": 5},
        "noise": {"enabled": False},
        "custom_trace_json": "trace.json",
        "input_token_config": {"min_length": 10, "max_length": 10, "distribution": "normal", "mean": 10, "std": 0},
        "analysis": {"percentiles": [50, 70, 90, 99]},
    }}
    path.write_text(yaml.safe_dump(cfg, allow_unicode=True), encoding="utf-8")
    return path


def trace_rec(rid: str, ts: float, model: str, prompt: str, max_out=None, phase="stable") -> dict:
    return {"request_id": rid, "timestamp": ts, "model_name": model, "prompt": prompt,
            "prompt_length": len(prompt.split()), "phase_type": phase, "max_output_tokens": max_out}
