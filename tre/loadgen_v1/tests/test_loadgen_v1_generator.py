"""The v1 trace generator (stage 1) still works from a config, offline.

Tokenizer download (modelscope) is replaced by a whitespace tokenizer stub; everything
else is the vendored v1 code path (IntegratedTraceGenerator2 = generate_mode "custom",
used by 5 of the 7 traces_v14 workloads).
"""

from __future__ import annotations

import json

import pytest

from loadgen_v1_testlib import V1_TRACE_FIELDS, write_config

pytest.importorskip("transformers")
pytest.importorskip("modelscope")


class _WordTokenizer:
    def encode(self, text, add_special_tokens=False):
        return text.split()

    def decode(self, tokens, skip_special_tokens=True):
        return " ".join(tokens)


def test_custom_mode_generates_v1_format_trace_files(tmp_path, monkeypatch):
    from tre_loadgen_v1 import cli
    from tre_loadgen_v1.data_generator import DataGenerator

    monkeypatch.setattr(DataGenerator, "load_tokenizer", lambda self, model_name: _WordTokenizer())
    cfg = write_config(tmp_path / "config.yaml", tmp_path / "base")
    text = cfg.read_text(encoding="utf-8").replace("duration_seconds: 10", "duration_seconds: 6")
    cfg.write_text(text, encoding="utf-8")
    (tmp_path / "trace.json").write_text(json.dumps({
        "ok": [{"start_time": 0, "end_time": 6, "rps": 3.0, "input_tokens": 12, "max_tokens": 5}],
        "ok-stop": [{"start_time": 0, "end_time": 3, "rps": 1.0}],
    }), encoding="utf-8")

    out = tmp_path / "gen"
    assert cli.main(["--config", str(cfg), "--stage", "trace", "--output", str(out)]) == 0

    traces = json.loads((out / "traces.json").read_text(encoding="utf-8"))
    assert traces, "no requests generated"
    for t in traces:
        assert list(t.keys()) == V1_TRACE_FIELDS
    assert {t["model_name"] for t in traces} <= {"ok", "ok-stop"}
    assert all(0 <= t["timestamp"] < 7 for t in traces)
    ok = [t for t in traces if t["model_name"] == "ok"]
    # v1 quirk kept: jitter (+0..0.9s) can push a request past its segment end, where the override
    # lookup finds no segment -> max_output_tokens None (-> model max_tokens at dispatch)
    assert ok and all(t["max_output_tokens"] in (5, None) for t in ok)
    assert sum(t["max_output_tokens"] == 5 for t in ok) > len(ok) // 2
    for name in ("load_timeline_rps.json", "load_timeline_token_rate.json"):
        timeline = json.loads((out / name).read_text(encoding="utf-8"))
        assert timeline and set(timeline[0]) == {"timestamp", "total_load", "phase_type", "model_loads"}


def test_trace_stage_rejects_trace_file(tmp_path):
    from tre_loadgen_v1 import cli

    cfg = write_config(tmp_path / "config.yaml", tmp_path / "base")
    assert cli.main(["--config", str(cfg), "--stage", "trace", "--trace-file", "x.json"]) == 2
