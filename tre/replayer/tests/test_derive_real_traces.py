from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts" / "derive_real_traces.py"
spec = importlib.util.spec_from_file_location("derive_real_traces", SCRIPT)
assert spec is not None and spec.loader is not None
real_traces = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = real_traces
spec.loader.exec_module(real_traces)


def test_azure_timestamp_is_unix_utc() -> None:
    assert real_traces._azure_timestamp("2024-05-12 00:00:00.001163+00:00", {}) == 1715472000.001163

def test_stable_model_keeps_session_assignment_and_is_deterministic() -> None:
    keys = [f"session-{index}" for index in range(2000)]
    first = [real_traces.stable_model(key, seed=17) for key in keys]
    second = [real_traces.stable_model(key, seed=17) for key in keys]

    assert first == second
    fractions = {model: first.count(model) / len(first) for model in real_traces.MODEL_ORDER}
    assert 0.35 < fractions["dsllama-8b"] < 0.45
    assert 0.30 < fractions["dsqwen-7b"] < 0.40
    assert 0.20 < fractions["dsqwen-14b"] < 0.30


def test_derive_both_is_byte_reproducible_and_matches_schema(tmp_path: Path) -> None:
    azure = tmp_path / "azure.csv"
    azure.write_text(
        "TIMESTAMP,ContextTokens,GeneratedTokens\n"
        "2024-05-12 00:00:00+00:00,100,10\n"
        "2024-05-12 00:00:10+00:00,200,20\n"
        "2024-05-12 00:00:20+00:00,300,30\n"
        "2024-05-12 00:00:30+00:00,400,40\n",
        encoding="utf-8",
    )
    burst = tmp_path / "burst.csv"
    burst.write_text(
        "Timestamp,Session ID,Elapsed time,Model,Request tokens,Response tokens,Total tokens,Log Type\n"
        "0,same,1,GPT-4,100,10,110,Conversation log\n"
        "10,same,1,GPT-4,200,20,220,Conversation log\n"
        "20,other,1,ChatGPT,300,30,330,Conversation log\n"
        "30,ignored,1,ChatGPT,400,0,400,Conversation log\n"
        "40,ignored-api,1,ChatGPT,400,40,440,API log\n",
        encoding="utf-8",
    )

    roots = [tmp_path / "run1", tmp_path / "run2"]
    for root in roots:
        real_traces.derive_both(
            azure,
            burst,
            root / "traces",
            root / "evidence",
            target_duration_s=20,
            bin_width_s=5,
            target_peak_rps=12.0,
            seed=17,
        )

    for name in ("t8_azure_conv", "t9_burstgpt"):
        first = roots[0] / "traces" / name / "trace.json"
        second = roots[1] / "traces" / name / "trace.json"
        assert first.read_bytes() == second.read_bytes()
        segments = real_traces.json.loads(first.read_text(encoding="utf-8"))
        assert list(segments) == list(real_traces.MODEL_ORDER)
        flattened = [segment for model_segments in segments.values() for segment in model_segments]
        assert max(segment["end_time"] for segment in flattened) == 20
        assert all(set(segment) == {"start_time", "end_time", "rps", "input_tokens", "max_tokens"} for segment in flattened)

    burst_report = real_traces.json.loads(
        (roots[0] / "evidence" / "manifest.json").read_text(encoding="utf-8")
    )["reports"]["t9_burstgpt"]
    assert burst_report["valid_source_rows"] == 3
    assert abs(burst_report["derived_peak_rps"] - 12.0) < 1e-6