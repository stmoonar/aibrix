"""History uses past source requests and the frozen transform, never test averages."""
import copy
import hashlib
from tre_baselines.tools.preserve_history_frozen import build_history


def test_each_model_uses_its_own_past_with_chat_token_floor_once(tmp_path):
    source = tmp_path / "source.csv"
    source.write_text("timestamp,input,output\n"
        "2024-05-01T00:00:00+00:00,1,1\n"
        "2024-05-01T00:01:10+00:00,10,7\n"
        "2024-05-01T00:02:10+00:00,256,11\n"
        "2024-05-01T00:03:10+00:00,9999,9999\n")
    spec = {"model_order": ["a", "b"], "prompt": {"min_input_tokens_by_model": {"a": 64, "b": 64}}}
    trace = {"name": "Real_code", "source": {"file": "source.csv", "kind": "code", "sha256": hashlib.sha256(source.read_bytes()).hexdigest()},
             "models": {"a": {"source_start_utc": "2024-05-01T00:02:00+00:00", "keep_fraction": 1},
                        "b": {"source_start_utc": "2024-05-01T00:03:00+00:00", "keep_fraction": 1}}}
    doc = build_history(spec, trace, 1, tmp_path, window_s=30, seq_len=2)
    assert doc["models"]["a"] == [[64, 7], [0, 0]]
    assert doc["models"]["b"] == [[256, 11], [0, 0]]
    assert doc["source"]["per_model"]["a"]["input_adjusted_requests"] == 1
    assert doc["source"]["per_model"]["b"]["end_utc_exclusive"] == trace["models"]["b"]["source_start_utc"]


def test_synthetic_history_does_not_read_later_peaks_or_test_counts(tmp_path):
    spec = {"model_order": ["a"], "prompt": {"min_input_tokens_by_model": {"a": 64}}}
    trace = {"name": "Spike", "models": {"a": {"segments": [
        {"start_s": 0, "end_s": 60, "rate": {"kind": "constant", "rps": 2}, "input_tokens": 100, "output_tokens": 20},
        {"start_s": 60, "end_s": 120, "rate": {"kind": "constant", "rps": 100}, "input_tokens": 8000, "output_tokens": 1000}]}}}
    one = build_history(spec, trace, 3, tmp_path)
    changed = copy.deepcopy(trace)
    changed["models"]["a"]["segments"][1]["rate"]["rps"] = 999
    assert build_history(spec, changed, 3, tmp_path) == one
    assert len(one["models"]["a"]) == 25
