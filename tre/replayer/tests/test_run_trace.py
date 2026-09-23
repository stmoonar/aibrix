from __future__ import annotations

import json

from tre_replayer.run_trace import run_trace


async def _instant_sleep(_seconds: float) -> None:
    return None


def test_run_trace_dry_run_scores_within_slo(tmp_path) -> None:
    trace = tmp_path / "trace.json"
    trace.write_text(
        json.dumps({"dsqwen-7b": [{"start_time": 0, "end_time": 2, "rps": 6, "input_tokens": 64, "max_tokens": 64}]})
    )
    out = tmp_path / "raw.jsonl"

    summary = run_trace(
        str(trace), gateway_url="http://x", out_path=str(out), seed=1, dry_run=True,
        window_ms=1000, step_ms=1000, trim_ramp_windows=0, sleep=_instant_sleep,
    )

    assert summary["requests"] > 0
    assert "dsqwen-7b" in summary["per_model"]
    assert out.exists()
    assert len(out.read_text().strip().splitlines()) == summary["requests"]
    # dry-run response (e2e = 80 + 64*3 = 272ms, tpot ~3ms) is well within dsqwen-7b SLO.
    assert summary["per_model"]["dsqwen-7b"]["violation_request_frac"] == 0.0


def _one_model_trace(tmp_path):
    trace = tmp_path / "trace.json"
    trace.write_text(
        json.dumps({"dsqwen-7b": [{"start_time": 0, "end_time": 1, "rps": 4, "input_tokens": 16, "max_tokens": 8}]})
    )
    return trace


def test_run_trace_routes_like_the_v1_client_by_default(tmp_path, monkeypatch) -> None:
    """v1 sent routing-strategy: least-gpu-cache on every request; the replayer does too,
    keeps the model header (tre-v2 per-model ORIGINAL_DST route) and tallies the pods the
    plugin reports, which is the post-deploy check that no sleeping pod is served."""
    import tre_replayer.run_trace as rt
    from tre_replayer.engine.http_sender import StreamResult

    seen: list[dict] = []

    def fake(url, headers, body, timeout_s):
        seen.append(dict(headers))
        return StreamResult(
            status=200, first_token_ms=50.0, done_ms=80.0, prompt_tokens=16, completion_tokens=8,
            target_pod="pod-a",
        )

    monkeypatch.setattr(rt, "_dry_stream_call", fake)
    common = dict(gateway_url="http://x", seed=1, dry_run=True, window_ms=1000, step_ms=1000,
                  trim_ramp_windows=0, sleep=_instant_sleep)
    summary = rt.run_trace(str(_one_model_trace(tmp_path)), **common)

    assert seen and all(h["routing-strategy"] == "least-gpu-cache" for h in seen)
    assert all(h["model"] == "dsqwen-7b" for h in seen)
    assert summary["routing_strategy"] == "least-gpu-cache"
    assert summary["target_pods"] == {"dsqwen-7b": {"pod-a": summary["requests"]}}

    seen.clear()
    summary = rt.run_trace(str(_one_model_trace(tmp_path)), routing_strategy=None, **common)
    assert seen and all("routing-strategy" not in h for h in seen)
    assert summary["routing_strategy"] is None


def test_run_trace_cli_defaults_to_tre_gateway_and_least_gpu_cache(tmp_path, monkeypatch, capsys) -> None:
    import tre_replayer.run_trace as rt

    calls: list[dict] = []
    monkeypatch.setattr(rt, "run_trace", lambda trace, **kw: calls.append(kw) or {"ok": True})
    rt.main(["--trace", "t.json"])
    rt.main(["--trace", "t.json", "--routing-strategy", "none"])
    assert calls[0]["gateway_url"] == "http://192.168.223.76:31094/v1/completions"
    assert calls[0]["routing_strategy"] == "least-gpu-cache"
    assert calls[1]["routing_strategy"] is None
