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
