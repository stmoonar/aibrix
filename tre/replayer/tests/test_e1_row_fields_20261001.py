"""Review P3 of the unified client (2026-10-01): the e1_v1 row keeps v1's token values
under ``*_v1``, ``output_tokens`` is the actual completion length and the trace's
bound is ``max_output_tokens``; the documented retry condition and the forking-thread
caveat of the process pool match the code."""

from __future__ import annotations

from tre_replayer.engine import procpool, transport
from tre_replayer.engine.schedule import ScheduledRequest
from tre_replayer.run_trace import e1_replay_row


def _record(**over):
    record = {
        "request_id": "r-1", "model_name": "m", "start_time": 100.0, "send_lateness_ms": 1.5,
        "ttft_strict_s": 0.2, "e2e_strict_s": 1.2, "input_tokens": 530, "output_tokens": 37,
        "http_status": 200, "http_status_strict": 200, "success_strict": True, "failure_strict": None,
        "error_message": None, "stream_error": None,
    }
    record.update(over)
    return record


def test_e1_row_token_fields():
    event = ScheduledRequest("r-1", "m", 3.0, prompt="p", prompt_tokens=512, max_output_tokens=256)
    row = e1_replay_row(_record(), event)
    assert row["input_tokens"] == 512            # the asked prompt length (replay meaning)
    assert row["input_tokens_v1"] == 530         # v1's usage value kept
    assert row["prompt_tokens"] == 530
    assert row["output_tokens"] == 37            # actual length, not the bound
    assert row["output_tokens_v1"] == 37 and row["completion_tokens"] == 37
    assert row["max_output_tokens"] == 256       # the trace's upper bound
    # what the scoring and the request-health gate read is unchanged
    assert row["ttft_ms"] == 200.0 and row["e2e_ms"] == 1200.0
    assert row["http_status"] == 200 and row["error"] is None


def test_e1_row_without_usage_has_no_invented_length():
    event = ScheduledRequest("r-1", "m", 3.0, prompt="p", prompt_tokens=512, max_output_tokens=256)
    row = e1_replay_row(_record(output_tokens=None, success_strict=False, failure_strict="http_5xx",
                                http_status_strict=503), event)
    assert row["output_tokens"] is None and row["max_output_tokens"] == 256
    assert row["http_status"] == 503 and row["error"].startswith("http_5xx")


def test_documented_retry_condition_and_fork_thread_caveat():
    import inspect

    source = inspect.getsource(transport)
    assert "send_request_headers.started" in source
    assert "not a byte of the request left" not in source
    assert "follows the **thread** that forked" in procpool.__doc__
    assert "main thread" in procpool.ProcessPoolRunner.__doc__
