"""Opt-in ``x-tre-bl-in-tokens``: the exact prompt length, counted before the run, sent
only when asked for, omitted (never guessed) when it cannot be counted; and the checker
that holds the sent value to ``usage.prompt_tokens``."""
from __future__ import annotations

import asyncio
import importlib.util
import json
import sys
from pathlib import Path

from tre_replayer.engine import model_tokenizer
from tre_replayer.engine.http_sender import StreamingHttpSender, StreamResult
from tre_replayer.engine.in_tokens import IN_TOKENS_HEADER, precount_in_tokens
from tre_replayer.engine.profiles import V1ChatOptions
from tre_replayer.engine.schedule import ScheduledRequest

SCRIPT = Path(__file__).parents[1] / "scripts" / "check_in_tokens_header.py"
_spec = importlib.util.spec_from_file_location("check_in_tokens_header", SCRIPT)
checker = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = checker
_spec.loader.exec_module(checker)


class WordTokenizer:
    """One token per whitespace-separated word; chat template ``<U> {content} <A>`` (+2)."""

    filler = " the"
    chat_prefix = "<U> "
    chat_suffix = " <A>"
    chat_error = None

    def __init__(self, model: str, path: str) -> None:
        self.model, self.path = model, path

    def encode_plain(self, text: str) -> list[str]:
        return text.split()


def _load(model, tokenizer_path=None):
    if model == "untokenizable":
        raise model_tokenizer.TokenizerUnavailable("no local tokenizer")
    return WordTokenizer(model, tokenizer_path or f"/fallback/{model}")


def _req(rid="m-0", model="m", prompt="one two three", count=None) -> ScheduledRequest:
    return ScheduledRequest(request_id=rid, model=model, scheduled_offset_s=0.0, prompt=prompt,
                            max_output_tokens=8, in_tokens_header=count)


def test_precount_is_the_templated_count_and_omits_what_it_cannot_count() -> None:
    events = [_req("m-0"), _req("m-1", prompt=""), _req("u-0", model="untokenizable")]
    out, summary = precount_in_tokens(events, api="chat", tokenizer_paths={"m": "/registry/m"}, load=_load)
    # chat: 3 content words + the template's 2; no prompt / no tokenizer -> None, never a guess
    assert [e.in_tokens_header for e in out] == [5, None, None]
    assert [e.prompt for e in out] == [e.prompt for e in events]  # nothing else changes
    assert summary["counted"] == 1 and summary["omitted"] == 2
    assert summary["omitted_by_model"] == {"m": 1, "untokenizable": 1}
    assert summary["tokenizers"] == {"m": "/registry/m"} and "untokenizable" in summary["errors"]
    # completions: the plain string, no template
    out, _ = precount_in_tokens(events[:1], api="completions", load=_load)
    assert out[0].in_tokens_header == 3


def _capture_sender(**kwargs):
    seen = []

    def fake(url, headers, body, timeout_s):
        seen.append(dict(headers))
        return StreamResult(200, 10.0, 20.0, prompt_tokens=7, completion_tokens=8)

    sender = StreamingHttpSender("http://gw/v1/completions", stream_call=fake, **kwargs)
    return sender, seen


def test_the_sender_sends_the_header_only_when_asked() -> None:
    sender, seen = _capture_sender(send_in_tokens=True)
    asyncio.run(sender(_req(count=7), 0.0, 0.0))
    asyncio.run(sender(_req("m-1", count=None), 0.0, 0.0))
    assert seen[0][IN_TOKENS_HEADER] == "7" and IN_TOKENS_HEADER not in seen[1]
    assert [r["in_tokens_header"] for r in sender.records] == [7, None]
    assert sender.provenance()["send_in_tokens"] is True

    sender, seen = _capture_sender()
    asyncio.run(sender(_req(count=7), 0.0, 0.0))
    assert IN_TOKENS_HEADER not in seen[0]
    assert "in_tokens_header" not in sender.records[0]  # a default row is unchanged


def test_e1_v1_sends_the_header_through_the_sdk_extra_headers() -> None:
    on = V1ChatOptions(send_in_tokens=True)
    assert on.kwargs_for("m", "p", 4, 12)["extra_headers"] == {IN_TOKENS_HEADER: "12"}
    assert "extra_headers" not in on.kwargs_for("m", "p", 4, None)
    assert "extra_headers" not in V1ChatOptions().kwargs_for("m", "p", 4, 12)
    assert V1ChatOptions().as_dict()["send_in_tokens"] is False


def test_run_trace_counts_with_the_registry_tokenizer_and_records_the_flag(tmp_path, monkeypatch) -> None:
    import tre_replayer.run_trace as rt
    from tre_common.registry import load_registry

    sent = []

    def fake(url, headers, body, timeout_s):
        sent.append((headers, json.loads(body)["prompt"]))
        return StreamResult(200, 10.0, 20.0, prompt_tokens=1, completion_tokens=1)

    monkeypatch.setattr(rt, "_dry_stream_call", fake)
    trace = tmp_path / "trace.json"
    trace.write_text(json.dumps({"dsqwen-7b": [{"start_time": 0, "end_time": 1, "rps": 4, "input_tokens": 16,
                                                "max_tokens": 8}]}))

    async def _instant(_s):
        return None

    common = dict(gateway_url="http://x", seed=1, dry_run=True, window_ms=1000, step_ms=1000,
                  trim_ramp_windows=0, sleep=_instant)
    assert rt.run_trace(str(trace), **common)["send_in_tokens"] is False
    assert all(IN_TOKENS_HEADER not in headers for headers, _ in sent)

    sent.clear()
    summary = rt.run_trace(str(trace), send_in_tokens=True, **common)
    weights = {m.name: m.weights_path for m in load_registry().models()}["dsqwen-7b"]
    assert summary["send_in_tokens"] is True
    assert summary["in_tokens_header"]["tokenizers"] == {"dsqwen-7b": weights}
    assert summary["in_tokens_header"]["counted"] == summary["requests"] == len(sent) > 0
    tok = model_tokenizer.load_tokenizer("dsqwen-7b", tokenizer_path=weights)
    for headers, prompt in sent:  # completions: the plain count of the string sent
        assert headers[IN_TOKENS_HEADER] == str(len(tok.encode_plain(prompt)))


def test_the_checker_compares_the_sent_value_with_usage(tmp_path) -> None:
    rows = [
        {"request_id": "a", "model": "m", "in_tokens_header": 9, "prompt_tokens": 9},        # replayer row
        {"request_id": "b", "model": "m", "in_tokens_header": 9, "prompt_tokens": 10},       # off by one
        {"request_id": "c", "model_name": "n", "in_tokens_header": 5, "input_tokens": 5},    # loadgen v1 line
        {"request_id": "d", "model_name": "n", "in_tokens_header": 5, "input_tokens": 0},    # failed: no usage
        {"request_id": "e", "model": "m", "in_tokens_header": None, "prompt_tokens": 4},     # header omitted
        {"request_id": "f", "model": "m", "prompt_tokens": 4},                               # flag off
    ]
    out = checker.check_rows(rows)
    assert (out["compared"], out["matched"], out["mismatched"]) == (3, 2, 1)
    assert (out["no_usage"], out["header_omitted"], out["without_field"]) == (1, 1, 1)
    assert out["mismatches"] == [{"request_id": "b", "model": "m", "sent": 9, "usage_prompt_tokens": 10, "diff": -1}]
    path = tmp_path / "records.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    assert checker.main([str(path)]) == 1
    assert checker.main([str(path), "--limit", "1"]) == 0
    path.write_text(json.dumps(rows[-1]) + "\n")
    assert checker.main([str(path)]) == 2
