"""End-to-end tests of the ported v1 ClientDispatcher against a local fake OpenAI server.

One multi-scenario dispatch run (module fixture) exercises the real multi-process /
batch-window scheduler once; individual tests then assert on its records.  Nothing here
talks to the cluster: the server is a subprocess bound to 127.0.0.1.
"""

from __future__ import annotations

import asyncio
import json
import uuid

import openai
import pytest

from loadgen_v1_testlib import AUDIT_FIELDS, V1_RESPONSE_FIELDS, write_config
from tre_loadgen_v1.client_dispatcher import ClientDispatcher, WorkerProcess
from tre_loadgen_v1.config_manager import ConfigManager
from tre_loadgen_v1.trace_types import RequestTrace

TIMEOUT_S = 1.0          # client timeout for this run (v1 default is 300s)
LATE_TS = 6.2            # > task_batch_window: forces the dynamic sliding-window path
N_CONCURRENT_RETRY = 10  # concurrent fail503x1 requests (attempt attribution under concurrency)


def _dispatch(cfg_path, traces):
    cm = ConfigManager(str(cfg_path))
    cm.load_config()
    dispatcher = ClientDispatcher(cm)
    results = asyncio.run(dispatcher.dispatch_traces(traces))
    return cm, results


def _trace(rid, ts, model, prompt, max_out=None):
    return RequestTrace(request_id=rid, timestamp=ts, model_name=model, prompt=prompt,
                        prompt_length=len(prompt.split()), phase_type="stable", max_output_tokens=max_out)


@pytest.fixture(scope="module")
def run(fake_server, tmp_path_factory):
    tag = uuid.uuid4().hex[:8]
    tmp = tmp_path_factory.mktemp("dispatch")
    cfg = write_config(tmp / "config.yaml", tmp / "out", timeout=TIMEOUT_S)
    cfg_text = cfg.read_text(encoding="utf-8").replace("http://localhost:8888", fake_server.url)
    cfg.write_text(cfg_text, encoding="utf-8")

    traces = []
    # normal streams; timestamps spread over ~2s (first batch window)
    for i in range(12):
        traces.append(_trace(f"ok-{i}", 0.3 + i * 0.15, "ok", f"{tag} ok prompt {i}", max_out=4 if i % 2 else None))
    traces.append(_trace("ok-late", LATE_TS, "ok", f"{tag} ok late"))
    traces.append(_trace("stop-0", 0.4, "ok-stop", f"{tag} stop", max_out=3))
    traces.append(_trace("r1-0", 0.5, "fail503x1", f"{tag} r1 single"))
    traces.append(_trace("r2-0", 0.5, "fail503x2", f"{tag} r2"))
    traces.append(_trace("r5-0", 0.5, "fail503x5", f"{tag} r5"))
    traces.append(_trace("e500-0", 0.5, "always500", f"{tag} e500"))
    traces.append(_trace("drop-0", 0.6, "drop", f"{tag} drop"))
    traces.append(_trace("stall-0", 0.6, "stall", f"{tag} stall"))
    traces.append(_trace("hang-0", 0.6, "hang", f"{tag} hang"))
    traces.append(_trace("sse-0", 0.6, "sseerror", f"{tag} sse"))
    for i in range(N_CONCURRENT_RETRY):
        traces.append(_trace(f"r1c-{i}", 1.0, "fail503x1", f"{tag} r1 concurrent {i}"))

    cm, results = _dispatch(cfg, traces)
    by_id = {r.request_id: r for r in results}
    metrics_path = cm.get_output_paths()["performance_metrics"]
    with open(metrics_path, encoding="utf-8") as f:
        lines = [json.loads(line) for line in f if line.strip()]
    prompts = {t.prompt for t in traces}
    server = fake_server.records_for(prompts)
    return {"traces": traces, "by_id": by_id, "lines": lines, "server": server, "cm": cm, "tag": tag}


def _server_reqs(run, prompt):
    return [r for r in run["server"] if r["body"]["messages"][0]["content"] == prompt]


def test_every_request_gets_exactly_one_record(run):
    assert len(run["by_id"]) == len(run["traces"])
    assert sorted(line["request_id"] for line in run["lines"]) == sorted(t.request_id for t in run["traces"])


def test_output_record_fields_are_v1_fields_then_audit_fields(run):
    for line in run["lines"]:
        assert list(line.keys()) == V1_RESPONSE_FIELDS + AUDIT_FIELDS


def test_request_wire_format_matches_v1(run):
    t = next(t for t in run["traces"] if t.request_id == "ok-1")  # per-request max_output_tokens=4
    (req,) = _server_reqs(run, t.prompt)
    assert req["method"] == "POST" and req["path"] == "/v1/chat/completions"
    body = req["body"]
    # exactly v1's kwargs: no ignore_eos, temperature from config (unset -> None -> JSON null, as v1)
    assert body == {
        "model": "ok",
        "messages": [{"role": "user", "content": t.prompt}],
        "temperature": None,
        "stream": True,
        "stream_options": {"include_usage": True},
        "max_tokens": 4,
    }
    headers = req["headers"]
    assert headers["routing-strategy"] == "least-gpu-cache"
    assert headers["authorization"] == "Bearer dummy-key-for-local-gateway"
    assert headers["x-stainless-retry-count"] == "0"


def test_null_max_output_tokens_falls_back_to_model_config(run):
    t = next(t for t in run["traces"] if t.request_id == "ok-0")  # max_output_tokens=None
    (req,) = _server_reqs(run, t.prompt)
    assert req["body"]["max_tokens"] == 6  # model max_tokens in config


def test_normal_stream_metrics_follow_v1_formulas(run):
    for rid in ("ok-0", "ok-1", "stop-0"):
        r = run["by_id"][rid]
        assert r.success is True and r.error_message is None
        assert r.attempts == 1 and r.http_status == 200
        assert r.stream_interrupted is False and r.stream_error is None
        assert r.target_pod == "10.0.0.9:8000"
        assert r.e2e_latency == pytest.approx(r.end_time - r.start_time)
        assert r.ttft is not None and 0 < r.ttft < r.e2e_latency
        first_token_time = r.start_time + r.ttft
        assert r.tpot == pytest.approx((r.end_time - first_token_time) / r.output_tokens)
        assert r.total_tokens == r.input_tokens + r.output_tokens
    assert run["by_id"]["ok-0"].output_tokens == 6
    assert run["by_id"]["ok-1"].output_tokens == 4
    assert run["by_id"]["ok-1"].finish_reason == "length"
    assert run["by_id"]["stop-0"].finish_reason == "stop"


def test_retries_are_counted_and_logged(run):
    r1 = run["by_id"]["r1-0"]
    assert r1.success is True and r1.attempts == 2 and r1.http_status == 200
    assert [a["status"] for a in r1.attempt_log] == [503, 200]
    r2 = run["by_id"]["r2-0"]
    assert r2.success is True and r2.attempts == 3
    assert [a["status"] for a in r2.attempt_log] == [503, 503, 200]
    assert len(_server_reqs(run, next(t.prompt for t in run["traces"] if t.request_id == "r2-0"))) == 3
    # retries exhausted (max_retries=2 -> 3 attempts), v1 failure record
    r5 = run["by_id"]["r5-0"]
    assert r5.success is False and r5.attempts == 3 and r5.http_status == 503
    assert r5.ttft is None and r5.output_tokens == 0 and r5.error_message
    e500 = run["by_id"]["e500-0"]
    assert e500.success is False and e500.attempts == 3 and e500.http_status == 500


def test_attempt_attribution_is_per_request_under_concurrency(run):
    for i in range(N_CONCURRENT_RETRY):
        r = run["by_id"][f"r1c-{i}"]
        assert r.success is True
        assert r.attempts == 2, (r.request_id, r.attempt_log)
        assert [a["status"] for a in r.attempt_log] == [503, 200]


def test_mid_stream_disconnect_is_success_but_flagged(run):
    r = run["by_id"]["drop-0"]
    # v1 semantics preserved: exception during iteration is swallowed -> success=True
    assert r.success is True
    assert r.stream_interrupted is True and r.stream_error
    assert r.attempts == 1 and r.http_status == 200
    assert r.ttft is not None and r.finish_reason is None
    # no usage chunk arrived -> v1 records 0 output tokens and tpot None
    assert r.output_tokens == 0 and r.tpot is None


def test_sse_error_event_mid_stream_is_success_but_flagged(run):
    r = run["by_id"]["sse-0"]
    assert r.success is True and r.stream_interrupted is True
    assert "engine died" in r.stream_error


def test_read_stall_mid_stream_times_out_as_flagged_success(run):
    r = run["by_id"]["stall-0"]
    # httpx timeout is per-read inactivity (not total); mid-body stall -> ReadTimeout in iteration
    assert r.success is True and r.stream_interrupted is True
    assert "Timeout" in r.stream_error
    assert r.attempts == 1
    assert r.e2e_latency == pytest.approx(TIMEOUT_S, abs=0.6)


def test_timeout_before_headers_retries_then_fails(run):
    r = run["by_id"]["hang-0"]
    assert r.success is False
    assert r.attempts == 3 and [a["status"] for a in r.attempt_log] == [None, None, None]
    assert r.http_status is None
    assert "timed out" in r.error_message.lower()
    assert r.e2e_latency >= 3 * TIMEOUT_S


def test_requests_are_sent_on_trace_timestamps(run):
    ok = [run["by_id"][t.request_id] for t in run["traces"] if t.model_name == "ok"]
    offsets = [r.start_time - r.timestamp for r in ok]  # = base_time + scheduling error
    # v1 worker loop blocks its event loop in Queue.get(timeout=0.1) whenever its queue is empty,
    # so up to ~0.1s scheduling jitter is inherent to v1 (kept on purpose, see report)
    assert max(offsets) - min(offsets) < 0.3, offsets
    # ... and the same blocking delays every await between start_time and the bytes hitting the wire
    # (connect / write): measured ~0.1s p50, ~0.23s max on 76.  v1-faithful; bounded here.
    by_prompt = {t.prompt: t for t in run["traces"]}
    start_by_prompt = {t.prompt: run["by_id"][t.request_id].start_time for t in run["traces"]}
    ok_server = [r for r in run["server"] if r["body"]["model"] == "ok"]
    wire_lag = [r["t"] - start_by_prompt[r["body"]["messages"][0]["content"]] for r in ok_server]
    assert all(-0.01 < lag < 0.6 for lag in wire_lag), wire_lag
    arr_off = [r["t"] - by_prompt[r["body"]["messages"][0]["content"]].timestamp for r in ok_server]
    assert max(arr_off) - min(arr_off) < 0.8, arr_off
    late = run["by_id"]["ok-late"]
    assert late.success and late.start_time - late.timestamp == pytest.approx(min(offsets), abs=0.3)


def test_client_construction_matches_v1_sdk_defaults(tmp_path):
    """Our client (custom httpx client for hooks) must be wire-equivalent to v1's SDK-built one."""
    cfg = write_config(tmp_path / "c.yaml", tmp_path / "out")
    cm = ConfigManager(str(cfg))
    config = cm.load_config()
    config.gateway_endpoint = "http://gw.example:80"
    worker = WorkerProcess(0, config, None, None, None, None)
    ours = worker.create_client()

    # exactly what v1's create_client() built
    v1 = openai.AsyncOpenAI(api_key="dummy-key-for-local-gateway", base_url="http://gw.example:80/v1",
                            max_retries=2, timeout=300.0)
    v1 = v1.with_options(default_headers={"routing-strategy": "least-gpu-cache"})

    assert str(ours.base_url) == str(v1.base_url)
    assert ours.max_retries == v1.max_retries == 2
    assert ours.timeout == v1.timeout
    def _headers(c):  # openai.Omit sentinels compare by identity; normalise them
        return {k: ("<omit>" if isinstance(v, openai.Omit) else v) for k, v in c.default_headers.items()}
    assert _headers(ours) == _headers(v1)
    assert _headers(ours)["routing-strategy"] == "least-gpu-cache"
    assert ours._client.timeout == v1._client.timeout
    assert ours._client.follow_redirects == v1._client.follow_redirects
    assert ours._client.trust_env == v1._client.trust_env
    p_ours, p_v1 = ours._client._transport._pool, v1._client._transport._pool
    assert p_ours._max_connections == p_v1._max_connections
    assert p_ours._max_keepalive_connections == p_v1._max_keepalive_connections
    assert p_ours._keepalive_expiry == p_v1._keepalive_expiry
    assert ours._client.event_hooks["request"] and ours._client.event_hooks["response"]


def test_max_retries_is_a_parameter(tmp_path):
    cfg = write_config(tmp_path / "c.yaml", tmp_path / "out", max_retries=5)
    config = ConfigManager(str(cfg)).load_config()
    config.gateway_endpoint = "http://gw.example:80"
    assert WorkerProcess(0, config, None, None, None, None).create_client().max_retries == 5


def test_non_streaming_path_records_audit_fields(fake_server, tmp_path):
    tag = uuid.uuid4().hex[:8]
    cfg = write_config(tmp_path / "c.yaml", tmp_path / "out", enable_streaming=False, timeout=TIMEOUT_S)
    cfg.write_text(cfg.read_text(encoding="utf-8").replace("http://localhost:8888", fake_server.url),
                   encoding="utf-8")
    traces = [_trace("ns-ok", 0.2, "ok", f"{tag} ns ok", max_out=3),
              _trace("ns-r1", 0.2, "fail503x1", f"{tag} ns r1")]
    _, results = _dispatch(cfg, traces)
    by_id = {r.request_id: r for r in results}
    ok = by_id["ns-ok"]
    assert ok.success and ok.ttft is None and ok.tpot is None and ok.output_tokens == 3
    assert ok.attempts == 1 and ok.http_status == 200 and ok.finish_reason == "length"
    r1 = by_id["ns-r1"]
    assert r1.success and r1.attempts == 2 and [a["status"] for a in r1.attempt_log] == [503, 200]
    (req,) = [r for r in fake_server.records_for({f"{tag} ns ok"})]
    assert "stream" not in req["body"] and req["body"]["max_tokens"] == 3
