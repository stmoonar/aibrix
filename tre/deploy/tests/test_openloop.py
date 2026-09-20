from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts import openloop, r3_grid


# --------------------------------------------------------------------- guards


def _record(status=200, e2e=100.0, pool_wait=1.0, request_id="r0",
            error_body=None, error_headers=None):
    return {
        "request_id": request_id,
        "http_status": status,
        "e2e_ms": e2e,
        "ttft_ms": 20.0,
        "pool_wait_ms": pool_wait,
        "actual_send_ts_ms": 1_000,
        "prompt_tokens": 256,
        "completion_tokens": 128,
        "error_body": error_body,
        "error_headers": error_headers,
    }


def test_guard_accepts_a_clean_cell() -> None:
    guard = openloop.check_cell(
        "i256_o128_c60", scheduled=100, records=[_record() for _ in range(100)], p99_delay_ms=5.0
    )
    assert guard.ok
    assert guard.completed == 100 and guard.errors == 0


def test_guard_rejects_a_cell_that_sent_nothing() -> None:
    # The closed-loop worker swallows every send exception, so a misconfigured sender
    # produces an empty raw file and a silently-zero row. In open-loop mode that must be
    # a hard, named failure.
    guard = openloop.check_cell("i256_o128_c60", scheduled=42, records=[], p99_delay_ms=0.0)
    assert not guard.ok
    assert any("0 requests were sent" in issue for issue in guard.issues)
    with pytest.raises(openloop.CellGuardError) as excinfo:
        openloop.raise_on_guard(guard)
    assert "i256_o128_c60" in str(excinfo.value)


def test_guard_rejects_an_empty_schedule() -> None:
    guard = openloop.check_cell("i0_o0_c95", scheduled=0, records=[], p99_delay_ms=0.0)
    assert any("0 requests" in issue for issue in guard.issues)


def test_guard_rejects_all_failed_sends() -> None:
    records = [_record(status=0, e2e=None) for _ in range(20)]
    guard = openloop.check_cell("i256_o128_c60", scheduled=20, records=records, p99_delay_ms=1.0)
    assert not guard.ok
    assert any("0/20 requests completed" in issue for issue in guard.issues)


def test_guard_rejects_an_error_rate_above_threshold() -> None:
    # 500 with an engine body: the request reached vLLM and failed there, so it counts
    # against the model budget. A bare 503 would not - that is a gateway shed.
    records = [_record() for _ in range(90)] + [
        _record(status=500, e2e=None, error_body='{"error": "engine failure"}') for _ in range(10)
    ]
    guard = openloop.check_cell(
        "i256_o128_c60", scheduled=100, records=records, p99_delay_ms=1.0,
        max_model_error_rate=0.05,
    )
    assert any("error rate" in issue for issue in guard.issues)


def test_guard_rejects_schedule_slip_and_pool_starvation() -> None:
    # Either of these means the driver, not the engine, limited the offered load: the
    # "open loop" silently became a closed loop and the cell measures nothing useful.
    slipped = openloop.check_cell(
        "i256_o128_c120", scheduled=10, records=[_record() for _ in range(10)], p99_delay_ms=900.0
    )
    assert any("dispatch delay" in issue for issue in slipped.issues)

    starved = openloop.check_cell(
        "i256_o128_c120",
        scheduled=10,
        records=[_record(pool_wait=4000.0) for _ in range(10)],
        p99_delay_ms=1.0,
    )
    assert any("pool wait" in issue for issue in starved.issues)


# ----------------------------------------------------------------- pod gauges


VLLM_METRICS = """
# HELP vllm:num_requests_running Number of requests currently running.
# TYPE vllm:num_requests_running gauge
vllm:num_requests_running{model_name="dsqwen-7b"} 12.0
# HELP vllm:num_requests_waiting Number of requests waiting to be processed.
# TYPE vllm:num_requests_waiting gauge
vllm:num_requests_waiting{model_name="dsqwen-7b"} 37.0
vllm:gpu_cache_usage_perc{model_name="dsqwen-7b"} 0.61
"""


def test_parse_pod_gauges_reads_waiting_and_running() -> None:
    gauges = openloop.parse_pod_gauges(VLLM_METRICS)
    assert gauges["waiting"] == 37.0
    assert gauges["running"] == 12.0
    # num_requests_swapped is absent on the V1 engine: genuinely zero, never an exception.
    assert gauges["swapping"] == 0.0


def test_parse_pod_gauges_sums_label_sets() -> None:
    text = (
        'vllm:num_requests_waiting{model_name="m",engine="0"} 3.0\n'
        'vllm:num_requests_waiting{model_name="m",engine="1"} 4.0\n'
    )
    assert openloop.parse_pod_gauges(text)["waiting"] == 7.0


def test_pod_sampler_sums_across_pods_and_survives_a_dead_pod() -> None:
    def fetch(url: str) -> str:
        if "dead" in url:
            raise OSError("connection refused")
        return VLLM_METRICS

    sampler = openloop.make_pod_metrics_sampler(
        ["http://a:8000/metrics", "http://b:8000/metrics", "http://dead:8000/metrics"],
        fetch=fetch,
    )
    snap = sampler(0)
    assert snap["waiting"] == 74.0  # two live pods
    assert snap["scrape_errors"] == 1.0
    assert snap["pods_scraped"] == 2.0


def test_pod_sampler_requires_at_least_one_endpoint() -> None:
    with pytest.raises(ValueError):
        openloop.make_pod_metrics_sampler([])


# ------------------------------------------------------------- grid alignment


def test_mark_live_grid_tags_one_sample_per_10s_bucket() -> None:
    samples = [{"ts_ms": 100_000 + 1000 * k, "waiting": 0.0} for k in range(25)]
    tagged = openloop.mark_live_grid(samples)
    on_grid = [s for s in tagged if s["on_live_grid"]]
    # 25 s of 1 Hz samples straddle three 10 s buckets.
    assert len(on_grid) == 3
    assert [s["ts_ms"] for s in on_grid] == [100_000, 110_000, 120_000]


def test_windows_observing_exposes_the_aliasing_the_bursts_target() -> None:
    # A 3 s waiting-queue spike sitting between two live grid ticks: visible at 1 Hz,
    # invisible at the gateway's 10 s cadence. This is exactly the failure mode that left
    # avg_waiting == 0 in ~100 % of the 14b calibration windows.
    samples = []
    for k in range(65):  # two full 30 s windows (the last partial window is dropped)
        ts = 0 + 1000 * k
        waiting = 9.0 if 32 <= k < 35 else 0.0
        samples.append({"ts_ms": ts, "waiting": waiting})
    tagged = openloop.mark_live_grid(samples)

    truth = openloop.windows_observing(tagged, window_ms=30_000)
    aliased = openloop.windows_observing(tagged, window_ms=30_000, grid_only=True)
    assert truth[0] == 1 and truth[1] == 2
    assert aliased[0] == 0  # the controller would have seen an empty queue
    assert aliased[1] == truth[1]


# -------------------------------------------------------------------- driver


class _FakeStream:
    """Deterministic stand-in for the SSE seam: never touches the network."""

    def __init__(self) -> None:
        self.calls: list[bytes] = []

    def __call__(self, url, headers, body, timeout):
        from tre_replayer.engine.http_sender import StreamResult

        self.calls.append(body)
        payload = json.loads(body)
        prompt = payload["prompt"]
        n_in = len(prompt) if isinstance(prompt, list) else len(prompt.split())
        return StreamResult(200, 11.0, 40.0, n_in, payload["max_tokens"])


def test_drive_cell_schedule_writes_the_r3_raw_schema(tmp_path: Path) -> None:
    from tre_replayer.engine.schedule import RpsSegment

    seg = RpsSegment("dsqwen-7b", 0.0, 0.4, 40.0, input_tokens=256, max_output_tokens=128)
    raw = tmp_path / "i256_o128_c60.jsonl"
    instant = tmp_path / "i256_o128_c60.instant.jsonl"
    stream = _FakeStream()
    samples: list[int] = []

    def sampler(now_ms: int) -> dict:
        samples.append(now_ms)
        return {"waiting": 2.0, "running": 1.0, "swapping": 0.0}

    start_ms, end_ms, guard = openloop.drive_cell_schedule(
        "http://gw/v1/completions", "dsqwen-7b", "i256_o128_c60", [seg],
        raw_path=raw, instant_path=instant,
        instant_sampler=sampler, instant_interval_s=0.05,
        stream_call=stream, prompt_mode="token_ids",
    )
    assert end_ms >= start_ms
    assert guard.ok, guard.issues
    assert guard.sent == guard.scheduled > 0

    rows = [json.loads(line) for line in raw.read_text().splitlines()]
    assert len(rows) == guard.sent
    # identical schema to the closed-loop path -> rewindow_from_raw needs no change
    assert set(rows[0]) == set(r3_grid.RAW_COLUMNS)
    assert rows[0]["cell_id"] == "i256_o128_c60"
    assert rows[0]["input_tokens"] == 256 and rows[0]["output_tokens"] == 128

    inst = [json.loads(line) for line in instant.read_text().splitlines()]
    assert inst and all("on_live_grid" in s for s in inst)
    assert all(s["waiting"] == 2.0 for s in inst)


def test_drive_cell_schedule_sends_a_unique_prompt_per_request(tmp_path: Path) -> None:
    # A shared prompt is served from the prefix cache and makes prefill free; the whole
    # capacity surface then inverts (the 14b prior pathology).
    from tre_replayer.engine.schedule import RpsSegment

    seg = RpsSegment("dsqwen-7b", 0.0, 0.3, 50.0, input_tokens=64, max_output_tokens=16)
    stream = _FakeStream()
    openloop.drive_cell_schedule(
        "http://gw/v1/completions", "dsqwen-7b", "i64_o16_c60", [seg], stream_call=stream
    )
    prompts = [tuple(json.loads(b)["prompt"]) for b in stream.calls]
    assert len(prompts) > 3
    assert len(set(prompts)) == len(prompts)


def test_drive_cell_schedule_superposes_overlapping_segments() -> None:
    # The bursts primitive relies on this: a spike segment laid on top of the base rate
    # must add requests, not replace them.
    from tre_replayer.engine.schedule import RpsSegment

    base = RpsSegment("m", 0.0, 0.4, 20.0, input_tokens=32, max_output_tokens=8)
    spike = RpsSegment("m", 0.1, 0.2, 200.0, input_tokens=32, max_output_tokens=8)
    stream = _FakeStream()
    _s, _e, only_base = openloop.drive_cell_schedule(
        "http://gw", "m", "i32_o8_c60", [base], stream_call=stream, prompt_mode="token_ids"
    )
    stream2 = _FakeStream()
    _s, _e, both = openloop.drive_cell_schedule(
        "http://gw", "m", "i32_o8_c60", [base, spike], stream_call=stream2, prompt_mode="token_ids"
    )
    assert both.sent > only_base.sent


def test_drive_cell_schedule_ignores_other_models_in_the_trace() -> None:
    from tre_replayer.engine.schedule import RpsSegment

    mine = RpsSegment("dsqwen-7b", 0.0, 0.2, 30.0, input_tokens=32, max_output_tokens=8)
    theirs = RpsSegment("dsllama-8b", 0.0, 0.2, 30.0, input_tokens=32, max_output_tokens=8)
    stream = _FakeStream()
    _s, _e, guard = openloop.drive_cell_schedule(
        "http://gw", "dsqwen-7b", "i32_o8_c60", [mine, theirs], stream_call=stream
    )
    assert guard.ok
    assert all(json.loads(b)["model"] == "dsqwen-7b" for b in stream.calls)


# --------------------------------------------------------------- failure classifier

#: The verbatim Envoy circuit-breaker response measured against the live gateway on
#: 2026-09-20 at in-flight 321. It carries NO x-envoy-* header, so a classifier keyed on
#: those headers alone would mis-attribute every shed in the campaign to the model.
MEASURED_SHED = {
    "http_status": 503,
    "e2e_ms": 3.0,
    "error": "HTTP 503",
    "error_body": (
        "upstream connect error or disconnect/reset before headers. "
        "reset reason: overflow"
    ),
    "error_headers": {
        "content-length": "81",
        "content-type": "text/plain",
        "date": "Sun, 20 Sep 2026 12:13:38 GMT",
        "connection": "close",
    },
}


def test_classifier_attributes_the_measured_shed_to_the_proxy() -> None:
    assert openloop.classify_failure(MEASURED_SHED) == openloop.FAILURE_PROXY


def test_classifier_attributes_an_engine_json_error_to_the_model() -> None:
    record = {
        "http_status": 503,
        "e2e_ms": 12.0,
        "error_body": '{"object": "error", "message": "no free blocks"}',
        "error_headers": {"content-type": "application/json"},
    }
    assert openloop.classify_failure(record) == openloop.FAILURE_MODEL


def test_classifier_attributes_a_timeout_to_the_model() -> None:
    # vLLM queues rather than shedding, so a request that never answered was waiting on
    # the engine, not rejected by the gateway.
    record = {"http_status": 0, "e2e_ms": None, "error": "TimeoutError"}
    assert openloop.classify_failure(record) == openloop.FAILURE_MODEL


def test_classifier_attributes_an_unknown_failure_to_the_model() -> None:
    # Attributing the unknown to the proxy would exempt it from the error budget.
    record = {"http_status": 418, "e2e_ms": 1.0, "error_body": "teapot"}
    assert openloop.classify_failure(record) == openloop.FAILURE_MODEL


def test_classifier_honours_an_explicit_envoy_marker_header() -> None:
    record = {"http_status": 429, "e2e_ms": 1.0, "error_headers": {"x-envoy-overloaded": "true"}}
    assert openloop.classify_failure(record) == openloop.FAILURE_PROXY


def test_served_request_is_not_a_failure() -> None:
    assert openloop.classify_failure(_record()) == openloop.FAILURE_NONE


# ------------------------------------------------------------------ split error budget


def test_proxy_sheds_do_not_fail_the_cell() -> None:
    # Half the cell shed by the gateway, and the cell still passes on the error budget:
    # the shed says the campaign hit the admission ceiling, not that the model failed.
    records = [_record() for _ in range(50)] + [dict(MEASURED_SHED) for _ in range(50)]
    guard = openloop.check_cell(
        "i256_o128_c60", scheduled=100, records=records, p99_delay_ms=1.0,
    )
    assert guard.proxy_errors == 50
    assert guard.model_errors == 0
    assert not any("error rate" in issue for issue in guard.issues)


def test_truncated_cell_passes_with_enough_slo_evidence() -> None:
    records = [_record() for _ in range(40)] + [dict(MEASURED_SHED)]
    guard = openloop.check_cell(
        "i256_o128_c120", scheduled=100, records=records, p99_delay_ms=1.0,
        truncated=True, truncated_at_offset_s=210.0, truncated_at_ts_ms=1700,
        censored=59, slo_windows=4,
    )
    assert guard.truncated
    assert guard.censored == 59
    assert guard.ok, guard.issues


def test_truncated_cell_fails_without_enough_slo_evidence() -> None:
    records = [_record() for _ in range(40)] + [dict(MEASURED_SHED)]
    guard = openloop.check_cell(
        "i256_o128_c120", scheduled=100, records=records, p99_delay_ms=1.0,
        truncated=True, truncated_at_offset_s=12.0, truncated_at_ts_ms=1700,
        censored=59, slo_windows=1,
    )
    assert not guard.ok
    assert any(openloop.TRUNCATION_EVIDENCE_ISSUE in issue for issue in guard.issues)


def test_censored_requests_are_not_counted_as_under_delivery() -> None:
    # Without the censored allowance the truncation itself would fail the cell for
    # "only 41/100 scheduled requests were sent".
    records = [_record() for _ in range(41)]
    guard = openloop.check_cell(
        "i256_o128_c120", scheduled=100, records=records, p99_delay_ms=1.0,
        truncated=True, censored=59, slo_windows=3,
    )
    assert guard.ok, guard.issues


def test_with_slo_windows_replaces_rather_than_accumulates_its_verdict() -> None:
    guard = openloop.check_cell(
        "i256_o128_c120", scheduled=10, records=[_record() for _ in range(10)],
        p99_delay_ms=1.0, truncated=True, slo_windows=0,
    )
    assert not guard.ok
    revised = guard.with_slo_windows(5)
    assert revised.ok, revised.issues
    assert revised.slo_windows == 5


# ---------------------------------------------------------------------- truncation


class _FakeSender:
    """Records what it was asked to send, and answers with a scripted status."""

    def __init__(self, shed_after: int) -> None:
        self.records: list = []
        self._shed_after = shed_after
        self.sent = 0

    async def __call__(self, request, scheduled_ts, actual_ts) -> None:
        self.sent += 1
        if self.sent > self._shed_after:
            self.records.append(dict(MEASURED_SHED, request_id=request.request_id))
        else:
            self.records.append(_record(request_id=request.request_id))


class _Request:
    def __init__(self, request_id: str, offset_s: float) -> None:
        self.request_id = request_id
        self.scheduled_offset_s = offset_s


def _drive(wrapper, offsets) -> None:
    import asyncio

    async def run() -> None:
        for index, offset in enumerate(offsets):
            await wrapper(_Request(f"r{index}", offset), 0.0, 0.0)

    asyncio.run(run())


def test_truncation_drops_pre_drain_requests_and_keeps_the_drain() -> None:
    sender = _FakeSender(shed_after=2)
    wrapper = openloop.TruncateOnProxyShed(sender, drain_start_s=100.0)
    _drive(wrapper, [0.0, 10.0, 20.0, 30.0, 40.0, 100.0, 110.0])

    assert wrapper.truncated
    assert wrapper.truncated_at_offset_s == 20.0
    # r0..r2 sent, r3/r4 censored (before the drain), r5/r6 sent (the drain itself).
    assert wrapper.censored == 2
    assert [r["request_id"] for r in sender.records] == ["r0", "r1", "r2", "r5", "r6"]


def test_truncation_without_a_drain_segment_stops_sending() -> None:
    sender = _FakeSender(shed_after=1)
    wrapper = openloop.TruncateOnProxyShed(sender, drain_start_s=None)
    _drive(wrapper, [0.0, 10.0, 20.0, 30.0])

    assert wrapper.truncated
    assert wrapper.censored == 2
    assert [r["request_id"] for r in sender.records] == ["r0", "r1"]


def test_no_truncation_when_every_request_is_served() -> None:
    sender = _FakeSender(shed_after=99)
    wrapper = openloop.TruncateOnProxyShed(sender, drain_start_s=100.0)
    _drive(wrapper, [0.0, 10.0, 20.0])

    assert not wrapper.truncated
    assert wrapper.censored == 0
    assert len(sender.records) == 3


def test_failure_signature_keeps_the_body_verbatim() -> None:
    signature = openloop.failure_signature(MEASURED_SHED)
    assert signature["failure_class"] == openloop.FAILURE_PROXY
    assert signature["error_body"] == MEASURED_SHED["error_body"]
    assert signature["error_headers"]["content-type"] == "text/plain"


def test_routing_balance_counts_requests_per_pod() -> None:
    records = [{"target_pod": "pod-a"}] * 6 + [{"target_pod": "pod-b"}] * 2
    balance = openloop.routing_balance(records)
    assert balance["pods"] == 2
    assert balance["per_pod"] == {"pod-a": 6, "pod-b": 2}
    assert balance["imbalance_ratio"] == 3.0
    assert balance["attributed"] == 8 and balance["unattributed"] == 0


def test_routing_balance_reports_unattributed_rather_than_claiming_balance() -> None:
    """The campaign's serving path names no pod. That must read as 'not measured', not
    as 'perfectly balanced' - the whole point of the check is to catch a hidden skew."""
    balance = openloop.routing_balance([{"target_pod": None}, {}, {"target_pod": ""}])
    assert balance["pods"] == 0
    assert balance["attributed"] == 0 and balance["unattributed"] == 3
    assert balance["imbalance_ratio"] is None


def test_cell_guard_artifact_surfaces_the_routing_balance() -> None:
    records = [_ok_record(pod="pod-a") for _ in range(4)] + [_ok_record(pod="pod-b")]
    guard = openloop.check_cell("i128_o128_c4", scheduled=5, records=records, p99_delay_ms=1.0)
    assert guard.ok, guard.issues
    artifact = guard.as_dict()
    assert artifact["routing_balance"]["per_pod"] == {"pod-a": 4, "pod-b": 1}
    assert artifact["routing_balance"]["imbalance_ratio"] == 4.0


def test_cell_guard_fails_an_imbalanced_cell_only_when_a_ceiling_is_set() -> None:
    """An aggregate capacity signal averages over pods, so a skewed router hides one
    pod's exploding p95 inside a healthy-looking Z. Off by default because nobody has
    calibrated a threshold yet - reporting it is the deliverable, gating on it is opt-in."""
    records = [_ok_record(pod="pod-a") for _ in range(9)] + [_ok_record(pod="pod-b")]
    assert openloop.check_cell("c", scheduled=10, records=records, p99_delay_ms=1.0).ok
    gated = openloop.check_cell(
        "c", scheduled=10, records=records, p99_delay_ms=1.0, max_routing_imbalance=3.0
    )
    assert not gated.ok
    assert any("imbalance" in issue for issue in gated.issues)


def test_cell_guard_does_not_gate_on_a_single_pod() -> None:
    records = [_ok_record(pod="pod-a") for _ in range(10)]
    guard = openloop.check_cell(
        "c", scheduled=10, records=records, p99_delay_ms=1.0, max_routing_imbalance=1.5
    )
    assert guard.ok, guard.issues


def _ok_record(pod: str | None = None) -> dict:
    return {
        "http_status": 200, "e2e_ms": 120.0, "error": None, "pool_wait_ms": 0.0,
        "target_pod": pod,
    }
