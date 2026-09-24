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


def test_kv_cache_usage_prefers_the_current_name_and_falls_back_to_the_deprecated_one() -> None:
    both = ('vllm:kv_cache_usage_perc{model_name="m"} 0.42\n'
            'vllm:gpu_cache_usage_perc{model_name="m"} 0.99\n')
    assert openloop.parse_pod_kv_cache_usage(both) == pytest.approx(0.42)
    # only the deprecated name (VLLM_METRICS): the fallback
    assert openloop.parse_pod_kv_cache_usage(VLLM_METRICS) == pytest.approx(0.61)
    # several label sets of one pod are averaged, not summed
    engines = ('vllm:kv_cache_usage_perc{model_name="m",engine="0"} 0.2\n'
               'vllm:kv_cache_usage_perc{model_name="m",engine="1"} 0.4\n')
    assert openloop.parse_pod_kv_cache_usage(engines) == pytest.approx(0.3)
    assert openloop.parse_pod_kv_cache_usage('vllm:num_requests_running{m="m"} 1\n') is None
    assert openloop.parse_pod_kv_cache_usage("# vllm:kv_cache_usage_perc 0.5\n") is None
    # the queue gauges are unchanged by it
    assert openloop.parse_pod_gauges(both) == {"waiting": 0.0, "running": 0.0, "swapping": 0.0}


def test_pod_sampler_reports_the_mean_kv_cache_usage_over_the_pods_that_have_it() -> None:
    bodies = {
        "a": VLLM_METRICS,                                                   # 0.61 (fallback)
        "b": VLLM_METRICS.replace("vllm:gpu_cache_usage_perc", "vllm:kv_cache_usage_perc")
                         .replace("0.61", "0.21"),                            # 0.21 (current)
        "c": 'vllm:num_requests_waiting{model_name="m"} 5.0\n',             # no KV gauge
    }

    def fetch(url: str) -> str:
        if "dead" in url:
            raise OSError("connection refused")
        return bodies[url.split("//")[1].split(":")[0]]

    urls = ["http://a:8000/metrics", "http://b:8000/metrics", "http://c:8000/metrics",
            "http://dead:8000/metrics"]
    snap = openloop.make_pod_metrics_sampler(urls, fetch=fetch)(0)
    assert snap["kv_cache_usage"] == pytest.approx((0.61 + 0.21) / 2)   # mean, not sum
    assert snap["waiting"] == 37.0 + 37.0 + 5.0 and snap["running"] == 24.0   # still summed
    assert snap["pods_scraped"] == 3.0 and snap["scrape_errors"] == 1.0
    none = openloop.make_pod_metrics_sampler(["http://c:8000/metrics"], fetch=fetch)(0)
    assert none["kv_cache_usage"] is None and none["waiting"] == 5.0


def test_the_sidecar_keeps_a_missing_kv_cache_usage_as_none() -> None:
    import time as _time

    snaps = iter([{"waiting": 1, "running": 2, "swapping": 0, "kv_cache_usage": None,
                   "pods_scraped": 1, "scrape_errors": 0}])
    sidecar = openloop._Sidecar(sampler=lambda _ts: next(snaps, None), interval_s=0.01,
                                now_ms=lambda: 10_000)
    sidecar.start()
    deadline = _time.time() + 2.0
    while not sidecar.samples and _time.time() < deadline:
        _time.sleep(0.01)
    rows = sidecar.stop()
    assert rows[0]["kv_cache_usage"] is None and rows[0]["waiting"] == 1.0
    assert rows[0]["running"] == 2.0 and rows[0]["on_live_grid"] is True


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
    # the closed-loop schema plus what the open-loop driver knows about each request
    assert set(rows[0]) == set(r3_grid.RAW_COLUMNS) | set(r3_grid.RAW_REQUEST_COLUMNS)
    assert rows[0]["outcome"] == "ok" and rows[0]["proxy_reason"] is None
    assert rows[0]["scheduled_send_ts_ms"] <= rows[0]["send_ts_ms"]
    assert rows[0]["in_flight_at_send"] >= 1
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
    assert openloop.classify_failure(MEASURED_SHED) == openloop.FAILURE_ADMISSION_OVERFLOW


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
    assert openloop.classify_failure(record) == openloop.FAILURE_ADMISSION_OVERFLOW


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
    assert signature["failure_class"] == openloop.FAILURE_ADMISSION_OVERFLOW
    # The wording the split was decided on travels with the evidence.
    assert signature["proxy_reason"] == "overflow"
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


# ==================================================================== failure handling
#
# One test per void rule. A void rule that silently stops firing fails nothing - it just
# lets a polluted cell into the fit, and theta moves without anyone seeing why.


def _served(**over):
    record = {
        "request_id": "r", "http_status": 200, "e2e_ms": 120.0, "ttft_ms": 100.0,
        "tpot_ms": 10.0, "completion_tokens": 8, "actual_send_ts_ms": 1_000,
        "pool_wait_ms": 0.0, "client_timeout": False, "in_flight_at_send": 3,
    }
    record.update(over)
    return record


def _shed(**over):
    return _served(
        http_status=503, e2e_ms=5.0, ttft_ms=None, tpot_ms=None,
        error_body="upstream connect error or disconnect/reset before headers. "
                   "reset reason: overflow",
        error_headers={"content-type": "text/plain"},
        **over,
    )


def _model_error(**over):
    return _served(
        http_status=500, e2e_ms=30.0, ttft_ms=None, tpot_ms=None,
        error_body='{"error": "engine died"}', error_headers={"content-type": "application/json"},
        **over,
    )


def _timeout(**over):
    return _served(
        http_status=0, e2e_ms=None, ttft_ms=None, tpot_ms=None,
        error="TimeoutError", client_timeout=True, **over,
    )


# ------------------------------------------------------------------ classification


def test_every_request_lands_in_exactly_one_of_the_four_outcomes() -> None:
    records = [_served(), _served(), _shed(), _model_error(), _timeout()]
    outcomes = openloop.count_outcomes(records)
    assert (outcomes.ok, outcomes.shed, outcomes.model_error, outcomes.client_timeout) == (
        2, 1, 1, 1
    )
    assert outcomes.ok + outcomes.shed + outcomes.model_error + outcomes.client_timeout == len(records)
    # offered is everything the driver emitted; admitted is everything the gateway let through
    assert outcomes.offered == 5
    assert outcomes.admitted == 4
    assert outcomes.completed == 2


def test_a_client_timeout_is_its_own_class_and_no_ones_error_budget() -> None:
    # Nothing is known about what the engine did with it, so charging it to the model
    # would read as an engine fault and charging it to the gateway as a shed.
    assert openloop.classify_failure(_timeout()) == openloop.FAILURE_CLIENT_TIMEOUT
    served, model_errors, proxy_errors = openloop.count_failures([_timeout()] * 4)
    assert (served, model_errors, proxy_errors) == (0, 0, 0)
    guard = openloop.check_cell(
        "c", scheduled=4, records=[_timeout()] * 3 + [_served()], p99_delay_ms=1.0,
    )
    assert guard.client_timeouts == 3
    assert openloop.VOID_MODEL_ERRORS not in guard.void_reasons


def test_the_sender_records_how_loaded_the_path_was_when_a_request_left() -> None:
    # The one quantity nothing downstream can reconstruct: the raw log has send and done
    # timestamps but not the driver's own view of concurrency at the instant of the send.
    signature = openloop.failure_signature(_shed(in_flight_at_send=311))
    assert signature["in_flight_at_send"] == 311
    assert signature["outcome"] == "shed"


# ------------------------------------------------------------------ void: shed


def test_any_shed_voids_a_calibration_cell() -> None:
    guard = openloop.check_cell(
        "c", scheduled=100, records=[_served()] * 99 + [_shed()], p99_delay_ms=1.0,
        shed_policy=openloop.SHED_POLICY_VOID,
    )
    assert guard.voided and openloop.VOID_SHED in guard.void_reasons
    assert not guard.ok


def test_a_voided_shed_cannot_be_rescued_by_collecting_enough_windows() -> None:
    # The "keep it if it already had >= 3 SLO windows" escape is exactly wrong here: the
    # windows before a shed are the HEALTHY ones, so keeping them biases theta upwards.
    guard = openloop.check_cell(
        "c", scheduled=10, records=[_served()] * 9 + [_shed()], p99_delay_ms=1.0,
        shed_policy=openloop.SHED_POLICY_VOID, truncated=True, truncated_at_offset_s=12.0,
    )
    rescued = guard.with_slo_windows(99)
    assert rescued.voided and openloop.VOID_SHED in rescued.void_reasons


def test_the_replay_policy_still_truncates_instead_of_voiding() -> None:
    guard = openloop.check_cell(
        "c", scheduled=10, records=[_served()] * 9 + [_shed()], p99_delay_ms=1.0,
        shed_policy=openloop.SHED_POLICY_TRUNCATE, truncated=True,
    )
    assert not guard.voided
    assert guard.proxy_errors == 1


def test_an_unknown_shed_policy_is_a_loud_failure() -> None:
    with pytest.raises(ValueError, match="unknown shed policy"):
        openloop.check_cell("c", scheduled=1, records=[_served()], p99_delay_ms=1.0,
                            shed_policy="whatever")


# ------------------------------------------------------------------ void: dispatch delay


def test_a_dispatch_delay_over_50ms_voids_a_calibration_cell() -> None:
    # Above this the generator is not keeping its schedule, so the cell did not offer the
    # load it is indexed by and is not an open loop at all.
    ok = openloop.check_cell(
        "c", scheduled=10, records=[_served()] * 10, p99_delay_ms=49.0,
        max_p99_delay_ms=openloop.CALIBRATION_MAX_P99_DELAY_MS,
    )
    assert not ok.voided and ok.ok

    late = openloop.check_cell(
        "c", scheduled=10, records=[_served()] * 10, p99_delay_ms=51.0,
        max_p99_delay_ms=openloop.CALIBRATION_MAX_P99_DELAY_MS,
    )
    assert late.voided and openloop.VOID_DISPATCH_DELAY in late.void_reasons
    assert openloop.CALIBRATION_MAX_P99_DELAY_MS == 50.0
    assert openloop.CALIBRATION_MAX_P99_DELAY_MS < openloop.DEFAULT_MAX_P99_DELAY_MS


# ------------------------------------------------------------------ void: model errors


def test_a_model_error_rate_over_five_percent_voids_the_cell() -> None:
    under = openloop.check_cell(
        "c", scheduled=100, records=[_served()] * 96 + [_model_error()] * 4,
        p99_delay_ms=1.0,
    )
    assert not under.voided

    over = openloop.check_cell(
        "c", scheduled=100, records=[_served()] * 93 + [_model_error()] * 7,
        p99_delay_ms=1.0,
    )
    assert over.voided and openloop.VOID_MODEL_ERRORS in over.void_reasons


def test_a_window_holding_a_model_error_is_marked_violating_and_kept() -> None:
    # A failed request contributes no latency sample, so a window whose slowest work all
    # errored out otherwise shows a comfortable p95 and is scored as healthy.
    rows = [
        {"window_start_ms": 0, "window_end_ms": 1000, "p95_ttft_client_ms": 50.0},
        {"window_start_ms": 1000, "window_end_ms": 2000, "p95_ttft_client_ms": 50.0},
    ]
    records = [_model_error(actual_send_ts_ms=1500), _served(actual_send_ts_ms=10)]
    marked = openloop.mark_unserved_request_windows(rows, records)
    assert len(marked) == 2  # kept, not dropped
    assert marked[0]["model_errors"] == 0 and _label(marked[0]) == "healthy"
    assert marked[1]["model_errors"] == 1 and _label(marked[1]) == "violated"


def _label(row: dict) -> str:
    from tre_common import slo_labels

    return slo_labels.window_slo_label(row, {"ttft_p95": 500.0})


def test_a_window_holding_a_client_timeout_is_marked_violating_and_kept() -> None:
    # The client gave up (>= 30 s): far beyond TTFT + (n-1) x TPOT SLO for every shape,
    # and without the count the slowest requests of the window vanish from its p95.
    rows = [{"window_start_ms": 0, "window_end_ms": 1000, "p95_ttft_client_ms": 50.0}]
    timed_out = {"actual_send_ts_ms": 500, "http_status": 0, "client_timeout": True,
                 "e2e_ms": 30100.0}
    marked = openloop.mark_unserved_request_windows(rows, [timed_out])
    assert marked[0]["client_timeouts"] == 1 and _label(marked[0]) == "violated"


def test_an_admission_overflow_does_not_mark_a_window_violating() -> None:
    # It never reached the engine, so it says nothing about the engine's health - and
    # the shed policy has already decided what happens to the cell as a whole.
    rows = [{"window_start_ms": 0, "window_end_ms": 1000, "p95_ttft_client_ms": 50.0}]
    marked = openloop.mark_unserved_request_windows(rows, [_shed(actual_send_ts_ms=500)])
    assert marked[0]["model_errors"] == 0 and _label(marked[0]) == "healthy"


# ------------------------------------------------------------------ void: sentinel


ENVOY_STATS = """
cluster.tre-v2-dsqwen-7b.upstream_rq_pending_overflow: 12
cluster.tre-v2-dsllama-8b.upstream_rq_pending_overflow: 4
cluster.tre-v2-dsqwen-7b.upstream_rq_total: 99999
"""


def test_the_overflow_sentinel_reads_envoys_own_account_of_refusing_work() -> None:
    assert openloop.parse_envoy_counters(ENVOY_STATS, openloop.PENDING_OVERFLOW_COUNTER) == 16
    assert openloop.parse_envoy_counters(
        ENVOY_STATS, openloop.PENDING_OVERFLOW_COUNTER, cluster_filter="dsqwen-7b"
    ) == 12


def test_a_moving_overflow_counter_voids_the_run() -> None:
    bodies = iter([ENVOY_STATS, ENVOY_STATS.replace(": 12", ": 19")])
    sentinel = openloop.PendingOverflowSentinel(read=lambda: next(bodies))
    assert sentinel.start() == 16
    delta = sentinel.delta()
    assert delta == 7

    guard = openloop.check_cell(
        "c", scheduled=10, records=[_served()] * 10, p99_delay_ms=1.0,
        pending_overflow_delta=delta,
    )
    assert guard.voided and openloop.VOID_PENDING_OVERFLOW in guard.void_reasons


def test_a_still_overflow_counter_leaves_the_cell_alone() -> None:
    sentinel = openloop.PendingOverflowSentinel(read=lambda: ENVOY_STATS)
    sentinel.start()
    assert sentinel.delta() == 0
    guard = openloop.check_cell(
        "c", scheduled=10, records=[_served()] * 10, p99_delay_ms=1.0,
        pending_overflow_delta=0,
    )
    assert not guard.voided


def test_an_unreadable_sentinel_reports_not_measured_rather_than_clean() -> None:
    def boom() -> str:
        raise OSError("admin port refused")

    sentinel = openloop.PendingOverflowSentinel(read=boom)
    assert sentinel.start() is None
    assert sentinel.delta() is None
    guard = openloop.check_cell(
        "c", scheduled=1, records=[_served()], p99_delay_ms=1.0,
        pending_overflow_delta=None,
    )
    assert guard.pending_overflow_delta is None
    assert not guard.voided


# ------------------------------------------------------------------ goodput


def test_goodput_divides_by_offered_so_a_rejection_is_a_loss() -> None:
    records = [_served()] * 6 + [_shed()] * 2 + [_model_error()] + [_timeout()]
    result = openloop.goodput(records, ttft_slo_ms=500.0, tpot_slo_ms=75.0)
    assert result.offered == 10 and result.admitted == 8 and result.completed == 6
    assert result.good == 6
    assert result.goodput == pytest.approx(0.6)
    # dividing by admitted instead would flatter a system that refuses its way to health
    assert result.good / result.admitted == pytest.approx(0.75)


def test_a_served_request_that_missed_an_slo_is_not_good() -> None:
    records = [_served(ttft_ms=900.0), _served(tpot_ms=200.0), _served()]
    result = openloop.goodput(records, ttft_slo_ms=500.0, tpot_slo_ms=75.0)
    assert result.completed == 3 and result.good == 1
    assert result.goodput == pytest.approx(1 / 3)


def test_a_served_request_with_no_latency_evidence_is_not_scored_as_healthy() -> None:
    record = _served(ttft_ms=None, tpot_ms=None, completion_tokens=None)
    assert not openloop.request_meets_slo(record, ttft_slo_ms=500.0, tpot_slo_ms=75.0)


def test_the_guard_carries_the_goodput_when_the_slo_is_supplied() -> None:
    guard = openloop.check_cell(
        "c", scheduled=4, records=[_served()] * 3 + [_shed()], p99_delay_ms=1.0,
        ttft_slo_ms=500.0, tpot_slo_ms=75.0, shed_policy=openloop.SHED_POLICY_TRUNCATE,
    )
    assert guard.goodput["offered"] == 4
    assert guard.goodput["goodput"] == pytest.approx(0.75)
    assert guard.outcomes["shed"] == 1


# ------------------------------------------- the deadline is on-wire, not dispatch-only


def _on_wire_record(on_wire_ms: float, **kwargs):
    record = _record(**kwargs)
    record.update({
        "scheduled_offset_s": 0.0,
        "body_build_ms": max(0.0, on_wire_ms - record["pool_wait_ms"]),
        "on_wire_delay_ms": on_wire_ms,
    })
    return record


def test_a_cell_that_fired_on_time_but_reached_the_wire_late_is_void() -> None:
    """The regression the on-wire gauge exists for: the dispatcher kept the schedule and
    the pool never starved, so the pre-existing numbers are both tiny - and every request
    still went out 200 ms after it was due because its prompt was built mid-send. Under
    the old rule this cell passed and its windows reached the theta fit."""
    records = [_on_wire_record(200.0, pool_wait=1.0) for _ in range(100)]
    guard = openloop.check_cell(
        "i256_o128_c60", scheduled=100, records=records,
        p99_delay_ms=2.0,  # what the dispatcher saw, and all the old rule looked at
        max_p99_delay_ms=openloop.CALIBRATION_MAX_P99_DELAY_MS,
    )
    assert guard.p99_on_wire_delay_ms == 200.0
    assert openloop.VOID_DISPATCH_DELAY in guard.void_reasons
    assert "on-wire" in " ".join(guard.issues)
    # the decomposition is kept, so the miss is attributable
    assert guard.p99_delay_ms == 2.0 and guard.p99_body_build_ms == 199.0


def test_the_dispatch_delay_still_stands_in_where_no_on_wire_figure_exists() -> None:
    """A request cannot reach the wire before it was fired, so on a capture that predates
    the gauge the dispatcher's own number is a valid lower bound - never a free pass."""
    guard = openloop.check_cell(
        "i256_o128_c120", scheduled=10, records=[_record() for _ in range(10)],
        p99_delay_ms=900.0,
    )
    assert guard.p99_on_wire_delay_ms == 900.0
    assert openloop.VOID_DISPATCH_DELAY in guard.void_reasons


def test_a_cell_inside_the_deadline_on_every_segment_passes() -> None:
    records = [_on_wire_record(40.0, pool_wait=1.0) for _ in range(10)]
    guard = openloop.check_cell(
        "c", scheduled=10, records=records, p99_delay_ms=2.0,
        max_p99_delay_ms=openloop.CALIBRATION_MAX_P99_DELAY_MS,
    )
    assert guard.ok and not guard.void_reasons


def test_the_guard_artifact_reports_the_whole_decomposition() -> None:
    guard = openloop.check_cell(
        "c", scheduled=2, records=[_on_wire_record(12.0), _on_wire_record(9.0)],
        p99_delay_ms=1.0, prompt_store_misses=0, rps_error_ratio=0.02,
    )
    artifact = guard.as_dict()
    for key in (
        "p99_delay_ms", "p99_pool_wait_ms", "p99_body_build_ms", "p99_on_wire_delay_ms",
        "prompt_store_misses", "rps_error_ratio",
    ):
        assert key in artifact
    assert artifact["prompt_store_misses"] == 0
    assert artifact["rps_error_ratio"] == 0.02


# -------------------------------------------------- prompts are materialised per cell


def test_drive_cell_schedule_materialises_every_prompt_before_it_sends(tmp_path) -> None:
    """The prompts land in the run's own output directory - never in the committed
    schedule tree, which stays a few kB of segments - and every request finds its own."""
    from tre_replayer.engine.schedule import RpsSegment

    seg = RpsSegment("dsqwen-7b", 0.0, 0.3, 50.0, input_tokens=64, max_output_tokens=16)
    stream = _FakeStream()
    prompts_dir = tmp_path / "prompts"
    _s, _e, guard = openloop.drive_cell_schedule(
        "http://gw", "dsqwen-7b", "i64_o16_c60", [seg],
        stream_call=stream, prompt_mode="token_ids", prompt_dir=prompts_dir,
    )
    path = openloop.prompt_file_path_for(prompts_dir, "i64_o16_c60")
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert len(rows) == guard.scheduled > 3
    assert guard.prompt_store_misses == 0
    # what was materialised is exactly what went out, request for request
    materialised = {row["request_id"]: row["prompt"] for row in rows}
    sent = [json.loads(body)["prompt"] for body in stream.calls]
    assert sorted(map(tuple, sent)) == sorted(
        tuple(prompt) for prompt in materialised.values()
    )


def test_a_runtime_generated_hold_schedule_takes_the_same_materialisation_path(tmp_path) -> None:
    """The boundary search builds its hold schedules while the campaign runs, so they
    never pass through the committed tree. If they skipped the materialisation they would
    quietly fall back to fitting every prompt mid-send - the exact regression - so the
    seam has to be the driver, not the schedule file."""
    from dataclasses import replace

    from scripts import gen_calibration_schedules as gen
    from tre_replayer.traces.loader import load_trace_segments

    body, _meta = gen.build_hold_schedule("dsqwen-7b", "S1", 10.0, 0.87, 120.0, stage="bisect")
    schedule_path = tmp_path / "hold.json"
    schedule_path.write_text(json.dumps(body), encoding="utf-8")
    segments = [
        # the probe holds 8.7 rps for two minutes; compress it into a fraction of a
        # second at a higher rate. What is under test is the driver, not the clock.
        replace(seg, start_s=seg.start_s / 300.0, end_s=seg.end_s / 300.0, rps=seg.rps * 20.0)
        for seg in load_trace_segments(schedule_path)
        if seg.model == "dsqwen-7b"
    ]
    prompts_dir = tmp_path / "prompts"
    _s, _e, guard = openloop.drive_cell_schedule(
        "http://gw", "dsqwen-7b", "hold_probe", segments,
        stream_call=_FakeStream(), prompt_mode="token_ids", prompt_dir=prompts_dir,
    )
    assert guard.sent > 0
    assert guard.prompt_store_misses == 0
    assert openloop.prompt_file_path_for(prompts_dir, "hold_probe").exists()


def test_without_a_prompt_directory_the_cell_still_runs(tmp_path) -> None:
    """No store is not an error - it is the fallback, and it reports itself as absent
    rather than as a clean zero."""
    from tre_replayer.engine.schedule import RpsSegment

    seg = RpsSegment("m", 0.0, 0.2, 30.0, input_tokens=32, max_output_tokens=8)
    _s, _e, guard = openloop.drive_cell_schedule(
        "http://gw", "m", "i32_o8_c60", [seg], stream_call=_FakeStream(), prompt_mode="token_ids"
    )
    assert guard.sent > 0 and guard.prompt_store_misses is None


# ------------------------------------------------------- the achieved arrival series


def test_drive_cell_schedule_writes_the_nominal_and_achieved_arrival_series(tmp_path) -> None:
    """The evidence that a cell offered the intensity its schedule describes."""
    import csv

    from tre_replayer.engine.schedule import RpsSegment

    seg = RpsSegment("m", 0.0, 0.4, 60.0, input_tokens=32, max_output_tokens=8)
    path = tmp_path / "cell.rps.csv"
    _s, _e, guard = openloop.drive_cell_schedule(
        "http://gw", "m", "i32_o8_c60", [seg], stream_call=_FakeStream(),
        prompt_mode="token_ids", rps_timeline_path=path, rps_window_s=0.1,
    )
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    assert rows and {row["model"] for row in rows} == {"m"}
    assert sum(int(row["scheduled_requests"]) for row in rows) == guard.scheduled
    assert sum(int(row["achieved_requests"]) for row in rows) == guard.sent
    assert guard.rps_error_ratio is not None


def test_achieved_offsets_are_read_off_the_schedule_grid() -> None:
    """Offset plus on-wire lateness, so the achieved series sits on the same axis as the
    nominal one without any clock conversion."""
    records = [
        {"scheduled_offset_s": 4.0, "on_wire_delay_ms": 250.0},
        {"scheduled_offset_s": 4.5, "on_wire_delay_ms": 0.0},
        {"on_wire_delay_ms": 10.0},  # pre-dates the field: skipped, never guessed
    ]
    assert openloop.achieved_arrival_offsets(records) == [4.25, 4.5]


# ============================== admission overflow vs transient proxy error
#
# Two different things the gateway does, which used to be one class. Each rule gets its
# own test, because collapsing them again fails nothing visibly: it just voids cells that
# offered exactly the load they were asked to, and a campaign then writes zero rows and
# still prints "complete".

#: The verbatim failure that voided a real calibration cell on 2026-09-21 - 1 request out
#: of 1296 in a 450 s dsqwen-7b steps cell. ``in_flight_at_send`` is 43, under a circuit
#: breaker admitting 4096 parallel / 1024 pending, so the breaker cannot arithmetically
#: have produced it; and the reset reason says the same thing in Envoy's own words.
MEASURED_CONNECTION_TERMINATION = {
    "request_id": "dsqwen-7b-001295",
    "http_status": 503,
    "error_body": (
        "upstream connect error or disconnect/reset before headers. "
        "reset reason: connection termination"
    ),
    "error_headers": {
        "content-length": "95",
        "content-type": "text/plain",
        "connection": "close",
    },
    "e2e_ms": 2.92,
    "in_flight_at_send": 43,
    "actual_send_ts_ms": 1_000,
    "ttft_ms": None,
    "tpot_ms": None,
}


def _terminated(**over):
    record = dict(MEASURED_CONNECTION_TERMINATION)
    record.update(over)
    return record


def test_the_measured_connection_termination_is_not_an_admission_decision() -> None:
    assert (
        openloop.classify_failure(MEASURED_CONNECTION_TERMINATION)
        == openloop.FAILURE_PROXY_TRANSIENT
    )
    assert (
        openloop.proxy_failure_reason(MEASURED_CONNECTION_TERMINATION)
        == "connection termination"
    )
    # Same sentence in front, opposite meaning behind it: the prefix cannot be the test.
    assert openloop.classify_failure(MEASURED_SHED) == openloop.FAILURE_ADMISSION_OVERFLOW
    assert openloop.proxy_failure_reason(MEASURED_SHED) == "overflow"


def test_one_connection_termination_does_not_void_a_calibration_cell() -> None:
    # The cell this is taken from: 1296 requests, 1 of them this one, and 450 s of
    # offered load thrown away for it.
    records = [_served() for _ in range(1295)] + [_terminated()]
    guard = openloop.check_cell(
        "i256_o128_c95", scheduled=1296, records=records, p99_delay_ms=1.0,
        shed_policy=openloop.SHED_POLICY_VOID,
    )
    assert not guard.voided
    assert guard.proxy_transient_errors == 1 and guard.proxy_errors == 0
    assert guard.outcomes["proxy_transient"] == 1 and guard.outcomes["shed"] == 0
    # ... and it is still counted against what the cell delivered, not forgiven.
    assert guard.outcomes["admitted"] == 1296 and guard.outcomes["ok"] == 1295


def test_one_admission_overflow_still_voids_a_calibration_cell() -> None:
    records = [_served() for _ in range(1295)] + [_shed()]
    guard = openloop.check_cell(
        "i256_o128_c95", scheduled=1296, records=records, p99_delay_ms=1.0,
        shed_policy=openloop.SHED_POLICY_VOID,
    )
    assert guard.voided and openloop.VOID_SHED in guard.void_reasons
    assert guard.proxy_errors == 1 and guard.proxy_transient_errors == 0


def test_transient_proxy_errors_past_their_budget_void_the_cell() -> None:
    # 1 % of the cell, twice the 0.5 % budget: the path itself was unhealthy and the
    # windows measured a system that was dropping connections.
    records = [_served() for _ in range(990)] + [_terminated() for _ in range(10)]
    guard = openloop.check_cell(
        "c", scheduled=1000, records=records, p99_delay_ms=1.0,
        shed_policy=openloop.SHED_POLICY_VOID,
    )
    assert guard.voided and openloop.VOID_PROXY_TRANSIENT in guard.void_reasons
    # and it is a reason of its own, never folded into the admission-overflow one
    assert openloop.VOID_SHED not in guard.void_reasons


def test_a_cell_is_never_voided_on_a_single_transient_error_however_small_it_is() -> None:
    # 1/150 is 0.67 %, above the rate - but one dropped connection is not a rate, and a
    # 60 s boundary probe must not void on it.
    records = [_served() for _ in range(149)] + [_terminated()]
    guard = openloop.check_cell(
        "c", scheduled=150, records=records, p99_delay_ms=1.0,
        shed_policy=openloop.SHED_POLICY_VOID,
    )
    assert not guard.voided
    # Two of them in the same small cell is a rate, and does void it.
    records = [_served() for _ in range(148)] + [_terminated(), _terminated()]
    guard = openloop.check_cell(
        "c", scheduled=150, records=records, p99_delay_ms=1.0,
        shed_policy=openloop.SHED_POLICY_VOID,
    )
    assert guard.voided and openloop.VOID_PROXY_TRANSIENT in guard.void_reasons


def test_an_unknown_proxy_wording_is_counted_as_transient_and_stays_visible() -> None:
    # A wording nobody has a rule for must not void the cell - one phrasing change in
    # Envoy would otherwise void every cell of a campaign - but it must not disappear
    # either, or the split would quietly stop being made on evidence.
    unknown = _served(
        http_status=503, e2e_ms=4.0, ttft_ms=None, tpot_ms=None,
        error_body="upstream connect error or disconnect/reset before headers. "
                   "reset reason: something nobody has seen",
        error_headers={"content-type": "text/plain"},
    )
    assert openloop.classify_failure(unknown) == openloop.FAILURE_PROXY_TRANSIENT
    guard = openloop.check_cell(
        "c", scheduled=100, records=[_served() for _ in range(99)] + [unknown],
        p99_delay_ms=1.0, shed_policy=openloop.SHED_POLICY_VOID,
    )
    assert not guard.voided
    assert guard.unrecognised_proxy_failures == 1
    assert guard.proxy_failure_reasons == {"something nobody has seen": 1}


def test_a_proxy_rejection_with_no_body_is_transient_and_flagged_unrecognised() -> None:
    # The connection went away before a body could be read, so there is no evidence
    # either way. That is the conservative side, and it is reported as a fallback.
    headless = _served(
        http_status=503, e2e_ms=1.0, ttft_ms=None, tpot_ms=None,
        error_body="", error_headers={},
    )
    assert openloop.classify_failure(headless) == openloop.FAILURE_PROXY_TRANSIENT
    guard = openloop.check_cell(
        "c", scheduled=10, records=[_served() for _ in range(9)] + [headless],
        p99_delay_ms=1.0, shed_policy=openloop.SHED_POLICY_VOID,
    )
    assert not guard.voided and guard.unrecognised_proxy_failures == 1
    assert guard.proxy_failure_reasons == {openloop.PROXY_REASON_EMPTY_BODY: 1}


def test_an_envoy_overload_header_is_read_as_an_admission_decision() -> None:
    # The one case where Envoy says "I shed this because of load" without a reset reason.
    record = _served(
        http_status=429, e2e_ms=1.0, ttft_ms=None, tpot_ms=None,
        error_headers={"x-envoy-overloaded": "true"}, error_body="",
    )
    assert openloop.classify_failure(record) == openloop.FAILURE_ADMISSION_OVERFLOW


def test_a_transient_error_is_a_goodput_loss_but_not_a_rejection() -> None:
    records = [_served()] * 8 + [_terminated()] + [_shed()]
    result = openloop.goodput(records, ttft_slo_ms=500.0, tpot_slo_ms=75.0)
    # admitted excludes only the shed: nothing refused the terminated request.
    assert result.offered == 10 and result.admitted == 9 and result.good == 8


# ------------------------------------------------------- window marking


def test_a_window_holding_a_transient_proxy_error_is_marked_violating_and_kept() -> None:
    # The request went unserved, so the window is a loss; but it is counted in its own
    # column, because it is not evidence that the ENGINE failed.
    rows = [
        {"window_start_ms": 0, "window_end_ms": 1000, "p95_ttft_client_ms": 50.0},
        {"window_start_ms": 1000, "window_end_ms": 2000, "p95_ttft_client_ms": 50.0},
    ]
    marked = openloop.mark_unserved_request_windows(
        rows, [_terminated(actual_send_ts_ms=1500)]
    )
    assert [_label(r) for r in marked] == ["healthy", "violated"]
    assert [r["proxy_transient_errors"] for r in marked] == [0, 1]
    assert [r["model_errors"] for r in marked] == [0, 0]


# ------------------------------------------------------- truncation


class _SenderFailingWith:
    """Fake sender that answers with ``failure`` from request ``after`` onwards."""

    def __init__(self, failure: dict, *, after: int) -> None:
        self.records: list = []
        self._failure = failure
        self._after = after
        self.sent = 0

    async def __call__(self, request, scheduled_ts, actual_ts) -> None:
        self.sent += 1
        if self.sent > self._after:
            self.records.append(dict(self._failure, request_id=request.request_id))
        else:
            self.records.append(_record(request_id=request.request_id))


def test_a_transient_proxy_error_does_not_truncate_the_cell() -> None:
    # The gateway went on admitting everything after it, so the rest of the cell is
    # still measuring the engine and there is nothing to cut short.
    sender = _SenderFailingWith(MEASURED_CONNECTION_TERMINATION, after=2)
    wrapper = openloop.TruncateOnProxyShed(sender, drain_start_s=100.0)
    _drive(wrapper, [0.0, 10.0, 20.0, 30.0, 40.0])

    assert not wrapper.truncated
    assert wrapper.censored == 0
    assert len(sender.records) == 5


def test_the_void_policy_skips_the_drain_because_its_only_purpose_is_evidence() -> None:
    # Under void the cell is re-run whatever it collected, so replaying the drain buys
    # no evidence and costs wall clock the re-run needs. Truncation itself stays: it
    # stops hammering a gateway that is already refusing.
    sender = _SenderFailingWith(MEASURED_SHED, after=2)
    wrapper = openloop.TruncateOnProxyShed(sender, drain_start_s=100.0, keep_drain=False)
    _drive(wrapper, [0.0, 10.0, 20.0, 30.0, 40.0, 100.0, 110.0])

    assert wrapper.truncated and not wrapper.keep_drain
    assert wrapper.censored == 4
    assert [r["request_id"] for r in sender.records] == ["r0", "r1", "r2"]


# ------------------------------------------------------- the sentinel as cross-check

#: What the node can actually reach: the gateway pod's metrics port. Measured
#: 2026-09-21 - the admin listener on :19000 refuses from off-pod and the envoy container
#: has no curl, so this Prometheus body is the only form the sentinel will ever see live.
ENVOY_PROMETHEUS_STATS = """
# TYPE envoy_cluster_upstream_rq_pending_overflow counter
envoy_cluster_upstream_rq_pending_overflow{envoy_cluster_name="httproute/tre-v2/dsqwen-7b-router/rule/0"} 80
envoy_cluster_upstream_rq_pending_overflow{envoy_cluster_name="httproute/tre-v2/dsqwen-14b-router/rule/0"} 0
envoy_cluster_upstream_rq_total{envoy_cluster_name="httproute/tre-v2/dsqwen-7b-router/rule/0"} 99999
"""


def test_the_sentinel_reads_the_listener_the_node_can_actually_reach() -> None:
    assert openloop.parse_envoy_counters(
        ENVOY_PROMETHEUS_STATS, openloop.PENDING_OVERFLOW_COUNTER
    ) == 80
    assert openloop.parse_envoy_counters(
        ENVOY_PROMETHEUS_STATS, openloop.PENDING_OVERFLOW_COUNTER,
        cluster_filter="dsqwen-14b",
    ) == 0


def test_a_stats_body_without_the_counter_is_not_measured_rather_than_clean() -> None:
    # The failure this closes: the admin-format parser fed a Prometheus body summed
    # nothing and returned 0, and the sentinel then certified every cell of a campaign.
    assert openloop.parse_envoy_counters(
        "envoy_cluster_upstream_rq_total{envoy_cluster_name=\"x\"} 5",
        openloop.PENDING_OVERFLOW_COUNTER,
    ) is None
    sentinel = openloop.PendingOverflowSentinel(
        read=lambda: "envoy_cluster_upstream_rq_total{envoy_cluster_name=\"x\"} 5"
    )
    assert sentinel.start() is None
    assert sentinel.delta() is None


def test_the_sentinel_contradicting_the_client_is_recorded_both_ways() -> None:
    # The client saw an overflow Envoy's own counter never accounted for.
    guard = openloop.check_cell(
        "c", scheduled=10, records=[_served()] * 9 + [_shed()], p99_delay_ms=1.0,
        pending_overflow_delta=0, shed_policy=openloop.SHED_POLICY_VOID,
    )
    assert guard.sentinel_contradiction
    assert "did not move" in guard.sentinel_contradiction

    # ... and the other way: Envoy refused work no request was attributed to.
    guard = openloop.check_cell(
        "c", scheduled=10, records=[_served()] * 10, p99_delay_ms=1.0,
        pending_overflow_delta=3,
    )
    assert guard.sentinel_contradiction and "rose by 3" in guard.sentinel_contradiction


def test_the_sentinel_agreeing_with_the_client_records_no_contradiction() -> None:
    guard = openloop.check_cell(
        "c", scheduled=10, records=[_served()] * 9 + [_shed()], p99_delay_ms=1.0,
        pending_overflow_delta=1, shed_policy=openloop.SHED_POLICY_VOID,
    )
    assert guard.sentinel_contradiction is None
    guard = openloop.check_cell(
        "c", scheduled=10, records=[_served()] * 10, p99_delay_ms=1.0,
        pending_overflow_delta=0,
    )
    assert guard.sentinel_contradiction is None


def test_an_unmeasured_sentinel_claims_no_contradiction_either() -> None:
    # "Not measured" must not read as "Envoy says this never happened".
    guard = openloop.check_cell(
        "c", scheduled=10, records=[_served()] * 9 + [_shed()], p99_delay_ms=1.0,
        pending_overflow_delta=None, shed_policy=openloop.SHED_POLICY_VOID,
    )
    assert guard.sentinel_contradiction is None
