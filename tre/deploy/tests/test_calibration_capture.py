"""scripts.calibration_capture: per-cell vLLM metrics, gateway redis dumps, controller
ticks, the cells/ layout and the compatibility reader."""
from __future__ import annotations

import json
import math
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import calibration_campaign as campaign
from scripts import calibration_capture as cc
from scripts import openloop, r3_grid, rewindow_from_raw
from tre_common.rediskeys import decision_hist_key, hist_key, inst_key, pods_key
from tre_common.tss import tss_terms, replica_factor


def _body(*, prompt=100.0, gen=50.0, ttft_buckets=(0, 1, 3, 3), running=2.0, extra: str = "") -> str:
    les = ("0.1", "0.5", "1.0", "+Inf")
    lines = [
        "# HELP vllm:prompt_tokens_total Number of prefill tokens processed.",
        "# TYPE vllm:prompt_tokens_total counter",
        f'vllm:prompt_tokens_total{{engine="0",model_name="m7"}} {prompt}',
        'vllm:prompt_tokens_created{engine="0",model_name="m7"} 1.79e+09',
        f'vllm:generation_tokens_total{{engine="0",model_name="m7"}} {gen}',
        'vllm:num_preemptions_total{engine="0",model_name="m7"} 0.0',
        'vllm:request_success_total{engine="0",finished_reason="stop",model_name="m7"} 3.0',
        f'vllm:num_requests_running{{engine="0",model_name="m7"}} {running}',
        'vllm:num_requests_waiting{engine="0",model_name="m7"} 0.0',
        'vllm:kv_cache_usage_perc{engine="0",model_name="m7"} 0.25',
        'vllm:iteration_tokens_total_bucket{engine="0",le="1.0",model_name="m7"} 4.0',
        'python_gc_objects_collected_total{generation="0"} 12.0',
    ]
    for le, v in zip(les, ttft_buckets):
        lines.append(f'vllm:time_to_first_token_seconds_bucket{{engine="0",le="{le}",model_name="m7"}} {float(v)}')
    lines.append(f'vllm:time_to_first_token_seconds_count{{engine="0",model_name="m7"}} {float(ttft_buckets[-1])}')
    lines.append('vllm:time_to_first_token_seconds_sum{engine="0",model_name="m7"} 1.25')
    lines.append('vllm:time_to_first_token_seconds_created{engine="0",model_name="m7"} 1.79e+09')
    # the 0.10 name of the TPOT histogram is kept too
    lines.append('vllm:time_per_output_token_seconds_bucket{engine="0",le="+Inf",model_name="m7"} 7.0')
    lines.append(extra)
    return "\n".join(lines) + "\n"


# ------------------------------------------------------------------------- parsing


def test_parse_vllm_metrics_keeps_counters_gauges_and_cumulative_buckets() -> None:
    st = cc.parse_vllm_metrics(_body())
    assert st["model_name"] == "m7"
    assert st["c"] == {
        'vllm:prompt_tokens_total{engine="0"}': 100,
        'vllm:generation_tokens_total{engine="0"}': 50,
        'vllm:num_preemptions_total{engine="0"}': 0,
        'vllm:request_success_total{engine="0",finished_reason="stop"}': 3,
    }
    # _created, python_* and the histogram named *_total (iteration_tokens_total) are not counters
    assert not any("created" in k or "python" in k or "iteration" in k for k in st["c"])
    assert st["g"]['vllm:num_requests_running{engine="0"}'] == 2
    assert st["g"]['vllm:kv_cache_usage_perc{engine="0"}'] == 0.25
    ttft = st["h"]['vllm:time_to_first_token_seconds{engine="0"}']
    assert ttft == {"b": {"0.1": 0, "0.5": 1, "1.0": 3, "+Inf": 3}, "s": 1.25, "n": 3}
    assert st["h"]['vllm:time_per_output_token_seconds{engine="0"}']["b"] == {"+Inf": 7}
    assert 'vllm:iteration_tokens_total{engine="0"}' not in st["h"]  # not in the default list


def test_parse_vllm_metrics_histogram_list_is_configurable_and_nan_is_a_string() -> None:
    st = cc.parse_vllm_metrics(
        _body(extra='vllm:kv_cache_usage_perc{engine="1",model_name="m7"} NaN'),
        histograms=("vllm:iteration_tokens_total",),
    )
    assert list(st["h"]) == ['vllm:iteration_tokens_total{engine="0"}']
    assert st["g"]['vllm:kv_cache_usage_perc{engine="1"}'] == "NaN"
    json.dumps(st, allow_nan=False)  # every value is JSON


def test_default_histograms_are_the_requested_families() -> None:
    for name in ("vllm:time_to_first_token_seconds", "vllm:inter_token_latency_seconds",
                 "vllm:e2e_request_latency_seconds", "vllm:request_prompt_tokens",
                 "vllm:request_generation_tokens"):
        assert name in cc.DEFAULT_VLLM_HISTOGRAMS


# --------------------------------------------------------------- delta encoding


def test_delta_rows_carry_only_what_changed_and_decode_back() -> None:
    bodies = [
        _body(),
        _body(),  # nothing moved
        _body(prompt=180.0, ttft_buckets=(0, 1, 4, 4)),  # one counter + two buckets + n
        _body(prompt=180.0, ttft_buckets=(0, 1, 4, 4), running=5.0),
    ]
    enc = cc.DeltaEncoder(keyframe_every=100)
    rows = [enc.encode(i, cc.parse_vllm_metrics(b)) for i, b in enumerate(bodies)]
    assert rows[0]["full"] is True
    assert rows[1] == {"ts_ms": 1}
    assert rows[2]["c"] == {'vllm:prompt_tokens_total{engine="0"}': 180}
    assert rows[2]["h"] == {'vllm:time_to_first_token_seconds{engine="0"}': {"b": {"1.0": 4, "+Inf": 4}, "n": 4}}
    assert "g" not in rows[2]
    assert rows[3] == {"ts_ms": 3, "g": {'vllm:num_requests_running{engine="0"}': 5}}
    lines = [json.dumps(r) for r in rows]
    decoded = list(cc.decode_vllm_metrics(lines))
    for i, b in enumerate(bodies):
        want = cc.parse_vllm_metrics(b)
        assert decoded[i]["ts_ms"] == i
        assert decoded[i]["c"] == want["c"] and decoded[i]["g"] == want["g"] and decoded[i]["h"] == want["h"]


def test_keyframe_on_a_new_series_and_periodically() -> None:
    enc = cc.DeltaEncoder(keyframe_every=3)
    base = cc.parse_vllm_metrics(_body())
    grown = cc.parse_vllm_metrics(_body(extra='vllm:num_requests_waiting{engine="1",model_name="m7"} 1'))
    flags = [bool(enc.encode(t, s).get("full")) for t, s in enumerate([base, base, grown, grown, grown, grown])]
    # first, then the new series, then every 3rd row
    assert flags == [True, False, True, False, False, True]


# ------------------------------------------------------------------- recorder


def test_sampler_hands_every_body_to_the_recorder_and_keeps_its_totals(tmp_path: Path) -> None:
    urls = ["http://10.0.0.1:8000/metrics", "http://10.0.0.2:8000/metrics"]
    rec = cc.VllmMetricsRecorder(tmp_path / "v", {urls[0]: "default/pod-a", urls[1]: "default/pod-b"},
                                 model="m7")

    def fetch(url: str) -> str:
        if url == urls[1]:
            raise OSError("connection refused")
        return _body()

    sample = openloop.make_pod_metrics_sampler(urls, fetch=fetch, recorder=rec)
    snap = sample(1000)
    sample(2000)
    rec.close()
    assert snap["running"] == 2.0 and snap["scrape_errors"] == 1.0
    a = (tmp_path / "v" / "default_pod-a.jsonl").read_text().splitlines()
    head = json.loads(a[0])
    assert head["kind"] == "header" and head["pod"] == "default/pod-a" and head["model_name_label"] == "m7"
    assert json.loads(a[1])["full"] is True and json.loads(a[2]) == {"ts_ms": 2000}
    b = [json.loads(x) for x in (tmp_path / "v" / "default_pod-b.jsonl").read_text().splitlines()]
    assert [r.get("error", "")[:7] for r in b[1:]] == ["OSError", "OSError"]
    s = rec.summary(tmp_path)
    assert s["samples"] == {"default/pod-a": 2} and s["errors"] == {"default/pod-b": 2}
    assert s["files"]["default/pod-a"] == "v/default_pod-a.jsonl"


def test_a_failing_recorder_never_costs_the_queue_sample() -> None:
    class Broken:
        def record(self, *a):
            raise RuntimeError("disk full")

        def record_error(self, *a):
            raise RuntimeError("disk full")

    sample = openloop.make_pod_metrics_sampler(["http://x/metrics"], fetch=lambda u: _body(), recorder=Broken())
    assert sample(0)["running"] == 2.0


def test_drive_cell_schedule_writes_per_pod_metrics_through_the_sidecar(tmp_path: Path) -> None:
    from tre_replayer.engine.schedule import RpsSegment

    from test_openloop import _FakeStream  # the suite's fake streaming seam

    url = "http://10.0.0.9:8000/metrics"
    rec = cc.VllmMetricsRecorder(tmp_path / "v", {url: "default/p"}, model="dsqwen-7b")
    sampler = openloop.make_pod_metrics_sampler([url], fetch=lambda u: _body(), recorder=rec)
    seg = RpsSegment("dsqwen-7b", 0.0, 0.3, 30.0, input_tokens=64, max_output_tokens=16)
    openloop.drive_cell_schedule(
        "http://gw/v1/completions", "dsqwen-7b", "i64_o16_c60", [seg],
        instant_path=tmp_path / "x.instant.jsonl", instant_sampler=sampler, instant_interval_s=0.05,
        stream_call=_FakeStream(), prompt_mode="token_ids",
    )
    rec.close()
    rows = (tmp_path / "v" / "default_p.jsonl").read_text().splitlines()
    assert len(rows) >= 3 and json.loads(rows[1])["full"] is True


# ------------------------------------------------------------------ fake redis


class FakeRedis:
    def __init__(self, now_ms: int = 1_000_000) -> None:
        self.sets: dict[str, set] = {}
        self.zsets: dict[str, dict[str, float]] = {}
        self.now_ms = now_ms

    def sadd(self, key, *members):
        self.sets.setdefault(key, set()).update(m.encode() for m in members)

    def smembers(self, key):
        return set(self.sets.get(key, set()))

    def zadd(self, key, mapping):
        self.zsets.setdefault(key, {}).update(mapping)

    def _sorted(self, key):
        return sorted(self.zsets.get(key, {}).items(), key=lambda kv: (kv[1], kv[0]))

    def zrangebyscore(self, key, lo, hi, withscores=False):
        items = [(m.encode(), s) for m, s in self._sorted(key) if float(lo) <= s <= float(hi)]
        return items if withscores else [m for m, _ in items]

    def zrevrangebyscore(self, key, hi, lo, start=0, num=None, withscores=False):
        items = [(m.encode(), s) for m, s in reversed(self._sorted(key))]
        items = items[start:start + num] if num is not None else items[start:]
        return items if withscores else [m for m, _ in items]

    def zrange(self, key, start, stop):
        items = [m.encode() for m, _ in self._sorted(key)]
        return items[start:] if stop == -1 else items[start:stop + 1]

    def time(self):
        return self.now_ms // 1000, (self.now_ms % 1000) * 1000


def _layout(tmp_path: Path) -> cc.CellLayout:
    return cc.CellLayout(tmp_path / "run" / "m7", "m7_S1_hold_c1000001_a1", "i0_o0_c1000001")


def _seed_gateway(r: FakeRedis, pods, stamps) -> None:
    for pod in pods:
        r.sadd(pods_key("m7"), pod)
        for s in stamps:
            r.zadd(hist_key(pod), {json.dumps({"timestamp": s, "model_histogram_metrics": {"m7/x": {"count": s}}}): s})
            r.zadd(inst_key(pod), {json.dumps({"timestamp": s, "model_metrics": {"m7/num_requests_running": 1}}): s})


def test_wait_for_gateway_write_times_the_next_round() -> None:
    r = FakeRedis(now_ms=1_000_000)
    _seed_gateway(r, ["default/p"], [980_000, 990_000])
    clock = {"t": 0.0}

    def sleep(dt):
        clock["t"] += dt
        if clock["t"] >= 1.0 and 1_000_000 not in r.zsets[inst_key("default/p")].values():
            r.zadd(inst_key("default/p"), {"new": 1_000_000})
            r.now_ms = 1_003_400  # the gateway's ticker runs 3.4 s after the round stamp

    out = cc.wait_for_gateway_write(r, ["default/p"], timeout_s=12, poll_s=0.25, sleep=sleep,
                                    monotonic=lambda: clock["t"])
    assert out["new_round_seen"] and out["new_round_ms"] == 1_000_000
    assert out["latest_round_before_ms"] == 990_000
    assert out["write_phase_ms"] == 3_400


def test_wait_for_gateway_write_gives_up_at_the_timeout() -> None:
    r = FakeRedis()
    _seed_gateway(r, ["default/p"], [990_000])
    clock = {"t": 0.0}
    out = cc.wait_for_gateway_write(r, ["default/p"], timeout_s=1.0, poll_s=0.25,
                                    sleep=lambda dt: clock.__setitem__("t", clock["t"] + dt),
                                    monotonic=lambda: clock["t"])
    assert not out["new_round_seen"] and out["waited_s"] >= 1.0
    assert cc.wait_for_gateway_write(r, ["default/p"], timeout_s=0)["reason"] == "wait disabled"


def test_dump_gateway_docs_keeps_only_the_cell_range_per_pod_and_kind(tmp_path: Path) -> None:
    r = FakeRedis()
    pods = ["default/pod-a", "default/pod-b"]
    _seed_gateway(r, pods, [900_000, 950_000, 960_000, 990_000])
    layout = _layout(tmp_path)
    s = cc.dump_gateway_docs(r, layout, pods, lo_ms=950_000, hi_ms=980_000)
    assert s["docs"] == {"hist": {p: 2 for p in pods}, "inst": {p: 2 for p in pods}}
    path = layout.cell_dir / "gateway_redis_dump" / "hist" / "default_pod-a.jsonl"
    assert s["files"]["hist"]["default/pod-a"] == "gateway_redis_dump/hist/default_pod-a.jsonl"
    rows = [json.loads(x) for x in path.read_text().splitlines()]
    assert rows[0]["kind"] == "header" and rows[0]["redis_key"] == hist_key("default/pod-a")
    assert [x["score"] for x in rows[1:]] == [950_000, 960_000]
    assert rows[1]["doc"]["model_histogram_metrics"]["m7/x"]["count"] == 950_000
    assert s["round_stamp_phases_ms"] == [0] and s["last_round_ms"] == 960_000


def test_raw_tss_comes_from_the_controller_or_is_derived_at_factor_one() -> None:
    terms = tss_terms(prompt_tokens=3000.0, generation_tokens=9000.0, avg_running=12.0, avg_waiting=2.0,
                      w_p=0.0, lambda_wait=3.0, qmin=1.0, factor=replica_factor(1, 1))
    # an old member (no trs_raw): y / q_ctl, NOT scaled by the bound count it carries
    member = {"y_m": terms.numerator, "q_ctl": terms.queue_ctl, "assigned_replicas": 8, "routable_pods": 1}
    raw, source = cc.tss_raw_of(member)
    assert raw == pytest.approx(terms.raw) and source == cc.TSS_RAW_DERIVED
    assert cc.tss_raw_of({**member, "trs_raw": 1234.5}) == (1234.5, cc.TSS_RAW_FROM_CONTROLLER)
    assert cc.tss_raw_of({**member, "trs_raw": None}) == (None, cc.TSS_RAW_FROM_CONTROLLER)
    assert cc.tss_raw_of({"y_m": None, "q_ctl": 1.0}) == (None, None)


def test_dump_controller_ticks_writes_members_with_the_derived_raw_tss(tmp_path: Path) -> None:
    r = FakeRedis()
    key = decision_hist_key("m7")
    for end, trs in ((940_000, 400.0), (950_000, 420.0), (990_000, 500.0)):
        r.zadd(key, {json.dumps({"window_end_ms": end, "ts": end, "trs": trs, "trs_z_m": trs / 500,
                                 "z_m": trs / 500, "state": "healthy", "y_m": 10_000.0, "q_ctl": 20.0,
                                 "assigned_replicas": 1, "routable_pods": 1}): end})
    layout = _layout(tmp_path)
    s = cc.dump_controller_ticks(r, layout, "m7", lo_ms=945_000, hi_ms=995_000)
    assert s["members"] == 2 and s["last_window_end_ms"] == 990_000
    rows = [json.loads(x) for x in layout.controller_ticks_path.read_text().splitlines()]
    assert rows[0]["kind"] == "header"
    assert [(x["window_end_ms"], x["trs"], x["tss_raw"], x["state"]) for x in rows[1:]] == [
        (950_000, 420.0, 500.0, "healthy"), (990_000, 500.0, 500.0, "healthy")]
    assert rows[1]["tss_raw_source"] == cc.TSS_RAW_DERIVED


class Cluster:
    """A fake redis fed by a gateway and a phase-aligned controller whose clocks are off
    redis TIME by ``gw_skew`` / ``ctl_skew``. The gateway writes round ``S`` (its own
    clock) at redis time ``S - gw_skew + gw_phase`` (``gw_phase`` = its ticker's write
    delay); the controller publishes window ``B`` (a gateway round) once its clock passed
    ``B + offset`` and the gateway wrote ``B``, and it reaches the decision history one
    tick later. :meth:`sleep` is the clock the capture waits on."""

    P = 10_000

    def __init__(self, *, pods=("default/p",), gw_skew=0, gw_phase=3_000, ctl_skew=0, ctl_offset=2_000,
                 ctl_tick=5_000, t0=900_000, controller=True) -> None:
        self.r = FakeRedis(now_ms=t0)
        self.pods = list(pods)
        self.gw_skew, self.gw_phase = gw_skew, gw_phase
        self.ctl_skew, self.ctl_offset, self.ctl_tick = ctl_skew, ctl_offset, ctl_tick
        self.controller = controller
        self.gw_next = (t0 + gw_skew - 300_000) // self.P * self.P
        self.ctl_next = self.gw_next
        self.writes: list[tuple[int, int]] = []  # (round stamp, redis time written)
        self.advance(t0)

    def advance(self, t: int) -> None:
        while self.gw_next - self.gw_skew + self.gw_phase <= t:
            _seed_gateway(self.r, self.pods, [self.gw_next])
            self.writes.append((self.gw_next, self.gw_next - self.gw_skew + self.gw_phase))
            self.gw_next += self.P
        while self.controller:
            b = self.ctl_next
            written = b - self.gw_skew + self.gw_phase
            give_up = b + self.P - 500 - self.ctl_skew  # PhaseAlignedSampler: next round minus retry
            if written > give_up:  # never fresh in time: a stale window, nothing written
                if give_up > t:
                    break
                self.ctl_next += self.P
                continue
            at = max(b + self.ctl_offset - self.ctl_skew, written) + self.ctl_tick
            if at > t:
                break
            self.r.zadd(decision_hist_key("m7"), {_tick(b): b})
            self.ctl_next += self.P
        self.r.now_ms = max(self.r.now_ms, t)

    def sleep(self, dt: float) -> None:
        self.advance(self.r.now_ms + int(round(dt * 1000)))


def _tick(end: int) -> str:
    return json.dumps({"window_end_ms": end, "ts": end, "trs": 1.0, "trs_raw": 2.0, "y_m": 2.0, "q_ctl": 1.0})


CFG = cc.ClockDomainConfig(window_ms=30_000)
RS, RE = 964_000, 994_000  # the cell in redis time


def _run_cell(c: Cluster, layout: cc.CellLayout, *, driver_skew: int = 0, end_at: int = RE, **kw) -> dict:
    """Drive a cell on ``c`` the way r3_grid does: start mark, load, end mark, capture."""
    now = lambda: c.r.now_ms + driver_skew  # noqa: E731 - the driver's clock
    c.advance(RS)
    start = cc.cell_clock_mark(c.r, "m7", CFG, now_ms=now, sleep=c.sleep)
    start_ms = now()
    c.advance(end_at)
    end_ms = now()
    end = cc.cell_clock_mark(c.r, "m7", CFG, start=start, now_ms=now, sleep=c.sleep)
    kw.setdefault("flush_wait_s", 0)
    return cc.capture_after_cell(layout, model="m7", start_ms=start_ms, end_ms=end_ms, window_ms=30_000,
                                 redis_client=c.r, clock_start=start, clock_end=end, clock_config=CFG,
                                 now_ms=now, sleep=c.sleep, **kw)


def _write_legacy(model_dir: Path, stem: str, cell_id: str, *, void: bool = False) -> None:
    raw = model_dir / "raw" / stem
    raw.mkdir(parents=True)
    (raw / (f"{cell_id}.jsonl" + (".void" if void else ""))).write_text('{"cell_id": "x"}\n')
    (raw / f"{cell_id}.instant.jsonl").write_text('{"ts_ms": 1}\n')
    (raw / f"{cell_id}.guard.json").write_text("{}\n")
    (raw / f"{cell_id}.rps.csv").write_text("a\n")
    (model_dir / f"{stem}.csv").write_text("scenario_id\n")


# ----------------------------------------------------------- cell meta and layout


def test_capture_after_cell_writes_the_cell_meta_and_the_reader_resolves_it(tmp_path: Path) -> None:
    layout = _layout(tmp_path)
    _write_legacy(layout.model_dir, layout.stem, layout.cell_id)
    c = Cluster()
    url = "http://10.0.0.1:8000/metrics"
    rec = cc.VllmMetricsRecorder(layout.vllm_metrics_dir, {url: "default/p"}, model="m7")
    rec.record(url, 960_000, _body())
    meta = _run_cell(c, layout, targets=[{"key": "default/p", "url": url}], recorder=rec,
                     info={"guard_voided": False})
    assert meta["errors"] == []
    on_disk = json.loads(layout.meta_path.read_text())
    assert on_disk["layout_version"] == cc.CAPTURE_LAYOUT_VERSION and on_disk["guard_voided"] is False
    assert on_disk["legacy"]["requests_jsonl"] == "../../raw/m7_S1_hold_c1000001_a1/i0_o0_c1000001.jsonl"
    clock = on_disk["clock"]
    assert clock["redis_start_ms"] == RS and clock["redis_end_ms"] == RE
    # redis time +- the margin (one window + one round + the check's 12 s blind spot)
    assert clock["range_ms"] == [RS - 52_000, RE + 52_000] == on_disk["gateway_redis_dump"]["range_ms"]
    assert clock["domain"] == {"gateway": "ok", "controller": "ok"} and "clock_domain_mismatch" not in on_disk
    assert clock["cell_start"]["late_write_baseline_docs"] > 0 and "late_write_baseline" not in clock["cell_start"]
    # cut at the cell end: the controller has not processed the tail windows yet
    ct = on_disk["controller_ticks"]
    assert ct["complete"] is False and ct["reached_tail"] is False and ct["tail_ms"] == 1_030_000
    assert (layout.cell_dir / cc.BACKFILL_MARKER).exists()
    got = cc.resolve_cell_artifacts(layout.model_dir, layout.stem)
    assert got["layout_version"] == 1 and got["cell_id"] == layout.cell_id
    assert got["requests_jsonl"] == (layout.raw_dir / f"{layout.cell_id}.jsonl").resolve()
    assert got["queue_1hz_jsonl"].name == f"{layout.cell_id}.instant.jsonl"
    assert got["failures_jsonl"] is None  # never written
    assert set(got["vllm_metrics"]) == {"default/p"}
    assert set(got["gateway_redis_dump"]["hist"]) == {"default/p"}
    assert got["controller_ticks"] == layout.controller_ticks_path


def test_capture_never_raises_without_redis(tmp_path: Path) -> None:
    layout = _layout(tmp_path)
    meta = cc.capture_after_cell(layout, model="m7", start_ms=0, end_ms=1, window_ms=30_000,
                                 redis_client=None, flush_wait_s=0)
    assert meta["errors"] and layout.meta_path.exists()

    class Down(FakeRedis):
        def smembers(self, key):
            raise ConnectionError("redis down")

        def time(self):
            raise ConnectionError("redis down")

    down = Down()
    start = cc.cell_clock_mark(down, "m7", CFG, sleep=lambda s: None)
    assert start["check"]["ok"] is False and "redis error" in start["check"]["reasons"][0]
    meta = cc.capture_after_cell(layout, model="m7", start_ms=0, end_ms=1, window_ms=30_000,
                                 redis_client=down, clock_start=start, flush_wait_s=0, sleep=lambda s: None)
    assert any("no redis-time span" in e for e in meta["errors"])
    assert "gateway_redis_dump" not in meta and not (layout.cell_dir / cc.BACKFILL_MARKER).exists()


def test_reader_resolves_a_legacy_run_without_cells(tmp_path: Path) -> None:
    model_dir = tmp_path / "run2" / "dsqwen-7b"
    stem = "dsqwen-7b_M_ladder_c1000008_a1"
    _write_legacy(model_dir, stem, "i0_o0_c1000008")
    got = cc.resolve_cell_artifacts(model_dir, stem)
    assert got["layout_version"] is None and got["cell_id"] == "i0_o0_c1000008"
    assert got["requests_jsonl"] == model_dir / "raw" / stem / "i0_o0_c1000008.jsonl"
    assert got["windows_csv"] == model_dir / f"{stem}.csv"
    assert got["vllm_metrics"] == {} and got["controller_ticks"] is None
    void_stem = "dsqwen-7b_M_ladder_c1000009_a1"
    _write_legacy(model_dir, void_stem, "i0_o0_c1000009", void=True)
    assert cc.resolve_cell_artifacts(model_dir, void_stem)["requests_jsonl"].name.endswith(".jsonl.void")


def test_the_rewindow_never_takes_capture_files_for_cells(tmp_path: Path) -> None:
    assert rewindow_from_raw.CAPTURE_DIRNAME == cc.CELLS_DIRNAME
    model_dir = tmp_path / "m"
    _write_legacy(model_dir, "m_S1_hold_c1_a1", "i0_o0_c1")
    layout = cc.CellLayout(model_dir, "m_S1_hold_c1_a1", "i0_o0_c1")
    layout.vllm_metrics_dir.mkdir(parents=True)
    layout.vllm_metrics_path("default/p").write_text("{}\n")
    layout.controller_ticks_path.write_text("{}\n")
    # the campaign's raw root: unchanged
    kept, _ = rewindow_from_raw.discover_cell_files(model_dir / "raw")
    assert [p.name for p in kept] == ["i0_o0_c1.jsonl"]
    # a raw root pointed at the run directory still sees no capture file as a cell
    kept, _ = rewindow_from_raw.discover_cell_files(model_dir)
    assert [p.name for p in kept] == ["i0_o0_c1.jsonl"]


# ---------------------------------------------------------------- run manifest


def test_run_manifest_is_written_once_with_code_registry_and_images(tmp_path: Path) -> None:
    reg = tmp_path / "registry.yaml"
    reg.write_text("models: {}\n")
    (tmp_path / "plan.json").write_text("{}\n")
    calls = []

    def run(argv, **kw):
        calls.append(argv)
        if argv[0] == "git":
            out = {"rev-parse": "abc123\n" if "HEAD" == argv[-1] and "--abbrev-ref" not in argv else "feat\n",
                   "status": ""}[argv[3]]
            return SimpleNamespace(returncode=0, stdout=out)
        doc = {"items": [{"metadata": {"name": "tre-v2-controller"},
                          "spec": {"template": {"spec": {"containers": [{"name": "c", "image": "ctl:1"}]}}}}]}
        return SimpleNamespace(returncode=0, stdout=json.dumps(doc))

    pods = [{"key": "default/p", "node": "n1", "containers": [{"name": "vllm", "image": "vllm:x", "image_id": "sha"}]}]
    path = cc.write_run_manifest(tmp_path, model="m7", config={"window_ms": 30000}, registry_path=reg,
                                 repo=tmp_path, model_pods=pods, control_namespace="ctl-ns", run=run)
    doc = json.loads(path.read_text())
    assert doc["layout_version"] == 1 and doc["model"] == "m7"
    assert doc["code"]["commit"] == "abc123" and doc["code"]["dirty"] is False
    assert doc["registry"]["sha256"] == cc.file_sha256(reg)
    assert doc["images"]["model_pods"][0]["containers"][0]["image"] == "vllm:x"
    assert doc["images"]["control_plane"]["deployments"]["tre-v2-controller"][0]["image"] == "ctl:1"
    assert doc["campaign_documents"]["plan.json"]["sha256"] == cc.file_sha256(tmp_path / "plan.json")
    assert doc["config"] == {"window_ms": 30000}
    assert any(a[:3] == ["kubectl", "-n", "ctl-ns"] for a in calls)
    assert cc.write_run_manifest(tmp_path, model="m7", config={}, registry_path=reg, repo=tmp_path,
                                 run=run) is None  # never overwritten


def test_discover_pod_targets_reads_names_nodes_and_images() -> None:
    doc = {"items": [
        {"metadata": {"name": "p1", "namespace": "default"}, "spec": {"nodeName": "n9", "containers": [
            {"name": "vllm-openai", "image": "vllm-openai-tre:1"}]},
         "status": {"podIP": "10.0.0.5", "containerStatuses": [{"name": "vllm-openai", "imageID": "docker://sha"}]}},
        {"metadata": {"name": "p0"}, "spec": {}, "status": {}},  # no IP yet: skipped
    ]}
    seen = {}

    def run(argv, **kw):
        seen["argv"] = argv
        return SimpleNamespace(stdout=json.dumps(doc))

    got = cc.discover_pod_targets("m7", "default", 8000, run=run)
    assert seen["argv"][-3:] == ["model.aibrix.ai/name=m7,tre.aibrix.io/routable=true", "-o", "json"]
    assert got == [{"key": "default/p1", "name": "p1", "namespace": "default", "ip": "10.0.0.5", "node": "n9",
                    "url": "http://10.0.0.5:8000/metrics",
                    "containers": [{"name": "vllm-openai", "image": "vllm-openai-tre:1",
                                    "image_id": "docker://sha"}]}]


# ------------------------------------------------------------- driver wiring


def test_campaign_cells_capture_next_to_the_online_csv_unless_disabled() -> None:
    cell = campaign.Cell(model="dsqwen-7b", shape="S1", primitive="hold", cell_id="i0_o0_c1", schedule="s.json",
                         duration_s=60.0, capacity_rps=1.0)

    class Args:
        gateway_url = "http://gw"
        raw_dir = "/r"
        window_ms = 30000
        fit_step_ms = 10000
        instant_sample_ms = 1000
        model_namespace = "default"
        guard_mode = "warn"
        min_slo_windows = 3
        out_dir = "/o"
        max_model_error_rate = 0.01
        ttft_slo_ms = 500.0
        tpot_slo_ms = 75.0
        registry = None
        redis_url = None
        controller_namespace = "ctl"

    command = campaign.cell_command(cell, Args(), Path("s.json"), Path("/o/m7/m7_S1_hold_a1.csv"))
    i = command.index("--capture-dir")
    assert Path(command[i + 1]) == Path("/o/m7/cells")
    assert command[command.index("--control-namespace") + 1] == "ctl"
    # the run checked the clocks once; a cell that fails its own check is marked, not refused
    assert command[command.index("--clock-domain-check") + 1] == "flag"
    off = Args()
    off.no_capture_extras = True
    assert "--capture-dir" not in campaign.cell_command(cell, off, Path("s.json"), Path("/o/x.csv"))


def test_r3_grid_capture_flags_and_layout() -> None:
    base = ["--model", "m7", "--gateway-url", "http://gw", "--output", "/o/m7/stem_a1.csv",
            "--schedule", "s.json", "--raw-dir", "/o/m7/raw", "--capture-dir", "/o/m7/cells"]
    args = r3_grid.parse_args(base)
    assert args.vllm_keyframe_every == cc.DEFAULT_KEYFRAME_EVERY
    assert args.gateway_flush_wait_s == cc.DEFAULT_CAPTURE_FLUSH_WAIT_S == 0
    assert args.clock_domain_check == "refuse"  # a standalone r3_grid cell is its own run
    cfg = r3_grid.capture_clock_config(args)
    assert cfg.margin == 52_000 and cfg.controller_lag_bounds == (-2_000, 31_500)
    assert r3_grid.capture_clock_config(r3_grid.parse_args(base + ["--capture-margin-ms", "60000"])).margin == 60_000
    with pytest.raises(SystemExit):  # a margin that can cut the first / last window is refused up front
        r3_grid.parse_args(base + ["--capture-margin-ms", "45000"])
    assert not (args.no_gateway_dump or args.no_controller_ticks or args.no_vllm_metrics_capture)
    layout = r3_grid.capture_layout_for(args, "i0_o0_c1")
    assert layout.cell_dir == Path("/o/m7/cells/stem_a1")
    assert layout.raw_dir == Path("/o/m7/raw/stem_a1")
    plain = r3_grid.parse_args(["--model", "m7", "--gateway-url", "http://gw", "--output", "/o/x.csv"])
    assert r3_grid.capture_layout_for(plain, "i0_o0_c1") is None


def test_capture_config_is_json_safe() -> None:
    args = SimpleNamespace(a=Path("/x"), b=1, c=[1, 2], d=None)
    assert json.loads(json.dumps(r3_grid._capture_config(args))) == {"a": str(Path("/x")), "b": 1, "c": [1, 2],
                                                                      "d": None}


# ------------------------------------------------------------------ the clock


def test_range_and_tail_are_in_redis_time() -> None:
    assert cc.dump_range_ms(960_000, 990_000, 40_000) == (920_000, 1_030_000)
    # without a blind spot the tail is the last 10 s grid window end over the cell
    assert cc.tail_ms(990_000, 30_000, 0) == 1_010_000
    assert cc.tail_ms(991_608, 30_000, 0) == 1_020_000
    # the default margin and tail also cover a gateway offset inside the check's blind spot
    assert CFG.margin == 52_000 and CFG.blind_spot_ms == 12_000
    assert cc.tail_ms(994_000, 30_000, CFG.blind_spot_ms) == 1_030_000
    assert CFG.gateway_write_delay_bounds == (-2_000, 12_000)
    assert CFG.controller_lag_bounds == (-2_000, 31_500)
    with pytest.raises(ValueError):  # below window + round + blind spot
        cc.ClockDomainConfig(window_ms=30_000, margin_ms=51_999)


def _controller_at(lag: int, *, end: int = 950_000) -> Cluster:
    """An in-sync gateway, and a controller whose newest window end is ``end``, ``lag``
    behind redis TIME."""
    c = Cluster(controller=False)
    c.advance(end + lag)
    for b in (end - 10_000, end):
        c.r.zadd(decision_hist_key("m7"), {_tick(b): b})
    return c


@pytest.mark.parametrize("controller_lag", [2_000, 5_000, 8_000, 11_600, 14_000, 17_000, 24_500, 29_500])
def test_a_controller_trailing_by_its_normal_lag_is_in_the_domain(controller_lag: int) -> None:
    # 2-17 s at the base read offset (2 s) and the 5 s rescue loop; up to 29.5 s with an
    # adapted offset (9.5 s) and only the 10 s fairness loop writing
    c = _controller_at(controller_lag)
    v = cc.check_clock_domains(c.r, "m7", CFG, sleep=c.sleep)
    assert v["ok"] and v["attempts"] == 1 and v["controller"]["lag_ms"] == controller_lag and v["reasons"] == []


@pytest.mark.parametrize("controller_lag", [-60_000, 40_000, 160_000])  # static: it stays off over the retries
def test_a_controller_ahead_or_far_behind_is_not(controller_lag: int) -> None:
    c = _controller_at(controller_lag)
    v = cc.check_clock_domains(c.r, "m7", CFG, sleep=c.sleep)
    assert not v["ok"] and v["gateway"]["ok"] and not v["controller"]["ok"]
    assert v["attempts"] == CFG.attempts and "controller" in v["reasons"][0]


@pytest.mark.parametrize("ends", [(928_731, 958_731), (930_000, 960_000)])
def test_a_free_running_controller_is_not(ends) -> None:
    # free_running stamps window ends with the controller's own clock: off the 10 s grid
    # (sliding) or on it but one 30 s window apart (tumbling)
    c = Cluster(controller=False)
    c.advance(964_000)
    for b in ends:
        c.r.zadd(decision_hist_key("m7"), {_tick(b): b})
    v = cc.check_clock_domains(c.r, "m7", CFG, sleep=c.sleep)
    assert not v["controller"]["ok"] and "gateway rounds" in v["reasons"][0]


def test_a_gateway_ticker_later_than_the_controller_retries_is_refused() -> None:
    # documented limit: past ~9.5 s the phase-aligned controller never sees its windows fresh
    c = Cluster(gw_phase=9_900)
    c.advance(RS)
    v = cc.check_clock_domains(c.r, "m7", CFG, sleep=c.sleep)
    assert v["gateway"]["ok"] and not v["controller"]["ok"]


@pytest.mark.parametrize("write_delay", [0, 5_000, 9_000, 9_400])
def test_any_gateway_ticker_phase_is_in_the_domain(write_delay: int) -> None:
    # the ticker starts where the gateway process started: any write delay the controller
    # can follow (up to period - retry = 9.5 s)
    c = Cluster(gw_phase=write_delay)
    c.advance(RS)
    for _ in range(12):  # probes all over the round
        v = cc.check_clock_domains(c.r, "m7", CFG, sleep=c.sleep)
        assert v["ok"] and v["attempts"] == 1, v
        assert write_delay <= v["gateway"]["write_delay_ms"] <= write_delay + 250
        c.advance(c.r.now_ms + 1_300)


def test_a_gateway_inside_the_blind_spot_still_lands_in_the_range_and_before_the_tail(tmp_path: Path) -> None:
    # 11 s fast with a 9.9 s ticker delay: indistinguishable from an in-sync late ticker
    c = Cluster(gw_skew=11_000, gw_phase=9_900)
    meta = _run_cell(c, _layout(tmp_path))
    assert meta["clock"]["domain"] == {"gateway": "ok", "controller": "ok"}
    lo, hi = meta["clock"]["range_ms"]
    during = [stamp for stamp, at in c.writes if RS - 30_000 <= at <= RE]
    first_after = min(stamp for stamp, at in c.writes if at >= RE)  # the round holding the cell's end
    assert all(lo <= stamp <= hi for stamp in during + [first_after])
    # the last controller window that holds that round ends before the tail
    assert first_after + 30_000 - 10_000 <= meta["clock"]["tail_ms"] <= hi


@pytest.mark.parametrize("gw_skew", [160_000, -160_000])
def test_a_gateway_160_s_off_is_refused_before_the_run(gw_skew: int, monkeypatch) -> None:
    c = Cluster(gw_skew=gw_skew)
    c.advance(RS)
    v = cc.check_clock_domains(c.r, "m7", CFG, sleep=c.sleep)
    assert not v["gateway"]["ok"] and abs(abs(v["gateway"]["write_delay_ms"]) - 160_000) <= 10_000
    with pytest.raises(cc.ClockDomainMismatch, match="gateway"):
        cc.require_clock_domains(c.r, ["m7"], CFG, sleep=c.sleep)
    # the campaign's pre-flight (every entry point calls it before driving)
    import redis

    monkeypatch.setattr(redis.Redis, "from_url", classmethod(lambda cls, url: c.r))
    args = SimpleNamespace(models="m7", window_ms=30_000, redis_url=None)
    monkeypatch.setattr(cc.time, "sleep", c.sleep)
    with pytest.raises(SystemExit, match="refusing to run"):
        campaign.require_capture_clock_domains(args)
    assert campaign.require_capture_clock_domains(SimpleNamespace(no_capture_extras=True)) is None


def test_the_campaign_pre_flight_passes_an_in_sync_cluster(monkeypatch) -> None:
    import redis

    c = Cluster()
    c.advance(RS)
    monkeypatch.setattr(redis.Redis, "from_url", classmethod(lambda cls, url: c.r))
    monkeypatch.setattr(cc.time, "sleep", c.sleep)
    got = campaign.require_capture_clock_domains(SimpleNamespace(models="m7", window_ms=30_000, redis_url=None))
    assert got["m7"]["ok"]


@pytest.mark.parametrize("gw_skew", [160_000, -160_000])
def test_a_gateway_160_s_off_marks_the_cell_and_never_completes_it(tmp_path: Path, gw_skew: int) -> None:
    layout = _layout(tmp_path)
    c = Cluster(gw_skew=gw_skew)
    meta = _run_cell(c, layout)
    assert meta["clock"]["domain"] == {"gateway": "clock_domain_mismatch", "controller": "clock_domain_mismatch"}
    assert meta["clock_domain_mismatch"]["dumps"] == ["controller_ticks", "gateway_redis_dump"]
    assert any("gateway" in r for r in meta["clock_domain_mismatch"]["reasons"])
    # dumped unshifted, in redis time
    assert meta["gateway_redis_dump"]["range_ms"] == [RS - 52_000, RE + 52_000]
    c.advance(1_400_000)
    cc.backfill_pending(layout.model_dir, c.r, sleep=c.sleep)
    on_disk = json.loads(layout.meta_path.read_text())
    for name in ("gateway_redis_dump", "controller_ticks"):
        assert on_disk[name]["complete"] is False, name
        assert on_disk[name]["clock_domain"] == cc.CLOCK_DOMAIN_MISMATCH
    assert on_disk["clock_domain_mismatch"]


@pytest.mark.parametrize("driver_skew", [160_000, -160_000])
def test_a_skewed_driver_does_not_move_the_redis_range(tmp_path: Path, driver_skew: int) -> None:
    ref = _run_cell(Cluster(), _layout(tmp_path / "ref"))
    layout = _layout(tmp_path / "skewed")
    c = Cluster()
    meta = _run_cell(c, layout, driver_skew=driver_skew)
    assert 0 <= meta["start_ms"] - driver_skew - RS <= 15_000  # the driver's own clock is recorded ...
    assert meta["clock"]["range_ms"] == ref["clock"]["range_ms"] == [RS - 52_000, RE + 52_000]  # ... not used
    assert meta["gateway_redis_dump"]["docs"] == ref["gateway_redis_dump"]["docs"]
    assert meta["controller_ticks"]["members"] == ref["controller_ticks"]["members"]
    assert meta["clock"]["domain"] == {"gateway": "ok", "controller": "ok"}
    probe = meta["clock"]["cell_start"]["probe"]
    assert probe["redis_minus_local_ms"] == -driver_skew  # audit only
    c.advance(1_040_000)
    done = cc.backfill_pending(layout.model_dir, c.r, sleep=c.sleep)
    assert done[0]["complete"] is True


def test_a_second_gateway_writer_behind_redis_fails_the_cell(tmp_path: Path) -> None:
    layout = _layout(tmp_path)
    c = Cluster()
    c.advance(RS)
    start = cc.cell_clock_mark(c.r, "m7", CFG, now_ms=lambda: c.r.now_ms, sleep=c.sleep)
    assert start["check"]["ok"]
    c.advance(RS + 10_000)
    # another writer 30 s behind redis: its newest stamp never beats the in-sync one ...
    c.r.zadd(inst_key("default/p"), {json.dumps({"timestamp": 940_000, "writer": "b"}): 940_000})
    c.advance(RE)
    end = cc.cell_clock_mark(c.r, "m7", CFG, start=start, now_ms=lambda: c.r.now_ms, sleep=c.sleep)
    # ... but it wrote a doc stamped before the cell while the cell ran
    assert not end["check"]["gateway"]["ok"] and end["check"]["gateway"]["late_writes"] == 1
    meta = cc.capture_after_cell(layout, model="m7", start_ms=RS, end_ms=RE, window_ms=30_000, redis_client=c.r,
                                 clock_start=start, clock_end=end, clock_config=CFG, flush_wait_s=0)
    assert meta["gateway_redis_dump"]["clock_domain"] == cc.CLOCK_DOMAIN_MISMATCH
    assert meta["controller_ticks"]["clock_domain"] == cc.CLOCK_DOMAIN_MISMATCH


def test_a_late_doc_after_the_end_mark_fails_the_capture_dump(tmp_path: Path) -> None:
    c = Cluster()
    c.advance(RS)
    start = cc.cell_clock_mark(c.r, "m7", CFG, now_ms=lambda: c.r.now_ms, sleep=c.sleep)
    c.advance(RE)
    end = cc.cell_clock_mark(c.r, "m7", CFG, start=start, now_ms=lambda: c.r.now_ms, sleep=c.sleep)
    assert end["check"]["ok"]
    # a writer ~70 s behind redis: nothing of it in the range during the cell, one doc now
    c.r.zadd(inst_key("default/p"), {json.dumps({"timestamp": 925_000, "writer": "b"}): 925_000})
    meta = cc.capture_after_cell(_layout(tmp_path), model="m7", start_ms=RS, end_ms=RE, window_ms=30_000,
                                 redis_client=c.r, clock_start=start, clock_end=end, clock_config=CFG,
                                 flush_wait_s=0)
    gd = meta["gateway_redis_dump"]
    assert gd["late_writes"] == 1 and gd["clock_domain"] == cc.CLOCK_DOMAIN_MISMATCH
    assert meta["controller_ticks"]["clock_domain"] == cc.CLOCK_DOMAIN_MISMATCH
    assert any("capture dump" in r for r in meta["clock_domain_mismatch"]["reasons"])


def test_a_late_doc_before_the_backfill_fails_the_backfilled_dumps(tmp_path: Path) -> None:
    layout = _layout(tmp_path)
    c = Cluster()
    meta = _run_cell(c, layout)
    assert meta["gateway_redis_dump"]["late_writes"] == 0 and meta["gateway_redis_dump"]["clock_domain"] == "ok"
    c.advance(1_040_000)
    c.r.zadd(hist_key("default/p"), {json.dumps({"timestamp": 970_000, "writer": "b"}): 970_000})
    done = cc.backfill_pending(layout.model_dir, c.r, sleep=c.sleep)
    assert done[0]["complete"] is False
    on_disk = json.loads(layout.meta_path.read_text())
    assert on_disk["gateway_redis_dump"]["late_writes"] == 1
    for name in ("gateway_redis_dump", "controller_ticks"):
        assert on_disk[name]["clock_domain"] == cc.CLOCK_DOMAIN_MISMATCH and not on_disk[name]["complete"]


def test_the_end_mark_is_taken_by_the_capture_when_not_given(tmp_path: Path) -> None:
    c = Cluster()
    c.advance(RS)
    start = cc.cell_clock_mark(c.r, "m7", CFG, now_ms=lambda: c.r.now_ms, sleep=c.sleep)
    c.advance(RE)
    meta = cc.capture_after_cell(_layout(tmp_path), model="m7", start_ms=RS, end_ms=RE, window_ms=30_000,
                                 redis_client=c.r, clock_start=start, flush_wait_s=0, now_ms=lambda: c.r.now_ms,
                                 sleep=c.sleep)
    assert meta["clock"]["redis_end_ms"] == RE and meta["clock"]["domain"]["gateway"] == "ok"


# ------------------------------------------------------------------ backfill


def test_a_dump_cut_at_the_cell_end_is_marked_and_backfilled_later(tmp_path: Path) -> None:
    layout = _layout(tmp_path)
    c = Cluster()
    meta = _run_cell(c, layout)
    assert meta["controller_ticks"]["complete"] is False and meta["gateway_redis_dump"]["complete"] is False
    assert meta["controller_ticks"]["tail_ms"] == 1_030_000
    assert cc.pending_cells(layout.model_dir) == [layout.cell_dir]
    c.advance(1_040_000)  # the controller and the gateway catch up
    done = cc.backfill_pending(layout.model_dir.parent, c.r, sleep=c.sleep)  # a run directory works too
    assert len(done) == 1 and done[0]["complete"] is True and done[0]["clock_check"]["ok"]
    assert done[0]["controller_ticks"]["members_after"] > done[0]["controller_ticks"]["members_before"]
    assert not (layout.cell_dir / cc.BACKFILL_MARKER).exists() and cc.pending_cells(layout.model_dir) == []
    on_disk = json.loads(layout.meta_path.read_text())
    assert on_disk["controller_ticks"]["complete"] and on_disk["gateway_redis_dump"]["complete"]
    assert on_disk["gateway_redis_dump"]["flush_wait"]["reason"] == "wait disabled"  # kept from the first dump
    assert len(on_disk["backfills"]) == 1
    rows = layout.controller_ticks_path.read_text().splitlines()[1:]
    ends = [json.loads(x)["window_end_ms"] for x in rows]
    assert ends[0] >= RS - 52_000 and ends[-1] == 1_030_000
    assert json.loads(rows[0])["tss_raw_source"] == cc.TSS_RAW_FROM_CONTROLLER
    assert cc.backfill_pending(layout.model_dir, c.r, sleep=c.sleep) == []  # nothing left


def test_the_backfill_lists_the_pods_again(tmp_path: Path) -> None:
    layout = _layout(tmp_path)
    c = Cluster()
    _run_cell(c, layout)
    c.pods.append("default/q")  # a pod that came up in the cell's tail
    c.advance(1_040_000)
    done = cc.backfill_pending(layout.model_dir, c.r, sleep=c.sleep)
    assert done[0]["gateway_docs"]["pods_before"] == 1 and done[0]["gateway_docs"]["pods_after"] == 2
    on_disk = json.loads(layout.meta_path.read_text())
    assert set(on_disk["gateway_redis_dump"]["files"]["inst"]) == {"default/p", "default/q"}
    assert on_disk["gateway_redis_dump"]["complete"] is True


def test_a_pod_first_writing_between_the_end_mark_and_the_dump_is_no_late_write(tmp_path: Path) -> None:
    layout = _layout(tmp_path)
    c = Cluster()
    c.advance(RS)
    start = cc.cell_clock_mark(c.r, "m7", CFG, now_ms=lambda: c.r.now_ms, sleep=c.sleep)
    c.advance(RE)
    end = cc.cell_clock_mark(c.r, "m7", CFG, start=start, now_ms=lambda: c.r.now_ms, sleep=c.sleep)
    c.pods.append("default/q")  # comes up now: first doc in the flush wait
    meta = cc.capture_after_cell(layout, model="m7", start_ms=RS, end_ms=RE, window_ms=30_000, redis_client=c.r,
                                 clock_start=start, clock_end=end, clock_config=CFG, flush_wait_s=12,
                                 now_ms=lambda: c.r.now_ms, sleep=c.sleep)
    gd = meta["gateway_redis_dump"]
    assert "default/q" in gd["pods"] and gd["late_writes"] == 0 and gd["clock_domain"] == "ok"
    c.advance(1_060_000)
    done = cc.backfill_pending(layout.model_dir, c.r, sleep=c.sleep)
    assert done[0]["complete"] is True


def test_every_campaign_entry_point_runs_the_clock_pre_flight() -> None:
    import inspect

    from scripts import (calibration_acceptance, calibration_ladder, calibration_supplement, calibration_t14,
                         calibration_training_supplement)

    for module in (calibration_acceptance, calibration_ladder, calibration_supplement, calibration_t14,
                   calibration_training_supplement):
        src = inspect.getsource(module)
        mode = src.index("campaign.controller_mode(args.controller_namespace)")
        pre = src.index("campaign.require_capture_clock_domains(args)")
        assert mode < pre < src.index("drive = drive or", mode), module.__name__
    src = inspect.getsource(campaign)
    for fn in (campaign.run_campaign, campaign.run_reprobe):
        body = inspect.getsource(fn)
        assert "require_capture_clock_domains(args" in body, fn.__name__


def test_a_cell_longer_than_the_gateway_retention_is_never_complete(tmp_path: Path) -> None:
    layout = _layout(tmp_path)
    c = Cluster()
    meta = _run_cell(c, layout, end_at=RS + 30 * 60 * 1000)
    assert meta["gateway_redis_dump"]["head_within_retention"] is False
    c.advance(RS + 32 * 60 * 1000)
    cc.backfill_pending(layout.model_dir, c.r, sleep=c.sleep)
    on_disk = json.loads(layout.meta_path.read_text())
    assert on_disk["gateway_redis_dump"]["reached_tail"] and not on_disk["gateway_redis_dump"]["complete"]
    assert on_disk["controller_ticks"]["complete"]  # kept ~24 h


def test_a_backfill_after_a_clock_step_marks_the_redumped_sources(tmp_path: Path) -> None:
    layout = _layout(tmp_path)
    c = Cluster()
    _run_cell(c, layout)
    c.gw_skew = 160_000  # the gateway's node steps its clock after the cell
    c.advance(1_040_000)
    done = cc.backfill_pending(layout.model_dir, c.r, sleep=c.sleep)
    assert not done[0]["clock_check"]["ok"] and done[0]["complete"] is False
    on_disk = json.loads(layout.meta_path.read_text())
    assert on_disk["gateway_redis_dump"]["clock_domain"] == cc.CLOCK_DOMAIN_MISMATCH
    assert on_disk["controller_ticks"]["clock_domain"] == cc.CLOCK_DOMAIN_MISMATCH
    assert any(r.startswith("backfill: gateway") for r in on_disk["clock_domain_mismatch"]["reasons"])


def test_a_backfill_never_shrinks_a_dump(tmp_path: Path) -> None:
    layout = _layout(tmp_path)
    c = Cluster()
    _run_cell(c, layout, gateway_dump=False)
    before = layout.controller_ticks_path.read_text()
    c.r.zsets[decision_hist_key("m7")].clear()  # retention trimmed it
    rec = cc.backfill_cell(layout.cell_dir, c.r, sleep=c.sleep)
    assert "controller_ticks_kept" in rec and rec["complete"] is False
    assert layout.controller_ticks_path.read_text() == before


def test_the_final_backfill_waits_for_the_controller_bounded(tmp_path: Path) -> None:
    layout = _layout(tmp_path)
    c = Cluster()
    _run_cell(c, layout, gateway_dump=False)
    slept = []

    def sleep(dt):
        slept.append(dt)
        c.advance(c.r.now_ms + int(dt * 1000))

    cc.backfill_pending(layout.model_dir, c.r, wait_s=90, sleep=sleep, monotonic=lambda: sum(slept))
    # waited until the tail (1_030_000) plus the controller's largest normal lag (31.5 s)
    assert c.r.now_ms >= 1_061_500 and max(slept) <= 5.0 and sum(slept) <= 90
    assert json.loads(layout.meta_path.read_text())["controller_ticks"]["complete"] is True


def test_old_markers_are_left_to_the_cli(tmp_path: Path) -> None:
    layout = _layout(tmp_path)
    layout.cell_dir.mkdir(parents=True)
    (layout.cell_dir / cc.BACKFILL_MARKER).write_text("x\n")
    assert cc.pending_cells(layout.model_dir, now=lambda: 1e12) == []
    assert cc.pending_cells(layout.model_dir, max_age_s=None, now=lambda: 1e12) == [layout.cell_dir]


def test_finalize_run_backfills_pending_cells_with_the_given_or_default_redis(tmp_path: Path, monkeypatch) -> None:
    import redis

    calls, urls = [], []
    pending = {"cells": []}
    monkeypatch.setattr(cc, "pending_cells", lambda root, **kw: pending["cells"])
    monkeypatch.setattr(cc, "backfill_pending", lambda root, client, **kw: calls.append((root, kw)) or [])
    monkeypatch.setattr(redis.Redis, "from_url", classmethod(lambda cls, url: urls.append(url) or object()))
    monkeypatch.setattr("scripts.calibration_dataset.build_dataset", lambda d: d)
    campaign.finalize_run(tmp_path, status="complete", exit_code=0)
    assert calls == []  # nothing pending: redis is not touched
    pending["cells"] = [tmp_path]
    campaign.finalize_run(tmp_path, status="complete", exit_code=0, redis_url="redis://10.0.0.1:6379/0")
    campaign.finalize_run(tmp_path, status="complete", exit_code=0)
    assert urls == ["redis://10.0.0.1:6379/0", cc.DEFAULT_REDIS_URL]
    assert calls[0][1]["wait_s"] == cc.DEFAULT_FINAL_BACKFILL_WAIT_S


def test_active_pods_skips_pods_long_gone() -> None:
    r = FakeRedis()
    _seed_gateway(r, ["default/live"], [990_000])
    _seed_gateway(r, ["default/gone"], [100_000])
    assert cc.active_pods(r, cc.model_pod_keys(r, "m7"), 900_000) == ["default/live"]


def test_a_rewrite_with_as_many_but_different_rows_is_not_a_superset(tmp_path: Path) -> None:
    layout = _layout(tmp_path)
    c = Cluster()
    _run_cell(c, layout, gateway_dump=False)
    key = decision_hist_key("m7")
    dumped = [kv for kv in c.r.zsets[key].items() if kv[1] >= RS - 52_000]
    c.r.zsets[key].pop(min(dumped, key=lambda kv: kv[1])[0])  # the oldest dumped row trimmed ...
    c.advance(RE + 10_000)  # ... while the tail arrived
    rec = cc.backfill_cell(layout.cell_dir, c.r, sleep=c.sleep)
    assert "controller_ticks_kept" in rec


def test_a_late_sample_after_close_is_dropped_not_written(tmp_path: Path) -> None:
    url = "http://10.0.0.1:8000/metrics"
    rec = cc.VllmMetricsRecorder(tmp_path / "v", {url: "default/p"})
    rec.record(url, 1, _body())
    rec.close()
    rec.record(url, 2, _body())
    rec.record_error(url, 3, "late")
    lines = (tmp_path / "v" / "default_p.jsonl").read_text().splitlines()
    assert len(lines) == 2 and rec.late_after_close == 2


def test_seals_never_cover_the_mutable_capture(tmp_path: Path) -> None:
    from scripts import calibration_acceptance, calibration_t14

    out = tmp_path / "m"
    (out / "raw" / "s_a1").mkdir(parents=True)
    (out / "raw" / "s_a1" / "c.jsonl").write_text("{}\n")
    (out / "cells.jsonl").write_text("{}\n")
    (out / "manifest.json").write_text("{}\n")
    cell = out / "cells" / "s_a1"
    cell.mkdir(parents=True)
    (cell / "cell_meta.json").write_text("{}\n")
    (cell / "controller_ticks.jsonl").write_text("{}\n")

    def names(files):
        return sorted(Path(f).resolve().relative_to(out.resolve()).as_posix() for f in files)

    want = ["cells.jsonl", "manifest.json", "raw/s_a1/c.jsonl"]
    assert names(calibration_acceptance.sealed_files(out, {})) == want
    assert names(calibration_t14.sealed_files(out, out / "raw", [])) == want


def test_capture_config_redacts_url_credentials() -> None:
    args = SimpleNamespace(redis_url="redis://user:s3cret@10.0.0.1:6379/0", gateway_url="http://gw:80/v1",
                           pod_endpoint=["http://a:b@10.0.0.2:8000/metrics"])
    cfg = r3_grid._capture_config(args)
    assert cfg["redis_url"] == "redis://***@10.0.0.1:6379/0" and cfg["gateway_url"] == "http://gw:80/v1"
    assert cfg["pod_endpoint"] == ["http://***@10.0.0.2:8000/metrics"]


def test_run_manifest_hashes_the_live_registry_configmap(tmp_path: Path) -> None:
    import hashlib

    def run(argv, **kw):
        if argv[0] == "git":
            return SimpleNamespace(returncode=1, stdout="")
        if "configmap" in argv:
            return SimpleNamespace(returncode=0, stdout=json.dumps({"data": {"registry.yaml": "models: {}\n"}}))
        return SimpleNamespace(returncode=0, stdout="{}")

    path = cc.write_run_manifest(tmp_path, model="m7", config={}, registry_path=None, repo=tmp_path,
                                 control_namespace="ns", registry_configmap="tre-v2-registry", run=run)
    live = json.loads(path.read_text())["registry"]["live_configmap"]
    assert live["name"] == "tre-v2-registry"
    assert live["data_sha256"]["registry.yaml"] == hashlib.sha256(b"models: {}\n").hexdigest()
