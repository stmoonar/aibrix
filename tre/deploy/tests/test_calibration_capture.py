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


# ----------------------------------------------------------- cell meta and layout


def _write_legacy(model_dir: Path, stem: str, cell_id: str, *, void: bool = False) -> None:
    raw = model_dir / "raw" / stem
    raw.mkdir(parents=True)
    (raw / (f"{cell_id}.jsonl" + (".void" if void else ""))).write_text('{"cell_id": "x"}\n')
    (raw / f"{cell_id}.instant.jsonl").write_text('{"ts_ms": 1}\n')
    (raw / f"{cell_id}.guard.json").write_text("{}\n")
    (raw / f"{cell_id}.rps.csv").write_text("a\n")
    (model_dir / f"{stem}.csv").write_text("scenario_id\n")


def test_capture_after_cell_writes_the_cell_meta_and_the_reader_resolves_it(tmp_path: Path) -> None:
    layout = _layout(tmp_path)
    _write_legacy(layout.model_dir, layout.stem, layout.cell_id)
    r = FakeRedis(now_ms=1_000_000)
    _seed_gateway(r, ["default/p"], [930_000, 960_000, 990_000])
    r.zadd(decision_hist_key("m7"), {json.dumps({"window_end_ms": 990_000, "y_m": 1.0, "q_ctl": 1.0}): 990_000})
    url = "http://10.0.0.1:8000/metrics"
    rec = cc.VllmMetricsRecorder(layout.vllm_metrics_dir, {url: "default/p"}, model="m7")
    rec.record(url, 960_000, _body())
    meta = cc.capture_after_cell(
        layout, model="m7", start_ms=960_000, end_ms=990_000, window_ms=30_000, redis_client=r,
        targets=[{"key": "default/p", "url": url}], recorder=rec, flush_wait_s=0,
        now_ms=lambda: 1_000_000, info={"guard_voided": False},
    )
    assert meta["errors"] == []
    on_disk = json.loads(layout.meta_path.read_text())
    assert on_disk["layout_version"] == cc.CAPTURE_LAYOUT_VERSION and on_disk["guard_voided"] is False
    assert on_disk["legacy"]["requests_jsonl"] == "../../raw/m7_S1_hold_c1000001_a1/i0_o0_c1000001.jsonl"
    # the dump reaches one window + one gateway round before the cell starts
    assert on_disk["gateway_redis_dump"]["range_ms"][0] == 960_000 - 30_000 - 10_000
    assert on_disk["clock"]["probe_before"]["redis_minus_local_ms"] == 0
    assert on_disk["clock"]["gateway"]["shift_ms"] == 0 and on_disk["clock"]["controller"]["shift_ms"] == 0
    assert "clock_skew_suspected" not in on_disk
    # cut at the cell end: the controller has not processed the tail windows yet
    assert on_disk["controller_ticks"]["complete"] is False and on_disk["controller_ticks"]["tail_ms"] == 1_010_000
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

    meta = cc.capture_after_cell(layout, model="m7", start_ms=0, end_ms=1, window_ms=30_000,
                                 redis_client=Down(), flush_wait_s=0)
    assert any("gateway_redis_dump" in e for e in meta["errors"])


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
    off = Args()
    off.no_capture_extras = True
    assert "--capture-dir" not in campaign.cell_command(cell, off, Path("s.json"), Path("/o/x.csv"))


def test_r3_grid_capture_flags_and_layout() -> None:
    args = r3_grid.parse_args(["--model", "m7", "--gateway-url", "http://gw", "--output", "/o/m7/stem_a1.csv",
                               "--schedule", "s.json", "--raw-dir", "/o/m7/raw",
                               "--capture-dir", "/o/m7/cells"])
    assert args.vllm_keyframe_every == cc.DEFAULT_KEYFRAME_EVERY
    assert args.gateway_flush_wait_s == cc.DEFAULT_GATEWAY_FLUSH_WAIT_S
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


# ------------------------------------------------------------------ backfill


def _tick(end: int) -> str:
    return json.dumps({"window_end_ms": end, "ts": end, "trs": 1.0, "trs_raw": 2.0, "y_m": 2.0, "q_ctl": 1.0})


def test_tail_is_the_last_grid_window_end_that_still_overlaps_the_cell() -> None:
    assert cc.tail_ms(990_000, 30_000) == 1_010_000
    assert cc.tail_ms(991_608, 30_000) == 1_020_000
    assert cc.dump_range_ms(960_000, 990_000, 30_000) == (920_000, 1_030_000)


def test_a_dump_cut_at_the_cell_end_is_marked_and_backfilled_later(tmp_path: Path) -> None:
    layout = _layout(tmp_path)
    r = FakeRedis(now_ms=995_000)
    _seed_gateway(r, ["default/p"], [950_000, 960_000, 970_000, 980_000, 990_000])
    for end in (970_000, 980_000, 990_000):
        r.zadd(decision_hist_key("m7"), {_tick(end): end})
    meta = cc.capture_after_cell(layout, model="m7", start_ms=960_000, end_ms=990_000, window_ms=30_000,
                                 redis_client=r, flush_wait_s=0, now_ms=lambda: 995_000)
    assert meta["controller_ticks"]["complete"] is False and meta["gateway_redis_dump"]["complete"] is False
    assert meta["controller_ticks"]["tail_ms"] == 1_010_000
    assert cc.pending_cells(layout.model_dir) == [layout.cell_dir]
    # the controller and the gateway catch up
    _seed_gateway(r, ["default/p"], [1_000_000, 1_010_000])
    for end in (1_000_000, 1_010_000):
        r.zadd(decision_hist_key("m7"), {_tick(end): end})
    r.now_ms = 1_040_000
    done = cc.backfill_pending(layout.model_dir.parent, r)  # a run directory works too
    assert len(done) == 1 and done[0]["complete"] is True
    assert done[0]["controller_ticks"] == {"members_before": 3, "members_after": 5}
    assert not (layout.cell_dir / cc.BACKFILL_MARKER).exists() and cc.pending_cells(layout.model_dir) == []
    on_disk = json.loads(layout.meta_path.read_text())
    assert on_disk["controller_ticks"]["complete"] and on_disk["gateway_redis_dump"]["complete"]
    assert on_disk["gateway_redis_dump"]["flush_wait"]["reason"] == "wait disabled"  # kept from the first dump
    assert len(on_disk["backfills"]) == 1
    rows = layout.controller_ticks_path.read_text().splitlines()[1:]
    assert [json.loads(x)["window_end_ms"] for x in rows] == [970_000, 980_000, 990_000, 1_000_000, 1_010_000]
    assert json.loads(rows[0])["tss_raw_source"] == cc.TSS_RAW_FROM_CONTROLLER
    assert cc.backfill_pending(layout.model_dir, r) == []  # nothing left


def test_a_backfill_never_shrinks_a_dump(tmp_path: Path) -> None:
    layout = _layout(tmp_path)
    r = FakeRedis()
    for end in (970_000, 980_000):
        r.zadd(decision_hist_key("m7"), {_tick(end): end})
    cc.capture_after_cell(layout, model="m7", start_ms=960_000, end_ms=990_000, window_ms=30_000,
                          redis_client=r, flush_wait_s=0, gateway_dump=False, now_ms=lambda: r.now_ms)
    before = layout.controller_ticks_path.read_text()
    r.zsets[decision_hist_key("m7")].clear()  # retention trimmed it
    rec = cc.backfill_cell(layout.cell_dir, r)
    assert "controller_ticks_kept" in rec and rec["complete"] is False
    assert layout.controller_ticks_path.read_text() == before


def test_the_final_backfill_waits_for_the_controller_bounded(tmp_path: Path) -> None:
    layout = _layout(tmp_path)
    r = FakeRedis(now_ms=995_000)
    cc.capture_after_cell(layout, model="m7", start_ms=960_000, end_ms=990_000, window_ms=30_000,
                          redis_client=r, flush_wait_s=0, gateway_dump=False, now_ms=lambda: r.now_ms)
    slept = []

    def sleep(dt):
        slept.append(dt)
        r.now_ms += int(dt * 1000)

    clock = {"t": 0.0}

    def monotonic():
        clock["t"] = sum(slept)
        return clock["t"]

    cc.backfill_pending(layout.model_dir, r, wait_s=90, sleep=sleep, monotonic=monotonic)
    # waited until two rounds past the tail (1_010_000 + 20_000), in steps of <= 5 s
    assert r.now_ms >= 1_030_000 and max(slept) <= 5.0 and sum(slept) <= 90


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


# ------------------------------------------------------------ clocks and review fixes


def test_source_clock_in_sync_is_not_shifted() -> None:
    c = cc.source_clock(990_000, 996_000, redis_minus_local_ms=3.0, write_phase_ms=1_700)
    assert c["shift_ms"] == 0 and c["suspect"] is None and c["lag_ms"] == 6_000


def test_source_clock_detects_a_source_ahead_by_160_s() -> None:
    # a gateway on a node 160 s fast stamps its rounds 160 s ahead of redis TIME
    c = cc.source_clock(1_160_000, 1_001_500, redis_minus_local_ms=0.0)
    assert c["suspect"] and abs(c["shift_ms"] - 160_000) <= 5_000
    # the same from the write phase of a live gateway (seen 158.3 s *before* its stamp)
    c = cc.source_clock(None, None, redis_minus_local_ms=0.0, write_phase_ms=-158_300)
    assert c["suspect"] and abs(c["shift_ms"] - 160_000) <= 5_000


def test_source_clock_adds_redis_offset_and_flags_a_stale_source() -> None:
    c = cc.source_clock(1_160_000, 1_165_000, redis_minus_local_ms=160_000.0)
    assert c["shift_ms"] == 160_000 and c["suspect"]  # redis and source on the fast node: both off the driver
    c = cc.source_clock(700_000, 1_000_000, redis_minus_local_ms=0.0)
    assert c["shift_ms"] == 0 and "stale" in c["suspect"]


def test_a_skewed_gateway_gets_a_shifted_dump_range(tmp_path: Path) -> None:
    layout = _layout(tmp_path)
    r = FakeRedis(now_ms=1_000_000)
    skew = 160_000
    _seed_gateway(r, ["default/p"], [s + skew for s in (930_000, 960_000, 990_000)])
    meta = cc.capture_after_cell(layout, model="m7", start_ms=960_000, end_ms=990_000, window_ms=30_000,
                                 redis_client=r, flush_wait_s=0, now_ms=lambda: 1_000_000,
                                 controller_ticks=False)
    gw = meta["clock"]["gateway"]
    assert meta["clock_skew_suspected"]["gateway"] and abs(gw["shift_ms"] - skew) <= 5_000
    # all three rounds (in the gateway's clock) are inside the shifted range
    assert sum(meta["gateway_redis_dump"]["docs"]["inst"].values()) == 3
    assert meta["gateway_redis_dump"]["tail_ms"] == cc.tail_ms(990_000, 30_000, gw["shift_ms"]) + 10_000


def test_active_pods_skips_pods_long_gone() -> None:
    r = FakeRedis()
    _seed_gateway(r, ["default/live"], [990_000])
    _seed_gateway(r, ["default/gone"], [100_000])
    assert cc.active_pods(r, cc.model_pod_keys(r, "m7"), 900_000) == ["default/live"]


def test_the_write_phase_is_measured_once_per_gateway_instance_set(tmp_path: Path) -> None:
    from tre_common.rediskeys import GW_INSTANCES_KEY

    r = FakeRedis(now_ms=1_000_000)
    _seed_gateway(r, ["default/p"], [980_000, 990_000])
    r.zadd(GW_INSTANCES_KEY, {"gw-a": 1.0})
    clock = {"t": 0.0}

    def sleep(dt):
        clock["t"] += dt
        if clock["t"] >= 1.0:
            r.zadd(inst_key("default/p"), {"new": 1_000_000})
            r.now_ms = 1_001_700

    first = cc.capture_after_cell(_layout(tmp_path), model="m7", start_ms=960_000, end_ms=990_000,
                                  window_ms=30_000, redis_client=r, sleep=sleep,
                                  monotonic=lambda: clock["t"], now_ms=lambda: 1_001_700,
                                  controller_ticks=False)
    assert first["gateway_redis_dump"]["flush_wait"]["write_phase_ms"] == 1_700
    second_layout = cc.CellLayout(tmp_path / "run" / "m7", "m7_S1_hold_c1000002_a1", "i0_o0_c1000002")
    second = cc.capture_after_cell(second_layout, model="m7", start_ms=990_000, end_ms=1_000_000,
                                   window_ms=30_000, redis_client=r, now_ms=lambda: 1_002_000,
                                   sleep=lambda dt: pytest.fail("waited although the phase is cached"),
                                   controller_ticks=False)
    assert second["gateway_redis_dump"]["flush_wait"]["skipped"]
    assert second["gateway_redis_dump"]["flush_wait"]["write_phase_ms"] == 1_700
    r.zadd(GW_INSTANCES_KEY, {"gw-b": 2.0})  # a gateway restart: measure again
    assert cc.cached_phase(second_layout.cell_dir.parent, cc.gateway_instances(r), now_ms=1_002_000) is None


def test_a_rewrite_with_as_many_but_different_rows_is_not_a_superset(tmp_path: Path) -> None:
    layout = _layout(tmp_path)
    r = FakeRedis()
    for end in (970_000, 980_000):
        r.zadd(decision_hist_key("m7"), {_tick(end): end})
    cc.capture_after_cell(layout, model="m7", start_ms=960_000, end_ms=990_000, window_ms=30_000,
                          redis_client=r, flush_wait_s=0, gateway_dump=False, now_ms=lambda: r.now_ms)
    r.zsets[decision_hist_key("m7")].pop(_tick(970_000))  # the oldest trimmed ...
    r.zadd(decision_hist_key("m7"), {_tick(990_000): 990_000})  # ... while the tail arrived
    rec = cc.backfill_cell(layout.cell_dir, r)
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


@pytest.mark.parametrize(
    "case, latest, redis_now, rml, phase, expected_lag, want_shift",
    [
        # redis 160 s ahead of the driver, controller in sync with the driver: no shift
        ("redis skewed, controller in sync", 990_000, 990_000 + 11_600 + 160_000, 160_000.0, None, 12_000, 0),
        # controller 160 s ahead, redis = driver
        ("controller ahead 160 s", 990_000 + 160_000, 1_001_600, 0.0, None, 12_000, 160_000),
        # controller 8 s ahead, redis = driver
        ("controller ahead 8 s", 990_000 + 8_000, 1_001_600, 0.0, None, 12_000, 8_000),
        # gateway in sync with the driver, redis 160 s ahead, phase measured
        ("gateway in sync, redis skewed", 990_000, 991_500 + 160_000, 160_000.0, 1_500 + 160_000, 5_000, 0),
    ],
)
def test_source_clock_is_judged_against_the_driver(case, latest, redis_now, rml, phase, expected_lag,
                                                    want_shift) -> None:
    c = cc.source_clock(latest, redis_now, redis_minus_local_ms=rml, write_phase_ms=phase,
                        expected_lag_ms=expected_lag, max_lag_ms=cc.CONTROLLER_MAX_LAG_MS)
    assert abs(c["shift_ms"] - want_shift) <= 1_000, (case, c)
    assert (c["suspect"] is None) == (want_shift == 0), (case, c)
