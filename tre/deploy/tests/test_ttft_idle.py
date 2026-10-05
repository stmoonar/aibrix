"""Idle TTFT(L) capture + fit (scripts.ttft_idle_capture / scripts.ttft_idle_fit)."""
from __future__ import annotations

import json
import random
from pathlib import Path

import yaml

from scripts import ttft_idle_capture as cap
from scripts import ttft_idle_fit as fit

REGISTRY = Path(__file__).resolve().parents[1] / "registry.yaml"
C, B = 40.0, 0.06


def _rec(send, length, ttft, e2e=None, **extra):
    e2e = ttft + 200.0 if e2e is None else e2e
    return {"send_ts_ms": send, "done_ts_ms": send + e2e, "ttft_ms": ttft, "input_tokens": length,
            "expected_prompt_tokens": length, "http_status": 200, **extra}


def _serial(lengths, per, *, gap_ms=2000.0, seed=1):
    rng = random.Random(seed)
    order = cap.block_order(lengths, per, seed)
    return [_rec(k * gap_ms, n, C + B * n + rng.gauss(0, 2.0)) for k, n in enumerate(order)]


def _write(path: Path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")


def test_fit_recovers_known_line_on_serial_capture(tmp_path):
    lengths = [128, 256, 512, 1024, 2048, 3072, 4096]
    recs = _serial(lengths, 30)
    recs[5]["ttft_ms"] += 400.0  # one outlier: Huber must not follow it
    _write(tmp_path / "m" / "raw" / "idle" / "i0_o16_c1.jsonl", recs)
    out = fit.fit_model(tmp_path / "m")
    assert out["n"] == len(recs)
    assert abs(out["c_ms"] - C) < 2.0 and abs(out["b_ms_per_token"] - B) < 0.002
    assert set(out["per_length"]) == {str(n) for n in lengths}
    assert out["inputs"][0]["sha256"] and out["inputs"][0]["kept"] == len(recs)


def test_non_isolated_and_excluded_requests_never_reach_the_fit(tmp_path):
    recs = _serial([256, 1024, 2048], 10)
    t = recs[-1]["send_ts_ms"] + 5000.0
    bad = [
        _rec(t, 1024, 900.0, e2e=3000.0),           # in flight when the next one is sent
        _rec(t + 100.0, 1024, 900.0),               # sent while another is in flight
        _rec(t + 10000.0, 2048, 900.0, e2e=950.0),  # the next one is sent inside its prefill
        _rec(t + 10500.0, 2048, 900.0),
        _rec(t + 20000.0, 256, 900.0, http_status=500),
        {**_rec(t + 30000.0, 256, 900.0), "expected_prompt_tokens": 300},
    ]
    _write(tmp_path / "m" / "raw" / "idle" / "c.jsonl", recs + bad)
    # Warmup and held-out cells are skipped wholesale, whatever they contain.
    _write(tmp_path / "m" / "raw" / "warmup" / "c.jsonl", [_rec(0.0, 256, 5000.0)])
    _write(tmp_path / "m" / "raw" / "m_M_steps" / "c.jsonl", [_rec(0.0, 256, 5000.0)])
    out = fit.fit_model(tmp_path / "m")
    assert out["n"] == len(recs)
    assert out["n_files"] == 1
    assert max(v["median_ms"] for v in out["per_length"].values()) < C + B * 2048 + 10
    assert out["dropped"]["inflight"] >= 2 and out["dropped"]["overlap_prefill"] >= 1
    assert out["dropped"]["not_served"] == 1 and out["dropped"]["prompt_mismatch"] == 1


def test_write_registry_changes_only_the_two_idle_fields(tmp_path):
    for m in ("dsqwen-7b", "dsqwen-14b"):
        _write(tmp_path / "root" / m / "raw" / "idle" / "c.jsonl", _serial([256, 1024, 2048, 4096], 8))
    reg = tmp_path / "registry.yaml"
    before = REGISTRY.read_text(encoding="utf-8")
    reg.write_text(before, encoding="utf-8")
    assert fit.main(["--root", str(tmp_path / "root"), "--models", "dsqwen-7b,dsqwen-14b",
                     "--out", str(tmp_path / "fit.json"), "--write-registry", str(reg)]) == 0
    after = reg.read_text(encoding="utf-8")
    changed = [(a, b) for a, b in zip(before.splitlines(), after.splitlines()) if a != b]
    assert len(before.splitlines()) == len(after.splitlines()) and len(changed) == 4
    assert all(b.split(":")[0] == a.split(":")[0] and b.split(":")[0].strip() in (fit.C_FIELD, fit.B_FIELD)
               for a, b in changed)
    slo = {e["name"]: e["slo"] for e in yaml.safe_load(after)["models"]}
    old = {e["name"]: e["slo"] for e in yaml.safe_load(before)["models"]}
    for m in ("dsqwen-7b", "dsqwen-14b"):
        assert abs(slo[m][fit.C_FIELD] - C) < 2.0 and abs(slo[m][fit.B_FIELD] - B) < 0.002
    assert slo["dsllama-8b"] == old["dsllama-8b"]


def test_capture_schedule_sends_one_request_per_slot_in_block_order(tmp_path):
    from tre_replayer.engine.schedule import build_deterministic_schedule
    from tre_replayer.traces.loader import load_trace_segments

    lengths = [128, 512, 4096]
    order = cap.block_order(lengths, 4, seed=7)
    for k in range(4):  # every block holds every length once
        assert sorted(order[3 * k:3 * k + 3]) == lengths
    path = tmp_path / "idle.json"
    path.write_text(json.dumps(cap.build_trace("m", order, gap_s=1.5, max_tokens=16)), encoding="utf-8")
    events = build_deterministic_schedule(load_trace_segments(path))
    assert [e.prompt_tokens for e in events] == order
    assert all(e.max_output_tokens == 16 for e in events)
    offsets = [e.scheduled_offset_s for e in events]
    assert all(abs((b - a) - 1.5) < 1e-6 for a, b in zip(offsets, offsets[1:]))
