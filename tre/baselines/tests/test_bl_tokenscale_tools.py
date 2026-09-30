from __future__ import annotations

import json

import pytest
import yaml

from tre_baselines.tools import tokenscale_buckets as tb
from tre_baselines.tools import tokenscale_profile as tp


def test_buckets_on_synthetic_records(tmp_path, capsys):
    recs = [{"model": "m", "in_tokens": i, "max_tokens": o} for i in range(1, 10) for o in (10, 20, 30)]
    f = tmp_path / "t.jsonl"
    f.write_text("\n".join(json.dumps(r) for r in recs), encoding="utf-8")
    assert tb.main(["--trace", str(f), "--model", "m"]) == 0
    out = yaml.safe_load(capsys.readouterr().out)
    assert out["bucket_edges"]["m"] == {"in": [3, 6], "out": [10, 20]}
    c = out["bucket_centers"]["m"]
    assert c[0][0] == [2, 10] and c[2][2] == [8, 30]


def test_buckets_on_segment_trace_and_dist(tmp_path, capsys):
    trace = {"m": [
        {"start_time": 0, "end_time": 10, "rps": 1, "input_tokens": 100, "max_tokens": 10},
        {"start_time": 10, "end_time": 20, "rps": 1, "input_tokens": 500, "max_tokens": 100},
        {"start_time": 20, "end_time": 30, "rps": 1, "input_tokens_dist": {"low": 1000, "high": 4000},
         "max_tokens_dist": {"low": 100, "high": 400}},
    ]}
    f = tmp_path / "t.json"
    f.write_text(json.dumps(trace), encoding="utf-8")
    assert tb.main(["--trace", str(f)]) == 0
    out = yaml.safe_load(capsys.readouterr().out)
    assert out["bucket_edges"]["m"]["in"] == [100, 500]
    assert out["bucket_centers"]["m"][2][2] == [2000, 200]


def test_buckets_bad_format(tmp_path, capsys):
    f = tmp_path / "bad.json"
    f.write_text(json.dumps([{"foo": 1}]), encoding="utf-8")
    assert tb.main(["--trace", str(f)]) == 2
    assert "needs one of" in capsys.readouterr().err
    f.write_text("not json\n{", encoding="utf-8")
    assert tb.main(["--trace", str(f)]) == 2


def _flat_sender(rates):
    """Sender whose tok/s at concurrency 1,2,4,.. follows ``rates``; records the calls."""
    calls = []

    def send(model, tin, tout, c, step_s, warmup_s):
        calls.append(c)
        w = step_s - warmup_s
        return int(rates[len(calls) - 1] * w / (tin + tout)), w

    send.calls = calls
    return send


def test_step_sequence_and_stop_rule():
    # 100, 200, 300, +2 %, +1 % -> stops after the second consecutive <3 % step
    s = _flat_sender([1000, 2000, 3000, 3060, 3090, 9990, 9990])
    v, steps = tp.profile_point(s, "m", 10, 10, 60, 15)
    assert s.calls == [1, 2, 4, 8, 16] and v == pytest.approx(3090, rel=0.01)
    assert len(steps) == 5


def test_stop_rule_resets_on_improvement():
    s = _flat_sender([1000, 1010, 2000, 2010, 2010, 5000])
    v, _ = tp.profile_point(s, "m", 10, 10, 60, 15)
    assert s.calls == [1, 2, 4, 8, 16] and v == pytest.approx(2010, rel=0.01)


def test_max_concurrency_caps():
    s = _flat_sender([1000 * 2 ** k for k in range(20)])
    tp.profile_point(s, "m", 10, 10, 60, 15, max_concurrency=8)
    assert s.calls == [1, 2, 4, 8]


def _centers_file(tmp_path):
    centers = [[[100 * (i + 1), 10 * (j + 1)] for j in range(3)] for i in range(3)]
    f = tmp_path / "c.yaml"
    f.write_text(yaml.safe_dump({"bucket_centers": {"m": centers}}), encoding="utf-8")
    return f


def test_profile_dry_run_writes_velocity_and_csv(tmp_path, capsys):
    out = tmp_path / "o"
    assert tp.main(["--model", "m", "--centers", str(_centers_file(tmp_path)), "--out-dir", str(out), "--dry-run"]) == 0
    assert "estimated GPU time" in capsys.readouterr().out
    vel = yaml.safe_load((out / "velocity.yaml").read_text(encoding="utf-8"))["velocity"]["m"]
    assert len(vel["buckets"]) == 3 and all(len(r) == 3 and all(v > 0 for v in r) for r in vel["buckets"])
    assert 9000 < vel["buckets"][0][0] <= 10000 and vel["v_prefill"] > 20000
    rows = (out / "profile_raw.csv").read_text(encoding="utf-8").splitlines()
    assert rows[0].startswith("model,kind,bucket") and len(rows) > 10


def test_profile_refuses_real_send_without_approval(tmp_path, capsys):
    out = tmp_path / "o"
    rc = tp.main(["--model", "m", "--centers", str(_centers_file(tmp_path)), "--out-dir", str(out), "--sender", "stub"])
    assert rc == 3 and not out.exists()
    assert "--i-have-user-approval" in capsys.readouterr().err


def test_profile_requires_out_dir():
    with pytest.raises(SystemExit):
        tp.main(["--model", "m", "--centers", "x", "--dry-run"])


def test_load_sender_spec():
    assert callable(tp.load_sender("stub"))
    with pytest.raises(ValueError):
        tp.load_sender("nocolon")
