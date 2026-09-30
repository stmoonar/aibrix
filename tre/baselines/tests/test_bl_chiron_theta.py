import json

import pytest

from tre_baselines.tools import chiron_theta as ct


def _write(tmp_path, name, obj, jsonl=False):
    p = tmp_path / name
    p.write_text("\n".join(json.dumps(x) for x in obj) if jsonl else json.dumps(obj))
    return p


def test_segment_trace_3x_spike(tmp_path):
    tr = {"a": [{"start_time": 0, "end_time": 30, "rps": 2},
                {"start_time": 30, "end_time": 60, "rps": 6},
                {"start_time": 60, "end_time": 90, "rps": 6}]}
    res = ct.compute_theta([_write(tmp_path, "t.json", tr)], interval_s=5)
    assert res["a"]["r"] == pytest.approx(3.0)
    assert res["a"]["theta"] == pytest.approx(1 / 3)


def test_jsonl_arrivals_3x_spike(tmp_path):
    rows = []
    for b in range(4):  # 10 arrivals per 5 s bin, then 30 per bin
        rows += [{"model": "m", "arrival_time": b * 5 + i * 0.4} for i in range(10)]
    for b in range(4, 8):
        rows += [{"model": "m", "arrival_time": b * 5 + i * 0.1} for i in range(30)]
    res = ct.compute_theta([_write(tmp_path, "t.jsonl", rows, jsonl=True)], interval_s=5)
    assert res["m"]["theta"] == pytest.approx(1 / 3, abs=1e-6)


def test_list_json_and_ms_unit(tmp_path):
    rows = [{"model": "m", "ts": t * 1000} for t in (0, 1, 2, 5, 6, 7, 8, 9, 10, 11)]
    res = ct.compute_theta([_write(tmp_path, "t.json", rows)], interval_s=5, time_field="ts",
                           time_scale=1e-3)
    assert "m" in res


def test_clamp_bounds(tmp_path):
    flat = {"a": [{"start_time": 0, "end_time": 50, "rps": 4}]}
    assert ct.compute_theta([_write(tmp_path, "f.json", flat)])["a"]["theta"] == 0.9
    huge = {"a": [{"start_time": 0, "end_time": 5, "rps": 1}, {"start_time": 5, "end_time": 10, "rps": 100}]}
    assert ct.compute_theta([_write(tmp_path, "h.json", huge)])["a"]["theta"] == 0.1


def test_multiple_traces_merge_per_model(tmp_path):
    a = _write(tmp_path, "a.json", {"a": [{"start_time": 0, "end_time": 10, "rps": 1}]})
    b = _write(tmp_path, "b.json", {"b": [{"start_time": 0, "end_time": 5, "rps": 1},
                                          {"start_time": 5, "end_time": 10, "rps": 2}]})
    res = ct.compute_theta([a, b])
    assert set(res) == {"a", "b"} and res["b"]["theta"] == pytest.approx(0.5)


def test_bad_inputs(tmp_path):
    with pytest.raises(ct.TraceFormatError, match="model field"):
        ct.compute_theta([_write(tmp_path, "x.json", [{"foo": 1}])])
    with pytest.raises(ct.TraceFormatError, match="arrival-time"):
        ct.compute_theta([_write(tmp_path, "y.json", [{"model": "m", "bar": 1}])])
    bad = tmp_path / "z.json"
    bad.write_text("not json\n{")
    with pytest.raises(ct.TraceFormatError):
        ct.compute_theta([bad])


def test_cli_prints_yaml(tmp_path, capsys):
    tr = {"a": [{"start_time": 0, "end_time": 5, "rps": 1}, {"start_time": 5, "end_time": 10, "rps": 3}]}
    assert ct.main(["--trace", str(_write(tmp_path, "t.json", tr)), "--interval-s", "5"]) == 0
    out = capsys.readouterr().out
    assert "theta:" in out and "a: 0.3333" in out
    assert ct.main(["--trace", str(tmp_path / "missing.json")]) == 2
