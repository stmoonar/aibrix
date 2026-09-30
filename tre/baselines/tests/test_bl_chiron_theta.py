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
    res = ct.compute_theta([_write(tmp_path, "t.json", tr)], interval_s=5, method="adjacent_p99")
    assert res["a"]["r"] == pytest.approx(3.0)
    assert res["a"]["theta"] == pytest.approx(1 / 3)
    # peak_mean over the same trace: mean (6 x 10 + 12 x 30) / 18 bins, peak 30
    res = ct.compute_theta([_write(tmp_path, "t.json", tr)], interval_s=5)
    assert res["a"]["method"] == "peak_mean" and res["a"]["peak"] == pytest.approx(30.0)
    assert res["a"]["theta"] == pytest.approx((60 + 360) / 18 / 30)


def test_jsonl_arrivals_3x_spike(tmp_path):
    rows = []
    for b in range(4):  # 10 arrivals per 5 s bin, then 30 per bin
        rows += [{"model": "m", "arrival_time": b * 5 + i * 0.4} for i in range(10)]
    for b in range(4, 8):
        rows += [{"model": "m", "arrival_time": b * 5 + i * 0.1} for i in range(30)]
    res = ct.compute_theta([_write(tmp_path, "t.jsonl", rows, jsonl=True)], interval_s=5,
                           method="adjacent_p99")
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
    assert ct.compute_theta([_write(tmp_path, "h.json", huge)], method="adjacent_p99")["a"]["theta"] == 0.1
    rare = {"a": [{"start_time": 0, "end_time": 10, "rps": 100}, {"start_time": 10, "end_time": 500, "rps": 0.1}]}
    assert ct.compute_theta([_write(tmp_path, "r.json", rare)])["a"]["theta"] == 0.1


def test_multiple_traces_merge_per_model(tmp_path):
    a = _write(tmp_path, "a.json", {"a": [{"start_time": 0, "end_time": 10, "rps": 1}]})
    b = _write(tmp_path, "b.json", {"b": [{"start_time": 0, "end_time": 5, "rps": 1},
                                          {"start_time": 5, "end_time": 10, "rps": 2}]})
    res = ct.compute_theta([a, b], method="adjacent_p99")
    assert set(res) == {"a", "b"} and res["b"]["theta"] == pytest.approx(0.5)
    assert ct.compute_theta([a, b])["b"]["theta"] == pytest.approx(7.5 / 10)


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
    path = str(_write(tmp_path, "t.json", tr))
    assert ct.main(["--trace", path, "--interval-s", "5", "--method", "adjacent_p99"]) == 0
    out = capsys.readouterr().out
    assert "theta:" in out and "a: 0.3333333333" in out and "method=adjacent_p99" in out
    assert ct.main(["--trace", path, "--interval-s", "5"]) == 0  # default peak_mean: 10 / 15
    out = capsys.readouterr().out
    assert "method=peak_mean" in out and "a: 0.6666666667" in out
    assert ct.main(["--trace", str(tmp_path / "missing.json")]) == 2


def test_peak_mean_paper_example_spike_3x(tmp_path):
    # a steady 2 rps with a 3x spike in 2 of 100 bins: theta ~ mean / peak ~ 1/3
    tr = {"a": [{"start_time": 0, "end_time": 490, "rps": 2}, {"start_time": 490, "end_time": 500, "rps": 6}]}
    res = ct.compute_theta([_write(tmp_path, "s.json", tr)], interval_s=5)["a"]
    assert res["peak"] == pytest.approx(30.0) and res["n_bins"] == 100
    assert res["mean"] == pytest.approx((98 * 10 + 2 * 30) / 100)
    assert res["theta"] == pytest.approx(10.4 / 30) and abs(res["theta"] - 1 / 3) < 0.02
    with pytest.raises(ValueError):
        ct.compute_theta([_write(tmp_path, "s.json", tr)], method="nope")
