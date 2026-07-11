import csv
import json

try:  # in-tree layout: deploy/scripts/analysis/e5_timeline_auroc.py
    from deploy.scripts.analysis.e5_timeline_auroc import (
        AlignedWindow,
        analyze,
        auroc,
        lead_labels,
        parse_float,
    )
except ModuleNotFoundError:  # /tmp/e5_dev dev layout (flat module beside tests/)
    import os
    import sys

    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts_analysis"))
    from e5_timeline_auroc import (
        AlignedWindow,
        analyze,
        auroc,
        lead_labels,
        parse_float,
    )

TIMELINE_HEADER = ["ts", "model", "z_m", "queue_len", "decode_tps", "prefill_tps", "replicas_awake"]
VIOL_HEADER = ["model", "window_end_ms", "n_requests", "violated"]


def _write_csv(path, header, rows):
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        for row in rows:
            writer.writerow(row)


def _make_run(run_dir, timeline_rows, viol_rows):
    run_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(run_dir / "timeline_signals.csv", TIMELINE_HEADER, timeline_rows)
    _write_csv(run_dir / "violation_windows.csv", VIOL_HEADER, viol_rows)


def _mid_ts(window_end_ms):
    # A ts that falls inside (window_end_ms-5000, window_end_ms].
    return (window_end_ms - 2500) / 1000.0


def _read_summary_rows(out_dir):
    with (out_dir / "summary.csv").open(newline="") as handle:
        return list(csv.DictReader(handle))


# --- unit-level: parse_float / auroc / lead_labels -------------------------------


def test_parse_float_treats_empty_and_nan_as_missing():
    assert parse_float("") is None
    assert parse_float("  ") is None
    assert parse_float("nan") is None
    assert parse_float("NaN") is None
    assert parse_float("inf") is None
    assert parse_float("1.5") == 1.5


def test_auroc_single_class_is_none():
    assert auroc([(0.1, False), (0.2, False)]) is None
    assert auroc([(0.1, True), (0.2, True)]) is None


def test_auroc_constant_scores_is_half():
    pairs = [(0.5, True), (0.5, False), (0.5, True), (0.5, False)]
    assert auroc(pairs) == 0.5


def test_auroc_perfect_separation_is_one():
    # Higher score => positive.
    pairs = [(9.0, True), (8.0, True), (1.0, False), (2.0, False)]
    assert auroc(pairs) == 1.0


def test_lead_labels_shift():
    aligned = [
        AlignedWindow(1, False, 0, {}, False),
        AlignedWindow(2, False, 0, {}, False),
        AlignedWindow(3, True, 0, {}, False),
        AlignedWindow(4, False, 0, {}, False),
    ]
    labels = lead_labels(aligned, 1)
    assert labels == [False, True, False, None]
    # Horizon 2: idx0 sees w1(F),w2(T) => True; last window has no lookahead.
    assert lead_labels(aligned, 2) == [True, True, False, None]


# --- pipeline: perfect separation, direction, macro -----------------------------


def test_zm_direction_flip_gives_perfect_auroc(tmp_path):
    evidence = tmp_path / "evidence"
    # violated windows have LOW z_m (near violation); score = -z_m must flip it.
    windows = [
        (5000, True, 0.10),
        (10000, True, 0.20),
        (15000, False, 0.80),
        (20000, False, 0.90),
    ]
    timeline_rows = [
        [_mid_ts(end), "m", zm, 10 if viol else 1, 0, 0, 1]
        for end, viol, zm in windows
    ]
    viol_rows = [["m", end, 5, str(viol)] for end, viol, _ in windows]
    _make_run(evidence / "t1_tre_seed1", timeline_rows, viol_rows)

    out = tmp_path / "out"
    analyze(evidence, out)
    rows = _read_summary_rows(out)
    zm = next(r for r in rows if r["signal"] == "z_m" and r["run_id"] == "t1_tre_seed1")
    assert float(zm["auroc"]) == 1.0
    # queue_len also perfectly separating (violated => high queue).
    ql = next(r for r in rows if r["signal"] == "queue_len" and r["run_id"] == "t1_tre_seed1")
    assert float(ql["auroc"]) == 1.0


def test_constant_signal_half_and_single_class_null(tmp_path):
    evidence = tmp_path / "evidence"
    # constant z_m across both classes -> 0.5; but violated all False in run2 -> null.
    windows = [(5000, True), (10000, False), (15000, True), (20000, False)]
    timeline_rows = [[_mid_ts(end), "m", 0.5, 0.5, 0, 0, 1] for end, _ in windows]
    viol_rows = [["m", end, 5, str(viol)] for end, viol in windows]
    _make_run(evidence / "t1_tre_seed1", timeline_rows, viol_rows)

    no_viol = [(5000, False), (10000, False)]
    tl2 = [[_mid_ts(end), "m", 0.3, 0.3, 0, 0, 1] for end, _ in no_viol]
    vw2 = [["m", end, 5, "False"] for end, _ in no_viol]
    _make_run(evidence / "t2_tre_seed1", tl2, vw2)

    out = tmp_path / "out"
    summary = analyze(evidence, out)
    rows = _read_summary_rows(out)
    zm1 = next(r for r in rows if r["signal"] == "z_m" and r["run_id"] == "t1_tre_seed1")
    assert float(zm1["auroc"]) == 0.5
    zm2 = next(r for r in rows if r["signal"] == "z_m" and r["run_id"] == "t2_tre_seed1")
    assert zm2["auroc"] == ""  # single class -> null -> empty cell

    # __macro__ row for z_m averages only the non-null (0.5) value.
    macro = next(r for r in rows if r["run_id"] == "__macro__" and r["signal"] == "z_m")
    assert float(macro["auroc"]) == 0.5
    assert macro["n_windows"] == "1"  # one contributing non-null AUROC
    assert summary["meta"]["runs_matched"] == 2


def test_fallback_within_10s_and_missing_beyond(tmp_path):
    evidence = tmp_path / "evidence"
    # window A: no in-window row, nearest prior 6s before -> used (not missing).
    # window B: nearest prior 15s before -> missing.
    timeline_rows = [
        [94000 / 1000.0, "m", 0.4, 3, 0, 0, 1],   # 6s before end=100000
        [185000 / 1000.0, "m", 0.6, 7, 0, 0, 1],  # 15s before end=200000
    ]
    viol_rows = [
        ["m", 100000, 5, "True"],
        ["m", 200000, 5, "False"],
    ]
    _make_run(evidence / "t1_tre_seed1", timeline_rows, viol_rows)

    out = tmp_path / "out"
    analyze(evidence, out)
    with (out / "aligned_t1_tre_seed1.csv").open(newline="") as handle:
        aligned = list(csv.DictReader(handle))
    by_end = {int(r["window_end_ms"]): r for r in aligned}
    assert by_end[100000]["missing_signal"] == "False"
    assert by_end[100000]["z_m"] != ""
    assert by_end[200000]["missing_signal"] == "True"
    assert by_end[200000]["z_m"] == ""

    rows = _read_summary_rows(out)
    zm = next(r for r in rows if r["signal"] == "z_m" and r["run_id"] == "t1_tre_seed1")
    assert zm["n_missing_signal"] == "1"
    assert zm["n_windows"] == "2"


def test_lead_windows_column_and_labels(tmp_path):
    evidence = tmp_path / "evidence"
    # No window is itself violated, but a violation appears one window ahead.
    # With lead=1 the signal becomes predictive; base auroc is null (no positives).
    windows = [
        (5000, False, 0.10),   # predicts the coming violation (low z_m)
        (10000, True, 0.90),   # the violation window (high z_m, healthy signal)
        (15000, False, 0.90),
    ]
    timeline_rows = [[_mid_ts(end), "m", zm, 0, 0, 0, 1] for end, _, zm in windows]
    viol_rows = [["m", end, 5, str(viol)] for end, viol, _ in windows]
    _make_run(evidence / "t1_tre_seed1", timeline_rows, viol_rows)

    out = tmp_path / "out"
    analyze(evidence, out, lead_windows=3)
    header = _read_summary_rows(out)[0].keys()
    assert "auroc_lead3" in header

    rows = _read_summary_rows(out)
    zm = next(r for r in rows if r["signal"] == "z_m" and r["run_id"] == "t1_tre_seed1")
    # base: only one True window -> both classes present? violated True at idx1,
    # others False -> both classes exist, auroc computable but not the point.
    # lead=3: labels = [ (w1..True? next windows include the True) ] -> idx0 True.
    assert zm["auroc_lead3"] != ""


def test_summary_files_written_and_macro_present(tmp_path):
    evidence = tmp_path / "evidence"
    windows = [(5000, True, 0.1), (10000, False, 0.9)]
    timeline_rows = [[_mid_ts(end), "m", zm, 0, 0, 0, 1] for end, _, zm in windows]
    viol_rows = [["m", end, 5, str(viol)] for end, viol, zm in windows]
    _make_run(evidence / "t1_tre_seed1", timeline_rows, viol_rows)

    out = tmp_path / "out"
    analyze(evidence, out)
    assert (out / "summary.json").exists()
    assert (out / "summary.csv").exists()
    assert (out / "aligned_t1_tre_seed1.csv").exists()

    data = json.loads((out / "summary.json").read_text())
    assert "meta" in data and "rows" in data
    macro_signals = {r["signal"] for r in data["rows"] if r["run_id"] == "__macro__"}
    assert macro_signals == {"z_m", "queue_len", "decode_tps", "prefill_tps"}
