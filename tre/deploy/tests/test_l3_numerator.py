"""L3: the TSS numerator from the pods' vLLM token counters (scripts.l3_numerator), built
into a standard dataset by ``calibration_dataset --numerator vllm_counter`` and kept apart
from the gateway numerator by dline_refit (trainset / freeze / accept)."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

import test_calibration_dataset as dsbase  # the standard-dataset fixture run
import test_dline_freeze_accept as base  # the D22 freeze / accept fixture world
from scripts import calibration_capture as capture
from scripts import calibration_dataset as dataset
from scripts import dline_refit as dl
from scripts import l3_numerator as l3

PROMPT_RATE, GEN_RATE = 1024.0, 512.0  # tokens per second of one pod


def _pod(samples, errors=()) -> l3.PodCounters:
    return l3.PodCounters("ns/p", [(t, p, g) for t, p, g in samples], list(errors))


def _steady(t0: int, t1: int, *, reset_at: int | None = None, skip=()) -> list:
    """1 Hz samples 1 ms after each second; counters grow at the fixed rates (from 0
    again after ``reset_at``)."""
    out = []
    for t in range(t0 + 1, t1, 1000):
        if t in skip:
            continue
        base_t = t0 if reset_at is None or t < reset_at else reset_at
        k = (t - base_t) // 1000  # whole seconds counted: integer counters, like vLLM's
        out.append((t, PROMPT_RATE * k, GEN_RATE * k))
    return out


def test_window_deltas_sum_pods_and_void_resets_holes_and_failed_scrapes() -> None:
    a, b = _pod(_steady(0, 120_000)), _pod(_steady(0, 120_000))
    src = l3.CounterTokenSource([a, b])
    # (30 s, 60 s]: boundary samples at 29.001 / 59.001 s -> exactly 30 s of counting, x 2 pods
    assert src(30_000, 60_000) == (2 * 30 * PROMPT_RATE, 2 * 30 * GEN_RATE)
    reset = _pod(_steady(0, 120_000, reset_at=45_500))
    hole = _pod(_steady(0, 120_000, skip={40_001, 41_001}))
    long_hole = _pod(_steady(0, 120_000, skip=set(range(40_001, 52_001, 1000))))
    failed = _pod(_steady(0, 120_000), errors=[50_500])
    assert reset.window_delta(30_000, 60_000)[0] == l3.STATUS_RESET
    assert reset.window_delta(60_000, 90_000)[0] == l3.STATUS_OK  # after the reset: fine again
    # 2026-10-28: a short interior hole or a failed scrape does not bias the delta (only
    # the boundary samples are read, as online); a hole long enough to hide a restart does.
    assert hole.window_delta(30_000, 60_000) == (l3.STATUS_OK, 30 * PROMPT_RATE, 30 * GEN_RATE)
    assert failed.window_delta(30_000, 60_000)[0] == l3.STATUS_OK
    assert long_hole.window_delta(30_000, 60_000)[0] == l3.STATUS_GAP
    assert a.window_delta(130_000, 160_000)[0] == l3.STATUS_NO_SAMPLE  # past the capture
    src = l3.CounterTokenSource([a, reset])
    assert src(30_000, 60_000) is None  # one void pod voids the model's window
    assert src(60_000, 90_000) == (60 * PROMPT_RATE, 60 * GEN_RATE)
    assert src.counts == {l3.STATUS_RESET: 1, l3.STATUS_OK: 1}
    assert l3.CounterTokenSource([a], missing_pods=["ns/gone"])(60_000, 90_000) is None


def _linear(times, prompt_rate=PROMPT_RATE, gen_rate=GEN_RATE) -> list:
    """Samples at ``times`` of counters growing exactly at the rates (per second)."""
    return [(t, prompt_rate * t / 1000.0, gen_rate * t / 1000.0) for t in times]


@pytest.mark.parametrize("phase", [1, 250, 999])
def test_window_value_is_the_rate_over_the_observed_span_as_a_window_total(phase) -> None:
    # the online rule: (v(end sample) - v(start sample)) * window / span of the two samples
    times = [phase + 1000 * k for k in range(60)]
    late = [t for t in times if t != max(t for t in times if t <= 40_000)]  # end sample 1-2 s old
    for samples in (times, late):
        pod = _pod(_linear(samples))
        status, p, g = pod.window_delta(20_000, 40_000)
        assert status == l3.STATUS_OK
        # a constant rate gives rate x window whatever the sample phase or a stale end sample
        assert p == pytest.approx(20 * PROMPT_RATE) and g == pytest.approx(20 * GEN_RATE)
    t0 = max(t for t in late if t <= 20_000)
    t1 = max(t for t in late if t <= 40_000)
    assert t1 - t0 != 20_000  # the raw delta alone would be off by the span error
    # one sample serves both boundaries (window shorter than the sample spacing): no rate
    assert _pod(_linear([0, 1000])).window_delta(100, 900)[0] == l3.STATUS_NO_SAMPLE


def test_a_missing_counter_or_a_reset_voids_the_window_never_zero() -> None:
    samples = _linear(range(1, 60_000, 1000))
    missing = _pod([(t, p, None if 25_000 < t < 27_000 else g) for t, p, g in samples])
    src = l3.CounterTokenSource([missing])
    assert src(20_000, 40_000) is None and src.counts[l3.STATUS_MISSING_SERIES] == 1
    assert src(40_000, 60_000) is not None
    reset = _pod([(t, (p if t < 30_000 else p - 30 * PROMPT_RATE), g) for t, p, g in samples])
    assert reset.window_delta(20_000, 40_000)[0] == l3.STATUS_RESET


def test_registry_stamp_of_each_numerator() -> None:
    assert l3.registry_numerator(l3.NUMERATOR_VLLM_COUNTER) == "step_counters"
    assert l3.registry_numerator(l3.NUMERATOR_GATEWAY) == "completion_histograms"
    with pytest.raises(ValueError):
        l3.registry_numerator("bogus")


def _write_pod_file(path: Path, samples) -> None:
    """A vllm_metrics_1hz file the way the capture writes it (header + delta rows)."""
    rec = capture.VllmMetricsRecorder(path.parent, {"http://x/metrics": path.stem}, model="m")
    for t, p, g in samples:
        rec.record("http://x/metrics", t,
                   f'vllm:prompt_tokens_total{{engine="0",model_name="m"}} {p}\n'
                   f'vllm:generation_tokens_total{{engine="0",model_name="m"}} {g}\n')
    rec.close()


def _l3_run(tmp_path: Path) -> Path:
    root = tmp_path / "run"
    campaign_dir = dsbase._campaign(root, "dsqwen-7b")
    start = dsbase.START
    vdir = campaign_dir / capture.CELLS_DIRNAME / "dsqwen-7b_S1_steps" / capture.VLLM_METRICS_DIRNAME
    _write_pod_file(vdir / "default_pod-a.jsonl", _steady(start - 15_000, start + 125_000))
    _write_pod_file(vdir / "default_pod-b.jsonl",
                    _steady(start - 15_000, start + 125_000, reset_at=start + 65_500))
    return root


def _rows(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def test_the_l3_dataset_changes_only_the_numerator_and_counts_what_it_drops(tmp_path) -> None:
    root = _l3_run(tmp_path)
    gw = dataset.build_dataset(root)
    out = dataset.build_dataset(root, overrides={"numerator": l3.NUMERATOR_VLLM_COUNTER})
    assert gw == root / "dataset" and out == root / "dataset_l3"
    gw_rows, l3_rows = _rows(gw / "windows.csv"), _rows(out / "windows.csv")
    assert list(gw_rows[0]) == dataset.WINDOW_COLUMNS  # the default is unchanged
    assert json.loads((gw / "manifest.json").read_text())["numerator"]["source"] == "gateway"

    s1 = [r for r in l3_rows if r["cell_id"] == "i256_o128_c95"]
    by_key = {(r["cell_id"], r["attempt"], r["window_start_ms"]): r for r in gw_rows}
    assert s1
    for r in s1:
        g = by_key[(r["cell_id"], r["attempt"], r["window_start_ms"])]
        # same window, labels and queue; the numerator is two pods x 30 s of counting
        for col in ("window_end_ms", "slo_label", "slo_label_fixed", "avg_running", "avg_waiting"):
            assert r[col] == g[col], col
        assert float(r["prompt_tokens_total"]) == 2 * 30 * PROMPT_RATE
        assert float(r["generation_tokens_total"]) == 2 * 30 * GEN_RATE
        assert r["prompt_tokens_gateway"] == g["prompt_tokens_total"]
        assert r["numerator_source"] == l3.NUMERATOR_VLLM_COUNTER
        assert not (int(r["window_start_ms"]) < dsbase.START + 65_500 <= int(r["window_end_ms"]))
    # the reset's windows are dropped and counted; cells with no capture drop every window
    man = json.loads((out / "manifest.json").read_text())
    s1_cell = next(c for c in man["cells"] if c["cell_id"] == "i256_o128_c95")
    s1_gw = [r for r in gw_rows if r["cell_id"] == "i256_o128_c95"]
    assert s1_cell["numerator"]["windows_void"].get(l3.STATUS_RESET) == 3
    assert s1_cell["numerator"]["windows_kept"] == len(s1) == len(s1_gw) - 3 - s1_cell["numerator"][
        "windows_void"].get(l3.STATUS_NO_SAMPLE, 0)
    assert man["numerator"]["source"] == l3.NUMERATOR_VLLM_COUNTER
    assert man["numerator"]["windows_kept"] == len(l3_rows)
    assert man["numerator"]["windows_void"][l3.STATUS_NO_METRICS] > 0
    assert any("L3 numerator" in d for d in man["discrepancies"])


def test_trainset_refuses_datasets_of_two_numerators(tmp_path) -> None:
    root = _l3_run(tmp_path)
    gw = dataset.build_dataset(root)
    l3_dir = dataset.build_dataset(root, overrides={"numerator": l3.NUMERATOR_VLLM_COUNTER})
    sources = [dl.DatasetSource.parse(f"gw={gw}", sealed_to_h2=True),
               dl.DatasetSource.parse(f"l3={l3_dir}", sealed_to_h2=True)]
    with pytest.raises(dl.TrainingSetError, match="different TSS numerators"):
        dl.build_training_set(sources, tmp_path / "fit")


def test_an_l3_freeze_is_accepted_only_on_an_l3_m_dataset(tmp_path, capsys) -> None:
    w = base._world(tmp_path, train_manifest={"numerator": {"source": l3.NUMERATOR_VLLM_COUNTER}})
    assert base._freeze(w) == 0
    entry = dl.verify_freeze(w["freeze"])["models"][base.MODEL]
    assert entry["numerator"] == l3.NUMERATOR_VLLM_COUNTER
    assert entry["registry"]["numerator"] == "step_counters"
    man = base._seal(w)
    base._refused(w, man, capsys, "TSS numerator")  # M was built with the gateway numerator
    mman = w["mdir"] / "manifest.json"
    mman.write_text(json.dumps({**json.loads(mman.read_text()),
                                "numerator": {"source": l3.NUMERATOR_VLLM_COUNTER}}))
    assert base._accept(w, man) == 0


def _fit_window(w: dict, window_ms: float) -> None:
    """The fit stages' recorded windowing (final.json; the fixture's alpha / wp carry none)."""
    path = w["out"] / base.MODEL / base.ARM / "final.json"
    path.write_text(json.dumps({**json.loads(path.read_text()), "windowing": dl.windowing(window_ms=window_ms)}))


def test_the_freeze_takes_the_datasets_window_and_stamps_it(tmp_path, capsys) -> None:
    cut20 = {"windowing": {"window_ms": 20_000, "step_ms": 10_000}}
    w = base._world(tmp_path, train_manifest={"numerator": {"source": l3.NUMERATOR_VLLM_COUNTER}, **cut20})
    _fit_window(w, 30_000)  # the fit stages left at the 30 s default: refused, nothing written
    assert base._freeze(w) == dl.EXIT_REFUSED
    assert "rerun alpha / wp / final" in capsys.readouterr().out and not w["freeze"].exists()
    _fit_window(w, 20_000)
    assert base._freeze(w) == 0
    entry = dl.verify_freeze(w["freeze"])["models"][base.MODEL]
    assert entry["registry"]["theta_window_ms"] == 20_000 and entry["registry"]["numerator"] == "step_counters"
    # an M dataset cut with another window than the freeze's is refused at accept
    man = base._seal(w)
    mman = w["mdir"] / "manifest.json"
    mman.write_text(json.dumps({**json.loads(mman.read_text()), "numerator": {"source": l3.NUMERATOR_VLLM_COUNTER},
                                "windowing": {"window_ms": 30_000, "step_ms": 10_000}}))
    base._refused(w, man, capsys, "was cut with")


def test_trainset_refuses_datasets_of_two_windows(tmp_path) -> None:
    root = _l3_run(tmp_path)
    d30 = dataset.build_dataset(root)
    d20 = dataset.build_dataset(root, out_dir=tmp_path / "d20", overrides={"window_ms": 20_000})
    assert json.loads((d20 / "manifest.json").read_text())["windowing"]["window_ms"] == 20_000
    sources = [dl.DatasetSource.parse(f"a={d30}", sealed_to_h2=True),
               dl.DatasetSource.parse(f"b={d20}", sealed_to_h2=True)]
    with pytest.raises(dl.TrainingSetError, match="different windows"):
        dl.build_training_set(sources, tmp_path / "fit")
