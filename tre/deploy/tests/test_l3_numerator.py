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
    failed = _pod(_steady(0, 120_000), errors=[50_500])
    assert reset.window_delta(30_000, 60_000)[0] == l3.STATUS_RESET
    assert reset.window_delta(60_000, 90_000)[0] == l3.STATUS_OK  # after the reset: fine again
    assert hole.window_delta(30_000, 60_000)[0] == l3.STATUS_GAP
    assert failed.window_delta(30_000, 60_000)[0] == l3.STATUS_GAP
    assert a.window_delta(130_000, 160_000)[0] == l3.STATUS_NO_SAMPLE  # past the capture
    src = l3.CounterTokenSource([a, reset])
    assert src(30_000, 60_000) is None  # one void pod voids the model's window
    assert src(60_000, 90_000) == (60 * PROMPT_RATE, 60 * GEN_RATE)
    assert src.counts == {l3.STATUS_RESET: 1, l3.STATUS_OK: 1}
    assert l3.CounterTokenSource([a], missing_pods=["ns/gone"])(60_000, 90_000) is None


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
    assert dl.verify_freeze(w["freeze"])["models"][base.MODEL]["numerator"] == l3.NUMERATOR_VLLM_COUNTER
    man = base._seal(w)
    base._refused(w, man, capsys, "TSS numerator")  # M was built with the gateway numerator
    mman = w["mdir"] / "manifest.json"
    mman.write_text(json.dumps({**json.loads(mman.read_text()),
                                "numerator": {"source": l3.NUMERATOR_VLLM_COUNTER}}))
    assert base._accept(w, man) == 0
