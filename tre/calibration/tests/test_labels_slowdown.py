"""Slowdown TTFT SLO of the shared label (plan 2026-09-21 §6.9h, §6.11 D6)."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import pytest

from tre_calibration.dataset import load_windows_from_csv
from tre_common.slo_labels import (
    LABEL_DEF_NAME,
    LabelDefinition,
    LabelInputError,
    add_label_arguments,
    count_label_exclusions,
    format_ttft_len_samples,
    label_cli_args,
    label_def_from_args,
    label_window,
    parse_ttft_len_samples,
)
from tre_common.registry import load_registry

SLOW = LabelDefinition(
    500.0, 75.0, ttft_slo_mode="slowdown", ttft_slowdown_k=3.0, ttft_floor_ms=150.0,
    ttft_idle_c_ms=30.0, ttft_idle_b_ms_per_token=0.1,
)
FIXED = LabelDefinition(500.0, 75.0)


def _row(pairs, *, tpot=20.0, n=None, unserved=False, p95_ttft=None):
    return {
        "ttft_len_samples": format_ttft_len_samples(pairs),
        "completed_requests": len(pairs) if n is None else n,
        "p95_tpot": tpot,
        "p95_ttft": p95_ttft if p95_ttft is not None else max((t for t, _ in pairs), default=""),
        # unserved evidence is the count columns; slo_violated is never read
        "client_timeouts": 1 if unserved else 0,
        "slo_violated": "False",
    }


def test_slowdown_formula_scales_the_idle_ttft_by_k() -> None:
    # 3 * (30 + 0.1 * L)
    assert SLOW.ttft_slo_ms(2048) == pytest.approx(3 * (30 + 204.8))
    assert SLOW.ttft_slo_ms(1600) == pytest.approx(570.0)
    k5 = SLOW.with_mode("slowdown", ttft_slowdown_k=5.0)
    assert k5.ttft_slo_ms(1600) == pytest.approx(5 * 190.0)
    # fixed mode ignores the length
    assert FIXED.ttft_slo_ms(10_000) == 500.0


def test_slowdown_floor_binds_for_short_prompts() -> None:
    # 3 * (30 + 0.1 * 50) = 105 < 150
    assert SLOW.ttft_slo_ms(50) == 150.0
    assert SLOW.ttft_slo_ms(0) == 150.0
    # just above the crossover (L = 200 -> 3 * 50 = 150)
    assert SLOW.ttft_slo_ms(300) == pytest.approx(180.0)


def test_window_ratio_is_the_p95_of_per_request_ratios() -> None:
    # 20 requests at L = 1000 (SLO 390 ms): 19 at 0.5x, one at 2x -> p95 (bucket_upper,
    # target 19 of 20) is the 19th sorted value = 0.5
    pairs = [(195.0, 1000)] * 19 + [(780.0, 1000)]
    lab = label_window(_row(pairs), SLOW)
    assert lab is not None and lab.slo_met
    assert lab.ttft_ratio == pytest.approx(0.5)
    # two slow requests out of 20 -> p95 lands on a slow one
    pairs = [(195.0, 1000)] * 18 + [(780.0, 1000)] * 2
    lab = label_window(_row(pairs), SLOW)
    assert lab is not None and not lab.slo_met and lab.violation_class == "ttft_only"
    assert lab.ttft_ratio == pytest.approx(2.0)


def test_same_ttft_is_healthy_for_a_long_prompt_and_violated_for_a_short_one() -> None:
    long_ok = label_window(_row([(400.0, 2048)] * 20), SLOW)  # SLO 704 ms
    short_bad = label_window(_row([(400.0, 256)] * 20), SLOW)  # SLO 166.8 ms
    assert long_ok is not None and long_ok.slo_met
    assert short_bad is not None and not short_bad.slo_met
    # the fixed label cannot tell them apart
    assert label_window(_row([(400.0, 2048)] * 20), FIXED).slo_met
    assert label_window(_row([(400.0, 256)] * 20), FIXED).slo_met


def test_violation_classes() -> None:
    ok = [(100.0, 1000)] * 20
    bad = [(1000.0, 1000)] * 20
    assert label_window(_row(ok, tpot=100.0), SLOW).violation_class == "tpot_only"
    assert label_window(_row(bad, tpot=100.0), SLOW).violation_class == "both"
    assert label_window(_row(ok, unserved=True), SLOW).violation_class == "unserved"
    assert label_window(_row(ok), SLOW).violation_class is None


def test_min_n_excludes_thin_windows_in_both_modes() -> None:
    thin = _row([(100.0, 1000)] * 19)
    assert SLOW.classify(thin) == (None, "low_n")
    assert FIXED.classify(thin) == (None, "low_n")
    assert label_window(_row([(100.0, 1000)] * 20), SLOW) is not None
    # a thin window that went unserved is kept as violated
    lab = label_window(_row([(100.0, 1000)] * 5, unserved=True), SLOW)
    assert lab is not None and not lab.slo_met and lab.unserved
    rows = [thin, _row([(100.0, 1000)] * 25), {"p95_tpot": "", "ttft_len_samples": "", "completed_requests": ""}]
    assert count_label_exclusions(rows, SLOW) == {"labelled": 1, "low_n": 1, "missing_latency": 1}


def test_min_n_is_configurable() -> None:
    lax = LabelDefinition(500.0, 75.0, min_completed_requests=10)
    assert label_window(_row([(100.0, 1000)] * 12), lax) is not None


def test_fixed_mode_is_unchanged_on_legacy_rows() -> None:
    # rows without the new columns: the definition object labels exactly as the legacy
    # mapping path always did (no min-n guard, same ratios, same verdicts)
    rows = [
        {"p95_ttft": 100, "p95_tpot": 20},
        {"p95_ttft": 600, "p95_tpot": 20},
        {"p95_ttft": 100, "p95_tpot": 80},
        {"p95_ttft": "", "p95_tpot": 20},
        {"p95_ttft": "", "p95_tpot": "", "model_errors": 1},
        {"p95_ttft": 100, "p95_tpot": 20, "proxy_transient_errors": "1"},
        # slo_violated alone is the label column, not unserved evidence
        {"p95_ttft": 100, "p95_tpot": 20, "slo_violated": "True"},
    ]
    for row in rows:
        new = label_window(row, FIXED)
        old = label_window(row, FIXED.latency_slo_ms())
        assert (new is None) == (old is None)
        if new is not None:
            assert (new.slo_met, new.ratio_max, new.unserved) == (old.slo_met, old.ratio_max, old.unserved)
    d = FIXED.as_dict()
    assert d["name"] == LABEL_DEF_NAME and d["mode"] == "fixed"
    assert d["violated_if"] == (
        "p95_ttft_client_ms > ttft_p95_ms or p95_tpot_client_ms > tpot_p95_ms or unserved"
    )


def test_slowdown_refuses_a_csv_without_per_request_samples() -> None:
    with pytest.raises(LabelInputError):
        label_window({"p95_ttft": 100, "p95_tpot": 20}, SLOW)


def test_slowdown_needs_the_idle_fit() -> None:
    with pytest.raises(ValueError):
        LabelDefinition(500.0, 75.0, ttft_slo_mode="slowdown")
    with pytest.raises(ValueError):
        LabelDefinition(500.0, 75.0, ttft_slo_mode="bogus")


def test_samples_round_trip() -> None:
    text = format_ttft_len_samples([(12.345, 256), (None, 10), (99.0, None), (7.0, 3072)])
    assert text == "12.35:256;7.00:3072"
    assert parse_ttft_len_samples(text) == [(12.35, 256.0), (7.0, 3072.0)]
    assert parse_ttft_len_samples("") == []


def test_label_def_records_every_parameter_and_round_trips() -> None:
    d = SLOW.as_dict()
    for key, value in {"mode": "slowdown", "k": 3.0, "floor_ms": 150.0, "c_ms": 30.0,
                       "b_ms_per_token": 0.1, "tpot_p95_ms": 75.0, "min_n": 20}.items():
        assert d[key] == value
    assert LabelDefinition.from_dict(d) == SLOW
    assert LabelDefinition.from_dict(FIXED.as_dict()) == FIXED
    # a pre-D6 artifact: fixed, no min_n -> the old behaviour (no guard)
    legacy = LabelDefinition.from_dict({"ttft_p95_ms": 500.0, "tpot_p95_ms": 75.0})
    assert legacy.ttft_slo_mode == "fixed" and legacy.min_completed_requests == 0


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    add_label_arguments(p)
    return p


def test_cli_args_rebuild_the_same_label() -> None:
    for label in (SLOW, FIXED, SLOW.with_mode("slowdown", ttft_slowdown_k=5.0, min_completed_requests=30)):
        args = _parser().parse_args(label_cli_args(label))
        assert label_def_from_args(args, None) == label


def test_per_model_idle_fit_comes_from_the_registry() -> None:
    reg = load_registry()
    args = _parser().parse_args(["--ttft-p95-ms", "500", "--tpot-p95-ms", "75", "--ttft-slo-mode", "slowdown"])
    seen = set()
    for model in ("dsqwen-7b", "dsllama-8b", "dsqwen-14b"):
        slo = reg.model(model).slo
        label = label_def_from_args(args, model)
        assert (label.ttft_idle_c_ms, label.ttft_idle_b_ms_per_token) == (slo.ttft_idle_c_ms, slo.ttft_idle_b_ms_per_token)
        assert label.ttft_slowdown_k == 5.0 and label.ttft_floor_ms == 500.0 and label.min_completed_requests == 20
        assert (label.ttft_slowdown_k, label.ttft_floor_ms) == (slo.ttft_slowdown_k, slo.ttft_floor_ms)
        seen.add(label.ttft_slo_ms(2048))
    assert len(seen) == 3
    # an explicit override wins over the registry
    args = _parser().parse_args(["--ttft-p95-ms", "500", "--tpot-p95-ms", "75", "--ttft-slo-mode", "slowdown",
                                 "--ttft-idle-c-ms", "30", "--ttft-idle-b-ms-per-token", "0.1",
                                 "--ttft-slowdown-k", "3", "--ttft-floor-ms", "150"])
    assert label_def_from_args(args, "dsqwen-7b") == SLOW


def test_dataset_loader_uses_the_definition(tmp_path: Path) -> None:
    path = tmp_path / "w.csv"
    fields = ["scenario_id", "scenario_family", "window_start_ms", "window_end_ms",
              "prompt_tokens_total", "generation_tokens_total", "p95_ttft", "p95_tpot",
              "trs", "client_timeouts", "slo_violated", "completed_requests", "ttft_len_samples"]
    rows = [
        _row([(400.0, 256)] * 20),   # slowdown-violated, fixed-healthy
        _row([(400.0, 2048)] * 20),  # healthy either way
        _row([(400.0, 2048)] * 8),   # too thin
    ]
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for i, r in enumerate(rows):
            w.writerow(dict(r, scenario_id="a", scenario_family="f", window_start_ms=i * 5000,
                            window_end_ms=i * 5000 + 30000, prompt_tokens_total=100,
                            generation_tokens_total=100, trs=50))
    slow = load_windows_from_csv(path, latency_slo_ms=SLOW)
    fixed = load_windows_from_csv(path, latency_slo_ms=FIXED)
    assert [w.slo_met for w in slow] == [False, True]
    assert [w.violation_class for w in slow] == ["ttft_only", None]
    assert [w.slo_met for w in fixed] == [True, True]
    assert slow[0].latency_ratio_avg is None


def test_d6_prime_primary_label_is_the_default() -> None:
    """Plan 6.11 D6': no mode flag -> slowdown, k = 5, floor 500 ms, TPOT 75 ms."""
    args = _parser().parse_args(["--ttft-p95-ms", "500", "--tpot-p95-ms", "75"])
    label = label_def_from_args(args, "dsqwen-7b")
    assert label.slowdown and label.ttft_slowdown_k == 5.0 and label.ttft_floor_ms == 500.0
    assert label.tpot_p95_ms == 75.0
    # 7b idle at 2048 tok: 36.4 + 0.0527 * 2048 = 144.3 ms -> 5x = 721.5 ms; short prompts sit on the floor
    assert abs(label.ttft_slo_ms(2048) - 5 * (36.4 + 0.0527 * 2048)) < 1e-9
    assert label.ttft_slo_ms(256) == 500.0
    # the fixed comparison column stays selectable
    fixed = label_def_from_args(_parser().parse_args(["--ttft-slo-mode", "fixed", "--ttft-p95-ms", "500", "--tpot-p95-ms", "75"]), "dsqwen-7b")
    assert not fixed.slowdown and fixed.ttft_p95_ms == 500.0 and fixed.as_dict()["mode"] == "fixed"
    # without a model the slowdown default cannot be built: refused, not silently fixed
    with pytest.raises(SystemExit):
        label_def_from_args(args, None)


def test_label_arms_primary_comparison_ablation() -> None:
    from tre_common.slo_labels import label_arms

    primary = label_def_from_args(_parser().parse_args(["--ttft-p95-ms", "500", "--tpot-p95-ms", "75"]), "dsllama-8b")
    arms = label_arms(primary)
    assert arms["primary"] == primary
    assert arms["fixed_comparison"].as_dict()["mode"] == "fixed"
    abl = arms["ablation_k3_floor150"]
    assert abl.slowdown and (abl.ttft_slowdown_k, abl.ttft_floor_ms) == (3.0, 150.0)
    assert (abl.ttft_idle_c_ms, abl.ttft_idle_b_ms_per_token) == (primary.ttft_idle_c_ms, primary.ttft_idle_b_ms_per_token)
