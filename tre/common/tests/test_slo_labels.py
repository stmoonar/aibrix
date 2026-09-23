from __future__ import annotations

import pytest

from tre_common import slo_labels

SLO = slo_labels.slo_targets(ttft_slo_ms=500.0, tpot_slo_ms=75.0)


def _row(ttft=100.0, tpot=20.0, **counts):
    return {slo_labels.P95_TTFT_CLIENT: ttft, slo_labels.P95_TPOT_CLIENT: tpot, **counts}


def test_a_window_under_every_slo_is_healthy() -> None:
    assert slo_labels.window_slo_label(_row(), SLO) == slo_labels.LABEL_HEALTHY


def test_a_window_over_any_slo_is_violated() -> None:
    assert slo_labels.window_slo_label(_row(tpot=75.01), SLO) == slo_labels.LABEL_VIOLATED
    assert slo_labels.window_slo_label(_row(ttft=900.0), SLO) == slo_labels.LABEL_VIOLATED
    # exactly at the SLO still meets it
    assert slo_labels.window_slo_label(_row(ttft=500.0, tpot=75.0), SLO) == slo_labels.LABEL_HEALTHY


@pytest.mark.parametrize("column", slo_labels.UNSERVED_COLUMNS)
def test_any_unserved_request_makes_the_window_violated(column) -> None:
    assert slo_labels.window_slo_label(_row(**{column: 1}), SLO) == slo_labels.LABEL_VIOLATED
    # even with no latency sample at all - the failed request left none
    assert slo_labels.window_slo_label(
        _row(ttft=None, tpot=None, **{column: "1"}), SLO
    ) == slo_labels.LABEL_VIOLATED


def test_a_missing_p95_is_unlabeled_not_healthy() -> None:
    assert slo_labels.window_slo_label(_row(tpot=None), SLO) == slo_labels.LABEL_UNLABELED
    assert slo_labels.window_slo_label(_row(ttft=""), SLO) == slo_labels.LABEL_UNLABELED


def test_the_server_columns_are_never_read() -> None:
    row = _row()
    row[slo_labels.P95_TPOT_SERVER] = 150.0  # the server histogram says "violated"
    assert slo_labels.window_slo_label(row, SLO) == slo_labels.LABEL_HEALTHY


def test_csv_strings_label_like_numbers() -> None:
    row = {slo_labels.P95_TTFT_CLIENT: "480.0", slo_labels.P95_TPOT_CLIENT: "80.5",
           "model_errors": "0", "proxy_transient_errors": "", "client_timeouts": "0"}
    assert slo_labels.window_slo_label(row, SLO) == slo_labels.LABEL_VIOLATED


def test_apply_label_writes_both_columns_consistently() -> None:
    for row, label, flag in (
        (_row(), "healthy", False),
        (_row(tpot=90.0), "violated", True),
        (_row(tpot=None), "unlabeled", None),
    ):
        assert slo_labels.apply_label(row, SLO) == label
        assert row[slo_labels.LABEL_COLUMN] == label
        assert row[slo_labels.VIOLATED_COLUMN] is flag


def test_an_unknown_slo_key_is_refused_not_ignored() -> None:
    with pytest.raises(ValueError):
        slo_labels.window_slo_label(_row(), {"p95_tpot": 75.0})


def test_the_label_definition_names_its_source() -> None:
    definition = slo_labels.label_definition(SLO, min_latency_samples=10)
    assert definition["latency_source"] == "client per-request"
    assert definition["slo_ms"] == {"p95_ttft_client_ms": 500.0, "p95_tpot_client_ms": 75.0}


# ---------------------------------------------------------- the merged label (2026-09-23)

PRIMARY = slo_labels.LabelDefinition(
    500.0, 75.0, ttft_slo_mode="slowdown", ttft_idle_c_ms=36.4, ttft_idle_b_ms_per_token=0.0527,
)


def _slow_row(pairs, *, tpot=20.0, **extra):
    return {
        slo_labels.P95_TPOT_CLIENT: tpot,
        slo_labels.P95_TTFT_CLIENT: max(t for t, _ in pairs),
        slo_labels.COMPLETED_REQUESTS_COLUMN: len(pairs),
        slo_labels.TTFT_LEN_SAMPLES_COLUMN: slo_labels.format_ttft_len_samples(pairs),
        **extra,
    }


def test_slo_violated_is_an_output_never_an_unserved_input() -> None:
    """The merge guard: a 09-23 window CSV writes slo_violated=True for a latency
    violation with every unserved count at 0. Reading that column as "unserved" (the 09-22
    meaning) would make every such window unserved - 100 % violations and not one error."""
    healthy_latency = _slow_row([(100.0, 256)] * 25, slo_violated="True", slo_label="violated",
                                model_errors=0, proxy_transient_errors=0, client_timeouts=0)
    for spec in (PRIMARY, SLO, slo_labels.LabelDefinition.from_targets(SLO)):
        assert not slo_labels.row_unserved(healthy_latency)
        assert slo_labels.window_slo_label(healthy_latency, spec) == slo_labels.LABEL_HEALTHY
    tpot_violated = _slow_row([(100.0, 256)] * 25, tpot=150.0, slo_violated="True",
                              model_errors=0, proxy_transient_errors=0, client_timeouts=0)
    lab = slo_labels.label_window(tpot_violated, PRIMARY)
    assert not lab.slo_met and not lab.unserved and lab.violation_class == "tpot_only"
    assert lab.ratio_max == pytest.approx(2.0)


def test_the_three_arms_are_written_by_one_call() -> None:
    # 20 requests at L = 2048: primary SLO 5 * (36.4 + 0.0527 * 2048) = 721.7 ms,
    # k = 3 SLO 433 ms, fixed 500 ms: a 600 ms TTFT is healthy / violated / violated.
    row = _slow_row([(600.0, 2048)] * 20)
    assert slo_labels.apply_label_arms(row, PRIMARY) == "healthy"
    assert row[slo_labels.LABEL_COLUMN] == "healthy" and row[slo_labels.VIOLATED_COLUMN] is False
    assert row[slo_labels.LABEL_COLUMN_FIXED] == "violated"
    assert row[slo_labels.LABEL_COLUMN_K3] == "violated"
    # a fixed primary is its own comparison arm and has no k = 3 arm
    row = _slow_row([(600.0, 2048)] * 20)
    slo_labels.apply_label_arms(row, SLO)
    assert row[slo_labels.LABEL_COLUMN] == row[slo_labels.LABEL_COLUMN_FIXED] == "violated"
    assert row[slo_labels.LABEL_COLUMN_K3] is None


def test_min_n_makes_a_thin_window_unlabeled_unless_it_went_unserved() -> None:
    thin = _slow_row([(100.0, 256)] * 19)
    assert slo_labels.window_slo_label(thin, PRIMARY) == slo_labels.LABEL_UNLABELED
    # the 09-23 threshold mapping has no min-n guard (it had only the p95 guard)
    assert slo_labels.window_slo_label(thin, SLO) == slo_labels.LABEL_HEALTHY
    thin["client_timeouts"] = 1
    assert slo_labels.window_slo_label(thin, PRIMARY) == slo_labels.LABEL_VIOLATED


def test_the_09_22_latency_names_are_read_only_when_the_client_column_is_absent() -> None:
    legacy = {"p95_ttft": 100.0, "p95_tpot": 80.0}
    assert slo_labels.window_slo_label(legacy, SLO) == slo_labels.LABEL_VIOLATED
    both = {"p95_ttft": 100.0, "p95_tpot": 80.0,
            slo_labels.P95_TTFT_CLIENT: 100.0, slo_labels.P95_TPOT_CLIENT: 20.0}
    assert slo_labels.window_slo_label(both, SLO) == slo_labels.LABEL_HEALTHY  # client wins
    assert slo_labels.window_slo_label({slo_labels.P95_TTFT_CLIENT: 1.0, "p95_tpot": 20.0,
                                        slo_labels.P95_TPOT_CLIENT: ""}, SLO) == slo_labels.LABEL_UNLABELED


def test_a_label_record_rebuilds_its_definition_in_both_formats() -> None:
    # the merged record (primary + arms) ...
    record = slo_labels.label_definition(PRIMARY, window_membership="(start, end]")
    assert slo_labels.LabelDefinition.from_dict(record) == PRIMARY
    assert slo_labels.LabelDefinition.from_dict(record["arms"]["fixed_comparison"]).latency_slo_ms() == SLO
    assert record["window_membership"]["latency"].endswith("(start, end]")
    assert record["slo_ms"] == {"p95_ttft_client_ms": 500.0, "p95_tpot_client_ms": 75.0}
    # ... and the 09-23 record of a threshold mapping: fixed, no min-n guard - exactly the
    # label that wrote the second run's online CSVs
    old = slo_labels.label_definition(SLO, min_latency_samples=10)
    rebuilt = slo_labels.LabelDefinition.from_dict(old)
    assert not rebuilt.slowdown and rebuilt.min_completed_requests == 0
    assert rebuilt.latency_slo_ms() == SLO


def test_the_cli_builds_the_registry_primary_and_its_arms(tmp_path) -> None:
    import argparse
    from pathlib import Path

    registry = Path(__file__).resolve().parents[2] / "deploy" / "registry.yaml"
    ap = argparse.ArgumentParser()
    ap.add_argument("--ttft-slo-ms", type=float, default=500.0)
    ap.add_argument("--tpot-slo-ms", type=float, default=75.0)
    slo_labels.add_label_arguments(ap, include_fixed=False)
    args = ap.parse_args(["--label-registry", str(registry)])
    primary = slo_labels.label_def_from_args(args, "dsqwen-7b")
    assert primary.slowdown and primary.min_completed_requests == 20
    assert (primary.ttft_slowdown_k, primary.ttft_floor_ms) == (5.0, 500.0)
    again = ap.parse_args(["--label-registry", str(registry), *slo_labels.label_mode_cli_args(primary)])
    assert slo_labels.label_def_from_args(again, "dsqwen-7b") == primary
    assert set(slo_labels.resolve_arms(primary)) == set(slo_labels.ARM_COLUMNS)
