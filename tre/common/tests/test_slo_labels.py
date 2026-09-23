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
