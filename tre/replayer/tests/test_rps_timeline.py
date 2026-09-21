"""Nominal vs achieved arrivals, and the blind spot the span ratio had.

The counter-example these tests are built around is the one that was measured on the
real code path: every request goes out a uniform amount late, the run's first-to-last
span is unchanged to the microsecond, and the old span-based ``actual_rps_error_ratio``
reports a perfect 0.0000 for a run that was offered entirely off its schedule.
"""
from __future__ import annotations

import csv
from dataclasses import replace

from tre_replayer.engine import rps_timeline
from tre_replayer.engine.dispatcher import DispatchRecord, DispatchReport


def _offsets(rate_hz: float, duration_s: float) -> list[float]:
    step = 1.0 / rate_hz
    count = int(round(duration_s * rate_hz))
    return [i * step for i in range(count)]


# ------------------------------------------------------------------------- the binning


def test_a_schedule_that_was_kept_scores_exactly_zero() -> None:
    """The two instant sets are the same events timestamped twice, so a kept schedule
    matches bin for bin - however ragged its Poisson arrivals are. Without that property
    the metric would fire on every honest run."""
    ragged = [0.0, 0.02, 0.03, 0.9, 2.4, 2.41, 2.9, 7.7]
    bins = rps_timeline.build_rps_timeline(ragged, list(ragged), window_s=1.0)
    assert rps_timeline.max_relative_rps_error(bins) == 0.0
    assert all(b.scheduled == b.achieved for b in bins)


def test_uniform_lateness_is_invisible_to_a_span_ratio_and_visible_here() -> None:
    """The exact defect: shifting every arrival by the same amount leaves both span
    rates identical, so the old ratio is 0 while the run is a third of a second behind
    its own schedule for its whole duration."""
    scheduled = _offsets(10.0, 30.0)
    late_s = 0.377
    achieved = [offset + late_s for offset in scheduled]

    # what the superseded definition computed: n / (last - first), both ways
    planned_rps = len(scheduled) / (scheduled[-1] - scheduled[0])
    achieved_rps = len(achieved) / (achieved[-1] - achieved[0])
    assert abs(achieved_rps - planned_rps) / planned_rps < 1e-12  # 0.0000, the blind spot

    bins = rps_timeline.build_rps_timeline(
        scheduled, achieved, window_s=rps_timeline.ERROR_WINDOW_S
    )
    assert rps_timeline.max_relative_rps_error(bins) > 0.05


def test_a_generator_that_falls_behind_shows_a_growing_deficit() -> None:
    """Drift, not a shift: the late-ness accumulates, so the deviation is in the middle
    of the run rather than only at its head."""
    scheduled = _offsets(10.0, 20.0)
    achieved = [offset * 1.15 for offset in scheduled]  # 15 % slow
    bins = rps_timeline.build_rps_timeline(scheduled, achieved, window_s=5.0)
    assert rps_timeline.max_relative_rps_error(bins) > 0.1


def test_a_stall_shows_up_in_the_windows_it_happened_in() -> None:
    scheduled = _offsets(10.0, 20.0)
    achieved = [offset if offset < 8.0 else offset + 3.0 for offset in scheduled]
    bins = rps_timeline.build_rps_timeline(scheduled, achieved, window_s=1.0)
    stalled = [b for b in bins if 8.0 <= b.start_s < 11.0]
    assert stalled and all(b.achieved < b.scheduled for b in stalled)


def test_a_window_the_schedule_left_empty_has_no_ratio_to_be_wrong_about() -> None:
    bins = rps_timeline.build_rps_timeline([0.1, 0.2], [0.1, 5.2], window_s=1.0)
    empty = [b for b in bins if b.scheduled == 0]
    assert empty and all(b.relative_error is None for b in empty)
    # the deficit that put the arrival there is still counted, in its own window
    assert rps_timeline.max_relative_rps_error(bins) == 0.5


def test_rates_are_per_second_not_per_window() -> None:
    bins = rps_timeline.build_rps_timeline([0.0, 0.5, 1.5], [0.0, 0.5, 1.5], window_s=0.5)
    assert bins[0].scheduled_rps == 2.0  # one request in a half-second window


# ------------------------------------------------------------------------ the artifact


def test_the_timeline_csv_carries_both_series_per_model(tmp_path) -> None:
    """This file is the evidence a run offered the intensity its trace describes, so it
    has to be plottable as it stands: one row per model per window, both series on it."""
    path = tmp_path / "cell.rps.csv"
    per_model = {
        "dsqwen-7b": rps_timeline.build_rps_timeline([0.0, 0.5, 1.2], [0.0, 0.5, 2.2], window_s=1.0),
        "dsllama-8b": rps_timeline.build_rps_timeline([0.0, 0.4], [0.0, 0.4], window_s=1.0),
    }
    written = rps_timeline.write_rps_timeline_csv(path, per_model)
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    assert written == len(rows)
    assert set(rows[0]) == set(rps_timeline.CSV_COLUMNS)
    assert {row["model"] for row in rows} == {"dsqwen-7b", "dsllama-8b"}
    first = next(r for r in rows if r["model"] == "dsqwen-7b")
    assert first["scheduled_requests"] == "2" and first["achieved_requests"] == "2"


# ----------------------------------------------------------- the dispatcher's own view


def test_dispatch_report_rps_error_sees_a_uniformly_late_loop() -> None:
    """The same counter-example on the report the drivers actually read."""
    base = 1000.0
    on_time = [
        DispatchRecord(
            request_id=f"r{i}", model="m", scheduled_ts=base + i * 0.1, actual_ts=base + i * 0.1
        )
        for i in range(200)
    ]
    span_s = on_time[-1].actual_ts - on_time[0].actual_ts
    kept = DispatchReport(
        records=on_time, planned_duration_s=span_s, actual_duration_s=span_s, base_ts=base
    )
    assert kept.actual_rps_error_ratio == 0.0

    late = [replace(record, actual_ts=record.actual_ts + 0.4) for record in on_time]
    # every span the old ratio looked at is unchanged; only the phase moved
    assert late[-1].actual_ts - late[0].actual_ts == span_s
    shifted = DispatchReport(
        records=late, planned_duration_s=span_s, actual_duration_s=span_s, base_ts=base
    )
    assert shifted.actual_rps_error_ratio > 0.05


def test_dispatch_report_is_still_quiet_for_an_empty_or_single_request_run() -> None:
    assert DispatchReport(records=[], planned_duration_s=0.0, actual_duration_s=0.0).actual_rps_error_ratio == 0.0
