from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts import adaptive_boundary as boundary
from scripts import admission_cap as admission
from scripts import calibration_campaign as campaign
from scripts import gen_calibration_schedules as gen
from scripts import openloop


# ------------------------------------------------------------------ capacity measurement

LEVELS = ((0.5, 90.0), (0.8, 120.0), (0.95, 240.0))


def _records(rates, *, levels=LEVELS, ttft_ms=100.0, tpot_ms=10.0, ttft_by_level=None):
    """Synthesise a steps cell's raw records delivering ``rates[k]`` rps in level k."""
    records = []
    origin = 1_000_000
    start_s = 0.0
    for index, (_rho, duration_s) in enumerate(levels):
        end_s = start_s + duration_s
        rate = rates[index]
        level_ttft = ttft_ms if ttft_by_level is None else ttft_by_level[index]
        count = int(rate * duration_s)
        for n in range(count):
            done_s = start_s + duration_s * (n + 0.5) / max(1, count)
            records.append({
                "send_ts_ms": origin + int(done_s * 1000) - 50,
                "done_ts_ms": origin + int(done_s * 1000),
                "ttft_ms": level_ttft,
                "tpot_ms": tpot_ms,
                "http_status": 200,
            })
        start_s = end_s
    # anchor the origin so the first level starts at offset 0
    records.append({"send_ts_ms": origin, "done_ts_ms": origin, "ttft_ms": ttft_ms,
                    "tpot_ms": tpot_ms, "http_status": 200})
    return records


def test_measurement_replaces_an_over_estimating_prior() -> None:
    # The prior claims 10 rps, but the top level only delivers 6 and misses the TTFT SLO:
    # the prior was an over-estimate and the ramp must be rebuilt on the measurement.
    records = _records([5.0, 6.0, 6.0], ttft_by_level=[100.0, 100.0, 900.0])
    measured = campaign.measure_capacity_from_steps(
        records, levels=LEVELS, capacity_prior_rps=10.0,
        ttft_slo_ms=500.0, tpot_slo_ms=75.0, transient_s=30.0,
    )
    assert measured.saturated
    assert measured.capacity_source == "measured_steps"
    assert measured.capacity_used_rps == pytest.approx(6.0, abs=0.3)
    assert measured.capacity_used_rps == measured.capacity_measured_rps


def test_an_unsaturated_cell_keeps_the_prior_and_says_it_is_censored() -> None:
    # Every level met the SLO and delivered what it was offered, so the cell only proves
    # C >= 0.95 * prior. Substituting that lower bound would shrink every downstream rho.
    records = _records([5.0, 8.0, 9.5])
    measured = campaign.measure_capacity_from_steps(
        records, levels=LEVELS, capacity_prior_rps=10.0,
        ttft_slo_ms=500.0, tpot_slo_ms=75.0, transient_s=30.0,
    )
    assert not measured.saturated
    assert measured.capacity_source == "prior_censored"
    assert measured.capacity_used_rps == 10.0
    assert measured.capacity_measured_rps == pytest.approx(9.5, abs=0.5)
    assert "under-estimates" in measured.note


def test_a_throughput_shortfall_alone_counts_as_saturation() -> None:
    # Latency stayed inside the SLO but the top level delivered barely half what it was
    # offered: the engine, not the schedule, is setting the rate.
    records = _records([5.0, 8.0, 5.0])
    measured = campaign.measure_capacity_from_steps(
        records, levels=LEVELS, capacity_prior_rps=10.0,
        ttft_slo_ms=500.0, tpot_slo_ms=75.0, transient_s=30.0,
    )
    assert measured.saturated
    assert measured.capacity_source == "measured_steps"
    assert measured.capacity_used_rps == pytest.approx(8.0, abs=0.5)


def test_a_cell_that_sent_nothing_is_a_loud_failure() -> None:
    with pytest.raises(ValueError, match="no requests"):
        campaign.measure_capacity_from_steps(
            [], levels=LEVELS, capacity_prior_rps=10.0,
            ttft_slo_ms=500.0, tpot_slo_ms=75.0,
        )


def test_only_the_steady_part_of_a_level_is_measured() -> None:
    # A level whose first 60 s are transient must not have them averaged in. Here the
    # whole of level 3 runs at 6 rps, so the steady measurement is 6 either way - what is
    # asserted is that the transient boundary is where it is claimed to be.
    records = _records([5.0, 6.0, 6.0], ttft_by_level=[100.0, 100.0, 900.0])
    measured = campaign.measure_capacity_from_steps(
        records, levels=LEVELS, capacity_prior_rps=10.0,
        ttft_slo_ms=500.0, tpot_slo_ms=75.0, transient_s=60.0,
    )
    top = measured.levels[-1]
    assert top.start_s == 210.0
    assert top.steady_start_s == 270.0
    assert top.end_s == 450.0


# -------------------------------------------------------------------------------- plan


def _index(cap=admission.GATEWAY_CAPPED):
    entries = []
    for model in ("dsqwen-7b", "dsllama-8b"):
        for shape in ("S1", "S3"):
            for primitive in ("ramp", "steps", "bursts"):
                skipped = primitive == "bursts" and shape == "S1"
                entry = {
                    "model": model, "shape": shape, "primitive": primitive,
                    "cell_id": f"i256_o128_c{ {'ramp': 120, 'steps': 95, 'bursts': 60}[primitive] }",
                    "duration_s": 400.0, "capacity_rps": 5.0,
                    "capacity_source": "prior_fit", "skipped": skipped,
                    "kv_cache_tokens": 147536,
                }
                if skipped:
                    entry["reason"] = "the spike would be shed by the gateway"
                else:
                    entry["path"] = f"{model}/{shape}_{primitive}.json"
                if primitive == "ramp":
                    entry["drain_start_s"] = 375.0
                entries.append(entry)
    return {
        "schedules": entries,
        "admission_cap": cap.as_dict(),
        "primitives": {"steps": {"levels": [{"rho": r, "duration_s": d} for r, d in LEVELS]}},
        "capacity_models": {},
    }


def test_plan_runs_steps_before_ramp_before_bursts_within_a_model() -> None:
    runnable, skipped = campaign.build_plan(_index(), ["dsqwen-7b", "dsllama-8b"])

    for model in ("dsqwen-7b", "dsllama-8b"):
        order = [c.primitive for c in runnable if c.model == model]
        assert order == ["steps", "steps", "ramp", "ramp", "bursts"]
    # and a model is finished before the next one starts
    assert [c.model for c in runnable][:5] == ["dsqwen-7b"] * 5
    assert {(c.model, c.shape) for c in skipped} == {("dsqwen-7b", "S1"), ("dsllama-8b", "S1")}
    assert all(c.skip_reason for c in skipped)


def test_plan_marks_ramps_for_regeneration_from_the_measurement() -> None:
    runnable, _ = campaign.build_plan(_index(), ["dsqwen-7b"])
    ramps = [c for c in runnable if c.primitive == "ramp"]
    assert ramps and all(c.regenerate_from_measured for c in ramps)
    assert not any(c.regenerate_from_measured for c in runnable if c.primitive != "ramp")
    assert all(c.drain_start_s == 375.0 for c in ramps)


def test_skipped_cells_never_reach_the_runnable_list() -> None:
    # A skipped entry has no "path"; treating it as runnable would crash mid-campaign.
    runnable, _ = campaign.build_plan(_index(), ["dsqwen-7b", "dsllama-8b"])
    assert all(cell.schedule for cell in runnable)
    assert not any(cell.skipped for cell in runnable)


def test_kv_cache_tokens_fall_back_to_the_schedule_entries() -> None:
    # A campaign whose bursts were all skipped still has to be able to say why.
    tokens = campaign.kv_cache_tokens_by_model(_index())
    assert tokens["dsllama-8b"] == 147536


def test_wall_clock_estimate_includes_the_cooldowns() -> None:
    runnable, _ = campaign.build_plan(_index(), ["dsqwen-7b"])
    assert campaign.estimate_wall_clock_s(runnable, 45.0) == pytest.approx(5 * 445.0)


# ---------------------------------------------------------------------------- fit plan


class _Args:
    out_dir = Path("/out")
    window_ms = 30000
    fit_step_ms = 5000
    instant_sample_ms = 1000
    ttft_slo_ms = 500.0
    tpot_slo_ms = 75.0
    max_model_error_rate = 0.05
    envoy_stats_url = None


def test_fit_plan_fits_on_the_live_grid_and_keeps_the_raw_stream_separate() -> None:
    plan = campaign.fit_plan(["dsqwen-7b"], Path("/out"), Path("/raw"), _Args())

    fitting = [r for r in plan["rewindow"] if "fitting" in r["purpose"]]
    aliasing = [r for r in plan["rewindow"] if "aliasing" in r["purpose"]]
    assert len(fitting) == 1 and len(aliasing) == 1

    # theta is a threshold on the signal the controller consumes, so the fit re-windows
    # the 10 s-aligned subsample; a mismatched divisor would scale queue averages by 10x.
    fit_cmd = fitting[0]["command"]
    assert "--instant-grid" in fit_cmd
    assert fit_cmd[fit_cmd.index("--instant-grid") + 1] == "live"
    assert fit_cmd[fit_cmd.index("--instant-sample-ms") + 1] == str(campaign.LIVE_GRID_MS)

    raw_cmd = aliasing[0]["command"]
    assert raw_cmd[raw_cmd.index("--instant-grid") + 1] == "raw"
    assert raw_cmd[raw_cmd.index("--instant-sample-ms") + 1] == "1000"


def test_fit_plan_pairs_the_primary_fit_with_a_lambda_wait_control() -> None:
    plan = campaign.fit_plan(["dsqwen-7b"], Path("/out"), Path("/raw"), _Args())
    by_label = {r["label"]: r for r in plan["refit"]}
    assert by_label["primary"]["lambda_wait"] == 3.0
    assert by_label["secondary"]["lambda_wait"] == 0.0
    assert plan["acceptance"]["tolerance"] == 0.05
    assert "inert" in plan["acceptance"]["statement"]


# ------------------------------------------------------------------------- cell command


def test_cell_command_passes_the_drain_offset_so_truncation_can_jump_to_it() -> None:
    runnable, _ = campaign.build_plan(_index(), ["dsqwen-7b"])
    ramp = next(c for c in runnable if c.primitive == "ramp")

    class Args(_Args):
        gateway_url = "http://gw/v1/completions"
        raw_dir = Path("/raw")
        model_namespace = "default"
        guard_mode = "warn"
        min_slo_windows = 3
        registry = None
        redis_url = None

    command = campaign.cell_command(ramp, Args(), Path("/s/S1_ramp.json"), Path("/o/out.csv"))
    assert "--drain-start-s" in command
    assert command[command.index("--drain-start-s") + 1] == "375.0"
    assert command[command.index("--min-slo-windows") + 1] == "3"


def test_every_calibration_cell_is_driven_with_the_strict_failure_rules() -> None:
    # These three flags are the campaign's discipline, not r3_grid's defaults. Each one
    # silently pollutes theta by its absence, so each is asserted here by name.
    runnable, _ = campaign.build_plan(_index(), ["dsqwen-7b"])

    class Args(_Args):
        gateway_url = "http://gw/v1/completions"
        raw_dir = Path("/raw")
        model_namespace = "default"
        guard_mode = "warn"
        min_slo_windows = 3
        registry = None
        redis_url = None
        envoy_stats_url = "http://envoy:19000/stats"

    for cell in runnable:
        command = campaign.cell_command(cell, Args(), Path("/s/x.json"), Path("/o/out.csv"))
        # A shed voids the whole cell: keeping the windows from before it keeps exactly
        # the healthy ones and biases theta towards health.
        assert command[command.index("--shed-policy") + 1] == openloop.SHED_POLICY_VOID
        # Ten times tighter than the replay default: a generator that fires late did not
        # offer the load the cell is indexed by.
        assert command[command.index("--max-p99-delay-ms") + 1] == str(
            openloop.CALIBRATION_MAX_P99_DELAY_MS
        )
        assert float(command[command.index("--max-p99-delay-ms") + 1]) == 50.0
        assert command[command.index("--max-model-error-rate") + 1] == "0.05"
        assert command[command.index("--ttft-slo-ms") + 1] == "500.0"
        assert command[command.index("--tpot-slo-ms") + 1] == "75.0"
        # the overflow sentinel travels with every cell when it is configured
        assert command[command.index("--envoy-stats-url") + 1] == "http://envoy:19000/stats"


def test_dry_run_writes_the_plan_without_driving_anything(tmp_path, monkeypatch) -> None:
    index_path = tmp_path / "INDEX.json"
    index_path.write_text(json.dumps(_index()), encoding="utf-8")

    def _no_subprocess(*args, **kwargs):  # pragma: no cover - must never be reached
        raise AssertionError("a dry run must not drive a cell")

    monkeypatch.setattr(campaign.subprocess, "run", _no_subprocess)
    exit_code = campaign.main([
        "--index", str(index_path), "--models", "dsqwen-7b",
        "--out-dir", str(tmp_path / "out"), "--dry-run",
    ])
    assert exit_code == 0
    plan = json.loads((tmp_path / "out" / "plan.json").read_text(encoding="utf-8"))
    assert [c["primitive"] for c in plan["cells"]] == ["steps", "steps", "ramp", "ramp", "bursts"]
    assert plan["admission_cap"]["name"] == admission.GATEWAY_CAPPED.name
    assert len(plan["skipped"]) == 1
    assert (tmp_path / "out" / "fit_plan.json").exists()


# ------------------------------------------------------------------ boundary search


def _rows(count: int, *, ttft: float = 100.0, tpot: float = 10.0, violated: bool = False):
    return [
        {
            "window_start_ms": 1000 * i,
            "window_end_ms": 1000 * (i + 1),
            "p95_ttft": ttft,
            "p95_tpot": tpot,
            "slo_violated": violated,
            "model_errors": 0,
        }
        for i in range(count)
    ]


def _fake_drive(violating_at: float, log=None):
    """A drive seam that violates at or above ``violating_at``."""

    def drive(probe, cell_id, body, meta):
        if log is not None:
            log.append((probe.stage, probe.rho, probe.attempt))
        violated = probe.rho >= violating_at
        rows = _rows(6, ttft=900.0 if violated else 100.0)
        return rows, {"goodput": {"goodput": 0.4 if violated else 0.99}}

    return drive


def _cap():
    return admission.get_cap(admission.DEFAULT_CAP_NAME)


def test_the_search_brackets_the_flip_then_bisects_then_dwells_under_it() -> None:
    log: list = []
    search = campaign.run_boundary_search(
        "dsqwen-7b", "S1", 10.0,
        drive=_fake_drive(0.95, log),
        cap=_cap(), ttft_slo_ms=500.0, tpot_slo_ms=75.0,
    )
    stages = [entry[0] for entry in log]
    assert stages == ["coarse"] * 3 + ["bisect"] * 2 + ["dwell"]
    # 0.6 and 0.9 are healthy, 1.1 violates -> bracket (0.9, 1.1), midpoint 1.0 violates
    # -> (0.9, 1.0), midpoint 0.95 violates -> rho* = 0.95
    assert [entry[1] for entry in log][:5] == [0.6, 0.9, 1.1, 1.0, 0.95]
    assert search.rho_star == 0.95
    assert search.boundary_found
    # the dwell sits UNDER the boundary, where windows land on both sides of it
    assert search.dwell_rho == pytest.approx(0.95 * boundary.DWELL_FRACTION)
    assert log[-1][1] == search.dwell_rho
    assert search.done and not search.stopped_reason


def test_the_three_stages_cost_what_the_plan_says_they_do() -> None:
    seconds = boundary.stage_seconds()
    assert seconds["coarse"] == 3 * 60.0
    assert seconds["bisect"] == 2 * 120.0
    assert seconds["dwell"] == 300.0
    assert boundary.shape_seconds() == 720.0
    assert boundary.probe_count() == 6


def test_a_voided_probe_is_re_driven_and_never_counted_as_healthy() -> None:
    # The failure mode this exists to stop: treating "we could not measure it" as "it did
    # not violate" walks the bracket upwards on every infrastructure hiccup.
    log: list = []
    voided_once = {"seen": False}

    def drive(probe, cell_id, body, meta):
        log.append((probe.rho, probe.attempt))
        if probe.rho == 1.1 and not voided_once["seen"]:
            voided_once["seen"] = True
            return [], {"void_reasons": ["gateway shed"]}
        violated = probe.rho >= 1.1
        return _rows(6, ttft=900.0 if violated else 100.0), {}

    search = campaign.run_boundary_search(
        "dsqwen-7b", "S1", 10.0, drive=drive,
        cap=_cap(), ttft_slo_ms=500.0, tpot_slo_ms=75.0,
    )
    assert (1.1, 1) in log and (1.1, 2) in log
    voided = [r for r in search.results if not r.valid]
    assert len(voided) == 1 and voided[0].void_reasons == ("gateway shed",)
    # the voided attempt moved neither end of the bracket
    assert search.violating_rho == 1.1
    assert search.healthy_rho is not None and search.healthy_rho < 1.1


def test_a_probe_that_keeps_voiding_stops_the_search_instead_of_guessing() -> None:
    search = campaign.run_boundary_search(
        "dsqwen-7b", "S1", 10.0,
        drive=lambda p, c, b, m: ([], {"void_reasons": ["envoy pending overflow"]}),
        cap=_cap(), ttft_slo_ms=500.0, tpot_slo_ms=75.0,
    )
    assert search.stopped_reason and "voided" in search.stopped_reason
    assert search.rho_star is None
    assert search.done


def test_a_shape_that_never_violates_says_so_instead_of_inventing_a_boundary() -> None:
    search = campaign.run_boundary_search(
        "dsqwen-7b", "S1", 10.0,
        drive=_fake_drive(99.0),
        cap=_cap(), ttft_slo_ms=500.0, tpot_slo_ms=75.0,
    )
    assert not search.boundary_found
    assert search.violating_rho is None
    # having failed to flip, the bisection rounds step UP rather than halving inward
    probed = [r.probe.rho for r in search.results]
    assert probed[3] > 1.1 and probed[4] > probed[3]


def test_a_probe_is_judged_on_the_fraction_of_its_windows_not_on_any_one() -> None:
    rows = _rows(9, ttft=100.0)
    rows[0]["p95_ttft"] = 900.0
    violated, violating, total = boundary.probe_violated(
        rows, ttft_slo_ms=500.0, tpot_slo_ms=75.0
    )
    assert (violating, total) == (1, 9) and not violated
    for row in rows[:5]:
        row["p95_ttft"] = 900.0
    violated, violating, _ = boundary.probe_violated(
        rows, ttft_slo_ms=500.0, tpot_slo_ms=75.0
    )
    assert violated and violating == 5


def test_a_window_holding_a_model_error_counts_as_a_violation() -> None:
    rows = _rows(4, ttft=100.0)
    for row in rows[:3]:
        row["slo_violated"] = True
    violated, violating, _ = boundary.probe_violated(
        rows, ttft_slo_ms=500.0, tpot_slo_ms=75.0
    )
    assert violated and violating == 3


def test_a_probe_with_too_few_windows_is_not_scored_as_healthy() -> None:
    result = campaign.probe_result_from_cell(
        boundary.Probe(1.0, 60.0, "coarse"), "i256_o128_c100", [], {},
        ttft_slo_ms=500.0, tpot_slo_ms=75.0,
    )
    assert not result.valid and result.void_reasons == ("no windows",)


def test_read_window_rows_of_a_voided_cell_is_empty_not_healthy(tmp_path) -> None:
    from scripts.r3_grid import CSV_COLUMNS

    path = tmp_path / "cell.csv"
    path.write_text(",".join(CSV_COLUMNS) + "\n", encoding="utf-8")
    assert campaign.read_window_rows(path) == []
    assert campaign.read_window_rows(tmp_path / "missing.csv") == []


# ------------------------------------------------------------------ stopping rule


def test_the_stopping_rule_needs_publish_rate_ci_and_both_families() -> None:
    ok = boundary.stop_rule(
        publish_rate=0.95, theta=1000.0, ci_half_width=50.0,
        family_boundary_windows={"prefill_heavy": 60, "decode_heavy": 55},
    )
    assert ok.satisfied and not ok.reasons and not ok.hold_cells

    thin = boundary.stop_rule(
        publish_rate=0.80, theta=1000.0, ci_half_width=50.0,
        family_boundary_windows={"prefill_heavy": 60, "decode_heavy": 55},
        boundaries={"S3": 1.0},
    )
    assert not thin.satisfied and any("publish rate" in r for r in thin.reasons)

    wide = boundary.stop_rule(
        publish_rate=0.95, theta=1000.0, ci_half_width=150.0,
        family_boundary_windows={"prefill_heavy": 60, "decode_heavy": 55},
        boundaries={"S3": 1.0},
    )
    assert not wide.satisfied and any("CI half width" in r for r in wide.reasons)


def test_a_short_family_is_topped_up_with_hold_cells_never_with_a_new_shape() -> None:
    verdict = boundary.stop_rule(
        publish_rate=0.95, theta=1000.0, ci_half_width=50.0,
        family_boundary_windows={"prefill_heavy": 60, "decode_heavy": 4},
        boundaries={"S4": 0.9, "S5": 1.1},
    )
    assert not verdict.satisfied
    assert any("decode_heavy" in r for r in verdict.reasons)
    assert verdict.hold_cells
    # every remedy is a hold at a shape ALREADY in the set
    assert {c["shape"] for c in verdict.hold_cells} <= {"S4", "S5"}
    assert all(c["stage"] == boundary.STAGE_DWELL for c in verdict.hold_cells)
    assert all(c["rho"] < 1.2 for c in verdict.hold_cells)


# ------------------------------------------------------------------ family verdict


def test_families_inside_the_ci_publish_the_merged_theta() -> None:
    verdict = campaign.family_theta_verdict(
        1000.0, 80.0, {"prefill_heavy": 1040.0, "decode_heavy": 960.0}
    )
    assert verdict["publish"] == "merged" and verdict["theta"] == 1000.0
    assert verdict["outside"] == []


def test_a_family_outside_the_ci_publishes_the_smallest_family_theta() -> None:
    # theta is a health threshold the controller must stay above, so under-claiming
    # health is the direction that fails safe.
    verdict = campaign.family_theta_verdict(
        1000.0, 20.0, {"prefill_heavy": 1300.0, "decode_heavy": 700.0}
    )
    assert verdict["publish"] == "min_family"
    assert verdict["theta"] == 700.0 and verdict["family"] == "decode_heavy"
    assert verdict["outside"] == ["decode_heavy", "prefill_heavy"]


# ------------------------------------------------------------------ held-out data


def test_the_fit_plan_keeps_the_held_out_shape_out_of_training() -> None:
    index = _index()
    for entry in index["schedules"]:
        entry["held_out"] = False
    index["schedules"].append({
        "model": "dsqwen-7b", "shape": gen.MIXTURE_NAME, "primitive": "steps",
        "cell_id": "i0_o0_c95", "held_out": True, "skipped": False,
        "path": "dsqwen-7b/M_steps.json", "duration_s": 450.0, "capacity_rps": 5.0,
    })
    plan = campaign.fit_plan(["dsqwen-7b"], Path("/out"), Path("/raw"), _Args(), index)

    assert plan["held_out_cell_ids"] == ["i0_o0_c95"]
    fitting = next(r for r in plan["rewindow"] if r["purpose"].startswith("fitting"))
    assert "--exclude-cell-id" in fitting["command"]
    assert fitting["command"][fitting["command"].index("--exclude-cell-id") + 1] == "i0_o0_c95"
    validation = next(r for r in plan["rewindow"] if "held-out" in r["purpose"])
    assert validation["command"].count("--only-cell-id") == 1
    assert validation["output"].endswith("_validation.csv")
    # and no refit ever reads the validation CSV
    assert not any(
        str(validation["output"]) in entry["command"] for entry in plan["refit"]
    )


def test_the_fit_plan_adds_a_diagnostic_fit_per_family() -> None:
    index = _index()
    for entry in index["schedules"]:
        entry["shape"] = "S3" if entry["shape"] == "S1" else "S4"
        entry["held_out"] = False
    plan = campaign.fit_plan(["dsqwen-7b"], Path("/out"), Path("/raw"), _Args(), index)
    families = {r["family"] for r in plan["refit"] if r["family"]}
    assert families == {"prefill_heavy", "decode_heavy"}
    for entry in plan["refit"]:
        if entry["family"]:
            assert "diagnostic only" in entry["purpose"]
    assert plan["acceptance"]["family_spread"]["tolerance"] == campaign.FAMILY_SPREAD_TOLERANCE
    assert plan["acceptance"]["stop_rule"]["min_publish_rate"] == boundary.MIN_PUBLISH_RATE


def test_the_held_out_shape_gets_no_boundary_search() -> None:
    # Its probe cells would be generated at campaign time, so their ids cannot be in the
    # index the fit builds its exclusion list from - they would be invisible to the
    # held-out filter and would end up training the fit.
    assert gen.is_held_out(gen.MIXTURE_NAME)
    shapes = [s for s in gen.ALL_SHAPES if not gen.is_held_out(s)]
    assert gen.MIXTURE_NAME not in shapes
    cells = campaign.boundary_plan(["dsqwen-7b"], shapes)
    assert gen.MIXTURE_NAME not in {c["shape"] for c in cells}


def test_the_boundary_plan_prices_the_search_honestly() -> None:
    cells = campaign.boundary_plan(["dsqwen-7b"], ["S1", "S3"])
    assert [c["probes"] for c in cells] == [boundary.probe_count()] * 2
    assert cells[0]["duration_s"] == boundary.shape_seconds() == 720.0
    # 720 s of offered load plus one cooldown per probe cell
    assert campaign.estimate_boundary_wall_clock_s(cells, 45.0) == pytest.approx(
        2 * (720.0 + 45.0 * 6)
    )


def test_every_cell_materialises_its_prompts_into_this_campaigns_output_dir() -> None:
    """Prompts are built before a cell sends, not inside its sends, and they are kept
    with the run that produced them rather than committed next to the schedules."""
    runnable, _ = campaign.build_plan(_index(), ["dsqwen-7b"])

    class Args(_Args):
        out_dir = Path("/campaign/run7")
        gateway_url = "http://gw/v1/completions"
        raw_dir = Path("/raw")
        model_namespace = "default"
        guard_mode = "warn"
        min_slo_windows = 3
        registry = None
        redis_url = None

    for cell in runnable:
        command = campaign.cell_command(cell, Args(), Path("/s/x.json"), Path("/o/out.csv"))
        assert command[command.index("--prompt-dir") + 1] == str(
            Path("/campaign/run7") / "prompts"
        )
