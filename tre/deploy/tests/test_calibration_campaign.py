from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts import admission_cap as admission
from scripts import calibration_campaign as campaign


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
    window_ms = 30000
    fit_step_ms = 5000
    instant_sample_ms = 1000


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
