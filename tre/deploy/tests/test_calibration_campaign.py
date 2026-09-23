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
    by_label = {r["label"]: r for r in plan["theta"]}
    assert by_label["primary"]["lambda_wait"] == 1.0  # the D-line's lambda_wait
    assert by_label["secondary"]["lambda_wait"] == 0.0
    assert plan["acceptance"]["tolerance"] == 0.05
    assert "inert" in plan["acceptance"]["statement"]


def test_fit_plan_runs_the_whole_pipeline_in_order_on_one_label() -> None:
    # B5: rewindow -> theta/delta (cli) -> bootstrap/stop-rule verdict -> alt -> hold-out.
    plan = campaign.fit_plan(["dsqwen-7b"], Path("/out"), Path("/raw"), _Args())
    assert plan["order"] == ["rewindow", "alpha", "dline", "theta", "verdict", "ablation", "alt", "holdout"]
    for entry in plan["theta"]:
        cmd = entry["command"]
        assert cmd[1:3] == ["-m", "tre_calibration.cli"]
        assert "--recompute-tss" in cmd and "--e2e-p95-ms" not in cmd
        assert cmd[cmd.index("--ttft-p95-ms") + 1] == "500.0"
        assert cmd[cmd.index("--tpot-p95-ms") + 1] == "75.0"
    scopes = {(e["family"], e["lambda_wait"]) for e in plan["theta"]}
    assert scopes == {(f, lw) for f in ("", "prefill_heavy", "decode_heavy") for lw in (1.0, 0.0)}
    [verdict] = plan["verdict"]
    assert "scripts.theta_verdict" in verdict["command"] and "verdict" in verdict["command"]
    assert verdict["command"].count("--family") == 2
    assert sorted(e["signal"] for e in plan["alt"]) == ["decode_tps", "prefill_tps", "queue_len"]
    for entry in plan["alt"]:
        assert "--registry" not in entry["command"]
        assert "--ttft-p95-ms" in entry["command"] and "--tpot-p95-ms" in entry["command"]
        # plan 6.9 item 5: the alt fit gets the families, the primary label and writes
        # verdicts the hold-out step scores - the same pipeline as TSS.
        assert entry["command"].count("--family") == 2
        assert entry["command"][entry["command"].index("--label-lambda-wait") + 1] == "1.0"
        assert "--verdict-dir" in entry["command"]
    arms = {e["arm"]: e for e in plan["ablation"]}
    assert set(arms) == {"tss_lw0", "tss_wp1"}
    assert (arms["tss_lw0"]["lambda_wait"], arms["tss_wp1"]["w_p"]) == (0.0, 1.0)
    for entry in plan["ablation"]:
        cmd = entry["command"]
        assert cmd[cmd.index("--label-lambda-wait") + 1] == "1.0"
        assert cmd.count("--family") == 2
    assert plan["label_def"]["e2e"] == "excluded"


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


def test_a_cell_is_windowed_like_the_fit_so_its_probe_verdict_is_the_fit_s_label() -> None:
    class Args(_Args):
        gateway_url = "http://gw/v1/completions"
        raw_dir = Path("/raw")
        model_namespace = "default"
        guard_mode = "warn"
        min_slo_windows = 3
        registry = None
        redis_url = None

    args = Args()
    cell = campaign.Cell("dsqwen-7b", "S1", "hold", "i256_o128_c1090", "s.json", 90.0, 10.0)
    command = campaign.cell_command(cell, args, Path("s.json"), Path("out.csv"))
    assert command[command.index("--window-ms") + 1] == str(args.window_ms)
    assert command[command.index("--step-ms") + 1] == str(args.fit_step_ms)
    plan = campaign.fit_plan(["dsqwen-7b"], Path("/out"), Path("/raw"), args, {})
    fitting = plan["rewindow"][0]["command"]
    assert fitting[fitting.index("--window-ms") + 1] == str(args.window_ms)
    assert fitting[fitting.index("--step-ms") + 1] == str(args.fit_step_ms)
    assert fitting[fitting.index("--tpot-slo-ms") + 1] == str(args.tpot_slo_ms)


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

    real_run = campaign.subprocess.run

    def _no_subprocess(argv, *args, **kwargs):
        # Reading the code commit for the plan's provenance is fine; driving is not.
        if argv and argv[0] == "git":
            return real_run(argv, *args, **kwargs)
        raise AssertionError("a dry run must not drive a cell")  # pragma: no cover

    monkeypatch.setattr(campaign.subprocess, "run", _no_subprocess)
    exit_code = campaign.main([
        "--design", "primitives",
        "--index", str(index_path), "--models", "dsqwen-7b",
        "--out-dir", str(tmp_path / "out"), "--dry-run",
    ])
    assert exit_code == 0
    plan = json.loads((tmp_path / "out" / "plan.json").read_text(encoding="utf-8"))
    assert [c["primitive"] for c in plan["cells"]] == ["steps", "steps", "ramp", "ramp", "bursts"]
    assert plan["admission_cap"]["name"] == admission.GATEWAY_CAPPED.name
    assert len(plan["skipped"]) == 1
    assert (tmp_path / "out" / "fit_plan.json").exists()
    # the plan says what it was made with and which ruler it labels with
    provenance = plan["provenance"]
    assert provenance["label"]["latency_source"] == "client per-request"
    assert provenance["registry_sha256"] and len(provenance["registry_sha256"]) == 64
    # D8: 30 s windows ending on the 10 s gateway grid; D6': the primary label is the
    # slowdown TTFT label, recorded per model
    assert (provenance["window_ms"], provenance["step_ms"]) == (30000, 10000)
    assert provenance["window_align"] == "grid"
    assert provenance["label"]["mode"] == "slowdown"
    assert set(provenance["label_by_model"]) == {"dsqwen-7b"}


# ------------------------------------------------------------------ boundary search


def _rows(count: int, *, ttft: float = 100.0, tpot: float = 10.0, model_errors: int = 0):
    return [
        {
            "window_start_ms": 1000 * i,
            "window_end_ms": 1000 * (i + 1),
            "p95_ttft_client_ms": ttft,
            "p95_tpot_client_ms": tpot,
            "model_errors": model_errors,
        }
        for i in range(count)
    ]


def _sliding_rows(duration_s: float, *, tpot: float, window_ms: int = 30000, step_ms: int = 5000,
                  start_ms: int = 1_790_011_888_589):
    """What rewindow_from_raw lays over a probe: 30 s windows sliding by 5 s."""
    rows = []
    w = start_ms
    while w + window_ms <= start_ms + int(duration_s * 1000):
        rows.append({
            "window_start_ms": w, "window_end_ms": w + window_ms,
            "p95_ttft_client_ms": 200.0, "p95_tpot_client_ms": tpot,
        })
        w += step_ms
    return rows


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
    assert seconds["coarse"] == 3 * 90.0
    assert seconds["bisect"] == 2 * 120.0
    assert seconds["dwell"] == 300.0
    assert boundary.shape_seconds() == 810.0
    assert boundary.probe_count() == 6


def test_a_coarse_probe_is_long_enough_to_be_conclusive_on_the_fitting_window() -> None:
    # Three disjoint 30 s windows need 90 s; the 60 s this started at could never be.
    assert boundary.COARSE_SECONDS >= boundary.min_probe_seconds(30000)
    assert boundary.min_probe_seconds(30000) == 90.0


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
    rows[0]["p95_ttft_client_ms"] = 900.0
    verdict = boundary.probe_verdict(rows, ttft_slo_ms=500.0, tpot_slo_ms=75.0)
    assert (verdict.violating_windows, verdict.labeled_windows) == (1, 9)
    assert verdict.verdict == boundary.VERDICT_HEALTHY
    for row in rows[:5]:
        row["p95_ttft_client_ms"] = 900.0
    verdict = boundary.probe_verdict(rows, ttft_slo_ms=500.0, tpot_slo_ms=75.0)
    assert verdict.verdict == boundary.VERDICT_VIOLATED and verdict.violating_windows == 5


def test_a_window_holding_a_model_error_counts_as_a_violation() -> None:
    rows = _rows(4, ttft=100.0)
    for row in rows[:3]:
        row["model_errors"] = 1
    verdict = boundary.probe_verdict(rows, ttft_slo_ms=500.0, tpot_slo_ms=75.0)
    assert verdict.verdict == boundary.VERDICT_VIOLATED and verdict.violating_windows == 3


def test_a_probe_with_no_windows_is_inconclusive_not_healthy_and_not_void() -> None:
    result = campaign.probe_result_from_cell(
        boundary.Probe(1.0, 90.0, "coarse"), "i256_o128_c100", [], {},
        ttft_slo_ms=500.0, tpot_slo_ms=75.0,
    )
    assert result.verdict == boundary.VERDICT_INCONCLUSIVE
    assert result.valid and not result.void_reasons


# --------------------------------------------- the 2026-09-21 coarse-probe regression


def test_the_2026_09_21_coarse_probe_that_violated_everywhere_is_not_recorded_healthy() -> None:
    # What every coarse probe of that campaign looked like: 60 s driven, judged on two
    # 30 s windows. dsqwen-7b T8 at rho 0.6 had BOTH windows violating (2/2) and the old
    # rule - "fewer than 3 rows -> not violated" - booked it as healthy, walking rho*
    # upwards. It must not be healthy now.
    rows = [
        {"window_start_ms": 1_790_011_888_589, "window_end_ms": 1_790_011_918_589,
         "p95_ttft_client_ms": 480.0, "p95_tpot_client_ms": 150.0},
        {"window_start_ms": 1_790_011_918_589, "window_end_ms": 1_790_011_948_589,
         "p95_ttft_client_ms": 480.0, "p95_tpot_client_ms": 150.0},
    ]
    verdict = boundary.probe_verdict(rows, ttft_slo_ms=500.0, tpot_slo_ms=75.0)
    assert verdict.violating_windows == 2
    assert verdict.verdict == boundary.VERDICT_INCONCLUSIVE

    search = boundary.BoundarySearch(model="dsqwen-7b", shape="T8")
    probe = search.next_probe()
    search.record(campaign.probe_result_from_cell(
        probe, "i1600_o112_c1060", rows, {}, ttft_slo_ms=500.0, tpot_slo_ms=75.0,
    ))
    assert search.healthy_rho is None and search.violating_rho is None
    redrive = search.next_probe()
    assert redrive.rho == probe.rho and redrive.attempt == 2
    assert redrive.duration_s == probe.duration_s * boundary.INCONCLUSIVE_DURATION_FACTOR


def test_sliding_rows_do_not_count_as_independent_evidence() -> None:
    # The same 60 s probe on the fitting windows (30 s sliding by 5 s) has 7 rows - more
    # than the 3-window floor - but only 2 disjoint 30 s spans. Counting rows would let it
    # through on two windows' worth of information.
    rows = _sliding_rows(60, tpot=150.0)
    assert len(rows) == 7
    verdict = boundary.probe_verdict(rows, ttft_slo_ms=500.0, tpot_slo_ms=75.0)
    assert verdict.independent_windows == 2
    assert verdict.verdict == boundary.VERDICT_INCONCLUSIVE
    # the 90 s coarse probe has three, and is decided
    verdict = boundary.probe_verdict(
        _sliding_rows(boundary.COARSE_SECONDS, tpot=150.0), ttft_slo_ms=500.0, tpot_slo_ms=75.0
    )
    assert verdict.independent_windows == 3 and verdict.verdict == boundary.VERDICT_VIOLATED


# ------------------------------------------------------------------ three-state verdict


def test_verdict_violated_moves_the_upper_end_of_the_bracket() -> None:
    search = boundary.BoundarySearch()
    probe = search.next_probe()
    search.record(campaign.probe_result_from_cell(
        probe, "c", _sliding_rows(90, tpot=150.0), {}, ttft_slo_ms=500.0, tpot_slo_ms=75.0,
    ))
    assert search.results[-1].verdict == boundary.VERDICT_VIOLATED
    assert search.violating_rho == probe.rho and search.healthy_rho is None
    assert search.next_probe().rho == search.coarse_rhos[1]


def test_verdict_healthy_moves_the_lower_end_of_the_bracket() -> None:
    search = boundary.BoundarySearch()
    probe = search.next_probe()
    search.record(campaign.probe_result_from_cell(
        probe, "c", _sliding_rows(90, tpot=20.0), {}, ttft_slo_ms=500.0, tpot_slo_ms=75.0,
    ))
    assert search.results[-1].verdict == boundary.VERDICT_HEALTHY
    assert search.healthy_rho == probe.rho and search.violating_rho is None


def test_verdict_inconclusive_moves_nothing_and_stops_when_it_repeats() -> None:
    search = boundary.BoundarySearch()
    for _ in range(2):
        probe = search.next_probe()
        # every window unlabeled: too few completions for a p95
        rows = [dict(r, p95_tpot_client_ms=None) for r in _sliding_rows(probe.duration_s, tpot=0.0)]
        search.record(campaign.probe_result_from_cell(
            probe, "c", rows, {}, ttft_slo_ms=500.0, tpot_slo_ms=75.0,
        ))
        assert search.results[-1].verdict == boundary.VERDICT_INCONCLUSIVE
        assert search.healthy_rho is None and search.violating_rho is None
    assert search.done and "inconclusive" in search.stopped_reason
    assert search.rho_star is None


def test_a_probe_result_cannot_claim_a_verdict_it_does_not_have() -> None:
    probe = boundary.Probe(1.0, 90.0, "coarse")
    with pytest.raises(ValueError):
        boundary.ProbeResult(probe=probe, verdict="false")
    with pytest.raises(ValueError):
        boundary.ProbeResult(probe=probe, verdict=boundary.VERDICT_VOID)  # no reasons
    with pytest.raises(ValueError):
        boundary.ProbeResult(probe=probe, verdict=boundary.VERDICT_HEALTHY,
                             void_reasons=("gateway shed",))


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
        publish_rate=0.95, theta=1000.0, ci_half_width=160.0,
        family_boundary_windows={"prefill_heavy": 60, "decode_heavy": 55},
        boundaries={"S3": 1.0},
    )
    assert not wide.satisfied and any("CI half width" in r for r in wide.reasons)


def test_d13_stop_gate_is_15_percent_and_10_percent_is_only_reported() -> None:
    assert boundary.MAX_CI_HALF_WIDTH_FRACTION == 0.15
    assert boundary.APPENDIX_CI_HALF_WIDTH_FRACTION == 0.10
    families = {"prefill_heavy": 60, "decode_heavy": 55}
    mid = boundary.stop_rule(publish_rate=0.95, theta=1000.0, ci_half_width=120.0,
                             family_boundary_windows=families)
    assert mid.satisfied and mid.appendix_ci_target_met is False
    d = mid.as_dict()
    assert d["ci_half_width_fraction"] == 0.12 and d["max_ci_half_width_fraction"] == 0.15
    assert d["appendix_ci_half_width_fraction"] == 0.10 and d["appendix_ci_target_met"] is False
    tight = boundary.stop_rule(publish_rate=0.95, theta=1000.0, ci_half_width=80.0,
                               family_boundary_windows=families)
    assert tight.satisfied and tight.appendix_ci_target_met is True


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


def test_a_family_outside_the_ci_publishes_the_largest_family_theta() -> None:
    # B1 (plan 6.3): Z = TSS/theta and Z < tau_crit is CRITICAL, so the larger theta is
    # the one that errs towards adding capacity - the conservative choice.
    verdict = campaign.family_theta_verdict(
        1000.0, 20.0, {"prefill_heavy": 1300.0, "decode_heavy": 700.0}
    )
    assert verdict["publish"] == "max_family"
    assert verdict["theta"] == 1300.0 and verdict["family"] == "prefill_heavy"
    assert verdict["outside"] == ["decode_heavy", "prefill_heavy"]


def test_the_published_family_theta_is_the_one_that_classifies_more_windows_critical() -> None:
    # Check the direction claim against the controller's own classifier, not a comment.
    from tre_controller.planning.classify import ModelState, TauThresholds, classify_model

    tau = TauThresholds.from_control(0.2, 0.25)
    verdict = campaign.family_theta_verdict(
        1000.0, 20.0, {"prefill_heavy": 1300.0, "decode_heavy": 700.0}
    )
    tss = 900.0  # Z = 0.69 under 1300 (CRITICAL), 1.29 under 700 (HIGH)

    def state(theta: float) -> ModelState:
        return classify_model(
            model_name="m", trs=tss, Z_m=tss / theta, eta_m=None, theta_m=theta, tau=tau
        ).state

    assert state(verdict["theta"]) == ModelState.CRITICAL
    assert state(min(verdict["family_thetas"].values())) != ModelState.CRITICAL


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
    # and no fitting step ever reads the validation CSV - only the hold-out step does
    for step in ("theta", "verdict", "ablation", "alt"):
        assert not any(str(validation["output"]) in entry["command"] for entry in plan[step])
    # one hold-out per verdict: TSS, the two ablation arms and every alt signal
    assert sorted(h["arm"] for h in plan["holdout"]) == sorted(
        ["tss", "tss_lw0", "tss_wp1", "queue_len", "decode_tps", "prefill_tps"]
    )
    for holdout in plan["holdout"]:
        assert str(validation["output"]) in holdout["command"]
    # the family CSVs drop the held-out cells too
    for entry in plan["rewindow"]:
        if entry.get("family"):
            assert "--exclude-cell-id" in entry["command"]


def test_the_fit_plan_adds_a_diagnostic_fit_per_family() -> None:
    index = _index()
    for entry in index["schedules"]:
        entry["shape"] = "S3" if entry["shape"] == "S1" else "S4"
        entry["held_out"] = False
    plan = campaign.fit_plan(["dsqwen-7b"], Path("/out"), Path("/raw"), _Args(), index)
    families = {r["family"] for r in plan["theta"] if r["family"]}
    assert families == {"prefill_heavy", "decode_heavy"}
    for entry in plan["theta"]:
        if entry["family"]:
            assert "diagnostic only" in entry["purpose"]
    # B3: family membership is resolved from the raw tree at fit time (--only-shape), so
    # boundary hold cells created after the index was written are included.
    for entry in plan["rewindow"]:
        if entry.get("family"):
            assert "--only-cell-id" not in entry["command"]
            shapes = [entry["command"][i + 1] for i, a in enumerate(entry["command"]) if a == "--only-shape"]
            assert shapes == list(gen.FAMILIES[entry["family"]])
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
    assert cells[0]["duration_s"] == boundary.shape_seconds() == 810.0
    # 810 s of offered load plus one cooldown per probe cell
    assert campaign.estimate_boundary_wall_clock_s(cells, 45.0) == pytest.approx(
        2 * (810.0 + 45.0 * 6)
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


# ================================================= a voided cell is re-run, not skipped
#
# The failure these replace: --guard-mode warn printed a WARNING per voided cell, wrote
# 0 rows, carried on, and finished with "campaign complete" after five hours.


def test_a_voided_cell_is_re_run_in_place() -> None:
    attempts: list[int] = []

    def drive(attempt: int) -> dict:
        attempts.append(attempt)
        return {"void_reasons": ["gateway admission overflow"]} if attempt == 1 else {}

    guard, attempt = campaign.drive_until_valid("i256_o128_c95", drive)
    assert attempts == [1, 2]
    assert attempt == 2 and guard == {}


def test_a_cell_that_voids_twice_stops_the_campaign_and_says_why() -> None:
    attempts: list[int] = []

    def drive(attempt: int) -> dict:
        attempts.append(attempt)
        return {"void_reasons": ["transient proxy error rate"]}

    with pytest.raises(campaign.CellVoided) as excinfo:
        campaign.drive_until_valid("i256_o128_c95", drive)
    assert attempts == [1, 2]
    assert excinfo.value.void_reasons == ("transient proxy error rate",)
    assert "voided on all 2 attempt(s)" in str(excinfo.value)


def test_the_scheduled_cells_retry_rule_is_the_boundary_search_s_own_rule() -> None:
    # One rule in one place. Two copies of "re-run once, then stop" drift, and the one
    # that drifts is the one nobody is watching.
    assert boundary.next_void_attempt(1) == 2
    assert boundary.next_void_attempt(2) is None
    assert boundary.MAX_VOID_RETRIES == 1


def test_a_re_run_writes_beside_the_attempt_it_replaces_not_over_it() -> None:
    # The raw JSONL is appended to, so a second attempt writing the same path would pool
    # the capture that failed with the one that replaced it.
    base = Path("/out/dsqwen-7b_S1_steps.csv")
    assert campaign.attempt_output_path(base, 1) == base
    assert campaign.attempt_output_path(base, 2) == Path("/out/dsqwen-7b_S1_steps_a2.csv")


def test_a_driver_that_died_without_writing_a_verdict_is_itself_a_void(tmp_path) -> None:
    # Absence of a verdict is not a passing verdict.
    guard = campaign.read_cell_guard(
        tmp_path, tmp_path / "dsqwen-7b_S1_steps.csv", "i256_o128_c95", returncode=1
    )
    assert guard["void_reasons"] == ["driver exited 1"]


def test_the_guard_artifact_is_read_back_from_where_the_driver_wrote_it(tmp_path) -> None:
    output = tmp_path / "dsqwen-7b_S1_steps.csv"
    cell_dir = tmp_path / output.stem
    cell_dir.mkdir()
    (cell_dir / "i256_o128_c95.guard.json").write_text(
        json.dumps({"void_reasons": ["model error rate"], "sent": 10}), encoding="utf-8"
    )
    guard = campaign.read_cell_guard(tmp_path, output, "i256_o128_c95")
    assert guard["void_reasons"] == ["model error rate"] and guard["sent"] == 10


def test_cooldown_shorter_than_the_metrics_window_is_rejected(tmp_path) -> None:
    with pytest.raises(SystemExit):
        campaign.main([
            "--index", str(tmp_path / "INDEX.json"), "--models", "dsqwen-7b",
            "--out-dir", str(tmp_path / "out"), "--dry-run",
            "--cooldown-s", "20", "--window-ms", "30000",
        ])
    assert not (tmp_path / "out").exists()
