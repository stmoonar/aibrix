"""Boundary search fixes of plan 2026-09-21 §6.11 (training supplement, gate 1): 90 s
coarse probes, outward extension, labelled rho*, re-probe mode, static-grid gate.
Nothing here drives load: every probe answers from a table."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts import adaptive_boundary as boundary
from scripts import admission_cap as admission
from scripts import calibration_campaign as campaign
from scripts import gen_calibration_schedules as gen
from scripts import static_grid


def _rows(count: int, *, violated: bool):
    return [{"window_start_ms": 30_000 * i, "p95_ttft": 900.0 if violated else 100.0,
             "p95_tpot": 10.0, "slo_violated": False, "model_errors": 0} for i in range(count)]


def _drive(pred, log=None, windows=lambda probe: 6):
    def drive(probe, cell_id, body, meta):
        if log is not None:
            log.append((probe.stage, probe.rho, probe.duration_s))
        return _rows(windows(probe), violated=pred(probe.rho)), {}
    return drive


def _search(pred, log=None, **kw):
    return campaign.run_boundary_search(
        "dsqwen-7b", "S1", 10.0, drive=_drive(pred, log, **kw),
        cap=admission.get_cap(admission.DEFAULT_CAP_NAME), ttft_slo_ms=500.0, tpot_slo_ms=75.0,
    )


def test_coarse_probes_are_90s_and_the_flag_reaches_the_search() -> None:
    assert boundary.COARSE_SECONDS == 90.0 and boundary.LEGACY_COARSE_SECONDS == 60.0
    log: list = []
    _search(lambda r: r >= 0.95, log)
    assert [d for st, _r, d in log if st == "coarse"] == [90.0] * 3
    args = campaign.argparse.Namespace(boundary_coarse_s=60.0)
    assert campaign.new_boundary_search("dsqwen-7b", "S1", args).coarse_seconds == 60.0


def test_a_too_short_probe_moves_neither_end_of_the_bracket() -> None:
    # 60 s probes (2 windows) were read as healthy even at 2/2 violating (7b T8 @ 0.6)
    s = boundary.BoundarySearch(extend_down_rhos=(), extend_up_rhos=())
    p = s.next_probe()
    s.record(boundary.ProbeResult(p, violated=False, windows=2, violating_windows=2, conclusive=False))
    assert s.healthy_rho is None and s.violating_rho is None
    r = campaign.probe_result_from_cell(p, "i256_o128_c1060", _rows(2, violated=True), {},
                                        ttft_slo_ms=500.0, tpot_slo_ms=75.0)
    assert r.valid and not r.conclusive


def test_search_extends_downward_when_the_lowest_coarse_probe_violates() -> None:
    log: list = []
    s = _search(lambda r: r >= 0.4, log)
    stages = [(st, r) for st, r, _d in log]
    assert stages[:5] == [("coarse", 0.6), ("coarse", 0.9), ("coarse", 1.1), ("extend", 0.45), ("extend", 0.3)]
    assert 0.3 <= s.healthy_rho < s.violating_rho <= 0.45
    assert [st for st, _ in stages[5:]] == ["bisect", "bisect", "dwell"]
    assert s.status()["status"] == boundary.RHO_STAR_MEASURED


def test_extension_stops_at_the_first_probe_that_closes_the_bracket() -> None:
    log: list = []
    s = _search(lambda r: r >= 1.8, log)
    ext = [r for st, r, _d in log if st == "extend"]
    assert ext == [1.5, 2.0]                      # 2.6 never driven
    assert s.as_dict()["rho_star_status"] == boundary.RHO_STAR_MEASURED
    assert 1.5 <= s.rho_star <= 2.0


def test_no_flip_after_extension_is_a_bound_not_a_bisection_point() -> None:
    log: list = []
    up = _search(lambda r: False, log)
    assert [r for st, r, _d in log if st == "extend"] == [1.5, 2.0, 2.6]
    assert not any(st in ("bisect", "dwell") for st, _r, _d in log)
    d = up.as_dict()
    assert d["rho_star_status"] == boundary.RHO_STAR_LOWER_BOUND and d["rho_star"] == 2.6
    assert "lower bound" in d["unresolved_reason"] and not d["stopped_reason"] and up.done
    down = _search(lambda r: True)
    assert down.as_dict()["rho_star_status"] == boundary.RHO_STAR_UPPER_BOUND
    assert down.rho_star == 0.3 and "upper bound" in down.unresolved_reason


def test_rho_star_status_of_the_20260921_campaign_records() -> None:
    def p(stage, rho, windows, violating):
        return {"stage": stage, "rho": rho, "windows": windows, "violating_windows": violating,
                "violated": windows >= 3 and violating >= 0.5 * windows, "valid": True}

    coarse = [p("coarse", 0.6, 2, 0), p("coarse", 0.9, 2, 1), p("coarse", 1.1, 2, 2)]
    # dsllama-8b T9: rho* 1.20175 only because the 2-window coarse probes read as healthy
    t9 = coarse + [p("bisect", 1.43, 4, 4), p("bisect", 1.265, 4, 4), p("dwell", 1.20175, 10, 10)]
    assert boundary.rho_star_status(t9)["status"] == boundary.RHO_STAR_GRID_ARTIFACT
    # dsllama-8b S1: nothing violated up to 1.859
    s1 = coarse[:1] + [p("bisect", 1.43, 4, 0), p("bisect", 1.859, 4, 0), p("dwell", 1.76605, 10, 2)]
    assert boundary.rho_star_status(s1)["status"] == boundary.RHO_STAR_LOWER_BOUND
    # dsqwen-7b S1: healthy 1.43, violating 1.76605 -> 19 % bracket, a grid point
    w = coarse[:1] + [p("bisect", 1.43, 4, 0), p("bisect", 1.859, 4, 4), p("dwell", 1.76605, 10, 10)]
    st = boundary.rho_star_status(w)
    assert st["status"] == boundary.RHO_STAR_GRID_ARTIFACT and st["bracket_rel_width"] > 0.1
    # dsqwen-14b T9: healthy dwell 1.20175 under violating 1.265 -> 5 % bracket, measured
    m = coarse[:1] + [p("bisect", 1.43, 4, 2), p("bisect", 1.265, 4, 2), p("dwell", 1.20175, 10, 0)]
    assert boundary.rho_star_status(m)["status"] == boundary.RHO_STAR_MEASURED
    assert boundary.saved_rho_star_status({"probes": m, "rho_star_status": "lower_bound"}) == "lower_bound"


# ------------------------------------------------------------------- re-probe mode


def _source(tmp_path: Path, statuses: dict) -> Path:
    root = tmp_path / "src"
    for model, by_shape in statuses.items():
        (root / model / "capacity").mkdir(parents=True)
        (root / model / "boundary").mkdir(parents=True)
        for shape in gen.TRAINING_SHAPES:
            (_w, i, o), = gen.shape_components(shape)
            cap = 1.0 / (gen._length_mean(i) / 20000.0 + gen._length_mean(o) / 2000.0)
            (root / model / "capacity" / f"{model}_{shape}.json").write_text(json.dumps({
                "model": model, "shape": shape, "capacity_prior_rps": cap, "capacity_measured_rps": cap,
                "capacity_used_rps": cap, "capacity_source": "measured_steps", "saturated": True,
                "ttft_slo_ms": 500.0, "tpot_slo_ms": 75.0, "levels": [], "note": ""}))
            status = by_shape.get(shape, "measured")
            (root / model / "boundary" / f"{model}_{shape}.json").write_text(json.dumps({
                "model": model, "shape": shape, "boundary_found": status != "lower_bound",
                "rho_star": 1.2, "stopped_reason": "", "rho_star_status": status}))
    return root


def test_parse_reprobe_shapes() -> None:
    assert campaign.parse_reprobe_shapes(["dsllama-8b:S1,S4", "dsqwen-7b:T8", "dsllama-8b:S4,T8"]) == {
        "dsllama-8b": ["S1", "S4", "T8"], "dsqwen-7b": ["T8"]}
    for bad in (["dsqwen-7b"], ["dsqwen-7b:M"], ["dsqwen-7b:G400x160"], ["dsqwen-7b:"]):
        with pytest.raises(ValueError):
            campaign.parse_reprobe_shapes(bad)


def test_reprobe_refuses_an_existing_or_overlapping_root(tmp_path) -> None:
    src = _source(tmp_path, {"dsqwen-7b": {}})
    with pytest.raises(ValueError):
        campaign.check_new_output_root(src / "dsqwen-7b" / "reprobe", src)
    with pytest.raises(ValueError):
        campaign.check_new_output_root(src, src)
    busy = tmp_path / "busy"
    busy.mkdir()
    (busy / "x").write_text("")
    with pytest.raises(ValueError):
        campaign.check_new_output_root(busy, src)
    campaign.check_new_output_root(tmp_path / "fresh", src)


def test_reprobe_dry_run_plans_only_the_listed_shapes(tmp_path, monkeypatch) -> None:
    src = _source(tmp_path, {"dsllama-8b": {}})
    monkeypatch.setattr(campaign.subprocess, "run", lambda *a, **k: pytest.fail("dry run drove a cell"))
    out = tmp_path / "reprobe_new"
    assert campaign.main(["--reprobe-shapes", "dsllama-8b:S1,S4", "--reprobe-source", str(src),
                          "--out-dir", str(out), "--dry-run"]) == 0
    plan = json.loads((out / "reprobe_plan.json").read_text())
    assert [(e["model"], e["shape"]) for e in plan["targets"]] == [("dsllama-8b", "S1"), ("dsllama-8b", "S4")]
    assert plan["coarse_seconds"] == 90.0 and plan["extend_up_rhos"] == [1.5, 2.0, 2.6]
    assert sorted(p.name for p in out.iterdir()) == ["reprobe_plan.json"]
    with pytest.raises(SystemExit):   # the same root again: not new any more
        campaign.main(["--reprobe-shapes", "dsllama-8b:S1", "--reprobe-source", str(src),
                       "--out-dir", str(out), "--dry-run"])


def test_reprobe_writes_new_boundaries_under_the_new_root_only(tmp_path, monkeypatch) -> None:
    src = _source(tmp_path, {"dsllama-8b": {}})
    before = {p: p.read_text() for p in src.rglob("*.json")}
    monkeypatch.setattr(campaign, "controller_mode", lambda ns: campaign.REQUIRED_CONTROLLER_MODE)
    monkeypatch.setattr(campaign.time, "sleep", lambda s: None)
    seen = []

    def fake_drive(cell, measured, args, *, cap, schedule_dir, out_dir):
        seen.append((cell.model, cell.shape, Path(args.raw_dir), Path(out_dir)))
        search = campaign.new_boundary_search(cell.model, cell.shape, args)
        return campaign.run_boundary_search(
            cell.model, cell.shape, measured.capacity_used_rps, drive=_drive(lambda r: r >= 2.2),
            cap=cap, ttft_slo_ms=500.0, tpot_slo_ms=75.0, search=search)

    monkeypatch.setattr(campaign, "drive_boundary_search", fake_drive)
    out = tmp_path / "reprobe_new"
    assert campaign.main(["--reprobe-shapes", "dsllama-8b:S4", "--reprobe-source", str(src),
                          "--out-dir", str(out), "--index", str(tmp_path / "no_index.json")]) == 0
    assert seen == [("dsllama-8b", "S4", out / "dsllama-8b" / "raw", out / "dsllama-8b")]
    body = json.loads((out / "dsllama-8b" / "boundary" / "dsllama-8b_S4.json").read_text())
    assert body["rho_star_status"] == "measured" and 2.0 <= body["rho_star"] <= 2.6
    assert (out / "dsllama-8b" / "capacity" / "dsllama-8b_S4.json").exists()
    assert {p: p.read_text() for p in src.rglob("*.json")} == before   # source untouched


# ------------------------------------------------------------------ static-grid gate


class _GridArgs:
    def __init__(self, source, overlay=None, allow=False):
        self.static_grid_source = source
        self.static_grid_reprobe = overlay
        self.static_grid_allow_unmeasured = allow
        self.static_grid_hold_s = gen.STATIC_GRID_HOLD_S
        self.registry = None


def test_static_grid_refuses_unmeasured_source_rho_star(tmp_path, capsys) -> None:
    src = _source(tmp_path, {"dsqwen-7b": {"T8": "grid_artifact", "S1": "lower_bound"}})
    with pytest.raises(SystemExit) as exc:
        campaign.plan_static_cells(_GridArgs(src), ["dsqwen-7b"])
    assert "S1=lower_bound" in str(exc.value) and "T8=grid_artifact" in str(exc.value)
    cells, surfaces = campaign.plan_static_cells(_GridArgs(src, allow=True), ["dsqwen-7b"])
    assert len(cells) == 12 and "WARNING" in capsys.readouterr().err
    assert surfaces["dsqwen-7b"].rho_star_status["T8"] == "grid_artifact"


def test_a_reprobe_overlay_replaces_the_unmeasured_shapes(tmp_path) -> None:
    src = _source(tmp_path, {"dsqwen-7b": {"T8": "grid_artifact"}})
    overlay = tmp_path / "reprobe"
    (overlay / "dsqwen-7b" / "boundary").mkdir(parents=True)
    (overlay / "dsqwen-7b" / "boundary" / "dsqwen-7b_T8.json").write_text(json.dumps({
        "model": "dsqwen-7b", "shape": "T8", "boundary_found": True, "rho_star": 0.5,
        "stopped_reason": "", "rho_star_status": "measured"}))
    cells, surfaces = campaign.plan_static_cells(_GridArgs(src, overlay=overlay), ["dsqwen-7b"])
    s = surfaces["dsqwen-7b"]
    assert static_grid.unmeasured_shapes(s) == {}
    assert s.boundary_files["T8"].startswith(str(overlay)) and len(cells) == 12
