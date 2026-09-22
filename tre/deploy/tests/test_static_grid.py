"""Opt-in static steady-state grid (plan 2026-09-21 §6.9g): planning, list mode, the
schedule it generates and its place in the fit plan. Nothing here drives load."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts import calibration_campaign as campaign
from scripts import gen_calibration_schedules as gen
from scripts import static_grid

MODELS = ("dsqwen-7b", "dsllama-8b", "dsqwen-14b")
#: Synthetic surfaces: capacity 1/C = i/P + o/D and a boundary at a fixed rho* per model.
SURFACE = {"dsqwen-7b": (20000.0, 2000.0, 1.25), "dsllama-8b": (15000.0, 1500.0, 1.2),
           "dsqwen-14b": (8000.0, 800.0, 1.2)}


def _source(tmp_path: Path, *, unfound: tuple[str, ...] = ()) -> Path:
    root = tmp_path / "campaign_20260921"
    for model, (p, d, rho_star) in SURFACE.items():
        (root / model / "capacity").mkdir(parents=True)
        (root / model / "boundary").mkdir(parents=True)
        for shape in gen.TRAINING_SHAPES:
            (_w, i, o), = gen.shape_components(shape)
            i, o = gen._length_mean(i), gen._length_mean(o)
            capacity = 1.0 / (i / p + o / d)
            (root / model / "capacity" / f"{model}_{shape}.json").write_text(json.dumps(
                {"model": model, "shape": shape, "capacity_used_rps": capacity}))
            found = shape not in unfound
            (root / model / "boundary" / f"{model}_{shape}.json").write_text(json.dumps(
                {"model": model, "shape": shape, "boundary_found": found,
                 "rho_star": rho_star if found else 1.859, "stopped_reason": ""}))
    return root


def _cells(tmp_path, **kw):
    source = _source(tmp_path, **kw)
    surfaces = {m: static_grid.load_surface(source, m) for m in MODELS}
    gpus = {"dsqwen-7b": 1, "dsllama-8b": 1, "dsqwen-14b": 2}
    return static_grid.plan_static_grid(MODELS, surfaces, gpus=gpus), surfaces


def test_static_shapes_are_opt_in_and_never_held_out() -> None:
    assert set(gen.STATIC_GRID_SHAPES) == {"G400x160", "G400x320", "G1200x160", "G1200x320"}
    for shape in gen.STATIC_GRID_SHAPES:
        assert shape not in gen.TRAINING_SHAPES and shape not in gen.ALL_SHAPES
        assert shape in gen.training_shapes(static_grid=True)
        assert not gen.is_held_out(shape) and "_" not in shape
    assert gen.families() == {k: tuple(v) for k, v in gen.FAMILIES.items()}
    fam = gen.families(static_grid=True)
    assert fam["prefill_heavy"] == (*gen.PREFILL_FAMILY, "G1200x160")
    assert fam["decode_heavy"] == (*gen.DECODE_FAMILY, "G400x320")
    # the committed families already satisfy the ratio rule the grid is assigned by
    for shape in gen.PREFILL_FAMILY:
        assert gen.static_grid_family(*gen.SHAPES[shape]) == "prefill_heavy"
    for shape in gen.DECODE_FAMILY:
        assert gen.static_grid_family(*gen.SHAPES[shape]) == "decode_heavy"


def test_plan_places_each_cell_relative_to_the_interpolated_boundary(tmp_path) -> None:
    cells, surfaces = _cells(tmp_path)
    assert len(cells) == 3 * 12
    for model, (p, d, rho_star) in SURFACE.items():
        mine = [c for c in cells if c.model == model]
        assert len(mine) == 12
        families = {}
        for c in mine:
            families[c.family] = families.get(c.family, 0) + 1
            capacity = 1.0 / (c.input_tokens / p + c.output_tokens / d)
            assert c.capacity_rps == pytest.approx(capacity, rel=1e-3)
            assert c.rho_star == pytest.approx(rho_star, rel=1e-3)
            assert c.offered_rps == pytest.approx(c.rho_over_rho_star * rho_star * capacity, rel=1e-3)
            assert c.duration_s == 300.0 and not c.held_out
            assert c.cell_id == f"i{c.input_tokens}_o{c.output_tokens}_c{2000 + round(100 * c.rho_over_rho_star)}"
        assert families == {"prefill_heavy": 3, "decode_heavy": 3, None: 6}
        assert sorted({c.rho_over_rho_star for c in mine}) == [0.85, 1.0, 1.1]
    assert len({(c.model, c.cell_id) for c in cells}) == len(cells)


def test_unfound_boundary_is_left_out_of_the_boundary_fit(tmp_path) -> None:
    _cells_, surfaces = _cells(tmp_path, unfound=("S1", "S4"))
    surface = surfaces["dsllama-8b"]
    assert "S1" not in surface.boundary_shapes and "S1" in surface.capacity_shapes
    assert "lower bound" in surface.excluded["S4"]


def test_estimate_counts_gpu_minutes_per_replica_gpus(tmp_path) -> None:
    cells, _ = _cells(tmp_path)
    est = static_grid.estimate(cells, cooldown_s=45.0)
    wall = 12 * (300.0 + 45.0) / 60.0  # 69 min
    assert est["per_model"]["dsqwen-7b"]["wall_min"] == pytest.approx(wall)
    assert est["per_model"]["dsqwen-7b"]["hold_min"] == pytest.approx(60.0)
    assert est["per_model"]["dsqwen-14b"]["gpu_min"] == pytest.approx(2 * wall)
    assert est["total"]["gpu_min"] == pytest.approx(4 * wall)
    assert est["total"]["cells"] == 36


def test_list_mode_prints_cells_and_gpu_minutes_without_driving(tmp_path, capsys, monkeypatch) -> None:
    source = _source(tmp_path)

    def _no_subprocess(*args, **kwargs):  # pragma: no cover - must never be reached
        raise AssertionError("list mode must not drive anything")

    monkeypatch.setattr(campaign.subprocess, "run", _no_subprocess)
    assert campaign.main([
        "--static-grid-list", "--static-grid-source", str(source),
        "--out-dir", str(tmp_path / "out"),
    ]) == 0
    out = capsys.readouterr().out
    print(out)  # the listing itself, visible with -s
    assert out.count("dsqwen-14b   G") == 12
    assert "i1200_o160_c2085" in out and "GPU-min" in out
    [line_14b] = [l for l in out.splitlines() if l.strip().startswith("dsqwen-14b") and "GPU-min" in l]
    assert "x2 GPU =" in line_14b and "138.0 GPU-min" in line_14b  # 12 x (300 + 45) s x 2 GPUs
    assert "total         36 cells" in out
    assert not (tmp_path / "out").exists()


def test_static_schedule_is_one_constant_rate_hold() -> None:
    body, meta = gen.build_schedule_from_capacity_rps(
        "dsqwen-7b", "G1200x160", gen.STATIC_PRIMITIVE, 4.0,
        hold_rho=1.1, hold_duration_s=300.0, static_fraction=0.85,
        capacity_source=static_grid.CAPACITY_SOURCE,
    )
    [segment] = body["dsqwen-7b"]
    assert (segment["start_time"], segment["end_time"], segment["rps"]) == (0, 300, 4.4)
    assert (segment["input_tokens"], segment["max_tokens"]) == (1200, 160)
    assert meta["cell_id"] == "i1200_o160_c2085" and not meta["held_out"]
    assert meta["rho_over_rho_star"] == 0.85 and meta["capacity_source"] == static_grid.CAPACITY_SOURCE
    with pytest.raises(ValueError):
        gen.build_schedule_from_capacity_rps("dsqwen-7b", "G1200x160", gen.STATIC_PRIMITIVE, 4.0,
                                             hold_rho=1.1, hold_duration_s=300.0)


class _Args:
    out_dir = Path("/out")
    window_ms = 30000
    fit_step_ms = 5000
    instant_sample_ms = 1000
    ttft_slo_ms = 500.0
    tpot_slo_ms = 75.0


def test_fit_plan_puts_static_shapes_in_training_and_families_only_when_enabled() -> None:
    off = campaign.fit_plan(["dsqwen-7b"], Path("/out"), Path("/raw"), _Args())
    assert off["training_shapes"] == list(gen.TRAINING_SHAPES)
    assert off["families"] == {k: list(v) for k, v in gen.FAMILIES.items()}
    assert off["static_grid"]["enabled"] is False

    args = _Args()
    args.static_grid = True
    on = campaign.fit_plan(["dsqwen-7b"], Path("/out"), Path("/raw"), args)
    assert set(gen.STATIC_GRID_SHAPES) <= set(on["training_shapes"])
    assert "G1200x160" in on["families"]["prefill_heavy"]
    assert "G400x320" in on["families"]["decode_heavy"]
    assert on["static_grid"]["shapes"]["G400x160"]["family"] is None
    family_rewindow = {r["family"]: r["command"] for r in on["rewindow"] if r.get("family")}
    cmd = family_rewindow["prefill_heavy"]
    assert cmd[cmd.index("--only-shape", cmd.index("--only-shape") + 1) + 1] == "T8"
    assert "G1200x160" in cmd
    theta = {e["label"]: e for e in on["theta"]}
    assert "G400x320" in theta["family_decode_heavy"]["shapes"]


def test_dry_run_with_static_grid_records_cells_and_estimate(tmp_path, monkeypatch) -> None:
    index = {
        "schedules": [
            {"model": "dsqwen-7b", "shape": "S1", "primitive": prim, "cell_id": f"i256_o128_c{code}",
             "duration_s": 400.0, "capacity_rps": 5.0, "skipped": False,
             "path": f"dsqwen-7b/S1_{prim}.json"}
            for prim, code in (("steps", 95), ("ramp", 120), ("bursts", 60))
        ],
        "primitives": {"steps": {"levels": [{"rho": 0.5, "duration_s": 90.0}]}},
        "capacity_models": {},
    }
    index_path = tmp_path / "INDEX.json"
    index_path.write_text(json.dumps(index), encoding="utf-8")
    monkeypatch.setattr(campaign.subprocess, "run", lambda *a, **k: (_ for _ in ()).throw(AssertionError))
    assert campaign.main([
        "--index", str(index_path), "--models", "dsqwen-7b",
        "--out-dir", str(tmp_path / "out"), "--dry-run",
        "--static-grid", "--static-grid-source", str(_source(tmp_path)),
    ]) == 0
    plan = json.loads((tmp_path / "out" / "plan.json").read_text(encoding="utf-8"))
    assert len(plan["cells"]) == 3  # the default stages are untouched
    assert len(plan["static_grid"]["cells"]) == 12
    assert plan["estimated_static_grid_wall_clock_s"] == pytest.approx(12 * 345.0)
    fit = json.loads((tmp_path / "out" / "fit_plan.json").read_text(encoding="utf-8"))
    assert fit["static_grid"]["enabled"] is True
