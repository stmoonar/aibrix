"""T14 - the held-out 14b test set (``scripts.calibration_t14``, 2026-09-24).

What these pin: the eight T14 shapes are held out, in no default shape list and collide
with no existing name; the composition is 8 shapes x {0.9, 1.0, 1.1} x 240 s holds with
fresh ids / seeds / prompt keys clear of M's serials; C^_s comes from a pre-registered
linear capacity prior (least squares with intercept, training shapes only) that the loader
re-derives and refuses when tampered with; the max-model-len, preregistration and
independent-output checks refuse what they must; a dry run drives nothing and offers
factor x C^_s; a complete run seals T14 like M (read-only manifest that
``dline_refit.check_m_manifest`` accepts); ``--routing-strategy`` reaches r3_grid only when
set; ``--t14-set`` does not combine with the other collection modes.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import pytest

from tre_common import slo_labels

from scripts import adaptive_boundary as boundary
from scripts import calibration_acceptance as ac
from scripts import calibration_campaign as campaign
from scripts import calibration_dataset as dataset
from scripts import calibration_design as design
from scripts import calibration_ladder as ladder
from scripts import calibration_supplement as supplement
from scripts import calibration_t14 as t14
from scripts import dline_refit as dl
from scripts import gen_calibration_schedules as gen

MODEL = "dsqwen-14b"
C0, A, B = -0.005, 1.2e-4, 2.2e-4    # synthetic 1/R = C0 + A*i + B*o (s)
C_S = 2.0                             # every training shape's run-2 C_s


def _write(path: Path, doc) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc), encoding="utf-8")
    return path


def _rate(shape):
    return 1.0 / sum(w * (C0 + A * gen._length_mean(i) + B * gen._length_mean(o))
                     for w, i, o in gen.shape_components(shape))


def _inputs(root: Path) -> dict:
    """A base run, a boundary supplement and a D6' table whose boundary rates follow the
    synthetic linear model exactly (the layout calibration_acceptance.boundary_rates reads)."""
    shapes = t14.PRIOR_SHAPES
    d = root / "base" / MODEL
    _write(d / ladder.DESIGN_RESULT, {"models": [{
        "model": MODEL, "status": "complete", "anchors": {s: 1.0 for s in shapes},
        "anchor_sources": {s: "midpoint" for s in shapes}}]})
    for s in shapes:
        _write(d / "boundary" / f"{MODEL}_{s}.json", {"anchor_rho": 1.0, "prior": {"capacity_rps": C_S}})
    _write(d / ladder.RUN_MANIFEST, {"rho_priors": {"path": "/x", "sha256": "ab" * 32, "parsed": {
        f"{MODEL}/{s}": {"capacity_rps": C_S} for s in shapes}}})
    _write(root / "supp" / MODEL / "boundary" / f"{MODEL}_S3.json", {
        "mode": supplement.MODE, "rho_star_status": boundary.RHO_STAR_MEASURED,
        "anchor_rho": _rate("S3") / C_S, "base": {"capacity_rps": C_S}})
    table = root / "table.csv"
    with open(table, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["model", "shape", "rho_star_fixed", "P_b50_rf"])
        for s in shapes:
            w.writerow([MODEL, s, 1.0, "" if s == "S3" else _rate(s) / C_S])
    return {"base_run": root / "base", "boundary_supplement_run": root / "supp",
            "boundary_table": table}


def _prior(tmp_path: Path, name: str = "prior.json", mutate=None) -> Path:
    ins = _inputs(tmp_path / "inputs")
    doc = t14.build_capacity_prior(MODEL, ins["base_run"], ins["boundary_supplement_run"],
                                   ins["boundary_table"])
    if mutate:
        mutate(doc)
    path = tmp_path / name
    t14.write_capacity_prior(doc, path)
    return path


def _sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _prereg(tmp_path: Path, prior: Path, *, freeze=None, refit=None, name="prereg.json",
            **over) -> Path:
    """A preregistration with its sidecar; ``a__b=value`` overrides doc["a"]["b"]."""
    doc = {"t14": {"capacity_prior": {"sha256": _sha(prior)}, "design_seed": 20260924,
                   "cell_serial_base": t14.CELL_SERIAL_BASE, "factors": [0.9, 1.0, 1.1],
                   "hold_s": 240, "shapes": {"interpolation": list(gen.T14_INTERPOLATION_SHAPES),
                                             "extrapolation": list(gen.T14_EXTRAPOLATION_SHAPES)},
                   "model": MODEL},
           "parameter_sets": {}}
    if freeze:
        doc["parameter_sets"]["freeze"] = {"sha256": _sha(freeze)}
    if refit:
        doc["parameter_sets"]["v1lambda"] = {"sha256": _sha(refit)}
    for dotted, value in over.items():
        cur = doc
        *head, last = dotted.split("__")
        for part in head:
            cur = cur.setdefault(part, {})
        cur[last] = value
    path = tmp_path / name
    path.write_text(json.dumps(doc), encoding="utf-8")
    Path(f"{path}.sha256").write_text(f"{_sha(path)}  {path.name}\n", encoding="utf-8")
    return path


@pytest.fixture
def frozen(tmp_path, monkeypatch):
    """Two parameter files (the D22 freeze, the v1-lambda set) that verify, frozen under the
    14b primary label."""
    label = dl.label_for(MODEL, "primary", None).as_dict()
    paths = {}
    for key in ("freeze", "refit"):
        path = tmp_path / f"{key}.json"
        path.write_text(json.dumps({"models": {MODEL: {"verdict_for_holdout": {"label_def": label}}},
                                    "freeze_sha256": key * 8}), encoding="utf-8")
        paths[key] = path
    monkeypatch.setattr(dl, "verify_freeze", lambda p: json.loads(Path(p).read_text()))
    return paths


def _args(tmp_path, **over):
    base = dict(
        models=MODEL, out_dir=tmp_path / "out", raw_dir=tmp_path / "out" / "raw",
        window_ms=30000, fit_step_ms=10000, instant_sample_ms=1000,
        ttft_slo_ms=500.0, tpot_slo_ms=75.0, index=None, cap=None, design_seed=20260924,
        cooldown_s=45.0, dry_run=False, stop_on_failure=True, controller_namespace="tre-v2",
        model_namespace="default", registry=None, redis_url=None,
        gateway_url="http://gw/v1/completions", guard_mode="warn", min_slo_windows=3,
        max_model_error_rate=0.05, envoy_stats_url=None, envoy_cluster_filter=None,
        freeze_file=None, refit_params_file=None, preregistration_json=None,
        capacity_prior_file=None, routing_strategy="least-gpu-cache",
    )
    base.update(over)
    return argparse.Namespace(**base)


def _windows(start_ms, seconds, *, ttft, tpot=20.0):
    rows, w = [], start_ms
    while w + 30000 <= start_ms + int(seconds * 1000):
        rows.append({"window_start_ms": w, "window_end_ms": w + 30000,
                     "p95_ttft_client_ms": ttft, "p95_tpot_client_ms": tpot,
                     "avg_running": 5.0, "completed_requests": 50,
                     "ttft_len_samples": slo_labels.format_ttft_len_samples([(ttft, 512)] * 50)})
        w += 10000
    return rows


class _Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t

    def sleep(self, s):
        self.t += float(s)


class _FakeCluster:
    def __init__(self):
        self.driven = []
        self.t_ms = 1_790_000_000_000

    def sample(self):
        return {"running": 0, "waiting": 0, "pods_scraped": 1, "scrape_errors": 0}

    def drive(self, cell, attempt, schedule_path, output, prompt_dir):
        body = json.loads(Path(schedule_path).read_text())
        self.driven.append((cell, body))
        raw = Path(output).parent / "raw" / cell.stem(attempt)
        raw.mkdir(parents=True, exist_ok=True)
        (raw / f"{cell.cell_id}.jsonl").write_text(f"{cell.cell_id}\n", encoding="utf-8")
        start = self.t_ms
        self.t_ms += int(cell.duration_s * 1000) + 60_000
        rows = _windows(start, cell.duration_s, ttft=20000.0 if cell.rho > 1.0 else 100.0)
        guard = {"start_ms": start, "end_ms": start + int(cell.duration_s * 1000),
                 "void_reasons": [], "sent": 10, "completed": 10}
        return rows, guard, 0


# ------------------------------------------------------------------------ shapes


def test_the_t14_shapes_are_held_out_and_in_no_default_list() -> None:
    assert t14.SHAPES == ("G512x256", "G1200x240", "G640x400", "G1800x160",
                          "G3072x96", "G4096x64", "G256x768", "G512x1024")
    assert set(gen.T14_SHAPES) == set(t14.SHAPES)
    existing = (set(gen.ACCEPTANCE_SHAPES) | set(gen.STATIC_GRID_SHAPES) | set(gen.SHAPES)
                | set(gen.SAMPLED_SHAPES) | {gen.MIXTURE_NAME})
    for shape in t14.SHAPES:
        assert gen.is_held_out(shape)
        assert shape not in gen.ALL_SHAPES and shape not in gen.TRAINING_SHAPES
        assert shape not in gen.training_shapes(static_grid=True)
        assert shape not in existing and "_" not in shape
        (w, i, o), = gen.shape_components(shape)
        assert w == 1.0 and shape == gen.static_grid_shape_name(i, o)
        assert not gen.is_mixture(shape)
        assert all(shape not in members for members in gen.families(static_grid=True).values())
    # the directory name of a T14 cell resolves to its shape
    from scripts import rewindow_from_raw as rw
    assert rw.raw_dir_shape("dsqwen-14b_G4096x64_acceptance_c1280501_a1", MODEL) == "G4096x64"


def test_a_t14_row_is_in_the_holdout_split_and_in_m_for_dline_refit() -> None:
    assert dataset._split("G4096x64") == design.SPLIT_HOLDOUT
    assert dataset._split("S1") == design.SPLIT_TRAIN
    for role in (design.ROLE_ACCEPTANCE, design.ROLE_LADDER, design.ROLE_BOUNDARY):
        assert design.split_for(role, "G512x256") == design.SPLIT_HOLDOUT
    row = {"model": MODEL, "cell_id": "i512_o256_c1280501", "attempt": "1", "shape": "G512x256",
           "primitive": "hold", "role": "acceptance", "split": "holdout", "stage": "acceptance"}
    assert dl.assign_set(row, sealed_to_h2=False, sentinels=True) == dl.SET_M
    with pytest.raises(dl.TrainingSetError):
        dl.assign_set({**row, "split": "train"}, sealed_to_h2=False, sentinels=True)


# -------------------------------------------------------------------- composition


def test_the_composition_is_24_fresh_holdout_holds() -> None:
    cells = t14.new_cells(MODEL, 20260924)
    assert len(cells) == 24
    assert sorted((c.shape, c.rho_factor) for c in cells) == sorted(
        (s, f) for s in t14.SHAPES for f in (0.9, 1.0, 1.1))
    for c in cells:
        assert c.duration_s == 240.0 and c.warmup_s == design.WARMUP_S == 60.0
        assert c.role == design.ROLE_ACCEPTANCE and c.split == design.SPLIT_HOLDOUT
        assert c.profile == design.PROFILE_HOLD and c.primitive == gen.HOLD_PRIMITIVE
        assert c.rho == c.rho_factor
        assert c.prompt_key == f"p20260924.{c.cell_id}"
        assert c.arrival_seed == design.derived_seed(20260924, MODEL, c.cell_id, "arrivals")
        (_w, i, o), = gen.shape_components(c.shape)
        assert c.cell_id == f"i{i}_o{o}_c{c.code}"
    serials = sorted(c.serial for c in cells)
    assert serials == list(range(80_501, 80_525))
    assert len({c.cell_id for c in cells}) == 24 and len({c.code for c in cells}) == 24
    assert len({c.arrival_seed for c in cells}) == 24
    # clear of every earlier collection's serials: M's probes 70_001+ and cells 70_501+,
    # the boundary supplement 50_001+, the training supplement 60_001+
    for other in (ac.SERIAL_BASE, ac.CELL_SERIAL_BASE, supplement.SUPPLEMENT_SERIAL_BASE, 60_000):
        assert not any(other < s < other + 500 for s in serials)
    t14.check_held_out(cells)
    with pytest.raises(ValueError, match="dsqwen-14b set"):
        t14.new_cells("dsqwen-7b", 20260924)


def test_the_order_interleaves_the_shapes_and_is_seeded() -> None:
    cells = t14.new_cells(MODEL, 20260924)
    order = t14.interleaved_order(cells, MODEL, 20260924)
    assert sorted(c.cell_id for c in order) == sorted(c.cell_id for c in cells)
    assert all(a.shape != b.shape for a, b in zip(order, order[1:]))
    again = t14.interleaved_order(t14.new_cells(MODEL, 20260924), MODEL, 20260924)
    assert [c.cell_id for c in again] == [c.cell_id for c in order]


def test_a_cell_that_is_not_held_out_is_refused() -> None:
    cells = t14.new_cells(MODEL, 20260924)
    f = design.CellFactory(MODEL, 20260924, serial_base=t14.CELL_SERIAL_BASE + 100)
    with pytest.raises(ValueError, match="not held out"):
        t14.check_held_out([*cells, f.new("S1", design.ROLE_LADDER, 240.0, rho=1.0)])


# ------------------------------------------------------------------ capacity prior


def test_the_linear_fit_recovers_known_coefficients() -> None:
    pts = [{"input_mean": i, "output_mean": o, "rate_rps": 1.0 / (C0 + A * i + B * o)}
           for i, o in ((256, 128), (768, 192), (2048, 96), (256, 448), (1600, 112), (900, 300))]
    c0, ci, co = t14.fit_linear_capacity(pts)
    assert c0 == pytest.approx(C0, abs=1e-12) and ci == pytest.approx(A, rel=1e-9)
    assert co == pytest.approx(B, rel=1e-9)
    with pytest.raises(ValueError, match="degenerate"):
        t14.fit_linear_capacity([{"input_mean": 100, "output_mean": 100, "rate_rps": 1.0}] * 4)
    assert t14.predicted_rate("G512x256", C0, A, B) == pytest.approx(1.0 / (C0 + A * 512 + B * 256))
    with pytest.raises(ValueError, match="non-positive"):
        t14.predicted_rate("G512x256", -1.0, A, B)


def test_the_prior_builder_fits_the_training_boundaries(tmp_path) -> None:
    path = _prior(tmp_path)
    doc = t14.load_capacity_prior(path, MODEL)
    co = doc["coefficients"]
    assert co["c0_s"] == pytest.approx(C0, rel=1e-4)
    assert co["c_in_s_per_token"] == pytest.approx(A, rel=1e-4)
    assert co["c_out_s_per_token"] == pytest.approx(B, rel=1e-4)
    assert [p["shape"] for p in doc["fit_points"]] == list(t14.PRIOR_SHAPES)
    assert all(abs(x["loo_error"]) < 1e-3 for x in doc["leave_one_out"])
    for shape in t14.SHAPES:
        assert doc["predicted_rps"][shape] == pytest.approx(_rate(shape), rel=1e-4)
    assert doc["form"] == t14.PRIOR_FORM and doc["sha256"] == _sha(path)
    assert oct(path.stat().st_mode & 0o777) == "0o444"
    with pytest.raises(ValueError, match="written once"):
        t14.write_capacity_prior(json.loads(path.read_text()), path)


@pytest.mark.parametrize("mutate, match", [
    (lambda d: d["fit_points"][0].update(shape="G800x240"), "not a training shape"),
    (lambda d: d["fit_points"][0].update(shape="M"), "not a training shape"),
    (lambda d: d["fit_points"][0].update(shape="G512x256"), "not a training shape"),
    (lambda d: d["predicted_rps"].update(G512x256=d["predicted_rps"]["G512x256"] * 1.01),
     "recomputed from the coefficients"),
    (lambda d: d.update(model="dsqwen-7b"), "capacity prior of 'dsqwen-7b'"),
    (lambda d: d["predicted_rps"].pop("G4096x64"), "misses T14 shapes"),
    (lambda d: d.update(fit_points=d["fit_points"][:3]), "need >= 4"),
    (lambda d: d["coefficients"].update(c0_s=d["coefficients"]["c0_s"] * 1.1),
     "not the least-squares fit"),
])
def test_the_prior_loader_refuses_a_tampered_prior(tmp_path, mutate, match) -> None:
    path = _prior(tmp_path, mutate=mutate)
    with pytest.raises(ValueError, match=match):
        t14.load_capacity_prior(path, MODEL)


def test_the_prior_loader_refuses_the_wrong_model(tmp_path) -> None:
    path = _prior(tmp_path)
    with pytest.raises(ValueError, match="not 'dsqwen-7b'"):
        t14.load_capacity_prior(path, "dsqwen-7b")


# ------------------------------------------------------------------------ checks


def test_max_model_len_is_checked_against_the_longest_request() -> None:
    assert t14.registry_max_model_len(MODEL) == 12288
    ok = t14.check_max_model_len(MODEL, t14.SHAPES)
    assert ok["ok"] and ok["max_model_len"] == 12288
    assert ok["longest_shape"] == "G4096x64" and ok["longest_request_tokens"] == 4160
    assert ok["needed"] == 4160 + t14.MAX_MODEL_LEN_MARGIN == 4224
    assert t14.check_max_model_len(MODEL, t14.SHAPES, max_model_len=12288)["ok"]
    t14.check_max_model_len(MODEL, t14.SHAPES, max_model_len=4224)
    with pytest.raises(ValueError, match="--max-model-len 4096 < 4224"):
        t14.check_max_model_len(MODEL, t14.SHAPES, max_model_len=4096)


def test_the_preregistration_binds_the_run(tmp_path, frozen) -> None:
    prior = _prior(tmp_path)
    fr = t14.check_param_file(frozen["freeze"], MODEL, None, "--freeze-file")
    rf = t14.check_param_file(frozen["refit"], MODEL, None, "--refit-params-file")
    kw = dict(capacity_sha256=_sha(prior), design_seed=20260924, freeze=fr, refit=rf)
    good = _prereg(tmp_path, prior, freeze=frozen["freeze"], refit=frozen["refit"])
    body = t14.check_preregistration(good, **kw)
    assert body["sha256"] == _sha(good) and body["unchecked"] == []
    assert set(body["checked_keys"]) == set(t14.PREREG_KEYS)
    # the list form of t14.shapes is accepted too
    t14.check_preregistration(_prereg(tmp_path, prior, name="list.json",
                                      t14__shapes=list(t14.SHAPES)), **kw)
    bad = {
        "capacity": dict(t14__capacity_prior={"sha256": "00" * 32}),
        "seed": dict(t14__design_seed=20260923),
        "serial": dict(t14__cell_serial_base=70_500),
        "factors": dict(t14__factors=[0.9, 1.0, 1.15]),
        "hold": dict(t14__hold_s=300),
        "shapes": dict(t14__shapes=list(t14.SHAPES)[::-1]),
        "model": dict(t14__model="dsqwen-7b"),
        "freeze": dict(parameter_sets__freeze={"sha256": "11" * 32}),
        "refit": dict(parameter_sets__v1lambda={"sha256": "22" * 32}),
    }
    for name, over in bad.items():
        path = _prereg(tmp_path, prior, name=f"bad_{name}.json", **over)
        with pytest.raises(ValueError, match="does not bind"):
            t14.check_preregistration(path, **kw)
    # the capacity prior the run loads is another file than the one preregistered
    with pytest.raises(ValueError, match="t14.capacity_prior.sha256"):
        t14.check_preregistration(good, **{**kw, "capacity_sha256": "ff" * 32})
    with pytest.raises(ValueError, match="t14.design_seed"):
        t14.check_preregistration(good, **{**kw, "design_seed": 1})
    # a missing key, a stale sidecar, no sidecar
    doc = json.loads(good.read_text())
    doc["t14"].pop("hold_s")
    missing = tmp_path / "missing.json"
    missing.write_text(json.dumps(doc))
    Path(f"{missing}.sha256").write_text(f"{_sha(missing)}  missing.json\n")
    with pytest.raises(ValueError, match="t14.hold_s missing"):
        t14.check_preregistration(missing, **kw)
    good.write_text(good.read_text() + " ")
    with pytest.raises(ValueError, match="changed after its sidecar"):
        t14.check_preregistration(good, **kw)
    Path(f"{missing}.sha256").unlink()
    with pytest.raises(ValueError, match="no sha256 sidecar"):
        t14.check_preregistration(missing, **kw)
    # a dry run without the refit file reports its preregistered hash as unchecked
    part = t14.check_preregistration(
        _prereg(tmp_path, prior, freeze=frozen["freeze"], refit=frozen["refit"], name="p.json"),
        **{**kw, "refit": None})
    assert part["unchecked"] == ["parameter_sets.v1lambda.sha256 (no --refit-params-file)"]


def test_a_parameter_file_frozen_under_another_label_is_refused(tmp_path, frozen) -> None:
    other = dl.label_for(MODEL, "fixed", None).as_dict()
    frozen["refit"].write_text(json.dumps({"models": {MODEL: {"verdict_for_holdout": {
        "label_def": other}}}}))
    with pytest.raises(ValueError, match="another label"):
        t14.check_param_file(frozen["refit"], MODEL, None, "--refit-params-file")
    with pytest.raises(ValueError, match="no parameter file"):
        t14.check_param_file(tmp_path / "absent.json", MODEL, None, "--freeze-file")


def test_the_output_root_is_independent(tmp_path) -> None:
    assert {p.name for p in t14.FORBIDDEN_ROOTS} >= {
        "calibration_rev2_20260923", "calibration_run2_main_20260923", "calibration_supp_20260923",
        "calibration_supp3_20260923", "calibration_M_20260923", "calibration_refit_final_20260923",
        "calibration_resplit_20260924", "calibration_freeze_20260923"}
    root = tmp_path / "calibration_M_20260923"
    root.mkdir()
    roots = (root,)
    with pytest.raises(ValueError, match="--out-dir .* overlaps"):
        t14.check_output_roots(root / "t14" / MODEL, tmp_path / "raw", roots)
    with pytest.raises(ValueError, match="--raw-dir .* overlaps"):
        t14.check_output_roots(tmp_path / "t14", root / "raw", roots)
    with pytest.raises(ValueError, match="overlaps"):
        t14.check_output_roots(tmp_path, tmp_path / "raw", roots)  # a root inside the out-dir
    busy = tmp_path / "busy"
    busy.mkdir()
    (busy / "x").write_text("x")
    with pytest.raises(ValueError, match="not empty"):
        t14.check_output_roots(busy, tmp_path / "raw", roots)
    (tmp_path / "empty").mkdir()
    t14.check_output_roots(tmp_path / "empty", tmp_path / "raw", roots)
    t14.check_output_roots(tmp_path / "new", tmp_path / "raw", roots)
    # the real list refuses the real roots
    with pytest.raises(ValueError, match="overlaps"):
        t14.check_output_roots(Path("/data/nfs_shared_data/xxy/calibration_M_20260923/t14"),
                               tmp_path / "raw")
    cells = t14.new_cells(MODEL, 20260924)
    (tmp_path / "raw" / cells[3].stem(1)).mkdir(parents=True)
    with pytest.raises(ValueError, match="already exist"):
        t14.check_fresh_raw(tmp_path / "raw", cells)


# ----------------------------------------------------------------------- the CLI


def test_routing_strategy_reaches_r3_grid_only_when_set(tmp_path) -> None:
    cell = campaign.Cell(MODEL, "G512x256", "hold", "i512_o256_c1280501", "s.json", 240.0, 0.0)
    plain = campaign.cell_command(cell, _args(tmp_path, routing_strategy=None),
                                  Path("s.json"), Path("o.csv"))
    assert "--routing-strategy" not in plain
    old = _args(tmp_path)
    delattr(old, "routing_strategy")            # a Namespace from before the flag existed
    assert campaign.cell_command(cell, old, Path("s.json"), Path("o.csv")) == plain
    routed = campaign.cell_command(cell, _args(tmp_path), Path("s.json"), Path("o.csv"))
    assert routed[routed.index("--routing-strategy") + 1] == "least-gpu-cache"
    assert routed[:len(plain)] == plain and len(routed) == len(plain) + 2
    dc = t14.new_cells(MODEL, 20260924)[0]
    full = ladder.design_cell_command(cell, dc, _args(tmp_path), Path("s.json"), Path("o.csv"),
                                      tmp_path / "p")
    assert full[full.index("--routing-strategy") + 1] == "least-gpu-cache"
    assert full[full.index("--prompt-key") + 1] == dc.prompt_key


@pytest.mark.parametrize("extra", [
    ["--acceptance-set"], ["--training-supplement"], ["--static-grid"], ["--static-grid-only"],
    ["--reprobe-shapes", "dsqwen-14b:S1"], ["--design", "primitives"],
    ["--base-run", "/x"], ["--boundary-table", "/x"], ["--retained-dataset", "/x"],
])
def test_t14_does_not_combine_with_the_other_modes(tmp_path, extra) -> None:
    argv = ["--t14-set", "--models", MODEL, "--capacity-prior-file", str(tmp_path / "p.json"),
            "--out-dir", str(tmp_path / "o"), "--dry-run", *extra]
    with pytest.raises(SystemExit):
        campaign.main(argv)
    assert not (tmp_path / "o").exists()


@pytest.mark.parametrize("argv", [
    ["--capacity-prior-file", "/x"], ["--refit-params-file", "/x"],
    ["--preregistration-json", "/x"], ["--t14-set"],
])
def test_t14_flags_belong_to_t14(tmp_path, argv) -> None:
    with pytest.raises(SystemExit):
        campaign.main(["--models", MODEL, "--out-dir", str(tmp_path / "o"), "--dry-run", *argv])


def test_the_cli_dry_run_plans_24_cells_and_drives_nothing(tmp_path, capsys) -> None:
    prior = _prior(tmp_path)
    out = tmp_path / "dry"
    argv = ["--t14-set", "--models", MODEL, "--capacity-prior-file", str(prior),
            "--out-dir", str(out), "--raw-dir", str(out / "raw"), "--dry-run",
            "--index", str(tmp_path / "absent-index.json")]
    assert campaign.main(argv) == 0
    text = capsys.readouterr().out
    assert "no --freeze-file" in text and "no --refit-params-file" in text
    assert "no --preregistration-json" in text and "--routing-strategy is None" in text
    assert "max-model-len check: 12288 >= 4224" in text and "wall clock" in text
    plan = json.loads((out / "plan.json").read_text())
    assert plan["mode"] == t14.MODE and plan["design_seed"] == 20260924
    assert plan["cell_serial_base"] == 80_500
    assert len(plan["order"]) == 24 and len(plan["cells_preview"]) == 24
    pred = json.loads(prior.read_text())["predicted_rps"]
    for c in plan["cells_preview"]:
        assert c["offered_rps"] == pytest.approx(c["rho_factor"] * pred[c["shape"]], rel=1e-12)
        assert c["split"] == design.SPLIT_HOLDOUT and c["prompt_key"].startswith("p20260924.")
    assert [c["cell_id"] for c in plan["cells_preview"]] == plan["order"]
    est = plan["estimate"]
    assert est["offered_load_s"] == 24 * 240.0
    tails = sum(design.request_timeout_s(s) for s in t14.SHAPES)
    assert est["seconds_expected"] == pytest.approx(24 * 240.0 + tails + 24 * (45.0 + 6.0))
    assert plan["capacity_prior"]["sha256"] == _sha(prior)
    assert not (out / ladder.RUN_MANIFEST).exists() and not (out / ladder.LEDGER).exists()
    assert not (out / "raw").exists() and not (out / t14.T14_MANIFEST).exists()


def test_a_real_run_needs_its_bindings(tmp_path, frozen) -> None:
    prior = _prior(tmp_path)
    prereg = _prereg(tmp_path, prior, freeze=frozen["freeze"], refit=frozen["refit"])
    full = dict(capacity_prior_file=prior, freeze_file=frozen["freeze"],
                refit_params_file=frozen["refit"], preregistration_json=prereg)
    for missing, match in (("freeze_file", "--freeze-file"),
                           ("refit_params_file", "--refit-params-file"),
                           ("preregistration_json", "--preregistration-json")):
        with pytest.raises(ValueError, match=match):
            t14.run_t14_set(_args(tmp_path, **{**full, missing: None}), check_controller=False)
    with pytest.raises(ValueError, match="least-gpu-cache"):
        t14.run_t14_set(_args(tmp_path, routing_strategy=None, **full), check_controller=False)
    with pytest.raises(ValueError, match="dsqwen-14b set"):
        t14.run_t14_set(_args(tmp_path, models="dsqwen-7b", **full), check_controller=False)
    with pytest.raises(ValueError, match="does not bind"):
        t14.run_t14_set(_args(tmp_path, design_seed=20260923, **full), check_controller=False)
    assert not (tmp_path / "out").exists()


def test_a_complete_run_offers_factor_x_c_hat_and_seals_t14(tmp_path, frozen) -> None:
    prior = _prior(tmp_path)
    prereg = _prereg(tmp_path, prior, freeze=frozen["freeze"], refit=frozen["refit"])
    args = _args(tmp_path, capacity_prior_file=prior, freeze_file=frozen["freeze"],
                 refit_params_file=frozen["refit"], preregistration_json=prereg)
    clock, fake = _Clock(), _FakeCluster()
    code = t14.run_t14_set(args, drive=fake.drive, sample_factory=lambda _m: fake.sample,
                           sleep=clock.sleep, clock=clock, check_controller=False)
    assert code == 0
    pred = json.loads(prior.read_text())["predicted_rps"]
    assert len(fake.driven) == 24
    for cell, body in fake.driven:
        (segs,) = body.values()
        assert len(segs) == 1 and segs[0]["end_time"] == 240.0
        assert segs[0]["rps"] == pytest.approx(cell.rho_factor * pred[cell.shape], abs=1e-4)
        (_w, i, o), = gen.shape_components(cell.shape)
        assert (segs[0]["input_tokens"], segs[0]["max_tokens"]) == (i, o)
    out = tmp_path / "out"
    records = [json.loads(x) for x in (out / ladder.LEDGER).read_text().splitlines()]
    assert len(records) == 24 and {r["split"] for r in records} == {design.SPLIT_HOLDOUT}
    for r in records:
        row = {k: r[k] for k in ("model", "cell_id", "attempt", "shape", "primitive", "role",
                                 "split", "stage")}
        assert dl.assign_set(row, sealed_to_h2=False, sentinels=True) == dl.SET_M
    sched = Path(records[0]["schedule_path"])
    meta = json.loads(sched.with_name(sched.stem + ".meta.json").read_text())
    assert meta["capacity_source"] == t14.CAPACITY_SOURCE and meta["held_out"]
    man_path = out / t14.T14_MANIFEST
    man = json.loads(man_path.read_text())
    assert oct(man_path.stat().st_mode & 0o777) == "0o444"
    assert oct((out / t14.T14_SHA256SUMS).stat().st_mode & 0o777) == "0o444"
    assert man["model"] == MODEL and len(man["cells"]) == 24 and man["sealed_probes"] == []
    assert man["capacity_prior"]["sha256"] == _sha(prior)
    assert man["parameter_sets"]["freeze"]["sha256"] == _sha(frozen["freeze"])
    assert man["parameter_sets"]["v1lambda"]["sha256"] == _sha(frozen["refit"])
    assert man["freeze"]["sha256"] == _sha(frozen["freeze"])
    assert man["preregistration"]["sha256"] == _sha(prereg)
    assert man["label_def"] == dl.label_for(MODEL, "primary", None).as_dict()
    assert man["label_def_sha256"] == dl.canonical_sha256(man["label_def"])
    assert man["run_manifest_sha256"] == _sha(out / ladder.RUN_MANIFEST)
    kinds = {c["shape"]: c["kind"] for c in man["cells"]}
    assert {s for s, k in kinds.items() if k == "interpolation"} == set(gen.T14_INTERPOLATION_SHAPES)
    for c in man["cells"]:
        assert c["raw_files"]
        assert c["offered_rps"] == pytest.approx(c["factor"] * pred[c["shape"]], rel=1e-4)
    listed = {}
    for line in (out / t14.T14_SHA256SUMS).read_text().splitlines():
        digest, path = line.split("  ", 1)
        listed[path] = digest
        assert _sha(Path(path)) == digest
    for c in man["cells"]:
        assert all(f in listed for f in c["raw_files"])
    assert str((out / ladder.LEDGER).resolve()) in listed
    assert str((out / ladder.RUN_MANIFEST).resolve()) in listed
    freeze_doc = json.loads(frozen["freeze"].read_text())
    _man, problems = dl.check_m_manifest(man_path, freeze_doc, _sha(frozen["freeze"]))
    assert problems == []
    status = json.loads((out / campaign.CAMPAIGN_STATUS_FILE).read_text())
    assert status["status"] == "complete"
    # an interrupted / repeated run is not resumed into the same root
    with pytest.raises(ValueError, match="not empty"):
        t14.run_t14_set(args, check_controller=False)
