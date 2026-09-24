#!/usr/bin/env python3
"""T14 - the held-out 14b test set (degraded version, 24 cells), 2026-09-24.

Entered from ``python -m scripts.calibration_campaign --t14-set --models dsqwen-14b
--capacity-prior-file <prior> --freeze-file <D22 freeze> --refit-params-file <v1-lambda
freeze> --preregistration-json <prereg> --routing-strategy least-gpu-cache ...``. Only for
evaluation: every T14 cell is split holdout (``gen.is_held_out``), nothing that fits may
read it.

Composition (:data:`SHAPES` x :data:`FACTORS`, 24 cells)
--------------------------------------------------------
Eight new fixed-length shapes (``gen.T14_SHAPES``, weight 1.0, static-grid naming):

* interpolation - G512x256, G1200x240, G640x400, G1800x160 (inside the seven training
  shapes' input 256-2048 / output 96-448 range);
* extrapolation - G3072x96, G4096x64 (longer prompts than S3), G256x768, G512x1024
  (longer generations than S4).

Each is driven as a constant-load hold at {0.9, 1.0, 1.1} x C^_s for :data:`HOLD_S` = 240 s,
standard hold warm-up (``design.WARMUP_S`` = 60 s dropped by the verdict / fit). The order
interleaves the shapes (seeded: ``derived_seed(design_seed, model, "t14-order")``,
``calibration_training_supplement.interleave``). Cell ids from serial
:data:`CELL_SERIAL_BASE` + 1 (M used 70_000 / 70_500, the supplements 50_000 / 60_000),
role ``acceptance`` (the role only names what the cell is; the split is holdout because
the shape is held out), arrival seed ``derived_seed(design_seed, model, cell_id,
"arrivals")`` and prompt key ``p<design_seed>.<cell_id>`` (default design seed 20260924).

C^_s: a pre-registered linear capacity model, no probing
-------------------------------------------------------
There are no boundary probes. C^_s (rps) comes from ``--capacity-prior-file``, a JSON the
``capacity-prior`` subcommand of this module builds once, before the run, from the seven
training shapes' D6' boundary rates (``calibration_acceptance.boundary_rates``: S1 S2 S3
S4 S5 T8 T9 - the same points M's prior used): ordinary least squares WITH intercept of
``1/R = c0 + c_in * input_mean + c_out * output_mean``, leave-one-shape-out errors, the
predicted rate of every T14 shape. :func:`load_capacity_prior` refuses a prior fitted on
anything but training shapes, with fewer than 4 points, whose coefficients are not the
least-squares fit of its points, whose ``predicted_rps`` differ from the coefficients, or
that misses a T14 shape. A cell's schedule capacity is C^_s and its rho is the factor, so
it offers ``factor x C^_s`` rps (``capacity_source`` ``t14_capacity_prior``).

    python -m scripts.calibration_t14 capacity-prior --model dsqwen-14b \\
        --base-run <run2 main> --boundary-supplement-run <supp> --boundary-table <csv> \\
        --out <file>                         (refuses to overwrite; the file is made read-only)

Checks before anything is driven (a dry run runs them too)
----------------------------------------------------------
* held out: every composed cell's shape is ``gen.is_held_out`` and its split holdout;
* ``--max-model-len`` of the model in the registry (``vllm_extra_args``) >= the longest
  input + output of the composition + :data:`MAX_MODEL_LEN_MARGIN` (G4096x64 needs 4160;
  14b pins 12288);
* the parameter files: ``--freeze-file`` (the D22 freeze) and ``--refit-params-file`` (the
  second parameter set, a ``dline_refit freeze`` of the v1-lambda refit) both verify
  (``dline_refit.verify_freeze``) and were frozen under the primary label this set is
  judged by (canonical sha256 equality). Required for a real run, warned about in a dry run;
* the preregistration (``--preregistration-json``, sidecar ``<file>.sha256`` in sha256sum
  format): :func:`check_preregistration` - the keys it checks are listed there
  (:data:`PREREG_KEYS`). Required for a real run;
* ``--routing-strategy least-gpu-cache`` (the tre-v2 ext_proc gateway path, NodePort 31094,
  headers ``model`` + ``routing-strategy``). Required for a real run;
* independent output: ``--out-dir`` absent or empty and, like ``--raw-dir``, outside every
  training / M / resplit / freeze root (:data:`FORBIDDEN_ROOTS`); no raw directory of a T14
  cell may pre-exist under ``--raw-dir``;
* (real run) the controller is in ``observe``.

What is written, and the seal
-----------------------------
As every ladder-machinery collection: ``plan.json``, the frozen ``run_manifest.json``
(before the first cell), ``cells.jsonl`` (one line per attempt, void rule included - a void
cell is re-driven once, a second void stops the run), ``schedules/``, the online CSVs,
``prompts/`` and the raw captures under ``--raw-dir`` (use ``<out-dir>/raw``, as M did).
After the last cell: ``T14_SHA256SUMS`` (sha256sum format, absolute paths) over every file
of the out-dir (``dataset/`` excluded) and every cell's raw capture, then
``T14_manifest.json`` (read-only): the cells (model / cell_id / attempt / shape /
interpolation-or-extrapolation / factor / rps / raw files), the capacity prior's sha256,
both parameter files' sha256, the preregistration's sha256, the label definition and its
canonical sha256, the run manifest's sha256 and how to evaluate. It carries the keys of an
M manifest (``dline_refit.check_m_manifest`` accepts it against the D22 freeze). Then
``training.finish`` - which, like M's, builds the standard dataset
(``campaign.finalize_run`` -> ``calibration_dataset.build_dataset``) into
``<out-dir>/dataset/``; to rebuild it by hand::

    python -m scripts.calibration_dataset <out-dir>

Interrupted runs: there is no resume
------------------------------------
The ladder machinery has none: ``run_manifest.json`` is written once and refused if it
exists, and this mode refuses a non-empty out-dir and pre-existing raw directories of its
cells. A run that stops (Ctrl-C, a second void, a driver failure) finishes its
``design_result.json`` / ``campaign_status.json`` and builds a dataset of what it has, but
is never sealed. The rule: re-run the whole set into a NEW out-dir (and raw-dir); the
partial root is kept, unsealed, as void evidence and never pooled with the complete one.
The same design seed gives the same cell ids and seeds in the new root (the preregistration
binds the seed), so the two roots must never be read together.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import stat
import sys
import time
from pathlib import Path
from typing import Callable, Mapping, Optional, Sequence

from scripts import calibration_acceptance as acceptance
from scripts import calibration_campaign as campaign
from scripts import calibration_design as design
from scripts import calibration_ladder as ladder
from scripts import calibration_training_supplement as training
from scripts import gen_calibration_schedules as gen

MODE = "t14_set"
MODEL = "dsqwen-14b"
CELL_SERIAL_BASE = 80_500
DEFAULT_DESIGN_SEED = 20260924
FACTORS: tuple[float, ...] = (0.9, 1.0, 1.1)
HOLD_S = 240.0
INTERPOLATION_SHAPES: tuple[str, ...] = tuple(gen.T14_INTERPOLATION_SHAPES)
EXTRAPOLATION_SHAPES: tuple[str, ...] = tuple(gen.T14_EXTRAPOLATION_SHAPES)
SHAPES: tuple[str, ...] = (*INTERPOLATION_SHAPES, *EXTRAPOLATION_SHAPES)
KIND_INTERPOLATION, KIND_EXTRAPOLATION = "interpolation", "extrapolation"
CAPACITY_SOURCE = "t14_capacity_prior"
ROUTING_STRATEGY = "least-gpu-cache"
MAX_MODEL_LEN_MARGIN = 64
MIN_FIT_POINTS = 4
PRIOR_SHAPES: tuple[str, ...] = tuple(acceptance.PRIOR_SHAPES)
PRIOR_FORM = "1/R = c0 + c_in*input_mean + c_out*output_mean (least squares, intercept)"
T14_MANIFEST = "T14_manifest.json"
T14_SHA256SUMS = "T14_SHA256SUMS"
MANIFEST_FORMAT_REVISION = 1   # the M manifest format (dline_refit.M_MANIFEST_FORMAT_REVISION)

#: Roots this collection must stay out of: the training rounds, their supplements, M, the
#: refits, the resplit and the freeze.
_DATA = Path("/data/nfs_shared_data/xxy")
FORBIDDEN_ROOTS: tuple[Path, ...] = tuple(_DATA / name for name in (
    "calibration_20260921", "calibration_run2_20260923", "calibration_rev2_20260923",
    "calibration_run2_main_20260923", "calibration_supp_20260923", "calibration_supp3_20260923",
    "calibration_M_20260923", "calibration_accept_20260923", "calibration_refit_prelim_20260923",
    "calibration_refit_final_20260923", "calibration_resplit_20260924",
    "calibration_freeze_20260923",
))

#: The preregistration keys :func:`check_preregistration` checks (dotted paths).
PREREG_KEYS_REQUIRED = ("t14.capacity_prior.sha256", "t14.design_seed", "t14.cell_serial_base",
                        "t14.factors", "t14.hold_s", "t14.shapes")
PREREG_KEYS_OPTIONAL = ("t14.model", "parameter_sets.freeze.sha256",
                        "parameter_sets.v1lambda.sha256")
PREREG_KEYS = (*PREREG_KEYS_REQUIRED, *PREREG_KEYS_OPTIONAL)


def shape_kind(shape: str) -> str:
    if shape in INTERPOLATION_SHAPES:
        return KIND_INTERPOLATION
    if shape in EXTRAPOLATION_SHAPES:
        return KIND_EXTRAPOLATION
    raise ValueError(f"{shape!r} is not a T14 shape")


def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


# ------------------------------------------------------------------- capacity prior


def _solve3(a: list[list[float]], b: list[float]) -> list[float]:
    """Gaussian elimination with partial pivoting of a 3x3 system."""
    m = [list(map(float, row)) + [float(v)] for row, v in zip(a, b)]
    n = 3
    scale = max(abs(x) for row in m for x in row[:n]) or 1.0
    for col in range(n):
        piv = max(range(col, n), key=lambda r: abs(m[r][col]))
        if abs(m[piv][col]) <= 1e-12 * scale:
            raise ValueError("capacity prior: degenerate (input, output) points")
        m[col], m[piv] = m[piv], m[col]
        for r in range(col + 1, n):
            f = m[r][col] / m[col][col]
            for c in range(col, n + 1):
                m[r][c] -= f * m[col][c]
    x = [0.0] * n
    for r in range(n - 1, -1, -1):
        x[r] = (m[r][n] - sum(m[r][c] * x[c] for c in range(r + 1, n))) / m[r][r]
    return x


def fit_linear_capacity(points: Sequence[Mapping]) -> tuple[float, float, float]:
    """Ordinary least squares of ``1/rate_rps = c0 + c_in*input_mean + c_out*output_mean``
    (with intercept) via the 3x3 normal equations. Returns (c0, c_in, c_out)."""
    if len(points) < 3:
        raise ValueError(f"capacity prior: {len(points)} points cannot fit 3 coefficients")
    xs = [(1.0, float(p["input_mean"]), float(p["output_mean"])) for p in points]
    ys = [1.0 / float(p["rate_rps"]) for p in points]
    ata = [[sum(x[i] * x[j] for x in xs) for j in range(3)] for i in range(3)]
    aty = [sum(x[i] * y for x, y in zip(xs, ys)) for i in range(3)]
    c0, c_in, c_out = _solve3(ata, aty)
    return c0, c_in, c_out


def predicted_rate(shape: str, c0: float, c_in: float, c_out: float) -> float:
    """C^_s: the shape's predicted D6' boundary rate (streams at their mean lengths,
    combined by load share); refuses a non-positive cost."""
    cost = sum(w * (c0 + c_in * gen._length_mean(i) + c_out * gen._length_mean(o))
               for w, i, o in gen.shape_components(shape))
    if not cost > 0.0:
        raise ValueError(f"capacity prior: non-positive cost per request for {shape} ({cost})")
    return 1.0 / cost


def build_capacity_prior(model: str, base_root: Path, supp_root: Path, table: Path) -> dict:
    """The capacity-prior document (see the module docstring)."""
    points = acceptance.boundary_rates(model, Path(base_root), Path(supp_root), Path(table))
    c0, c_in, c_out = fit_linear_capacity(points)
    loo = []
    for k, p in enumerate(points):
        kc = fit_linear_capacity(points[:k] + points[k + 1:])
        pred = 1.0 / (kc[0] + kc[1] * p["input_mean"] + kc[2] * p["output_mean"])
        loo.append({"shape": p["shape"], "rate_rps": p["rate_rps"], "loo_rps": round(pred, 4),
                    "loo_error": round(pred / p["rate_rps"] - 1.0, 4)})
    fitted = [{"shape": p["shape"], "fitted_rps": round(
        1.0 / (c0 + c_in * p["input_mean"] + c_out * p["output_mean"]), 4)} for p in points]
    return {
        "what": ("T14 capacity prior (C^_s per held-out 14b shape): pre-registered before T14 is "
                 "collected, fitted on the training shapes' D6' boundary rates only; T14 drives "
                 "each shape at {0.9, 1.0, 1.1} x C^_s without probing"),
        "model": model,
        "form": PRIOR_FORM,
        "coefficients": {"c0_s": c0, "c_in_s_per_token": c_in, "c_out_s_per_token": c_out},
        "fit_points": points,
        "fitted": fitted,
        "leave_one_out": loo,
        "predicted_rps": {s: predicted_rate(s, c0, c_in, c_out) for s in SHAPES},
        "t14_shapes": {s: {"input": gen.shape_components(s)[0][1],
                           "output": gen.shape_components(s)[0][2], "kind": shape_kind(s)}
                       for s in SHAPES},
        "inputs": {"boundary_table": {"path": str(Path(table).resolve()),
                                      "sha256": _file_sha256(Path(table))},
                   "base_run": str(Path(base_root).resolve()),
                   "boundary_supplement_run": str(Path(supp_root).resolve()),
                   "prior_shapes": list(PRIOR_SHAPES),
                   "rates_from": "scripts.calibration_acceptance.boundary_rates"},
        "created_at_utc": campaign.utc_iso(),
        "code": campaign.git_state(Path(__file__).resolve().parents[2]),
    }


def write_capacity_prior(doc: Mapping, out: Path) -> str:
    """Write once (refuse to overwrite), read-only; returns the file's sha256."""
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    data = (json.dumps(doc, indent=2) + "\n").encode("utf-8")
    try:
        with open(out, "xb") as fh:
            fh.write(data)
    except FileExistsError:
        raise ValueError(f"{out} exists; a capacity prior is written once - pick a new path")
    os.chmod(out, stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
    return hashlib.sha256(data).hexdigest()


def _close(a: float, b: float) -> bool:
    return math.isclose(float(a), float(b), rel_tol=1e-6, abs_tol=1e-12)


def load_capacity_prior(path: Path, model: str) -> dict:
    """The capacity prior, checked (ValueError on any problem); adds ``path`` / ``sha256``."""
    path = Path(path)
    if not path.is_file():
        raise ValueError(f"no capacity prior at {path}")
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except ValueError as exc:
        raise ValueError(f"{path}: not JSON ({exc})")
    if not isinstance(doc, dict):
        raise ValueError(f"{path}: not a capacity prior")
    if doc.get("model") != model:
        raise ValueError(f"{path}: capacity prior of {doc.get('model')!r}, not {model!r}")
    points = doc.get("fit_points")
    if not isinstance(points, list) or len(points) < MIN_FIT_POINTS:
        n = len(points) if isinstance(points, list) else 0
        raise ValueError(f"{path}: {n} fit points, need >= {MIN_FIT_POINTS}")
    for p in points:
        shape = p.get("shape") if isinstance(p, Mapping) else None
        if shape not in gen.TRAINING_SHAPES or gen.is_held_out(shape):
            raise ValueError(f"{path}: fit point {shape!r} is not a training shape - the prior "
                             "may be fitted on training rho* only")
        try:
            ok = float(p["rate_rps"]) > 0 and p["input_mean"] is not None and p["output_mean"] is not None
        except (KeyError, TypeError, ValueError):
            ok = False
        if not ok:
            raise ValueError(f"{path}: fit point {shape} lacks input_mean / output_mean / a "
                             "positive rate_rps")
    coef = doc.get("coefficients") or {}
    try:
        c0, c_in, c_out = (float(coef[k]) for k in ("c0_s", "c_in_s_per_token", "c_out_s_per_token"))
    except (KeyError, TypeError, ValueError):
        raise ValueError(f"{path}: coefficients c0_s / c_in_s_per_token / c_out_s_per_token missing")
    refit = fit_linear_capacity(points)
    if not all(_close(a, b) for a, b in zip((c0, c_in, c_out), refit)):
        raise ValueError(f"{path}: coefficients {(c0, c_in, c_out)} are not the least-squares fit "
                         f"of its fit_points {refit}")
    predicted = doc.get("predicted_rps")
    if not isinstance(predicted, dict):
        raise ValueError(f"{path}: no predicted_rps")
    missing = [s for s in SHAPES if s not in predicted]
    if missing:
        raise ValueError(f"{path}: predicted_rps misses T14 shapes {missing}")
    for shape, value in predicted.items():
        try:
            gen.shape_components(shape)
        except KeyError:
            raise ValueError(f"{path}: predicted_rps names an unknown shape {shape!r}")
        try:
            value = float(value)
        except (TypeError, ValueError):
            raise ValueError(f"{path}: predicted rate of {shape} is not a number")
        if not value > 0:
            raise ValueError(f"{path}: predicted rate of {shape} is {value}, not positive")
        want = predicted_rate(shape, c0, c_in, c_out)
        if not math.isclose(value, want, rel_tol=1e-6):
            raise ValueError(f"{path}: predicted_rps[{shape}] = {value} != {want} recomputed from "
                             "the coefficients")
    return {**doc, "path": str(path.resolve()), "sha256": _file_sha256(path)}


# ------------------------------------------------------------------------ the cells


def new_cells(model: str, design_seed: int) -> list[design.DesignCell]:
    """The 24 cells, shape-major then factor; rho = factor (rho of C^_s)."""
    if model != MODEL:
        raise ValueError(f"T14 is a {MODEL} set; got {model!r}")
    factory = design.CellFactory(model, design_seed, serial_base=CELL_SERIAL_BASE)
    cells = []
    for shape in SHAPES:
        for factor in FACTORS:
            cells.append(factory.new(
                shape, design.ROLE_ACCEPTANCE, HOLD_S, rho_factor=factor, rho=factor,
                warmup_s=design.WARMUP_S,
                note=f"T14: {shape} ({shape_kind(shape)}) hold {factor:g} x C^_s, {HOLD_S:g} s"))
    return cells


def interleaved_order(cells: Sequence[design.DesignCell], model: str,
                      design_seed: int) -> list[design.DesignCell]:
    rng = random.Random(design.derived_seed(design_seed, model, "t14-order"))
    return training.interleave(cells, rng)


def check_held_out(cells: Sequence[design.DesignCell]) -> None:
    bad = [f"{c.cell_id} ({c.shape}, split {c.split})" for c in cells
           if not gen.is_held_out(c.shape)
           or design.split_for(c.role, c.shape) != design.SPLIT_HOLDOUT
           or c.split != design.SPLIT_HOLDOUT]
    if bad:
        raise ValueError(f"T14 cells not held out: {bad}")


# ------------------------------------------------------------------------- checks


def registry_max_model_len(model: str, registry: Optional[str] = None) -> Optional[int]:
    """``--max-model-len`` of ``model`` in the registry's ``vllm_extra_args`` (None if not
    pinned)."""
    from tre_common import registry as tre_registry

    args = list(tre_registry.load_registry(registry).model(model).vllm_extra_args)
    for k, arg in enumerate(args):
        if arg == "--max-model-len" and k + 1 < len(args):
            return int(args[k + 1])
        if arg.startswith("--max-model-len="):
            return int(arg.split("=", 1)[1])
    return None


def _longest(length) -> int:
    return int(length.high) if hasattr(length, "high") else int(length)


def check_max_model_len(model: str, shapes: Sequence[str], registry: Optional[str] = None,
                        *, max_model_len: Optional[int] = None) -> dict:
    """Every request of the composition fits: max(input + output) + margin <= max-model-len."""
    if max_model_len is None:
        max_model_len = registry_max_model_len(model, registry)
    longest, worst = 0, None
    for shape in shapes:
        for _w, i, o in gen.shape_components(shape):
            if _longest(i) + _longest(o) > longest:
                longest, worst = _longest(i) + _longest(o), shape
    need = longest + MAX_MODEL_LEN_MARGIN
    if max_model_len is None:
        raise ValueError(f"{model}: the registry does not pin --max-model-len; T14 needs >= {need}")
    if int(max_model_len) < need:
        raise ValueError(f"{model}: --max-model-len {max_model_len} < {need} ({worst}: {longest} "
                         f"tokens + {MAX_MODEL_LEN_MARGIN})")
    return {"max_model_len": int(max_model_len), "longest_request_tokens": longest,
            "longest_shape": worst, "margin": MAX_MODEL_LEN_MARGIN, "needed": need, "ok": True}


def check_param_file(path, model: str, registry: Optional[str], what: str) -> dict:
    """A ``dline_refit freeze`` file: verifies, and froze ``model`` under the primary label
    (the check ``calibration_acceptance.check_freeze`` makes)."""
    from scripts import dline_refit

    if not path or not Path(path).is_file():
        raise ValueError(f"{what}: no parameter file at {path}")
    try:
        doc = dline_refit.verify_freeze(Path(path))
    except dline_refit.FreezeError as exc:
        raise ValueError(f"{what}: {path} does not verify ({exc})")
    ours = dline_refit.label_for(model, "primary", registry).as_dict()
    theirs = acceptance.frozen_label_def(doc, model)
    if dline_refit.canonical_sha256(ours) != dline_refit.canonical_sha256(theirs):
        raise ValueError(f"{what}: {path}: {model} was frozen under another label definition "
                         "than the one T14 is judged by")
    return {"path": str(Path(path).resolve()), "sha256": _file_sha256(Path(path)),
            "freeze_sha256": doc.get("freeze_sha256"),
            "label_def_sha256": dline_refit.canonical_sha256(theirs)}


def _dig(doc: Mapping, dotted: str):
    cur = doc
    for part in dotted.split("."):
        if not isinstance(cur, Mapping) or part not in cur:
            raise KeyError(dotted)
        cur = cur[part]
    return cur


def _has(doc: Mapping, dotted: str) -> bool:
    try:
        _dig(doc, dotted)
        return True
    except KeyError:
        return False


def _is_int(value, want: int) -> bool:
    return (not isinstance(value, bool) and isinstance(value, (int, float))
            and float(value) == float(want))


def _floats_equal(value, want: Sequence[float]) -> bool:
    return (isinstance(value, list) and len(value) == len(want)
            and all(not isinstance(a, bool) and isinstance(a, (int, float))
                    and math.isclose(float(a), float(b), rel_tol=0.0, abs_tol=1e-12)
                    for a, b in zip(value, want)))


def _shapes_match(value) -> bool:
    if isinstance(value, Mapping):
        return (set(value) == {KIND_INTERPOLATION, KIND_EXTRAPOLATION}
                and list(value[KIND_INTERPOLATION]) == list(INTERPOLATION_SHAPES)
                and list(value[KIND_EXTRAPOLATION]) == list(EXTRAPOLATION_SHAPES))
    return isinstance(value, list) and value == list(SHAPES)


#: The keys an amendment may override (``overrides``, dotted): only what binds the second
#: parameter set. The design (``t14.*``: shapes, factors, seeds, capacity prior) is never
#: amendable - a different design is a new preregistration, not an amendment.
AMENDABLE_KEYS = ("parameter_sets.v1lambda.sha256", "parameter_sets.v1lambda.path",
                  "parameter_sets.v1lambda.freeze_sha256")


def _read_sidecar_checked(path: Path, what: str) -> tuple[dict, str]:
    """(JSON document, sha256) of ``path`` after checking its sha256sum sidecar."""
    side = Path(f"{path}.sha256")
    if not path.is_file():
        raise ValueError(f"no {what} at {path}")
    if not side.is_file():
        raise ValueError(f"{side} is missing: the {what} has no sha256 sidecar")
    parts = side.read_text(encoding="utf-8").split()
    if len(parts) != 2 or len(parts[0]) != 64:
        raise ValueError(f"{side}: not a sha256sum line ('<hex>  <name>')")
    want, name = parts[0].lower(), parts[1].lstrip("*")
    got = _file_sha256(path)
    if Path(name).name != path.name:
        raise ValueError(f"{side} names {name!r}, not {path.name!r}")
    if got != want:
        raise ValueError(f"{path}: sha256 {got} != {want} in {side.name}: the {what} changed "
                         "after its sidecar was written")
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except ValueError as exc:
        raise ValueError(f"{path}: not JSON ({exc})")
    if not isinstance(doc, dict):
        raise ValueError(f"{path}: not a {what} document")
    return doc, got


def check_amendment(path, *, prereg_path, prereg_sha256: str) -> tuple[dict, dict]:
    """(amendment summary, overrides) of an amendment to the preregistration: its own
    sha256 sidecar must match, ``amends.sha256`` must be the preregistration's sha256 (and
    ``amends.file`` its file name), and every ``overrides`` key must be in
    :data:`AMENDABLE_KEYS`. The frozen preregistration itself is never rewritten."""
    path = Path(path)
    doc, got = _read_sidecar_checked(path, "preregistration amendment")
    amends = doc.get("amends") or {}
    if str(amends.get("sha256", "")).lower() != prereg_sha256:
        raise ValueError(f"{path}: amends sha256 {amends.get('sha256')!r}, the preregistration "
                         f"has {prereg_sha256}")
    if amends.get("file") and Path(str(amends["file"])).name != Path(prereg_path).name:
        raise ValueError(f"{path}: amends {amends.get('file')!r}, not {Path(prereg_path).name!r}")
    overrides = doc.get("overrides") or {}
    if not isinstance(overrides, Mapping):
        raise ValueError(f"{path}: overrides is not a mapping")
    bad = sorted(k for k in overrides if k not in AMENDABLE_KEYS)
    if bad:
        raise ValueError(f"{path}: overrides {bad} are not amendable (only {list(AMENDABLE_KEYS)})")
    return ({"path": str(path.resolve()), "sha256": got, "overrides": dict(overrides)},
            dict(overrides))


def _set(doc: dict, dotted: str, value) -> None:
    cur = doc
    parts = dotted.split(".")
    for part in parts[:-1]:
        cur = cur.setdefault(part, {})
    cur[parts[-1]] = value


def check_preregistration(path, *, capacity_sha256: str, design_seed: int,
                          freeze: Optional[Mapping], refit: Optional[Mapping],
                          model: str = MODEL, amendment=None) -> dict:
    """The preregistration JSON, bound to this run (ValueError on any mismatch).

    Its sidecar ``<path>.sha256`` (sha256sum format, naming the file) must match. Then the
    required keys (:data:`PREREG_KEYS_REQUIRED`) must equal this run's values:
    ``t14.capacity_prior.sha256`` = sha256 of ``--capacity-prior-file``; ``t14.design_seed``
    = ``--design-seed``; ``t14.cell_serial_base`` = :data:`CELL_SERIAL_BASE`; ``t14.factors``
    = :data:`FACTORS` (a list); ``t14.hold_s`` = :data:`HOLD_S`; ``t14.shapes`` = the eight
    names, either the list in :data:`SHAPES` order or ``{"interpolation": [...],
    "extrapolation": [...]}``. The optional keys, when present: ``t14.model`` = the model;
    ``parameter_sets.freeze.sha256`` / ``parameter_sets.v1lambda.sha256`` = the sha256 of
    ``--freeze-file`` / ``--refit-params-file`` (present but its file not given - only
    possible in a dry run - is reported as unchecked)."""
    path = Path(path)
    side = Path(f"{path}.sha256")
    if not path.is_file():
        raise ValueError(f"no preregistration at {path}")
    if not side.is_file():
        raise ValueError(f"{side} is missing: the preregistration has no sha256 sidecar")
    parts = side.read_text(encoding="utf-8").split()
    if len(parts) != 2 or len(parts[0]) != 64:
        raise ValueError(f"{side}: not a sha256sum line ('<hex>  <name>')")
    want, name = parts[0].lower(), parts[1].lstrip("*")
    got = _file_sha256(path)
    if Path(name).name != path.name:
        raise ValueError(f"{side} names {name!r}, not {path.name!r}")
    if got != want:
        raise ValueError(f"{path}: sha256 {got} != {want} in {side.name}: the preregistration "
                         "changed after its sidecar was written")
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except ValueError as exc:
        raise ValueError(f"{path}: not JSON ({exc})")
    if not isinstance(doc, dict):
        raise ValueError(f"{path}: not a preregistration document")
    amended = None
    if amendment is not None:
        amended, overrides = check_amendment(amendment, prereg_path=path, prereg_sha256=got)
        doc = json.loads(json.dumps(doc))
        for key, value in overrides.items():
            _set(doc, key, value)
    expected = {
        "t14.capacity_prior.sha256": (lambda v: isinstance(v, str) and v.lower() == capacity_sha256,
                                      capacity_sha256),
        "t14.design_seed": (lambda v: _is_int(v, int(design_seed)), int(design_seed)),
        "t14.cell_serial_base": (lambda v: _is_int(v, CELL_SERIAL_BASE), CELL_SERIAL_BASE),
        "t14.factors": (lambda v: _floats_equal(v, FACTORS), list(FACTORS)),
        "t14.hold_s": (lambda v: _is_int(v, HOLD_S), HOLD_S),
        "t14.shapes": (_shapes_match, list(SHAPES)),
    }
    problems, unchecked, checked = [], [], []
    for key, (ok, want_value) in expected.items():
        if not _has(doc, key):
            problems.append(f"{key} missing")
            continue
        value = _dig(doc, key)
        checked.append(key)
        if not ok(value):
            problems.append(f"{key} = {value!r}, this run has {want_value!r}")
    if _has(doc, "t14.model"):
        checked.append("t14.model")
        if _dig(doc, "t14.model") != model:
            problems.append(f"t14.model = {_dig(doc, 't14.model')!r}, this run is {model!r}")
    for key, given, flag in (("parameter_sets.freeze.sha256", freeze, "--freeze-file"),
                             ("parameter_sets.v1lambda.sha256", refit, "--refit-params-file")):
        if not _has(doc, key):
            continue
        value = _dig(doc, key)
        if given is None:
            unchecked.append(f"{key} (no {flag})")
            continue
        checked.append(key)
        if not isinstance(value, str) or value.lower() != given["sha256"]:
            problems.append(f"{key} = {value!r}, {flag} has sha256 {given['sha256']}")
    if problems:
        raise ValueError(f"{path}: the preregistration does not bind this run: " + "; ".join(problems))
    return {"path": str(path.resolve()), "sha256": got, "checked_keys": checked,
            "unchecked": unchecked, "amendment": amended}


def _overlaps(path: Path, root: Path) -> bool:
    return path == root or root in path.parents or path in root.parents


def check_output_roots(out_dir: Path, raw_dir: Path,
                       roots: Optional[Sequence[Path]] = None) -> None:
    """``--out-dir`` absent or empty; neither it nor ``--raw-dir`` inside (or around) a
    training / M / resplit / freeze root (:data:`FORBIDDEN_ROOTS`)."""
    roots = FORBIDDEN_ROOTS if roots is None else roots
    out, raw = Path(out_dir).resolve(), Path(raw_dir).resolve()
    for root in roots:
        root = Path(root).resolve()
        if _overlaps(out, root):
            raise ValueError(f"--out-dir {out} overlaps {root}: T14 gets an independent root")
        if _overlaps(raw, root):
            raise ValueError(f"--raw-dir {raw} overlaps {root}: T14 gets an independent root")
    if out.exists() and (not out.is_dir() or any(out.iterdir())):
        raise ValueError(f"--out-dir {out} exists and is not empty - T14 is collected into a new "
                         "root (an interrupted run is not resumed)")


def check_fresh_raw(raw_dir: Path, cells: Sequence[design.DesignCell]) -> None:
    """No raw directory of a T14 cell (either attempt) exists yet under ``raw_dir``."""
    taken = [str(Path(raw_dir) / c.stem(a)) for c in cells for a in (1, 2)
             if (Path(raw_dir) / c.stem(a)).exists()]
    if taken:
        raise ValueError(f"raw directories of T14 cells already exist under {raw_dir}: {taken[:3]}"
                         f"{' ...' if len(taken) > 3 else ''} - use a new --raw-dir")


# -------------------------------------------------------------------------- plan


def estimate(order: Sequence[design.DesignCell],
             cooldown_s: float = campaign.DEFAULT_COOLDOWN_S) -> dict:
    """Wall clock: the load, a gap before every cell (expected: the cooldown floor, upper:
    the drain limit; plus the driver's start-up), and a tail after the cells above C^_s
    (expected: factor > 1, upper: every cell) while their backlog finishes."""
    gap_exp = max(cooldown_s, 0.0) + ladder.DRIVER_OVERHEAD_S
    gap_up = max(cooldown_s, design.DRAIN_LIMIT_S) + ladder.DRIVER_OVERHEAD_S
    load = sum(c.duration_s for c in order)
    tails_exp = sum(design.request_timeout_s(c.shape) for c in order if (c.rho_factor or 0) > 1.0)
    tails_up = sum(design.request_timeout_s(c.shape) for c in order)
    return {"cells": len(order), "offered_load_s": load,
            "gap_expected_s": gap_exp, "gap_upper_s": gap_up,
            "tails_expected_s": tails_exp, "tails_upper_s": tails_up,
            "seconds_expected": round(load + tails_exp + len(order) * gap_exp, 1),
            "seconds_upper": round(load + tails_up + len(order) * gap_up, 1)}


def preview(order: Sequence[design.DesignCell], prior: Mapping) -> list[dict]:
    out = []
    for c in order:
        cs = float(prior["predicted_rps"][c.shape])
        out.append({**c.as_dict(), "kind": shape_kind(c.shape), "capacity_rps": cs,
                    "offered_rps": float(c.rho_factor) * cs})
    return out


def _prior_summary(prior: Mapping) -> dict:
    keep = ("what", "model", "form", "coefficients", "fit_points", "leave_one_out",
            "predicted_rps", "inputs", "path", "sha256")
    return {k: prior.get(k) for k in keep}


def build_plan(args, model: str, cells: Sequence[design.DesignCell],
               order: Sequence[design.DesignCell], prior: Mapping, labels: Mapping,
               freeze: Optional[Mapping], refit: Optional[Mapping], prereg: Optional[Mapping],
               mml: Mapping, cap) -> tuple[dict, dict]:
    """(plan.json, run manifest)."""
    composition = {
        "cells": len(cells),
        "shapes": {KIND_INTERPOLATION: list(INTERPOLATION_SHAPES),
                   KIND_EXTRAPOLATION: list(EXTRAPOLATION_SHAPES)},
        "factors": list(FACTORS), "hold_s": HOLD_S, "warmup_s": design.WARMUP_S,
        "unit": "C^_s from the capacity prior (rps); rho = factor, offered rps = factor x C^_s",
        "probes": "none - C^_s is a pre-registered prediction, not a measured boundary",
    }
    common = {
        "design": ladder.DESIGN_NAME,
        "mode": MODE,
        "models": [model],
        "design_seed": int(args.design_seed),
        "cell_serial_base": CELL_SERIAL_BASE,
        "composition": composition,
        "capacity_prior": _prior_summary(prior),
        "capacity_source": CAPACITY_SOURCE,
        "max_model_len_check": dict(mml),
        "parameter_sets": {"freeze": freeze, "v1lambda": refit},
        "preregistration": prereg,
        "routing_strategy": getattr(args, "routing_strategy", None),
        "gateway_url": getattr(args, "gateway_url", None),
        "label": {**labels, "window_ms": args.window_ms, "step_ms": args.fit_step_ms},
        "order": [c.cell_id for c in order],
        "provenance": campaign.run_provenance(args),
        "admission_cap": cap.as_dict(),
    }
    plan = {
        **common,
        "generated_at_utc": campaign.utc_iso(),
        "cooldown_s": args.cooldown_s,
        "run_manifest": ladder.RUN_MANIFEST,
        # the cells in composition order; ``order`` / ``cells_preview`` are the run order
        "static_cells": {model: [c.as_dict() for c in cells]},
        "cells_preview": preview(order, prior),
        "estimate": estimate(order, args.cooldown_s),
    }
    manifest = {
        **common,
        "written_at_utc": campaign.utc_iso(),
        "why": ("T14 (2026-09-24): the held-out 14b test set - 8 new shapes x {0.9, 1.0, 1.1} x "
                "C^_s of a pre-registered linear capacity prior, 240 s holds; evaluation only"),
        "seed_derivation": ("as the ladder design (calibration_design.CellFactory): arrival seed "
                            "derived_seed(design_seed, model, cell_id, 'arrivals'), prompt key "
                            "p<design_seed>.<cell_id>; order from derived_seed(design_seed, "
                            "model, 't14-order'), shapes interleaved"),
        "static_plan": {model: [c.as_dict() for c in cells]},
        "cooldown_floor_s": args.cooldown_s,
    }
    return plan, manifest


def print_plan(plan: Mapping, model: str) -> None:
    prior = plan["capacity_prior"]
    co = prior["coefficients"]
    print(f"{model}: T14 capacity prior {prior['form']}: c0={co['c0_s']:.6g} s, "
          f"c_in={co['c_in_s_per_token']:.6g} s/token, c_out={co['c_out_s_per_token']:.6g} s/token")
    print(f"  prior file {prior['path']} sha256 {prior['sha256']}")
    print("  leave-one-out: " + ", ".join(f"{x['shape']} {x['loo_error']:+.0%}"
                                          for x in prior["leave_one_out"]))
    for shape in SHAPES:
        print(f"  {shape:10} {shape_kind(shape):13} C^_s = {prior['predicted_rps'][shape]:.3f} rps")
    m = plan["max_model_len_check"]
    print(f"  max-model-len check: {m['max_model_len']} >= {m['needed']} ({m['longest_shape']}: "
          f"{m['longest_request_tokens']} tokens + {m['margin']}) - ok")
    print(f"  {len(plan['order'])} cells in run order (all held out, split holdout, "
          f"{HOLD_S:g} s, first {design.WARMUP_S:g} s warm-up):")
    for i, c in enumerate(plan["cells_preview"], start=1):
        print(f"    {i:2d} {c['cell_id']:22} {c['shape']:10} {c['kind']:13} {c['rho_factor']:g} x "
              f"C^_s = {c['offered_rps']:.3f} rps  {c['duration_s']:.0f}s")
    est = plan["estimate"]
    print(f"  wall clock ~{est['seconds_expected'] / 3600:.2f} h expected "
          f"({est['seconds_expected'] / 60:.0f} min), <= {est['seconds_upper'] / 3600:.2f} h "
          f"({est['offered_load_s'] / 60:.0f} min of load, gap {est['gap_expected_s']:.0f}-"
          f"{est['gap_upper_s']:.0f} s per cell; re-drives of void cells excluded)")
    for key, flag in (("freeze", "--freeze-file"), ("v1lambda", "--refit-params-file")):
        p = plan["parameter_sets"][key]
        print(f"  parameter set {key}: " + (f"{p['path']} sha256 {p['sha256']}" if p
                                             else f"NOT CHECKED (no {flag}; dry run)"))
    pr = plan.get("preregistration")
    if pr:
        extra = f" (unchecked: {pr['unchecked']})" if pr.get("unchecked") else ""
        print(f"  preregistration: {pr['path']} sha256 {pr['sha256']}; checked "
              f"{pr['checked_keys']}{extra}")
        am = pr.get("amendment")
        if am:
            print(f"  preregistration amendment: {am['path']} sha256 {am['sha256']}; overrides "
                  f"{sorted(am['overrides'])}")
    else:
        print("  preregistration: NOT CHECKED (no --preregistration-json; dry run)")
    print(f"  routing strategy: {plan['routing_strategy']} via {plan['gateway_url']}")


# --------------------------------------------------------------------------- sealing


def sealed_files(out_dir: Path, raw_dir: Path, records: Sequence[Mapping]) -> list[Path]:
    """Every file of the out-dir (``dataset/`` and the seal itself excluded) and every
    attempt's raw capture."""
    out_dir = Path(out_dir).resolve()
    skip = {T14_MANIFEST, T14_SHA256SUMS}
    files = {p.resolve() for p in out_dir.rglob("*")
             if p.is_file() and p.name not in skip
             and "dataset" not in p.relative_to(out_dir).parts[:1]
             and not p.name.startswith(".")}
    for r in records:
        raw = Path(raw_dir) / str(r["stem"])
        if raw.is_dir():
            files |= {p.resolve() for p in raw.rglob("*") if p.is_file()}
    return sorted(files)


def write_sha256sums(out_dir: Path, files: Sequence[Path]) -> tuple[Path, str]:
    body = "".join(f"{_file_sha256(f)}  {Path(f).resolve()}\n" for f in files)
    path = Path(out_dir) / T14_SHA256SUMS
    path.write_text(body, encoding="utf-8")
    os.chmod(path, stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
    return path, hashlib.sha256(body.encode("utf-8")).hexdigest()


def cell_entry(record: Mapping, raw_dir: Path) -> dict:
    raw = Path(raw_dir) / str(record["stem"])
    return {"model": record["model"], "cell_id": record["cell_id"],
            "attempt": int(record["attempt"]), "shape": record["shape"],
            "kind": shape_kind(record["shape"]),
            "primitive": record["primitive"], "role": record["role"], "split": record["split"],
            "origin": "collected", "seen_before": False, "note": record.get("note", ""),
            "factor": record.get("rho_factor"), "rho": record.get("rho"),
            "capacity_rps": record.get("capacity_rps"), "offered_rps": record.get("offered_rps"),
            "duration_s": record.get("duration_s"), "warmup_s": record.get("warmup_s"),
            "online_csv": record.get("online_csv"), "schedule_path": record.get("schedule_path"),
            "possibly_contaminated": record.get("possibly_contaminated"),
            "raw_files": sorted(str(p.resolve()) for p in raw.rglob("*") if p.is_file())
            if raw.is_dir() else []}


def seal(out_dir: Path, raw_dir: Path, model: str, run: "T14Run", labels: Mapping,
         plan: Mapping) -> dict:
    """T14_SHA256SUMS and T14_manifest.json (both read-only) - before the dataset."""
    collected = [cell_entry(run.final_records[c.cell_id], raw_dir) for c in run.cells]
    files = sealed_files(out_dir, raw_dir, run.records)
    sums_path, sums_sha = write_sha256sums(out_dir, files)
    ps = plan["parameter_sets"]
    prior = plan["capacity_prior"]
    doc = {
        "what": ("T14, the held-out 14b test set (2026-09-24): evaluation only - nothing that fits "
                 "may read it; its split is holdout"),
        "model": model,
        "mode": MODE,
        "format_revision": MANIFEST_FORMAT_REVISION,
        "written_at_utc": campaign.utc_iso(),
        # "freeze" is the M-manifest key dline_refit.check_m_manifest reads: the D22 freeze.
        "freeze": {"path": ps["freeze"]["path"], "sha256": ps["freeze"]["sha256"]},
        "parameter_sets": {k: {"path": v["path"], "sha256": v["sha256"],
                               "freeze_sha256": v.get("freeze_sha256")} for k, v in ps.items()},
        "capacity_prior": {"path": prior["path"], "sha256": prior["sha256"],
                           "form": prior["form"], "coefficients": prior["coefficients"],
                           "predicted_rps": prior["predicted_rps"]},
        "preregistration": {"path": plan["preregistration"]["path"],
                            "sha256": plan["preregistration"]["sha256"],
                            "checked_keys": plan["preregistration"]["checked_keys"],
                            "amendment": plan["preregistration"].get("amendment")},
        "label_def": labels["label_def"],
        "label_def_sha256": labels["label_def_sha256"],
        "composition": plan["composition"],
        "design_seed": plan["design_seed"],
        "cell_serial_base": CELL_SERIAL_BASE,
        "routing_strategy": plan["routing_strategy"],
        "gateway_url": plan["gateway_url"],
        "max_model_len_check": plan["max_model_len_check"],
        "cells": collected,
        "sealed_probes": [],
        "attempts": [{"cell_id": r["cell_id"], "attempt": r["attempt"],
                      "void_reasons": r.get("void_reasons")} for r in run.records],
        "run_manifest_sha256": plan.get("run_manifest_sha256"),
        "sha256sums_file": T14_SHA256SUMS,
        "sha256sums_sha256": sums_sha,
        "how_to_evaluate": (
            "Verify T14_SHA256SUMS first (dline_refit.check_m_manifest accepts this file against "
            "the D22 freeze). Read this run's standard dataset (<out-dir>/dataset, built right "
            "after this seal; rebuild: python -m scripts.calibration_dataset <out-dir>): every "
            "row is split holdout; keep the cells listed here (their final attempts) and the "
            "windows after warmup_s, labelled with label_def. Evaluate both frozen parameter "
            "sets (parameter_sets.freeze = the D22 freeze, parameter_sets.v1lambda = the "
            "v1-lambda refit) on the same windows, per shape and per kind (interpolation / "
            "extrapolation). Never train on it, never pass it as --h2-dataset."),
    }
    path = Path(out_dir) / T14_MANIFEST
    path.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
    os.chmod(path, stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
    return {"manifest": str(path), "manifest_sha256": _file_sha256(path),
            "sha256sums": str(sums_path), "sha256sums_sha256": sums_sha, "files": len(files)}


# ------------------------------------------------------------------------------ run


class T14Run(training.PlannedRun):
    """The 24 holds on the ladder's per-cell machinery (gap / drain, ledger, void rule)."""

    def __init__(self, *a, cells: Sequence[design.DesignCell], **kw) -> None:
        super().__init__(*a, **kw)
        self.cells = list(cells)
        self.final_records: dict[str, dict] = {}

    def drive_set(self, order: Sequence[design.DesignCell]) -> None:
        for cell in order:
            self.drive_planned(cell)
            self.final_records[cell.cell_id] = self.records[-1]

    def result(self, status: str) -> dict:
        return {"model": self.model, "status": status, "cells": self.outcomes,
                "possibly_contaminated_cells": self.contaminated,
                "attempts": len(self.records)}


def _required(value, flag: str, what: str, dry: bool) -> bool:
    """True when ``value`` is given; a real run refuses without it, a dry run warns."""
    if value:
        return True
    if not dry:
        raise ValueError(f"--t14-set needs {flag} ({what}) for a real run")
    print(f"WARNING: no {flag} ({what}); the real run refuses to start without it")
    return False


def run_t14_set(args, *, drive: Optional[Callable] = None,
                sample_factory: Optional[Callable[[str], Callable]] = None,
                sleep: Callable[[float], None] = time.sleep,
                clock: Callable[[], float] = time.monotonic,
                check_controller: bool = True) -> int:
    """``--t14-set``: see the module docstring."""
    ladder.check_label_args(args)
    model = training.single_model(args)
    if model != MODEL:
        raise ValueError(f"--t14-set is a {MODEL} set; got --models {model}")
    training.check_primary_label(args, model)
    if getattr(args, "design_seed", None) is None:
        args.design_seed = DEFAULT_DESIGN_SEED
    seed = int(args.design_seed)
    dry = bool(args.dry_run)
    out_dir, raw_dir = Path(args.out_dir), Path(args.raw_dir)
    check_output_roots(out_dir, raw_dir)
    if not getattr(args, "capacity_prior_file", None):
        raise ValueError("--t14-set needs --capacity-prior-file")
    prior = load_capacity_prior(Path(args.capacity_prior_file), model)
    cells = new_cells(model, seed)
    order = interleaved_order(cells, model, seed)
    check_held_out(cells)
    check_fresh_raw(raw_dir, cells)
    registry = getattr(args, "registry", None)
    mml = check_max_model_len(model, SHAPES, registry)
    freeze = refit = prereg = None
    if _required(getattr(args, "freeze_file", None), "--freeze-file", "the D22 freeze", dry):
        freeze = check_param_file(args.freeze_file, model, registry, "--freeze-file")
    if _required(getattr(args, "refit_params_file", None), "--refit-params-file",
                 "the v1-lambda parameter set", dry):
        refit = check_param_file(args.refit_params_file, model, registry, "--refit-params-file")
    if _required(getattr(args, "preregistration_json", None), "--preregistration-json",
                 "the preregistration", dry):
        prereg = check_preregistration(Path(args.preregistration_json),
                                       capacity_sha256=prior["sha256"], design_seed=seed,
                                       freeze=freeze, refit=refit, model=model,
                                       amendment=getattr(args, "preregistration_amendment_json", None))
    elif getattr(args, "preregistration_amendment_json", None):
        raise ValueError("--preregistration-amendment-json needs --preregistration-json")
    routing = getattr(args, "routing_strategy", None)
    if routing != ROUTING_STRATEGY:
        if not dry:
            raise ValueError(f"--t14-set needs --routing-strategy {ROUTING_STRATEGY} (the tre-v2 "
                             f"ext_proc gateway path); got {routing!r}")
        print(f"WARNING: --routing-strategy is {routing!r}; the real run requires "
              f"{ROUTING_STRATEGY!r}")
    labels = acceptance.label_documents(args, model)
    cap = training.resolve_cap(args)
    plan, manifest = build_plan(args, model, cells, order, prior, labels, freeze, refit, prereg,
                                mml, cap)
    out_dir.mkdir(parents=True, exist_ok=True)
    print_plan(plan, model)
    if dry:
        plan["run_manifest_preview"] = manifest
        (out_dir / "plan.json").write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")
        print(f"dry run: wrote {out_dir / 'plan.json'}; nothing driven (the run manifest is only "
              "written when a run starts)")
        return 0

    if check_controller:
        mode = campaign.controller_mode(args.controller_namespace)
        if mode != campaign.REQUIRED_CONTROLLER_MODE:
            raise SystemExit(f"controller mode is {mode!r}, refusing to run (need "
                             f"{campaign.REQUIRED_CONTROLLER_MODE!r})")
        print(f"controller mode: {mode}")
    plan["run_manifest_sha256"] = ladder.write_frozen(out_dir / ladder.RUN_MANIFEST, manifest)
    (out_dir / "plan.json").write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")
    raw_dir.mkdir(parents=True, exist_ok=True)
    drive = drive or ladder.subprocess_drive(args, raw_dir)
    sample_factory = sample_factory or (lambda m: ladder.make_engine_sampler(m, args.model_namespace))
    run = T14Run(
        args, model, {s: float(prior["predicted_rps"][s]) for s in SHAPES},
        factory=design.CellFactory(model, seed, serial_base=CELL_SERIAL_BASE), cap=cap,
        out_dir=out_dir, raw_dir=raw_dir, drive=drive, sample=sample_factory(model),
        capacity_source=CAPACITY_SOURCE, sleep=sleep, clock=clock, cells=cells)
    status, code = "failed", 1
    result = None
    try:
        try:
            run.drive_set(order)
        except ladder.CampaignStopped as stop:
            print(f"STOPPED: {stop} - T14 is not sealed; re-run into a NEW out-dir", flush=True)
            status, code = "stopped", stop.code or 1
            result = run.result(f"stopped: {stop}")
            return code
        sealed = seal(out_dir, raw_dir, model, run, labels, plan)
        result = run.result("complete")
        result["sealed"] = sealed
        print(f"[{model}] T14 sealed: {sealed['manifest']} ({sealed['files']} files in "
              f"{T14_SHA256SUMS})", flush=True)
        status, code = "complete", 0
        return 0
    except KeyboardInterrupt:
        status = "interrupted"
        raise
    finally:
        if result is None:
            result = run.result(status)
        training.finish(out_dir, plan, result, status, code)


# ---------------------------------------------------------------------------- CLI


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)
    cp = sub.add_parser("capacity-prior", help="build the T14 capacity prior (written once)")
    cp.add_argument("--model", default=MODEL)
    cp.add_argument("--base-run", type=Path, required=True)
    cp.add_argument("--boundary-supplement-run", type=Path, required=True)
    cp.add_argument("--boundary-table", type=Path, required=True)
    cp.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    if args.model != MODEL:
        ap.error(f"T14 is a {MODEL} set")
    try:
        doc = build_capacity_prior(args.model, args.base_run, args.boundary_supplement_run,
                                   args.boundary_table)
        sha = write_capacity_prior(doc, args.out)
        load_capacity_prior(args.out, args.model)
    except ValueError as exc:
        ap.error(str(exc))
    co = doc["coefficients"]
    print(f"{args.out}  sha256 {sha}")
    print(f"  {PRIOR_FORM}: c0={co['c0_s']:.6g} c_in={co['c_in_s_per_token']:.6g} "
          f"c_out={co['c_out_s_per_token']:.6g}")
    print("  leave-one-out: " + ", ".join(f"{x['shape']} {x['loo_error']:+.1%}"
                                          for x in doc["leave_one_out"]))
    for s in SHAPES:
        print(f"  {s:10} {shape_kind(s):13} {doc['predicted_rps'][s]:.3f} rps")
    return 0


if __name__ == "__main__":
    sys.exit(main())
