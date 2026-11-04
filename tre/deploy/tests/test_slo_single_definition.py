"""Guard: the SLO is defined once (``tre_common.slo``) and the registry's ``slo_mode`` drives
every consumer (user rules 2026-10-29 / 2026-11-04).

1. No consumer builds a threshold itself: outside an allow-list (each entry says why it
   may touch an SLO parameter: it parses, plumbs or records one, never computes a
   threshold), no file under tre/ reads the SLO parameters of a ``models[].slo`` block,
   calls the threshold primitives or defines an SLO constant (``TTFT_SLO_MS = 500.0``). A new hit fails here: route it through
   ``tre_common.slo`` (or ``LabelDefinition.thresholds()``), or justify it below.
2. Data contract: every ``models[].slo`` field has a reader, every repo-registry model has
   the ``idle_multiple`` parameters, and the switch is written in both repo registry files.
3. One switch: changing ``slo_mode`` changes the thresholds of every consumer at once.
4. No SLO value in code (2026-11-04): outside an allow-list, no file under tre/ holds a numeric
   literal in an SLO context - compared with, or dividing / multiplying, a value named after
   TTFT / TPOT (``max(ttft / 500.0, tpot / 75.0)``), as the default of a TTFT / TPOT option or
   parameter, as a ``*ttft*`` / ``*tpot*`` keyword argument, or as a TTFT / TPOT module
   constant - and no file compares a TTFT / TPOT value with an SLO threshold itself (instead
   of ``tre_common.slo.judge_request`` / ``violated_array`` or the window label).
"""
from __future__ import annotations

import ast
import dataclasses
import re
from pathlib import Path

import pytest
import yaml

from tre_common import slo
from tre_common.registry import IdleMultipleSlo, SloSpec, load_registry

TRE = Path(__file__).resolve().parents[2]
REGISTRY = TRE / "deploy" / "registry.yaml"
PARAMS = TRE / "deploy" / "overlays" / "tre-v2" / "params.yaml"
MODELS = ("dsqwen-7b", "dsllama-8b", "dsqwen-14b")

#: Threshold ingredients (any read is a hit) and the fixed thresholds (a hit when read from
#: something named ``slo``), the retired per-model switch, the threshold primitives.
PARAMETERS = {"ttft_slowdown_k", "ttft_floor_ms", "ttft_idle_c_ms", "ttft_idle_b_ms_per_token",
              "tpot_idle_ms", "tpot_k", "ttft_k"}
FIXED = {"ttft_p95_ms", "tpot_p95_ms"}
RETIRED = {"ttft_slo_mode"}
PRIMITIVES = {"request_ttft_slo_ms", "tpot_idle_multiple_ms"}
#: A module constant that is an SLO threshold (``TTFT_SLO_MS = 500.0``, ``TTFT_SLO, TPOT_SLO = 500.0, 75.0``).
SLO_CONSTANT = re.compile(r"(TTFT|TPOT).*SLO|SLO.*(TTFT|TPOT)", re.IGNORECASE)

ALLOWED = {
    "common/tre_common/slo.py": "the implementation",
    "common/tre_common/slo_labels.py": "the window label: LabelDefinition.thresholds() builds through tre_common.slo",
    "common/tre_common/registry.py": "parses models[].slo",
    "baselines/tre_baselines/tools/preserve_nl_slo.py": "records the SloThresholds parameters in its output",
    "baselines/tre_baselines/tools/tokenscale_profile.py": "reads the label's TPOT threshold (LabelDefinition)",
    "baselines/tre_baselines/tools/velocity_ladder.py": "the TokenScale paper's own SLO (D52 disclosure column)",
    "deploy/scripts/calibration_supplement.py": "records the D6' label's parameters in the plan",
    "deploy/scripts/calibration_training_supplement.py": "records the D6' label's parameters in the plan",
    "deploy/scripts/dline_refit.py": "records the label's parameters in the freeze",
    "deploy/scripts/r3_grid.py": "fixed-mode values as the label CLI's --ttft/--tpot-slo-ms defaults",
    "deploy/scripts/rewindow_from_raw.py": "fixed-mode values as the label CLI's --ttft/--tpot-slo-ms defaults",
    "deploy/registry_smoke.py": "compares the registry with the v1 profiles (no verdict)",
    "deploy/scripts/calibration_design.py": "the D6' design record's fixed-mode values (10-03 calibration)",
    # Not on tre_common.slo (2026-11-04, listed in the change): the latency_p95 alternative
    # signal normalises by the fixed p95 limits; the replayer's V_sys (tre_replayer.scoring,
    # v1 parity: window p95 + e2e) is a diagnostic next to the scorer (tre_eval.metrics).
    "controller/tre_controller/signals/sources.py": "latency_p95 alt signal: fixed p95 limits (v1)",
    "replayer/tre_replayer/run_trace.py": "v1-parity V_sys diagnostic (fixed + e2e)",
    "deploy/scripts/analysis/score_request_trace.py": "v1-parity V_sys diagnostic (fixed + e2e)",
}


#: Rule 4. Numbers that are units or neutral values, never an SLO (ms <-> s, ratios).
UNIT_NUMBERS = {0, 1, 2, 10, 100, 1000, 1e6, 1e-3, 1e-6, 0.5, 60}
TTFT_TPOT = re.compile(r"ttft|tpot", re.IGNORECASE)
LIMIT = re.compile(r"slo|thr|limit", re.IGNORECASE)

#: Rule 4 allow-lists (file -> why). LITERALS: an SLO-context number that is not an SLO
#: definition. COMPARES: code that compares a TTFT / TPOT value with a threshold it got from
#: tre_common.slo / the label (window-level or a diagnostic), not a request verdict of its own.
LITERALS = {
    "controller/tre_controller/planning/safescale_evidence.py":
        "NO_REGISTRY_TTFT_MS / _TPOT_MS: the last resort of a SafeScale construction without a registry "
        "(tests / offline); the controller judges with the registry resolver",
    "baselines/tre_baselines/tools/tokenscale_profile.py": "make_stub_measure: the dry run's stub latency model, not an SLO",
    "common/tre_common/slo_labels.py":
        "ABLATION_TTFT_*: the D6 ablation arm (k 3 / floor 150 ms, plan 2026-09-21 6.9h), a label column of the "
        "10-03 design, not an SLO mode",
    "deploy/scripts/calibration_design.py": "TTFT_SLO_MS / TPOT_SLO_MS: the 10-03 D6' design record (frozen calibration)",
}
COMPARES = {
    "common/tre_common/slo.py": "the implementation",
    "controller/tre_controller/planning/safescale.py":
        "SafeScale probe: window p95 against the resolver's thresholds (tre_common.slo; window-level, unchanged 2026-11-04)",
    "baselines/tre_baselines/tools/tokenscale_profile.py": "TokenScale profile: a step's p95 against the label's thresholds",
    "deploy/scripts/calibration_campaign.py": "10-03 capacity prior: a window p95 against the run's fixed thresholds",
    "deploy/scripts/dline_refit.py": "D6' validation diagnostic: per-bucket TTFT attainment against LabelDefinition.ttft_slo_ms",
    "deploy/scripts/openloop.py": "10-03 open-loop ladder goodput (fixed thresholds passed in by the caller; kept to reproduce)",
    "replayer/tre_replayer/scoring.py": "v1-parity V_sys diagnostic (fixed + e2e)",
}


def _number(node: ast.AST) -> bool:
    return (isinstance(node, ast.Constant) and isinstance(node.value, (int, float))
            and not isinstance(node.value, bool) and abs(node.value) not in UNIT_NUMBERS
            and abs(node.value) >= 0.01)   # tolerances (1e-9) are not SLO values


def _named(node: ast.AST, rx: re.Pattern) -> bool:
    if isinstance(node, (ast.Constant, ast.Set, ast.List, ast.Tuple, ast.Dict)):
        return False
    return bool(rx.search(ast.unparse(node)))


def _literal_and_compare_hits(tree: ast.AST) -> tuple[set[str], set[str]]:
    literals: set[str] = set()
    compares: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and _number(node.value):
            for name in (n.id for t in node.targets for n in ast.walk(t) if isinstance(n, ast.Name)):
                if TTFT_TPOT.search(name):
                    literals.add(f"{name} = {node.value.value}")
        elif isinstance(node, ast.Compare):
            ops = [node.left, *node.comparators]
            if any(_named(o, TTFT_TPOT) for o in ops) and any(_number(o) for o in ops):
                literals.add(ast.unparse(node)[:80])
            elif (any(_named(o, TTFT_TPOT) and not _named(o, LIMIT) for o in ops)
                  and any(_named(o, LIMIT) for o in ops)):
                compares.add(ast.unparse(node)[:80])
        elif isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Div, ast.Mult)):
            if ((_named(node.left, TTFT_TPOT) and _number(node.right))
                    or (_named(node.right, TTFT_TPOT) and _number(node.left))):
                literals.add(ast.unparse(node)[:80])
        elif isinstance(node, ast.Call):
            func = node.func
            name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", "")
            if name == "add_argument":
                flags = " ".join(a.value for a in node.args if isinstance(a, ast.Constant) and isinstance(a.value, str))
                for kw in node.keywords:
                    if kw.arg == "default" and _number(kw.value) and TTFT_TPOT.search(flags):
                        literals.add(f"add_argument({flags}, default={kw.value.value})")
            for kw in node.keywords:
                if kw.arg and TTFT_TPOT.search(kw.arg) and _number(kw.value):
                    literals.add(f"{name}({kw.arg}={kw.value.value})")
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            a = node.args
            pos = a.posonlyargs + a.args
            pairs = list(zip(pos[len(pos) - len(a.defaults):], a.defaults))
            pairs += [(x, d) for x, d in zip(a.kwonlyargs, a.kw_defaults) if d is not None]
            for arg, default in pairs:
                if TTFT_TPOT.search(arg.arg) and _number(default):
                    literals.add(f"def {node.name}({arg.arg}={default.value})")
    return literals, compares


def _scan_rule4() -> tuple[dict[str, set[str]], dict[str, set[str]]]:
    lit, cmp_ = {}, {}
    for path in sorted(TRE.rglob("*.py")):
        rel = path.relative_to(TRE).as_posix()
        if "/tests/" in rel or rel.startswith("docs/") or "/." in rel:
            continue
        a, b = _literal_and_compare_hits(ast.parse(path.read_text(encoding="utf-8")))
        if a:
            lit[rel] = a
        if b:
            cmp_[rel] = b
    return lit, cmp_


def test_no_slo_value_in_code_and_no_verdict_of_its_own() -> None:
    lit, cmp_ = _scan_rule4()
    new_lit = {f: sorted(h) for f, h in lit.items() if f not in LITERALS}
    assert not new_lit, f"SLO values in code (read them from the registry through tre_common.slo): {new_lit}"
    new_cmp = {f: sorted(h) for f, h in cmp_.items() if f not in COMPARES}
    assert not new_cmp, (f"TTFT / TPOT compared with a threshold outside tre_common.slo (use judge_request / "
                         f"violated_array, or justify in COMPARES): {new_cmp}")
    stale = sorted((set(LITERALS) - set(lit)) | (set(COMPARES) - set(cmp_)))
    assert not stale, f"rule-4 allow-list entries that no longer match (remove them): {stale}"


@pytest.mark.parametrize("source, kind", [
    ("x = max(ttft / 500.0, tpot / 75.0)", "literal"),           # calibration_priors before 2026-11-04
    ("v = tt.fillna(0.0) > TTFT_SLO_MS", None),                   # a constant: rule 1 (SLO_CONSTANT)
    ("v = np.nan_to_num(tpot) > TPOT_SLO_MS", "compare"),         # supp_lib before 2026-11-04
    ("ap.add_argument('--ttft-slo-ms', type=float, default=500.0)", "literal"),
    ("f(model, ttft_p95_ms=500.0, tpot_p95_ms=75.0)", "literal"),  # dline_refit before 2026-11-04
    ("def g(rows, ttft_slo_ms=500.0):\n    pass", "literal"),
    ("NO_REGISTRY_TTFT_MS = 500.0", "literal"),
    ("ms = ttft_s * 1000.0", None),                               # a unit
    ("ok = ttft_ms is not None", None),
])
def test_rule4_catches_the_inline_forms(source, kind) -> None:
    lit, cmp_ = _literal_and_compare_hits(ast.parse(source))
    assert (("literal" if lit else None) or ("compare" if cmp_ else None)) == kind, (lit, cmp_)


def _key(node: ast.AST):
    return node.value if isinstance(node, ast.Constant) and isinstance(node.value, str) else None


def _named_slo(node: ast.AST) -> bool:
    return "slo" in ast.unparse(node).lower()


def _numbers(node: ast.AST) -> bool:
    if isinstance(node, ast.Tuple):
        return bool(node.elts) and all(_numbers(e) for e in node.elts)
    return isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool)


def _hits(tree: ast.AST) -> set[str]:
    out: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and _numbers(node.value):
            names = [n.id for t in node.targets for n in ast.walk(t) if isinstance(n, ast.Name)]
            out.update(f"{n} = <number>" for n in names if SLO_CONSTANT.search(n))
        if isinstance(node, ast.Attribute):
            if node.attr in PARAMETERS or (node.attr in FIXED and _named_slo(node.value)):
                out.add(node.attr)
        elif isinstance(node, ast.Subscript):
            key = _key(node.slice)
            if key in PARAMETERS | FIXED | RETIRED and _named_slo(node.value):
                out.add(key)
        elif isinstance(node, ast.Call):
            func = node.func
            name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", None)
            if name in PRIMITIVES:
                out.add(f"{name}()")
            if name == "get" and isinstance(func, ast.Attribute) and node.args:
                key = _key(node.args[0])
                if key in PARAMETERS | FIXED | RETIRED and _named_slo(func.value):
                    out.add(key)
    return out


def _scan() -> dict[str, set[str]]:
    found = {}
    for path in sorted(TRE.rglob("*.py")):
        rel = path.relative_to(TRE).as_posix()
        if "/tests/" in rel or rel.startswith("docs/") or "/." in rel:
            continue
        hits = _hits(ast.parse(path.read_text(encoding="utf-8")))
        if hits:
            found[rel] = hits
    return found


def test_no_consumer_computes_an_slo_threshold_on_its_own() -> None:
    found = _scan()
    new = {f: sorted(h) for f, h in found.items() if f not in ALLOWED}
    assert not new, f"SLO parameters read outside tre_common.slo (route through it or justify in ALLOWED): {new}"
    stale = sorted(set(ALLOWED) - set(found))
    assert not stale, f"ALLOWED entries that no longer touch an SLO parameter (remove them): {stale}"
    # the primitives are called by the implementation only
    assert {f for f, h in found.items() if any(x.endswith("()") for x in h)} <= {
        "common/tre_common/slo.py", "common/tre_common/slo_labels.py"}


def test_every_slo_field_has_a_reader_and_the_registry_writes_them() -> None:
    source = (TRE / "common" / "tre_common" / "slo.py").read_text(encoding="utf-8")
    for field in (f.name for f in dataclasses.fields(SloSpec)):
        if field == "e2e_p95_ms":  # diagnostic, no SLO mode judges it
            assert '"e2e_p95_ms"' in (TRE / "eval" / "tre_eval" / "metrics.py").read_text(encoding="utf-8")
            continue
        assert f'"{field}"' in source, f"models[].slo.{field} has no reader in tre_common.slo"
    assert {f.name for f in dataclasses.fields(IdleMultipleSlo)} == set(slo.IDLE_MULTIPLE_FIELDS)
    for field in slo.IDLE_MULTIPLE_FIELDS:  # the reader requires each one
        block = {"ttft_idle_c_ms": 40.0, "ttft_idle_b_ms_per_token": 0.05,
                 "idle_multiple": {k: 1.0 for k in slo.IDLE_MULTIPLE_FIELDS if k != field}}
        with pytest.raises(slo.SloConfigError, match=field):
            slo.slo_thresholds(block, "idle_multiple")
    # written: both repo registry files carry the switch (the live value) and complete blocks
    params = yaml.safe_load(yaml.safe_load(PARAMS.read_text(encoding="utf-8"))["data"]["registry.yaml"])
    for raw in (yaml.safe_load(REGISTRY.read_text(encoding="utf-8")), params):
        assert raw["slo_mode"] == "fixed"
        assert slo.resolve_slo_mode(raw).source == "slo_mode"
        for model in raw["models"]:
            assert set(model["slo"]["idle_multiple"]) == set(slo.IDLE_MULTIPLE_FIELDS), model["name"]
    # read: every consumer of "the definition in force" asks the registry's switch
    readers = {
        "controller/tre_controller/config.py": "slo_mode_resolution()",
        "eval/tre_eval/load.py": "resolve_slo_mode(",
        "eval/runner/score_pilot.py": "resolve_slo_mode(",
        "common/tre_common/slo_labels.py": '"slo_mode"',
        "controller/tre_controller/planning/safescale_evidence.py": "registry.slo_mode()",
        "baselines/tre_baselines/tools/slo_judge.py": "registry.slo_mode()",
    }
    for rel, needle in readers.items():
        assert needle in (TRE / rel).read_text(encoding="utf-8"), rel


def _registry_file(tmp_path: Path, mode: str) -> Path:
    raw = yaml.safe_load(REGISTRY.read_text(encoding="utf-8"))
    raw["slo_mode"] = mode
    path = tmp_path / f"registry-{mode}.yaml"
    path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    return path


@pytest.mark.parametrize("mode", slo.SLO_MODES)
def test_the_switch_changes_every_consumer_at_once(tmp_path, mode) -> None:
    from tre_baselines.config import model_limits
    from tre_baselines.tools import slo_judge
    from tre_common.slo_labels import label_def_for_model
    from tre_controller.planning.safescale_evidence import RegistryThresholds
    from tre_eval import metrics

    path = _registry_file(tmp_path, mode)
    reg = load_registry(str(path))
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    eval_mode = slo.resolve_slo_mode(raw).mode
    blocks = {m["name"]: m["slo"] for m in raw["models"]}
    probe = RegistryThresholds(reg)
    limits = model_limits(reg)
    for model in MODELS:
        want = slo.slo_thresholds(reg.model(model).slo, mode)
        label = label_def_for_model(model, registry=reg)
        judge = slo_judge.unified_slo(model, reg)
        assert label.ttft_slo_mode == judge.ttft_slo_mode == limits[model].slo.ttft_slo_mode == eval_mode == mode
        for L in (256, 2048, 6000):
            ttft = want.ttft_ms(L)
            assert label.ttft_slo_ms(L) == judge.ttft_slo_ms(L) == limits[model].slo.ttft_slo_ms(L) == ttft
            assert probe.resolve(model, float(L))["ttft_ms"] == ttft
            under = (ttft - 1.0, ttft - 1.0 + 9 * (want.tpot_ms - 0.1))   # 1 ms / 0.1 ms under the limits
            row = {"input_tokens": L, "ttft": under[0] / 1000.0, "e2e_latency": under[1] / 1000.0,
                   "output_tokens": 10, "success_strict": True}
            scored = metrics.score_request(row, blocks[model], eval_mode)
            assert scored["ttft_thr_ms"] == ttft and not scored["viol"]
            over = metrics.score_request({**row, "ttft": (ttft + 1.0) / 1000.0}, blocks[model], eval_mode)
            assert over["v_ttft"] and over["viol"]
            slow = metrics.score_request({**row, "e2e_latency": (under[0] + 9 * (want.tpot_ms + 0.1)) / 1000.0},
                                         blocks[model], eval_mode)
            assert slow["v_tpot"] and not slow["v_ttft"]
        assert label.tpot_p95_ms == judge.tpot_p95_ms == limits[model].tpot_slo_ms == want.tpot_ms
        assert probe.resolve(model, None)["tpot_ms"] == want.tpot_ms


def test_params_set_publishes_the_switch_and_the_idle_multiple_parameters() -> None:
    from scripts import registry_params as rp

    legacy = yaml.safe_load(REGISTRY.read_text(encoding="utf-8"))
    legacy.pop("slo_mode")
    legacy["safescale"]["slo_mode"] = "fixed"
    for model in legacy["models"]:
        model["slo"]["ttft_slo_mode"] = "slowdown"
    text = yaml.safe_dump(legacy, sort_keys=False)
    names = [m["name"] for m in legacy["models"]]
    edits = rp.parse_sets(["slo_mode=idle_multiple", "dsqwen-14b.slo.idle_multiple.ttft_k=4"], names)
    new = yaml.safe_load(rp.apply_and_validate(text, edits))
    assert new["slo_mode"] == "idle_multiple" and "slo_mode" not in new["safescale"]
    assert all("ttft_slo_mode" not in m["slo"] for m in new["models"])     # one switch left
    assert new["models"][2]["slo"]["idle_multiple"]["ttft_k"] == 4.0
    changes = rp.changed_fields(text, rp.apply_and_validate(text, edits))
    assert ("registry", "slo_mode", None, "idle_multiple") in changes
    with pytest.raises(rp.ParamValidationError):
        rp.apply_and_validate(text, rp.parse_sets(["slo_mode=median"], names))
