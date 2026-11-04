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
4. No SLO value in code (2026-11-04): outside an exact allow-list of (file, hit), no file under
   tre/ holds a numeric literal in an SLO context - compared with, or dividing / multiplying, a
   value named after a latency (TTFT / TPOT / first token / ITL / latency; ``max(ttft / 500.0,
   tpot / 75.0)``, seconds included: ``r["ttft"] > 0.5``), as the default of such an option or
   parameter, as such a keyword or positional argument (matched to the callee's parameter
   names), as a dict value under such a key, as the constant of ``max`` / ``min`` with an
   idle / floor / SLO term (a TTFT floor), or as such a module constant; no ``.sh`` / ``.yaml``
   passes ``--ttft-*-ms <number>``; and no file compares a latency with an SLO threshold itself
   (instead of ``tre_common.slo.judge_request`` / ``violated_array`` or the window label).
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


#: Rule 4. Numbers that are units or neutral values, never an SLO (ms <-> s, ratios). Next to a
#: latency, 0.5 is not neutral: it is 500 ms in seconds (``r["ttft"] > 0.5``).
UNIT_NUMBERS = {0, 1, 2, 10, 100, 1000, 1e6, 1e-3, 1e-6, 0.5, 60}
LATENCY_UNIT_NUMBERS = UNIT_NUMBERS - {0.5}
#: What a latency is called (TTFT / TPOT and their other names: first token, ITL, latency).
LATENCY = re.compile(r"ttft|tpot|first_token|(?<![a-z])itl(?![a-z])|latency", re.IGNORECASE)
LIMIT = re.compile(r"slo|thr|limit", re.IGNORECASE)
#: ``max`` / ``min`` of a constant and a term named so: a TTFT floor (``max(450.0, k * idle)``).
FLOOR_CONTEXT = re.compile(r"ttft|tpot|first_token|(?<![a-z])itl(?![a-z])|latency|idle|slo|floor", re.IGNORECASE)
#: Shell scripts / YAML (campaign specs, drivers): an SLO value on a command line.
SHELL_FLAG = re.compile(r"--(?:ttft|tpot)-[a-z0-9-]*-ms[ =]+[\"']?\d+(?:\.\d+)?")

#: Rule 4 allow-lists, exact: (file, hit) -> why; the scan must find exactly these. LITERALS: an
#: SLO-context number that is not an SLO definition. COMPARES: code that compares a latency with a
#: threshold it got from tre_common.slo / the label (window-level or a diagnostic), not a request
#: verdict of its own. Not caught (documented limit): a number reaching a latency comparison
#: through a variable whose name says nothing (``lim = 500; lat > lim`` is a COMPARES hit only
#: when the variable is named like a limit).
LITERALS = {
    ("baselines/tre_baselines/tools/tokenscale_profile.py", "def make_stub_measure(ttft_ms_per_c=30.0)"):
        "the dry run's stub latency model, not an SLO",
    ("calibration/tre_calibration/fit.py", "DEFAULT_SURPLUS_LATENCY_QUANTILE = 0.35"): "a quantile, not a latency",
    ("common/tre_common/rediskeys.py", "SM_SLEEP_ACK_LATENCY_MAX = 5000"): "SM sleep-ack timeout (ms), not an SLO",
    ("common/tre_common/slo_labels.py", "ABLATION_TTFT_FLOOR_MS = 150.0"):
        "the D6 ablation arm (k 3 / floor 150 ms, plan 2026-09-21 6.9h): a label column of the 10-03 design",
    ("common/tre_common/slo_labels.py", "ABLATION_TTFT_SLOWDOWN_K = 3.0"): "the D6 ablation arm (as above)",
    ("deploy/scripts/calibration_design.py", "TTFT_SLO_MS = 500.0"):
        "the frozen D-line design record (preregistered_parameters, calibration_ladder.check_label_args); "
        "no verdict defaults to it (2026-11-04)",
    ("deploy/scripts/calibration_design.py", "TPOT_SLO_MS = 75.0"): "the frozen D-line design record (as above)",
    ("deploy/scripts/calibration_design.py", "{'median_p95_tpot_relative': 0.2}"): "sentinel drift tolerance (a ratio)",
    ("eval/runner/selftest_bl_tools.py", "r['phases'][1]['release_latency_s'] == 4"): "offline self-test fixture",
    ("eval/runner/selftest_bl_tools.py", "{'e2e_latency': 150.0}"): "offline self-test fixture",
    ("eval/runner/selftest_bl_tools.py", "{'ttft': 0.1}"): "offline self-test fixture",
    ("replayer/tre_replayer/run_trace.py", "StreamResult(first_token_ms=80.0)"): "dry-run synthetic response",
}
COMPARES = {
    ("common/tre_common/slo.py", "float(ttft_ms) > ttft_thr"): "the implementation",
    ("common/tre_common/slo.py", "np.nan_to_num(tpot, nan=0.0) > float(thresholds.tpot_ms)"): "the implementation",
    ("common/tre_common/slo.py", "np.nan_to_num(ttft, nan=0.0) > thr"): "the implementation",
    ("common/tre_common/slo.py", "tpot > thresholds.tpot_ms"): "the implementation",
    ("controller/tre_controller/planning/safescale.py", "tpot_p95_ms > limit_ms"):
        "SafeScale probe: window TPOT p95 (bucket_upper) against E = tre_common.slo.tpot_bucket_threshold_ms("
        "TPOT_slo, the histogram's edges) - a derived quantization of the one TPOT_slo, not a second "
        "definition (TPOT only, user 2026-11-04; per-request TPOT and E, user 2026-11-05)",
    ("baselines/tre_baselines/tools/tokenscale_profile.py", "s.tpot_p95_ms <= tpot_slo_ms"):
        "TokenScale profile: a step's p95 against the label's thresholds",
    ("baselines/tre_baselines/tools/tokenscale_profile.py", "s.ttft_p95_ms <= ttft_slo_ms"): "TokenScale profile (as above)",
    ("deploy/scripts/calibration_campaign.py", "p95_tpot <= tpot_slo_ms"):
        "10-03 capacity prior: a window p95 against the run's fixed thresholds",
    ("deploy/scripts/calibration_campaign.py", "p95_ttft <= ttft_slo_ms"): "10-03 capacity prior (as above)",
    ("deploy/scripts/dline_refit.py", "ttft <= label.ttft_slo_ms(length)"):
        "D6' validation diagnostic: per-bucket TTFT attainment against LabelDefinition.ttft_slo_ms",
    ("deploy/scripts/openloop.py", "float(tpot) <= tpot_slo_ms"):
        "10-03 open-loop ladder goodput (fixed thresholds passed in by the caller; kept to reproduce)",
    ("deploy/scripts/openloop.py", "float(ttft) <= ttft_slo_ms"): "10-03 open-loop ladder goodput (as above)",
    ("replayer/tre_replayer/scoring.py", "m['tpot_ms'] > tpot_slo_ms"): "v1-parity V_sys diagnostic (fixed + e2e)",
    ("replayer/tre_replayer/scoring.py", "m['ttft_ms'] > ttft_slo_ms"): "v1-parity V_sys diagnostic (as above)",
}
#: Shell / YAML hits (file, hit) -> why: none.
SHELL_FLAGS: dict[tuple[str, str], str] = {}


def _number(node: ast.AST, units: set = UNIT_NUMBERS) -> bool:
    return (isinstance(node, ast.Constant) and isinstance(node.value, (int, float))
            and not isinstance(node.value, bool) and abs(node.value) not in units
            and abs(node.value) >= 0.01)   # tolerances (1e-9) are not SLO values


def _lat_number(node: ast.AST) -> bool:
    return _number(node, LATENCY_UNIT_NUMBERS)


def _named(node: ast.AST, rx: re.Pattern) -> bool:
    if isinstance(node, (ast.Constant, ast.Set, ast.List, ast.Tuple, ast.Dict)):
        return False
    return bool(rx.search(ast.unparse(node)))


def _signatures(trees) -> dict[str, list[list[str]]]:
    """Callee simple name -> positional parameter names (functions, ``__init__`` without self,
    dataclass-style fields), so ``f(m, 500.0, 75.0)`` is matched to ``f``'s parameter names."""
    sig: dict[str, list[list[str]]] = {}
    for tree in trees:
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                sig.setdefault(node.name, []).append([x.arg for x in node.args.posonlyargs + node.args.args])
            elif isinstance(node, ast.ClassDef):
                init = [n for n in node.body if isinstance(n, ast.FunctionDef) and n.name == "__init__"]
                if init:
                    sig.setdefault(node.name, []).append(
                        [x.arg for x in init[0].args.posonlyargs + init[0].args.args][1:])
                else:
                    fields = [n.target.id for n in node.body
                              if isinstance(n, ast.AnnAssign) and isinstance(n.target, ast.Name)]
                    if fields:
                        sig.setdefault(node.name, []).append(fields)
    return sig


_MEMBERSHIP = (ast.In, ast.NotIn, ast.Is, ast.IsNot)


def _literal_and_compare_hits(tree: ast.AST, sig: dict | None = None) -> tuple[set[str], set[str]]:
    sig = sig if sig is not None else _signatures([tree])
    literals: set[str] = set()
    compares: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and _lat_number(node.value):
            for name in (n.id for t in node.targets for n in ast.walk(t) if isinstance(n, ast.Name)):
                if LATENCY.search(name):
                    literals.add(f"{name} = {node.value.value}")
        elif isinstance(node, ast.Compare) and not any(isinstance(op, _MEMBERSHIP) for op in node.ops):
            ops = [node.left, *node.comparators]
            if any(_named(o, LATENCY) for o in ops) and any(_lat_number(o) for o in ops):
                literals.add(ast.unparse(node)[:80])
            elif (any(_named(o, LATENCY) and not _named(o, LIMIT) for o in ops)
                  and any(_named(o, LIMIT) for o in ops)):
                compares.add(ast.unparse(node)[:80])
        elif isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Div, ast.Mult)):
            if ((_named(node.left, LATENCY) and _lat_number(node.right))
                    or (_named(node.right, LATENCY) and _lat_number(node.left))):
                literals.add(ast.unparse(node)[:80])
        elif isinstance(node, ast.Dict):
            for k, v in zip(node.keys, node.values):
                if (isinstance(k, ast.Constant) and isinstance(k.value, str) and LATENCY.search(k.value)
                        and _lat_number(v)):
                    literals.add(f"{{{k.value!r}: {v.value}}}")
        elif isinstance(node, ast.Call):
            func = node.func
            name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", "")
            if name == "add_argument":
                flags = " ".join(a.value for a in node.args if isinstance(a, ast.Constant) and isinstance(a.value, str))
                for kw in node.keywords:
                    if kw.arg == "default" and _lat_number(kw.value) and LATENCY.search(flags):
                        literals.add(f"add_argument({flags}, default={kw.value.value})")
            for kw in node.keywords:
                if kw.arg and LATENCY.search(kw.arg) and _lat_number(kw.value):
                    literals.add(f"{name}({kw.arg}={kw.value.value})")
            if name in ("max", "min") and not node.keywords:
                others = [a for a in node.args if not isinstance(a, ast.Constant)]
                if any(_lat_number(a) for a in node.args) and any(_named(a, FLOOR_CONTEXT) for a in others):
                    literals.add(ast.unparse(node)[:80])
            for params in sig.get(name, ()):
                for i, arg in enumerate(node.args):
                    if i < len(params) and LATENCY.search(params[i]) and _lat_number(arg):
                        literals.add(f"{name}(<{params[i]}>={arg.value})")
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            a = node.args
            pos = a.posonlyargs + a.args
            pairs = list(zip(pos[len(pos) - len(a.defaults):], a.defaults))
            pairs += [(x, d) for x, d in zip(a.kwonlyargs, a.kw_defaults) if d is not None]
            for arg, default in pairs:
                if LATENCY.search(arg.arg) and _lat_number(default):
                    literals.add(f"def {node.name}({arg.arg}={default.value})")
    return literals, compares


def _shell_hits(text: str) -> set[str]:
    return {m.group(0) for line in text.splitlines() if not line.lstrip().startswith("#")
            for m in SHELL_FLAG.finditer(line)}


def _scanned(pattern: str):
    for path in sorted(TRE.rglob(pattern)):
        rel = path.relative_to(TRE).as_posix()
        if "/tests/" in rel or rel.startswith("docs/") or "/." in rel:
            continue
        yield rel, path


def _scan_rule4() -> tuple[set[tuple[str, str]], set[tuple[str, str]], set[tuple[str, str]]]:
    trees = {rel: ast.parse(path.read_text(encoding="utf-8")) for rel, path in _scanned("*.py")}
    sig = _signatures(trees.values())
    lit: set[tuple[str, str]] = set()
    cmp_: set[tuple[str, str]] = set()
    for rel, tree in trees.items():
        a, b = _literal_and_compare_hits(tree, sig)
        lit |= {(rel, h) for h in a}
        cmp_ |= {(rel, h) for h in b}
    shell = {(rel, h) for pattern in ("*.sh", "*.yaml", "*.yml") for rel, path in _scanned(pattern)
             for h in _shell_hits(path.read_text(encoding="utf-8", errors="replace"))}
    return lit, cmp_, shell


def test_no_slo_value_in_code_and_no_verdict_of_its_own() -> None:
    lit, cmp_, shell = _scan_rule4()
    assert not lit - set(LITERALS), (f"SLO values in code (read them from the registry through tre_common.slo, "
                                     f"or justify the exact hit in LITERALS): {sorted(lit - set(LITERALS))}")
    assert not cmp_ - set(COMPARES), (f"a latency compared with a threshold outside tre_common.slo (use "
                                      f"judge_request / violated_array, or justify in COMPARES): "
                                      f"{sorted(cmp_ - set(COMPARES))}")
    assert not shell - set(SHELL_FLAGS), f"SLO values on a command line (read the registry): {sorted(shell - set(SHELL_FLAGS))}"
    stale = sorted((set(LITERALS) - lit) | (set(COMPARES) - cmp_) | (set(SHELL_FLAGS) - shell))
    assert not stale, f"rule-4 allow-list entries that no longer match (remove them): {stale}"


@pytest.mark.parametrize("source, kind", [
    ("x = max(ttft / 500.0, tpot / 75.0)", "literal"),           # calibration_priors before 2026-11-04
    ("v = tt.fillna(0.0) > TTFT_SLO_MS", None),                   # a constant: rule 1 (SLO_CONSTANT)
    ("v = np.nan_to_num(tpot) > TPOT_SLO_MS", "compare"),         # supp_lib before 2026-11-04
    ("ap.add_argument('--ttft-slo-ms', type=float, default=500.0)", "literal"),
    ("f(model, ttft_p95_ms=500.0, tpot_p95_ms=75.0)", "literal"),  # dline_refit before 2026-11-04
    ("def g(rows, ttft_slo_ms=500.0):\n    pass", "literal"),
    ("NO_REGISTRY_TTFT_MS = 500.0", "literal"),
    # review 2026-11-04: forms the first rule-4 scan missed
    ("late = r['ttft'] > 0.5", "literal"),                        # seconds: 0.5 is 500 ms next to a latency
    ("cfg = {'ttft_p95_ms': 500}", "literal"),                    # a dict literal
    ("def f(model, ttft_slo_ms, tpot_slo_ms):\n    pass\nf(m, 500.0, 75.0)", "literal"),   # positional
    ("late = first_token_ms > 450.0", "literal"),                 # other names of a latency
    ("bad = itl_ms > 61.0", "literal"),
    ("slow = latency_ms > 3000", "literal"),
    ("thr = max(450.0, k * idle)", "literal"),                    # a TTFT floor
    # not hits
    ("ms = ttft_s * 1000.0", None),                               # a unit
    ("ok = ttft_ms is not None", None),
    ("half = width * 0.5", None),                                 # 0.5 away from a latency
    ("ok = slo_key not in LATENCY_COLUMNS", None),                # membership
    ("n = max(5, len(rows))", None),                              # max of a count
    ("wide = title_len > 40", None),                              # "itl" inside a word
])
def test_rule4_catches_the_inline_forms(source, kind) -> None:
    lit, cmp_ = _literal_and_compare_hits(ast.parse(source))
    assert (("literal" if lit else None) or ("compare" if cmp_ else None)) == kind, (lit, cmp_)


@pytest.mark.parametrize("line, hit", [
    ("python3 fit.py --ttft-p95-ms 500 --model m", True),         # run_alphafit_w10.sh before 2026-11-04
    ("  --tpot-slo-ms=75.0 \\", True),
    ("# --ttft-p95-ms 500 (comment)", False),
    ("--ttft-slo-mode fixed", False),
])
def test_rule4_catches_slo_values_on_a_command_line(line, hit) -> None:
    assert bool(_shell_hits(line)) is hit


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
    repo = yaml.safe_load(REGISTRY.read_text(encoding="utf-8"))
    assert repo["slo_mode"] == params["slo_mode"] and repo["slo_mode"] in slo.SLO_MODES   # one value, both files
    for raw in (repo, params):
        assert slo.resolve_slo_mode(raw).source == "slo_mode"
        for model in raw["models"]:
            assert set(model["slo"]["idle_multiple"]) == set(slo.IDLE_MULTIPLE_FIELDS), model["name"]
    # read: every consumer of "the definition in force" asks the registry's switch
    readers = {
        "controller/tre_controller/config.py": "slo_mode_resolution()",
        "controller/tre_controller/app.py": "registry.slo_mode()",
        "deploy/scripts/r3_grid.py": "registry.slo_mode()",
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
    from tre_controller.planning.safescale import tpot_limit_ms
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
        # SafeScale consumes the TPOT component only (user 2026-11-04): no TTFT threshold.
        assert probe.resolve(model) == {"mode": mode, "tpot_ms": want.tpot_ms, "source": "registry"}
        # Its threshold is E (user 2026-11-05): the smallest edge of the per-request TPOT
        # histogram the pods export (vLLM 0.30's) >= that same TPOT_slo - derived, one place.
        limit = tpot_limit_ms(probe.resolve(model), VLLM_030_TPOT_EDGES)
        assert limit == slo.tpot_bucket_threshold_ms(want.tpot_ms, VLLM_030_TPOT_EDGES) >= want.tpot_ms
        assert limit / 1000.0 in VLLM_030_TPOT_EDGES
        assert limit == SAFESCALE_E_MS[mode][model]


#: What the probe compares with, per mode, on the 0.30 edges (25 / 50 / 75 / 100 ms ...):
#: idle_multiple TPOT_slo 60.9 / 63.4 / 42.0 ms -> 75 / 75 / 50; fixed / slowdown 75 -> 75.
SAFESCALE_E_MS = {
    slo.SLO_MODE_FIXED: dict.fromkeys(MODELS, 75.0),
    slo.SLO_MODE_SLOWDOWN: dict.fromkeys(MODELS, 75.0),
    slo.SLO_MODE_IDLE_MULTIPLE: {"dsqwen-7b": 75.0, "dsllama-8b": 75.0, "dsqwen-14b": 50.0},
}


def _vllm_030_tpot_edges() -> tuple[float, ...]:
    from tre_controller.planning.safescale_direct import parse_vllm_metrics

    text = (TRE / "deploy" / "tests" / "fixtures" / "vllm_metrics" / "vllm-0.30.0.prom").read_text(encoding="utf-8")
    hist = parse_vllm_metrics(text, ts_ms=0).tpot
    return tuple(le for le, _ in hist.buckets if le != float("inf"))


VLLM_030_TPOT_EDGES = _vllm_030_tpot_edges()


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


def test_offline_label_tools_follow_the_switch_or_name_the_mode(tmp_path) -> None:
    """S2 (review 2026-11-04): r3_grid labels under the loaded registry's slo_mode; the fit tools whose
    output is compared with a frozen record (D6' reproductions) require --ttft-slo-mode."""
    from types import SimpleNamespace

    from scripts import r3_grid

    for mode in ("fixed", "slowdown"):
        reg = load_registry(str(_registry_file(tmp_path, mode)))
        args = SimpleNamespace(ttft_slo_ms=None, tpot_slo_ms=None, registry_slo_mode=reg.slo_mode())
        assert r3_grid.primary_label(args, reg.model("dsqwen-7b")).ttft_slo_mode == mode
    for rel in ("deploy/scripts/rewindow_from_raw.py", "deploy/scripts/refit_trs_params.py",
                "deploy/scripts/theta_verdict.py", "deploy/scripts/alpha_fit.py"):
        assert "require_mode=True" in (TRE / rel).read_text(encoding="utf-8"), rel
