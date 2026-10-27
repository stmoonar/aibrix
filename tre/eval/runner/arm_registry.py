#!/usr/bin/env python3
"""Per-arm registry parameters (parameter-sensitivity arms): resolve, apply, verify, restore.

The live registry (ConfigMap tre-v2-registry) is the only config source (AGENTS.md): an arm that
perturbs a parameter changes the LIVE registry through deploy/scripts/release/params_set.py
(``--apply --restart --backup-dir``: one ConfigMap write, then restart-to-apply SM, then controller)
and puts the original values back the same way after the arm. Campaign file::

  arm_defs:
    sens-theta-m20:
      base: tre
      registry_params:            # params_set paths; "*" = every model of the registry
        "*.trs.theta_m": x0.8     #   "x<f>": the live value times f (rounded to 6 decimals)
        "*.trs.w_p": 1.0          #   a number: that absolute value
      registry_transforms:        # named transforms
        tau_band: 0.8             #   tau_crit and tau_high x f together, tau_low unchanged (v1 TAU rule);
                                  #   refused unless 0 < tau_crit' < tau_low < tau_high'
  registry_expect:                # the canonical registry this campaign was planned for
    registry_id: 0123456789abcdef #   `arm_registry.py id` on the live ConfigMap (null = not filled in: refused)
    present: ["*.alt_thresholds.queue_len.delta_crit"]   # paths that must hold a value

The factors apply to the canonical (live, unperturbed) values at the start of each arm; nothing
is refitted. Crash safety: before the first write the original values go into the annotation
``tre.dev/arm-registry-restore`` of the registry ConfigMap. ``restore`` (after every arm, in
reset_canonical.sh, hence also at the next pre-check after a crash) writes them back through
params_set, verifies them and removes the annotation; while the annotation is there the cluster
is not canonical.

CLI (reset_canonical.sh, operators):
  arm_registry.py check                          exit 0 = no perturbation in effect, 1 = in effect
  arm_registry.py restore --backup-dir DIR       put the recorded values back (no-op when none)
  arm_registry.py id [--registry-file F]         registry id of the live ConfigMap (or of a file)
  arm_registry.py preview --rule PATH=VALUE ... [--tau-band F] [--registry-file F]
                                                 the concrete --set list an arm would apply
Environment: TRE_NS (tre-v2), REGISTRY_CM (tre-v2-registry), KUBECTL (kubectl), TRE_DIR (the tree
whose deploy/scripts/release/params_set.py is used; default the tree this file is in), plus what
params_set.py reads (SM_DEPLOY, CONTROLLER_DEPLOY, ROLLOUT_TIMEOUT).
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

HERE = Path(__file__).resolve().parent
ANNOTATION = "tre.dev/arm-registry-restore"
TRANSFORMS = ("tau_band",)
_PATH = re.compile(r"(?P<model>\*|[A-Za-z0-9][A-Za-z0-9_.-]*?)\.(?P<path>(trs|slo)\.\w+|alt_thresholds\.\w+\.\w+|"
                   r"min_replicas|max_awake_replicas)")
_FACTOR = re.compile(r"x(?P<f>[0-9]*\.?[0-9]+(?:[eE][-+]?\d+)?)")
_ROUND = 6


class RegistryError(Exception):
    pass


def _yaml():
    import yaml  # noqa: PLC0415
    return yaml


# ------------------------------------------------------------------ spec (offline)
@dataclass(frozen=True)
class Rule:
    model: str          # a registry model name or "*"
    path: str           # params_set path below the model, e.g. trs.theta_m
    op: str             # "mul" | "set" | "tau_band"
    value: float


def parse_spec(where: str, params: Any, transforms: Any) -> list[Rule]:
    """Validate an arm's registry_params / registry_transforms; returns the rules (empty = no change)."""
    rules: list[Rule] = []
    if params:
        if not isinstance(params, dict):
            raise RegistryError(f"{where}.registry_params must be a mapping of <model|*>.<path>: <value | x<factor>>")
        for key, raw in params.items():
            m = _PATH.fullmatch(str(key))
            if not m:
                raise RegistryError(f"{where}.registry_params: {key!r} is not <model|*>.<params_set path> (trs.<f>, "
                                    "slo.<f>, alt_thresholds.<signal>.<f>, min_replicas, max_awake_replicas)")
            if m.group("path") == "trs.ema_tau_ms":
                raise RegistryError(f"{where}.registry_params: trs.ema_tau_ms is locked (window-coupled)")
            if isinstance(raw, str):
                f = _FACTOR.fullmatch(raw.strip())
                if not f or not float(f.group("f")) > 0:
                    raise RegistryError(f"{where}.registry_params.{key}: {raw!r} is neither a number nor x<factor > 0>")
                rules.append(Rule(m.group("model"), m.group("path"), "mul", float(f.group("f"))))
            elif isinstance(raw, (int, float)) and not isinstance(raw, bool) and math.isfinite(raw):
                rules.append(Rule(m.group("model"), m.group("path"), "set", raw))
            else:
                raise RegistryError(f"{where}.registry_params.{key}: {raw!r} is neither a number nor x<factor>")
    if transforms:
        if not isinstance(transforms, dict):
            raise RegistryError(f"{where}.registry_transforms must be a mapping {{name: factor}}")
        for name, f in transforms.items():
            if name not in TRANSFORMS:
                raise RegistryError(f"{where}.registry_transforms: unknown {name!r} (known {list(TRANSFORMS)})")
            if isinstance(f, bool) or not isinstance(f, (int, float)) or not f > 0:
                raise RegistryError(f"{where}.registry_transforms.{name} must be a factor > 0")
            rules.append(Rule("*", "trs.tau_crit+tau_high", "tau_band", float(f)))
    return rules


def describe(rules: list[Rule]) -> list[str]:
    out = []
    for r in rules:
        if r.op == "tau_band":
            out.append(f"tau_band x{r.value:g} (tau_crit, tau_high; tau_low unchanged)")
        else:
            out.append(f"{r.model}.{r.path} {'x' if r.op == 'mul' else '='}{r.value:g}")
    return out


# ------------------------------------------------------------------ resolution against a registry text
def _models(data: dict) -> dict[str, dict]:
    return {m.get("name"): m for m in (data.get("models") or []) if isinstance(m, dict)}


def get_path(model: dict, path: str) -> Any:
    cur: Any = model
    for part in path.split("."):
        if not isinstance(cur, dict):
            return None
        cur = cur.get(part)
    return cur


def split_key(key: str, names: list[str]) -> tuple[str, str]:
    """``<model>.<path>`` -> (model, path); the longest model name that prefixes the key (params_set's rule)."""
    model = max((n for n in names if key.startswith(f"{n}.")), key=len, default=None)
    if model is None:
        raise RegistryError(f"{key}: no registry model prefixes it")
    return model, key[len(model) + 1:]


def _num(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def resolve(rules: list[Rule], registry_yaml: str) -> tuple[dict[str, Any], dict[str, Any]]:
    """The concrete edits of ``rules`` on this registry: ({"<model>.<path>": new}, {"<model>.<path>": old}).
    Entries whose new value equals the old one are left out. Refuses a path set twice, a missing value,
    a factor on a non-numeric field (params_set rejects a fraction on an integer field), and a tau band that breaks 0 < tau_crit < tau_low < tau_high."""
    data = _yaml().safe_load(registry_yaml) or {}
    models = _models(data)
    sets: dict[str, Any] = {}
    olds: dict[str, Any] = {}

    def put(name: str, path: str, new: Any) -> None:
        key = f"{name}.{path}"
        if key in sets:
            raise RegistryError(f"{key} is set by two rules")
        old = get_path(models[name], path)
        olds[key] = old
        sets[key] = new

    for r in rules:
        names = sorted(models) if r.model == "*" else [r.model]
        for name in names:
            if name not in models:
                raise RegistryError(f"{r.model}.{r.path}: model {name!r} is not in the registry")
            if r.op == "tau_band":
                trs = models[name].get("trs") or {}
                tc, tl, th = trs.get("tau_crit"), trs.get("tau_low"), trs.get("tau_high")
                if not all(_num(x) for x in (tc, tl, th)):
                    raise RegistryError(f"{name}: tau_band needs numeric trs.tau_crit / tau_low / tau_high")
                nc, nh = round(tc * r.value, _ROUND), round(th * r.value, _ROUND)
                if not 0.0 < nc < tl < nh:
                    raise RegistryError(f"{name}: tau_band x{r.value:g} gives tau_crit {nc} / tau_high {nh}; "
                                        f"needs 0 < tau_crit < tau_low ({tl}) < tau_high")
                put(name, "trs.tau_crit", nc)
                put(name, "trs.tau_high", nh)
                continue
            old = get_path(models[name], r.path)
            if r.op == "mul":
                if not _num(old):
                    raise RegistryError(f"{name}.{r.path}: x{r.value:g} needs a numeric value in the registry (got {old!r})")
                put(name, r.path, round(old * r.value, _ROUND))
            else:
                if old is None:
                    raise RegistryError(f"{name}.{r.path}: not in the registry (an arm only changes existing values)")
                put(name, r.path, r.value)
    changed = {k: v for k, v in sets.items() if not same(olds[k], v)}
    return changed, {k: olds[k] for k in changed}


def same(a: Any, b: Any) -> bool:
    if _num(a) and _num(b):
        return abs(float(a) - float(b)) <= 1e-9 * max(1.0, abs(float(b)))
    return a == b


def values_at(registry_yaml: str, keys: list[str]) -> dict[str, Any]:
    data = _yaml().safe_load(registry_yaml) or {}
    models = _models(data)
    out = {}
    for k in keys:
        name, path = split_key(k, list(models))
        out[k] = get_path(models[name], path)
    return out


def _norm(v: Any) -> Any:
    """Every non-bool number as float (params_set's coercion writes a float field as float: lambda_wait: 3 ->
    3.0) and no empty mappings (params_set's edit adds empty trs / slo / alt_thresholds sections to a model
    that has none): neither changes the id."""
    if isinstance(v, dict):
        return {k: _norm(x) for k, x in v.items() if x != {}}
    if isinstance(v, list):
        return [_norm(x) for x in v]
    return float(v) if _num(v) else v


def registry_id(registry_yaml: str) -> str:
    """Content id of a registry: sha256 of its parsed content (key order, comments, float spelling and
    int-vs-float do not count; a params_set re-render keeps it), first 16 hex."""
    data = _norm(_yaml().safe_load(registry_yaml) or {})
    return hashlib.sha256(json.dumps(data, sort_keys=True, separators=(",", ":"), default=str).encode()).hexdigest()[:16]


def set_arg(key: str, value: Any) -> str:
    """``key=VALUE`` for params_set (VALUE is parsed as YAML there: dump it as YAML so 1e-05 stays a float)."""
    return f"{key}={_yaml().safe_dump(value).splitlines()[0]}"


# ------------------------------------------------------------------ campaign-level expectation
def expect_spec_problems(expect: Any) -> list[str]:
    """Offline problems of the campaign's registry_expect (placeholders not filled in = refused)."""
    if expect is None:
        return []
    if not isinstance(expect, dict):
        return ["registry_expect must be a mapping (registry_id, present)"]
    out = []
    rid = expect.get("registry_id")
    if not rid:
        out.append("registry_expect.registry_id is not filled in: publish the recalibrated registry, then put "
                   "`arm_registry.py id` here (the campaign refuses to start without it)")
    elif not re.fullmatch(r"[0-9a-f]{16}", str(rid)):
        out.append(f"registry_expect.registry_id {rid!r} is not 16 hex digits (arm_registry.py id)")
    pres = expect.get("present") or []
    if not isinstance(pres, list) or any(not _PATH.fullmatch(str(p)) for p in pres):
        out.append("registry_expect.present must be a list of <model|*>.<params_set path>")
    return out


def expect_problems(expect: Optional[dict], registry_yaml: str) -> list[str]:
    """Live problems: the canonical registry is not the expected one / a required value is missing."""
    if not expect:
        return []
    out = []
    rid = registry_id(registry_yaml)
    if expect.get("registry_id") and rid != expect["registry_id"]:
        out.append(f"live registry id {rid} != registry_expect.registry_id {expect['registry_id']} "
                   "(not the registry this campaign was planned for)")
    data = _yaml().safe_load(registry_yaml) or {}
    models = _models(data)
    for p in expect.get("present") or []:
        m = _PATH.fullmatch(str(p))
        if not m:
            continue
        for name in (sorted(models) if m.group("model") == "*" else [m.group("model")]):
            if name not in models or get_path(models[name], m.group("path")) is None:
                out.append(f"{name}.{m.group('path')} missing in the live registry")
    return out


# ------------------------------------------------------------------ cluster access (faked in the tests)
class Cluster:
    def __init__(self, env: Optional[dict] = None):
        self.e = dict(os.environ if env is None else env)

    def _kubectl(self, args: list[str], stdin: Optional[str] = None) -> str:
        cmd = [self.e.get("KUBECTL", "kubectl"), "-n", self.e.get("TRE_NS", "tre-v2"), *args]
        try:
            r = subprocess.run(cmd, input=stdin, capture_output=True, text=True, timeout=120)
        except (OSError, subprocess.SubprocessError) as exc:   # TimeoutExpired, kubectl missing
            raise RegistryError(f"{' '.join(cmd[:6])}: {type(exc).__name__}: {exc}") from exc
        if r.returncode != 0:
            raise RegistryError(f"{' '.join(cmd[:6])}: {r.stderr.strip()[:300]}")
        return r.stdout

    def read_cm(self) -> dict:
        out = self._kubectl(["get", "configmap", self.e.get("REGISTRY_CM", "tre-v2-registry"), "-o", "json"])
        try:
            return json.loads(out)
        except ValueError as exc:
            raise RegistryError(f"registry ConfigMap: not JSON: {exc}") from exc

    def annotate(self, value: Optional[str], resource_version: Optional[str] = None) -> None:
        name = self.e.get("REGISTRY_CM", "tre-v2-registry")
        if value is None:
            self._kubectl(["annotate", "configmap", name, f"{ANNOTATION}-"])
            return
        args = ["annotate", "configmap", name, f"{ANNOTATION}={value}", "--overwrite"]
        if resource_version:
            args.append(f"--resource-version={resource_version}")
        self._kubectl(args)

    def params_set(self, args: list[str]) -> tuple[int, str]:
        tre = Path(self.e.get("TRE_DIR") or HERE.parents[1])
        try:
            r = subprocess.run([sys.executable, str(tre / "deploy/scripts/release/params_set.py"), *args],
                               capture_output=True, text=True, timeout=1800, env=self.e)
        except (OSError, subprocess.SubprocessError) as exc:
            raise RegistryError(f"params_set {' '.join(args[:3])}: {type(exc).__name__}: {exc}") from exc
        return r.returncode, (r.stdout + r.stderr)


def _data(cm: dict) -> str:
    return (cm.get("data") or {}).get("registry.yaml", "")


def _annotation(cm: dict) -> Optional[dict]:
    raw = ((cm.get("metadata") or {}).get("annotations") or {}).get(ANNOTATION)
    if not raw:
        return None
    try:
        rec = json.loads(raw)
    except ValueError as exc:
        raise RegistryError(f"annotation {ANNOTATION} is not JSON: {raw[:200]}") from exc
    if not isinstance(rec, dict) or not isinstance(rec.get("restore"), dict):
        raise RegistryError(f"annotation {ANNOTATION} has no restore map: {raw[:200]}")
    return rec


def _tail(text: str, n: int = 12) -> str:
    return "\n".join(text.strip().splitlines()[-n:])


def in_effect(cl: Cluster) -> Optional[dict]:
    """The recorded perturbation (annotation), or None when the registry is canonical."""
    return _annotation(cl.read_cm())


def _status_in_sync(cl: Cluster) -> tuple[bool, str]:
    rc, out = cl.params_set(["--status"])
    last = out.strip().splitlines()[-1] if out.strip() else ""
    return rc == 0 and last.strip() == "in sync", _tail(out, 4)


def verify(cl: Cluster, want: dict[str, Any]) -> str:
    """The live ConfigMap holds ``want`` and both Deployments were restarted with it (params_set
    --status in sync). Returns the live registry text; raises RegistryError otherwise."""
    text = _data(cl.read_cm())
    got = values_at(text, list(want))
    bad = {k: (got[k], v) for k, v in want.items() if not same(got[k], v)}
    if bad:
        raise RegistryError("live ConfigMap differs from the wanted values: "
                            + "; ".join(f"{k} = {g!r}, want {w!r}" for k, (g, w) in sorted(bad.items())))
    ok, st = _status_in_sync(cl)
    if not ok:
        raise RegistryError(f"params_set --status is not in sync (restart-to-apply pending?): {st}")
    return text


def apply(cl: Cluster, rules: list[Rule], *, label: str, backup_dir: str,
          log: Callable[[str], None] = lambda s: None) -> dict:
    """Apply an arm's rules to the live registry (restart-to-apply), verify, and return the record.
    The restore map is in the ConfigMap annotation before the first write."""
    cm = cl.read_cm()
    prev = _annotation(cm)
    if prev:
        raise RegistryError(f"a registry perturbation is already in effect ({prev.get('label')}); restore it first")
    text = _data(cm)
    sets, olds = resolve(rules, text)
    rec: dict = {"label": label, "at": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
                 "rules": describe(rules), "canonical_id": registry_id(text), "set": sets, "restore": olds,
                 "backup_dir": backup_dir, "applied": False}
    if not sets:
        log(f"registry params of {label}: no value changes")
        return rec
    ann = json.dumps({k: rec[k] for k in ("label", "at", "canonical_id", "set", "restore")}, sort_keys=True,
                     separators=(",", ":"))
    cl.annotate(ann, (cm.get("metadata") or {}).get("resourceVersion"))
    args = [x for k, v in sets.items() for x in ("--set", set_arg(k, v))]
    rc, out = cl.params_set(args + ["--apply", "--restart", "--backup-dir", backup_dir, "--reason", f"arm {label}"[:200]])
    rec["params_set_output"] = _tail(out)
    if rc != 0:
        raise RegistryError(f"params_set --apply --restart rc={rc}: {_tail(out, 4)}")
    rec["perturbed_id"] = registry_id(verify(cl, sets))
    rec["applied"] = True
    log(f"registry params of {label} applied and verified: " + ", ".join(f"{k}={v}" for k, v in sets.items()))
    return rec


def restore(cl: Cluster, *, backup_dir: str, reason: str = "arm-registry restore", expect_label: Optional[str] = None,
            log: Callable[[str], None] = lambda s: None) -> Optional[dict]:
    """Put the values recorded in the annotation back (params_set --apply --restart, or only --restart
    when the values are back but a restart is pending), verify, remove the annotation. Idempotent;
    None when nothing was in effect.

    Each recorded key is classified on the live registry: ``set`` (still the arm's value), ``want``
    (already the original) or ``other`` (a third value: someone else changed it). ``other`` keys are
    never overwritten: the ``set`` keys are written back, the annotation stays and RegistryError
    names the keys (a human decides; the pre-check blocks meanwhile). ``held`` = every key was still
    the arm's value, i.e. the perturbation held for the whole arm. With ``expect_label`` a record of
    another label (another runner's or attempt's perturbation) is left alone: {"skipped": ...}."""
    cm = cl.read_cm()
    rec = _annotation(cm)
    if rec is None:
        return None
    if expect_label is not None and rec.get("label") != expect_label:
        log(f"registry restore: the record in effect is {rec.get('label')!r}, not {expect_label!r}; left alone")
        return {"skipped": f"record of {rec.get('label')!r} in effect", "label": rec.get("label")}
    want = rec["restore"]
    applied = rec.get("set") or {}
    text = _data(cm)
    got = values_at(text, list(want))
    cls = {k: ("set" if k in applied and same(got[k], applied[k]) else "want" if same(got[k], v) else "other")
           for k, v in want.items()}
    other = {k: got[k] for k, c in cls.items() if c == "other"}
    held = all(c == "set" for c in cls.values())
    back = {k: v for k, v in want.items() if cls[k] == "set"}
    out = ""
    if back:
        args = [x for k, v in back.items() for x in ("--set", set_arg(k, v))]
        rc, out = cl.params_set(args + ["--apply", "--restart", "--backup-dir", backup_dir, "--reason", f"{reason} {rec.get('label')}"[:200]])
        if rc != 0:
            raise RegistryError(f"restore: params_set --apply --restart rc={rc}: {_tail(out, 4)}")
    elif not _status_in_sync(cl)[0]:
        rc, out = cl.params_set(["--restart", "--reason", f"{reason} {rec.get('label')}"[:200]])
        if rc != 0:
            raise RegistryError(f"restore: params_set --restart rc={rc}: {_tail(out, 4)}")
    if other:
        raise RegistryError(f"restore of {rec.get('label')}: changed by someone else during the arm, NOT overwritten: "
                            + "; ".join(f"{k} = {v!r} (arm set {applied.get(k)!r}, original {want[k]!r})"
                                        for k, v in sorted(other.items()))
                            + f". Decide, then `kubectl annotate configmap <registry> {ANNOTATION}-`")
    live = registry_id(verify(cl, want))
    cl.annotate(None)
    res = {"label": rec.get("label"), "restored": want, "written": back, "classes": cls, "held": held,
           "canonical_id": rec.get("canonical_id"), "live_id": live,
           "canonical_id_match": live == rec.get("canonical_id"), "params_set_output": _tail(out)}
    log(f"registry restored after {rec.get('label')}: {len(back)} value(s) written"
        + ("" if res["canonical_id_match"] else f"; WARN live id {live} != canonical {rec.get('canonical_id')} "
           "(another change to the registry meanwhile?)"))
    return res


def signal_log_theta_check(arm_dir: str | Path, sets: dict[str, Any]) -> tuple[list[str], list[str]]:
    """The theta the controller actually used (signal_log.jsonl theta_m per model) vs the applied
    ``<model>.trs.theta_m`` values (v1 check_run.py did this with the startup log). Returns
    (mismatches, unverified)."""
    want = {k[: -len(".trs.theta_m")]: v for k, v in sets.items() if k.endswith(".trs.theta_m")}
    if not want:
        return [], []
    seen: dict[str, set] = {m: set() for m in want}
    try:
        with open(Path(arm_dir) / "signal_log.jsonl") as fh:
            for line in fh:
                try:
                    r = json.loads(line)
                    if r.get("model") in seen and r.get("theta_m") not in (None, "", "nan"):
                        seen[r["model"]].add(round(float(r["theta_m"]), 6))
                except (ValueError, TypeError, AttributeError):
                    continue
    except OSError:
        return [], [f"no signal_log.jsonl: theta of {sorted(want)} not verified"]
    bad, unver = [], []
    for m, v in want.items():
        if not seen[m]:
            unver.append(f"{m}: no theta_m in signal_log.jsonl")
        elif any(not same(x, v) for x in seen[m]):
            bad.append(f"{m}: signal_log theta_m {sorted(seen[m])} != applied {v}")
    return bad, unver


# ------------------------------------------------------------------ CLI
def main(argv: Optional[list] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("check")
    p_r = sub.add_parser("restore")
    p_r.add_argument("--backup-dir", required=True)
    p_r.add_argument("--reason", default="arm-registry restore")
    p_i = sub.add_parser("id")
    p_i.add_argument("--registry-file")
    p_p = sub.add_parser("preview")
    p_p.add_argument("--rule", action="append", default=[], metavar="<model|*>.<path>=<value|x<f>>")
    p_p.add_argument("--tau-band", type=float)
    p_p.add_argument("--registry-file")
    a = ap.parse_args(argv)
    cl = Cluster()
    try:
        if a.cmd == "check":
            rec = in_effect(cl)
            if rec is None:
                print("no registry perturbation in effect")
                return 0
            print(f"registry perturbation in effect: {rec.get('label')} since {rec.get('at')}: {rec.get('set')}")
            return 1
        if a.cmd == "restore":
            res = restore(cl, backup_dir=a.backup_dir, reason=a.reason, log=print)
            if res is None:
                print("no registry perturbation in effect")
            return 0
        text = Path(a.registry_file).read_text() if getattr(a, "registry_file", None) else _data(cl.read_cm())
        if a.cmd == "id":
            print(registry_id(text))
            return 0
        params = {}
        for r in a.rule:
            k, sep, v = r.partition("=")
            if not sep:
                ap.error(f"--rule {r!r}: expected PATH=VALUE")
            params[k] = v if v.startswith("x") else float(v)
        rules = parse_spec("preview", params, {"tau_band": a.tau_band} if a.tau_band else None)
        sets, olds = resolve(rules, text)
        for k, v in sets.items():
            print(f"{k}: {olds[k]} -> {v}")
        print(" ".join(set_arg(k, v) for k, v in sets.items()) or "(no change)")
        return 0
    except RegistryError as exc:
        print(f"arm_registry: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
