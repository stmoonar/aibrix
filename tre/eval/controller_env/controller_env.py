#!/usr/bin/env python3
"""The env of the controller Deployment as a file: snapshot it, diff it, apply it in ONE rollout.

Why: a run is only reproducible if the controller env is on record. Part of it is not in the overlay
(deploy/overlays/tre-v2/controller.yaml) but was set on the live Deployment (``kubectl set env`` / ``patch``), and
the runner changes a few switches per arm (TRE_SAFESCALE_SCOPE, TRE_SIGNAL_SOURCE, ... see campaign.py
CONTROLLER_ENV_KEYS) and puts them back afterwards. The files next to this one are such records (README.md).

  controller_env.py snapshot [--out FILE]     the live env as NAME=VALUE lines (read only)
  controller_env.py diff FILE [--registry-file F]
                                              what applying FILE would change (read only); exit 0 = live env equals FILE,
                                              1 = it differs, 2 = FILE cannot be applied (see the message)
  controller_env.py apply FILE --backup-dir DIR [--registry-file F]
                                              WRITES THE CLUSTER: make the live env equal FILE with one ``kubectl patch``
                                              (one Recreate rollout of the controller). The patch carries the resourceVersion
                                              of the Deployment it was planned on, so a concurrent edit makes it fail instead of
                                              being overwritten. The env before goes to DIR/controller-env.before.env and the
                                              Deployment (the same read) to DIR/deploy.before.json;
                                              rollback = ``apply DIR/controller-env.before.env``.

FILE: ``NAME=VALUE`` per line, ``#`` comments and blank lines ignored, every line is the whole state: a live variable
that FILE does not name is removed (an entry fed by ``valueFrom`` is never touched). ``apply`` is refused when
  * an exclusive-window marker file exists (TRE_EXCLUSIVE_WINDOW_FILE, default $CAMPAIGN_DATA_ROOT/TRE_EXCLUSIVE_WINDOW): another
    session owns the cluster;
  * a campaign / arm runner or a campaign driver shell is active (the runner owns the per-arm switches while it runs);
  * the EFFECTIVE TRE_METRICS_WINDOW_MS differs from a model's trs.theta_window_ms in the registry (live ConfigMap, or
    --registry-file). Effective = the value in FILE; if FILE does not name it, applying unsets it and the controller falls
    back to its code default (read from this checkout's tre_controller.config, an older image may differ - name the key
    in FILE to be sure). The controller refuses to start on that mismatch (tre_common.tss.theta_basis_errors) and, being
    ``Recreate``, the cluster would be left without a controller. Publish the registry first, or both in one go.
``diff`` applies the window check only.
Environment: TRE_NS (tre-v2), CONTROLLER_DEPLOY (tre-v2-controller), CONTROLLER_CONTAINER (controller),
REGISTRY_CM (tre-v2-registry), KUBECTL (kubectl), TRE_EXCLUSIVE_WINDOW_FILE / CAMPAIGN_DATA_ROOT (marker location).
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Optional

NS = os.environ.get("TRE_NS", "tre-v2")
DEPLOY = os.environ.get("CONTROLLER_DEPLOY", "tre-v2-controller")
CONTAINER = os.environ.get("CONTROLLER_CONTAINER", "controller")
REGISTRY_CM = os.environ.get("REGISTRY_CM", "tre-v2-registry")
KUBECTL = os.environ.get("KUBECTL", "kubectl")
WINDOW_KEY = "TRE_METRICS_WINDOW_MS"
_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


class EnvError(Exception):
    pass


def kubectl(args: list[str], timeout: int = 300) -> str:
    r = subprocess.run([KUBECTL, "-n", NS, *args], capture_output=True, text=True, timeout=timeout)
    if r.returncode != 0:
        raise EnvError(f"kubectl {' '.join(args[:4])}: {r.stderr.strip()[:300]}")
    return r.stdout


def parse_env_file(text: str) -> dict[str, str]:
    env: dict[str, str] = {}
    for n, raw in enumerate(text.splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        name, sep, value = line.partition("=")
        if not sep or not _NAME.fullmatch(name):
            raise EnvError(f"line {n}: not NAME=VALUE: {raw[:80]!r}")
        if name in env:
            raise EnvError(f"line {n}: {name} twice")
        env[name] = value
    return env


def render_env_file(env: dict[str, str], header: str = "") -> str:
    head = "".join(f"# {l}\n" if l else "#\n" for l in header.splitlines())
    return head + "".join(f"{k}={env[k]}\n" for k in sorted(env))


def live_env(deploy: Optional[dict] = None) -> tuple[dict[str, str], list[str]]:
    """(plain NAME -> value entries of the controller container, names fed by valueFrom)."""
    d = deploy if deploy is not None else json.loads(kubectl(["get", "deploy", DEPLOY, "-o", "json"]))
    cs = [c for c in d["spec"]["template"]["spec"]["containers"] if c["name"] == CONTAINER]
    if not cs:
        raise EnvError(f"deployment {DEPLOY} has no container {CONTAINER}")
    plain: dict[str, str] = {}
    other: list[str] = []
    for e in cs[0].get("env") or []:
        if "valueFrom" in e:
            other.append(e["name"])
        else:
            plain[e["name"]] = e.get("value", "")
    return plain, other


def plan(live: dict[str, str], want: dict[str, str]) -> dict:
    return {
        "set": {k: v for k, v in want.items() if live.get(k) != v},
        "unset": sorted(k for k in live if k not in want),
        "same": sorted(k for k in want if live.get(k) == want[k]),
    }


def registry_window_errors(registry_text: str, window: str) -> list[str]:
    import yaml  # noqa: PLC0415
    data = yaml.safe_load(registry_text) or {}
    errs = []
    for m in data.get("models") or []:
        w = ((m.get("trs") or {}).get("theta_window_ms"))
        if w is None or float(w) != float(window):
            errs.append(f"{m.get('name')}: trs.theta_window_ms={w!r} but {WINDOW_KEY}={window}")
    return errs


#: Anchored (``^``) so the shell that wraps this very command line does not match. A driver is a shell script under
#: eval/runner/drivers/ (run_campaign-*.sh, run_all3.sh, ...); run_campaign.sh is the generic launcher.
_ACTIVE_PATTERNS = (
    r"^python3 .*campaign\.py run",
    r"^bash .*run_arm_pilot\.sh",
    r"^bash .*/drivers/.*\.sh",
    r"^bash .*run_campaign\.sh",
    r"^bash .*run_all3\.sh",
)


def runner_active() -> list[str]:
    own = {str(os.getpid()), str(os.getppid())}
    out: list[str] = []
    for pat in _ACTIVE_PATTERNS:
        r = subprocess.run(["pgrep", "-f", pat], capture_output=True, text=True)
        out += [pid for pid in r.stdout.split() if pid not in own and pid not in out]
    return out


def exclusive_window() -> Optional[str]:
    """The first line of the exclusive-window marker if it exists (None: no marker, or its location is not configured)."""
    path = os.environ.get("TRE_EXCLUSIVE_WINDOW_FILE")
    if not path and os.environ.get("CAMPAIGN_DATA_ROOT"):
        path = str(Path(os.environ["CAMPAIGN_DATA_ROOT"]) / "TRE_EXCLUSIVE_WINDOW")
    if not path or not Path(path).exists():
        return None
    try:
        return (Path(path).read_text().splitlines() or [""])[0].strip() or "(empty marker)"
    except OSError:
        return "(unreadable marker)"


def controller_default_window() -> str:
    """TRE_METRICS_WINDOW_MS as the controller code of this checkout falls back to when the variable is unset."""
    tre = Path(__file__).resolve().parents[2]
    for sub in ("common", "controller"):
        if str(tre / sub) not in sys.path:
            sys.path.insert(0, str(tre / sub))
    from tre_controller.config import ControllerConfig  # noqa: PLC0415

    return str(ControllerConfig.from_env({}).metrics_window_ms)


def _registry_text(registry_file: Optional[str]) -> str:
    if registry_file:
        return Path(registry_file).read_text()
    return json.loads(kubectl(["get", "cm", REGISTRY_CM, "-o", "json"]))["data"]["registry.yaml"]


def _problems(want: dict[str, str], registry_file: Optional[str]) -> list[str]:
    window = want.get(WINDOW_KEY)
    how = ""
    if window is None:
        try:
            window = controller_default_window()
        except Exception as exc:  # noqa: BLE001 - any failure to read the code default refuses
            return [f"FILE does not name {WINDOW_KEY} (applying unsets it) and the controller default cannot be read: {exc}"]
        how = f" (not in FILE: applying unsets it, the controller falls back to its code default {window})"
    try:
        return [m + how for m in registry_window_errors(_registry_text(registry_file), window)]
    except EnvError as exc:
        return [f"registry unreadable: {exc}"]


def env_patch(resource_version: str, p: dict) -> dict:
    """The strategic-merge patch for plan ``p``: containers and env entries merge by name, ``$patch: delete`` removes an entry,
    and the resourceVersion makes the API server refuse the patch when the Deployment changed since it was read."""
    entries = [{"name": k, "value": v} for k, v in sorted(p["set"].items())]
    entries += [{"name": k, "$patch": "delete"} for k in p["unset"]]
    return {"metadata": {"resourceVersion": resource_version},
            "spec": {"template": {"spec": {"containers": [{"name": CONTAINER, "env": entries}]}}}}


def _print_plan(p: dict, unmanaged: list[str]) -> None:
    for k, v in p["set"].items():
        print(f"set    {k}={v}")
    for k in p["unset"]:
        print(f"unset  {k}")
    if unmanaged:
        print(f"info   not touched (valueFrom): {', '.join(unmanaged)}")
    print(f"{len(p['set'])} to set, {len(p['unset'])} to unset, {len(p['same'])} already equal")


def cmd_snapshot(a) -> int:
    live, other = live_env()
    text = render_env_file(live, f"live env of {NS}/{DEPLOY} container {CONTAINER} (controller_env.py snapshot)")
    if a.out:
        Path(a.out).write_text(text)
    else:
        sys.stdout.write(text)
    if other:
        print(f"# not written (valueFrom): {', '.join(other)}", file=sys.stderr)
    return 0


def cmd_diff(a) -> int:
    want = parse_env_file(Path(a.file).read_text())
    live, other = live_env()
    p = plan(live, {k: v for k, v in want.items() if k not in other})
    _print_plan(p, other)
    probs = _problems(want, a.registry_file)
    for m in probs:
        print(f"REFUSED {m}")
    if probs:
        return 2
    return 1 if (p["set"] or p["unset"]) else 0


def cmd_apply(a) -> int:
    want = parse_env_file(Path(a.file).read_text())
    probs = _problems(want, a.registry_file)
    marker = exclusive_window()
    if marker:
        probs.append(f"an exclusive window is active: {marker}")
    act = runner_active()
    if act:
        probs.append(f"a campaign / arm runner or driver is active (pid {', '.join(act)})")
    if not os.environ.get("TRE_EXCLUSIVE_WINDOW_FILE") and not os.environ.get("CAMPAIGN_DATA_ROOT"):
        print("info   exclusive-window marker not checked (set TRE_EXCLUSIVE_WINDOW_FILE or CAMPAIGN_DATA_ROOT)")
    if probs:
        for m in probs:
            print(f"REFUSED {m}")
        return 2
    deploy = json.loads(kubectl(["get", "deploy", DEPLOY, "-o", "json"]))
    rv = deploy["metadata"]["resourceVersion"]
    live, other = live_env(deploy)
    p = plan(live, {k: v for k, v in want.items() if k not in other})
    _print_plan(p, other)
    if not p["set"] and not p["unset"]:
        print("nothing to do")
        return 0
    bdir = Path(a.backup_dir)
    bdir.mkdir(parents=True, exist_ok=True)
    (bdir / "controller-env.before.env").write_text(render_env_file(live, f"env before controller_env.py apply {a.file}"))
    (bdir / "deploy.before.json").write_text(json.dumps(deploy, indent=2) + "\n")
    (bdir / "rollback.sh").write_text(f"#!/usr/bin/env bash\nexec python3 {Path(__file__).resolve()} apply {bdir}/controller-env.before.env "
                                      f"--backup-dir {bdir}/rollback\n")
    os.chmod(bdir / "rollback.sh", 0o755)
    kubectl(["patch", f"deploy/{DEPLOY}", "--type", "strategic", "-p", json.dumps(env_patch(rv, p))])
    kubectl(["rollout", "status", f"deploy/{DEPLOY}", "--timeout=300s"], timeout=330)
    after, _ = live_env()
    left = plan(after, {k: v for k, v in want.items() if k not in other})
    if left["set"] or left["unset"]:
        print(f"WARNING live env still differs after the rollout: {left}")
        return 1
    print(f"applied; rollback: {bdir}/rollback.sh")
    return 0


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("snapshot")
    s.add_argument("--out")
    s.set_defaults(fn=cmd_snapshot)
    d = sub.add_parser("diff")
    d.add_argument("file")
    d.add_argument("--registry-file")
    d.set_defaults(fn=cmd_diff)
    p = sub.add_parser("apply")
    p.add_argument("file")
    p.add_argument("--backup-dir", required=True)
    p.add_argument("--registry-file")
    p.set_defaults(fn=cmd_apply)
    a = ap.parse_args(argv)
    try:
        return a.fn(a)
    except EnvError as exc:
        print(f"controller_env: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
