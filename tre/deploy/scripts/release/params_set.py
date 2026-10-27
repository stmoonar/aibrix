#!/usr/bin/env python3
"""Edit the per-model TUNABLES of the live registry (ConfigMap tre-v2-registry) and
restart-to-apply. Replaces the removed console's PUT /api/params and
POST /api/ops/controller/restart (2026-10-20).

  params_set.py --show [MODEL]                     editable fields: value and bounds
  params_set.py --set dsqwen-7b.trs.theta_m=800 \\
                --set dsqwen-7b.max_awake_replicas=4   dry run: validate, print the diff
  params_set.py --set ... --apply                  replace the ConfigMap (resourceVersion lock)
  params_set.py --set ... --apply --restart        ... then restart controller + SM
  params_set.py --restart [--reason TEXT]          restart-to-apply the current ConfigMap
  params_set.py --status                           ConfigMap hash vs the hash each Deployment
                                                   was last restarted with (pending restart?)
  params_set.py --registry-file F --set ...        validate against a local file (no cluster)

Editable (deploy/scripts/registry_params.py): <model>.trs.<f> (not ema_tau_ms),
<model>.slo.{ttft_p95_ms,tpot_p95_ms,e2e_p95_ms,ttft_idle_c_ms,ttft_idle_b_ms_per_token},
<model>.alt_thresholds.<signal>.{theta,direction,delta_crit,delta_high,basis.definition,
basis.window_ms} (an alt arm starts only on a matching basis stamp),
<model>.{min_replicas,max_awake_replicas}. Several --set go out in ONE ConfigMap write, so a
recalibration (theta, lambda, tau, w_p and c/b) is published atomically. max_replicas (the
GPU layout) is structural. VALUE is parsed as YAML. The edit
is checked against the bounds and cross-field rules, re-parsed with tre_common.registry,
and rejected if Registry.validate() reports an error the live registry does not have.
The ConfigMap is re-rendered (yaml.safe_dump): comments of the live text are not kept.
--apply first saves the live ConfigMap (YAML, as read) under --backup-dir.
Run it from a checkout at the same sha as the live images (its parser and bounds are
the ones the controller / SM will load).

Structural keys (images, engine args, placement / gateway / service_manager): use
merge_live_registry.py or registry_set.py instead. Restart-to-apply restarts the
service-manager, then the controller (both read the registry only at start), stamping
the pod template with kubectl.kubernetes.io/restartedAt and tre.dev/params-hash
(sha256 of registry.yaml, first 16 hex) and waiting for each rollout.

Environment: TRE_NS (tre-v2), REGISTRY_CM (tre-v2-registry), CONTROLLER_DEPLOY
(tre-v2-controller), SM_DEPLOY (tre-v2-service-manager), KUBECTL (kubectl),
ROLLOUT_TIMEOUT (300s), TRE_BACKUP_ROOT (default --backup-dir = $TRE_BACKUP_ROOT/params-set;
without it --apply needs --backup-dir). Exit 0 ok / dry run, 1 rejected, 2 cluster error or conflict.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml

_TRE = Path(__file__).resolve().parents[3]
for _p in (_TRE / "common", _TRE / "deploy"):  # after PYTHONPATH: an explicit release wins
    if str(_p) not in sys.path:
        sys.path.append(str(_p))

from scripts import registry_params as rp  # noqa: E402

HASH_ANNOTATION = "tre.dev/params-hash"
DATA_KEY = "registry.yaml"


class ClusterError(RuntimeError):
    pass


def _env(name: str, default: str) -> str:
    return os.environ.get(name) or default


def kubectl(args: list[str], *, stdin: str | None = None) -> str:
    cmd = [_env("KUBECTL", "kubectl"), "-n", _env("TRE_NS", "tre-v2"), *args]
    proc = subprocess.run(cmd, input=stdin, capture_output=True, text=True)
    if proc.returncode != 0:
        raise ClusterError(f"{' '.join(cmd[:5])} ...: {proc.stderr.strip()[:500]}")
    return proc.stdout


def read_configmap() -> dict:
    return json.loads(kubectl(["get", "configmap", _env("REGISTRY_CM", "tre-v2-registry"), "-o", "json"]))


#: Server-owned metadata dropped from the PUT body; everything else (labels, annotations,
#: ownerReferences, finalizers, resourceVersion, ...) is sent back unchanged.
_SERVER_METADATA = ("uid", "managedFields", "creationTimestamp")


def replacement(cm: dict, new_text: str) -> dict:
    """The ConfigMap object to PUT: the metadata that was read (minus server-owned fields),
    the other data keys, the edited registry.yaml, and the resourceVersion that was read
    (the API server answers 409 Conflict if the ConfigMap changed since)."""
    meta = {k: v for k, v in (cm.get("metadata") or {}).items() if k not in _SERVER_METADATA}
    out = {k: v for k, v in cm.items() if k not in ("metadata", "data")}
    out.update({"apiVersion": "v1", "kind": "ConfigMap", "metadata": meta,
                "data": {**(cm.get("data") or {}), DATA_KEY: new_text}})
    return out


def backup(cm: dict, backup_dir: str) -> Path:
    """Save the live ConfigMap (as read) before it is replaced; returns the file."""
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    rv = (cm.get("metadata") or {}).get("resourceVersion", "unknown")
    path = Path(backup_dir) / f"{cm['metadata']['name']}-{stamp}-rv{rv}.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(cm, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return path


def deployments() -> list[str]:
    # SM first, then the controller (release order: SM before controller).
    return [_env("SM_DEPLOY", "tre-v2-service-manager"), _env("CONTROLLER_DEPLOY", "tre-v2-controller")]


def template_annotations(name: str) -> dict:
    d = json.loads(kubectl(["get", "deployment", name, "-o", "json"]))
    return ((((d.get("spec") or {}).get("template") or {}).get("metadata") or {}).get("annotations")) or {}


def status() -> int:
    cm = read_configmap()
    current = rp.params_hash((cm.get("data") or {}).get(DATA_KEY, ""))
    print(f"ConfigMap {cm['metadata']['name']}: resourceVersion {cm['metadata'].get('resourceVersion')}, "
          f"params-hash {current}")
    pending = []
    for name in deployments():
        ann = template_annotations(name)
        applied = ann.get(HASH_ANNOTATION)
        print(f"  {name}: params-hash {applied or '-'}  restartedAt {ann.get('kubectl.kubernetes.io/restartedAt', '-')}"
              f"  by {ann.get('tre.dev/restarted-by', '-')}")
        if applied != current:
            pending.append(name)
    print("in sync" if not pending else f"PENDING restart-to-apply: {', '.join(pending)} "
          "(or restarted without this script)")
    return 0


def restart(reason: str) -> int:
    text = (read_configmap().get("data") or {}).get(DATA_KEY, "")
    now = datetime.now(timezone.utc).isoformat()
    patch = {"spec": {"template": {"metadata": {"annotations": {
        "kubectl.kubernetes.io/restartedAt": now,
        "tre.dev/restarted-by": "params_set.py",
        "tre.dev/restart-reason": reason[:200],
        HASH_ANNOTATION: rp.params_hash(text),
    }}}}}
    timeout = _env("ROLLOUT_TIMEOUT", "300s")
    names = deployments()
    done: list[str] = []
    for name in names:
        print(f"restart {name} (params-hash {rp.params_hash(text)})")
        try:
            kubectl(["patch", "deployment", name, "--type", "strategic", "-p", json.dumps(patch)])
            print(kubectl(["rollout", "status", f"deployment/{name}", f"--timeout={timeout}"]).rstrip())
        except ClusterError as exc:
            print(f"ERROR: restart of {name} failed: {exc}", file=sys.stderr)
            print(f"  restarted and rolled out: {', '.join(done) or 'none'}", file=sys.stderr)
            print(f"  failed or not restarted:  {', '.join(names[len(done):])}", file=sys.stderr)
            print("  fix the cause, then rerun: params_set.py --restart", file=sys.stderr)
            return 2
        done.append(name)
    return status()


def show(text: str, model: str | None) -> int:
    for name, view in rp.build_view(text).items():
        if model and name != model:
            continue
        print(name)
        for field, spec in view["editable"].items():
            if spec.get("type") == "enum":
                print(f"  {field} = {spec['value']}   choices {spec['choices']}")
                continue
            lo = "(" if spec["min_exclusive"] else "["
            hi = ")" if spec["max_exclusive"] else "]"
            print(f"  {field} = {spec['value']}   {spec['type']} {lo}{spec['min']}, {spec['max']}{hi}")
        print("  constraints: " + "; ".join(view["constraints"]))
    return 0


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--set", action="append", default=[], metavar="MODEL.PATH=VALUE")
    parser.add_argument("--apply", action="store_true", help="write the ConfigMap (default: dry run)")
    parser.add_argument("--restart", action="store_true", help="restart-to-apply: SM, then controller")
    parser.add_argument("--reason", default="params_set", help="tre.dev/restart-reason annotation")
    parser.add_argument("--status", action="store_true")
    parser.add_argument("--show", nargs="?", const="", metavar="MODEL")
    parser.add_argument("--registry-file", help="validate against this file; no cluster access")
    root = os.environ.get("TRE_BACKUP_ROOT")
    parser.add_argument("--backup-dir", default=str(Path(root) / "params-set") if root else None,
                        help="--apply saves the live ConfigMap here first (default $TRE_BACKUP_ROOT/params-set)")
    args = parser.parse_args(argv)

    if args.registry_file and (args.apply or args.restart or args.status):
        parser.error("--registry-file is offline: no --apply / --restart / --status")
    if args.set and args.restart and not args.apply:
        parser.error("--set with --restart needs --apply")
    if args.apply and not args.backup_dir:
        parser.error("--apply needs --backup-dir (or TRE_BACKUP_ROOT)")
    if not (args.set or args.restart or args.status or args.show is not None):
        parser.error("nothing to do: --set, --show, --status or --restart")

    try:
        if args.status:
            return status()
        cm = None
        if args.registry_file:
            text = Path(args.registry_file).read_text(encoding="utf-8")
        else:
            cm = read_configmap()
            text = (cm.get("data") or {}).get(DATA_KEY, "")
        if args.show is not None:
            return show(text, args.show or None)
        if args.set:
            try:
                names = [m.get("name") for m in rp.load_models(text)]
                new_text = rp.apply_and_validate(text, rp.parse_sets(args.set, names))
            except rp.ParamValidationError as exc:
                for error in exc.errors:
                    print("REJECTED: " + json.dumps(error, ensure_ascii=False), file=sys.stderr)
                return 1
            new_errors = rp.new_validation_errors(text, new_text)
            if new_errors:
                for error in new_errors:
                    print(f"REJECTED by Registry.validate(): {error}", file=sys.stderr)
                return 1
            changes = rp.changed_fields(text, new_text)
            if not changes:
                print("no change (values already set)")
            for model, field, old, new in changes:
                print(f"{model}.{field}: {old} -> {new}")
            if changes:
                print(rp.unified_diff(text, new_text), end="")
            if changes and not args.apply:
                print("dry run: nothing written (add --apply)")
                return 0
            if changes:
                rv = cm["metadata"].get("resourceVersion")
                print(f"backup of the live ConfigMap: {backup(cm, args.backup_dir)}")
                try:
                    kubectl(["replace", "-f", "-"], stdin=json.dumps(replacement(cm, new_text)))
                except ClusterError as exc:
                    if "Conflict" in str(exc) or "has been modified" in str(exc):
                        print(f"CONFLICT: the ConfigMap changed since it was read (resourceVersion {rv}); "
                              "nothing written, re-run", file=sys.stderr)
                        return 2
                    raise
                print(f"ConfigMap replaced (was resourceVersion {rv}); params-hash {rp.params_hash(new_text)}"
                      + ("" if args.restart else "; takes effect after --restart"))
        if args.restart:
            return restart(args.reason)
        return 0
    except ClusterError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
