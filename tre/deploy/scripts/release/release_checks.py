#!/usr/bin/env python3
"""Read-only checks used by the release plans (deploy/RELEASE-*.md).

  release_checks.py manifests --models-dir DIR --node NODE [--expect-env NAME=VALUE ...]
      Print (stdout, one per line) the model manifest files whose Deployments run on NODE
      (label tre.aibrix.io/node). Exit 1 unless every such Deployment's engine container
      (the first container) carries each expected env var with that exact value.

  release_checks.py wave-checkpoint [--node NODE] [--wait-s S] [--interval-s S]
      The model pods of NODE (all managed pods without --node) have settled: no GPU lease
      in phase "starting" (for NODE), no pod still annotated
      tre.aibrix.io/startup-admitted-uid, no running SM operation, empty wake journal.
      Retried until it passes or --wait-s elapses (exit 1 then, with the blockers).

  release_checks.py gpu-truth --nodes "N1 N2" [--gap-s S]
      tre:gpu_truth:<node> exists with a TTL on every node and its seq advances within
      --gap-s (two reads; each node compared only with itself - node clocks differ).

Environment: TRE_NS (tre-v2), MODEL_NS (default), REDIS_DEPLOY (tre-v2-redis),
SM_URL (default: ClusterIP of $TRE_NS/$SM_SERVICE:$SM_PORT, tre-v2-service-manager:8000).
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import subprocess
import sys
import time
import urllib.request

import yaml

TRE_NS = os.environ.get("TRE_NS", "tre-v2")
MODEL_NS = os.environ.get("MODEL_NS", "default")
REDIS_DEPLOY = os.environ.get("REDIS_DEPLOY", "tre-v2-redis")
NODE_LABEL = "tre.aibrix.io/node"
ADMITTED = "tre.aibrix.io/startup-admitted-uid"
LEASES_KEY = "tre:v2:sm:gpu_leases"      # tre_common.rediskeys.SM_GPU_LEASES_KEY
WAKE_OPS_KEY = "tre:v2:sm:wake_ops"      # tre_common.rediskeys.SM_WAKE_OPS_KEY
TRUTH_PREFIX = "tre:gpu_truth:"          # tre_common.rediskeys.GPU_TRUTH_KEY_PREFIX


def log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def redis(*args: str) -> list[str]:
    out = subprocess.check_output(
        ["kubectl", "-n", TRE_NS, "exec", f"deploy/{REDIS_DEPLOY}", "--", "redis-cli", "--raw", *args], text=True
    )
    return out.splitlines()


def sm_get(path: str):
    base = os.environ.get("SM_URL")
    if not base:
        ip = subprocess.check_output(
            ["kubectl", "-n", TRE_NS, "get", "svc", os.environ.get("SM_SERVICE", "tre-v2-service-manager"),
             "-o", "jsonpath={.spec.clusterIP}"], text=True).strip()
        base = f"http://{ip}:{os.environ.get('SM_PORT', '8000')}"
    with urllib.request.urlopen(base.rstrip("/") + path, timeout=15) as r:
        return json.load(r)


# ---------------------------------------------------------------- manifests


def manifests(args) -> int:
    expect = dict(item.split("=", 1) for item in args.expect_env)
    files, bad = [], []
    for path in sorted(glob.glob(os.path.join(args.models_dir, "*.yaml"))):
        docs = [d for d in yaml.safe_load_all(open(path, encoding="utf-8")) if d]
        deploys = [d for d in docs if d.get("kind") == "Deployment"
                   and (d.get("metadata", {}).get("labels") or {}).get(NODE_LABEL) == args.node]
        if not deploys:
            continue
        files.append(path)
        for d in deploys:
            engine = d["spec"]["template"]["spec"]["containers"][0]
            env = {e["name"]: str(e.get("value")) for e in engine.get("env") or [] if "name" in e}
            for name, value in expect.items():
                if env.get(name) != value:
                    bad.append(f"{os.path.basename(path)}: {d['metadata']['name']} {engine['name']} {name}={env.get(name)!r}, want {value!r}")
    if not files:
        log(f"no Deployment of node {args.node} under {args.models_dir}")
        return 1
    for line in bad:
        log("FAIL " + line)
    if bad:
        return 1
    log(f"OK {len(files)} manifest files for {args.node}; expected env present: {sorted(expect)}")
    print("\n".join(files))
    return 0


# ---------------------------------------------------------------- wave checkpoint


def _blockers(node: str | None) -> list[str]:
    out: list[str] = []
    raw = redis("HGETALL", LEASES_KEY)
    for field, value in zip(raw[0::2], raw[1::2]):
        try:
            lease = json.loads(value)
        except ValueError:
            out.append(f"lease {field}: unreadable {value[:80]!r}")
            continue
        if lease.get("phase") == "starting" and (node is None or lease.get("node") == node):
            out.append(f"starting lease {field} binding={lease.get('binding_id')} (never expires; see plan)")
    selector = "tre.aibrix.io/managed=true" + (f",{NODE_LABEL}={node}" if node else "")
    pods = json.loads(subprocess.check_output(
        ["kubectl", "-n", MODEL_NS, "get", "pods", "-l", selector, "-o", "json"], text=True))["items"]
    for pod in pods:
        annotations = pod["metadata"].get("annotations") or {}
        if annotations.get(ADMITTED):
            out.append(f"pod {pod['metadata']['name']} still annotated {ADMITTED} (startup not converged)")
    for op in sm_get("/v2/operations?limit=200").get("operations", []):
        if op.get("status") == "running":
            out.append(f"running SM operation {op.get('kind')} {op.get('operation_id')} since {op.get('started_at')}")
    wake_ops = redis("HLEN", WAKE_OPS_KEY)
    if wake_ops and wake_ops[0].strip() not in ("", "0"):
        out.append(f"wake journal {WAKE_OPS_KEY} has {wake_ops[0].strip()} entries")
    return out


def wave_checkpoint(args) -> int:
    deadline = time.monotonic() + args.wait_s
    while True:
        blockers = _blockers(args.node)
        if not blockers:
            log(f"OK wave checkpoint ({args.node or 'all nodes'}): no starting lease, no admission annotation, "
                "no running operation, empty wake journal")
            return 0
        if time.monotonic() >= deadline:
            for line in blockers:
                log("BLOCKED " + line)
            return 1
        log(f"{time.strftime('%H:%M:%S')} waiting: {len(blockers)} blocker(s), e.g. {blockers[0]}")
        time.sleep(args.interval_s)


# ---------------------------------------------------------------- gpu-truth


def _truth(node: str) -> tuple[int, int | None]:
    key = TRUTH_PREFIX + node
    ttl = int(redis("TTL", key)[0])
    raw = redis("GET", key)
    seq = json.loads(raw[0]).get("seq") if raw and raw[0] else None
    return ttl, seq


def gpu_truth(args) -> int:
    nodes = args.nodes.split()
    first = {node: _truth(node) for node in nodes}
    time.sleep(args.gap_s)
    ok = True
    for node in nodes:
        (ttl0, seq0), (ttl1, seq1) = first[node], _truth(node)
        fresh = ttl1 > 0 and seq0 is not None and seq1 is not None and seq1 > seq0
        ok &= fresh
        log(f"{'OK ' if fresh else 'FAIL'} {node}: ttl {ttl0}->{ttl1} seq {seq0}->{seq1}"
            + ("" if fresh else "  (Running agent with a frozen seq = NVML died: force-delete that gpu-truth pod)"))
    return 0 if ok else 1


def main(argv) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("manifests")
    p.add_argument("--models-dir", required=True)
    p.add_argument("--node", required=True)
    p.add_argument("--expect-env", action="append", default=[])
    p = sub.add_parser("wave-checkpoint")
    p.add_argument("--node")
    p.add_argument("--wait-s", type=float, default=600)
    p.add_argument("--interval-s", type=float, default=15)
    p = sub.add_parser("gpu-truth")
    p.add_argument("--nodes", required=True)
    p.add_argument("--gap-s", type=float, default=15)
    args = parser.parse_args(argv)
    return {"manifests": manifests, "wave-checkpoint": wave_checkpoint, "gpu-truth": gpu_truth}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
