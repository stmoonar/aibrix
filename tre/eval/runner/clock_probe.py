#!/usr/bin/env python3
"""clock_probe.py <clock_offsets.json> <start|end>: offset of every node's clock vs this host.

CLOCK_NODES = "<k8s node>=<ssh target> ..." ("local" = this host, offset 0). Per node: 3 rounds
of ``t0 = now; remote = ssh <target> date +%s.%N; t1 = now``; the round with the smallest RTT
gives ``offset_s = remote - (t0 + t1) / 2`` (seconds the node is ahead of this host). The file
keeps one block per phase and ``offsets_s`` = the mean of the phases present (what the report
subtracts from that node's timestamps). Read-only; a node that fails is recorded with its error.
"""
from __future__ import annotations

import json
import os
import shlex
import socket
import subprocess
import sys
import time


def probe(target: str, ssh_opts: list[str], rounds: int = 3) -> dict:
    if target == "local":
        return {"offset_s": 0.0, "rtt_s": 0.0, "target": "local"}
    best = None
    errors = []
    for _ in range(rounds):
        t0 = time.time()
        out = subprocess.run(["ssh", *ssh_opts, target, "date +%s.%N"], text=True, capture_output=True, timeout=15)
        t1 = time.time()
        try:
            remote = float(out.stdout.strip())
        except ValueError:
            errors.append((out.stderr or out.stdout).strip()[:120])
            continue
        sample = {"offset_s": round(remote - (t0 + t1) / 2, 4), "rtt_s": round(t1 - t0, 4), "target": target}
        if best is None or sample["rtt_s"] < best["rtt_s"]:
            best = sample
    return best or {"offset_s": None, "rtt_s": None, "target": target, "error": "; ".join(errors)}


def main(path: str, phase: str) -> int:
    pairs = [p.split("=", 1) for p in os.environ.get("CLOCK_NODES", "").split() if "=" in p]
    ssh_opts = shlex.split(os.environ.get("CLOCK_SSH_OPTS", "-o BatchMode=yes -o ConnectTimeout=5"))
    try:
        doc = json.load(open(path))
    except (OSError, ValueError):
        doc = {}
    doc.setdefault("ref_host", socket.gethostname())
    doc["method"] = "ssh date +%s.%N, midpoint of the min-RTT of 3 rounds; offset = node - reference"
    doc[phase] = {"ts": time.time(), "nodes": {node: probe(target, ssh_opts) for node, target in pairs}}
    offsets = {}
    for node in {n for ph in ("start", "end") for n in (doc.get(ph) or {}).get("nodes", {})}:
        vals = [doc[ph]["nodes"][node]["offset_s"] for ph in ("start", "end")
                if doc.get(ph) and node in doc[ph]["nodes"] and doc[ph]["nodes"][node].get("offset_s") is not None]
        if vals:
            offsets[node] = round(sum(vals) / len(vals), 4)
    doc["offsets_s"] = offsets
    json.dump(doc, open(path, "w"), indent=1)
    print("clock offsets", phase, json.dumps({n: v.get("offset_s") for n, v in doc[phase]["nodes"].items()}))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1], sys.argv[2]))
