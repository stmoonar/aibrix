#!/usr/bin/env python3
"""Move / restore the awake set through the service-manager API
(``PUT /v2/bindings/<serve_id>/power``). Used by the release waves and rollback.

  awake_ctl.py show
  awake_ctl.py off <node> [--dry-run]       every awake binding on <node>: wake a sleeping
                                            binding of the same model (same GPU count, on GPUs
                                            with no awake binding) on another node first, then
                                            sleep the one on <node> (path scale_down)
  awake_ctl.py restore <sm-state.json> [--dry-run]
                                            awake set = the one recorded in that file
                                            (GET /v2/state saved earlier), matched by binding_id
  awake_ctl.py restore-ids <binding_id>... [--dry-run]
                                            awake set = exactly these binding ids
  awake_ctl.py expect <sm-state.json> <binding_id>...
                                            exit 0 iff the awake set recorded in the file is
                                            exactly these ids (else print the difference, exit 1)

Environment: SM_URL (default: http://<ClusterIP of $TRE_NS/$SM_SERVICE>:$SM_PORT),
TRE_NS (tre-v2), SM_SERVICE (tre-v2-service-manager), SM_PORT (8000).
"""
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request

_SM = None


def sm_url():
    global _SM
    if _SM:
        return _SM
    if os.environ.get("SM_URL"):
        _SM = os.environ["SM_URL"].rstrip("/")
        return _SM
    ns = os.environ.get("TRE_NS", "tre-v2")
    svc = os.environ.get("SM_SERVICE", "tre-v2-service-manager")
    port = os.environ.get("SM_PORT", "8000")
    ip = subprocess.check_output(
        ["kubectl", "-n", ns, "get", "svc", svc, "-o", "jsonpath={.spec.clusterIP}"], text=True
    ).strip()
    _SM = f"http://{ip}:{port}"
    return _SM


def state():
    with urllib.request.urlopen(sm_url() + "/v2/state", timeout=15) as r:
        return json.load(r)["bindings"]


def power(serve_id, awake, dry):
    body = json.dumps({"awake": awake, "sleep_path": "scale_down"}).encode()
    what = f"{'wake ' if awake else 'sleep'} {serve_id}"
    if dry:
        print(f"[dry-run] {what}")
        return
    req = urllib.request.Request(f"{sm_url()}/v2/bindings/{serve_id}/power", data=body, method="PUT",
                                 headers={"Content-Type": "application/json"})
    t = time.time()
    try:
        with urllib.request.urlopen(req, timeout=300) as r:
            code = r.status
            r.read()
    except urllib.error.HTTPError as e:
        print(f"{what}: HTTP {e.code} {e.read().decode()[:300]}")
        raise SystemExit(1)
    print(f"{time.strftime('%H:%M:%S')} {what}: HTTP {code} {time.time() - t:.1f}s", flush=True)


def show(bs):
    for b in sorted(bs, key=lambda x: x["binding_id"]):
        if b["awake"]:
            print("awake", b["binding_id"], "hidden" if b["hidden"] else "")


def off(node, dry):
    bs = state()
    here = [b for b in bs if b["node"] == node and b["awake"]]
    busy = {(c["node"], g) for c in bs if c["awake"] for g in c["gpu_ids"]}
    plan = []
    for b in here:
        cands = [c for c in bs if c["model"] == b["model"] and c["node"] != node and not c["awake"]
                 and len(c["gpu_ids"]) == len(b["gpu_ids"])
                 and not any((c["node"], g) in busy for g in c["gpu_ids"])]
        if not cands:
            raise SystemExit(f"no free sleeping binding of {b['model']} off {node}")
        cands.sort(key=lambda c: (c["node"], c["gpu_ids"]))
        plan.append((cands[0], b))
        busy |= {(cands[0]["node"], g) for g in cands[0]["gpu_ids"]}
    for w, s in plan:
        print(f"plan: wake {w['binding_id']}  then sleep {s['binding_id']}")
    for w, _ in plan:
        power(w["serve_id"], True, dry)
    for _, s in plan:
        power(s["serve_id"], False, dry)


def awake_in_file(path):
    return {b["binding_id"] for b in json.load(open(path))["bindings"] if b["awake"]}


def restore(want, dry):
    """Four passes, so a wanted binding whose GPUs hold an unwanted awake one still
    comes up and no model drops to zero awake replicas: (1) wake wanted bindings on free
    GPUs; (2) sleep unwanted awake bindings whose model has another awake replica;
    (3) wake the remaining wanted: an occupant of its GPUs is slept first, and an occupant
    that is its model's last awake replica is first moved to a free slot (a sleeping binding
    of that model on GPUs that are free and in no wanted binding), so a swap (A on B's
    slot, B on A's) resolves; no such slot = exit 3 before that step; (4) sleep the
    remaining unwanted. Never two awake on one GPU, never a model at zero awake."""
    bs = state()
    unknown = sorted(want - {b["binding_id"] for b in bs})
    if unknown:
        raise SystemExit(f"unknown binding ids (not in /v2/state): {unknown}")
    if dry:
        for b in bs:
            if b["binding_id"] in want and not b["awake"]:
                power(b["serve_id"], True, dry)
        for b in bs:
            if b["awake"] and b["binding_id"] not in want:
                power(b["serve_id"], False, dry)
        print("target awake:", sorted(want))
        return

    def busy(bs):
        return {(c["node"], g) for c in bs if c["awake"] for g in c["gpu_ids"]}

    for b in bs:  # 1
        if b["binding_id"] in want and not b["awake"] and not any((b["node"], g) in busy(bs) for g in b["gpu_ids"]):
            power(b["serve_id"], True, dry)
            bs = state()
    for b in list(bs):  # 2
        if b["awake"] and b["binding_id"] not in want:
            others = [c for c in state() if c["model"] == b["model"] and c["awake"] and c["binding_id"] != b["binding_id"]]
            if others:
                power(b["serve_id"], False, dry)
    def gpus(b):
        return {(b["node"], g) for g in b["gpu_ids"]}

    targets = set().union(*(gpus(b) for b in bs if b["binding_id"] in want))
    for w in [b for b in state() if b["binding_id"] in want and not b["awake"]]:  # 3
        cur = state()
        for o in [c for c in cur if c["awake"] and gpus(c) & gpus(w)]:
            if o["binding_id"] in want:
                print(f"wanted bindings overlap: {o['binding_id']} and {w['binding_id']}", file=sys.stderr)
                raise SystemExit(3)
            cur = state()
            if not any(c["model"] == o["model"] and c["awake"] and c["binding_id"] != o["binding_id"] for c in cur):
                free = [c for c in cur if c["model"] == o["model"] and not c["awake"]
                        and not gpus(c) & (busy(cur) | targets)]
                if not free:
                    print(f"cannot wake {w['binding_id']}: its GPUs hold {o['binding_id']}, the last awake "
                          f"{o['model']}, and no free non-target slot of {o['model']} exists", file=sys.stderr)
                    raise SystemExit(3)
                free.sort(key=lambda c: (c["node"], c["gpu_ids"]))
                print(f"relocate: wake {free[0]['binding_id']} (free slot) before sleeping {o['binding_id']}")
                power(free[0]["serve_id"], True, dry)
            power(o["serve_id"], False, dry)
        power(w["serve_id"], True, dry)
    for b in state():  # 4
        if b["awake"] and b["binding_id"] not in want:
            power(b["serve_id"], False, dry)
    print("target awake:", sorted(want))


def main(argv):
    dry = "--dry-run" in argv
    args = [a for a in argv if a != "--dry-run"]
    if not args:
        raise SystemExit(__doc__)
    cmd = args[0]
    if cmd == "show":
        show(state())
        return 0
    if cmd == "expect":
        have = awake_in_file(args[1])
        want = set(args[2:])
        if have == want:
            print("awake set matches:", sorted(want))
            return 0
        print("awake set differs: missing", sorted(want - have), "extra", sorted(have - want))
        return 1
    if cmd == "off":
        off(args[1], dry)
    elif cmd == "restore":
        restore(awake_in_file(args[1]), dry)
    elif cmd == "restore-ids":
        restore(set(args[1:]), dry)
    else:
        raise SystemExit(__doc__)
    if not dry:
        show(state())
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
