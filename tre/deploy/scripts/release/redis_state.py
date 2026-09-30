#!/usr/bin/env python3
"""Key-level snapshot / restore of the TRE Redis state, and removal of release keys.

The TRE Redis Deployment has no volume (``/data`` is the container's own layer),
``appendonly no`` and the DEBUG command disabled: copying a ``dump.rdb`` into the pod
and restarting it does NOT restore anything (the restarted container starts on an
empty ``/data``). State is therefore restored key by key (DUMP / RESTORE REPLACE).

  redis_state.py dump <out.json> [--prefix tre:] [--all-ttl]
      every key under the prefix WITHOUT a TTL (the durable state: desired / observed /
      leases / journals / run mode); --all-ttl also keeps keys with a TTL (metrics).
  redis_state.py restore <in.json> [--dry-run]
      RESTORE ... REPLACE every key of the snapshot (a key with a recorded TTL gets it
      again). Stop controller and service-manager first (plan / rollback.sh do).
  redis_state.py delete-keys <key>... [--scan PATTERN]... [--dry-run]
      DEL the keys and every key matching each SCAN pattern.

A backup RDB (BGSAVE copy) is read the same way: start a scratch Redis on it
(``docker run -d --rm -p 127.0.0.1:<port>:6379 -v <dir>:/data redis:7.2-alpine``, the
file named ``dump.rdb``) and ``REDIS_URL=redis://127.0.0.1:<port>/0 redis_state.py dump``.

Environment: REDIS_URL (default redis://<ClusterIP of $TRE_NS/$REDIS_SERVICE>:6379/0),
TRE_NS (tre-v2), REDIS_SERVICE (tre-v2-redis).
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import subprocess
import sys
import time


def client():
    import redis  # type: ignore[import-not-found]

    url = os.environ.get("REDIS_URL")
    if not url:
        ns = os.environ.get("TRE_NS", "tre-v2")
        svc = os.environ.get("REDIS_SERVICE", "tre-v2-redis")
        ip = subprocess.check_output(
            ["kubectl", "-n", ns, "get", "svc", svc, "-o", "jsonpath={.spec.clusterIP}"], text=True
        ).strip()
        url = f"redis://{ip}:6379/0"
    return redis.Redis.from_url(url)


def dump(args) -> int:
    r = client()
    out, skipped = {}, 0
    for key in r.scan_iter(match=args.prefix + "*", count=1000):
        pttl = r.pttl(key)
        if pttl == -2:
            continue  # gone meanwhile
        if pttl > 0 and not args.all_ttl:
            skipped += 1
            continue
        payload = r.dump(key)
        if payload is None:
            continue
        out[key.decode()] = {"pttl": pttl if pttl > 0 else 0, "dump": base64.b64encode(payload).decode()}
    meta = {"taken_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "prefix": args.prefix, "keys": len(out),
            "skipped_with_ttl": skipped}
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump({"meta": meta, "keys": out}, fh, sort_keys=True)
    print(json.dumps(meta))
    return 0


def restore(args) -> int:
    data = json.load(open(args.inp, encoding="utf-8"))
    r = client()
    for key, item in sorted(data["keys"].items()):
        if args.dry_run:
            print(f"[dry-run] RESTORE {key} ttl={item['pttl']}")
            continue
        r.restore(key, int(item["pttl"]), base64.b64decode(item["dump"]), replace=True)
    print(f"restored {len(data['keys'])} keys (snapshot {data['meta']})" + (" [dry-run]" if args.dry_run else ""))
    return 0


def delete_keys(args) -> int:
    r = client()
    keys = [k for k in args.keys]
    for pattern in args.scan:
        keys += [k.decode() for k in r.scan_iter(match=pattern, count=1000)]
    for key in keys:
        if args.dry_run:
            print(f"[dry-run] DEL {key} (exists={r.exists(key)})")
        else:
            print(f"DEL {key} -> {r.delete(key)}")
    return 0


def main(argv) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("dump")
    p.add_argument("out")
    p.add_argument("--prefix", default="tre:")
    p.add_argument("--all-ttl", action="store_true")
    p = sub.add_parser("restore")
    p.add_argument("inp")
    p.add_argument("--dry-run", action="store_true")
    p = sub.add_parser("delete-keys")
    p.add_argument("keys", nargs="*")
    p.add_argument("--scan", action="append", default=[])
    p.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    return {"dump": dump, "restore": restore, "delete-keys": delete_keys}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
