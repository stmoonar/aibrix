#!/usr/bin/env python3
"""Set / remove one STRUCTURAL key of a registry file and validate the result with
the release's own parser (tre_common.registry) before it may reach the ConfigMap.

  registry_set.py --in live.yaml --out new.yaml --set service_manager.test_hooks=true
  registry_set.py --in live.yaml --out new.yaml --unset service_manager.test_hooks

For keys without a console path (e.g. service_manager.test_hooks). Tunables
(trs / slo / alt_thresholds / replica bounds) go through console PUT /api/params.
The value is parsed as YAML (true, 30, [a, b], ...). Run with
PYTHONPATH=<tre>/common:<tre>/deploy of the release that will read the file.
Exit 1 (nothing written) when the parser rejects the result.
"""
from __future__ import annotations

import argparse
import os
import sys
import tempfile

import yaml


def main(argv) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--in", dest="inp", required=True)
    parser.add_argument("--out", required=True)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--set", metavar="PATH=VALUE")
    group.add_argument("--unset", metavar="PATH")
    args = parser.parse_args(argv)

    data = yaml.safe_load(open(args.inp, encoding="utf-8")) or {}
    path = (args.set.split("=", 1)[0] if args.set else args.unset).split(".")
    node = data
    for part in path[:-1]:
        node = node.setdefault(part, {})
        if not isinstance(node, dict):
            raise SystemExit(f"{'.'.join(path)}: {part} is not a mapping")
    if args.set:
        node[path[-1]] = yaml.safe_load(args.set.split("=", 1)[1])
    else:
        node.pop(path[-1], None)
    text = yaml.safe_dump(data, sort_keys=False, allow_unicode=True)

    from tre_common.registry import load_registry  # the release's parser

    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False, encoding="utf-8") as fh:
        fh.write(text)
        tmp = fh.name
    try:
        registry = load_registry(tmp)
        errors = list(registry.validate())
    except Exception as exc:  # noqa: BLE001 - reported, nothing written
        print(f"REJECTED by {load_registry.__module__}: {exc}", file=sys.stderr)
        return 1
    finally:
        os.unlink(tmp)
    if errors:
        for error in errors:
            print(f"REJECTED: {error}", file=sys.stderr)
        return 1
    with open(args.out, "w", encoding="utf-8") as fh:
        fh.write(text)
    shown = args.set if args.set else f"{args.unset} (removed)"
    print(f"OK {shown}: validated, written to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
