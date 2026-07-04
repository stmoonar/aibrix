from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Iterable

import yaml


GPU_LINE_RE = re.compile(r"^GPU\s+(\d+):.*\(UUID:\s*(GPU-[^)]+)\)")


def parse_nvidia_smi_l(output: str) -> tuple[str, ...]:
    indexed: dict[int, str] = {}
    for line in output.splitlines():
        match = GPU_LINE_RE.match(line.strip())
        if match:
            indexed[int(match.group(1))] = match.group(2)
    if not indexed:
        raise ValueError("no GPU UUIDs found in nvidia-smi -L output")
    expected = list(range(max(indexed) + 1))
    if sorted(indexed) != expected:
        raise ValueError(f"GPU indexes must be contiguous from 0: got {sorted(indexed)}")
    return tuple(indexed[index] for index in expected)


def apply_gpu_uuids(registry: dict, node_uuids: dict[str, tuple[str, ...]]) -> dict:
    updated = dict(registry)
    cluster = dict(updated.get("cluster") or {})
    nodes = []
    for node in cluster.get("nodes", []):
        next_node = dict(node)
        name = str(next_node["name"])
        if name in node_uuids:
            uuids = list(node_uuids[name])
            if len(uuids) != int(next_node["gpus"]):
                raise ValueError(f"{name}: {len(uuids)} UUIDs for {next_node['gpus']} GPUs")
            next_node["gpu_uuids"] = uuids
        nodes.append(next_node)
    missing = set(node_uuids) - {str(node["name"]) for node in nodes}
    if missing:
        raise ValueError(f"unknown registry node(s): {sorted(missing)}")
    cluster["nodes"] = nodes
    updated["cluster"] = cluster
    return updated


def parse_node_output_arg(raw: str) -> tuple[str, Path]:
    if "=" not in raw:
        raise argparse.ArgumentTypeError("node output must be NODE=PATH")
    node, path = raw.split("=", 1)
    if not node:
        raise argparse.ArgumentTypeError("node name must not be empty")
    return node, Path(path)


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry", default="tre/deploy/registry.yaml")
    parser.add_argument("--node-output", action="append", type=parse_node_output_arg, default=[])
    args = parser.parse_args(list(argv) if argv is not None else None)

    registry_path = Path(args.registry)
    registry = yaml.safe_load(registry_path.read_text(encoding="utf-8")) or {}
    node_uuids = {
        node: parse_nvidia_smi_l(path.read_text(encoding="utf-8"))
        for node, path in args.node_output
    }
    updated = apply_gpu_uuids(registry, node_uuids)
    registry_path.write_text(yaml.safe_dump(updated, sort_keys=False), encoding="utf-8")


if __name__ == "__main__":
    main()
