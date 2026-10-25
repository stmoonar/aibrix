#!/usr/bin/env python3
"""components.py <components.json> <start|end>: where every component runs, and its images.

Lists the pods of TRE_NS, ENVOY_NS, AIBRIX_NS and the model pods (MODEL_NS, MODEL_SELECTOR)
and records per pod its node, phase and per container image + imageID (the pod status, so the
image IDs of every node are covered - not only the docker of this host). Top level:
``component_nodes`` {component: node} (from the last phase recorded; a component with pods on
several nodes, e.g. a DaemonSet, is listed per node as ``<component>@<node>``) and
``image_ids_by_node`` {node: {image: imageID}}; a component seen in an earlier phase and gone
in this one keeps its node. ``baseline_scaler`` {pod, node, phase}: the baseline scaler's pod
when one was running in a recorded phase (the runner records a ``bl_enabled`` phase right after
the shell starts; the scaler is at 0 replicas at ``start`` and ``end``). Read-only.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time

from k8s_names import component_of


def pods(ns: str, selector: str | None = None) -> list[dict]:
    args = ["kubectl", "-n", ns, "get", "pods", "-o", "json"] + (["-l", selector] if selector else [])
    out = subprocess.run(args, text=True, capture_output=True, timeout=30)
    if out.returncode != 0:
        raise RuntimeError(out.stderr.strip()[:200])
    rows = []
    for p in json.loads(out.stdout).get("items", []):
        status = {c["name"]: c for c in p.get("status", {}).get("containerStatuses") or ()}
        rows.append({
            "namespace": ns, "name": p["metadata"]["name"], "component": component_of(p["metadata"]["name"]),
            "node": p["spec"].get("nodeName"), "phase": p.get("status", {}).get("phase"),
            "containers": [{"name": c["name"], "image": c.get("image"),
                            "image_id": (status.get(c["name"]) or {}).get("imageID")}
                           for c in p["spec"].get("containers", [])],
        })
    return rows


def main(path: str, phase: str) -> int:
    e = os.environ.get
    groups = [(e("TRE_NS", "tre-v2"), None), (e("ENVOY_NS", "envoy-gateway-system"), None),
              (e("AIBRIX_NS", "aibrix-system"), None),
              (e("MODEL_NS", "default"), e("MODEL_SELECTOR", "tre.aibrix.io/managed=true"))]
    if e("BL_NS") and e("BL_NS") not in {ns for ns, _ in groups}:
        groups.append((e("BL_NS"), None))
    rows, errors = [], {}
    for ns, sel in groups:
        try:
            rows += [dict(r, model_pod=sel is not None) for r in pods(ns, sel)]
        except Exception as exc:  # noqa: BLE001
            errors[ns] = str(exc)[:200]
    try:
        doc = json.load(open(path))
    except (OSError, ValueError):
        doc = {}
    doc[phase] = {"ts": time.time(), "pods": rows, **({"errors": errors} if errors else {})}
    nodes_of: dict = {}
    images: dict = {}
    for r in rows:
        if not r["model_pod"]:  # model pods: in pods[] and image_ids_by_node, not a component each
            nodes_of.setdefault(r["component"], set()).add(r["node"])
        for c in r["containers"]:
            if r["node"] and c.get("image") and c.get("image_id"):
                images.setdefault(r["node"], {})[c["image"]] = c["image_id"]
    comp = {}
    for name, nodes in sorted(nodes_of.items()):
        nodes = sorted(n for n in nodes if n)
        if len(nodes) == 1:
            comp[name] = nodes[0]
        else:
            comp.update({f"{name}@{n}": n for n in nodes})
    # 2026-10-26: a component recorded in an earlier phase and gone now (the baseline scaler is
    # scaled to 0 before the "end" phase) keeps its node; the current phase wins for the others.
    prev = doc.get("component_nodes") or {}
    comp = {**{k: v for k, v in prev.items() if k.split("@", 1)[0] not in nodes_of}, **comp}
    doc["component_nodes"] = comp
    # the baseline scaler's pod + node (its log / decision timestamps are that node's clock)
    scaler = os.environ.get("BL_DEPLOY", "tre-v2-baseline-scaler")
    for r in rows:
        if r["namespace"] == e("BL_NS", "tre-v2") and r["component"] == component_of(scaler) \
                and r["phase"] in ("Running", "Pending"):
            doc["baseline_scaler"] = {"pod": r["name"], "node": r["node"], "phase": phase, "pod_phase": r["phase"]}
    merged = doc.get("image_ids_by_node") or {}
    for node, imgs in images.items():
        merged.setdefault(node, {}).update(imgs)
    doc["image_ids_by_node"] = merged
    json.dump(doc, open(path, "w"), indent=1)
    print("components", phase, json.dumps(comp))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1], sys.argv[2]))
