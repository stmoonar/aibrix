from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

from tre_common.registry import ClusterTopology
from tre_sm.allocator.slots import SlotAllocator
from tre_sm.state.reconcile import POD_STATE_AWAKE, PodRecord


STATE_ANNOTATION = "tre.aibrix.io/state"
GPU_IDS_ANNOTATION = "tre.aibrix.io/gpu-ids"
CUDA_VISIBLE_DEVICES = "CUDA_VISIBLE_DEVICES"


@dataclass(frozen=True)
class K8sPodSnapshot:
    name: str
    model: str
    node: str
    env: Mapping[str, str]
    annotations: Mapping[str, str] = field(default_factory=dict)
    pod_ip: str | None = None


def pod_records_from_snapshots(
    topology: ClusterTopology,
    snapshots: list[K8sPodSnapshot],
) -> list[PodRecord]:
    known_nodes = {node.name for node in topology.nodes}
    records: list[PodRecord] = []
    for snapshot in sorted(snapshots, key=lambda item: item.name):
        if snapshot.node not in known_nodes:
            raise ValueError(f"unknown node for pod {snapshot.name}: {snapshot.node}")
        gpu_ids = snapshot.annotations.get(GPU_IDS_ANNOTATION) or snapshot.env.get(CUDA_VISIBLE_DEVICES)
        if not gpu_ids:
            raise ValueError(f"pod {snapshot.name} missing {GPU_IDS_ANNOTATION} or {CUDA_VISIBLE_DEVICES}")
        records.append(
            PodRecord(
                serve_id=snapshot.name,
                model=snapshot.model,
                node=snapshot.node,
                cuda_visible_devices=gpu_ids,
                state=snapshot.annotations.get(STATE_ANNOTATION, POD_STATE_AWAKE),
            )
        )

    SlotAllocator(topology, [record.to_binding() for record in records])
    return records
