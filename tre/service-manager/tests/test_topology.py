import pytest

from tre_common.registry import ClusterTopology, NodeSpec
from tre_sm.allocator.topology import (
    GPU_IDS_ANNOTATION,
    STATE_ANNOTATION,
    K8sPodSnapshot,
    pod_records_from_snapshots,
)
from tre_sm.state.reconcile import PodRecord


def topology():
    return ClusterTopology(nodes=(NodeSpec(name="node-a", gpus=4, two_gpu_slots=((0, 1), (2, 3))),))


def test_pod_records_from_snapshots_uses_gpu_annotation_over_cuda_env():
    records = pod_records_from_snapshots(
        topology(),
        [
            K8sPodSnapshot(
                name="serve-a",
                model="dsqwen-14b",
                node="node-a",
                env={"CUDA_VISIBLE_DEVICES": "0,1"},
                annotations={GPU_IDS_ANNOTATION: "2,3", STATE_ANNOTATION: "hidden"},
            )
        ],
    )

    assert records == [
        PodRecord(
            serve_id="serve-a",
            model="dsqwen-14b",
            node="node-a",
            cuda_visible_devices="2,3",
            state="hidden",
        )
    ]


def test_pod_records_from_snapshots_falls_back_to_cuda_env_without_gpu_annotation():
    records = pod_records_from_snapshots(
        topology(),
        [
            K8sPodSnapshot(
                name="serve-a",
                model="dsqwen-14b",
                node="node-a",
                env={"CUDA_VISIBLE_DEVICES": "0,1"},
            )
        ],
    )

    assert records == [
        PodRecord(
            serve_id="serve-a",
            model="dsqwen-14b",
            node="node-a",
            cuda_visible_devices="0,1",
            state="awake",
        )
    ]


def test_pod_records_from_snapshots_rejects_unknown_node_or_invalid_gpu_slot():
    with pytest.raises(ValueError, match="unknown node"):
        pod_records_from_snapshots(
            topology(),
            [
                K8sPodSnapshot(
                    name="serve-a",
                    model="dsqwen-7b",
                    node="node-missing",
                    env={"CUDA_VISIBLE_DEVICES": "0"},
                )
            ],
        )

    with pytest.raises(ValueError, match="invalid slot"):
        pod_records_from_snapshots(
            topology(),
            [
                K8sPodSnapshot(
                    name="serve-a",
                    model="dsqwen-14b",
                    node="node-a",
                    env={"CUDA_VISIBLE_DEVICES": "0,1"},
                    annotations={GPU_IDS_ANNOTATION: "0,2"},
                )
            ],
        )
