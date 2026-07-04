from __future__ import annotations

from tre_common.registry import ClusterTopology, NodeSpec
from tre_sm.allocator.topology import K8sPodSnapshot
from tre_sm.server import K8sPodClientFromOps
from tre_sm.state.reconcile import PodRecord


class FakeOps:
    def list_pod_snapshots(self):
        return [
            K8sPodSnapshot(
                name="serve-a",
                model="dsqwen-7b",
                node="nscc-ds-4a100-node9",
                env={"CUDA_VISIBLE_DEVICES": "0"},
            )
        ]


def test_k8s_pod_client_from_ops_converts_snapshots_to_reconcile_records() -> None:
    topology = ClusterTopology(
        nodes=(NodeSpec(name="nscc-ds-4a100-node9", gpus=4, two_gpu_slots=((0, 1), (2, 3))),)
    )
    client = K8sPodClientFromOps(topology, FakeOps())

    assert client.list_pods() == [
        PodRecord(
            serve_id="serve-a",
            model="dsqwen-7b",
            node="nscc-ds-4a100-node9",
            cuda_visible_devices="0",
        )
    ]
