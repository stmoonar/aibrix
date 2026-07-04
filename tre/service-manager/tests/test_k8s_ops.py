from tre_sm.allocator.slots import Binding, Slot
from tre_sm.allocator.topology import GPU_IDS_ANNOTATION, STATE_ANNOTATION, K8sPodSnapshot
from tre_sm.ops.k8s_ops import MODEL_LABEL, K8sOps


class FakeK8sApi:
    def __init__(self, pods):
        self.pods = pods
        self.patches = []

    def list_namespaced_pod(self, *, namespace, label_selector=None):
        self.last_list = (namespace, label_selector)
        return self.pods

    def patch_namespaced_pod(self, *, name, namespace, body):
        self.patches.append((name, namespace, body))


def pod_dict(name, model, node, cuda, *, phase="Running", annotations=None, deleting=False):
    return {
        "metadata": {
            "name": name,
            "labels": {MODEL_LABEL: model},
            "annotations": annotations or {},
            "deletionTimestamp": "2026-07-04T00:00:00Z" if deleting else None,
        },
        "spec": {
            "nodeName": node,
            "containers": [
                {
                    "name": "vllm",
                    "env": [{"name": "CUDA_VISIBLE_DEVICES", "value": cuda}],
                }
            ],
        },
        "status": {"phase": phase},
    }


class PodListObject:
    def __init__(self, items):
        self.items = items


class SectionObject:
    def __init__(self, payload):
        self._payload = payload

    def to_dict(self):
        return dict(self._payload)


class PodObject:
    def __init__(self):
        self.metadata = SectionObject(
            {
                "name": "serve-a",
                "labels": {MODEL_LABEL: "dsqwen-7b"},
                "annotations": {},
                "deletion_timestamp": None,
            }
        )
        self.spec = SectionObject(
            {
                "node_name": "node-a",
                "containers": [
                    {
                        "name": "vllm",
                        "env": [{"name": "CUDA_VISIBLE_DEVICES", "value": "0"}],
                    }
                ],
            }
        )
        self.status = SectionObject({"phase": "Running"})


def test_k8s_ops_lists_running_pod_snapshots_and_applies_model_selector():
    api = FakeK8sApi(
        [
            pod_dict(
                "serve-a",
                "dsqwen-7b",
                "node-a",
                "0",
                annotations={STATE_ANNOTATION: "sleeping"},
            ),
            pod_dict("serve-b", "dsqwen-7b", "node-a", "1", deleting=True),
            pod_dict("serve-c", "dsqwen-14b", "node-a", "2,3", phase="Pending"),
        ]
    )
    ops = K8sOps(api=api, namespace="tre-v2")

    snapshots = ops.list_pod_snapshots(model="dsqwen-7b")

    assert api.last_list == ("tre-v2", f"{MODEL_LABEL}=dsqwen-7b")
    assert snapshots == [
        K8sPodSnapshot(
            name="serve-a",
            model="dsqwen-7b",
            node="node-a",
            env={"CUDA_VISIBLE_DEVICES": "0"},
            annotations={STATE_ANNOTATION: "sleeping"},
        )
    ]


def test_k8s_ops_accepts_kubernetes_pod_list_objects():
    api = FakeK8sApi([pod_dict("serve-a", "dsqwen-7b", "node-a", "0")])
    api.pods = PodListObject(api.pods)
    ops = K8sOps(api=api, namespace="default")

    assert ops.list_pod_snapshots() == [
        K8sPodSnapshot(
            name="serve-a",
            model="dsqwen-7b",
            node="node-a",
            env={"CUDA_VISIBLE_DEVICES": "0"},
            annotations={},
        )
    ]


def test_k8s_ops_accepts_kubernetes_client_snake_case_objects():
    api = FakeK8sApi(PodListObject([PodObject()]))
    ops = K8sOps(api=api, namespace="default")

    assert ops.list_pod_snapshots() == [
        K8sPodSnapshot(
            name="serve-a",
            model="dsqwen-7b",
            node="node-a",
            env={"CUDA_VISIBLE_DEVICES": "0"},
            annotations={},
        )
    ]


def test_k8s_ops_writes_binding_annotations():
    api = FakeK8sApi([])
    ops = K8sOps(api=api, namespace="tre-v2")

    ops.write_binding_annotations(
        Binding(
            serve_id="serve-a",
            model="dsqwen-14b",
            slot=Slot("node-a", (0, 1)),
            awake=True,
        ),
        state="hidden",
    )

    assert api.patches == [
        (
            "serve-a",
            "tre-v2",
            {
                "metadata": {
                    "annotations": {
                        GPU_IDS_ANNOTATION: "0,1",
                        STATE_ANNOTATION: "hidden",
                    }
                }
            },
        )
    ]
