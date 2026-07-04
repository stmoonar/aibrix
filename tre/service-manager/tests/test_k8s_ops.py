from tre_common.registry import ClusterTopology, ModelSpec, NodeSpec, Registry, SloSpec, TrsParams
from tre_sm.allocator.slots import Binding, Slot
from tre_sm.allocator.topology import GPU_IDS_ANNOTATION, STATE_ANNOTATION, K8sPodSnapshot
from tre_sm.ops.k8s_ops import MODEL_LABEL, K8sOps


class FakeK8sApi:
    def __init__(self, pods):
        self.pods = pods
        self.patches = []
        self.deleted_deployments = []
        self.created_deployments = []

    def list_namespaced_pod(self, *, namespace, label_selector=None):
        self.last_list = (namespace, label_selector)
        return self.pods

    def patch_namespaced_pod(self, *, name, namespace, body):
        self.patches.append((name, namespace, body))

    def delete_namespaced_deployment(self, *, name, namespace):
        self.deleted_deployments.append((name, namespace))

    def create_namespaced_deployment(self, *, namespace, body):
        self.created_deployments.append((namespace, body))


def pod_dict(name, model, node, cuda, *, phase="Running", annotations=None, labels=None, deleting=False):
    return {
        "metadata": {
            "name": name,
            "labels": {MODEL_LABEL: model, **(labels or {})},
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
        "status": {"phase": phase, "podIP": "10.0.0.9"},
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
            pod_ip="10.0.0.9",
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
            pod_ip="10.0.0.9",
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


def test_k8s_ops_uses_gpu_id_label_as_annotation_fallback():
    api = FakeK8sApi(
        [
            pod_dict(
                "serve-a",
                "dsqwen-7b",
                "node-a",
                "0",
                labels={GPU_IDS_ANNOTATION: "2"},
            )
        ]
    )
    ops = K8sOps(api=api, namespace="default")

    assert ops.list_pod_snapshots() == [
        K8sPodSnapshot(
            name="serve-a",
            model="dsqwen-7b",
            node="node-a",
            env={"CUDA_VISIBLE_DEVICES": "0"},
            annotations={GPU_IDS_ANNOTATION: "2"},
            pod_ip="10.0.0.9",
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
                    },
                    "labels": {"tre.aibrix.io/routable": "false"},
                }
            },
        )
    ]


def test_k8s_ops_marks_awake_pods_routable():
    api = FakeK8sApi([])
    ops = K8sOps(api=api, namespace="tre-v2")

    ops.write_binding_annotations(
        Binding(
            serve_id="serve-a",
            model="dsqwen-7b",
            slot=Slot("node-a", (0,)),
            awake=True,
        ),
        state="awake",
    )

    assert api.patches[0][2]["metadata"]["labels"] == {"tre.aibrix.io/routable": "true"}


def test_k8s_ops_deletes_and_creates_model_deployments_from_manifest_template():
    api = FakeK8sApi([])
    ops = K8sOps(api=api, namespace="default", registry=registry())

    old = Binding("serve-old", "m1", Slot("node-a", (0,)), awake=True)
    created_name = ops.create_model_deployment("m1", Slot("node-a", (1,)))
    ops.delete_model_deployment(old)

    assert created_name == "m1-node-a-gpu-1"
    assert api.deleted_deployments == [("m1-node-a-gpu-0", "default")]
    [(namespace, body)] = api.created_deployments
    assert namespace == "default"
    assert body["metadata"]["name"] == "m1-node-a-gpu-1"
    assert body["spec"]["template"]["spec"]["nodeName"] == "node-a"
    container = body["spec"]["template"]["spec"]["containers"][0]
    assert {"name": "NVIDIA_VISIBLE_DEVICES", "value": "GPU-a1"} in container["env"]
    assert "nvidia.com/gpu" not in str(container.get("resources", {}))


def test_k8s_ops_wait_pod_ready_resolves_pod_by_created_deployment_app_label():
    api = FakeK8sApi(
        [
            pod_dict(
                "m1-node-a-gpu-1-rs-pod",
                "m1",
                "node-a",
                "0",
                labels={"app": "m1-node-a-gpu-1"},
                annotations={GPU_IDS_ANNOTATION: "1"},
            )
        ]
    )
    ops = K8sOps(api=api, namespace="default", registry=registry())

    snapshot = ops.wait_pod_ready("m1-node-a-gpu-1", timeout_s=0.01, interval_s=0.001)

    assert api.last_list == ("default", "app=m1-node-a-gpu-1")
    assert snapshot.name == "m1-node-a-gpu-1-rs-pod"
    assert snapshot.annotations[GPU_IDS_ANNOTATION] == "1"


def registry():
    trs = TrsParams(
        w_p=0.04,
        w_d=1.0,
        lambda_wait=2.625,
        qmin=1.0,
        ema_alpha=0.5,
        theta_m=1.0,
        tau_crit=0.8,
        tau_low=1.0,
        tau_high=1.25,
        qsat=4.0,
        epsat=0.05,
        hsat=3,
    )
    slo = SloSpec(ttft_p95_ms=1, tpot_p95_ms=1, e2e_p95_ms=1)
    return Registry(
        ClusterTopology(
            nodes=(NodeSpec("node-a", 4, ((0, 1), (2, 3)), ("GPU-a0", "GPU-a1", "GPU-a2", "GPU-a3")),)
        ),
        [
            ModelSpec(
                name="m1",
                weights_path="/m1",
                tp_size=1,
                min_replicas=0,
                max_replicas=4,
                vllm_image="image",
                slo=slo,
                trs=trs,
            )
        ],
    )
