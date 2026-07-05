from tre_common.registry import ClusterTopology, NodeSpec
from tre_sm.allocator.slots import Binding, Slot
from tre_sm.state.reconcile import PodRecord, reconcile_state
from tre_sm.state.store import StateStore


class FakeRedis:
    def __init__(self):
        self.values = {}
        self.hashes = {}

    def get(self, key):
        value = self.values.get(key)
        return None if value is None else str(value).encode("utf-8")

    def set(self, key, value):
        self.values[key] = str(value)

    def delete(self, key):
        self.hashes.pop(key, None)

    def hgetall(self, key):
        return dict(self.hashes.get(key, {}))

    def hset(self, key, mapping):
        bucket = self.hashes.setdefault(key, {})
        for field, value in mapping.items():
            bucket[str(field).encode("utf-8")] = str(value).encode("utf-8")


class FakeK8sClient:
    def __init__(self, pods):
        self._pods = list(pods)

    def list_pods(self):
        return list(self._pods)


def topology():
    return ClusterTopology(
        nodes=(
            NodeSpec(
                name="node-a",
                gpus=4,
                two_gpu_slots=((0, 1), (2, 3)),
                gpu_uuids=("GPU-0", "GPU-1", "GPU-2", "GPU-3"),
            ),
        )
    )


class FakeGpuTruth:
    def __init__(self, used_by_uuid):
        self._used_by_uuid = dict(used_by_uuid)

    def used_mib(self, *, node, gpu_id, gpu_uuid):
        return self._used_by_uuid.get((node, gpu_uuid))


def test_reconcile_prefers_existing_pod_cuda_env_over_stale_store_and_persists():
    store = StateStore(FakeRedis())
    store.save(
        [
            Binding(
                serve_id="serve-a",
                model="dsqwen-14b",
                slot=Slot("node-a", (2, 3)),
                awake=False,
            )
        ],
        expected_version=0,
    )
    k8s = FakeK8sClient(
        [
            PodRecord(
                serve_id="serve-a",
                model="dsqwen-14b",
                node="node-a",
                cuda_visible_devices="0,1",
                state="awake",
            )
        ]
    )

    result = reconcile_state(topology(), store, k8s)

    assert result.version == 2
    assert result.bindings == [
        Binding(
            serve_id="serve-a",
            model="dsqwen-14b",
            slot=Slot("node-a", (0, 1)),
            awake=True,
        )
    ]
    assert store.load().bindings == result.bindings
    assert result.warnings == [
        "serve-a: pod reality overrides persisted binding",
    ]


def test_reconcile_keeps_persisted_binding_when_pod_observation_is_missing():
    store = StateStore(FakeRedis())
    persisted = Binding(
        serve_id="serve-sleeping",
        model="dsqwen-7b",
        slot=Slot("node-a", (0,)),
        awake=False,
    )
    store.save([persisted], expected_version=0)

    result = reconcile_state(topology(), store, FakeK8sClient([]))

    assert result.version == 1
    assert result.bindings == [persisted]
    assert result.allocator.feasible_wake("serve-sleeping") is True
    assert result.warnings == [
        "serve-sleeping: persisted binding has no matching pod observation",
    ]


def test_reconcile_drops_stale_binding_when_replacement_pod_reuses_slot():
    store = StateStore(FakeRedis())
    store.save(
        [
            Binding(
                serve_id="serve-old",
                model="dsqwen-7b",
                slot=Slot("node-a", (0,)),
                awake=True,
            )
        ],
        expected_version=0,
    )
    k8s = FakeK8sClient(
        [
            PodRecord(
                serve_id="serve-new",
                model="dsqwen-7b",
                node="node-a",
                cuda_visible_devices="0",
                state="awake",
            )
        ]
    )

    result = reconcile_state(topology(), store, k8s)

    assert result.version == 2
    assert result.bindings == [
        Binding(
            serve_id="serve-new",
            model="dsqwen-7b",
            slot=Slot("node-a", (0,)),
            awake=True,
        )
    ]
    assert store.load().bindings == result.bindings
    assert result.warnings == [
        "serve-old: dropped stale persisted binding that overlaps pod observation",
    ]


def test_reconcile_auto_sleeps_later_pod_when_two_awake_pods_share_gpu():
    store = StateStore(FakeRedis())
    store.save([], expected_version=0)
    k8s = FakeK8sClient(
        [
            PodRecord(
                serve_id="serve-a",
                model="dsqwen-7b",
                node="node-a",
                cuda_visible_devices="0",
                state="awake",
            ),
            PodRecord(
                serve_id="serve-b",
                model="dsqwen-7b",
                node="node-a",
                cuda_visible_devices="0",
                state="awake",
            ),
        ]
    )

    result = reconcile_state(topology(), store, k8s)

    assert result.bindings == [
        Binding("serve-a", "dsqwen-7b", Slot("node-a", (0,)), awake=True),
        Binding("serve-b", "dsqwen-7b", Slot("node-a", (0,)), awake=False),
    ]
    assert result.warnings == [
        "serve-b: auto-slept to preserve single awake GPU invariant on node-a/0",
    ]


def test_reconcile_warns_sleep_leak_when_sleeping_only_gpu_has_high_truth_usage():
    store = StateStore(FakeRedis())
    store.save([], expected_version=0)
    k8s = FakeK8sClient(
        [
            PodRecord(
                serve_id="serve-sleeping",
                model="dsqwen-7b",
                node="node-a",
                cuda_visible_devices="2",
                state="sleeping",
            )
        ]
    )

    result = reconcile_state(
        topology(),
        store,
        k8s,
        gpu_truth=FakeGpuTruth({("node-a", "GPU-2"): 24000}),
        sleep_leak_used_mib=8192,
    )

    assert result.bindings == [Binding("serve-sleeping", "dsqwen-7b", Slot("node-a", (2,)), awake=False)]
    assert result.warnings == [
        "sleep_leak:serve-sleeping: node-a/GPU-2 used_mib=24000 threshold_mib=8192",
    ]


def test_reconcile_does_not_warn_sleep_leak_when_gpu_has_awake_binding():
    store = StateStore(FakeRedis())
    store.save([], expected_version=0)
    k8s = FakeK8sClient(
        [
            PodRecord("serve-awake", "dsqwen-7b", "node-a", "2", state="awake"),
            PodRecord("serve-sleeping", "dsllama-8b", "node-a", "2", state="sleeping"),
        ]
    )

    result = reconcile_state(
        topology(),
        store,
        k8s,
        gpu_truth=FakeGpuTruth({("node-a", "GPU-2"): 39000}),
        sleep_leak_used_mib=8192,
    )

    assert all(not warning.startswith("sleep_leak:") for warning in result.warnings)
