from types import SimpleNamespace

from tre_common.registry import ClusterTopology, NodeSpec
from tre_sm.allocator.slots import Binding, Slot
from tre_sm.api.v2 import ServiceManagerV2
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


class FakeRegistry:
    def __init__(self):
        self._topology = ClusterTopology(
            nodes=(
                NodeSpec(
                    name="nscc-ds-4a100-node9",
                    gpus=4,
                    two_gpu_slots=((0, 1), (2, 3)),
                ),
                NodeSpec(
                    name="nscc-ds-4a100-node10",
                    gpus=4,
                    two_gpu_slots=((0, 1), (2, 3)),
                ),
            )
        )

    def model(self, _model):
        return SimpleNamespace(max_replicas=8, tp_size=1)

    def topology(self):
        return self._topology


class FakeRuntimeOps:
    def __init__(self):
        self.annotations = []

    def write_binding_annotations(self, binding, *, state):
        self.annotations.append((binding.serve_id, state))


def _service(bindings, *, runtime_ops=None):
    store = StateStore(FakeRedis())
    store.save(bindings, expected_version=0)
    return ServiceManagerV2(FakeRegistry(), store, runtime_ops=runtime_ops), store


def _binding(serve_id, model, node, gpu_ids, *, awake, hidden=False):
    return Binding(serve_id, model, Slot(node, gpu_ids), awake=awake, hidden=hidden)


def test_wake_prefers_node_with_fewer_cluster_wide_awake_bindings():
    node9 = "nscc-ds-4a100-node9"
    node10 = "nscc-ds-4a100-node10"
    service, _store = _service(
        [
            _binding("other-node9-gpu-0", "other", node9, (0,), awake=True),
            _binding("other-node10-gpu-0", "other", node10, (0,), awake=True),
            _binding("other-node10-gpu-1", "other", node10, (1,), awake=True),
            _binding("other-node10-gpu-2", "other", node10, (2,), awake=True),
            _binding("target-node10-gpu-0", "target", node10, (0,), awake=False),
            _binding("target-node10-gpu-3", "target", node10, (3,), awake=False),
            _binding("target-node9-gpu-1", "target", node9, (1,), awake=False),
            _binding("target-node9-gpu-2", "target", node9, (2,), awake=False),
        ]
    )

    result = service.put_model_target("target", wake_replicas=1)

    assert result["actions"] == [{"action": "wake", "serve_id": "target-node9-gpu-1"}]


def test_wake_equal_node_counts_use_natural_serve_id_order():
    service, _store = _service(
        [
            _binding(
                "target-nscc-ds-4a100-node10-gpu-0",
                "target",
                "nscc-ds-4a100-node10",
                (0,),
                awake=False,
            ),
            _binding(
                "target-nscc-ds-4a100-node9-gpu-0",
                "target",
                "nscc-ds-4a100-node9",
                (0,),
                awake=False,
            ),
        ]
    )

    result = service.put_model_target("target", wake_replicas=1)

    assert result["actions"] == [
        {"action": "wake", "serve_id": "target-nscc-ds-4a100-node9-gpu-0"}
    ]


def test_wake_skips_infeasible_candidate_on_least_loaded_node():
    node9 = "nscc-ds-4a100-node9"
    node10 = "nscc-ds-4a100-node10"
    service, _store = _service(
        [
            _binding("other-node9-gpu-0", "other", node9, (0,), awake=True),
            _binding("other-node10-gpu-0", "other", node10, (0,), awake=True),
            _binding("other-node10-gpu-1", "other", node10, (1,), awake=True),
            _binding("target-node9-gpu-0", "target", node9, (0,), awake=False),
            _binding("target-node10-gpu-2", "target", node10, (2,), awake=False),
        ]
    )

    result = service.put_model_target("target", wake_replicas=1)

    assert result["actions"] == [{"action": "wake", "serve_id": "target-node10-gpu-2"}]


def test_state_store_loads_serve_ids_in_natural_order():
    store = StateStore(FakeRedis())
    store.save(
        [
            _binding("x-node10-a", "target", "nscc-ds-4a100-node10", (0,), awake=False),
            _binding("x-node9-b", "target", "nscc-ds-4a100-node9", (1,), awake=False),
            _binding("x-node9-a", "target", "nscc-ds-4a100-node9", (0,), awake=False),
        ],
        expected_version=0,
    )

    assert [binding.serve_id for binding in store.load().bindings] == [
        "x-node9-a",
        "x-node9-b",
        "x-node10-a",
    ]


def test_tp2_wake_uses_same_least_loaded_node_ordering():
    node9 = "nscc-ds-4a100-node9"
    node10 = "nscc-ds-4a100-node10"
    service, _store = _service(
        [
            _binding("other-node9-gpu-0", "other", node9, (0,), awake=True),
            _binding("other-node10-gpu-0", "other", node10, (0,), awake=True),
            _binding("other-node10-gpu-1", "other", node10, (1,), awake=True),
            _binding("tp2-node10-gpu-2-3", "tp2", node10, (2, 3), awake=False),
            _binding("tp2-node9-gpu-2-3", "tp2", node9, (2, 3), awake=False),
        ]
    )

    result = service.put_model_target("tp2", wake_replicas=1)

    assert result["actions"] == [{"action": "wake", "serve_id": "tp2-node9-gpu-2-3"}]


def test_unhide_sleeping_binding_does_not_require_wake_feasibility():
    node9 = "nscc-ds-4a100-node9"
    runtime_ops = FakeRuntimeOps()
    service, store = _service(
        [
            _binding("awake-other", "other", node9, (0,), awake=True),
            _binding("sleeping-hidden", "target", node9, (0,), awake=False, hidden=True),
        ],
        runtime_ops=runtime_ops,
    )

    result = service.put_model_routable("target", hidden_pods=[])

    assert result["actions"] == [{"action": "unhide", "serve_id": "sleeping-hidden"}]
    assert runtime_ops.annotations == [("sleeping-hidden", "sleeping")]
    binding = next(item for item in store.load().bindings if item.serve_id == "sleeping-hidden")
    assert binding.awake is False
    assert binding.hidden is False


def test_7b_target_five_uses_all_four_node9_slots():
    node9 = "nscc-ds-4a100-node9"
    node10 = "nscc-ds-4a100-node10"
    service, store = _service(
        [
            _binding("awake-14b", "dsqwen-14b", node10, (0, 1), awake=True),
            _binding("awake-7b", "target", node10, (2,), awake=True),
            _binding("awake-llama", "dsllama-8b", node10, (3,), awake=True),
            _binding("target-node9-gpu-0", "target", node9, (0,), awake=False),
            _binding("target-node9-gpu-1", "target", node9, (1,), awake=False),
            _binding("target-node9-gpu-2", "target", node9, (2,), awake=False),
            _binding("target-node9-gpu-3", "target", node9, (3,), awake=False),
            _binding("target-node10-gpu-0", "target", node10, (0,), awake=False),
            _binding("target-node10-gpu-1", "target", node10, (1,), awake=False),
            _binding("target-node10-gpu-3", "target", node10, (3,), awake=False),
        ]
    )

    result = service.put_model_target("target", wake_replicas=5)

    assert result["wake_replicas"] == 5
    assert [action["serve_id"] for action in result["actions"]] == [
        "target-node9-gpu-0",
        "target-node9-gpu-1",
        "target-node9-gpu-2",
        "target-node9-gpu-3",
    ]
    awake = [
        binding
        for binding in store.load().bindings
        if binding.model == "target" and binding.awake
    ]
    assert len(awake) == 5
    assert sum(binding.slot.node == node9 for binding in awake) == 4