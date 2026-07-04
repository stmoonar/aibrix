from __future__ import annotations

from collect_gpu_uuids import apply_gpu_uuids, parse_nvidia_smi_l


def test_parse_nvidia_smi_l_returns_uuids_by_gpu_index() -> None:
    output = """
    GPU 0: NVIDIA A100-SXM4-40GB (UUID: GPU-a)
    GPU 1: NVIDIA A100-SXM4-40GB (UUID: GPU-b)
    """

    assert parse_nvidia_smi_l(output) == ("GPU-a", "GPU-b")


def test_apply_gpu_uuids_updates_matching_registry_nodes() -> None:
    registry = {
        "cluster": {
            "nodes": [
                {"name": "node-a", "gpus": 2, "two_gpu_slots": [[0, 1]]},
                {"name": "node-b", "gpus": 1, "two_gpu_slots": []},
            ]
        }
    }

    updated = apply_gpu_uuids(registry, {"node-a": ("GPU-a", "GPU-b")})

    assert updated["cluster"]["nodes"][0]["gpu_uuids"] == ["GPU-a", "GPU-b"]
    assert "gpu_uuids" not in updated["cluster"]["nodes"][1]
