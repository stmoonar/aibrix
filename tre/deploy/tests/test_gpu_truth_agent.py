from scripts.gpu_truth_agent import build_payload, parse_nvidia_smi_csv


def test_parse_nvidia_smi_csv_extracts_uuid_used_and_total():
    text = """GPU-a, 10, 40536
GPU-b, 20 MiB, 40536 MiB
bad row
"""

    assert parse_nvidia_smi_csv(text) == [
        {"uuid": "GPU-a", "used_mib": 10, "total_mib": 40536},
        {"uuid": "GPU-b", "used_mib": 20, "total_mib": 40536},
    ]


def test_build_payload_includes_node_and_gpus():
    payload = build_payload("node-a", [{"uuid": "GPU-a", "used_mib": 10, "total_mib": 40536}], now=123.4)

    assert payload == {
        "node": "node-a",
        "timestamp": 123.4,
        "gpus": [{"uuid": "GPU-a", "used_mib": 10, "total_mib": 40536}],
    }
