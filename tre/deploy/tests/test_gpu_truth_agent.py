import json
import subprocess

from scripts import gpu_truth_agent
from scripts.gpu_truth_agent import (
    build_payload,
    encode_setex_command,
    parse_nvidia_smi_csv,
    run_agent,
)


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


def test_encode_setex_command_uses_resp_arrays():
    encoded = encode_setex_command("tre:gpu_truth:node-a", 120, '{"ok":true}')

    assert encoded == (
        b"*4\r\n"
        b"$5\r\nSETEX\r\n"
        b"$20\r\ntre:gpu_truth:node-a\r\n"
        b"$3\r\n120\r\n"
        b"$11\r\n{\"ok\":true}\r\n"
    )


class RecordingSetexClient:
    def __init__(self):
        self.calls = []

    def setex(self, key, ttl_s, value):
        self.calls.append((key, ttl_s, value))


def _ok_gpus():
    return [{"uuid": "GPU-a", "used_mib": 10, "total_mib": 40536}]


def test_run_agent_keeps_polling_after_a_transient_nvidia_smi_failure():
    attempts = []

    def collect():
        attempts.append(1)
        if len(attempts) == 1:
            raise subprocess.CalledProcessError(255, ["nvidia-smi"])
        return _ok_gpus()

    run_agent(
        RecordingSetexClient(),
        node="node-a",
        ttl_s=120,
        interval_s=30.0,
        collect=collect,
        sleep=lambda _s: None,
        iterations=2,
    )

    assert len(attempts) == 2


def test_run_agent_does_not_refresh_the_key_when_collection_fails():
    attempts = []

    def collect():
        attempts.append(1)
        if len(attempts) == 1:
            raise subprocess.CalledProcessError(255, ["nvidia-smi"])
        return _ok_gpus()

    client = RecordingSetexClient()
    run_agent(
        client,
        node="node-a",
        ttl_s=120,
        interval_s=30.0,
        collect=collect,
        sleep=lambda _s: None,
        iterations=2,
    )

    assert len(client.calls) == 1
    assert json.loads(client.calls[0][2])["gpus"] == _ok_gpus()


def test_run_agent_keeps_polling_after_a_redis_publish_failure():
    class FailingOnceClient(RecordingSetexClient):
        def setex(self, key, ttl_s, value):
            super().setex(key, ttl_s, value)
            if len(self.calls) == 1:
                raise OSError("redis unreachable")

    client = FailingOnceClient()
    run_agent(
        client,
        node="node-a",
        ttl_s=120,
        interval_s=30.0,
        collect=_ok_gpus,
        sleep=lambda _s: None,
        iterations=2,
    )

    assert len(client.calls) == 2


def test_run_agent_reports_collection_failures_on_stderr(capsys):
    def collect():
        raise subprocess.CalledProcessError(255, ["nvidia-smi"])

    run_agent(
        RecordingSetexClient(),
        node="node-a",
        ttl_s=120,
        interval_s=30.0,
        collect=collect,
        sleep=lambda _s: None,
        iterations=1,
    )

    assert "nvidia-smi" in capsys.readouterr().err


def test_collect_nvidia_smi_bounds_the_subprocess_with_a_timeout(monkeypatch):
    seen = {}

    def fake_check_output(cmd, **kwargs):
        seen.update(kwargs)
        return "GPU-a, 10, 40536\n"

    monkeypatch.setattr(gpu_truth_agent.subprocess, "check_output", fake_check_output)

    assert gpu_truth_agent.collect_nvidia_smi() == _ok_gpus()
    assert seen.get("timeout", 0) > 0
