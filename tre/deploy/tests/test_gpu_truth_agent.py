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


# ------------------------------------------------ B3: on-demand refresh protocol
import io
import socket
import socketserver
import threading

import pytest

from tre_common import rediskeys


def test_agent_key_literals_match_rediskeys():
    assert gpu_truth_agent.GPU_TRUTH_KEY_PREFIX == rediskeys.GPU_TRUTH_KEY_PREFIX
    assert gpu_truth_agent.GPU_TRUTH_REFRESH_KEY_PREFIX == rediskeys.GPU_TRUTH_REFRESH_KEY_PREFIX
    # The refresh counter must never look like a node payload to SCAN tre:gpu_truth:*.
    assert not rediskeys.gpu_truth_refresh_key("n").startswith(rediskeys.GPU_TRUTH_KEY_PREFIX)


class DictRedis:
    """GET / SETEX / INCR over a dict; ``on_get`` lets a test INCR mid-run."""

    def __init__(self):
        self.values = {}
        self.setex_calls = []
        self.on_get = None

    def get(self, key):
        if self.on_get is not None:
            self.on_get(key)
        value = self.values.get(key)
        return None if value is None else str(value).encode()

    def setex(self, key, ttl_s, value):
        self.values[key] = value
        self.setex_calls.append((key, ttl_s, json.loads(value)))

    def incr(self, key):
        self.values[key] = int(self.values.get(key, 0)) + 1
        return self.values[key]


class FakeClock:
    def __init__(self):
        self.now = 100.0
        self.slept = []

    def monotonic(self):
        return self.now

    def sleep(self, seconds):
        self.slept.append(seconds)
        self.now += max(seconds, 0.001)


REFRESH_KEY = "tre:gpu_truth_refresh:node-a"


def test_refresh_request_is_served_by_a_sample_taken_after_it():
    client = DictRedis()
    clock = FakeClock()
    samples = []
    requested_at = []

    def collect():
        samples.append(clock.now)
        return _ok_gpus()

    def on_get(key):
        # The SM asks for a refresh 1 s after the first periodic sample.
        if key == REFRESH_KEY and clock.now >= 101.0 and REFRESH_KEY not in client.values:
            client.incr(REFRESH_KEY)
            requested_at.append(clock.now)

    client.on_get = on_get
    run_agent(
        client, node="node-a", ttl_s=120, interval_s=10.0, collect=collect,
        sleep=clock.sleep, monotonic=clock.monotonic, refresh_poll_s=0.25, iterations=2,
    )

    first, second = [call[2] for call in client.setex_calls]
    assert first["refresh_seq"] == 0 and first["seq"] == 1  # periodic, refresh-aware
    assert second["refresh_seq"] == 1 and second["seq"] == 2
    assert samples[1] >= requested_at[0]  # sampled after the request was read
    assert samples[1] - requested_at[0] < 0.3  # within one poll, not the 10 s period
    assert max(clock.slept) <= 0.25


def test_periodic_samples_keep_their_interval_and_carry_the_last_refresh_served():
    client = DictRedis()
    client.values[REFRESH_KEY] = 7  # requests from before this agent started
    clock = FakeClock()
    run_agent(
        client, node="node-a", ttl_s=120, interval_s=10.0, collect=_ok_gpus,
        sleep=clock.sleep, monotonic=clock.monotonic, iterations=3,
    )
    payloads = [call[2] for call in client.setex_calls]
    assert [p["seq"] for p in payloads] == [1, 2, 3]
    assert [p["refresh_seq"] for p in payloads] == [7, 7, 7]
    assert clock.now == pytest.approx(120.0, abs=0.5)  # 2 periodic intervals


def test_a_reset_refresh_counter_is_served_again():
    client = DictRedis()
    client.values[REFRESH_KEY] = 5
    clock = FakeClock()

    def on_get(key):
        if clock.now >= 102.0 and client.values.get(REFRESH_KEY) == 5:
            client.values[REFRESH_KEY] = 1  # Redis flushed, then one INCR

    client.on_get = on_get
    run_agent(
        client, node="node-a", ttl_s=120, interval_s=10.0, collect=_ok_gpus,
        sleep=clock.sleep, monotonic=clock.monotonic, iterations=2,
    )
    assert [call[2]["refresh_seq"] for call in client.setex_calls] == [5, 1]


def test_a_client_without_get_publishes_periodically_without_refresh_seq():
    client = RecordingSetexClient()
    run_agent(client, node="node-a", ttl_s=120, interval_s=30.0, collect=_ok_gpus,
              sleep=lambda _s: None, iterations=2)
    payloads = [json.loads(call[2]) for call in client.calls]
    assert [p["seq"] for p in payloads] == [1, 2]
    assert all("refresh_seq" not in p for p in payloads)  # old-agent semantics for the SM


def test_refresh_poll_failures_do_not_stop_periodic_samples(capsys):
    class FlakyGet(DictRedis):
        def get(self, key):
            raise OSError("redis blip")

    client = FlakyGet()
    clock = FakeClock()
    run_agent(client, node="node-a", ttl_s=120, interval_s=10.0, collect=_ok_gpus,
              sleep=clock.sleep, monotonic=clock.monotonic, iterations=2)
    assert len(client.setex_calls) == 2
    assert "refresh poll failed" in capsys.readouterr().err


def test_read_reply_parses_resp2():
    reader = io.BytesIO(
        b"+OK\r\n:42\r\n$5\r\nhello\r\n$-1\r\n*2\r\n:1\r\n$1\r\nx\r\n-ERR boom\r\n"
    )
    read = gpu_truth_agent.read_reply
    assert read(reader) == "OK"
    assert read(reader) == 42
    assert read(reader) == b"hello"
    assert read(reader) is None
    assert read(reader) == [1, b"x"]
    error = read(reader)
    assert isinstance(error, gpu_truth_agent.RedisReplyError) and "boom" in str(error)
    with pytest.raises(ConnectionError):
        read(io.BytesIO(b""))


class _MiniRedisHandler(socketserver.StreamRequestHandler):
    def handle(self):
        server = self.server
        server.connections += 1
        server.last_conn = self.connection
        while True:
            try:
                command = gpu_truth_agent.read_reply(self.rfile)
            except (ConnectionError, OSError):
                return
            name = command[0].decode().upper()
            args = [part.decode() for part in command[1:]]
            server.commands.append([name] + args)
            if name in ("SELECT", "AUTH"):
                self.wfile.write(b"+OK\r\n")
            elif name == "SETEX":
                server.data[args[0]] = args[2].encode()
                self.wfile.write(b"+OK\r\n")
            elif name == "GET":
                value = server.data.get(args[0])
                if value is None:
                    self.wfile.write(b"$-1\r\n")
                else:
                    self.wfile.write(b"$" + str(len(value)).encode() + b"\r\n" + value + b"\r\n")
            elif name == "INCR":
                value = int(server.data.get(args[0], b"0")) + 1
                server.data[args[0]] = str(value).encode()
                self.wfile.write(b":" + str(value).encode() + b"\r\n")
            else:
                self.wfile.write(b"-ERR unknown\r\n")


@pytest.fixture
def mini_redis():
    server = socketserver.ThreadingTCPServer(("127.0.0.1", 0), _MiniRedisHandler)
    server.daemon_threads = True
    server.data, server.commands, server.connections = {}, [], 0
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield server
    server.shutdown()
    server.server_close()


def test_raw_client_get_setex_incr_on_one_connection(mini_redis):
    host, port = mini_redis.server_address
    client = gpu_truth_agent.RawRedisClient(f"redis://{host}:{port}/0")
    assert client.get(REFRESH_KEY) is None
    assert client.incr(REFRESH_KEY) == 1
    assert client.get(REFRESH_KEY) == b"1"
    client.setex("tre:gpu_truth:node-a", 120, '{"ok":true}')
    assert client.get("tre:gpu_truth:node-a") == b'{"ok":true}'
    client.close()
    assert mini_redis.connections == 1
    assert gpu_truth_agent.RawRedisSetexClient is gpu_truth_agent.RawRedisClient
    # read_refresh_request works on the raw client like on redis-py.
    client2 = gpu_truth_agent.RawRedisClient(f"redis://{host}:{port}")
    assert gpu_truth_agent.read_refresh_request(client2, "node-a") == 1
    client2.close()


def test_raw_client_selects_the_db_and_authenticates(mini_redis):
    host, port = mini_redis.server_address
    client = gpu_truth_agent.RawRedisClient(f"redis://:s3cret@{host}:{port}/2")
    client.get("k")
    client.close()
    assert mini_redis.commands[:3] == [["AUTH", "s3cret"], ["SELECT", "2"], ["GET", "k"]]


def test_raw_client_reconnects_after_a_dropped_connection(mini_redis):
    host, port = mini_redis.server_address
    client = gpu_truth_agent.RawRedisClient(f"redis://{host}:{port}/0")
    client.incr("c")
    mini_redis.last_conn.shutdown(socket.SHUT_RDWR)  # the server drops the connection
    with pytest.raises((ConnectionError, OSError)):
        client.get("c")
    assert client._sock is None  # dropped; the next command reconnects
    assert client.get("c") == b"1"  # a fresh connection
    client.close()


def test_raw_client_raises_redis_errors(mini_redis):
    host, port = mini_redis.server_address
    client = gpu_truth_agent.RawRedisClient(f"redis://{host}:{port}/0")
    with pytest.raises(gpu_truth_agent.RedisReplyError):
        client.execute("FLUSHALL")
    client.close()
