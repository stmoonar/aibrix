from __future__ import annotations

import argparse
import json
import socket
import subprocess
import sys
import time
from urllib.parse import unquote, urlparse

# Redis keys (kept equal to tre_common.rediskeys by deploy/tests/test_gpu_truth_agent.py;
# this script runs standalone from a ConfigMap and cannot import tre_common).
GPU_TRUTH_KEY_PREFIX = "tre:gpu_truth:"
GPU_TRUTH_REFRESH_KEY_PREFIX = "tre:gpu_truth_refresh:"

#: How often the agent polls its refresh counter (a cheap GET) between samples.
DEFAULT_REFRESH_POLL_S = 0.25
DEFAULT_INTERVAL_S = 10.0


def parse_nvidia_smi_csv(text: str) -> list[dict]:
    rows: list[dict] = []
    for line in text.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 3:
            continue
        uuid, used, total = parts
        try:
            rows.append(
                {
                    "uuid": uuid,
                    "used_mib": _parse_mib(used),
                    "total_mib": _parse_mib(total),
                }
            )
        except ValueError:
            continue
    return rows


def build_payload(
    node: str,
    gpus: list[dict],
    *,
    now: float | None = None,
    seq: int | None = None,
    refresh_seq: int | None = None,
) -> dict:
    """``seq`` increases with every publish of this agent process; ``refresh_seq``
    is the highest refresh request (``tre:gpu_truth_refresh:<node>``) answered
    by a sample taken after that request was read. Its presence tells the
    service-manager that this agent serves refreshes (an old agent omits it)."""
    payload = {
        "node": node,
        "timestamp": time.time() if now is None else now,
        "gpus": gpus,
    }
    if seq is not None:
        payload["seq"] = seq
    if refresh_seq is not None:
        payload["refresh_seq"] = refresh_seq
    return payload


NVIDIA_SMI_TIMEOUT_S = 20.0


def collect_nvidia_smi(*, timeout_s: float = NVIDIA_SMI_TIMEOUT_S) -> list[dict]:
    # timeout_s bounds a wedged driver: nvidia-smi can hang indefinitely on a
    # stuck GPU, which would silently freeze the publish loop.
    output = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=uuid,memory.used,memory.total",
            "--format=csv,noheader,nounits",
        ],
        text=True,
        timeout=timeout_s,
    )
    return parse_nvidia_smi_csv(output)


def publish_once(
    redis_client,
    *,
    node: str,
    ttl_s: int,
    collect=None,
    seq: int | None = None,
    refresh_seq: int | None = None,
) -> dict:
    collect = collect or collect_nvidia_smi
    payload = build_payload(node, collect(), seq=seq, refresh_seq=refresh_seq)
    redis_client.setex(
        f"{GPU_TRUTH_KEY_PREFIX}{node}", ttl_s, json.dumps(payload, separators=(",", ":"))
    )
    return payload


def read_refresh_request(redis_client, node: str) -> int | None:
    """The node's refresh counter (0 when never requested); None when the client
    cannot GET (refresh unsupported) or the value is not an integer."""
    getter = getattr(redis_client, "get", None)
    if not callable(getter):
        return None
    raw = getter(f"{GPU_TRUTH_REFRESH_KEY_PREFIX}{node}")
    if raw is None:
        return 0
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8")
    try:
        return int(raw)
    except (TypeError, ValueError):
        return None


def run_agent(
    redis_client,
    *,
    node: str,
    ttl_s: int,
    interval_s: float,
    collect=None,
    sleep=None,
    monotonic=None,
    refresh_poll_s: float = DEFAULT_REFRESH_POLL_S,
    iterations: int | None = None,
    error_stream=None,
) -> int:
    """Publish GPU truth forever, surviving transient collection/publish faults.

    A single nvidia-smi hiccup (exit 255 on a driver blip) used to kill the
    process and put the DaemonSet into CrashLoopBackOff. On failure we log and
    retry instead, and deliberately leave the Redis key untouched: letting its
    TTL expire is precisely the signal service-manager needs to treat GPU truth
    as unavailable and fail its startup-headroom gate closed.

    Samples are published every ``interval_s`` AND on demand: between samples
    the agent polls ``tre:gpu_truth_refresh:<node>`` every ``refresh_poll_s``;
    when the counter differs from the last request served (the service-manager
    INCRed it, or it was reset) the agent samples at once and publishes the
    counter value it read as ``refresh_seq``. The sample is taken after that
    read, hence after the request: a reader that sees ``refresh_seq >= N`` has a
    sample newer than its request N. Any publish restarts the periodic interval.
    A client without GET (refresh unsupported) just samples every
    ``interval_s``. ``iterations`` bounds the number of sampling rounds (tests).
    """
    sleep = sleep or time.sleep
    monotonic = monotonic or time.monotonic
    stream = error_stream if error_stream is not None else sys.stderr
    if not callable(getattr(redis_client, "get", None)):
        return _run_periodic(
            redis_client, node=node, ttl_s=ttl_s, interval_s=interval_s,
            collect=collect, sleep=sleep, iterations=iterations, stream=stream,
        )
    served = 0
    seq = 0
    completed = 0
    next_sample = monotonic()
    while iterations is None or completed < iterations:
        requested = None
        try:
            requested = read_refresh_request(redis_client, node)
        except Exception as exc:  # noqa: BLE001 - keep sampling periodically
            print(
                f"gpu_truth: refresh poll failed for node={node}: {exc!r}",
                file=stream,
                flush=True,
            )
        wanted = requested is not None and requested != served
        if wanted or monotonic() >= next_sample:
            if requested is not None:
                # Read BEFORE sampling: the sample below is newer than request `requested`.
                served = requested
            seq += 1
            _publish_logged(
                redis_client, node=node, ttl_s=ttl_s, collect=collect, stream=stream,
                seq=seq, refresh_seq=served,
            )
            completed += 1
            next_sample = monotonic() + interval_s
            if iterations is not None and completed >= iterations:
                break
        sleep(max(0.0, min(refresh_poll_s, next_sample - monotonic())))
    return 0


def _run_periodic(redis_client, *, node, ttl_s, interval_s, collect, sleep, iterations, stream) -> int:
    """Periodic samples only (a client without GET: no on-demand refresh; the
    payload carries no ``refresh_seq``)."""
    completed = 0
    seq = 0
    while iterations is None or completed < iterations:
        seq += 1
        _publish_logged(redis_client, node=node, ttl_s=ttl_s, collect=collect, stream=stream, seq=seq)
        completed += 1
        if iterations is not None and completed >= iterations:
            break
        sleep(interval_s)
    return 0


def _publish_logged(redis_client, *, node, ttl_s, collect, stream, seq, refresh_seq=None) -> None:
    try:
        publish_once(
            redis_client,
            node=node,
            ttl_s=ttl_s,
            collect=collect,
            seq=seq,
            refresh_seq=refresh_seq,
        )
    except Exception as exc:  # noqa: BLE001 - the loop must outlive any fault
        print(
            f"gpu_truth: publish failed for node={node}: {exc!r}",
            file=stream,
            flush=True,
        )


class RawRedisClient:
    """Minimal RESP client (GET / SETEX / INCR) for images without redis-py.

    Keeps one connection open (the refresh poll runs several times a second)
    and reconnects after any error. Honours the URL's password and db index.
    """

    def __init__(self, redis_url: str, *, timeout_s: float = 5.0) -> None:
        parsed = urlparse(redis_url)
        if parsed.scheme != "redis":
            raise ValueError(f"unsupported redis url scheme: {parsed.scheme}")
        self._host = parsed.hostname or "localhost"
        self._port = parsed.port or 6379
        self._username = unquote(parsed.username) if parsed.username else None
        self._password = unquote(parsed.password) if parsed.password else None
        path = (parsed.path or "").strip("/")
        self._db = int(path) if path else 0
        self._timeout_s = timeout_s
        self._sock = None
        self._reader = None

    def setex(self, key: str, ttl_s: int, value: str) -> None:
        reply = self.execute("SETEX", key, str(ttl_s), value)
        if reply != "OK":
            raise RuntimeError(f"redis SETEX failed: {reply!r}")

    def get(self, key: str) -> bytes | None:
        return self.execute("GET", key)

    def incr(self, key: str) -> int:
        return int(self.execute("INCR", key))

    def close(self) -> None:
        sock, self._sock, self._reader = self._sock, None, None
        if sock is not None:
            try:
                sock.close()
            except OSError:
                pass

    def execute(self, *parts: str):
        if self._sock is None:
            self._connect()
        try:
            return self._roundtrip(parts)
        except Exception:
            self.close()
            raise

    def _connect(self) -> None:
        sock = socket.create_connection((self._host, self._port), timeout=self._timeout_s)
        self._sock, self._reader = sock, sock.makefile("rb")
        try:
            if self._password is not None:
                auth = ("AUTH", self._username, self._password) if self._username else ("AUTH", self._password)
                self._roundtrip(auth)
            if self._db:
                self._roundtrip(("SELECT", str(self._db)))
        except Exception:
            self.close()
            raise

    def _roundtrip(self, parts):
        self._sock.sendall(encode_command(*parts))
        reply = read_reply(self._reader)
        if isinstance(reply, RedisReplyError):
            raise reply
        return reply


#: Backwards-compatible name (the agent used to only SETEX).
RawRedisSetexClient = RawRedisClient


class RedisReplyError(RuntimeError):
    pass


def encode_command(*parts: str) -> bytes:
    encoded = [str(part).encode("utf-8") for part in parts]
    out = [f"*{len(encoded)}\r\n".encode("ascii")]
    for part in encoded:
        out.append(f"${len(part)}\r\n".encode("ascii"))
        out.append(part)
        out.append(b"\r\n")
    return b"".join(out)


def encode_setex_command(key: str, ttl_s: int, value: str) -> bytes:
    return encode_command("SETEX", key, str(ttl_s), value)


def read_reply(reader):
    """One RESP2 reply: simple string -> str, error -> RedisReplyError,
    integer -> int, bulk string -> bytes | None, array -> list."""
    line = reader.readline()
    if not line.endswith(b"\r\n"):
        raise ConnectionError("redis connection closed")
    kind, body = line[:1], line[1:-2]
    if kind == b"+":
        return body.decode("utf-8")
    if kind == b"-":
        return RedisReplyError(body.decode("utf-8", "replace"))
    if kind == b":":
        return int(body)
    if kind == b"$":
        length = int(body)
        if length < 0:
            return None
        data = reader.read(length + 2)
        if len(data) != length + 2:
            raise ConnectionError("redis connection closed")
        return data[:-2]
    if kind == b"*":
        count = int(body)
        if count < 0:
            return None
        return [read_reply(reader) for _ in range(count)]
    raise RuntimeError(f"unexpected redis reply: {line!r}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Publish node GPU memory truth to TRE Redis.")
    parser.add_argument("--redis-url", required=True)
    parser.add_argument("--node", default=socket.gethostname())
    parser.add_argument("--interval-s", type=float, default=DEFAULT_INTERVAL_S)
    parser.add_argument(
        "--refresh-poll-s",
        type=float,
        default=DEFAULT_REFRESH_POLL_S,
        help="poll interval of the on-demand refresh counter (0 disables refreshes)",
    )
    parser.add_argument("--ttl-s", type=int, default=120)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()

    try:
        import redis  # type: ignore[import-not-found]

        client = redis.Redis.from_url(args.redis_url)
    except ModuleNotFoundError:
        client = RawRedisClient(args.redis_url)
    if args.once:
        # --once stays fail-fast so operators can use it as a diagnostic probe.
        publish_once(client, node=args.node, ttl_s=args.ttl_s)
        return 0
    if args.refresh_poll_s <= 0:
        client = _SetexOnly(client)
    return run_agent(
        client,
        node=args.node,
        ttl_s=args.ttl_s,
        interval_s=args.interval_s,
        refresh_poll_s=args.refresh_poll_s if args.refresh_poll_s > 0 else DEFAULT_REFRESH_POLL_S,
    )


class _SetexOnly:
    """Hides GET: periodic samples only (--refresh-poll-s 0)."""

    def __init__(self, client) -> None:
        self._client = client

    def setex(self, key, ttl_s, value):
        return self._client.setex(key, ttl_s, value)


def _parse_mib(value: str) -> int:
    return int(value.replace("MiB", "").strip())


if __name__ == "__main__":
    raise SystemExit(main())
