from __future__ import annotations

import argparse
import json
import socket
import subprocess
import time


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


def build_payload(node: str, gpus: list[dict], *, now: float | None = None) -> dict:
    return {
        "node": node,
        "timestamp": time.time() if now is None else now,
        "gpus": gpus,
    }


def collect_nvidia_smi() -> list[dict]:
    output = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=uuid,memory.used,memory.total",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    )
    return parse_nvidia_smi_csv(output)


def publish_once(redis_client, *, node: str, ttl_s: int) -> dict:
    payload = build_payload(node, collect_nvidia_smi())
    redis_client.setex(f"tre:gpu_truth:{node}", ttl_s, json.dumps(payload, separators=(",", ":")))
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Publish node GPU memory truth to TRE Redis.")
    parser.add_argument("--redis-url", required=True)
    parser.add_argument("--node", default=socket.gethostname())
    parser.add_argument("--interval-s", type=float, default=30.0)
    parser.add_argument("--ttl-s", type=int, default=120)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()

    import redis  # type: ignore[import-not-found]

    client = redis.Redis.from_url(args.redis_url)
    while True:
        publish_once(client, node=args.node, ttl_s=args.ttl_s)
        if args.once:
            return 0
        time.sleep(args.interval_s)


def _parse_mib(value: str) -> int:
    return int(value.replace("MiB", "").strip())


if __name__ == "__main__":
    raise SystemExit(main())
