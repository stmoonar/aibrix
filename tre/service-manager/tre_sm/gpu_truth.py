from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Mapping, Protocol


GPU_TRUTH_KEY_PREFIX = "tre:gpu_truth:"


@dataclass(frozen=True)
class NodeGpuTruth:
    """Parsed GPU truth for one node.

    Its existence is the availability signal: providers return None when a
    node has no usable truth, which lets callers fail closed. Freshness is
    delegated to the Redis TTL the publisher sets (single clock, server side)
    rather than to the payload timestamp -- node clocks in this cluster are
    known to drift by minutes relative to the control plane, so a
    timestamp-based staleness check would block cold starts spuriously.
    """

    node: str
    used_by_uuid: Mapping[str, int]
    timestamp: float | None = None

    def used_mib(self, gpu_uuid: str) -> int | None:
        return self.used_by_uuid.get(gpu_uuid)


class GpuTruthProvider(Protocol):
    def used_mib(self, *, node: str, gpu_id: int, gpu_uuid: str) -> int | None: ...

    def node_truth(self, *, node: str) -> NodeGpuTruth | None: ...


class NullGpuTruth:
    def used_mib(self, *, node: str, gpu_id: int, gpu_uuid: str) -> int | None:
        return None

    def node_truth(self, *, node: str) -> NodeGpuTruth | None:
        return None


class RedisGpuTruth:
    def __init__(self, redis_client, *, key_prefix: str = GPU_TRUTH_KEY_PREFIX) -> None:
        self._redis = redis_client
        self._key_prefix = key_prefix

    def node_truth(self, *, node: str) -> NodeGpuTruth | None:
        raw = self._redis.get(f"{self._key_prefix}{node}")
        if raw is None:
            return None
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8")
        try:
            payload = json.loads(str(raw))
        except (TypeError, ValueError):
            return None
        if not isinstance(payload, dict):
            return None
        used_by_uuid: dict[str, int] = {}
        for item in payload.get("gpus", []):
            if not isinstance(item, dict):
                continue
            try:
                used_by_uuid[str(item["uuid"])] = int(item["used_mib"])
            except (KeyError, TypeError, ValueError):
                continue
        timestamp = payload.get("timestamp")
        return NodeGpuTruth(
            node=node,
            used_by_uuid=used_by_uuid,
            timestamp=float(timestamp) if isinstance(timestamp, (int, float)) else None,
        )

    def used_mib(self, *, node: str, gpu_id: int, gpu_uuid: str) -> int | None:
        truth = self.node_truth(node=node)
        if truth is None:
            return None
        return truth.used_mib(gpu_uuid)
