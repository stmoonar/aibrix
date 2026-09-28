from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Mapping, Protocol

from tre_common import rediskeys


GPU_TRUTH_KEY_PREFIX = rediskeys.GPU_TRUTH_KEY_PREFIX
GPU_TRUTH_REFRESH_KEY_PREFIX = rediskeys.GPU_TRUTH_REFRESH_KEY_PREFIX
LOG = logging.getLogger(__name__)


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
    #: Total memory per GPU (gpu-truth agent ``total_mib``); empty when not reported.
    total_by_uuid: Mapping[str, int] = field(default_factory=dict)
    #: Publish counter of the agent process (None: agent predates it).
    seq: int | None = None
    #: Highest refresh request this sample answers (it was taken after the agent
    #: read that request). None: the agent does not serve refresh requests.
    refresh_seq: int | None = None

    def used_mib(self, gpu_uuid: str) -> int | None:
        return self.used_by_uuid.get(gpu_uuid)

    def total_mib(self, gpu_uuid: str) -> int | None:
        return self.total_by_uuid.get(gpu_uuid)


class GpuTruthProvider(Protocol):
    """Optionally also ``request_refresh(*, node) -> int | None`` (see
    :meth:`RedisGpuTruth.request_refresh`); providers without it are read as is."""

    def used_mib(self, *, node: str, gpu_id: int, gpu_uuid: str) -> int | None: ...

    def node_truth(self, *, node: str) -> NodeGpuTruth | None: ...


class NullGpuTruth:
    def used_mib(self, *, node: str, gpu_id: int, gpu_uuid: str) -> int | None:
        return None

    def node_truth(self, *, node: str) -> NodeGpuTruth | None:
        return None


class RedisGpuTruth:
    def __init__(
        self,
        redis_client,
        *,
        key_prefix: str = GPU_TRUTH_KEY_PREFIX,
        refresh_key_prefix: str = GPU_TRUTH_REFRESH_KEY_PREFIX,
    ) -> None:
        self._redis = redis_client
        self._key_prefix = key_prefix
        self._refresh_key_prefix = refresh_key_prefix

    def request_refresh(self, *, node: str) -> int | None:
        """Ask the node's gpu-truth agent for a sample now: INCR the node's refresh
        counter and return the new value N. A payload with ``refresh_seq >= N``
        was sampled after this request. None when the request could not be sent."""
        try:
            return int(self._redis.incr(f"{self._refresh_key_prefix}{node}"))
        except Exception:  # the gate falls back to re-reading the periodic sample
            LOG.warning("gpu-truth refresh request for node %s failed", node, exc_info=True)
            return None

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
        total_by_uuid: dict[str, int] = {}
        for item in payload.get("gpus", []):
            if not isinstance(item, dict):
                continue
            try:
                used_by_uuid[str(item["uuid"])] = int(item["used_mib"])
            except (KeyError, TypeError, ValueError):
                continue
            try:
                total_by_uuid[str(item["uuid"])] = int(item["total_mib"])
            except (KeyError, TypeError, ValueError):
                pass
        timestamp = payload.get("timestamp")
        return NodeGpuTruth(
            node=node,
            used_by_uuid=used_by_uuid,
            total_by_uuid=total_by_uuid,
            timestamp=float(timestamp) if isinstance(timestamp, (int, float)) else None,
            seq=_int_or_none(payload.get("seq")),
            refresh_seq=_int_or_none(payload.get("refresh_seq")),
        )

    def used_mib(self, *, node: str, gpu_id: int, gpu_uuid: str) -> int | None:
        truth = self.node_truth(node=node)
        if truth is None:
            return None
        return truth.used_mib(gpu_uuid)


def _int_or_none(value) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value
