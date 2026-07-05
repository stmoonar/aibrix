from __future__ import annotations

import json
from typing import Protocol


GPU_TRUTH_KEY_PREFIX = "tre:gpu_truth:"


class GpuTruthProvider(Protocol):
    def used_mib(self, *, node: str, gpu_id: int, gpu_uuid: str) -> int | None: ...


class NullGpuTruth:
    def used_mib(self, *, node: str, gpu_id: int, gpu_uuid: str) -> int | None:
        return None


class RedisGpuTruth:
    def __init__(self, redis_client, *, key_prefix: str = GPU_TRUTH_KEY_PREFIX) -> None:
        self._redis = redis_client
        self._key_prefix = key_prefix

    def used_mib(self, *, node: str, gpu_id: int, gpu_uuid: str) -> int | None:
        raw = self._redis.get(f"{self._key_prefix}{node}")
        if raw is None:
            return None
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8")
        try:
            payload = json.loads(str(raw))
        except (TypeError, ValueError):
            return None
        for item in payload.get("gpus", []):
            if str(item.get("uuid")) != gpu_uuid:
                continue
            try:
                return int(item["used_mib"])
            except (KeyError, TypeError, ValueError):
                return None
        return None
