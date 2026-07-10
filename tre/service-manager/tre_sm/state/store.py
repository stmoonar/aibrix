from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Iterable, Mapping, Protocol

from tre_common import rediskeys
from tre_sm.allocator.slots import Binding, Slot

_NAT_SPLIT = re.compile(r"(\d+)")


class RedisStateClient(Protocol):
    def get(self, key: str): ...

    def set(self, key: str, value: str) -> None: ...

    def delete(self, key: str) -> None: ...

    def hgetall(self, key: str) -> Mapping[object, object]: ...

    def hset(self, key: str, mapping: Mapping[str, str]) -> None: ...


@dataclass(frozen=True)
class StateSnapshot:
    version: int
    bindings: list[Binding]


class StateConflict(RuntimeError):
    def __init__(self, *, expected_version: int, current_version: int) -> None:
        super().__init__(
            f"state version conflict: expected {expected_version}, current {current_version}"
        )
        self.expected_version = expected_version
        self.current_version = current_version


class StateStore:
    def __init__(self, redis: RedisStateClient) -> None:
        self._redis = redis

    def load(self) -> StateSnapshot:
        return StateSnapshot(version=self._current_version(), bindings=self._load_bindings())

    def save(self, bindings: Iterable[Binding], *, expected_version: int) -> int:
        current_version = self._current_version()
        if current_version != expected_version:
            raise StateConflict(
                expected_version=expected_version,
                current_version=current_version,
            )

        mapping = self._encode_bindings(bindings)
        self._redis.delete(rediskeys.SM_STATE_KEY)
        if mapping:
            self._redis.hset(rediskeys.SM_STATE_KEY, mapping=mapping)
        next_version = expected_version + 1
        self._redis.set(rediskeys.SM_VERSION_KEY, str(next_version))
        return next_version

    def _current_version(self) -> int:
        raw = self._redis.get(rediskeys.SM_VERSION_KEY)
        if raw is None:
            return 0
        return int(_to_text(raw))

    def _load_bindings(self) -> list[Binding]:
        raw = self._redis.hgetall(rediskeys.SM_STATE_KEY) or {}
        bindings: list[Binding] = []
        for raw_serve_id, raw_payload in sorted(raw.items(), key=lambda item: _natural_key(item[0])):
            serve_id = _to_text(raw_serve_id)
            payload = json.loads(_to_text(raw_payload))
            bindings.append(
                Binding(
                    serve_id=serve_id,
                    model=str(payload["model"]),
                    slot=Slot(
                        node=str(payload["node"]),
                        gpu_ids=tuple(int(gpu) for gpu in payload["gpu_ids"]),
                    ),
                    awake=bool(payload["awake"]),
                    hidden=bool(payload.get("hidden", False)),
                )
            )
        return bindings

    def _encode_bindings(self, bindings: Iterable[Binding]) -> dict[str, str]:
        encoded: dict[str, str] = {}
        for binding in bindings:
            if binding.serve_id in encoded:
                raise ValueError(f"duplicate serve_id: {binding.serve_id}")
            encoded[binding.serve_id] = json.dumps(
                {
                    "model": binding.model,
                    "node": binding.slot.node,
                    "gpu_ids": list(binding.slot.gpu_ids),
                    "awake": binding.awake,
                    "hidden": binding.hidden,
                },
                sort_keys=True,
                separators=(",", ":"),
            )
        return encoded


def _natural_key(value: object) -> tuple[object, ...]:
    return tuple(int(part) if part.isdigit() else part for part in _NAT_SPLIT.split(_to_text(value)))


def _to_text(value: object) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)
