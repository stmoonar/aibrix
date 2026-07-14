from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Iterable, Mapping, Protocol

from tre_common import rediskeys
from tre_sm.allocator.slots import Binding, Slot, natural_key
from tre_sm.state.operations import current_fence


_SAVE_SCRIPT = r"""
local current = tonumber(redis.call('GET', KEYS[2]) or '0')
local expected = tonumber(ARGV[1])
if current ~= expected then
  return {0, current}
end
if ARGV[3] == '1' then
  if redis.call('GET', KEYS[3]) ~= ARGV[4] then
    return {-1, current}
  end
end
redis.call('DEL', KEYS[1])
local index = 5
while index <= #ARGV do
  redis.call('HSET', KEYS[1], ARGV[index], ARGV[index + 1])
  index = index + 2
end
local next_version = tonumber(ARGV[2])
redis.call('SET', KEYS[2], tostring(next_version))
return {1, next_version}
"""


class RedisStateClient(Protocol):
    def get(self, key: str): ...

    def set(self, key: str, value: str) -> None: ...

    def delete(self, key: str) -> None: ...

    def hgetall(self, key: str) -> Mapping[object, object]: ...

    def hset(self, key: str, mapping: Mapping[str, str]) -> None: ...
    def eval(self, script: str, numkeys: int, *keys_and_args): ...


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


class StateFenceError(RuntimeError):
    pass


class StateStore:
    def __init__(self, redis: RedisStateClient, *, require_fence: bool = False) -> None:
        self._redis = redis
        self._require_fence = require_fence

    def load(self) -> StateSnapshot:
        return StateSnapshot(version=self._current_version(), bindings=self._load_bindings())

    def save(self, bindings: Iterable[Binding], *, expected_version: int) -> int:
        mapping = self._encode_bindings(bindings)
        eval_method = getattr(self._redis, "eval", None)
        if callable(eval_method):
            return self._save_atomic(
                mapping=mapping, expected_version=expected_version
            )
        if self._require_fence:
            raise StateFenceError("atomic Redis EVAL is required for fenced state writes")

        # In-memory test clients may omit EVAL. Production always enables the
        # Lua path above and refuses this compatibility fallback.
        current_version = self._current_version()
        if current_version != expected_version:
            raise StateConflict(
                expected_version=expected_version,
                current_version=current_version,
            )

        self._redis.delete(rediskeys.SM_STATE_KEY)
        if mapping:
            self._redis.hset(rediskeys.SM_STATE_KEY, mapping=mapping)
        next_version = expected_version + 1
        self._redis.set(rediskeys.SM_VERSION_KEY, str(next_version))
        return next_version

    def _save_atomic(self, *, mapping: dict[str, str], expected_version: int) -> int:
        fence = current_fence()
        if self._require_fence and fence is None:
            raise StateFenceError("state write requires an active writer fence")
        require_fence = self._require_fence or fence is not None
        args: list[str] = [
            str(expected_version),
            str(expected_version + 1),
            "1" if require_fence else "0",
            fence.lock_value if fence is not None else "",
        ]
        for field, payload in mapping.items():
            args.extend((field, payload))
        result = self._redis.eval(
            _SAVE_SCRIPT,
            3,
            rediskeys.SM_STATE_KEY,
            rediskeys.SM_VERSION_KEY,
            rediskeys.SM_WRITER_LOCK_KEY,
            *args,
        )
        status = int(result[0])
        value = int(result[1])
        if status == 0:
            raise StateConflict(
                expected_version=expected_version,
                current_version=value,
            )
        if status == -1:
            raise StateFenceError("writer fence is no longer active")
        return value

    def _current_version(self) -> int:
        raw = self._redis.get(rediskeys.SM_VERSION_KEY)
        if raw is None:
            return 0
        return int(_to_text(raw))

    def _load_bindings(self) -> list[Binding]:
        raw = self._redis.hgetall(rediskeys.SM_STATE_KEY) or {}
        bindings: list[Binding] = []
        for raw_serve_id, raw_payload in sorted(raw.items(), key=lambda item: natural_key(_to_text(item[0]))):
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

def _to_text(value: object) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)
