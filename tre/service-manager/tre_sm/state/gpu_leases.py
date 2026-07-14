from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from typing import Mapping, Protocol

from tre_common import rediskeys
from tre_sm.allocator.slots import Binding
from tre_sm.state.operations import current_fence
from tre_sm.state.store import StateFenceError


_ACQUIRE_GPU_SCRIPT = r"""
if redis.call('GET', KEYS[2]) ~= ARGV[1] then
  return {-1, 'writer_fence_lost', ''}
end
local now_parts = redis.call('TIME')
local now_ms = tonumber(now_parts[1]) * 1000 + math.floor(tonumber(now_parts[2]) / 1000)
local ttl_ms = tonumber(ARGV[7])
local expires_at_ms = ttl_ms == 0 and 0 or now_ms + ttl_ms
local count = tonumber(ARGV[8])
for index = 1, count do
  local field = ARGV[8 + index]
  local raw = redis.call('HGET', KEYS[1], field)
  if raw then
    local existing = cjson.decode(raw)
    local active = tonumber(existing.expires_at_ms) == 0 or tonumber(existing.expires_at_ms) > now_ms
    if active and existing.binding_id ~= ARGV[2] then
      return {0, field, existing.binding_id}
    end
  end
end
local gpu_ids = cjson.decode(ARGV[6])
local record = cjson.encode({
  binding_id=ARGV[2], node=ARGV[3], gpu_ids=gpu_ids,
  owner=ARGV[4], fencing_token=tonumber(ARGV[5]), phase=ARGV[6 + count + 3],
  expires_at_ms=expires_at_ms
})
for index = 1, count do
  redis.call('HSET', KEYS[1], ARGV[8 + index], record)
end
return {1, tostring(expires_at_ms), ''}
"""

# The acquire script's argument layout is intentionally explicit. Phase lives
# after the variable-length GPU field list, so TP2 checks and writes both fields
# within one indivisible Lua execution.

_RELEASE_GPU_SCRIPT = r"""
if redis.call('GET', KEYS[2]) ~= ARGV[1] then
  return -1
end
local count = tonumber(ARGV[4])
for index = 1, count do
  local field = ARGV[4 + index]
  local raw = redis.call('HGET', KEYS[1], field)
  if raw then
    local existing = cjson.decode(raw)
    if existing.binding_id == ARGV[2] and tonumber(existing.fencing_token) <= tonumber(ARGV[3]) then
      redis.call('HDEL', KEYS[1], field)
    end
  end
end
return 1
"""

_REBUILD_GPU_SCRIPT = r"""
if redis.call('GET', KEYS[2]) ~= ARGV[1] then
  return -1
end
redis.call('DEL', KEYS[1])
local index = 2
while index <= #ARGV do
  redis.call('HSET', KEYS[1], ARGV[index], ARGV[index + 1])
  index = index + 2
end
return 1
"""


class GpuLeaseRedis(Protocol):
    def hgetall(self, key: str) -> Mapping[object, object]: ...
    def eval(self, script: str, numkeys: int, *keys_and_args): ...


@dataclass(frozen=True)
class GpuLease:
    binding_id: str
    node: str
    gpu_ids: tuple[int, ...]
    owner: str
    fencing_token: int
    phase: str
    expires_at_ms: int


class GpuLeaseConflict(RuntimeError):
    def __init__(self, *, gpu: str, occupant: str) -> None:
        super().__init__(f"GPU lease conflict on {gpu}: held by {occupant}")
        self.gpu = gpu
        self.occupant = occupant


class GpuLeaseStore:
    def __init__(self, redis_client: GpuLeaseRedis, *, transient_ttl_ms: int = 120_000) -> None:
        self._redis = redis_client
        self._transient_ttl_ms = transient_ttl_ms

    def acquire(self, binding: Binding, *, phase: str) -> GpuLease:
        fence = current_fence()
        if fence is None:
            raise StateFenceError("GPU lease acquisition requires an active writer fence")
        ttl_ms = 0 if phase == "awake" else self._transient_ttl_ms
        fields = [_gpu_field(binding.slot.node, gpu) for gpu in binding.slot.gpu_ids]
        # ARGV layout: fence, binding, node, owner, token, gpu_ids_json,
        # ttl, count, fields..., phase. Lua derives a single expiry for all GPUs.
        result = self._redis.eval(
            _ACQUIRE_GPU_SCRIPT,
            2,
            rediskeys.SM_GPU_LEASES_KEY,
            rediskeys.SM_WRITER_LOCK_KEY,
            fence.lock_value,
            binding.binding_id,
            binding.slot.node,
            fence.owner,
            str(fence.token),
            json.dumps(list(binding.slot.gpu_ids)),
            str(ttl_ms),
            str(len(fields)),
            *fields,
            phase,
        )
        status = int(result[0])
        if status == -1:
            raise StateFenceError("writer fence is no longer active")
        if status == 0:
            raise GpuLeaseConflict(
                gpu=_text(result[1]), occupant=_text(result[2])
            )
        return GpuLease(
            binding_id=binding.binding_id,
            node=binding.slot.node,
            gpu_ids=binding.slot.gpu_ids,
            owner=fence.owner,
            fencing_token=fence.token,
            phase=phase,
            expires_at_ms=int(result[1]),
        )

    def release(self, binding: Binding) -> None:
        fence = current_fence()
        if fence is None:
            raise StateFenceError("GPU lease release requires an active writer fence")
        fields = [_gpu_field(binding.slot.node, gpu) for gpu in binding.slot.gpu_ids]
        result = self._redis.eval(
            _RELEASE_GPU_SCRIPT,
            2,
            rediskeys.SM_GPU_LEASES_KEY,
            rediskeys.SM_WRITER_LOCK_KEY,
            fence.lock_value,
            binding.binding_id,
            str(fence.token),
            str(len(fields)),
            *fields,
        )
        if int(result) == -1:
            raise StateFenceError("writer fence is no longer active")

    def rebuild_awake(self, bindings: list[Binding]) -> None:
        fence = current_fence()
        if fence is None:
            raise StateFenceError("GPU lease rebuild requires an active writer fence")
        mapping: dict[str, str] = {}
        for binding in bindings:
            if not binding.awake:
                continue
            record = GpuLease(
                binding_id=binding.binding_id,
                node=binding.slot.node,
                gpu_ids=binding.slot.gpu_ids,
                owner=fence.owner,
                fencing_token=fence.token,
                phase="awake",
                expires_at_ms=0,
            )
            payload = json.dumps(
                {**asdict(record), "gpu_ids": list(record.gpu_ids)},
                sort_keys=True,
                separators=(",", ":"),
            )
            for gpu_id in binding.slot.gpu_ids:
                field = _gpu_field(binding.slot.node, gpu_id)
                if field in mapping:
                    other = json.loads(mapping[field])["binding_id"]
                    raise GpuLeaseConflict(gpu=field, occupant=other)
                mapping[field] = payload
        args = [fence.lock_value]
        for field, payload in sorted(mapping.items()):
            args.extend((field, payload))
        result = self._redis.eval(
            _REBUILD_GPU_SCRIPT,
            2,
            rediskeys.SM_GPU_LEASES_KEY,
            rediskeys.SM_WRITER_LOCK_KEY,
            *args,
        )
        if int(result) == -1:
            raise StateFenceError("writer fence is no longer active")

    def load(self) -> list[GpuLease]:
        raw = self._redis.hgetall(rediskeys.SM_GPU_LEASES_KEY) or {}
        unique: dict[str, GpuLease] = {}
        for payload in raw.values():
            data = json.loads(_text(payload))
            data["gpu_ids"] = tuple(int(gpu) for gpu in data["gpu_ids"])
            lease = GpuLease(**data)
            unique[lease.binding_id] = lease
        return sorted(unique.values(), key=lambda item: item.binding_id)


def _gpu_field(node: str, gpu_id: int) -> str:
    return f"{node}/{gpu_id}"


def _text(value: object) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)
