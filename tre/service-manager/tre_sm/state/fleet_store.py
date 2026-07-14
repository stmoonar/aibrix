from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
import json
from typing import Iterable, Mapping, Protocol

from tre_common import rediskeys
from tre_sm.allocator.slots import Binding, Slot
from tre_sm.state.operations import current_fence
from tre_sm.state.store import StateFenceError


_SAVE_HASH_SCRIPT = r"""
local current = tonumber(redis.call('GET', KEYS[2]) or '0')
local expected = tonumber(ARGV[1])
if current ~= expected then
  return {0, current}
end
if redis.call('GET', KEYS[3]) ~= ARGV[3] then
  return {-1, current}
end
redis.call('DEL', KEYS[1])
local index = 4
while index <= #ARGV do
  redis.call('HSET', KEYS[1], ARGV[index], ARGV[index + 1])
  index = index + 2
end
local next_version = tonumber(ARGV[2])
redis.call('SET', KEYS[2], tostring(next_version))
return {1, next_version}
"""


class FleetRedis(Protocol):
    def get(self, key: str): ...
    def hgetall(self, key: str) -> Mapping[object, object]: ...
    def eval(self, script: str, numkeys: int, *keys_and_args): ...


@dataclass(frozen=True)
class DesiredBinding:
    binding_id: str
    model: str
    node: str
    gpu_ids: tuple[int, ...]
    lifecycle: str
    power: str
    hidden: bool
    generation: int
    updated_at: str
    updated_by: str
    reason: str

    @classmethod
    def from_binding(
        cls,
        binding: Binding,
        *,
        generation: int = 1,
        updated_by: str = "bootstrap",
        reason: str = "legacy_state_migration",
    ) -> "DesiredBinding":
        return cls(
            binding_id=binding.binding_id,
            model=binding.model,
            node=binding.slot.node,
            gpu_ids=binding.slot.gpu_ids,
            lifecycle="resident",
            power="awake" if binding.awake else "sleeping",
            hidden=binding.hidden,
            generation=generation,
            updated_at=_utc_now(),
            updated_by=updated_by,
            reason=reason,
        )

    def with_intent(
        self,
        *,
        power: str | None = None,
        hidden: bool | None = None,
        lifecycle: str | None = None,
        updated_by: str,
        reason: str,
    ) -> "DesiredBinding":
        candidate = replace(
            self,
            power=self.power if power is None else power,
            hidden=self.hidden if hidden is None else hidden,
            lifecycle=self.lifecycle if lifecycle is None else lifecycle,
        )
        if (
            candidate.power == self.power
            and candidate.hidden == self.hidden
            and candidate.lifecycle == self.lifecycle
        ):
            return self
        return replace(
            candidate,
            generation=self.generation + 1,
            updated_at=_utc_now(),
            updated_by=updated_by,
            reason=reason,
        )


@dataclass(frozen=True)
class ObservedBinding:
    binding_id: str
    model: str
    node: str
    gpu_ids: tuple[int, ...]
    pod_name: str | None
    pod_uid: str | None
    pod_ip: str | None
    phase: str
    ready: bool
    restart_count: int
    physical_power: str
    routable: bool | None
    hidden: bool
    error: str | None = None

    @classmethod
    def from_binding(
        cls, binding: Binding, *, source: str = "legacy_bootstrap"
    ) -> "ObservedBinding":
        return cls(
            binding_id=binding.binding_id,
            model=binding.model,
            node=binding.slot.node,
            gpu_ids=binding.slot.gpu_ids,
            pod_name=binding.serve_id,
            pod_uid=None,
            pod_ip=None,
            phase="Running",
            ready=True,
            restart_count=0,
            physical_power="awake" if binding.awake else "sleeping",
            routable=binding.awake and not binding.hidden,
            hidden=binding.hidden,
            error=source if source != "legacy_bootstrap" else None,
        )


@dataclass(frozen=True)
class DesiredSnapshot:
    version: int
    bindings: list[DesiredBinding]


@dataclass(frozen=True)
class ObservedSnapshot:
    version: int
    bindings: list[ObservedBinding]


class FleetStateConflict(RuntimeError):
    def __init__(self, *, expected_version: int, current_version: int) -> None:
        super().__init__(
            f"fleet state conflict: expected {expected_version}, current {current_version}"
        )
        self.expected_version = expected_version
        self.current_version = current_version


class FleetStateStore:
    def __init__(self, redis_client: FleetRedis) -> None:
        self._redis = redis_client

    def load_desired(self) -> DesiredSnapshot:
        return DesiredSnapshot(
            version=self._version(rediskeys.SM_DESIRED_VERSION_KEY),
            bindings=self._load_records(
                rediskeys.SM_DESIRED_KEY, DesiredBinding
            ),
        )

    def load_observed(self) -> ObservedSnapshot:
        return ObservedSnapshot(
            version=self._version(rediskeys.SM_OBSERVED_VERSION_KEY),
            bindings=self._load_records(
                rediskeys.SM_OBSERVED_KEY, ObservedBinding
            ),
        )

    def save_desired(
        self,
        bindings: Iterable[DesiredBinding],
        *,
        expected_version: int,
    ) -> int:
        return self._save(
            state_key=rediskeys.SM_DESIRED_KEY,
            version_key=rediskeys.SM_DESIRED_VERSION_KEY,
            bindings=bindings,
            expected_version=expected_version,
        )

    def save_observed(
        self,
        bindings: Iterable[ObservedBinding],
        *,
        expected_version: int,
    ) -> int:
        return self._save(
            state_key=rediskeys.SM_OBSERVED_KEY,
            version_key=rediskeys.SM_OBSERVED_VERSION_KEY,
            bindings=bindings,
            expected_version=expected_version,
        )

    def bootstrap(self, legacy_bindings: list[Binding]) -> dict[str, int]:
        desired = self.load_desired()
        observed = self.load_observed()
        desired_version = desired.version
        observed_version = observed.version
        if desired.version == 0:
            desired_version = self.save_desired(
                [DesiredBinding.from_binding(binding) for binding in legacy_bindings],
                expected_version=desired.version,
            )
        if observed.version == 0:
            observed_version = self.save_observed(
                [ObservedBinding.from_binding(binding) for binding in legacy_bindings],
                expected_version=observed.version,
            )
        return {
            "desired_version": desired_version,
            "observed_version": observed_version,
        }

    def _save(
        self,
        *,
        state_key: str,
        version_key: str,
        bindings: Iterable[object],
        expected_version: int,
    ) -> int:
        fence = current_fence()
        if fence is None:
            raise StateFenceError("fleet state write requires an active writer fence")
        mapping: dict[str, str] = {}
        for binding in bindings:
            binding_id = str(getattr(binding, "binding_id"))
            if binding_id in mapping:
                raise ValueError(f"duplicate stable binding_id: {binding_id}")
            payload = asdict(binding)
            payload["gpu_ids"] = list(payload["gpu_ids"])
            mapping[binding_id] = json.dumps(
                payload, sort_keys=True, separators=(",", ":")
            )
        args = [
            str(expected_version),
            str(expected_version + 1),
            fence.lock_value,
        ]
        for field, payload in sorted(mapping.items()):
            args.extend((field, payload))
        result = self._redis.eval(
            _SAVE_HASH_SCRIPT,
            3,
            state_key,
            version_key,
            rediskeys.SM_WRITER_LOCK_KEY,
            *args,
        )
        status, value = int(result[0]), int(result[1])
        if status == 0:
            raise FleetStateConflict(
                expected_version=expected_version,
                current_version=value,
            )
        if status == -1:
            raise StateFenceError("writer fence is no longer active")
        return value

    def _version(self, key: str) -> int:
        raw = self._redis.get(key)
        return 0 if raw is None else int(_text(raw))

    def _load_records(self, key: str, record_type):
        raw = self._redis.hgetall(key) or {}
        records = []
        for raw_binding_id, raw_payload in sorted(
            raw.items(), key=lambda item: _text(item[0])
        ):
            payload = json.loads(_text(raw_payload))
            payload["gpu_ids"] = tuple(int(gpu) for gpu in payload["gpu_ids"])
            payload["binding_id"] = _text(raw_binding_id)
            records.append(record_type(**payload))
        return records


def desired_to_binding(desired: DesiredBinding, *, serve_id: str) -> Binding:
    return Binding(
        serve_id=serve_id,
        model=desired.model,
        slot=Slot(desired.node, desired.gpu_ids),
        awake=desired.power == "awake",
        hidden=desired.hidden,
    )


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _text(value: object) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)
