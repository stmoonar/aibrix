from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import threading
from typing import Iterator, Mapping, Protocol
from uuid import uuid4

from tre_common import rediskeys


_ACQUIRE_SCRIPT = r"""
local token = redis.call('INCR', KEYS[2])
local value = ARGV[1] .. ':' .. tostring(token)
local acquired = redis.call('SET', KEYS[1], value, 'NX', 'PX', ARGV[2])
if not acquired then
  return {0, redis.call('GET', KEYS[1]) or ''}
end
local record = cjson.encode({
  operation_id=ARGV[3], kind=ARGV[4], owner=ARGV[1],
  fencing_token=token, status='running', phase='acquired',
  started_at=ARGV[5], updated_at=ARGV[5]
})
redis.call('HSET', KEYS[3], ARGV[3], record)
return {token, value}
"""

_RENEW_SCRIPT = r"""
if redis.call('GET', KEYS[1]) ~= ARGV[1] then
  return 0
end
redis.call('PEXPIRE', KEYS[1], ARGV[2])
return 1
"""

_UPDATE_SCRIPT = r"""
if redis.call('GET', KEYS[1]) ~= ARGV[1] then
  return 0
end
redis.call('HSET', KEYS[2], ARGV[2], ARGV[3])
return 1
"""

_FINISH_SCRIPT = r"""
if redis.call('GET', KEYS[1]) ~= ARGV[1] then
  return 0
end
redis.call('HSET', KEYS[2], ARGV[2], ARGV[3])
redis.call('DEL', KEYS[1])
return 1
"""


class RedisOperationClient(Protocol):
    def eval(self, script: str, numkeys: int, *keys_and_args): ...
    def hgetall(self, key: str) -> Mapping[object, object]: ...
    def hget(self, key: str, field: str): ...


@dataclass(frozen=True)
class WriterFence:
    operation_id: str
    owner: str
    token: int
    lock_value: str


_CURRENT_FENCE: ContextVar[WriterFence | None] = ContextVar(
    "tre_sm_current_fence", default=None
)


def current_fence() -> WriterFence | None:
    return _CURRENT_FENCE.get()


class OperationBusy(RuntimeError):
    def __init__(self, current_writer: str) -> None:
        super().__init__(f"service-manager writer lease is held by {current_writer}")
        self.current_writer = current_writer


class OperationFenceLost(RuntimeError):
    pass


class OperationHandle:
    def __init__(
        self,
        coordinator: "OperationCoordinator",
        *,
        operation_id: str,
        kind: str,
        fence: WriterFence,
        started_at: str,
    ) -> None:
        self._coordinator = coordinator
        self.operation_id = operation_id
        self.kind = kind
        self.fence = fence
        self.started_at = started_at
        self._lost = threading.Event()
        self._stop = threading.Event()
        self._thread = threading.Thread(
            target=self._renew_loop,
            name=f"tre-sm-lease-{operation_id}",
            daemon=True,
        )

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=self._coordinator.renew_interval_s + 1.0)

    def assert_active(self) -> None:
        if self._lost.is_set():
            raise OperationFenceLost(
                f"writer fence lost for operation {self.operation_id}"
            )

    def advance(self, phase: str, *, details: dict | None = None) -> None:
        self.assert_active()
        record = self._record(status="running", phase=phase, details=details)
        if not self._coordinator._update(self.fence, record):
            self._lost.set()
            self.assert_active()

    def finish(self, *, status: str, error: str | None = None) -> None:
        record = self._record(
            status=status,
            phase=status,
            details={"error": error} if error else None,
            finished=True,
        )
        if not self._coordinator._finish(self.fence, record):
            self._lost.set()
            raise OperationFenceLost(
                f"writer fence lost while finishing operation {self.operation_id}"
            )

    def _record(
        self,
        *,
        status: str,
        phase: str,
        details: dict | None,
        finished: bool = False,
    ) -> dict:
        now = _utc_now()
        record = {
            "operation_id": self.operation_id,
            "kind": self.kind,
            "owner": self.fence.owner,
            "fencing_token": self.fence.token,
            "status": status,
            "phase": phase,
            "started_at": self.started_at,
            "updated_at": now,
        }
        if details:
            record["details"] = details
        if finished:
            record["finished_at"] = now
        return record

    def _renew_loop(self) -> None:
        while not self._stop.wait(self._coordinator.renew_interval_s):
            if not self._coordinator._renew(self.fence):
                self._lost.set()
                return


class OperationCoordinator:
    def __init__(
        self,
        redis_client: RedisOperationClient,
        *,
        owner: str,
        lease_ttl_ms: int = 30_000,
    ) -> None:
        if lease_ttl_ms < 3_000:
            raise ValueError("lease_ttl_ms must be at least 3000")
        self._redis = redis_client
        self.owner = owner
        self.lease_ttl_ms = lease_ttl_ms
        self.renew_interval_s = lease_ttl_ms / 3000.0

    @contextmanager
    def operation(self, kind: str) -> Iterator[OperationHandle]:
        handle = self.acquire(kind)
        token = _CURRENT_FENCE.set(handle.fence)
        handle.start()
        error: BaseException | None = None
        try:
            yield handle
            handle.assert_active()
        except BaseException as exc:
            error = exc
            raise
        finally:
            handle.stop()
            try:
                handle.finish(
                    status="failed" if error else "succeeded",
                    error=str(error) if error else None,
                )
            finally:
                _CURRENT_FENCE.reset(token)

    def acquire(self, kind: str) -> OperationHandle:
        operation_id = str(uuid4())
        started_at = _utc_now()
        result = self._redis.eval(
            _ACQUIRE_SCRIPT,
            3,
            rediskeys.SM_WRITER_LOCK_KEY,
            rediskeys.SM_FENCE_COUNTER_KEY,
            rediskeys.SM_OPERATIONS_KEY,
            self.owner,
            str(self.lease_ttl_ms),
            operation_id,
            kind,
            started_at,
        )
        token = int(result[0])
        lock_value = _text(result[1])
        if token == 0:
            raise OperationBusy(lock_value or "unknown")
        fence = WriterFence(
            operation_id=operation_id,
            owner=self.owner,
            token=token,
            lock_value=lock_value,
        )
        return OperationHandle(
            self,
            operation_id=operation_id,
            kind=kind,
            fence=fence,
            started_at=started_at,
        )

    def list_operations(self, *, limit: int = 100) -> list[dict]:
        raw = self._redis.hgetall(rediskeys.SM_OPERATIONS_KEY) or {}
        records = [json.loads(_text(value)) for value in raw.values()]
        records.sort(key=lambda item: item.get("started_at", ""), reverse=True)
        return records[:limit]

    def get_operation(self, operation_id: str) -> dict | None:
        raw = self._redis.hget(rediskeys.SM_OPERATIONS_KEY, operation_id)
        return None if raw is None else json.loads(_text(raw))

    def _renew(self, fence: WriterFence) -> bool:
        result = self._redis.eval(
            _RENEW_SCRIPT,
            1,
            rediskeys.SM_WRITER_LOCK_KEY,
            fence.lock_value,
            str(self.lease_ttl_ms),
        )
        return int(result) == 1

    def _update(self, fence: WriterFence, record: dict) -> bool:
        result = self._redis.eval(
            _UPDATE_SCRIPT,
            2,
            rediskeys.SM_WRITER_LOCK_KEY,
            rediskeys.SM_OPERATIONS_KEY,
            fence.lock_value,
            fence.operation_id,
            json.dumps(record, sort_keys=True, separators=(",", ":")),
        )
        return int(result) == 1

    def _finish(self, fence: WriterFence, record: dict) -> bool:
        result = self._redis.eval(
            _FINISH_SCRIPT,
            2,
            rediskeys.SM_WRITER_LOCK_KEY,
            rediskeys.SM_OPERATIONS_KEY,
            fence.lock_value,
            fence.operation_id,
            json.dumps(record, sort_keys=True, separators=(",", ":")),
        )
        return int(result) == 1


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _text(value: object) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)
