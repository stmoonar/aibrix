from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import threading
import time
from typing import Callable, Iterator, Mapping, Protocol
from uuid import uuid4

from tre_common import rediskeys


# Fair (first-come first-served) writer lock (review 2 P2-4). A caller that is
# willing to wait passes a ticket (ARGV[7]); it joins the waiter queue KEYS[4]
# (sorted by arrival) and keeps its place while it polls within ARGV[8] ms
# (KEYS[5]: ticket -> deadline, Redis TIME). The lock is granted only to the
# head of the queue - or, with no live waiter, to a caller without a ticket - so
# a drained sleep's commit phase is not starved by other writers polling faster.
_ACQUIRE_SCRIPT = r"""
local t = redis.call('TIME')
local now = tonumber(t[1]) * 1000 + math.floor(tonumber(t[2]) / 1000)
local ticket = ARGV[7]
local waiters = redis.call('ZRANGE', KEYS[4], 0, -1)
for _, waiter in ipairs(waiters) do
  local deadline = tonumber(redis.call('HGET', KEYS[5], waiter) or '0')
  if deadline <= now then
    redis.call('ZREM', KEYS[4], waiter)
    redis.call('HDEL', KEYS[5], waiter)
  end
end
if ticket ~= '' then
  if not redis.call('ZSCORE', KEYS[4], ticket) then
    redis.call('ZADD', KEYS[4], redis.call('INCR', KEYS[6]), ticket)
  end
  redis.call('HSET', KEYS[5], ticket, now + tonumber(ARGV[8]))
end
local holder = redis.call('GET', KEYS[1])
if holder then
  return {0, holder}
end
local head = redis.call('ZRANGE', KEYS[4], 0, 0)[1]
if head and head ~= ticket then
  return {0, 'queued:' .. head}
end
local token = redis.call('INCR', KEYS[2])
local value = ARGV[1] .. ':' .. tostring(token)
redis.call('SET', KEYS[1], value, 'PX', ARGV[2])
if ticket ~= '' then
  redis.call('ZREM', KEYS[4], ticket)
  redis.call('HDEL', KEYS[5], ticket)
end
local record = cjson.encode({
  operation_id=ARGV[3], kind=ARGV[4], owner=ARGV[1],
  fencing_token=token, status='running', phase='acquired',
  started_at=ARGV[5], updated_at=ARGV[5], request=cjson.decode(ARGV[6])
})
redis.call('HSET', KEYS[3], ARGV[3], record)
return {token, value}
"""

_LEAVE_QUEUE_SCRIPT = r"""
redis.call('ZREM', KEYS[1], ARGV[1])
redis.call('HDEL', KEYS[2], ARGV[1])
return 1
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
    def get(self, key: str): ...


@dataclass(frozen=True)
class WriterFence:
    operation_id: str
    owner: str
    token: int
    lock_value: str


_CURRENT_FENCE: ContextVar[WriterFence | None] = ContextVar(
    "tre_sm_current_fence", default=None
)
_CURRENT_OPERATION: ContextVar["OperationHandle | None"] = ContextVar(
    "tre_sm_current_operation", default=None
)


#: Who called the SM HTTP API (X-TRE-Actor header, else User-Agent + remote
#: address), set per request by the API middleware (2026-09-28). Recorded as
#: ``request.actor`` of every operation the call starts.
_CURRENT_ACTOR: ContextVar[str | None] = ContextVar("tre_sm_current_actor", default=None)


def set_current_actor(actor: str | None):
    return _CURRENT_ACTOR.set(actor)


def reset_current_actor(token) -> None:
    _CURRENT_ACTOR.reset(token)


def current_fence() -> WriterFence | None:
    return _CURRENT_FENCE.get()


def current_operation() -> "OperationHandle | None":
    return _CURRENT_OPERATION.get()


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
        request: dict | None = None,
    ) -> None:
        self._coordinator = coordinator
        self.operation_id = operation_id
        self.kind = kind
        self.fence = fence
        self.started_at = started_at
        self.request = dict(request or {})
        #: Details that stay on the record whatever the phase (``note``): what the
        #: operation changed (binding, placement, phase durations, error code).
        self._notes: dict = {}
        self._phase = "acquired"
        self._phase_details: dict | None = None
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
        self._phase = phase
        self._phase_details = dict(details) if details else None
        record = self._record(status="running", phase=phase, details=details)
        if not self._coordinator._update(self.fence, record):
            self._lost.set()
            self.assert_active()

    def note(self, **fields) -> None:
        """Add details that stay on the operation record until it finishes (the
        phase's own details win on a key clash). Best effort: a lost fence is
        reported by the operation's end, not here."""
        if self._lost.is_set():
            return
        self._notes.update(fields)
        record = self._record(status="running", phase=self._phase, details=self._phase_details)
        if not self._coordinator._update(self.fence, record):
            self._lost.set()

    def supersede(self, operation_id: str) -> None:
        """Close a journal entry left running by a dead service-manager."""
        self.assert_active()
        if not self._coordinator._supersede(
            self.fence, operation_id, replacement_id=self.operation_id
        ):
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
        merged = {**self._notes, **(details or {})}
        if merged:
            record["details"] = merged
        if self.request:
            record["request"] = self.request
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
        poll_interval_s: float = 0.1,
        waiter_ttl_ms: int = 2_000,
        max_records: int | None = None,
    ) -> None:
        if lease_ttl_ms < 3_000:
            raise ValueError("lease_ttl_ms must be at least 3000")
        if waiter_ttl_ms <= poll_interval_s * 1000:
            raise ValueError("waiter_ttl_ms must exceed the poll interval")
        self._redis = redis_client
        self.owner = owner
        self.lease_ttl_ms = lease_ttl_ms
        #: A waiting caller re-polls this often ...
        self.poll_interval_s = float(poll_interval_s)
        #: ... and loses its queue place when it stops polling for this long.
        self.waiter_ttl_ms = int(waiter_ttl_ms)
        self.renew_interval_s = lease_ttl_ms / 3000.0
        #: Keep at most this many records in the journal (registry
        #: service_manager.operations.max_records); None = never trimmed.
        self.max_records = None if max_records is None else max(1, int(max_records))
        self._finished_since_trim = 0
        self._submitted: dict[str, threading.Thread] = {}
        self._submitted_lock = threading.Lock()

    @contextmanager
    def operation(
        self, kind: str, *, request: dict | None = None, wait_s: float = 0.0
    ) -> Iterator[OperationHandle]:
        handle = self.acquire(kind, request=request, wait_s=wait_s)
        fence_token = _CURRENT_FENCE.set(handle.fence)
        operation_token = _CURRENT_OPERATION.set(handle)
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
                _CURRENT_OPERATION.reset(operation_token)
                _CURRENT_FENCE.reset(fence_token)

    def submit(
        self,
        kind: str,
        target: Callable[[OperationHandle], None],
        *,
        request: dict | None = None,
    ) -> str:
        """Acquire the writer fence synchronously, then run work in background."""
        handle = self.acquire(kind, request=request)
        thread = threading.Thread(
            target=self._run_submitted,
            args=(handle, target),
            name=f"tre-sm-operation-{handle.operation_id}",
            daemon=True,
        )
        with self._submitted_lock:
            self._submitted[handle.operation_id] = thread
        thread.start()
        return handle.operation_id

    def wait(self, operation_id: str, *, timeout_s: float | None = None) -> bool:
        with self._submitted_lock:
            thread = self._submitted.get(operation_id)
        if thread is None:
            return True
        thread.join(timeout=timeout_s)
        return not thread.is_alive()

    def acquire(
        self, kind: str, *, request: dict | None = None, wait_s: float = 0.0
    ) -> OperationHandle:
        """Take the writer lock; wait up to ``wait_s`` while another writer holds
        it (phases hold it briefly), then raise OperationBusy. Waiters are served
        first-come first-served (a ticket in the Redis waiter queue); a caller
        that does not wait never jumps a queued waiter."""
        actor = _CURRENT_ACTOR.get()
        if actor and "actor" not in (request or {}):
            request = {**(request or {}), "actor": actor}
        wait_s = max(0.0, float(wait_s))
        deadline = time.monotonic() + wait_s
        ticket = uuid4().hex if wait_s > 0 else ""
        try:
            while True:
                try:
                    handle = self._acquire_once(kind, request=request, ticket=ticket)
                    ticket = ""  # dequeued by the script
                    return handle
                except OperationBusy:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise
                    time.sleep(min(self.poll_interval_s, remaining))
        finally:
            if ticket:
                self._leave_queue(ticket)

    def _leave_queue(self, ticket: str) -> None:
        try:
            self._redis.eval(
                _LEAVE_QUEUE_SCRIPT,
                2,
                rediskeys.SM_WRITER_QUEUE_KEY,
                rediskeys.SM_WRITER_QUEUE_DEADLINES_KEY,
                ticket,
            )
        except Exception:  # its deadline expires by itself
            pass

    def _acquire_once(
        self, kind: str, *, request: dict | None = None, ticket: str = ""
    ) -> OperationHandle:
        operation_id = str(uuid4())
        started_at = _utc_now()
        result = self._redis.eval(
            _ACQUIRE_SCRIPT,
            6,
            rediskeys.SM_WRITER_LOCK_KEY,
            rediskeys.SM_FENCE_COUNTER_KEY,
            rediskeys.SM_OPERATIONS_KEY,
            rediskeys.SM_WRITER_QUEUE_KEY,
            rediskeys.SM_WRITER_QUEUE_DEADLINES_KEY,
            rediskeys.SM_WRITER_QUEUE_SEQ_KEY,
            self.owner,
            str(self.lease_ttl_ms),
            operation_id,
            kind,
            started_at,
            json.dumps(request or {}, sort_keys=True, separators=(",", ":")),
            ticket,
            str(self.waiter_ttl_ms),
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
            request=request,
        )

    def list_operations(self, *, limit: int = 100) -> list[dict]:
        raw = self._redis.hgetall(rediskeys.SM_OPERATIONS_KEY) or {}
        records = [json.loads(_text(value)) for value in raw.values()]
        records.sort(key=lambda item: item.get("started_at", ""), reverse=True)
        return records[:limit]

    def get_operation(self, operation_id: str) -> dict | None:
        raw = self._redis.hget(rediskeys.SM_OPERATIONS_KEY, operation_id)
        return None if raw is None else json.loads(_text(raw))

    def active_operation(self, *, kind: str | None = None) -> dict | None:
        """Return the journal record protected by the live writer lease."""
        raw_lock = self._redis.get(rediskeys.SM_WRITER_LOCK_KEY)
        if raw_lock is None:
            return None
        lock_value = _text(raw_lock)
        for record in self.list_operations(limit=1000):
            expected = f"{record.get('owner')}:{record.get('fencing_token')}"
            if expected != lock_value or record.get("status") != "running":
                continue
            if kind is not None and record.get("kind") != kind:
                return None
            return record
        return None

    def stale_running_operations(self, *, kind: str | None = None) -> list[dict]:
        active = self.active_operation()
        active_id = None if active is None else active.get("operation_id")
        return [
            record
            for record in self.list_operations(limit=1000)
            if record.get("status") == "running"
            and record.get("operation_id") != active_id
            and (kind is None or record.get("kind") == kind)
        ]

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
        finished = int(result) == 1
        if finished:
            self._maybe_trim()
        return finished

    #: Trim the journal at most once per this many finished operations.
    TRIM_EVERY = 50

    def _maybe_trim(self) -> None:
        if self.max_records is None:
            return
        self._finished_since_trim += 1
        if self._finished_since_trim < self.TRIM_EVERY:
            return
        self._finished_since_trim = 0
        try:
            self.trim()
        except Exception:  # noqa: BLE001 - the next finish tries again
            pass

    def trim(self) -> int:
        """Drop the oldest finished records beyond ``max_records`` (running
        records are never dropped). Returns how many were dropped."""
        if self.max_records is None:
            return 0
        counter = getattr(self._redis, "hlen", None)
        if callable(counter) and int(counter(rediskeys.SM_OPERATIONS_KEY)) <= self.max_records:
            return 0
        records = self.list_operations(limit=10**9)  # newest first
        drop = [
            str(record.get("operation_id"))
            for record in records[self.max_records:]
            if record.get("status") != "running" and record.get("operation_id")
        ]
        for start in range(0, len(drop), 500):
            self._redis.hdel(rediskeys.SM_OPERATIONS_KEY, *drop[start:start + 500])
        return len(drop)

    def _supersede(
        self, fence: WriterFence, operation_id: str, *, replacement_id: str
    ) -> bool:
        raw = self._redis.hget(rediskeys.SM_OPERATIONS_KEY, operation_id)
        if raw is None:
            return True
        record = json.loads(_text(raw))
        if record.get("status") != "running":
            return True
        now = _utc_now()
        record.update(
            {
                "status": "superseded",
                "phase": "superseded",
                "updated_at": now,
                "finished_at": now,
                "replacement_operation_id": replacement_id,
            }
        )
        return self._update_record_under_fence(fence, operation_id, record)

    def _update_record_under_fence(
        self, fence: WriterFence, operation_id: str, record: dict
    ) -> bool:
        result = self._redis.eval(
            _UPDATE_SCRIPT,
            2,
            rediskeys.SM_WRITER_LOCK_KEY,
            rediskeys.SM_OPERATIONS_KEY,
            fence.lock_value,
            operation_id,
            json.dumps(record, sort_keys=True, separators=(",", ":")),
        )
        return int(result) == 1

    def _run_submitted(
        self,
        handle: OperationHandle,
        target: Callable[[OperationHandle], None],
    ) -> None:
        fence_token = _CURRENT_FENCE.set(handle.fence)
        operation_token = _CURRENT_OPERATION.set(handle)
        handle.start()
        error: BaseException | None = None
        try:
            target(handle)
            handle.assert_active()
        except BaseException as exc:  # background result is persisted in journal.
            error = exc
        finally:
            handle.stop()
            try:
                handle.finish(
                    status="failed" if error else "succeeded",
                    error=str(error) if error else None,
                )
            except OperationFenceLost:
                pass
            finally:
                _CURRENT_OPERATION.reset(operation_token)
                _CURRENT_FENCE.reset(fence_token)
                with self._submitted_lock:
                    self._submitted.pop(handle.operation_id, None)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _text(value: object) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)
