"""Per-binding sleep reservations (review P1-3).

A sleep holds the SM writer lock only to hide the pod and, later, to call
/sleep and record the result. The gateway ack and the drain (up to the route
timeout) run WITHOUT the writer lock, so a slow scale-down of one model does
not block unrelated writes. While the lock is released, the draining binding is
fenced by a reservation instead:

* one record per binding in ``tre:v2:sm:sleep_reservations`` (field =
  binding_id) with the binding's node / GPUs, a random token, the owner and an
  expiry computed from Redis ``TIME`` (one clock for every SM replica);
* acquired all-or-nothing for the bindings of one sleep; refused when any live
  reservation covers the same binding or overlaps its GPUs;
* renewed on every drain poll (TTL ``service_manager.sleep.reservation_ttl_s``);
  a failed renewal means ownership was lost and the drain rolls back;
* checked by every conflicting operation: a wake on an overlapping GPU, another
  sleep / wake / hide / unhide of the binding, a defrag touching it, a model
  target of its model, startup admission on its GPUs, a fleet repair.

A reservation whose owner died expires by itself after the TTL.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import threading
import time
from typing import Callable, Iterable
from uuid import uuid4

from tre_common import rediskeys
from tre_sm.allocator.slots import Binding


_ACQUIRE_SCRIPT = r"""
local t = redis.call('TIME')
local now = tonumber(t[1]) * 1000 + math.floor(tonumber(t[2]) / 1000)
local count = tonumber(ARGV[5])
local wanted = {}
for i = 1, count do
  local base = 5 + (i - 1) * 4
  wanted[i] = {id = ARGV[base + 1], node = ARGV[base + 2],
               gpus = cjson.decode(ARGV[base + 3]), serve = ARGV[base + 4]}
end
local all = redis.call('HGETALL', KEYS[1])
for j = 1, #all, 2 do
  local rec = cjson.decode(all[j + 1])
  if tonumber(rec.expires_at_ms) <= now then
    redis.call('HDEL', KEYS[1], all[j])
  elseif rec.token ~= ARGV[1] then
    for i = 1, count do
      local w = wanted[i]
      if rec.binding_id == w.id then
        return {0, rec.binding_id}
      end
      if rec.node == w.node then
        for _, a in ipairs(rec.gpu_ids) do
          for _, b in ipairs(w.gpus) do
            if tonumber(a) == tonumber(b) then
              return {0, rec.binding_id}
            end
          end
        end
      end
    end
  end
end
local expires = now + tonumber(ARGV[4])
for i = 1, count do
  local w = wanted[i]
  redis.call('HSET', KEYS[1], w.id, cjson.encode({
    binding_id = w.id, serve_id = w.serve, node = w.node, gpu_ids = w.gpus,
    token = ARGV[1], owner = ARGV[2], operation_id = ARGV[3], expires_at_ms = expires
  }))
end
return {1, tostring(expires)}
"""

_RENEW_SCRIPT = r"""
local t = redis.call('TIME')
local now = tonumber(t[1]) * 1000 + math.floor(tonumber(t[2]) / 1000)
for i = 3, #ARGV do
  local raw = redis.call('HGET', KEYS[1], ARGV[i])
  if not raw then
    return 0
  end
  local rec = cjson.decode(raw)
  if rec.token ~= ARGV[1] or tonumber(rec.expires_at_ms) <= now then
    return 0
  end
end
for i = 3, #ARGV do
  local rec = cjson.decode(redis.call('HGET', KEYS[1], ARGV[i]))
  rec.expires_at_ms = now + tonumber(ARGV[2])
  redis.call('HSET', KEYS[1], ARGV[i], cjson.encode(rec))
end
return 1
"""

_RELEASE_SCRIPT = r"""
local released = 0
for i = 2, #ARGV do
  local raw = redis.call('HGET', KEYS[1], ARGV[i])
  if raw then
    local rec = cjson.decode(raw)
    if rec.token == ARGV[1] then
      redis.call('HDEL', KEYS[1], ARGV[i])
      released = released + 1
    end
  end
end
return released
"""


class ReservationConflict(RuntimeError):
    """A live sleep reservation covers the binding or overlaps its GPUs (HTTP 409)."""

    def __init__(self, message: str, *, binding_id: str | None = None) -> None:
        super().__init__(message)
        self.binding_id = binding_id


@dataclass(frozen=True)
class Reservation:
    binding_id: str
    serve_id: str
    node: str
    gpu_ids: tuple[int, ...]
    token: str
    owner: str
    operation_id: str | None
    expires_at_ms: int

    def overlaps(self, node: str, gpu_ids: Iterable[int]) -> bool:
        return node == self.node and bool(set(self.gpu_ids).intersection(int(g) for g in gpu_ids))


class SleepReservations:
    """Reservation store: Redis (Lua, Redis TIME) or in-process when redis is None."""

    def __init__(
        self,
        redis_client=None,
        *,
        wall_ms: Callable[[], int] | None = None,
    ) -> None:
        self._redis = redis_client
        self._wall_ms = wall_ms or (lambda: int(time.time() * 1000))
        self._memory: dict[str, dict] = {}
        self._lock = threading.Lock()

    # ------------------------------------------------------------- mutation
    def acquire(
        self,
        bindings: list[Binding],
        *,
        owner: str,
        operation_id: str | None,
        ttl_s: float,
    ) -> str:
        """All-or-nothing reservation of ``bindings``; returns the token."""
        token = uuid4().hex
        ttl_ms = max(1, int(ttl_s * 1000))
        if self._redis is None:
            with self._lock:
                now = self._wall_ms()
                self._expire_memory(now)
                for record in self._memory.values():
                    self._raise_on_conflict(record, bindings, token)
                for binding in bindings:
                    self._memory[binding.binding_id] = _record(
                        binding, token, owner, operation_id, now + ttl_ms
                    )
            return token
        args: list[str] = [token, owner, operation_id or "", str(ttl_ms), str(len(bindings))]
        for binding in bindings:
            args.extend(
                (
                    binding.binding_id,
                    binding.slot.node,
                    json.dumps(list(binding.slot.gpu_ids)),
                    binding.serve_id,
                )
            )
        result = self._redis.eval(_ACQUIRE_SCRIPT, 1, rediskeys.SM_SLEEP_RESERVATIONS_KEY, *args)
        if int(result[0]) != 1:
            holder = _text(result[1])
            raise ReservationConflict(
                f"binding {holder} is reserved by a sleep in progress", binding_id=holder
            )
        return token

    def renew(self, binding_ids: list[str], token: str, *, ttl_s: float) -> bool:
        if not binding_ids:
            return True
        ttl_ms = max(1, int(ttl_s * 1000))
        if self._redis is None:
            with self._lock:
                now = self._wall_ms()
                records = [self._memory.get(binding_id) for binding_id in binding_ids]
                if any(
                    record is None
                    or record["token"] != token
                    or int(record["expires_at_ms"]) <= now
                    for record in records
                ):
                    return False
                for record in records:
                    record["expires_at_ms"] = now + ttl_ms
            return True
        result = self._redis.eval(
            _RENEW_SCRIPT, 1, rediskeys.SM_SLEEP_RESERVATIONS_KEY, token, str(ttl_ms), *binding_ids
        )
        return int(result) == 1

    def release(self, binding_ids: list[str], token: str) -> None:
        if not binding_ids:
            return
        if self._redis is None:
            with self._lock:
                for binding_id in binding_ids:
                    record = self._memory.get(binding_id)
                    if record is not None and record["token"] == token:
                        self._memory.pop(binding_id, None)
            return
        self._redis.eval(_RELEASE_SCRIPT, 1, rediskeys.SM_SLEEP_RESERVATIONS_KEY, token, *binding_ids)

    # ---------------------------------------------------------------- reads
    def active(self) -> dict[str, Reservation]:
        """Unexpired reservations by binding_id."""
        if self._redis is None:
            with self._lock:
                now = self._wall_ms()
                self._expire_memory(now)
                raw = {key: dict(value) for key, value in self._memory.items()}
        else:
            now = self._redis_now_ms()
            raw = {}
            for field_name, payload in (self._redis.hgetall(rediskeys.SM_SLEEP_RESERVATIONS_KEY) or {}).items():
                try:
                    raw[_text(field_name)] = json.loads(_text(payload))
                except (TypeError, ValueError):
                    continue
        result: dict[str, Reservation] = {}
        for binding_id, record in raw.items():
            try:
                reservation = _reservation(record)
            except (KeyError, TypeError, ValueError):
                continue
            if reservation.expires_at_ms > now:
                result[binding_id] = reservation
        return result

    def conflict(
        self,
        *,
        node: str,
        gpu_ids: Iterable[int],
        binding_id: str | None = None,
        model: str | None = None,
        exclude_token: str | None = None,
    ) -> Reservation | None:
        """The first live reservation covering ``binding_id``, a binding of ``model``
        or any GPU of ``node``/``gpu_ids``."""
        gpus = [int(gpu) for gpu in gpu_ids]
        for reservation in self.active().values():
            if exclude_token is not None and reservation.token == exclude_token:
                continue
            if binding_id is not None and reservation.binding_id == binding_id:
                return reservation
            if model is not None and reservation.binding_id.split("/", 1)[0] == model:
                return reservation
            if gpus and reservation.overlaps(node, gpus):
                return reservation
        return None

    def assert_free(
        self,
        *,
        node: str,
        gpu_ids: Iterable[int],
        binding_id: str | None = None,
        model: str | None = None,
        what: str,
    ) -> None:
        reservation = self.conflict(node=node, gpu_ids=gpu_ids, binding_id=binding_id, model=model)
        if reservation is not None:
            raise ReservationConflict(
                f"{what}: binding {reservation.binding_id} ({reservation.node}/"
                f"{','.join(str(g) for g in reservation.gpu_ids)}) is draining for sleep "
                f"(owner {reservation.owner})",
                binding_id=reservation.binding_id,
            )

    # -------------------------------------------------------------- helpers
    def _redis_now_ms(self) -> int:
        redis_time = getattr(self._redis, "time", None)
        if callable(redis_time):
            try:
                seconds, micros = redis_time()
                return int(seconds) * 1000 + int(micros) // 1000
            except Exception:
                pass
        return self._wall_ms()

    def _expire_memory(self, now: int) -> None:
        for binding_id in [key for key, value in self._memory.items() if int(value["expires_at_ms"]) <= now]:
            self._memory.pop(binding_id, None)

    @staticmethod
    def _raise_on_conflict(record: dict, bindings: list[Binding], token: str) -> None:
        if record["token"] == token:
            return
        for binding in bindings:
            if record["binding_id"] == binding.binding_id or (
                record["node"] == binding.slot.node
                and set(record["gpu_ids"]).intersection(binding.slot.gpu_ids)
            ):
                raise ReservationConflict(
                    f"binding {record['binding_id']} is reserved by a sleep in progress",
                    binding_id=record["binding_id"],
                )


def _record(binding: Binding, token: str, owner: str, operation_id: str | None, expires: int) -> dict:
    return {
        "binding_id": binding.binding_id,
        "serve_id": binding.serve_id,
        "node": binding.slot.node,
        "gpu_ids": list(binding.slot.gpu_ids),
        "token": token,
        "owner": owner,
        "operation_id": operation_id or "",
        "expires_at_ms": expires,
    }


def _reservation(record: dict) -> Reservation:
    return Reservation(
        binding_id=str(record["binding_id"]),
        serve_id=str(record.get("serve_id", "")),
        node=str(record["node"]),
        gpu_ids=tuple(int(gpu) for gpu in record.get("gpu_ids") or ()),
        token=str(record["token"]),
        owner=str(record.get("owner", "")),
        operation_id=(str(record["operation_id"]) or None) if record.get("operation_id") else None,
        expires_at_ms=int(float(record["expires_at_ms"])),
    )


def _text(value: object) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)
