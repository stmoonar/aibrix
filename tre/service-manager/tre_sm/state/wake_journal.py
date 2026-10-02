"""Journal of the wakes (crash evidence).

A wake holds the service-manager writer lock from its checks to its commit
(whole-lock, 2026-10-02): the account is checked, the binding's ``awake`` GPU
lease taken and this entry written, ``/wake_up`` + ``/is_sleeping`` run (every
binding of the request concurrently), and the outcome is recorded - all in one
lock hold. An entry outlives the lock only when the service-manager died
mid-wake or the engine's state could not be read; it is resolved by
``ServiceManagerV2.recover_wake_journal`` from the pod's physical state
(``/is_sleeping``): awake -> the wake is completed, asleep -> rolled back
(desired power restored from the entry), unknown -> kept for the next pass.

Redis HASH ``tre:v2:sm:wake_ops`` (field = binding id), or process memory.
"""

from __future__ import annotations

import json
import threading

from tre_common import rediskeys


class WakeJournal:
    def __init__(
        self,
        redis_client=None,
        *,
        key: str = rediskeys.SM_WAKE_OPS_KEY,
        stats_key: str = rediskeys.SM_WAKE_STATS_KEY,
    ) -> None:
        self._redis = redis_client
        self._key = key
        self._stats_key = stats_key
        self._memory: dict[str, dict] = {}
        self._stats: dict[str, int] = {}
        self._lock = threading.Lock()

    # ------------------------------------------------------------ counters
    def incr(self, name: str, amount: int = 1) -> None:
        """Best effort: a counter never fails a wake."""
        with self._lock:
            self._stats[name] = self._stats.get(name, 0) + int(amount)
        if self._redis is not None:
            try:
                self._redis.hincrby(self._stats_key, name, int(amount))
            except Exception:  # noqa: BLE001 - observability only
                pass

    def record_max(self, name: str, value: int) -> None:
        """Keep the largest ``value`` seen under ``name`` (e.g. wake_parallel_max)."""
        with self._lock:
            self._stats[name] = max(self._stats.get(name, 0), int(value))
        if self._redis is not None:
            try:
                current = self._redis.hget(self._stats_key, name)
                if current is None or int(_text(current)) < int(value):
                    self._redis.hset(self._stats_key, name, int(value))
            except Exception:  # noqa: BLE001 - observability only
                pass

    def stats(self) -> dict[str, int]:
        if self._redis is None:
            with self._lock:
                return dict(self._stats)
        try:
            raw = self._redis.hgetall(self._stats_key) or {}
            return {_text(key): int(_text(value)) for key, value in raw.items()}
        except Exception:  # noqa: BLE001 - fall back to this process's counters
            with self._lock:
                return dict(self._stats)

    def begin(self, binding_id: str, record: dict) -> None:
        self._write(binding_id, dict(record))

    def update(self, binding_id: str, **fields) -> None:
        record = dict(self.get(binding_id) or {})
        record.update(fields)
        self._write(binding_id, record)

    def end(self, binding_id: str) -> None:
        with self._lock:
            self._memory.pop(binding_id, None)
        if self._redis is not None:
            self._redis.hdel(self._key, binding_id)

    def get(self, binding_id: str) -> dict | None:
        if self._redis is None:
            with self._lock:
                record = self._memory.get(binding_id)
            return None if record is None else dict(record)
        raw = self._redis.hget(self._key, binding_id)
        return None if raw is None else _decode(raw)

    def entries(self) -> dict[str, dict]:
        if self._redis is None:
            with self._lock:
                return {key: dict(value) for key, value in self._memory.items()}
        return {
            _text(field_name): _decode(raw)
            for field_name, raw in (self._redis.hgetall(self._key) or {}).items()
        }

    def _write(self, binding_id: str, record: dict) -> None:
        with self._lock:
            self._memory[binding_id] = record
        if self._redis is not None:
            self._redis.hset(self._key, binding_id, json.dumps(record, sort_keys=True))


class RestartLedger:
    """Container restart counts last seen per Pod UID (Redis HASH
    ``tre:v2:sm:restart_seen``, or process memory): the restart guard compares
    against it, so a restart while the service-manager was down is still seen."""

    def __init__(self, redis_client=None, *, key: str = rediskeys.SM_RESTART_SEEN_KEY) -> None:
        self._redis = redis_client
        self._key = key
        self._memory: dict[str, int] = {}

    def load(self) -> dict[str, int]:
        if self._redis is None:
            return dict(self._memory)
        out: dict[str, int] = {}
        for field_name, raw in (self._redis.hgetall(self._key) or {}).items():
            try:
                out[_text(field_name)] = int(_text(raw))
            except (TypeError, ValueError):
                continue
        return out

    def set(self, uid: str, count: int) -> None:
        self._memory[uid] = int(count)
        if self._redis is not None:
            self._redis.hset(self._key, uid, int(count))

    def drop(self, uid: str) -> None:
        self._memory.pop(uid, None)
        if self._redis is not None:
            self._redis.hdel(self._key, uid)


def _decode(raw) -> dict:
    try:
        value = json.loads(_text(raw))
    except (TypeError, ValueError):
        return {"corrupt": True}
    return value if isinstance(value, dict) else {"corrupt": True}


def _text(value) -> str:
    return value.decode("utf-8") if isinstance(value, bytes) else str(value)
