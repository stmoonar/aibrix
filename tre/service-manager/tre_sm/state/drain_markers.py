"""Persistent ``draining`` markers for the staged (lock-releasing) sleep.

A binding being slept by the staged flow stays ``awake=True, hidden=True`` in
the legacy state (so every free-capacity check keeps treating its GPU as
occupied) and gets one marker here while the service-manager drains it
without holding the writer lock. The marker carries a fencing ``token``: the
commit phase only sleeps the binding when the marker it wrote is still there,
so a reclaim (target grew again), a wake or a drain recovery that ran in the
meantime is detected and wins.

All markers live in one JSON document under a single key (plain GET/SET, so
every Redis fake works). Real Redis writes go through a Lua script that
checks the service-manager writer fence like every other SM store; writes
only ever happen under the writer lock.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import threading
from typing import Mapping, Protocol

from tre_common import rediskeys
from tre_sm.state.operations import current_fence
from tre_sm.state.store import StateFenceError

# SM-private key (kept here instead of tre_common.rediskeys on purpose: no
# other component reads it; the controller sees draining via /v2/state).
DRAIN_MARKERS_KEY = "tre:v2:sm:draining"

_SAVE_MARKERS_SCRIPT = r"""
if ARGV[1] ~= '' and redis.call('GET', KEYS[2]) ~= ARGV[1] then
  return -1
end
redis.call('SET', KEYS[1], ARGV[2])
return 1
"""


class MarkerRedis(Protocol):
    def get(self, key: str): ...
    def set(self, key: str, value: str) -> None: ...


@dataclass(frozen=True)
class DrainMarker:
    binding_id: str
    serve_id: str
    model: str
    token: str
    instance: str
    started_at: float
    deadline_at: float
    reason: str
    operation_id: str | None = None
    prior_hidden: bool = False
    # The async operation (TRE_SM_ASYNC_OPS) that staged this sleep, if any: when
    # that operation is found orphaned (its SM died) the marker is recovered at
    # once instead of waiting for deadline + grace.
    async_op_id: str | None = None

    def to_dict(self) -> dict:
        record = asdict(self)
        if record.get("async_op_id") is None:
            # Keep the stored document identical to the pre-async format so an
            # older SM (DrainMarker(**item)) can still read it after a rollback.
            record.pop("async_op_id", None)
        return record


class DrainMarkerStore:
    def __init__(
        self,
        redis_client: MarkerRedis | None = None,
        *,
        key: str = DRAIN_MARKERS_KEY,
        require_fence: bool = False,
    ) -> None:
        self._redis = redis_client
        self._key = key
        self._require_fence = require_fence
        self._memory: dict[str, DrainMarker] = {}
        self._lock = threading.Lock()

    def load(self) -> dict[str, DrainMarker]:
        if self._redis is None:
            with self._lock:
                return dict(self._memory)
        raw = self._redis.get(self._key)
        if raw is None:
            return {}
        text = raw.decode("utf-8") if isinstance(raw, bytes) else str(raw)
        if not text.strip():
            return {}
        payload = json.loads(text)
        markers: dict[str, DrainMarker] = {}
        for binding_id, item in (payload or {}).items():
            markers[str(binding_id)] = DrainMarker(**item)
        return markers

    def save(self, markers: Mapping[str, DrainMarker]) -> None:
        if self._redis is None:
            with self._lock:
                self._memory = dict(markers)
            return
        document = json.dumps(
            {binding_id: marker.to_dict() for binding_id, marker in sorted(markers.items())},
            sort_keys=True,
            separators=(",", ":"),
        )
        fence = current_fence()
        if self._require_fence and fence is None:
            raise StateFenceError("draining marker write requires an active writer fence")
        eval_method = getattr(self._redis, "eval", None)
        if callable(eval_method) and fence is not None:
            result = eval_method(
                _SAVE_MARKERS_SCRIPT,
                2,
                self._key,
                rediskeys.SM_WRITER_LOCK_KEY,
                fence.lock_value,
                document,
            )
            if int(result) == -1:
                raise StateFenceError("writer fence is no longer active")
            return
        if self._require_fence:
            raise StateFenceError("atomic Redis EVAL is required for fenced marker writes")
        self._redis.set(self._key, document)
