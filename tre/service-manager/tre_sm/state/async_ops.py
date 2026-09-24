"""Asynchronous service-manager operations (``TRE_SM_ASYNC_OPS``, default off).

With the flag on, ``PUT /v2/models/{m}/target`` and ``PUT /v2/bindings/{id}/power``
accept ``?async=1`` (or ``Prefer: respond-async``): the request is validated, persisted
here as an operation record and answered with ``202 {operation_id, plan}``; a
background worker then runs the (staged) sleep / wake. ``GET /v2/operations/{id}``
returns the record.

Scheduling (per model):

* Operations of different models run concurrently (the writer lock still serialises
  their short locked phases).
* Per model at most one operation runs in a LOCKED stage (plan + hide + wakes, or the
  commit). An operation whose staged sleep entered its lock-free ``draining`` stage no
  longer blocks the model: the next queued operation of that model starts right away
  (latest wins - the staged sleep's phase 1 reclaims draining bindings that are wanted
  again, and the older operation's commit sees its token gone and abandons them, so
  its status ends ``superseded``). A feasible wake therefore never waits behind
  another binding's drain.
* Queued (not yet started) operations collapse: a newer model target replaces an
  older queued model target of the same model, a newer binding power request an
  older queued one of the same binding (the older one ends ``superseded``).

Records live in one Redis hash (``tre:v2:sm:async_ops``; SM-private bookkeeping, not
fleet state, so plain HSET without the writer fence). Active records carry the owning
SM instance and a heartbeat; ``recover_orphans`` (supervisor tick / reconcile) marks
records of a dead instance ``failed`` so the service can finish their draining sleeps
through the drain recovery (desired state wins).
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import logging
import threading
import time
from typing import Callable, Mapping
from uuid import uuid4

ASYNC_OPS_KEY = "tre:v2:sm:async_ops"
ACTIVE_STATUSES = frozenset({"pending", "running"})
TERMINAL_STATUSES = frozenset({"succeeded", "failed", "superseded"})
# Stages in which a running operation holds (or waits for) the writer lock.
DRAINING_STAGE = "draining"

_LOGGER = logging.getLogger("tre_sm.async_ops")
_TRUTHY = {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class AsyncOpsConfig:
    enabled: bool = False
    # How long an operation keeps retrying a busy writer lock before it fails.
    lock_wait_s: float = 120.0
    lock_retry_s: float = 0.5
    heartbeat_s: float = 5.0
    # An active record of ANOTHER SM instance whose heartbeat is older than this is
    # orphaned (that instance died or was replaced mid-operation).
    orphan_after_s: float = 60.0
    max_records: int = 500

    def __post_init__(self) -> None:
        for name in ("lock_wait_s", "lock_retry_s", "heartbeat_s", "orphan_after_s"):
            if not float(getattr(self, name)) > 0:
                raise ValueError(f"AsyncOpsConfig.{name} must be positive")
        if self.orphan_after_s <= self.heartbeat_s:
            raise ValueError("AsyncOpsConfig.orphan_after_s must exceed heartbeat_s")
        if int(self.max_records) < 10:
            raise ValueError("AsyncOpsConfig.max_records must be at least 10")

    @classmethod
    def from_env(cls, env: Mapping[str, str]) -> "AsyncOpsConfig":
        defaults = cls()
        return cls(
            enabled=str(env.get("TRE_SM_ASYNC_OPS", "")).strip().lower() in _TRUTHY,
            lock_wait_s=_env_float(env, "TRE_SM_ASYNC_LOCK_WAIT_S", defaults.lock_wait_s),
            heartbeat_s=_env_float(env, "TRE_SM_ASYNC_HEARTBEAT_S", defaults.heartbeat_s),
            orphan_after_s=_env_float(
                env, "TRE_SM_ASYNC_ORPHAN_AFTER_S", defaults.orphan_after_s
            ),
            max_records=int(
                _env_float(env, "TRE_SM_ASYNC_MAX_RECORDS", float(defaults.max_records))
            ),
        )


def _env_float(env: Mapping[str, str], name: str, default: float) -> float:
    raw = env.get(name)
    if raw is None or not str(raw).strip():
        return default
    try:
        return float(str(raw).strip())
    except ValueError as exc:
        raise ValueError(f"{name} must be a number, got {raw!r}") from exc


def _text(value: object) -> str:
    return value.decode("utf-8") if isinstance(value, bytes) else str(value)


def _iso(ts: float | None) -> str | None:
    if ts is None:
        return None
    return datetime.fromtimestamp(ts, timezone.utc).isoformat()


class AsyncOpJournal:
    """Operation records in one Redis hash (or memory without Redis)."""

    def __init__(self, redis_client=None, *, key: str = ASYNC_OPS_KEY) -> None:
        self._redis = redis_client
        self._key = key
        self._memory: dict[str, str] = {}
        self._lock = threading.Lock()

    def save(self, record: dict) -> None:
        document = json.dumps(record, sort_keys=True, separators=(",", ":"), default=str)
        if self._redis is None:
            with self._lock:
                self._memory[str(record["operation_id"])] = document
            return
        self._redis.hset(self._key, mapping={str(record["operation_id"]): document})

    def get(self, operation_id: str) -> dict | None:
        if self._redis is None:
            with self._lock:
                raw = self._memory.get(operation_id)
            return None if raw is None else json.loads(raw)
        hget = getattr(self._redis, "hget", None)
        if callable(hget):
            raw = hget(self._key, operation_id)
        else:
            raw = None
            for field, value in (self._redis.hgetall(self._key) or {}).items():
                if _text(field) == operation_id:
                    raw = value
                    break
        return None if raw is None else json.loads(_text(raw))

    def list(self) -> list[dict]:
        if self._redis is None:
            with self._lock:
                values = list(self._memory.values())
        else:
            values = list((self._redis.hgetall(self._key) or {}).values())
        records = [json.loads(_text(value)) for value in values]
        records.sort(key=lambda item: float(item.get("created_ts") or 0.0), reverse=True)
        return records

    def delete(self, operation_ids: list[str]) -> None:
        if not operation_ids:
            return
        if self._redis is None:
            with self._lock:
                for operation_id in operation_ids:
                    self._memory.pop(operation_id, None)
            return
        hdel = getattr(self._redis, "hdel", None)
        if callable(hdel):
            hdel(self._key, *operation_ids)


ProgressFn = Callable[[str, dict | None], None]
# executor(record, progress) -> (status, payload); payload is merged into the record.
Executor = Callable[[dict, ProgressFn], tuple[str, dict]]


class AsyncOperationManager:
    def __init__(
        self,
        journal: AsyncOpJournal,
        executor: Executor,
        *,
        config: AsyncOpsConfig,
        instance: str,
        wall_clock: Callable[[], float] = time.time,
    ) -> None:
        self._journal = journal
        self._executor = executor
        self._config = config
        self._instance = instance
        self._wall_clock = wall_clock
        self._lock = threading.RLock()
        self._records: dict[str, dict] = {}  # local, not yet finished
        self._queues: dict[str, deque[str]] = {}
        self._running: dict[str, dict[str, str]] = {}  # model -> {op_id: stage}
        self._threads: dict[str, threading.Thread] = {}
        self._done: dict[str, threading.Event] = {}
        self._heartbeat: threading.Thread | None = None

    @property
    def config(self) -> AsyncOpsConfig:
        return self._config

    # ------------------------------------------------------------------ API

    def submit(
        self,
        *,
        kind: str,
        model: str,
        target_key: str,
        request: dict,
        plan: dict | None = None,
    ) -> dict:
        now = self._wall_clock()
        record = {
            "operation_id": uuid4().hex,
            "async": True,
            "kind": kind,
            "model": model,
            "target_key": target_key,
            "request": dict(request),
            "plan": dict(plan or {}),
            "status": "pending",
            "phase": "queued",
            "instance": self._instance,
            "created_ts": now,
            "created_at": _iso(now),
            "updated_ts": now,
            "heartbeat_ts": now,
            "supersedes": [],
        }
        with self._lock:
            queue = self._queues.setdefault(model, deque())
            for older_id in list(queue):
                older = self._records.get(older_id)
                if older is not None and older.get("target_key") == target_key:
                    queue.remove(older_id)
                    record["supersedes"].append(older_id)
                    self._finish_locked(
                        older,
                        status="superseded",
                        payload={"superseded_by": record["operation_id"]},
                    )
            self._records[record["operation_id"]] = record
            self._done[record["operation_id"]] = threading.Event()
            queue.append(record["operation_id"])
            self._journal.save(record)
            self._pump_locked(model)
            self._ensure_heartbeat_locked()
        self._prune()
        return dict(record)

    def get(self, operation_id: str) -> dict | None:
        with self._lock:
            local = self._records.get(operation_id)
            if local is not None:
                return dict(local)
        return self._journal.get(operation_id)

    def list(self, *, limit: int = 100) -> list[dict]:
        return self._journal.list()[:limit]

    def wait(self, operation_id: str, *, timeout_s: float | None = None) -> bool:
        event = self._done.get(operation_id)
        if event is None:
            return True
        return event.wait(timeout_s)

    def active_models(self) -> set[str]:
        with self._lock:
            return {
                model
                for model in set(self._queues) | set(self._running)
                if self._queues.get(model) or self._running.get(model)
            }

    def recover_orphans(self) -> list[dict]:
        """Mark active records of dead SM instances ``failed`` (returns them)."""
        now = self._wall_clock()
        orphaned: list[dict] = []
        for record in self._journal.list():
            if record.get("status") not in ACTIVE_STATUSES:
                continue
            operation_id = str(record.get("operation_id"))
            with self._lock:
                if operation_id in self._records:
                    continue  # ours and alive
            heartbeat = float(record.get("heartbeat_ts") or record.get("updated_ts") or 0.0)
            if (
                record.get("instance") != self._instance
                and now - heartbeat < self._config.orphan_after_s
            ):
                continue  # another live instance (e.g. mid rolling update)
            phase = record.get("phase")
            record.update(
                {
                    "status": "failed",
                    "phase": "orphaned",
                    "orphaned_in_phase": phase,
                    "recovered_by": self._instance,
                    "error": (
                        "service-manager instance "
                        f"{record.get('instance')} stopped during phase {phase!r}; "
                        + (
                            "never started, nothing was changed"
                            if phase == "queued"
                            else "the desired state it persisted is finished by the "
                            "drain recovery / reconcile"
                        )
                    ),
                    "updated_ts": now,
                    "finished_ts": now,
                    "finished_at": _iso(now),
                }
            )
            self._journal.save(record)
            _LOGGER.warning("async op orphaned: %s", json.dumps(record, default=str))
            orphaned.append(record)
        return orphaned

    # ------------------------------------------------------------ internals

    def _pump_locked(self, model: str) -> None:
        queue = self._queues.get(model)
        running = self._running.setdefault(model, {})
        while queue:
            if any(stage != DRAINING_STAGE for stage in running.values()):
                return
            operation_id = queue.popleft()
            record = self._records.get(operation_id)
            if record is None:
                continue
            running[operation_id] = "starting"
            thread = threading.Thread(
                target=self._run,
                args=(operation_id,),
                name=f"tre-sm-async-{operation_id[:8]}",
                daemon=True,
            )
            self._threads[operation_id] = thread
            thread.start()

    def _progress(self, operation_id: str, stage: str, details: dict | None = None) -> None:
        with self._lock:
            record = self._records.get(operation_id)
            if record is None:
                return
            model = record["model"]
            running = self._running.setdefault(model, {})
            if operation_id in running:
                running[operation_id] = stage
            now = self._wall_clock()
            record["phase"] = stage
            record["updated_ts"] = now
            record["heartbeat_ts"] = now
            if details:
                record.setdefault("progress", []).append({"phase": stage, **details})
            self._journal.save(record)
            if stage == DRAINING_STAGE:
                self._pump_locked(model)

    def _run(self, operation_id: str) -> None:
        with self._lock:
            record = self._records[operation_id]
            now = self._wall_clock()
            record.update(
                {
                    "status": "running",
                    "phase": "running",
                    "started_ts": now,
                    "started_at": _iso(now),
                    "updated_ts": now,
                    "heartbeat_ts": now,
                }
            )
            self._journal.save(record)
            snapshot = dict(record)
        status = "failed"
        payload: dict = {}
        try:
            status, payload = self._executor(
                snapshot,
                lambda stage, details=None: self._progress(operation_id, stage, details),
            )
        except Exception as exc:  # the record is the result channel
            payload = {"error": f"{type(exc).__name__}: {exc}"}
            status = "failed"
        with self._lock:
            model = record["model"]
            self._running.get(model, {}).pop(operation_id, None)
            self._finish_locked(record, status=status, payload=payload)
            self._pump_locked(model)

    def _finish_locked(self, record: dict, *, status: str, payload: dict) -> None:
        now = self._wall_clock()
        record.update(payload)
        record["status"] = status
        record["phase"] = status
        record["updated_ts"] = now
        record["finished_ts"] = now
        record["finished_at"] = _iso(now)
        if record.get("started_ts") is not None:
            record["duration_s"] = round(now - float(record["started_ts"]), 3)
        record["latency_s"] = round(now - float(record["created_ts"]), 3)
        self._journal.save(record)
        operation_id = record["operation_id"]
        self._records.pop(operation_id, None)
        self._threads.pop(operation_id, None)
        event = self._done.pop(operation_id, None)
        if event is not None:
            event.set()
        _LOGGER.info(
            json.dumps(
                {
                    "event": "sm_async_op",
                    "operation_id": operation_id,
                    "kind": record.get("kind"),
                    "model": record.get("model"),
                    "status": status,
                    "latency_s": record.get("latency_s"),
                    "summary": record.get("summary"),
                    "error": record.get("error"),
                },
                default=str,
                separators=(",", ":"),
            )
        )

    def _ensure_heartbeat_locked(self) -> None:
        if self._heartbeat is not None and self._heartbeat.is_alive():
            return
        self._heartbeat = threading.Thread(
            target=self._heartbeat_loop, name="tre-sm-async-heartbeat", daemon=True
        )
        self._heartbeat.start()

    def _heartbeat_loop(self) -> None:
        while True:
            time.sleep(self._config.heartbeat_s)
            with self._lock:
                if not self._records:
                    self._heartbeat = None
                    return
                now = self._wall_clock()
                for record in self._records.values():
                    record["heartbeat_ts"] = now
                    self._journal.save(record)

    def _prune(self) -> None:
        try:
            records = self._journal.list()
            terminal = [
                record for record in records if record.get("status") in TERMINAL_STATUSES
            ]
            excess = len(records) - int(self._config.max_records)
            if excess > 0:
                victims = sorted(
                    terminal, key=lambda item: float(item.get("created_ts") or 0.0)
                )[:excess]
                self._journal.delete([str(item["operation_id"]) for item in victims])
        except Exception:  # pragma: no cover - pruning is best effort
            _LOGGER.exception("async op journal pruning failed")
