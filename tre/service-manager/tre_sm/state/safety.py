from __future__ import annotations

import json
import logging
import threading
import time
from typing import Callable, Iterable, Protocol

from tre_common import rediskeys
from tre_sm.state.actuation import SmActuation
from tre_sm.state.operations import OperationHandle

LOG = logging.getLogger(__name__)

#: Default TTL of the SM maintenance lock and its renewal period (2026-09-28):
#: a repair renews every 15 s (and at each safety check), so a dead SM's lock
#: expires by itself within 60 s. Env: TRE_SM_MAINTENANCE_TTL_S / _RENEW_S.
MAINTENANCE_TTL_S = 60.0
MAINTENANCE_RENEW_S = 15.0

# Maintenance lock scripts. The value is the JSON {operation_id, kind, owner,
# since_ms}; ownership is its ``operation_id``.
#
# Acquire (KEYS[1]; ARGV[1] value, ARGV[2] ttl ms, ARGV[3..] operation ids
# whose lock may be taken over - the caller's own id and the stale operations
# of a dead SM that the caller recovers). A free key is SET PX (NX); a held
# key is taken over only when its holder is one of ARGV[3..], when it has no
# TTL (written by a pre-TTL service-manager or by hand) or is not a JSON
# object; otherwise it is refused. Returns {1, ''} (free), {2, old} (taken
# over) or {0, current} (refused).
_MAINTENANCE_ACQUIRE_SCRIPT = r"""
local cur = redis.call('GET', KEYS[1])
if cur then
  local takeover = redis.call('PTTL', KEYS[1]) == -1
  if not takeover then
    local ok, rec = pcall(cjson.decode, cur)
    if not ok or type(rec) ~= 'table' then
      takeover = true
    else
      for i = 3, #ARGV do
        if rec.operation_id == ARGV[i] then
          takeover = true
        end
      end
    end
  end
  if not takeover then
    return {0, cur}
  end
end
redis.call('SET', KEYS[1], ARGV[1], 'PX', ARGV[2])
if cur then
  return {2, cur}
end
return {1, ''}
"""

# Renew (compare-and-pexpire): KEYS[1]; ARGV[1] operation id, ARGV[2] ttl ms.
_MAINTENANCE_RENEW_SCRIPT = r"""
local cur = redis.call('GET', KEYS[1])
if not cur then
  return 0
end
local ok, rec = pcall(cjson.decode, cur)
if not ok or type(rec) ~= 'table' or rec.operation_id ~= ARGV[1] then
  return 0
end
redis.call('PEXPIRE', KEYS[1], ARGV[2])
return 1
"""

# Release (compare-and-delete): KEYS[1]; ARGV[1] operation id.
_MAINTENANCE_RELEASE_SCRIPT = r"""
local cur = redis.call('GET', KEYS[1])
if not cur then
  return 0
end
local ok, rec = pcall(cjson.decode, cur)
if not ok or type(rec) ~= 'table' or rec.operation_id ~= ARGV[1] then
  return 0
end
redis.call('DEL', KEYS[1])
return 1
"""


class SafetyRedis(Protocol):
    def get(self, key: str): ...
    def set(self, key: str, value: str): ...
    def delete(self, *keys: str): ...
    def eval(self, script: str, numkeys: int, *keys_and_args): ...


class PressureSource(Protocol):
    def node_pressure_reasons(self) -> dict[str, list[str]]: ...


class MaintenanceLockLost(RuntimeError):
    """The SM maintenance lock of a running fleet repair was cleared, expired or
    taken over (an operator deleted ``tre:v2:sm:maintenance`` to abort it)."""


class MaintenanceLockBusy(RuntimeError):
    """The SM maintenance lock is held (with a live TTL) by another operation
    that the caller does not recover; the repair does not start."""

    def __init__(self, message: str, *, holder: dict | None = None) -> None:
        super().__init__(message)
        self.holder = holder


class _MaintenanceRenewer:
    """Background compare-and-pexpire of one held maintenance lock."""

    def __init__(self, renew: Callable[[], bool], interval_s: float, operation_id: str) -> None:
        self._renew = renew
        self._interval_s = interval_s
        self.lost = threading.Event()
        self._stop = threading.Event()
        self._thread = threading.Thread(
            target=self._loop, name=f"tre-sm-maintenance-{operation_id}", daemon=True
        )

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread.is_alive() and self._thread is not threading.current_thread():
            self._thread.join(timeout=self._interval_s + 1.0)

    def _loop(self) -> None:
        while not self._stop.wait(self._interval_s):
            try:
                held = self._renew()
            except Exception:  # transient Redis error: retry; the TTL bounds it
                LOG.warning("renewing the SM maintenance lock failed; retrying", exc_info=True)
                continue
            if not held:
                self.lost.set()
                return


class PressureWaitTimeout(RuntimeError):
    pass


class NodePressureActive(RuntimeError):
    pass


class ClusterSafetyGate:
    """Cluster-level guards of the service-manager: node pressure, the SM
    maintenance lock held by a fleet repair, and the SM actuation switch.

    The fleet repair used to require (and the supervisor used to force) the
    controller mode ``observe`` as its lock; it now holds its own maintenance
    lock ``tre:v2:sm:maintenance`` and never writes the controller mode (user
    decision 2026-09-28). While it runs, the repair also holds the SM writer
    lock, so every other SM write (controller, APA, operator) gets a retriable
    409 meanwhile."""

    def __init__(
        self,
        redis_client: SafetyRedis,
        pressure_source: PressureSource,
        *,
        clear_hysteresis_s: float = 60.0,
        pressure_timeout_s: float = 3600.0,
        poll_interval_s: float = 5.0,
        monotonic: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
        actuation: SmActuation | None = None,
        wall_ms: Callable[[], int] = lambda: int(time.time() * 1000),
        maintenance_ttl_s: float = MAINTENANCE_TTL_S,
        maintenance_renew_s: float = MAINTENANCE_RENEW_S,
    ) -> None:
        if maintenance_renew_s <= 0 or maintenance_ttl_s < 2 * maintenance_renew_s:
            raise ValueError(
                "maintenance lock: renew interval must be > 0 and at most half the TTL "
                f"(ttl {maintenance_ttl_s:g}s, renew {maintenance_renew_s:g}s)"
            )
        self._redis = redis_client
        self._pressure_source = pressure_source
        self._clear_hysteresis_s = clear_hysteresis_s
        self._pressure_timeout_s = pressure_timeout_s
        self._poll_interval_s = poll_interval_s
        self._monotonic = monotonic
        self._sleep = sleep
        self._wall_ms = wall_ms
        self._actuation = actuation if actuation is not None else SmActuation(redis_client)
        self._maintenance_ttl_ms = int(maintenance_ttl_s * 1000)
        self._maintenance_renew_s = float(maintenance_renew_s)
        self._renewers: dict[str, _MaintenanceRenewer] = {}
        self._renewers_lock = threading.Lock()

    # ------------------------------------------------------ maintenance lock
    def acquire_maintenance(
        self,
        operation_id: str,
        *,
        kind: str,
        owner: str = "",
        takeover_operation_ids: Iterable[str] = (),
    ) -> None:
        """Take the SM maintenance lock for ``operation_id`` atomically (Lua:
        SET PX when free) with a TTL that a background thread renews while the
        repair runs. A lock held by another operation is taken over only when
        that operation is one the caller recovers (``takeover_operation_ids``:
        the stale repairs of a dead SM), or when the key has no TTL (left by a
        pre-TTL SM) - otherwise :class:`MaintenanceLockBusy`. A lock whose SM
        died expires by itself after the TTL."""
        value = json.dumps(
            {"operation_id": operation_id, "kind": kind, "owner": owner, "since_ms": self._wall_ms()},
            sort_keys=True,
        )
        allowed = [operation_id, *(str(item) for item in takeover_operation_ids)]
        result = self._redis.eval(
            _MAINTENANCE_ACQUIRE_SCRIPT, 1, rediskeys.SM_MAINTENANCE_KEY,
            value, str(self._maintenance_ttl_ms), *allowed,
        )
        code = int(result[0])
        previous = _text(result[1]) if len(result) > 1 and result[1] is not None else ""
        if code == 0:
            holder = _parse_holder(previous)
            raise MaintenanceLockBusy(
                f"SM maintenance lock is held by {holder.get('operation_id', holder)}; "
                f"{kind} {operation_id} not started",
                holder=holder,
            )
        if code == 2:
            LOG.warning("SM maintenance lock taken over by %s (previous holder: %s)", operation_id, previous)
        renewer = _MaintenanceRenewer(
            lambda: self._renew_maintenance(operation_id), self._maintenance_renew_s, operation_id
        )
        with self._renewers_lock:
            stale = self._renewers.pop(operation_id, None)
            self._renewers[operation_id] = renewer
        if stale is not None:
            stale.stop()
        renewer.start()

    def _renew_maintenance(self, operation_id: str) -> bool:
        """Compare-and-pexpire: extend the TTL only while ``operation_id`` holds it."""
        return int(
            self._redis.eval(
                _MAINTENANCE_RENEW_SCRIPT, 1, rediskeys.SM_MAINTENANCE_KEY,
                operation_id, str(self._maintenance_ttl_ms),
            )
        ) == 1

    def maintenance(self) -> dict | None:
        raw = self._redis.get(rediskeys.SM_MAINTENANCE_KEY)
        if raw is None:
            return None
        try:
            value = json.loads(_text(raw))
        except ValueError:
            return {"invalid": _text(raw)}
        return value if isinstance(value, dict) else {"invalid": _text(raw)}

    def assert_maintenance_held(self, operation_id: str) -> None:
        """Safety check of a running repair; also renews the TTL (Lua
        compare-and-pexpire). Raises :class:`MaintenanceLockLost` once the lock
        was deleted, expired or taken over."""
        with self._renewers_lock:
            renewer = self._renewers.get(operation_id)
        if (renewer is None or not renewer.lost.is_set()) and self._renew_maintenance(operation_id):
            return
        if renewer is not None:
            renewer.lost.set()
        try:
            holder = (self.maintenance() or {}).get("operation_id")
        except Exception:  # the message only
            holder = "unknown"
        raise MaintenanceLockLost(
            f"SM maintenance lock of {operation_id} lost (holder: {holder}); fleet repair aborted"
        )

    def release_maintenance(self, operation_id: str) -> None:
        """Stop renewing and delete the lock only if ``operation_id`` still holds
        it (Lua compare-and-delete)."""
        with self._renewers_lock:
            renewer = self._renewers.pop(operation_id, None)
        if renewer is not None:
            renewer.stop()
        self._redis.eval(_MAINTENANCE_RELEASE_SCRIPT, 1, rediskeys.SM_MAINTENANCE_KEY, operation_id)

    # ------------------------------------------------------- actuation switch
    def actuation_mode(self) -> str:
        return self._actuation.mode()

    def actuation_state(self) -> dict:
        mode, source = self._actuation.resolve()
        return {"mode": mode, "source": source, "suppressed": self._actuation.recent_suppressed()[:20]}

    def record_suppressed(self, action: str, detail: dict) -> bool:
        return self._actuation.record_suppressed(action, detail)

    # ---------------------------------------------------------- node pressure
    def assert_no_pressure(self) -> None:
        reasons = self._pressure_source.node_pressure_reasons()
        if reasons:
            raise NodePressureActive(f"node pressure blocks cold start: {reasons}")

    def wait_until_healthy(self, operation: OperationHandle) -> None:
        """Pause during pressure and require a continuous clear hysteresis window.
        Aborts (MaintenanceLockLost) once the operation no longer holds the SM
        maintenance lock."""
        operation_id = operation.operation_id
        self.assert_maintenance_held(operation_id)
        started = self._monotonic()
        clear_since: float | None = None
        saw_pressure = False
        last_report: tuple[tuple[str, tuple[str, ...]], ...] | None = None
        while True:
            operation.assert_active()
            self.assert_maintenance_held(operation_id)
            reasons = self._pressure_source.node_pressure_reasons()
            now = self._monotonic()
            if reasons:
                saw_pressure = True
                clear_since = None
                report = tuple(
                    (node, tuple(values)) for node, values in sorted(reasons.items())
                )
                if report != last_report:
                    operation.advance(
                        "waiting_node_pressure",
                        details={"node_pressure": reasons},
                    )
                    last_report = report
            else:
                if not saw_pressure:
                    operation.advance("cluster_healthy")
                    return
                if clear_since is None:
                    clear_since = now
                    operation.advance("pressure_clear_hysteresis")
                if now - clear_since >= self._clear_hysteresis_s:
                    operation.advance("cluster_healthy")
                    return
            if now - started >= self._pressure_timeout_s:
                raise PressureWaitTimeout(
                    f"node pressure did not clear within {self._pressure_timeout_s}s"
                )
            self._sleep(self._poll_interval_s)


def _parse_holder(raw: str) -> dict:
    try:
        value = json.loads(raw)
    except ValueError:
        return {"invalid": raw}
    return value if isinstance(value, dict) else {"invalid": raw}


def _text(value: object) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)
