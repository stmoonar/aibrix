from __future__ import annotations

import asyncio
import json
import logging
import time
from collections.abc import Awaitable, Callable, Iterable, Mapping
from typing import Any

from tre_common import rediskeys
from tre_controller.store.state_store import ControllerStateStore

_SCAN_INTERVAL_S = 60.0
_LOG = logging.getLogger(__name__)


class HiddenOrphanDetector:
    def __init__(
        self,
        redis_client: Any,
        *,
        grace_s: float = 600.0,
        now: Callable[[], float] = time.time,
        logger: logging.Logger = _LOG,
    ) -> None:
        if grace_s <= 0:
            raise ValueError("grace_s must be positive")
        self._redis = redis_client
        self._grace_s = float(grace_s)
        self._now = now
        self._logger = logger
        self._probe_store = ControllerStateStore(redis_client)

    def scan(self, *, now: float | None = None) -> tuple[str, ...]:
        detected_ts = float(self._now() if now is None else now)
        self._probe_store.gc_resolved_probes(now_ts=detected_ts)
        sm_state = _decode_hash(self._redis.hgetall(rediskeys.SM_STATE_KEY))
        probes = _decode_hash(
            self._redis.hgetall(rediskeys.CONTROLLER_SAFESCALE_PROBES_KEY)
        )
        watch = _decode_timestamps(
            self._redis.hgetall(rediskeys.CONTROLLER_ORPHAN_WATCH_KEY)
        )
        alerts = _decode_hash(
            self._redis.hgetall(rediskeys.CONTROLLER_HIDDEN_ORPHAN_ALERTS_KEY)
        )
        probing_ids, probing_models = _probing_targets(probes.values())

        candidates: dict[str, dict[str, Any]] = {}
        for serve_id, entry in sm_state.items():
            if not bool(entry.get("hidden", False)):
                continue
            model = str(entry.get("model", "unknown"))
            if serve_id in probing_ids or model in probing_models:
                continue
            candidates[serve_id] = entry

        stale_ids = (set(watch) | set(alerts)) - set(candidates)
        _delete_hash_fields(
            self._redis, rediskeys.CONTROLLER_ORPHAN_WATCH_KEY, stale_ids
        )
        _delete_hash_fields(
            self._redis, rediskeys.CONTROLLER_HIDDEN_ORPHAN_ALERTS_KEY, stale_ids
        )

        newly_alerted: list[str] = []
        for serve_id, entry in sorted(candidates.items()):
            first_seen = watch.get(serve_id)
            if first_seen is None:
                first_seen = detected_ts
                self._redis.hset(
                    rediskeys.CONTROLLER_ORPHAN_WATCH_KEY,
                    mapping={serve_id: str(first_seen)},
                )
            hidden_for_s = max(0.0, detected_ts - first_seen)
            if hidden_for_s <= self._grace_s or serve_id in alerts:
                continue

            model = str(entry.get("model", "unknown"))
            payload = {
                "model": model,
                "since_ts": first_seen,
                "detected_ts": detected_ts,
                "mode": "alert_only",
            }
            self._redis.hset(
                rediskeys.CONTROLLER_HIDDEN_ORPHAN_ALERTS_KEY,
                mapping={serve_id: _to_json(payload)},
            )
            self._logger.error(
                "TRE_ORPHAN_HIDDEN model=%s serve_id=%s hidden_for_s=%d "
                "probe=absent action=alert_only",
                model,
                serve_id,
                int(hidden_for_s),
            )
            newly_alerted.append(serve_id)

        return tuple(newly_alerted)

    async def run(
        self,
        *,
        interval_s: float = _SCAN_INTERVAL_S,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        while True:
            try:
                self.scan()
            except Exception:
                self._logger.exception("TRE_ORPHAN_SCAN_FAILED action=alert_only")
            await sleep(interval_s)


def _probing_targets(
    probes: Iterable[dict[str, Any]],
) -> tuple[set[str], set[str]]:
    serve_ids: set[str] = set()
    fallback_models: set[str] = set()
    for probe in probes:
        if probe.get("status") != "probing":
            continue
        probe_ids: set[str] = set()
        raw_pods = probe.get("pods", ())
        if isinstance(raw_pods, (list, tuple)):
            probe_ids.update(str(pod) for pod in raw_pods)
        raw_serve_id = probe.get("serve_id")
        if raw_serve_id:
            probe_ids.add(str(raw_serve_id))
        serve_ids.update(probe_ids)
        if not probe_ids and probe.get("model"):
            fallback_models.add(str(probe["model"]))
    return serve_ids, fallback_models


def _decode_hash(raw: Mapping[object, object] | None) -> dict[str, dict[str, Any]]:
    decoded: dict[str, dict[str, Any]] = {}
    for raw_key, raw_value in (raw or {}).items():
        try:
            value = json.loads(_to_text(raw_value))
        except (TypeError, ValueError, UnicodeDecodeError):
            continue
        if isinstance(value, dict):
            decoded[_to_text(raw_key)] = value
    return decoded


def _decode_timestamps(raw: Mapping[object, object] | None) -> dict[str, float]:
    decoded: dict[str, float] = {}
    for raw_key, raw_value in (raw or {}).items():
        try:
            decoded[_to_text(raw_key)] = float(_to_text(raw_value))
        except (TypeError, ValueError, UnicodeDecodeError):
            continue
    return decoded


def _delete_hash_fields(redis_client: Any, key: str, fields: set[str]) -> None:
    if fields:
        redis_client.hdel(key, *sorted(fields))


def _to_json(value: Mapping[str, Any]) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _to_text(value: object) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)
