from __future__ import annotations

import json
from typing import Any, Mapping, Protocol

from tre_common import rediskeys


class RedisProbeClient(Protocol):
    def hset(
        self,
        name: str,
        key: str | None = None,
        value: str | None = None,
        mapping: Mapping[str, str] | None = None,
    ) -> Any: ...

    def hgetall(self, name: str) -> Mapping[object, object]: ...

    def hdel(self, name: str, *keys: str) -> Any: ...

    def rpush(self, name: str, *values: str) -> Any: ...

    def lrange(self, name: str, start: int, end: int) -> list[object]: ...


class ControllerStateStore:
    def __init__(self, redis: RedisProbeClient) -> None:
        self._redis = redis

    def save_probe(self, request_id: str, record: dict[str, Any]) -> None:
        payload = dict(record)
        payload["request_id"] = request_id
        self._redis.hset(
            rediskeys.CONTROLLER_SAFESCALE_PROBES_KEY,
            mapping={request_id: _to_json(payload)},
        )

    def delete_probe(self, request_id: str) -> None:
        self._redis.hdel(rediskeys.CONTROLLER_SAFESCALE_PROBES_KEY, request_id)

    def list_unresolved_probes(self) -> list[dict[str, Any]]:
        try:
            raw = self._redis.hgetall(rediskeys.CONTROLLER_SAFESCALE_PROBES_KEY) or {}
        except Exception:
            return []

        records: list[tuple[str, dict[str, Any]]] = []
        for raw_request_id, raw_payload in raw.items():
            request_id = _to_text(raw_request_id)
            record = _decode_mapping(raw_payload)
            if record is None:
                continue
            if str(record.get("status", "probing")) != "probing":
                continue
            record.setdefault("request_id", request_id)
            records.append((request_id, record))
        return [record for _, record in sorted(records, key=lambda item: item[0])]

    def append_probe_journal(self, request_id: str, record: dict[str, Any]) -> None:
        self._redis.rpush(
            rediskeys.controller_safescale_probe_journal_key(request_id),
            _to_json(record),
        )

    def load_probe_journal(self, request_id: str) -> list[dict[str, Any]]:
        try:
            raw_entries = self._redis.lrange(
                rediskeys.controller_safescale_probe_journal_key(request_id),
                0,
                -1,
            )
        except Exception:
            return []

        entries: list[dict[str, Any]] = []
        for raw_entry in raw_entries:
            record = _decode_mapping(raw_entry)
            if record is not None:
                entries.append(record)
        return entries


def _to_json(record: Mapping[str, Any]) -> str:
    return json.dumps(record, sort_keys=True, separators=(",", ":"))


def _decode_mapping(raw: object) -> dict[str, Any] | None:
    try:
        decoded = json.loads(_to_text(raw))
    except (TypeError, json.JSONDecodeError, UnicodeDecodeError):
        return None
    if not isinstance(decoded, dict):
        return None
    return decoded


def _to_text(value: object) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)
