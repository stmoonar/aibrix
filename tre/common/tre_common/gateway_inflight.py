"""Gateway in-flight counts (written by the gateway plugin, TRE-PATCH P3-GW-007/009).

``tre:v2:gw:inflight:<pod>`` is a HASH (field = plugin instance id, value JSON
``{"total", "non_continuable", "ts"}``): a write-on-change mirror of the plugin's
in-memory per-pod count of routed, unfinished requests. It does not depend on the
pod's /metrics scrape. A field counts only while its instance is live in
``tre:v2:gw:instances`` (ZSET, score = the plugin's heartbeat in Redis TIME ms, every
2 s). Liveness compares that score with Redis TIME read now - one clock, never the
reader's wall clock. A field's ``ts`` is not a freshness test (the mirror is written
on change only).

TODO(after the SM-B merge): the service-manager's reader
(``tre_sm.ops.sleep_primitive.GatewayState.inflight``) should switch to
:func:`pod_inflight_fields` here.
"""

from __future__ import annotations

import json
from typing import Any, Collection

from tre_common import rediskeys


def _text(value: Any) -> str:
    return value.decode() if isinstance(value, bytes) else str(value)


def redis_now_ms(redis: Any) -> int:
    return time_reply_ms(redis.time())


def time_reply_ms(reply: Any) -> int:
    """A Redis ``TIME`` reply (seconds, microseconds) in ms."""
    seconds, micros = reply
    return int(seconds) * 1000 + int(micros) // 1000


#: ``service_manager.sleep.instance_staleness_s`` default (registry).
DEFAULT_INSTANCE_MAX_AGE_MS = 10_000


def instance_max_age_ms(registry: Any) -> int:
    """THE liveness bound of a gateway plugin instance (heartbeat age, ms): the
    registry's ``service_manager.sleep.instance_staleness_s`` - the bound the
    service-manager applies before a sleep. Every controller reader of the in-flight
    mirror (early commit, frozen-scrape demand, idle window) uses it (review
    2026-10-06 P3-2). The default when the registry has no service-manager section."""
    service_manager = getattr(registry, "service_manager", None)
    if not callable(service_manager):
        return DEFAULT_INSTANCE_MAX_AGE_MS
    return max(1, int(1000 * float(service_manager().sleep.instance_staleness_s)))


def live_gateway_instances(redis: Any, *, max_age_ms: int) -> set[str]:
    """Instance ids whose last heartbeat is at most ``max_age_ms`` old (Redis TIME)."""
    now_ms = redis_now_ms(redis)
    members = redis.zrange(rediskeys.GW_INSTANCES_KEY, 0, -1, withscores=True)
    return live_instances_at(now_ms, members, max_age_ms=max_age_ms)


def live_instances_at(now_ms: int, members: Any, *, max_age_ms: int) -> set[str]:
    """:func:`live_gateway_instances` from replies already read: ``now_ms`` (Redis TIME)
    and the ``GW_INSTANCES_KEY`` ZRANGE ... WITHSCORES reply."""
    return {_text(member) for member, score in (members or ()) if now_ms - float(score) <= max_age_ms}


def pod_inflight_fields(redis: Any, pod: str) -> dict[str, dict | None]:
    """instance id -> the parsed field value (None when unreadable)."""
    return inflight_fields(redis.hgetall(rediskeys.gw_inflight_key(pod)))


def inflight_fields(raw_hash: Any) -> dict[str, dict | None]:
    """:func:`pod_inflight_fields` of an HGETALL reply already read."""
    out: dict[str, dict | None] = {}
    for field_name, raw in (raw_hash or {}).items():
        try:
            payload = json.loads(_text(raw))
        except (TypeError, ValueError):
            payload = None
        out[_text(field_name)] = payload if isinstance(payload, dict) else None
    return out


def pod_inflight(redis: Any, pod: str, live_instances: Collection[str]) -> int | None:
    """Routed, unfinished requests on ``pod`` summed over the live gateway instances.
    None (unknown, review 2026-10-06 P3-1) when a LIVE instance's field cannot be read
    (not JSON, no integer ``total``): that instance may hold requests there. A field of
    a dead instance is ignored either way; no field at all is 0 (nothing routed). A
    caller that turns the count into demand counts unknown as no demand."""
    return inflight_total(pod_inflight_fields(redis, pod), live_instances)


def inflight_total(fields: dict[str, dict | None], live_instances: Collection[str]) -> int | None:
    """:func:`pod_inflight` of the pod's parsed fields (:func:`inflight_fields`)."""
    total = 0
    for instance, payload in fields.items():
        if instance not in live_instances:
            continue
        value = None if payload is None else payload.get("total")
        if isinstance(value, bool) or not isinstance(value, (int, float)) or value != value:
            return None
        total += max(0, int(value))
    return total
