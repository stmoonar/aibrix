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
    seconds, micros = redis.time()
    return int(seconds) * 1000 + int(micros) // 1000


def live_gateway_instances(redis: Any, *, max_age_ms: int) -> set[str]:
    """Instance ids whose last heartbeat is at most ``max_age_ms`` old (Redis TIME)."""
    now_ms = redis_now_ms(redis)
    members = redis.zrange(rediskeys.GW_INSTANCES_KEY, 0, -1, withscores=True) or ()
    return {_text(member) for member, score in members if now_ms - float(score) <= max_age_ms}


def pod_inflight_fields(redis: Any, pod: str) -> dict[str, dict | None]:
    """instance id -> the parsed field value (None when unreadable)."""
    out: dict[str, dict | None] = {}
    for field_name, raw in (redis.hgetall(rediskeys.gw_inflight_key(pod)) or {}).items():
        try:
            payload = json.loads(_text(raw))
        except (TypeError, ValueError):
            payload = None
        out[_text(field_name)] = payload if isinstance(payload, dict) else None
    return out


def pod_inflight(redis: Any, pod: str, live_instances: Collection[str]) -> int:
    """Routed, unfinished requests on ``pod`` summed over the live gateway instances."""
    total = 0
    for instance, payload in pod_inflight_fields(redis, pod).items():
        if instance not in live_instances or payload is None:
            continue
        try:
            total += max(0, int(payload.get("total", 0)))
        except (TypeError, ValueError):
            continue
    return total
