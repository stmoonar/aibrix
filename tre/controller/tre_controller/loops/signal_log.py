from __future__ import annotations

import logging
import math
from typing import Any

from tre_common import rediskeys
from tre_common.metrics_schema import MetricsSnapshot
from tre_controller.planning.planner import ScaleAction

SIGNAL_LOG_FIELDS = (
    "ts",
    "window_id",
    "model",
    "signal_source",
    "raw_signal",
    "theta",
    "z",
    "tss",
    "theta_m",
    "z_m",
    "queue_len",
    "decode_tps",
    "prefill_tps",
    "replicas_awake",
    "replicas_target",
    "tier",
    "eta_m",
    "action",
)

_LOG = logging.getLogger(__name__)


class SignalLogWriter:
    def __init__(
        self,
        redis_client: Any,
        *,
        key: str = rediskeys.CONTROLLER_SIGNAL_LOG_KEY,
        maxlen: int = 200_000,
    ) -> None:
        self._redis = redis_client
        self._key = key
        self._maxlen = int(maxlen)
        self._last_window_by_model: dict[str, int] = {}

    def write(self, snapshot: MetricsSnapshot, result: Any) -> int:
        contexts = getattr(result, "model_contexts", {}) or {}
        classifications = getattr(result, "classifications", {}) or {}
        action_by_model, delta_by_model = _summarize_actions(
            getattr(result, "actions", ()) or ()
        )
        written = 0
        for model, context in sorted(contexts.items()):
            metrics = snapshot.models.get(model)
            window_id = int(
                getattr(metrics, "window_end_ms", None) or snapshot.ts_ms
            )
            if window_id <= self._last_window_by_model.get(model, -1):
                continue
            source = str(context.get("signal_source") or "unknown")
            if source == "zm":
                raw_signal = context.get("trs")
                active_z = context.get("trs_z_m")
            else:
                raw_signal = context.get("signal_raw_value")
                active_z = context.get("z_m")
            theta = context.get("signal_theta")
            replicas_awake = int(context.get("routable_pods") or 0)
            replicas_target = max(
                0, replicas_awake + delta_by_model.get(model, 0)
            )
            classification = classifications.get(model)
            fields = {
                "ts": _format(snapshot.ts_ms / 1000.0),
                "window_id": str(window_id),
                "model": model,
                "signal_source": source,
                "raw_signal": _format(raw_signal),
                "theta": _format(theta),
                "z": _format(active_z),
                "tss": _format(context.get("trs")),
                "theta_m": _format(context.get("theta_m")),
                "z_m": _format(context.get("trs_z_m")),
                "queue_len": _format(context.get("Q")),
                "decode_tps": _format(context.get("decode_tps")),
                "prefill_tps": _format(context.get("prefill_tps")),
                "replicas_awake": str(replicas_awake),
                "replicas_target": str(replicas_target),
                "tier": _tier(classification),
                "eta_m": _format(context.get("eta_m")),
                "action": action_by_model.get(model, "none"),
            }
            try:
                self._redis.xadd(
                    self._key,
                    fields,
                    maxlen=self._maxlen,
                    approximate=True,
                )
            except Exception as exc:
                _LOG.warning("signal_log_write_failed:%s: %s", model, exc)
                continue
            self._last_window_by_model[model] = window_id
            written += 1
        return written


def _summarize_actions(actions: tuple[Any, ...]) -> tuple[dict[str, str], dict[str, int]]:
    action_by_model: dict[str, str] = {}
    delta_by_model: dict[str, int] = {}
    for action in actions:
        if not isinstance(action, ScaleAction):
            continue
        delta_by_model[action.model] = delta_by_model.get(action.model, 0) + int(
            action.delta
        )
        if action.receiver and action.donor:
            label = f"transfer:{action.donor}->{action.receiver}"
        elif action.delta > 0:
            label = "scale_up"
        elif action.delta < 0:
            label = "scale_down"
        else:
            label = "none"
        action_by_model[action.model] = label
    return action_by_model, delta_by_model


def _tier(classification: Any) -> str:
    state = getattr(getattr(classification, "state", None), "value", None)
    if state == "critical":
        return "crit"
    if state == "high":
        return "high"
    return "healthy"


def _format(value: Any) -> str:
    if value is None:
        return "nan"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not math.isfinite(number):
        return "nan"
    return repr(number)
