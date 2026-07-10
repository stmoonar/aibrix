from __future__ import annotations

import json
import logging
from typing import Any

from tre_common.metrics_schema import MetricsSnapshot
from tre_common.rediskeys import (
    DECISION_HIST_RETENTION_MS,
    DECISION_HIST_TTL_SECONDS,
    DECISION_LATEST_KEY,
    decision_hist_key,
)
from tre_controller.loops.tick import LoopTickResult
from tre_controller.loops.signal_log import SignalLogWriter
from tre_controller.planning.planner import DefragAction, HideAction, ScaleAction, UnhideAction


_LOGGER = logging.getLogger("tre_controller.decision")


def build_decision_snapshot(loop_name: str, snapshot: MetricsSnapshot, result: LoopTickResult) -> dict[str, str]:
    return {
        "ts_ms": str(snapshot.ts_ms),
        "loop": loop_name,
        "stale": "true" if snapshot.stale else "false",
        "submitted": str(result.submitted),
        "actions": json.dumps([_action_to_dict(action) for action in result.actions], separators=(",", ":")),
        "events": json.dumps(list(result.events), separators=(",", ":")),
        "model_states": json.dumps(
            _model_states(result.model_contexts, getattr(result, "classifications", {}), snapshot),
            separators=(",", ":"),
            sort_keys=True,
        ),
    }


class DecisionSnapshotWriter:
    def __init__(
        self,
        redis_client: Any,
        key: str = DECISION_LATEST_KEY,
        *,
        signal_log_writer: SignalLogWriter | None = None,
    ) -> None:
        self._redis = redis_client
        self._key = key
        self._signal_log_writer = signal_log_writer or SignalLogWriter(redis_client)

    def write(self, loop_name: str, snapshot: MetricsSnapshot, result: LoopTickResult) -> None:
        payload = build_decision_snapshot(loop_name, snapshot, result)
        try:
            self._redis.hset(self._key, mapping=payload)
        except Exception as exc:
            _LOGGER.warning("decision_snapshot_redis_write_failed: %s", exc)
        self._append_history(snapshot, result)
        self._signal_log_writer.write(snapshot, result)
        _LOGGER.info(json.dumps({"event": "trs_calc_result", **payload}, separators=(",", ":")))

    def _append_history(self, snapshot: MetricsSnapshot, result: LoopTickResult) -> None:
        # S5.1: per-model decision time-series, scored by window_end_ms. rescue and fairness
        # both write per tick; on the same window they usually collapse to one member, but a
        # scale action or stale-hold divergence between the two loops can leave two members at
        # the same score (review F3) -> readers dedup by window_end_ms. Trimmed to ~24h + TTL.
        states = _model_states(result.model_contexts, getattr(result, "classifications", {}), snapshot)
        for model, state in states.items():
            score = state.get("window_end_ms") or snapshot.ts_ms
            member = json.dumps({"ts": snapshot.ts_ms, "model": model, **state}, sort_keys=True, separators=(",", ":"))
            key = decision_hist_key(model)
            try:
                self._redis.zadd(key, {member: float(score)})
                self._redis.zremrangebyscore(key, "-inf", int(score) - DECISION_HIST_RETENTION_MS)
                self._redis.expire(key, DECISION_HIST_TTL_SECONDS)
            except Exception as exc:  # noqa: BLE001 - history is best-effort, never blocks decisions.
                _LOGGER.warning("decision_hist_write_failed:%s: %s", model, exc)


def _action_to_dict(action: object) -> dict[str, Any]:
    if isinstance(action, ScaleAction):
        return {
            "kind": "scale",
            "model": action.model,
            "delta": action.delta,
            "reason": action.reason,
            "source_loop": action.source_loop,
            "requires_safescale": action.requires_safescale,
            "receiver": action.receiver,
            "donor": action.donor,
        }
    if isinstance(action, HideAction):
        return {
            "kind": "hide",
            "model": action.model,
            "pods": list(action.pods),
            "reason": action.reason,
            "source_loop": action.source_loop,
        }
    if isinstance(action, UnhideAction):
        return {
            "kind": "unhide",
            "model": action.model,
            "pods": list(action.pods),
            "reason": action.reason,
            "source_loop": action.source_loop,
        }
    if isinstance(action, DefragAction):
        return {
            "kind": "defrag",
            "reason": action.reason,
            "source_loop": action.source_loop,
            "migrations": [_migration_to_dict(migration) for migration in action.migrations],
        }
    raise TypeError(f"unsupported decision action: {type(action).__name__}")


def _model_states(
    model_contexts: dict[str, dict[str, Any]],
    classifications: dict[str, Any] | None = None,
    snapshot: MetricsSnapshot | None = None,
) -> dict[str, dict[str, Any]]:
    classifications = classifications or {}
    models = snapshot.models if snapshot is not None else {}
    out: dict[str, dict[str, Any]] = {}
    for model, context in sorted(model_contexts.items()):
        cls = classifications.get(model)
        state = getattr(getattr(cls, "state", None), "value", None)
        window = models.get(model)
        out[model] = {
            "z_m": context.get("z_m"),
            "trs_z_m": context.get("trs_z_m"),
            "trs": context.get("trs"),
            "q_ctl": context.get("Q_ctl"),
            "y_m": context.get("Y_m"),
            "eta_m": context.get("eta_m"),
            "theta_m": context.get("theta_m"),
            "routable_pods": context.get("routable_pods"),
            "assigned_replicas": context.get("assigned_replicas"),
            "signal_warm": context.get("signal_warm"),
            "state": state,
            "signal_source": context.get("signal_source"),
            "signal_unavailable_reason": context.get("signal_unavailable_reason"),
            "window_end_ms": getattr(window, "window_end_ms", None),
        }
    return out


def _migration_to_dict(migration: Any) -> dict[str, Any]:
    return {
        "serve_id": migration.serve_id,
        "from_slot": _slot_to_dict(migration.from_slot),
        "to_slot": _slot_to_dict(migration.to_slot),
    }


def _slot_to_dict(slot: Any) -> dict[str, Any]:
    return {"node": slot.node, "gpu_ids": list(slot.gpu_ids)}
