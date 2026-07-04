from __future__ import annotations

import json
import logging
from typing import Any

from tre_common.metrics_schema import MetricsSnapshot
from tre_common.rediskeys import DECISION_LATEST_KEY
from tre_controller.loops.tick import LoopTickResult
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
    }


class DecisionSnapshotWriter:
    def __init__(self, redis_client: Any, key: str = DECISION_LATEST_KEY) -> None:
        self._redis = redis_client
        self._key = key

    def write(self, loop_name: str, snapshot: MetricsSnapshot, result: LoopTickResult) -> None:
        payload = build_decision_snapshot(loop_name, snapshot, result)
        self._redis.hset(self._key, mapping=payload)
        _LOGGER.info(json.dumps({"event": "trs_calc_result", **payload}, separators=(",", ":")))


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


def _migration_to_dict(migration: Any) -> dict[str, Any]:
    return {
        "serve_id": migration.serve_id,
        "from_slot": _slot_to_dict(migration.from_slot),
        "to_slot": _slot_to_dict(migration.to_slot),
    }


def _slot_to_dict(slot: Any) -> dict[str, Any]:
    return {"node": slot.node, "gpu_ids": list(slot.gpu_ids)}
