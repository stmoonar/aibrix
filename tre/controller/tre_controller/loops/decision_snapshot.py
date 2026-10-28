from __future__ import annotations

import json
import logging
import threading
import time
from concurrent.futures import Executor
from typing import Any

from tre_common.metrics_schema import MetricsSnapshot
from tre_common.rediskeys import (
    DECISION_HIST_RETENTION_MS,
    DECISION_HIST_TTL_SECONDS,
    decision_hist_key,
)
from tre_controller.loops.action_queue import _members
from tre_controller.loops.tick import LoopTickResult
from tre_controller.loops.signal_log import SignalLogWriter, run_pipelined
from tre_controller.planning.planner import DefragAction, HideAction, ScaleAction, UnhideAction


_LOGGER = logging.getLogger("tre_controller.decision")


def build_decision_snapshot(loop_name: str, snapshot: MetricsSnapshot, result: LoopTickResult) -> dict[str, str]:
    return {
        "ts_ms": str(snapshot.ts_ms),
        "loop": loop_name,
        "stale": "true" if snapshot.stale else "false",
        "submitted": str(result.submitted),
        "actions": json.dumps(
            [_action_to_dict(member) for action in result.actions for member in _members(action)],
            separators=(",", ":"),
        ),
        "events": json.dumps(list(result.events), separators=(",", ":")),
        "model_states": json.dumps(
            _model_states(result.model_contexts, getattr(result, "classifications", {}), snapshot),
            separators=(",", ":"),
            sort_keys=True,
        ),
    }


class DecisionSnapshotWriter:
    """Records each decision: the ``trs_calc_result`` log line (on the caller's thread,
    in decision order), the per-model decision history and the signal log (Redis,
    best-effort). The Redis writes of one decision go in one pipelined round trip;
    with ``executor`` (the app: one worker thread, so decisions land in order) they run
    off the event loop and the decision never waits for Redis. Without it (tests,
    offline) they run inline. At most ``max_pending`` decisions wait for the executor;
    beyond that (Redis down or slow) a decision's Redis writes are dropped and logged."""

    def __init__(
        self,
        redis_client: Any,
        *,
        signal_log_writer: SignalLogWriter | None = None,
        executor: Executor | None = None,
        max_pending: int = 32,
    ) -> None:
        self._redis = redis_client
        self._signal_log_writer = signal_log_writer or SignalLogWriter(redis_client)
        self._executor = executor
        self._max_pending = int(max_pending)
        self._pending = 0
        self._pending_lock = threading.Lock()

    def write(self, loop_name: str, snapshot: MetricsSnapshot, result: LoopTickResult) -> None:
        payload = build_decision_snapshot(loop_name, snapshot, result)
        history = self._history_calls(snapshot, result)
        signal_rows = self._signal_log_writer.claim(snapshot, result)
        if history or signal_rows:
            self._send(history, signal_rows)
        # ``now_ms``: wall clock of the decision (``ts_ms`` is the metrics window
        # boundary the tick read, grid-aligned) - evaluation data, log line only.
        _LOGGER.info(json.dumps({"event": "trs_calc_result", "now_ms": str(int(time.time() * 1000)), **payload},
                                separators=(",", ":")))

    def _send(self, history: list[tuple[str, list]], signal_rows: list) -> None:
        if self._executor is None:
            self._flush(history, signal_rows)
            return
        with self._pending_lock:
            if self._pending >= self._max_pending:
                full = True
            else:
                full = False
                self._pending += 1
        if full:
            _LOGGER.warning("decision_redis_write_dropped: %d decisions waiting for redis", self._max_pending)
            self._signal_log_writer.settle(signal_rows, [RuntimeError("dropped")] * len(signal_rows))
            return
        try:
            self._executor.submit(self._flush_pending, history, signal_rows)
        except RuntimeError as exc:  # executor shut down (controller stopping)
            with self._pending_lock:
                self._pending -= 1
            self._signal_log_writer.settle(signal_rows, [exc] * len(signal_rows))

    def _flush_pending(self, history: list[tuple[str, list]], signal_rows: list) -> None:
        try:
            self._flush(history, signal_rows)
        finally:
            with self._pending_lock:
                self._pending -= 1

    def _flush(self, history: list[tuple[str, list]], signal_rows: list) -> None:
        """One round trip: every model's history calls, then the signal-log entries."""
        calls = [call for _model, model_calls in history for call in model_calls]
        signal_calls = self._signal_log_writer.calls(signal_rows)
        try:
            replies = run_pipelined(self._redis, calls + signal_calls)
            index = 0
            for model, model_calls in history:
                failed = [reply for reply in replies[index:index + len(model_calls)] if isinstance(reply, Exception)]
                index += len(model_calls)
                if failed:
                    _LOGGER.warning("decision_hist_write_failed:%s: %s", model, failed[0])
            self._signal_log_writer.settle(signal_rows, replies[index:])
        except Exception:  # noqa: BLE001 - best-effort, never stops the writer
            _LOGGER.exception("decision redis write failed")

    def _history_calls(self, snapshot: MetricsSnapshot, result: LoopTickResult) -> list[tuple[str, list]]:
        # S5.1: per-model decision time-series, scored by window_end_ms: one member per
        # planner call (one decision loop since 2026-10-28; a re-read of the same window
        # can still leave two members at one score -> readers dedup by window_end_ms).
        # Trimmed to ~24h + TTL.
        states = _model_states(result.model_contexts, getattr(result, "classifications", {}), snapshot)
        out: list[tuple[str, list]] = []
        for model, state in states.items():
            score = state.get("window_end_ms") or snapshot.ts_ms
            member = json.dumps({"ts": snapshot.ts_ms, "model": model, **state}, sort_keys=True, separators=(",", ":"))
            key = decision_hist_key(model)
            out.append((model, [
                ("zadd", (key, {member: float(score)}), {}),
                ("zremrangebyscore", (key, "-inf", int(score) - DECISION_HIST_RETENTION_MS), {}),
                ("expire", (key, DECISION_HIST_TTL_SECONDS), {}),
            ]))
        return out


def _action_to_dict(action: object) -> dict[str, Any]:
    if isinstance(action, ScaleAction):
        payload = {
            "kind": "scale",
            "model": action.model,
            "delta": action.delta,
            "reason": action.reason,
            "source_loop": action.source_loop,
            "requires_safescale": action.requires_safescale,
            "receiver": action.receiver,
            "donor": action.donor,
        }
        if action.pods:
            payload["pods"] = list(action.pods)
        if action.group:
            # sm-plans section 10: the transfer (plan group) this item belongs to.
            payload["group"] = action.group
        rescue = getattr(action, "rescue", None)
        if rescue is not None:
            # C1: the absolute rescue target this scale-up belongs to.
            payload["rescue"] = {
                "target": rescue.target, "desired": rescue.desired,
                "base": rescue.base, "covered": rescue.covered,
            }
        return payload
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
            "trs_raw": context.get("trs_raw"),
            "q_ctl": context.get("Q_ctl"),
            "y_m": context.get("Y_m"),
            "eta_m": context.get("eta_m"),
            "theta_m": context.get("theta_m"),
            "routable_pods": context.get("routable_pods"),
            "assigned_replicas": context.get("assigned_replicas"),
            "sample_quality_hold": context.get("sample_quality_hold"),
            "state": state,
            "signal_source": context.get("signal_source"),
            "signal_unavailable_reason": context.get("signal_unavailable_reason"),
            "window_end_ms": getattr(window, "window_end_ms", None),
        }
        if context.get("floor_headroom") is not None:
            # 2026-10-02: the SM replica floor view the planner bounded donors by.
            out[model]["floor"] = context.get("floor")
            out[model]["floor_headroom"] = context.get("floor_headroom")
    return out


def _migration_to_dict(migration: Any) -> dict[str, Any]:
    return {
        "serve_id": migration.serve_id,
        "from_slot": _slot_to_dict(migration.from_slot),
        "to_slot": _slot_to_dict(migration.to_slot),
    }


def _slot_to_dict(slot: Any) -> dict[str, Any]:
    return {"node": slot.node, "gpu_ids": list(slot.gpu_ids)}
