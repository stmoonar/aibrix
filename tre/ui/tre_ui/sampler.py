"""Isolated live-data sampler for the console (architect spec).

ONE in-process background thread owns ALL upstream reads at fixed low rates and publishes a
cached composite snapshot. Browsers consume the cache (via /api/snapshot + SSE) and therefore
NEVER cause an upstream read -- cost is constant regardless of how many tabs are open, and the
console is physically incapable of touching the control loop: it only does read-only ops on
Redis keys the controller writes (plus the SM's cheap state read).

The snapshot-building is split into pure helpers (testable with fakes); the thread is thin timing.
"""
from __future__ import annotations

import json
import threading
import time
from collections import deque
from typing import Any, Callable

from tre_common.rediskeys import CONTROLLER_SAFESCALE_PROBES_KEY, DECISION_LATEST_KEY, decision_hist_key

# per-source cadence (seconds) -- all far below the controller's own Redis load per tick.
_RATES = {"decision": 1.0, "hist": 2.0, "sm": 2.0, "gpu": 5.0, "probes": 2.0}
_HIST_RING = 2000  # points kept per model in memory (~1h at 2s)
_HIST_TAIL = 240   # points embedded in each snapshot (browser gets the rest via /api/signal/history)
_EVENTS = 5000
_EVENT_MARKERS = ("critical", "suppress", "safescale", "leak", "defrag", "hide", "unhide",
                  "capacity", "incomplete", "stale", "warmup")


def _text(value: Any) -> str:
    return value.decode("utf-8") if isinstance(value, bytes) else str(value)


def _loads(value: Any) -> Any:
    if value is None:
        return None
    try:
        return json.loads(_text(value))
    except (json.JSONDecodeError, TypeError, ValueError):
        return None


def decode_decision(raw: dict[Any, Any]) -> dict[str, Any]:
    hashmap = {_text(k): _text(v) for k, v in (raw or {}).items()}
    if not hashmap:
        return {"ts_ms": None, "loop": None, "model_states": {}, "actions": [], "events": []}
    ts_raw = hashmap.get("ts_ms", "")
    return {
        "ts_ms": int(ts_raw) if ts_raw.lstrip("-").isdigit() else None,
        "loop": hashmap.get("loop"),
        "stale": hashmap.get("stale") == "true",
        "submitted": int(hashmap["submitted"]) if hashmap.get("submitted", "").isdigit() else 0,
        "model_states": _loads(hashmap.get("model_states")) or {},
        "actions": _loads(hashmap.get("actions")) or [],
        "events": _loads(hashmap.get("events")) or [],
    }


def diff_events(decision: dict[str, Any], seen_key: tuple | None) -> tuple[list[dict], tuple | None]:
    """Return (new_events, new_seen_key). A decision is new when (ts_ms, loop) changes."""
    key = (decision.get("ts_ms"), decision.get("loop"))
    if decision.get("ts_ms") is None or key == seen_key:
        return [], seen_key
    ts, loop = decision["ts_ms"], decision.get("loop")
    out: list[dict] = []
    for action in decision.get("actions", []):
        out.append({
            "ts_ms": ts, "loop": loop, "kind": action.get("kind", "action"),
            "model": action.get("model"), "delta": action.get("delta"),
            "reason": action.get("reason"), "text": _action_text(action),
        })
    for event in decision.get("events", []):
        text = event if isinstance(event, str) else json.dumps(event)
        if any(marker in text for marker in _EVENT_MARKERS):
            out.append({"ts_ms": ts, "loop": loop, "kind": "event", "model": _event_model(text), "text": text})
    return out, key


def _action_text(action: dict) -> str:
    kind = action.get("kind", "action")
    delta = action.get("delta")
    delta_txt = f"+{delta}" if isinstance(delta, int) and delta > 0 else (str(delta) if delta is not None else "")
    reason = action.get("reason")
    core = f"{kind} {action.get('model', '')} {delta_txt}".strip()
    return f"{core} ({reason})" if reason else core


def _event_model(text: str) -> str | None:
    return text.rsplit(":", 1)[-1] if ":" in text else None


def merge_hist(existing: list[dict], new_points: list[Any]) -> list[dict]:
    """Append decoded new points, dedup by window_end_ms keeping the max ts (review F3)."""
    by_window: dict[int, dict] = {}
    for point in existing:
        window = point.get("window_end_ms") or point.get("ts")
        if window is not None:
            by_window[window] = point
    for raw in new_points:
        point = _loads(raw)
        if point is None:
            continue
        window = point.get("window_end_ms") or point.get("ts")
        if window is None:
            continue
        prev = by_window.get(window)
        if prev is None or (point.get("ts") or 0) >= (prev.get("ts") or 0):
            by_window[window] = point
    merged = [by_window[w] for w in sorted(by_window)]
    return merged[-_HIST_RING:]


class Sampler:
    """Owns all upstream reads on one background thread; publishes a versioned cached snapshot."""

    def __init__(
        self,
        redis_client: Any,
        sm_get_state: Callable[[], dict[str, Any]],
        *,
        model_names: list[str],
        clock: Callable[[], float] = time.monotonic,
        now_ms: Callable[[], int] = lambda: int(time.time() * 1000),
    ) -> None:
        self._redis = redis_client
        self._sm_get_state = sm_get_state
        self._models = model_names
        self._clock = clock
        self._now_ms = now_ms
        self._lock = threading.Lock()
        self._version = 0
        self._snapshot: dict[str, Any] = {"version": 0}
        self._hist: dict[str, list[dict]] = {m: [] for m in model_names}
        self._events: deque[dict] = deque(maxlen=_EVENTS)
        self._seen_decision: tuple | None = None
        self._parts: dict[str, Any] = {}
        self._ages: dict[str, int] = {}
        self._next: dict[str, float] = {k: 0.0 for k in _RATES}
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    # ---- lifecycle ----
    def start(self) -> None:
        self.sample_once()  # first paint before serving
        self._thread = threading.Thread(target=self._run, name="tre-ui-sampler", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2)

    def _run(self) -> None:
        while not self._stop.wait(0.25):
            try:
                self.sample_once()
            except Exception:  # noqa: BLE001 - a bad read must never kill the sampler
                pass

    # ---- reads (each guarded; a failure marks that source stale but never raises) ----
    def sample_once(self) -> None:
        now = self._clock()
        changed = False
        for source, reader in (
            ("decision", self._read_decision), ("hist", self._read_hist), ("sm", self._read_sm),
            ("gpu", self._read_gpu), ("probes", self._read_probes),
        ):
            if now >= self._next[source]:
                self._next[source] = now + _RATES[source]
                changed |= reader()
        if changed:
            self._rebuild()

    def _read_decision(self) -> bool:
        try:
            decision = decode_decision(self._redis.hgetall(DECISION_LATEST_KEY))
        except Exception:  # noqa: BLE001
            return False
        self._parts["decision"] = decision
        self._ages["decision"] = self._now_ms()
        new_events, self._seen_decision = diff_events(decision, self._seen_decision)
        for event in new_events:
            self._events.appendleft(event)
        return True

    def _read_hist(self) -> bool:
        for model in self._models:
            try:
                raw = self._redis.zrangebyscore(decision_hist_key(model), "-inf", "+inf")
            except Exception:  # noqa: BLE001
                continue
            self._hist[model] = merge_hist(self._hist[model], raw)
        return True

    def _read_sm(self) -> bool:
        try:
            self._parts["sm"] = self._sm_get_state()
            self._ages["sm"] = self._now_ms()
        except Exception as exc:  # noqa: BLE001
            self._parts["sm"] = {"error": str(exc)}
        return True

    def _read_gpu(self) -> bool:
        nodes: list[dict] = []
        try:
            for key in self._redis.scan_iter("tre:gpu_truth:*"):
                doc = _loads(self._redis.get(_text(key)))
                if doc is not None:
                    nodes.append(doc)
        except Exception:  # noqa: BLE001
            pass
        nodes.sort(key=lambda n: str(n.get("node", "")))
        self._parts["gpu_truth"] = {"nodes": nodes}
        self._ages["gpu"] = self._now_ms()
        return True

    def _read_probes(self) -> bool:
        try:
            raw = self._redis.hgetall(CONTROLLER_SAFESCALE_PROBES_KEY) or {}
            self._parts["probes"] = [p for p in (_loads(v) for v in raw.values()) if p]
        except Exception:  # noqa: BLE001
            self._parts["probes"] = []
        return True

    def _rebuild(self) -> None:
        now = self._now_ms()
        decision = self._parts.get("decision") or {"ts_ms": None, "model_states": {}}
        model_states = decision.get("model_states") or {}
        snapshot = {
            "sampled_at_ms": now,
            "decision": {"latest": decision, "age_ms": self._age(now, "decision")},
            "models": {
                model: {"hist_tail": self._hist[model][-_HIST_TAIL:], "state": model_states.get(model, {})}
                for model in self._models
            },
            "sm": {"state": self._parts.get("sm", {}), "age_ms": self._age(now, "sm")},
            "gpu_truth": {**(self._parts.get("gpu_truth") or {"nodes": []}), "age_ms": self._age(now, "gpu")},
            "probes": self._parts.get("probes", []),
            "events_head": list(self._events)[:80],
        }
        with self._lock:
            self._version += 1
            snapshot["version"] = self._version
            self._snapshot = snapshot

    def _age(self, now: int, source: str) -> int | None:
        return (now - self._ages[source]) if source in self._ages else None

    # ---- consumers (cache-only; never read upstream) ----
    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return dict(self._snapshot)

    def version(self) -> int:
        with self._lock:
            return self._version

    def history(self, model: str, since_ms: int = 0) -> list[dict]:
        return [p for p in self._hist.get(model, []) if (p.get("window_end_ms") or p.get("ts") or 0) >= since_ms]

    def events(self, limit: int = 200) -> list[dict]:
        return list(self._events)[:limit]
