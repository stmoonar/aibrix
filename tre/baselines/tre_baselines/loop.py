"""The tick loop: gather -> decide -> clamp -> scale-downs before scale-ups -> dispatch.

* Dry-run (the default) logs every decision and never calls the SM. A shell configured
  to actuate falls back to dry-run for any tick in which it does not hold the owner lock
  ``tre:v2:bl:owner`` (``SET NX PX``, renewed every tick), so two shells never actuate.
* The dispatcher is asynchronous: a model whose previous SM call is still running gets
  ``inflight_skip`` (not queued); the tick never waits for the SM.
* No arbiter in the MVP: scale-downs are submitted before scale-ups in the same tick and
  SM refusals are only logged (``sm_result`` on the model's next decision line).
* One JSONL line per model per tick to ``$TRE_BL_LOG_DIR/decisions-<policy>-<YYYYMMDD>.jsonl``
  (date of the Redis clock, UTC), mirrored to ``tre:v2:bl:decision:<model>`` (TTL 1 h)
  unless ``TRE_BL_WRITE_REDIS=false``.
* A tick that raises is logged and skipped; ``max_tick_failures`` consecutive failures
  turn ``/healthz`` into 503.
"""
from __future__ import annotations

import datetime as _dt
import json
import logging
import os
import socket
import threading
import time
import uuid
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Mapping, Optional, Protocol

from tre_baselines.keys import DECISION_TTL_S, OWNER_KEY, decision_key
from tre_baselines.policies.base import Decision
from tre_baselines.sm_client import Completed, Dispatcher
from tre_baselines.snapshot import ClusterSnapshot

LOG = logging.getLogger(__name__)

REVERSAL_WINDOW_MS = 60_000
ACTIONS = ("none", "up", "down", "inflight_skip", "dry_run")


class Source(Protocol):
    def gather(self, tick: int = 0) -> ClusterSnapshot: ...


def clamp(desired: int, lo: int, hi: int) -> int:
    return max(int(lo), min(int(hi), int(desired)))


class OwnerLock:
    """``SET key token NX PX ttl``; renewed by ``PEXPIRE`` while the value is ours."""

    def __init__(self, redis: Any, ttl_s: float, *, key: str = OWNER_KEY, token: Optional[str] = None) -> None:
        self._redis = redis
        self._ttl_ms = max(1000, int(ttl_s * 1000))
        self.key = key
        self.token = token or f"{socket.gethostname()}:{os.getpid()}:{uuid.uuid4().hex[:8]}"

    @staticmethod
    def _text(value: Any) -> Any:
        return value.decode("utf-8", "replace") if isinstance(value, bytes) else value

    def ensure(self) -> bool:
        try:
            if self._text(self._redis.get(self.key)) == self.token:
                return bool(self._redis.pexpire(self.key, self._ttl_ms))
            return bool(self._redis.set(self.key, self.token, nx=True, px=self._ttl_ms))
        except Exception as exc:
            LOG.warning("owner lock check failed: %s", exc)
            return False

    def release(self) -> None:
        try:
            if self._text(self._redis.get(self.key)) == self.token:
                self._redis.delete(self.key)
        except Exception as exc:
            LOG.warning("owner lock release failed: %s", exc)


@dataclass
class ShellStats:
    ticks: int = 0
    tick_failures: int = 0
    consecutive_failures: int = 0
    decisions: int = 0
    sm_calls: int = 0
    sm_failures: int = 0
    redis_write_failures: int = 0
    effective_dry_run: bool = True
    owner: bool = False
    last_tick_ms: Optional[int] = None
    last_error: Optional[str] = None
    actions: dict[tuple[str, str], int] = field(default_factory=dict)
    reversals: dict[str, int] = field(default_factory=dict)
    event_lag_s: dict[str, float] = field(default_factory=dict)
    scrape_failures: int = 0


class DecisionLog:
    def __init__(self, log_dir: str, policy: str) -> None:
        self._dir = Path(log_dir)
        self._policy = policy
        self._dir.mkdir(parents=True, exist_ok=True)

    def path_for(self, ts_ms: int) -> Path:
        day = _dt.datetime.fromtimestamp(ts_ms / 1000.0, tz=_dt.timezone.utc).strftime("%Y%m%d")
        return self._dir / f"decisions-{self._policy}-{day}.jsonl"

    def write(self, lines: list[dict]) -> None:
        if not lines:
            return
        path = self.path_for(int(lines[0]["ts_ms"]))
        with path.open("a", encoding="utf-8") as fh:
            for line in lines:
                fh.write(json.dumps(line, sort_keys=True, default=str) + "\n")


class BaselineShell:
    def __init__(
        self,
        config: Any,
        source: Source,
        policy: Any,
        dispatcher: Dispatcher,
        redis: Any = None,
        *,
        lock: Optional[OwnerLock] = None,
        decision_log: Optional[DecisionLog] = None,
    ) -> None:
        self.config = config
        self.source = source
        self.policy = policy
        self.dispatcher = dispatcher
        self.redis = redis
        self.lock = lock
        self.log = decision_log or DecisionLog(config.log_dir, config.policy)
        self.stats = ShellStats()
        self._tick = 0
        self._last_dispatch: dict[str, tuple[str, int]] = {}
        self._stop = threading.Event()
        self._stats_lock = threading.Lock()

    # -- one tick ---------------------------------------------------------------------

    def _collect_results(self) -> dict[str, Completed]:
        latest: dict[str, Completed] = {}
        for done in self.dispatcher.drain_results():
            with self._stats_lock:
                if not done.result.ok:
                    self.stats.sm_failures += 1
            if not done.result.ok:
                LOG.warning("SM refused %s %s->%s: %s", done.model, done.direction, done.target,
                            done.result.as_dict())
            latest[done.model] = done
        return latest

    def _owner(self) -> bool:
        if self.config.dry_run:
            return False
        if self.lock is None:
            return True
        return self.lock.ensure()

    def tick_once(self) -> list[dict]:
        results = self._collect_results()
        owner = self._owner()
        effective_dry = bool(self.config.dry_run or not owner)
        snap = self.source.gather(self._tick)
        decisions: Mapping[str, Decision] = self.policy.decide(snap) or {}

        planned: list[dict] = []
        for model, ms in snap.models.items():
            decision = decisions.get(model)
            raw = None if decision is None else int(decision.desired)
            clamped = ms.awake if raw is None else clamp(raw, ms.min_replicas, ms.max_replicas)
            direction = "up" if clamped > ms.awake else "down" if clamped < ms.awake else "none"
            planned.append({
                "model": model, "awake": ms.awake, "raw_desired": raw, "clamped": clamped,
                "direction": direction,
                "reason": "no_decision" if decision is None else decision.reason,
                "inputs": {} if decision is None else dict(decision.inputs),
            })
        order = {"down": 0, "up": 1, "none": 2}
        planned.sort(key=lambda p: (order[p["direction"]], p["model"]))

        lines: list[dict] = []
        for item in planned:
            model, direction = item["model"], item["direction"]
            if direction == "none":
                action = "none"
            elif effective_dry:
                action = "dry_run"
            elif self.dispatcher.submit(model, direction, item["clamped"]):
                action = direction
                self._note_dispatch(model, direction, snap.now_ms)
            else:
                action = "inflight_skip"
            line = {
                "ts_ms": snap.now_ms,
                "tick": snap.tick,
                "policy": self.config.policy,
                **item,
                "action": action,
                "dry_run": effective_dry,
                "owner": owner,
                "inflight": self.dispatcher.inflight(model),
            }
            done = results.get(model)
            if done is not None:
                line["sm_result"] = {
                    "direction": done.direction, "target": done.target, **done.result.as_dict()
                }
            lines.append(line)
            with self._stats_lock:
                key = (model, action)
                self.stats.actions[key] = self.stats.actions.get(key, 0) + 1
                self.stats.decisions += 1

        self.log.write(lines)
        self._write_redis(lines)
        extra = snap.extra or {}
        with self._stats_lock:
            self.stats.ticks += 1
            self.stats.consecutive_failures = 0
            self.stats.effective_dry_run = effective_dry
            self.stats.owner = owner
            self.stats.last_tick_ms = snap.now_ms
            self.stats.event_lag_s = dict(extra.get("event_lag_s") or {})
            self.stats.scrape_failures += int(extra.get("scrape_failed") or 0)
        self._tick += 1
        return lines

    def _note_dispatch(self, model: str, direction: str, now_ms: int) -> None:
        with self._stats_lock:
            self.stats.sm_calls += 1
            prev = self._last_dispatch.get(model)
            if prev is not None and prev[0] != direction and now_ms - prev[1] <= REVERSAL_WINDOW_MS:
                self.stats.reversals[model] = self.stats.reversals.get(model, 0) + 1
            self._last_dispatch[model] = (direction, now_ms)

    def _write_redis(self, lines: list[dict]) -> None:
        if not self.config.write_redis or self.redis is None:
            return
        for line in lines:
            try:
                self.redis.set(decision_key(line["model"]), json.dumps(line, sort_keys=True, default=str),
                               ex=DECISION_TTL_S)
            except Exception as exc:
                with self._stats_lock:
                    self.stats.redis_write_failures += 1
                LOG.warning("decision key write failed for %s: %s", line["model"], exc)

    def safe_tick(self) -> Optional[list[dict]]:
        try:
            return self.tick_once()
        except Exception as exc:
            LOG.exception("tick %d failed; skipped", self._tick)
            with self._stats_lock:
                self.stats.tick_failures += 1
                self.stats.consecutive_failures += 1
                self.stats.last_error = repr(exc)[:500]
            self._tick += 1
            return None

    # -- run --------------------------------------------------------------------------

    def run(self, max_ticks: Optional[int] = None) -> None:
        period = float(self.config.tick_s)
        next_at = time.monotonic()
        done = 0
        while not self._stop.is_set():
            self.safe_tick()
            done += 1
            if max_ticks is not None and done >= max_ticks:
                break
            next_at += period
            delay = next_at - time.monotonic()
            if delay < 0:  # overran: skip the missed slots, never burst
                next_at = time.monotonic()
                delay = 0.0
            self._stop.wait(delay)

    def stop(self) -> None:
        self._stop.set()

    # -- health / metrics -------------------------------------------------------------

    def healthy(self) -> bool:
        with self._stats_lock:
            return self.stats.consecutive_failures < int(self.config.max_tick_failures)

    def health_doc(self) -> dict:
        with self._stats_lock:
            s = self.stats
            return {
                "ok": s.consecutive_failures < int(self.config.max_tick_failures),
                "policy": self.config.policy,
                "ticks": s.ticks,
                "consecutive_failures": s.consecutive_failures,
                "last_error": s.last_error,
                "dry_run_configured": bool(self.config.dry_run),
                "dry_run_effective": s.effective_dry_run,
                "owner": s.owner,
                "last_tick_ms": s.last_tick_ms,
            }

    def metrics_text(self) -> str:
        with self._stats_lock:
            s = self.stats
            policy = self.config.policy
            lines = [
                "# TYPE tre_bl_ticks_total counter",
                f'tre_bl_ticks_total{{policy="{policy}"}} {s.ticks}',
                "# TYPE tre_bl_tick_failures_total counter",
                f'tre_bl_tick_failures_total{{policy="{policy}"}} {s.tick_failures}',
                "# TYPE tre_bl_consecutive_tick_failures gauge",
                f'tre_bl_consecutive_tick_failures{{policy="{policy}"}} {s.consecutive_failures}',
                "# TYPE tre_bl_decisions_total counter",
                f'tre_bl_decisions_total{{policy="{policy}"}} {s.decisions}',
                "# TYPE tre_bl_sm_calls_total counter",
                f'tre_bl_sm_calls_total{{policy="{policy}"}} {s.sm_calls}',
                "# TYPE tre_bl_sm_failures_total counter",
                f'tre_bl_sm_failures_total{{policy="{policy}"}} {s.sm_failures}',
                "# TYPE tre_bl_scrape_failures_total counter",
                f'tre_bl_scrape_failures_total{{policy="{policy}"}} {s.scrape_failures}',
                "# TYPE tre_bl_redis_write_failures_total counter",
                f'tre_bl_redis_write_failures_total{{policy="{policy}"}} {s.redis_write_failures}',
                "# TYPE tre_bl_inflight gauge",
                f'tre_bl_inflight{{policy="{policy}"}} {self.dispatcher.inflight_count()}',
                "# TYPE tre_bl_dry_run gauge",
                f'tre_bl_dry_run{{policy="{policy}"}} {int(s.effective_dry_run)}',
                "# TYPE tre_bl_owner gauge",
                f'tre_bl_owner{{policy="{policy}"}} {int(s.owner)}',
                "# TYPE tre_bl_actions_total counter",
            ]
            for (model, action), count in sorted(s.actions.items()):
                lines.append(f'tre_bl_actions_total{{policy="{policy}",model="{model}",action="{action}"}} {count}')
            lines.append("# TYPE tre_bl_direction_reversals_60s_total counter")
            for model, count in sorted(s.reversals.items()):
                lines.append(f'tre_bl_direction_reversals_60s_total{{policy="{policy}",model="{model}"}} {count}')
            lines.append("# TYPE tre_bl_event_lag_seconds gauge")
            for model, lag in sorted(s.event_lag_s.items()):
                lines.append(f'tre_bl_event_lag_seconds{{policy="{policy}",model="{model}"}} {lag:.3f}')
        return "\n".join(lines) + "\n"


def make_http_server(shell: BaselineShell, port: int, host: str = "0.0.0.0") -> ThreadingHTTPServer:
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802
            if self.path.startswith("/healthz"):
                doc = shell.health_doc()
                body = json.dumps(doc, sort_keys=True).encode("utf-8")
                self._send(200 if doc["ok"] else 503, body, "application/json")
            elif self.path.startswith("/metrics"):
                self._send(200, shell.metrics_text().encode("utf-8"), "text/plain; version=0.0.4")
            else:
                self._send(404, b"not found\n", "text/plain")

        def _send(self, code: int, body: bytes, ctype: str) -> None:
            self.send_response(code)
            self.send_header("content-type", ctype)
            self.send_header("content-length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, fmt: str, *args: Any) -> None:
            LOG.debug("http: " + fmt, *args)

    server = ThreadingHTTPServer((host, port), Handler)
    server.daemon_threads = True
    return server
