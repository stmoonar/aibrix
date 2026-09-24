"""Drain-before-sleep for vLLM hot switching.

Sleeping a vLLM engine aborts every in-flight request. With the flag on, the
service-manager first makes the pod unroutable, then waits (bounded) for the
engine's running + waiting queues to empty before calling ``/sleep``.

Everything in this module is dependency-free: the metric parsers are pure
functions over the Prometheus text exposition, and ``SleepDrainer`` takes its
clock and sleep function as seams so tests never really wait.
"""

from __future__ import annotations

import inspect
import json
import logging
import re
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Callable, Mapping

from tre_sm.allocator.slots import Binding
from tre_sm.state.operations import current_operation
from tre_sm.state.reconcile import POD_STATE_HIDDEN


_TRUTHY = {"1", "true", "yes", "on"}
_RUNNING_METRIC = "vllm:num_requests_running"
_WAITING_METRIC = "vllm:num_requests_waiting"
_E2E_BUCKET_METRIC = "vllm:e2e_request_latency_seconds_bucket"
_SAMPLE = re.compile(
    r"^(?P<name>[A-Za-z_:][A-Za-z0-9_:]*)(?:\{(?P<labels>[^}]*)\})?\s+(?P<value>\S+)"
)
_LE_LABEL = re.compile(r'(?:^|,)\s*le\s*=\s*"(?P<le>[^"]*)"')
SNAPSHOT_TEXT_LIMIT = 4096
AUDIT_RING_SIZE = 256

_audit_logger = logging.getLogger("tre_sm.drain")


@dataclass(frozen=True)
class DrainConfig:
    enabled: bool = False
    default_timeout_s: float = 60.0
    min_timeout_s: float = 30.0
    max_timeout_s: float = 300.0
    poll_interval_s: float = 1.0
    unroutable_timeout_s: float = 30.0

    def __post_init__(self) -> None:
        for name in (
            "default_timeout_s",
            "min_timeout_s",
            "max_timeout_s",
            "poll_interval_s",
            "unroutable_timeout_s",
        ):
            if not float(getattr(self, name)) > 0:
                raise ValueError(f"DrainConfig.{name} must be positive")
        if self.min_timeout_s > self.max_timeout_s:
            raise ValueError(
                "DrainConfig.min_timeout_s must not exceed max_timeout_s "
                f"({self.min_timeout_s} > {self.max_timeout_s})"
            )

    @classmethod
    def from_env(cls, env: Mapping[str, str]) -> "DrainConfig":
        defaults = cls()
        enabled = (
            str(env.get("TRE_SM_DRAIN_BEFORE_SLEEP", "")).strip().lower() in _TRUTHY
        )
        return cls(
            enabled=enabled,
            default_timeout_s=_env_float(
                env, "TRE_SM_DRAIN_DEFAULT_S", defaults.default_timeout_s
            ),
            min_timeout_s=_env_float(env, "TRE_SM_DRAIN_MIN_S", defaults.min_timeout_s),
            max_timeout_s=_env_float(env, "TRE_SM_DRAIN_MAX_S", defaults.max_timeout_s),
            poll_interval_s=_env_float(
                env, "TRE_SM_DRAIN_POLL_S", defaults.poll_interval_s
            ),
        )


def _env_float(env: Mapping[str, str], name: str, default: float) -> float:
    raw = env.get(name)
    if raw is None or not str(raw).strip():
        return default
    try:
        return float(str(raw).strip())
    except ValueError as exc:
        raise ValueError(f"{name} must be a number, got {raw!r}") from exc


def _samples(metrics_text: str):
    for line in (metrics_text or "").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        match = _SAMPLE.match(line)
        if match is None:
            continue
        try:
            value = float(match.group("value"))
        except ValueError:
            continue
        yield match.group("name"), match.group("labels") or "", value


def parse_vllm_load(metrics_text: str) -> tuple[int, int] | None:
    """Sum vLLM running/waiting request gauges over all label sets.

    Returns None when neither gauge is present (not a vLLM metrics page).
    """
    running = 0.0
    waiting = 0.0
    seen = False
    for name, _labels, value in _samples(metrics_text):
        if name == _RUNNING_METRIC:
            running += value
            seen = True
        elif name == _WAITING_METRIC:
            waiting += value
            seen = True
    if not seen:
        return None
    return int(round(running)), int(round(waiting))


def parse_e2e_p95_s(metrics_text: str) -> float | None:
    """p95 of vllm:e2e_request_latency_seconds from its cumulative histogram.

    Buckets are summed across label sets per ``le``; the quantile is linearly
    interpolated inside the bucket that crosses it. When it lands in the +Inf
    bucket the largest finite bound is returned. None when there are no
    observations.
    """
    buckets: dict[float, float] = {}
    for name, labels, value in _samples(metrics_text):
        if name != _E2E_BUCKET_METRIC:
            continue
        match = _LE_LABEL.search(labels)
        if match is None:
            continue
        try:
            le = float(match.group("le"))
        except ValueError:
            continue
        buckets[le] = buckets.get(le, 0.0) + value
    if not buckets:
        return None
    bounds = sorted(buckets)
    total = buckets[bounds[-1]]
    if total <= 0:
        return None
    target = 0.95 * total
    finite = [bound for bound in bounds if bound != float("inf")]
    previous_le = 0.0
    previous_count = 0.0
    for bound in bounds:
        count = buckets[bound]
        if count >= target:
            if bound == float("inf"):
                return finite[-1] if finite else None
            if count <= previous_count:
                return bound
            fraction = (target - previous_count) / (count - previous_count)
            return previous_le + (bound - previous_le) * fraction
        previous_le = bound
        previous_count = count
    return finite[-1] if finite else None


def drain_timeout_s(p95_s: float | None, cfg: DrainConfig) -> float:
    raw = 2.0 * p95_s if p95_s else cfg.default_timeout_s
    return min(max(raw, cfg.min_timeout_s), cfg.max_timeout_s)


def parse_sleep_snapshot(text: str | None) -> list[dict] | None:
    """Parse the custom vLLM /sleep response (JSON list of aborted requests)."""
    if not text or not str(text).strip():
        return None
    try:
        payload = json.loads(text)
    except (TypeError, ValueError):
        return None
    if not isinstance(payload, list):
        return None
    if not all(isinstance(item, dict) for item in payload):
        return None
    return payload


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class SleepDrainer:
    def __init__(
        self,
        runtime_ops,
        vllm_ops,
        cfg: DrainConfig,
        *,
        monotonic: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
        logger: logging.Logger | None = None,
    ) -> None:
        self._runtime_ops = runtime_ops
        self._vllm_ops = vllm_ops
        self._cfg = cfg
        self._monotonic = monotonic
        self._sleep = sleep
        self._logger = logger or _audit_logger

    @property
    def config(self) -> DrainConfig:
        return self._cfg

    def now(self) -> float:
        return self._monotonic()

    def hide(self, binding: Binding) -> float:
        """Take the pod out of routing; returns the monotonic hide time."""
        self._runtime_ops.write_binding_annotations(binding, state=POD_STATE_HIDDEN)
        return self._monotonic()

    def drain(
        self,
        binding: Binding,
        pod_ip: str,
        *,
        hidden_at: float | None = None,
        unroutable_confirmed: bool = False,
    ) -> dict:
        """Wait (bounded) for the engine queues to empty; never raises on timeout.

        The drain window is anchored at ``hidden_at`` (the hide time) when
        given, so several bindings hidden together drain concurrently.
        """
        started = self._monotonic() if hidden_at is None else hidden_at
        if not unroutable_confirmed:
            unroutable_confirmed = self._wait_unroutable(binding)

        record: dict = {
            "serve_id": binding.serve_id,
            "binding_id": binding.binding_id,
            "model": binding.model,
            "pod_ip": pod_ip,
            "drain_timeout_s": None,
            "p95_e2e_s": None,
            "waited_s": 0.0,
            "drained": False,
            "interrupted_running": None,
            "interrupted_waiting": None,
            "unroutable_confirmed": unroutable_confirmed,
            "metrics_available": False,
        }

        text = self._scrape(pod_ip)
        load = parse_vllm_load(text) if text is not None else None
        p95 = parse_e2e_p95_s(text) if text is not None else None
        timeout = drain_timeout_s(p95, self._cfg)
        record["drain_timeout_s"] = timeout
        record["p95_e2e_s"] = p95
        if load is None:
            # No metrics: do not wait blindly, sleep right away.
            record["waited_s"] = max(0.0, self._monotonic() - started)
            return record

        record["metrics_available"] = True
        deadline = started + timeout
        while True:
            running, waiting = load
            record["interrupted_running"] = running
            record["interrupted_waiting"] = waiting
            if running + waiting == 0:
                record["drained"] = True
                break
            remaining = deadline - self._monotonic()
            if remaining <= 0:
                break
            self._sleep(min(self._cfg.poll_interval_s, remaining))
            text = self._scrape(pod_ip)
            next_load = parse_vllm_load(text) if text is not None else None
            if next_load is None:
                record["metrics_available"] = False
                break
            load = next_load
        record["waited_s"] = max(0.0, self._monotonic() - started)
        return record

    def _wait_unroutable(self, binding: Binding) -> bool:
        wait = getattr(self._runtime_ops, "wait_pod_unroutable", None)
        if not callable(wait):
            return False
        try:
            if _accepts_kwarg(wait, "timeout_s"):
                wait(binding, timeout_s=self._cfg.unroutable_timeout_s)
            else:
                wait(binding)
        except TimeoutError:
            self._logger.warning(
                "drain: %s still routable after %.0fs; draining anyway",
                binding.serve_id,
                self._cfg.unroutable_timeout_s,
            )
            return False
        return True

    def _scrape(self, pod_ip: str) -> str | None:
        metrics = getattr(self._vllm_ops, "metrics", None)
        if not callable(metrics):
            return None
        try:
            return metrics(pod_ip, port=8000)
        except Exception:  # pragma: no cover - VllmOps.metrics already swallows.
            return None


def _accepts_kwarg(func: Callable, name: str) -> bool:
    try:
        parameters = inspect.signature(func).parameters.values()
    except (TypeError, ValueError):
        return False
    return any(
        parameter.name == name or parameter.kind is inspect.Parameter.VAR_KEYWORD
        for parameter in parameters
    )


class SleepAuditLog:
    """In-memory ring of the last /sleep calls (never persisted)."""

    def __init__(self, maxlen: int = AUDIT_RING_SIZE) -> None:
        self._records: deque[dict] = deque(maxlen=maxlen)

    def records(self, limit: int | None = None) -> list[dict]:
        records = list(self._records)
        if limit is not None:
            records = records[-limit:] if limit > 0 else []
        return records

    def record_sleep(
        self,
        binding: Binding,
        pod_ip: str,
        result,
        drain_record: dict | None,
    ) -> dict:
        """Append one audit record; journal + log only when a drain ran.

        With ``drain_record is None`` (flag off) this is a pure in-memory
        append and has no other side effect.
        """
        message = getattr(result, "message", "")
        message = message if isinstance(message, str) else str(message or "")
        snapshot = parse_sleep_snapshot(message)
        if drain_record is None:
            record: dict = {
                "serve_id": binding.serve_id,
                "binding_id": binding.binding_id,
                "model": binding.model,
                "pod_ip": pod_ip,
                "drain_enabled": False,
            }
        else:
            record = dict(drain_record)
            record["drain_enabled"] = True
        record.update(
            {
                "action": "sleep",
                "ts": _utc_now_iso(),
                "sleep_status_code": getattr(result, "status_code", None),
                "sleep_success": bool(getattr(result, "success", False)),
                "sleep_snapshot": (
                    snapshot
                    if snapshot is not None
                    else (message[:SNAPSHOT_TEXT_LIMIT] or None)
                ),
                "aborted_count": len(snapshot) if snapshot is not None else None,
            }
        )
        self._records.append(record)
        if drain_record is not None:
            operation = current_operation()
            if operation is not None:
                operation.advance("sleep_drained", details=record)
            _audit_logger.info(json.dumps(record, sort_keys=True, default=str))
        return record
