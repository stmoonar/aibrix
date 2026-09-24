"""Hide-before-sleep and drain-before-sleep for vLLM hot switching.

Sleeping a vLLM engine aborts every in-flight request. Two opt-in flags:

* ``TRE_SM_HIDE_BEFORE_SLEEP``: before every ``/sleep`` the service-manager
  hides the pod (routable=false + hidden annotation), waits until it is out
  of the routable selector, and sends ``X-TRE-Hidden: 1`` on the ``/sleep``
  call (the reissue sidecar refuses a ``/sleep`` without it).
* ``TRE_SM_DRAIN_BEFORE_SLEEP`` (requires the hide flag): additionally waits,
  bounded, for the engine's running + waiting queues to empty.

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


HIDDEN_SLEEP_HEADER = "X-TRE-Hidden"


class SleepConfigError(ValueError):
    """Inconsistent hide/drain/sidecar configuration; raised at startup."""


@dataclass(frozen=True)
class DrainConfig:
    # ``enabled`` is the drain flag (TRE_SM_DRAIN_BEFORE_SLEEP); it requires
    # ``hide_before_sleep`` (TRE_SM_HIDE_BEFORE_SLEEP).
    enabled: bool = False
    default_timeout_s: float = 60.0
    min_timeout_s: float = 30.0
    max_timeout_s: float = 300.0
    poll_interval_s: float = 1.0
    unroutable_timeout_s: float = 30.0
    hide_before_sleep: bool = False
    # One deadline per SM call for hide + wait-unroutable + drain + /sleep of
    # every binding it sleeps. Must stay below the controller's
    # TRE_SM_SLOW_TIMEOUT_SECONDS (default 300 s) with margin.
    sleep_deadline_s: float = 240.0
    # Tail of the deadline kept free for re-acquiring the writer lock and the
    # /sleep calls themselves; the drain window ends this much earlier.
    commit_reserve_s: float = 30.0
    lock_retry_interval_s: float = 0.5
    # A draining marker older than its deadline + this grace (SM crashed or
    # lost the lock mid-drain) is recovered by the drain recovery.
    stale_grace_s: float = 30.0

    def __post_init__(self) -> None:
        for name in (
            "default_timeout_s",
            "min_timeout_s",
            "max_timeout_s",
            "poll_interval_s",
            "unroutable_timeout_s",
            "sleep_deadline_s",
            "commit_reserve_s",
            "lock_retry_interval_s",
            "stale_grace_s",
        ):
            if not float(getattr(self, name)) > 0:
                raise ValueError(f"DrainConfig.{name} must be positive")
        if self.min_timeout_s > self.max_timeout_s:
            raise ValueError(
                "DrainConfig.min_timeout_s must not exceed max_timeout_s "
                f"({self.min_timeout_s} > {self.max_timeout_s})"
            )
        if self.commit_reserve_s >= self.sleep_deadline_s:
            raise ValueError(
                "DrainConfig.commit_reserve_s must be below sleep_deadline_s "
                f"({self.commit_reserve_s} >= {self.sleep_deadline_s})"
            )
        if self.enabled and not self.hide_before_sleep:
            # Draining a pod that still receives traffic never converges and
            # only delays the sleep: fail closed at startup.
            raise SleepConfigError(
                "TRE_SM_DRAIN_BEFORE_SLEEP=true requires TRE_SM_HIDE_BEFORE_SLEEP=true"
            )

    @property
    def drain_before_sleep(self) -> bool:
        return self.enabled

    @classmethod
    def from_env(cls, env: Mapping[str, str]) -> "DrainConfig":
        defaults = cls()
        enabled = (
            str(env.get("TRE_SM_DRAIN_BEFORE_SLEEP", "")).strip().lower() in _TRUTHY
        )
        hide = (
            str(env.get("TRE_SM_HIDE_BEFORE_SLEEP", "")).strip().lower() in _TRUTHY
        )
        return cls(
            enabled=enabled,
            hide_before_sleep=hide,
            default_timeout_s=_env_float(
                env, "TRE_SM_DRAIN_DEFAULT_S", defaults.default_timeout_s
            ),
            min_timeout_s=_env_float(env, "TRE_SM_DRAIN_MIN_S", defaults.min_timeout_s),
            max_timeout_s=_env_float(env, "TRE_SM_DRAIN_MAX_S", defaults.max_timeout_s),
            poll_interval_s=_env_float(
                env, "TRE_SM_DRAIN_POLL_S", defaults.poll_interval_s
            ),
            unroutable_timeout_s=_env_float(
                env, "TRE_SM_UNROUTABLE_TIMEOUT_S", defaults.unroutable_timeout_s
            ),
            sleep_deadline_s=_env_float(
                env, "TRE_SM_SLEEP_DEADLINE_S", defaults.sleep_deadline_s
            ),
            commit_reserve_s=_env_float(
                env, "TRE_SM_SLEEP_COMMIT_RESERVE_S", defaults.commit_reserve_s
            ),
            stale_grace_s=_env_float(
                env, "TRE_SM_DRAIN_STALE_GRACE_S", defaults.stale_grace_s
            ),
        )


def check_reissue_coupling(registry, cfg: DrainConfig) -> None:
    """Fail closed: the reissue sidecar refuses /sleep without X-TRE-Hidden
    (409), which VllmOps reads as an idempotent success that then never
    converges. So the sidecar requires TRE_SM_HIDE_BEFORE_SLEEP."""
    reissue = getattr(registry, "reissue_sidecar", None)
    if bool(getattr(reissue, "enabled", False)) and not cfg.hide_before_sleep:
        raise SleepConfigError(
            "registry reissue_sidecar.enabled=true requires "
            "TRE_SM_HIDE_BEFORE_SLEEP=true (the sidecar only accepts /sleep for "
            "a pod the service-manager hid first, marked by X-TRE-Hidden: 1)"
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

    def sleep_for(self, seconds: float) -> None:
        self._sleep(seconds)

    def drain(
        self,
        binding: Binding,
        pod_ip: str,
        *,
        hidden_at: float | None = None,
        unroutable_confirmed: bool = False,
        deadline: float | None = None,
        should_continue: Callable[[], bool] | None = None,
    ) -> dict:
        """Wait for unroutable, then (drain flag) for empty queues; bounded.

        Never raises on timeout. The drain window is anchored at
        ``hidden_at`` (the hide time) when given, so several bindings hidden
        together drain concurrently. ``deadline`` (same clock) caps both the
        unroutable wait and the drain window; ``should_continue`` is polled
        between scrapes and stops the drain early (the binding was reclaimed).
        """
        started = self._monotonic() if hidden_at is None else hidden_at
        if not unroutable_confirmed:
            timeout = self._cfg.unroutable_timeout_s
            if deadline is not None:
                timeout = min(timeout, deadline - self._monotonic())
            unroutable_confirmed = (
                self.wait_unroutable(binding, timeout_s=timeout) if timeout > 0 else False
            )

        record: dict = {
            "serve_id": binding.serve_id,
            "binding_id": binding.binding_id,
            "model": binding.model,
            "pod_ip": pod_ip,
            "hide_enabled": True,
            "drain_enabled": bool(self._cfg.enabled),
            "drain_timeout_s": None,
            "p95_e2e_s": None,
            "waited_s": 0.0,
            "drained": False,
            "interrupted_running": None,
            "interrupted_waiting": None,
            "unroutable_confirmed": unroutable_confirmed,
            "metrics_available": False,
            "deadline_capped": False,
            "cancelled": False,
        }
        if not self._cfg.enabled:
            # Hide-only mode: unroutable is all we wait for.
            record["waited_s"] = max(0.0, self._monotonic() - started)
            return record

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
        end = started + timeout
        if deadline is not None and deadline < end:
            end = deadline
            record["deadline_capped"] = True
        while True:
            running, waiting = load
            record["interrupted_running"] = running
            record["interrupted_waiting"] = waiting
            if running + waiting == 0:
                record["drained"] = True
                break
            remaining = end - self._monotonic()
            if remaining <= 0:
                break
            if should_continue is not None and not _safe_call(should_continue):
                record["cancelled"] = True
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

    def wait_unroutable(self, binding: Binding, *, timeout_s: float | None = None) -> bool:
        """True once the pod left the routable selector; False on timeout."""
        timeout = self._cfg.unroutable_timeout_s if timeout_s is None else timeout_s
        wait = getattr(self._runtime_ops, "wait_pod_unroutable", None)
        if not callable(wait):
            return False
        try:
            if _accepts_kwarg(wait, "timeout_s"):
                wait(binding, timeout_s=timeout)
            else:
                wait(binding)
        except TimeoutError:
            self._logger.warning(
                "drain: %s still routable after %.0fs; continuing",
                binding.serve_id,
                timeout,
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


def _safe_call(predicate: Callable[[], bool]) -> bool:
    try:
        return bool(predicate())
    except Exception:  # a flaky marker read must not cut the drain short.
        return True


def accepts_kwarg(func: Callable, name: str) -> bool:
    return _accepts_kwarg(func, name)


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
        """Append one audit record; journal + log only when hide/drain ran.

        With ``drain_record is None`` (flags off) this is a pure in-memory
        append and has no other side effect.

        The /sleep response of the 0.10.1-sleep image is NOT a reliable
        abort count: ``AsyncLLM.pause_generation`` calls ``abort()`` first,
        whose ``scheduler.finish_requests`` already frees the requests, so
        the later snapshot usually comes back ``[]`` even when requests were
        cut off. The raw snapshot is kept, ``[]`` is not read as "nothing
        aborted", and the SM-side estimate is the drain's last observed
        running + waiting. The authoritative count is the reissue sidecar's
        ``tre_reissue_*`` metrics.
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
            record["drain_enabled"] = bool(drain_record.get("drain_enabled", True))
        estimate = None
        if drain_record is not None and drain_record.get("interrupted_running") is not None:
            estimate = int(drain_record.get("interrupted_running") or 0) + int(
                drain_record.get("interrupted_waiting") or 0
            )
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
                # A non-empty snapshot is a lower bound; [] means "unknown".
                "aborted_count": len(snapshot) if snapshot else None,
                "aborted_count_reported": (
                    len(snapshot) if snapshot is not None else None
                ),
                "aborted_count_source": "vllm_sleep_response_unreliable",
                "interrupted_estimate": estimate,
                "interrupted_estimate_source": (
                    "sm_drain_last_poll" if estimate is not None else None
                ),
                "authoritative_count_source": "reissue_sidecar_tre_reissue_metrics",
            }
        )
        self._records.append(record)
        if drain_record is not None:
            operation = current_operation()
            if operation is not None:
                operation.advance("sleep_drained", details=record)
            _audit_logger.info(json.dumps(record, sort_keys=True, default=str))
        return record
