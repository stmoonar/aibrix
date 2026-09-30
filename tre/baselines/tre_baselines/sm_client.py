"""Service-manager client and the per-model asynchronous dispatcher.

Contract (service manager on main, ``tre_sm/api/v2.py``):

* ``PUT /v2/models/{m}/target`` body ``{wake_replicas, at_least, sleep_path,
  drain_budget_s}``. ``at_least`` = grow-only (no-op when the model already has that many
  awake), so a re-sent scale-up never shrinks a model. A scale-down is an absolute
  ``wake_replicas`` with a ``sleep_path`` from ``EXTERNAL_SLEEP_PATHS``; the handler
  blocks until the drain ends and wake / create run synchronously inside it, so a call
  can take minutes (the controller uses a 300 s timeout for the same call).
* ``GET /v2/state`` -> ``{"version", "models": {m: {"awake", "bound"}}, "bindings":
  [{"serve_id" (= pod name), "model", "node", "gpu_ids", "awake", "hidden"}], ...}``.
* Refusals are HTTP 409 (writer lock busy, reservation, floor violation, wake conflict,
  cap). Today most bodies are ``{"detail": "<text>"}``; floor violations carry ``error``.
  A structured body (``detail, error, reason, node, gpu_ids, scope, binding_id,
  blocking_binding_id, retry_after_s``) is PROVISIONAL (the T1 line is not merged):
  :func:`parse_sm_error` reads whichever keys are present, nested under ``detail`` or not,
  and falls back to the raw text.

The dispatcher gives every model one worker thread and at most one SM call in flight;
the tick never waits for the SM. :class:`Backoff` spaces out retries after refusals.
"""
from __future__ import annotations

import json
import logging
import queue
import socket
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Optional
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

LOG = logging.getLogger(__name__)

ACTOR = "tre-baseline-scaler"
_STRUCTURED_KEYS = ("error", "reason", "node", "gpu_ids", "scope", "retry_after_s")


@dataclass(frozen=True)
class SMResult:
    ok: bool
    code: Optional[int] = None
    error: Optional[str] = None
    reason: Optional[str] = None
    node: Optional[str] = None
    gpu_ids: Optional[tuple[int, ...]] = None
    scope: Optional[str] = None
    retry_after_s: Optional[float] = None
    detail: Optional[str] = None
    raw: Any = None
    elapsed_s: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        out = {
            "ok": self.ok, "code": self.code, "error": self.error, "reason": self.reason,
            "node": self.node, "gpu_ids": list(self.gpu_ids) if self.gpu_ids is not None else None,
            "scope": self.scope, "retry_after_s": self.retry_after_s, "detail": self.detail,
            "elapsed_s": round(self.elapsed_s, 3),
        }
        if isinstance(self.raw, dict):
            for key in ("binding_id", "blocking_binding_id", "actions", "at_least", "awake"):
                if key in self.raw:
                    out[key] = self.raw[key]
        return out


def _as_int_tuple(value: Any) -> Optional[tuple[int, ...]]:
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        try:
            return tuple(int(v) for v in value)
        except (TypeError, ValueError):
            return None
    try:
        return (int(value),)
    except (TypeError, ValueError):
        return None


def _as_float(value: Any) -> Optional[float]:
    try:
        return None if value is None else float(value)
    except (TypeError, ValueError):
        return None


def parse_sm_error(code: Optional[int], body: bytes | str | None, elapsed_s: float = 0.0) -> SMResult:
    """An SM error answer -> :class:`SMResult`. JSON if possible (structured keys at the
    top level or inside a dict ``detail``; unknown keys ignored), else the text."""
    text = body.decode("utf-8", "replace") if isinstance(body, bytes) else (body or "")
    try:
        parsed = json.loads(text) if text else None
    except ValueError:
        parsed = None
    if not isinstance(parsed, dict):
        return SMResult(ok=False, code=code, error="http_error", detail=text[:2000] or None,
                        raw=text[:2000], elapsed_s=elapsed_s)
    merged: dict[str, Any] = {}
    detail = parsed.get("detail")
    if isinstance(detail, dict):
        merged.update(detail)
        detail_text = detail.get("detail") or detail.get("message")
    else:
        detail_text = detail
    merged.update({k: v for k, v in parsed.items() if k != "detail"})
    return SMResult(
        ok=False,
        code=code,
        error=str(merged["error"]) if merged.get("error") is not None else "http_error",
        reason=str(merged["reason"]) if merged.get("reason") is not None else None,
        node=str(merged["node"]) if merged.get("node") is not None else None,
        gpu_ids=_as_int_tuple(merged.get("gpu_ids")),
        scope=str(merged["scope"]) if merged.get("scope") is not None else None,
        retry_after_s=_as_float(merged.get("retry_after_s")),
        detail=None if detail_text is None else str(detail_text)[:2000],
        raw=parsed,
        elapsed_s=elapsed_s,
    )


class SMClient:
    """Blocking HTTP client (stdlib). Every call is made from a worker thread."""

    def __init__(self, base_url: str, *, timeout_s: float = 300.0, state_timeout_s: float = 5.0,
                 actor: str = ACTOR) -> None:
        self._base = base_url.rstrip("/")
        self.timeout_s = float(timeout_s)
        self.state_timeout_s = float(state_timeout_s)
        self._actor = actor

    def _open(self, method: str, path: str, body: Optional[dict], timeout_s: float):
        data = None if body is None else json.dumps(body).encode("utf-8")
        headers = {"x-tre-actor": self._actor, "accept": "application/json"}
        if data is not None:
            headers["content-type"] = "application/json"
        req = Request(self._base + path, data=data, method=method, headers=headers)
        return urlopen(req, timeout=timeout_s)

    def get_state(self) -> dict:
        """``GET /v2/state``; raises on any failure (the caller skips the tick)."""
        with self._open("GET", "/v2/state", None, self.state_timeout_s) as resp:
            state = json.loads(resp.read().decode("utf-8"))
        if not isinstance(state, dict) or not isinstance(state.get("models"), dict):
            raise ValueError("malformed /v2/state: no models object")
        return state

    def put_target(self, model: str, body: Mapping[str, Any], timeout_s: Optional[float] = None) -> SMResult:
        started = time.monotonic()
        timeout = self.timeout_s if timeout_s is None else float(timeout_s)
        try:
            with self._open("PUT", f"/v2/models/{model}/target", dict(body), timeout) as resp:
                raw = resp.read()
                code = resp.status
        except HTTPError as exc:
            return parse_sm_error(exc.code, exc.read(), time.monotonic() - started)
        except (socket.timeout, TimeoutError) as exc:
            return SMResult(ok=False, error="timeout", detail=str(exc), elapsed_s=time.monotonic() - started)
        except URLError as exc:
            reason = exc.reason
            kind = "timeout" if isinstance(reason, (socket.timeout, TimeoutError)) else "transport"
            return SMResult(ok=False, error=kind, detail=str(reason), elapsed_s=time.monotonic() - started)
        except OSError as exc:
            return SMResult(ok=False, error="transport", detail=str(exc), elapsed_s=time.monotonic() - started)
        elapsed = time.monotonic() - started
        try:
            parsed = json.loads(raw.decode("utf-8")) if raw else {}
        except ValueError:
            parsed = raw.decode("utf-8", "replace")[:2000]
        return SMResult(ok=True, code=code, raw=parsed, elapsed_s=elapsed)


def target_body(direction: str, target: int, *, sleep_path: str, drain_budget_s: Optional[float]) -> dict:
    """Request body of one scale action (``up`` = grow-only, ``down`` = absolute)."""
    if direction == "up":
        return {"wake_replicas": int(target), "at_least": True}
    if direction == "down":
        body: dict[str, Any] = {"wake_replicas": int(target), "sleep_path": sleep_path}
        if drain_budget_s is not None:
            body["drain_budget_s"] = float(drain_budget_s)
        return body
    raise ValueError(f"unknown direction {direction!r}")


DEFAULT_BACKOFF_MAX_S = 60.0


class Backoff:
    """Per-model exponential backoff after a failed SM call (the MVP has no arbiter, so a
    model the SM keeps refusing would otherwise be re-sent every tick).

    The first failure waits ``max(retry_after_s, tick_s)``; each further consecutive
    failure doubles the previous wait (capped at ``max_s``) but never waits less than the
    SM's ``retry_after_s``. :meth:`reset` (a successful call, or the policy's desired count
    back at the awake count) clears it. Times are the snapshot's Redis-clock ms.
    """

    def __init__(self, tick_s: float, max_s: float = DEFAULT_BACKOFF_MAX_S) -> None:
        self.tick_s = max(0.0, float(tick_s))
        self.max_s = max(self.tick_s, float(max_s))
        self._delay_s: dict[str, float] = {}
        self._until_ms: dict[str, int] = {}

    def failed(self, model: str, now_ms: int, retry_after_s: Optional[float] = None) -> float:
        retry = max(0.0, float(retry_after_s)) if retry_after_s is not None else 0.0
        prev = self._delay_s.get(model)
        delay = max(retry, self.tick_s) if prev is None else max(retry, prev * 2.0)
        delay = max(min(delay, self.max_s), retry)  # the SM's own retry_after is honoured
        self._delay_s[model] = delay
        self._until_ms[model] = int(now_ms) + int(round(delay * 1000.0))
        return delay

    def remaining_s(self, model: str, now_ms: int) -> Optional[float]:
        """Seconds still to wait, or None when a call may be sent."""
        until = self._until_ms.get(model)
        if until is None or int(now_ms) >= until:
            return None
        return (until - int(now_ms)) / 1000.0

    def reset(self, model: str) -> None:
        self._delay_s.pop(model, None)
        self._until_ms.pop(model, None)

    def delay_s(self, model: str) -> Optional[float]:
        return self._delay_s.get(model)


@dataclass(frozen=True)
class Completed:
    model: str
    direction: str
    target: int
    body: Mapping[str, Any]
    result: SMResult
    submitted_seq: int


@dataclass
class _Worker:
    thread: threading.Thread
    jobs: "queue.Queue[Optional[tuple[int, str, int, dict]]]"


class Dispatcher:
    """One worker thread per model; at most one SM call in flight per model.

    :meth:`submit` returns False (the caller logs ``inflight_skip``) when the model
    already has a call queued or running; it never blocks. Finished calls are collected
    with :meth:`drain_results`.
    """

    def __init__(self, put_target: Callable[[str, Mapping[str, Any]], SMResult], *,
                 sleep_path: str = "scale_down", drain_budget_s: Optional[float] = None) -> None:
        self._put = put_target
        self._sleep_path = sleep_path
        self._drain_budget_s = drain_budget_s
        self._lock = threading.Lock()
        self._inflight: dict[str, tuple[str, int]] = {}
        self._workers: dict[str, _Worker] = {}
        self._results: "queue.Queue[Completed]" = queue.Queue()
        self._seq = 0
        #: Submission log (seq, model, direction, target), for tests and debugging.
        self.submitted: list[tuple[int, str, str, int]] = []
        self._closed = False

    def inflight(self, model: str) -> bool:
        with self._lock:
            return model in self._inflight

    def inflight_count(self) -> int:
        with self._lock:
            return len(self._inflight)

    def submit(self, model: str, direction: str, target: int) -> bool:
        body = target_body(direction, target, sleep_path=self._sleep_path, drain_budget_s=self._drain_budget_s)
        with self._lock:
            if self._closed:
                raise RuntimeError("dispatcher is closed")
            if model in self._inflight:
                return False
            self._inflight[model] = (direction, int(target))
            self._seq += 1
            seq = self._seq
            self.submitted.append((seq, model, direction, int(target)))
            worker = self._workers.get(model)
            if worker is None:
                jobs: "queue.Queue[Optional[tuple[int, str, int, dict]]]" = queue.Queue()
                thread = threading.Thread(
                    target=self._run, args=(model, jobs), name=f"bl-sm-{model}", daemon=True
                )
                worker = _Worker(thread=thread, jobs=jobs)
                self._workers[model] = worker
                thread.start()
        worker.jobs.put((seq, direction, int(target), body))
        return True

    def _run(self, model: str, jobs: "queue.Queue[Optional[tuple[int, str, int, dict]]]") -> None:
        while True:
            job = jobs.get()
            if job is None:
                return
            seq, direction, target, body = job
            try:
                result = self._put(model, body)
            except Exception as exc:  # the client should not raise; never lose the flag
                LOG.exception("SM call for %s raised", model)
                result = SMResult(ok=False, error="exception", detail=repr(exc)[:2000])
            self._results.put(Completed(model, direction, target, body, result, seq))
            with self._lock:
                self._inflight.pop(model, None)

    def drain_results(self) -> list[Completed]:
        out: list[Completed] = []
        while True:
            try:
                out.append(self._results.get_nowait())
            except queue.Empty:
                return out

    def close(self, join_s: float = 0.0) -> None:
        with self._lock:
            self._closed = True
            workers = list(self._workers.values())
        for worker in workers:
            worker.jobs.put(None)
        if join_s > 0:
            deadline = time.monotonic() + join_s
            for worker in workers:
                worker.thread.join(max(0.0, deadline - time.monotonic()))
