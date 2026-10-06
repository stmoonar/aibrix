"""Service-manager client and the per-model asynchronous dispatcher.

Contract (service manager on main, ``tre_sm/api/v2.py``):

* ``PUT /v2/models/{m}/target`` body ``{wake_replicas, at_least}`` (scale-up) or
  ``{wake_replicas, sleep_path}`` (scale-down). ``at_least`` = grow-only (no-op when the
  model already has that many awake), so a re-sent scale-up never shrinks a model. A
  scale-down is an absolute ``wake_replicas`` on the **abort sleep path**: no drain. The SM
  hides the pod, waits for the gateway ack and sleeps it with ``mode=abort``; requests in
  flight are cut off and continued by the reissue sidecar (the transparent sleep every arm
  shares). ``sleep_path`` names a path in the registry's
  ``service_manager.sleep.no_drain_paths`` (default ``urgent``; the SM on main drains on
  the other paths, the whole-lock SM never drains); ``drain_budget_s`` is never sent. The
  handler returns once the sleep is committed (GPU released) or the wake done.
* ``GET /v2/state`` -> ``{"version", "models": {m: {"awake", "bound"}}, "bindings":
  [{"serve_id" (= pod name), "model", "node", "gpu_ids", "awake", "hidden"}], ...}``.
* Refusals are HTTP 409 (writer lock busy ``writer_busy``, wake conflict, cap,
  ``routable_unknown``). Wake conflicts and ``writer_busy`` carry a structured body
  (``detail, error, reason, node, gpu_ids, scope, binding_id, blocking_binding_id,
  retry_after_s``); others only ``{"detail": "<text>"}``. :func:`parse_sm_error` reads
  whichever keys are present, nested under ``detail`` or not, and falls back to the raw
  text. The whole-lock SM answers some outcomes with 200, not 409: a shrink clamped at the
  replica floor (``taken``, ``clamped_by_floor``) and a grow-only scale-up that could not
  place every wake (``unfilled``, ``refusals``); :meth:`SMResult.as_dict` logs those keys.

The dispatcher gives every model one worker thread and at most one SM call in flight;
the tick never waits for the SM. A scale-up from ``awake`` to ``target`` goes out one
replica at a time (``awake+1``, ``awake+2``, ... each a grow-only call) and stops at the
first refusal: the SM refuses a whole target when any wake it needs is blocked
(``tre_sm/api/v2.py`` ``_plan_target``), so one call for several replicas would leave a GPU
that is free right now unused while another is still held. The answer reports how far it
got (``Completed.reached``; ``partial_fill`` = some granted, then refused). This is a
substrate adaptation for the baseline arms; the SM is unchanged. Right before each call the worker asks the dispatcher's
``guard`` (set by the shell: still the owner-lock holder, controller still in observe);
when the answer is a reason, the call is dropped (``SMResult.error = "dropped"``,
``reason`` = why) and never reaches the SM. :class:`Backoff` spaces out retries after
refusals.
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
#: ``SMResult.error`` of a call the dispatcher's guard dropped before it reached the SM.
DROPPED = "dropped"
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
            for key in ("binding_id", "blocking_binding_id", "actions", "at_least", "awake",
                        "taken", "clamped_by_floor", "unfilled", "refusals"):
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


#: Default sleep path of a scale-down: a no-drain (abort) path on the SM on main
#: (``DEFAULT_NO_DRAIN_PATHS``) and the whole-lock SM's default. The whole-lock SM clamps a
#: shrink at the replica floor (200, ``clamped_by_floor`` in the logged result).
DEFAULT_ABORT_SLEEP_PATH = "urgent"


def target_body(direction: str, target: int, *, abort_sleep_path: str = DEFAULT_ABORT_SLEEP_PATH) -> dict:
    """Request body of one scale action (``up`` = grow-only, ``down`` = absolute, abort
    sleep: hide -> gateway ack -> ``/sleep mode=abort``, no drain)."""
    if direction == "up":
        return {"wake_replicas": int(target), "at_least": True}
    if direction == "down":
        return {"wake_replicas": int(target), "sleep_path": abort_sleep_path}
    raise ValueError(f"unknown direction {direction!r}")


#: Cap of the fallback wait. Not a hold: a refusal is retried as soon as the SM state
#: changes (the shell clears it); this cap only bounds the wait when the cause of a
#: refusal is not visible in ``/v2/state`` (writer lock busy, a gpu-truth sample not yet
#: fresh) or the SM is unreachable - a few ticks, not the cold-start minute.
DEFAULT_BACKOFF_MAX_S = 10.0


class Backoff:
    """Per-model exponential backoff after a failed SM call (the MVP has no arbiter, so a
    model the SM keeps refusing would otherwise be re-sent every tick).

    The first failure waits ``max(retry_after_s, tick_s)``; each further consecutive
    failure doubles the previous wait (capped at ``max_s``) but never waits less than the
    SM's ``retry_after_s``. :meth:`reset` (a successful call, the policy's desired count
    back at the awake count, or - for an SM refusal - any change of the SM state version)
    clears it. Times are the snapshot's Redis-clock ms.
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
    #: The answer to the last call made (a refusal when the steps stopped early).
    result: SMResult
    submitted_seq: int
    #: Awake count the scale-up started from (None: one call for ``target``).
    start: Optional[int] = None
    #: Highest target the SM granted (None: no call granted).
    reached: Optional[int] = None
    #: SM calls made for this job (a dropped step is not a call).
    steps: int = 1

    @property
    def progressed(self) -> bool:
        """At least one replica was added (the SM state changed because of us)."""
        return self.reached is not None and (self.start is None or self.reached > self.start)

    @property
    def partial_fill(self) -> bool:
        """A stepped scale-up that got some replicas, then was refused or dropped."""
        return (not self.result.ok) and self.start is not None and self.progressed

    def as_dict(self) -> dict[str, Any]:
        out = {"direction": self.direction, "target": self.target, **self.result.as_dict()}
        if self.start is not None:
            out.update(start=self.start, reached=self.reached, steps=self.steps,
                       partial_fill=self.partial_fill)
        return out


@dataclass
class _Worker:
    thread: threading.Thread
    jobs: "queue.Queue[Optional[tuple[int, str, int, Optional[int]]]]"


class Dispatcher:
    """One worker thread per model; at most one SM call in flight per model.

    :meth:`submit` returns False (the caller logs ``inflight_skip``) when the model
    already has a call queued or running; it never blocks. Finished calls are collected
    with :meth:`drain_results`.
    """

    def __init__(self, put_target: Callable[[str, Mapping[str, Any]], SMResult], *,
                 abort_sleep_path: str = DEFAULT_ABORT_SLEEP_PATH,
                 guard: Optional[Callable[[], Optional[str]]] = None) -> None:
        self._put = put_target
        #: Called right before every SM call: None = go, a string = drop the call (why).
        self.guard = guard
        self._abort_sleep_path = abort_sleep_path
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

    def inflight_downs(self) -> list[str]:
        """Models whose scale-down call is queued or running (not yet answered)."""
        with self._lock:
            return sorted(m for m, (direction, _t) in self._inflight.items() if direction == "down")

    def submit(self, model: str, direction: str, target: int, start: Optional[int] = None) -> bool:
        """Queue one scale action. ``start`` (scale-up only) = the awake count the decision
        saw: the worker then asks for ``start+1``, ``start+2``, ... ``target`` one call at a
        time and stops at the first refusal. Without it, one call for ``target``."""
        target_body(direction, target, abort_sleep_path=self._abort_sleep_path)  # validates
        if direction != "up" or start is None or int(start) >= int(target):
            start = None
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
        worker.jobs.put((seq, direction, int(target), None if start is None else int(start)))
        return True

    def _check_guard(self, model: str, direction: str, target: int) -> Optional[str]:
        try:
            why = self.guard() if self.guard is not None else None
        except Exception as exc:  # cannot tell whether we may act: do not act
            why = f"guard_failed: {exc!r}"[:200]
        if why:
            LOG.warning("SM call %s %s->%s dropped: %s", model, direction, target, why)
        return why or None

    def _call(self, model: str, body: dict) -> SMResult:
        try:
            return self._put(model, body)
        except Exception as exc:  # the client should not raise; never lose the flag
            LOG.exception("SM call for %s raised", model)
            return SMResult(ok=False, error="exception", detail=repr(exc)[:2000])

    def _run(self, model: str, jobs: "queue.Queue[Optional[tuple[int, str, int, Optional[int]]]]") -> None:
        while True:
            job = jobs.get()
            if job is None:
                return
            seq, direction, target, start = job
            try:
                done = self._execute(model, seq, direction, target, start)
            except Exception as exc:  # never lose the in-flight flag
                LOG.exception("SM job for %s failed", model)
                body = target_body(direction, target, abort_sleep_path=self._abort_sleep_path)
                done = Completed(model, direction, target, body,
                                 SMResult(ok=False, error="exception", detail=repr(exc)[:2000]), seq,
                                 start=start)
            self._results.put(done)
            with self._lock:
                self._inflight.pop(model, None)

    def _execute(self, model: str, seq: int, direction: str, target: int, start: Optional[int]) -> Completed:
        steps = [target] if start is None else list(range(start + 1, target + 1))
        reached: Optional[int] = None
        calls = 0
        body = target_body(direction, steps[0], abort_sleep_path=self._abort_sleep_path)
        result = SMResult(ok=False, error=DROPPED, reason="no_step")
        for step in steps:
            # Owner lock and controller mode are checked again before every call.
            why = self._check_guard(model, direction, step)
            if why:
                result = SMResult(ok=False, error=DROPPED, reason=str(why))
                break
            body = target_body(direction, step, abort_sleep_path=self._abort_sleep_path)
            result = self._call(model, body)
            calls += 1
            if not result.ok:
                if reached is not None:
                    LOG.info("SM partial fill %s: %s of %s->%s, then %s", model, reached, start, target,
                             result.as_dict())
                break
            reached = step
        return Completed(model, direction, target, body, result, seq, start=start,
                         reached=reached, steps=calls)

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
