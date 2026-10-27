"""Service-manager client and the asynchronous plan dispatcher.

Contract (service manager, ``tre_sm/api/v2.py``, design ``docs/design/20261027-sm-plans.md``):

* ``POST /v2/plans`` body ``{plan_id, sleeps: [...], wakes: [...]}``, ONE writer-lock hold for
  the whole plan, at most one item per model. A scale-up is a wake item ``{model, at_least}``
  (grow-only: a no-op when the model already has that many awake, so a re-sent scale-up
  never shrinks a model; the SM fills it best effort - free GPUs first, then GPUs the plan's
  own sleeps release, in the same hold). A scale-down is a sleep item ``{model, at_most,
  sleep_path}`` on the **abort sleep path**: no drain. The SM hides the pod, waits for the
  gateway ack and sleeps it with ``mode=abort``; requests in flight are cut off and continued
  by the reissue sidecar (the transparent sleep every arm shares). ``sleep_path`` names a
  path in the registry's ``service_manager.sleep.no_drain_paths`` (default ``urgent``). The
  SM clamps an ``at_most`` at the replica floor (``clamped_by_floor``). The handler returns
  once every sleep is committed (GPU released) and every wake done.
* Response: ``{plan_id, version, status, sleeps: [item], wakes: [item], phases_ms}``; an item
  has ``model``, ``status`` (``done | noop | clamped | partial | refused | failed``) and, on
  a wake item, ``awake_before`` / ``awake_after`` / ``woke`` / ``unfilled`` / ``refusals``,
  on a sleep item ``taken`` / ``clamped_by_floor``; a refused or failed item carries the
  structured 409 body of the old single-model endpoint under ``error``. HTTP 200 when
  anything changed or every item was a no-op / clamp; 409 ``error: partial`` (the full
  response in the body) when nothing changed and an item was refused or failed; 409
  ``writer_busy`` / ``routable_unknown`` and 400 before anything was done (no items).
* ``GET /v2/state`` -> ``{"version", "models": {m: {"awake", "bound"}}, "bindings":
  [{"serve_id" (= pod name), "model", "node", "gpu_ids", "awake", "hidden"}], ...}``.
* :func:`parse_sm_error` reads a structured 409 body (``detail, error, reason, node, gpu_ids,
  scope, binding_id, blocking_binding_id, retry_after_s``), nested under ``detail`` or not,
  and falls back to the raw text. :func:`plan_item_results` turns one plan answer into one
  :class:`SMResult` per model: an item refused / failed reads like the old endpoint's 409
  (``code`` 409, located by its ``error``), a partial wake like the old 200 with
  ``unfilled`` (located by its first refusal), a whole-plan answer without items (409
  ``writer_busy``, 400, transport error, timeout) is every model's answer.

The dispatcher sends ONE plan per tick covering every model's decision, so a baseline gets
the actuation of the TRE controller (one decision, one plan, one lock hold): the comparison
is about policy, not actuation. One plan at a time: a model whose item is in the plan in
flight gets ``inflight_skip`` (its next decision sees the answer, as with one call per model
before); every other model's decision waits for the next plan, and only the latest tick's
decisions are kept (latest wins - an older decision of the same model, or of a model that no
longer wants to scale, is never sent). The next plan goes out as soon as the one in flight
returns; no batching timer. The tick never waits for the SM. Right before each POST the
worker asks the dispatcher's ``guard`` (set by the shell: still the owner-lock holder,
controller still in observe); when the answer is a reason, the plan is dropped
(``SMResult.error = "dropped"``, ``reason`` = why) and never reaches the SM.

After a scale-up refusal (an SM answer) :class:`RefusalGate` holds the next scale-up of
that model until the SM state shows a change (no timer when ``/v2/state`` can show the
cause); :class:`Backoff` spaces out retries of failed scale-downs and of calls the SM
never answered.
"""
from __future__ import annotations

import json
import logging
import queue
import socket
import threading
import time
import uuid
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
    #: The binding that holds the refused GPU (structured wake conflicts), if named.
    blocking_binding_id: Optional[str] = None

    def as_dict(self) -> dict[str, Any]:
        out = {
            "ok": self.ok, "code": self.code, "error": self.error, "reason": self.reason,
            "node": self.node, "gpu_ids": list(self.gpu_ids) if self.gpu_ids is not None else None,
            "scope": self.scope, "retry_after_s": self.retry_after_s, "detail": self.detail,
            "elapsed_s": round(self.elapsed_s, 3),
        }
        if isinstance(self.raw, dict):
            for key in ("binding_id", "blocking_binding_id", "plan_id", "plan_status", "http_status", "status",
                        "awake_before", "awake_after", "taken", "clamped_by_floor", "unfilled", "refusals"):
                if key in self.raw:
                    out[key] = self.raw[key]
            woke = self.raw.get("woke")
            if isinstance(woke, list):
                out["woke"] = [w.get("serve_id") if isinstance(w, dict) else w for w in woke]
        if self.blocking_binding_id is not None:
            out.setdefault("blocking_binding_id", self.blocking_binding_id)
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
        blocking_binding_id=(str(merged["blocking_binding_id"])
                             if merged.get("blocking_binding_id") is not None else None),
    )


def located_refusal(result: SMResult, where: Any) -> SMResult:
    """``result`` as a refusal located by the structured body ``where`` (a plan item's
    ``error`` or its first ``refusals`` entry); ``raw`` / ``code`` / ``elapsed_s`` kept."""
    located = parse_sm_error(result.code, json.dumps(where), result.elapsed_s) if isinstance(where, dict) else None
    return SMResult(
        ok=False, code=result.code, error=(located.error if located else result.error or "unfilled"),
        reason=located.reason if located else result.reason or "unfilled", node=located.node if located else None,
        gpu_ids=located.gpu_ids if located else None, scope=located.scope if located else None,
        retry_after_s=located.retry_after_s if located else None,
        detail=located.detail if located else result.detail, raw=result.raw, elapsed_s=result.elapsed_s,
        blocking_binding_id=located.blocking_binding_id if located else None,
    )


#: Item statuses that are a success for the caller (``clamped``: a shrink the SM stopped at
#: the replica floor, as the old ``/target`` 200 with ``clamped_by_floor``).
ITEM_OK = frozenset({"done", "noop", "clamped"})


def _plan_items(raw: Any) -> dict[str, dict]:
    """``model -> item`` of a plan answer (both lists; empty when it has none)."""
    out: dict[str, dict] = {}
    if isinstance(raw, dict):
        for kind in ("sleeps", "wakes"):
            for item in raw.get(kind) or ():
                if isinstance(item, dict) and item.get("model") is not None:
                    out[str(item["model"])] = item
    return out


def _item_result(plan: SMResult, item: dict) -> SMResult:
    """One plan item as the answer of the old single-model call."""
    raw = dict(item)
    whole = plan.raw if isinstance(plan.raw, dict) else {}
    raw.update(plan_id=whole.get("plan_id"), plan_status=whole.get("status"), http_status=plan.code)
    error = item.get("error") if isinstance(item.get("error"), dict) else None
    if error is not None and error.get("binding_id") is not None:
        raw.setdefault("binding_id", error["binding_id"])
    status = item.get("status")
    base = SMResult(ok=True, code=200, raw=raw, elapsed_s=plan.elapsed_s)
    if status in ITEM_OK:
        return base
    if status == "partial":
        # Some wakes done, the rest unplaced: like the old 200 with ``unfilled``.
        refusals = item.get("refusals")
        first = refusals[0] if isinstance(refusals, list) and refusals and isinstance(refusals[0], dict) else None
        return located_refusal(SMResult(ok=False, code=200, error="unfilled", reason="unfilled", raw=raw,
                                        elapsed_s=plan.elapsed_s), error or first)
    # refused / failed (or a status this client does not know): the old endpoint's 409.
    return located_refusal(SMResult(ok=False, code=409, error=str(status or "unknown_status"),
                                    reason=str(status or "unknown_status"), raw=raw, elapsed_s=plan.elapsed_s),
                           error)


def plan_item_results(plan: SMResult, models: list[str] | tuple[str, ...]) -> dict[str, SMResult]:
    """One :class:`SMResult` per model of a plan answer. An answer that lists items (200, or
    409 ``partial``) gives each model its item; one without items (409 ``writer_busy`` /
    ``routable_unknown``, 400, 5xx, transport error, timeout, dropped) is every model's."""
    items = _plan_items(plan.raw)
    out: dict[str, SMResult] = {}
    for model in models:
        item = items.get(model)
        if item is not None:
            out[model] = _item_result(plan, item)
        elif items or plan.ok:
            out[model] = SMResult(ok=False, code=plan.code, error="missing_item",
                                  detail=f"the plan answer has no item for {model}", raw=plan.raw,
                                  elapsed_s=plan.elapsed_s)
        else:
            out[model] = plan
    return out


class SMClient:
    """Blocking HTTP client (stdlib). Every call is made from a worker thread."""

    def __init__(self, base_url: str, *, timeout_s: float = 360.0, state_timeout_s: float = 5.0,
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

    def post_plan(self, body: Mapping[str, Any], timeout_s: Optional[float] = None) -> SMResult:
        """``POST /v2/plans``; never raises. 200 -> ``ok`` with the plan response in ``raw``;
        an HTTP error -> :func:`parse_sm_error` (a 409 ``partial`` keeps its items in ``raw``)."""
        started = time.monotonic()
        timeout = self.timeout_s if timeout_s is None else float(timeout_s)
        try:
            with self._open("POST", "/v2/plans", dict(body), timeout) as resp:
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


#: Default sleep path of a scale-down: a no-drain (abort) path (``DEFAULT_NO_DRAIN_PATHS``).
#: The SM clamps a shrink at the replica floor (item ``clamped``, ``clamped_by_floor``).
DEFAULT_ABORT_SLEEP_PATH = "urgent"


def plan_item(model: str, direction: str, target: int, *,
              abort_sleep_path: str = DEFAULT_ABORT_SLEEP_PATH) -> dict:
    """The plan item of one scale action: ``up`` = wake item, grow-only ``at_least``;
    ``down`` = sleep item ``at_most`` on the abort path (hide -> gateway ack -> ``/sleep
    mode=abort``, no drain)."""
    if direction == "up":
        return {"model": model, "at_least": int(target)}
    if direction == "down":
        return {"model": model, "at_most": int(target), "sleep_path": abort_sleep_path}
    raise ValueError(f"unknown direction {direction!r}")


#: Cap of the fallback wait of :class:`Backoff` (failed scale-downs, calls the SM never
#: answered). Not a hold: a refused scale-down is retried as soon as the SM state changes
#: (the shell clears it); this cap only bounds the wait when the SM is unreachable or the
#: cause is not visible in ``/v2/state`` - a few ticks, not the cold-start minute.
DEFAULT_BACKOFF_MAX_S = 10.0


class Backoff:
    """Per-model exponential backoff after a failed scale-down or a call the SM never
    answered (transport error, timeout). Scale-up refusals use :class:`RefusalGate`.

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


#: Refusal reasons whose cause ``/v2/state`` ``gpus[]`` shows (an awake binding, a load or
#: wake in flight, unexplained memory use): a refused scale-up waits for a state change,
#: never for a timer.
STATE_VISIBLE_REASONS = frozenset({
    "slot_occupied", "resident_awake", "gpu_truth_used", "lease_starting", "lease_waking",
    "wake_in_progress", "lease_conflict",
})


@dataclass(frozen=True)
class StateView:
    """What a refused scale-up is re-checked against, from one tick's ``/v2/state``.

    ``key`` = store version + wakeable GPU set (None = the source reports no SM state);
    ``awake_bindings`` = binding ids awake on some GPU; ``free_gpus`` = per model, the
    ``node/gpu`` of its sleeping bindings whose GPUs are all wakeable. None = unknown."""

    key: Any = None
    awake_bindings: Optional[frozenset] = None
    free_gpus: Optional[Mapping[str, frozenset]] = None

    def free(self, model: str) -> Optional[frozenset]:
        if self.free_gpus is None or self.free_gpus.get(model) is None:
            return None
        return frozenset(self.free_gpus[model])


@dataclass(frozen=True)
class Refusal:
    """One refused scale-up: where (``node`` / ``gpus`` as ``node/gpu``), by whom
    (``blocking``), and the state view the refused call was decided on."""

    view: StateView
    at_ms: int
    reason: Optional[str]
    node: Optional[str]
    scope: Optional[str]
    gpus: frozenset
    blocking: Optional[str]
    #: The cause shows in ``/v2/state``: retried on evidence only, no timer.
    visible: bool
    retry_after_s: Optional[float]

    def as_dict(self, now_ms: int) -> dict[str, Any]:
        return {"reason": self.reason, "node": self.node, "gpus": sorted(self.gpus),
                "blocking_binding_id": self.blocking, "visible": self.visible,
                "since_s": round((int(now_ms) - self.at_ms) / 1000.0, 3)}


class RefusalGate:
    """Per-model record of the last scale-up refusal; the next scale-up goes out on the
    first tick with evidence that the state changed (:meth:`evidence`):

    * ``blocker_asleep`` - the blocking binding is no longer awake;
    * ``gpu_freed`` - a GPU where the model has a sleeping binding is free now that was
      neither free nor refused when the refused call was decided;
    * ``state_changed`` - the SM store version or the wakeable GPU set differs from the
      view the refused call was decided on (also covers a change between the refusal and
      the tick that collected it).

    No timer when the cause is state-visible (:data:`STATE_VISIBLE_REASONS` or a named
    blocking binding, and the source reports the state): an unchanged state is never
    re-sent, so no busy loop. Otherwise (a cause ``/v2/state`` cannot show: writer lock,
    a failed wake, a text-only 409) ``retry_after`` also opens it once the SM's
    ``retry_after_s`` has passed (no hint: the next tick) - the SM's own word on when its
    hidden cause may be gone, at most one call per tick. The record is cleared by a
    successful call or the desired count back at awake."""

    def __init__(self) -> None:
        self._last: dict[str, Refusal] = {}

    def refused(self, model: str, result: "SMResult", view: StateView, now_ms: int) -> Refusal:
        node = result.node
        gpus = frozenset(f"{node}/{g}" for g in (result.gpu_ids or ())) if node else frozenset()
        blocking = result.blocking_binding_id
        visible = view.key is not None and (blocking is not None or result.reason in STATE_VISIBLE_REASONS)
        record = Refusal(view=view, at_ms=int(now_ms), reason=result.reason, node=node, scope=result.scope,
                         gpus=gpus, blocking=blocking, visible=visible, retry_after_s=result.retry_after_s)
        self._last[model] = record
        return record

    def get(self, model: str) -> Optional[Refusal]:
        return self._last.get(model)

    def clear(self, model: str) -> None:
        self._last.pop(model, None)

    def evidence(self, model: str, view: StateView, now_ms: int) -> Optional[str]:
        """Why the next scale-up may go (None = keep waiting). No refusal = ``"none"``."""
        rec = self._last.get(model)
        if rec is None:
            return "none"
        if rec.blocking is not None and view.awake_bindings is not None and rec.blocking not in view.awake_bindings:
            return "blocker_asleep"
        free_now, free_then = view.free(model), rec.view.free(model)
        if free_now is not None:
            refused_node = rec.node if rec.scope == "node" else None
            new = {g for g in free_now - (free_then or frozenset()) - rec.gpus
                   if refused_node is None or not g.startswith(f"{refused_node}/")}
            if new:
                return "gpu_freed"
        if view.key is not None and rec.view.key is not None and view.key != rec.view.key:
            return "state_changed"
        if not rec.visible:
            wait_ms = int(round(max(0.0, rec.retry_after_s or 0.0) * 1000.0))
            if int(now_ms) >= rec.at_ms + wait_ms:
                return "retry_after"
        return None


@dataclass(frozen=True)
class Completed:
    model: str
    direction: str
    target: int
    #: The plan item of this decision.
    body: Mapping[str, Any]
    #: This model's answer: its plan item, or the whole-plan answer (no items).
    result: SMResult
    submitted_seq: int
    #: The awake count it started from - the item's ``awake_before`` (the SM's count
    #: under the lock), else the count the decision saw (scale-up; None for a
    #: scale-down without an item).
    start: Optional[int] = None
    #: The awake count after the plan (the item's ``awake_after``; sleep items carry it
    #: too since 2026-10-28); None when the plan gave no item (dropped, whole-plan
    #: refusal, transport error).
    reached: Optional[int] = None
    plan_id: Optional[str] = None
    #: Items in the plan this decision went out with.
    plan_size: int = 1

    @property
    def progressed(self) -> bool:
        """At least one replica was added (scale-up) or put to sleep (scale-down): the
        SM state changed because of us."""
        if self.reached is None or self.start is None:
            return False
        return self.reached > self.start if self.direction == "up" else self.reached < self.start

    @property
    def partial_fill(self) -> bool:
        """A scale-up that got some replicas, then the SM could place no more."""
        return (not self.result.ok) and self.direction == "up" and self.progressed

    def as_dict(self) -> dict[str, Any]:
        out = {"direction": self.direction, "target": self.target, **self.result.as_dict(),
               "plan_id": self.plan_id, "plan_size": self.plan_size}
        out.update(start=self.start, reached=self.reached)
        if self.direction == "up":
            out["partial_fill"] = self.partial_fill
        return out


@dataclass(frozen=True)
class _Job:
    seq: int
    model: str
    direction: str
    target: int
    #: Awake count the decision saw (scale-up), for ``Completed.start``.
    start: Optional[int]


class Dispatcher:
    """One plan in flight at a time, latest-wins coalescing (see the module docstring).

    :meth:`dispatch` takes one tick's decisions and never blocks. Finished decisions are
    collected with :meth:`drain_results` (one :class:`Completed` per model of a plan).
    """

    def __init__(self, post_plan: Callable[[Mapping[str, Any]], SMResult], *,
                 abort_sleep_path: str = DEFAULT_ABORT_SLEEP_PATH,
                 guard: Optional[Callable[[], Optional[str]]] = None) -> None:
        self._post = post_plan
        #: Called right before every POST: None = go, a string = drop the plan (why).
        self.guard = guard
        self._abort_sleep_path = abort_sleep_path
        self._cond = threading.Condition()
        #: The plan in flight (model -> job), empty when none.
        self._sending: dict[str, _Job] = {}
        #: The latest tick's decisions waiting for the next plan.
        self._pending: dict[str, _Job] = {}
        self._results: "queue.Queue[Completed]" = queue.Queue()
        self._seq = 0
        #: Decisions handed over (seq, model, direction, target), for tests and debugging; a
        #: queued decision a later tick replaced is listed but never sent.
        self.submitted: list[tuple[int, str, str, int]] = []
        self._thread: Optional[threading.Thread] = None
        self._closed = False

    def inflight(self, model: str) -> bool:
        """The model has a decision in the plan in flight or waiting for the next plan."""
        with self._cond:
            return model in self._sending or model in self._pending

    def inflight_count(self) -> int:
        """Models with a decision not answered yet (in flight or waiting)."""
        with self._cond:
            return len(set(self._sending) | set(self._pending))

    def dispatch(self, decisions: Mapping[str, tuple[str, int, Optional[int]]]) -> dict[str, str]:
        """Hand over one tick's decisions ``{model: (direction, target, start)}``.

        Per model: ``sent`` (no plan in flight: it goes out now, with the tick's other
        decisions, in one plan), ``queued`` (a plan is in flight: it goes in the next one,
        unless a later tick replaces it) or ``inflight`` (its item is in the plan in flight:
        not taken; the next decision sees the answer). The tick's decisions replace every
        decision still waiting (latest wins): a model waiting from an older tick and absent
        here - its desired count is back at awake, or a gate holds it now - is not sent."""
        for model, (direction, target, _start) in decisions.items():
            plan_item(model, direction, target, abort_sleep_path=self._abort_sleep_path)  # validates
        out: dict[str, str] = {}
        with self._cond:
            if self._closed:
                raise RuntimeError("dispatcher is closed")
            busy = bool(self._sending)
            pending: dict[str, _Job] = {}
            for model in sorted(decisions):
                direction, target, start = decisions[model]
                if model in self._sending:
                    out[model] = "inflight"
                    continue
                self._seq += 1
                pending[model] = _Job(self._seq, model, direction, int(target),
                                      None if start is None else int(start))
                self.submitted.append((self._seq, model, direction, int(target)))
                out[model] = "queued" if busy else "sent"
            self._pending = pending
            if pending:
                if self._thread is None:
                    self._thread = threading.Thread(target=self._run, name="bl-sm-plans", daemon=True)
                    self._thread.start()
                self._cond.notify_all()
        return out

    def _check_guard(self, plan_id: str) -> Optional[str]:
        try:
            why = self.guard() if self.guard is not None else None
        except Exception as exc:  # cannot tell whether we may act: do not act
            why = f"guard_failed: {exc!r}"[:200]
        if why:
            LOG.warning("SM plan %s dropped: %s", plan_id, why)
        return why or None

    def _call(self, body: dict) -> SMResult:
        try:
            return self._post(body)
        except Exception as exc:  # the client should not raise; never lose the plan's answers
            LOG.exception("SM plan %s raised", body.get("plan_id"))
            return SMResult(ok=False, error="exception", detail=repr(exc)[:2000])

    def _run(self) -> None:
        while True:
            with self._cond:
                while not self._pending and not self._closed:
                    self._cond.wait()
                if self._closed:
                    return
                jobs, self._pending = self._pending, {}
                self._sending = jobs
            try:
                done = self._execute(jobs)
            except Exception as exc:  # never lose the answers
                LOG.exception("SM plan failed")
                done = [Completed(job.model, job.direction, job.target,
                                  plan_item(job.model, job.direction, job.target,
                                            abort_sleep_path=self._abort_sleep_path),
                                  SMResult(ok=False, error="exception", detail=repr(exc)[:2000]), job.seq,
                                  start=job.start, plan_size=len(jobs))
                        for job in jobs.values()]
            for item in done:
                self._results.put(item)
            with self._cond:
                self._sending = {}
                self._cond.notify_all()

    def _execute(self, jobs: Mapping[str, _Job]) -> list[Completed]:
        plan_id = f"bl-{uuid.uuid4().hex[:12]}"
        models = sorted(jobs)
        items = {m: plan_item(m, jobs[m].direction, jobs[m].target, abort_sleep_path=self._abort_sleep_path)
                 for m in models}
        body = {
            "plan_id": plan_id,
            "sleeps": [items[m] for m in models if jobs[m].direction == "down"],
            "wakes": [items[m] for m in models if jobs[m].direction == "up"],
        }
        # Owner lock and controller mode are checked again right before the POST.
        why = self._check_guard(plan_id)
        whole = SMResult(ok=False, error=DROPPED, reason=str(why)) if why else self._call(body)
        answers = plan_item_results(whole, models)
        out: list[Completed] = []
        for model in models:
            job, result = jobs[model], answers[model]
            # Wake and sleep items both carry awake_before / awake_after.
            raw = result.raw if isinstance(result.raw, dict) else {}
            before, after = raw.get("awake_before"), raw.get("awake_after")
            start = before if isinstance(before, int) else job.start
            reached = after if isinstance(after, int) else None
            done = Completed(model, job.direction, job.target, items[model], result, job.seq,
                             start=start, reached=reached, plan_id=plan_id, plan_size=len(models))
            if done.partial_fill:
                LOG.info("SM partial fill %s: %s of %s->%s, then %s", model, reached, start, job.target,
                         result.as_dict())
            out.append(done)
        return out

    def drain_results(self) -> list[Completed]:
        out: list[Completed] = []
        while True:
            try:
                out.append(self._results.get_nowait())
            except queue.Empty:
                return out

    def close(self, join_s: float = 0.0) -> None:
        """Stop the worker after the plan in flight; decisions still waiting are not sent."""
        with self._cond:
            self._closed = True
            self._pending = {}
            thread = self._thread
            self._cond.notify_all()
        if join_s > 0 and thread is not None:
            thread.join(join_s)
