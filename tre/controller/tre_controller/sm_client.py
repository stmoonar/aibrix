from __future__ import annotations

import asyncio
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
import json
import socket
from typing import Any, Iterator, Protocol
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

#: HTTP statuses of the SM that mean "try again later" (review 2 P2-5): 409 = writer
#: lock busy / sleep reservation / drain rolled back, 503 = shutting down.
RETRIABLE_STATUSES = frozenset({409, 503})

#: ``error`` codes of a structured 409 wake refusal / failure (S3, 2026-09-30): the
#: body locates it (node, gpu_ids, scope). Since 2026-10-02 the controller only
#: records it as an observation event (``wake_refused``); the next tick re-plans from
#: a new view (no GPU cooldown). ``wake_conflict`` is accepted for a service-manager
#: of the same series that sent the generic code.
WAKE_ERROR_CODES = frozenset({
    "gpu_busy", "resident_loading", "lease_conflict", "resident_awake",
    "truth_unavailable", "wake_failed", "wake_conflict",
})

#: 409 ``error`` codes the service-manager answers BEFORE it changes anything
#: (2026-10-02): the routable view unreadable (fail closed), the replica floor, the
#: writer lock busy, and the located wake refusals of the prepare step. ``wake_failed``
#: (the wake itself failed and was settled) and ``partial`` (a transfer: see its
#: pairs) are not in it.
NOT_EXECUTED_CODES = frozenset({
    "routable_unknown", "floor_violation", "writer_busy",
    "gpu_busy", "resident_loading", "lease_conflict", "resident_awake", "truth_unavailable", "wake_conflict",
})

#: Caller identity of the SM calls made in this context (``X-TRE-Actor``,
#: recorded as ``request.actor`` of the SM operation): controller/<loop>/<reason>.
_ACTOR: ContextVar[str | None] = ContextVar("tre_controller_sm_actor", default=None)
DEFAULT_ACTOR = "tre-controller"


@contextmanager
def sm_actor(actor: str | None) -> Iterator[None]:
    """SM calls made inside the block send ``X-TRE-Actor: <actor>``."""
    token = _ACTOR.set(actor)
    try:
        yield
    finally:
        _ACTOR.reset(token)


class ServiceManagerError(Exception):
    """An SM call failed. ``retriable``: a conflict / busy (409), shutting down
    (503), a timeout or a connection error - the same call may succeed later.
    Anything else (400 invalid request, 404, 5xx, a malformed answer) is permanent."""

    def __init__(
        self,
        message: str,
        *,
        status: int | None = None,
        timeout: bool = False,
        transport: bool = False,
        body: dict | None = None,
    ) -> None:
        super().__init__(message)
        self.status = status
        self.timeout = timeout
        self.transport = transport
        #: The SM's JSON error body, when it sent one (e.g. sleep ``outcomes``).
        self.body = body if isinstance(body, dict) else None

    @property
    def floor_violation(self) -> bool:
        """The SM refused because the call would take a model below its replica
        floor (409 ``error: floor_violation``, 2026-09-29). Not retried: the planner
        re-plans on its next tick from a fresh view."""
        return (self.body or {}).get("error") == "floor_violation"

    @property
    def retriable(self) -> bool:
        if self.floor_violation:
            return False
        if (self.body or {}).get("error") == "wake_failed":
            # The wake itself failed on that GPU (the SM settled it, possibly with a
            # compensating sleep): not re-sent at once - the planner re-plans from a
            # new view (review P3-11).
            return False
        return self.timeout or self.transport or self.status in RETRIABLE_STATUSES

    @property
    def wake_conflict(self) -> dict | None:
        """The located wake refusal of a structured 409 (S3), or None - also for
        an older service-manager whose 409 is plain text (still retriable)."""
        body = self.body or {}
        if self.status != 409 or body.get("error") not in WAKE_ERROR_CODES:
            return None
        node = body.get("node")
        raw = body.get("gpu_ids", body.get("gpu"))
        if isinstance(raw, (int, str)) and not isinstance(raw, bool):
            raw = [raw]
        try:
            gpus = [int(gpu) for gpu in raw or ()]
        except (TypeError, ValueError):
            gpus = []
        if not node:
            return None
        retry = body.get("retry_after_s")
        return {
            "error": str(body.get("error")),
            "reason": body.get("reason"),
            "node": str(node),
            "gpu_ids": gpus,
            "scope": "node" if body.get("scope") == "node" else "gpu",
            "retry_after_s": float(retry) if isinstance(retry, (int, float)) and not isinstance(retry, bool) else None,
            "binding_id": body.get("binding_id"),
            "blocking_binding_id": body.get("blocking_binding_id"),
        }

    @property
    def not_executed(self) -> bool:
        """The SM refused the call BEFORE changing anything (2026-10-02): a 400 / 404, a
        503 while shutting down, or a 409 whose code says it was refused up front
        (``routable_unknown``, ``floor_violation``, ``writer_busy``, a located wake
        refusal other than ``wake_failed``) or a plain ``RetryLater`` 409 (``...; retry``;
        ``/target`` and ``/v2/transfers`` no longer send it - concurrent requests queue on
        the SM writer lock). The controller then records no change (no view-pending / O1
        stamp, no last action) and re-plans on the next tick."""
        if self.status in (400, 404, 503):
            return True
        if self.status != 409:
            return False
        body = self.body or {}
        code = body.get("error")
        if code is None:
            return str(body.get("detail") or "").rstrip().endswith("retry")
        return code in NOT_EXECUTED_CODES

    def result(self) -> dict:
        result = {
            "ok": False,
            "error": str(self),
            "status": self.status,
            "retriable": self.retriable,
        }
        code = (self.body or {}).get("error")
        if isinstance(code, str) and code:
            result["code"] = code  # the SM's structured error code (writer_busy, ...)
        if self.not_executed:
            result["not_executed"] = True
        if self.floor_violation:
            result["floor_violation"] = (self.body or {}).get("floor")
        conflict = self.wake_conflict
        if conflict is not None:
            result["wake_conflict"] = conflict
        outcomes = (self.body or {}).get("outcomes")
        if isinstance(outcomes, list):
            # Per-pod sleep outcomes of a failed sleep (review 4 P2-2): which pods
            # slept, rolled back, or stay hidden unconfirmed.
            result["outcomes"] = outcomes
        return result


class AsyncTransport(Protocol):
    async def request(self, method: str, url: str, *, json: dict | None = None, timeout_s: float) -> dict: ...


class UrllibTransport:
    async def request(self, method: str, url: str, *, json: dict | None = None, timeout_s: float) -> dict:
        return await asyncio.to_thread(_request_json, method, url, json, timeout_s)


class ServiceManagerClient:
    def __init__(
        self,
        base_url: str,
        *,
        transport: AsyncTransport | None = None,
        timeout_s: float = 5.0,
        slow_timeout_s: float = 300.0,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._transport = transport or UrllibTransport()
        self._timeout_s = timeout_s
        # B1: wake/create (target) and defrag run synchronously inside the SM handler for
        # minutes; the default 5s timeout would fire mid-migration and free the inflight,
        # letting the controller replan against a stale view. Give them a long timeout.
        self._slow_timeout_s = slow_timeout_s

    async def get_state(self) -> dict:
        return await self._request("GET", "/v2/state")

    async def get_state_result(self) -> dict:
        try:
            return {"ok": True, "response": await self.get_state()}
        except ServiceManagerError as exc:
            return exc.result()

    async def scale_model(
        self,
        model: str,
        delta: int,
        *,
        sleep_path: str | None = None,
        drain_budget_s: float | None = None,
    ) -> dict:
        """RELATIVE scale (current awake + ``delta``, read at dispatch). Not
        idempotent: callers never retry it (review 3 P2-1) - a retried upscale
        uses :meth:`scale_model_to` with an absolute target."""
        try:
            counts = await self._model_counts(model)
            current = counts["awake"]
            serving_floor = 1 if counts["bound"] > 0 and current > 0 and int(delta) < 0 else 0
            target = max(serving_floor, current + int(delta))
            payload: dict = {"wake_replicas": target}
            payload.update(_sleep_fields(sleep_path, drain_budget_s))
            response = await self._request(
                "PUT", f"/v2/models/{model}/target", json=payload, timeout_s=self._slow_timeout_s
            )
            return {"ok": True, "response": response}
        except ServiceManagerError as exc:
            return exc.result()

    async def scale_model_to(self, model: str, target: int) -> dict:
        """Grow ``model`` to at least ``target`` awake replicas (absolute and
        grow-only, review 3 P2-1): re-sending it after a success that timed out
        on the client is a no-op, and it never shrinks a model that meanwhile grew
        past the target. The SM applies ``at_least`` under its writer lock; the
        pre-check also keeps a service-manager without ``at_least`` from shrinking."""
        try:
            target = int(target)
            counts = await self._model_counts(model)
            if counts["awake"] >= target:
                return {"ok": True, "response": {"model": model, "wake_replicas": target, "actions": [], "noop": True}}
            response = await self._request(
                "PUT",
                f"/v2/models/{model}/target",
                json={"wake_replicas": target, "at_least": True},
                timeout_s=self._slow_timeout_s,
            )
            return {"ok": True, "response": response}
        except ServiceManagerError as exc:
            return exc.result()

    async def scale_model_hinted(self, model: str, delta: int, *, hints: tuple[str, ...]) -> dict:
        """Grow ``model`` by ``delta`` awake replicas, waking the sleeping bindings
        ``hints`` names when the service-manager can (S5): it picks the GPUs itself
        (registry placement policy) and substitutes a hint it cannot wake; the
        response's ``picked`` says where the replicas went. Relative (awake read at
        dispatch) and grow-only (``at_least``); never retried."""
        try:
            counts = await self._model_counts(model)
            target = counts["awake"] + int(delta)
            response = await self._request(
                "PUT",
                f"/v2/models/{model}/target",
                json={"wake_replicas": target, "at_least": True, "hints": list(hints)},
                timeout_s=self._slow_timeout_s,
            )
            return {"ok": True, "response": response}
        except ServiceManagerError as exc:
            return exc.result()

    async def transfer(
        self,
        donor_model: str,
        receiver_model: str,
        count: int,
        *,
        sleep_path: str = "urgent",
    ) -> dict:
        """``POST /v2/transfers`` (2026-10-02, design 20261002-controller-transfer): the
        service-manager hands ``count`` donor replicas' GPUs to the receiver model -
        it picks the pairs (same GPUs, TP coverage), sleeps the donors and wakes the
        receivers in one request under its global writer lock: when the call returns,
        the relay has completed or failed (no follow-up tracking). Never retried (not
        idempotent; the planner re-plans). A writer-lock wait that timed out is a 409
        ``writer_busy`` (``not_executed``).

        Returns ``{"ok": True, "response": body}`` on 200 (which may still be a
        PARTIAL transfer - account by ``done`` / ``taken`` / ``unfilled``, never by
        ``count``). A 409 that carries the transfer response (``error: partial``, no
        pair completed) returns ``ok: False`` with that body as ``response`` and
        ``partial: True``. A 404 (service-manager without the endpoint) returns
        ``ok: False`` with ``unsupported: True``."""
        try:
            response = await self._request(
                "POST",
                "/v2/transfers",
                json={
                    "donor_model": donor_model,
                    "receiver_model": receiver_model,
                    "count": int(count),
                    "sleep_path": sleep_path,
                },
                timeout_s=self._slow_timeout_s,
            )
            return {"ok": True, "response": response}
        except ServiceManagerError as exc:
            result = exc.result()
            # A transfer is never re-sent as is (the planner re-plans from a new view).
            result["retriable"] = False
            body = exc.body or {}
            if exc.status == 404:
                result["unsupported"] = True
            elif exc.status == 409 and isinstance(body.get("pairs"), list):
                result["partial"] = True
                result["response"] = body
            return result

    async def model_awake(self, model: str) -> dict:
        """{"ok": True, "awake": n} from the SM state, or a failed result."""
        try:
            counts = await self._model_counts(model)
            return {"ok": True, "awake": counts["awake"]}
        except ServiceManagerError as exc:
            return exc.result()

    async def _model_counts(self, model: str) -> dict[str, int]:
        """Awake / bound counts of ``model`` from GET /v2/state. A malformed
        answer is a (permanent) ServiceManagerError, never a raw exception."""
        state = await self.get_state()
        models = state.get("models", {})
        counts = models.get(model, {}) if isinstance(models, dict) else None
        if not isinstance(counts, dict):
            raise ServiceManagerError(f"malformed /v2/state for {model}: models entry is not an object")
        try:
            return {"awake": int(counts.get("awake", 0)), "bound": int(counts.get("bound", 0))}
        except (TypeError, ValueError) as exc:
            raise ServiceManagerError(f"malformed /v2/state counts for {model}: {counts!r}") from exc

    async def set_binding_power(
        self,
        serve_id: str,
        *,
        awake: bool,
        sleep_path: str | None = None,
        drain_budget_s: float | None = None,
    ) -> dict:
        # Binding-level power (PUT /v2/bindings/{serve_id}/power): used to sleep exactly
        # one chosen binding (safescale commit of the hidden pod, slot-targeted donor).
        try:
            response = await self._request(
                "PUT",
                f"/v2/bindings/{serve_id}/power",
                json={"awake": bool(awake), **_sleep_fields(sleep_path, drain_budget_s)},
                timeout_s=self._slow_timeout_s,
            )
            return {"ok": True, "response": response}
        except ServiceManagerError as exc:
            return exc.result()

    async def set_routable(self, model: str, hidden_pods: tuple[str, ...]) -> dict:
        try:
            response = await self._request(
                "PUT",
                f"/v2/models/{model}/routable",
                json={"hidden_pods": list(hidden_pods)},
            )
            return {"ok": True, "response": response}
        except ServiceManagerError as exc:
            return exc.result()

    async def defrag(self, migrations: tuple) -> dict:
        del migrations
        try:
            response = await self._request(
                "POST", "/v2/defrag", json={"tp_size": 2}, timeout_s=self._slow_timeout_s
            )
            return {"ok": True, "response": response}
        except ServiceManagerError as exc:
            if exc.status == 404 or "HTTP 404" in str(exc):
                return {
                    "ok": False,
                    "error": "defrag endpoint is not implemented in service-manager v2",
                    "status": 404,
                    "retriable": False,
                }
            return exc.result()

    async def _request(self, method: str, path: str, *, json: dict | None = None, timeout_s: float | None = None) -> dict:
        url = f"{self._base_url}{path}"
        try:
            response = await self._transport.request(
                method, url, json=json, timeout_s=self._timeout_s if timeout_s is None else timeout_s
            )
        except ServiceManagerError:
            raise
        except (TimeoutError, asyncio.TimeoutError, socket.timeout) as exc:
            raise ServiceManagerError(f"request timed out: {exc}", timeout=True) from exc
        except Exception as exc:
            # A transport failure (connection refused / reset while the SM restarts).
            raise ServiceManagerError(str(exc), transport=True) from exc
        if not isinstance(response, dict):
            raise ServiceManagerError("service-manager response must be a JSON object")
        return response


@dataclass(frozen=True)
class ModelFloor:
    """One model's replica-floor view of ``/v2/state`` (2026-10-02): ``routable`` is the
    service-manager's own floor-check count, ``floor`` the enforced ``min_replicas`` (0
    while the floor is off), ``floor_headroom = routable - floor``. None = not readable."""

    routable: int | None = None
    floor: int | None = None
    floor_headroom: int | None = None


@dataclass(frozen=True)
class StateRoutable:
    """The routable / floor fields of one ``/v2/state`` answer.

    ``routable_ids`` (serve ids the SM counts routable) is None when the SM did not
    report a per-binding ``routable`` (an older SM: key missing) or could not compute
    it (``routable: null`` with ``routable_error``); ``error`` then says why and the
    controller falls back to its own count (awake and not hidden)."""

    routable_ids: frozenset[str] | None
    models: dict[str, ModelFloor] = field(default_factory=dict)
    floor_enforced: bool | None = None
    error: str | None = None


def parse_state_routable(state: dict) -> StateRoutable:
    """The SM routable / floor view of a ``/v2/state`` answer (never raises)."""
    if not isinstance(state, dict):
        return StateRoutable(routable_ids=None, error="malformed_state")
    enforced = state.get("floor_enforced")
    enforced = bool(enforced) if isinstance(enforced, bool) else None
    models: dict[str, ModelFloor] = {}
    raw_models = state.get("models")
    for model, entry in (raw_models.items() if isinstance(raw_models, dict) else ()):
        if isinstance(entry, dict):
            models[str(model)] = ModelFloor(
                routable=_opt_int(entry.get("routable")),
                floor=_opt_int(entry.get("floor")),
                floor_headroom=_opt_int(entry.get("floor_headroom")),
            )
    error: str | None = None
    ids: set[str] = set()
    bindings = state.get("bindings")
    for item in bindings if isinstance(bindings, list) else ():
        if not isinstance(item, dict):
            continue
        if "routable" not in item:
            error = "routable_missing"
            break
        value = item.get("routable")
        if not isinstance(value, bool):
            error = f"routable_unavailable: {state.get('routable_error') or 'null'}"
            break
        if value:
            ids.add(str(item.get("serve_id")))
    if error is None and state.get("routable_error"):
        error = f"routable_unavailable: {state.get('routable_error')}"
    return StateRoutable(
        routable_ids=None if error is not None else frozenset(ids),
        models=models,
        floor_enforced=enforced,
        error=error,
    )


def _opt_int(value: Any) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _sleep_fields(sleep_path: str | None, drain_budget_s: float | None) -> dict:
    fields: dict = {}
    if sleep_path is not None:
        fields["sleep_path"] = sleep_path
    if drain_budget_s is not None:
        fields["drain_budget_s"] = float(drain_budget_s)
    return fields


def _request_json(method: str, url: str, payload: dict | None, timeout_s: float) -> dict[str, Any]:
    body = None
    # Caller identity for the SM operation log (2026-09-28): the SM HTTP write
    # API is shared by the TRE controller, the APA arm and operators.
    headers = {"Accept": "application/json", "X-TRE-Actor": (_ACTOR.get() or DEFAULT_ACTOR)[:200]}
    if payload is not None:
        body = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
    request = Request(url, data=body, headers=headers, method=method)
    try:
        with urlopen(request, timeout=timeout_s) as response:
            data = response.read()
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        try:
            body = json.loads(detail)
        except ValueError:
            body = None
        raise ServiceManagerError(
            f"HTTP {exc.code}: {detail}", status=int(exc.code), body=body if isinstance(body, dict) else None
        ) from exc
    except URLError as exc:
        timed_out = isinstance(exc.reason, (TimeoutError, socket.timeout))
        raise ServiceManagerError(
            str(exc.reason), timeout=timed_out, transport=not timed_out
        ) from exc
    except (TimeoutError, socket.timeout) as exc:
        raise ServiceManagerError("request timed out", timeout=True) from exc

    if not data:
        return {}
    try:
        decoded = json.loads(data.decode("utf-8"))
    except json.JSONDecodeError as exc:
        raise ServiceManagerError("service-manager returned invalid JSON") from exc
    if not isinstance(decoded, dict):
        raise ServiceManagerError("service-manager response must be a JSON object")
    return decoded
