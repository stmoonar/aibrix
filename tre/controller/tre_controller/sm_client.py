from __future__ import annotations

import asyncio
import json
import socket
from typing import Any, Protocol
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

#: HTTP statuses of the SM that mean "try again later" (review 2 P2-5): 409 = writer
#: lock busy / sleep reservation / drain rolled back, 503 = shutting down.
RETRIABLE_STATUSES = frozenset({409, 503})


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
    def retriable(self) -> bool:
        return self.timeout or self.transport or self.status in RETRIABLE_STATUSES

    def result(self) -> dict:
        result = {
            "ok": False,
            "error": str(self),
            "status": self.status,
            "retriable": self.retriable,
        }
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


def _sleep_fields(sleep_path: str | None, drain_budget_s: float | None) -> dict:
    fields: dict = {}
    if sleep_path is not None:
        fields["sleep_path"] = sleep_path
    if drain_budget_s is not None:
        fields["drain_budget_s"] = float(drain_budget_s)
    return fields


def _request_json(method: str, url: str, payload: dict | None, timeout_s: float) -> dict[str, Any]:
    body = None
    headers = {"Accept": "application/json"}
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
