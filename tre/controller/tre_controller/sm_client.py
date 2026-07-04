from __future__ import annotations

import asyncio
import json
from typing import Any, Protocol
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


class ServiceManagerError(Exception):
    pass


class AsyncTransport(Protocol):
    async def request(self, method: str, url: str, *, json: dict | None = None, timeout_s: float) -> dict: ...


class UrllibTransport:
    async def request(self, method: str, url: str, *, json: dict | None = None, timeout_s: float) -> dict:
        return await asyncio.to_thread(_request_json, method, url, json, timeout_s)


class ServiceManagerClient:
    def __init__(self, base_url: str, *, transport: AsyncTransport | None = None, timeout_s: float = 5.0) -> None:
        self._base_url = base_url.rstrip("/")
        self._transport = transport or UrllibTransport()
        self._timeout_s = timeout_s

    async def get_state(self) -> dict:
        return await self._request("GET", "/v2/state")

    async def get_state_result(self) -> dict:
        try:
            return {"ok": True, "response": await self.get_state()}
        except ServiceManagerError as exc:
            return {"ok": False, "error": str(exc)}

    async def scale_model(self, model: str, delta: int) -> dict:
        try:
            state = await self.get_state()
            current = int(state.get("models", {}).get(model, {}).get("awake", 0))
            target = max(0, current + int(delta))
            response = await self._request("PUT", f"/v2/models/{model}/target", json={"wake_replicas": target})
            return {"ok": True, "response": response}
        except ServiceManagerError as exc:
            return {"ok": False, "error": str(exc)}

    async def set_routable(self, model: str, hidden_pods: tuple[str, ...]) -> dict:
        try:
            response = await self._request(
                "PUT",
                f"/v2/models/{model}/routable",
                json={"hidden_pods": list(hidden_pods)},
            )
            return {"ok": True, "response": response}
        except ServiceManagerError as exc:
            return {"ok": False, "error": str(exc)}

    async def defrag(self, migrations: tuple) -> dict:
        del migrations
        return {"ok": False, "error": "defrag endpoint is not implemented in service-manager v2"}

    async def _request(self, method: str, path: str, *, json: dict | None = None) -> dict:
        url = f"{self._base_url}{path}"
        try:
            response = await self._transport.request(method, url, json=json, timeout_s=self._timeout_s)
        except ServiceManagerError:
            raise
        except Exception as exc:
            raise ServiceManagerError(str(exc)) from exc
        if not isinstance(response, dict):
            raise ServiceManagerError("service-manager response must be a JSON object")
        return response


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
        raise ServiceManagerError(f"HTTP {exc.code}: {detail}") from exc
    except URLError as exc:
        raise ServiceManagerError(str(exc.reason)) from exc
    except TimeoutError as exc:
        raise ServiceManagerError("request timed out") from exc

    if not data:
        return {}
    try:
        decoded = json.loads(data.decode("utf-8"))
    except json.JSONDecodeError as exc:
        raise ServiceManagerError("service-manager returned invalid JSON") from exc
    if not isinstance(decoded, dict):
        raise ServiceManagerError("service-manager response must be a JSON object")
    return decoded
