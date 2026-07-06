from __future__ import annotations

import json
import os
from typing import Any
from urllib.request import Request, urlopen

from fastapi import FastAPI

from tre_common.registry import load_registry
from tre_ui.app import create_ui_app


class ServiceManagerStateClient:
    def __init__(self, base_url: str) -> None:
        self._base_url = base_url.rstrip("/")

    def get_state(self) -> dict[str, Any]:
        return self.request("GET", "/v2/state")

    def request(self, method: str, path: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        body = None
        headers = {"Accept": "application/json"}
        if payload is not None:
            body = json.dumps(payload).encode("utf-8")
            headers["Content-Type"] = "application/json"
        # Operate calls (target/defrag) can run for minutes in the SM; observe calls are fast.
        timeout = 5.0 if method == "GET" else 300.0
        request = Request(f"{self._base_url}{path}", data=body, headers=headers, method=method)
        with urlopen(request, timeout=timeout) as response:
            raw = response.read().decode("utf-8")
        decoded = json.loads(raw) if raw else {}
        if not isinstance(decoded, dict):
            raise RuntimeError("service-manager response must be a JSON object")
        return decoded


def create_app() -> FastAPI:
    try:
        import redis  # type: ignore[import-not-found]
    except ModuleNotFoundError as exc:
        raise RuntimeError("redis package is required for the UI server") from exc

    registry = load_registry(os.environ.get("TRE_REGISTRY_PATH"))
    redis_url = os.environ.get("TRE_REDIS_URL", "redis://aibrix-redis-master:6379/0")
    sm_url = os.environ.get("TRE_SERVICE_MANAGER_URL", "http://aibrix-tre-service-manager:8000")
    # In-cluster kubernetes access for param editing (restart-to-apply). Absent outside a
    # pod (e.g. local dev) -> param endpoints return 503, everything else still works.
    k8s_client: Any | None = None
    try:
        from tre_ui.k8s_client import InClusterK8sClient

        if os.path.exists("/var/run/secrets/kubernetes.io/serviceaccount/token"):
            k8s_client = InClusterK8sClient()
    except Exception:  # noqa: BLE001
        k8s_client = None
    return create_ui_app(registry, redis.Redis.from_url(redis_url), ServiceManagerStateClient(sm_url), k8s_client)
