from __future__ import annotations

import json
import os
from typing import Any
from urllib.request import urlopen

from fastapi import FastAPI

from tre_common.registry import load_registry
from tre_ui.app import create_ui_app


class ServiceManagerStateClient:
    def __init__(self, base_url: str) -> None:
        self._base_url = base_url.rstrip("/")

    def get_state(self) -> dict[str, Any]:
        with urlopen(f"{self._base_url}/v2/state", timeout=5.0) as response:
            payload = json.loads(response.read().decode("utf-8"))
        if not isinstance(payload, dict):
            raise RuntimeError("service-manager state response must be a JSON object")
        return payload


def create_app() -> FastAPI:
    try:
        import redis  # type: ignore[import-not-found]
    except ModuleNotFoundError as exc:
        raise RuntimeError("redis package is required for the UI server") from exc

    registry = load_registry(os.environ.get("TRE_REGISTRY_PATH"))
    redis_url = os.environ.get("TRE_REDIS_URL", "redis://aibrix-redis-master:6379/0")
    sm_url = os.environ.get("TRE_SERVICE_MANAGER_URL", "http://aibrix-tre-service-manager:8000")
    return create_ui_app(registry, redis.Redis.from_url(redis_url), ServiceManagerStateClient(sm_url))
