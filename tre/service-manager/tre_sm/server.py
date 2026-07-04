from __future__ import annotations

import os

from fastapi import FastAPI

from tre_common.registry import load_registry
from tre_sm.app import create_service_app
from tre_sm.state.store import StateStore


def create_app() -> FastAPI:
    try:
        import redis  # type: ignore[import-not-found]
    except ModuleNotFoundError as exc:
        raise RuntimeError("redis package is required for the service-manager server") from exc

    registry = load_registry(os.environ.get("TRE_REGISTRY_PATH"))
    redis_url = os.environ.get("TRE_REDIS_URL", "redis://aibrix-redis-master:6379/0")
    return create_service_app(registry, StateStore(redis.Redis.from_url(redis_url)))
