from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Protocol

from fastapi import FastAPI
from fastapi.responses import HTMLResponse

from tre_common.registry import Registry


class RedisHashClient(Protocol):
    def hgetall(self, key: str) -> dict[Any, Any]: ...


class ServiceManagerStateClient(Protocol):
    def get_state(self) -> dict[str, Any]: ...


def create_ui_app(
    registry: Registry,
    redis_client: RedisHashClient,
    service_manager_client: ServiceManagerStateClient,
) -> FastAPI:
    app = FastAPI(title="TRE UI")

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        return (Path(__file__).parent / "static" / "index.html").read_text(encoding="utf-8")

    @app.get("/healthz")
    def healthz() -> dict[str, bool]:
        return {"ok": True}

    @app.get("/api/cluster")
    def cluster() -> dict[str, Any]:
        return {
            "topology": _topology_payload(registry),
            "service_manager": service_manager_client.get_state(),
        }

    @app.get("/api/models")
    def models() -> dict[str, Any]:
        return {"models": [_model_payload(model) for model in registry.models()]}

    @app.get("/api/decision/latest")
    def latest_decision() -> dict[str, Any]:
        return _decision_payload(redis_client.hgetall("tre:v2:decision:latest"))

    @app.get("/api/experiments")
    def experiments() -> dict[str, Any]:
        return {"available": False, "reason": "orchestrate integration is stubbed in P8"}

    return app


def _topology_payload(registry: Registry) -> dict[str, Any]:
    return {
        "nodes": [
            {
                "name": node.name,
                "gpus": node.gpus,
                "two_gpu_slots": [list(slot) for slot in node.two_gpu_slots],
            }
            for node in registry.topology().nodes
        ]
    }


def _model_payload(model: Any) -> dict[str, Any]:
    return {
        "name": model.name,
        "weights_path": model.weights_path,
        "tp_size": model.tp_size,
        "min_replicas": model.min_replicas,
        "max_replicas": model.max_replicas,
        "slo": {
            "ttft_p95_ms": model.slo.ttft_p95_ms,
            "tpot_p95_ms": model.slo.tpot_p95_ms,
            "e2e_p95_ms": model.slo.e2e_p95_ms,
        },
        "trs": {
            "w_p": model.trs.w_p,
            "w_d": model.trs.w_d,
            "lambda_wait": model.trs.lambda_wait,
            "qmin": model.trs.qmin,
            "ema_alpha": model.trs.ema_alpha,
            "theta_m": model.trs.theta_m,
            "tau_crit": model.trs.tau_crit,
            "tau_low": model.trs.tau_low,
            "tau_high": model.trs.tau_high,
            "qsat": model.trs.qsat,
            "epsat": model.trs.epsat,
            "hsat": model.trs.hsat,
        },
    }


def _decision_payload(raw: dict[Any, Any]) -> dict[str, Any]:
    normalized = {_to_text(key): _to_text(value) for key, value in raw.items()}
    payload_text = normalized.get("payload")
    payload: Any = None if payload_text is None else json.loads(payload_text)
    return {
        "ts_ms": int(normalized["ts_ms"]) if "ts_ms" in normalized else None,
        "payload": payload,
    }


def _to_text(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)
