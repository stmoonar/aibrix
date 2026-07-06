from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Protocol

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from tre_common.registry import Registry

_AUDIT = logging.getLogger("tre_ui.audit")


class RedisClient(Protocol):
    def hgetall(self, key: str) -> dict[Any, Any]: ...

    def zrangebyscore(self, key: str, minimum: Any, maximum: Any) -> list[Any]: ...

    def scan_iter(self, match: str) -> Any: ...

    def get(self, key: str) -> Any: ...


class ServiceManagerClient(Protocol):
    def get_state(self) -> dict[str, Any]: ...

    def request(self, method: str, path: str, payload: dict[str, Any] | None = None) -> dict[str, Any]: ...


class _TargetBody(BaseModel):
    wake_replicas: int


class _RoutableBody(BaseModel):
    hidden_pods: list[str] = []


class _DefragBody(BaseModel):
    tp_size: int = 2


def create_ui_app(
    registry: Registry,
    redis_client: RedisClient,
    service_manager_client: ServiceManagerClient,
) -> FastAPI:
    app = FastAPI(title="TRE Console")
    model_names = [m.name for m in registry.models()]
    model_max = {m.name: m.max_replicas for m in registry.models()}

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        return (Path(__file__).parent / "static" / "index.html").read_text(encoding="utf-8")

    @app.get("/healthz")
    def healthz() -> dict[str, bool]:
        return {"ok": True}

    # ---- observe ----

    @app.get("/api/cluster")
    def cluster() -> dict[str, Any]:
        return {"topology": _topology_payload(registry), "service_manager": _safe(service_manager_client.get_state)}

    @app.get("/api/models")
    def models() -> dict[str, Any]:
        return {"models": [_model_payload(model) for model in registry.models()]}

    @app.get("/api/decision/latest")
    def latest_decision() -> dict[str, Any]:
        return _decode_decision(redis_client.hgetall("tre:v2:decision:latest"))

    @app.get("/api/signal/history")
    def signal_history(model: str, since_ms: int = 0, until_ms: int | None = None, limit: int = 1000) -> dict[str, Any]:
        if model not in model_names:
            raise HTTPException(status_code=404, detail="unknown model")
        maximum: Any = "+inf" if until_ms is None else until_ms
        try:
            raw = redis_client.zrangebyscore(f"tre:v2:decision:hist:{model}", since_ms, maximum)
        except Exception:  # noqa: BLE001
            raw = []
        points = [p for p in (_decode_json(item) for item in raw) if p is not None]
        return {"model": model, "points": points[-limit:]}

    @app.get("/api/gputruth")
    def gpu_truth() -> dict[str, Any]:
        nodes: list[dict[str, Any]] = []
        try:
            for key in redis_client.scan_iter("tre:gpu_truth:*"):
                doc = _decode_json(redis_client.get(_to_text(key)))
                if doc is not None:
                    nodes.append(doc)
        except Exception:  # noqa: BLE001
            pass
        nodes.sort(key=lambda n: str(n.get("node", "")))
        return {"nodes": nodes}

    # ---- operate (audited; the frontend confirms before calling) ----

    @app.post("/api/ops/models/{model}/target")
    def op_target(model: str, body: _TargetBody) -> dict[str, Any]:
        _require_model(model, model_names)
        target = max(0, min(int(body.wake_replicas), model_max.get(model, int(body.wake_replicas))))
        _AUDIT.info(json.dumps({"op": "target", "model": model, "wake_replicas": target}))
        return _proxy(service_manager_client, "PUT", f"/v2/models/{model}/target", {"wake_replicas": target})

    @app.post("/api/ops/models/{model}/routable")
    def op_routable(model: str, body: _RoutableBody) -> dict[str, Any]:
        _require_model(model, model_names)
        _AUDIT.info(json.dumps({"op": "routable", "model": model, "hidden_pods": body.hidden_pods}))
        return _proxy(service_manager_client, "PUT", f"/v2/models/{model}/routable", {"hidden_pods": body.hidden_pods})

    @app.post("/api/ops/reconcile")
    def op_reconcile() -> dict[str, Any]:
        _AUDIT.info(json.dumps({"op": "reconcile"}))
        return _proxy(service_manager_client, "POST", "/v2/reconcile", None)

    @app.post("/api/ops/defrag")
    def op_defrag(body: _DefragBody) -> dict[str, Any]:
        _AUDIT.info(json.dumps({"op": "defrag", "tp_size": body.tp_size}))
        return _proxy(service_manager_client, "POST", "/v2/defrag", {"tp_size": body.tp_size})

    return app


def _require_model(model: str, model_names: list[str]) -> None:
    if model not in model_names:
        raise HTTPException(status_code=404, detail="unknown model")


def _proxy(client: ServiceManagerClient, method: str, path: str, payload: dict[str, Any] | None) -> dict[str, Any]:
    try:
        return {"ok": True, "response": client.request(method, path, payload)}
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=502, detail=f"service-manager {method} {path} failed: {exc}") from exc


def _safe(fn: Any) -> Any:
    try:
        return fn()
    except Exception as exc:  # noqa: BLE001
        return {"error": str(exc)}


def _topology_payload(registry: Registry) -> dict[str, Any]:
    return {
        "nodes": [
            {"name": node.name, "gpus": node.gpus, "two_gpu_slots": [list(slot) for slot in node.two_gpu_slots]}
            for node in registry.topology().nodes
        ]
    }


def _model_payload(model: Any) -> dict[str, Any]:
    return {
        "name": model.name,
        "tp_size": model.tp_size,
        "min_replicas": model.min_replicas,
        "max_replicas": model.max_replicas,
        "slo": {
            "ttft_p95_ms": model.slo.ttft_p95_ms,
            "tpot_p95_ms": model.slo.tpot_p95_ms,
            "e2e_p95_ms": model.slo.e2e_p95_ms,
        },
        "trs": {
            "theta_m": model.trs.theta_m,
            "tau_crit": model.trs.tau_crit,
            "tau_low": model.trs.tau_low,
            "tau_high": model.trs.tau_high,
            "qsat": model.trs.qsat,
            "ema_tau_ms": getattr(model.trs, "ema_tau_ms", None),
        },
    }


def _decode_decision(raw: dict[Any, Any]) -> dict[str, Any]:
    hashmap = {_to_text(k): _to_text(v) for k, v in raw.items()}
    if not hashmap:
        return {"ts_ms": None, "loop": None, "model_states": {}, "actions": [], "events": []}
    ts_raw = hashmap.get("ts_ms", "")
    return {
        "ts_ms": int(ts_raw) if ts_raw.lstrip("-").isdigit() else None,
        "loop": hashmap.get("loop"),
        "stale": hashmap.get("stale") == "true",
        "submitted": int(hashmap["submitted"]) if hashmap.get("submitted", "").isdigit() else 0,
        "model_states": _decode_json(hashmap.get("model_states")) or {},
        "actions": _decode_json(hashmap.get("actions")) or [],
        "events": _decode_json(hashmap.get("events")) or [],
    }


def _decode_json(value: Any) -> Any:
    if value is None:
        return None
    try:
        return json.loads(_to_text(value))
    except (json.JSONDecodeError, TypeError, ValueError):
        return None


def _to_text(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)
