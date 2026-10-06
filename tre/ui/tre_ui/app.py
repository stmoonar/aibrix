from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from tre_common.registry import Registry
from tre_common.run_mode import RUN_MODES, run_mode_view, write_run_mode
from tre_ui import params as params_mod
from tre_ui.sampler import Sampler

_AUDIT = logging.getLogger("tre_ui.audit")
_STATIC = Path(__file__).parent / "static"
_REGISTRY_CM = "tre-v2-registry"
_CONTROLLER_DEPLOY = "tre-v2-controller"
#: Restart-to-apply restarts the service-manager too: it reads the registry
#: (placement, service_manager, gateway, models) only at start.
_SM_DEPLOY = "tre-v2-service-manager"
_HASH_ANNOTATION = "tre.dev/params-hash"
_PARAMS_AUDIT_KEY = "tre:v2:audit:params"
_PARAMS_APPLIED_KEY = "tre:v2:ui:params-applied-hash"


class RedisClient(Protocol):
    def hgetall(self, key: str) -> dict[Any, Any]: ...

    def zrangebyscore(self, key: str, minimum: Any, maximum: Any) -> list[Any]: ...

    def scan_iter(self, match: str) -> Any: ...

    def get(self, key: str) -> Any: ...

    def set(self, key: str, value: str) -> Any: ...

    def rpush(self, key: str, value: str) -> Any: ...


class ServiceManagerClient(Protocol):
    def get_state(self) -> dict[str, Any]: ...

    def request(self, method: str, path: str, payload: dict[str, Any] | None = None) -> dict[str, Any]: ...


class _TargetBody(BaseModel):
    wake_replicas: int


class _RoutableBody(BaseModel):
    hidden_pods: list[str] = []


class _DefragBody(BaseModel):
    tp_size: int = 2
    #: Forwarded to the service-manager: run even while registry
    #: placement.defrag.enabled is false (otherwise it refuses with 409).
    force: bool = False


class _ModeBody(BaseModel):
    mode: str  # "active" | "observe"


class _RunModeBody(BaseModel):
    """Convenience "set both": either may be omitted (= left unchanged)."""
    controller: str | None = None
    sm_actuation: str | None = None


class _ParamsBody(BaseModel):
    expected_resource_version: str | None = None
    models: dict[str, Any] = {}


class _RestartBody(BaseModel):
    reason: str = ""


def _thresholds_payload(registry: Registry) -> dict[str, Any]:
    """SM GPU thresholds for display. The cold-start limit is per GPU and model
    (B9): an absolute override, else total x (1 - util) - margin."""
    config = registry.service_manager() if callable(getattr(registry, "service_manager", None)) else None
    return {
        "create_max_used_mib": getattr(config, "create_max_used_mib", None),
        "create_margin_mib": getattr(config, "create_margin_mib", None),
        "sleep_leak_used_mib": 8192,
    }


def create_ui_app(
    registry: Registry,
    redis_client: RedisClient,
    service_manager_client: ServiceManagerClient,
    k8s_client: Any | None = None,
) -> FastAPI:
    model_names = [m.name for m in registry.models()]
    # Scaling cap (max_awake_replicas; falls back to the layout size max_replicas).
    model_max = {m.name: m.scale_max_replicas for m in registry.models()}
    sampler = Sampler(redis_client, service_manager_client, model_names=model_names)

    @asynccontextmanager
    async def _lifespan(_app: FastAPI):
        sampler.start()
        try:
            yield
        finally:
            sampler.stop()

    app = FastAPI(title="TRE Console", lifespan=_lifespan)
    app.state.sampler = sampler
    # One mount instead of a route per asset: the console is now several ES
    # modules, and hand-rolling a route for each would be pure friction.
    app.mount("/static", StaticFiles(directory=str(_STATIC)), name="static")

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        return (_STATIC / "index.html").read_text(encoding="utf-8")

    @app.get("/healthz")
    def healthz() -> dict[str, bool]:
        return {"ok": True}

    # ---- live: served from the in-pod sampler cache; browsers NEVER trigger an upstream read ----

    @app.get("/api/snapshot")
    def snapshot(request: Request) -> Response:
        snap = sampler.snapshot()
        etag = f'W/"{snap.get("version", 0)}"'
        if request.headers.get("if-none-match") == etag:
            return Response(status_code=304, headers={"ETag": etag})
        return Response(json.dumps(snap, separators=(",", ":")), media_type="application/json",
                        headers={"ETag": etag})

    @app.get("/api/stream")
    async def stream() -> StreamingResponse:
        async def gen():
            last = -1
            beats = 0
            while True:
                version = sampler.version()
                if version != last:
                    last = version
                    yield f"data: {json.dumps(sampler.snapshot(), separators=(',', ':'))}\n\n"
                    beats = 0
                else:
                    beats += 1
                    if beats % 30 == 0:  # ~15s keep-alive when idle
                        yield ": keep-alive\n\n"
                await asyncio.sleep(0.5)

        return StreamingResponse(gen(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no",
                                          "Connection": "keep-alive"})

    @app.get("/api/events")
    def events(limit: int = 200) -> dict[str, Any]:
        return {"events": sampler.events(limit)}

    @app.get("/api/meta")
    def meta() -> dict[str, Any]:
        return {
            "models": [_model_payload(model) for model in registry.models()],
            "topology": _topology_payload(registry),
            "thresholds": _thresholds_payload(registry),
            "sampler_version": sampler.version(),
        }

    audit_cache: dict[str, Any] = {"ran_at_ms": None, "result": None}

    @app.get("/api/ops/audit")
    def get_audit() -> dict[str, Any]:
        return dict(audit_cache)

    @app.post("/api/ops/audit")
    def run_audit() -> dict[str, Any]:
        """Run the fleet audit on demand.

        Deliberately manual and never on a timer: /v2/audit lists k8s pods and
        HTTP-probes every vLLM pod, so polling it would load the model pods and
        perturb scaling experiments.
        """
        try:
            result = service_manager_client.request("GET", "/v2/audit")
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=502, detail=str(exc)) from exc
        audit_cache["ran_at_ms"] = int(time.time() * 1000)
        audit_cache["result"] = result
        return dict(audit_cache)

    # ---- run mode: controller mode and SM actuation are INDEPENDENT switches ----
    # (user decision 2026-09-28; tre/docs/design/20260928-observe-mode-semantics.md)

    def _run_mode() -> dict[str, Any]:
        """``mode`` / ``controller`` = what the controller acts on, ``sm_actuation``
        = what the SM supervisor acts on (each: absent / unreadable = observe,
        fail-closed; the SM never derives its switch from the controller mode);
        ``raw`` = the stored values (None = absent); ``warnings`` names every
        missing key."""
        return run_mode_view(redis_client)

    def _write_run_mode(controller: str | None, sm_actuation: str | None) -> dict[str, Any]:
        for name, value in (("controller mode", controller), ("sm_actuation", sm_actuation)):
            if value is not None and value not in RUN_MODES:
                raise HTTPException(status_code=400, detail=f"{name} must be active or observe")
        if controller is None and sm_actuation is None:
            raise HTTPException(status_code=400, detail="set controller and/or sm_actuation")
        _AUDIT.info(json.dumps({"op": "run_mode", "controller": controller, "sm_actuation": sm_actuation}))
        try:
            written = write_run_mode(redis_client, controller_mode=controller, sm_actuation=sm_actuation)
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=502, detail=f"redis set failed: {exc}") from exc
        return {"ok": True, "written": written, **_run_mode()}

    @app.get("/api/ops/controller/mode")
    def get_mode() -> dict[str, Any]:
        return _run_mode()

    @app.post("/api/ops/controller/mode")
    def set_mode(body: _ModeBody) -> dict[str, Any]:
        """Sets ONLY the controller mode (the SM actuation switch is unchanged)."""
        return _write_run_mode(body.mode, None)

    @app.get("/api/ops/sm/actuation")
    def get_sm_actuation() -> dict[str, Any]:
        return _run_mode()

    @app.post("/api/ops/sm/actuation")
    def set_sm_actuation(body: _ModeBody) -> dict[str, Any]:
        """Sets ONLY the SM actuation switch (the controller mode is unchanged)."""
        return _write_run_mode(None, body.mode)

    @app.get("/api/ops/run-mode")
    def get_run_mode() -> dict[str, Any]:
        return _run_mode()

    @app.post("/api/ops/run-mode")
    def set_run_mode(body: _RunModeBody) -> dict[str, Any]:
        """Convenience: set both switches (one MULTI) or either one."""
        return _write_run_mode(body.controller, body.sm_actuation)

    # ---- params: edit per-model registry via ConfigMap, restart-to-apply ----

    def _require_k8s() -> Any:
        if k8s_client is None:
            raise HTTPException(status_code=503, detail="param editing unavailable (no kubernetes access)")
        return k8s_client

    @app.get("/api/params")
    def get_params() -> dict[str, Any]:
        client = _require_k8s()
        try:
            cm = client.get_configmap(_REGISTRY_CM)
            registry_yaml = (cm.get("data") or {}).get("registry.yaml", "")
            rv = (cm.get("metadata") or {}).get("resourceVersion")
            deploy = client.get_deployment(_CONTROLLER_DEPLOY)
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=502, detail=f"kubernetes read failed: {exc}") from exc
        current_hash = _hash(registry_yaml)
        # "Applied" hash is tracked in Redis (UI-writable, no extra RBAC) and seeded to the
        # current config on first read, so a fresh deploy reads as in-sync and the first edit
        # correctly flips to pending. The controller restart also stamps it.
        applied = _get_applied_hash(redis_client)
        if applied is None:
            _set_applied_hash(redis_client, current_hash)
            applied = current_hash
        return {
            "models": params_mod.build_view(registry_yaml),
            "resource_version": rv,
            "params_hash": current_hash,
            "applied_hash": applied,
            "pending_restart": applied != current_hash,
            "last_restart": {k: v for k, v in _template_annotations(deploy).items() if k.startswith("tre.dev/")},
        }

    @app.put("/api/params")
    def put_params(body: _ParamsBody) -> dict[str, Any]:
        client = _require_k8s()
        try:
            cm = client.get_configmap(_REGISTRY_CM)
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=502, detail=f"kubernetes read failed: {exc}") from exc
        rv = (cm.get("metadata") or {}).get("resourceVersion")
        if body.expected_resource_version is not None and body.expected_resource_version != rv:
            raise HTTPException(status_code=409, detail="configmap changed since load; refresh and retry")
        registry_yaml = (cm.get("data") or {}).get("registry.yaml", "")
        try:
            new_yaml = params_mod.apply_and_validate(registry_yaml, body.models)
        except params_mod.ParamValidationError as exc:
            raise HTTPException(status_code=422, detail={"errors": exc.errors}) from exc
        _AUDIT.info(json.dumps({"op": "params_edit", "models": list(body.models), "diff": body.models}))
        _record_audit(redis_client, {"op": "params_edit", "diff": body.models, "prev_rv": rv})
        try:
            client.replace_configmap(_REGISTRY_CM, {"registry.yaml": new_yaml}, rv)
        except Exception as exc:  # noqa: BLE001
            status = getattr(exc, "status", 502)
            raise HTTPException(status_code=409 if status == 409 else 502, detail=f"configmap write failed: {exc}") from exc
        return get_params()

    @app.post("/api/ops/controller/restart")
    def restart_controller(body: _RestartBody) -> dict[str, Any]:
        """Restart-to-apply: rolls the controller AND the service-manager. Both
        read the registry only at start (the SM: placement, service_manager,
        gateway and the models), so a registry change is applied only once
        both restarted."""
        client = _require_k8s()
        try:
            cm = client.get_configmap(_REGISTRY_CM)
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=502, detail=f"restart failed: {exc}") from exc
        registry_yaml = (cm.get("data") or {}).get("registry.yaml", "")
        now = datetime.now(timezone.utc).isoformat()
        patch = {"spec": {"template": {"metadata": {"annotations": {
            "kubectl.kubernetes.io/restartedAt": now,
            "tre.dev/restarted-by": "tre-v2-ui",
            "tre.dev/restart-reason": body.reason[:200],
            _HASH_ANNOTATION: _hash(registry_yaml),
        }}}}}
        _AUDIT.info(json.dumps({"op": "controller_restart", "reason": body.reason,
                                "deployments": [_CONTROLLER_DEPLOY, _SM_DEPLOY]}))
        generations: dict[str, Any] = {}
        for name in (_CONTROLLER_DEPLOY, _SM_DEPLOY):
            try:
                deploy = client.patch_deployment(name, patch)
            except Exception as exc:  # noqa: BLE001
                done = ", ".join(generations) or "none"
                raise HTTPException(
                    status_code=502, detail=f"restart of {name} failed (already restarted: {done}): {exc}"
                ) from exc
            generations[name] = (deploy.get("metadata") or {}).get("generation")
        _set_applied_hash(redis_client, _hash(registry_yaml))
        return {"ok": True, "generation": generations[_CONTROLLER_DEPLOY], "generations": generations,
                "restarted": list(generations), "restarted_at": now}

    @app.get("/api/ops/controller/rollout")
    def controller_rollout() -> dict[str, Any]:
        """Rollout of the restart-to-apply: top-level fields are the controller's
        (compatible), ``state`` is the combined state of both Deployments."""
        client = _require_k8s()
        try:
            controller = _rollout_state(client.get_deployment(_CONTROLLER_DEPLOY))
            service_manager = _rollout_state(client.get_deployment(_SM_DEPLOY))
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=502, detail=f"kubernetes read failed: {exc}") from exc
        states = {controller["state"], service_manager["state"]}
        combined = "failed" if "failed" in states else ("ready" if states == {"ready"} else "progressing")
        return {**controller, "state": combined, "controller": controller, "service_manager": service_manager}

    # ---- observe (legacy per-request reads; superseded by /api/stream, kept for compatibility) ----

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

    @app.get("/api/signal/timeline")
    def signal_timeline(model: str, since_ms: int = 0) -> dict[str, Any]:
        """Richer per-window series (queue_len, throughput, tier, action).

        Served from the sampler's in-memory ring so browsers never trigger an
        upstream read, however many tabs are open.
        """
        if model not in model_names:
            raise HTTPException(status_code=404, detail="unknown model")
        return {"model": model, "points": sampler.timeline(model, since_ms=since_ms)}

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
        _AUDIT.info(json.dumps({"op": "defrag", "tp_size": body.tp_size, "force": body.force}))
        return _proxy(
            service_manager_client, "POST", "/v2/defrag", {"tp_size": body.tp_size, "force": body.force}
        )

    return app


def _require_model(model: str, model_names: list[str]) -> None:
    if model not in model_names:
        raise HTTPException(status_code=404, detail="unknown model")


def _proxy(client: ServiceManagerClient, method: str, path: str, payload: dict[str, Any] | None) -> dict[str, Any]:
    try:
        return {"ok": True, "response": client.request(method, path, payload)}
    except Exception as exc:  # noqa: BLE001
        partial = _partial_body(exc)
        if partial is not None:
            # An exact target the SM filled in part (409 ``partial``, 2026-10-06):
            # what woke stays awake - its answer (actions, unfilled, refusals) is
            # passed through as a 409, not a 502.
            raise HTTPException(status_code=409, detail=partial) from exc
        raise HTTPException(status_code=502, detail=f"service-manager {method} {path} failed: {exc}") from exc


def _partial_body(exc: Exception) -> dict[str, Any] | None:
    """The body of an SM 409 ``error: partial`` (urllib HTTPError), else None."""
    if getattr(exc, "code", None) != 409 or not callable(getattr(exc, "read", None)):
        return None
    try:
        body = json.loads(exc.read().decode("utf-8"))
    except Exception:  # noqa: BLE001
        return None
    return body if isinstance(body, dict) and body.get("error") == "partial" else None


def _hash(text: str) -> str:
    return hashlib.sha256((text or "").encode("utf-8")).hexdigest()[:16]


def _template_annotations(deploy: dict) -> dict[str, str]:
    return (((deploy.get("spec") or {}).get("template") or {}).get("metadata") or {}).get("annotations") or {}


def _rollout_state(deploy: dict) -> dict[str, Any]:
    status = deploy.get("status") or {}
    spec = deploy.get("spec") or {}
    desired = spec.get("replicas", 1)
    updated = status.get("updatedReplicas", 0)
    ready = status.get("readyReplicas", 0)
    conditions = {c.get("type"): c for c in status.get("conditions", [])}
    progressing = conditions.get("Progressing", {})
    if progressing.get("reason") == "ProgressDeadlineExceeded":
        state = "failed"
    elif updated >= desired and ready >= desired and status.get("observedGeneration", 0) >= (deploy.get("metadata") or {}).get("generation", 0):
        state = "ready"
    else:
        state = "progressing"
    return {"state": state, "ready_replicas": ready, "desired": desired,
            "message": progressing.get("message", ""), "observed_generation": status.get("observedGeneration")}


def _get_applied_hash(redis_client: Any) -> str | None:
    try:
        raw = redis_client.get(_PARAMS_APPLIED_KEY)
    except Exception:  # noqa: BLE001
        return None
    return raw.decode() if isinstance(raw, bytes) else raw


def _set_applied_hash(redis_client: Any, value: str) -> None:
    try:
        redis_client.set(_PARAMS_APPLIED_KEY, value)
    except Exception:  # noqa: BLE001
        pass


def _record_audit(redis_client: Any, entry: dict) -> None:
    try:
        redis_client.rpush(_PARAMS_AUDIT_KEY, json.dumps(entry, separators=(",", ":")))
    except Exception:  # noqa: BLE001 - audit is best-effort, never blocks the edit
        pass


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
        "max_awake_replicas": model.scale_max_replicas,
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
