from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query



def create_v1_compat_router(service) -> APIRouter:
    # v2 imports this module: its errors are imported here, once v2 is loaded.
    from tre_sm.api.v2 import TargetPartial, WakeConflict, WakeFailed
    from tre_sm.state.gpu_leases import GpuLeaseConflict

    def _wake_refusal(exc: BaseException) -> bool:
        if isinstance(exc, GpuLeaseConflict):
            return True  # a lease taken on the GPU (the prepare's _prepare_wake)
        return not isinstance(exc, WakeFailed) and exc.reason != "routable_unknown"

    router = APIRouter()

    @router.post("/models_replicas")
    def models_replicas(models: str = Query(...)) -> dict[str, int]:
        # The light state (store only, no Pod LIST): awake and not hidden - the
        # base /scale_service converts a delta from (at arrival).
        state = service.get_state()
        result: dict[str, int] = {}
        for model in _split_models(models):
            result[model] = _awake_count(state, model)
        return result

    @router.post("/scale_service")
    def scale_service(
        model_name: str = Query(...),
        scale_type: str = Query(...),
        scale_value: int = Query(...),
    ) -> dict[str, int]:
        if scale_value < 0:
            raise HTTPException(status_code=400, detail="scale_value must be non-negative")
        # The delta becomes an absolute target HERE, at arrival, from the same
        # count /models_replicas reports (2026-10-04, I2): APA computed its delta
        # from that count, so a retried or duplicated call converts to the same
        # target and is harmless. (Converting under the writer lock instead let
        # retries stack: two "+1" from one base woke two.) Remaining race: two
        # DIFFERENT deltas computed from one base by different callers - only an
        # absolute desiredReplicas from APA removes it (deferred, design note).
        state = service.get_state()
        current = _awake_count(state, model_name)
        if scale_type == "up":
            target = current + scale_value
        elif scale_type == "down":
            target = max(0, current - scale_value)
        else:
            raise HTTPException(status_code=400, detail="scale_type must be up or down")
        try:
            # APA scale-downs take the "apa" sleep path (no drain, like every
            # sleep of the service-manager); a scale-up ignores it. ``actual``
            # is the number of wake / sleep actions of THIS call.
            response = service.put_model_target(model_name, wake_replicas=target, sleep_path="apa")
        except TargetPartial as exc:
            # v1 contract (execute_scale_up, best effort): 200 with what woke; the
            # APA client warns on actual < requested.
            response = exc.response
        except (WakeConflict, GpuLeaseConflict) as exc:
            # No sleeping binding could be placed (refused before any change): v1
            # answered 200 with actual 0, and so does this (the APA client warns and
            # re-reconciles; an error only makes it requeue with backoff). Only for a
            # wake refusal of a real growth (target above every awake binding,
            # hidden ones included): a shrink-side refusal (``routable_unknown``) or
            # a wake that itself failed keeps its old status.
            if _wake_refusal(exc) and target > _awake_total(state, model_name):
                return {"requested": scale_value, "actual": 0}
            if isinstance(exc, GpuLeaseConflict):
                raise  # the app's GpuLeaseConflict handler, as before
            raise HTTPException(status_code=400, detail=str(exc)) from exc  # a ValueError
        except (KeyError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {"requested": scale_value, "actual": len(response["actions"])}

    @router.post("/wake_up")
    def wake_up(
        model_name: str = Query(...),
        kind: int = Query(0),
        queue_len: int = Query(0),
    ) -> dict:
        del kind, queue_len
        # One more than awake and not hidden, converted at arrival like
        # /scale_service (a replay is harmless); never past the model's bindings
        # (no cold create) - "delayed" then, as before.
        current = _awake_count(service.get_state(), model_name)
        try:
            response = service.put_model_target(model_name, wake_replicas=current + 1, within_bindings=True)
        except (KeyError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        if response.get("at_bindings_limit"):
            return _wake_response(success=False, delayed=True, wake_ids=[])
        wake_ids = [action["serve_id"] for action in response["actions"] if action["action"] == "wake"]
        return _wake_response(success=bool(wake_ids), delayed=False, wake_ids=wake_ids)

    return router


def _split_models(models: str) -> list[str]:
    return [model.strip() for model in models.split(",") if model.strip()]


def _awake_total(state: dict, model: str) -> int:
    """Awake bindings of ``model``, hidden ones included."""
    return sum(1 for binding in state["bindings"] if binding["model"] == model and binding["awake"])


def _awake_count(state: dict, model: str) -> int:
    return sum(
        1
        for binding in state["bindings"]
        if binding["model"] == model and binding["awake"] and not binding.get("hidden", False)
    )


def _wake_response(*, success: bool, delayed: bool, wake_ids: list[str]) -> dict:
    return {
        "success": success,
        "delayed": delayed,
        "strategy_type": "wake_up",
        "strategy": {"serves_to_sleep": [], "serves_to_wakeup": wake_ids},
        "total_cost": 0.0,
        "wake_up_time": 0.0,
    }
