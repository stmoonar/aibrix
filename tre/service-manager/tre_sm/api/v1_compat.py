from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query



def create_v1_compat_router(service) -> APIRouter:
    router = APIRouter()

    @router.post("/models_replicas")
    def models_replicas(models: str = Query(...)) -> dict[str, int]:
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
        current = _awake_count(service.get_state(), model_name)
        if scale_type == "up":
            target = current + scale_value
        elif scale_type == "down":
            target = max(0, current - scale_value)
        else:
            raise HTTPException(status_code=400, detail="scale_type must be up or down")
        try:
            response = service.put_model_target(model_name, wake_replicas=target)
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
        state = service.get_state()
        current = _awake_count(state, model_name)
        bound = state["models"].get(model_name, {}).get("bound", 0)
        if current >= bound:
            return _wake_response(success=False, delayed=True, wake_ids=[])
        try:
            response = service.put_model_target(model_name, wake_replicas=current + 1)
        except (KeyError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        wake_ids = [action["serve_id"] for action in response["actions"] if action["action"] == "wake"]
        return _wake_response(success=bool(wake_ids), delayed=False, wake_ids=wake_ids)

    return router


def _split_models(models: str) -> list[str]:
    return [model.strip() for model in models.split(",") if model.strip()]


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
