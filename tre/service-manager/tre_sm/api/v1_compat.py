from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query



def create_v1_compat_router(service) -> APIRouter:
    router = APIRouter()

    @router.post("/models_replicas")
    def models_replicas(models: str = Query(...)) -> dict[str, int]:
        # The light state (store only, no Pod LIST): awake and not hidden - the
        # base /scale_service converts a delta from.
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
        if scale_type == "up":
            delta = scale_value
        elif scale_type == "down":
            delta = -scale_value
        else:
            raise HTTPException(status_code=400, detail="scale_type must be up or down")
        try:
            # A delta, converted under the writer lock from the awake and not
            # hidden count (2026-10-02; it was converted here, outside the lock,
            # from the same count): the same target as before for one call, and
            # two concurrent calls add up instead of converting from one base.
            # APA scale-downs take the "apa" sleep path (no drain, like every
            # sleep of the service-manager); a scale-up ignores it. ``actual``
            # stays the number of wake / sleep actions of THIS call.
            response = service.put_model_target(model_name, delta=delta, sleep_path="apa")
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
        try:
            # One more than awake and not hidden, under the writer lock; never past
            # the model's bindings (no cold create) - "delayed" then, as before.
            response = service.put_model_target(model_name, delta=1, within_bindings=True)
        except (KeyError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        if response.get("at_bindings_limit"):
            return _wake_response(success=False, delayed=True, wake_ids=[])
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
