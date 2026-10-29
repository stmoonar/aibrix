"""Test-only bridge from ``POST /v2/plans`` to the per-endpoint fakes written before it.

Many queue tests script a fake service-manager per legacy call (``scale_model``,
``scale_model_to``, ``scale_model_hinted``, ``set_binding_power``, ``transfer``) and assert
the controller's semantics (stamps, retries, rescue credit, relay holds, events) on its
answers. :class:`PlanShim` keeps those fakes: it answers ``get_state_result`` from fixed
awake counts and ``post_plan`` by calling the legacy method of each item - sleeps first,
then wakes, in request order, one call per item (per pod for named items, stopping at the
first failure) - and translating the answers into plan item results (design
20261027-sm-plans §3).

Conservative SM model: when a named sleep of the plan failed, the plan's wakes are refused
``donor_sleep_failed`` without a call (the old commit never sent its receivers then). That is
the real SM's answer only for receivers on the failed donor's GPUs: the legacy fakes have no GPU
map. The real SM wakes a receiver on a free GPU (wave 1) even when the donor sleep fails; that
case is tested with plan-native fakes (``test_plans_20261027.py``
``test_a_receiver_on_a_free_gpu_may_wake_when_the_donor_sleep_fails``) and end to end in
``service-manager/tests/test_plan_execute_20261027.py``. Do not use the shim to assert it. An
unstructured failure of the FIRST call (timeout, transport, writer_busy, routable_unknown,
400 / 404 / 503, a raised exception) is the answer of the whole plan; later in the plan it
fails only its item. New behaviour of the plan path is tested with plan-native fakes."""

from __future__ import annotations

from typing import Any, Callable

from tre_controller.loops.action_queue import ActionQueue
from tre_controller.loops.plan_items import RECEIVER_REASON


class _Counts(dict):
    """``/v2/state`` ``models``: a fixed entry for every model asked for."""

    def __init__(self, shim: "PlanShim") -> None:
        super().__init__()
        self._shim = shim

    def get(self, model, default=None):  # noqa: D401 - dict API
        return {"awake": self._shim.awake_of(model), "bound": self._shim.bound_of(model)}


class _Abort(Exception):
    def __init__(self, response: Any) -> None:
        super().__init__("whole-plan answer")
        self.response = response


def _structured(response: dict) -> bool:
    return bool(
        response.get("partial") or isinstance(response.get("outcomes"), list)
        or "floor_violation" in response or response.get("wake_conflict")
    )


def _failed_item(response: dict, *, status: str = "failed") -> dict:
    item: dict[str, Any] = {"status": status, "retriable": bool(response.get("retriable", False))}
    code = response.get("code")
    if "floor_violation" in response:
        item.update({"status": "refused", "error": "floor_violation", "floor": response.get("floor_violation")})
    elif response.get("wake_conflict"):
        conflict = dict(response["wake_conflict"])
        item.update({"status": "refused", "error": conflict.get("error"), "refusals": [conflict]})
    else:
        item["error"] = code or str(response.get("error") or "failed")
    if response.get("not_executed") and item["status"] == "failed":
        item["status"] = "refused"
    if isinstance(response.get("outcomes"), list):
        item["outcomes"] = list(response["outcomes"])
    return item


class PlanShim:
    """Wraps a pre-plan fake; every other attribute is the fake's own."""

    def __init__(
        self,
        inner,
        *,
        awake: dict[str, int] | None = None,
        default_awake: int = 8,
        bound: int = 8,
        state: Callable[[], dict] | None = None,
    ):
        self.inner = inner
        self.awake = awake if awake is not None else {}  # kept by reference: a test may change it
        self.default_awake = default_awake
        self.bound = bound
        #: A real ``/v2/state`` (e.g. an in-process service's): counts come from it.
        self.state = state
        self.plans: list[dict] = []
        self.state_reads = 0

    def __getattr__(self, name):
        if name == "inner":
            raise AttributeError(name)
        return getattr(self.inner, name)

    def _entry(self, model: str) -> dict | None:
        if self.state is None:
            return None
        entry = (self.state().get("models") or {}).get(model)
        return entry if isinstance(entry, dict) else {"awake": 0, "bound": 0}

    def awake_of(self, model: str) -> int:
        entry = self._entry(model)
        if entry is not None:
            return int(entry.get("awake", 0))
        return int(self.awake.get(model, self.default_awake))

    def bound_of(self, model: str) -> int:
        entry = self._entry(model)
        return int(entry.get("bound", 0)) if entry is not None else self.bound

    async def get_state_result(self) -> dict:
        self.state_reads += 1
        return {"ok": True, "response": {"models": _Counts(self)}}

    async def post_plan(self, request: dict) -> dict:
        self.plans.append(request)
        self._calls = 0
        sleeps, wakes = [], []
        named_sleep_failed = False
        try:
            for index, item in enumerate(request.get("sleeps") or ()):
                result = await self._sleep(item)
                if item.get("pods") and result["status"] not in ("done", "noop", "clamped"):
                    named_sleep_failed = True
                sleeps.append({"index": index, "model": item["model"], "reason": item.get("reason", ""), **result})
            for index, item in enumerate(request.get("wakes") or ()):
                if named_sleep_failed:
                    result = {"status": "refused", "error": "donor_sleep_failed", "retriable": True,
                              "woke": [], "unfilled": 0}
                else:
                    result = await self._wake(item)
                wakes.append({"index": index, "model": item["model"], "reason": item.get("reason", ""), **result})
        except _Abort as abort:
            if isinstance(abort.response, BaseException):
                raise abort.response
            return abort.response
        plan_id = getattr(self, "_plan_id", None) or "pl-shim"
        return {"ok": True, "status": 200, "response": {"plan_id": plan_id, "status": "done",
                                                         "sleeps": sleeps, "wakes": wakes}}

    async def _legacy(self, method: str, *args, **kwargs) -> dict:
        first = self._calls == 0
        self._calls += 1
        try:
            response = await getattr(self.inner, method)(*args, **kwargs)
        except Exception as exc:  # noqa: BLE001 - the old dispatch caught it per call
            if first:
                raise _Abort(exc)
            return {"ok": False, "error": f"dispatch_exception: {type(exc).__name__}: {exc}", "retriable": False}
        if not isinstance(response, dict):
            response = {"ok": False, "error": "malformed"}
        if first and not response.get("ok", False) and not _structured(response):
            raise _Abort(response)
        return response

    async def _sleep(self, item: dict) -> dict:
        model = item["model"]
        path = item.get("sleep_path")
        kwargs = {} if path in (None, "scale_down") else {"sleep_path": path}
        if item.get("pods"):
            outcomes = []
            for pod in item["pods"]:
                response = await self._legacy("set_binding_power", pod, awake=False, **kwargs)
                if not response.get("ok", False):
                    failed = _failed_item(response)
                    failed["outcomes"] = outcomes + list(failed.get("outcomes") or ())
                    return failed
                outcomes.append({"serve_id": pod, "status": "slept"})
            return {"status": "done", "taken": len(item["pods"]), "outcomes": outcomes}
        awake = self.awake_of(model)
        response = await self._legacy("scale_model", model, int(item["at_most"]) - awake, **kwargs)
        if not response.get("ok", False):
            return _failed_item(response)
        body = response.get("response") if isinstance(response.get("response"), dict) else {}
        taken = int(body["taken"]) if "taken" in body else awake - int(item["at_most"])
        return {"status": "clamped" if body.get("clamped_by_floor") else "done", "taken": taken,
                "clamped_by_floor": bool(body.get("clamped_by_floor")), "floor": body.get("floor")}

    async def _wake(self, item: dict) -> dict:
        model = item["model"]
        hinted_by_power = item.get("hints") and not callable(getattr(self.inner, "scale_model_hinted", None))
        if item.get("pods") or hinted_by_power:
            # Named wakes; a fake without a hinted call wakes the hints by binding power
            # (the SM honours hints first).
            pods = item.get("pods") or item["hints"][: max(0, int(item["at_least"]) - self.awake_of(model))]
            woke = []
            for pod in pods:
                response = await self._legacy("set_binding_power", pod, awake=True)
                if not response.get("ok", False):
                    return {**_failed_item(response), "woke": woke}
                woke.append({"serve_id": pod})
            return {"status": "done", "woke": woke, "unfilled": 0}
        awake = self.awake_of(model)
        target = int(item["at_least"])
        if item.get("hints") and callable(getattr(self.inner, "scale_model_hinted", None)):
            response = await self._legacy("scale_model_hinted", model, target - awake, hints=tuple(item["hints"]))
        elif item.get("reason") == RECEIVER_REASON and callable(getattr(self.inner, "scale_model_to", None)):
            response = await self._legacy("scale_model_to", model, target)
        else:
            response = await self._legacy("scale_model", model, target - awake)
        return self._growth_item(response)

    def _growth_item(self, response: dict) -> dict:
        body = response.get("response") if isinstance(response.get("response"), dict) else {}
        if not response.get("ok", False) and not response.get("partial"):
            return {**_failed_item(response), "woke": []}
        woke = [dict(entry) for entry in body.get("picked") or () if isinstance(entry, dict) and entry.get("serve_id")]
        if not woke:
            woke = [{"serve_id": str(entry["serve_id"])} for entry in body.get("actions") or ()
                    if isinstance(entry, dict) and entry.get("action") == "wake" and entry.get("serve_id")]
        unfilled = int(body.get("unfilled") or 0)
        status = "noop" if body.get("noop") else "partial" if (unfilled > 0 or response.get("partial")) else "done"
        if status == "partial" and not woke:
            status = "refused"
        return {"status": status, "woke": woke, "unfilled": unfilled,
                "refusals": list(body.get("refusals") or ())}

class LegacyFakeQueue(ActionQueue):
    """An ActionQueue whose pre-plan fake client is wrapped in a :class:`PlanShim`
    (a client with ``post_plan`` is used as is)."""

    def __init__(self, client, *args, **kwargs):
        if not callable(getattr(type(client), "post_plan", None)):
            # A fake may pin the awake counts its old ``model_awake`` answered.
            client = PlanShim(
                client,
                default_awake=int(getattr(client, "default_awake", 8)),
                awake=getattr(client, "awake_counts", None),
            )
        super().__init__(client, *args, **kwargs)


class PlanSM:
    """A plan-native fake service-manager: fixed awake counts for the plan's GET, every
    POST recorded; by default every item is done (a sleep takes its pods / down to
    ``at_most``, a wake wakes up to ``at_least`` / its pods). ``respond(request)`` may
    return any ``post_plan`` answer instead; ``gate`` (an asyncio.Event) holds the POST."""

    def __init__(self, awake: dict[str, int] | None = None, *, bound: int = 8, respond=None, gate=None) -> None:
        self.awake = awake if awake is not None else {}
        self.bound = bound
        self.respond = respond
        self.gate = gate
        self.plans: list[dict] = []
        self.state_reads = 0
        self.routable: list[tuple] = []

    async def get_state_result(self) -> dict:
        self.state_reads += 1
        models = {model: {"awake": awake, "bound": self.bound} for model, awake in self.awake.items()}
        return {"ok": True, "response": {"models": models}}

    async def set_routable(self, model, hidden_pods):
        self.routable.append((model, tuple(hidden_pods)))
        return {"ok": True}

    async def defrag(self, migrations):
        return {"ok": True}

    async def post_plan(self, request: dict) -> dict:
        self.plans.append(request)
        if self.gate is not None:
            await self.gate.wait()
        if self.respond is not None:
            return self.respond(request)
        return {"ok": True, "status": 200, "response": done_plan(request, self.awake)}


def done_plan(request: dict, awake: dict[str, int] | None = None, *, plan_id: str = "pl-test") -> dict:
    """A plan response with every item of ``request`` done."""
    awake = awake or {}
    sleeps = []
    for index, item in enumerate(request.get("sleeps") or ()):
        if item.get("pods"):
            taken = len(item["pods"])
            outcomes = [{"serve_id": pod, "status": "slept"} for pod in item["pods"]]
        else:
            taken = max(0, awake.get(item["model"], 0) - int(item["at_most"]))
            outcomes = []
        sleeps.append({"index": index, "model": item["model"], "status": "done", "taken": taken,
                       "outcomes": outcomes})
    wakes = []
    for index, item in enumerate(request.get("wakes") or ()):
        model = item["model"]
        if item.get("pods"):
            names = list(item["pods"])
        else:
            names = [f"{model}-w{i}" for i in range(awake.get(model, 0), int(item["at_least"]))]
        woke = [{"serve_id": name, "node": "n", "gpu_ids": [i], "wave": 1, "source": "free"}
                for i, name in enumerate(names)]
        wakes.append({"index": index, "model": model, "status": "done", "woke": woke,
                      "unfilled": 0, "refusals": []})
    return {"plan_id": plan_id, "status": "done", "sleeps": sleeps, "wakes": wakes}
