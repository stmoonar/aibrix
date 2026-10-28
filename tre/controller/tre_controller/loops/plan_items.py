"""Plan items of the controller's power actions (design 20261027-sm-plans §5).

Every power dispatch of the ActionQueue is one ``POST /v2/plans``: each member
action becomes exactly ONE item (strict 1:1, no same-model merging):

* ``ScaleAction +k`` with pods  -> wake ``pods`` (exactly these bindings: the controller
  names every pod, sm-plans section 10)
* ``ScaleAction +k``            -> wake ``at_least = awake + k`` (a cold create only: the
  receiver has no sleeping binding at the target)
* ``ScaleAction -k`` with pods  -> sleep ``pods`` (a transfer's donor pods, a SafeScale
  commit donor, a named shrink)
* ``ScaleAction -k``            -> sleep ``at_most = max(serving floor, awake - k)``
* ``TransferIntent D->R``       -> wake of R ``at_least = awake_R + pairs``, ``donors=[{D, count}]``
* ``ReceiverTarget``            -> wake ``pods`` = its bindings on the donor pods' GPUs;
  without any, ``at_least = target`` (resolved once, then frozen)

A member's ``group`` (one transfer: donor sleeps + receiver wake) is sent as the item's
``group``.

Relative fields are resolved from ONE ``GET /v2/state`` per plan; an item whose target
is already met is dropped locally (no SM call for it). Pure helpers: the ActionQueue
keeps the events, stats and stamps."""

from __future__ import annotations

from typing import Any, Mapping

from tre_controller.planning.planner import ReceiverTarget, ScaleAction, TransferIntent

#: Item statuses of a plan response that changed nothing or did what was asked.
SLEEP_OK_STATUSES = frozenset({"done", "noop", "clamped"})
WAKE_OK_STATUSES = frozenset({"done", "noop"})
#: Item reason of a SafeScale receiver (a ReceiverTarget carries no planner reason).
RECEIVER_REASON = "safescale_receiver"
#: Refusal of a wake item whose GPUs a failed / unconfirmed plan sleep still holds: the
#: same commit re-sends it after a retry of the donor sleep.
DONOR_SLEEP_FAILED = "donor_sleep_failed"


class SleepPathRefused(ValueError):
    """A scale-down that names no sleep path: refused before any SM call."""


class StateUnreadable(ValueError):
    """``/v2/state`` has no usable count for a model an item must resolve."""


def sleep_path_of(action: ScaleAction) -> str:
    """SM sleep path of a scale-down (plan 2026-09-27 D1). A negative delta must name its
    path (explicitly or via an "*_immediate" reason, the rescue-section donor paths:
    "urgent"): there is no implicit "scale_down" (a path-less shrink would silently take
    the SM default - SM-side drain - and bypass SafeScale)."""
    path = action.sleep_path or ("urgent" if str(action.reason).endswith("_immediate") else None)
    if path is None:
        raise SleepPathRefused(
            f"scale-down of {action.model} ({action.reason}) has no sleep_path: "
            "refusing the implicit SM scale_down default"
        )
    return path


def plan_item(action) -> tuple[str, dict] | None:
    """(``"sleep"`` | ``"wake"``, item) of one member action; relative parts are kept as
    ``_delta`` / ``_cap`` for :func:`resolve_item`. None = nothing to ask (delta 0).
    Raises :class:`SleepPathRefused` for a path-less scale-down."""
    if isinstance(action, ReceiverTarget):
        item: dict[str, Any] = {"model": action.model, "reason": RECEIVER_REASON}
        if action.hints:
            # Named (sm-plans section 10): a retry of an awake binding is a no-op.
            item["pods"] = list(action.hints)
            return "wake", item
        if action.target is not None:
            item["at_least"] = int(action.target)
        else:
            item["_delta"] = max(0, int(action.delta))
            item["_cap"] = action.cap
        return "wake", item
    if isinstance(action, TransferIntent):
        return "wake", {
            "model": action.receiver_model,
            "reason": str(action.reason),
            "_delta": int(action.pairs),
            "donors": [{"model": action.donor_model, "max": int(action.count), "sleep_path": action.sleep_path}],
        }
    if not isinstance(action, ScaleAction):
        raise TypeError(f"not a power action: {type(action).__name__}")
    delta = int(action.delta)
    if delta > 0:
        item = {"model": action.model, "reason": str(action.reason)}
        if action.pods:
            item["pods"] = list(action.pods)
        else:
            item["_delta"] = delta
        if action.group:
            item["group"] = action.group
        return "wake", item
    if delta < 0:
        item = {"model": action.model, "reason": str(action.reason), "sleep_path": sleep_path_of(action)}
        if action.pods:
            item["pods"] = list(action.pods)
        else:
            item["_delta"] = delta
        if action.group:
            item["group"] = action.group
        return "sleep", item
    return None


def needs_state(kind: str, item: Mapping[str, Any]) -> bool:
    """The item must read the model's awake count (relative or absolute target)."""
    return "_delta" in item or "at_least" in item


def state_counts(state: Any, models) -> dict[str, tuple[int, int]]:
    """model -> (awake, bound) of a ``/v2/state`` answer for ``models`` (a model without a
    readable entry is left out)."""
    entries = state.get("models") if isinstance(state, dict) else None
    counts: dict[str, tuple[int, int]] = {}
    if entries is None or not hasattr(entries, "get"):
        return counts
    for model in models:
        entry = entries.get(model)
        if not isinstance(entry, dict):
            continue
        try:
            counts[str(model)] = (int(entry.get("awake", 0)), int(entry.get("bound", 0)))
        except (TypeError, ValueError):
            continue
    return counts


def resolve_item(kind: str, item: dict, counts: Mapping[str, tuple[int, int]]) -> dict | None:
    """The request form of ``item`` (relative parts resolved against ``counts``), or None
    when its target is already met (dropped locally). Raises :class:`StateUnreadable`."""
    if not needs_state(kind, item):
        return {key: value for key, value in item.items() if not key.startswith("_")}
    model = item["model"]
    if model not in counts:
        raise StateUnreadable(f"malformed /v2/state: no counts for {model}")
    awake, bound = counts[model]
    out = {key: value for key, value in item.items() if not key.startswith("_")}
    if kind == "wake":
        if "_delta" in item:
            target = awake + int(item["_delta"])
            cap = item.get("_cap")
            if cap is not None:
                target = min(target, max(int(cap), awake))
            out["at_least"] = target
        return None if int(out["at_least"]) <= awake else out
    # Model-level shrink: one awake bound replica keeps serving (as the old /target call).
    serving_floor = 1 if bound > 0 and awake > 0 else 0
    out["at_most"] = max(serving_floor, awake + int(item["_delta"]))
    return None if out["at_most"] >= awake else out


def item_code(item: Mapping[str, Any]) -> str | None:
    """The error code of a refused / failed item: its ``error`` (a code or a structured
    body), else the code of its first refusal."""
    error = item.get("error")
    if isinstance(error, dict):
        error = error.get("error")
    if isinstance(error, str) and error:
        return error
    for refusal in item.get("refusals") or ():
        if isinstance(refusal, dict) and refusal.get("error"):
            return str(refusal["error"])
    return None


def item_retriable(item: Mapping[str, Any]) -> bool:
    """The item's ``retriable`` (design §3: the SM always sends it and mirrors the old
    per-endpoint classification). Fallback for an answer without it: a refusal is
    retriable unless it is a floor violation, a failure is not (the wake failed on that
    GPU and was settled; the planner re-plans)."""
    value = item.get("retriable")
    if isinstance(value, bool):
        return value
    code = item_code(item)
    if code == DONOR_SLEEP_FAILED:
        return True
    if code in ("floor_violation", "wake_failed"):
        return False
    return item.get("status") == "refused"


def item_failure(item: Mapping[str, Any]) -> dict:
    """A failed item in the shape of ``ServiceManagerError.result()`` (what the
    ActionQueue's ``_dispatch_result`` reads)."""
    code = item_code(item)
    status = str(item.get("status") or "failed")
    error = item.get("error")
    detail = error.get("reason") or error.get("detail") if isinstance(error, dict) else None
    result: dict[str, Any] = {
        "ok": False,
        "error": f"{status}: {code or 'unknown'}" + (f" ({detail})" if detail else ""),
        "status": 409,
        "retriable": item_retriable(item),
    }
    if code:
        result["code"] = code
    if status == "refused":
        # A refused item changed nothing for its model (refused before any hide / wake).
        result["not_executed"] = True
    if code == "floor_violation":
        result["floor_violation"] = item.get("floor")
    outcomes = item.get("outcomes")
    if isinstance(outcomes, list):
        result["outcomes"] = outcomes
    refusal = next((entry for entry in item.get("refusals") or () if isinstance(entry, dict)), None)
    if refusal is not None and refusal.get("node"):
        result["wake_conflict"] = refusal_conflict(refusal)
    return result


def refusal_conflict(refusal: Mapping[str, Any]) -> dict:
    """A structured refusal body (``refusals[]`` of a wake item) in the shape of
    ``ServiceManagerError.wake_conflict``."""
    raw = refusal.get("gpu_ids", refusal.get("gpu"))
    if isinstance(raw, (int, str)) and not isinstance(raw, bool):
        raw = [raw]
    return {
        "node": refusal.get("node"), "gpu_ids": list(raw or ()),
        "scope": refusal.get("scope"), "error": refusal.get("error"), "reason": refusal.get("reason"),
        "blocking_binding_id": refusal.get("blocking_binding_id"),
    }


def woke_entries(item: Mapping[str, Any]) -> tuple[dict, ...]:
    return tuple(entry for entry in item.get("woke") or () if isinstance(entry, dict) and entry.get("serve_id"))


def woke_ids(item: Mapping[str, Any]) -> tuple[str, ...]:
    """Serve ids a wake item woke."""
    return tuple(str(entry["serve_id"]) for entry in woke_entries(item))


def slept_ids(item: Mapping[str, Any]) -> tuple[str, ...]:
    """Serve ids a sleep item confirmed asleep (its per-pod sleep outcomes)."""
    return tuple(
        str(entry.get("serve_id")) for entry in item.get("outcomes") or ()
        if isinstance(entry, dict) and entry.get("status") == "slept" and entry.get("serve_id")
    )


def num(item: Mapping[str, Any], key: str) -> int:
    try:
        return max(0, int(item.get(key) or 0))
    except (TypeError, ValueError):
        return 0


def transfer_summary(item: Mapping[str, Any], plan_id: Any) -> dict:
    """The accounting view of a relay's wake item (the summary ``POST /v2/transfers``
    used to give): ``done`` = receivers woken, ``taken`` = donor replicas slept, one
    pair per woken receiver with the donor bindings it took."""
    woke = woke_entries(item)
    donors_slept = [entry for entry in item.get("donors_slept") or () if isinstance(entry, dict)]
    taken = sum(1 for entry in donors_slept if entry.get("status") in (None, "slept", "done"))
    pairs = []
    for entry in woke:
        donors = entry.get("donor_binding_ids") or ()
        pairs.append({
            "donors": [str(donor) for donor in donors],
            "receiver": str(entry.get("serve_id") or ""),
            "node": str(entry.get("node") or ""),
            "gpu_ids": list(entry.get("gpu_ids") or ()),
            "status": "done",
            "source": str(entry.get("source") or ""),
        })
    skipped = item.get("skipped") if isinstance(item.get("skipped"), dict) else {}
    status = str(item.get("status") or "")
    refusals = [entry for entry in item.get("refusals") or () if isinstance(entry, dict)]
    # A refused item (nothing woke, no donor taken) stays outcome "partial", as a 409
    # ``partial`` of /v2/transfers did: the relay hold reads clamped / skipped / refusal
    # codes, the stamps skip both sides (no pair, nothing taken).
    summary = {
        "transfer_id": plan_id,
        "outcome": "ok" if woke else "partial",
        "done": len(woke),
        "taken": taken,
        "unfilled": num(item, "unfilled"),
        "clamped_by_floor": bool(item.get("clamped_by_floor")),
        "pairs": pairs,
        "refusals": len(refusals),
        "refusal_codes": [[str(entry.get("error") or ""), str(entry.get("reason") or "")] for entry in refusals],
        "skipped": {str(key): value for key, value in skipped.items()},
        "picked": [entry.get("serve_id") for entry in woke],
        "item_status": status,
    }
    if status == "failed":
        summary["wake_failed"] = True
    return summary
