"""Baseline / APA compatibility of ``POST /v2/plans`` (2026-10-28, design
20261027-sm-plans.md section 11): the request shapes the TokenScale / PreServe
dispatcher (``tre_baselines.sm_client``: ``{plan_id, sleeps: [{model, at_most,
sleep_path}], wakes: [{model, at_least}]}``) and the APA controller (Go
``workload_scale.go``: one-item plans ``{model, at_least, reason}`` /
``{model, at_most, sleep_path: "apa", reason}``) send must see exactly the
selection and the response of the code before plan groups and named multi-items
(snapshot taken from aibrix 674f80b5, ``golden/plan_baseline_20261028.json``).

Regenerate only on purpose: ``TRE_REGEN_GOLDEN=1 pytest <this file>``."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from tre_common.registry import ClusterTopology, NodeSpec
from tre_sm.allocator.slots import Binding, Slot, release_order
from tre_sm.ops.plan import SleepItem, WakeItem, select_plan

from test_wholelock_20261002 import World

GOLDEN = Path(__file__).parent / "golden" / "plan_baseline_20261028.json"
TOPOLOGY = ClusterTopology(
    nodes=(NodeSpec("node-a", 4, ((0, 1), (2, 3)), ("GPU-0", "GPU-1", "GPU-2", "GPU-3")),)
)

#: d serves on GPUs 0 and 1; r sleeps on 0, 1 and 2; x on 2; t (TP2) on 2-3.
FLEET = (
    ("d-0", "d", (0,), True),
    ("d-1", "d", (1,), True),
    ("r-0", "r", (0,), False),
    ("r-1", "r", (1,), False),
    ("r-2", "r", (2,), False),
    ("x-2", "x", (2,), False),
    ("t-23", "t", (2, 3), False),
)


def _fleet():
    return [Binding(name, model, Slot("node-a", gpus), awake=awake) for name, model, gpus, awake in FLEET]


def _blocker(binding, planning, _ignored):
    for other in planning:
        if other.awake and other.serve_id != binding.serve_id and set(other.slot.gpu_ids) & set(binding.slot.gpu_ids):
            return f"slot_occupied by {other.serve_id}"
    return None


def _select(sleeps=(), wakes=(), d_floor=None):
    fleet = _fleet()
    routable = [b.binding_id for b in fleet if b.model == "d" and b.awake]
    return select_plan(
        fleet,
        sleeps=sleeps,
        wakes=wakes,
        topology=TOPOLOGY,
        floors={"d": (d_floor, routable)},
        caps={"d": 4, "r": 4, "x": 4, "t": 2},
        wake_blocker=_blocker,
        shrink_order=lambda serving, books, released: release_order(
            serving, bindings=books, topology=TOPOLOGY, already_released=released
        ),
    )


def _selection_view(selection) -> dict:
    return {
        "sleeps": [
            {
                "index": choice.index, "model": choice.item.model,
                "members": [b.serve_id for b in choice.members],
                "already_asleep": [b.serve_id for b in choice.already_asleep],
                "kept_awake": [b.serve_id for b in choice.kept_awake],
                "clamped": choice.clamped,
                "refusal": getattr(choice.refusal, "reason", None),
            }
            for choice in selection.sleeps
        ],
        "wakes": [
            {
                "index": choice.index, "model": choice.item.model,
                "picks": [
                    [p.binding.serve_id, p.wave, list(p.depends_on), p.source, p.hinted, list(p.donor_ids)]
                    for p in choice.picks
                ],
                "already_awake": [b.serve_id for b in choice.already_awake],
                "unfilled": choice.unfilled,
                "refusals": [str(getattr(r, "reason", r)) for r in choice.refusals],
                "skipped": dict(choice.skipped),
            }
            for choice in selection.wakes
        ],
    }


SELECT_CASES = {
    # TokenScale: a mixed plan, one item per model, count forms only.
    "tokenscale_mixed": dict(sleeps=[SleepItem("d", "urgent", at_most=1)], wakes=[WakeItem("r", at_least=2)]),
    # PreServe: several wakes and a full shrink in one plan.
    "preserve_multi": dict(
        sleeps=[SleepItem("d", "urgent", at_most=0)],
        wakes=[WakeItem("x", at_least=1), WakeItem("t", at_least=1), WakeItem("r", at_least=1)],
    ),
    "apa_up": dict(wakes=[WakeItem("r", at_least=1, reason="apa")]),
    "apa_down": dict(sleeps=[SleepItem("d", "apa", at_most=1, reason="apa")]),
    "floor_clamp": dict(sleeps=[SleepItem("d", "urgent", at_most=0)], d_floor=2),
    "noop_targets": dict(sleeps=[SleepItem("d", "urgent", at_most=2)], wakes=[WakeItem("r", at_least=0)]),
}

#: (World kwargs, request) of the HTTP cases; one fresh World each.
WORLD_PODS = (
    ("d-0", "d", (0,), "awake"),
    ("r-0", "r", (0,), "sleeping"),
    ("x-0", "x", (0,), "sleeping"),
    ("d-1", "d", (1,), "awake"),
    ("r-1", "r", (1,), "sleeping"),
    ("r-2", "r", (2,), "sleeping"),
    ("x-2", "x", (2,), "sleeping"),
)
HTTP_CASES = {
    "tokenscale_mixed": ({}, {
        "plan_id": "bl-g1",
        "sleeps": [{"model": "d", "at_most": 1, "sleep_path": "urgent"}],
        "wakes": [{"model": "r", "at_least": 2}],
    }),
    "preserve_multi": ({}, {
        "plan_id": "bl-g2",
        "sleeps": [{"model": "d", "at_most": 0, "sleep_path": "urgent"}],
        "wakes": [{"model": "x", "at_least": 1}, {"model": "r", "at_least": 1}],
    }),
    "apa_up": ({}, {"plan_id": "apa-g3", "wakes": [{"model": "r", "at_least": 1, "reason": "apa"}]}),
    "apa_down": ({}, {
        "plan_id": "apa-g4", "sleeps": [{"model": "d", "at_most": 1, "sleep_path": "apa", "reason": "apa"}],
    }),
    "floor_clamp": ({"d_min": 2}, {
        "plan_id": "bl-g5", "sleeps": [{"model": "d", "at_most": 0, "sleep_path": "urgent"}],
    }),
    "same_model_twice_400": ({}, {
        "plan_id": "bl-g6",
        "sleeps": [{"model": "d", "at_most": 1, "sleep_path": "urgent"}],
        "wakes": [{"model": "d", "at_least": 2}],
    }),
    "two_wakes_one_model_400": ({}, {
        "plan_id": "bl-g7", "wakes": [{"model": "r", "at_least": 1}, {"model": "r", "at_least": 2}],
    }),
    "unknown_model_400": ({}, {"plan_id": "bl-g8", "wakes": [{"model": "nope", "at_least": 1}]}),
    "noop": ({}, {"plan_id": "bl-g9", "wakes": [{"model": "r", "at_least": 0}]}),
    "unknown_field_ignored": ({}, {
        "plan_id": "bl-g10", "extra": 1, "wakes": [{"model": "r", "at_least": 1, "foo": "bar"}],
    }),
}


def _shape(value):
    """The key structure of a JSON value (lists: the union of their elements' shapes)."""
    if isinstance(value, dict):
        return {key: _shape(item) for key, item in sorted(value.items())}
    if isinstance(value, list):
        merged: dict = {}
        scalars = set()
        for item in value:
            shape = _shape(item)
            if isinstance(shape, dict):
                merged.update(shape)
            else:
                scalars.add(shape)
        return [merged] if merged else sorted(scalars)
    return type(value).__name__


def _http_view(response) -> dict:
    body = response.json()
    view: dict = {"code": response.status_code, "shape": _shape(body)}
    if response.status_code == 400:
        view["detail"] = body.get("detail")
        return view
    body = body.get("detail", body) if response.status_code == 409 else body
    if isinstance(body, dict) and "response" in body:
        body = body["response"]
    view["status"] = body.get("status")
    view["sleeps"] = [
        {key: item.get(key) for key in ("index", "model", "status", "taken", "clamped_by_floor", "awake_before",
                                         "awake_after", "retriable", "sleep_path", "reason")}
        | {"slept": sorted(o.get("serve_id") for o in item.get("outcomes", []) if o.get("status") == "slept"),
           "error": (item.get("error") or {}).get("error")}
        for item in body.get("sleeps", [])
    ]
    view["wakes"] = [
        {key: item.get(key) for key in ("index", "model", "status", "unfilled", "awake_before", "awake_after",
                                         "retriable", "clamped_by_floor", "skipped", "reason")}
        | {"woke": [[w["serve_id"], w["wave"], w["source"], w["hinted"], w["donor_binding_ids"]]
                    for w in item.get("woke", [])],
           "refusals": [r.get("error") for r in item.get("refusals", [])],
           "error": (item.get("error") or {}).get("error")}
        for item in body.get("wakes", [])
    ]
    return view


def _snapshot() -> dict:
    out: dict = {"select": {}, "http": {}}
    for name, case in SELECT_CASES.items():
        case = dict(case)
        d_floor = case.pop("d_floor", None)
        out["select"][name] = _selection_view(_select(d_floor=d_floor, **case))
    for name, (kwargs, request) in HTTP_CASES.items():
        world = World(pods=WORLD_PODS, **kwargs)
        out["http"][name] = _http_view(world.client.post("/v2/plans", json=request))
        if name == "apa_up":
            # A replay of the same absolute target is a no-op.
            out["http"]["apa_up_replay"] = _http_view(world.client.post("/v2/plans", json=request))
    return json.loads(json.dumps(out, sort_keys=True))


def test_baseline_and_apa_plans_behave_as_before_plan_groups():
    current = _snapshot()
    if os.environ.get("TRE_REGEN_GOLDEN") == "1":
        GOLDEN.parent.mkdir(exist_ok=True)
        GOLDEN.write_text(json.dumps(current, indent=1, sort_keys=True) + "\n")
        pytest.skip("golden regenerated")
    assert current == json.loads(GOLDEN.read_text())
