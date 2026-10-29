"""TSS suffix (2026-10-29, design docs/design/20261029-tss-suffix.md): the invariants.

(a) After a change of a model's pod set (awake, not hidden) the model is *settling*: Z is
    unknown and the model takes no action until the samples since the change span
    ``scaling.settle_ms`` (10 s); other models act meanwhile. One event per episode.
(b) The model's EMA restarts at the change and seeds on the first decision-valid suffix.
(c) A suffix with fewer than two samples of a serving pod is unknown, never Z = 0
    (CRITICAL).
(d) A SafeScale probe judges its donor on the remaining pods only: remaining pods that
    stall while the hidden pod drains give no commit.
(e) Scope ``healthy_and_shrink``: a HEALTHY donor gives only to a CRITICAL receiver and
    only through a probe; a HIGH donor relays to a receiver at once; a LOW receiver never
    takes a HEALTHY donor.
(f) A TP block with two probed donors: both commit in one SM plan; one failure rolls
    both back and the receiver is not woken; the block's immediate half sleeps only at
    the commit.
"""
from __future__ import annotations

import asyncio
from dataclasses import replace
from types import SimpleNamespace

import pytest

from tre_common.metrics_schema import MetricsSnapshot, ModelWindowMetrics, PodWindowMetrics
from tre_common.registry import ClusterTopology, ModelSpec, NodeSpec, Registry, SloSpec, TrsParams
from tre_common.window_pods import aggregate_pods
from section_ticks import run_rescue_tick
from tre_controller.loops.action_queue import ActionQueue
from tre_controller.loops.safescale_task import (
    _commands_to_actions,
    _rollback_aborted,
    run_safescale_observation_tick,
)
from tre_controller.loops.tick import compute_model_signal
from tre_controller.planning.planner import ClusterView, ScaleAction, UnhideAction
from tre_controller.planning.safescale import BLOCK_PARTNER_ROLLBACK
from tre_controller.signals.trs import SignalState
from tre_sm.allocator.slots import Binding, Slot

from plan_shim import PlanSM
from test_single_loop_20261028 import CRIT, HEALTHY, HIGH, LOW, TOPOLOGY8, _binding, _tp2
from test_tss_only_20261029 import _Queue, _tp1, _donor_moves

T0 = 2_000_000
W = 20_000
GRID = 2_000
THETA = 100.0
#: Per pod: 100 tok/s, Q = 30 + 2.625 * 20 = 82.5 -> Z = 100 * 20 / 82.5 / 100 = 0.24 (CRITICAL).
BURST = (100.0, 30.0, 20.0)
#: Per pod: Z = 11 * 20 / 2 / 100 = 1.1 (HEALTHY).
CALM = (11.0, 2.0, 0.0)


# ------------------------------------------------------------------- fixtures
def _registry() -> Registry:
    slo = SloSpec(ttft_p95_ms=1200.0, tpot_p95_ms=100.0, e2e_p95_ms=10_000.0)
    trs = TrsParams(
        w_p=0.04, w_d=1.0, lambda_wait=2.625, qmin=1.0, ema_alpha=0.5, theta_m=THETA,
        tau_crit=0.8, tau_low=1.0, tau_high=1.25, qsat=4.0, epsat=0.05, hsat=1, ema_tau_ms=10_000.0,
    )
    specs = [
        ModelSpec(name=name, weights_path="/w", tp_size=1, min_replicas=1, max_replicas=3, vllm_image="img",
                  slo=slo, trs=trs)
        for name in ("m", "o")
    ]
    topology = ClusterTopology(nodes=(NodeSpec(name="node-a", gpus=6, two_gpu_slots=((0, 1), (2, 3), (4, 5))),))
    return Registry(topology, specs)


def _pod(name: str, start: int, end: int, load, *, ticks: tuple[int, ...] | None = None) -> PodWindowMetrics:
    """A pod's window (start, end]: tokens = its rate over the window length (the store
    scales a pod's counter delta to the window), queue = its mean, an instant doc every
    500 ms (``ticks`` overrides)."""
    rate, running, waiting = load
    return PodWindowMetrics(
        pod=name, prompt_tokens=0.0, generation_tokens=rate * (end - start) / 1000.0,
        avg_waiting=waiting, avg_running=running, avg_swapping=0.0, kv_cache_hit_rate=0.0,
        ttft_p95_ms=100.0, tpot_p95_ms=10.0, e2e_p95_ms=1000.0, request_count=10.0,
        instant_ticks_ms=tuple(range(start + 500, end + 1, 500)) if ticks is None else ticks,
    )


def _window(model: str, end: int, pods: dict, *, change_ms: int | None = None, ticks=None) -> ModelWindowMetrics:
    """``model``'s 20 s window ending at ``end``; ``pods`` = name -> (load, serving since
    (None = the whole window)). With ``change_ms`` the store's suffix from the first 2 s
    boundary at or after it (``MetricsStore(suffix_change=...)``); ``ticks`` = name ->
    the pod's instant docs in that suffix."""
    start = end - W

    def agg(lo: int, hi: int, own_ticks=None) -> ModelWindowMetrics:
        per_pod = {}
        for name, (load, since) in pods.items():
            if since is not None and since >= hi:
                continue
            first = lo if since is None else max(lo, since)
            # A pod's tokens are scaled to the (sub)window it is read for.
            per_pod[name] = replace(_pod(name, first, hi, load, ticks=(own_ticks or {}).get(name)),
                                    generation_tokens=load[0] * (hi - lo) / 1000.0)
        return aggregate_pods(model, lo, hi, per_pod)

    window = agg(start, end)
    if change_ms is not None and start < change_ms < end:
        boundary = -(-change_ms // GRID) * GRID
        if boundary < end:
            window = replace(window, suffix_windows=(agg(boundary, end, ticks),))
    return window


def _view(awake: dict[str, tuple[int, ...]], asleep: dict[str, tuple[int, ...]], *, state_ms: int,
          hidden=()) -> ClusterView:
    bindings = [
        Binding(serve_id=pod, model=pod.split("-")[0], slot=Slot("node-a", (gpu,)), awake=True, hidden=pod in hidden)
        for pod, (gpu,) in awake.items()
    ] + [
        Binding(serve_id=pod, model=pod.split("-")[0], slot=Slot("node-a", (gpu,)), awake=False, hidden=False)
        for pod, (gpu,) in asleep.items()
    ]
    return ClusterView(topology=_registry().topology(), bindings=tuple(bindings), fetched_ms=state_ms + 100,
                       state_ms=state_ms)


def _ups(actions) -> list[tuple[str, int]]:
    return [(a.model, a.delta) for a in actions if isinstance(a, ScaleAction) and a.delta > 0]


def _snap(*windows: ModelWindowMetrics) -> MetricsSnapshot:
    return MetricsSnapshot(ts_ms=windows[0].window_end_ms, models={w.model: w for w in windows}, stale=False)


ONE_M = ({"m-0": (0,), "o-3": (3,)}, {"m-1": (1,), "m-2": (2,), "o-4": (4,)})
TWO_M = ({"m-0": (0,), "m-1": (1,), "o-3": (3,)}, {"m-2": (2,), "o-4": (4,)})
WOKEN = T0 + 1_000  # done_ms of the controller's wake of m-1


def _after_wake(state: SignalState, queue: _Queue, at: int, m_load=BURST, o_load=BURST, ticks=None):
    """A rescue tick at ``at``: m-0 and m-1 (awake since the wake) serve ``m_load``."""
    m = _window("m", at, {"m-0": (m_load, None), "m-1": (m_load, WOKEN)}, change_ms=WOKEN, ticks=ticks)
    o = _window("o", at, {"o-3": (o_load, None)})
    return run_rescue_tick(_snap(m, o), queue=queue, registry=_registry(),
                           cluster_view=_view(*TWO_M, state_ms=at - 1_000), signal_state=state)


def _woken(state: SignalState, queue: _Queue):
    """Tick at T0 (m-0 alone, m CRITICAL, o HEALTHY) -> m wakes m-1, done at T0 + 1 s."""
    first = run_rescue_tick(
        _snap(_window("m", T0, {"m-0": (BURST, None)}), _window("o", T0, {"o-3": (CALM, None)})),
        queue=queue, registry=_registry(), cluster_view=_view(*ONE_M, state_ms=T0 - 100), signal_state=state,
    )
    assert _ups(first.actions) == [("m", 1)]
    queue.last["m"] = (WOKEN, "up")
    return first


# ------------------------------------------------------------------- (a) settling
def test_after_its_wake_a_model_settles_for_settle_ms_while_other_models_act():
    state, queue = SignalState(settle_ms=10_000), _Queue()
    _woken(state, queue)

    # +4 s (3 s of post-change samples): m is settling - unknown, no action; o acts.
    early = _after_wake(state, queue, T0 + 5_000)
    assert early.classifications["m"].state.value == "unknown"
    assert early.model_contexts["m"]["settling"] == "settle_span"
    assert _ups(early.actions) == [("o", 1)]
    assert [e for e in early.events if e.startswith("settling_")] == ["settling_start:m:settle_span"]
    # +8 s: still settling, and no second start event for the same episode.
    queue.inflight.add("o")
    mid = _after_wake(state, queue, T0 + 9_000)
    assert mid.classifications["m"].state.value == "unknown" and _ups(mid.actions) == []
    assert not [e for e in mid.events if e.startswith("settling_")]
    # +12 s (10 s observed since the change, from the 2 s boundary after it): valid.
    valid = _after_wake(state, queue, T0 + 13_000)
    assert valid.classifications["m"].state.value == "critical"
    assert _ups(valid.actions) == [("m", 1)]
    assert [e for e in valid.events if e.startswith("settling_")] == ["settling_end:m:valid:evidence_ms=11000"]
    assert valid.model_contexts["m"]["settling"] is None


# ------------------------------------------------------------------- (b) EMA restart
def test_the_ema_restarts_at_the_change_and_seeds_on_the_first_valid_suffix():
    spec, state = _registry().model("m"), SignalState(settle_ms=10_000)
    state.note_pod_sets({"m": {"m-0"}}, view_ms=T0 - 30_000)
    for at in range(T0 - 30_000, T0 + 1, 5_000):  # a long CRITICAL history on m-0 alone
        old = compute_model_signal("m", _window("m", at, {"m-0": (BURST, None)}), spec, signal_source="zm",
                                   signal_state=state)
    assert old.result.Z_m == pytest.approx(0.242, abs=1e-3)
    state.note_pod_sets({"m": {"m-0", "m-1"}}, view_ms=T0 + 4_000, done_ms={"m": WOKEN})
    assert state.change_ms("m") == WOKEN

    pods = {"m-0": (CALM, None), "m-1": (CALM, WOKEN)}
    settling = compute_model_signal("m", _window("m", T0 + 5_000, pods, change_ms=WOKEN), spec,
                                    signal_source="zm", signal_state=state)
    assert settling.settling == "settle_span" and settling.result.Z_m is None
    first = compute_model_signal("m", _window("m", T0 + 13_000, pods, change_ms=WOKEN), spec,
                                 signal_source="zm", signal_state=state)
    # Seeded on the suffix value alone (1.1), not blended with the pre-change 0.24.
    assert first.settling is None
    assert first.result.Z_m == pytest.approx(1.1, abs=1e-6) == pytest.approx(first.result.TRS_raw / THETA)


# ------------------------------------------------------------------- (c) <2 samples
def test_a_suffix_with_one_sample_of_a_serving_pod_is_unknown_not_critical():
    state, queue = SignalState(settle_ms=10_000), _Queue()
    _woken(state, queue)
    # m-1 stalls with a queue and zero tokens, and the gateway read it once since the
    # change: Z would be ~0 (CRITICAL), but one sample is no evidence.
    stalled = (0.0, 30.0, 20.0)
    m = _window("m", T0 + 13_000, {"m-0": (stalled, None), "m-1": (stalled, WOKEN)}, change_ms=WOKEN,
                ticks={"m-1": (T0 + 12_500,)})
    o = _window("o", T0 + 13_000, {"o-3": (CALM, None)})
    result = run_rescue_tick(_snap(m, o), queue=queue, registry=_registry(),
                             cluster_view=_view(*TWO_M, state_ms=T0 + 12_000), signal_state=state)
    assert result.model_contexts["m"]["settling"] == "samples"
    assert result.classifications["m"].state.value == "unknown"
    assert _ups(result.actions) == []


# ------------------------------------------------------------------- (d) remaining pods
class _Probe:
    model = "m"
    pods = ("m-1",)


class _Observer:
    def __init__(self) -> None:
        self.observations = []

    def active_probes(self):
        return (_Probe(),)

    def tail_start_ms(self, _model):
        return T0 - W  # the hide precedes the window: the whole window is the tail

    def observe(self, model, observation, *, now_ms):
        self.observations.append(observation)
        return SimpleNamespace(reason="observing", commands=(), status="probing")


@pytest.mark.parametrize("view_shows_the_hide", [True, False])
def test_a_probe_is_judged_on_its_remaining_pods_only(view_shows_the_hide):
    # Regression: m-1 (hidden) still drains its requests at full speed while m-0, the
    # remaining pod, stalls with a queue. Counted, m-1's tokens give Z = 10000 / 84.5 /
    # 100 = 1.18 >= tau_low (a commit); on the remaining pod alone Z = 0, so the commit
    # gate fails (z_below_tau_low) - also when the view does not show the hide yet.
    m = _window("m", T0, {"m-0": ((0.0, 30.0, 20.0), None), "m-1": ((500.0, 2.0, 0.0), None)})
    view = _view({"m-0": (0,), "m-1": (1,)}, {}, state_ms=T0 - 500,
                 hidden=("m-1",) if view_shows_the_hide else ())
    observer = _Observer()
    run_safescale_observation_tick(_snap(m), queue=_Queue(), registry=_registry(), safescale=observer,
                                   cluster_view=view)
    [obs] = observer.observations
    # The commit gate's tail: Z of the remaining pod, under traffic -> below tau_low.
    assert obs.tail_z_m == pytest.approx(0.0) and obs.tail_has_traffic


# ------------------------------------------------------------------- (e) donor scope
def test_scope_healthy_and_shrink_a_healthy_donor_gives_only_to_critical_through_a_probe():
    critical = _tp1((*CRIT, 0.3), (*HEALTHY, 1.1), relay_high_donors=True)
    moves = [a for a in critical.actions if isinstance(a, ScaleAction) and a.model == "d" and a.delta < 0]
    assert moves and all(a.requires_safescale and a.receiver == "r" for a in moves)
    # A LOW receiver never takes a HEALTHY donor, probe or not.
    low = _tp1((*LOW, 0.9), (*HEALTHY, 1.1), relay_high_donors=True)
    assert _donor_moves(low) == [] and not low.probe_upscale_plans


@pytest.mark.parametrize("receiver", [(*CRIT, 0.3), (*LOW, 0.9)], ids=["critical", "low"])
def test_scope_healthy_and_shrink_a_high_donor_relays_to_a_receiver_at_once(receiver):
    plan = _tp1(receiver, (*HIGH, 1.4), relay_high_donors=True)
    [relay] = plan.actions  # the donor pod's sleep and the receiver's wake, one plan, no probe
    assert relay.sleeps == (("d", "d-1"),) and relay.probe is None and relay.receiver_pods == ("r-1",)
    assert relay.reason.endswith("_relay") and not plan.probe_upscale_plans
    # Scope all: the same donor is a probe for the receiver.
    probed = _tp1(receiver, (*HIGH, 1.4))
    assert probed.probe_upscale_plans == {"d": {"r": 1}}


# ------------------------------------------------------------------- (f) joint block
class _AcceptingQueue(_Queue):
    def submit(self, actions):
        self.submitted.extend(actions)
        return SimpleNamespace(accepted=len(tuple(actions)))


def _joint():
    donors = (_binding("s7-0", "s7", (0,), True), _binding("s8-1", "s8", (1,), True))
    plans, _events, machine = _tp2(*donors, *(_binding(f"x-{gpu}", "x", (gpu,), True) for gpu in (2, 3)),
                                   classes={"s7": (*HIGH, 1.4), "s8": (*HIGH, 1.5)})
    view = ClusterView(TOPOLOGY8, (
        _binding("tp2-4", "tp2", (4, 5), True), _binding("tp2-0", "tp2", (0, 1), False),
        _binding("tp2-2", "tp2", (2, 3), False), *(_binding(f"x-{gpu}", "x", (gpu,), True) for gpu in (2, 3, 6, 7)),
        *donors,
    ))
    probes = [machine.active_probe("s7"), machine.active_probe("s8")]
    leader = next(p for p in probes if p.pending_upscales)
    member = next(p for p in probes if not p.pending_upscales)
    return plans, machine, view, leader, member


def test_a_joint_block_commits_both_donors_and_the_receiver_in_one_plan():
    _plans, machine, view, leader, member = _joint()
    # The first probe to pass waits for its partner; nothing sleeps yet.
    waiting = machine._commit(member, reason="formal_commit_gate_passed")
    assert (waiting.status, waiting.reason, waiting.commands) == ("probing", "block_waiting", ())
    decision = machine._commit(machine.active_probe(leader.model), reason="formal_commit_gate_passed")
    assert decision.status == "commit"
    [commit] = _commands_to_actions(decision.commands, cluster_view=view)
    assert commit.companions == ((member.model, member.pods[0]),)

    sm = PlanSM({"s7": 1, "s8": 1, "tp2": 1, "x": 4})
    queue = ActionQueue(sm, now_ms=lambda: 5_000)
    queue.submit((commit,))
    asyncio.run(queue.drain_once())
    [plan] = sm.plans  # one SM plan: both donor sleeps, then the receiver wake
    assert sorted((i["model"], tuple(i.get("pods") or ())) for i in plan["sleeps"]) == [
        ("s7", ("s7-0",)), ("s8", ("s8-1",))]
    assert [(i["model"], i.get("pods")) for i in plan["wakes"]] == [("tp2", ["tp2-0"])]

    assert machine.resolve(leader.model, status="commit", reason="done", now_ms=6_000)
    assert machine.active_probe("s7") is None and machine.active_probe("s8") is None


@pytest.mark.parametrize("failing", ["leader", "member"])
def test_one_failed_probe_of_a_joint_block_rolls_both_back_and_wakes_no_receiver(failing):
    _plans, machine, _view8, leader, member = _joint()
    other = member if failing == "leader" else leader
    machine._commit(other, reason="formal_commit_gate_passed")  # the partner had passed
    failed = leader if failing == "leader" else member
    rollback = machine._rollback(machine.active_probe(failed.model), reason="immediate_slo_rollback")
    assert [c.kind for c in rollback.commands] == ["unhide"]
    machine.resolve(failed.model, status="rollback", reason="immediate_slo_rollback", now_ms=5_000)
    assert machine.active_probe(other.model).abort_reason == BLOCK_PARTNER_ROLLBACK

    queue = _AcceptingQueue()
    events: list[str] = []
    out = _rollback_aborted(queue, machine, now_ms=5_000, events=events)
    assert [(type(a), a.model) for a in out] == [(UnhideAction, other.model)]
    assert not [a for a in queue.submitted if isinstance(a, ScaleAction) and a.delta > 0]
    assert machine.active_probe(other.model) is None and machine.active_probe("tp2") is None


def test_the_immediate_half_of_a_probed_block_sleeps_at_the_commit_not_at_the_start():
    plans, _events, machine = _tp2(_binding("h-1", "h", (1,), True), _binding("i-0", "i", (0,), True),
                                   *(_binding(f"x-{gpu}", "x", (gpu,), True) for gpu in (2, 3)),
                                   classes={"h": (*HIGH, 1.4)})
    assert not [item for items in plans for item in items if item[0] == "sleep"]  # nothing sleeps at start
    decision = machine._commit(machine.active_probe("h"), reason="formal_commit_gate_passed")
    kinds = [(c.kind, c.model, c.pods) for c in decision.commands]
    assert kinds[:2] == [("scale_down", "h", ("h-1",)), ("companion_sleep", "i", ("i-0",))]
    assert kinds[2][:2] == ("scale_up", "tp2")
