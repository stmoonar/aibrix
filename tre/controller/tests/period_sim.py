"""Closed-loop scenario renderer for the period-invariance test (round 4,
docs/design/20261025-instant-sampling.md "Evidence, not ticks").

One scenario - the demand of each model over time, gateway stalls - is rendered into
gateway docs at a given evidence period P (instant docs every 0.5 s, histogram docs
every P, ``scraped_ms`` 50 ms before the write, the cadence hash with ``last_round_ms``)
and run through the real controller path: MetricsStore -> PhaseAlignedSampler -> the
rescue (5 s) / fairness (10 s) planner ticks (``loops/replay.py``, one SignalState and
PaperStateCache per run) -> SafeScale observation ticks (2 s). A mock SM applies every
action at once (relay, wake, sleep, hide, unhide), so the load follows the decisions.

The engine model is deliberately simple: a pod's concurrency is the model's demand
split over its routable pods; up to 16 run, the rest wait; per-request decode speed
falls with the batch; TTFT grows with the queue.
"""
from __future__ import annotations

import asyncio
import bisect
import json
from types import SimpleNamespace
from dataclasses import dataclass, field, replace

from tre_common.registry import ClusterTopology, ModelSpec, NodeSpec, Registry, SloSpec, TrsParams
from tre_controller.config import ControllerConfig, SafeScaleConfig
from tre_controller.loops.metrics_task import PhaseAlignedSampler, SnapshotBox, default_phase_offset_ms
from tre_controller.loops.replay import TickReplayStep, run_tick_replay
from tre_controller.loops.safescale_task import run_safescale_observation_tick
from tre_controller.loops.tick import PaperStateCache
from tre_controller.planning.planner import ClusterView, HideAction, ScaleAction, TransferIntent, UnhideAction
from tre_controller.planning.safescale import SafeScaleStateMachine
from tre_controller.signals.saturation import SaturationRescueConfig, SaturationTracker
from tre_controller.signals.trs import BreakpointWindowConfig, SignalState
from tre_controller.store.evidence_grid import EvidenceGrid
from tre_controller.store.metrics_store import MetricsStore
from tre_sm.allocator.slots import Binding, Slot

INSTANT = 500
WINDOW = 30_000
T0 = 1_000_000_000_000  # epoch-aligned: every period divides it
MAX_RUNNING = 16
#: Tokens per request: d's long answers take ~20 s, so an onset on d completes nothing
#: for a while (the TSS numerator is 0: the onset saturation rescue decides).
OUTPUT_TOKENS = {"r": 200.0, "d": 2000.0}
PROMPT_TOKENS = 500.0
TTFT_BOUNDS = (0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0)
TPOT_BOUNDS = (0.005, 0.01, 0.025, 0.05, 0.1, 0.25)
E2E_BOUNDS = (1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0)
MODELS = ("r", "d")


def registry() -> Registry:
    specs = [
        ModelSpec(
            name=name, weights_path="/w", tp_size=1, min_replicas=1, max_replicas=4, vllm_image="img",
            slo=SloSpec(ttft_p95_ms=1000.0, tpot_p95_ms=100.0, e2e_p95_ms=20_000.0),
            trs=TrsParams(w_p=0.04, w_d=1.0, lambda_wait=2.625, qmin=1.0, ema_alpha=1.0, theta_m=3000.0,
                          tau_crit=0.8, tau_low=1.0, tau_high=1.25, qsat=4.0, epsat=0.05, hsat=1, ema_tau_ms=10_000.0),
        )
        for name in MODELS
    ]
    return Registry(ClusterTopology(nodes=(NodeSpec(name="n", gpus=4, two_gpu_slots=((0, 1), (2, 3))),)), specs)


@dataclass(frozen=True)
class Scenario:
    """``demand[model]`` = ((from_s, concurrency), ...); ``awake`` = the model awake on
    each GPU 0-3 at the start; ``stalls`` = ((from_s, to_s), ...) gateway outages."""

    demand: dict[str, tuple[tuple[float, float], ...]]
    awake: tuple[str, str, str, str]
    duration_s: float
    stalls: tuple[tuple[float, float], ...] = ()

    def concurrency(self, model: str, t_s: float) -> float:
        steps = self.demand.get(model, ())
        level = steps[0][1] if steps else 0.0  # the first level also before 0
        for start, value in steps:
            if t_s >= start:
                level = value
        return level


def _bucket_add(buckets: list[float], bounds: tuple[float, ...], value: float, n: float) -> None:
    for i, bound in enumerate(bounds):
        if value <= bound:
            buckets[i] += n
            return
    buckets[-1] += n


@dataclass
class _Pod:
    name: str
    model: str
    gpu: int
    awake: bool
    hidden: bool = False
    count: float = 0.0
    gen: float = 0.0
    ttft_sum: float = 0.0
    busy_since: int | None = None
    e2e_sum: float = 0.0
    ttft: list = field(default_factory=lambda: [0.0] * (len(TTFT_BOUNDS) + 1))
    tpot: list = field(default_factory=lambda: [0.0] * (len(TPOT_BOUNDS) + 1))
    e2e: list = field(default_factory=lambda: [0.0] * (len(E2E_BOUNDS) + 1))
    gauges: tuple = (0.0, 0.0, 0.0)


class FastRedis:
    """The Redis subset the store and the sampler read, with sorted sorted-sets."""

    def __init__(self) -> None:
        self.sets: dict = {}
        self.hashes: dict = {}
        self._scores: dict = {}
        self._members: dict = {}

    def sadd(self, key, *values):
        self.sets.setdefault(key, set()).update(values)

    def smembers(self, key):
        return set(self.sets.get(key, set()))

    def hgetall(self, key):
        return dict(self.hashes.get(key, {}))

    def zadd(self, key, doc, score):
        scores, members = self._scores.setdefault(key, []), self._members.setdefault(key, [])
        i = bisect.bisect_right(scores, float(score))
        scores.insert(i, float(score))
        members.insert(i, json.dumps(doc))

    def zrangebyscore(self, key, lo, hi):
        scores, members = self._scores.get(key, []), self._members.get(key, [])
        return members[bisect.bisect_left(scores, float(lo)):bisect.bisect_right(scores, float(hi))]

    def zrevrangebyscore(self, key, hi, lo, start=0, num=None):
        found = list(reversed(self.zrangebyscore(key, lo, hi)))
        return found[start:] if num is None else found[start:start + num]


class _Queue:
    def __init__(self, sim: "Sim") -> None:
        self._sim = sim

    def inflight_models(self) -> set:
        return set()

    def routable_changes(self) -> dict:
        """When the last call that could change each model's routable count returned
        (``ActionQueue.routable_changes``): the O1 breakpoint hint."""
        return dict(self._sim.changes)

    def submit(self, actions):
        actions = tuple(actions)
        for action in actions:
            self._sim.apply(action)
        return SimpleNamespace(accepted=len(actions), submitted=len(actions))


@dataclass
class Run:
    """What one run did: (sim time ms, event) - the planner loops' events prefixed
    ``rescue|`` / ``fairness|``, the SafeScale observation events, the mock SM's applied
    actions, ``hold:<model>:<O1 hold reason>`` per planner tick, ``stale`` - and the
    probe verdicts in order."""

    events: list = field(default_factory=list)
    verdicts: list = field(default_factory=list)


class Sim:
    def __init__(self, scenario: Scenario, period_ms: int) -> None:
        self.scenario, self.period = scenario, int(period_ms)
        self.registry = registry()
        self.redis = FastRedis()
        self.pods = [_Pod(f"{m}-{g}", m, g, awake=scenario.awake[g] == m) for g in range(4) for m in MODELS]
        for pod in self.pods:
            self.redis.sadd(f"tre:v2:pods:{pod.model}", f"default/{pod.name}")
        self.now = T0
        self.rendered = T0 - WINDOW - 2 * self.period
        self.run = Run()
        self.changes: dict = {}  # model -> (ms, routable direction) of the last applied action
        cfg = ControllerConfig.from_env({})
        self.rescue_ms = int(cfg.rescue_interval_s * 1000)
        self.fairness_ms = int(cfg.fairness_interval_s * 1000)
        self.poll_ms = int(cfg.safescale.probe_poll_seconds * 1000)
        self.signal_state = SignalState(
            warmup_ms=cfg.signal_warmup_ms,
            breakpoint=BreakpointWindowConfig.from_registry(self.registry),
            saturation=SaturationTracker(SaturationRescueConfig.from_registry(self.registry)),
        )
        self.safescale = SafeScaleStateMachine(
            config=SafeScaleConfig(ttft_p95_slo_ms=1000.0, tpot_p95_slo_ms=100.0), grid=EvidenceGrid(self.period),
        )
        self.store = MetricsStore(
            self.redis, self.registry, instant_sample_interval_ms=INSTANT, evidence_period_ms=self.period,
            expected_cadence=(INSTANT, self.period), cadence_enforce=True,
            min_sample_ratio=float(self.registry.scaling().sample_min_ratio or 0.0),
            suffix_period_ms=self.period,
            suffix_breakpoint=lambda m: self.signal_state.breakpoint_ms(m) if self.signal_state.o1_active else None,
            suffix_tail=self.safescale.tail_span_ms,
        )
        self.box = SnapshotBox()
        self.queue = _Queue(self)
        self.papers = {loop: PaperStateCache(max_stale_ms=cfg.paper_stale_max_ms) for loop in ("rescue", "fairness")}

        async def fetch(start, end):
            return self.store.read_snapshot(start, end, use_cache=False, start_exclusive=True)

        self.sampler = PhaseAlignedSampler(
            self.store, self.box, window_ms=WINDOW, period_ms=self.period,
            offset_ms=default_phase_offset_ms(self.period), clock_ms=lambda: self.now,
            sleep=self._sleep, fetch=fetch,
        )

    # ---------------------------------------------------------------- the world

    def _stalled(self, ts: int) -> bool:
        t = (ts - T0) / 1000.0
        return any(a <= t < b for a, b in self.scenario.stalls)

    def _render(self, ts: int) -> None:
        dt = INSTANT / 1000.0
        t_s = (ts - T0) / 1000.0
        for model in MODELS:
            routable = [p for p in self.pods if p.model == model and p.awake and not p.hidden]
            per_pod = self.scenario.concurrency(model, t_s) / max(1, len(routable))
            for pod in (p for p in self.pods if p.model == model):
                c = per_pod if pod in routable else 0.0
                running, waiting = min(c, MAX_RUNNING), max(0.0, c - MAX_RUNNING)
                speed = 300.0 / (1.0 + running / 8.0)  # decode tokens/s per request
                out = OUTPUT_TOKENS[model]
                if c <= 0.0:
                    pod.busy_since = None
                elif pod.busy_since is None:
                    pod.busy_since = ts
                # The first requests of a busy period complete only after one e2e.
                warm = pod.busy_since is not None and ts - pod.busy_since >= 1000.0 * out / speed
                done = running * speed / out * dt if warm else 0.0
                pod.count += done
                pod.gen += done * out
                ttft = 0.08 + 0.25 * waiting
                _bucket_add(pod.ttft, TTFT_BOUNDS, ttft, done)
                pod.ttft_sum += ttft * done
                pod.e2e_sum += (ttft + out / speed) * done
                _bucket_add(pod.tpot, TPOT_BOUNDS, 1.0 / speed, done * out)
                _bucket_add(pod.e2e, E2E_BOUNDS, ttft + out / speed, done)
                pod.gauges = (running, waiting, min(1.0, (running + 4.0 * waiting) / 18.0))
        if self._stalled(ts):
            return
        for pod in self.pods:
            base = {"timestamp": ts, "written_ms": ts + 5, "pod_name": pod.name, "scraped_ms": ts - 50}
            running, waiting, kv = pod.gauges
            self.redis.zadd(f"tre:v2:inst:default/{pod.name}", {**base, "model_metrics": {
                f"{pod.model}/num_requests_running": running, f"{pod.model}/num_requests_waiting": waiting,
                f"{pod.model}/gpu_cache_usage_perc": kv,
            }}, ts)
            if ts % self.period == 0:
                self.redis.zadd(f"tre:v2:hist:default/{pod.name}", {**base, "model_histogram_metrics": {
                    f"{pod.model}/request_prompt_tokens": _hist(pod.count, pod.count * PROMPT_TOKENS, (1e9,)),
                    f"{pod.model}/request_generation_tokens": _hist(pod.count, pod.gen, (1e9,)),
                    f"{pod.model}/time_to_first_token_seconds": _hist(pod.count, pod.ttft_sum, TTFT_BOUNDS, pod.ttft),
                    f"{pod.model}/time_per_output_token_seconds": _hist(pod.gen, pod.gen / 150.0, TPOT_BOUNDS, pod.tpot),
                    f"{pod.model}/e2e_request_latency_seconds": _hist(pod.count, pod.e2e_sum, E2E_BOUNDS, pod.e2e),
                }}, ts)
        self.redis.hashes["tre:v2:gateway:cadence"] = {
            "instant_ms": str(INSTANT), "evidence_ms": str(self.period), "written_ms": str(ts + 5),
            "last_round_ms": str(ts), "version": "sim",
        }

    def view(self) -> ClusterView:
        bindings = tuple(
            Binding(serve_id=p.name, model=p.model, slot=Slot("n", (p.gpu,)), awake=p.awake, hidden=p.hidden)
            for p in self.pods
        )
        return replace(ClusterView(topology=self.registry.topology(), bindings=bindings),
                       state_ms=self.now, fetched_ms=self.now)

    def _pod(self, name: str) -> _Pod:
        return next(p for p in self.pods if p.name == name)

    def _wake(self, model: str, gpu: int | None = None) -> None:
        busy = {p.gpu for p in self.pods if p.awake}
        free = [p for p in self.pods if p.model == model and not p.awake and (gpu is None and p.gpu not in busy
                                                                            or p.gpu == gpu)]
        if free:
            free[0].awake = True

    def apply(self, action) -> None:
        t = self.now
        if isinstance(action, TransferIntent):
            self.changes[action.donor_model] = (t, -1)
            self.changes[action.receiver_model] = (t, 1)
        elif isinstance(action, (ScaleAction, HideAction, UnhideAction)):
            up = isinstance(action, UnhideAction) or (isinstance(action, ScaleAction) and action.delta > 0)
            self.changes[action.model] = (t, 1 if up else -1)
        if isinstance(action, TransferIntent):
            donors = sorted((p for p in self.pods if p.model == action.donor_model and p.awake and not p.hidden),
                            key=lambda p: -p.gpu)[: int(action.count)]
            for pod in donors:
                pod.awake = False
                self._wake(action.receiver_model, pod.gpu)
            self.run.events.append((t, f"relay:{action.donor_model}->{action.receiver_model}:{len(donors)}"))
        elif isinstance(action, ScaleAction) and action.delta > 0:
            for _ in range(action.delta):
                self._wake(action.model)
            self.run.events.append((t, f"scale_up:{action.model}:{action.delta}"))
        elif isinstance(action, ScaleAction):
            pods = [self._pod(n) for n in action.pods] or sorted(
                (p for p in self.pods if p.model == action.model and p.awake), key=lambda p: -p.gpu)[: -action.delta]
            for pod in pods:
                pod.awake, pod.hidden = False, False
            self.run.events.append((t, f"scale_down:{action.model}:{len(pods)}"))
        elif isinstance(action, HideAction):
            for name in action.pods:
                self._pod(name).hidden = True
            self.run.events.append((t, f"hide:{action.model}"))
        elif isinstance(action, UnhideAction):
            for name in set(action.pods) - set(action.keep_hidden):
                self._pod(name).hidden = False
            self.run.events.append((t, f"unhide:{action.model}"))

    # ---------------------------------------------------------------- the controller

    def _tick(self, t: int) -> None:
        snapshot = self.box.get()
        if snapshot is None:
            return
        if snapshot.stale:
            if not any(event == "stale" for _, event in self.run.events):
                self.run.events.append((t, "stale"))
            return
        if (t - T0) % self.poll_ms == 0:
            result = run_safescale_observation_tick(
                snapshot, queue=self.queue, registry=self.registry, safescale=self.safescale,
                signal_state=self.signal_state, cluster_view=self.view(),
            )
            for event in result.events:
                self.run.events.append((t, event))
                if event.startswith(("safescale_formal_commit_gate", "safescale_slo_violation",
                                     "safescale_early_commit")):
                    self.run.verdicts.append(event)
        for loop, due in (("rescue", self.rescue_ms), ("fairness", self.fairness_ms)):
            if (t - T0) % due:
                continue
            step = TickReplayStep(snapshot=snapshot, rescue_due=loop == "rescue", fairness_due=loop == "fairness",
                                  cluster_view=self.view())
            replay = run_tick_replay([step], registry=self.registry, safescale=self.safescale,
                                     signal_state=self.signal_state, paper_state_cache=self.papers[loop],
                                     queue=self.queue)
            for event in replay.events:
                self.run.events.append((t, f"{loop}|{event}"))
            for result in replay.results:
                for model, context in (result.model_contexts or {}).items():
                    # O1: the reason the model's window decides nothing (None = it decides).
                    self.run.events.append((t, f"hold:{model}:{context.get('signal_hold_reason')}"))

    def _advance(self, until: int) -> None:
        while self.now < until:
            self.now += 250
            if self.now % INSTANT == 0 and self.now > self.rendered:
                self._render(self.now)
                self.rendered = self.now
            if self.now % 1000 == 250:
                self._tick(self.now - 250)

    async def _sleep(self, seconds: float) -> None:
        self._advance(self.now + int(round(seconds * 1000)))

    def go(self) -> Run:
        for ts in range(self.rendered + INSTANT, T0 + 1, INSTANT):
            self._render(ts)
        self.rendered = T0
        end = T0 + int(self.scenario.duration_s * 1000)

        async def main():
            while self.now < end:
                await self.sampler.run_once()

        asyncio.run(main())
        return self.run


def _hist(count: float, sum_: float, bounds: tuple[float, ...], buckets: list | None = None) -> dict:
    cumulative, running = {}, 0.0
    values = buckets if buckets is not None else [count] + [0.0] * len(bounds)
    for bound, n in zip((*bounds, float("inf")), values):
        running += n
        cumulative["+Inf" if bound == float("inf") else repr(bound)] = running
    return {"count": count, "sum": sum_, "buckets": cumulative}
