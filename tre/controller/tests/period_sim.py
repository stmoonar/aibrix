"""Closed-loop scenario renderer for the period-invariance test (round 4,
docs/design/20261025-instant-sampling.md "Evidence, not ticks").

One scenario - the demand of each model over time, gateway stalls, a gateway clock
offset - is rendered into gateway docs at a given evidence period P (instant docs every
0.5 s, histogram docs every P, ``scraped_ms`` 50 ms before the write, the cadence hash
with ``last_hist_ms``; every gateway stamp on the gateway clock = controller clock +
``gateway_offset_ms``) and run through the real controller path: MetricsStore ->
PhaseAlignedSampler -> the single decision loop (``loops/decision_task.py``: on each
published snapshot, the rescue section after 5 s and the rebalance section after 10 s
of observation time; one SignalState and PaperStateCache per run) -> SafeScale
observation ticks (2 s; with ``direct``, the direct evidence path: the real
DirectEvidenceCollector scraping the sim pods, early commit on). A mock SM applies every
action at once (relay, wake, sleep, hide, unhide), so the load follows the decisions.

The engine model is deliberately simple: a pod's concurrency is the model's demand
split over its routable pods; up to 16 run, the rest wait; per-request decode speed
falls with the batch; TTFT grows with the queue. Round 5 (M3): requests complete in
whole numbers (Poisson per 0.5 s step, seeded by pod and time: the same world at every
period until the decisions differ).
"""
from __future__ import annotations

import asyncio
import bisect
import json
import math
import random
from types import SimpleNamespace
from dataclasses import dataclass, field, replace

from tre_common.registry import ClusterTopology, ModelSpec, NodeSpec, Registry, SloSpec, TrsParams
from tre_controller.config import ControllerConfig, SafeScaleConfig
from tre_controller.loops.metrics_task import PhaseAlignedSampler, SnapshotBox, default_phase_offset_ms
from tre_common.persistence import SPAN_TOLERANCE_MS
from tre_controller.loops.action_queue import PlanBatch
from tre_controller.loops.decision_task import DecisionLoop
from tre_controller.loops.replay import _active_probe_models
from tre_controller.loops.safescale_task import run_safescale_observation_tick
from tre_controller.planning.planner import ClusterView, HideAction, ScaleAction, TransferIntent, UnhideAction
from tre_controller.planning.safescale import SafeScaleStateMachine
from tre_controller.planning.safescale_direct import DirectEvidenceCollector, Hist, PodScrape
from tre_controller.planning.safescale_evidence import HideAnchor
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
    #: Gateway clock - controller clock (a multiple of the 0.5 s instant step).
    gateway_offset_ms: int = 0
    #: SafeScale on the direct evidence path (early commit on).
    direct: bool = False
    #: O1 (``scaling.breakpoint_window``) on.
    o1: bool = True
    #: SafeScale's TPOT p95 threshold (config mode).
    tpot_slo_ms: float = 100.0

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
        self.poll_ms = int(cfg.safescale.probe_poll_seconds * 1000)
        self.offset = int(scenario.gateway_offset_ms)
        breakpoint = BreakpointWindowConfig.from_registry(self.registry)
        self.signal_state = SignalState(
            warmup_ms=cfg.signal_warmup_ms,
            breakpoint=breakpoint if scenario.o1 else replace(breakpoint, enabled=False),
            saturation=SaturationTracker(SaturationRescueConfig.from_registry(self.registry)),
        )
        slo = dict(ttft_p95_slo_ms=1000.0, tpot_p95_slo_ms=float(scenario.tpot_slo_ms))
        self.collector = None
        if scenario.direct:
            self.safescale = SafeScaleStateMachine(
                config=SafeScaleConfig(**slo, evidence_source="direct", early_commit=True, baseline_delay_ms=0.0,
                                       min_window_ms=60_000.0, window_ceiling_ms=90_000.0),
                grid=EvidenceGrid(self.period), evidence=_SimEvidence(self), wall_clock_ms=lambda: self.now,
            )
            self.collector = DirectEvidenceCollector(
                self.safescale, _SimScraper(self),
                lambda model, exclude: {p.name: p.name for p in self.pods
                                        if p.model == model and p.awake and not p.hidden and p.name not in exclude},
                urls=lambda model, pods: {name: name for name in pods},
                poll_ms=float(self.poll_ms), clock_ms=lambda: self.now, hidden_scrape=True,
                gateway_inflight=lambda pods: 0.0,  # the mock SM drains a hidden pod at once
            )
        else:
            self.safescale = SafeScaleStateMachine(config=SafeScaleConfig(**slo), grid=EvidenceGrid(self.period))
        self.store = MetricsStore(
            self.redis, self.registry, instant_sample_interval_ms=INSTANT, evidence_period_ms=self.period,
            expected_cadence=(INSTANT, self.period), cadence_enforce=True,
            min_sample_ratio=float(self.registry.scaling().sample_min_ratio or 0.0),
            suffix_period_ms=self.period,
            suffix_breakpoint=lambda m: self.signal_state.breakpoint_ms(m) if self.signal_state.o1_active else None,
            suffix_tail=self.safescale.tail_start_ms,
        )
        self.box = SnapshotBox()
        self.queue = _Queue(self)
        self.decisions = DecisionLoop(
            queue=self.queue, registry=self.registry, cfg=cfg,
            cluster_view_box=SimpleNamespace(get=self.view),
            active_probe_models=lambda: set(_active_probe_models(self.safescale)),
            safescale=self.safescale, signal_state=self.signal_state,
            tolerance_ms=min(SPAN_TOLERANCE_MS, self.period // 2),
        )
        self._seen_version = 0

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
                rng = random.Random(f"{pod.name}:{ts}")
                done = float(_poisson(rng, running * speed / out * dt)) if warm else 0.0
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
        ts = ts + self.offset  # every gateway stamp is on the gateway clock
        for pod in self.pods:
            base = {"timestamp": ts, "written_ms": ts + 5, "pod_name": pod.name, "scraped_ms": ts - 50}
            running, waiting, kv = pod.gauges
            self.redis.zadd(f"tre:v2:inst:default/{pod.name}", {**base, "model_metrics": {
                f"{pod.model}/num_requests_running": running, f"{pod.model}/num_requests_waiting": waiting,
                f"{pod.model}/gpu_cache_usage_perc": kv,
            }}, ts)
            if ts % self.period == 0:
                self.redis.zadd(f"tre:v2:hist:default/{pod.name}", {**base, "model_counter_metrics": {
                    f"{pod.model}/prompt_tokens_total": pod.count * PROMPT_TOKENS,
                    f"{pod.model}/generation_tokens_total": pod.gen,
                }, "model_histogram_metrics": {
                    f"{pod.model}/request_prompt_tokens": _hist(pod.count, pod.count * PROMPT_TOKENS, (1e9,)),
                    f"{pod.model}/request_generation_tokens": _hist(pod.count, pod.gen, (1e9,)),
                    f"{pod.model}/time_to_first_token_seconds": _hist(pod.count, pod.ttft_sum, TTFT_BOUNDS, pod.ttft),
                    f"{pod.model}/time_per_output_token_seconds": _hist(pod.gen, pod.gen / 150.0, TPOT_BOUNDS, pod.tpot),
                    f"{pod.model}/e2e_request_latency_seconds": _hist(pod.count, pod.e2e_sum, E2E_BOUNDS, pod.e2e),
                }}, ts)
        cadence = self.redis.hashes.setdefault("tre:v2:gateway:cadence", {})
        cadence.update({"instant_ms": str(INSTANT), "evidence_ms": str(self.period), "written_ms": str(ts + 5),
                        "version": "sim"})
        if ts % self.period == 0:
            cadence["last_hist_ms"] = str(ts)  # the boundary's histogram docs are written

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
        if isinstance(action, PlanBatch):  # a same-slot release: its items in order
            for member in action.members:
                self.apply(member)
            return
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
            if self.collector is not None:
                # The SM confirmed the hide (ActionQueue on_hide_done): anchor + baseline.
                self.collector.on_hide_done(action.model, tuple(action.pods))
        elif isinstance(action, UnhideAction):
            for name in set(action.pods) - set(action.keep_hidden):
                self._pod(name).hidden = False
            self.run.events.append((t, f"unhide:{action.model}"))

    # ---------------------------------------------------------------- the controller

    async def _tick(self, t: int) -> None:
        snapshot = self.box.get()
        if snapshot is None:
            return
        if snapshot.stale:
            if not any(event == "stale" for _, event in self.run.events):
                self.run.events.append((t, "stale"))
            return
        if (t - T0) % self.poll_ms == 0:
            polls = await self.collector.poll() if self.collector is not None else None
            result = run_safescale_observation_tick(
                snapshot, queue=self.queue, registry=self.registry, safescale=self.safescale,
                signal_state=self.signal_state, cluster_view=self.view(), direct_polls=polls,
            )
            for event in result.events:
                self.run.events.append((t, event))
                if event.startswith(("safescale_formal_commit_gate", "safescale_slo_violation",
                                     "safescale_early_commit")):
                    self.run.verdicts.append(event)
        if self.box.version == self._seen_version:
            return
        self._seen_version = self.box.version  # a new snapshot was published
        result = self.decisions.on_snapshot(snapshot)
        if result is None:
            return
        loop = "rescue" if "rescue_skipped_by_cadence" not in result.events else "fairness"
        for event in result.events:
            self.run.events.append((t, f"{loop}|{event}"))
        for model, context in (result.model_contexts or {}).items():
            # O1: the reason the model's window decides nothing (None = it decides).
            self.run.events.append((t, f"hold:{model}:{context.get('signal_hold_reason')}"))

    async def _advance(self, until: int) -> None:
        while self.now < until:
            self.now += 250
            if self.now % INSTANT == 0 and self.now > self.rendered:
                self._render(self.now)
                self.rendered = self.now
            if self.now % 1000 == 250:
                await self._tick(self.now - 250)
            if self.collector is not None:
                for _ in range(4):  # let the baseline task of a hide just confirmed run now
                    await asyncio.sleep(0)

    async def _sleep(self, seconds: float) -> None:
        await self._advance(self.now + int(round(seconds * 1000)))

    def pod_scrape(self, name: str) -> PodScrape:
        """A direct /metrics read of a sim pod (controller clock)."""
        pod = self._pod(name)
        running, waiting, kv = pod.gauges
        return PodScrape(
            ts_ms=self.now, started_ms=self.now, ttft=_cumulative(pod.count, TTFT_BOUNDS, pod.ttft),
            tpot=_cumulative(pod.gen, TPOT_BOUNDS, pod.tpot), prompt=(pod.count * PROMPT_TOKENS, pod.count),
            kv_cache=kv, awake=pod.awake, in_flight=running + waiting,
        )

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


class _SimScraper:
    def __init__(self, sim: Sim) -> None:
        self._sim = sim

    async def scrape(self, targets, *, model_name=None):
        return {name: self._sim.pod_scrape(name) for name in targets}


class _SimEvidence:
    """The hide anchor of the direct path (``MetricsEvidenceReader.hide_anchor``):
    Redis TIME (the gateway node's clock here) and the model's newest histogram doc
    stamp (gateway clock), with the controller clock beside them."""

    def __init__(self, sim: Sim) -> None:
        self._sim = sim

    def hide_anchor(self, model: str) -> HideAnchor:
        sim = self._sim
        newest = None
        for pod in (p for p in sim.pods if p.model == model):
            docs = sim.redis.zrevrangebyscore(f"tre:v2:hist:default/{pod.name}", float("inf"), float("-inf"), 0, 1)
            if docs:
                stamp = int(json.loads(docs[0])["timestamp"])
                newest = stamp if newest is None else max(newest, stamp)
        return HideAnchor(ts_ms=sim.now + sim.offset, source="redis_time", newest_doc_ts_ms=newest,
                          controller_ts_ms=sim.now)

    def read(self, model, *, start_ms, end_ms, exclude_pods):  # the Redis path only
        raise AssertionError("the direct path never reads the Redis evidence")


def _poisson(rng: random.Random, lam: float) -> int:
    if lam <= 0.0:
        return 0
    limit, k, product = math.exp(-lam), 0, rng.random()
    while product > limit:
        k += 1
        product *= rng.random()
    return k


def _cumulative(count: float, bounds: tuple[float, ...], buckets: list) -> Hist:
    running, out = 0.0, []
    for bound, n in zip((*bounds, float("inf")), buckets):
        running += n
        out.append((float(bound), running))
    return Hist(count=float(count), buckets=tuple(out))


def _hist(count: float, sum_: float, bounds: tuple[float, ...], buckets: list | None = None) -> dict:
    cumulative, running = {}, 0.0
    values = buckets if buckets is not None else [count] + [0.0] * len(bounds)
    for bound, n in zip((*bounds, float("inf")), values):
        running += n
        cumulative["+Inf" if bound == float("inf") else repr(bound)] = running
    return {"count": count, "sum": sum_, "buckets": cumulative}
