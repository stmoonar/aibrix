"""SafeScale direct evidence (2026-09-29, plan B+D).

B: at the SM's hide confirmation the controller scrapes the remaining pods' vLLM
/metrics (baseline), then every poll; the difference is the probe's latency / KV
evidence. D: a judged SLO violation rolls back at once. Deadline = confirmation + W on
the controller clock, extended one poll period while short, 60 s ceiling. Fallback to
the Redis evidence path when every remaining pod fails; fail-closed without it.
"""
from __future__ import annotations

import asyncio
import json
import math
import time
from pathlib import Path

import pytest
import yaml

from tre_common.metrics_schema import MetricsSnapshot, ModelWindowMetrics
from tre_common.registry import SafeScaleRegistryConfig, load_registry, parse_safescale_config
from tre_controller.config import ControllerConfig, SafeScaleConfig
from tre_controller.loops.cluster_view_task import cluster_view_from_state
from tre_controller.loops.safescale_task import run_safescale_observation_tick
from tre_controller.planning.safescale import ProbeObservation, SafeScaleStateMachine
from tre_controller.planning.safescale_direct import (
    DirectEvidenceCollector,
    DirectState,
    Hist,
    PodMetricsScraper,
    cluster_view_targets,
    compact_hist,
    expand_hist,
    hist_delta,
    parse_vllm_metrics,
)
from tre_controller.planning.safescale_evidence import HideAnchor

from test_safescale import FakeProbeStore
from test_safescale_evidence_20260929 import FakeEvidence, _slo_registry, _window

TESTS = Path(__file__).resolve().parent
TRE_DIR = TESTS.parents[1]
LIVE_030 = TESTS / "fixtures" / "vllm-0.30.0-awake.prom"
OLD_0101 = TRE_DIR / "deploy" / "tests" / "fixtures" / "vllm_metrics" / "vllm-0.10.1.prom"

MODEL = "m"
START = 100_000  # probe planned on the snapshot of boundary 100 s
HIDE = 103_000  # the SM confirmed the hide 3 s later (controller clock)
W = 20_000

# vLLM 0.30 bucket grids (from the live fixture).
TTFT_LES = (0.001, 0.005, 0.01, 0.02, 0.04, 0.06, 0.08, 0.1, 0.25, 0.5, 0.75, 1.0, 2.5, 5.0, 7.5, 10.0,
            20.0, 40.0, 80.0, 160.0, 640.0, 2560.0)
ITL_LES = (0.01, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.75, 1.0, 2.5, 5.0, 7.5, 10.0, 20.0,
           40.0, 80.0)


# ----------------------------------------------------------------------- helpers
class PodSim:
    """A vLLM pod's cumulative counters, rendered as /metrics text."""

    def __init__(self, *, kv: float = 0.2) -> None:
        self.ttft: list[float] = []
        self.itl: list[float] = []
        self.prompt: list[float] = []
        self.kv = kv
        self.awake = True

    def serve(self, n: int, *, ttft_s: float = 0.05, itl_s: float = 0.02, prompt: float = 500.0) -> None:
        for _ in range(n):
            self.ttft.append(ttft_s)
            self.itl.extend([itl_s] * 10)
            self.prompt.append(prompt)

    def restart(self) -> None:
        self.ttft, self.itl, self.prompt = [], [], []

    def text(self, *, tpot_family: str = "vllm:inter_token_latency_seconds") -> str:
        lines = [f'vllm:engine_sleep_state{{engine="0",model_name="m",sleep_state="awake"}} '
                 f'{1.0 if self.awake else 0.0}',
                 f'vllm:kv_cache_usage_perc{{engine="0",model_name="m"}} {self.kv}']
        lines += _hist_lines("vllm:time_to_first_token_seconds", TTFT_LES, self.ttft)
        lines += _hist_lines(tpot_family, ITL_LES, self.itl)
        # Per-request TPOT (0.30): a different family that must not be mistaken for it.
        lines += _hist_lines("vllm:request_time_per_output_token_seconds", ITL_LES, [9.0] * len(self.ttft))
        lines.append(f'vllm:request_prompt_tokens_count{{engine="0",model_name="m"}} {float(len(self.prompt))}')
        lines.append(f'vllm:request_prompt_tokens_sum{{engine="0",model_name="m"}} {float(sum(self.prompt))}')
        return "\n".join(lines) + "\n"


def _hist_lines(name: str, les, values) -> list[str]:
    out = []
    for le in les:
        count = sum(1 for value in values if value <= le)
        out.append(f'{name}_bucket{{engine="0",le="{le}",model_name="m"}} {float(count)}')
    out.append(f'{name}_bucket{{engine="0",le="+Inf",model_name="m"}} {float(len(values))}')
    out.append(f'{name}_count{{engine="0",model_name="m"}} {float(len(values))}')
    out.append(f'{name}_sum{{engine="0",model_name="m"}} {float(sum(values))}')
    return out


class Clock:
    def __init__(self, now: int) -> None:
        self.now = now

    def __call__(self) -> int:
        return self.now


class SimScraper:
    """Collector scraper over PodSims: pod -> PodScrape, or the scripted error."""

    def __init__(self, harness: "Harness") -> None:
        self.h = harness
        self.calls: list[tuple[int, tuple[str, ...]]] = []

    async def scrape(self, targets):
        self.calls.append((self.h.clock(), tuple(sorted(targets))))
        out = {}
        for pod in targets:
            if pod in self.h.failing:
                out[pod] = self.h.failing[pod]
            else:
                out[pod] = parse_vllm_metrics(self.h.sims[pod].text(), ts_ms=self.h.clock())
        return out


class RaisingEvidence(FakeEvidence):
    def read(self, model, *, start_ms, end_ms, exclude_pods):
        raise ConnectionError("redis down")


def _cfg(**overrides) -> SafeScaleConfig:
    values = dict(min_window_ms=float(W), window_ceiling_ms=60_000.0, hq=0.25, tau_low=1.0,
                  evidence_source="direct", evidence_poll_ms=2_000.0, min_latency_samples=10,
                  min_commit_samples=20)
    values.update(overrides)
    return SafeScaleConfig(**values)


def _obs(boundary: int, *, z: float = 2.0, traffic: bool = True, kv: float | None = 0.1) -> ProbeObservation:
    return ProbeObservation(ts_ms=boundary, ttft_p95_ms=100.0, tpot_p95_ms=10.0, z_m=z, has_traffic=traffic,
                            avg_gpu_cache_norm=kv, window_start_ms=boundary - 30_000, window_end_ms=boundary)


def _anchor() -> HideAnchor:
    return HideAnchor(ts_ms=HIDE, source="redis_time", newest_doc_ts_ms=100_000, controller_ts_ms=HIDE)


class Harness:
    def __init__(self, pods=("m-0", "m-2"), *, evidence=None, store=None, **cfg) -> None:
        self.clock = Clock(START)
        self.sims = {pod: PodSim() for pod in pods}
        self.failing: dict[str, str] = {}
        self.evidence = evidence if evidence is not None else FakeEvidence([_window(50)], anchor=_anchor())
        self.store = store
        self.machine = self._machine(cfg)
        self.cfg = cfg
        self.scraper = SimScraper(self)
        self.collector = self._collector(self.machine)
        self.obs_kwargs: dict = {}

    def _machine(self, cfg):
        return SafeScaleStateMachine(config=_cfg(**cfg), store=self.store, evidence=self.evidence,
                                     wall_clock_ms=self.clock)

    def _collector(self, machine):
        return DirectEvidenceCollector(
            machine, self.scraper,
            lambda model, exclude: {pod: f"http://{pod}" for pod in self.sims if pod not in exclude},
            poll_ms=2_000.0, clock_ms=self.clock,
        )

    def start(self, *, pre_hide: int = 0, pre_hide_ttft: float = 2.0):
        self.machine.start_probe(model=MODEL, pods=("m-1",), now_ms=START)
        for sim in self.sims.values():
            sim.serve(pre_hide, ttft_s=pre_hide_ttft)
        self.clock.now = HIDE
        # No running loop here: the baseline is taken by the collector (as the tick would).
        assert self.collector.on_hide_done(MODEL, ("m-1",))
        asyncio.run(self.collector.take_baselines())
        return self.machine.active_probe(MODEL)

    def tick(self, at_ms: int, *, serve: int = 0, ttft_s: float = 0.05, obs: ProbeObservation | None = None):
        self.clock.now = at_ms
        for sim in self.sims.values():
            sim.serve(serve, ttft_s=ttft_s)
        polls = asyncio.run(self.collector.poll())
        boundary = at_ms // 10_000 * 10_000
        observation = obs or _obs(boundary, **self.obs_kwargs)
        return self.machine.observe(MODEL, observation, now_ms=boundary, direct_poll=polls.get(MODEL))

    def run_until(self, end_ms: int, *, serve: int = 0, ttft_s: float = 0.05, first: int = HIDE + 2_000):
        decisions = []
        for at in range(first, end_ms + 1, 2_000):
            decision = self.tick(at, serve=serve, ttft_s=ttft_s)
            decisions.append((at, decision))
            if decision.status in ("commit", "rollback"):
                break
        return decisions


# ============================================================ /metrics format
def test_the_live_030_fixture_parses_to_the_evidence_families() -> None:
    scrape = parse_vllm_metrics(LIVE_030.read_text(encoding="utf-8"), ts_ms=1)
    assert scrape.ttft.count == 6.0
    assert dict(scrape.ttft.buckets)[0.04] == 5.0 and dict(scrape.ttft.buckets)[0.08] == 6.0
    assert dict(scrape.ttft.buckets)[math.inf] == 6.0
    # TPOT = the per-token inter_token_latency family (218 tokens), not the per-request
    # request_time_per_output_token_seconds (6 requests).
    assert scrape.tpot_name == "vllm:inter_token_latency_seconds"
    assert scrape.tpot.count == 218.0
    assert scrape.prompt == (72.0, 6.0)
    assert scrape.kv_cache == 0.0 and scrape.awake is True


def test_old_vllm_names_are_accepted() -> None:
    scrape = parse_vllm_metrics(OLD_0101.read_text(encoding="utf-8"), ts_ms=1)
    assert scrape.tpot_name == "vllm:time_per_output_token_seconds"
    assert scrape.ttft is not None and scrape.kv_cache is not None
    sim = PodSim()
    sim.serve(3)
    assert parse_vllm_metrics(sim.text(tpot_family="vllm:time_per_output_token_seconds"), ts_ms=1).tpot.count == 30


def test_the_compact_baseline_is_small_json_and_differences_exactly_like_the_full_one() -> None:
    before = PodSim()
    before.serve(40, ttft_s=0.03)
    before.serve(5, ttft_s=0.9)
    base = parse_vllm_metrics(before.text(), ts_ms=1)
    record = compact_hist(base.ttft)
    assert len(record[1]) == 2 and json.loads(json.dumps(record)) == record  # 2 steps, no Infinity
    before.serve(30, ttft_s=3.0)
    now = parse_vllm_metrics(before.text(), ts_ms=2)
    assert hist_delta(expand_hist(record), now.ttft) == hist_delta(base.ttft, now.ttft)
    assert hist_delta(base.ttft, now.ttft)[0] == 30.0
    assert hist_delta(now.ttft, base.ttft) is None  # counters backwards = restart


# ============================================================ baseline / difference
def test_pre_hide_requests_never_enter_the_evidence() -> None:
    h = Harness()
    probe = h.start(pre_hide=100, pre_hide_ttft=2.0)  # slow before the hide
    assert probe.direct.baseline_ts_ms == HIDE and sorted(probe.direct.baseline) == ["m-0", "m-2"]
    decisions = h.run_until(HIDE + W, serve=3, ttft_s=0.05)
    at, decision = decisions[-1]
    assert (at, decision.status, decision.reason) == (HIDE + W, "commit", "formal_commit_gate_passed")
    assert decision.details["evidence_ttft_p95_ms"] == pytest.approx(60.0)  # bucket_upper 0.06 s
    assert decision.details["latency_samples"] == 2 * 3 * 10
    assert decision.details["evidence_source_used"] == "direct"
    assert decision.details["tail_pre_hide_fraction"] == 0.0
    assert decision.details["evidence_start_ms"] == HIDE


def test_the_probe_commits_exactly_at_hide_confirmation_plus_w() -> None:
    h = Harness()
    probe = h.start()
    assert probe.deadline_ms == HIDE + W  # not aligned to the 10 s gateway grid
    decisions = h.run_until(HIDE + W + 10_000, serve=5)
    assert [d.status for _, d in decisions[:-1]] == ["probing"] * (len(decisions) - 1)
    assert decisions[-1][0] == HIDE + W and decisions[-1][1].status == "commit"


def test_short_evidence_extends_by_one_poll_period() -> None:
    h = Harness(pods=("m-0",))
    h.start()
    decisions = h.run_until(HIDE + W, serve=1)  # n = 10 < 20 at the deadline
    at, decision = decisions[-1]
    assert (at, decision.reason) == (HIDE + W, "evidence_extended")
    assert decision.details["deadline_ms"] == HIDE + W + 2_000
    assert decision.details["extend_reason"] == "insufficient_samples"
    final = h.tick(HIDE + W + 2_000, serve=10)
    assert final.status == "commit" and final.details["extensions"] == 1
    assert final.details["latency_samples"] == 20


def test_at_the_ceiling_traffic_skips_latency_and_idle_commits() -> None:
    h = Harness(pods=("m-0",))
    h.start()
    decisions = h.run_until(HIDE + 60_000, serve=0)  # traffic in flight, never 20 samples
    at, decision = decisions[-1]
    assert at == HIDE + 60_000 and decision.status == "commit"
    assert (decision.details["latency_gate"], decision.details["latency_skip_reason"]) == (
        "skipped", "insufficient_samples")
    assert decision.details["clamped"] is True
    idle = Harness(pods=("m-0",))
    idle.obs_kwargs = {"traffic": False}
    idle.start()
    at, decision = idle.run_until(HIDE + 60_000)[-1]
    assert (at, decision.status, decision.details["latency_skip_reason"]) == (HIDE + 60_000, "commit", "idle")


# ============================================================ immediate rollback (D)
@pytest.mark.parametrize("per_pod_per_tick, rollback_after_ms", [(8, 4_000), (20, 2_000)])
def test_a_violation_rolls_back_within_2_to_4_s_at_7b_load(per_pod_per_tick, rollback_after_ms) -> None:
    # 3 remaining pods at 4 / 10 req/s each (the 7b load range); TTFT 2 s > 500 ms.
    # Per-pod p95 needs TRE_MIN_LATENCY_SAMPLES (10) requests of that pod.
    h = Harness(pods=("m-0", "m-2", "m-3"))
    h.start()
    decisions = h.run_until(HIDE + W, serve=per_pod_per_tick, ttft_s=2.0)
    at, decision = decisions[-1]
    assert decision.status == "rollback" and at - HIDE == rollback_after_ms
    reason = decision.details["rollback_reason"]
    assert reason["code"] == "slo_violation_direct" and reason["metrics"] == ["ttft"]
    assert reason["ttft_p95_ms"] == pytest.approx(2500.0) and reason["ttft_threshold_ms"] == 500.0
    assert decision.details["evidence_source_used"] == "direct"


def test_below_min_commit_samples_a_violation_is_not_judged_yet() -> None:
    h = Harness(pods=("m-0",))
    h.start()
    assert h.tick(HIDE + 2_000, serve=12, ttft_s=2.0).status == "probing"  # n = 12 < 20
    assert h.tick(HIDE + 4_000, serve=12, ttft_s=2.0).status == "rollback"


# ============================================================ failures
def test_a_failing_pod_is_left_out_and_recorded() -> None:
    h = Harness()
    h.start()
    h.failing["m-2"] = "timeout"
    at, decision = h.run_until(HIDE + W, serve=5)[-1]
    assert decision.status == "commit" and decision.details["latency_samples"] == 50  # m-0 only
    assert decision.details["direct_excluded_pods"] == {"m-2": "timeout"}
    assert list(decision.details["direct_pods"]) == ["m-0"]
    assert decision.details["direct_pods"]["m-0"]["n"] == 50


def test_a_counter_reset_drops_the_pod_for_good() -> None:
    h = Harness()
    h.start(pre_hide=10, pre_hide_ttft=0.05)
    h.tick(HIDE + 2_000, serve=5)
    h.sims["m-2"].restart()
    h.tick(HIDE + 4_000, serve=5)
    probe = h.machine.active_probe(MODEL)
    assert probe.direct.dropped["m-2"]["reason"] == "counter_reset"
    assert probe.direct.live_pods() == ("m-0",)
    at, decision = h.run_until(HIDE + W, serve=5, first=HIDE + 6_000)[-1]
    assert decision.status == "commit" and decision.details["direct_excluded_pods"]["m-2"] == "counter_reset"
    assert "m-2" not in h.scraper.calls[-1][1]  # no longer polled


def test_every_pod_failing_falls_back_to_the_redis_evidence() -> None:
    h = Harness()
    h.start()
    h.tick(HIDE + 2_000, serve=5)
    h.failing.update({"m-0": "timeout", "m-2": "error:URLError"})
    assert h.tick(HIDE + 4_000).status == "probing"
    probe = h.machine.active_probe(MODEL)
    assert probe.direct.fallback["reason"] == "all_pods_failed"
    polls_before = len(h.scraper.calls)
    h.tick(HIDE + 6_000)
    assert len(h.scraper.calls) == polls_before  # sticky: no more scraping
    # The Redis path (687cbd9c logic) judges at the snapshot deadline.
    decision = h.tick(130_000, obs=_obs(130_000))
    assert decision.status == "commit"
    assert decision.details["evidence_source_used"] == "redis_fallback"
    assert decision.details["latency_source"] == "evidence"
    assert h.evidence.reads and h.evidence.reads[-1]["start_ms"] == 110_000


def test_a_failed_baseline_falls_back_at_once() -> None:
    h = Harness()
    h.failing.update({"m-0": "timeout", "m-2": "no_endpoint"})
    probe = h.start()
    assert probe.direct.fallback["reason"] == "baseline_failed"
    assert probe.direct.dropped == {"m-0": {"reason": "baseline_timeout", "ts_ms": HIDE},
                                    "m-2": {"reason": "baseline_no_endpoint", "ts_ms": HIDE}}


def test_redis_evidence_failing_too_fails_closed() -> None:
    h = Harness(evidence=RaisingEvidence([_window(50)], anchor=_anchor()))
    h.failing.update({"m-0": "timeout", "m-2": "timeout"})
    h.start()
    decision = h.tick(130_000, obs=_obs(130_000))
    assert decision.status == "rollback"
    assert decision.details["rollback_reason"]["code"] == "evidence_unavailable"
    assert decision.details["evidence_source_used"] == "redis_fallback"
    # No Redis evidence reader at all: fail-closed as well.
    bare = Harness()
    bare.machine = SafeScaleStateMachine(config=_cfg(), evidence=None, wall_clock_ms=bare.clock)
    bare.collector = bare._collector(bare.machine)
    bare.failing.update({"m-0": "timeout", "m-2": "timeout"})
    bare.start()
    decision = bare.tick(130_000, obs=_obs(130_000))
    assert decision.details["rollback_reason"] == {"code": "evidence_unavailable",
                                                   "detail": "no_redis_evidence_reader"}


# ============================================================ KV / Z gates
def test_the_kv_gate_reads_the_direct_scrape() -> None:
    h = Harness(kv_cache_max=0.8)
    for sim in h.sims.values():
        sim.kv = 0.9
    h.start()
    at, decision = h.run_until(HIDE + W, serve=5)[-1]  # the tail snapshots say 0.1
    assert decision.status == "rollback"
    assert decision.details["rollback_reason"]["gates"] == ["kv_cache"]
    assert (decision.details["kv_source"], decision.details["kv_direct"]) == ("direct", pytest.approx(0.9))
    assert decision.details["kv_redis_tail_max"] == 0.1 and decision.details["kv_ts_ms"] == HIDE + W
    ok = Harness(kv_cache_max=0.8)
    ok.obs_kwargs = {"kv": 0.95}
    ok.start()
    at, decision = ok.run_until(HIDE + W, serve=5)[-1]
    assert decision.status == "commit" and decision.details["kv_redis_tail_max"] == 0.95


def test_the_z_gate_still_reads_the_snapshot_tail() -> None:
    h = Harness()
    h.obs_kwargs = {"z": 0.5}
    h.start()
    at, decision = h.run_until(HIDE + W, serve=5)[-1]
    assert decision.status == "rollback" and decision.details["rollback_reason"]["gates"] == ["z_below_tau_low"]
    assert decision.details["z_source"] == "redis_snapshot_tail"
    assert decision.details["z_ts_ms"] == 120_000


# ============================================================ restart
def test_a_restarted_controller_resumes_from_the_persisted_baseline() -> None:
    store = FakeProbeStore()
    h = Harness(store=store)
    probe = h.start(pre_hide=50, pre_hide_ttft=2.0)
    h.tick(HIDE + 2_000, serve=4)
    record = json.loads(json.dumps(store.records[probe.request_id]))  # through JSON, as Redis
    assert set(record["direct_evidence"]) >= {"baseline", "baseline_ts_ms", "scrapes"}
    restored = SafeScaleStateMachine(
        config=_cfg(), store=FakeProbeStore(unresolved=[record], journal=store.journal),
        evidence=h.evidence, wall_clock_ms=h.clock,
    )
    assert restored.restore() == 1
    again = restored.active_probe(MODEL)
    assert again.deadline_ms == HIDE + W and again.direct.baseline_ts_ms == HIDE
    assert again.direct.baseline["m-0"].ttft.count == 50  # the pre-hide requests stay out
    h.machine, h.collector = restored, h._collector(restored)
    at, decision = h.run_until(HIDE + W, serve=4, first=HIDE + 4_000)[-1]
    assert decision.status == "commit" and decision.details["latency_samples"] == 2 * 4 * 10
    assert decision.details["evidence_ttft_p95_ms"] == pytest.approx(60.0)


# ============================================================ non-blocking scrapes
def test_scrapes_are_concurrent_bounded_and_never_block_the_event_loop() -> None:
    def slow_fetch(url: str, timeout_s: float) -> str:
        time.sleep(2.0 if "dead" in url else 0.15)
        return LIVE_030.read_text(encoding="utf-8")

    scraper = PodMetricsScraper(timeout_s=0.3, fetch=slow_fetch)

    async def scenario():
        beats = 0
        stop = asyncio.Event()

        async def heartbeat():
            nonlocal beats
            while not stop.is_set():
                beats += 1
                await asyncio.sleep(0.01)

        beat = asyncio.create_task(heartbeat())
        started = time.monotonic()
        results = await scraper.scrape({f"p{i}": f"http://ok-{i}" for i in range(6)} | {"d": "http://dead",
                                                                                         "n": None})
        elapsed = time.monotonic() - started
        stop.set()
        await beat
        return results, elapsed, beats

    results, elapsed, beats = asyncio.run(scenario())
    scraper.close()
    assert results["d"] == "timeout" and results["n"] == "no_endpoint"
    assert all(results[f"p{i}"].ttft.count == 6.0 for i in range(6))
    assert elapsed < 0.6  # 6 x 0.15 s in parallel, the dead pod cut at 0.3 s
    assert beats >= 10  # the loop kept running meanwhile


def test_the_hide_callback_starts_the_baseline_on_the_running_loop() -> None:
    h = Harness()
    h.machine.start_probe(model=MODEL, pods=("m-1",), now_ms=START)
    h.clock.now = HIDE

    async def scenario():
        assert h.collector.on_hide_done(MODEL, ("m-1",))
        assert h.machine.active_probe(MODEL).direct.baseline is None  # not blocking the caller
        await asyncio.gather(*list(h.collector._tasks))

    asyncio.run(scenario())
    assert h.machine.active_probe(MODEL).direct.baseline_ts_ms == HIDE
    assert h.machine.direct_baseline_due() == ()


# ============================================================ wiring / config
def test_the_observation_tick_hands_the_polls_to_the_machine_and_logs_the_direct_rollback() -> None:
    from test_safescale_task import FakeQueue

    h = Harness(pods=("m-0", "m-2", "m-3"))
    h.start()
    registry = _slo_registry()
    snapshot = MetricsSnapshot(ts_ms=100_000, stale=False, models={MODEL: ModelWindowMetrics(
        model=MODEL, window_start_ms=70_000, window_end_ms=100_000, prompt_tokens=0.0, generation_tokens=100.0,
        avg_waiting=0.0, avg_running=1.0, avg_swapping=0.0, kv_cache_hit_rate=0.0, ttft_p95_ms=100.0,
        tpot_p95_ms=10.0, e2e_p95_ms=500.0, routable_pods=1, assigned_replicas=1, per_pod={},
    )})
    h.clock.now = HIDE + 2_000
    for sim in h.sims.values():
        sim.serve(20, ttft_s=2.0)
    polls = asyncio.run(h.collector.poll())
    queue = FakeQueue()
    result = run_safescale_observation_tick(snapshot, queue=queue, registry=registry, safescale=h.machine,
                                            direct_polls=polls)
    assert any(e.startswith("safescale_rollback_reason:m:slo_violation_direct:ttft") for e in result.events)
    assert any(e.startswith("safescale_evidence:m:") for e in result.events)
    assert queue.submitted and queue.submitted[0][0].pods == ("m-1",)


def test_the_cluster_view_carries_pod_ips_and_targets_skip_probe_and_sleeping_pods() -> None:
    from tre_common.registry import ClusterTopology, NodeSpec

    state = {
        "bindings": [
            {"serve_id": "m-0", "model": "m", "node": "n", "gpu_ids": [0], "awake": True, "hidden": False},
            {"serve_id": "m-1", "model": "m", "node": "n", "gpu_ids": [1], "awake": True, "hidden": True},
            {"serve_id": "m-2", "model": "m", "node": "n", "gpu_ids": [2], "awake": False, "hidden": False},
            {"serve_id": "m-3", "model": "m", "node": "n", "gpu_ids": [3], "awake": True, "hidden": False},
        ],
        "fleet": {"observed": [{"pod_name": "m-0", "pod_ip": "10.0.0.5"}, {"pod_name": "m-1", "pod_ip": "10.0.0.6"},
                               {"pod_name": "m-2", "pod_ip": "10.0.0.7"}]},
    }
    view = cluster_view_from_state(state, ClusterTopology(nodes=(NodeSpec(name="n", gpus=4, two_gpu_slots=((0, 1), (2, 3))),)))
    assert view.pod_ips == {"m-0": "10.0.0.5", "m-1": "10.0.0.6", "m-2": "10.0.0.7"}
    targets = cluster_view_targets(lambda: view, port=8123)("m", ("m-1",))
    assert targets == {"m-0": "http://10.0.0.5:8123/metrics", "m-3": None}
    assert cluster_view_from_state({"bindings": []}, view.topology).pod_ips == {}


def test_registry_direct_keys_defaults_and_validation() -> None:
    defaults = SafeScaleRegistryConfig()
    assert (defaults.evidence_source, defaults.evidence_poll_s, defaults.scrape_timeout_s, defaults.metrics_port) == (
        "direct", 2.0, 1.0, 8000)
    tuned = parse_safescale_config({"evidence_source": "redis", "evidence_poll_s": 3, "scrape_timeout_s": 0.5,
                                    "metrics_port": 9000})
    assert (tuned.evidence_source, tuned.evidence_poll_s, tuned.scrape_timeout_s, tuned.metrics_port) == (
        "redis", 3.0, 0.5, 9000)
    for bad in ({"evidence_source": "gateway"}, {"evidence_poll_s": 0}, {"scrape_timeout_s": 2},
                {"scrape_timeout_s": -1}, {"metrics_port": 70000}, {"metrics_port": True}):
        with pytest.raises(ValueError):
            parse_safescale_config(bad)
    registry = yaml.safe_load((TRE_DIR / "deploy" / "registry.yaml").read_text(encoding="utf-8"))
    params = yaml.safe_load((TRE_DIR / "deploy" / "overlays" / "tre-v2" / "params.yaml").read_text(encoding="utf-8"))
    params_registry = yaml.safe_load(next(iter(params["data"].values())))
    assert params_registry["safescale"] == registry["safescale"]
    assert parse_safescale_config(registry["safescale"]) == defaults


def test_config_maps_the_registry_keys(tmp_path) -> None:
    raw = yaml.safe_load((TRE_DIR / "deploy" / "registry.yaml").read_text(encoding="utf-8"))
    raw["safescale"].update(evidence_poll_s=3, scrape_timeout_s=0.5, metrics_port=9000)
    path = tmp_path / "registry.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    cfg = ControllerConfig.from_env({"TRE_REGISTRY_PATH": str(path), "TRE_MIN_LATENCY_SAMPLES": "7"}).safescale
    assert (cfg.evidence_source, cfg.evidence_poll_ms, cfg.scrape_timeout_s, cfg.metrics_port) == (
        "direct", 3000.0, 0.5, 9000)
    assert (cfg.min_latency_samples, cfg.percentile_mode) == (7, "bucket_upper")


def test_the_app_wires_the_collector_and_redis_mode_leaves_it_out(tmp_path) -> None:
    from test_controller_app import EmptyRedis
    from test_controller_app import REGISTRY_PATH as APP_REGISTRY
    from tre_controller.app import create_controller_dependencies

    cfg = ControllerConfig.from_env({"TRE_REGISTRY_PATH": str(APP_REGISTRY)})
    deps = create_controller_dependencies(cfg, redis_client=EmptyRedis())
    assert isinstance(deps.direct_evidence, DirectEvidenceCollector)
    assert deps.queue._on_hide_done == deps.direct_evidence.on_hide_done
    raw = yaml.safe_load(Path(APP_REGISTRY).read_text(encoding="utf-8"))
    raw["safescale"]["evidence_source"] = "redis"
    path = tmp_path / "registry.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    cfg = ControllerConfig.from_env({"TRE_REGISTRY_PATH": str(path)})
    deps = create_controller_dependencies(cfg, redis_client=EmptyRedis())
    assert deps.direct_evidence is None and not deps.safescale.direct_mode()


def test_direct_state_record_roundtrip() -> None:
    sim = PodSim()
    sim.serve(12)
    from tre_controller.planning.safescale_direct import take_baseline

    state = take_baseline({"a": parse_vllm_metrics(sim.text(), ts_ms=5), "b": "timeout"}, ts_ms=6)
    again = DirectState.from_record(json.loads(json.dumps(state.as_record())))
    assert again.baseline["a"].ttft.count == 12 and again.dropped == {"b": {"reason": "baseline_timeout", "ts_ms": 6}}
    assert again.baseline_ts_ms == 5 and isinstance(again.baseline["a"].ttft, Hist)
