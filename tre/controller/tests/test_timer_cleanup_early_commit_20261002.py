"""Timer cleanup item 3 (2026-10-02, docs/design/20261002-timer-cleanup.md): SafeScale
early commit on the direct evidence path.

Before the deadline (hide confirmation + W) a probe commits once, on the same poll,
(a) min_commit_samples requests of the remaining pods are judged, (b) every formal
commit gate passes on the evidence so far, (c) the hidden pods have nothing in flight
(gateway count of the live plugin instances and vLLM running + waiting, both known and
0) and (d) one evidence floor (Q4, 2026-10-06): the newest snapshot window holds
scaling.min_evidence_grids complete post-hide grids and at least the donor's p95 e2e
passed since the confirmation (no W / 2 term, no minimum observation time). Rollback
checks are unchanged."""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import replace

import pytest

from tre_common.metrics_schema import MetricsSnapshot, ModelWindowMetrics
from tre_common.registry import SafeScaleRegistryConfig, load_registry, parse_safescale_config
from tre_controller.config import ControllerConfig
from tre_controller.loops.safescale_task import run_safescale_observation_tick
from tre_controller.planning.safescale import SafeScaleDecision
from tre_controller.planning.safescale_direct import DirectEvidenceCollector, GatewayInflightReader

from test_safescale_direct_20260929 import HIDE, MODEL, TRE_DIR, Harness, PodSim
from test_safescale_evidence_20260929 import _slo_registry

LONG_W = 60_000  # a probe window long enough to see the early commit clearly


class HiddenSim(PodSim):
    """The hidden probe pod: also renders vLLM's running / waiting gauges."""

    def __init__(self) -> None:
        super().__init__()
        self.running = 0.0
        self.waiting = 0.0
        self.gauges = True

    def text(self, **kwargs) -> str:
        text = super().text(**kwargs)
        if not self.gauges:
            return text
        return text + (f'vllm:num_requests_running{{engine="0",model_name="m"}} {self.running}\n'
                       f'vllm:num_requests_waiting{{engine="0",model_name="m"}} {self.waiting}\n')


class EarlyHarness(Harness):
    """The direct-evidence harness with the hidden pod m-1 scraped and a gateway count."""

    def __init__(self, pods=("m-0", "m-2"), **cfg) -> None:
        cfg.setdefault("min_window_ms", float(LONG_W))
        cfg.setdefault("early_commit", True)
        self.gateway: float | None = 0.0
        self.gateway_reads: list[tuple[str, ...]] = []
        super().__init__(pods=pods, **cfg)
        self.hidden = HiddenSim()
        self.sims["m-1"] = self.hidden

    def _collector(self, machine):
        def gateway(pods):
            self.gateway_reads.append(tuple(pods))
            return self.gateway

        return DirectEvidenceCollector(
            machine, self.scraper,
            lambda model, exclude: {pod: f"http://{pod}" for pod in self.sims if pod not in exclude},
            poll_ms=2_000.0, clock_ms=self.clock, hidden_scrape=True, gateway_inflight=gateway,
        )

    def start(self, **kwargs):
        probe = super().start(**kwargs)
        assert probe.deadline_ms == HIDE + LONG_W
        return probe


def _first_terminal(h: EarlyHarness, end_ms: int, *, serve: int = 5, ttft_s: float = 0.05):
    return h.run_until(end_ms, serve=serve, ttft_s=ttft_s)[-1]


def test_commits_at_the_first_poll_that_meets_every_condition():
    h = EarlyHarness()
    h.start()
    at, decision = _first_terminal(h, HIDE + LONG_W)
    # Polls every 2 s from 105 s: 20 samples are there early; the newest snapshot window
    # holds two post-hide grids (110-130 s) from the 131 s poll, and one p95 e2e has long
    # passed (Q4: no W / 2 term) -> the 131 s poll.
    assert (at, decision.status, decision.reason) == (131_000, "commit", "formal_commit_gate_passed")
    early = decision.details["early_commit"]
    assert early["elapsed_ms"] == 28_000 and early["min_elapsed_ms"] < 28_000 and early["samples"] >= 20
    assert early["post_hide_grids"] == 2
    assert early["planned_deadline_ms"] == HIDE + LONG_W
    assert (early["hidden_in_flight"], early["gateway_in_flight"]) == (0.0, 0.0)
    assert decision.commands[0].kind == "scale_down" and decision.commands[0].pods == ("m-1",)
    assert h.gateway_reads and set(h.gateway_reads) == {("m-1",)}


def test_the_evidence_never_contains_the_hidden_pod():
    h = EarlyHarness()
    h.start()
    h.clock.now = HIDE + 2_000
    polls = asyncio.run(h.collector.poll())
    poll = polls[MODEL]
    assert sorted(poll.results) == ["m-0", "m-2"] and sorted(poll.hidden) == ["m-1"]
    assert poll.hidden["m-1"].in_flight == 0.0 and poll.gateway_inflight == 0.0


@pytest.mark.parametrize("blocker", ["vllm_running", "vllm_waiting", "gateway", "gateway_unknown", "no_gauges"])
def test_requests_on_the_hidden_pod_keep_the_probe_running_until_they_finish(blocker):
    h = EarlyHarness()
    h.start()
    if blocker == "vllm_running":
        h.hidden.running = 1.0
    elif blocker == "vllm_waiting":
        h.hidden.waiting = 2.0
    elif blocker == "gateway":
        h.gateway = 1.0
    elif blocker == "gateway_unknown":
        h.gateway = None
    else:
        h.hidden.gauges = False
    decisions = h.run_until(141_000, serve=5)
    assert all(d.status == "probing" for _, d in decisions)
    h.hidden.running = h.hidden.waiting = 0.0
    h.hidden.gauges = True
    h.gateway = 0.0
    assert h.tick(143_000, serve=5).status == "commit"


def test_fewer_than_min_commit_samples_never_commit_early():
    h = EarlyHarness(pods=("m-0",))
    h.start()
    decisions = h.run_until(141_000, serve=0)  # nothing completes on the remaining pod
    assert all(d.status == "probing" for _, d in decisions)
    assert h.tick(143_000, serve=10).status == "probing"  # 10 judged < 20
    decision = h.tick(145_000, serve=10)
    assert decision.status == "commit" and decision.details["early_commit"]["samples"] == 20


def test_a_long_e2e_model_never_commits_before_one_p95_e2e():
    # Review P2-3: the remaining pods' concurrency needs about one end-to-end latency to
    # reach its new steady state - an early commit waits at least p95 e2e (45 s here).
    h = EarlyHarness()
    probe = h.start()
    terms = {**probe.window_terms, "inputs": {**probe.window_terms.get("inputs", {}), "p95_e2e_ms": 45_000.0}}
    h.machine._probes[MODEL] = replace(probe, window_terms=terms)
    at, decision = _first_terminal(h, HIDE + LONG_W)
    assert decision.status == "commit" and at == 149_000
    assert decision.details["early_commit"]["min_elapsed_ms"] == 45_000


def test_the_post_hide_grids_follow_the_o1_warm_rule():
    # Three post-hide grids required: the newest window must end at 140 s.
    h = EarlyHarness(early_commit_min_grids=3)
    h.start()
    at, decision = _first_terminal(h, HIDE + LONG_W)
    assert decision.status == "commit" and at == 141_000


def test_switched_off_the_probe_commits_at_the_deadline_only():
    h = EarlyHarness(early_commit=False)
    h.start()
    at, decision = _first_terminal(h, HIDE + LONG_W)
    assert (at, decision.status) == (HIDE + LONG_W, "commit")
    assert "early_commit" not in decision.details


def test_a_failing_gate_is_not_acted_on_early_and_rolls_back_at_the_deadline():
    h = EarlyHarness(kv_cache_max=0.8)
    for sim in h.sims.values():
        sim.kv = 0.9
    h.start()
    at, decision = _first_terminal(h, HIDE + LONG_W)
    assert (at, decision.status) == (HIDE + LONG_W, "rollback")
    assert decision.details["rollback_reason"]["gates"] == ["kv_cache"]


def test_a_violation_still_rolls_back_at_once():
    h = EarlyHarness(pods=("m-0", "m-2", "m-3"))
    h.start()
    at, decision = _first_terminal(h, HIDE + LONG_W, serve=8, ttft_s=2.0)
    assert decision.status == "rollback" and at == HIDE + 2_000
    assert decision.details["rollback_reason"]["code"] == "slo_violation_direct"


def test_the_observation_tick_logs_the_early_commit():
    class _Probe:
        model = MODEL
        pods = ("m-1",)
        request_id = "m-1"

    class _Machine:
        def active_probes(self):
            return (_Probe(),)

        def observe(self, model, observation, *, now_ms, **_kwargs):
            return SafeScaleDecision(status="commit", reason="formal_commit_gate_passed",
                                     details={"early_commit": {"elapsed_ms": 17_000, "samples": 42.0}})

        def resolve(self, *args, **kwargs):
            return True

    class _Queue:
        def submit(self, actions):
            return None

    snapshot = MetricsSnapshot(ts_ms=120_000, stale=False, models={MODEL: ModelWindowMetrics(
        model=MODEL, window_start_ms=90_000, window_end_ms=120_000, prompt_tokens=0.0, generation_tokens=100.0,
        avg_waiting=0.0, avg_running=1.0, avg_swapping=0.0, kv_cache_hit_rate=0.0, ttft_p95_ms=100.0,
        tpot_p95_ms=10.0, e2e_p95_ms=500.0, routable_pods=1, assigned_replicas=1, per_pod={},
    )})
    result = run_safescale_observation_tick(snapshot, queue=_Queue(), registry=_slo_registry(), safescale=_Machine())
    assert "safescale_early_commit:m:elapsed_ms=17000:samples=42" in result.events


def test_the_state_machine_logs_a_json_event(caplog):
    h = EarlyHarness()
    h.start()
    with caplog.at_level(logging.INFO, logger="tre_controller.safescale"):
        _first_terminal(h, HIDE + LONG_W)
    events = [json.loads(r.getMessage()) for r in caplog.records if "safescale_early_commit" in r.getMessage()]
    assert events and events[0]["model"] == MODEL and events[0]["elapsed_ms"] == 28_000


NOW_MS = 1_000_000


class _FakeRedis:
    """``instances``: plugin instance id -> heartbeat score (Redis TIME ms)."""

    def __init__(self, instances=None, hashes=None, fail=False):
        self.instances = {"gw-a": NOW_MS, "gw-b": NOW_MS} if instances is None else instances
        self.hashes = hashes or {}
        self.fail = fail

    def time(self):
        if self.fail:
            raise ConnectionError("down")
        return NOW_MS // 1000, (NOW_MS % 1000) * 1000

    def zrange(self, key, start, end, withscores=False):
        assert key == "tre:v2:gw:instances" and withscores
        return [(name.encode(), float(score)) for name, score in self.instances.items()]

    def hgetall(self, key):
        return self.hashes.get(key, {})


def test_gateway_inflight_reader_counts_live_instances_only():
    """Early commit reads the hidden pods' gateway in-flight count over the live plugin
    instances only: a field a dead instance left behind never blocks the commit, a
    live instance's nonzero count does; no live instance (or a read error) is unknown."""
    key = "tre:v2:gw:inflight:m-1"
    fields = {b"gw-a": b'{"total":0,"non_continuable":0,"ts":1}', "gw-b": '{"total":2,"non_continuable":0,"ts":1}'}
    live = {"gw-a": NOW_MS, "gw-b": NOW_MS - 2_000}
    assert GatewayInflightReader(_FakeRedis(live, {key: fields}))(("m-1",)) == 2.0
    dead_b = {"gw-a": NOW_MS, "gw-b": NOW_MS - 60_000}  # gw-b stopped heartbeating
    assert GatewayInflightReader(_FakeRedis(dead_b, {key: fields}), max_age_ms=10_000)(("m-1",)) == 0.0
    assert GatewayInflightReader(_FakeRedis())(("m-1",)) == 0.0  # no field: nothing routed there
    assert GatewayInflightReader(_FakeRedis({}))(("m-1",)) is None  # nobody counts
    assert GatewayInflightReader(_FakeRedis({"gw-a": NOW_MS - 60_000}))(("m-1",)) is None
    assert GatewayInflightReader(_FakeRedis(fail=True))(("m-1",)) is None


@pytest.mark.parametrize("raw", [b"not-json", b'{"total":"x"}', b'{"non_continuable":0}'])
def test_an_unreadable_live_inflight_field_is_unknown_and_blocks_early_commit(raw):
    """Review 2026-10-06 P3-1: a live instance whose in-flight field cannot be read may
    hold requests on the hidden pod - unknown (no early commit), never 0. The same
    field of a dead instance is ignored."""
    key = "tre:v2:gw:inflight:m-1"
    fields = {b"gw-a": b'{"total":0,"non_continuable":0,"ts":1}', b"gw-b": raw}
    assert GatewayInflightReader(_FakeRedis(hashes={key: fields}))(("m-1",)) is None
    dead_b = {"gw-a": NOW_MS, "gw-b": NOW_MS - 60_000}
    assert GatewayInflightReader(_FakeRedis(dead_b, {key: fields}), max_age_ms=10_000)(("m-1",)) == 0.0


def test_registry_and_config_keys(tmp_path):
    assert SafeScaleRegistryConfig().early_commit is True
    shipped = load_registry(str(TRE_DIR / "deploy" / "registry.yaml"))
    assert shipped.safescale().early_commit is True and shipped.scaling().min_evidence_grids == 2
    assert parse_safescale_config({"early_commit": False}).early_commit is False
    with pytest.raises(ValueError):
        parse_safescale_config({"early_commit": "yes"})
    # Q4: the post-hide grids are O1's scaling.min_evidence_grids; a registry that still
    # carries the removed safescale.early_commit_min_grids loads (unknown key: a warning).
    text = (TRE_DIR / "deploy" / "registry.yaml").read_text()
    assert "\n  min_evidence_grids: 2\n" in text and "\n  early_commit: true\n" in text
    text = text.replace("\n  min_evidence_grids: 2\n", "\n  min_evidence_grids: 3\n")
    text = text.replace("\n  early_commit: true\n", "\n  early_commit: true\n  early_commit_min_grids: 5\n")
    path = tmp_path / "registry.yaml"
    path.write_text(text)
    cfg = ControllerConfig.from_env({"TRE_REGISTRY_PATH": str(path)}).safescale
    assert cfg.early_commit is True and cfg.early_commit_min_grids == 3
