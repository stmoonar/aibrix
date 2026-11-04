"""SafeScale evidence rework (2026-09-29, Fable list items 1-6).

1. one observation per snapshot; the immediate rollback judges only post-hide snapshots;
2. the latency gate reads the post-hide evidence window (remaining pods only), with
   deadline extensions while it is short and the two outcomes at the ceiling;
3. W ceiling 60 s from the registry (extensions included);
4. thresholds from the registry (labels / fixed), env overrides optional;
5. evidence clock check (fail-closed);
6. audit fields in the decision, the probe record and window_terms.
"""
from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest
import yaml

from tre_common.metrics_schema import MetricsSnapshot, ModelWindowMetrics
from tre_common.rediskeys import hist_key, pods_key
from tre_common.registry import (
    ClusterTopology,
    ModelSpec,
    NodeSpec,
    Registry,
    SafeScaleRegistryConfig,
    SloSpec,
    TrsParams,
    load_registry,
    parse_safescale_config,
)
from tre_common.slo import SloConfigError
from tre_common.slo_labels import label_def_for_model
from safescale_thresholds import STATIC_TPOT_75
from tre_controller.config import ControllerConfig, SafeScaleConfig
from tre_controller.loops.action_queue import ActionQueue
from tre_controller.loops.safescale_task import (
    _observation_from_metrics,
    format_evidence_event,
    run_safescale_observation_tick,
)
from tre_controller.planning.planner import HideAction
from tre_controller.planning.safescale import (
    ProbeObservation,
    ProbeWindowInputs,
    SafeScaleStateMachine,
    calc_probe_window_details,
)
from tre_controller.planning.safescale_evidence import (
    EvidenceWindow,
    HideAnchor,
    MetricsEvidenceReader,
    RegistryThresholds,
    StaticThresholds,
)
from tre_controller.store.metrics_store import MetricsStore

from test_action_queue_review3 import ScriptedSM
from test_safescale import FakeProbeStore

TRE_DIR = Path(__file__).resolve().parents[2]
MODEL = "m"
START = 100_000  # probe planned on the snapshot of boundary 100 s
HIDE = 103_000  # the SM confirmed the hide 3 s later (Redis TIME)
#: vLLM 0.30's per-request TPOT bucket edges (s).
VLLM_TPOT_EDGES_S = (0.01, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.75, 1.0, 2.5, 5.0, 7.5, 10.0,
                     20.0, 40.0, 80.0)


# ----------------------------------------------------------------------- helpers
def _cfg(**overrides) -> SafeScaleConfig:
    values = dict(min_window_ms=20_000.0, window_ceiling_ms=60_000.0, hq=0.25, tau_low=1.0)
    values.update(overrides)
    return SafeScaleConfig(**values)


def _obs(ts_ms: int, *, tpot: float = 10.0, z: float | None = 2.0, traffic: bool = True,
         window_ms: int = 30_000, gateway: tuple[float, float] | None = None
         ) -> ProbeObservation:
    return ProbeObservation(
        ts_ms=ts_ms,
        tpot_p95_ms=tpot,
        z_m=z,
        has_traffic=traffic,
        window_start_ms=ts_ms - window_ms,
        window_end_ms=ts_ms,
        gateway_requests=gateway[0] if gateway else None,
        gateway_errors=gateway[1] if gateway else None,
    )


class FakeEvidence:
    """Scripted evidence source: one window per read; records the reads."""

    def __init__(self, windows=(), *, anchor: HideAnchor | None = None) -> None:
        self.windows = list(windows)
        self.reads: list[dict] = []
        self.anchor = anchor or HideAnchor(ts_ms=HIDE, source="redis_time", newest_doc_ts_ms=100_000)

    def hide_anchor(self, model):
        return self.anchor

    def read(self, model, *, start_ms, end_ms, exclude_pods):
        self.reads.append({"model": model, "start_ms": start_ms, "end_ms": end_ms, "exclude": tuple(exclude_pods)})
        window = self.windows.pop(0) if len(self.windows) > 1 else self.windows[0]
        # The gateway writes a doc of every remaining pod every period: by default each
        # pod's docs reach the window end (a fresh view's remaining pods must, to commit).
        last = window.get("last_doc_ts_ms", {pod: end_ms for pod in window.get("pods", ())})
        # A source with the per-request TPOT series has its bucket edges (vLLM 0.30's);
        # the real Redis reader has none (fail closed).
        edges = window.get("tpot_edges_s", VLLM_TPOT_EDGES_S)
        return EvidenceWindow(**{**window, "start_ms": start_ms, "end_ms": end_ms, "last_doc_ts_ms": last,
                                 "tpot_edges_s": edges})


def _window(n: float, *, tpot: float | None = 10.0, first_doc: int = 110_000) -> dict:
    return dict(
        start_ms=0, end_ms=0, pods=("m-0",), excluded_pods=("m-1",), tpot_p95_ms=tpot, request_count=n,
        first_doc_ts_ms={"m-0": first_doc} if first_doc is not None else {},
    )


def _remaining(*pods: str):
    """A fresh cluster view's remaining pods (Redis mode: the pods a commit needs)."""
    return lambda model, exclude: tuple(pod for pod in pods if pod not in exclude)


def _machine(evidence, *, store=None, thresholds=None, remaining=("m-0",), **cfg) -> SafeScaleStateMachine:
    clock = iter(range(1_000_000, 10_000_000, 7_000))
    return SafeScaleStateMachine(
        config=_cfg(**cfg), store=store, evidence=evidence,
        thresholds=thresholds if thresholds is not None else STATIC_TPOT_75,
        wall_clock_ms=lambda: next(clock),
        remaining_pods=_remaining(*remaining) if remaining is not None else None,
    )


def _started(machine, *, p95_e2e: float | None = None, hide: bool = True):
    machine.start_probe(
        model=MODEL, pods=("m-1",), now_ms=START,
        window_inputs=ProbeWindowInputs(p95_e2e_ms=p95_e2e) if p95_e2e else None,
    )
    if hide:
        assert machine.mark_hidden(MODEL, pods=("m-1",))
    return machine.active_probe(MODEL)


# ============================================================ 1. dedupe + pre-hide gate
def test_one_observation_per_snapshot_and_the_journal_only_grows_per_snapshot() -> None:
    store = FakeProbeStore()
    machine = _machine(FakeEvidence([_window(50)]), store=store)
    probe = _started(machine)
    for _ in range(5):  # the 2 s loop re-reads the same 10 s snapshot
        assert machine.observe(MODEL, _obs(110_000), now_ms=110_000).reason == "probe_pending"
    assert len(machine.active_probe(MODEL).observations) == 1
    assert len(store.journal[probe.request_id]) == 1
    machine.observe(MODEL, _obs(120_000), now_ms=120_000)
    assert [o.window_end_ms for o in machine.active_probe(MODEL).observations] == [110_000, 120_000]
    assert len(store.journal[probe.request_id]) == 2


def test_donor_health_still_runs_on_every_tick_of_a_repeated_snapshot() -> None:
    machine = _machine(FakeEvidence([_window(50)]))
    _started(machine, p95_e2e=30_000.0)
    assert machine.observe(MODEL, _obs(110_000, gateway=(100.0, 0.0)), now_ms=110_000).status == "probing"
    # Same snapshot, fresh gateway counters: 10 % errors over 50 requests -> rollback.
    decision = machine.observe(MODEL, _obs(110_000, gateway=(150.0, 5.0)), now_ms=110_000)
    assert (decision.status, decision.reason) == ("rollback", "donor_health")
    assert decision.details["rollback_reason"]["code"] == "donor_health"
    assert len(machine.active_probe(MODEL).observations) == 1


def test_a_pre_hide_snapshot_never_triggers_the_immediate_rollback() -> None:
    machine = _machine(FakeEvidence([_window(50)]))
    _started(machine, p95_e2e=30_000.0)  # W = 60 s -> deadline 160 s
    # Windows (80, 110], (90, 120], (100, 130] overlap the hide (103 s): recorded, not judged.
    for ts in (110_000, 120_000, 130_000):
        decision = machine.observe(MODEL, _obs(ts, tpot=500.0), now_ms=ts)
        assert decision.reason == "probe_pending", ts
    # (110, 140] follows the hide completely: judged.
    decision = machine.observe(MODEL, _obs(140_000, tpot=500.0), now_ms=140_000)
    assert (decision.status, decision.reason) == ("rollback", "slo_violation")
    reason = decision.details["rollback_reason"]
    assert (reason["code"], reason["metrics"], reason["window_start_ms"], reason["evidence_start_ms"]) == (
        "slo_violation", ["tpot"], 110_000, 110_000,
    )


def test_no_immediate_rollback_before_the_hide_is_confirmed() -> None:
    machine = _machine(FakeEvidence([_window(50)]))
    _started(machine, p95_e2e=30_000.0, hide=False)
    assert machine.observe(MODEL, _obs(140_000, tpot=500.0), now_ms=140_000).reason == "probe_pending"


# ============================================================ 2. evidence window + extensions
def test_the_evidence_window_starts_at_the_first_boundary_after_the_hide_and_ends_at_the_newest_snapshot() -> None:
    evidence = FakeEvidence([_window(50)])
    machine = _machine(evidence)
    _started(machine)  # W = 20 s -> deadline 120 s
    machine.observe(MODEL, _obs(110_000), now_ms=110_000)
    decision = machine.observe(MODEL, _obs(120_000), now_ms=120_000)
    assert decision.status == "commit"
    assert evidence.reads == [{"model": MODEL, "start_ms": 110_000, "end_ms": 120_000, "exclude": ("m-1",)}]
    assert decision.details["latency_gate"] == "evaluated"
    assert decision.details["latency_samples"] == 50


def test_evidence_latency_violation_rolls_back_even_though_the_tail_snapshots_look_fine() -> None:
    machine = _machine(FakeEvidence([_window(50, tpot=900.0)]))
    _started(machine)
    machine.observe(MODEL, _obs(110_000), now_ms=110_000)
    decision = machine.observe(MODEL, _obs(120_000), now_ms=120_000)
    assert (decision.status, decision.reason) == ("rollback", "formal_commit_gate_failed")
    reason = decision.details["rollback_reason"]
    assert reason["code"] == "formal_commit_gate_failed" and reason["gates"] == ["latency"]
    assert reason["latency"]["evidence_tpot_p95_ms"] == 900.0


def test_pre_hide_slowness_in_the_tail_no_longer_blocks_the_commit() -> None:
    # Old gate: the tail snapshots (windows reaching 30 s back) carried the pre-hide
    # slowness and rolled back; now only the post-hide evidence is judged for latency.
    machine = _machine(FakeEvidence([_window(50)]))
    _started(machine)
    machine.observe(MODEL, _obs(110_000, tpot=3_000.0), now_ms=110_000)
    decision = machine.observe(MODEL, _obs(120_000, tpot=3_000.0), now_ms=120_000)
    assert decision.status == "commit"


def test_short_evidence_extends_the_deadline_one_gateway_period_until_enough_samples() -> None:
    evidence = FakeEvidence([_window(8), _window(15), _window(26)])
    store = FakeProbeStore()
    machine = _machine(evidence, store=store)
    probe = _started(machine)
    machine.observe(MODEL, _obs(110_000), now_ms=110_000)
    first = machine.observe(MODEL, _obs(120_000), now_ms=120_000)
    assert (first.status, first.reason) == ("probing", "evidence_extended")
    assert (first.details["extensions"], first.details["deadline_ms"], first.details["extend_reason"]) == (
        1, 130_000, "insufficient_samples",
    )
    assert store.records[probe.request_id]["deadline_ms"] == 130_000
    assert store.records[probe.request_id]["extensions"] == 1
    # A repeated snapshot does not re-read or re-extend.
    assert machine.observe(MODEL, _obs(120_000), now_ms=120_000).reason == "probe_pending"
    assert machine.observe(MODEL, _obs(130_000), now_ms=130_000).reason == "evidence_extended"
    decision = machine.observe(MODEL, _obs(140_000), now_ms=140_000)
    assert decision.status == "commit"
    assert (decision.details["extensions"], decision.details["latency_samples"]) == (2, 26)
    assert [read["end_ms"] for read in evidence.reads] == [120_000, 130_000, 140_000]
    assert {read["start_ms"] for read in evidence.reads} == {110_000}


def test_at_the_ceiling_an_idle_model_commits() -> None:
    machine = _machine(FakeEvidence([_window(0, tpot=None)]))
    _started(machine)
    for ts in range(110_000, 170_000, 10_000):
        decision = machine.observe(MODEL, _obs(ts, z=None, traffic=False), now_ms=ts)
    assert (decision.status, decision.reason) == ("commit", "formal_commit_gate_passed")
    assert decision.details["latency_gate"] == "skipped"
    assert decision.details["latency_skip_reason"] == "idle"
    assert decision.details["extensions"] == 4  # 120 -> 130 -> 140 -> 150 -> 160 s (= start + 60 s)
    assert decision.details["clamped"] is True


def test_at_the_ceiling_with_traffic_the_few_samples_are_judged_and_z_decides() -> None:
    # 2026-09-29 review: the latency gate is no longer skipped at the ceiling - the few
    # samples there are are judged (max of per pod and pooled, no minimum).
    ok = _machine(FakeEvidence([_window(5, tpot=60.0)]))
    _started(ok)
    for ts in range(110_000, 170_000, 10_000):
        decision = ok.observe(MODEL, _obs(ts, z=1.5), now_ms=ts)
    assert decision.status == "commit"
    assert decision.details["latency_gate"] == "evaluated_low_samples"
    assert (decision.details["low_sample_commit"], decision.details["latency_samples"]) == (True, 5)
    slow = _machine(FakeEvidence([_window(5, tpot=500.0)]))
    _started(slow)
    for ts in range(110_000, 170_000, 10_000):
        decision = slow.observe(MODEL, _obs(ts, z=1.5), now_ms=ts)
    assert (decision.status, decision.reason) == ("rollback", "formal_commit_gate_failed")
    assert decision.details["rollback_reason"]["gates"] == ["latency"]
    low = _machine(FakeEvidence([_window(5, tpot=60.0)]))
    _started(low)
    for ts in range(110_000, 170_000, 10_000):
        decision = low.observe(MODEL, _obs(ts, z=0.6), now_ms=ts)
    assert (decision.status, decision.reason) == ("rollback", "formal_commit_gate_failed")
    assert decision.details["rollback_reason"]["gates"] == ["z_below_tau_low"]


def test_an_unconfirmed_hide_extends_then_rolls_back_at_the_ceiling() -> None:
    machine = _machine(FakeEvidence([_window(50)]))
    _started(machine, hide=False)
    for ts in range(110_000, 170_000, 10_000):
        decision = machine.observe(MODEL, _obs(ts), now_ms=ts)
    assert (decision.status, decision.reason) == ("rollback", "hide_unconfirmed")


def test_an_unreadable_evidence_window_fails_closed() -> None:
    class Broken(FakeEvidence):
        def read(self, *args, **kwargs):
            raise ConnectionError("redis down")

    machine = _machine(Broken([_window(50)]))
    _started(machine)
    machine.observe(MODEL, _obs(110_000), now_ms=110_000)
    decision = machine.observe(MODEL, _obs(120_000), now_ms=120_000)
    assert (decision.status, decision.reason) == ("rollback", "evidence_unavailable")


# ============================================================ 3. the 60 s ceiling
def test_w_is_capped_at_the_registry_ceiling_and_extensions_never_pass_it() -> None:
    terms = calc_probe_window_details(ProbeWindowInputs(p95_e2e_ms=45_000.0), hidden_count=1, config=_cfg())
    assert (terms["W"], terms["W_max"], terms["clamped"], terms["dominant"]) == (60_000.0, 60_000.0, True, "ceiling")
    machine = _machine(FakeEvidence([_window(3)]))
    probe = _started(machine, p95_e2e=45_000.0)
    assert probe.deadline_ms == START + 60_000
    for ts in range(110_000, 170_000, 10_000):
        decision = machine.observe(MODEL, _obs(ts), now_ms=ts)
    # No extension possible: judged at 160 s with the latency gate skipped.
    assert decision.status == "commit"
    assert (decision.details["extensions"], decision.details["clamped"], decision.details["window_clamped"]) == (
        0, True, True,
    )


def test_ceiling_multiplier_and_floor_come_from_config_and_registry(tmp_path) -> None:
    cfg = ControllerConfig.from_env({})
    assert cfg.safescale.window_ceiling_ms == 60_000.0
    assert (cfg.safescale.min_window_ms, cfg.safescale.e2e_multiplier) == (20_000.0, 2.0)
    # The probe's SLO mode is the registry's one switch (repo registry: fixed, the live one).
    assert load_registry(str(TRE_DIR / "deploy" / "registry.yaml")).slo_mode() == "fixed"
    assert cfg.safescale.min_commit_samples == 20
    assert cfg.safescale.evidence_clock_tolerance_ms == 20_000.0
    raw = yaml.safe_load((TRE_DIR / "deploy" / "registry.yaml").read_text(encoding="utf-8"))
    raw["slo_mode"] = "idle_multiple"
    raw["safescale"] = {"window_ceiling_s": 90, "min_commit_samples": 30, "evidence_clock_tolerance_s": 15}
    path = tmp_path / "r.yaml"
    path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    tuned = ControllerConfig.from_env({
        "TRE_REGISTRY_PATH": str(path), "SAFE_SCALE_E2E_MULTIPLIER": "3", "SAFE_SCALE_WINDOW_FLOOR_MS": "25000",
    }).safescale
    assert load_registry(str(path)).slo_mode() == "idle_multiple"
    assert (tuned.window_ceiling_ms, tuned.min_commit_samples) == (90_000.0, 30)
    assert (tuned.evidence_clock_tolerance_ms, tuned.e2e_multiplier, tuned.min_window_ms) == (15_000.0, 3.0, 25_000.0)


def test_a_legacy_registry_whose_probe_and_scorer_modes_disagree_is_refused(tmp_path) -> None:
    # The live registry before 2026-11-04: probe safescale.slo_mode fixed, scorer / labels
    # models[].slo.ttft_slo_mode slowdown. Two switches -> the controller refuses it.
    raw = yaml.safe_load((TRE_DIR / "deploy" / "registry.yaml").read_text(encoding="utf-8"))
    raw.pop("slo_mode")
    raw["safescale"] = {"slo_mode": "fixed"}
    for model in raw["models"]:
        model["slo"]["ttft_slo_mode"] = "slowdown"
    path = tmp_path / "legacy.yaml"
    path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    with pytest.raises(ValueError, match="slo_mode"):
        ControllerConfig.from_env({"TRE_REGISTRY_PATH": str(path)})
    raw["safescale"] = {"slo_mode": "labels"}  # agreeing legacy keys: that mode
    path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    ControllerConfig.from_env({"TRE_REGISTRY_PATH": str(path)})  # accepted
    assert load_registry(str(path)).slo_mode() == "slowdown"


def test_registry_safescale_section_parsing() -> None:
    assert parse_safescale_config(None) == SafeScaleRegistryConfig()
    assert load_registry(str(TRE_DIR / "deploy" / "registry.yaml")).safescale() == SafeScaleRegistryConfig()
    for bad in ({"window_ceiling_s": 0}, {"min_commit_samples": 2.5},
                {"min_commit_samples": -1}, {"evidence_clock_tolerance_s": -3}):
        with pytest.raises(ValueError):
            parse_safescale_config(bad)


# ============================================================ 4. thresholds
def _slo_registry(**slo) -> Registry:
    values = dict(ttft_p95_ms=800.0, tpot_p95_ms=90.0, e2e_p95_ms=12_000.0, ttft_idle_c_ms=40.0,
                  ttft_idle_b_ms_per_token=0.05, ttft_slowdown_k=5.0, ttft_floor_ms=500.0)
    values.update(slo)
    spec = ModelSpec(
        name=MODEL, weights_path="/w", tp_size=1, min_replicas=1, max_replicas=2, vllm_image="img",
        slo=SloSpec(**values),
        trs=TrsParams(w_p=0.04, w_d=1.0, lambda_wait=2.0, qmin=1.0, ema_alpha=0.0, theta_m=100.0, tau_crit=0.8,
                      tau_low=1.0, tau_high=1.25, qsat=4.0, epsat=0.05, hsat=1),
    )
    return Registry(ClusterTopology(nodes=(NodeSpec(name="n", gpus=4, two_gpu_slots=((0, 1), (2, 3))),)), [spec],
                    slo_mode="slowdown")


def test_the_probe_threshold_is_the_modes_tpot_slo_and_nothing_else() -> None:
    # SafeScale judges TPOT only (user 2026-11-04): the resolver gives TPOT_slo, no TTFT.
    assert RegistryThresholds(_slo_registry()).resolve(MODEL) == {"mode": "slowdown", "tpot_ms": 90.0,
                                                                   "source": "registry"}
    shipped = load_registry(str(TRE_DIR / "deploy" / "registry.yaml"))
    model = shipped.models()[0].name
    for mode in ("fixed", "slowdown", "idle_multiple"):
        got = RegistryThresholds(shipped, mode=mode).resolve(model)
        assert got["tpot_ms"] == pytest.approx(label_def_for_model(model, mode=mode, registry=shipped).tpot_p95_ms)
        assert "ttft_ms" not in got


def test_fixed_mode_uses_models_slo_and_an_unbuildable_slo_raises() -> None:
    fixed = RegistryThresholds(_slo_registry(), mode="fixed").resolve(MODEL)
    assert (fixed["mode"], fixed["tpot_ms"]) == ("fixed", 90.0)
    # No idle fit -> the mode's block cannot be built: no fallback, the error propagates.
    with pytest.raises(SloConfigError):
        RegistryThresholds(_slo_registry(ttft_idle_c_ms=None, ttft_idle_b_ms_per_token=None)).resolve(MODEL)


def test_static_thresholds_resolve_the_same_values_for_every_model() -> None:
    record = StaticThresholds(80.0).resolve(MODEL)
    assert record == {"mode": "static", "tpot_ms": 80.0, "source": "static"}


# ============================================================ 2+5. the real reader (fake redis)
class DocRedis:
    def __init__(self, *, now_ms: int = HIDE) -> None:
        self.sets: dict[str, set] = {}
        self.zsets: dict[str, list] = {}
        self.now_ms = now_ms

    def time(self):
        return (self.now_ms // 1000, (self.now_ms % 1000) * 1000)

    def smembers(self, key):
        return set(self.sets.get(key, set()))

    def zrangebyscore(self, key, minimum, maximum):
        return [m for s, m in self.zsets.get(key, []) if float(minimum) <= s <= float(maximum)]

    def zrevrangebyscore(self, key, maximum, minimum, start=0, num=None, withscores=False):
        rows = sorted(self.zsets.get(key, []), key=lambda item: -item[0])[start:]
        rows = rows[:num] if num is not None else rows
        return [(m, s) for s, m in rows] if withscores else [m for _, m in rows]

    def add(self, pod: str, ts: int, *, fast: int, slow: int, prompt: float, model: str = MODEL,
            ttft_s: float = 0.05, tpot: str = "samples") -> None:
        """Cumulative counts: ``fast`` / ``slow`` requests with a 10 ms / 500 ms TPOT (the
        judged latency); every request's TTFT is ``ttft_s`` (not judged by SafeScale).
        ``tpot``: ``samples`` (one per request), ``empty`` (the family with no sample: one-
        token outputs) or ``absent`` (no TPOT family in the doc)."""
        self.sets.setdefault(pods_key(model), set()).add(pod)
        count = fast + slow
        ttft_le = next(le for le in (0.1, 5.0, 60.0) if ttft_s <= le)
        doc = {
            "pod_name": pod,
            "timestamp": ts,
            "model_counter_metrics": {f"{model}/prompt_tokens_total": prompt, f"{model}/generation_tokens_total": 0.0},
            "model_histogram_metrics": {
                f"{model}/request_prompt_tokens": {"sum": prompt, "count": count, "buckets": {"+Inf": count}},
                f"{model}/time_to_first_token_seconds": {
                    "sum": count * ttft_s, "count": count,
                    "buckets": {**{str(le): (count if le >= ttft_le else 0) for le in (0.1, 5.0, 60.0)},
                                "+Inf": count},
                },
            },
        }
        if tpot != "absent":
            n_fast, n_all = (fast, count) if tpot == "samples" else (0, 0)
            doc["model_histogram_metrics"][f"{model}/time_per_output_token_seconds"] = {
                "sum": (fast * 0.01 + slow * 0.5) if tpot == "samples" else 0.0, "count": n_all,
                "buckets": {"0.02": n_fast, "1.0": n_all, "+Inf": n_all},
            }
        self.zsets.setdefault(hist_key(pod), []).append((float(ts), json.dumps(doc)))


def _history(redis: DocRedis, *, skip: tuple[int, ...] = ()) -> None:
    # m-0 (stays): slow before the hide, fast after; m-1 (probe pod) and m-2 (asleep): slow.
    cumulative = {"m-0": [0, 0, 0.0], "m-1": [0, 0, 0.0], "m-2": [0, 0, 0.0]}
    for ts in range(70_000, 140_001, 10_000):
        for pod, row in cumulative.items():
            if pod == "m-0":
                if ts <= 110_000:
                    row[1] += 30  # pre-hide (and hide-straddling) slow requests
                else:
                    row[0] += 15
                row[2] += 15 * 1_000.0 if ts > 110_000 else 30 * 400.0
            else:
                row[1] += 10
                row[2] += 10 * 9_000.0
            if pod == "m-0" and ts in skip:
                continue  # the gateway missed this tick for m-0
            redis.add(pod, ts, fast=row[0], slow=row[1], prompt=row[2])


def _reader(redis: DocRedis) -> MetricsEvidenceReader:
    # as the controller builds it (app.py): latency histograms only, no instant docs
    store = MetricsStore(redis, _slo_registry(), instant_sample_interval_ms=10_000, min_latency_samples=10,
                         histogram_lookback_ms=0, read_instant=False)
    return MetricsEvidenceReader(store, redis_client=redis, sleeping_pods=lambda model: {"m-2"})


def test_reader_differences_the_remaining_pods_from_the_first_boundary_after_the_hide() -> None:
    redis = DocRedis()
    _history(redis)
    window = _reader(redis).read(MODEL, start_ms=110_000, end_ms=130_000, exclude_pods=("m-1",))
    assert window.pods == ("m-0",) and window.excluded_pods == ("m-1", "m-2")
    assert window.request_count == 30  # 15 + 15 after the 110 s boundary doc, none before it
    assert window.first_doc_ts_ms == {"m-0": 110_000}  # the boundary doc is the baseline
    everything = MetricsEvidenceReader(_reader(redis)._store, redis_client=redis).read(
        MODEL, start_ms=110_000, end_ms=130_000, exclude_pods=()
    )
    assert everything.request_count == 70


def test_hide_anchor_is_redis_time_with_the_newest_gateway_doc() -> None:
    redis = DocRedis(now_ms=103_250)
    for pod in ("m-0", "m-1"):
        redis.add(pod, 90_000, fast=1, slow=0, prompt=1.0)
        redis.add(pod, 100_000, fast=2, slow=0, prompt=2.0)
    anchor = _reader(redis).hide_anchor(MODEL)
    assert (anchor.ts_ms, anchor.source, anchor.newest_doc_ts_ms) == (103_250, "redis_time", 100_000)


def _real_machine(redis: DocRedis) -> SafeScaleStateMachine:
    machine = SafeScaleStateMachine(
        config=_cfg(), evidence=_reader(redis), thresholds=RegistryThresholds(_slo_registry()),
        wall_clock_ms=lambda: 0, remaining_pods=_remaining("m-0"),
    )
    machine.start_probe(model=MODEL, pods=("m-1",), now_ms=START)
    machine.mark_hidden(MODEL, pods=("m-1",), anchor=HideAnchor(HIDE, "redis_time", 100_000))
    return machine


def test_one_missed_boundary_tick_is_tolerated_the_delta_starts_at_the_next_doc() -> None:
    redis = DocRedis()
    _history(redis, skip=(110_000,))
    machine = _real_machine(redis)
    machine.observe(MODEL, _obs(110_000), now_ms=110_000)
    assert machine.observe(MODEL, _obs(120_000), now_ms=120_000).reason == "evidence_extended"  # 1 doc: n = 0
    assert machine.observe(MODEL, _obs(130_000), now_ms=130_000).reason == "evidence_extended"  # n = 15
    decision = machine.observe(MODEL, _obs(140_000), now_ms=140_000)
    assert decision.status != "commit"  # no per-request TPOT in the gateway docs
    # Never the pre-hide 100 s doc as the baseline (no lookback).
    assert (decision.details["evidence_first_doc_ts_ms"], decision.details["latency_samples"]) == (120_000, 30)


def test_two_missed_ticks_fail_closed() -> None:
    redis = DocRedis()
    _history(redis, skip=(110_000, 120_000))
    machine = _real_machine(redis)
    machine.observe(MODEL, _obs(110_000), now_ms=110_000)
    machine.observe(MODEL, _obs(120_000), now_ms=120_000)  # no m-0 doc yet: extended
    decision = machine.observe(MODEL, _obs(130_000), now_ms=130_000)
    assert (decision.status, decision.reason) == ("rollback", "evidence_clock_skew")
    reason = decision.details["rollback_reason"]
    assert (reason["check"], reason["first_doc_ts_ms"], reason["latest_allowed_ms"]) == (
        "first_doc_outside", {"m-0": 130_000}, 120_000,
    )


def test_the_evidence_reader_refuses_a_store_with_histogram_lookback() -> None:
    store = MetricsStore(DocRedis(), _slo_registry(), instant_sample_interval_ms=10_000, histogram_lookback_ms=90_000)
    with pytest.raises(ValueError):
        MetricsEvidenceReader(store)


def test_real_reader_never_commits_the_gateway_docs_lack_the_per_request_tpot() -> None:
    # User 2026-11-05: SafeScale judges the per-request TPOT; the gateway docs carry only
    # the per-token TPOT. Healthy post-hide traffic still never commits on this path.
    redis = DocRedis()
    _history(redis)
    machine = _real_machine(redis)
    decisions = [machine.observe(MODEL, _obs(ts), now_ms=ts) for ts in range(110_000, 200_001, 10_000)]
    assert all(d.status != "commit" for d in decisions)
    final = next(d for d in decisions if d.status == "rollback")
    assert final.details["rollback_reason"]["code"] == "evidence_incomplete:pods_incomplete"
    assert final.details["rollback_reason"]["detail"] == {"m-0": "no_tpot_histogram"}


# ============================================================ 5. clock check
def test_first_doc_outside_the_tolerance_fails_closed() -> None:
    machine = _machine(FakeEvidence([_window(50, first_doc=140_000)]))  # > hide + 20 s
    _started(machine)
    machine.observe(MODEL, _obs(110_000), now_ms=110_000)
    decision = machine.observe(MODEL, _obs(120_000), now_ms=120_000)
    assert (decision.status, decision.reason) == ("rollback", "evidence_clock_skew")
    assert decision.details["rollback_reason"]["check"] == "first_doc_outside"


def test_the_evidence_start_follows_the_gateway_stamps_not_redis_time() -> None:
    # Redis 160 s ahead of the gateway (Redis on the fast node): S comes from the doc
    # stamps, so the evidence is unaffected; the skew is alerted, not acted on.
    ahead = HideAnchor(ts_ms=100_000 + 160_000, source="redis_time", newest_doc_ts_ms=100_000,
                       controller_ts_ms=103_000)
    evidence = FakeEvidence([_window(50)], anchor=ahead)
    machine = _machine(evidence)
    probe = _started(machine)
    assert probe.window_terms["clock_skew_alert"] is True
    assert probe.window_terms["gateway_offset_ms"] == 160_000
    assert probe.deadline_ms == START + 20_000  # the window counts on the controller clock
    machine.observe(MODEL, _obs(110_000), now_ms=110_000)
    decision = machine.observe(MODEL, _obs(120_000), now_ms=120_000)
    assert decision.status == "commit"
    assert evidence.reads[0]["start_ms"] == 110_000


def test_a_gateway_ahead_of_the_controller_never_yields_evidence_and_rolls_back_at_the_ceiling() -> None:
    # node9-style skew: the gateway stamps docs 160 s ahead of Redis TIME / the controller.
    ahead = HideAnchor(ts_ms=HIDE, source="redis_time", newest_doc_ts_ms=HIDE + 160_000, controller_ts_ms=HIDE)
    machine = _machine(FakeEvidence([_window(50)], anchor=ahead))
    probe = _started(machine)
    assert probe.window_terms["clock_skew_alert"] is True
    for ts in range(110_000, 170_000, 10_000):
        decision = machine.observe(MODEL, _obs(ts), now_ms=ts)
    assert (decision.status, decision.reason) == ("rollback", "evidence_empty")


def test_an_unverifiable_anchor_fails_closed() -> None:
    broken = HideAnchor(ts_ms=HIDE, source="redis_time", newest_doc_ts_ms=None, newest_doc_error=True)
    machine = _machine(FakeEvidence([_window(50)], anchor=broken))
    _started(machine)
    # Rolled back at the next tick, not at the deadline.
    decision = machine.observe(MODEL, _obs(110_000), now_ms=110_000)
    assert (decision.status, decision.reason) == ("rollback", "evidence_clock_skew")
    assert decision.details["rollback_reason"]["check"] == "anchor_unverified"


def test_clock_only_anchors_fail_closed_with_an_evidence_source() -> None:
    class Raising(FakeEvidence):
        def hide_anchor(self, model):
            raise ConnectionError("redis down")

    machine = _machine(Raising([_window(50)]))
    _started(machine)
    assert machine.active_probe(MODEL).hide_anchor.newest_doc_error is True
    assert machine.observe(MODEL, _obs(110_000), now_ms=110_000).reason == "evidence_clock_skew"


def test_no_doc_of_any_remaining_pod_by_the_ceiling_rolls_back_instead_of_committing() -> None:
    empty = {**_window(0, tpot=None, first_doc=None), "pods": (), "judged_count": 0.0}
    for traffic in (True, False):
        machine = _machine(FakeEvidence([empty]))
        _started(machine)
        for ts in range(110_000, 170_000, 10_000):
            decision = machine.observe(MODEL, _obs(ts, traffic=traffic, z=2.0 if traffic else None), now_ms=ts)
        assert (decision.status, decision.reason) == ("rollback", "evidence_empty"), traffic
        assert decision.details["rollback_reason"]["check"] == "no_docs"


def test_a_restored_probe_keeps_its_late_window_base() -> None:
    late = HideAnchor(ts_ms=START + 45_500, source="redis_time", newest_doc_ts_ms=START + 40_000,
                      controller_ts_ms=START + 45_500)
    store = FakeProbeStore()
    machine = _machine(FakeEvidence([_window(50)], anchor=late), store=store)
    probe = _started(machine)
    restored = SafeScaleStateMachine(thresholds=STATIC_TPOT_75, config=_cfg(), store=FakeProbeStore(unresolved=[store.records[probe.request_id]]),
                                     evidence=FakeEvidence([_window(50)]))
    assert restored.restore() == 1
    again = restored.active_probe(MODEL)
    assert (again.window_base_ms, again.deadline_ms, again.hide_anchor.controller_ts_ms) == (
        START + 40_000, START + 60_000, START + 45_500,
    )


# ============================================================ review follow-ups
def test_idle_needs_no_requests_in_flight_either() -> None:
    # n = 0 but the tail shows traffic (queue): not idle - stalled, rolled back.
    machine = _machine(FakeEvidence([_window(0, tpot=None)]))
    _started(machine)
    for ts in range(110_000, 170_000, 10_000):
        decision = machine.observe(MODEL, _obs(ts, z=None, traffic=True), now_ms=ts)
    assert (decision.status, decision.reason) == ("rollback", "insufficient_evidence:stalled")
    assert decision.details["rollback_reason"]["code"] == "insufficient_evidence:stalled"


def test_a_snapshot_clock_jump_does_not_burn_extensions_on_the_same_evidence() -> None:
    evidence = FakeEvidence([_window(8)])
    machine = _machine(evidence)
    _started(machine)
    machine.observe(MODEL, _obs(110_000), now_ms=110_000)
    # Stale hold, then the next published snapshot is 40 s later.
    first = machine.observe(MODEL, _obs(150_000), now_ms=150_000)
    assert (first.reason, first.details["deadline_ms"]) == ("evidence_extended", 160_000)
    for _ in range(4):  # the 2 s loop re-reads the same snapshot
        assert machine.observe(MODEL, _obs(150_000), now_ms=150_000).reason == "probe_pending"
    assert machine.active_probe(MODEL).extensions == 1 and len(evidence.reads) == 1


def test_n_counts_only_the_pods_whose_p95_is_judged() -> None:
    # 24 requests spread over pods below the per-pod p95 minimum: no p95 is judged.
    machine = _machine(FakeEvidence([{**_window(24, tpot=None), "judged_count": 0.0}]))
    _started(machine)
    machine.observe(MODEL, _obs(110_000), now_ms=110_000)
    decision = machine.observe(MODEL, _obs(120_000), now_ms=120_000)
    assert (decision.reason, decision.details["extend_reason"]) == ("evidence_extended", "p95_unavailable")
    # A slow pod below the per-pod minimum does not let a fast pod fill the quota alone.
    partial = _machine(FakeEvidence([{**_window(20), "judged_count": 11.0}]))
    _started(partial)
    partial.observe(MODEL, _obs(110_000), now_ms=110_000)
    assert partial.observe(MODEL, _obs(120_000), now_ms=120_000).reason == "evidence_extended"


def test_a_late_hide_moves_the_window_with_it() -> None:
    late = HideAnchor(ts_ms=START + 45_500, source="redis_time", newest_doc_ts_ms=START + 40_000,
                      controller_ts_ms=START + 45_500)
    machine = _machine(FakeEvidence([_window(50, first_doc=START + 50_000)], anchor=late))
    probe = _started(machine)
    assert (probe.window_base_ms, probe.deadline_ms) == (START + 40_000, START + 60_000)
    for ts in range(110_000, 160_000, 10_000):
        assert machine.observe(MODEL, _obs(ts), now_ms=ts).status == "probing"
    decision = machine.observe(MODEL, _obs(160_000), now_ms=160_000)
    assert decision.status == "commit"
    assert decision.details["evidence_start_ms"] == START + 50_000


def test_registry_safescale_section_ignores_unknown_keys_with_a_warning(caplog) -> None:
    # Forward compatibility (2026-09-29 B+D): a key added by a newer release must not
    # stop the SM / UI / an older controller that parse this section too.
    with caplog.at_level("WARNING"):
        parsed = parse_safescale_config({"window_ceiling": 90, "min_commit_samples": 5})
    assert parsed == SafeScaleRegistryConfig(min_commit_samples=5)
    assert "window_ceiling" in caplog.text


# ============================================================ 6. audit fields
AUDIT_KEYS = (
    "evidence_start_ms", "evidence_end_ms", "latency_samples", "latency_gate", "extensions", "clamped",
    "threshold_mode", "tpot_threshold_ms", "probe_wall_clock_ms", "tail_pre_hide_fraction",
)


def test_audit_fields_reach_the_decision_the_record_and_window_terms() -> None:
    store = FakeProbeStore()
    machine = _machine(FakeEvidence([_window(8), _window(50, tpot=900.0)]), store=store)
    probe = _started(machine)
    machine.observe(MODEL, _obs(110_000), now_ms=110_000)
    machine.observe(MODEL, _obs(120_000), now_ms=120_000)
    decision = machine.observe(MODEL, _obs(130_000), now_ms=130_000)
    assert decision.status == "rollback"
    machine.mark_committing(MODEL, status="rollback", reason=decision.reason, now_ms=130_000)
    record = store.records[probe.request_id]
    for where in (decision.details, record["window_terms"], record["terminal_details"]):
        missing = [key for key in AUDIT_KEYS if key not in where]
        assert not missing, missing
        assert where["tail_pre_hide_fraction"] == 0.0  # regression assertion
        assert where["rollback_reason"]["code"] == "formal_commit_gate_failed"
    assert (decision.details["evidence_start_ms"], decision.details["evidence_end_ms"]) == (110_000, 130_000)
    assert (decision.details["extensions"], decision.details["latency_gate"]) == (1, "evaluated")
    assert decision.details["probe_wall_clock_ms"] > 0
    assert record["hide_anchor"] == {"ts_ms": HIDE, "source": "redis_time", "newest_doc_ts_ms": 100_000}
    event = format_evidence_event(MODEL, decision.details)
    assert event.startswith("safescale_evidence:m:start=110000:end=130000:n=50:gate=evaluated:ext=1:clamped=0")
    assert event.endswith(":pre_hide=0")
    json.dumps(record)  # the record stays JSON-serialisable


def test_a_restored_probe_keeps_its_anchor_extensions_and_deduplicated_observations() -> None:
    store = FakeProbeStore()
    machine = _machine(FakeEvidence([_window(8)]), store=store)
    probe = _started(machine)
    machine.observe(MODEL, _obs(110_000), now_ms=110_000)
    machine.observe(MODEL, _obs(120_000), now_ms=120_000)  # extended
    # An older controller journalled one entry per 2 s tick: duplicate it.
    store.journal[probe.request_id].append(dict(store.journal[probe.request_id][-1]))
    restored = SafeScaleStateMachine(
        thresholds=STATIC_TPOT_75,
        config=_cfg(), store=FakeProbeStore(unresolved=[store.records[probe.request_id]],
                                            journal=store.journal), evidence=FakeEvidence([_window(50)])
    )
    assert restored.restore() == 1
    again = restored.active_probe(MODEL)
    assert (again.hide_anchor.ts_ms, again.extensions, again.deadline_ms) == (HIDE, 1, 130_000)
    assert [o.window_end_ms for o in again.observations] == [110_000, 120_000]


# ============================================================ wiring
def test_the_queue_reports_a_confirmed_hide_and_only_a_confirmed_one() -> None:
    done, failed = [], []

    async def scenario(results):
        queue = ActionQueue(ScriptedSM(results=results), on_hide_done=lambda *a: done.append(a),
                            on_hide_failed=lambda *a: failed.append(a))
        queue.submit((HideAction("donor", ("pod-a",), "probe_started", "fairness"),))
        await queue.drain_once()

    asyncio.run(scenario({}))
    assert done == [("donor", ("pod-a",))] and failed == []
    asyncio.run(scenario({"routable:donor": [{"ok": False, "error": "HTTP 409"}]}))
    assert done == [("donor", ("pod-a",))] and len(failed) == 1


def test_mark_hidden_anchors_once_and_only_its_own_probe() -> None:
    evidence = FakeEvidence([_window(50)])
    machine = _machine(evidence)
    _started(machine, hide=False)
    assert not machine.mark_hidden(MODEL, pods=("other",))
    assert machine.mark_hidden(MODEL, pods=("m-1",))
    evidence.anchor = HideAnchor(ts_ms=999_999, source="redis_time")
    assert machine.mark_hidden(MODEL, pods=("m-1",))  # a retried hide does not move it
    assert machine.active_probe(MODEL).hide_anchor.ts_ms == HIDE


def test_the_app_wires_the_evidence_reader_thresholds_and_hide_callback() -> None:
    from test_controller_app import EmptyRedis
    from test_controller_app import REGISTRY_PATH as APP_REGISTRY
    from tre_controller.app import create_controller_dependencies

    cfg = ControllerConfig.from_env({"TRE_REGISTRY_PATH": str(APP_REGISTRY)})
    deps = create_controller_dependencies(cfg, redis_client=EmptyRedis())
    assert isinstance(deps.safescale._evidence, MetricsEvidenceReader)
    assert isinstance(deps.safescale._thresholds, RegistryThresholds)
    assert deps.queue._on_hide_done is not None


def test_the_observation_tick_emits_the_evidence_and_rollback_reason_events() -> None:
    from test_safescale_task import FakeQueue

    machine = _machine(FakeEvidence([_window(50, tpot=900.0)]))
    _started(machine)
    snapshot = lambda ts: MetricsSnapshot(ts_ms=ts, stale=False, models={MODEL: ModelWindowMetrics(
        model=MODEL, window_start_ms=ts - 30_000, window_end_ms=ts, prompt_tokens=0.0, generation_tokens=100.0,
        avg_waiting=0.0, avg_running=1.0, avg_swapping=0.0, kv_cache_hit_rate=0.0, ttft_p95_ms=100.0,
        tpot_p95_ms=10.0, e2e_p95_ms=500.0, routable_pods=1, assigned_replicas=1, per_pod={},
    )})
    registry = _slo_registry()
    run_safescale_observation_tick(snapshot(110_000), queue=FakeQueue(), registry=registry, safescale=machine)
    result = run_safescale_observation_tick(snapshot(120_000), queue=FakeQueue(), registry=registry, safescale=machine)
    assert any(event.startswith("safescale_evidence:m:start=110000:end=120000:n=50:gate=evaluated")
               for event in result.events)
    assert any(event.startswith("safescale_rollback_reason:m:formal_commit_gate_failed:latency")
               for event in result.events)


@pytest.mark.parametrize("mode", ["fixed", "idle_multiple"])
def test_the_app_probe_thresholds_come_from_the_registry(tmp_path, mode) -> None:
    """re-review 2026-11-04: the controller built from a registry judges probes with that registry's
    slo_mode and models[].slo (no built-in threshold, no env override)."""
    from test_controller_app import EmptyRedis
    from tre_controller.app import create_controller_dependencies

    raw = yaml.safe_load((TRE_DIR / "deploy" / "registry.yaml").read_text(encoding="utf-8"))
    raw["slo_mode"] = mode
    path = tmp_path / "registry.yaml"
    path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    reg = load_registry(str(path))
    deps = create_controller_dependencies(ControllerConfig.from_env({"TRE_REGISTRY_PATH": str(path)}),
                                          redis_client=EmptyRedis())
    for model in ("dsqwen-7b", "dsqwen-14b"):
        want = reg.slo_thresholds(model)
        got = deps.safescale._thresholds_for(model)
        assert got == {"mode": mode, "tpot_ms": want.tpot_ms, "source": "registry"}


def test_a_threshold_error_never_commits_the_probe() -> None:
    """re-review 2026-11-04: _thresholds_for has no fallback; a resolver error propagates out of the judge,
    so the commit gate is never passed on it (the same evidence commits with a working resolver, above)."""
    class Broken:
        def resolve(self, model):
            raise SloConfigError(f"{model}: no thresholds")

    machine = _machine(FakeEvidence([_window(50)]), thresholds=Broken())
    _started(machine)
    raised = 0
    for ts in (110_000, 120_000):   # the commit deadline is 120 s
        try:
            decision = machine.observe(MODEL, _obs(ts), now_ms=ts)
        except SloConfigError:
            raised += 1
            continue
        assert decision.status != "commit", decision
    assert raised and machine.active_probe(MODEL).status == "probing"     # not committed, not resolved
