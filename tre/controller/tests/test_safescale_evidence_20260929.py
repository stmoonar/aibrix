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
from tre_common.slo_labels import label_def_for_model
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
)
from tre_controller.store.metrics_store import MetricsStore

from test_action_queue_review3 import ScriptedSM
from test_safescale import FakeProbeStore

TRE_DIR = Path(__file__).resolve().parents[2]
MODEL = "m"
START = 100_000  # probe planned on the snapshot of boundary 100 s
HIDE = 103_000  # the SM confirmed the hide 3 s later (Redis TIME)


# ----------------------------------------------------------------------- helpers
def _cfg(**overrides) -> SafeScaleConfig:
    values = dict(min_window_ms=20_000.0, window_ceiling_ms=60_000.0, hq=0.25, tau_low=1.0)
    values.update(overrides)
    return SafeScaleConfig(**values)


def _obs(ts_ms: int, *, ttft: float = 100.0, tpot: float = 10.0, z: float | None = 2.0, traffic: bool = True,
         window_ms: int = 30_000, prompt: float | None = None, gateway: tuple[float, float] | None = None
         ) -> ProbeObservation:
    return ProbeObservation(
        ts_ms=ts_ms,
        ttft_p95_ms=ttft,
        tpot_p95_ms=tpot,
        z_m=z,
        has_traffic=traffic,
        window_start_ms=ts_ms - window_ms,
        window_end_ms=ts_ms,
        mean_prompt_tokens=prompt,
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
        return EvidenceWindow(**{**window, "start_ms": start_ms, "end_ms": end_ms})


def _window(n: float, *, ttft: float | None = 100.0, tpot: float | None = 10.0, first_doc: int = 110_000,
            prompt_tokens: float | None = None, prompt_count: float | None = None) -> dict:
    return dict(
        start_ms=0, end_ms=0, pods=("m-0",), excluded_pods=("m-1",), ttft_p95_ms=ttft, tpot_p95_ms=tpot,
        ttft_count=n, prompt_tokens=prompt_tokens, prompt_count=prompt_count,
        first_doc_ts_ms={"m-0": first_doc} if first_doc is not None else {},
    )


def _machine(evidence, *, store=None, thresholds=None, **cfg) -> SafeScaleStateMachine:
    clock = iter(range(1_000_000, 10_000_000, 7_000))
    return SafeScaleStateMachine(
        config=_cfg(**cfg), store=store, evidence=evidence, thresholds=thresholds,
        wall_clock_ms=lambda: next(clock),
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
        decision = machine.observe(MODEL, _obs(ts, ttft=5_000.0, tpot=500.0), now_ms=ts)
        assert decision.reason == "probe_pending", ts
    # (110, 140] follows the hide completely: judged.
    decision = machine.observe(MODEL, _obs(140_000, ttft=5_000.0), now_ms=140_000)
    assert (decision.status, decision.reason) == ("rollback", "slo_violation")
    reason = decision.details["rollback_reason"]
    assert (reason["code"], reason["metrics"], reason["window_start_ms"], reason["evidence_start_ms"]) == (
        "slo_violation", ["ttft"], 110_000, 110_000,
    )


def test_no_immediate_rollback_before_the_hide_is_confirmed() -> None:
    machine = _machine(FakeEvidence([_window(50)]))
    _started(machine, p95_e2e=30_000.0, hide=False)
    assert machine.observe(MODEL, _obs(140_000, ttft=5_000.0), now_ms=140_000).reason == "probe_pending"


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
    machine = _machine(FakeEvidence([_window(50, ttft=900.0)]))
    _started(machine)
    machine.observe(MODEL, _obs(110_000), now_ms=110_000)
    decision = machine.observe(MODEL, _obs(120_000, ttft=100.0), now_ms=120_000)
    assert (decision.status, decision.reason) == ("rollback", "formal_commit_gate_failed")
    reason = decision.details["rollback_reason"]
    assert reason["code"] == "formal_commit_gate_failed" and reason["gates"] == ["latency"]
    assert reason["latency"]["evidence_ttft_p95_ms"] == 900.0


def test_pre_hide_slowness_in_the_tail_no_longer_blocks_the_commit() -> None:
    # Old gate: the tail snapshots (windows reaching 30 s back) carried the pre-hide
    # slowness and rolled back; now only the post-hide evidence is judged for latency.
    machine = _machine(FakeEvidence([_window(50, ttft=120.0)]))
    _started(machine)
    machine.observe(MODEL, _obs(110_000, ttft=3_000.0), now_ms=110_000)
    decision = machine.observe(MODEL, _obs(120_000, ttft=3_000.0), now_ms=120_000)
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
    machine = _machine(FakeEvidence([_window(0, ttft=None, tpot=None)]))
    _started(machine)
    for ts in range(110_000, 170_000, 10_000):
        decision = machine.observe(MODEL, _obs(ts, z=None, traffic=False), now_ms=ts)
    assert (decision.status, decision.reason) == ("commit", "formal_commit_gate_passed")
    assert decision.details["latency_gate"] == "skipped"
    assert decision.details["latency_skip_reason"] == "idle"
    assert decision.details["extensions"] == 4  # 120 -> 130 -> 140 -> 150 -> 160 s (= start + 60 s)
    assert decision.details["clamped"] is True


def test_at_the_ceiling_with_traffic_latency_is_skipped_and_z_decides() -> None:
    ok = _machine(FakeEvidence([_window(5, ttft=5_000.0)]))
    _started(ok)
    for ts in range(110_000, 170_000, 10_000):
        decision = ok.observe(MODEL, _obs(ts, z=1.5), now_ms=ts)
    assert decision.status == "commit"
    assert (decision.details["latency_gate"], decision.details["latency_skip_reason"]) == (
        "skipped", "insufficient_samples",
    )
    low = _machine(FakeEvidence([_window(5, ttft=5_000.0)]))
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
    assert (cfg.safescale.slo_mode, cfg.safescale.min_commit_samples) == ("labels", 20)
    assert cfg.safescale.evidence_clock_tolerance_ms == 20_000.0
    assert cfg.safescale.evidence_step_ms == 10_000.0
    raw = yaml.safe_load((TRE_DIR / "deploy" / "registry.yaml").read_text(encoding="utf-8"))
    raw["safescale"] = {"slo_mode": "fixed", "window_ceiling_s": 90, "min_commit_samples": 30,
                        "evidence_clock_tolerance_s": 15}
    path = tmp_path / "r.yaml"
    path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    tuned = ControllerConfig.from_env({
        "TRE_REGISTRY_PATH": str(path), "SAFE_SCALE_E2E_MULTIPLIER": "3", "SAFE_SCALE_WINDOW_FLOOR_MS": "25000",
    }).safescale
    assert (tuned.window_ceiling_ms, tuned.slo_mode, tuned.min_commit_samples) == (90_000.0, "fixed", 30)
    assert (tuned.evidence_clock_tolerance_ms, tuned.e2e_multiplier, tuned.min_window_ms) == (15_000.0, 3.0, 25_000.0)


def test_registry_safescale_section_parsing() -> None:
    assert parse_safescale_config(None) == SafeScaleRegistryConfig()
    assert load_registry(str(TRE_DIR / "deploy" / "registry.yaml")).safescale() == SafeScaleRegistryConfig()
    for bad in ({"slo_mode": "median"}, {"window_ceiling_s": 0}, {"min_commit_samples": 2.5},
                {"min_commit_samples": -1}, {"evidence_clock_tolerance_s": -3}):
        with pytest.raises(ValueError):
            parse_safescale_config(bad)


# ============================================================ 4. thresholds
def _slo_registry(**slo) -> Registry:
    values = dict(ttft_p95_ms=800.0, tpot_p95_ms=90.0, e2e_p95_ms=12_000.0, ttft_idle_c_ms=40.0,
                  ttft_idle_b_ms_per_token=0.05, ttft_slo_mode="slowdown", ttft_slowdown_k=5.0,
                  ttft_floor_ms=500.0)
    values.update(slo)
    spec = ModelSpec(
        name=MODEL, weights_path="/w", tp_size=1, min_replicas=1, max_replicas=2, vllm_image="img",
        slo=SloSpec(**values),
        trs=TrsParams(w_p=0.04, w_d=1.0, lambda_wait=2.0, qmin=1.0, ema_alpha=0.0, theta_m=100.0, tau_crit=0.8,
                      tau_low=1.0, tau_high=1.25, qsat=4.0, epsat=0.05, hsat=1),
    )
    return Registry(ClusterTopology(nodes=(NodeSpec(name="n", gpus=4, two_gpu_slots=((0, 1), (2, 3))),)), [spec])


def test_labels_mode_uses_the_calibration_label_rule_with_the_evidence_prompt_length() -> None:
    labels = RegistryThresholds(_slo_registry())
    long = labels.resolve(MODEL, 4_000.0)  # 5 * (40 + 0.05 * 4000) = 1200
    assert (long["mode"], long["ttft_ms"], long["tpot_ms"]) == ("labels", pytest.approx(1_200.0), 75.0)
    assert labels.resolve(MODEL, 100.0)["ttft_ms"] == 500.0  # 5 * 45 = 225 -> floor 500
    assert labels.resolve(MODEL, None)["ttft_ms"] == 500.0  # L unknown -> the floor
    # The same function the theta labels use, on the shipped registry.
    shipped = load_registry(str(TRE_DIR / "deploy" / "registry.yaml"))
    model = shipped.models()[0].name
    assert RegistryThresholds(shipped).resolve(model, 3_000.0)["ttft_ms"] == pytest.approx(
        label_def_for_model(model, registry=shipped).ttft_slo_ms(3_000.0)
    )


def test_fixed_mode_uses_models_slo_and_env_overrides_either_mode() -> None:
    fixed = RegistryThresholds(_slo_registry(), mode="fixed").resolve(MODEL, 4_000.0)
    assert (fixed["mode"], fixed["ttft_ms"], fixed["tpot_ms"]) == ("fixed", 800.0, 90.0)
    over = RegistryThresholds(_slo_registry(), ttft_override_ms=1_234.0).resolve(MODEL, 4_000.0)
    assert (over["ttft_ms"], over["tpot_ms"], over["source"]) == (1_234.0, 75.0, "env_override")
    # No idle fit -> the label cannot be built: fixed values, recorded.
    fallback = RegistryThresholds(_slo_registry(ttft_idle_c_ms=None, ttft_idle_b_ms_per_token=None)).resolve(MODEL, 1.0)
    assert (fallback["mode"], fallback["ttft_ms"]) == ("fixed", 800.0) and "fallback" in fallback


def test_env_thresholds_are_optional_overrides() -> None:
    assert ControllerConfig.from_env({}).safescale.ttft_p95_slo_ms is None
    assert ControllerConfig.from_env({}).safescale.tpot_p95_slo_ms is None
    tuned = ControllerConfig.from_env({"SAFE_SCALE_TTFT_P95_SLO_MS": "700", "SAFE_SCALE_TPOT_P95_SLO_MS": "80"})
    assert (tuned.safescale.ttft_p95_slo_ms, tuned.safescale.tpot_p95_slo_ms) == (700.0, 80.0)


def test_the_gate_judges_evidence_against_the_labels_threshold_of_its_prompt_length() -> None:
    # 4000-token prompts: TTFT threshold 1200 ms, so a 1000 ms p95 passes (fixed 800 would fail).
    labels = _machine(FakeEvidence([_window(50, ttft=1_000.0, prompt_tokens=200_000.0, prompt_count=50.0)]),
                      thresholds=RegistryThresholds(_slo_registry()))
    _started(labels)
    labels.observe(MODEL, _obs(110_000), now_ms=110_000)
    decision = labels.observe(MODEL, _obs(120_000), now_ms=120_000)
    assert decision.status == "commit"
    assert (decision.details["threshold_mode"], decision.details["mean_prompt_tokens"]) == ("labels", 4_000.0)
    assert decision.details["ttft_threshold_ms"] == pytest.approx(1_200.0)
    fixed = _machine(FakeEvidence([_window(50, ttft=1_000.0, prompt_tokens=200_000.0, prompt_count=50.0)]),
                     thresholds=RegistryThresholds(_slo_registry(), mode="fixed"))
    _started(fixed)
    fixed.observe(MODEL, _obs(110_000), now_ms=110_000)
    assert fixed.observe(MODEL, _obs(120_000), now_ms=120_000).status == "rollback"


def test_snapshot_observations_carry_their_mean_prompt_length() -> None:
    from test_safescale_task import _registry

    window = ModelWindowMetrics(
        model="donor", window_start_ms=0, window_end_ms=30_000, prompt_tokens=6_000.0, generation_tokens=10.0,
        avg_waiting=0.0, avg_running=1.0, avg_swapping=0.0, kv_cache_hit_rate=0.0, ttft_p95_ms=1.0,
        tpot_p95_ms=1.0, e2e_p95_ms=1.0, routable_pods=1, assigned_replicas=1, per_pod={}, request_count=4.0,
    )
    observation = _observation_from_metrics(30_000, window, _registry().model("donor"), "zm")
    assert observation.mean_prompt_tokens == 1_500.0


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

    def add(self, pod: str, ts: int, *, fast: int, slow: int, prompt: float, model: str = MODEL) -> None:
        self.sets.setdefault(pods_key(model), set()).add(pod)
        count = fast + slow
        doc = {
            "pod_name": pod,
            "timestamp": ts,
            "model_histogram_metrics": {
                f"{model}/request_prompt_tokens": {"sum": prompt, "count": count, "buckets": {"+Inf": count}},
                f"{model}/time_to_first_token_seconds": {
                    "sum": fast * 0.05 + slow * 4.0, "count": count,
                    "buckets": {"0.1": fast, "5.0": count, "+Inf": count},
                },
                f"{model}/time_per_output_token_seconds": {
                    "sum": count * 0.01, "count": count, "buckets": {"0.02": count, "+Inf": count},
                },
            },
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
    store = MetricsStore(redis, _slo_registry(), instant_sample_interval_ms=10_000, min_latency_samples=10,
                         histogram_lookback_ms=0)
    return MetricsEvidenceReader(store, redis_client=redis, sleeping_pods=lambda model: {"m-2"})


def test_reader_differences_the_remaining_pods_from_the_first_boundary_after_the_hide() -> None:
    redis = DocRedis()
    _history(redis)
    window = _reader(redis).read(MODEL, start_ms=110_000, end_ms=130_000, exclude_pods=("m-1",))
    assert window.pods == ("m-0",) and window.excluded_pods == ("m-1", "m-2")
    assert window.ttft_count == 30  # 15 + 15 after the 110 s boundary doc, none before it
    assert window.ttft_p95_ms == 100.0  # only fast requests (the slow pods are excluded)
    assert window.first_doc_ts_ms == {"m-0": 110_000}  # the boundary doc is the baseline
    assert window.mean_prompt_tokens == 1_000.0
    # Without the exclusions the probe / sleeping pods' slow requests would dominate.
    everything = MetricsEvidenceReader(_reader(redis)._store, redis_client=redis).read(
        MODEL, start_ms=110_000, end_ms=130_000, exclude_pods=()
    )
    assert everything.ttft_p95_ms == 5_000.0 and everything.ttft_count == 70


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
        wall_clock_ms=lambda: 0,
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
    assert decision.status == "commit"
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


def test_real_reader_commits_on_post_hide_evidence() -> None:
    redis = DocRedis()
    _history(redis)
    machine = _real_machine(redis)
    # Snapshots still show the pre-hide slowness (5 s TTFT): not judged, not in the gate.
    machine.observe(MODEL, _obs(110_000, ttft=5_000.0), now_ms=110_000)
    assert machine.observe(MODEL, _obs(120_000, ttft=5_000.0), now_ms=120_000).reason == "evidence_extended"
    decision = machine.observe(MODEL, _obs(130_000, ttft=5_000.0), now_ms=130_000)
    assert decision.status == "commit"
    assert (decision.details["latency_samples"], decision.details["evidence_ttft_p95_ms"]) == (30, 100.0)
    assert decision.details["tail_pre_hide_fraction"] == 0.0


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
    machine.observe(MODEL, _obs(110_000), now_ms=110_000)
    decision = machine.observe(MODEL, _obs(120_000), now_ms=120_000)
    assert (decision.status, decision.reason) == ("rollback", "evidence_clock_skew")
    assert decision.details["rollback_reason"]["check"] == "anchor_unverified"


# ============================================================ review follow-ups
def test_idle_needs_no_requests_in_flight_either() -> None:
    # n = 0 but the tail shows traffic (queue) and no Z: not idle -> z_missing rollback.
    machine = _machine(FakeEvidence([_window(0, ttft=None, tpot=None)]))
    _started(machine)
    for ts in range(110_000, 170_000, 10_000):
        decision = machine.observe(MODEL, _obs(ts, z=None, traffic=True), now_ms=ts)
    assert (decision.status, decision.reason) == ("rollback", "formal_commit_gate_failed")
    assert decision.details["rollback_reason"]["gates"] == ["z_missing"]
    assert decision.details["latency_skip_reason"] == "insufficient_samples"


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
    machine = _machine(FakeEvidence([{**_window(24, ttft=None, tpot=None), "judged_count": 0.0}]))
    _started(machine)
    machine.observe(MODEL, _obs(110_000), now_ms=110_000)
    decision = machine.observe(MODEL, _obs(120_000), now_ms=120_000)
    assert (decision.reason, decision.details["extend_reason"]) == ("evidence_extended", "p95_unavailable")
    # A slow pod below the per-pod minimum does not let a fast pod fill the quota alone.
    partial = _machine(FakeEvidence([{**_window(20, ttft=100.0), "judged_count": 11.0}]))
    _started(partial)
    partial.observe(MODEL, _obs(110_000), now_ms=110_000)
    assert partial.observe(MODEL, _obs(120_000), now_ms=120_000).reason == "evidence_extended"


def test_reader_judged_count_excludes_pods_below_the_per_pod_minimum() -> None:
    redis = DocRedis()
    for pod, fast in (("m-0", 6), ("m-3", 12)):
        redis.add(pod, 110_000, fast=0, slow=0, prompt=0.0)
        redis.add(pod, 120_000, fast=fast, slow=0, prompt=100.0 * fast)
    window = _reader(redis).read(MODEL, start_ms=110_000, end_ms=120_000, exclude_pods=())
    assert (window.ttft_count, window.judged_count) == (18.0, 12.0)


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


def test_registry_safescale_section_rejects_unknown_keys() -> None:
    with pytest.raises(ValueError):
        parse_safescale_config({"window_ceiling": 90})


# ============================================================ 6. audit fields
AUDIT_KEYS = (
    "evidence_start_ms", "evidence_end_ms", "latency_samples", "latency_gate", "extensions", "clamped",
    "threshold_mode", "ttft_threshold_ms", "tpot_threshold_ms", "probe_wall_clock_ms", "tail_pre_hide_fraction",
)


def test_audit_fields_reach_the_decision_the_record_and_window_terms() -> None:
    store = FakeProbeStore()
    machine = _machine(FakeEvidence([_window(8), _window(50, ttft=900.0)]), store=store)
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

    machine = _machine(FakeEvidence([_window(50, ttft=900.0)]))
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
