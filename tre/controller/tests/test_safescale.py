from __future__ import annotations

from tre_controller.config import SafeScaleConfig
from tre_controller.planning.safescale import (
    ProbeObservation,
    ProbeWindowInputs,
    SafeScaleCommand,
    SafeScaleDecision,
    SafeScaleStateMachine,
    calc_probe_window_details,
    format_window_event,
)


class FakeProbeStore:
    def __init__(self, unresolved: list[dict] | None = None, journal: dict[str, list[dict]] | None = None) -> None:
        self.records: dict[str, dict] = {}
        self.deleted: list[str] = []
        self.unresolved = unresolved or []
        self.journal = journal or {}

    def save_probe(self, request_id: str, record: dict) -> None:
        self.records[request_id] = dict(record)

    def delete_probe(self, request_id: str) -> None:
        self.deleted.append(request_id)
        self.records.pop(request_id, None)

    def list_unresolved_probes(self) -> list[dict]:
        return [dict(item) for item in self.unresolved]

    def append_probe_journal(self, request_id: str, record: dict) -> None:
        self.journal.setdefault(request_id, []).append(dict(record))

    def load_probe_journal(self, request_id: str) -> list[dict]:
        return [dict(item) for item in self.journal.get(request_id, [])]


def _cfg() -> SafeScaleConfig:
    return SafeScaleConfig(
        ttft_p95_slo_ms=1000.0,
        tpot_p95_slo_ms=100.0,
        default_window_ms=60_000.0,
        min_window_ms=15_000.0,
        max_window_ms=300_000.0,
        hq=0.5,
        tau_low=1.0,
    )


def _healthy_observation(ts_ms: int = 61_000) -> ProbeObservation:
    return ProbeObservation(
        ts_ms=ts_ms,
        ttft_p95_ms=500.0,
        tpot_p95_ms=50.0,
        z_m=1.2,
        q_ctl=0.0,
        has_traffic=True,
        avg_gpu_cache_norm=0.5,
    )


def test_safescale_starts_probe_and_persists_hidden_pods() -> None:
    store = FakeProbeStore()
    machine = SafeScaleStateMachine(config=_cfg(), store=store)

    decision = machine.start_probe(
        model="donor",
        pods=("pod-a", "pod-b"),
        now_ms=1_000,
        pending_upscales={"receiver": 2},
    )

    assert decision.status == "probing"
    assert decision.reason == "probe_started"
    assert decision.commands == (
        SafeScaleCommand(kind="hide", model="donor", pods=("pod-a", "pod-b"), reason="probe_started"),
    )
    probe = machine.active_probe("donor")
    assert probe is not None
    assert probe.deadline_ms == 61_000
    assert probe.request_id in store.records
    assert store.records[probe.request_id]["pending_upscales"] == {"receiver": 2}


def test_safescale_rolls_back_immediately_on_slo_violation() -> None:
    store = FakeProbeStore()
    machine = SafeScaleStateMachine(config=_cfg(), store=store)
    machine.start_probe(model="donor", pods=("pod-a",), now_ms=1_000)

    decision = machine.observe(
        "donor",
        ProbeObservation(
            ts_ms=2_000,
            ttft_p95_ms=1200.0,
            tpot_p95_ms=50.0,
            z_m=1.5,
            q_ctl=0.0,
            has_traffic=True,
        ),
        now_ms=2_000,
    )

    assert decision == SafeScaleDecision(
        status="rollback",
        reason="slo_violation",
        commands=(SafeScaleCommand(kind="unhide", model="donor", pods=("pod-a",), reason="slo_violation"),),
    )
    assert machine.active_probe("donor") is not None
    assert store.deleted == []
    assert machine.resolve(
        "donor", status="rollback", reason=decision.reason, now_ms=2_000
    )
    assert machine.active_probe("donor") is None
    record = next(iter(store.records.values()))
    assert record["status"] == "resolved"
    assert record["resolution"] == "rollback"
    assert record["resolved_ts"] == 2.0


def test_safescale_commits_after_deadline_when_tail_is_healthy() -> None:
    store = FakeProbeStore()
    machine = SafeScaleStateMachine(config=_cfg(), store=store)
    machine.start_probe(model="donor", pods=("pod-a",), now_ms=1_000, pending_upscales={"receiver": 1})
    assert machine.observe("donor", _healthy_observation(ts_ms=20_000), now_ms=20_000).status == "probing"

    decision = machine.observe("donor", _healthy_observation(ts_ms=61_000), now_ms=61_000)

    assert decision.status == "commit"
    assert decision.reason == "formal_commit_gate_passed"
    assert decision.commands == (
        SafeScaleCommand(kind="scale_down", model="donor", pods=("pod-a",), delta=-1, reason="formal_commit_gate_passed"),
        SafeScaleCommand(kind="scale_up", model="receiver", delta=1, reason="safescale_followup_upscale"),
    )
    assert machine.active_probe("donor") is not None
    assert store.deleted == []
    assert machine.resolve(
        "donor", status="commit", reason=decision.reason, now_ms=61_000
    )
    assert machine.active_probe("donor") is None
    record = next(iter(store.records.values()))
    assert record["status"] == "resolved"
    assert record["resolution"] == "commit"
    assert record["resolved_ts"] == 61.0


def test_safescale_rolls_back_at_deadline_when_tail_health_fails() -> None:
    machine = SafeScaleStateMachine(config=_cfg(), store=FakeProbeStore())
    machine.start_probe(model="donor", pods=("pod-a",), now_ms=1_000)

    decision = machine.observe(
        "donor",
        ProbeObservation(
            ts_ms=61_000,
            ttft_p95_ms=500.0,
            tpot_p95_ms=50.0,
            z_m=0.8,
            q_ctl=0.0,
            has_traffic=True,
        ),
        now_ms=61_000,
    )

    assert decision.status == "rollback"
    assert decision.reason == "formal_commit_gate_failed"
    assert decision.commands == (
        SafeScaleCommand(kind="unhide", model="donor", pods=("pod-a",), reason="formal_commit_gate_failed"),
    )


def test_safescale_restores_unresolved_probe_and_commits() -> None:
    unresolved = [
        {
            "model": "donor",
            "request_id": "probe-1",
            "pods": ["pod-a"],
            "start_ms": 1_000,
            "deadline_ms": 61_000,
            "status": "probing",
            "pending_upscales": {"receiver": 1},
        }
    ]
    journal = {
        "probe-1": [
            {
                "last_observation": {
                    "ts_ms": 20_000,
                    "ttft_p95_ms": 500.0,
                    "tpot_p95_ms": 50.0,
                    "z_m": 1.2,
                    "q_ctl": 0.0,
                    "has_traffic": True,
                }
            }
        ]
    }
    store = FakeProbeStore(unresolved=unresolved, journal=journal)
    machine = SafeScaleStateMachine(config=_cfg(), store=store)

    assert machine.restore() == 1
    decision = machine.observe("donor", _healthy_observation(ts_ms=61_000), now_ms=61_000)

    assert decision.status == "commit"
    assert decision.commands[0] == SafeScaleCommand(
        kind="scale_down",
        model="donor",
        pods=("pod-a",),
        delta=-1,
        reason="formal_commit_gate_passed",
    )


def test_safescale_restored_tail_blocks_commit_on_prior_latency_violation() -> None:
    unresolved = [
        {
            "model": "donor",
            "request_id": "probe-violation",
            "pods": ["pod-a"],
            "start_ms": 1_000,
            "deadline_ms": 61_000,
            "status": "probing",
        }
    ]
    journal = {
        "probe-violation": [
            {
                "last_observation": {
                    "ts_ms": 20_000,
                    "ttft_p95_ms": 1200.0,
                    "tpot_p95_ms": 50.0,
                    "z_m": 1.4,
                    "q_ctl": 0.0,
                    "has_traffic": True,
                }
            }
        ]
    }
    store = FakeProbeStore(unresolved=unresolved, journal=journal)
    machine = SafeScaleStateMachine(config=_cfg(), store=store)
    assert machine.restore() == 1

    decision = machine.observe("donor", _healthy_observation(ts_ms=61_000), now_ms=61_000)

    assert decision.status == "rollback"
    assert decision.reason == "formal_commit_gate_failed"


# --- A6: adaptive probe window (port of v1 _calc_probe_window_details) -------------------

_V1_CFG = SafeScaleConfig()  # W_lo 60 s, W_hi 120 s, default 60 s, cw2 fallback 60 s, cdec 2


def _inputs(**overrides) -> ProbeWindowInputs:
    # 4 serving pods, Y_m = 30000 weighted tokens over a 30 s window -> arrival 1000/s;
    # Z = 2 -> capacity 2000/s = 500/s per pod; hiding 1 leaves 1500/s -> gap 500/s.
    values = dict(
        p95_e2e_ms=5_000.0,
        p95_tpot_ms=50.0,
        q=10.0,
        y_total=30_000.0,
        y_per_pod=7_500.0,
        z_m=2.0,
        routable_pods=4,
        interval_s=30.0,
    )
    values.update(overrides)
    return ProbeWindowInputs(**values)


def test_window_without_metrics_is_the_default_window() -> None:
    terms = calc_probe_window_details(ProbeWindowInputs(), hidden_count=1, config=_V1_CFG)
    assert terms["W"] == 60_000.0
    assert (terms["W1"], terms["W2"], terms["cW2"], terms["decode_term_ms"]) == (60_000.0, 60_000.0, None, None)
    assert (terms["dominant"], terms["clamped"]) == ("default", None)


def test_window_terms_follow_v1_formula_and_clamp_up_to_w_lo() -> None:
    terms = calc_probe_window_details(_inputs(), hidden_count=1, config=_V1_CFG)
    assert terms["rate_gap_per_second"] == 500.0
    assert terms["cW2"] == 10.0 / 500.0 * 1000.0  # Q / rate_gap in ms
    assert terms["decode_term_ms"] == 100.0  # cdec * p95_tpot
    assert terms["W1"] == 10_000.0  # 2 * p95_e2e
    assert terms["W_raw"] == 10_000.0
    assert (terms["W"], terms["dominant"], terms["clamped"]) == (60_000.0, "e2e", "lo")


def test_window_e2e_term_inside_the_band_and_clamped_at_w_hi() -> None:
    inside = calc_probe_window_details(_inputs(p95_e2e_ms=40_000.0), hidden_count=1, config=_V1_CFG)
    assert (inside["W"], inside["dominant"], inside["clamped"]) == (80_000.0, "e2e", None)
    above = calc_probe_window_details(_inputs(p95_e2e_ms=90_000.0), hidden_count=1, config=_V1_CFG)
    assert (above["W"], above["W_raw"], above["clamped"]) == (120_000.0, 180_000.0, "hi")


def test_window_queue_term_dominates_when_the_post_hide_gap_is_small() -> None:
    import pytest

    # Z = 1.3336: capacity 1333.6/s, 333.4/s per pod; 3 left -> 1000.2/s -> gap 0.2/s.
    terms = calc_probe_window_details(_inputs(q=20.0, z_m=1.3336), hidden_count=1, config=_V1_CFG)
    assert terms["rate_gap_per_second"] == pytest.approx(0.2)
    assert terms["cW2"] == pytest.approx(100_000.0)  # 20 / 0.2 per s
    assert terms["W"] == pytest.approx(100_000.0)
    assert (terms["dominant"], terms["clamped"], terms["cW2_fallback"]) == ("queue", None, False)


def test_window_uses_the_60s_fallback_when_there_is_no_post_hide_spare_rate() -> None:
    # Z <= 1: v1 treats capacity == arrival, so hiding a pod leaves a negative gap -> 0.
    no_spare = calc_probe_window_details(_inputs(z_m=0.9), hidden_count=1, config=_V1_CFG)
    assert no_spare["rate_gap_per_second"] == 0.0
    assert (no_spare["cW2"], no_spare["cW2_fallback"], no_spare["W"], no_spare["dominant"]) == (
        60_000.0,
        True,
        60_000.0,
        "queue",
    )
    # Unknown gap (no token counts) with a queue -> fallback as well; the fallback is capped
    # at W_hi like v1 (min(max_window, cw2_fallback)).
    unknown = calc_probe_window_details(_inputs(y_total=None, y_per_pod=None), hidden_count=1, config=_V1_CFG)
    assert (unknown["rate_gap_per_second"], unknown["cW2"], unknown["cW2_fallback"]) == (None, 60_000.0, True)
    wide = SafeScaleConfig(cw2_fallback_ms=300_000.0)
    capped = calc_probe_window_details(_inputs(z_m=0.9), hidden_count=1, config=wide)
    assert (capped["cW2"], capped["W"], capped["clamped"]) == (120_000.0, 120_000.0, None)
    # A lone pod cannot be hidden with anything left: no gap either.
    lone = calc_probe_window_details(_inputs(routable_pods=1), hidden_count=1, config=_V1_CFG)
    assert lone["rate_gap_per_second"] is None and lone["cW2_fallback"] is True


def test_window_y_per_pod_backs_up_a_missing_y_total_like_v1() -> None:
    terms = calc_probe_window_details(_inputs(y_total=None), hidden_count=1, config=_V1_CFG)
    assert terms["rate_gap_per_second"] == 500.0


def test_start_probe_uses_the_adaptive_window_and_persists_its_terms() -> None:
    store = FakeProbeStore()
    machine = SafeScaleStateMachine(config=_V1_CFG, store=store)

    decision = machine.start_probe(
        model="donor", pods=("pod-a",), now_ms=1_000, window_inputs=_inputs(p95_e2e_ms=45_000.0)
    )

    probe = machine.active_probe("donor")
    assert probe.window_ms == 90_000.0
    assert probe.deadline_ms == 91_000
    assert decision.details["W"] == 90_000.0 and decision.details["dominant"] == "e2e"
    record = store.records[probe.request_id]
    assert record["window_ms"] == 90_000.0
    assert record["window_terms"]["W1"] == 90_000.0
    assert record["window_terms"]["inputs"]["routable_pods"] == 4
    assert format_window_event("donor", decision.details).startswith(
        "safescale_probe_window:donor:W=90000:dominant=e2e:clamped=none:e2e=90000:queue=20:decode=100:gap=500"
    )

    restored = SafeScaleStateMachine(
        config=_V1_CFG, store=FakeProbeStore(unresolved=[record])
    )
    assert restored.restore() == 1
    assert restored.active_probe("donor").window_ms == 90_000.0
    assert restored.active_probe("donor").window_terms["dominant"] == "e2e"



def test_tail_gate_failures_lists_every_failing_check_in_v1_order() -> None:
    from tre_controller.planning.safescale import ProbeTailSummary, tail_gate_failures

    def summary(**kw) -> ProbeTailSummary:
        values = dict(latency_ok=True, z_min=1.2, has_traffic=True, sample_count=4, tail_count=2, gpu_cache_max=0.5)
        values.update(kw)
        return ProbeTailSummary(**values)

    assert tail_gate_failures(summary(), tau_low=1.0) == ()
    assert tail_gate_failures(summary(gpu_cache_max=0.81), tau_low=1.0) == ("kv_cache",)
    assert tail_gate_failures(summary(gpu_cache_max=0.8), tau_low=1.0) == ()  # v1: > 0.8 fails
    assert tail_gate_failures(summary(gpu_cache_max=0.7), tau_low=1.0, kv_cache_max=0.6) == ("kv_cache",)
    assert tail_gate_failures(summary(latency_ok=False, z_min=0.9, gpu_cache_max=0.9), tau_low=1.0) == (
        "latency",
        "z_below_tau_low",
        "kv_cache",
    )
    assert tail_gate_failures(summary(z_min=None), tau_low=1.0) == ("z_missing",)
    # No Z and no traffic commits (v1), whatever the cache says.
    assert tail_gate_failures(summary(z_min=None, has_traffic=False, gpu_cache_max=0.99), tau_low=1.0) == ()



# --- A13: donor-health guard + rollback backoff ------------------------------------------


def _gw(ts_ms: int, requests: float, errors: float, **kw) -> ProbeObservation:
    values = dict(
        ts_ms=ts_ms,
        ttft_p95_ms=500.0,
        tpot_p95_ms=50.0,
        z_m=1.2,
        q_ctl=0.0,
        has_traffic=True,
        gateway_requests=requests,
        gateway_errors=errors,
    )
    values.update(kw)
    return ProbeObservation(**values)


def test_donor_health_rolls_back_on_gateway_error_ratio_since_probe_start() -> None:
    store = FakeProbeStore()
    machine = SafeScaleStateMachine(config=_cfg(), store=store)
    machine.start_probe(model="donor", pods=("pod-a",), now_ms=0)

    # Baseline taken at the first observation: errors before the probe never count.
    assert machine.observe("donor", _gw(2_000, 10_000, 500), now_ms=2_000).reason == "probe_pending"
    # 19 new requests, 5 errors: below the 20-request minimum -> keep probing.
    assert machine.observe("donor", _gw(4_000, 10_019, 505), now_ms=4_000).reason == "probe_pending"
    # 100 requests, 1 error = 1 % -> not above the 1 % ceiling.
    assert machine.observe("donor", _gw(6_000, 10_100, 501), now_ms=6_000).reason == "probe_pending"
    decision = machine.observe("donor", _gw(8_000, 10_200, 503), now_ms=8_000)

    assert (decision.status, decision.reason) == ("rollback", "donor_health")
    assert decision.commands == (SafeScaleCommand(kind="unhide", model="donor", pods=("pod-a",), reason="donor_health"),)
    probe = machine.active_probe("donor")
    assert probe.terminal_details["donor_health"] == {"requests": 200.0, "errors": 3.0, "error_rate": 0.015}
    assert machine.resolve("donor", status="rollback", reason="donor_health", now_ms=8_000)
    (record,) = store.records.values()
    assert record["terminal_reason"] == "donor_health"
    assert record["gateway_baseline"] == [10_000.0, 500.0]
    assert record["terminal_details"]["donor_health"]["errors"] == 3.0


def test_donor_health_fails_open_without_counters_and_rebaselines_after_reset() -> None:
    machine = SafeScaleStateMachine(config=_cfg())
    machine.start_probe(model="donor", pods=("pod-a",), now_ms=0)
    assert machine.observe("donor", _gw(2_000, None, None), now_ms=2_000).reason == "probe_pending"
    assert machine.active_probe("donor").gateway_baseline is None
    machine.observe("donor", _gw(4_000, 5_000, 10), now_ms=4_000)
    # Envoy restarted: counters dropped -> new baseline, no bogus negative / huge delta.
    assert machine.observe("donor", _gw(6_000, 40, 0), now_ms=6_000).reason == "probe_pending"
    assert machine.active_probe("donor").gateway_baseline == (40.0, 0.0)


def test_commit_gate_records_donor_health_alongside_gate_failures() -> None:
    machine = SafeScaleStateMachine(config=_cfg())
    machine.start_probe(model="donor", pods=("pod-a",), now_ms=0)
    machine.observe("donor", _gw(2_000, 1_000, 0), now_ms=2_000)
    decision = machine.observe("donor", _gw(61_000, 1_500, 1), now_ms=61_000)

    assert decision.reason == "formal_commit_gate_passed"
    details = machine.active_probe("donor").terminal_details
    assert details["gate_failures"] == []
    assert details["donor_health"] == {"requests": 500.0, "errors": 1.0, "error_rate": 0.002}


def test_rollback_backoff_window_follows_the_last_rollback() -> None:
    machine = SafeScaleStateMachine(config=SafeScaleConfig(rollback_backoff_ms=60_000.0))
    machine.start_probe(model="donor", pods=("pod-a",), now_ms=0)
    machine.observe("donor", _gw(1_000, 0, 0, ttft_p95_ms=5_000.0), now_ms=1_000)
    machine.resolve("donor", status="rollback", reason="slo_violation", now_ms=1_000)

    assert machine.rollback_backoff_models(1_000) == {"donor"}
    assert machine.rollback_backoff_models(60_999) == {"donor"}
    assert machine.rollback_backoff_models(61_000) == set()
    # A commit does not start a backoff; 0 disables it.
    machine.start_probe(model="other", pods=("pod-o",), now_ms=0)
    machine.resolve("other", status="commit", reason="formal_commit_gate_passed", now_ms=2_000)
    assert "other" not in machine.rollback_backoff_models(2_000)
    off = SafeScaleStateMachine(config=SafeScaleConfig(rollback_backoff_ms=0.0))
    off.start_probe(model="donor", pods=("pod-a",), now_ms=0)
    off.resolve("donor", status="rollback", reason="slo_violation", now_ms=1_000)
    assert off.rollback_backoff_models(1_000) == set()



def test_preemption_request_is_idempotent_and_survives_restore() -> None:
    store = FakeProbeStore()
    machine = SafeScaleStateMachine(config=_cfg(), store=store)
    assert machine.request_preemption("donor") == 0  # no probe
    machine.start_probe(model="donor", pods=("pod-a", "pod-b"), now_ms=0)
    assert machine.request_preemption("donor") == 2
    assert machine.request_preemption("donor") == 2
    record = store.records[machine.active_probe("donor").request_id]
    assert record["preempt_reason"] == "receiver_need_upscale"

    restored = SafeScaleStateMachine(config=_cfg(), store=FakeProbeStore(unresolved=[record]))
    restored.restore()
    decision = restored.observe("donor", _healthy_observation(ts_ms=1_000), now_ms=1_000)
    assert (decision.status, decision.reason) == ("rollback", "receiver_need_upscale")
