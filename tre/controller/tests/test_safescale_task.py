from __future__ import annotations

from tre_common.metrics_schema import MetricsSnapshot, ModelWindowMetrics
from tre_common.registry import ClusterTopology, ModelSpec, NodeSpec, Registry, SloSpec, TrsParams
from tre_controller.config import SafeScaleConfig
from tre_controller.loops.action_queue import SubmitResult
from tre_controller.loops.safescale_task import run_safescale_observation_tick
from tre_controller.planning.planner import ScaleAction, UnhideAction
from tre_controller.planning.safescale import SafeScaleStateMachine


class FakeQueue:
    def __init__(self) -> None:
        self.submitted: list[tuple] = []

    def submit(self, actions) -> SubmitResult:
        self.submitted.append(tuple(actions))
        return SubmitResult(accepted=len(actions))


def _registry() -> Registry:
    spec = ModelSpec(
        name="donor",
        weights_path="/weights",
        tp_size=1,
        min_replicas=0,
        max_replicas=4,
        vllm_image="image",
        slo=SloSpec(ttft_p95_ms=1000.0, tpot_p95_ms=100.0, e2e_p95_ms=10_000.0),
        trs=TrsParams(
            w_p=0.04,
            w_d=1.0,
            lambda_wait=2.0,
            qmin=1.0,
            ema_alpha=0.0,
            theta_m=100.0,
            tau_crit=0.8,
            tau_low=1.0,
            tau_high=1.25,
            qsat=4.0,
            epsat=0.05,
            hsat=1,
        ),
    )
    return Registry(ClusterTopology(nodes=(NodeSpec(name="node-a", gpus=4, two_gpu_slots=((0, 1), (2, 3))),)), [spec])


def _metrics(*, ts_ms: int, generation: float = 120.0, ttft: float = 500.0, tpot: float = 50.0) -> MetricsSnapshot:
    return MetricsSnapshot(
        ts_ms=ts_ms,
        stale=False,
        models={
            "donor": ModelWindowMetrics(
                model="donor",
                window_start_ms=0,
                window_end_ms=60_000,
                prompt_tokens=0.0,
                generation_tokens=generation,
                avg_waiting=0.0,
                avg_running=1.0,
                avg_swapping=0.0,
                kv_cache_hit_rate=0.0,
                ttft_p95_ms=ttft,
                tpot_p95_ms=tpot,
                e2e_p95_ms=1000.0,
                routable_pods=1,
                assigned_replicas=1,
                per_pod={},
            )
        },
    )


def _machine() -> SafeScaleStateMachine:
    return SafeScaleStateMachine(
        config=SafeScaleConfig(
            ttft_p95_slo_ms=1000.0,
            tpot_p95_slo_ms=100.0,
            default_window_ms=1000.0,
            min_window_ms=1000.0,
            hq=0.5,
            tau_low=1.0,
        )
    )


def test_safescale_observation_tick_submits_commit_actions_after_deadline() -> None:
    queue = FakeQueue()
    machine = _machine()
    machine.start_probe(model="donor", pods=("pod-a",), now_ms=0, pending_upscales={"receiver": 1})

    pending = run_safescale_observation_tick(_metrics(ts_ms=500), queue=queue, registry=_registry(), safescale=machine)
    committed = run_safescale_observation_tick(_metrics(ts_ms=1000), queue=queue, registry=_registry(), safescale=machine)

    assert pending.submitted == 0
    assert pending.events == ("safescale_probe_pending:donor",)
    assert committed.submitted == 2
    assert queue.submitted == [
        (
            ScaleAction("donor", -1, "formal_commit_gate_passed", "safescale", pods=("pod-a",)),
            ScaleAction("receiver", 1, "safescale_followup_upscale", "safescale"),
        )
    ]
    # No per-pod KV-cache data in this window: the gate passes, and says so (P2-a).
    assert committed.events == (
        "safescale_formal_commit_gate_passed:donor",
        "safescale_kv_cache_unavailable:donor",
    )


def test_safescale_observation_tick_submits_rollback_unhide_on_slo_violation() -> None:
    queue = FakeQueue()
    machine = _machine()
    machine.start_probe(model="donor", pods=("pod-a",), now_ms=0)

    result = run_safescale_observation_tick(
        _metrics(ts_ms=500, ttft=1500.0),
        queue=queue,
        registry=_registry(),
        safescale=machine,
    )

    assert result.submitted == 1
    assert queue.submitted == [(UnhideAction("donor", ("pod-a",), "slo_violation", "safescale"),)]
    assert result.events == ("safescale_slo_violation:donor",)


def _with_pod_kv(snapshot: MetricsSnapshot, fills: dict[str, float | None]) -> MetricsSnapshot:
    from dataclasses import replace

    from tre_common.metrics_schema import PodWindowMetrics

    window = snapshot.models["donor"]
    per_pod = {
        pod: PodWindowMetrics(
            pod=pod,
            prompt_tokens=0.0,
            generation_tokens=0.0,
            avg_waiting=0.0,
            avg_running=0.0,
            avg_swapping=0.0,
            kv_cache_hit_rate=0.0,
            ttft_p95_ms=None,
            tpot_p95_ms=None,
            e2e_p95_ms=None,
            gpu_cache_usage=fill,
        )
        for pod, fill in fills.items()
    }
    return replace(snapshot, models={"donor": replace(window, per_pod=per_pod)})


def test_remaining_pods_kv_cache_averages_the_serving_pods_only() -> None:
    from tre_controller.loops.safescale_task import remaining_pods_kv_cache

    window = _with_pod_kv(_metrics(ts_ms=0), {"pod-a": 0.99, "pod-b": 0.5, "pod-c": 0.7, "pod-d": None}).models["donor"]
    assert abs(remaining_pods_kv_cache(window, ("pod-a",)) - 0.6) < 1e-9  # hidden pod-a excluded
    assert remaining_pods_kv_cache(window, ("pod-a", "pod-b", "pod-c")) is None


def test_safescale_kv_cache_guard_blocks_commit_like_v1() -> None:
    # A12: avg_gpu_cache_norm was hard-wired None, so the commit gate's KV-cache check
    # (v1: tail max <= 0.8) never fired. The hidden pod's own fill does not count.
    queue = FakeQueue()
    machine = _machine()
    machine.start_probe(model="donor", pods=("pod-a",), now_ms=0)
    hot = {"pod-a": 0.1, "pod-b": 0.95}

    pending = run_safescale_observation_tick(
        _with_pod_kv(_metrics(ts_ms=500), hot), queue=queue, registry=_registry(), safescale=machine
    )
    assert pending.events == ("safescale_probe_pending:donor",)  # no immediate KV rollback (v1)
    result = run_safescale_observation_tick(
        _with_pod_kv(_metrics(ts_ms=1000), hot), queue=queue, registry=_registry(), safescale=machine
    )

    assert queue.submitted == [(UnhideAction("donor", ("pod-a",), "formal_commit_gate_failed", "safescale"),)]
    assert result.events == (
        "safescale_formal_commit_gate_failed:donor",
        "safescale_gate_failures:donor:kv_cache",
    )

    cool_queue = FakeQueue()
    cool = _machine()
    cool.start_probe(model="donor", pods=("pod-a",), now_ms=0)
    for ts in (500, 1000):
        run_safescale_observation_tick(
            _with_pod_kv(_metrics(ts_ms=ts), {"pod-a": 0.99, "pod-b": 0.8}),
            queue=cool_queue,
            registry=_registry(),
            safescale=cool,
        )
    assert cool_queue.submitted[0][0].reason == "formal_commit_gate_passed"


def test_safescale_resolution_record_carries_the_gate_failures() -> None:
    class Store:
        def __init__(self) -> None:
            self.records: dict[str, dict] = {}

        def save_probe(self, request_id, record):
            self.records[request_id] = dict(record)

        def delete_probe(self, request_id):
            self.records.pop(request_id, None)

        def list_unresolved_probes(self):
            return []

        def append_probe_journal(self, request_id, record):
            pass

        def load_probe_journal(self, request_id):
            return []

    store = Store()
    machine = SafeScaleStateMachine(
        config=SafeScaleConfig(
            ttft_p95_slo_ms=1000.0, tpot_p95_slo_ms=100.0, default_window_ms=1000.0, min_window_ms=1000.0, hq=0.5
        ),
        store=store,
    )
    machine.start_probe(model="donor", pods=("pod-a",), now_ms=0)
    for ts in (500, 1000):
        run_safescale_observation_tick(
            _with_pod_kv(_metrics(ts_ms=ts), {"pod-b": 0.9}), queue=FakeQueue(), registry=_registry(), safescale=machine
        )

    (record,) = store.records.values()
    assert (record["status"], record["resolution"], record["terminal_reason"]) == (
        "resolved",
        "rollback",
        "formal_commit_gate_failed",
    )
    assert record["terminal_details"]["gate_failures"] == ["kv_cache"]
    assert record["terminal_details"]["tail"]["gpu_cache_max"] == 0.9



def test_observation_tick_feeds_gateway_counters_to_the_donor_health_guard() -> None:
    from tre_controller.gateway_health import GatewayCounters

    queue = FakeQueue()
    machine = SafeScaleStateMachine(
        config=SafeScaleConfig(ttft_p95_slo_ms=1000.0, tpot_p95_slo_ms=100.0, default_window_ms=60_000.0)
    )
    machine.start_probe(model="donor", pods=("pod-a",), now_ms=0)

    run_safescale_observation_tick(
        _metrics(ts_ms=500),
        queue=queue,
        registry=_registry(),
        safescale=machine,
        gateway_counters={"donor": GatewayCounters(requests=1_000, errors=0)},
    )
    result = run_safescale_observation_tick(
        _metrics(ts_ms=2_500),
        queue=queue,
        registry=_registry(),
        safescale=machine,
        gateway_counters={"donor": GatewayCounters(requests=1_050, errors=5)},
    )

    assert queue.submitted == [(UnhideAction("donor", ("pod-a",), "donor_health", "safescale"),)]
    assert result.events == (
        "safescale_donor_health:donor",
        "safescale_donor_health:donor:errors=5:requests=50:rate=0.1000",
    )
    assert machine.active_probe("donor") is None
    assert machine.rollback_backoff_models(2_500) == {"donor"}



def test_kv_cache_unavailable_passes_the_gate_but_is_reported() -> None:
    # Review P2-a: no pod reports gpu_cache_usage -> the KV check cannot run. Still
    # fail-open like v1, but no longer silent: event + kv_cache=unavailable in the record.
    queue = FakeQueue()
    machine = _machine()
    machine.start_probe(model="donor", pods=("pod-a",), now_ms=0)
    run_safescale_observation_tick(
        _with_pod_kv(_metrics(ts_ms=500), {"pod-b": None}), queue=queue, registry=_registry(), safescale=machine
    )
    probe_holder = {}
    original_resolve = machine.resolve

    def capture(model, **kw):
        probe_holder["details"] = dict(machine.active_probe(model).terminal_details)
        return original_resolve(model, **kw)

    machine.resolve = capture
    result = run_safescale_observation_tick(
        _with_pod_kv(_metrics(ts_ms=1000), {"pod-b": None}), queue=queue, registry=_registry(), safescale=machine
    )

    assert result.events == (
        "safescale_formal_commit_gate_passed:donor",
        "safescale_kv_cache_unavailable:donor",
    )
    assert probe_holder["details"]["kv_cache"] == "unavailable"
    assert queue.submitted[-1][0].reason == "formal_commit_gate_passed"
