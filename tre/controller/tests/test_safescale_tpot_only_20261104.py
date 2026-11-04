"""SafeScale judges TPOT only (user 2026-11-04, a design change).

Invariants, on every evidence path (direct scrapes - the live default -, the Redis
evidence window, the tail-snapshot fallback), with real histograms:

1. the latency verdict is the window's TPOT p95 against the model's TPOT_slo: a window
   whose TTFT is far over any TTFT_slo but whose TPOT is under TPOT_slo commits; the same
   window with TPOT over TPOT_slo rolls back;
2. fail closed on the judged latency (review of 9ad3ec32): a remaining pod without the
   TPOT family never lets the probe commit; a window with requests but no TPOT sample at
   all never commits and rolls back ``insufficient_evidence:p95_unavailable`` at the
   ceiling.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

from safescale_thresholds import STATIC_TPOT_75
from test_safescale_direct_20260929 import HIDE, START, W, Harness
from test_safescale_evidence_20260929 import (
    DocRedis,
    _obs,
    _reader,
    _real_machine,
    _remaining,
    _slo_registry,
)
from tre_controller.config import SafeScaleConfig
from tre_controller.planning.safescale import ProbeObservation, SafeScaleStateMachine
from tre_controller.planning.safescale_evidence import HideAnchor, RegistryThresholds

MODEL = "m"
#: 5 s first-token time: over every TTFT_slo of every mode (floors 450 / 500 ms, k * idle).
SLOW_TTFT_S = 5.0
#: The W ceiling of both harnesses (start / confirmed hide + 60 s).
DIRECT_CAP = HIDE + 60_000
REDIS_CAP = START + 60_000
P95_UNAVAILABLE = "insufficient_evidence:p95_unavailable"


# ============================================================ 1. TPOT is the verdict
@pytest.mark.parametrize("itl_s, want", [(0.02, "commit"), (2.0, "rollback")])
def test_direct_path_judges_tpot_only(itl_s, want) -> None:
    h = Harness()
    h.start(pre_hide_itl=0.02)
    at, decision = h.run_until(HIDE + W, serve=5, itl_s=itl_s, ttft_s=SLOW_TTFT_S)[-1]
    assert decision.status == want, decision
    if want == "commit":
        assert at == HIDE + W and decision.reason == "formal_commit_gate_passed"
        assert decision.details["latency_violations"] == []
    else:
        reason = decision.details["rollback_reason"]
        assert (reason["code"], reason["metrics"]) == ("slo_violation_direct", ["tpot"])
    assert not any("ttft" in key for key in decision.details if key != "ttft_count")


def _docs(redis: DocRedis, *, end_ms: int = 130_000, tpot: str = "samples", slow_tpot: bool = False,
          pods: tuple[str, ...] = ("m-0",), tpot_by_pod: dict | None = None) -> None:
    """The remaining pods serve 15 requests per 10 s after the 110 s boundary, each with
    a 5 s TTFT; their TPOT is 10 ms (fast) or 500 ms (slow), or has no sample / no
    family (``tpot``, per pod in ``tpot_by_pod``). m-1 is the probe pod."""
    for ts in range(70_000, end_ms + 1, 10_000):
        done = 15 * max(0, (ts - 110_000) // 10_000)
        fast, slow = (0, done) if slow_tpot else (done, 0)
        for pod in pods:
            redis.add(pod, ts, fast=fast, slow=slow, prompt=0.0, ttft_s=SLOW_TTFT_S,
                      tpot=(tpot_by_pod or {}).get(pod, tpot))
        redis.add("m-1", ts, fast=0, slow=0, prompt=0.0)


@pytest.mark.parametrize("slow_tpot, want", [(False, "commit"), (True, "rollback")])
def test_redis_evidence_path_judges_tpot_only(slow_tpot, want) -> None:
    redis = DocRedis()
    _docs(redis, slow_tpot=slow_tpot)
    machine = _real_machine(redis)
    machine.observe(MODEL, _obs(110_000), now_ms=110_000)
    machine.observe(MODEL, _obs(120_000), now_ms=120_000)
    decision = machine.observe(MODEL, _obs(130_000), now_ms=130_000)
    assert decision.status == want, decision
    assert decision.details["latency_samples"] == 30
    assert decision.details["latency_violations"] == ([] if want == "commit" else ["tpot"])


# ============================================================ 2. fail closed on TPOT
def _strip_tpot(h: Harness, pod: str) -> None:
    original = h.sims[pod].text
    h.sims[pod].text = lambda **kw: "\n".join(
        line for line in original(**kw).splitlines() if "inter_token_latency" not in line) + "\n"


def test_direct_a_pod_without_the_tpot_family_never_lets_the_probe_commit() -> None:
    # B1: m-0 healthy, m-2 has TTFT but no TPOT family - its requests are not judged.
    h = Harness()
    _strip_tpot(h, "m-2")
    h.start(pre_hide_itl=0.02)
    decisions = h.run_until(DIRECT_CAP, serve=5)
    assert all(d.status != "commit" for _, d in decisions)
    assert decisions[-1][1].status == "rollback"
    assert h.machine.active_probe(MODEL).direct.last.missing.get("m-2") == "pending_baseline"


def _redis_machine(redis: DocRedis, remaining: tuple[str, ...]) -> SafeScaleStateMachine:
    machine = SafeScaleStateMachine(
        config=SafeScaleConfig(min_window_ms=20_000.0, window_ceiling_ms=60_000.0, hq=0.25, tau_low=1.0),
        evidence=_reader(redis), thresholds=RegistryThresholds(_slo_registry()),
        wall_clock_ms=lambda: 0, remaining_pods=_remaining(*remaining),
    )
    machine.start_probe(model=MODEL, pods=("m-1",), now_ms=START)
    machine.mark_hidden(MODEL, pods=("m-1",), anchor=HideAnchor(HIDE, "redis_time", 100_000))
    return machine


def _redis_run(machine: SafeScaleStateMachine, end_ms: int = REDIS_CAP):
    out = []
    for ts in range(110_000, end_ms + 1, 10_000):
        decision = machine.observe(MODEL, _obs(ts), now_ms=ts)
        out.append(decision)
        if decision.status in ("commit", "rollback"):
            break
    return out


def test_redis_a_required_pod_without_the_tpot_family_never_lets_the_probe_commit() -> None:
    redis = DocRedis()
    _docs(redis, end_ms=REDIS_CAP + 20_000, pods=("m-0", "m-3"), tpot_by_pod={"m-3": "absent"})
    window = _reader(redis).read(MODEL, start_ms=110_000, end_ms=130_000, exclude_pods=("m-1",))
    assert window.tpot_missing_pods == ("m-3",) and window.judged_count == 30.0  # m-0 only
    decisions = _redis_run(_redis_machine(redis, ("m-0", "m-3")), end_ms=REDIS_CAP + 20_000)
    assert all(d.status != "commit" for d in decisions)
    assert decisions[-1].status == "rollback"
    assert decisions[-1].details["rollback_reason"]["detail"] == {"m-3": "no_tpot_histogram"}


def _tail_snapshot_run():
    """No evidence source: the gate reads the tail snapshots - traffic, no TPOT p95."""
    machine = SafeScaleStateMachine(thresholds=STATIC_TPOT_75, config=SafeScaleConfig(
        min_window_ms=20_000.0, window_ceiling_ms=60_000.0, hq=0.25, tau_low=1.0))
    machine.start_probe(model=MODEL, pods=("m-1",), now_ms=0)
    out = []
    for ts in range(10_000, 70_001, 10_000):
        observation = ProbeObservation(ts_ms=ts, tpot_p95_ms=None, z_m=2.0, has_traffic=True,
                                       window_start_ms=ts - 10_000, window_end_ms=ts)
        decision = machine.observe(MODEL, observation, now_ms=ts)
        out.append(decision)
        if decision.status in ("commit", "rollback"):
            break
    return out


def _direct_no_tpot_samples():
    h = Harness()
    h.start(pre_hide_itl=0.02)
    h.tokens = 0  # one-token outputs: requests complete, no TPOT sample
    return [d for _, d in h.run_until(DIRECT_CAP, serve=5)]


def _redis_no_tpot_samples():
    redis = DocRedis()
    _docs(redis, end_ms=REDIS_CAP + 10_000, tpot="empty")
    return _redis_run(_real_machine(redis))


@pytest.mark.parametrize("run", [_direct_no_tpot_samples, _redis_no_tpot_samples, _tail_snapshot_run],
                         ids=["direct", "redis", "tail_snapshots"])
def test_requests_without_any_tpot_sample_never_commit_and_roll_back_at_the_ceiling(run) -> None:
    # TTFT present (5 s, or none in the snapshot), TPOT absent: no latency verdict.
    decisions = run()
    assert all(d.status != "commit" for d in decisions)
    assert decisions[-1].status == "rollback"
    assert decisions[-1].details["rollback_reason"]["code"] == P95_UNAVAILABLE


# ============================================================ structure
def test_the_safescale_code_has_no_ttft_threshold_or_ttft_p95() -> None:
    # The TTFT histogram is SafeScale's request counter only (ttft / ttft_count).
    tre = Path(__file__).resolve().parents[1] / "tre_controller"
    files = sorted((tre / "planning").glob("safescale*.py")) + [tre / "loops" / "safescale_task.py"]
    hits = {f.name: re.findall(r"\w*ttft_(?:ms|p95|threshold|slo)\w*", f.read_text(encoding="utf-8"))
            for f in files}
    assert not {name: found for name, found in hits.items() if found}, hits
