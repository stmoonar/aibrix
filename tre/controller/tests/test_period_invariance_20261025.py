"""Period invariance (round 4, docs/design/20261025-instant-sampling.md "Evidence, not
ticks"): the same scenario run at evidence periods of 1, 2, 5 and 10 s gives the same
decisions at about the same times.

``period_sim`` renders each scenario into gateway docs at the period and runs it through
the real store, sampler, the single decision loop (2026-10-28, woken by each published
snapshot) and SafeScale (closed loop, mock SM; requests complete in whole numbers,
Poisson). Checked:

* single loop (design docs/design/20261028-single-loop.md, invariant 6): at a 1 s and a
  2 s snapshot cadence the mock SM applies the same actions in the same order;

* the key events - onset saturation rescue, first CRITICAL scale-up, O1 warm, probe
  starts and verdicts, the first stale snapshot - happen in every run; each one's
  delay from its cause is within one period of the coarser run (10 s) plus one rescue
  tick (5 s) of the 10 s run's;
* the probe verdicts are the same, in the same order;
* round 5 (M3): each run against the sim's own ground truth - a rule's delay from its
  cause is its evidence span, plus at most the run's period and one loop tick
  (``span - TOL <= delay <= span + P + tick``; a rule that counts windows is 5x
  shorter at 2 s, one that waits for windows 5x longer at 10 s): the saturation
  rescue (10 s full), the stale snapshot (20 s hold), the rollback-retry floor (20 s of
  new evidence; O1 off, so O1's own 20 s does not cover it), the direct-path early
  commit (20 s of post-hide observation with the gateway clock 4 s ahead: measured on
  the gateway clock only, the data a snapshot carries is 4 s older on the controller
  clock), O1 warm (>= 20 s);
* no stale snapshot without a gateway stall.
"""
from __future__ import annotations

import functools

import pytest

import period_sim as ps
from tre_common.persistence import SPAN_TOLERANCE_MS as TOL

PERIODS = (1_000, 2_000, 5_000, 10_000)
REFERENCE = 10_000
#: One period of the coarser run + one rescue tick.
TOLERANCE_MS = REFERENCE + 5_000
RESCUE_TICK_MS = 5_000
SAFESCALE_TICK_MS = 2_000
GATEWAY_AHEAD_MS = 4_000

SCENARIOS = {
    # d idle (shrunk to its floor), then 40 requests at once: the onset completes
    # nothing for ~20 s (saturation rescue), then CRITICAL; a 40 s gateway stall.
    "spike": ps.Scenario(demand={"r": ((0, 8.0),), "d": ((0, 0.0), (60, 40.0))}, awake=("r", "d", "d", "d"),
                         duration_s=300, stalls=((260, 300),)),
    # d HIGH on 3 pods: two probes, both commit, down to its floor.
    "shrink": ps.Scenario(demand={"r": ((0, 8.0),), "d": ((0, 9.0),)}, awake=("r", "d", "d", "d"),
                          duration_s=170),
    # d HIGH on 2 pods: the probe overloads the last pod and rolls back; the same
    # evidence holds the retry until the load drops; then the probe commits.
    "retry": ps.Scenario(demand={"r": ((0, 8.0),), "d": ((0, 22.0), (200, 8.0))}, awake=("r", "r", "d", "d"),
                         duration_s=330),
    # d HIGH on 3 pods, a TPOT threshold the probe breaks: rolled back, d stays HIGH, so
    # only the rollback-retry floor (then same_evidence) holds the retry. O1 off.
    "hold": ps.Scenario(demand={"r": ((0, 8.0),), "d": ((0, 9.0),)}, awake=("r", "d", "d", "d"),
                        duration_s=90, o1=False, tpot_slo_ms=7.0),
    # r HIGH on 3 pods, direct evidence path, early commit; the gateway clock 4 s ahead.
    "direct": ps.Scenario(demand={"r": ((0, 4.0),), "d": ((0, 8.0),)}, awake=("r", "r", "r", "d"),
                          duration_s=110, direct=True, gateway_offset_ms=GATEWAY_AHEAD_MS),
}


@functools.lru_cache(maxsize=None)
def _run(name: str, period: int) -> ps.Run:
    return ps.Sim(SCENARIOS[name], period).go()


def _times(run: ps.Run, predicate) -> list[int]:
    return [t for t, event in run.events if predicate(event)]


def _verdicts(run: ps.Run) -> list[str]:
    """The verdicts as ``event:model`` (an early commit's event also carries its counts)."""
    return [":".join(verdict.split(":")[:2]) for verdict in run.verdicts]


def _changes(run: ps.Run, model: str) -> list[int]:
    """When the mock SM changed ``model``'s routable count."""
    return _times(run, lambda e: e.startswith((f"scale_up:{model}", f"scale_down:{model}", f"hide:{model}",
                                               f"unhide:{model}")))


def _o1_warm(run: ps.Run, model: str, after_ms: int) -> int | None:
    """The first planner tick after ``after_ms`` whose window for ``model`` decides."""
    return next((t for t, event in run.events if t > after_ms and event == f"hold:{model}:None"), None)


def _key_delays(name: str, run: ps.Run) -> dict[str, list[int]]:
    """Each key event as the delay from its cause: a scenario input (onset, stall,
    start), the probe start for a verdict, the last routable change for O1 warm. A
    chained event is compared on its own delay, not on its absolute time (that adds
    the delays of the events before it)."""
    starts = _times(run, lambda e: "safescale_probe_started:" in e)
    verdicts = [t for t, e in run.events if e in run.verdicts]
    delays: dict[str, list[int]] = {
        "probe_start": [t - ps.T0 for t in starts],
        "verdict": [t - max(s for s in starts if s <= t) for t in verdicts],
    }
    if name == "spike":
        onset, stall = ps.T0 + 60_000, ps.T0 + 260_000
        saturation = _times(run, lambda e: e.startswith("rescue|saturation_rescue:d"))[:1]
        critical = _times(run, lambda e: e.startswith("rescue|rescue_target:d") and ":z=none:" not in e)[:1]
        delays["saturation_rescue"] = [t - onset for t in saturation]
        delays["critical_scale_up"] = [t - onset for t in critical]
        changed = max(t for t in _changes(run, "d") if t <= critical[0] + 250)
        warm = _o1_warm(run, "d", changed)
        delays["o1_warm"] = [warm - changed] if warm is not None else [None]
        delays["stale"] = [t - stall for t in _times(run, lambda e: e == "stale")]
    return delays


def _ground_truth(name: str, run: ps.Run, period: int) -> list[tuple[str, int, int, int]]:
    """(rule, delay from its cause, evidence span, upper slack) of each rule the
    scenario exercises; the check is ``span - TOL <= delay <= span + slack + TOL``."""
    out: list[tuple[str, int, int, int]] = []
    if name == "spike":
        onset, stall = ps.T0 + 60_000, ps.T0 + 260_000
        fired = _times(run, lambda e: e.startswith("rescue|saturation_rescue:d"))[0]
        out.append(("saturation_rescue", fired - onset, 10_000, period + RESCUE_TICK_MS))
        stale = _times(run, lambda e: e == "stale")[0]
        out.append(("stale", stale - stall, 20_000, period + 1_000))
    if name == "hold":
        # The cause is the rollback tick; the floor counts from the rollback's snapshot,
        # up to one period + one SafeScale tick before it (lower bound widened by P).
        rolled = _times(run, lambda e: e.startswith(("safescale_formal_commit_gate_failed:d",
                                                     "safescale_slo_violation:d")))[0]
        free = next(t for t, e in run.events if t > rolled and e.startswith("rescue|safescale_rollback_hold:d:")
                    and not e.endswith(":no_new_evidence"))
        out.append(("retry_floor", free - rolled + period, 20_000, period + RESCUE_TICK_MS))
    if name == "direct":
        hides = _times(run, lambda e: e == "hide:r")
        for commit in _times(run, lambda e: e.startswith("safescale_early_commit:r")):
            hide = max(t for t in hides if t <= commit)
            # 20 s observed on the gateway clock from the first doc boundary after the
            # hide (<= one period later); a snapshot's data is GATEWAY_AHEAD_MS old.
            out.append(("early_commit", commit - hide, 20_000 + GATEWAY_AHEAD_MS,
                        2 * period + SAFESCALE_TICK_MS))
    return out


def _applied(run: ps.Run) -> list[str]:
    """The actions the mock SM applied, in order (no times)."""
    kinds = ("scale_up:", "scale_down:", "hide:", "unhide:", "relay:")
    return [event for _, event in run.events if event.startswith(kinds)]


@pytest.mark.parametrize("name", sorted(SCENARIOS))
def test_one_and_two_second_snapshots_give_the_same_actions(name):
    fine, coarse = _applied(_run(name, 1_000)), _applied(_run(name, 2_000))
    assert fine and fine == coarse


@pytest.mark.parametrize("name", sorted(SCENARIOS))
def test_the_same_decisions_at_about_the_same_times_at_every_period(name):
    reference = _key_delays(name, _run(name, REFERENCE))
    assert reference["verdict"] or name == "spike", "the scenario must reach a probe verdict"
    for period in PERIODS:
        run = _run(name, period)
        assert _verdicts(run) == _verdicts(_run(name, REFERENCE)), period
        got = _key_delays(name, run)
        for key, delays in reference.items():
            assert len(got[key]) == len(delays) and None not in got[key], (period, key, got[key], delays)
            for delay, ref in zip(got[key], delays):
                assert abs(delay - ref) <= TOLERANCE_MS, (period, key, delay, ref)


@pytest.mark.parametrize("period", PERIODS)
@pytest.mark.parametrize("name", ["spike", "hold", "direct"])
def test_each_rule_waits_its_evidence_span_and_at_most_a_period_and_a_tick_more(name, period):
    truth = _ground_truth(name, _run(name, period), period)
    assert truth, "the scenario must exercise its rules"
    for rule, delay, span, slack in truth:
        assert span - TOL <= delay <= span + slack + TOL, (rule, period, delay, span, slack)


@pytest.mark.parametrize("period", PERIODS)
def test_the_evidence_spans_hold_at_every_period(period):
    spike = _run("spike", period)
    # Saturation: 10 s of full observations (the event reports the held span).
    fired = next(e for _, e in spike.events if e.startswith("rescue|saturation_rescue:d"))
    assert int(fired.rsplit("held_ms=", 1)[1].split(":")[0]) >= 10_000 - TOL
    # O1: the first deciding window after a routable change carries >= 20 s of
    # observations after it (the change is seen by the next view; warm >= 20 s later).
    for changed in (t for t in _changes(spike, "d") if t < ps.T0 + 120_000):
        warm = _o1_warm(spike, "d", changed)
        assert warm is not None and warm - changed >= 20_000 - TOL
    # SafeScale: the rolled-back probe is held by the same evidence (not by a newer
    # window), until the load drops.
    retry = _run("retry", period)
    assert any("safescale_rollback_hold:d:same_evidence" in e for _, e in retry.events)


@pytest.mark.parametrize("period", PERIODS)
def test_no_stale_snapshot_without_a_gateway_stall(period):
    for name in ("shrink", "retry", "hold", "direct"):
        assert not _times(_run(name, period), lambda e: e == "stale"), (name, period)
    stall_start = ps.T0 + 260_000
    stale = _times(_run("spike", period), lambda e: e == "stale")
    assert stale and stale[0] > stall_start + 20_000  # the 20 s stale hold, then stale
