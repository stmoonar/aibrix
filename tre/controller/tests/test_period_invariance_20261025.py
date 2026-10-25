"""Period invariance (round 4, docs/design/20261025-instant-sampling.md "Evidence, not
ticks"): the same scenario run at evidence periods of 2, 5 and 10 s gives the same
decisions at about the same times.

``period_sim`` renders each scenario into gateway docs at the period and runs it through
the real store, sampler, planner loops and SafeScale (closed loop, mock SM). Checked:

* the key events - onset saturation rescue, first CRITICAL scale-up, O1 warm, probe
  starts and verdicts, the first stale snapshot - happen in every run; each one's
  delay from its cause is within one period of the coarser run (10 s) plus one rescue
  tick (5 s) of the 10 s run's;
* the probe verdicts are the same, in the same order;
* the evidence spans hold at every period (a rule that counts windows is 5x shorter at
  2 s): O1 warms >= 20 s of observations after the breakpoint, the saturation rescue
  fires on >= 10 s of full observations, a rolled-back probe is held by its evidence;
* no stale snapshot without a gateway stall.
"""
from __future__ import annotations

import functools

import pytest

import period_sim as ps
from tre_common.persistence import SPAN_TOLERANCE_MS as TOL

PERIODS = (2_000, 5_000, 10_000)
REFERENCE = 10_000
#: One period of the coarser run + one rescue tick.
TOLERANCE_MS = REFERENCE + 5_000

SCENARIOS = {
    # d idle (shrunk to its floor), then 40 requests at once: the onset completes
    # nothing for ~20 s (saturation rescue), then CRITICAL; a 40 s gateway stall.
    "spike": ps.Scenario(demand={"r": ((0, 8.0),), "d": ((0, 0.0), (60, 40.0))}, awake=("r", "d", "d", "d"),
                         duration_s=300, stalls=((260, 300),)),
    # d HIGH on 3 pods: two probes, both commit, down to its floor.
    "shrink": ps.Scenario(demand={"r": ((0, 8.0),), "d": ((0, 9.0),)}, awake=("r", "d", "d", "d"),
                          duration_s=130),
    # d HIGH on 2 pods: the probe overloads the last pod and rolls back; the same
    # evidence holds the retry until the load drops; then the probe commits.
    "retry": ps.Scenario(demand={"r": ((0, 8.0),), "d": ((0, 22.0), (200, 8.0))}, awake=("r", "r", "d", "d"),
                         duration_s=290),
}


@functools.lru_cache(maxsize=None)
def _run(name: str, period: int) -> ps.Run:
    return ps.Sim(SCENARIOS[name], period).go()


def _times(run: ps.Run, predicate) -> list[int]:
    return [t for t, event in run.events if predicate(event)]


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


@pytest.mark.parametrize("name", sorted(SCENARIOS))
def test_the_same_decisions_at_about_the_same_times_at_every_period(name):
    reference = _key_delays(name, _run(name, REFERENCE))
    assert reference["verdict"] or name == "spike", "the scenario must reach a probe verdict"
    for period in PERIODS:
        run = _run(name, period)
        assert run.verdicts == _run(name, REFERENCE).verdicts, period
        got = _key_delays(name, run)
        for key, delays in reference.items():
            assert len(got[key]) == len(delays) and None not in got[key], (period, key, got[key], delays)
            for delay, ref in zip(got[key], delays):
                assert abs(delay - ref) <= TOLERANCE_MS, (period, key, delay, ref)


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
    for name in ("shrink", "retry"):
        assert not _times(_run(name, period), lambda e: e == "stale"), (name, period)
    stall_start = ps.T0 + 260_000
    stale = _times(_run("spike", period), lambda e: e == "stale")
    assert stale and stale[0] > stall_start + 20_000  # the 20 s stale hold, then stale
