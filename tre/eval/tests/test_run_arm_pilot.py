"""run_arm_pilot.sh / reset_canonical.sh on the offline harness (runner/selftest/make_harness.sh: fake
kubectl, curl, redis-py, loadgen; no cluster). Invariants (2026-11-02, docs runner-overhead-20261102 C2):

* one controller rollout per arm: the arm's controller is started by the ``set env`` rollout when a switch
  differs, else by a restart; the startup-line check passes either way;
* CTL_ENV_KEEP=1 (campaign): an arm leaves its switches set and the next arm's rollout writes every one of
  them; reset_canonical.sh leaves them with CTL_ENV_KEEP=1 and puts them back without it; without
  CTL_ENV_KEEP an arm puts them back at its end, as before;
* a switch compares as the controller reads it (_get_bool): the overlay's "0" is false - no rollout, no
  deviation; a value the controller does not know is reported;
* every collector file is still written, and an arm that dies in its collection puts its switches back
  (CTL_ENV_KEEP=1 too).
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

RUNNER = Path(__file__).resolve().parents[1] / "runner"
EVAL = RUNNER.parent

pytestmark = pytest.mark.skipif(sys.platform != "linux" or not shutil.which("git"),
                                reason="the harness needs Linux (GNU sed / date) and git")

# what the collection after POST_S writes (the harness stubs the sampler: no layout / pod files)
COLLECTED = ("clock_offsets.json", "components.json", "snap_after.json", "controller.log", "sm.log", "sm.ts.log",
             "gateway-plugins.log", "sidecar", "EMFILE_PODS", "aibrix-controller-manager.log", "apa_crs_after.yaml",
             "safescale.json", "signal_log.jsonl", "safescale_summary.json", "gateway_events", "decision_hist.jsonl",
             "sm_operations.jsonl", "collect_start_epoch", "analyze.out", "score.json", "run_check.json")


@pytest.fixture()
def harness(tmp_path):
    h = tmp_path / "h"
    subprocess.run(["bash", str(RUNNER / "selftest" / "make_harness.sh")], check=True, capture_output=True,
                   env={**os.environ, "H": str(h), "SRC": str(RUNNER)})
    return h


def env_of(h: Path, **kw) -> dict:
    return {**os.environ, "PATH": f"{h}/bin:{os.environ['PATH']}", "PYTHONPATH": f"{h}/pyfake:{EVAL}",
            "TRE_DIR": str(h / "tre"), "LOADGEN_TRE_DIR": str(h / "tre"), "TOOLS": str(h / "tools"),
            "PILOT_ROOT": str(h / "pilot"), "REQUIRE_MARKER": "0", "MARKER": "/nonexistent", "ICSE": str(h / "icse"),
            "IDLE_S": "0", "POST_S": "0", "CLOCK_NODES": "fake-node=local", "RUNNER_ENV": "/nonexistent",
            "CONTROLLER_DEPLOY": "tre-v2-controller", **kw}


def calls(h: Path) -> list[str]:
    p = h / "state" / "calls.log"
    return p.read_text().splitlines() if p.is_file() else []


def run_arm(h: Path, arm: str, n: int, **kw) -> tuple[int, Path, list[str]]:
    """One arm; returns rc, its directory and the controller rollouts it caused (set env / rollout restart)."""
    before = len(calls(h))
    d = h / "pilot" / f"a{n}"
    r = subprocess.run(["bash", str(h / "tools" / "run_arm_pilot.sh"), arm, "Alt"], cwd=h, capture_output=True,
                       env=env_of(h, ARM_OUT_DIR=str(d), **kw), timeout=300)
    new = calls(h)[before:]
    rollouts = [c for c in new if "deploy/tre-v2-controller" in c and (" set env " in c or " rollout restart " in c)]
    return r.returncode, d, rollouts


def switch(h: Path, key: str = "TRE_ABLATION_DISABLE_SLOW_LOOP") -> str:
    p = h / "state" / f"ctl_{key}"
    return p.read_text() if p.is_file() else ""


def startup_line_matched(d: Path) -> bool:
    import json
    return json.loads((d / "ablation_switches.json").read_text())["match"] is True


SWITCH_DEVIATION = "controller ablation switches not false"


def reset_check(h: Path, **kw) -> str:
    """reset_canonical.sh --check output (the harness lacks the run-mode / registry tools, so it is never
    canonical there: only its switch line is looked at)."""
    r = subprocess.run(["bash", str(h / "tools" / "reset_canonical.sh"), "--check"], cwd=h, capture_output=True,
                       env=env_of(h, **kw))
    return r.stdout.decode() + r.stderr.decode()


def test_one_controller_rollout_per_arm_and_switches_kept_for_the_next_arm_in_campaign_mode(harness):
    h = harness
    # campaign mode: an ablation arm -> its `set env` rollout is the arm's restart; the switch stays set
    rc, d, ro = run_arm(h, "tre", 1, CTL_ENV_KEEP="1", ARM_CONTROLLER_ENV="TRE_ABLATION_DISABLE_SLOW_LOOP=true")
    assert rc == 0 and len(ro) == 1 and " set env " in ro[0], ro
    assert (d / "controller_restart_epoch").is_file() and startup_line_matched(d)
    assert switch(h) == "true"
    # reset between arms (campaign): the kept switch is not a deviation; without CTL_ENV_KEEP it is
    assert SWITCH_DEVIATION not in reset_check(h, CTL_ENV_KEEP="1")
    assert SWITCH_DEVIATION in reset_check(h)
    # next arm without switches: ONE rollout that writes the switch back to its default and starts the arm
    rc, d, ro = run_arm(h, "apa", 2, CTL_ENV_KEEP="1")
    assert rc == 0 and len(ro) == 1 and "TRE_ABLATION_DISABLE_SLOW_LOOP=false" in ro[0], ro
    assert switch(h) == "false" and startup_line_matched(d)
    # nothing to change: a plain restart, still one rollout
    rc, d, ro = run_arm(h, "tre", 3, CTL_ENV_KEEP="1")
    assert rc == 0 and len(ro) == 1 and " rollout restart " in ro[0], ro
    # the end of a campaign (no CTL_ENV_KEEP): reset_canonical.sh restores a switch left set
    (h / "state" / "ctl_TRE_ABLATION_DISABLE_SLOW_LOOP").write_text("true")
    subprocess.run(["bash", str(h / "tools" / "reset_canonical.sh")], cwd=h, capture_output=True, env=env_of(h, CTL_ENV_KEEP="1"))
    assert switch(h) == "true"
    subprocess.run(["bash", str(h / "tools" / "reset_canonical.sh")], cwd=h, capture_output=True, env=env_of(h))
    assert switch(h) == "false"


def test_standalone_arm_puts_its_switches_back_and_collects_every_file(harness):
    h = harness
    rc, d, ro = run_arm(h, "tre", 1, ARM_CONTROLLER_ENV="TRE_ABLATION_DISABLE_SLOW_LOOP=true")
    assert rc == 0, (d / "runner.log").read_text()[-1500:]
    # the merged start rollout + the end-of-arm restore (was: set env + restart + restore)
    assert len(ro) == 2 and "SLOW_LOOP=true" in ro[0] and "SLOW_LOOP=false" in ro[1], ro
    assert switch(h) == "false" and startup_line_matched(d)
    missing = [f for f in COLLECTED if not (d / f).exists()]
    assert not missing, missing


def test_an_arm_that_dies_in_its_collection_puts_its_switches_back_in_campaign_mode(harness):
    h = harness
    (h / "tools" / "snap.py").write_text(
        "import json, sys\nif sys.argv[1].endswith('snap_after.json'):\n    sys.exit(5)\njson.dump({}, open(sys.argv[1], 'w'))\n")
    rc, d, _ = run_arm(h, "tre", 1, ARM_CONTROLLER_ENV="TRE_ABLATION_DISABLE_SLOW_LOOP=true", CTL_ENV_KEEP="1")
    log = (d / "runner.log").read_text()
    # snap_after is fatal (as before); the exit trap puts the switch back although CTL_ENV_KEEP=1
    assert rc == 5 and switch(h) == "false" and "TRAP rc=5" in log


def test_a_switch_compares_as_the_controller_reads_it(harness):
    h = harness
    sup = "TRE_SAFESCALE_SUPPRESS_HOT_PROACTIVE"
    (h / "state" / f"ctl_{sup}").write_text("0")          # the overlay's value: false for the controller
    rc, d, ro = run_arm(h, "tre", 1)
    assert rc == 0, (d / "runner.log").read_text()[-1500:]
    # no `set env` at the start or the end: the arm's single rollout is its restart
    assert len(ro) == 1 and " rollout restart " in ro[0], ro
    assert switch(h, sup) == "0" and startup_line_matched(d)
    assert SWITCH_DEVIATION not in reset_check(h)
    # a value the controller does not know is still reported
    (h / "state" / f"ctl_{sup}").write_text("maybe")
    assert f"{sup}=maybe" in reset_check(h)
