"""run_arm_pilot.sh / reset_canonical.sh on the offline harness (runner/selftest/make_harness.sh: fake
kubectl, curl, redis-py, loadgen; no cluster). Invariants (2026-11-02, docs runner-overhead-20261102 C2):

* one controller rollout per arm, after the arm's run mode is set (note 16a): the arm's controller is started
  by the ``set env`` rollout when a switch differs, else by a restart; the startup-line check passes either way;
* CTL_ENV_KEEP=1 (campaign): an arm leaves its switches set and the next arm's rollout writes every one of
  them; reset_canonical.sh leaves them with CTL_ENV_KEEP=1 and puts them back without it; without
  CTL_ENV_KEEP an arm puts them back at its end, as before;
* a switch compares as the controller reads it (_get_bool): the overlay's "0" is false - no rollout, no
  deviation; a value the controller does not know is reported;
* every collector file is still written, and an arm that dies in its collection puts its switches back
  (CTL_ENV_KEEP=1 too);
* (eta removed 2026-11-02) TRE_DISABLE_ETA_GATE left on the Deployment is unset in the arm's one rollout and by
  the final reset; a controller image that still logs disable_eta_gate never matches.
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
    assert rc == 0 and len(ro) == 1, ro
    assert (d / "controller_restart_epoch").is_file() and startup_line_matched(d)
    assert switch(h) == "true"
    # reset between arms (campaign): the kept switch is not a deviation; without CTL_ENV_KEEP it is
    assert SWITCH_DEVIATION not in reset_check(h, CTL_ENV_KEEP="1")
    assert SWITCH_DEVIATION in reset_check(h)
    # next arm without switches: ONE rollout that writes the switch back to its default and starts the arm
    rc, d, ro = run_arm(h, "apa", 2, CTL_ENV_KEEP="1")
    assert rc == 0 and len(ro) == 1, ro
    assert switch(h) == "false" and startup_line_matched(d)
    # nothing to change: still one rollout
    rc, d, ro = run_arm(h, "tre", 3, CTL_ENV_KEEP="1")
    assert rc == 0 and len(ro) == 1 and startup_line_matched(d), ro
    # the end of a campaign (no CTL_ENV_KEEP): reset_canonical.sh restores a switch left set
    (h / "state" / "ctl_TRE_ABLATION_DISABLE_SLOW_LOOP").write_text("true")
    subprocess.run(["bash", str(h / "tools" / "reset_canonical.sh")], cwd=h, capture_output=True, env=env_of(h, CTL_ENV_KEEP="1"))
    assert switch(h) == "true"
    subprocess.run(["bash", str(h / "tools" / "reset_canonical.sh")], cwd=h, capture_output=True, env=env_of(h))
    assert switch(h) == "false"


def test_the_arms_controller_starts_after_its_run_mode_is_set_with_one_rollout(harness):
    """note 16a (campaign mode): whether the switches change (set env) or not (restart), the arm's run mode is
    set before the one rollout that starts the arm's controller."""
    h = harness
    for n, (arm, kw) in enumerate((("tre", {"ARM_CONTROLLER_ENV": "TRE_ABLATION_DISABLE_SLOW_LOOP=true"}),
                                   ("apa", {}), ("tre", {})), 1):
        before = len(calls(h))
        rc, d, ro = run_arm(h, arm, n, CTL_ENV_KEEP="1", **kw)
        assert rc == 0 and len(ro) == 1, (n, ro)
        new = calls(h)[before:]
        modes = [c for c in new[:new.index(ro[0])] if c.startswith("set_run_mode ")]
        assert modes and modes[-1] == ("set_run_mode active active" if arm == "tre" else "set_run_mode observe active"), (n, modes)


def test_standalone_arm_puts_its_switches_back_and_collects_every_file(harness):
    h = harness
    rc, d, ro = run_arm(h, "tre", 1, ARM_CONTROLLER_ENV="TRE_ABLATION_DISABLE_SLOW_LOOP=true")
    assert rc == 0, (d / "runner.log").read_text()[-1500:]
    # the merged start rollout + the end-of-arm restore (was: set env + restart + restore)
    assert len(ro) == 2, ro
    assert switch(h) == "false" and startup_line_matched(d)
    missing = [f for f in COLLECTED if not (d / f).exists()]
    assert not missing, missing


def test_an_arm_that_dies_in_its_collection_puts_its_switches_back_in_campaign_mode(harness):
    h = harness
    (h / "tools" / "snap.py").write_text(
        "import json, sys\nif sys.argv[1].endswith('snap_after.json'):\n    sys.exit(5)\njson.dump({}, open(sys.argv[1], 'w'))\n")
    rc, d, _ = run_arm(h, "tre", 1, ARM_CONTROLLER_ENV="TRE_ABLATION_DISABLE_SLOW_LOOP=true", CTL_ENV_KEEP="1")
    log = (d / "runner.log").read_text()
    # snap_after is fatal (as before): the arm fails, and the exit trap puts the switch back although CTL_ENV_KEEP=1
    assert rc != 0 and switch(h) == "false" and "TRAP rc=" in log


def test_a_switch_compares_as_the_controller_reads_it(harness):
    h = harness
    sup = "TRE_SAFESCALE_SUPPRESS_HOT_PROACTIVE"
    (h / "state" / f"ctl_{sup}").write_text("0")          # the overlay's value: false for the controller
    rc, d, ro = run_arm(h, "tre", 1)
    assert rc == 0, (d / "runner.log").read_text()[-1500:]
    # no `set env` at the start or the end: one rollout (the arm's restart), the "0" left as it is
    assert len(ro) == 1 and not [c for c in ro if " set env " in c], ro
    assert switch(h, sup) == "0" and startup_line_matched(d)
    assert SWITCH_DEVIATION not in reset_check(h)
    # a value the controller does not know is still reported
    (h / "state" / f"ctl_{sup}").write_text("maybe")
    assert f"{sup}=maybe" in reset_check(h)


def test_a_removed_switch_left_on_the_deployment_is_unset_in_the_arms_one_rollout_and_by_the_final_reset(harness):
    """A stale TRE_DISABLE_ETA_GATE would crash-loop the current controller ("true") or linger ("false")."""
    h = harness
    stale = h / "state" / "ctl_TRE_DISABLE_ETA_GATE"
    stale.write_text("true")
    assert SWITCH_DEVIATION not in reset_check(h, CTL_ENV_KEEP="1")      # between arms: the next arm unsets it
    assert "TRE_DISABLE_ETA_GATE=true(removed)" in reset_check(h)
    rc, d, ro = run_arm(h, "tre", 1, CTL_ENV_KEEP="1")
    assert rc == 0 and len(ro) == 1 and "TRE_DISABLE_ETA_GATE-" in ro[0].split(), ro
    assert not stale.exists() and startup_line_matched(d)
    stale.write_text("false")
    subprocess.run(["bash", str(h / "tools" / "reset_canonical.sh")], cwd=h, capture_output=True, env=env_of(h))
    assert not stale.exists()


def test_a_controller_image_from_before_the_eta_removal_never_matches(harness):
    """Its startup line still logs disable_eta_gate: donor order not IDLE then Z. An arm with switches stops."""
    import json
    h = harness
    (h / "state" / "old_image_line").write_text("")
    rc, d, _ = run_arm(h, "tre", 1, ARM_CONTROLLER_ENV="TRE_ABLATION_DISABLE_SLOW_LOOP=true")
    sw = json.loads((d / "ablation_switches.json").read_text())
    assert rc == 4 and sw["match"] is False and "eta removal" in (sw["problem"] or ""), sw
