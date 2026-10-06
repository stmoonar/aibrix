"""set_run_mode.sh / toggle_tre_apa.sh / deploy_models.sh run-mode handling,
against a fake kubectl (hermetic: no cluster, no Redis).

Controller mode and SM actuation are independent switches (2026-09-28):
TRE arm = active + active, APA arm = observe + active, maintenance = observe +
observe."""
from __future__ import annotations

import os
import shutil
import stat
import subprocess
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
CTRL = "tre:v2:controller:mode"
SM = "tre:v2:sm:actuation"

FAKE_KUBECTL = r"""#!/usr/bin/env bash
echo "$*" >> "$FAKE_DIR/calls.log"
args=("$@")
for i in "${!args[@]}"; do
  if [[ "${args[$i]}" == "redis-cli" ]]; then
    cmd=("${args[@]:$((i + 2))}")   # skip redis-cli --raw
    op="${cmd[0]}"
    mkdir -p "$FAKE_DIR/kv"
    case "$op" in
      GET) [[ -f "$FAKE_DIR/redis_down" ]] && exit 1
           [[ -f "$FAKE_DIR/kv/${cmd[1]}" ]] && cat "$FAKE_DIR/kv/${cmd[1]}" || echo "" ;;
      SET) printf '%s\n' "${cmd[2]}" > "$FAKE_DIR/kv/${cmd[1]}"; echo OK ;;
      MSET)
        j=1
        while [[ $j -lt ${#cmd[@]} ]]; do
          printf '%s\n' "${cmd[$((j + 1))]}" > "$FAKE_DIR/kv/${cmd[$j]}"
          j=$((j + 2))
        done
        echo OK ;;
    esac
    exit 0
  fi
done
case "$*" in
  *"get podautoscalers"*)
    [[ -f "$FAKE_DIR/apa_crs" ]] && cat "$FAKE_DIR/apa_crs" ;;
esac
exit 0
"""


@pytest.fixture()
def fake(tmp_path):
    if shutil.which("bash") is None:
        pytest.skip("bash not available")
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    kubectl = bin_dir / "kubectl"
    kubectl.write_text(FAKE_KUBECTL, encoding="utf-8")
    kubectl.chmod(kubectl.stat().st_mode | stat.S_IEXEC)
    env = {**os.environ, "PATH": f"{bin_dir}{os.pathsep}{os.environ.get('PATH', '')}",
           "FAKE_DIR": str(tmp_path), "APA_DIR": str(tmp_path)}

    def run(script, *args, check=True):
        return subprocess.run(["bash", str(SCRIPTS / script), *args], env=env, text=True,
                              capture_output=True, check=check)

    def kv(key):
        path = tmp_path / "kv" / key
        return path.read_text(encoding="utf-8").strip() if path.exists() else None

    def calls():
        log = tmp_path / "calls.log"
        return log.read_text(encoding="utf-8").splitlines() if log.exists() else []

    return run, kv, calls


def test_set_run_mode_writes_both_in_one_mset_and_reads_back(fake):
    run, kv, calls = fake
    out = run("set_run_mode.sh", "observe", "active").stdout
    assert (kv(CTRL), kv(SM)) == ("observe", "active")
    assert "controller_mode=observe" in out and "sm_actuation=active" in out
    assert sum(" MSET " in line for line in calls()) == 1


def test_set_run_mode_can_set_one_switch_and_rejects_bad_values(fake):
    run, kv, _calls = fake
    run("set_run_mode.sh", "-", "active")
    assert (kv(CTRL), kv(SM)) == (None, "active")
    assert run("set_run_mode.sh", "on", "active", check=False).returncode != 0
    assert run("set_run_mode.sh", "-", "-", check=False).returncode != 0
    status = run("set_run_mode.sh", "status")
    assert f"{CTRL} missing -> controller treats as observe" in status.stderr


def test_toggle_apa_sets_controller_observe_first_and_sm_active(fake):
    run, kv, calls = fake
    run("toggle_tre_apa.sh", "apa")
    assert (kv(CTRL), kv(SM)) == ("observe", "active")
    log = calls()
    mset = next(i for i, line in enumerate(log) if " MSET " in line)
    apply_apa = next(i for i, line in enumerate(log) if " apply -f " in line)
    assert mset < apply_apa  # the controller stops acting before the source switch
    assert not any("set env" in line for line in log)  # run mode alone, no env switch


def test_toggle_tre_sets_both_active_after_the_switch(fake):
    run, kv, calls = fake
    run("toggle_tre_apa.sh", "tre")
    assert (kv(CTRL), kv(SM)) == ("active", "active")
    log = calls()
    mset = next(i for i, line in enumerate(log) if " MSET " in line)
    restart = next(i for i, line in enumerate(log) if "rollout restart" in line)
    assert mset > restart


def test_toggle_refuses_apa_while_the_controller_is_active_or_a_baseline_shell_owns(fake, tmp_path):
    run, kv, calls = fake
    run("set_run_mode.sh", "active", "active")
    assert run("toggle_tre_apa.sh", "apa", "--keep-run-mode", check=False).returncode != 0
    run("set_run_mode.sh", "observe", "active")
    (tmp_path / "kv" / "tre:v2:bl:owner").write_text("scaler-pod:1:abcd\n", encoding="utf-8")
    assert run("toggle_tre_apa.sh", "apa", "--keep-run-mode", check=False).returncode != 0
    assert run("toggle_tre_apa.sh", "tre", "--keep-run-mode", check=False).returncode != 0
    assert not any(" apply -f " in line for line in calls())


@pytest.mark.parametrize(
    "mode, crs, owner, source",
    [
        ("observe", "", "", "NONE"),
        ("active", "", "", "TRE"),
        ("observe", "podautoscaler/a\n", "", "APA"),
        ("observe", "", "shell:1", "BASELINE"),
        ("active", "", "shell:1", "CONFLICT"),
    ],
)
def test_toggle_status_decision_source_is_run_mode_apa_crs_and_owner_lock(fake, tmp_path, mode, crs, owner, source):
    run, _kv, _calls = fake
    run("set_run_mode.sh", mode, "active")
    (tmp_path / "apa_crs").write_text(crs, encoding="utf-8")
    if owner:
        (tmp_path / "kv" / "tre:v2:bl:owner").write_text(owner + "\n", encoding="utf-8")
    out = run("toggle_tre_apa.sh", "status").stdout
    assert f"active decision source: {source}" in out


def test_toggle_keep_run_mode_leaves_both_keys_alone(fake):
    run, kv, calls = fake
    run("toggle_tre_apa.sh", "apa", "--keep-run-mode")
    run("toggle_tre_apa.sh", "tre", "--keep-run-mode")
    assert (kv(CTRL), kv(SM)) == (None, None)
    assert not any("MSET" in line or " SET " in line for line in calls())
    assert run("toggle_tre_apa.sh", "tre", "--bogus", check=False).returncode == 2


def test_deploy_models_plain_apply_sets_the_requested_run_mode(fake, tmp_path):
    run, kv, _calls = fake
    run("deploy_models.sh", "--run-mode", "observe:active")
    assert (kv(CTRL), kv(SM)) == ("observe", "active")
    assert run("deploy_models.sh", "--run-mode", "observe", check=False).returncode != 0


def test_deploy_models_plain_apply_without_run_mode_warns_on_missing_keys(fake):
    run, kv, _calls = fake
    result = run("deploy_models.sh")
    assert (kv(CTRL), kv(SM)) == (None, None)
    assert f"{SM} missing -> SM treats as observe" in result.stderr


def test_toggle_fails_closed_when_redis_is_unreadable(fake, tmp_path):
    """An unreadable controller-mode key must abort `apa` (no CR apply) and `status`."""
    run, _kv, calls = fake
    (tmp_path / "redis_down").write_text("", encoding="utf-8")
    apa = run("toggle_tre_apa.sh", "apa", "--keep-run-mode", check=False)
    assert apa.returncode != 0 and "redis unreadable" in apa.stderr
    assert run("toggle_tre_apa.sh", "status", check=False).returncode != 0
    assert not any(" apply -f " in line for line in calls())
