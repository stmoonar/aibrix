"""Calibration launchers (deploy/scripts/calibration/): the pre-flight guards refuse a run
the cluster is not ready for, against fakes of kubectl / pgrep / nc / curl (hermetic: no
cluster, no Redis). Tests the external contract only: which states start and which refuse.
"""
from __future__ import annotations

import json
import os
import shutil
import stat
import subprocess
from pathlib import Path

import pytest

CALIB = Path(__file__).resolve().parents[1] / "scripts" / "calibration"

FAKE_KUBECTL = r"""#!/usr/bin/env bash
case "$*" in
  *"get svc"*) echo 10.0.0.1 ;;
  *"envoy-gateway-system get pods"*) echo "envoy-tre-v2-tre-aibrix-eg-x 10.0.0.2" ;;
  *"redis-cli --raw GET tre:v2:controller:mode"*) cat "$FAKE_DIR/ctrl" 2>/dev/null ;;
  *"redis-cli --raw GET tre:v2:sm:actuation"*) cat "$FAKE_DIR/sm" 2>/dev/null ;;
  *"get pods -l"*"routable=true"*) echo dsqwen-7b-pod-0 ;;
  *"get pods -l"*) echo dsqwen-7b-pod-0 ;;
  *"exec dsqwen-7b-pod-0"*) printf 'vllm:num_requests_running{x="y"} %s\nvllm:num_requests_waiting{x="y"} 0\n' \
                              "$(cat "$FAKE_DIR/running" 2>/dev/null || echo 0)" ;;
esac
exit 0
"""


@pytest.fixture()
def cluster(tmp_path):
    if shutil.which("bash") is None:
        pytest.skip("bash not available")
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    fakes = {
        "kubectl": FAKE_KUBECTL,
        "pgrep": "#!/usr/bin/env bash\nexit 1\n",
        "nc": "#!/usr/bin/env bash\ncat >/dev/null; echo +PONG\n",
        "curl": "#!/usr/bin/env bash\necho 'envoy_cluster_upstream_rq_pending_overflow{} 0'\n",
    }
    for name, body in fakes.items():
        p = bin_dir / name
        p.write_text(body, encoding="utf-8")
        p.chmod(p.stat().st_mode | stat.S_IEXEC)
    (tmp_path / "ctrl").write_text("observe\n")
    (tmp_path / "sm").write_text("observe\n")
    marker = tmp_path / "TRE_EXCLUSIVE_WINDOW"
    marker.write_text("calibration\n")
    env = {**os.environ, "PATH": f"{bin_dir}{os.pathsep}{os.environ.get('PATH', '')}",
           "FAKE_DIR": str(tmp_path), "TRE_EXCLUSIVE_WINDOW_FILE": str(marker)}

    def preflight(out: Path, dry: int = 0, **extra):
        script = f'source "{CALIB / "lib.sh"}"; calib_preflight dsqwen-7b "{out}" {dry}'
        return subprocess.run(["bash", "-c", script], env={**env, **extra}, text=True,
                              capture_output=True)

    return tmp_path, preflight


def test_ready_cluster_starts_and_reports_the_pinning(cluster):
    tmp, preflight = cluster
    r = preflight(tmp / "out", CALIB_TASKSET="48-63")
    assert r.returncode == 0, r.stderr
    assert "taskset=48-63" in r.stdout


@pytest.mark.parametrize("key,value,needle", [
    ("ctrl", "active", "controller mode"),
    ("sm", "active", "SM actuation"),
    ("sm", "", "SM actuation"),            # a missing key is not "written observe"
    ("running", "3", "not idle"),
])
def test_refuses_a_cluster_not_in_calibration_state(cluster, key, value, needle):
    tmp, preflight = cluster
    (tmp / key).write_text(value + "\n")
    r = preflight(tmp / "out")
    assert r.returncode != 0
    assert needle in r.stderr


def test_refuses_a_non_empty_out_dir_on_a_real_run_only(cluster):
    tmp, preflight = cluster
    out = tmp / "out"
    out.mkdir()
    (out / "partial.csv").write_text("x\n")
    assert "not empty" in preflight(out).stderr
    assert preflight(out, dry=1).returncode == 0


def test_real_run_needs_the_exclusive_window_marker(cluster):
    tmp, preflight = cluster
    (tmp / "TRE_EXCLUSIVE_WINDOW").unlink()
    r = preflight(tmp / "out")
    assert r.returncode != 0 and "exclusive-window" in r.stderr
    assert preflight(tmp / "out", dry=1).returncode == 0


def _supplement_chain(tmp_path, *, git_repo: bool, blocks: str):
    """run_chain_supp.sh --dry-run dsqwen-7b against fakes of run_stage3.sh (block A fails)
    and kubectl; returns (CompletedProcess, [(step, event, rc, detail)])."""
    if shutil.which("bash") is None or shutil.which("flock") is None or shutil.which("git") is None:
        pytest.skip("bash / flock / git not available")
    repo = tmp_path / "repo"
    repo.mkdir()
    if git_repo:
        subprocess.run(["git", "init", "-q", str(repo)], check=True)  # CALIB_REPO: a clean repo
    cal = tmp_path / "cal"
    cal.mkdir()
    for name in ("run_chain_supp.sh", "lib.sh"):
        shutil.copy(CALIB / name, cal / name)
    (cal / "run_stage3.sh").write_text(
        '#!/usr/bin/env bash\n[ "$TRAINING_PLAN" = supp-a-20261103 ] && exit 1\nexit 0\n', encoding="utf-8")
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    kubectl = bin_dir / "kubectl"
    kubectl.write_text("#!/usr/bin/env bash\necho 'dsqwen-7b-pod-0 vllm-openai=0,'\n", encoding="utf-8")
    kubectl.chmod(kubectl.stat().st_mode | stat.S_IEXEC)
    logs = tmp_path / "logs"
    env = {**os.environ, "PATH": f"{bin_dir}{os.pathsep}{os.environ.get('PATH', '')}",
           "CHAIN_LOG_DIR": str(logs), "CALIB_ROOT": str(tmp_path / "root"), "BASE_RUN": str(tmp_path / "base"),
           "CHAIN_BLOCKS": blocks, "CHAIN_LOCK_DIR": str(tmp_path / "lock"), "CALIB_REPO": str(repo)}
    r = subprocess.run(["bash", str(cal / "run_chain_supp.sh"), "--dry-run", "dsqwen-7b"], env=env, text=True,
                       capture_output=True)
    path = logs / "dryrun" / "logs" / "CHAIN_STATUS.jsonl"
    assert path.exists(), f"no status line at all (rc {r.returncode}): {r.stderr}"
    status = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    return r, [(d["step"], d["event"], d["rc"], d["detail"]) for d in status]


def test_supplement_chain_records_a_failed_block_and_runs_the_next(tmp_path):
    """run_chain_supp.sh: a block that exits non-zero gets its `end` and `block_failed`
    status lines, the next block still runs, and the chain exits 1 (2026-11-03: `step`
    turned `set -e` back on before returning, so the model's chain died at the failed
    block without a status line)."""
    r, status = _supplement_chain(tmp_path, git_repo=True, blocks="A B")
    events = [(step, event, rc) for step, event, rc, _ in status]
    assert ("A", "end", 1) in events and ("A", "block_failed", 1) in events, (events, r.stderr)
    assert ("B", "end", 0) in events
    assert ("chain", "done", 1) in events
    assert r.returncode == 1


def test_supplement_chain_reports_an_unreadable_repo(tmp_path):
    """A CALIB_REPO git cannot read (not a repo; "dubious ownership" on another uid's NFS
    checkout) is reported as a status line - a dry run warns and goes on, a real run refuses
    through the same line - instead of ending the script before its first line (exit 128)."""
    r, status = _supplement_chain(tmp_path, git_repo=False, blocks="B")
    assert any(step == "code" and event == "warning" and "git status failed" in detail
               for step, event, _, detail in status), (status, r.stderr)
    assert r.returncode == 0, r.stderr


@pytest.mark.parametrize("script", sorted(p.name for p in CALIB.glob("run_*.sh")))
def test_launchers_parse_and_hard_code_no_site_path(script):
    if shutil.which("bash") is None:
        pytest.skip("bash not available")
    subprocess.run(["bash", "-n", str(CALIB / script)], check=True)
    text = (CALIB / script).read_text(encoding="utf-8")
    for literal in ("/data/nfs_shared_data", "192.168.", "nscc-ds-"):
        assert literal not in text, f"{script} hard-codes {literal}"
