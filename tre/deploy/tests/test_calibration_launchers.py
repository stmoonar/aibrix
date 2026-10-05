"""Calibration launchers (deploy/scripts/calibration/): the pre-flight guards refuse a run
the cluster is not ready for, against fakes of kubectl / pgrep / nc / curl (hermetic: no
cluster, no Redis). Tests the external contract only: which states start and which refuse.
"""
from __future__ import annotations

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


@pytest.mark.parametrize("script", sorted(p.name for p in CALIB.glob("run_*.sh")))
def test_launchers_parse_and_hard_code_no_site_path(script):
    if shutil.which("bash") is None:
        pytest.skip("bash not available")
    subprocess.run(["bash", "-n", str(CALIB / script)], check=True)
    text = (CALIB / script).read_text(encoding="utf-8")
    for literal in ("/data/nfs_shared_data", "192.168.", "nscc-ds-"):
        assert literal not in text, f"{script} hard-codes {literal}"
