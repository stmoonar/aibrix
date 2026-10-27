"""run_freeze_m2.sh hands one TSS numerator and one window to every stage that has the
notion: the dataset builder (--numerator / --window-ms / --step-ms) and the dline_refit fit
stages (--window-ms / --step-ms; the freeze records them). Hermetic: python3 is a fake that
logs its arguments, so nothing is built or fitted."""
from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "calibration" / "run_freeze_m2.sh"


def _run(tmp_path: Path, **env: str) -> subprocess.CompletedProcess:
    if shutil.which("bash") is None:
        pytest.skip("bash not available")
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    fake = bin_dir / "python3"
    fake.write_text('#!/usr/bin/env bash\necho "$*" >> "$FAKE_LOG"\n')
    fake.chmod(0o755)
    (tmp_path / "calib").mkdir()
    full = {**os.environ, "PATH": f"{bin_dir}:{os.environ['PATH']}", "FAKE_LOG": str(tmp_path / "calls.log"),
            "CALIB_ROOT": str(tmp_path / "calib"), "OUT_DIR": str(tmp_path / "out"), "MODELS": "dsqwen-7b",
            "JOBS": "1", **env}
    return subprocess.run(["bash", str(SCRIPT)], env=full, text=True, capture_output=True, timeout=60)


def _calls(tmp_path: Path, module: str) -> list[list[str]]:
    lines = (tmp_path / "calls.log").read_text().splitlines()
    return [ln.split() for ln in lines if f"-m {module} " in f"{ln} "]


def _flag(argv: list[str], name: str) -> str:
    assert name in argv, argv
    return argv[argv.index(name) + 1]


@pytest.mark.parametrize("env,numerator,window", [
    ({}, "vllm_counter", "20000"),  # the 2026-10-28 default: step counters on 20 s windows
    ({"NUMERATOR": "gateway", "WINDOW_MS": "30000"}, "gateway", "30000"),  # reproduces the 2026-10-25 freeze
])
def test_every_stage_gets_the_numerator_and_the_window(tmp_path, env, numerator, window) -> None:
    proc = _run(tmp_path, **env)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    builds = _calls(tmp_path, "scripts.calibration_dataset")
    assert len(builds) == 4  # run1 run2 supp p1 of the one model
    for argv in builds:
        assert _flag(argv, "--numerator") == numerator and _flag(argv, "--window-ms") == window
        assert _flag(argv, "--step-ms") == "10000"
    fits = [a for a in _calls(tmp_path, "scripts.dline_refit") if a[2] in ("alpha", "wp", "final")]
    assert sorted(a[2] for a in fits) == ["alpha", "final", "wp"]
    for argv in fits:
        assert float(_flag(argv, "--window-ms")) == float(window) and _flag(argv, "--step-ms") == "10000"
    record = (tmp_path / "out" / "run_record.txt").read_text()
    assert f"numerator={numerator} window_ms={window}" in record


def test_an_unknown_numerator_is_refused_before_anything_runs(tmp_path) -> None:
    proc = _run(tmp_path, NUMERATOR="completion")
    assert proc.returncode == 2 and "NUMERATOR" in proc.stderr
    assert not (tmp_path / "calls.log").exists()
