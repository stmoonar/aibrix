"""deploy/scripts/merge_live_registry.py: a release never resets live console tunables."""
from __future__ import annotations

import copy
from pathlib import Path

import yaml

from scripts import merge_live_registry as mlr

DEPLOY_ROOT = Path(__file__).resolve().parents[1]


def _release() -> dict:
    return yaml.safe_load((DEPLOY_ROOT / "registry.yaml").read_text(encoding="utf-8"))


def test_live_tunables_win_and_release_structure_stays():
    release = _release()
    live = copy.deepcopy(release)
    for section in ("vllm", "gateway", "reissue", "service_manager"):
        live.pop(section, None)
    model = live["models"][0]
    model["vllm_image"] = "old:image"
    model["trs"]["theta_m"] = 123.0
    model["alt_thresholds"]["queue_len"]["theta"] = 7.0
    model["slo"]["tpot_p95_ms"] = 60.0
    model["max_awake_replicas"] = 3
    merged, report = mlr.merge(live, release)
    out = merged["models"][0]
    assert out["vllm_image"] == release["models"][0]["vllm_image"]
    assert out["trs"]["theta_m"] == 123.0
    assert out["alt_thresholds"]["queue_len"]["theta"] == 7.0
    assert out["slo"]["tpot_p95_ms"] == 60.0
    assert out["max_awake_replicas"] == 3
    assert {"vllm", "gateway", "reissue", "service_manager"} <= set(merged)
    # B9: a live registry without service_manager.create gets the release section.
    assert merged["service_manager"]["create"] == release["service_manager"]["create"]
    assert any("trs.theta_m" in line for line in report)
    assert mlr._validate(merged) == []


def test_max_replicas_drift_is_warned():
    release = _release()
    live = copy.deepcopy(release)
    live["models"][0]["max_replicas"] = 6
    _, report = mlr.merge(live, release)
    assert any(line.startswith("WARN") and "max_replicas" in line for line in report)


def test_cli_writes_a_valid_registry(tmp_path):
    release_path = DEPLOY_ROOT / "registry.yaml"
    live_path = tmp_path / "live.yaml"
    live_path.write_text(release_path.read_text(encoding="utf-8"), encoding="utf-8")
    out = tmp_path / "merged.yaml"
    assert mlr.main(["--live", str(live_path), "--release", str(release_path), "--out", str(out)]) == 0
    assert yaml.safe_load(out.read_text(encoding="utf-8")) == yaml.safe_load(release_path.read_text(encoding="utf-8"))
