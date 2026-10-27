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
    for section in ("vllm", "gateway", "reissue", "service_manager", "placement"):
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
    assert {"vllm", "gateway", "reissue", "service_manager", "placement"} <= set(merged)
    # B9: a live registry without service_manager.create gets the release section.
    assert merged["service_manager"]["create"] == release["service_manager"]["create"]
    assert any("trs.theta_m" in line for line in report)
    assert mlr._validate(merged) == []


def test_max_replicas_drift_is_warned():
    release = _release()
    live = copy.deepcopy(release)
    live["models"][0]["max_replicas"] = 6
    merged, report = mlr.merge(live, release)
    assert any(line.startswith("WARN") and "max_replicas" in line for line in report)
    # structural (params_set.py cannot edit it): the release layout wins
    assert merged["models"][0]["max_replicas"] == release["models"][0]["max_replicas"]


def test_cli_writes_a_valid_registry(tmp_path):
    release_path = DEPLOY_ROOT / "registry.yaml"
    live_path = tmp_path / "live.yaml"
    live_path.write_text(release_path.read_text(encoding="utf-8"), encoding="utf-8")
    out = tmp_path / "merged.yaml"
    assert mlr.main(["--live", str(live_path), "--release", str(release_path), "--out", str(out)]) == 0
    assert yaml.safe_load(out.read_text(encoding="utf-8")) == yaml.safe_load(release_path.read_text(encoding="utf-8"))


def test_live_theta_without_basis_stamps_keeps_its_true_basis():
    # H1 (10-28): a live theta_m fitted before the stamps is a completion / 30 s theta; the
    # release stamps (step_counters / 20000) must not relabel it, else the new controller
    # starts with every Z off by ~x1.5 instead of refusing.
    release = _release()
    live = copy.deepcopy(release)
    for model in live["models"]:
        model["trs"].pop("numerator", None)
        model["trs"].pop("theta_window_ms", None)
        model["trs"]["theta_m"] = 321.0
    merged, report = mlr.merge(live, release)
    for model in merged["models"]:
        assert model["trs"]["theta_m"] == 321.0
        assert model["trs"]["numerator"] == "completion_histograms"
        assert model["trs"]["theta_window_ms"] == 30000
    assert any(line.startswith("WARN") and "trs.numerator" in line for line in report)
    # a stamped live theta keeps its own stamps
    stamped, _ = mlr.merge(release, release)
    assert stamped["models"] == release["models"]


def test_an_alt_threshold_is_merged_as_one_fit_never_key_by_key():
    # M1 (10-28): a stale live theta (no bands, no basis) mixed with the release bands and
    # basis passed the controller's start check. The fit is one group: whole live or whole
    # release; a live theta without a basis keeps no basis, so an alt arm refuses to start.
    from tre_common.alt_signals import ALT_SIGNAL_DEFINITIONS, alt_basis_errors
    from tre_common.registry import _parse_registry

    release = _release()
    fit = {"theta": 216.2, "direction": "lower_is_healthier", "delta_crit": 0.81, "delta_high": 0.98,
           "basis": {"definition": ALT_SIGNAL_DEFINITIONS["ttft_mean"], "window_ms": 20000, "ema_tau_ms": 10000.0}}
    for model in release["models"]:
        model.setdefault("alt_thresholds", {})["ttft_mean"] = copy.deepcopy(fit)
    live = copy.deepcopy(release)
    stale = {"theta": 999.0, "direction": "lower_is_healthier"}  # a 07-10 fit: no bands, no basis
    for model in live["models"]:
        model["alt_thresholds"]["ttft_mean"] = copy.deepcopy(stale)
    merged, report = mlr.merge(live, release)
    for model in merged["models"]:
        assert model["alt_thresholds"]["ttft_mean"] == stale
    models = _parse_registry(merged).models()
    assert len(alt_basis_errors(models, "ttft_mean", window_ms=20000)) == len(models)
    assert any(line.startswith("WARN") and "no basis stamp" in line for line in report)
    # a live entry without theta: the release fit whole
    for model in live["models"]:
        model["alt_thresholds"]["ttft_mean"] = {"delta_crit": 0.5}
    merged, report = mlr.merge(live, release)
    assert all(m["alt_thresholds"]["ttft_mean"] == fit for m in merged["models"])
    assert any("has no theta" in line for line in report)
    # a stamped live fit is kept whole (its own bands and basis)
    newer = {**fit, "theta": 230.0, "delta_crit": 0.7, "basis": {**fit["basis"], "ema_tau_ms": 0.0}}
    for model in live["models"]:
        model["alt_thresholds"]["ttft_mean"] = copy.deepcopy(newer)
    merged, _ = mlr.merge(live, release)
    assert all(m["alt_thresholds"]["ttft_mean"] == newer for m in merged["models"])
