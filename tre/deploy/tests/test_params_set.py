"""deploy/scripts/release/params_set.py (replaces the console's PUT /api/params, 2026-10-20):
the --set parsing, the dry-run diff and the ConfigMap it would PUT. No cluster."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import yaml

from scripts import registry_params as rp

_P = Path(__file__).resolve().parents[1] / "scripts" / "release" / "params_set.py"
_spec = importlib.util.spec_from_file_location("params_set_under_test", _P)
params_set = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(params_set)

DEPLOY_ROOT = Path(__file__).resolve().parents[1]
REGISTRY = (DEPLOY_ROOT / "registry.yaml").read_text(encoding="utf-8")
NAMES = [m["name"] for m in rp.load_models(REGISTRY)]


def _edit(*sets: str) -> str:
    return rp.apply_and_validate(REGISTRY, rp.parse_sets(list(sets), NAMES))


def test_dry_run_changes_exactly_the_set_fields() -> None:
    model = NAMES[0]
    new = _edit(f"{model}.trs.theta_m=812.5", f"{model}.max_awake_replicas=3",
                f"{model}.alt_thresholds.queue_len.theta=7.25")
    changed = {(m, f) for m, f, _, _ in rp.changed_fields(REGISTRY, new)}
    assert changed == {(model, "trs.theta_m"), (model, "max_awake_replicas"),
                       (model, "alt_thresholds.queue_len.theta")}
    # everything else, other sections included, survives the re-render unchanged
    old_doc, new_doc = yaml.safe_load(REGISTRY), yaml.safe_load(new)
    assert {k: v for k, v in old_doc.items() if k != "models"} == {k: v for k, v in new_doc.items() if k != "models"}
    diff = rp.unified_diff(REGISTRY, new)
    assert sum(1 for line in diff.splitlines() if line.startswith("+") and not line.startswith("+++")) == 3
    assert rp.new_validation_errors(REGISTRY, new) == []


def test_setting_the_live_value_is_no_change() -> None:
    model = rp.load_models(REGISTRY)[0]
    new = _edit(f"{model['name']}.trs.theta_m={model['trs']['theta_m']}")
    assert rp.changed_fields(REGISTRY, new) == [] and rp.unified_diff(REGISTRY, new) == ""


@pytest.mark.parametrize("item, error", [
    ("nope.trs.theta_m=1", "unknown_model"),
    ("{m}.trs=1", "unknown_field"),
    ("{m}.foo=1", "unknown_field"),
    ("{m}.trs.theta_m", "expected_MODEL.PATH=VALUE"),
    ("{m}.max_awake_replicas=2.5", "not_a_int"),
    ("{m}.tp_size=2", "locked"),
    ("{m}.trs.ema_tau_ms=1000", "locked"),
])
def test_rejected_edits(item: str, error: str) -> None:
    with pytest.raises(rp.ParamValidationError) as exc:
        _edit(item.format(m=NAMES[0]))
    assert error in {e["error"] for e in exc.value.errors}


def test_model_names_with_dots_match_the_longest_prefix() -> None:
    edits = rp.parse_sets(["m.v2.trs.theta_m=5", "m.slo.ttft_p95_ms=600"], ["m", "m.v2"])
    assert edits == {"m.v2": {"trs": {"theta_m": 5}}, "m": {"slo": {"ttft_p95_ms": 600}}}


def test_only_new_validate_errors_block() -> None:
    class Reg:
        def __init__(self, path):
            self.text = Path(path).read_text(encoding="utf-8")

        def validate(self):
            return [line for line in self.text.splitlines() if line.startswith("bad")]

    assert rp.new_validation_errors("bad: old\n", "bad: old\n", loader=Reg) == []
    assert rp.new_validation_errors("bad: old\n", "bad: old\nbad: new\n", loader=Reg) == ["bad: new"]


def test_put_keeps_identity_and_the_resource_version_lock() -> None:
    cm = {"metadata": {"name": "tre-v2-registry", "namespace": "tre-v2", "resourceVersion": "42",
                       "uid": "u", "managedFields": [{}], "labels": {"a": "b"}},
          "data": {"registry.yaml": "old", "other": "x"}}
    obj = params_set.replacement(cm, "new")
    assert obj["metadata"] == {"name": "tre-v2-registry", "namespace": "tre-v2", "labels": {"a": "b"},
                               "resourceVersion": "42"}
    assert obj["data"] == {"registry.yaml": "new", "other": "x"}


def test_params_hash_is_the_console_annotation() -> None:
    # same value the console stamped (sha256 of registry.yaml, first 16 hex), so --status
    # compares correctly against Deployments the console restarted
    assert rp.params_hash("x") == "2d711642b726b044"
    assert rp.params_hash("") == rp.params_hash(None)
