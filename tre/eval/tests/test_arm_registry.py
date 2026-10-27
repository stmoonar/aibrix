"""Per-arm registry params (tre/eval/runner/arm_registry.py), against a fake ConfigMap + params_set.

Invariants:
* after an arm - success, a failed apply, or a crash followed by the reset's restore - the live
  registry has its canonical content again, both Deployments run it (status in sync) and the
  restore record (ConfigMap annotation) is gone; a second restore is a no-op;
* the record of the original values is on the ConfigMap before the first write;
* tau_band scales tau_crit and tau_high together and never touches tau_low; a band that would
  cross tau_low is refused;
* "*" applies a factor to every model's own canonical value; a value set twice is refused.
"""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import pytest

yaml = pytest.importorskip("yaml")
_TRE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_TRE / "eval" / "runner"))
for _p in (_TRE / "common", _TRE / "deploy"):
    if str(_p) not in sys.path:
        sys.path.append(str(_p))

import arm_registry as AR  # noqa: E402
from scripts import registry_params as rp  # noqa: E402  the real params_set edit (coercion, bounds, render)

REGISTRY = """\
# comment lines are dropped by a params_set re-render; the registry id does not change
models:
- name: dsqwen-7b
  tp_size: 1
  trs: {theta_m: 771.0746050443564, w_p: 0.02, w_d: 1.0, lambda_wait: 2.375, tau_crit: 0.63, tau_low: 1.0,
        tau_high: 1.86, ema_tau_ms: 10000}
  alt_thresholds: {queue_len: {theta: 2.5, direction: lower_is_healthier}}
- name: dsqwen-14b
  tp_size: 2
  trs: {theta_m: 642.6049429483861, w_p: 0.0025, w_d: 1.0, lambda_wait: 2.25, tau_crit: 0.64, tau_low: 1.0,
        tau_high: 2.02, ema_tau_ms: 10000}
"""


def _hash(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:16]


class FakeCluster:
    """The registry ConfigMap (data + annotations + resourceVersion) and the two Deployments' applied hash.
    params_set: --set k=v / --apply / --restart / --status; the edit itself is the real params_set one
    (registry_params.parse_sets + apply_and_validate: coercion, bounds, re-render; the controller's full
    registry parser is skipped, the fixture is not a complete registry);
    ``fail`` = a predicate on the args that makes that call return rc 1 before it changes anything."""

    def __init__(self, text: str = REGISTRY, fail=None):
        self.text = text
        self.annotations: dict = {}
        self.rv = 1
        self.applied = {"sm": _hash(text), "controller": _hash(text)}
        self.fail = fail
        self.calls: list = []
        self.annotated_before_write: list = []

    def read_cm(self) -> dict:
        return {"metadata": {"name": "tre-v2-registry", "resourceVersion": str(self.rv), "annotations": dict(self.annotations)},
                "data": {"registry.yaml": self.text}}

    def annotate(self, value, resource_version=None):
        if resource_version is not None and resource_version != str(self.rv):
            raise AR.RegistryError("conflict")
        if value is None:
            self.annotations.pop(AR.ANNOTATION, None)
        else:
            self.annotations[AR.ANNOTATION] = value
        self.rv += 1

    def params_set(self, args):
        self.calls.append(list(args))
        if self.fail and self.fail(args):
            return 1, "REJECTED"
        if args == ["--status"]:
            ok = all(h == _hash(self.text) for h in self.applied.values())
            return 0, "ConfigMap ...\nin sync" if ok else "ConfigMap ...\nPENDING restart-to-apply: sm"
        sets = [args[i + 1] for i, a in enumerate(args) if a == "--set"]
        if sets:
            assert "--apply" in args and "--backup-dir" in args
            self.annotated_before_write.append(AR.ANNOTATION in self.annotations)
            names = [m["name"] for m in rp.load_models(self.text)]
            self.text = rp.apply_and_validate(self.text, rp.parse_sets(sets, names), loader=lambda path: None)
            self.rv += 1
        if "--restart" in args:
            self.applied = {k: _hash(self.text) for k in self.applied}
        return 0, "ok\nin sync"

    def value(self, key):
        return AR.values_at(self.text, [key])[key]


def rules(params=None, transforms=None):
    return AR.parse_spec("arm_defs.x", params, transforms)


def canonical(cl: FakeCluster) -> bool:
    return (AR.registry_id(cl.text) == AR.registry_id(REGISTRY) and AR.ANNOTATION not in cl.annotations
            and all(h == _hash(cl.text) for h in cl.applied.values()))


# ------------------------------------------------------------------ transforms
def test_tau_band_scales_crit_and_high_together_and_leaves_tau_low():
    sets, olds = AR.resolve(rules(transforms={"tau_band": 0.8}), REGISTRY)
    assert sets == {"dsqwen-14b.trs.tau_crit": 0.512, "dsqwen-14b.trs.tau_high": 1.616,
                    "dsqwen-7b.trs.tau_crit": 0.504, "dsqwen-7b.trs.tau_high": 1.488}
    assert not any(k.endswith("tau_low") for k in sets)
    assert olds["dsqwen-7b.trs.tau_crit"] == 0.63
    with pytest.raises(AR.RegistryError, match="tau_low"):          # 0.64 x 1.6 = 1.024 > tau_low
        AR.resolve(rules(transforms={"tau_band": 1.6}), REGISTRY)
    with pytest.raises(AR.RegistryError, match="tau_low"):          # 1.86 x 0.5 = 0.93 < tau_low
        AR.resolve(rules(transforms={"tau_band": 0.5}), REGISTRY)


def test_star_factor_uses_each_models_own_value_and_absolute_values_and_conflicts():
    sets, _ = AR.resolve(rules({"*.trs.theta_m": "x1.2", "*.trs.w_p": 1.0}), REGISTRY)
    assert sets["dsqwen-7b.trs.theta_m"] == pytest.approx(925.289526)
    assert sets["dsqwen-14b.trs.theta_m"] == pytest.approx(771.125932)
    assert sets["dsqwen-7b.trs.w_p"] == 1.0 and sets["dsqwen-14b.trs.w_p"] == 1.0
    with pytest.raises(AR.RegistryError, match="two rules"):
        AR.resolve(rules({"*.trs.tau_crit": "x0.9"}, {"tau_band": 0.9}), REGISTRY)
    assert AR.resolve(rules({"*.trs.w_d": 1.0}), REGISTRY) == ({}, {})   # no value change -> nothing to apply


@pytest.mark.parametrize("params, transforms, match", [
    ({"*.trs.ema_tau_ms": "x0.5"}, None, "locked"),
    ({"*.trs.theta": "x0.8"}, None, None),                      # unknown field: resolve refuses (not in registry)
    ({"*.image": 1}, None, "not <model"),
    ({"*.trs.theta_m": "0.8"}, None, "x<factor"),
    ({"*.trs.theta_m": True}, None, "neither"),
    (None, {"tau_scale": 0.8}, "unknown"),
])
def test_spec_validation(params, transforms, match):
    if match is None:
        with pytest.raises(AR.RegistryError, match="needs a numeric"):
            AR.resolve(rules(params, transforms), REGISTRY)
        return
    with pytest.raises(AR.RegistryError, match=match):
        rules(params, transforms)


# ------------------------------------------------------------------ apply / restore
def test_apply_then_restore_returns_the_canonical_registry():
    cl = FakeCluster()
    rec = AR.apply(cl, rules({"*.trs.theta_m": "x0.8"}), label="c/Simul_spike_s1/sens-theta-m20/a1", backup_dir="/b")
    assert rec["applied"] and rec["canonical_id"] == AR.registry_id(REGISTRY) != rec["perturbed_id"]
    assert cl.value("dsqwen-7b.trs.theta_m") == pytest.approx(616.859684)
    assert cl.annotated_before_write == [True]                  # restore map recorded before the first write
    assert all(h == _hash(cl.text) for h in cl.applied.values())  # restart-to-apply done (status in sync)
    res = AR.restore(cl, backup_dir="/b")
    assert res["canonical_id_match"] and canonical(cl)
    assert AR.restore(cl, backup_dir="/b") is None              # idempotent


def test_failed_apply_is_restored():
    cl = FakeCluster(fail=lambda a: "--apply" in a and "--set" in a and len(cl.calls) == 1)
    with pytest.raises(AR.RegistryError, match="rc=1"):
        AR.apply(cl, rules({"*.trs.lambda_wait": "x1.2"}), label="x", backup_dir="/b")
    assert AR.ANNOTATION in cl.annotations                      # the record survives the failure
    AR.restore(cl, backup_dir="/b")
    assert canonical(cl)


def test_crash_after_apply_is_seen_and_restored_by_the_reset():
    cl = FakeCluster()
    AR.apply(cl, rules(transforms={"tau_band": 1.2}), label="x", backup_dir="/b")
    # the campaign process dies here: no restore. A fresh process (reset_canonical.sh) sees it ...
    assert AR.in_effect(cl)["label"] == "x"
    AR.restore(cl, backup_dir="/b")
    assert canonical(cl) and AR.in_effect(cl) is None


def test_restore_after_values_written_back_but_restart_missing_only_restarts():
    cl = FakeCluster()
    AR.apply(cl, rules({"*.trs.w_p": 1.0}), label="x", backup_dir="/b")
    rec = AR.in_effect(cl)
    AR.restore(cl, backup_dir="/b")
    # simulate a crash between the restoring write and its restart: values back, restart pending, record kept
    cl.annotations[AR.ANNOTATION] = __import__("json").dumps(rec)
    cl.applied["controller"] = "stale"
    cl.calls.clear()
    AR.restore(cl, backup_dir="/b")
    assert [c for c in cl.calls if "--set" in c] == [] and any("--restart" in c for c in cl.calls)
    assert canonical(cl)


def test_apply_refuses_while_another_perturbation_is_in_effect():
    cl = FakeCluster()
    AR.apply(cl, rules({"*.trs.w_p": 1.0}), label="a", backup_dir="/b")
    with pytest.raises(AR.RegistryError, match="already in effect"):
        AR.apply(cl, rules({"*.trs.theta_m": "x0.8"}), label="b", backup_dir="/b")


def test_registry_id_ignores_comments_and_rendering_and_expectations():
    assert AR.registry_id(REGISTRY) == AR.registry_id(yaml.safe_dump(yaml.safe_load(REGISTRY)))
    rid = AR.registry_id(REGISTRY)
    assert AR.expect_problems({"registry_id": rid, "present": ["*.alt_thresholds.queue_len.theta"]}, REGISTRY) == [
        "dsqwen-14b.alt_thresholds.queue_len.theta missing in the live registry"]
    assert "not the registry" in AR.expect_problems({"registry_id": "0" * 16}, REGISTRY)[0]
    assert "not filled in" in AR.expect_spec_problems({"registry_id": None})[0]
    assert AR.expect_spec_problems({"registry_id": rid, "present": ["*.trs.theta_m"]}) == []


def test_set_arg_keeps_small_floats_floats_for_params_set():
    k, _, v = AR.set_arg("m.trs.w_p", 1e-05).partition("=")
    assert yaml.safe_load(v) == 1e-05 and isinstance(yaml.safe_load(v), float)


def test_int_valued_float_field_keeps_the_registry_id_after_restore():
    """params_set coerces float fields to float (lambda_wait: 3 -> 3.0): the id must not change, or every arm
    after the first perturbed one would fail the registry_expect pre-check."""
    text = REGISTRY.replace("lambda_wait: 2.375", "lambda_wait: 3")
    cl = FakeCluster(text)
    AR.apply(cl, rules({"*.trs.lambda_wait": "x1.2"}), label="x", backup_dir="/b")
    AR.restore(cl, backup_dir="/b")
    assert "lambda_wait: 3.0" in cl.text and AR.registry_id(cl.text) == AR.registry_id(text)


def test_restore_never_overwrites_a_third_value_and_reports_held():
    cl = FakeCluster()
    AR.apply(cl, rules({"*.trs.theta_m": "x0.8", "*.trs.w_p": 1.0}), label="x", backup_dir="/b")
    # someone else sets 7b's w_p during the arm
    names = [m["name"] for m in rp.load_models(cl.text)]
    cl.text = rp.apply_and_validate(cl.text, rp.parse_sets(["dsqwen-7b.trs.w_p=0.5"], names), loader=lambda p: None)
    with pytest.raises(AR.RegistryError, match="NOT overwritten: dsqwen-7b.trs.w_p = 0.5"):
        AR.restore(cl, backup_dir="/b")
    assert cl.value("dsqwen-7b.trs.w_p") == 0.5 and AR.ANNOTATION in cl.annotations     # kept for a human
    assert cl.value("dsqwen-7b.trs.theta_m") == pytest.approx(771.0746050443564)        # the arm's own keys went back
    # a perturbation that held to the end: held True; one undone during the arm: held False
    cl = FakeCluster()
    AR.apply(cl, rules({"*.trs.w_p": 1.0}), label="a", backup_dir="/b")
    assert AR.restore(cl, backup_dir="/b")["held"] is True
    AR.apply(cl, rules({"*.trs.w_p": 1.0}), label="b", backup_dir="/b")
    rec = cl.annotations[AR.ANNOTATION]
    AR.restore(cl, backup_dir="/b")
    cl.annotations[AR.ANNOTATION] = rec                                                  # values back early
    assert AR.restore(cl, backup_dir="/b")["held"] is False and canonical(cl)


def test_restore_leaves_another_runners_record_alone():
    cl = FakeCluster()
    AR.apply(cl, rules({"*.trs.w_p": 1.0}), label="other-campaign/a1", backup_dir="/b")
    n = len(cl.calls)
    assert "skipped" in AR.restore(cl, backup_dir="/b", expect_label="mine/a1")
    assert len(cl.calls) == n and AR.ANNOTATION in cl.annotations


def test_signal_log_theta_check(tmp_path):
    import json
    rows = [{"model": "dsqwen-7b", "theta_m": "616.859684"}, {"model": "dsqwen-14b", "theta_m": "642.6"},
            {"model": "dsqwen-7b", "theta_m": "nan"}]
    (tmp_path / "signal_log.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    sets = {"dsqwen-7b.trs.theta_m": 616.859684, "dsqwen-14b.trs.theta_m": 514.083954, "dsllama-8b.trs.theta_m": 1.0}
    bad, unver = AR.signal_log_theta_check(tmp_path, sets)
    assert bad == ["dsqwen-14b: signal_log theta_m [642.6] != applied 514.083954"]
    assert unver == ["dsllama-8b: no theta_m in signal_log.jsonl"]
    assert AR.signal_log_theta_check(tmp_path / "none", sets)[1][0].startswith("no signal_log")


def test_cluster_wraps_timeouts_as_registry_errors(monkeypatch):
    import subprocess

    def boom(*a, **k):
        raise subprocess.TimeoutExpired("kubectl", 120)
    monkeypatch.setattr(AR.subprocess, "run", boom)
    with pytest.raises(AR.RegistryError, match="TimeoutExpired"):
        AR.Cluster({}).read_cm()
    with pytest.raises(AR.RegistryError, match="TimeoutExpired"):
        AR.Cluster({}).params_set(["--status"])
