"""tools/preserve_mu on a tiny synthetic calibration capture (tests/fixtures/preserve_calib).

Fixture (model dsqwen-7b, two 30 s windows per cell, 1 replica):
  c100 hold: window 1 clean, 3 x (in 1000, out 100)  -> p 100, d 10, t 110 tok/s
             window 2 has a request with TPOT 100 ms > 75 ms -> excluded
  c101 hold: window 1 clean, 3 x (in 500, out 300)   -> p 50, d 30, t 80 tok/s
             window 2 has a model_error sent in it   -> excluded
  c102 hold but void, c103 static: huge loads        -> ignored
So mu = {p: 100, d: 30, t: 110}: each rate is its own maximum over clean windows.
"""
from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest
import yaml

from tre_baselines.tools import preserve_mu as pm

FIX = Path(__file__).resolve().parent / "fixtures" / "preserve_calib"
REGISTRY = str(Path(__file__).resolve().parents[2] / "deploy" / "registry.yaml")


@pytest.fixture(scope="module")
def label():
    from tre_common.registry import load_registry

    return pm.label_for("dsqwen-7b", load_registry(REGISTRY))


def test_request_criterion(label) -> None:
    prof = pm.profile_model(FIX / "dsqwen-7b", label)
    assert prof.model == "dsqwen-7b"
    assert prof.cells == 2 and prof.windows == 4 and prof.clean == 2
    assert prof.skipped_cells == {"void": 1, "primitive": 1}
    assert prof.as_mu() == {"p": 100.0, "d": 30.0, "t": 110.0}


def test_replicas_and_contamination(label) -> None:
    prof = pm.profile_model(FIX / "dsqwen-7b", label, replicas=2)
    assert prof.as_mu() == {"p": 50.0, "d": 15.0, "t": 55.0}
    strict = dataclasses.replace(label, tpot_p95_ms=15.0)  # c101's TPOT 10 ms passes, c100's 20 ms not
    assert pm.profile_model(FIX / "dsqwen-7b", strict).as_mu() == {"p": 50.0, "d": 30.0, "t": 80.0}


def test_label_criterion_agrees_on_the_fixture(label) -> None:
    lab = dataclasses.replace(label, min_completed_requests=0)
    prof = pm.profile_model(FIX / "dsqwen-7b", lab, criterion="label", min_latency_samples=1)
    assert prof.clean == 2 and prof.as_mu() == {"p": 100.0, "d": 30.0, "t": 110.0}
    # with the calibration's min-n guard (20) these tiny windows are unlabeled: no mu
    assert pm.profile_model(FIX / "dsqwen-7b", label, criterion="label").clean == 0


def test_cli_prints_the_policy_mu_format(capsys) -> None:
    assert pm.main([str(FIX), "--registry", REGISTRY]) == 0
    out = yaml.safe_load(capsys.readouterr().out)
    assert out == {"mu": {"dsqwen-7b": {"p": 100.0, "d": 30.0, "t": 110.0}}}
    # the output plugs straight into the policy's fail-closed parser
    from tre_baselines.policies.preserve_tier1 import parse_mu

    assert parse_mu(out["mu"], ["dsqwen-7b"])["dsqwen-7b"].d == 30.0
    # a model without any clean window -> exit 1, no entry
    assert pm.main([str(FIX / "dsqwen-7b"), "--registry", REGISTRY, "--window-s", "300"]) == 1
    assert yaml.safe_load(capsys.readouterr().out) == {"mu": {}}
    with pytest.raises(SystemExit):
        pm.main([str(FIX / "nowhere"), "--registry", REGISTRY])
