"""Per-call drain (design note 20260924-reissue-sidecar.md section 3.3): hiding stays a
global SM switch, waiting for in-flight requests is chosen by the caller (``drain_s``)."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from tre_sm.api.v2 import create_app
from tre_sm.ops.drain import DrainConfig, SleepConfigError, normalize_drain_s

_HERE = Path(__file__).resolve().parent


def _load(name: str):
    spec = importlib.util.spec_from_file_location(f"_per_call_helpers_{name}", _HERE / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_staged = _load("test_staged_sleep")
_drain = _load("test_drain_before_sleep")
_make = _staged._make
_bindings = _staged._bindings
metrics_text = _drain.metrics_text
FakeVllmOps = _drain.FakeVllmOps


def _busy_then_idle(ip="10.0.0.2"):
    return {ip: [metrics_text(2, 1), metrics_text(1, 0), metrics_text(0, 0)]}


def _scrapes(events, ip="10.0.0.2"):
    return events.count(("metrics", ip))


def test_call_drain_budget_waits_for_in_flight_requests_even_with_default_off():
    events: list = []
    service, _runtime, vllm = _make(events, hide=True, drain=False, pages=_busy_then_idle())

    result = service.put_binding_power("serve-b", awake=False, drain_s=30)

    (outcome,) = result["sleep_outcomes"]
    assert outcome["outcome"] == "slept" and outcome["drained"] is True
    assert outcome["drain_budget_s"] == 30.0 and outcome["drain_budget_source"] == "call"
    assert outcome["interrupted"] == 0
    assert outcome["drained_s"] is not None and outcome["drained_s"] > 0
    assert _scrapes(events) == 3
    assert vllm.sleep_kwargs == [{"hidden": True}]


def test_call_drain_zero_skips_the_drain_even_with_default_on():
    events: list = []
    service, _runtime, vllm = _make(events, hide=True, drain=True, pages=_busy_then_idle())

    result = service.put_binding_power("serve-b", awake=False, drain_s=0)

    (outcome,) = result["sleep_outcomes"]
    assert outcome["outcome"] == "slept" and outcome["drained"] is False
    assert outcome["drain_budget_source"] == "call" and outcome["drain_budget_s"] is None
    assert _scrapes(events) == 0
    # hide -> wait unroutable -> /sleep with the header, nothing in between.
    assert events.index(("wait_unroutable", "serve-b")) < events.index(("sleep", "10.0.0.2"))
    assert vllm.sleep_kwargs == [{"hidden": True}]


@pytest.mark.parametrize("default_on, scrapes", [(True, 3), (False, 0)])
def test_calls_without_drain_s_use_the_env_default(default_on, scrapes):
    events: list = []
    service, _runtime, _vllm = _make(events, hide=True, drain=default_on, pages=_busy_then_idle())

    result = service.put_model_target("m1", wake_replicas=1)

    (outcome,) = result["sleep_outcomes"]
    assert outcome["drain_budget_source"] == "default"
    assert _scrapes(events) == scrapes


def test_default_drain_requires_explicit_opt_in():
    """Review L4: a non-zero default drain without TRE_SM_ALLOW_DEFAULT_DRAIN=1 fails
    at startup (TRE callers without drain_s would drain, APA never does)."""
    for value in ("true", "45"):
        with pytest.raises(SleepConfigError, match="TRE_SM_ALLOW_DEFAULT_DRAIN"):
            DrainConfig.from_env({"TRE_SM_HIDE_BEFORE_SLEEP": "1", "TRE_SM_DRAIN_BEFORE_SLEEP": value})
    # 0 / unset needs no opt-in
    assert DrainConfig.from_env({"TRE_SM_HIDE_BEFORE_SLEEP": "1", "TRE_SM_DRAIN_BEFORE_SLEEP": "0"}).enabled is False


def test_fixed_default_budget_from_env_number():
    allow = {"TRE_SM_HIDE_BEFORE_SLEEP": "1", "TRE_SM_ALLOW_DEFAULT_DRAIN": "1"}
    cfg = DrainConfig.from_env({**allow, "TRE_SM_DRAIN_BEFORE_SLEEP": "45"})
    assert cfg.enabled is True and cfg.default_budget_s == 45.0
    for off in ("", "0", "false", "no"):
        cfg = DrainConfig.from_env({"TRE_SM_HIDE_BEFORE_SLEEP": "1", "TRE_SM_DRAIN_BEFORE_SLEEP": off})
        assert cfg.enabled is False and cfg.default_budget_s is None
    auto = DrainConfig.from_env({**allow, "TRE_SM_DRAIN_BEFORE_SLEEP": "true"})
    assert auto.enabled is True and auto.default_budget_s is None
    with pytest.raises(ValueError):
        DrainConfig.from_env({"TRE_SM_HIDE_BEFORE_SLEEP": "1", "TRE_SM_DRAIN_BEFORE_SLEEP": "soon"})
    with pytest.raises(ValueError):
        DrainConfig.from_env({"TRE_SM_HIDE_BEFORE_SLEEP": "1", "TRE_SM_DRAIN_BEFORE_SLEEP": "-3"})
    # DRAIN default without HIDE still fails closed at startup (numbers included).
    with pytest.raises(ValueError):
        DrainConfig.from_env({"TRE_SM_DRAIN_BEFORE_SLEEP": "45"})

    events: list = []
    service, _runtime, _vllm = _make(
        events, hide=True, drain=True, default_budget_s=12.0, pages=_busy_then_idle()
    )
    (outcome,) = service.put_model_target("m1", wake_replicas=1)["sleep_outcomes"]
    assert outcome["drain_budget_s"] == 12.0 and outcome["drain_budget_source"] == "default"


def test_call_budget_is_capped_by_max_timeout_and_the_call_deadline():
    events: list = []
    stuck = {"10.0.0.2": [metrics_text(3, 0)]}
    service, _runtime, _vllm = _make(events, hide=True, drain=False, pages=stuck, max_timeout_s=50.0)

    (outcome,) = service.put_binding_power("serve-b", awake=False, drain_s=500)["sleep_outcomes"]

    assert outcome["drain_budget_s"] == 50.0
    assert outcome["drained"] is False and outcome["interrupted"] == 3
    assert outcome["drained_s"] == pytest.approx(50.0)


def test_drain_request_without_hide_is_refused_and_zero_keeps_the_legacy_path():
    events: list = []
    service, _runtime, vllm = _make(events, hide=False)

    with pytest.raises(ValueError, match="TRE_SM_HIDE_BEFORE_SLEEP"):
        service.put_binding_power("serve-b", awake=False, drain_s=5)
    with pytest.raises(ValueError, match="TRE_SM_HIDE_BEFORE_SLEEP"):
        service.put_model_target("m1", wake_replicas=1, drain_s=5)
    assert events == []

    # drain 0 == what the legacy (hide off) path already does: plain /sleep.
    result = service.put_binding_power("serve-b", awake=False, drain_s=0)
    assert result["actions"] == [{"action": "sleep", "serve_id": "serve-b"}]
    assert vllm.sleep_kwargs == [{}]
    assert "sleep_outcomes" not in result


@pytest.mark.parametrize("bad", [-1, float("nan"), float("inf"), "x", True])
def test_invalid_drain_values_are_rejected(bad):
    with pytest.raises(ValueError):
        normalize_drain_s(bad)


def test_api_accepts_drain_s_and_maps_errors_to_400():
    events: list = []
    service, _runtime, _vllm = _make(events, hide=True, drain=False, pages=_busy_then_idle())
    client = TestClient(create_app(service))

    bad = client.put("/v2/bindings/serve-b/power", json={"awake": False, "drain_s": -1})
    assert bad.status_code == 400
    ok = client.put("/v2/bindings/serve-b/power", json={"awake": False, "drain_s": 20})
    assert ok.status_code == 200, ok.text
    (outcome,) = ok.json()["sleep_outcomes"]
    assert outcome["drain_budget_s"] == 20.0 and outcome["drained"] is True

    events2: list = []
    legacy, _r, _v = _make(events2, hide=False)
    refused = TestClient(create_app(legacy)).put(
        "/v2/models/m1/target", json={"wake_replicas": 1, "drain_s": 3}
    )
    assert refused.status_code == 400
    assert "TRE_SM_HIDE_BEFORE_SLEEP" in refused.json()["detail"]


def test_apa_scale_service_never_drains_and_stays_synchronous():
    events: list = []
    # Default drain ON: the APA path must still pass drain_s=0.
    service, _runtime, vllm = _make(events, hide=True, drain=True, pages=_busy_then_idle())
    client = TestClient(create_app(service))

    response = client.post(
        "/scale_service", params={"model_name": "m1", "scale_type": "down", "scale_value": 1}
    )

    assert response.status_code == 200, response.text
    assert _scrapes(events) == 0
    assert vllm.sleep_kwargs == [{"hidden": True}]
    # Applied before the response returns: /models_replicas already sees it.
    assert client.post("/models_replicas", params={"models": "m1"}).json() == {"m1": 1}
    assert _bindings(service)["serve-b"].awake is False
