"""The service-manager never drains (2026-10-02): every sleep path - SafeScale
commit, urgent donors, APA, maintenance - hides, waits for the gateway ack, reads
the load once for the record and sends ONE /sleep (mode=abort). ``no_drain_paths``
still parses (the baseline scaler reads it) but the service-manager ignores it; the
other drain-era keys are retired (not read)."""

from pathlib import Path

import pytest

from tre_common.registry import (
    DEFAULT_NO_DRAIN_PATHS,
    SLEEP_PATHS,
    load_registry,
    parse_service_manager_config,
)

from test_sleep_primitive import World

REPO_REGISTRY = Path(__file__).resolve().parents[2] / "deploy" / "registry.yaml"


def sleep_modes(world):
    return [call[2] for call in world.vllm.calls if call[0] == "sleep"]


def test_the_deprecated_drain_settings_still_parse():
    parsed = parse_service_manager_config(
        {"sleep": {"no_drain_paths": []}}
    )
    assert parsed.sleep.no_drain_paths == ()
    assert parse_service_manager_config(None).sleep.no_drain_paths == DEFAULT_NO_DRAIN_PATHS
    with pytest.raises(ValueError):
        parse_service_manager_config({"sleep": {"no_drain_paths": ["nope"]}})


@pytest.mark.parametrize("path", SLEEP_PATHS)
def test_every_path_aborts_at_once_whatever_the_drain_settings(path):
    world = World(no_drain_paths=())
    world.gateway.inflight("pod-a", "gw-1", total=2, non_continuable=1)
    world.vllm.load["10.0.0.1"] = 2

    [outcome] = world.sleep(path=path, drain_budget_s=60.0)

    assert sleep_modes(world) == ["abort"]
    assert outcome["status"] == "slept" and outcome["forced_abort"] is True
    assert outcome["waited_s"] < 1.0
    assert outcome["drain_policy"] == "no_drain"


def test_repo_registry_parses_with_the_whole_lock_timeouts():
    config = load_registry(REPO_REGISTRY).service_manager()
    assert config.sleep.ack_timeout_s == 5.0
    assert config.sleep.sleep_call_timeout_s == 20.0
    assert config.sleep.physical_confirm_timeout_s == 8.0
    assert config.wake_call_timeout_s == 10.0
    assert config.sleep.sleep_mode_when_idle == "abort"
