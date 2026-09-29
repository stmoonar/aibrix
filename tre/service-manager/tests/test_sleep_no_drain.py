"""No-drain sleep paths (v1 / paper semantics, 2026-09-29).

Only the SafeScale probe window drains a TRE scale-down (the pod is hidden while
it runs); the SafeScale commit, the fast-loop donors (urgent) and APA
(/scale_service -> "apa") hide, wait for the gateway ack and then sleep with
mode=abort at once. Non-continuable requests are cut off (counted), never waited
for and never a reason to roll back. Maintenance paths keep draining, and
``service_manager.sleep.no_drain_paths: []`` restores the old behaviour.
"""

from pathlib import Path

import pytest

from tre_common.registry import (
    DEFAULT_NO_DRAIN_PATHS,
    ServiceManagerConfig,
    SleepPolicy,
    _validate_service_manager,
    load_registry,
    parse_service_manager_config,
)
from tre_sm.ops.sleep_primitive import GatewayAckTimeout, SleepIncomplete

from sm_test_fakes import Result
from test_sleep_primitive import World

REPO_REGISTRY = Path(__file__).resolve().parents[2] / "deploy" / "registry.yaml"


def v1_world(**overrides):
    """A World on the production default: the v1 no-drain paths."""
    overrides.setdefault("no_drain_paths", DEFAULT_NO_DRAIN_PATHS)
    return World(**overrides)


def sleep_modes(world):
    return [call[2] for call in world.vllm.calls if call[0] == "sleep"]


# ------------------------------------------------------------------ defaults
def test_production_default_no_drain_paths_are_commit_urgent_and_apa():
    assert SleepPolicy().no_drain_paths == ("safescale_commit", "urgent", "apa")
    parsed = parse_service_manager_config(None)
    assert parsed.sleep.no_drain_paths == DEFAULT_NO_DRAIN_PATHS
    for path in ("safescale_commit", "urgent", "apa"):
        assert parsed.sleep.no_drain(path)
        assert parsed.sleep.soft_budget_s(path, 60.0) == 0.0  # caller budget ignored
    for path in ("scale_down", "defrag", "repair", "startup", "default"):
        assert not parsed.sleep.no_drain(path)
    assert parsed.sleep.soft_budget_s("defrag") == 30.0
    with pytest.raises(ValueError):  # a bad caller budget is still rejected
        parsed.sleep.soft_budget_s("urgent", -1)


def test_no_drain_paths_can_be_switched_off_and_are_validated():
    assert parse_service_manager_config({"sleep": {"no_drain_paths": []}}).sleep.no_drain_paths == ()
    assert parse_service_manager_config({"sleep": {"no_drain_paths": None}}).sleep.no_drain_paths == ()
    only = parse_service_manager_config({"sleep": {"no_drain_paths": ["urgent", "urgent"]}})
    assert only.sleep.no_drain_paths == ("urgent",)
    with pytest.raises(ValueError, match="unknown sleep path 'fast'"):
        parse_service_manager_config({"sleep": {"no_drain_paths": ["fast"]}})
    with pytest.raises(ValueError, match="must be a list"):
        parse_service_manager_config({"sleep": {"no_drain_paths": "urgent"}})
    good = _validate_service_manager(
        parse_service_manager_config({"sleep": {"no_drain_paths": ["urgent"]}})
    )
    assert not [e for e in good if "no_drain_paths" in e]
    bad = _validate_service_manager(
        ServiceManagerConfig(sleep=SleepPolicy(no_drain_paths=("bogus",)))
    )
    assert "service_manager.sleep.no_drain_paths: unknown sleep path bogus" in bad


def test_repo_registry_ships_the_v1_no_drain_default():
    registry = load_registry(str(REPO_REGISTRY))
    assert registry.service_manager().sleep.no_drain_paths == DEFAULT_NO_DRAIN_PATHS


# ------------------------------------------------------- CRIT donor (urgent)
def test_urgent_donor_does_not_wait_and_aborts_continuable_requests():
    world = v1_world()
    world.gateway.inflight("pod-a", "gw-1", total=3)  # never finishes
    world.vllm.load["10.0.0.1"] = 3

    [outcome] = world.sleep(path="urgent", drain_budget_s=30.0)

    assert outcome["status"] == "slept"
    assert outcome["drain_policy"] == "no_drain"
    assert outcome["sleep_mode"] == "abort" and outcome["forced_abort"] is True
    assert outcome["waited_s"] < 1.0  # no drain wait (was the 30 s urgent budget)
    assert outcome["aborted"] == {
        "state_known": True, "in_flight": 3, "continuable": 3, "non_continuable": 0,
        "unclassified": 0,
    }
    stats = world.stats()
    assert stats["no_drain_sleeps_total"] == 1
    assert stats["forced_abort_requests_total"] == 3
    assert "no_drain_non_continuable_aborted_total" not in stats
    kinds = [event[:3] for event in world.events]
    assert kinds[0] == ("patch", "pod-a", "hidden")  # hide + ack still first


def test_urgent_donor_aborts_non_continuable_instead_of_waiting_or_rolling_back():
    world = v1_world(hard_cap_s=60.0)
    world.gateway.inflight("pod-a", "gw-1", total=3, non_continuable=2)
    world.vllm.load["10.0.0.1"] = 4  # one request the gateway did not count

    [outcome] = world.sleep(path="urgent")

    assert outcome["status"] == "slept"
    assert sleep_modes(world) == ["abort"]
    assert outcome["waited_s"] < 1.0
    assert outcome["aborted"] == {
        "state_known": True, "in_flight": 4, "continuable": 1, "non_continuable": 2,
        "unclassified": 1,
    }
    assert world.runtime.patches[-1][1] == "sleeping"
    stats = world.stats()
    assert stats["no_drain_non_continuable_aborted_total"] == 2
    assert stats["no_drain_unclassified_aborted_total"] == 1
    assert "non_continuable_rollback_total" not in stats
    assert "rollback_total" not in stats


def test_urgent_donor_with_unknown_drain_state_aborts_and_records_it():
    world = v1_world()
    world.vllm.metrics_down.add("10.0.0.1")  # engine metrics unavailable

    [outcome] = world.sleep(path="urgent")

    assert outcome["status"] == "slept" and outcome["sleep_mode"] == "abort"
    assert outcome["aborted"]["state_known"] is False
    assert outcome["waited_s"] < 1.0
    assert world.stats()["no_drain_unknown_state_abort_total"] == 1
    assert "drain_unknown_rollback_total" not in world.stats()


def test_no_drain_with_nothing_in_flight_sleeps_with_mode_wait():
    world = v1_world()

    [outcome] = world.sleep(path="urgent")

    assert sleep_modes(world) == ["wait"]
    assert outcome["drained"] is True and outcome["forced_abort"] is False
    assert outcome["aborted"] is None


def test_no_drain_wait_failure_aborts_even_with_unknown_state():
    world = v1_world()
    world.vllm.sleep_results = [Result(False, "timed out")]
    original = world.vllm.sleep

    def sleep(pod_ip, **kwargs):
        world.vllm.metrics_down.add(pod_ip)  # engine unreadable after the call
        return original(pod_ip, **kwargs)

    world.vllm.sleep = sleep

    [outcome] = world.sleep(path="urgent")

    assert sleep_modes(world) == ["wait", "abort"]
    assert outcome["status"] == "slept" and outcome["forced_abort"] is True
    assert outcome["aborted"]["state_known"] is False


def test_no_drain_abort_that_errors_but_slept_still_counts_what_it_cut_off():
    world = v1_world()
    world.gateway.inflight("pod-a", "gw-1", total=2, non_continuable=1)
    world.vllm.sleep_results = [Result(False, "read timeout")]
    probes = []
    original_probe = world.vllm.is_sleeping

    def is_sleeping(pod_ip, **kwargs):
        # The first probe (right after the failed /sleep) still sees it awake; the
        # rollback's re-probe finds it asleep -> recorded as slept, not re-routed.
        probes.append(pod_ip)
        if len(probes) == 1:
            return False
        world.vllm.sleeping[pod_ip] = True
        return original_probe(pod_ip, **kwargs)

    world.vllm.is_sleeping = is_sleeping

    [outcome] = world.sleep(path="urgent")

    assert len(probes) >= 2
    assert outcome["status"] == "slept" and outcome["forced_abort"] is True
    assert outcome["aborted"]["non_continuable"] == 1
    assert world.stats()["no_drain_non_continuable_aborted_total"] == 1


def test_no_drain_still_rolls_back_when_the_gateway_never_acks_the_hide():
    world = v1_world(auto_ack=False)

    with pytest.raises(GatewayAckTimeout):
        world.sleep(path="urgent")

    assert not any(call[0] == "sleep" for call in world.vllm.calls)
    assert world.runtime.patches[-1][:2] == ("pod-a", "awake")


def test_no_drain_without_sleep_mode_support_sends_a_plain_sleep_at_once():
    world = v1_world(vllm_sleep_mode_param="false")
    world.gateway.inflight("pod-a", "gw-1", total=1, non_continuable=1)

    [outcome] = world.sleep(path="urgent")

    assert sleep_modes(world) == [None]
    assert outcome["status"] == "slept" and outcome["forced_abort"] is True
    assert outcome["waited_s"] < 1.0


# ------------------------------------------------------------ SafeScale / APA
def test_safescale_commit_ignores_the_probe_window_budget_and_commits_at_once():
    world = v1_world()
    world.gateway.inflight("pod-a", "gw-1", total=2, non_continuable=1)

    [outcome] = world.sleep(path="safescale_commit", drain_budget_s=60.0)

    assert outcome["status"] == "slept" and sleep_modes(world) == ["abort"]
    assert outcome["waited_s"] < 1.0  # was: up to one more probe window
    assert outcome["aborted"]["non_continuable"] == 1
    assert world.stats()["sleeps_path_safescale_commit"] == 1


def test_apa_path_does_not_drain():
    world = v1_world()
    world.gateway.inflight("pod-a", "gw-1", total=1, non_continuable=1)

    [outcome] = world.sleep(path="apa")

    assert outcome["status"] == "slept" and outcome["drain_policy"] == "no_drain"
    assert outcome["waited_s"] < 1.0


# ---------------------------------------------------------- maintenance paths
@pytest.mark.parametrize("path", ["defrag", "repair", "startup", "scale_down", "default"])
def test_maintenance_paths_still_wait_for_non_continuable_and_roll_back(path):
    world = v1_world(hard_cap_s=60.0)
    world.gateway.inflight("pod-a", "gw-1", total=1, non_continuable=1)

    with pytest.raises(SleepIncomplete) as info:
        world.sleep(path=path)

    [outcome] = info.value.outcomes
    assert outcome["status"] == "rolled_back" and outcome["drain_policy"] == "drain"
    assert 60.0 <= outcome["waited_s"] < 61.0
    assert not any(call[0] == "sleep" for call in world.vllm.calls)
    assert world.stats()["non_continuable_rollback_total"] == 1


def test_maintenance_path_keeps_its_soft_budget_for_continuable_requests():
    world = v1_world()
    world.gateway.inflight("pod-a", "gw-1", total=2)

    [outcome] = world.sleep(path="defrag")

    assert outcome["forced_abort"] is True and outcome["drain_policy"] == "drain"
    assert 30.0 <= outcome["waited_s"] < 31.0
    assert "no_drain_sleeps_total" not in world.stats()


# ------------------------------------------------------ switch back (config)
def test_empty_no_drain_paths_restores_the_draining_urgent_path():
    world = World(no_drain_paths=(), budgets_s={"urgent": 30.0}, hard_cap_s=60.0)
    world.gateway.inflight("pod-a", "gw-1", total=1)

    [outcome] = world.sleep(path="urgent")

    assert outcome["drain_policy"] == "drain"
    assert 30.0 <= outcome["waited_s"] < 31.0


def test_empty_no_drain_paths_restores_rollback_over_non_continuable():
    world = World(no_drain_paths=(), hard_cap_s=60.0)
    world.gateway.inflight("pod-a", "gw-1", total=1, non_continuable=1)

    with pytest.raises(SleepIncomplete) as info:
        world.sleep(path="urgent")

    assert info.value.outcomes[0]["status"] == "rolled_back"


def test_empty_no_drain_paths_restores_the_probe_window_commit_drain():
    world = World(no_drain_paths=(), hard_cap_s=150.0)
    world.gateway.inflight("pod-a", "gw-1", total=1)

    [outcome] = world.sleep(path="safescale_commit", drain_budget_s=45.0)

    assert 45.0 <= outcome["waited_s"] < 46.0
