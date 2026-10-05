"""Release guard (review 2026-10-06 P1-1): the shipped registry.yaml must stay
startable by the PREVIOUS service-manager / controller image (20261001, code
ba5b558f), the rollback target while the whole-lock images are switched in.

That image validates ``service_manager:`` at start with its own rules
(``_validate_service_manager`` / ``sleep_call_timeout_errors`` in
``tre/common/tre_common/registry.py`` at ba5b558f) and refuses to start on any
error. The checks below re-implement exactly the constraints of ba5b558f that
the whole-lock values can break (the new service-manager ignores the drain-era
keys, so nothing else keeps them valid); absent keys take ba5b558f's defaults.
Checked by hand on 2026-10-06: ba5b558f's loader + ``validate()`` on this
registry return no error (with ``reservation_ttl_s: 30`` they reported
"reservation_ttl_s (30) must exceed the longest renewal gap 37.5s")."""

from pathlib import Path

import yaml

REGISTRY = Path(__file__).resolve().parents[1] / "registry.yaml"

#: ba5b558f's ``SleepPolicy`` / ``ServiceManagerConfig`` defaults (absent keys).
OLD_SLEEP_DEFAULTS = {
    "ack_timeout_s": 10.0,
    "instance_staleness_s": 10.0,
    "poll_interval_s": 0.5,
    "sleep_call_timeout_s": 45.0,
    "probe_timeout_s": 5.0,
    "io_margin_s": 5.0,
    "physical_confirm_timeout_s": 15.0,
    "reservation_ttl_s": 30.0,
}
OLD_WRITER_LOCK_WAIT_S = 10.0
OLD_API_CALL_TIMEOUT_S = 360.0
OLD_ROUTE_TIMEOUT_S = 150.0
#: ba5b558f's SLEEP_PATHS: budgets_s / no_drain_paths naming another path is a
#: start error there.
OLD_SLEEP_PATHS = {"safescale_commit", "urgent", "scale_down", "apa", "defrag", "repair", "startup", "default"}


def _old_view():
    raw = yaml.safe_load(REGISTRY.read_text(encoding="utf-8"))
    section = raw.get("service_manager") or {}
    sleep_raw = section.get("sleep") or {}
    sleep = {key: float(sleep_raw.get(key, default)) for key, default in OLD_SLEEP_DEFAULTS.items()}
    route_timeout = float((raw.get("gateway") or {}).get("route_timeout_s", OLD_ROUTE_TIMEOUT_S))
    hard_cap = sleep_raw.get("hard_cap_s")
    sleep["hard_cap_s"] = route_timeout if hard_cap is None else float(hard_cap)
    writer = float(section.get("writer_lock_wait_s", OLD_WRITER_LOCK_WAIT_S))
    commit = section.get("commit_lock_wait_s")
    commit_wait = writer if commit is None else float(commit)
    api = float(section.get("api_call_timeout_s", OLD_API_CALL_TIMEOUT_S))
    return sleep, sleep_raw, writer, commit_wait, api, route_timeout


def test_the_previous_image_accepts_the_reservation_ttl():
    sleep, _raw, _writer, commit_wait, _api, _route = _old_view()
    for key, value in sleep.items():
        assert value > 0, key
    assert sleep["reservation_ttl_s"] > 2 * sleep["poll_interval_s"]
    # ba5b558f review 2 P2-1: the longest gap between two reservation renewals.
    renew_gap = commit_wait + sleep["poll_interval_s"] + sleep["probe_timeout_s"] + sleep["io_margin_s"]
    assert sleep["reservation_ttl_s"] > renew_gap, (sleep["reservation_ttl_s"], renew_gap)


def test_the_previous_image_accepts_the_worst_case_call():
    """ba5b558f's worst case of one sleeping call (drain + split commit) must stay
    below api_call_timeout_s, and its drain hard cap within the route timeout."""
    sleep, _raw, writer, commit_wait, api, route_timeout = _old_view()
    assert sleep["hard_cap_s"] <= route_timeout
    drain = sleep["ack_timeout_s"] + sleep["hard_cap_s"] + sleep["probe_timeout_s"]
    commit = (
        5 * sleep["probe_timeout_s"] + 2 * sleep["sleep_call_timeout_s"]
        + sleep["physical_confirm_timeout_s"] + sleep["poll_interval_s"] + 2 * sleep["probe_timeout_s"]
    )
    worst = writer + drain + commit_wait + commit + sleep["io_margin_s"]
    assert worst < api, (worst, api)


def test_the_previous_image_knows_every_sleep_path():
    _sleep, sleep_raw, *_rest = _old_view()
    assert set(sleep_raw.get("budgets_s") or {}) <= OLD_SLEEP_PATHS
    assert set(sleep_raw.get("no_drain_paths") or ()) <= OLD_SLEEP_PATHS
