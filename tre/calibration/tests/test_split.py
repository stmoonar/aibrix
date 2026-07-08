from __future__ import annotations

from tre_calibration.dataset import (
    CalibrationWindow,
    select_test_scenarios,
    split_by_scenario,
)


def _windows() -> list[CalibrationWindow]:
    rows: list[CalibrationWindow] = []
    # 10 scenarios across 2 families, several windows each.
    for idx in range(5):
        family = "steady"
        rows.append(CalibrationWindow(f"steady-{idx}", family, 100.0 + idx, idx % 2 == 0, 0.5))
        rows.append(CalibrationWindow(f"steady-{idx}", family, 101.0 + idx, idx % 2 == 0, 0.6))
    for idx in range(5):
        family = "burst"
        rows.append(CalibrationWindow(f"burst-{idx}", family, 90.0 + idx, idx % 2 == 1, 0.4))
        rows.append(CalibrationWindow(f"burst-{idx}", family, 91.0 + idx, idx % 2 == 1, 0.5))
    return rows


def test_select_test_scenarios_is_deterministic() -> None:
    windows = _windows()
    first = select_test_scenarios(windows, test_fraction=0.2, seed="tre-v2-ranking")
    second = select_test_scenarios(list(reversed(windows)), test_fraction=0.2, seed="tre-v2-ranking")
    # Same CSV content in any row order -> identical split (hash of scenario id only).
    assert first == second
    # 10 scenarios * 0.2 = 2 held out, clamped into [1, n-1].
    assert len(first) == 2


def test_select_test_scenarios_never_leaks_a_scenario() -> None:
    windows = _windows()
    test_scenarios = select_test_scenarios(windows, test_fraction=0.3)
    train, test = split_by_scenario(windows, test_scenarios=test_scenarios)

    train_ids = {window.scenario_id for window in train}
    test_ids = {window.scenario_id for window in test}
    assert train_ids and test_ids
    assert not (train_ids & test_ids)
    # Every window of a held-out scenario went to test (no partial cell leak).
    assert test_ids == set(test_scenarios)
    assert train_ids | test_ids == {window.scenario_id for window in windows}


def test_select_test_scenarios_seed_changes_split() -> None:
    windows = _windows()
    a = select_test_scenarios(windows, test_fraction=0.4, seed="seed-a")
    b = select_test_scenarios(windows, test_fraction=0.4, seed="seed-b")
    # Different seeds generally produce different holdouts (same size).
    assert len(a) == len(b) == 4
    assert a != b


def test_select_test_scenarios_single_scenario_returns_empty() -> None:
    windows = [
        CalibrationWindow("only", "steady", 100.0, True, 0.9),
        CalibrationWindow("only", "steady", 101.0, True, 0.9),
    ]
    assert select_test_scenarios(windows, test_fraction=0.2) == set()


def test_select_test_scenarios_clamps_fraction_to_keep_both_sets() -> None:
    windows = _windows()
    # A tiny fraction still yields at least one test scenario...
    tiny = select_test_scenarios(windows, test_fraction=0.01)
    assert len(tiny) == 1
    # ...and a huge fraction still leaves at least one train scenario.
    huge = select_test_scenarios(windows, test_fraction=0.99)
    assert len(huge) == 9
