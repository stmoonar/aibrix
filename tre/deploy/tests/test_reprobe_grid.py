"""The S3 supplement grid rule brackets the flip the 2026-09-23 supplement measured."""
from __future__ import annotations

import pytest

from scripts.analysis.reprobe_grid import grid_for


@pytest.mark.parametrize("row, measured_flip", [
    ({"model": "dsqwen-7b", "shape": "S3", "rf_max": "1.3", "P_b50_rf": "1.395"}, 1.356),
    ({"model": "dsqwen-14b", "shape": "S3", "rf_max": "1.3", "P_b50_rf": "1.587"}, 1.987),
    ({"model": "dsllama-8b", "shape": "S3", "rf_max": "1.3", "P_b50_rf": ""}, 1.575),
])
def test_grid_brackets_the_measured_flip(row, measured_flip):
    grid = grid_for(row)["grid"]
    assert len(grid) == 4 and grid == sorted(grid)
    assert grid[0] < measured_flip < grid[-1]


def test_no_basis_refuses():
    with pytest.raises(ValueError):
        grid_for({"model": "m", "shape": "S3", "rf_max": "", "P_b50_rf": ""})
