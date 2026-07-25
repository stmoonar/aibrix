"""Run the chart module's pure scale maths under node, when node is available.

The console has no JS test stack, and adding one is out of scope. But the chart
scale helpers are deliberately DOM-free, so a plain `node` run covers them --
and syntax-checks every module, which is otherwise only discovered in a browser.
Skips cleanly where node is absent so `make check` stays portable.
"""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

_STATIC_JS = Path(__file__).resolve().parents[1] / "tre_ui" / "static" / "js"
_HARNESS = Path(__file__).resolve().parent / "chart_check.mjs"

pytestmark = pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")


def test_every_console_module_parses(tmp_path) -> None:
    failures = []
    for module in sorted(_STATIC_JS.glob("*.js")):
        # node --check treats .js as CommonJS, so ES modules need the .mjs suffix.
        staged = tmp_path / (module.stem + ".mjs")
        staged.write_text(module.read_text(encoding="utf-8"), encoding="utf-8")
        result = subprocess.run(
            ["node", "--check", str(staged)], capture_output=True, text=True
        )
        if result.returncode != 0:
            failures.append(f"{module.name}: {result.stderr.strip()}")
    assert not failures, "\n".join(failures)


def test_chart_scale_maths_holds() -> None:
    result = subprocess.run(
        ["node", str(_HARNESS), str(_STATIC_JS / "chart.js")],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "chart maths OK" in result.stdout
