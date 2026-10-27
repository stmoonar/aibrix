"""Guard (round 4, docs/design/20261025-instant-sampling.md "Evidence, not ticks"): the
decision layer reads observation-time spans, never the evidence period or a count of
windows / ticks.

The evidence period is a storage detail (``store/``, the sampler in
``loops/metrics_task.py``, ``store/evidence_grid.py`` for the SafeScale Redis path).
``planning/safescale.py`` holds an ``EvidenceGrid`` object (``self._grid``) and asks it
for doc boundaries only - the first doc after the hide, the next doc after the newest
evidence (wait for a NEW snapshot) - never for the period as a span of evidence. A
rule that counts windows or ticks, or that reads the period, changes its meaning when
the period changes (10 s -> 2 s made "2 consecutive windows" mean 2 s instead of 10 s).
The whole file text is checked, comments included: a comment that explains a rule in
windows is how the next count-based rule gets written.
"""
from __future__ import annotations

import re
from pathlib import Path

PACKAGE = Path(__file__).resolve().parents[1] / "tre_controller"
DECISION_LAYER = (
    *sorted((PACKAGE / "signals").glob("*.py")),
    *sorted((PACKAGE / "planning").glob("*.py")),
    PACKAGE / "loops" / "tick.py",
    PACKAGE / "loops" / "rescue_task.py",
    PACKAGE / "loops" / "fairness_task.py",
    # Round 5 (L4): the SafeScale observation task (tail terms, observation fields).
    PACKAGE / "loops" / "safescale_task.py",
)
FORBIDDEN = re.compile(
    r"grid_ms|evidence_period|period_ms|EVIDENCE_PERIOD_MS|INSTANT_SAMPLE_INTERVAL_MS"
    r"|SCRAPE_INTERVAL_MS|streak|consecutive|n_windows|grids"
    # 2026-10-28: nor the configured window length - a rule reads its window's own
    # bounds (window_end_ms - window_start_ms), so 30 s -> 20 s needs no decision edit.
    r"|metrics_window|METRICS_WINDOW"
)


def test_the_decision_layer_never_reads_the_period_or_counts_windows():
    assert len(DECISION_LAYER) > 5 and all(path.is_file() for path in DECISION_LAYER)
    hits = [
        f"{path.relative_to(PACKAGE)}:{number}: {line.strip()}"
        for path in DECISION_LAYER
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1)
        if FORBIDDEN.search(line)
    ]
    assert hits == [], "\n".join(hits)
