"""Guard (2026-10-07): the controller's scaling env switch is gone.

Actuation follows the run mode (``tre:v2:controller:mode``) only; the old switch stopped
the whole decision pipeline (signals, decision snapshots, signal log), so the APA and
baseline arms lost their TSS record. No deploy manifest, deploy script or experiment
runner may set or read it again (the controller ignores it with one warning, see
``controller/tests/test_config.py``)."""
from __future__ import annotations

from pathlib import Path

TRE = Path(__file__).resolve().parents[2]
REMOVED = "ENABLE_TRE_SCALING"
SCANNED = (
    TRE / "deploy" / "overlays",
    TRE / "deploy" / "models",
    TRE / "deploy" / "baselines",
    TRE / "deploy" / "scripts",
    TRE / "eval" / "runner",
)
SUFFIXES = {".yaml", ".yml", ".sh", ".py", ".env", ".example", ".sample", ""}


def test_no_manifest_script_or_runner_mentions_the_removed_switch() -> None:
    hits = []
    for root in SCANNED:
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if not path.is_file() or path.suffix not in SUFFIXES or "__pycache__" in path.parts:
                continue
            if REMOVED in path.read_text(encoding="utf-8", errors="replace"):
                hits.append(str(path.relative_to(TRE)))
    assert hits == []
