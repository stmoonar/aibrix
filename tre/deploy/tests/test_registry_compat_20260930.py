"""Release guard (review P2-6, 2026-09-30): the shipped registry.yaml must stay
readable by the components released before the placement / parallel-wake
branch (images 20260930-f8ccb0ca): their ``parse_placement_config`` refuses any
placement key but ``reserve_tp_pairs`` and ``defrag`` (a UI / controller / SM
still on the old image would not start). The new keys stay commented out until
all three images run this release (deploy/RELEASE-20260930-placement-parallel-wake.md)."""

from pathlib import Path

import yaml

#: What ``parse_placement_config`` accepted before 2026-09-30 (main 19781f50).
LEGACY_PLACEMENT_KEYS = {"reserve_tp_pairs", "defrag"}
LEGACY_DEFRAG_KEYS = {"enabled"}

REGISTRY = Path(__file__).resolve().parents[1] / "registry.yaml"


def test_registry_yaml_placement_is_readable_by_the_previous_release():
    raw = yaml.safe_load(REGISTRY.read_text(encoding="utf-8"))
    placement = raw.get("placement") or {}
    assert set(placement) <= LEGACY_PLACEMENT_KEYS, sorted(set(placement) - LEGACY_PLACEMENT_KEYS)
    assert set(placement.get("defrag") or {}) <= LEGACY_DEFRAG_KEYS


def test_new_keys_are_documented_but_commented_out():
    text = REGISTRY.read_text(encoding="utf-8")
    for key in ("# placement_penalty:", "# wake_cooldown:", "# test_hooks: false", "#   max_records: 20000"):
        assert key in text, key
