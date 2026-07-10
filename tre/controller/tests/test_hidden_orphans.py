from __future__ import annotations

import json
import logging
from types import SimpleNamespace

from tre_common import rediskeys
from tre_controller.app import build_controller_task_specs
from tre_controller.config import ControllerConfig
from tre_controller.reconcile.hidden_orphans import HiddenOrphanDetector


class FakeRedis:
    def __init__(self):
        self.hashes = {}

    def hgetall(self, key):
        return dict(self.hashes.get(key, {}))

    def hset(self, name, key=None, value=None, mapping=None):
        bucket = self.hashes.setdefault(name, {})
        values = mapping if mapping is not None else {key: value}
        for field, payload in values.items():
            bucket[str(field).encode("utf-8")] = str(payload).encode("utf-8")
        return len(values)

    def hdel(self, name, *keys):
        bucket = self.hashes.setdefault(name, {})
        removed = 0
        for key in keys:
            removed += int(bucket.pop(str(key).encode("utf-8"), None) is not None)
        return removed


def _put_json(redis, key, field, payload):
    redis.hset(key, mapping={field: json.dumps(payload, sort_keys=True)})


def _state(redis, *, hidden=True):
    _put_json(
        redis,
        rediskeys.SM_STATE_KEY,
        "serve-a",
        {"model": "model-a", "awake": False, "hidden": hidden},
    )


def test_hidden_binding_with_live_probe_does_not_alert():
    redis = FakeRedis()
    _state(redis)
    _put_json(
        redis,
        rediskeys.CONTROLLER_SAFESCALE_PROBES_KEY,
        "probe-a",
        {"model": "model-a", "pods": ["serve-a"], "status": "probing"},
    )
    detector = HiddenOrphanDetector(redis, grace_s=600)

    assert detector.scan(now=1_000) == ()
    assert redis.hgetall(rediskeys.CONTROLLER_ORPHAN_WATCH_KEY) == {}
    assert redis.hgetall(rediskeys.CONTROLLER_HIDDEN_ORPHAN_ALERTS_KEY) == {}


def test_hidden_binding_within_grace_creates_watch_without_alert():
    redis = FakeRedis()
    _state(redis)
    detector = HiddenOrphanDetector(redis, grace_s=600)

    assert detector.scan(now=1_000) == ()

    assert redis.hgetall(rediskeys.CONTROLLER_ORPHAN_WATCH_KEY) == {
        b"serve-a": b"1000.0"
    }
    assert redis.hgetall(rediskeys.CONTROLLER_HIDDEN_ORPHAN_ALERTS_KEY) == {}


def test_hidden_binding_beyond_grace_logs_and_persists_alert(caplog):
    redis = FakeRedis()
    _state(redis)
    detector = HiddenOrphanDetector(redis, grace_s=600)
    detector.scan(now=1_000)

    with caplog.at_level(logging.ERROR):
        assert detector.scan(now=1_601) == ("serve-a",)

    alert = json.loads(
        redis.hgetall(rediskeys.CONTROLLER_HIDDEN_ORPHAN_ALERTS_KEY)[b"serve-a"]
    )
    assert alert == {
        "model": "model-a",
        "since_ts": 1_000.0,
        "detected_ts": 1_601.0,
        "mode": "alert_only",
    }
    assert (
        "TRE_ORPHAN_HIDDEN model=model-a serve_id=serve-a hidden_for_s=601 "
        "probe=absent action=alert_only"
    ) in caplog.text


def test_unhidden_binding_clears_watch_and_alert():
    redis = FakeRedis()
    _state(redis)
    detector = HiddenOrphanDetector(redis, grace_s=10)
    detector.scan(now=100)
    detector.scan(now=111)
    _state(redis, hidden=False)

    assert detector.scan(now=112) == ()
    assert redis.hgetall(rediskeys.CONTROLLER_ORPHAN_WATCH_KEY) == {}
    assert redis.hgetall(rediskeys.CONTROLLER_HIDDEN_ORPHAN_ALERTS_KEY) == {}


def test_resolved_probe_starts_watch_and_alerts_after_grace():
    redis = FakeRedis()
    _state(redis)
    probe = {"model": "model-a", "pods": ["serve-a"], "status": "probing"}
    _put_json(redis, rediskeys.CONTROLLER_SAFESCALE_PROBES_KEY, "probe-a", probe)
    detector = HiddenOrphanDetector(redis, grace_s=600)
    detector.scan(now=100)

    probe["status"] = "resolved"
    _put_json(redis, rediskeys.CONTROLLER_SAFESCALE_PROBES_KEY, "probe-a", probe)
    assert detector.scan(now=200) == ()
    assert detector.scan(now=801) == ("serve-a",)


def test_config_exposes_orphan_scan_knobs():
    defaults = ControllerConfig.from_env({})
    assert defaults.orphan_scan_enabled is True
    assert defaults.orphan_grace_s == 600.0

    configured = ControllerConfig.from_env(
        {"TRE_ORPHAN_SCAN_ENABLED": "false", "TRE_ORPHAN_GRACE_S": "42"}
    )
    assert configured.orphan_scan_enabled is False
    assert configured.orphan_grace_s == 42.0


def test_orphan_task_runs_even_when_scaling_is_disabled():
    class FakeDetector:
        async def run(self):
            return None

    deps = SimpleNamespace(
        store=object(),
        snapshot_box=object(),
        profiler=None,
        hidden_orphan_detector=FakeDetector(),
    )
    cfg = SimpleNamespace(
        enable_tre_scaling=False,
        orphan_scan_enabled=True,
    )

    specs = build_controller_task_specs(deps, cfg)

    assert tuple(spec.name for spec in specs) == ("metrics", "hidden_orphans")
