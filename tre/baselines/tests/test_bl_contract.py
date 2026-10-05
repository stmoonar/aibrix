from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest

from tre_baselines.config import load_config
from tre_baselines.policies import POLICIES, Policy, build_policy
from tre_baselines.policies.base import Decision
from tre_baselines.snapshot import COUNTER_KEYS, ClusterSnapshot, ModelSnapshot, PodSnapshot

TRE_ROOT = Path(__file__).resolve().parents[2]
REGISTRY = str(TRE_ROOT / "deploy" / "registry.yaml")
ENV = {
    "TRE_SM_URL": "http://sm.test:8000/",
    "TRE_REDIS_URL": "redis://redis.test:6379/0",
    "TRE_BL_POLICY": "static",
    "TRE_REGISTRY_PATH": REGISTRY,
}


def _model(name: str, awake: int) -> ModelSnapshot:
    pod = PodSnapshot(
        pod=f"{name}-p0", model=name, node="n0", gpu_ids=(0,), running=1.0, waiting=0.0,
        kv_usage=0.1, counters={"gen_tokens": 10.0}, num_gpu_blocks=100, block_size=16,
        scraped_at_ms=1000,
    )
    return ModelSnapshot(
        model=name, awake=awake, min_replicas=1, max_replicas=4, gpus_per_replica=1,
        ttft_slo_ms=500.0, tpot_slo_ms=75.0, max_num_seqs=256, pods=(pod,), events=(),
    )


def test_snapshot_types_are_frozen() -> None:
    snap = ClusterSnapshot(now_ms=1, tick_s=2.0, models={"m": _model("m", 2)})
    with pytest.raises(dataclasses.FrozenInstanceError):
        snap.now_ms = 2  # type: ignore[misc]
    with pytest.raises(dataclasses.FrozenInstanceError):
        snap.models["m"].awake = 3  # type: ignore[misc]
    assert "gen_tokens" in COUNTER_KEYS and len(set(COUNTER_KEYS)) == len(COUNTER_KEYS)


def test_static_policy_holds_awake_and_satisfies_protocol() -> None:
    config = load_config(ENV)
    policy = build_policy("static", config)
    assert isinstance(policy, Policy)
    snap = ClusterSnapshot(now_ms=1, tick_s=2.0, models={"a": _model("a", 2), "b": _model("b", 1)})
    out = policy.decide(snap)
    assert out == {
        "a": Decision(desired=2, reason="static_hold", inputs={"awake": 2}),
        "b": Decision(desired=1, reason="static_hold", inputs={"awake": 1}),
    }
    assert "static" in POLICIES
    with pytest.raises(KeyError):
        build_policy("nope", config)


def test_config_reads_registry_limits_through_common_readers() -> None:
    config = load_config(ENV)
    assert config.dry_run is True  # default
    assert config.tick_s == 2.0
    assert config.sm_url == "http://sm.test:8000"
    assert config.abort_sleep_path == "urgent"  # a no-drain path: hide, ack, abort + continue
    assert config.scrape_timeout_s == 2.5 and config.decision_stream is True
    assert config.backoff_max_s == 10.0 and config.liveness_stall_s == 120.0
    assert load_config({**ENV, "TRE_BL_DECISION_STREAM": "false"}).decision_stream is False
    from tre_common.registry import load_registry

    registry = load_registry(REGISTRY)
    assert set(config.models) == {m.name for m in registry.models()}
    for spec in registry.models():
        lim = config.models[spec.name]
        assert lim.min_replicas == spec.min_replicas
        assert lim.max_replicas == spec.scale_max_replicas
        assert lim.gpus_per_replica == spec.tp_size
        assert lim.ttft_slo_ms == spec.slo.ttft_p95_ms
        assert lim.tpot_slo_ms == spec.slo.tpot_p95_ms
        assert lim.slo is not None


def test_config_rejects_bad_values(tmp_path) -> None:
    with pytest.raises(ValueError):
        load_config({**ENV, "TRE_BL_ABORT_SLEEP_PATH": "repair"})
    with pytest.raises(ValueError, match="drains"):  # scale_down drains on the SM on main
        load_config({**ENV, "TRE_BL_ABORT_SLEEP_PATH": "scale_down"})
    for retired in ("TRE_BL_SLEEP_PATH", "TRE_BL_DRAIN_BUDGET_S"):
        with pytest.raises(ValueError, match="retired"):
            load_config({**ENV, retired: "30"})
    with pytest.raises(ValueError):
        load_config({k: v for k, v in ENV.items() if k != "TRE_SM_URL"})
    with pytest.raises(ValueError):
        load_config({**ENV, "TRE_BL_DRY_RUN": "maybe"})
    with pytest.raises(ValueError):
        load_config({**ENV, "TRE_BL_MODELS": "no-such-model"})
    params = tmp_path / "p.yaml"
    params.write_text("theta: 0.8\nwindow_s: 30\n", encoding="utf-8")
    cfg = load_config({**ENV, "TRE_BL_POLICY_CONFIG": str(params), "TRE_BL_DRY_RUN": "false"})
    assert cfg.policy_params == {"theta": 0.8, "window_s": 30}
    assert cfg.dry_run is False
    assert load_config({**ENV, "TRE_BL_POLICY_CONFIG": str(tmp_path / "missing.yaml")}).policy_params == {}


def test_controller_mode_key_matches_tre_common() -> None:
    from tre_common.rediskeys import CONTROLLER_MODE_KEY as COMMON_KEY
    from tre_baselines.keys import CONTROLLER_MODE_KEY

    assert CONTROLLER_MODE_KEY == COMMON_KEY


def test_missing_idle_ttft_fit_refuses_an_actuating_shell(tmp_path) -> None:
    """c/b are live values (the TRE arm's TTFT SLO): no silent 500/75 fallback when acting."""
    text = Path(REGISTRY).read_text(encoding="utf-8")
    stripped = "\n".join(ln for ln in text.splitlines()
                          if "ttft_idle_c_ms:" not in ln and "ttft_idle_b_ms_per_token:" not in ln)
    reg = tmp_path / "registry.yaml"
    reg.write_text(stripped + "\n", encoding="utf-8")
    env = {**ENV, "TRE_REGISTRY_PATH": str(reg)}
    with pytest.raises(ValueError, match="ttft_idle_c_ms"):
        load_config({**env, "TRE_BL_DRY_RUN": "false"})
    assert load_config(env).dry_run is True  # dry-run: warns, falls back to the fixed arm
