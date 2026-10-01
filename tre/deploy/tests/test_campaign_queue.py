import gzip
import json
from pathlib import Path

import pytest

from scripts.campaign_queue import (
    MAINTENANCE_RUN_MODE,
    CampaignRunner,
    RunSpec,
    _parse_baseline_overrides,
    arm_config,
    baseline_errors,
    derive_actual_actions,
    deterministic_gzip,
    generate_baseline,
    load_manifest,
    parse_controller_decisions,
    pod_is_ready,
    redis_keys_to_clear,
    request_health,
    resolve_baseline,
    select_runs,
)
from tre_common.registry import ClusterTopology, ModelSpec, NodeSpec, Registry, SloSpec, TrsParams


def _registry() -> Registry:
    """Synthetic 2 nodes x 4 GPUs, models A/B (tp1) and C (tp2)."""
    trs = TrsParams(
        w_p=0.0, w_d=1.0, lambda_wait=3.0, qmin=1.0, ema_alpha=0.2, theta_m=100.0,
        tau_crit=0.5, tau_low=1.0, tau_high=2.0, qsat=4.0, epsat=0.05, hsat=3,
    )
    slo = SloSpec(ttft_p95_ms=500.0, tpot_p95_ms=75.0, e2e_p95_ms=12000.0)
    nodes = tuple(
        NodeSpec(name=name, gpus=4, two_gpu_slots=((0, 1), (2, 3)),
                 gpu_uuids=tuple(f"{name}-{gpu}" for gpu in range(4)))
        for name in ("n-a", "n-b")
    )
    return Registry(
        ClusterTopology(nodes=nodes),
        [
            ModelSpec(name=name, weights_path="/w", tp_size=tp, min_replicas=1,
                      max_replicas=hi, max_awake_replicas=4, vllm_image="image", slo=slo, trs=trs)
            for name, tp, hi in (("A", 1, 8), ("B", 1, 8), ("C", 2, 4))
        ],
    )


REGISTRY = _registry()
#: An explicit (manifest) baseline: live serve_ids.
BASELINE = {"A": "a-pod-0", "B": "b-pod-1", "C": "c-pod-4"}
_SLOTS = {"A": ("n-a", [0]), "B": ("n-a", [1]), "C": ("n-b", [0, 1])}


def _state(*, awake=None, hidden=()):
    awake = set(BASELINE.values()) if awake is None else set(awake)
    hidden = set(hidden)
    bindings = []
    for model, serve_id in BASELINE.items():
        node, gpu_ids = _SLOTS[model]
        bindings.append(
            {
                "binding_id": f"{model}/{node}/{','.join(str(g) for g in gpu_ids)}",
                "serve_id": serve_id,
                "model": model,
                "node": node,
                "gpu_ids": gpu_ids,
                "awake": serve_id in awake,
                "hidden": serve_id in hidden,
            }
        )
    return {"version": 1, "bindings": bindings}


def _write_manifest(path: Path, **overrides):
    value = {
        "frozen_sha": "a" * 40,
        "params_hash": "params",
        "images": {
            "controller": "controller:tag",
            "service-manager": "sm:tag",
            "ui": "ui:tag",
        },
        "baseline": BASELINE,
        "cooldown_s": 600,
        "post_drain_s": 30,
        "runs": [
            {"id": "t1_tre_seed1", "trace": "trace.json", "arm": "tre", "seed": 1},
            {"id": "t1_apa_seed1", "trace": "trace.json", "arm": "apa", "seed": 1},
        ],
    }
    value.update(overrides)
    path.write_text(json.dumps(value), encoding="utf-8")


def test_manifest_pins_freeze_baseline_and_unique_runs(tmp_path):
    path = tmp_path / "manifest.json"
    _write_manifest(path)

    manifest = load_manifest(path, registry=REGISTRY)

    assert manifest.frozen_sha == "a" * 40
    assert manifest.cooldown_s == 600
    assert manifest.baseline == BASELINE
    assert manifest.baseline_source == "manifest"
    assert [run.run_id for run in manifest.runs] == ["t1_tre_seed1", "t1_apa_seed1"]

    _write_manifest(
        path,
        runs=[
            {"id": "duplicate", "trace": "a", "arm": "tre", "seed": 1},
            {"id": "duplicate", "trace": "b", "arm": "apa", "seed": 1},
        ],
    )
    with pytest.raises(ValueError, match="duplicate run IDs"):
        load_manifest(path, registry=REGISTRY)


def test_generated_baseline_places_one_replica_per_model_by_the_placement_policy():
    # Registry model order A, B, C on an empty cluster: A and B share one pair of the
    # first node, C (tp2) takes a pair on the other node -- not all on one node.
    assert generate_baseline(REGISTRY) == {"A": "A/n-a/0", "B": "B/n-a/1", "C": "C/n-b/0,1"}


def test_manifest_without_baseline_uses_the_generated_one_and_cli_overrides(tmp_path):
    path = tmp_path / "manifest.json"
    _write_manifest(path, baseline=None)

    manifest = load_manifest(path, registry=REGISTRY)
    assert manifest.baseline == generate_baseline(REGISTRY)
    assert manifest.baseline_source == "generated"

    manifest = load_manifest(
        path, registry=REGISTRY, baseline_override=_parse_baseline_overrides(["A=A/n-b/3"])
    )
    assert manifest.baseline["A"] == "A/n-b/3"
    assert manifest.baseline_source == "cli"

    _write_manifest(path, baseline={"A": "a", "B": "b"})
    with pytest.raises(ValueError, match="exactly the registered models"):
        load_manifest(path, registry=REGISTRY)
    with pytest.raises(ValueError, match="MODEL="):
        _parse_baseline_overrides(["A"])


def test_binding_id_baseline_resolves_to_the_live_serve_ids():
    by_binding = {model: f"{model}/{node}/{','.join(map(str, gpus))}" for model, (node, gpus) in _SLOTS.items()}
    assert resolve_baseline(_state(), by_binding) == BASELINE
    assert baseline_errors(_state(), by_binding) == []
    assert "missing baseline binding A/n-b/3" in ";".join(
        baseline_errors(_state(), {**by_binding, "A": "A/n-b/3"})
    )


def test_arm_configs_share_the_tre_gateway_and_keep_apa_counterfactual_logging():
    tre = arm_config("tre")
    apa = arm_config("apa")
    queue = arm_config("queue_len")

    assert tre.gateway.endswith(":31094/v1/completions")
    assert tre.mode == "active" and not tre.disable_eta_gate
    # Same gateway for both arms (v1 parity): same routes, timeout, admission limits and
    # least-gpu-cache pod choice; only the scaler differs.
    assert apa.gateway == tre.gateway
    assert apa.mode == "observe" and apa.apa_enabled
    assert apa.signal_source == "zm"
    assert queue.gateway == tre.gateway
    assert queue.signal_source == "queue_len" and queue.disable_eta_gate


def test_every_experiment_arm_runs_the_sm_actuation_active():
    # 2026-09-28: controller mode and SM actuation are independent; both arms
    # get the same SM self-heal, only the controller mode differs.
    assert (arm_config("tre").controller_mode, arm_config("tre").sm_actuation) == ("active", "active")
    assert (arm_config("apa").controller_mode, arm_config("apa").sm_actuation) == ("observe", "active")
    assert arm_config("queue_len").sm_actuation == "active"
    assert MAINTENANCE_RUN_MODE == ("observe", "observe")


class _ModeRedis:
    def __init__(self, drop=None):
        self.kv = {}
        self.transactions = []
        self.drop = drop

    def get(self, key):
        value = self.kv.get(key)
        return value.encode() if isinstance(value, str) else value

    def pipeline(self, transaction=True):
        redis = self

        class _Pipe:
            def __init__(self):
                self.commands = []

            def set(self, key, value):
                self.commands.append((key, value))

            def execute(self):
                redis.transactions.append((transaction, list(self.commands)))
                for key, value in self.commands:
                    if key != redis.drop:
                        redis.kv[key] = value

        return _Pipe()


def _runner(redis):
    runner = object.__new__(CampaignRunner)
    runner.redis = redis
    return runner


def test_set_mode_writes_both_switches_in_one_multi_and_reads_them_back():
    redis = _ModeRedis()
    runner = _runner(redis)
    assert runner.set_mode("observe", "active") == {"controller_mode": "observe", "sm_actuation": "active"}
    assert redis.transactions == [
        (True, [("tre:v2:controller:mode", "observe"), ("tre:v2:sm:actuation", "active")])
    ]
    assert runner.read_run_mode() == {"controller_mode": "observe", "sm_actuation": "active"}
    with pytest.raises(ValueError):
        runner.set_mode("active", "bogus")


def test_set_mode_fails_when_the_read_back_differs():
    runner = _runner(_ModeRedis(drop="tre:v2:sm:actuation"))
    with pytest.raises(RuntimeError, match="read back"):
        runner.set_mode("active", "active")


def test_toggle_leaves_the_run_mode_to_the_queue(monkeypatch):
    runner = _runner(_ModeRedis())
    calls = []
    monkeypatch.setattr(runner, "_command", lambda command, **kw: calls.append(command), raising=False)
    runner.toggle("apa")
    assert calls == [["bash", "deploy/scripts/toggle_tre_apa.sh", "apa", "--keep-run-mode"]]


def test_baseline_gate_rejects_extra_awake_and_hidden_bindings():
    assert baseline_errors(_state(), BASELINE) == []
    extra = _state()
    extra["bindings"].append(
        {
            "serve_id": "extra",
            "model": "A",
            "node": "n-b",
            "gpu_ids": [0],
            "awake": True,
            "hidden": False,
        }
    )
    assert "unexpected awake bindings" in ";".join(
        baseline_errors(extra, BASELINE)
    )
    target = next(iter(BASELINE.values()))
    assert "not awake+routable" in ";".join(
        baseline_errors(_state(hidden={target}), BASELINE)
    )


def test_redis_clear_selection_never_selects_service_manager_truth():
    selected = redis_keys_to_clear(
        {
            "tre:v2:sm:state",
            "tre:v2:sm:version",
            "tre:v2:decision:hist:dsqwen-7b",
            "tre:v2:controller:safescale:probe:req-1:journal",
            "tre:v2:hist:pod-a",
        }
    )

    assert "tre:v2:sm:state" not in selected
    assert "tre:v2:sm:version" not in selected
    assert "tre:v2:hist:pod-a" not in selected
    assert "tre:v2:decision:hist:dsqwen-7b" in selected
    assert "tre:v2:controller:signal_log" in selected
    assert "tre:v2:controller:safescale:probe:req-1:journal" in selected


def test_pod_ready_requires_running_ready_and_not_terminating():
    pod = {
        "metadata": {"name": "pod"},
        "status": {"phase": "Running", "conditions": [{"type": "Ready", "status": "True"}]},
    }
    assert pod_is_ready(pod)
    pod["metadata"]["deletionTimestamp"] = "now"
    assert not pod_is_ready(pod)


def test_layout_transition_derivation_tracks_actual_wakes_and_sleeps():
    rows = [
        {"ts": "1", "model": "m", "awake_serve_ids": "a", "hidden_serve_ids": ""},
        {"ts": "2", "model": "m", "awake_serve_ids": "a;b", "hidden_serve_ids": "a"},
        {"ts": "3", "model": "m", "awake_serve_ids": "b", "hidden_serve_ids": ""},
    ]

    assert derive_actual_actions(rows) == [
        {"ts": "2", "model": "m", "action": "wake", "serve_id": "b"},
        {"ts": "2", "model": "m", "action": "hide", "serve_id": "a"},
        {"ts": "3", "model": "m", "action": "sleep", "serve_id": "a"},
        {"ts": "3", "model": "m", "action": "unhide", "serve_id": "a"},
    ]


def test_controller_log_parser_decodes_nested_decision_actions():
    payload = {
        "event": "trs_calc_result",
        "ts_ms": "150",
        "loop": "rescue",
        "actions": json.dumps([{"kind": "scale", "model": "m", "delta": 1}]),
        "events": json.dumps(["event-a"]),
        "model_states": json.dumps({"m": {"z_m": 0.5}}),
    }
    line = json.dumps({"message": json.dumps(payload)})

    decisions, actions = parse_controller_decisions(
        "not-json\n" + line, start_ms=100, end_ms=200
    )

    assert decisions[0]["events"] == ["event-a"]
    assert decisions[0]["model_states"]["m"]["z_m"] == 0.5
    assert actions == [
        {"ts_ms": 150, "loop": "rescue", "kind": "scale", "model": "m", "delta": 1}
    ]


def test_request_health_and_deterministic_gzip(tmp_path):
    source = tmp_path / "requests.jsonl"
    source.write_text(
        "\n".join(
            json.dumps({"http_status": status}) for status in (200, 503, 0, 200)
        ) + "\n",
        encoding="utf-8",
    )

    assert request_health(source) == {
        "requests": 4,
        "http_5xx": 1,
        "http_5xx_frac": 0.25,
        "status_zero": 1,
        "status_zero_frac": 0.25,
    }
    first = tmp_path / "first.gz"
    second = tmp_path / "second.gz"
    deterministic_gzip(source, first)
    deterministic_gzip(source, second)
    assert first.read_bytes() == second.read_bytes()
    with gzip.open(first, "rt", encoding="utf-8") as stream:
        assert stream.read() == source.read_text(encoding="utf-8")


def test_run_selection_supports_resume_boundary_and_limit():
    runs = tuple(RunSpec(f"run-{index}", "trace", "tre", index) for index in range(4))

    assert [run.run_id for run in select_runs(runs, start_at="run-2", limit=1)] == ["run-2"]
    with pytest.raises(ValueError, match="unknown --start-at"):
        select_runs(runs, start_at="missing", limit=None)
    with pytest.raises(ValueError, match="positive"):
        select_runs(runs, start_at=None, limit=0)

def test_manifest_pins_the_prompt_corpus_of_the_replays(tmp_path):
    path = tmp_path / "manifest.json"
    _write_manifest(path)
    manifest = load_manifest(path, registry=REGISTRY)
    assert (manifest.corpus_lang, manifest.zh_ratio) == ("mix", 0.5)
    _write_manifest(path, corpus_lang="en")
    assert (load_manifest(path, registry=REGISTRY).corpus_lang,
            load_manifest(path, registry=REGISTRY).zh_ratio) == ("en", 0.0)
    _write_manifest(path, corpus_lang="fr")
    with pytest.raises(ValueError, match="corpus_lang"):
        load_manifest(path, registry=REGISTRY)
    _write_manifest(path, corpus_lang="mix", zh_ratio=1.5)
    with pytest.raises(ValueError, match="zh_ratio"):
        load_manifest(path, registry=REGISTRY)


def test_the_arms_send_v1s_request_unless_the_manifest_says_replay(tmp_path):
    """E1 = v1's request (chat, stream, no ignore_eos, max_tokens from the trace): every
    arm sends the e1_v1 profile unless the manifest names the old replay request."""
    import inspect

    from scripts import campaign_queue

    path = tmp_path / "manifest.json"
    _write_manifest(path)
    assert load_manifest(path, registry=REGISTRY).client_profile == "e1_v1"
    _write_manifest(path, client_profile="replay")
    assert load_manifest(path, registry=REGISTRY).client_profile == "replay"
    _write_manifest(path, client_profile="calib")
    with pytest.raises(ValueError, match="client_profile"):
        load_manifest(path, registry=REGISTRY)
    # the run command and its command.json carry the profile
    source = inspect.getsource(campaign_queue.CampaignRunner.run_one)
    assert '"--client-profile", self.manifest.client_profile' in source
    assert '"client_profile": self.manifest.client_profile' in source
    from tre_replayer import run_trace

    with pytest.raises(SystemExit):  # the flag exists and refuses unknown profiles
        run_trace.main(["--trace", "t.json", "--client-profile", "calib"])
