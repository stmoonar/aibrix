import gzip
import json
from pathlib import Path

import pytest

from scripts.campaign_queue import (
    DEFAULT_BASELINE,
    RunSpec,
    arm_config,
    baseline_errors,
    derive_actual_actions,
    deterministic_gzip,
    load_manifest,
    parse_controller_decisions,
    pod_is_ready,
    redis_keys_to_clear,
    request_health,
    select_runs,
)


def _state(*, awake=None, hidden=()):
    awake = set(DEFAULT_BASELINE.values()) if awake is None else set(awake)
    hidden = set(hidden)
    bindings = []
    for index, (model, serve_id) in enumerate(DEFAULT_BASELINE.items()):
        bindings.append(
            {
                "serve_id": serve_id,
                "model": model,
                "node": "node9",
                "gpu_ids": [index],
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
        "baseline": DEFAULT_BASELINE,
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

    manifest = load_manifest(path)

    assert manifest.frozen_sha == "a" * 40
    assert manifest.cooldown_s == 600
    assert manifest.baseline == DEFAULT_BASELINE
    assert [run.run_id for run in manifest.runs] == ["t1_tre_seed1", "t1_apa_seed1"]

    _write_manifest(
        path,
        runs=[
            {"id": "duplicate", "trace": "a", "arm": "tre", "seed": 1},
            {"id": "duplicate", "trace": "b", "arm": "apa", "seed": 1},
        ],
    )
    with pytest.raises(ValueError, match="duplicate run IDs"):
        load_manifest(path)


def test_arm_configs_separate_gateways_and_keep_apa_counterfactual_logging():
    tre = arm_config("tre")
    apa = arm_config("apa")
    queue = arm_config("queue_len")

    assert tre.gateway.endswith(":31094/v1/completions")
    assert tre.mode == "active" and not tre.disable_eta_gate
    assert apa.gateway.endswith(":31592/v1/completions")
    assert apa.mode == "observe" and apa.apa_enabled
    assert apa.signal_source == "zm"
    assert queue.gateway == tre.gateway
    assert queue.signal_source == "queue_len" and queue.disable_eta_gate


def test_baseline_gate_rejects_extra_awake_and_hidden_bindings():
    assert baseline_errors(_state(), DEFAULT_BASELINE) == []
    extra = _state()
    extra["bindings"].append(
        {
            "serve_id": "extra",
            "model": "dsqwen-7b",
            "node": "node10",
            "gpu_ids": [0],
            "awake": True,
            "hidden": False,
        }
    )
    assert "unexpected awake bindings" in ";".join(
        baseline_errors(extra, DEFAULT_BASELINE)
    )
    target = next(iter(DEFAULT_BASELINE.values()))
    assert "not awake+routable" in ";".join(
        baseline_errors(_state(hidden={target}), DEFAULT_BASELINE)
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