"""E1 runner sampler (tre/eval/runner/sampler.py): the collector behaviours the report needs.

* hidden pods (SafeScale / SM sleep hides them, they still decode) are sampled, with their
  routable label, so KV / running / waiting are not lost while hidden (gap G2/G3);
* the pilot's routable-only set is unchanged (pod_gauges.jsonl stays comparable);
* the per-GPU map and the SM routable view come from one /v2/state sample (G2/G3);
* container CPU comes from the cumulative kubelet counter (G8).
"""
from __future__ import annotations

import sys
from pathlib import Path

RUNNER = Path(__file__).resolve().parents[1] / "runner"
sys.path.insert(0, str(RUNNER))

import sampler  # noqa: E402

STATE = {
    "version": 7,
    "models": {"m7": {"awake": 2}},
    "bindings": [
        {"binding_id": "m7/n1/0", "serve_id": "m7-n1-gpu-0", "model": "m7", "node": "n1", "gpu_ids": ["0"],
         "awake": True, "hidden": False, "routable": True},
        {"binding_id": "m7/n1/1", "serve_id": "m7-n1-gpu-1", "model": "m7", "node": "n1", "gpu_ids": ["1"],
         "awake": True, "hidden": True, "routable": False},
        {"binding_id": "m7/n2/0", "serve_id": "m7-n2-gpu-0", "model": "m7", "node": "n2", "gpu_ids": ["0"],
         "awake": False, "hidden": False, "routable": False},
    ],
}
PODS = {
    "m7-n1-gpu-0-6f8f8b8fdb-277c4": {"ip": "10.0.0.1", "node": "n1", "model": "m7", "routable_label": "true"},
    "m7-n1-gpu-1-6f8f8b8fdb-x7k2p": {"ip": "10.0.0.2", "node": "n1", "model": "m7", "routable_label": "false"},
    "m7-n2-gpu-0-6f8f8b8fdb-q2w8x": {"ip": "10.0.0.3", "node": "n2", "model": "m7", "routable_label": "false"},
}


def test_hidden_awake_pods_are_sampled_with_their_routable_label():
    chosen = sampler.select_pods(PODS, STATE)
    hidden = chosen["m7-n1-gpu-1-6f8f8b8fdb-x7k2p"]
    assert (hidden["sm_hidden"], hidden["routable_label"], hidden["binding_id"]) == (True, "false", "m7/n1/1")
    assert "m7-n1-gpu-0-6f8f8b8fdb-277c4" in chosen
    assert "m7-n2-gpu-0-6f8f8b8fdb-q2w8x" not in chosen  # asleep and not routable


def test_the_pilot_routable_only_set_is_unchanged():
    assert set(sampler.select_pods(PODS, STATE, routable_only=True)) == {"m7-n1-gpu-0-6f8f8b8fdb-277c4"}


def test_layout_and_gpu_map_from_one_state_sample():
    row = sampler.layout_row(1.0, STATE)
    assert row["models"]["m7"]["awake"] == ["m7/n1/0", "m7/n1/1"]
    assert row["models"]["m7"]["hidden"] == ["m7/n1/1"]
    assert row["models"]["m7"]["routable"] == ["m7/n1/0"]
    gm = sampler.gpu_map(STATE)
    assert gm["n1/1"] == {"awake": ["m7/n1/1"], "hidden": ["m7/n1/1"]}
    assert "n2/0" not in gm


def test_container_cpu_is_the_cumulative_counter_rate():
    prev: dict = {}

    def summary(nanos, stamp):
        return {"pods": [{"podRef": {"namespace": "tre", "name": "tre-v2-controller-bfdd97594-dqp54"},
                          "containers": [{"name": "controller",
                                          "cpu": {"usageCoreNanoSeconds": nanos, "usageNanoCores": 5e8, "time": stamp},
                                          "memory": {"rssBytes": 2 ** 21}}]},
                         {"podRef": {"namespace": "other", "name": "x-1"}, "containers": []}]}

    first = sampler.resource_rows(0.0, "n1", summary(10 ** 9, "2026-10-07T00:00:00Z"), ["tre"], prev)
    second = sampler.resource_rows(5.0, "n1", summary(3 * 10 ** 9, "2026-10-07T00:00:05Z"), ["tre"], prev)
    assert [r["component"] for r in first] == ["controller"]
    assert first[0]["cpu_cores"] == 0.5          # first sample: kubelet's own rate
    assert second[0]["cpu_cores"] == 0.4         # 2 core-seconds over 5 s
    assert second[0]["rss_mib"] == 2.0
