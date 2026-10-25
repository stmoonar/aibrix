"""E1 runner sampler (tre/eval/runner/sampler.py): the collector behaviours the report needs.

* hidden pods (SafeScale / SM sleep hides them, they still decode) are sampled, with their
  routable label, so KV / running / waiting are not lost while hidden (gap G2/G3);
* the pilot's routable-only set is unchanged (pod_gauges.jsonl stays comparable);
* the per-GPU map and the SM routable view come from one /v2/state sample (G2/G3);
* container CPU comes from the cumulative kubelet counter (G8).
"""
from __future__ import annotations

import json
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


def test_cadence_row_counts_docs_against_the_declared_periods_and_records_a_missing_hash():
    """gateway_cadence.jsonl: per pod the docs of the last period vs the declared instant / evidence
    periods; a gateway without the cadence hash (images before 2026-10-25) is recorded as null and
    judged on the legacy 10 s grid, never an error."""
    doc = '{"timestamp": 1000000, "written_ms": 1000030, "scraped_ms": 999800}'
    pods = {"default/p1": "m7", "default/p2": "m7"}
    docs = {"default/p1": {"inst_n": 20, "hist_n": 5, "inst_last": (doc, 1000000), "hist_last_ms": 1000000},
            "default/p2": {"inst_n": 12, "hist_n": 5, "inst_last": None, "hist_last_ms": None}}
    # The gateway's declared progress anchors the window (2026-10-26): last_hist_ms = the
    # newest complete round's boundary, never the Redis clock that races the 500 ms grid.
    cad = {"instant_ms": "500", "evidence_ms": "2000", "written_ms": "1001535", "last_hist_ms": "1000000",
           "version": "v"}
    row = sampler.cadence_row(1.0, 1000400, cad, pods, docs, 10.0)
    assert row["declared"] and row["expected"] == {"inst": 20, "hist": 5} and row["window_end_ms"] == 1000000
    assert row["window_end_source"] == "last_hist_ms"
    pre_round5 = {k: v for k, v in cad.items() if k != "last_hist_ms"}
    assert sampler.cadence_window_anchor(pre_round5, 1000400) == (1001535, "written_ms", None)
    assert row["summary"] == {"pods": 2, "inst_full": 1, "hist_full": 2}
    assert (row["pods"]["default/p1"]["write_lag_ms"], row["pods"]["default/p1"]["scrape_age_ms"]) == (30, 200)
    assert row["pods"]["default/p2"]["write_lag_ms"] is None
    legacy = sampler.cadence_row(1.0, 1000400, None, pods, docs, 10.0)
    assert legacy["cadence"] is None and not legacy["declared"] and legacy["expected"] == {"inst": 1, "hist": 1}
    assert legacy["window_end_ms"] == 1000400 and legacy["window_end_source"] == "redis_time"


def test_cadence_row_falls_back_to_written_ms_on_a_frozen_last_hist_ms(tmp_path):
    """HSET never deletes last_hist_ms: more than 2 evidence periods behind written_ms it is
    frozen (gateway rollback, stalled histogram rounds) - the window ends at written_ms and
    first_run_check flags the run."""
    import first_run_check as F

    pods = {"default/p1": "m7"}
    docs = {"default/p1": {"inst_n": 20, "hist_n": 0, "inst_last": None, "hist_last_ms": 990000}}
    frozen = {"instant_ms": "500", "evidence_ms": "2000", "written_ms": "1000035", "last_hist_ms": "990000"}
    row = sampler.cadence_row(1.0, 1000400, frozen, pods, docs, 10.0)
    assert (row["window_end_ms"], row["window_end_source"], row["anchor_lag_ms"]) == (
        1000035, "written_ms_hist_frozen", 10035)
    live = sampler.cadence_row(1.0, 1000400, dict(frozen, last_hist_ms="998000"), pods, docs, 10.0)
    assert live["window_end_source"] == "last_hist_ms"  # 2 periods behind: a late round, not frozen
    path = tmp_path / "gateway_cadence.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in (row, live)) + "\n")
    scan = F.cadence_scan(path)
    assert scan["hist_frozen_samples"] == 1 and scan["anchor_lag_ms_max"] == 10035
