"""Every vLLM metric TRE reads resolves against a real vLLM 0.30 and a 0.10.1 /metrics body.

vLLM 0.11+ (and the 0.30 fork the model pods run) dropped ``gpu_cache_usage_perc``,
``time_per_output_token_seconds`` and ``gpu_prefix_cache_*``. A reader that still names
only the old family reads nothing - silently (APA scales on 0, least-gpu-cache routes at
random, the TPOT histogram stays empty). The name table lives in
``tre_common.vllm_metrics`` (Go: ``pkg/metrics/engine_fetcher.go``); these tests pin it,
and the readers that parse /metrics text themselves, to the two samples in
``fixtures/vllm_metrics`` (0.30: live pod capture; 0.10.1: the stock stat logger rendered
offline, see the fixture header).
"""
from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from scripts import openloop
from tre_common.vllm_metrics import (
    GATEWAY_DOC_KEYS,
    OPTIONAL_VLLM_METRICS,
    VLLM_METRICS,
    doc_key_candidates,
    doc_lookup,
    resolve_vllm_name,
    sample_names,
    vllm_candidates,
)
from tre_sm.ops.sleep_primitive import parse_vllm_load

TRE_ROOT = Path(__file__).resolve().parents[2]
FIXTURES = Path(__file__).resolve().parent / "fixtures" / "vllm_metrics"
SAMPLES = {
    "0.30.0": FIXTURES / "vllm-0.30.0.prom",
    "0.10.1": FIXTURES / "vllm-0.10.1.prom",
}
APA_DIR = TRE_ROOT / "deploy" / "baselines" / "apa"


def _text(version: str) -> str:
    return SAMPLES[version].read_text()


@pytest.mark.parametrize("version", sorted(SAMPLES))
def test_every_required_metric_resolves(version: str) -> None:
    names = sample_names(_text(version))
    missing = {canonical: raw for canonical, raw in VLLM_METRICS.items() if resolve_vllm_name(canonical, names) is None}
    assert not missing, f"vLLM {version} sample lacks every candidate of {missing}"


def test_030_resolves_to_the_new_names() -> None:
    names = sample_names(_text("0.30.0"))
    assert resolve_vllm_name("kv_cache_usage_perc", names) == "vllm:kv_cache_usage_perc"
    assert resolve_vllm_name("inter_token_latency_seconds", names) == "vllm:inter_token_latency_seconds"
    assert resolve_vllm_name("prefix_cache_queries", names) == "vllm:prefix_cache_queries_total"
    assert resolve_vllm_name("prefix_cache_hits", names) == "vllm:prefix_cache_hits_total"
    # The e2e histogram is exported under its unchanged name (SafeScale / latency_p95 e2e).
    assert "vllm:e2e_request_latency_seconds_bucket" in names
    # The old names really are gone on 0.30 - the reason for the fallback table.
    for old in ("vllm:gpu_cache_usage_perc", "vllm:time_per_output_token_seconds_bucket",
                "vllm:gpu_prefix_cache_queries_total"):
        assert old not in names


def test_0101_keeps_the_old_tpot_name_and_both_cache_names() -> None:
    names = sample_names(_text("0.10.1"))
    assert resolve_vllm_name("inter_token_latency_seconds", names) == "vllm:time_per_output_token_seconds"
    # 0.10.1 exports both KV-cache gauges; the new one is preferred.
    assert {"vllm:kv_cache_usage_perc", "vllm:gpu_cache_usage_perc"} <= names
    assert resolve_vllm_name("kv_cache_usage_perc", names) == "vllm:kv_cache_usage_perc"


def test_candidates_are_newest_first_and_unique() -> None:
    seen: set[str] = set()
    for canonical, raws in {**VLLM_METRICS, **OPTIONAL_VLLM_METRICS}.items():
        assert raws and all(raw.startswith("vllm:") for raw in raws), canonical
        assert not seen & set(raws), f"{canonical} shares a raw name with another entry"
        seen |= set(raws)
    assert vllm_candidates("kv_cache_usage_perc")[-1] == "vllm:gpu_cache_usage_perc"
    assert vllm_candidates("inter_token_latency_seconds")[-1] == "vllm:time_per_output_token_seconds"
    with pytest.raises(KeyError):
        vllm_candidates("gpu_cache_usage_perc")


def test_gateway_doc_keys_prefer_new_identifier_and_fall_back() -> None:
    for written, candidates in GATEWAY_DOC_KEYS.items():
        assert candidates[-1] == written  # what the gateway writes today stays readable
    new_and_old = {"m/inter_token_latency_seconds": "new", "m/time_per_output_token_seconds": "old"}
    assert doc_lookup(new_and_old, "m", "time_per_output_token_seconds") == "new"
    assert doc_lookup({"m/time_per_output_token_seconds": "old"}, "m", "time_per_output_token_seconds") == "old"
    assert doc_lookup({"m/kv_cache_usage_perc": 0.3}, "m", "gpu_cache_usage_perc") == 0.3
    assert doc_lookup({"m/e2e_request_latency_seconds": 1}, "m", "e2e_request_latency_seconds") == 1
    assert doc_lookup({}, "m", "e2e_request_latency_seconds") is None
    assert doc_key_candidates("num_requests_waiting") == ("num_requests_waiting",)


@pytest.mark.parametrize("version", sorted(SAMPLES))
def test_apa_target_metric_is_exported(version: str) -> None:
    """The AIBrix APA fetcher maps targetMetric to exactly one raw name (no fallback in the
    aibrix-system build) and reads a missing metric as 0: the name must exist verbatim."""
    names = sample_names(_text(version))
    crs = sorted(APA_DIR.glob("*-apa.yaml"))
    assert crs
    for path in crs:
        cr = yaml.safe_load(path.read_text())
        for source in cr["spec"]["metricsSources"]:
            assert "vllm:" + source["targetMetric"] in names, (path.name, version)


@pytest.mark.parametrize("version", sorted(SAMPLES))
def test_text_parsers_read_the_samples(version: str) -> None:
    text = _text(version)
    # SM sleep drain check: running + waiting found (0 on these idle samples, not None =
    # "no gauges"); num_requests_waiting_by_reason (0.30) must not be summed in.
    assert parse_vllm_load(text) == 0
    assert parse_vllm_load(text + 'vllm:num_requests_waiting_by_reason{reason="capacity"} 5.0\n') == 0
    # openloop sidecar KV-cache diagnostic reads a value on both builds.
    assert openloop.parse_pod_kv_cache_usage(text) == 0.0
    assert openloop.POD_KV_CACHE_USAGE_GAUGES == vllm_candidates("kv_cache_usage_perc")
    gauges = openloop.parse_pod_gauges(text)
    assert gauges == {"waiting": 0.0, "running": 0.0, "swapping": 0.0}
