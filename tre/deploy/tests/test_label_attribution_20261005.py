"""Label v2 (hybrid attribution, 2026-10-05): the invariants the next calibration round rests on.

* a burst followed by its drain tail: under the completion attribution (v1) the tail window
  is labelled a violation by the burst's late completions; under hybrid it carries only the
  TTFTs whose first token falls in it, so it is unlabelled (min-n counts the TTFT set) -
  while the TSS numerator (tokens of the completions) and the TPOT evidence do not move;
* the v1 label is reproducible byte for byte: the frozen 2026-10-03 label definition still
  hashes to the sha256 sealed in the freeze, and the hybrid label hashes to another.
"""
from __future__ import annotations

import json

from tre_common import slo_labels
from tre_common.registry import load_registry
from scripts import dline_refit, r3_grid, rewindow_from_raw

from pathlib import Path

REGISTRY = Path(__file__).resolve().parents[1] / "registry.yaml"
MODEL = "dsqwen-7b"

#: The dsqwen-14b label definition of calib_20261003/freeze/params_freeze.json and the
#: label_def_sha256 the freeze sealed for it.
FROZEN_LABEL_DEF = json.loads(
    '{"b_ms_per_token": 0.0879, "c_ms": 45.3, "e2e": "excluded", "floor_ms": 500.0, "k": 5.0, '
    '"latency_columns": ["p95_ttft_client_ms", "p95_tpot_client_ms"], "min_n": 20, '
    '"missing_latency": "unlabeled unless unserved (also when completed_requests < min_n)", '
    '"mode": "slowdown", "name": "p95_ttft_slowdown_tpot_plus_unserved_v1", '
    '"percentile_mode": "bucket_upper", "tpot_p95_ms": 75.0, "ttft_p95_ms": 500.0, '
    '"ttft_p95_ms_unused": true, '
    '"ttft_slo_ms": "max(floor_ms, k * (c_ms + b_ms_per_token * prompt_tokens))", '
    '"unserved": "violated when any of model_errors, proxy_transient_errors, client_timeouts > 0 '
    '(requests sent in the window); slo_violated is never read", '
    '"violated_if": "p95_i(ttft_i / ttft_slo_ms(L_i)) > 1 or p95_tpot_client_ms > tpot_p95_ms or unserved"}')
FROZEN_LABEL_SHA256 = "b108dc24b56567f035b50e61f04ed4ebe319a8be84f2c08b0569e14e176183f9"


def _req(send_s: float, ttft_s: float, decode_s: float, *, out_tokens: int = 100) -> dict:
    send = int(send_s * 1000)
    first = send + int(ttft_s * 1000)
    done = first + int(decode_s * 1000)
    return {"send_ts_ms": send, "recv_first_token_ts_ms": first, "done_ts_ms": done,
            "ttft_ms": ttft_s * 1000.0, "tpot_ms": decode_s * 1000.0 / (out_tokens - 1),
            "e2e_ms": float(done - send), "input_tokens": 512, "output_tokens": out_tokens,
            "http_status": 200, "outcome": "ok"}


def _burst_then_tail() -> list[dict]:
    # 30 requests of a burst sent in [1, 10) s: first token 19 s later (queued), done in the
    # next window [30, 60) s - the drain tail; plus 5 fresh fast requests sent in the tail.
    burst = [_req(1.0 + 0.3 * k, 19.0, 15.0) for k in range(30)]
    fresh = [_req(40.0 + 2.0 * k, 0.2, 2.0) for k in range(5)]
    return burst + fresh


def _label(attribution: str) -> slo_labels.LabelDefinition:
    return slo_labels.LabelDefinition(ttft_p95_ms=500.0, tpot_p95_ms=75.0, min_completed_requests=20,
                                      attribution=attribution)


def _rows(attribution: str) -> list[dict]:
    spec = load_registry(str(REGISTRY)).model(MODEL)
    return rewindow_from_raw.label_cell(
        _burst_then_tail(), [], r3_grid.GridCell.from_scenario_id("i512_o100_c9"), spec,
        label=_label(attribution), window_ms=30_000, step_ms=30_000, percentile_mode="bucket_upper",
        min_latency_samples=10, instant_sample_interval_ms=1_000, instant_grid="raw",
        start_ms=0, end_ms=60_000)


def test_drain_tail_is_violated_by_completion_and_unlabelled_by_hybrid() -> None:
    done, hybrid = _rows("completion"), _rows("hybrid")
    tail_done, tail_hybrid = done[1], hybrid[1]
    assert (tail_done["window_start_ms"], tail_hybrid["window_start_ms"]) == (30_000, 30_000)
    # v1: the burst's 30 completions (19 s TTFT) + 5 fresh ones label the recovered window
    assert tail_done[slo_labels.COMPLETED_REQUESTS_COLUMN] == 35
    assert tail_done[slo_labels.LABEL_COLUMN] == slo_labels.LABEL_VIOLATED
    # v2: only the 5 first tokens of the tail are its TTFT set -> below min-n -> no label
    assert tail_hybrid[slo_labels.COMPLETED_REQUESTS_COLUMN] == 5
    assert tail_hybrid[slo_labels.LABEL_COLUMN] == slo_labels.LABEL_UNLABELED
    # the burst's TTFTs now sit in the burst window, where their first tokens arrived
    assert hybrid[0][slo_labels.COMPLETED_REQUESTS_COLUMN] == 30
    assert hybrid[0]["p95_ttft_client_ms"] >= 19_000.0
    # what the attribution must not move: the TSS numerator and the TPOT evidence
    for a, b in zip(done, hybrid):
        for col in ("prompt_tokens_total", "generation_tokens_total", "p95_tpot_client_ms",
                    "p95_e2e_client_ms", "trs"):
            assert a.get(col) == b.get(col), col


def test_v1_label_reproduces_the_frozen_sha_and_v2_differs() -> None:
    v1 = slo_labels.LabelDefinition.from_dict(FROZEN_LABEL_DEF)
    assert v1.attribution == slo_labels.ATTRIBUTION_COMPLETION
    assert v1.as_dict() == FROZEN_LABEL_DEF
    assert dline_refit.canonical_sha256(v1.as_dict()) == FROZEN_LABEL_SHA256
    # the registry route builds the same v1 label (no attribution given = completion)
    built = slo_labels.label_def_for_model("dsqwen-14b", ttft_p95_ms=500.0, tpot_p95_ms=75.0,
                                           c=45.3, b=0.0879, mode="slowdown", k=5.0, floor=500.0)
    assert dline_refit.canonical_sha256(built.as_dict()) == FROZEN_LABEL_SHA256
    v2 = slo_labels.LabelDefinition.from_dict({**FROZEN_LABEL_DEF, "attribution": "hybrid"})
    assert v2.hybrid and v2.as_dict()["name"].endswith("_v2")
    assert dline_refit.canonical_sha256(v2.as_dict()) != FROZEN_LABEL_SHA256
    assert slo_labels.LabelDefinition.from_dict(v2.as_dict()) == v2
    # every arm of a v2 primary is v2, and a v1 command line carries no attribution flag
    assert {a.attribution for a in slo_labels.label_arms(v2).values()} == {"hybrid"}
    assert "--label-attribution" not in slo_labels.label_cli_args(v1)
    assert slo_labels.label_cli_args(v2)[-2:] == ["--label-attribution", "hybrid"]
