"""Score a replayer run against per-model SLOs (audit blocker B3).

Pure functions over the per-request JSONL the StreamingHttpSender produces. Reports SLO
violation as both a time fraction (fraction of windows in violation) and a request fraction,
and the paper's oracle-normalized score (V_static - V_sys) / (V_static - V_oracle).

TPOT DEFINITION (review F6, ruling): here tpot is the per-request MEAN inter-token latency
(e2e - ttft) / (completion_tokens - 1), and its p95 is taken ACROSS requests. This is the
standard serving-benchmark definition and is used identically for every system under test, so
TRE-vs-baseline comparisons are fair. Note it is NOT the same statistic as the controller/R3
`tpot_p95` column, which is vLLM's per-token-gap histogram p95 (metrics_store). Both are compared
to the same registry SLO (75ms); p95-of-per-request-means <= p95-of-per-token-gaps, so this scorer
is the more lenient of the two. The paper must state this single SLO definition; R3's histogram
`tpot_p95` is a calibration input, not the reported SLO metric.
"""
from __future__ import annotations

from typing import Any


def request_metrics(record: dict[str, Any]) -> dict[str, Any]:
    ttft = record.get("ttft_ms")
    e2e = record.get("e2e_ms")
    completion = record.get("completion_tokens")
    tpot = None
    if ttft is not None and e2e is not None and completion and completion > 1:
        tpot = (e2e - ttft) / (completion - 1)
    ok = record.get("http_status") == 200 and not record.get("error")
    return {"ts_ms": record.get("actual_send_ts_ms"), "ttft_ms": ttft, "tpot_ms": tpot, "e2e_ms": e2e, "ok": ok}


def _violates(m: dict[str, Any], ttft_slo_ms: float, tpot_slo_ms: float, e2e_slo_ms: float) -> bool:
    if not m["ok"]:
        return True
    return (
        (m["ttft_ms"] is not None and m["ttft_ms"] > ttft_slo_ms)
        or (m["tpot_ms"] is not None and m["tpot_ms"] > tpot_slo_ms)
        or (m["e2e_ms"] is not None and m["e2e_ms"] > e2e_slo_ms)
    )


def _p95(values: list[Any]) -> float | None:
    vals = sorted(v for v in values if v is not None)
    if not vals:
        return None
    return vals[min(len(vals) - 1, int(len(vals) * 0.95))]


def window_violations(
    records: list[dict[str, Any]],
    *,
    ttft_slo_ms: float,
    tpot_slo_ms: float,
    e2e_slo_ms: float,
    window_ms: int,
    step_ms: int,
    t0_ms: int | None = None,
    t1_ms: int | None = None,
    min_samples: int = 5,
) -> list[dict[str, Any]]:
    metrics = [m for m in (request_metrics(r) for r in records) if m["ts_ms"] is not None]
    if not metrics:
        return []
    t0 = t0_ms if t0_ms is not None else min(m["ts_ms"] for m in metrics)
    t1 = t1_ms if t1_ms is not None else max(m["ts_ms"] for m in metrics)
    windows: list[dict[str, Any]] = []
    end = t0 + window_ms
    while end <= t1 + step_ms:
        start = end - window_ms
        win = [m for m in metrics if start < m["ts_ms"] <= end]
        p95t, p95p, p95e = _p95([m["ttft_ms"] for m in win]), _p95([m["tpot_ms"] for m in win]), _p95([m["e2e_ms"] for m in win])
        errors = sum(1 for m in win if not m["ok"])
        violated = False
        if len(win) >= min_samples:
            violated = (
                (p95t is not None and p95t > ttft_slo_ms)
                or (p95p is not None and p95p > tpot_slo_ms)
                or (p95e is not None and p95e > e2e_slo_ms)
                or errors > 0
            )
        windows.append(
            {"window_end_ms": end, "n_requests": len(win), "violated": violated,
             "ttft_p95": p95t, "tpot_p95": p95p, "e2e_p95": p95e, "errors": errors}
        )
        end += step_ms
    return windows


def compute_v_sys(
    records: list[dict[str, Any]],
    *,
    ttft_slo_ms: float,
    tpot_slo_ms: float,
    e2e_slo_ms: float,
    window_ms: int = 30_000,
    step_ms: int = 5_000,
    min_samples: int = 5,
) -> dict[str, Any]:
    metrics = [request_metrics(r) for r in records]
    n_req = len(metrics)
    req_viol = sum(1 for m in metrics if _violates(m, ttft_slo_ms, tpot_slo_ms, e2e_slo_ms))
    windows = window_violations(
        records, ttft_slo_ms=ttft_slo_ms, tpot_slo_ms=tpot_slo_ms, e2e_slo_ms=e2e_slo_ms,
        window_ms=window_ms, step_ms=step_ms, min_samples=min_samples,
    )
    scored = [w for w in windows if w["n_requests"] >= min_samples]
    # F7: None (not 0.0) when nothing was scored -> a too-short / too-sparse run must not read
    # as "perfect". Callers/aggregators must handle None explicitly.
    time_frac = (sum(1 for w in scored if w["violated"]) / len(scored)) if scored else None
    return {
        "violation_time_frac": time_frac,
        "violation_request_frac": (req_viol / n_req) if n_req else None,
        "n_requests": n_req,
        "n_windows_scored": len(scored),
    }


def oracle_normalized_score(v_static: float, v_sys: float, v_oracle: float) -> float:
    """(V_static - V_sys) / (V_static - V_oracle). 1.0 = matches the post-hoc optimum,
    0.0 = no better than the static baseline. Degenerate V_static==V_oracle: 1.0 if the
    system is at least as good as the oracle, else 0.0."""
    denom = v_static - v_oracle
    if abs(denom) < 1e-12:
        return 1.0 if v_sys <= v_oracle + 1e-12 else 0.0
    return (v_static - v_sys) / denom
