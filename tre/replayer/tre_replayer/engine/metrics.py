"""Two metric bases over one answer: v1's and the strict one.

Every request the core sends is measured both ways, from the same
:class:`~tre_replayer.engine.stream.StreamResult`, and the column names say which:

========================  ==========================================  ==========================================
                          v1 basis (``ttft``/``tpot``/``success``;     strict basis (``*_strict*``; calibration
                          ``v1_*`` on calibration rows)               labels, ``slo_labels``)
========================  ==========================================  ==========================================
TTFT                      first chunk with ``delta.content`` not      first chunk carrying text / content /
                          None - the role-only chunk included         reasoning (``TTFT_BASIS``)
TPOT                      (end - first) / completion_tokens           (E2E - TTFT) / (completion_tokens - 1)
end                       end of the body (after ``[DONE]``)           the ``[DONE]`` line
success                   any 2xx whose headers arrived: a stream      2xx, complete (``[DONE]`` or a finish
                          cut mid-way, an in-stream error chunk and    reason), no in-stream error, >= 1
                          zero output are all successes               completion token
missing TTFT              not counted (dropped from percentiles)       a violation (``ttft_missing_strict``)
retries                   inside TTFT / E2E (timed from the first      excluded: timed from the last attempt;
                          attempt)                                    ``retries`` / ``retry_wait`` reported
========================  ==========================================  ==========================================

Units: the v1 columns keep v1's seconds; strict columns carry their unit in the name.
"""
from __future__ import annotations

import math
from typing import Any, Iterable, Optional, Sequence

from tre_replayer.engine.stream import TTFT_BASIS, V1_TTFT_BASIS, StreamResult

STRICT_FAILURE_HTTP = "http_error"
STRICT_FAILURE_TIMEOUT = "client_timeout"
STRICT_FAILURE_TRANSPORT = "transport_error"
STRICT_FAILURE_STREAM_ERROR = "stream_error"
STRICT_FAILURE_INCOMPLETE = "incomplete_stream"
STRICT_FAILURE_ZERO_OUTPUT = "zero_output"

BASES = {
    "v1": {"ttft": V1_TTFT_BASIS, "tpot": "(end - first_chunk) / completion_tokens",
           "success": "2xx headers received (interrupted / error-chunk / zero-output streams count)",
           "missing_ttft": "excluded", "retries": "included in TTFT / E2E"},
    "strict": {"ttft": TTFT_BASIS, "tpot": "(e2e - ttft) / (completion_tokens - 1)",
               "success": "2xx + complete + no in-stream error + completion_tokens >= 1",
               "missing_ttft": "violation", "retries": "excluded (timed from the last attempt)"},
}


def strict_failure(res: StreamResult) -> Optional[str]:
    """Why ``res`` is not a strict success (None = it is)."""
    status = int(res.status or 0)
    if not 200 <= status < 300:
        if status:
            return STRICT_FAILURE_HTTP
        return STRICT_FAILURE_TIMEOUT if res.timed_out else STRICT_FAILURE_TRANSPORT
    if res.stream_error is not None:
        return STRICT_FAILURE_STREAM_ERROR
    if res.stream_interrupted or not (res.done_seen or res.finish_reason):
        return STRICT_FAILURE_INCOMPLETE
    if not res.completion_tokens or int(res.completion_tokens) < 1:
        return STRICT_FAILURE_ZERO_OUTPUT
    return None


def strict_view_ms(res: StreamResult) -> dict[str, Any]:
    """Strict basis, milliseconds, timed from the start of the last attempt."""
    base = float(res.last_attempt_offset_ms or 0.0)
    failure = strict_failure(res)
    ok = failure is None
    ttft = None if (not ok or res.first_token_ms is None) else float(res.first_token_ms) - base
    e2e = None if res.done_ms is None else float(res.done_ms) - base
    n = res.completion_tokens
    tpot = None
    if ok and ttft is not None and e2e is not None and n is not None and n > 1:
        tpot = (e2e - ttft) / (float(n) - 1.0)
    return {"success": ok, "failure": failure, "ttft_ms": ttft, "tpot_ms": tpot, "e2e_ms": e2e,
            "ttft_missing": ok and ttft is None, "retries": max(0, int(res.attempts or 1) - 1),
            "retry_wait_ms": base}


def v1_view_s(res: StreamResult) -> dict[str, Any]:
    """v1 basis, seconds, timed from before the first attempt (v1's ``start_time``)."""
    success = bool(res.v1_success)
    done_ms = res.v1_done_ms if res.v1_done_ms is not None else res.done_ms
    e2e = None if done_ms is None else float(done_ms) / 1000.0
    ttft = tpot = None
    prompt = completion = total = 0
    if success:
        prompt, completion, total = res.v1_prompt_tokens or 0, res.v1_completion_tokens or 0, res.v1_total_tokens or 0
        first = res.v1_first_token_ms
        if first is not None:
            ttft = float(first) / 1000.0
            if completion > 0 and done_ms is not None:
                tpot = (float(done_ms) - float(first)) / 1000.0 / completion
    return {"success": success, "ttft_s": ttft, "tpot_s": tpot, "e2e_s": e2e, "input_tokens": prompt,
            "output_tokens": completion, "total_tokens": total}


def dual_fields_ms(res: StreamResult) -> dict[str, Any]:
    """Both bases as extra columns of a calibration / replay row (``dual_metrics``)."""
    v1 = v1_view_s(res)
    strict = strict_view_ms(res)
    ms = (lambda s: None if s is None else s * 1000.0)
    return {
        "v1_success": v1["success"], "v1_ttft_ms": ms(v1["ttft_s"]), "v1_tpot_ms": ms(v1["tpot_s"]),
        "v1_e2e_ms": ms(v1["e2e_s"]),
        "strict_success": strict["success"], "strict_failure": strict["failure"],
        "strict_ttft_ms": strict["ttft_ms"], "strict_tpot_ms": strict["tpot_ms"],
        "strict_e2e_ms": strict["e2e_ms"], "strict_ttft_missing": strict["ttft_missing"],
        "stream_interrupted": bool(res.stream_interrupted),
    }


# ------------------------------------------------------------------------ summaries


def percentile(values: Iterable[Optional[float]], p: float) -> Optional[float]:
    """Linear-interpolated percentile of the non-None values (None when there are none)."""
    xs = sorted(float(v) for v in values if v is not None and not (isinstance(v, float) and math.isnan(v)))
    if not xs:
        return None
    k = (len(xs) - 1) * p / 100.0
    lo = math.floor(k)
    hi = min(lo + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def summarize_v1_records(records: Sequence[dict], percentiles: Sequence[float] = (50, 95, 99)) -> dict:
    """Both bases over ``performance_metrics.json`` lines (the ``e1_v1`` record)."""
    out: dict[str, Any] = {"requests": len(records)}
    ok_v1 = [r for r in records if r.get("success")]
    ok_strict = [r for r in records if r.get("success_strict")]
    out["v1"] = {"success": len(ok_v1), "failed": len(records) - len(ok_v1),
                 "stream_interrupted_but_success": sum(1 for r in ok_v1 if r.get("stream_interrupted"))}
    out["strict"] = {"success": len(ok_strict), "failed": len(records) - len(ok_strict),
                     "ttft_missing": sum(1 for r in ok_strict if r.get("ttft_missing_strict")),
                     "failures": _count(r.get("failure_strict") for r in records if not r.get("success_strict"))}
    for p in percentiles:
        tag = f"p{int(p)}"
        out["v1"][f"ttft_{tag}_s"] = percentile((r.get("ttft") for r in ok_v1), p)
        out["v1"][f"tpot_{tag}_s"] = percentile((r.get("tpot") for r in ok_v1), p)
        out["v1"][f"e2e_{tag}_s"] = percentile((r.get("e2e_latency") for r in ok_v1), p)
        out["strict"][f"ttft_{tag}_s"] = percentile((r.get("ttft_strict_s") for r in ok_strict), p)
        out["strict"][f"tpot_{tag}_s"] = percentile((r.get("tpot_strict_s") for r in ok_strict), p)
        out["strict"][f"e2e_{tag}_s"] = percentile((r.get("e2e_strict_s") for r in ok_strict), p)
    out["retried_requests"] = sum(1 for r in records if (r.get("retries") or 0) > 0)
    out["total_attempts"] = sum(int(r.get("attempts") or 0) for r in records)
    lateness = [r.get("send_lateness_ms") for r in records]
    out["send_lateness_ms"] = {f"p{int(p)}": percentile(lateness, p) for p in (50, 99)}
    out["send_lateness_ms"]["max"] = max((x for x in lateness if x is not None), default=None)
    out["tre_continued_requests"] = sum(1 for r in records if r.get("tre_continued"))
    out["tre_retried_requests"] = sum(1 for r in records if r.get("tre_retried"))
    out["bases"] = BASES
    return out


def _count(values: Iterable[Any]) -> dict:
    out: dict[str, int] = {}
    for value in values:
        out[str(value)] = out.get(str(value), 0) + 1
    return out
