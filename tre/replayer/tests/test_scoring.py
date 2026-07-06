from __future__ import annotations

from tre_replayer.scoring import compute_v_sys, oracle_normalized_score, request_metrics, window_violations


def _rec(ts, ttft, e2e, comp, status=200, error=None):
    return {"actual_send_ts_ms": ts, "ttft_ms": ttft, "e2e_ms": e2e, "completion_tokens": comp, "http_status": status, "error": error}


def test_request_metrics_computes_tpot() -> None:
    m = request_metrics(_rec(0, 100.0, 900.0, 9))  # tpot = (900-100)/(9-1) = 100
    assert m["tpot_ms"] == 100.0 and m["ttft_ms"] == 100.0 and m["ok"] is True


def test_request_metrics_error_is_not_ok() -> None:
    assert request_metrics(_rec(0, None, None, None, status=500, error="HTTP 500"))["ok"] is False


def test_compute_v_sys_request_and_time_fractions() -> None:
    recs = [_rec(i * 1000, 50.0, 400.0, 10) for i in range(20)]  # first 20s within SLO
    recs += [_rec(20000 + i * 1000, 50.0, 900.0, 10) for i in range(20)]  # next 20s violate e2e>500
    v = compute_v_sys(recs, ttft_slo_ms=500, tpot_slo_ms=200, e2e_slo_ms=500, window_ms=10000, step_ms=10000, min_samples=3)
    assert 0.4 <= v["violation_request_frac"] <= 0.6
    assert v["violation_time_frac"] > 0.0
    assert v["n_requests"] == 40


def test_window_violations_flags_high_p95() -> None:
    recs = [_rec(i * 500, 50.0, 900.0, 10) for i in range(20)]  # all violate e2e>500
    wins = window_violations(recs, ttft_slo_ms=500, tpot_slo_ms=200, e2e_slo_ms=500, window_ms=5000, step_ms=5000, min_samples=3)
    assert any(w["violated"] for w in wins)


def test_oracle_normalized_score() -> None:
    assert oracle_normalized_score(0.5, 0.1, 0.0) == 0.8  # (0.5-0.1)/(0.5-0)
    assert oracle_normalized_score(0.5, 0.5, 0.0) == 0.0  # no better than static
    assert oracle_normalized_score(0.5, 0.0, 0.0) == 1.0  # matches oracle
    assert oracle_normalized_score(0.3, 0.2, 0.3) == 1.0  # degenerate v_static==v_oracle, sys<=oracle
    assert oracle_normalized_score(0.3, 0.4, 0.3) == 0.0  # degenerate, sys>oracle
