from __future__ import annotations

import json

from scripts.analysis import safescale_summary


def _record(request_id: str, resolution: str | None, *, code: str | None = None, terminal: str = "",
            gate: str | None = None, pre_hide: float | None = None, extensions: int = 0, gates=()) -> str:
    terms: dict = {"extensions": extensions}
    if gate:
        terms.update(latency_gate=gate, latency_source="evidence", threshold_mode="labels", latency_samples=25)
    if pre_hide is not None:
        terms["tail_pre_hide_fraction"] = pre_hide
    if code:
        terms["rollback_reason"] = {"code": code, "gates": list(gates)}
    record = {"request_id": request_id, "status": "resolved" if resolution else "probing",
              "terminal_reason": terminal, "window_terms": terms,
              "terminal_details": {"rollback_reason": terms["rollback_reason"]} if code else {}}
    if resolution:
        record["resolution"] = resolution
    return json.dumps(record)


def test_summary_reports_rollback_rate_and_reason_distribution(tmp_path) -> None:
    snapshot = {
        "probes": {
            "a": _record("a", "commit", gate="evaluated", pre_hide=0.0, extensions=1),
            "b": _record("b", "rollback", code="formal_commit_gate_failed", gates=("latency",), gate="evaluated",
                         pre_hide=0.0),
            "c": _record("c", "rollback", code="evidence_clock_skew"),
            "d": _record("d", "rollback", terminal="observe_entered"),  # pre-2026-09-29 / no structured reason
            "e": _record("e", None),  # still probing: not counted in the rate
        },
        "journals": {},
    }
    path = tmp_path / "safescale.json"
    path.write_text(json.dumps(snapshot), encoding="utf-8")
    summary = safescale_summary.summarize(safescale_summary.load_probe_records(json.loads(path.read_text())))
    assert (summary["probes"], summary["decided"], summary["commits"], summary["rollbacks"]) == (5, 4, 1, 3)
    assert summary["rollback_rate"] == 0.75
    assert summary["rollback_reasons"] == {
        "formal_commit_gate_failed": 1, "evidence_clock_skew": 1, "observe_entered": 1,
    }
    assert summary["formal_gate_failures"] == {"latency": 1}
    assert summary["latency_gate"] == {"evaluated": 2}
    assert summary["extensions_total"] == 1
    assert summary["evidence_pre_hide_fraction_max"] == 0.0
    assert safescale_summary.main([str(path)]) == 0


def test_summary_groups_the_resolved_probes_by_evidence_source() -> None:
    def record(request_id, resolution, source, code=None):
        terms = {"evidence_source_used": source} if source else {}
        if code:
            terms["rollback_reason"] = {"code": code}
        return {"request_id": request_id, "status": "resolved", "resolution": resolution, "window_terms": terms}

    summary = safescale_summary.summarize([
        record("a", "commit", "direct"),
        record("b", "rollback", "direct", "slo_violation_direct"),
        record("c", "rollback", "redis_fallback", "evidence_unavailable"),
        record("d", "commit", None),
    ])
    assert summary["by_evidence_source"] == {
        "direct": {"decided": 2, "commits": 1, "rollbacks": 1, "rollback_rate": 0.5,
                   "rollback_reasons": {"slo_violation_direct": 1}},
        "redis_fallback": {"decided": 1, "commits": 0, "rollbacks": 1, "rollback_rate": 1.0,
                           "rollback_reasons": {"evidence_unavailable": 1}},
        "unknown": {"decided": 1, "commits": 1, "rollbacks": 0, "rollback_rate": 0.0, "rollback_reasons": {}},
    }


def test_summary_reports_the_pre_hide_share_of_direct_probes() -> None:
    def record(request_id, source, pre_hide):
        terms = {"latency_source": source, "tail_pre_hide_fraction": pre_hide, "evidence_source_used": "direct"}
        return {"request_id": request_id, "status": "resolved", "resolution": "commit", "window_terms": terms}

    summary = safescale_summary.summarize([record("a", "direct", 0.0), record("b", "direct", 0.25)])
    assert summary["evidence_pre_hide_fraction_max"] == 0.25
    summary = safescale_summary.summarize([record("c", "gateway", 0.5)])
    assert summary["evidence_pre_hide_fraction_max"] is None
