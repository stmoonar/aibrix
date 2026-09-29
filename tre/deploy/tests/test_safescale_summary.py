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
