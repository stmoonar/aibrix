"""Post-run check (tre/eval/runner/first_run_check.py) and the recording it reads: the
invariants the campaign verdict depends on.

* a baseline whose ticks cover less than 0.9 x load / tick, or reports more than 5 % frozen
  ticks, makes the arm suspect; a baseline that reports no frozen field gives null, not 0;
* the baseline scaler's node, recorded while it runs, survives the end phase of
  components.json (it is at 0 replicas then) - its log clock is that node's.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

RUNNER = Path(__file__).resolve().parents[1] / "runner"
sys.path.insert(0, str(RUNNER))

import components  # noqa: E402
import first_run_check as F  # noqa: E402


def _arm(tmp: Path, ticks: int, frozen_every: int | None) -> Path:
    d = tmp / "Simul_spike_s1" / "preserve"
    (d / "client").mkdir(parents=True)
    (d / "baseline" / "scaler-pod").mkdir(parents=True)
    (d / "arm_meta.json").write_text(json.dumps({"arm": "preserve"}))
    (d / "load_start_epoch").write_text("1000\n")
    (d / "load_end_epoch").write_text("1500\n")
    (d / "controller_restart_epoch").write_text("900\n")
    reqs = [{"model_name": "m7", "start_time": 1000 + i, "success": True, "e2e_latency": 2.0, "ttft": 0.2,
             "output_tokens": 11, "http_status": 200} for i in range(50)]
    (d / "client" / "performance_metrics.json").write_text("\n".join(json.dumps(r) for r in reqs) + "\n")
    lines = []
    for k in range(ticks):
        r = {"ts_ms": int((1000 + 5 * k) * 1000), "tick": k, "model": "m7", "action": "none", "reason": "hold",
             "inputs": {}}
        if frozen_every:
            r["inputs"]["frozen"] = k % frozen_every == 0
        lines.append(json.dumps(r))
    (d / "baseline" / "scaler-pod" / "decisions-preserve-20261026.jsonl").write_text("\n".join(lines) + "\n")
    (d / "baseline" / "run_validity.json").write_text(json.dumps({"events_valid": True}))
    return d


def test_baseline_tick_coverage_and_frozen_share_gate_the_verdict(tmp_path):
    ok = F.check(_arm(tmp_path / "a", ticks=100, frozen_every=None))["validity"]   # 100 ticks over 500 s / 5 s
    assert ok["tick_coverage"]["baseline:m7"] == 1.0 and ok["frozen_tick_share"] is None
    assert not [s for s in ok["suspect"] if "baseline m7" in s]
    bad = F.check(_arm(tmp_path / "b", ticks=80, frozen_every=10))["validity"]     # 0.8 coverage, 10 % frozen
    assert any("ticks" in s and "baseline m7" in s for s in bad["suspect"])
    assert bad["frozen_tick_share"]["m7"] == 0.1 and any("frozen-tick share" in s for s in bad["suspect"])


def test_scaler_node_recorded_while_running_survives_the_end_phase(tmp_path, monkeypatch):
    path = tmp_path / "components.json"
    scaler = {"namespace": "tre-v2", "name": "tre-v2-baseline-scaler-5cfbfc4896-djmk8", "component": "baseline-scaler",
              "node": "node9", "phase": "Running", "containers": []}
    ctrl = {"namespace": "tre-v2", "name": "tre-v2-controller-664f855b95-q9nx6", "component": "controller",
            "node": "node10", "phase": "Running", "containers": []}
    phases = {"bl_enabled": [ctrl, scaler], "end": [ctrl]}
    monkeypatch.setenv("TRE_NS", "tre-v2")
    monkeypatch.setenv("BL_NS", "tre-v2")
    for phase in ("bl_enabled", "end"):
        monkeypatch.setattr(components, "pods", lambda ns, sel=None, _p=phase: [r for r in phases[_p] if ns == "tre-v2" and sel is None])
        components.main(str(path), phase)
    doc = json.loads(path.read_text())
    assert doc["component_nodes"]["baseline-scaler"] == "node9" and doc["component_nodes"]["controller"] == "node10"
    assert doc["baseline_scaler"]["node"] == "node9" and doc["baseline_scaler"]["phase"] == "bl_enabled"


def test_expected_rescue_interval_follows_the_overlay_snapshot_period():
    """The check's default controller tick is the single loop's rescue spacing on the
    overlay's snapshot grid (TRE_EVIDENCE_PERIOD_MS): exactly 5 s at 1 s snapshots."""
    import inspect
    import re

    overlay = (Path(__file__).resolve().parents[2] / "deploy" / "overlays" / "tre-v2" / "controller.yaml").read_text()
    period_ms = int(re.search(r'name: TRE_EVIDENCE_PERIOD_MS\s+value: "(\d+)"', overlay).group(1))
    default_s = inspect.signature(F.expected_rescue_interval_s).parameters["snapshot_s"].default
    assert default_s * 1000 == period_ms
    assert F.expected_rescue_interval_s(snapshot_s=1.0) == 5.0


def test_settling_episodes_are_counted_per_model(tmp_path):
    # TSS suffix (2026-10-29): one settling_start / settling_end per episode in the tick events.
    def tick(events):
        message = {"event": "trs_calc_result", "loop": "rescue", "now_ms": 1_000_000,
                   "events": json.dumps(events), "actions": "[]", "model_states": "{}"}
        return json.dumps({"message": json.dumps(message)})

    log = tmp_path / "controller.log"
    log.write_text("\n".join([
        tick(["settling_start:m7:settle_span", "rescue_target:x"]),
        tick(["settling_end:m7:valid:evidence_ms=10200", "settling_start:m8:no_suffix"]),
    ]) + "\n")
    scan = F.controller_scan(log, None, None)
    assert scan["settling"] == {"m7": {"settling_start": 1, "settling_end": 1}, "m8": {"settling_start": 1}}
