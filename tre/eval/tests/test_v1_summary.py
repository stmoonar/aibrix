"""v1-format cross-arm summary (tre_eval/v1_summary.py) on a small synthetic TRE arm.

The row's numbers are the tre_eval metric functions' (no second definition): latency percentiles and
timeouts from score_requests, replicas / GPUs from gpu_accounting, scale up / down from the layout,
probes from safescale_summary.json, action classes from the controller's trs_calc_result actions.
"""
import json
import os

import pytest

pytest.importorskip("numpy")

from tre_eval import load as L  # noqa: E402
from tre_eval import metrics as M  # noqa: E402
from tre_eval import v1_summary as V  # noqa: E402

from test_eval_metrics import A0, A1, B0, B23, T0, _jl, lay, req, write_arm  # noqa: E402


def make_tre_arm(root, name="sens-theta-m20"):
    d = str(root / "Simul_spike_s1" / name)
    layout = [(float(t), lay([A0, A1] if 30 <= t < 60 else [A0], [B0, B23] if t >= 50 else [B0])) for t in range(-5, 101)]
    rows = [req(i, model="m-a" if i % 2 else "m-b", t=float(i), e2e=5.0 + (i % 10), ttft=0.1 + 0.01 * (i % 7))
            for i in range(95)]
    rows.append({**req(95, model="m-b", t=60.0, e2e=150.0, ttft=0.2), "success_strict": False})       # cut by the route timeout
    write_arm(d, layout, rows)
    acts = [(31, [{"model": "m-a", "delta": 1, "reason": "critical_sleeping_capacity"},
                  {"model": "m-b", "delta": -1, "reason": "critical_donor_immediate"}]),     # donor side: not urgent
            (50, [{"receiver": "m-b", "donor": "m-a", "reason": "low_fairness_idle_capacity"},
                  {"model": "m-b", "reason": "probe_started", "kind": "hide"}]),
            (60, [{"model": "m-a", "delta": -1, "reason": "high_proactive_safescale"}]),
            (200, [{"model": "m-a", "delta": 1, "reason": "critical_sleeping_capacity"}])]   # after the load: not counted
    _jl(os.path.join(d, "controller.log"), [{"message": json.dumps(
        {"event": "trs_calc_result", "ts_ms": str(int((T0 + t) * 1000)), "now_ms": str(int((T0 + t) * 1000)),
         "loop": "rescue", "submitted": 1, "actions": json.dumps(a)})} for t, a in acts])
    with open(os.path.join(d, "safescale_summary.json"), "w") as fh:
        json.dump({"probes": 3, "commits": 2, "rollbacks": 1}, fh)
    with open(os.path.join(d, "campaign_manifest.json"), "w") as fh:
        json.dump({"registry_rules": ["*.trs.theta_m x0.8"],
                   "controller_env": {"TRE_SAFESCALE_SCOPE": "receiverless_shrink"}}, fh)
    with open(os.path.join(d, "CAMPAIGN_DONE.json"), "w") as fh:
        json.dump({"validity": {"verdict": "suspect" if name == "tre" else "valid"}}, fh)
    return d


def test_v1_row_reuses_tre_eval_metrics_and_classifies_actions(tmp_path):
    d = make_tre_arm(tmp_path)
    r = V.arm_row(d)
    arm = L.load_arm(d)
    _, summ = M.score_requests(arm.requests, arm.slo, trim_s=30.0, t_ref=arm.t_load)
    a = summ["ALL"]
    assert r["e2e_p90"] == pytest.approx(a["e2e_p90_ms"] / 1000) and r["ttft_p99"] == pytest.approx(a["ttft_p99_ms"] / 1000, abs=1e-3)
    assert r["tpot_p50"] == pytest.approx(a["tpot_p50_ms"], abs=0.1)
    assert r["timeouts"] == 1 and r["timeout_pct"] == pytest.approx(100 / a["n"], abs=1e-3)
    # m-a 1 -> 2 at 30 s, 2 -> 1 at 60 s; m-b 1 -> 2 at 50 s (a TP-2 binding): replica-s / load, GPUs with TP
    assert (r["scale_up"], r["scale_down"]) == (2, 1)
    assert r["replicas"] == pytest.approx((100 + 30 + 100 + 50) / 100, abs=0.02)
    assert r["gpus"] == pytest.approx((100 + 30 + 100 + 2 * 50) / 100, abs=0.02)
    assert (r["probes"], r["probe_commits"], r["probe_rollbacks"]) == (3, 2, 1)
    assert (r["act_urgent"], r["act_fairness"], r["act_high_shrink"], r["act_other"]) == (1, 1, 1, 1)
    assert r["tpot_v1_p95"] == (None if a["tpot_v1_p95_ms"] is None else pytest.approx(a["tpot_v1_p95_ms"], abs=0.1))
    assert r["verdict"] == "valid"
    assert r["setting"] == "*.trs.theta_m x0.8"


def test_v1_summary_files_from_a_results_root(tmp_path):
    make_tre_arm(tmp_path / "res", "tre")
    make_tre_arm(tmp_path / "res", "sens-theta-m20")
    rows = [V.arm_row(p) for p in V.find_arms([str(tmp_path / "res")])]
    assert [r["arm"] for r in rows] == ["sens-theta-m20", "tre"]
    paths = V.write(rows, str(tmp_path / "out"))
    md = open(paths["md"]).read()
    assert "## Simul_spike_s1" in md and "| sens-theta-m20 (*.trs.theta_m x0.8) |" in md
    assert md.count("| E2E (s) |") == 1 and "scale up / down" in md and "m-a | m-b" in md
    assert open(paths["csv"]).readline().startswith("trace,arm,label,setting")
    doc = json.load(open(paths["json"]))
    assert len(doc["rows"]) == 2 and "115 s" in doc["notes"] and "(e2e - ttft)" in doc["notes"]
    assert "| TRE [suspect] |" in md or "(*.trs.theta_m x0.8) [suspect] |" in md
    assert "150 s route timeout" in md and "TPOT v1 (ms)" in md
