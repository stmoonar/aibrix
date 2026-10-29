"""Cross-arm summary in the v1 ablation / sensitivity table format (tre_v1 results/ablation-20261023.md).

One row per arm, grouped by trace. Every number comes from the tre_eval metric functions (the
report's own definitions): ``score_requests`` (latency percentiles, timeouts, V_req), ``gpu_accounting``
(replica-s, GPU-s), ``interruption`` (sidecar reissues), ``layout_series`` (scale up / down), the
arm's ``safescale_summary.json`` (probes) and the controller's ``trs_calc_result`` actions.

Columns (v1 basis first, v2 basis next to it):
  replicas            mean awake replicas summed over the models over the load window (v1 "GPU")
  GPUs                mean awake GPUs (14b TP2 counts 2; the v2 cost basis)
  E2E / TTFT (s), TPOT (ms)   p50 / p90 / p95 / p99 over the strictly complete scored requests (first trim_s
                      seconds dropped, as the report). TPOT = v2 definition (e2e - ttft) / (output_tokens - 1);
                      "TPOT v1" = the client's own TPOT field (score_requests tpot_v1, p50 / p95 / p99)
  reissues            sidecar continuations (tre_reissue events) in the load window (v1 "重发")
  completion / cuts   complete and truncated shares of all scored requests, plus failed/unknown counts
  timeouts            truncated requests with e2e >= 149 s: a legacy diagnostic compatible with the
                      150 s route timeout, not proof of its cause (v1 counted "> 115 s")
  V_req %             the v2 primary metric
  scale up / down     increases / decreases of a model's awake replicas in the layout samples (v1 definition)
  probes              SafeScale probes started / committed / rolled back (safescale_summary.json)
  verdict             the campaign's validity verdict; a "suspect" arm is marked in the tables
  actions             controller actions by reason (TRE arms): urgent = critical_* scale-ups only (receiver side,
                      as v1's rescue gain; donor-side critical_* actions count as other), HIGH shrink =
                      high_proactive*, fairness = low_fairness*, LOW rescue = low_rescue*, IDLE shrink = idle*
Per model: mean awake replicas; E2E p50 / p95 / p99 (s); TTFT p95 (s).

CLI: ``python3 tre/eval/v1_summary.py <results_root | arm_dir>... --out DIR [--trim-s 30]``
(a results root: every ``<trace>/<arm>/`` with a CAMPAIGN_DONE.json). Writes v1_summary.md,
v1_summary.json ({"notes", "rows"}) and v1_summary.csv.
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
from collections import Counter
from typing import Any

from . import load as L
from . import metrics as M

ACTION_CLASSES = (("urgent", ("critical_",)), ("high_shrink", ("high_proactive",)), ("fairness", ("low_fairness",)),
                  ("low_rescue", ("low_rescue",)), ("idle_shrink", ("idle",)))
PCTS = (50, 90, 95, 99)
V1_PCTS = (50, 95, 99)       # score_requests' tpot_v1 percentiles
NOTES = ("Latency cells: p50 / p90 / p95 / p99 of the strictly complete scored requests (first trim_s seconds dropped). "
         "TPOT = v2 definition (e2e - ttft) / (output_tokens - 1); TPOT v1 = the client's TPOT field (p50 / p95 / p99). "
         "replicas = mean awake replicas summed over the models (v1 \"GPU\"); GPUs = mean awake GPUs (TP2 = 2). "
         "Completion and truncation rates share the scored-request denominator; failed and unknown are separate. "
         "timeouts = truncated with e2e >= 149 s (150 s route timeout diagnostic, not proof of cause); v1 counted > 115 s "
         "(its route cut), not the same threshold. Normal finish_reason=length is complete. "
         "urgent = critical_* scale-ups only (v1 rescue gain). "
         "[suspect] = campaign verdict suspect. Definitions: tre_eval/v1_summary.py.")


def action_class(reason: str | None) -> str:
    r = str(reason or "")
    for name, prefixes in ACTION_CLASSES:
        if r.startswith(prefixes):
            return name
    return "other"


def _verdict(arm_dir: str) -> str | None:
    for f in ("CAMPAIGN_DONE.json", "campaign_manifest.json"):
        v = ((L._read_json(os.path.join(arm_dir, f)) or {}).get("validity") or {}).get("verdict")
        if v:
            return v
    return None


def _setting(arm_dir: str) -> str:
    """What this arm changes (campaign_manifest.json): registry rules and non-default controller settings."""
    man = L._read_json(os.path.join(arm_dir, "campaign_manifest.json")) or {}
    parts = list(man.get("registry_rules") or [])
    env = man.get("controller_env") or {}
    sig = env.get("TRE_SIGNAL_SOURCE")
    if sig:
        parts.insert(0, f"signal={sig}")
    if str(env.get("TRE_DISABLE_ETA_GATE", "false")).lower() == "true":
        parts.append("eta gate off")
    return "; ".join(parts)


def arm_row(path: str, trim_s: float = 30.0) -> dict[str, Any]:
    arm = L.load_arm(path)
    max_tok = {t["request_id"]: t.get("max_output_tokens") for t in arm.traces}
    recs, summ = M.score_requests(arm.requests, arm.slo, trim_s=trim_s, max_tok=max_tok, t_ref=arm.t_load)
    gpu = M.gpu_accounting(arm, summ)
    inter = M.interruption(arm, recs)
    a = summ.get(M.ALL) or {}
    load_s = (gpu.get(M.ALL) or {}).get("load_s")
    row: dict[str, Any] = {"trace": arm.trace_name, "arm": arm.name, "label": arm.label, "setting": _setting(arm.dir),
                           "dir": arm.dir, "verdict": _verdict(arm.dir), "n": a.get("n"), "fail": a.get("fail"), "V_req_pct": a.get("V_req_pct")}
    for key in ("complete", "truncated", "failed", "unknown", "completion_rate_pct", "truncation_rate_pct", "scoring_basis"):
        row[key] = a.get(key)
    rep_s = sum((gpu.get(m) or {}).get("replica_s") or 0.0 for m in arm.models)
    row["replicas"] = round(rep_s / load_s, 2) if load_s else None
    row["gpus"] = round((gpu.get(M.ALL) or {}).get("mean_gpus"), 2) if (gpu.get(M.ALL) or {}).get("mean_gpus") is not None else None
    for m, scale in (("e2e", 1e-3), ("ttft", 1e-3), ("tpot", 1.0)):
        for p in PCTS:
            v = a.get(f"{m}_p{p}_ms")
            row[f"{m}_p{p}"] = None if v is None else round(v * scale, 3 if scale != 1.0 else 1)
    for p in V1_PCTS:
        v = a.get(f"tpot_v1_p{p}_ms")
        row[f"tpot_v1_p{p}"] = None if v is None else round(v, 1)
    row["reissues"] = inter.get("sidecar_reissue_events")
    row["continued_requests"] = inter.get("client_continued_requests")
    row["timeouts"] = a.get("censored_ge_149s")
    row["timeout_pct"] = round(100.0 * a["censored_ge_149s"] / a["n"], 3) if a.get("n") else None
    # scale up / down: changes of each model's awake replica count in the load window (v1 definition)
    ups = downs = 0
    for s in M.layout_series(arm).values():
        win = [x for x in s if arm.t_load is not None and arm.t_end is not None and arm.t_load <= x[0] <= arm.t_end]
        for p, q in zip(win, win[1:]):
            ups += q[1] > p[1]
            downs += q[1] < p[1]
    row["scale_up"], row["scale_down"] = ups, downs
    ss = arm.files.get("safescale_summary.json") or {}
    row["probes"], row["probe_commits"], row["probe_rollbacks"] = ss.get("probes"), ss.get("commits"), ss.get("rollbacks")
    acts: Counter = Counter()
    by_reason: Counter = Counter()
    if M.decision_source(arm) == "tre":
        for t in arm.ctrl_ticks:
            ts = M.tick_time(t)
            if arm.t_load is not None and arm.t_end is not None and not arm.t_load <= ts <= arm.t_end:
                continue
            for x in t["actions"]:
                if isinstance(x, dict) and x.get("reason") != "probe_started":   # probes are counted above
                    cls = action_class(x.get("reason"))
                    if cls == "urgent" and not (x.get("delta") or 0) > 0:
                        # Donor side of a CRITICAL action (a transfer's donor sleeps carry
                        # the receiver too): v1 counted the gain only.
                        cls = "other"
                    acts[cls] += 1
                    by_reason[str(x.get("reason"))] += 1
    row.update({f"act_{name}": acts.get(name, 0) for name, _ in ACTION_CLASSES})
    row["act_other"] = acts.get("other", 0)
    row["actions_by_reason"] = dict(sorted(by_reason.items()))
    row["models"] = {}
    for m in arm.models:
        s = summ.get(m) or {}
        g = gpu.get(m) or {}
        sec = lambda v: None if v is None else round(v / 1000.0, 2)  # noqa: E731
        row["models"][m] = {"replicas": round(g["replica_s"] / load_s, 2) if load_s and g.get("replica_s") is not None else None,
                            **{k: s.get(k) for k in ("complete", "truncated", "failed", "unknown", "completion_rate_pct", "truncation_rate_pct")},
                            "e2e_p50": sec(s.get("e2e_p50_ms")), "e2e_p95": sec(s.get("e2e_p95_ms")),
                            "e2e_p99": sec(s.get("e2e_p99_ms")), "ttft_p95": sec(s.get("ttft_p95_ms"))}
    row["warnings"] = list(arm.warnings)
    return row


def find_arms(paths: list[str]) -> list[str]:
    out = []
    for p in paths:
        p = p.rstrip("/")
        if os.path.isfile(os.path.join(p, "client", "performance_metrics.json")):
            out.append(p)
            continue
        for m in sorted(glob.glob(os.path.join(p, "*", "*", "CAMPAIGN_DONE.json"))):
            out.append(os.path.dirname(m))
    return out


# ---------------------------------------------------------------- rendering
def _f(v: Any) -> str:
    if v is None:
        return "-"
    if isinstance(v, float):
        return f"{v:.2f}" if abs(v) >= 1 or v == 0 else f"{v:.3f}"
    return str(v)


def _quad(r: dict, m: str) -> str:
    return " / ".join(_f(r.get(f"{m}_p{p}")) for p in PCTS)


def _name(r: dict) -> str:
    return (r["label"] + (f" ({r['setting']})" if r.get("setting") else "")
            + (" [suspect]" if r.get("verdict") == "suspect" else ""))


def markdown(rows: list[dict]) -> str:
    traces: dict[str, list] = {}
    for r in rows:
        traces.setdefault(r["trace"], []).append(r)
    out = ["# Cross-arm summary (v1 table format)", "", NOTES, ""]
    for t, rs in traces.items():
        models = sorted({m for r in rs for m in r["models"]})
        out += [f"## {t}", "",
                "| arm | replicas | GPUs | E2E (s) | TTFT (s) | TPOT (ms) | TPOT v1 (ms) | reissues | timeouts | V_req % |",
                "|---|---|---|---|---|---|---|---|---|---|"]
        for r in rs:
            v1 = " / ".join(_f(r.get(f"tpot_v1_p{p}")) for p in V1_PCTS)
            out.append(f"| {_name(r)} | {_f(r['replicas'])} | {_f(r['gpus'])} | {_quad(r, 'e2e')} | {_quad(r, 'ttft')} | "
                       f"{_quad(r, 'tpot')} | {v1} | {_f(r['reissues'])} | {_f(r['timeouts'])} ({_f(r['timeout_pct'])} %) | "
                       f"{_f(r['V_req_pct'])} |")
        out += ["", "Completion (all scored requests):", "",
                "| arm | complete | truncated | failed | unknown | completion % | truncation % |",
                "|---|---|---|---|---|---|---|"]
        for r in rs:
            out.append(f"| {_name(r)} | " + " | ".join(_f(r.get(k)) for k in
                       ("complete", "truncated", "failed", "unknown", "completion_rate_pct", "truncation_rate_pct")) + " |")
        out += ["", "Controller actions:", "",
                "| arm | scale up / down | probes started / committed / rolled back | HIGH shrink | urgent (CRITICAL) "
                "| fairness | LOW rescue | IDLE shrink | other |",
                "|---|---|---|---|---|---|---|---|---|"]
        for r in rs:
            out.append(f"| {_name(r)} | {r['scale_up']} / {r['scale_down']} | {_f(r['probes'])} / {_f(r['probe_commits'])} / "
                       f"{_f(r['probe_rollbacks'])} | {r['act_high_shrink']} | {r['act_urgent']} | {r['act_fairness']} | "
                       f"{r['act_low_rescue']} | {r['act_idle_shrink']} | {r['act_other']} |")
        out += ["", "Per model (mean awake replicas; E2E p50 / p95 / p99 (s); TTFT p95 (s)):", "",
                "| arm | " + " | ".join(models) + " |", "|---|" + "---|" * len(models)]
        for r in rs:
            cells = []
            for m in models:
                x = r["models"].get(m) or {}
                cells.append(f"{_f(x.get('replicas'))}; {_f(x.get('e2e_p50'))} / {_f(x.get('e2e_p95'))} / "
                             f"{_f(x.get('e2e_p99'))}; {_f(x.get('ttft_p95'))}")
            out.append(f"| {_name(r)} | " + " | ".join(cells) + " |")
        out.append("")
    return "\n".join(out)


CSV_FIELDS = ["trace", "arm", "label", "setting", "verdict", "n", "fail", "complete", "truncated", "failed", "unknown",
              "completion_rate_pct", "truncation_rate_pct", "scoring_basis", "V_req_pct", "replicas", "gpus",
              *[f"{m}_p{p}" for m in ("e2e", "ttft", "tpot") for p in PCTS],
              *[f"tpot_v1_p{p}" for p in V1_PCTS], "reissues", "continued_requests",
              "timeouts", "timeout_pct", "scale_up", "scale_down", "probes", "probe_commits", "probe_rollbacks",
              *[f"act_{n}" for n, _ in ACTION_CLASSES], "act_other", "dir"]


def write(rows: list[dict], out: str) -> dict[str, str]:
    os.makedirs(out, exist_ok=True)
    paths = {k: os.path.join(out, f"v1_summary.{k}") for k in ("md", "json", "csv")}
    with open(paths["md"], "w") as fh:
        fh.write(markdown(rows) + "\n")
    with open(paths["json"], "w") as fh:
        json.dump({"notes": NOTES, "rows": rows}, fh, indent=1, default=str)
    with open(paths["csv"], "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=CSV_FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    return paths


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("paths", nargs="+", help="campaign results roots and / or arm directories")
    ap.add_argument("--out", required=True)
    ap.add_argument("--trim-s", type=float, default=30.0)
    a = ap.parse_args(argv)
    arms = find_arms(a.paths)
    if not arms:
        ap.error("no arm directories found")
    rows = [arm_row(p, a.trim_s) for p in arms]
    for k, p in write(rows, a.out).items():
        print(f"{k}: {p}")
    return 0
