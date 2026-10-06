#!/usr/bin/env python3
"""Pilot comparison (10-06 evening): one row per arm dir, per-model + ALL.

Reads score.json (score_pilot.py output), analyze.out (layout, client, sm sleeps), sm.log
(409 counts), and for baseline arms baseline/ (decision JSONL, *.metrics.txt).
Usage: compare_arms.py <arm_dir> [<arm_dir> ...] [--json out.json]
"""
import glob
import json
import os
import re
import sys
from collections import Counter

TP = {"dsqwen-14b": 2}
MODELS = ("dsqwen-7b", "dsllama-8b", "dsqwen-14b")


def load(p):
    try:
        with open(p) as fh:
            return json.load(fh)
    except Exception:
        return None


def decisions(d):
    lines = []
    files = glob.glob(os.path.join(d, "baseline", "*", "decisions-*.jsonl"))
    for f in files:
        with open(f) as fh:
            for raw in fh:
                try:
                    lines.append(json.loads(raw))
                except ValueError:
                    pass
    lines.sort(key=lambda l: (l.get("ts_ms", 0), l.get("model", "")))
    return lines


def metrics(d):
    out = {}
    for f in glob.glob(os.path.join(d, "baseline", "*.metrics.txt")):
        for line in open(f):
            if line.startswith("#"):
                continue
            m = re.match(r'(\w+)\{([^}]*)\}\s+([-0-9.e+]+)', line)
            if not m:
                continue
            labels = dict(re.findall(r'(\w+)="([^"]*)"', m.group(2)))
            key = (m.group(1), labels.get("model"), labels.get("name") or labels.get("action"))
            out[key] = out.get(key, 0.0) + float(m.group(3))
    return out


def arm_row(d):
    score = load(os.path.join(d, "score.json")) or {}
    an = load(os.path.join(d, "analyze.out")) or {}
    row = {"dir": d, "label": open(os.path.join(d, "arm_label")).read().strip()
           if os.path.exists(os.path.join(d, "arm_label")) else os.path.basename(d),
           "valid": score.get("valid"), "models": {}}
    load_s = an.get("load_s") or 0
    gpu_s = 0.0
    layout = an.get("layout") or {}
    for m in ("ALL",) + MODELS:
        s = (score.get("models") or {}).get(m, {})
        r = {k: s.get(k) for k in ("n", "fail", "V_req_pct", "ttft_p50_ms", "ttft_p95_ms", "ttft_p99_ms",
                                   "tpot_p95_ms")}
        if m in layout:
            r["layout_changes"] = layout[m].get("n_changes")
            r["replica_s"] = layout[m].get("replica_s")
            gpu_s += (layout[m].get("replica_s") or 0) * TP.get(m, 1)
            r["mean_gpus"] = round((layout[m].get("replica_s") or 0) * TP.get(m, 1) / load_s, 2) if load_s else None
        c = (an.get("client") or {}).get(m, {})
        r["continued"] = c.get("tre_continued")
        r["interrupted"] = c.get("interrupted")
        row["models"][m] = r
    row["mean_gpus"] = round(gpu_s / load_s, 2) if load_s else None
    row["layout_changes"] = sum((layout[m].get("n_changes") or 0) for m in layout)
    row["sm_sleeps"] = (an.get("sm_sleep_delta") or {}).get("sleeps_total")
    row["sidecar_delta"] = an.get("sidecar_delta")
    # SM 409s from the SM access log
    sm409 = Counter()
    smlog = os.path.join(d, "sm.log")
    if os.path.exists(smlog):
        for line in open(smlog, errors="replace"):
            if '" 409' in line or " 409 " in line:
                if "/target" in line:
                    sm409["target"] += 1
                elif "/scale_service" in line:
                    sm409["scale_service"] += 1
                else:
                    sm409["other"] += 1
    row["sm_409"] = dict(sm409)
    lines = decisions(d)
    if lines:
        mt = metrics(d)
        bl = {"lines": len(lines)}
        act = Counter((l["model"], l["action"]) for l in lines)
        bl["actions"] = {f"{m}:{a}": n for (m, a), n in sorted(act.items()) if a not in ("none",)}
        res = [l["sm_result"] for l in lines if "sm_result" in l]
        bl["sm_refusals"] = sum(1 for r in res if r.get("code") == 409)
        bl["partial_fill"] = sum(1 for r in res if r.get("partial_fill"))
        bl["partial_fill_by_model"] = dict(Counter(l["model"] for l in lines
                                                   if l.get("sm_result", {}).get("partial_fill")))
        bl["reversals_60s"] = {k[1]: v for k, v in mt.items() if k[0] == "tre_bl_direction_reversals_60s_total"}
        bl["policy_events"] = {f"{k[1]}:{k[2]}": v for k, v in mt.items()
                               if k[0] == "tre_bl_policy_events_total" and v}
        # flips: dispatched direction changes per model
        flips = Counter()
        last = {}
        for l in lines:
            if l["action"] in ("up", "down"):
                if last.get(l["model"]) and last[l["model"]] != l["action"]:
                    flips[l["model"]] += 1
                last[l["model"]] = l["action"]
        bl["flips"] = dict(flips)
        pol = lines[0].get("policy")
        if pol == "preserve":
            split = Counter()
            below = 0
            for l in lines:
                if l["action"] in ("up", "down"):
                    rsn = l.get("reason", "")
                    tier = "tier1" if rsn.startswith("tier1") else "tier2" if rsn.startswith("tier2") else rsn
                    split[f"{tier}:{l['action']}"] += 1
                    n = (l.get("inputs") or {}).get("tier1_n")
                    if n is not None and l["clamped"] < n:
                        below += 1
            bl["preserve_action_split"] = dict(split)
            bl["preserve_below_N_actions"] = below
        row["baseline"] = bl
    return row


def main(argv):
    out = None
    if "--json" in argv:
        i = argv.index("--json")
        out = argv[i + 1]
        argv = argv[:i] + argv[i + 2:]
    rows = [arm_row(d.rstrip("/")) for d in argv]
    hdr = f"{'arm':24} {'model':11} {'n':>6} {'Vreq%':>7} {'ttft50':>7} {'ttft95':>8} {'ttft99':>8} {'tpot95':>7} {'gpus':>5} {'chg':>4} {'cont':>5}"
    print(hdr)
    for r in rows:
        for m, v in r["models"].items():
            f = lambda x, p=1: "-" if x is None else (f"{x:.{p}f}" if isinstance(x, float) else str(x))
            print(f"{r['label'][:24]:24} {m:11} {f(v['n']):>6} {f(v['V_req_pct'],2):>7} {f(v['ttft_p50_ms'],0):>7} "
                  f"{f(v['ttft_p95_ms'],0):>8} {f(v['ttft_p99_ms'],0):>8} {f(v['tpot_p95_ms'],1):>7} "
                  f"{f(v.get('mean_gpus') if m != 'ALL' else r['mean_gpus'],2):>5} "
                  f"{f(v.get('layout_changes') if m != 'ALL' else r['layout_changes']):>4} {f(v['continued']):>5}")
        print(f"  valid={r['valid']} sm_sleeps={r['sm_sleeps']} sm_409={r['sm_409']}")
        if "baseline" in r:
            b = dict(r["baseline"])
            b.pop("actions", None)
            print("  baseline:", json.dumps(b, sort_keys=True))
    if out:
        with open(out, "w") as fh:
            json.dump(rows, fh, indent=1, sort_keys=True)


if __name__ == "__main__":
    main(sys.argv[1:])
