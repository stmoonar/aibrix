#!/usr/bin/env python3
"""Analyze S1.2 W-freeze trials: compute onset->first-non-null-z_m lag (primary)
plus state-elevation / settle context, and P50/P95/max."""
import json

trials = json.load(open("/tmp/s1wf_trials.json"))
raw = json.load(open("/tmp/s1wf_raw_hist.json"))

def pct(vals, q):
    if not vals: return None
    s = sorted(vals)
    return s[min(len(s) - 1, int(round((len(s) - 1) * q)))]

rows = []
for t in trials:
    m = t["model"]; onset = t["onset_ms"]; load_end = t["load_end_ms"]
    ents = [e for e in raw[m] if onset - 20000 <= e["ts"] <= load_end + 20000]
    ents.sort(key=lambda e: e["ts"])
    # baseline: last entry with ts <= onset must be idle (z_m null)
    pre = [e for e in ents if e["ts"] <= onset]
    baseline_idle = (not pre) or (pre[-1].get("z_m") is None)
    post = [e for e in ents if e["ts"] > onset]
    first_zm = next((e for e in post if e.get("z_m") is not None), None)
    first_state = next((e for e in post if e.get("state") not in (None, "idle")), None)
    # plateau: max z_m over post entries within load period
    inload = [e for e in post if e["ts"] <= load_end and e.get("z_m") is not None]
    plat = max((e["z_m"] for e in inload), default=None)
    settle = None
    if plat is not None:
        thr = 0.9 * plat
        s = next((e for e in inload if e["z_m"] >= thr), None)
        settle = (s["ts"] - onset) / 1000.0 if s else None
    row = {
        "trial": t["trial"], "model": m,
        "baseline_idle": baseline_idle,
        "lag_first_s": round((first_zm["ts"] - onset) / 1000.0, 1) if first_zm else None,
        "first_zm": round(first_zm["z_m"], 3) if first_zm else None,
        "first_state": first_zm["state"] if first_zm else None,
        "lag_state_s": round((first_state["ts"] - onset) / 1000.0, 1) if first_state else None,
        "plateau_zm": round(plat, 2) if plat is not None else None,
        "lag_settle_s": round(settle, 1) if settle is not None else None,
        "load_ok": t["ok"], "load_err": t["err"], "rps": t["rps"], "p95_ms": t["lat_p95_ms"],
    }
    rows.append(row)

print("%-3s %-11s %-6s %-9s %-8s %-7s %-9s %-8s %-7s %-5s" % (
    "T", "model", "idle?", "lag1st_s", "first_zm", "state", "lagstate", "plateau", "settle", "err"))
for r in rows:
    print("%-3s %-11s %-6s %-9s %-8s %-7s %-9s %-8s %-7s %-5s" % (
        r["trial"], r["model"], r["baseline_idle"], r["lag_first_s"], r["first_zm"],
        r["first_state"], r["lag_state_s"], r["plateau_zm"], r["lag_settle_s"], r["load_err"]))

lag_first = [r["lag_first_s"] for r in rows if r["lag_first_s"] is not None]
lag_state = [r["lag_state_s"] for r in rows if r["lag_state_s"] is not None]
lag_settle = [r["lag_settle_s"] for r in rows if r["lag_settle_s"] is not None]

def stats(name, vals):
    print(f"{name}: n={len(vals)} P50={pct(vals,0.5)} P95={pct(vals,0.95)} max={max(vals) if vals else None} min={min(vals) if vals else None}")

print("\n=== AGGREGATE (all models pooled) ===")
stats("lag_first  (onset->first non-null z_m; PRIMARY, target<=35s)", lag_first)
stats("lag_state  (onset->state leaves idle)", lag_state)
stats("lag_settle (onset->z_m>=90% plateau)", lag_settle)
print("all trials baseline_idle:", all(r["baseline_idle"] for r in rows))
print("all trials 0 errors:", all(r["load_err"] == 0 for r in rows))

summary = {
    "experiment": "S1.2 authoritative W-freeze (ADR-0012, plan15 §1.2)",
    "date": "2026-07-08", "window_ms": 30000, "refresh_s": 5, "window_mode": "sliding",
    "controller_image": "tre-v2-controller:20260707-07717371", "mode": "observe",
    "load": {"workers": WORKERS if False else 20, "max_tokens": 128, "input_tokens": 64,
             "duration_s": 60, "cooldown_s": 45},
    "n_trials": len(rows), "models": ["dsqwen-7b", "dsllama-8b", "dsqwen-14b"], "rounds": 4,
    "primary_metric": "lag_first = onset -> first decision entry with z_m != null",
    "lag_first_s": {"n": len(lag_first), "p50": pct(lag_first, 0.5), "p95": pct(lag_first, 0.95),
                    "max": max(lag_first), "min": min(lag_first), "values": sorted(lag_first)},
    "lag_state_s": {"p50": pct(lag_state, 0.5), "p95": pct(lag_state, 0.95), "max": max(lag_state)},
    "lag_settle_s": {"p50": pct(lag_settle, 0.5), "p95": pct(lag_settle, 0.95), "max": max(lag_settle)},
    "target_s": 35, "all_baseline_idle": all(r["baseline_idle"] for r in rows),
    "all_zero_errors": all(r["load_err"] == 0 for r in rows),
    "rows": rows,
}
json.dump(summary, open("/tmp/s1wf_summary.json", "w"), indent=2)
print("\nSAVED /tmp/s1wf_summary.json")
