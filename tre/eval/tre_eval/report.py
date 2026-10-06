"""report.py <arm_dir> [<arm_dir> ...] --out DIR: per-arm + cross-arm evaluation report.

Read-only over the arm directories. Writes under ``--out``:
  index.html                     self-contained (figures embedded), links to every file below
  comparison.csv / .json         one row per arm x model (+ ALL): V_req, percentiles, goodput, GPU-s, ...
  paired.csv                     A - ref on the same request plan, block-bootstrap CIs; multi-seed t-CIs
  onsets.csv, decision_points.csv
  validation.json                recomputed numbers vs each arm's score.json (and --compare-json)
  cross_*.png/.pdf               CDF overlays, per-model bars with CIs, GPU-s vs V_req, onset boxes, ...
  arms/<arm>/                    requests.csv, timeseries_1s.csv, events.csv, gpu_intervals.csv,
                                 onsets.csv, summary.json, manifest.json, figures
Optional inputs that are missing only remove the parts that need them (see warnings in index.html).

Example:
  PYTHONPATH=tre/eval python3 -m tre_eval.report RUN/trace/{tre,apa,chiron} --out RUN/report \
      --capacity tre/eval/configs/capacity-v030-492x400.yaml --clock-offset node9=162
"""

from __future__ import annotations

import argparse
import base64
import csv
import datetime as _dt
import html
import json
import os
import re
import sys
from collections import Counter, defaultdict
from typing import Any

from . import load as L
from . import metrics as M
from . import timeseries as T
from . import style as S

try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None


def _num_eq(a, b, tol=1e-6) -> bool:
    if a is None or b is None:
        return a is None and b is None
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return abs(float(a) - float(b)) <= tol * max(1.0, abs(float(b)))
    return a == b


def _write_csv(path: str, rows: list[dict], fields: list[str] | None = None) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    rows = list(rows)
    if fields is None:
        fields = []
        for r in rows:
            for k in r:
                if k not in fields:
                    fields.append(k)
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: (json.dumps(v, default=str) if isinstance(v, (dict, list, tuple)) else v) for k, v in r.items()})


def _jdump(path: str, obj: Any) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as fh:
        json.dump(obj, fh, indent=1, default=lambda o: list(o) if isinstance(o, (set, tuple)) else str(o))
        fh.write("\n")


def _load_yaml(path: str | None) -> dict:
    if not path:
        return {}
    with open(path) as fh:
        return (yaml.safe_load(fh) if yaml else json.load(fh)) or {}


# ---------------------------------------------------------------- manifest
def manifest(arm: L.Arm) -> dict:
    f = arm.files
    lg = dict(re.findall(r"^(\w+)=(.*)$", f.get("loadgen_sha", ""), re.M))
    images = dict(line.split(None, 1) for line in f.get("images.txt", "").splitlines() if len(line.split(None, 1)) == 2)
    ids: dict[str, dict] = {}
    for k, v in f.items():
        mt = re.match(r"image_ids_(.+)\.txt$", k)
        if mt and v:
            node_ids = dict(line.split(None, 1) for line in v.splitlines() if len(line.split(None, 1)) == 2)
            ids[mt.group(1)] = {img: node_ids.get(img) for img in images.values()}
    model_imgs = Counter()
    for line in f.get("model_images.txt", "").splitlines():
        p = line.split()
        if len(p) >= 3:
            model_imgs[(p[1], p[2].replace("docker://", ""))] += 1
    meta = f.get("arm_meta.json") or {}
    return {
        "arm": arm.name, "label": arm.label, "trace": arm.trace_name, "dir": arm.dir,
        "load_start_epoch": arm.t_load, "load_end_epoch": arm.t_end, "load_s": (arm.t_end - arm.t_load) if arm.t_end and arm.t_load else None,
        "controller_restart_epoch": arm.t_restart, "start_iso": (f.get("start_iso") or "").strip() or None,
        "tre_sha": (f.get("tre_sha") or "").strip() or None,
        "loadgen": lg or None,
        "images": images or None, "image_ids_by_node": ids or None,
        "model_images": [{"image": a, "image_id": b, "pods": n} for (a, b), n in model_imgs.items()] or None,
        "registry_sha256": arm.registry_sha256, "traces_sha256": arm.traces_sha256,
        "trace_segments_sha256": f.get("trace_segments_sha256"),
        "policy_file": f.get("policy_file"), "policy_sha256": f.get("policy_sha256"),
        "policy_cm_sha": (f.get("policy_cm_sha") or "").strip() or None,
        "arm_meta": meta or None, "run_validity": f.get("run_validity.json"),
        "decision_source": M.decision_source(arm),
        "clock_offsets_s": arm.clock_offsets or None,
        "clock_reference": "runner host (client + sampler)",
        "component_nodes": arm.components or None,
        "emfile_pods": [x for x in (f.get("EMFILE_PODS") or "").splitlines() if x.strip()],
        "inputs_present": arm.present,
        "missing_manifest_fields": [k for k, ok in (("clock_offsets_s", bool(arm.clock_offsets)),
                                                     ("component_nodes", bool(arm.components)),
                                                     ("image_ids_all_nodes", len(ids) >= 2),
                                                     ("trace_segments", bool(arm.trace_segments))) if not ok],
    }


# ---------------------------------------------------------------- one arm
def analyse_arm(path: str, args, capacity: dict, dp_cfg: dict) -> dict:
    arm = L.load_arm(path, clock_offsets=args.clock_offset, trace_segments=args.trace_segments)
    max_tok = {t["request_id"]: t.get("max_output_tokens") for t in arm.traces}
    recs, summary = M.score_requests(arm.requests, arm.slo, trim_s=args.trim_s, max_tok=max_tok, t_ref=arm.t_load)
    if not arm.slo:
        arm.warn("no SLO block: V_req / percentiles empty")
    gpu = M.gpu_accounting(arm, summary)
    onsets = M.detect_onsets(arm) if arm.layout else []
    onset_rows = M.onset_latencies(arm, recs, onsets) if onsets else []
    dps = []
    if dp_cfg.get("decision_points") and re.search(dp_cfg.get("applies_to", "."), arm.trace_name):
        dps = M.decision_points(onset_rows, dp_cfg["decision_points"])
    inter = M.interruption(arm, recs)
    sw = M.switch_durations(arm)
    over = M.control_overhead(arm)
    ts = T.build(arm, recs, capacity)
    giv = T.gpu_intervals(arm)
    ev = T.events(arm)
    dec = [{**d, "ts_rel": d["ts"] - arm.t_load} for d in M.decisions(arm)]
    ci = {}
    for m in arm.models + [M.ALL]:
        b = M.block_bootstrap(recs, m, {"ci_vreq": M.vreq_stat, "ci_ttft_p95": M.ttft_p95_stat},
                              block_s=args.block_s, reps=args.reps, seed=args.seed)
        ci[m] = b
    # per-model continued / interrupted over all rows (analyze.py basis, trimmed rows included)
    client_counts = {}
    for m in arm.models + [M.ALL]:
        rs = [r for r in recs if m == M.ALL or r["model"] == m]
        client_counts[m] = {"continued": sum(1 for r in rs if r.get("continued")),
                            "interrupted": sum(1 for r in rs if r.get("interrupted"))}
    return {"arm": arm, "name": arm.name, "label": arm.label, "trace": arm.trace_name, "models": arm.models,
            "records": recs, "summary": summary, "gpu": gpu, "onsets": onsets, "onset_rows": onset_rows,
            "decision_points": dps, "interruption": inter, "switch": sw, "overhead": over, "ts": ts,
            "gpu_intervals": giv, "events": ev, "decisions": dec, "ci": ci, "trs": arm.trs,
            "client_counts": client_counts, "manifest": manifest(arm), "warnings": arm.warnings}


def validate(res: dict, compare_rows: list[dict] | None) -> dict:
    """Recomputed numbers vs the arm's score.json (every key) and the compare_arms.py row."""
    out = {"score_json": None, "compare_json": None, "mismatches": []}
    sc = res["arm"].files.get("score.json")
    if sc:
        n = 0
        for m, row in (sc.get("models") or {}).items():
            mine = res["summary"].get(m) or {}
            for k, v in row.items():
                n += 1
                if not _num_eq(mine.get(k), v):
                    out["mismatches"].append({"source": "score.json", "model": m, "key": k, "theirs": v, "ours": mine.get(k)})
        out["score_json"] = {"checked": n, "mismatched": sum(1 for x in out["mismatches"] if x["source"] == "score.json")}
    row = next((r for r in compare_rows or [] if os.path.basename(str(r.get("dir", "")).rstrip("/")) == res["name"]), None)
    if row:
        n = 0
        g = res["gpu"]
        for m, v in (row.get("models") or {}).items():
            mine_s = res["summary"].get(m) or {}
            pairs = {k: mine_s.get(k) for k in ("n", "fail", "V_req_pct", "ttft_p50_ms", "ttft_p95_ms", "ttft_p99_ms",
                                                 "tpot_p95_ms")}
            pairs["continued"] = res["client_counts"].get(m, {}).get("continued")
            pairs["interrupted"] = res["client_counts"].get(m, {}).get("interrupted")
            if m != M.ALL:
                pairs["layout_changes"] = (g.get(m) or {}).get("compat_layout_changes")
                pairs["replica_s"] = (g.get(m) or {}).get("compat_replica_s")
                pairs["mean_gpus"] = (g.get(m) or {}).get("compat_mean_gpus")
            for k, ours in pairs.items():
                if k not in v:
                    continue
                theirs = v[k]
                if theirs is None and k in ("continued", "interrupted"):
                    continue  # pre-unified-client rows carry None
                n += 1
                if not _num_eq(ours, theirs):
                    out["mismatches"].append({"source": "compare_json", "model": m, "key": k, "theirs": theirs, "ours": ours})
        for k, ours in (("mean_gpus", (g.get(M.ALL) or {}).get("compat_mean_gpus")),
                        ("layout_changes", (g.get(M.ALL) or {}).get("compat_layout_changes"))):
            if k in row:
                n += 1
                if not _num_eq(ours, row[k]):
                    out["mismatches"].append({"source": "compare_json", "model": "ALL", "key": k, "theirs": row[k], "ours": ours})
        out["compare_json"] = {"checked": n, "mismatched": sum(1 for x in out["mismatches"] if x["source"] == "compare_json")}
    return out


def write_arm(res: dict, out: str, multi_trace: bool) -> dict:
    from . import plots as P
    sub = f"{res['trace']}__{res['name']}" if multi_trace else res["name"]
    d = os.path.join(out, "arms", sub)
    os.makedirs(d, exist_ok=True)
    res["subdir"] = os.path.relpath(d, out)
    fields = ["request_id", "model", "t_send", "t_end", "trimmed", "ok", "ttft_ms", "tpot_ms", "tpot_v1_ms", "e2e_ms",
              "in", "out", "max_tokens", "ttft_thr_ms", "v_ttft", "v_tpot", "v_e2e", "censored", "viol", "continued",
              "interrupted", "http_status", "finish_reason", "target_pod", "send_lateness_ms"]
    _write_csv(os.path.join(d, "requests.csv"), res["records"], fields)
    _write_csv(os.path.join(d, "timeseries_1s.csv"), T.to_rows(res["ts"]), ["t", "model", *T.TS_FIELDS])
    _write_csv(os.path.join(d, "events.csv"), res["events"],
               ["t", "t_epoch", "source", "kind", "model", "target", "value", "detail"])
    giv_rows = []
    for g, ivs in res["gpu_intervals"].items():
        for iv in ivs:
            giv_rows.append({"gpu": g, "model": iv["model"], "binding": iv["binding"], "t0": iv["t0"], "t1": iv["t1"],
                             "dur_s": round(iv["t1"] - iv["t0"], 3), "hidden_s": round(sum(b - a for a, b in iv["hidden"]), 3),
                             "hidden": iv["hidden"], "open_end": iv.get("open_end", False)})
    _write_csv(os.path.join(d, "gpu_intervals.csv"), giv_rows)
    _write_csv(os.path.join(d, "onsets.csv"), res["onset_rows"])
    _write_csv(os.path.join(d, "wakes.csv"), res["switch"]["wakes"])
    summ = {"arm": res["name"], "label": res["label"], "trace": res["trace"], "summary": res["summary"],
            "ci": res["ci"], "gpu": res["gpu"], "interruption": res["interruption"],
            "switch": {k: v for k, v in res["switch"].items() if k != "wakes"}, "overhead": res["overhead"],
            "decision_points": res["decision_points"], "n_onsets": len(res["onset_rows"]),
            "n_decisions": {f"{m}:{'up' if k > 0 else 'down'}": n
                            for (m, k), n in Counter((d["model"], d["dir"]) for d in res["decisions"]).items()},
            "warnings": res["warnings"], "validation": res.get("validation")}
    _jdump(os.path.join(d, "summary.json"), summ)
    _jdump(os.path.join(d, "manifest.json"), res["manifest"])
    figs = {}
    for key, fn in (("timeseries", P.arm_timeseries), ("gpu_map", P.arm_gpu_map), ("cdfs", P.arm_cdfs),
                    ("switch", P.arm_switch), ("lengths", P.arm_lengths)):
        try:
            p = fn(res, os.path.join(d, f"fig_{key}"))
        except Exception as e:  # noqa: BLE001 - a broken optional figure must not stop the report
            res["warnings"].append(f"figure {key} failed: {type(e).__name__}: {e}")
            p = None
        if p:
            figs[key] = p
    res["figs"] = figs
    return res


# ---------------------------------------------------------------- cross arm
COMP_FIELDS = ["arm", "label", "trace", "model", "n", "fail", "V_req_pct", "V_req_ci_lo", "V_req_ci_hi", "slo_attainment_pct",
               "V_req_with_e2e_pct", "v_ttft", "v_tpot", "censored_ge_149s", "ttft_p50_ms", "ttft_p95_ms",
               "ttft_p95_ci_lo", "ttft_p95_ci_hi", "ttft_p99_ms", "tpot_p50_ms", "tpot_p95_ms", "tpot_p99_ms",
               "e2e_p50_ms", "e2e_p95_ms", "e2e_p99_ms", "ttft_mean_ms", "e2e_mean_ms", "goodput_rps",
               "good_output_tokens_per_s", "gpu_s", "routable_gpu_s", "hidden_gpu_s", "mean_gpus", "gpu_s_per_good_req",
               "max_awake", "continued", "onsets", "lat_decision_s_med", "lat_awake_s_med", "lat_routable_s_med",
               "lat_donor_s_med", "valid"]


def comparison_rows(results: list[dict]) -> list[dict]:
    import statistics as st
    rows = []
    for res in results:
        for m in res["models"] + [M.ALL]:
            s = res["summary"].get(m) or {}
            g = res["gpu"].get(m) or {}
            ci = res["ci"].get(m) or {}
            ons = [o for o in res["onset_rows"] if (m == M.ALL or o["model"] == m) and not o.get("at_start")]
            med = lambda k: (round(st.median([o[k] for o in ons if o.get(k) is not None]), 3)  # noqa: E731
                             if any(o.get(k) is not None for o in ons) else None)
            rows.append({"arm": res["name"], "label": res["label"], "trace": res["trace"], "model": m,
                         **{k: s.get(k) for k in COMP_FIELDS if k in s},
                         "V_req_ci_lo": (ci.get("ci_vreq") or (None, None))[0], "V_req_ci_hi": (ci.get("ci_vreq") or (None, None))[1],
                         "ttft_p95_ci_lo": (ci.get("ci_ttft_p95") or (None, None))[0],
                         "ttft_p95_ci_hi": (ci.get("ci_ttft_p95") or (None, None))[1],
                         **{k: g.get(k) for k in ("gpu_s", "routable_gpu_s", "hidden_gpu_s", "mean_gpus", "gpu_s_per_good_req", "max_awake")},
                         "continued": res["client_counts"].get(m, {}).get("continued"), "onsets": len(ons),
                         "lat_decision_s_med": med("lat_decision_s"), "lat_awake_s_med": med("lat_awake_s"),
                         "lat_routable_s_med": med("lat_routable_s"), "lat_donor_s_med": med("lat_donor_s"),
                         "valid": (res["arm"].files.get("score.json") or {}).get("valid")})
    return rows


def paired_rows(results: list[dict], ref_name: str, args) -> list[dict]:
    """Per (arm, seed) paired differences vs the reference arm on the same request plan; then the
    multi-seed summary (mean of per-seed differences, t-CI) per arm x model."""
    by_key = defaultdict(dict)  # arm name -> seed -> res
    for r in results:
        seed = r["arm"].traces_sha256 or r["trace"]
        by_key[r["name"]][seed] = r
    ref = by_key.get(ref_name, {})
    rows, multi = [], []
    for name, seeds in by_key.items():
        if name == ref_name:
            continue
        diffs = defaultdict(list)
        for seed, r in seeds.items():
            rr = ref.get(seed)
            if rr is None:
                continue
            for m in r["models"] + [M.ALL]:
                p = M.paired_bootstrap(r["records"], rr["records"], m, block_s=args.block_s, reps=args.reps, seed=args.seed)
                rows.append({"arm": name, "arm_label": _full_label(r), "ref": ref_name, "seed": seed[:12], "model": m, **p})
                if p.get("n_pairs"):
                    diffs[m].append(p["d_vreq_pp"])
        for m, v in diffs.items():
            multi.append({"arm": name, "ref": ref_name, "model": m, "metric": "d_vreq_pp", **M.seed_ci(v)})
    return rows, multi


def _full_label(res: dict) -> str:
    return res["label"] if res["label"].lower() == res["name"].lower() else f"{res['label']} ({res['name']})"


# ---------------------------------------------------------------- html
def _img(path: str) -> str:
    with open(path, "rb") as fh:
        b = base64.b64encode(fh.read()).decode()
    pdf = os.path.basename(path)[:-4] + ".pdf"
    return (f'<figure><img alt="{html.escape(os.path.basename(path))}" src="data:image/png;base64,{b}">'
            f'<figcaption>{html.escape(os.path.basename(path))} · PDF next to the PNG ({html.escape(pdf)})</figcaption></figure>')


def _table(rows: list[dict], fields: list[str] | None = None, max_rows: int = 400) -> str:
    if not rows:
        return "<p class=muted>none</p>"
    fields = fields or list(dict.fromkeys(k for r in rows for k in r))
    h = ["<div class=tw><table><thead><tr>" + "".join(f"<th>{html.escape(str(f))}</th>" for f in fields) + "</tr></thead><tbody>"]
    for r in rows[:max_rows]:
        cells = []
        for f in fields:
            v = r.get(f)
            if isinstance(v, float):
                v = f"{v:.3f}".rstrip("0").rstrip(".")
            elif isinstance(v, (dict, list, tuple)):
                v = json.dumps(v, default=str)
            cells.append(f"<td>{html.escape('' if v is None else str(v))}</td>")
        h.append("<tr>" + "".join(cells) + "</tr>")
    h.append("</tbody></table></div>")
    if len(rows) > max_rows:
        h.append(f"<p class=muted>{len(rows) - max_rows} more rows in the CSV</p>")
    return "".join(h)


CSS = """
:root{--bg:#fcfcfb;--ink:#0b0b0b;--ink2:#52514e;--line:#e6e5e1;--card:#ffffff;--warn:#8a5a00;--bad:#b42318}
@media (prefers-color-scheme: dark){:root:not([data-theme="light"]){--bg:#1a1a19;--ink:#ffffff;--ink2:#c3c2b7;--line:#383835;--card:#232322;--warn:#e0a640;--bad:#f07c7c}}
:root[data-theme="dark"]{--bg:#1a1a19;--ink:#ffffff;--ink2:#c3c2b7;--line:#383835;--card:#232322;--warn:#e0a640;--bad:#f07c7c}
body{background:var(--bg);color:var(--ink);font:14px/1.45 system-ui,-apple-system,Segoe UI,sans-serif;margin:0 auto;max-width:1400px;padding:16px}
h1{font-size:20px}h2{font-size:17px;border-top:1px solid var(--line);padding-top:14px;margin-top:28px}h3{font-size:15px}
.muted{color:var(--ink2)}.warn{color:var(--warn)}.bad{color:var(--bad)}
figure{margin:8px 0 18px;background:#fff;border:1px solid var(--line);border-radius:6px;padding:6px;overflow-x:auto}
figure img{max-width:100%;height:auto;display:block}figcaption{font-size:12px;color:#52514e}
.tw{overflow-x:auto;margin:6px 0 14px}table{border-collapse:collapse;font-size:12px;white-space:nowrap}
th,td{border-bottom:1px solid var(--line);padding:3px 7px;text-align:right}th{position:sticky;top:0;background:var(--card);text-align:left}
td:first-child,th:first-child{text-align:left}a{color:inherit}nav a{margin-right:10px}code{font-size:12px}
details{margin:6px 0}
"""


def write_html(out: str, results: list[dict], comp: list[dict], paired: list[dict], multi: list[dict],
               cross_figs: dict, validation: dict, args) -> str:
    now = _dt.datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
    p = [f"<!doctype html><html lang=en><head><meta charset=utf-8><meta name=viewport content='width=device-width,initial-scale=1'>"
         f"<title>TRE eval report</title><style>{CSS}</style></head><body>",
         f"<h1>TRE evaluation report</h1><p class=muted>generated {now} by tre_eval.report · {len(results)} arm(s) · "
         f"trim {args.trim_s} s · bootstrap {args.reps} reps, {args.block_s} s blocks · ref arm <code>{html.escape(args.ref or '')}</code></p>",
         "<nav>" + "".join(f"<a href='#arm-{html.escape(r['subdir'])}'>{html.escape(_full_label(r))}</a>" for r in results) + "</nav>"]
    bad = [m for v in validation.values() for m in v.get("mismatches", [])]
    p.append("<h2>Validation against existing scores</h2>")
    p.append(_table([{"arm": k, "score.json checked": (v.get("score_json") or {}).get("checked"),
                      "score.json mismatched": (v.get("score_json") or {}).get("mismatched"),
                      "compare_json checked": (v.get("compare_json") or {}).get("checked"),
                      "compare_json mismatched": (v.get("compare_json") or {}).get("mismatched")} for k, v in validation.items()]))
    if bad:
        p.append("<p class=bad>mismatches:</p>" + _table([{"arm": k, **m} for k, v in validation.items() for m in v["mismatches"]]))
    p.append("<h2>Cross-arm comparison</h2><p><a href='comparison.csv'>comparison.csv</a> · <a href='paired.csv'>paired.csv</a> · "
             "<a href='onsets.csv'>onsets.csv</a> · <a href='decision_points.csv'>decision_points.csv</a> · <a href='validation.json'>validation.json</a></p>")
    p.append(_table([r for r in comp if r["model"] == M.ALL],
                    ["arm", "label", "n", "fail", "V_req_pct", "V_req_ci_lo", "V_req_ci_hi", "ttft_p50_ms", "ttft_p95_ms",
                     "ttft_p99_ms", "tpot_p95_ms", "e2e_p95_ms", "goodput_rps", "gpu_s", "mean_gpus", "gpu_s_per_good_req",
                     "continued", "lat_decision_s_med", "lat_awake_s_med", "lat_donor_s_med", "valid"]))
    for k in ("cdfs", "bars", "scatter", "onsets", "paired", "replicas"):
        if cross_figs.get(k):
            p.append(_img(cross_figs[k]))
    p.append("<details><summary>per-model comparison table</summary>" + _table(comp, COMP_FIELDS) + "</details>")
    p.append("<h3>Paired differences vs reference (same request plan)</h3>" + _table(paired))
    if multi:
        p.append("<h3>Multi-seed (mean of per-seed differences, t 95 % CI)</h3>" + _table(multi))
    dps = [{"arm": r["name"], **d} for r in results for d in r["decision_points"]]
    if dps:
        p.append("<h3>Pre-registered decision points</h3>" + _table(dps))
    for r in results:
        sd = r["subdir"]
        p.append(f"<h2 id='arm-{html.escape(sd)}'>{html.escape(_full_label(r))} <span class=muted>· {html.escape(r['trace'])}</span></h2>")
        p.append(f"<p>files: " + " · ".join(f"<a href='{html.escape(sd)}/{f}'>{f}</a>" for f in
                 ("requests.csv", "timeseries_1s.csv", "events.csv", "gpu_intervals.csv", "onsets.csv", "wakes.csv",
                  "summary.json", "manifest.json")) + "</p>")
        if r["warnings"]:
            p.append("<details><summary class=warn>" + f"{len(r['warnings'])} warning(s)</summary><ul>" +
                     "".join(f"<li>{html.escape(w)}</li>" for w in r["warnings"]) + "</ul></details>")
        man = r["manifest"]
        p.append("<details><summary>manifest</summary>" + _table([{"key": k, "value": v} for k, v in man.items()]) + "</details>")
        p.append(_table([{"model": m, **{k: (r["summary"].get(m) or {}).get(k) for k in
                                         ("n", "fail", "V_req_pct", "slo_attainment_pct", "ttft_p50_ms", "ttft_p95_ms", "ttft_p99_ms",
                                          "tpot_p95_ms", "e2e_p95_ms", "goodput_rps")},
                          **{k: (r["gpu"].get(m) or {}).get(k) for k in ("gpu_s", "mean_gpus", "gpu_s_per_good_req")}}
                         for m in r["models"] + [M.ALL]]))
        for k in ("timeseries", "gpu_map", "cdfs", "switch", "lengths"):
            if r["figs"].get(k):
                p.append(_img(r["figs"][k]))
        p.append("<h3>Hot-onset latency breakdown</h3>" + _table(r["onset_rows"],
                 ["model", "t_on", "t_off", "at_start", "awake_at_onset", "lat_decision_s", "lat_awake_s", "lat_routable_s",
                  "lat_donor_s", "donor_model", "decision_to_awake_s", "lat_slo_ok_s", "max_awake_in_phase", "n_req_phase",
                  "V_req_phase_pct", "ttft_p95_phase_ms"]))
        inter = {k: v for k, v in r["interruption"].items() if k != "sidecar_sleep_durations_s"}
        p.append("<h3>Interruption cost</h3>" + _table([{"key": k, "value": v} for k, v in inter.items()]))
        sw = {k: v for k, v in r["switch"].items() if k != "wakes"}
        p.append("<h3>Hot-switch durations</h3>" + _table([{"key": k, "value": v} for k, v in sw.items()]))
        p.append("<h3>Control overhead</h3>" + (_table([{"component": c, **v} for c, v in r["overhead"].items()])
                                                if r["overhead"] else "<p class=muted>no profiler / resource samples for this arm</p>"))
    p.append("</body></html>")
    path = os.path.join(out, "index.html")
    with open(path, "w") as fh:
        fh.write("\n".join(p))
    return path


# ---------------------------------------------------------------- main
def _kv_float(s: str) -> tuple[str, float]:
    k, _, v = s.partition("=")
    if not k or not v:
        raise argparse.ArgumentTypeError("expected NODE=SECONDS")
    return k, float(v)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("arms", nargs="+", help="arm directories (client/performance_metrics.json inside)")
    ap.add_argument("--out", required=True, help="output directory (created)")
    ap.add_argument("--ref", help="reference arm (directory basename) for paired differences; default: first arm")
    ap.add_argument("--capacity", help="YAML with capacity_rps_per_replica: {model: req/s at ρ=1}")
    ap.add_argument("--decision-points", help="YAML with applies_to (regex on trace name) + decision_points")
    ap.add_argument("--clock-offset", type=_kv_float, action="append", default=[],
                    help="NODE=SECONDS the node's clock is ahead of the reference clock (repeatable)")
    ap.add_argument("--trace-segments", help="phase table JSON {model: [{start_time, end_time, rps}]} (trace-relative s)")
    ap.add_argument("--compare-json", help="compare_arms.py --json output to cross-check")
    ap.add_argument("--trim-s", type=float, default=30.0)
    ap.add_argument("--block-s", type=float, default=30.0, help="bootstrap block length (s)")
    ap.add_argument("--reps", type=int, default=1000, help="bootstrap replicates")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    args.clock_offset = dict(args.clock_offset)
    cap = _load_yaml(args.capacity).get("capacity_rps_per_replica") or {}
    dp_cfg = _load_yaml(args.decision_points)
    compare_rows = json.load(open(args.compare_json)) if args.compare_json else None
    os.makedirs(args.out, exist_ok=True)
    results = []
    for a in args.arms:
        try:
            res = analyse_arm(a, args, cap, dp_cfg)
        except FileNotFoundError as e:
            print(f"SKIP {a}: {e}", file=sys.stderr)
            continue
        if not cap:
            res["warnings"].append("no --capacity: ρ (offered load in replica-equivalents) not drawn")
        res["validation"] = validate(res, compare_rows)
        res["label_full"] = _full_label(res)
        results.append(res)
        v = res["validation"]
        print(f"{res['name']}: n={res['summary'].get('ALL', {}).get('n')} V_req={res['summary'].get('ALL', {}).get('V_req_pct')}% "
              f"gpu_s={res['gpu'].get('ALL', {}).get('gpu_s')} onsets={len(res['onset_rows'])} "
              f"score.json={v['score_json']} compare={v['compare_json']} warnings={len(res['warnings'])}")
    if not results:
        print("no arm with client results", file=sys.stderr)
        return 2
    args.ref = args.ref or results[0]["name"]
    multi_trace = len({r["trace"] for r in results}) > 1 and len({r["name"] for r in results}) < len(results)
    styles = S.ArmStyles([(r["name"], r["label"]) for r in results])
    for r in results:
        write_arm(r, args.out, multi_trace)
    from . import plots as P
    comp = comparison_rows(results)
    _write_csv(os.path.join(args.out, "comparison.csv"), comp, COMP_FIELDS)
    _jdump(os.path.join(args.out, "comparison.json"), comp)
    paired, multi = paired_rows(results, args.ref, args)
    _write_csv(os.path.join(args.out, "paired.csv"), paired)
    _write_csv(os.path.join(args.out, "paired_multiseed.csv"), multi)
    _write_csv(os.path.join(args.out, "onsets.csv"), [{"arm": r["name"], **o} for r in results for o in r["onset_rows"]])
    _write_csv(os.path.join(args.out, "decision_points.csv"), [{"arm": r["name"], **d} for r in results for d in r["decision_points"]])
    validation = {r["name"]: r["validation"] for r in results}
    _jdump(os.path.join(args.out, "validation.json"), validation)
    ref_res = next((r for r in results if r["name"] == args.ref), results[0])
    cross = {}
    for key, fn, extra in (("cdfs", P.cross_cdfs, ()), ("bars", P.cross_bars, ()), ("scatter", P.cross_scatter, ()),
                           ("onsets", P.cross_onsets, ()), ("replicas", P.cross_replicas, ())):
        try:
            cross[key] = fn(results, styles, os.path.join(args.out, f"cross_{key}"))
        except Exception as e:  # noqa: BLE001
            print(f"WARN cross figure {key} failed: {type(e).__name__}: {e}", file=sys.stderr)
    try:
        cross["paired"] = P.cross_paired(paired, styles, _full_label(ref_res), os.path.join(args.out, "cross_paired"))
    except Exception as e:  # noqa: BLE001
        print(f"WARN cross figure paired failed: {type(e).__name__}: {e}", file=sys.stderr)
    idx = write_html(args.out, results, comp, paired, multi, cross, validation, args)
    n_bad = sum(len(v["mismatches"]) for v in validation.values())
    print(f"report: {idx}  (validation mismatches: {n_bad})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
