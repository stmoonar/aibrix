"""Render ``REPORT.md`` in the data directory from ``manifest.json``."""

from __future__ import annotations

import json
import os


def _fmt(v) -> str:
    return "-" if v is None else (f"{v:.1f}" if isinstance(v, (int, float)) else str(v))


def write_report(out_dir: str) -> str:
    with open(os.path.join(out_dir, "manifest.json")) as f:
        m = json.load(f)
    L = ["# PreServe mLSTM predictors", ""]
    L.append(f"Created {m['created_utc']}; code {m['code'].get('sha')} "
             f"(tre_preserve/train dirty: {m['code'].get('dirty_tre_preserve_train')}); "
             f"torch {m['env']['torch']}, numpy {m['env']['numpy']}, python {m['env']['python']}.")
    L.append("")
    L.append(f"Window W = {m['window_s']} s (real time). seq_len {m['model']['seq_len']}, hidden "
             f"{m['model']['hidden']}, epochs {m['training']['epochs']}, batch {m['training']['batch_size']}, "
             f"Adam lr {m['training']['lr']}, seed {m['training']['seed']}. {m['model']['arch']}.")
    L.append("")
    L.append("## Data")
    L.append("")
    L.append("| stream | file sha256 | windows | excluded (UTC) | excluded windows | train windows | held-out windows "
             "| train sequences | held-out targets | held-out from |")
    L.append("|---|---|---|---|---|---|---|---|---|---|")
    for name, d in m["datasets"].items():
        sp = d["split"]
        ex = "; ".join(f"{e['start_utc']} – {e['end_utc']}" for e in d["exclusions"]) or "-"
        L.append(f"| {name} | `{d['sha256'][:12]}…` | {d['windows']} | {ex} | {sp['excluded_windows']} | "
                 f"{sp['train_windows']} | {sp['held_out_windows']} | {sp['train_sequences']} | "
                 f"{sp['held_out_targets']} | {sp['held_out_starts_utc']} |")
    if m.get("protocol") == "paper_past":
        L.append("")
        L.append("## Protocol paper_past (train on the past, predict the future)")
        L.append("")
        L.append("The per-stream split columns above are not used; each set's split:")
        L.append("")
        L.append("| set | stream | rule | train until (UTC) | train windows | train sequences | held-out targets |")
        L.append("|---|---|---|---|---|---|---|")
        for set_name, recs in m["set_splits"].items():
            for stream, r in recs.items():
                L.append(f"| {set_name} | {stream} | {r['rule']} | {r['train_until_utc']} | {r['train_windows']} | "
                         f"{r['train_sequences']} | {r['held_out_targets']} |")
        for d in m.get("deviations") or []:
            L.append("")
            L.append(f"Deviation: {d}.")
    L.append("")
    L.append("## Held-out MAPE (%)")
    L.append("")
    L.append(m["mape_definition"] + ".")
    L.append("")
    L.append("| set | evaluated on | normalization | prompt | response | total |")
    L.append("|---|---|---|---|---|---|")
    for set_name, by_norm in m["held_out_mape"].items():
        for norm, by_stream in by_norm.items():
            for stream, r in by_stream.items():
                tag = " (shipped)" if norm == m["normalization"]["shipped"] else ""
                L.append(f"| {set_name} | {stream} | {norm}{tag} | {_fmt(r['prompt'])} | {_fmt(r['response'])} "
                         f"| {_fmt(r['total'])} |")
    ref = m["upstream_reference_total_mape"]
    L.append("")
    L.append("Upstream reference (total MAPE, their windows): " +
             "; ".join(f"{k} 5 s {v['5s']} / 600 s {v['600s']}" for k, v in ref.items()) + ".")
    L.append("")
    L.append("numpy vs torch (shipped models, max relative diff on held-out histories): " +
             ", ".join(f"{k} {v:.1e}" for k, v in m["numpy_vs_torch_max_rel_diff"].items()) + ".")
    te = m.get("trace_eval")
    if te:
        L.append("")
        L.append("## v1 traces, 1-step-ahead MAPE (%) with loop prefill")
        L.append("")
        if m.get("trace_eval_stale"):
            L.append("**Stale: predictors were retrained after this evaluation.**")
            L.append("")
        L.append("Persistence = predict the previous window's total (cyclic); shown for scale only.")
        L.append("")
        L.append("| trace | set | model | P | D | total | persistence total | mean P | mean D |")
        L.append("|---|---|---|---|---|---|---|---|---|")
        for name, rec in te["traces"].items():
            for model, r in rec["models"].items():
                L.append(f"| {name} | {rec['set']} | {model} | {_fmt(r['prompt'])} | {_fmt(r['response'])} | "
                         f"{_fmt(r['total'])} | {_fmt(r['total_persistence_baseline'])} | {r['mean_P']:.0f} | "
                         f"{r['mean_D']:.0f} |")
        L.append("")
        L.append("Trace files (trace-stage `traces.json`; regenerating a trace changes per-window prompt sums by "
                 "<= 0.5 %, counts and max_tokens not at all):")
        for name, rec in te["traces"].items():
            L.append(f"- {name}: `{rec['traces_json']}` sha256 `{rec['traces_json_sha256'][:12]}…`, "
                     f"{rec['windows']} windows")
    L.append("")
    text = "\n".join(L)
    path = os.path.join(out_dir, "REPORT.md")
    with open(path, "w") as f:
        f.write(text)
    return path
