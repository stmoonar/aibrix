#!/usr/bin/env python3
"""Score a campaign with tre_eval.metrics (strict-completion-v1).

Only success_strict=True enters latency distributions. All requests remain in the
SLO denominator; truncated, failed and unknown-completion rows are violations.
Normal finish_reason=length is a complete response. Long E2E alone is not a cut.
--validity attaches the baseline infrastructure validity independently of request success.
"""
import argparse, json, os, sys
import yaml

ap = argparse.ArgumentParser()
ap.add_argument("src")
ap.add_argument("--registry", default=os.environ.get("SCORE_REGISTRY"),
                help="registry whose slo block (c/b, k, floor, tpot, e2e) is used (the runner passes the live "
                     "registry recorded at arm start; default: env SCORE_REGISTRY)")
ap.add_argument("--trim-s", type=float, default=30.0)
ap.add_argument("--out")
ap.add_argument("--traces", help="traces.json with per-request max_output_tokens (default: next to the metrics file)")
ap.add_argument("--validity", help="run_validity.json of a baseline arm (adds run_validity / valid)")
ap.add_argument("--arm-label", help="arm label recorded with --validity (e.g. Chiron-global)")
a = ap.parse_args()
if not a.registry:
    ap.error("--registry (or env SCORE_REGISTRY) is required")
src = a.src if a.src.endswith(".json") else os.path.join(a.src, "performance_metrics.json")
traces_path = a.traces or os.path.join(os.path.dirname(os.path.abspath(src)), "traces.json")
max_tok = {}
if os.path.exists(traces_path):
    max_tok = {t["request_id"]: t.get("max_output_tokens") for t in json.load(open(traces_path))}
reg = yaml.safe_load(open(a.registry))
slo = {m["name"]: m["slo"] for m in reg["models"]}
rows = [json.loads(l) for l in open(src) if l.strip()]
# Share the exact completion and SLO semantics with the full evaluator.
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tre_eval.metrics import score_requests

_, summary = score_requests(rows, slo, trim_s=a.trim_s, max_tok=max_tok)
res = {"source": os.path.abspath(src), "registry": a.registry, "trim_s": a.trim_s,
       "traces": traces_path if max_tok else None, "scoring_basis": "strict-completion-v1", "models": summary}
if a.validity:
    rv = json.load(open(a.validity)) if os.path.exists(a.validity) else None
    res["arm_label"] = a.arm_label
    res["run_validity"] = rv
    why = ["run_validity.json missing: " + a.validity] if rv is None else \
        ([] if rv.get("events_valid") else ["events_valid false: " + "; ".join(rv.get("invalid_because") or ["?"])])
    res["valid"] = not why
    res["invalid_because"] = why
txt = json.dumps(res, indent=1, default=int)
if a.out: open(a.out, "w").write(txt + "\n")
print(txt)
