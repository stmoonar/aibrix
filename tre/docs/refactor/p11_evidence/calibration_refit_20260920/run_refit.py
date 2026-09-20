"""Offline refit of theta_m / delta_crit / delta_high using the in-repo implementation.

Writes one artifact per model plus a combined summary, and (optionally) verifies the
four-way E1 numbers.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

_TRE_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(_TRE_ROOT / "calibration"))

from tre_calibration.dataset import load_windows_from_csv  # noqa: E402
from tre_calibration.evaluate import evaluate_signal_direction  # noqa: E402
from tre_calibration.fit import (  # noqa: E402
    DEFAULT_HEALTHY_QUANTILE_CANDIDATES,
    fit_delta_margins,
    fit_theta_by_balanced_accuracy,
    fit_theta_by_reliability,
)
from tre_calibration.profile import build_profile_patch  # noqa: E402
from tre_calibration.signals import ParameterCandidateScore  # noqa: E402

EXP = "/root/tre-experiments"
TRIM = 0
OUT = Path(__file__).resolve().parent

MODELS = [
    dict(
        model="dsqwen-7b",
        csv=f"{EXP}/r3_7b_slide_convprobe2.csv",
        slo={"ttft_p95": 500.0, "tpot_p95": 75.0, "e2e_p95": 12000.0},
        theta_live=993.4687597800112,
        w_p=0.02, lambda_wait=3.0, qmin=1.0,
    ),
    dict(
        model="dsllama-8b",
        csv=f"{EXP}/r3_llama_slide_supp.csv",
        slo={"ttft_p95": 500.0, "tpot_p95": 75.0, "e2e_p95": 12000.0},
        theta_live=1290.9145158066265,
        w_p=0.02, lambda_wait=3.0, qmin=1.0,
    ),
    dict(
        model="dsqwen-14b",
        csv=f"{EXP}/r3_14b_slide_supp3.csv",
        slo={"ttft_p95": 500.0, "tpot_p95": 75.0, "e2e_p95": 15000.0},
        theta_live=1020.2350905172023,
        w_p=0.0575, lambda_wait=3.0, qmin=1.0,
    ),
]

GENERATED_AT = "2026-09-20T00:00:00+00:00"


def sha256(path):
    d = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            d.update(chunk)
    return d.hexdigest()


def metrics_at(scores, labels, thr):
    tp = fp = tn = fn = 0
    for s, y in zip(scores, labels):
        pred = s < thr
        if y and pred:
            tp += 1
        elif y:
            fn += 1
        elif pred:
            fp += 1
        else:
            tn += 1
    rec = tp / (tp + fn) if tp + fn else 0.0
    spec = tn / (tn + fp) if tn + fp else 0.0
    prec = tp / (tp + fp) if tp + fp else 0.0
    return {"threshold": thr, "recall": rec, "precision": prec,
            "balanced_accuracy": 0.5 * (rec + spec), "n": len(scores)}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    summary = {
        "generated_at": GENERATED_AT,
        "trim_ramp_windows": TRIM,
        "theta_criterion": "balanced_accuracy",
        "delta_floor_mode": "soft",
        "min_healthy_recall": 0.0,
        "healthy_quantile_candidates": list(DEFAULT_HEALTHY_QUANTILE_CANDIDATES),
        "models": {},
    }
    for cfg in MODELS:
        windows = load_windows_from_csv(
            cfg["csv"],
            latency_slo_ms=cfg["slo"],
            signal_column="trs",
            trim_ramp_windows=TRIM,
            lambda_wait=cfg["lambda_wait"],
        )
        # comparison baseline: the containment criterion that produced the live theta
        rel = fit_theta_by_reliability(
            windows, reliability_target=0.9, min_support=3, min_confidence=0.9,
            min_scenario_families=2, max_single_scenario_ratio=0.7,
        )
        ba = fit_theta_by_balanced_accuracy(windows, min_healthy_recall=0.0)
        ba_floor = fit_theta_by_balanced_accuracy(windows, min_healthy_recall=0.90)
        delta = fit_delta_margins(windows, theta=ba.theta)
        # kept only as a comparison: the lexicographic floor that collapsed 14b's LOW band
        delta_strict = fit_delta_margins(windows, theta=ba.theta, floor_mode="strict")

        direction = evaluate_signal_direction(windows)
        score = ParameterCandidateScore(
            w_p=cfg["w_p"], lambda_wait=cfg["lambda_wait"], qmin=cfg["qmin"],
            objective=(direction.spearman_health + 1.0) / 2.0,
            spearman_health=direction.spearman_health, auroc=direction.auroc,
            scored_windows=windows,
        )
        fit_config = {
            "critical_violation_quantile": 0.65,
            "delta_floor_mode": "soft",
            "fit_delta": True,
            "healthy_quantile_candidates": list(DEFAULT_HEALTHY_QUANTILE_CANDIDATES),
            "latency_slo_ms": dict(sorted(cfg["slo"].items())),
            "max_single_scenario_ratio": 0.7,
            "min_critical_recall": 0.85,
            "min_healthy_recall": 0.0,
            "min_scenario_families": 2,
            "min_surplus_precision": 0.80,
            "signal_column": "trs",
            "surplus_latency_quantile": 0.35,
            "surplus_queue_quantile": 0.50,
            "theta_criterion": "balanced_accuracy",
            "trim_ramp_windows": TRIM,
        }
        inputs = {
            "csv_path": cfg["csv"],
            "csv_sha256": sha256(cfg["csv"]),
            "window_count": len(windows),
            "scenario_count": len({w.scenario_id for w in windows}),
            "lambda_wait_used_for_queue_raw": cfg["lambda_wait"],
            "lambda_wait_validated": False,
            "lambda_wait_note": (
                "lambda_wait is NOT identifiable from this closed-loop R3 data: avg_waiting "
                "is exactly 0 in 81.5% / 90.7% / 100.0% of windows (7b/8b/14b), so the term "
                "multiplies a zero. The inherited 3.0 is carried over unvalidated; "
                "re-deriving it needs an open-loop (fixed-rate) sweep."
            ),
        }
        patch = build_profile_patch(
            cfg["model"], theta_fit=ba, parameter_score=score,
            generated_at=GENERATED_AT, fit_config=fit_config,
            delta_fit=delta, inputs=inputs,
        )
        patch["comparison"] = {
            "delta_with_strict_floor": {
                "delta_crit": delta_strict.crit.delta,
                "delta_high": delta_strict.high.delta,
                "tau_crit": delta_strict.crit.tau,
                "tau_high": delta_strict.high.tau,
                "crit_balanced_accuracy": delta_strict.crit.balanced_accuracy,
                "crit_recall_pos": delta_strict.crit.recall_pos,
                "crit_clamped": delta_strict.crit.clamped,
                "crit_clamp_reason": delta_strict.crit.clamp_reason,
            },
            "theta_live": cfg["theta_live"],
            "theta_ratio_vs_live": ba.theta / cfg["theta_live"],
            "reliability_criterion_theta": rel.theta,
            "reliability_reproduces_live": abs(rel.theta - cfg["theta_live"]) < 1e-6,
            "balanced_accuracy_with_recall_floor_0_90": {
                "theta": ba_floor.theta,
                "healthy_quantile": ba_floor.healthy_quantile,
                "balanced_accuracy": ba_floor.balanced_accuracy,
                "recall_good": ba_floor.recall_good,
            },
        }
        (OUT / f"fit_{cfg['model']}.json").write_text(
            json.dumps(patch, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        summary["models"][cfg["model"]] = {
            "delta_crit_clamped": delta.crit.clamped,
            "delta_crit_clamp_reason": delta.crit.clamp_reason,
            "delta_crit_meets_recall_floor": delta.crit.meets_target_floor,
            "delta_crit_strict_floor": delta_strict.crit.delta,
            "delta_floor_mode": "soft",
            "delta_high_clamped": delta.high.clamped,
            "delta_high_strict_floor": delta_strict.high.delta,
            "csv_path": cfg["csv"],
            "csv_sha256": inputs["csv_sha256"],
            "window_count": len(windows),
            "scenario_count": inputs["scenario_count"],
            "theta_live": cfg["theta_live"],
            "theta_m": ba.theta,
            "theta_ratio_vs_live": ba.theta / cfg["theta_live"],
            "healthy_quantile": ba.healthy_quantile,
            "in_sample_balanced_accuracy": ba.balanced_accuracy,
            "in_sample_recall_good": ba.recall_good,
            "delta_crit": delta.crit.delta,
            "delta_high": delta.high.delta,
            "tau_crit": delta.crit.tau,
            "tau_high": delta.high.tau,
            "delta_crit_used_fallback": delta.crit.used_fallback,
            "delta_high_used_fallback": delta.high.used_fallback,
            "reliability_criterion_theta": rel.theta,
            "reliability_reproduces_live": abs(rel.theta - cfg["theta_live"]) < 1e-6,
        }
        print(
            f"{cfg['model']}: n={len(windows)} cells={inputs['scenario_count']} "
            f"theta={ba.theta:.4f} ({ba.theta / cfg['theta_live']:.3f}x live) "
            f"q={ba.healthy_quantile} BA={ba.balanced_accuracy:.4f} rec={ba.recall_good:.3f} "
            f"| d_crit={delta.crit.delta:.4f} (strict {delta_strict.crit.delta:.4f}, "
            f"clamped={delta.crit.clamped}) d_high={delta.high.delta:.4f} "
            f"| reliability={rel.theta:.6f} repro={abs(rel.theta - cfg['theta_live']) < 1e-6}"
        )

    # ---- E1 four-way check -------------------------------------------------
    e1_path = Path("/tmp/theta_forensics/windows.json")
    if e1_path.exists():
        e1 = json.loads(e1_path.read_text())
        by_model = defaultdict(list)
        for r in e1:
            by_model[r["model"]].append(r)
        pooled_z, pooled_y, pooled_m = [], [], []
        pooled_zlive = []
        per_model = {}
        for cfg in MODELS:
            rows = by_model.get(cfg["model"], [])
            if not rows:
                continue
            theta = summary["models"][cfg["model"]]["theta_m"]
            y = [r["vrate"] > 0.10 for r in rows]
            z = [r["trs"] / theta for r in rows]
            zlive = [r["trs"] / cfg["theta_live"] for r in rows]
            per_model[cfg["model"]] = metrics_at(z, y, 1.0)
            pooled_z += z
            pooled_y += y
            pooled_m += [cfg["model"]] * len(rows)
            pooled_zlive += zlive
        pooled = metrics_at(pooled_z, pooled_y, 1.0)
        deployed = metrics_at(pooled_zlive, pooled_y, 0.8)
        summary["e1_validation"] = {
            "source": str(e1_path),
            "label": "vrate > 0.10",
            "window_count": len(pooled_z),
            "refit_shared_threshold_Z_lt_1": pooled,
            "refit_per_model_at_Z_lt_1": per_model,
            "deployed_shared_threshold_Z_lt_0_8": deployed,
        }
        print(f"POOLED refit Z<1: n={pooled['n']} BA={pooled['balanced_accuracy']:.4f} "
              f"rec={pooled['recall']:.4f} prec={pooled['precision']:.4f}")
        for m, v in per_model.items():
            print(f"   {m}: BA={v['balanced_accuracy']:.4f} rec={v['recall']:.4f}")
        print(f"POOLED deployed Z<0.8: BA={deployed['balanced_accuracy']:.4f} "
              f"rec={deployed['recall']:.4f}")
    else:
        print("E1 windows.json absent; skipping validation block")

    summary["repo_head"] = subprocess.run(
        ["git", "-C", str(_TRE_ROOT), "rev-parse", "HEAD"],
        capture_output=True, text=True,
    ).stdout.strip()
    (OUT / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
