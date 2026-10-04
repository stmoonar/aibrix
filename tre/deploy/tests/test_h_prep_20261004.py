"""Stage H preparation (2026-10-04, DRAFT tools): the zero-token drop audit, the
conservative variant and the T14 scorer's censoring audit / refusals, on tiny fixtures."""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import pytest

from scripts import b_prime
from scripts import dline_refit as dl
from scripts import theta_verdict as tv
from scripts.analysis import h_conservative_score as hcs
from scripts.analysis import h_dropped_windows as hd
from scripts.analysis import t14_score as t14
from tre_common import slo_labels

COLUMNS = ["model", "shape", "primitive", "role", "stage", "cell_id", "attempt", "split", "cell_status", "in_warmup", "scenario_id",
           "scenario_family", "window_start_ms", "window_end_ms", "prompt_tokens_total", "generation_tokens_total",
           "avg_waiting", "avg_running", "completed_requests", "p95_ttft_client_ms", "p95_tpot_client_ms",
           "p95_e2e_client_ms", "model_errors", "proxy_transient_errors", "client_timeouts"]
LABEL = slo_labels.LabelDefinition(ttft_p95_ms=500.0, tpot_p95_ms=75.0)   # fixed 500 / 75, min-n 20
TAU_CRIT = 0.73


def _rows(cell: str, *, n: int, gen: float, running: float, tpot: float, start: int = 0, **over) -> list[dict]:
    out = []
    for k in range(n):
        s = 1_790_000_000_000 + start + 10_000 * k
        out.append({"model": "m", "shape": "S1", "primitive": "hold", "role": "ladder", "stage": "ladder",
                    "cell_id": cell, "attempt": 1, "split": "train",
                    "cell_status": "valid", "in_warmup": "False", "scenario_id": cell, "scenario_family": "f",
                    "window_start_ms": s, "window_end_ms": s + 30_000, "prompt_tokens_total": gen,
                    "generation_tokens_total": gen, "avg_waiting": 0.0, "avg_running": running,
                    "completed_requests": 40, "p95_ttft_client_ms": 200.0, "p95_tpot_client_ms": tpot,
                    "p95_e2e_client_ms": 1.0, "model_errors": 0, "proxy_transient_errors": 0,
                    "client_timeouts": 0, **over})
    return out


def _stall(cell: str, start: int, *, errors: int, running: float = 40.0, **over) -> dict:
    """A window in which nothing completed: 0 tokens, no latency sample."""
    return _rows(cell, n=1, gen=0.0, running=running, tpot=0.0, start=start, completed_requests=0,
                 p95_ttft_client_ms="", p95_tpot_client_ms="", model_errors=errors, **over)[0]


def _write(path: Path, rows: list[dict]) -> Path:
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)
    return path


def _world(tmp_path: Path) -> tuple[dict, Path]:
    """Two healthy cells (TSS 3000), two violating ones (TSS 75, TPOT 100 ms); the second
    violating cell then stalls: one zero-token window with a model error, one with a
    backlog only. theta sits between the two levels."""
    rows = (_rows("h1", n=6, gen=30_000, running=10, tpot=30) + _rows("h2", n=6, gen=30_000, running=10, tpot=30)
            + _rows("v1", n=6, gen=3_000, running=40, tpot=100)
            + _rows("v2", n=6, gen=3_000, running=40, tpot=100)
            + [_stall("v2", 60_000, errors=1), _stall("v2", 70_000, errors=0)])
    path = _write(tmp_path / "w.csv", rows)
    spec = dl.spec_for(10.0, 0.0, 1.0)
    theta = math.sqrt(75.0 * 3000.0)
    vh = {"model": "m", "signal_spec": spec.as_dict(), "label_def": LABEL.as_dict(), "trim_ramp_windows": 1,
          "fit_config": {"direction": "higher_is_healthier"}, "merged": {"theta": theta},
          "published": {"theta_m": theta, "tau_crit": TAU_CRIT}}
    entry = {"verdict_for_holdout": vh, "train_ba_at_published": 1.0, "stop_rule": {"satisfied": True},
             "windowing": {"dwell_windows": 2, "window_ms": 30_000.0}}
    return entry, path


def _cfg(cut: float) -> dict:
    return {"dwell_windows": 1, "severity_cut": cut, "gate": dict(b_prime.DEFAULT_GATE)}


def test_audit_counts_the_zero_token_drops_by_evidence_and_skips_warmup_invalid_and_holdout(tmp_path) -> None:
    rows = (_rows("a", n=3, gen=100, running=5, tpot=30)
            + [_stall("a", 30_000, errors=2),                       # backlog + failure: unserved-violated
               _stall("a", 40_000, errors=0),                       # backlog only: label None (low n)
               _stall("a", 50_000, errors=0, running=0.0),          # idle: no backlog, no failure
               _stall("a", 60_000, errors=3, in_warmup="True"),     # warm-up: not counted
               _stall("a", 70_000, errors=3, cell_status="inconclusive"),
               _stall("a", 80_000, errors=3, split="holdout")]      # M / H2 sealed: not read
            + _rows("a", n=1, gen=100, running=5, tpot=30, start=90_000, completed_requests=5))  # low-n, backlog
    entry, _ = _world(tmp_path)
    a = hd.audit_csv(_write(tmp_path / "a.csv", rows), entry, model="m", sealed_to_h2=False)
    t = a["total"]
    assert (t["kept"], t["filtered"], t["invalid_cell"], t["holdout_not_read"]) == (3, 1, 1, 1)
    assert t["invalid_cell_zero_token"] == 1
    z = t["zero_token"]
    assert (z["total"], z["with_backlog"], z["with_failure_evidence"], z["unserved_violated"],
            z["backlog_or_failure"], z["label_unlabeled"]) == (3, 2, 1, 1, 2, 2)
    assert t["unlabeled"]["low_n_with_backlog"] == 1
    assert a["by_set"][dl.SET_TRAINING]["zero_token"]["total"] == 3
    # the loader really drops all three (the rule audited is the loader's own)
    spec, label, _ = hd.frozen_spec_and_label(entry)
    assert len(spec.load(tmp_path / "a.csv", label, 0)) == 3


def test_phantoms_are_violating_never_critical_and_skip_the_ramp_trim(tmp_path) -> None:
    entry, path = _world(tmp_path)
    spec, label, trim = hd.frozen_spec_and_label(entry)
    theta = entry["verdict_for_holdout"]["published"]["theta_m"]
    rows = hd.read_rows(path)
    # a stalled FIRST window of a new cell is an onset window: the trim drops it, never added
    rows.insert(0, {**_stall("v3", 0, errors=1)})
    ph, tiers, added = hd.phantom_windows(rows, hd.row_signals(rows, spec), label, theta=theta, trim=trim)
    assert tiers == [hd.TIER_FAILURE, hd.TIER_BACKLOG_ONLY] and added["not_added_ramp_trim"] == 1
    assert all(not w.slo_met and w.violation_class == "unserved" for w in ph)
    assert all(w.latency_ratio_p95 == slo_labels.UNSERVED_MIN_RATIO for w in ph)
    flags = tv.critical_dwell_flags(ph, theta=theta, tau_crit=TAU_CRIT, direction="higher_is_healthier",
                                    dwell_windows=1)
    assert flags == [False, False] and all(w.signal / theta >= 1.0 for w in ph)


def test_conservative_variant_counts_stalls_as_misses_and_frozen_equals_the_accept_path(tmp_path) -> None:
    entry, path = _world(tmp_path)
    r = hcs.score_model(entry, path, b_prime_cfg=_cfg(1.2), n_resamples=30, seed=dl.SEED)
    assert r["accept_path_check"] == {"matches_evaluate_model": True, "differences": []}
    f, cf, c = (r["variants"][k] for k in hcs.VARIANTS)
    # 4 cells x 5 windows (first trimmed); 10 violating, every one CRITICAL
    assert (f["windows"], f["violating"], f["A_ba"], f["recall_all_violations_gate_dwell"]) == (20, 10, 1.0, 1.0)
    assert (cf["windows"], cf["violating"]) == (21, 11) and (c["windows"], c["violating"]) == (22, 12)
    assert c["A_ba"] == pytest.approx((1.0 + 10 / 12) / 2)
    assert c["recall_all_violations_gate_dwell"] == pytest.approx(10 / 12)
    # severity 2.0 >= cut 1.2: the stalls are severe and B' misses them too ...
    assert c["B_prime_recall_severe"] == pytest.approx(10 / 12)
    # ... with the cut above 2.0 (this freeze: 8.8-12.6) B' cannot see them at all
    hi = hcs.score_model(entry, path, b_prime_cfg=_cfg(1.3), n_resamples=30, seed=dl.SEED, check_accept_path=False)
    assert hi["variants"]["conservative"]["B_prime_severe_windows"] == 12
    hi = hcs.score_model(entry, path, b_prime_cfg=_cfg(2.5), n_resamples=30, seed=dl.SEED, check_accept_path=False)
    assert hi["variants"]["conservative"]["B_prime_severe_windows"] == hi["variants"]["frozen"]["B_prime_severe_windows"] == 0


def test_t14_censoring_audit_cuts_only_at_the_route_timeout(tmp_path) -> None:
    head = ["cell_id", "attempt", "outcome", "e2e_ms"]
    reqs = ([("c1", 1, "ok", 1000.0)] * 90 + [("c1", 1, "model_error", 150_000.0)] * 6
            + [("c1", 1, "model_error", 149_999.0)] * 2 + [("c1", 1, "model_error", "")] * 2
            + [("c1", 2, "model_error", 1.0)] * 50                     # another attempt: not read
            + [("c2", 1, "ok", 1.0)] * 94 + [("c2", 1, "model_error", 1.0)] * 6)
    p = tmp_path / "requests.csv"
    with p.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(head)
        w.writerows(reqs)
    cells = {("c1", 1): {"shape": "G512x256", "kind": "interpolation"},
             ("c2", 1): {"shape": "G3072x96", "kind": "extrapolation"}}
    a = t14.censoring_audit(p, cells)
    c1, c2 = a["cells"]
    assert (c1["sent"], c1["model_error"], c1["cut"], c1["non_cut"], c1["non_cut_no_e2e"]) == (100, 10, 6, 4, 2)
    assert c1["non_cut_rate"] == 0.04 and not c1["void_at_audit"] and not c1["runtime_limit_exceeded"]
    assert c2["non_cut_rate"] == 0.06 and c2["void_at_audit"]
    assert a["void_at_audit"] == ["c2"]


def test_t14_scorer_refuses_a_changed_prereg_and_writes_nothing(tmp_path, capsys) -> None:
    pre = tmp_path / "preregistration.json"
    pre.write_text(json.dumps({"t14": {"model": "m"}, "evaluation": {}}))
    (tmp_path / "preregistration.json.sha256").write_text("0" * 64 + "  preregistration.json\n")
    add = tmp_path / "ADDENDUM.json"
    add.write_text(json.dumps({"amends": {"sha256": "x"}, "audit_rule": {}}))
    out = tmp_path / "score.json"
    rc = t14.main(["--prereg", str(pre), "--addendum", str(add), "--freeze-file", str(tmp_path / "nofreeze.json"),
                   "--dry-run-dataset", str(tmp_path), "--out", str(out)])
    text = capsys.readouterr().out
    assert rc == 1 and "REFUSED" in text
    assert "!= its sidecar" in text and "amends" in text and "cross" in text   # every problem listed
    assert not out.exists() and not Path(f"{out}.d").exists()


def _win(cell: str, k: int, *, signal: float, ok: bool):
    from tre_calibration.dataset import CalibrationWindow

    return CalibrationWindow(scenario_id=cell, scenario_family="f", signal=signal, slo_met=ok,
                             health_score=0.5, window_start_ms=1_790_000_000_000 + 10_000 * k,
                             latency_ratio_p95=0.5 if ok else 1.5, violation_class=None if ok else "tpot_only")


def test_t14_single_class_shape_makes_the_claim_not_evaluable_and_is_never_dropped(tmp_path) -> None:
    entry, _ = _world(tmp_path)
    theta = entry["verdict_for_holdout"]["published"]["theta_m"]
    windows, shape_of = [], {}
    for shape, cells in (("GA", ("a1", "a2")), ("GC", ("c1", "c2")), ("GB", ("b1", "b2"))):
        for n, cell in enumerate(cells):
            shape_of[cell] = shape
            bad = shape != "GB" and n == 1           # GB: healthy windows only
            windows += [_win(cell, k, signal=(0.1 if bad else 5.0) * theta, ok=not bad) for k in range(4)]
    kind_of = {"GA": "interpolation", "GB": "interpolation", "GC": "interpolation"}
    out = t14.cross_shape(entry, windows, shape_of, kind_of, n_resamples=20, seed=dl.SEED)["interpolation"]
    gb = out["per_shape"]["GB"]
    assert gb["single_class"] and gb["ba"] is None and gb["auroc"] is None
    assert out["per_shape"]["GA"]["ba"] == 1.0 and out["per_shape"]["GA"]["auroc"] == 1.0
    c = out["claim"]
    # GA and GC alone would claim (SD 0 <= half width 0): the single-class shape is not dropped
    assert c["one_theta_transfers"] == t14.CLAIM_NOT_EVALUABLE and c["single_class_shapes"] == ["GB"]
    assert c["disclosure_only_over_remaining_shapes"]["shapes"] == 2 and "sd_sample" not in c
    full = t14.claim({s: t for s, t in out["per_shape"].items() if s != "GB"})
    assert full["one_theta_transfers"] is True                  # the rule itself works on two-class shapes
    assert out["pooled_disclosure"]["ba"] == 1.0                # pooled numbers keep every window


def test_t14_void_at_audit_follows_the_prereg_void_rule_per_cell() -> None:
    def audit(*cells):
        return {"cells": [{"cell_id": c, "attempt": a, "void_at_audit": v} for c, a, v in cells]}

    assert t14.void_status(audit(("c1", 1, False), ("c2", 2, False)))["status"] == t14.STATUS_EVALUATED
    one = t14.void_status(audit(("c1", 1, True), ("c2", 1, False)))
    assert one["status"] == "void_redrive_required:c1" and one["status_if_two_void_cells_stop_the_run"] == one["status"]
    # a re-driven attempt void at audit is that cell's second void: the run is stopped
    assert t14.void_status(audit(("c1", 2, True), ("c2", 1, True)))["status"] == t14.STATUS_RUN_VOID
    # two cells void on their first attempt: each re-driven once (prereg, per cell); the
    # "two void cells stop the run" reading is reported, not applied
    two = t14.void_status(audit(("c1", 1, True), ("c3", 1, True)))
    assert two["status"] == "void_redrive_required:c1,c3"
    assert two["status_if_two_void_cells_stop_the_run"] == t14.STATUS_RUN_VOID
    assert "second void stops the run" in two["void_rule"]
