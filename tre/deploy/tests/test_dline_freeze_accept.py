"""D22 parameter freeze and the one-shot plan §6.9f A-D acceptance of scripts.dline_refit
(``freeze`` / ``verify-freeze`` / ``accept``)."""
from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import stat
from pathlib import Path

import pytest

from scripts import dline_refit as dl
from scripts import r3_grid

TRE_ROOT = Path(__file__).resolve().parents[2]
REGISTRY = TRE_ROOT / "deploy" / "registry.yaml"
MODEL = "dsqwen-7b"
ARM = "fixed"          # the fixed 500 / 75 ms label: the windows' outcome is set by two p95s
TAU_CRIT = 0.73
TRAIN_BA = 0.9
BOOT = 40

_IDENTITY = ["model", "shape", "primitive", "stage", "rho", "cell_id", "attempt", "split", "cell_status", "role",
             "rho_factor", "replicate", "possibly_contaminated", "in_warmup"]
COLUMNS = _IDENTITY + [c for c in r3_grid.CSV_COLUMNS if c not in _IDENTITY]
_SHAPE_IO = {"S1": (256, 128), "S3": (2048, 96), "S4": (256, 448), "S5": (768, 384), "M": (0, 0)}


def _cell_rows(code: int, *, shape: str, split: str, role: str, stage: str, signal: str = "hi",
               ttft: float = 200.0, tpot: float = 30.0, status: str = "valid", n: int = 12,
               primitive: str = "hold") -> list[dict]:
    """One constant-load cell: ``signal`` hi (high TSS) or lo (40x lower); the window label
    is set by ``ttft`` / ``tpot`` against the fixed 500 / 75 ms SLO."""
    i, o = _SHAPE_IO[shape]
    sid = f"i{i}_o{o}_c{code}"
    gen, running = (30_000.0, 10.0) if signal == "hi" else (3_000.0, 40.0)
    start = 1_790_000_000_000 + code * 1_000_000
    rows = []
    for k in range(n):
        s = start + 10_000 * k
        rows.append({
            "model": MODEL, "shape": shape, "primitive": primitive, "stage": stage, "rho": 1.0,
            "cell_id": sid, "attempt": 1, "split": split, "cell_status": status, "role": role,
            "rho_factor": "", "replicate": "", "possibly_contaminated": "False", "in_warmup": "False",
            "scenario_id": sid, "scenario_family": f"i{i}_o{o}", "input_tokens": i, "output_tokens": o,
            "concurrency": code, "window_start_ms": s, "window_end_ms": s + 30_000,
            "prompt_tokens_total": gen, "generation_tokens_total": gen, "avg_waiting": 0.0,
            "avg_running": running, "avg_swapping": 0.0, "queue_control": running, "trs": "",
            "completed_requests": 40, "p95_ttft_client_ms": ttft, "p95_tpot_client_ms": tpot,
            "p95_e2e_client_ms": 1.0, "model_errors": 0, "proxy_transient_errors": 0, "client_timeouts": 0,
        })
    return rows


def _write_dataset(d: Path, rows: list[dict]) -> Path:
    d.mkdir(parents=True, exist_ok=True)
    with (d / "windows.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    (d / "manifest.json").write_text(json.dumps({"format_revision": 2}))
    return d


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _dump(path: Path, doc) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc, indent=1))


#: The M cells: (code, signal, ttft, tpot, origin). Healthy cells sit far above theta, the
#: violating ones far below theta * tau_crit, so every outcome below is known exactly.
_HEALTHY = [(3_000_001 + k, "hi", 200.0, 30.0) for k in range(6)]
_TPOT = [(3_000_101 + k, "lo", 200.0, 100.0) for k in range(5)]
_TTFT_ONLY = [(3_000_201, "lo", 900.0, 30.0)]
_PROBE = 3_000_301


def _world(tmp_path: Path, *, stop_ok: bool = True, holdout_evaluated: bool = False,
           false_alarm_cells: int = 0) -> dict:
    """A trainset fit dir, a refit tree (the four stage outputs of one model) and an M
    dataset. ``false_alarm_cells`` healthy M cells carry a low signal (CRITICAL false alarms)."""
    # training set, cut by the trainset stage from a small dataset
    train = []
    for n, shape in enumerate(("S1", "S3", "S4", "S5", "S3", "S4")):
        train += _cell_rows(1_100_000 + n, shape=shape, split="train", role="ladder", stage="ladder")
    _write_dataset(tmp_path / "run2" / "dataset", train)
    fit = tmp_path / "fit"
    assert dl.main(["trainset", "--fit-dir", str(fit), "--h2-dataset", str(tmp_path / "run2")]) == 0

    # M: the evaluated cells, one sealed probe and one training row in the same dataset
    m_rows, cells = [], []
    specs = [(c, "lo" if k < false_alarm_cells else s, t, p) for k, (c, s, t, p) in enumerate(_HEALTHY)]
    for code, sig, ttft, tpot in specs + _TPOT + _TTFT_ONLY:
        m_rows += _cell_rows(code, shape="M", split="holdout", role="ladder", stage="ladder", signal=sig,
                             ttft=ttft, tpot=tpot)
        cells.append({"model": MODEL, "cell_id": f"i0_o0_c{code}", "attempt": 1, "shape": "M",
                      "primitive": "hold", "role": "ladder",
                      "origin": "retained" if code == _HEALTHY[0][0] else "collected",
                      "seen_before": code == _HEALTHY[0][0],
                      "note": "first-round M cell, looked at once by the 09-22 refit" if code == _HEALTHY[0][0] else ""})
    m_rows += _cell_rows(_PROBE, shape="M", split="holdout", role="boundary", stage="bisect", signal="lo",
                         tpot=30.0)
    m_rows += _cell_rows(1_200_000, shape="S1", split="train", role="ladder", stage="ladder", signal="lo")
    mdir = _write_dataset(tmp_path / "mrun" / "dataset", m_rows)

    # the refit tree: theta between the two signal levels (read off the M rows here, in the
    # test only - it pins the geometry, it is not how a theta is fitted)
    label = dl.label_for(MODEL, ARM, str(REGISTRY))
    spec = dl.spec_for(10.0, 0.0, 1.0)
    ws = spec.load(mdir / "windows.csv", label, dl.TRIM_RAMP_WINDOWS)
    lo = max(w.signal for w in ws if w.scenario_id.endswith(str(_TPOT[0][0])))
    hi = min(w.signal for w in ws if w.scenario_id.endswith(str(_HEALTHY[-1][0])))
    theta = math.sqrt(lo * hi)
    assert lo < 0.5 * TAU_CRIT * theta and hi > 2.0 * theta
    stop = ({"satisfied": True, "reasons": [], "hold_cells": [], "ci_half_width_fraction": 0.1}
            if stop_ok else
            {"satisfied": False, "reasons": ["CI half width 82.58 >= 15% of theta (75.77)",
                                             "family 'prefill_heavy' has 12 window(s) near the boundary, needs >= 30"],
             "hold_cells": [], "ci_half_width_fraction": 0.16})
    published = {"signal": "tss", "direction": "higher_is_healthier", "theta_m": theta, "tau_crit": TAU_CRIT,
                 "delta_crit": 1 - TAU_CRIT, "tau_high": 1.59, "delta_high": 0.59, "source": "merged",
                 "stop_rule_satisfied": stop_ok, "family_rule_theta": theta, "d5_merged_published": True}
    verdict = {"model": MODEL, "signal": "tss", "direction": "higher_is_healthier", "signal_spec": spec.as_dict(),
               "label_def": label.as_dict(), "trim_ramp_windows": dl.TRIM_RAMP_WINDOWS,
               "fit_config": {"direction": "higher_is_healthier", "theta_criterion": "balanced_accuracy"},
               "merged": {"theta": theta, "windows": 99,
                          "opposite_direction": {"direction": "lower_is_healthier", "publish": True,
                                                 "theta": 3 * theta, "fit": {}}},
               "stop_rule": stop, "published": published}
    final = {"model": MODEL, "arm": ARM, "tau_s": 10.0, "alpha": dl.alpha_of(10.0), "w_p": 0.0, "lambda_wait": 1.0,
             "theta_merged": theta, "theta_published": theta, "source": "merged", "train_ba": TRAIN_BA,
             "ci_half_frac": stop["ci_half_width_fraction"], "publish_rate": 1.0, "theta_P": 1.05 * theta,
             "theta_D": 0.97 * theta, "family_gap_frac": 0.08, "delta_crit": 1 - TAU_CRIT, "delta_high": 0.59,
             "tau_crit": TAU_CRIT, "stop_rule": stop, "stop_rule_15": stop_ok, "theta_family_rule": theta,
             "train_ba_at_published": TRAIN_BA, "holdout_evaluated": holdout_evaluated,
             "label_def": label.as_dict(), "training_set": dl.check_training_inputs(MODEL, dl.paths(fit, MODEL))}
    out = tmp_path / "refit"
    d = out / MODEL / ARM
    _dump(d / "alpha.json", dl.publish_alpha({"rule": "d4prime", "chosen_tau_s": 10.0, "alpha_fit": {"curve": []}},
                                             10.0))
    _dump(d / "wp.json", {"model": MODEL, "tau_s": 10.0, "w_p_star": 0.0, "w_p_used": 0.0, "lambda_star": 1.0})
    _dump(d / "final.json", final)
    _dump(d / "verdict_final.json", verdict)
    return {"fit": fit, "out": out, "mdir": mdir, "cells": cells, "theta": theta, "label": label.as_dict(),
            "freeze": tmp_path / "freeze" / "params_freeze.json", "mroot": tmp_path / "M"}


def _freeze(w: dict) -> int:
    return dl.main(["freeze", "--model", MODEL, "--arm", ARM, "--fit-dir", str(w["fit"]), "--out-dir", str(w["out"]),
                    "--freeze-file", str(w["freeze"])])


def _seal(w: dict, *, label_def=None, freeze_sha=None, cells=None, sums_extra: list[Path] = ()) -> Path:
    """The collection side's M manifest for ``MODEL`` (its contract, written by hand)."""
    mdir = w["mroot"] / MODEL
    mdir.mkdir(parents=True, exist_ok=True)
    sums = mdir / "SHA256SUMS"
    files = [w["mdir"] / "windows.csv", *sums_extra]
    sums.write_text("".join(f"{_sha(f)}  {f}\n" for f in files))
    label = w["label"] if label_def is None else label_def
    man = {"model": MODEL, "format_revision": 1,
           "freeze": {"path": str(w["freeze"]), "sha256": freeze_sha or _sha(w["freeze"])},
           "label_def": label, "label_def_sha256": dl.canonical_sha256(label),
           "cells": w["cells"] if cells is None else cells,
           "sealed_probes": [{"model": MODEL, "cell_id": f"i0_o0_c{_PROBE}", "attempt": 1}],
           "sha256sums_file": sums.name, "sha256sums_sha256": _sha(sums)}
    path = mdir / "M_manifest.json"
    if path.exists():
        path.unlink()
    path.write_text(json.dumps(man, indent=1))
    return path


def _accept(w: dict, man: Path, *extra: str) -> int:
    return dl.main(["accept", "--freeze-file", str(w["freeze"]), "--dataset", f"mrun={w['mdir']}",
                    "--m-manifest", str(man), "--accept-resamples", str(BOOT), *extra])


def _written(w: dict) -> list[str]:
    return sorted(p.name for p in w["freeze"].parent.iterdir()) if w["freeze"].parent.exists() else []


# ------------------------------------------------------------------------- freeze


def test_canonical_sha256_is_the_sorted_compact_utf8_json() -> None:
    obj = {"b": [1, 2.5, "é"], "a": {"y": None, "x": True}}
    raw = '{"a":{"x":true,"y":null},"b":[1,2.5,"é"]}'.encode("utf-8")
    assert dl.canonical_sha256(obj) == hashlib.sha256(raw).hexdigest()


def test_freeze_writes_one_self_hashed_read_only_document(tmp_path) -> None:
    w = _world(tmp_path)
    assert _freeze(w) == 0
    ff = w["freeze"]
    assert _written(w) == ["params_freeze.json", "params_freeze.json.sha256"]
    assert stat.S_IMODE(ff.stat().st_mode) == 0o444
    assert (ff.parent / "params_freeze.json.sha256").read_text() == f"{_sha(ff)}  params_freeze.json\n"
    doc = dl.verify_freeze(ff)
    assert doc["freeze_sha256"] == dl.canonical_sha256({k: v for k, v in doc.items() if k != "freeze_sha256"})
    assert doc["arm"] == ARM and doc["code"]["commit"] and doc["command"][:3] == ["python", "-m", "scripts.dline_refit"]
    e = doc["models"][MODEL]
    assert e["published"] == {"signal": "tss", "direction": "higher_is_healthier", "theta": w["theta"], "w_p": 0.0,
                              "tau_s": 10.0, "alpha": dl.alpha_of(10.0), "lambda_wait": 1.0,
                              "delta_crit": pytest.approx(1 - TAU_CRIT), "delta_high": 0.59, "tau_crit": TAU_CRIT,
                              "tau_high": 1.59}
    assert e["registry"] == {"ema_tau_ms": 10_000.0, "ema_alpha": round(dl.alpha_of(10.0), 6)}
    assert e["stop_rule"]["satisfied"] and e["train_ba_at_published"] == TRAIN_BA
    vh = e["verdict_for_holdout"]
    assert set(vh) == {"model", "signal", "label_def", "signal_spec", "trim_ramp_windows", "fit_config",
                       "published", "merged"}
    assert vh["merged"]["opposite_direction"] == {"direction": "lower_is_healthier", "publish": True,
                                                  "theta": 3 * w["theta"]}
    assert e["label_def_sha256"] == dl.canonical_sha256(w["label"])
    assert set(e["stage_files"]) == {"alpha", "wp", "final", "verdict_final"}
    assert set(e["training_inputs"]) == {"fitting", "family_decode_heavy", "family_prefill_heavy",
                                         "training_ledger", "trainset_manifest", "h2_manifest"}
    assert e["training_inputs"]["fitting"]["sha256"] == _sha(w["fit"] / f"{MODEL}_fitting.csv")
    h2 = json.loads((w["fit"] / dl.H2_MANIFEST).read_text())
    assert e["h2"] == {"rows_sha256": h2["rows_sha256"], "cells_sha256": h2["cells_sha256"],
                       "manifest_sha256": _sha(w["fit"] / dl.H2_MANIFEST)}
    assert dl.main(["verify-freeze", "--freeze-file", str(ff)]) == 0


def test_freeze_refuses_an_unsatisfied_stop_rule_and_lists_the_reasons(tmp_path, capsys) -> None:
    w = _world(tmp_path, stop_ok=False)
    assert _freeze(w) == dl.EXIT_REFUSED
    out = capsys.readouterr().out
    assert "REFUSED" in out and "D13 stop rule not satisfied" in out
    assert "CI half width 82.58 >= 15% of theta (75.77)" in out and "needs >= 30" in out
    assert _written(w) == []


def test_freeze_refuses_a_final_that_read_m(tmp_path, capsys) -> None:
    w = _world(tmp_path, holdout_evaluated=True)
    assert _freeze(w) == dl.EXIT_REFUSED
    assert "M was read before the freeze" in capsys.readouterr().out
    assert _written(w) == []


def test_freeze_lists_every_problem_of_every_model(tmp_path, capsys) -> None:
    w = _world(tmp_path, stop_ok=False, holdout_evaluated=True)
    rc = dl.main(["freeze", "--model", MODEL, "--model", "dsqwen-14b", "--arm", ARM, "--fit-dir", str(w["fit"]),
                  "--out-dir", str(w["out"]), "--freeze-file", str(w["freeze"])])
    out = capsys.readouterr().out
    assert rc == dl.EXIT_REFUSED
    assert f"{MODEL}: D13" in out and f"{MODEL}: final ran with the hold-out" in out
    assert "dsqwen-14b: " in out and "final.json is missing" in out


def test_freeze_refuses_training_inputs_changed_since_final(tmp_path, capsys) -> None:
    w = _world(tmp_path)
    fitting = w["fit"] / f"{MODEL}_fitting.csv"
    with fitting.open("a") as fh:
        fh.write(fitting.read_text().splitlines()[-1] + "\n")
    assert _freeze(w) == dl.EXIT_REFUSED
    assert "not the file the trainset stage wrote" in capsys.readouterr().out
    # a fit dir other than the one final ran on is refused by the trainset manifest hash
    w2 = _world(tmp_path / "other")
    other_fit = tmp_path / "other" / "fit"
    (other_fit / dl.TRAINSET_MANIFEST).write_text((other_fit / dl.TRAINSET_MANIFEST).read_text() + " ")
    assert _freeze(w2) == dl.EXIT_REFUSED
    assert "is not the trainset manifest final ran on" in capsys.readouterr().out


def test_freeze_never_overwrites(tmp_path, capsys) -> None:
    w = _world(tmp_path)
    assert _freeze(w) == 0
    before = w["freeze"].read_bytes()
    assert _freeze(w) == dl.EXIT_REFUSED
    assert "already exists" in capsys.readouterr().out
    assert w["freeze"].read_bytes() == before


def test_a_one_byte_edit_breaks_the_freeze_for_verify_and_accept(tmp_path) -> None:
    w = _world(tmp_path)
    assert _freeze(w) == 0
    man = _seal(w)
    ff = w["freeze"]
    data = bytearray(ff.read_bytes())
    k = data.index(b'"arm": "fixed"') + len('"arm": "')
    data[k:k + 1] = b"F"
    os.chmod(ff, 0o644)
    ff.write_bytes(bytes(data))
    with pytest.raises(dl.FreezeError, match="changed after it was written"):
        dl.verify_freeze(ff)
    assert dl.main(["verify-freeze", "--freeze-file", str(ff)]) == dl.EXIT_REFUSED
    assert _accept(w, man) == dl.EXIT_REFUSED
    # the sidecar re-pointed at the edited bytes: the embedded self hash still catches it
    side = Path(f"{ff}.sha256")
    os.chmod(side, 0o644)
    side.write_text(f"{_sha(ff)}  {ff.name}\n")
    with pytest.raises(dl.FreezeError, match="embedded freeze_sha256"):
        dl.verify_freeze(ff)
    side.unlink()
    with pytest.raises(dl.FreezeError, match="sidecar"):
        dl.verify_freeze(ff)
    assert not Path(f"{ff}.accepted").exists()


# ------------------------------------------------------------------------- accept


def test_accept_runs_once_and_scores_a_known_m(tmp_path, capsys) -> None:
    w = _world(tmp_path)
    assert _freeze(w) == 0
    man = _seal(w)
    assert _accept(w, man) == 0
    fp = dl.freeze_paths(w["freeze"])
    assert fp["result"].name == "params_freeze.accept.json" and fp["marker"].name == "params_freeze.json.accepted"
    assert stat.S_IMODE(fp["result"].stat().st_mode) == 0o444
    marker = json.loads(fp["marker"].read_text())
    assert marker["result_sha256"] == _sha(fp["result"]) and marker["result"] == str(fp["result"])
    res = json.loads(fp["result"].read_text())
    assert res["passed"] is True and res["failed"] == []
    assert res["freeze"]["sha256"] == _sha(w["freeze"]) and res["bootstrap"] == {"n_resamples": BOOT, "seed": dl.SEED}
    assert res["datasets"][0]["windows_csv_sha256"] == _sha(w["mdir"] / "windows.csv")
    assert res["datasets"][0]["covered_by_m_sha256sums"] is True
    r = res["models"][MODEL]
    assert r["m_manifest"]["sha256"] == _sha(man)
    csv_path = Path(r["validation_csv"])
    assert r["validation_csv_sha256"] == _sha(csv_path) and csv_path.parent == fp["work"]
    with csv_path.open(newline="") as fh:
        vrows = list(csv.DictReader(fh))
    assert list(vrows[0]) == COLUMNS                                   # the dataset header
    ids = list(dict.fromkeys(x["cell_id"] for x in vrows))
    assert ids == [c["cell_id"] for c in w["cells"]]                   # dataset order, manifest cells only
    # 12 M cells x 11 windows (the first of each trimmed); healthy 6, both/TPOT 5, TTFT-only 1
    m = r["M"]
    assert (m["windows"], m["cells"], m["violating"]) == (132, 12, 66) and m["violating_fraction"] == 0.5
    assert set(m["cell_windows"].values()) == {11} and m["sealed_probes_not_evaluated"] == 1
    assert m["seen_before"] == [{"cell_id": f"i0_o0_c{_HEALTHY[0][0]}", "attempt": 1, "origin": "retained",
                                 "seen_before": True, "note": "first-round M cell, looked at once by the 09-22 refit"}]
    c = r["criteria"]
    # dwell 2: the first CRITICAL window of each violating cell is not confirmed -> 10 / 11
    assert c["A"]["passed"] and c["A"]["criteria"][0]["value"] == 1.0
    assert c["B"]["passed"] and c["B"]["criteria"][0]["value"] == pytest.approx(50 / 55)
    assert c["B"]["criteria"][1]["value"] == pytest.approx(10 / 11)   # every resample: 10 / 11
    assert c["B"]["criteria"][2]["value"] == 0.0 and c["B"]["criteria"][3]["value"] == 0.0
    assert c["B"]["all_violating_recall"] == {"value": pytest.approx(10 / 11), "target": 0.70, "met": True,
                                              "ci95": [pytest.approx(10 / 11)] * 2, "gating": False}
    assert c["C"]["critical_recall_ttft_only"] == pytest.approx(10 / 11) and c["C"]["windows"] == 11
    assert c["C"]["independent_windows"] == pytest.approx(11 / 3) and c["C"]["gating"] is False
    assert c["D"]["passed"] and c["D"]["family_gap_within_ci_half_width"] is True
    assert r["holdout_report"]["with_dwell"]["dwell_windows"] == 2
    # once only
    capsys.readouterr()
    assert _accept(w, man) == dl.EXIT_REFUSED
    assert "M is evaluated once" in capsys.readouterr().out
    assert json.loads(fp["result"].read_text()) == res
    # --recheck on unchanged inputs reproduces the stored result and writes nothing
    listing = sorted(p.name for p in w["freeze"].parent.rglob("*"))
    assert _accept(w, man, "--recheck") == 0
    assert "identical" in capsys.readouterr().out
    assert sorted(p.name for p in w["freeze"].parent.rglob("*")) == listing


def test_recheck_reports_every_difference_after_m_changed(tmp_path, capsys) -> None:
    w = _world(tmp_path)
    assert _freeze(w) == 0
    man = _seal(w)
    assert _accept(w, man) == 0
    # a healthy M window turns violating; the sums are re-pointed so nothing refuses earlier
    wcsv = w["mdir"] / "windows.csv"
    lines = wcsv.read_text().splitlines(keepends=True)
    k = next(i for i, x in enumerate(lines) if f"i0_o0_c{_HEALTHY[2][0]}" in x and ",200.0,30.0," in x)
    lines[k + 3] = lines[k + 3].replace(",200.0,30.0,", ",200.0,100.0,")
    wcsv.write_text("".join(lines))
    man = _seal(w)
    capsys.readouterr()
    assert _accept(w, man, "--recheck") == dl.EXIT_RECHECK_DIFFERS
    out = capsys.readouterr().out
    assert "models.dsqwen-7b.M.violating: stored 66 != recomputed 67" in out
    assert "validation_csv_sha256" in out and "datasets[0].windows_csv_sha256" in out


def test_result_differences_ignore_only_volatile_keys() -> None:
    a = {"evaluated_at_utc": "t0", "work_dir": "/a", "command": ["x"], "code": {"commit": "1"},
         "models": {"m": {"validation_csv": "/a/v.csv", "validation_csv_sha256": "s", "x": [1.0, float("nan")],
                          "holdout_report": {"generated_at": "t0", "windows": 3}}}}
    b = json.loads(json.dumps(a))
    b.update(evaluated_at_utc="t1", work_dir="/b", command=["y"], code={"commit": "2"})
    b["models"]["m"].update(validation_csv="/b/v.csv")
    b["models"]["m"]["holdout_report"]["generated_at"] = "t1"
    b["models"]["m"]["x"] = [1.0, float("nan")]
    assert dl.result_differences(a, b) == []
    b["models"]["m"]["holdout_report"]["windows"] = 4
    b["models"]["m"]["x"] = [1.0]
    b["extra"] = 1
    assert dl.result_differences(a, b) == [
        "extra: only in the recomputed result",
        "models.m.holdout_report.windows: stored 3 != recomputed 4",
        "models.m.x: 2 items stored, 1 recomputed"]


def test_accept_fails_b_on_false_alarms_and_exits_3(tmp_path, capsys) -> None:
    w = _world(tmp_path, false_alarm_cells=2)
    assert _freeze(w) == 0
    assert _accept(w, _seal(w)) == dl.EXIT_ACCEPT_FAILED
    out = capsys.readouterr().out
    assert "acceptance FAILED" in out and f"{MODEL}: B failed" in out
    res = json.loads(dl.freeze_paths(w["freeze"])["result"].read_text())
    c = res["models"][MODEL]["criteria"]
    assert res["passed"] is False and not c["B"]["passed"]
    fa = c["B"]["criteria"][2]
    assert fa["value"] == pytest.approx(20 / 66) and not fa["met"]
    # BA without dwell: 44 of 66 healthy windows on the healthy side, every violating one caught
    assert c["A"]["criteria"][0]["value"] == pytest.approx(0.5 * (44 / 66 + 1.0))
    assert Path(f"{w['freeze']}.accepted").exists()


def test_accept_refuses_training_inputs_changed_after_the_freeze(tmp_path, capsys) -> None:
    w = _world(tmp_path)
    assert _freeze(w) == 0
    man = _seal(w)
    fam = w["fit"] / f"{MODEL}_fitting_prefill_heavy.csv"
    fam.write_text(fam.read_text().replace(",30.0,", ",31.0,", 1))
    assert _accept(w, man) == dl.EXIT_REFUSED
    out = capsys.readouterr().out
    assert f"{MODEL}: training input family_prefill_heavy" in out and "changed after the freeze" in out
    assert not dl.freeze_paths(w["freeze"])["result"].exists()
    assert not dl.freeze_paths(w["freeze"])["work"].exists()


def _refused(w: dict, man: Path, capsys, needle: str) -> None:
    capsys.readouterr()
    assert _accept(w, man) == dl.EXIT_REFUSED
    out = capsys.readouterr().out
    assert needle in out, out
    fp = dl.freeze_paths(w["freeze"])
    assert not fp["result"].exists() and not fp["marker"].exists() and not fp["work"].exists()


def test_accept_refuses_a_broken_m_manifest(tmp_path, capsys) -> None:
    w = _world(tmp_path)
    assert _freeze(w) == 0
    other_label = dict(w["label"], tpot_p95_ms=150.0)
    _refused(w, _seal(w, label_def=other_label), capsys, "M was sealed under another label")
    _refused(w, _seal(w, freeze_sha="0" * 64), capsys, "sealed under another freeze")
    # raw data changed after sealing: the sums no longer match
    man = _seal(w)
    wcsv = w["mdir"] / "windows.csv"
    wcsv.write_text(wcsv.read_text().replace(",30.0,", ",30.5,", 1))
    _refused(w, man, capsys, "M data changed after it was sealed")
    # the sums file itself edited
    man = _seal(w)
    sums = w["mroot"] / MODEL / "SHA256SUMS"
    sums.write_text(sums.read_text() + "\n")
    _refused(w, man, capsys, "sha256sums_sha256")
    # a cell the datasets do not hold
    missing = [*w["cells"], {**w["cells"][0], "cell_id": "i0_o0_c3999999"}]
    _refused(w, _seal(w, cells=missing), capsys, "i0_o0_c3999999 a1: not found in any dataset")
    # a cell outside the holdout split / not valid
    train_cell = {**w["cells"][0], "cell_id": "i256_o128_c1200000", "shape": "S1"}
    _refused(w, _seal(w, cells=[*w["cells"][1:], train_cell]), capsys, "rows of split ['train']")
    # no manifest at all
    capsys.readouterr()
    assert dl.main(["accept", "--freeze-file", str(w["freeze"]), "--dataset", str(w["mdir"])]) == dl.EXIT_REFUSED
    assert f"{MODEL}: no M manifest" in capsys.readouterr().out


def test_accept_refuses_an_invalid_cell_and_a_cell_in_two_datasets(tmp_path, capsys) -> None:
    w = _world(tmp_path)
    wcsv = w["mdir"] / "windows.csv"
    text = wcsv.read_text()
    code = _TPOT[0][0]
    wcsv.write_text("".join(x.replace(",holdout,valid,", ",holdout,inconclusive,") if f"c{code}," in x else x
                            for x in text.splitlines(keepends=True)))
    assert _freeze(w) == 0
    _refused(w, _seal(w), capsys, "cell_status ['inconclusive']")
    wcsv.write_text(text)
    copy = _write_dataset(tmp_path / "copy" / "dataset", [])
    (copy / "windows.csv").write_text(text)
    man = _seal(w)
    capsys.readouterr()
    assert dl.main(["accept", "--freeze-file", str(w["freeze"]), "--dataset", f"mrun={w['mdir']}",
                    "--dataset", f"copy={copy}", "--m-manifest", str(man)]) == dl.EXIT_REFUSED
    assert "found in 2 datasets" in capsys.readouterr().out


def test_criteria_a_and_b_on_constructed_reports() -> None:
    entry = {"train_ba_at_published": 0.95, "stop_rule": {"satisfied": False, "reasons": ["short"]},
             "ci_half_frac": 0.16, "family_gap_frac": 0.31, "publish_rate": 1.0}
    h = {"windows": 100, "violating": 40, "at_published_theta": {"balanced_accuracy": 0.85},
         "with_dwell": {"critical_recall_both_tpot": None, "both_tpot_windows": 0,
                        "critical_false_alarm_on_healthy": 0.01, "healthy_windows": 60,
                        "critical_recall_of_violating": 0.6, "dwell_windows": 2,
                        "violation_classes": {"ttft_only": {"windows": 40, "critical_recall": 0.6}}}}
    boot = {"metrics": {"balanced_accuracy": {"ci95": [0.76, 0.9]},
                        "critical_recall_both_tpot": {"ci95": [None, None]},
                        "critical_false_alarm_on_healthy": {"ci95": [0.0, 0.03]},
                        "critical_recall_of_violating": {"ci95": [0.5, 0.7]},
                        "critical_recall_ttft_only": {"ci95": [0.5, 0.7]}}}
    c = dl.acceptance_criteria(entry, h, boot)
    # A: 0.85 >= 0.80 and CI low 0.76 >= 0.75, but 0.85 < 0.95 - 0.08
    assert [x["met"] for x in c["A"]["criteria"]] == [True, True, False] and not c["A"]["passed"]
    assert c["A"]["criteria"][2]["threshold"] == pytest.approx(0.87)
    # B: no both/TPOT-only window -> not evaluable -> not passed, whatever the false alarm
    assert c["B"]["evaluable"] is False and c["B"]["passed"] is False
    assert c["B"]["all_violating_recall"]["met"] is False
    assert c["C"]["independent_windows"] == pytest.approx(40 / 3)
    assert c["D"]["passed"] is False and c["D"]["family_gap_within_ci_half_width"] is False
    h["with_dwell"].update(critical_recall_both_tpot=0.9, both_tpot_windows=20)
    boot["metrics"]["critical_recall_both_tpot"]["ci95"] = [0.74, 1.0]
    c = dl.acceptance_criteria(entry, h, boot)
    assert c["B"]["evaluable"] and [x["met"] for x in c["B"]["criteria"]] == [True, False, True, True]
    boot["metrics"]["critical_recall_both_tpot"]["ci95"] = [0.75, 1.0]
    assert dl.acceptance_criteria(entry, h, boot)["B"]["passed"] is True
    boot["metrics"]["critical_false_alarm_on_healthy"]["ci95"] = [0.0, 0.081]
    assert dl.acceptance_criteria(entry, h, boot)["B"]["passed"] is False
    # A is not evaluable on an M holding one class only
    h.update(violating=0)
    assert dl.acceptance_criteria(entry, h, boot)["A"]["evaluable"] is False


def test_the_cell_bootstrap_resamples_cells_and_scores_fixed_dwell_flags() -> None:
    from tre_calibration.dataset import CalibrationWindow

    def cell(name, n, signal, cls):
        return [CalibrationWindow(name, "f", signal, cls is None, window_start_ms=t * 10_000, violation_class=cls)
                for t in range(n)]

    ws = cell("a", 4, 200.0, None) + cell("b", 4, 10.0, "both") + cell("c", 3, 10.0, "tpot_only")
    from scripts import theta_verdict as tv

    crit = tv.critical_dwell_flags(ws, theta=100.0, tau_crit=0.5, direction="higher_is_healthier")
    boot = dl.acceptance_bootstrap(ws, crit, theta=100.0, direction="higher_is_healthier", n_resamples=200, seed=1)
    m = boot["metrics"]
    assert boot["cells"] == 3
    assert m["critical_false_alarm_on_healthy"]["ci95"] == [0.0, 0.0]
    lo, hi = m["critical_recall_both_tpot"]["ci95"]      # cell b: 3/4, cell c: 2/3
    assert 2 / 3 - 1e-12 <= lo <= hi <= 0.75 + 1e-12
    assert m["balanced_accuracy"]["ci95"] == [1.0, 1.0]
    assert m["balanced_accuracy"]["resamples_used"] < 200   # one-class resamples have no BA
    assert m["critical_recall_ttft_only"]["resamples_used"] == 0
    assert dl.acceptance_bootstrap(ws, crit, theta=100.0, direction="higher_is_healthier",
                                   n_resamples=200, seed=1) == boot
