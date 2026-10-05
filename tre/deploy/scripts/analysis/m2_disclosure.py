#!/usr/bin/env python3
"""M2 forking-path disclosure (design 2026-10-05, "M2 discloses the three attributions and the
steady-only theta"). DISCLOSURE ONLY: not part of ``dline_refit accept``, never changes its
verdict, never selects anything.

Next to the accept verdict (hybrid label, pooled hybrid theta - the frozen parameters) it scores
the same M2 cells two more ways, with the accept arithmetic (``dline_refit.evaluate_model``:
onset gate, window false alarm, A, D, B', three-way verdict):

(a) ``completion_label``: the COMPLETION (label v1) windows of M2 under the frozen hybrid
    parameters (theta, tau_crit, lambda, w_p, severity cut - the cut is the hybrid training
    one, said so in the output);
(b) ``holdonly_theta``: the hybrid windows under a HOLD-ONLY hybrid freeze (``dline_refit
    trainset`` without ``--train-dynamic`` -> the same pipeline -> ``freeze``), fitted and
    sealed BEFORE M2 exists - the tool refuses a hold-only freeze created after any M2
    manifest, one fitted with dynamic training cells, or one under another label.

Inputs: the main freeze and the M2 manifests are checked exactly as accept checks them
(``dline_refit._accept_inputs`` on the hybrid datasets); the completion datasets must carry the
completion attribution. A collection run builds both datasets (``campaign.finalize_run``:
``dataset/`` and, for a hybrid run, ``dataset_hybrid/``); by hand::

    python -m scripts.calibration_dataset <M2 out-dir>                       # dataset/
    python -m scripts.calibration_dataset <M2 out-dir> --attribution hybrid  # dataset_hybrid/

Usage (from tre/deploy)::

    python3 -m scripts.analysis.m2_disclosure --freeze-file F --holdonly-freeze-file H \\
        --dataset M2.<model>=<root>/<model>/dataset_hybrid --completion-dataset M2.<model>=<root>/<model>/dataset \\
        --m-manifest <root>/<model>/M_manifest.json ... --out <file>.json

Writes ``--out`` (read-only, once) and ``<out>.d/`` (the validation CSVs).
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import sys
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

WHAT = ("M2 forking-path DISCLOSURE (design 2026-10-05): not acceptance evidence, not part of dline_refit "
        "accept, never changes its verdict; (a) completion-label windows under the frozen hybrid parameters, "
        "(b) hybrid windows under a hold-only hybrid freeze sealed before M2")


class Refused(RuntimeError):
    def __init__(self, problems: Sequence[str]):
        super().__init__("; ".join(problems))
        self.problems = list(problems)


def holdonly_problems(main: Mapping[str, Any], hold: Mapping[str, Any],
                      manifests: Mapping[str, Mapping[str, Any]]) -> list[str]:
    """Why the hold-only freeze cannot serve disclosure (b) (every reason)."""
    from scripts import dline_refit as dl

    out = [f"hold-only freeze: {x}" for x in dl.accept_gate_problems(hold)]
    created = str(hold.get("created_at_utc") or "")
    for model, entry in sorted(main["models"].items()):
        h = (hold.get("models") or {}).get(model)
        if h is None:
            out.append(f"{model}: not in the hold-only freeze")
            continue
        if h.get("label_def_sha256") != entry.get("label_def_sha256"):
            out.append(f"{model}: the hold-only freeze's label {h.get('label_def_sha256')} != the main freeze's "
                       f"{entry.get('label_def_sha256')}")
        if (h.get("dynamic_training") or {}).get("admitted"):
            out.append(f"{model}: the hold-only freeze was fitted with dynamic training cells (--train-dynamic)")
        written = str((manifests.get(model) or {}).get("written_at_utc") or "")
        if not created or (written and created >= written):
            out.append(f"{model}: the hold-only freeze ({created or 'no created_at_utc'}) is not older than the M2 "
                       f"manifest ({written}): disclosure (b) must be fixed before M2 exists")
    return out


def summary(ev: Mapping[str, Any]) -> dict:
    """The numbers a reader compares across the variants."""
    c = ev["criteria"]
    on, fa, a = c.get("onset") or {}, c.get("window_fa") or {}, c.get("A") or {}
    bp = ((c.get("B_prime") or {}).get("by_dwell") or {}).get("1") or {}
    return {"verdict": ev.get("verdict"),
            "onset": {"passed": on.get("passed"), "evaluable": on.get("evaluable"),
                      "episodes": (on.get("onset") or {}).get("episodes"),
                      "within_budget": (on.get("onset") or {}).get("within_budget"),
                      "lag_max_s": (on.get("onset") or {}).get("lag_max_s"),
                      "n_eff": on.get("n_eff"), "clopper_pearson_lower": on.get("clopper_pearson_lower")},
            "window_fa": {"passed": fa.get("passed"), "criteria": fa.get("criteria")},
            "A": {"passed": a.get("passed"), "criteria": a.get("criteria")},
            "D": {"passed": (c.get("D") or {}).get("passed"), "ci_half_frac": (c.get("D") or {}).get("ci_half_frac")},
            "B_prime_dwell1": {k: bp.get(k) for k in ("recall_severe", "recall_severe_ci95", "false_alarm",
                                                      "false_alarm_ci95")},
            "windows": ev["M"]["windows"], "violating": ev["M"]["violating"]}


def completion_entry(entry: Mapping[str, Any]) -> dict:
    """The frozen entry with its label's attribution set to completion (label v1); every
    parameter - theta, tau, lambda, w_p, the severity cut - stays the hybrid freeze's."""
    import dataclasses

    from tre_common import slo_labels

    out = copy.deepcopy(dict(entry))
    ld = slo_labels.LabelDefinition.from_dict(out["verdict_for_holdout"]["label_def"])
    out["verdict_for_holdout"]["label_def"] = dataclasses.replace(
        ld, attribution=slo_labels.ATTRIBUTION_COMPLETION).as_dict()
    return out


def disclose(freeze_file: Path, holdonly_file: Path, datasets: Sequence[str], completion: Sequence[str],
             m_manifests: Sequence[str], work: Path, *, n_resamples: int, seed: int) -> dict:
    from tre_common import slo_labels

    from scripts import dline_refit as dl

    inp, problems = dl._accept_inputs(freeze_file, datasets, m_manifests)
    doc = inp.get("doc") or {}
    hold: dict = {}
    try:
        hold = dl.verify_freeze(holdonly_file)
    except dl.FreezeError as exc:
        problems += [f"hold-only freeze: {x}" for x in exc.problems]
    if doc and hold:
        problems += holdonly_problems(doc, hold, inp.get("manifests") or {})
    csrc = []
    for text in completion:
        try:
            s = dl.DatasetSource.parse(text, sealed_to_h2=False)
        except dl.TrainingSetError as exc:
            problems.append(f"completion dataset {text}: {exc}")
            continue
        if dl.dataset_attribution(s.directory) != slo_labels.ATTRIBUTION_COMPLETION:
            problems.append(f"completion dataset {s.directory} carries {dl.dataset_attribution(s.directory)!r}")
        csrc.append(s)
    cm: dict = {}
    if not problems:
        cm, pr = dl.collect_m_rows(csrc, inp["manifests"])
        problems += [f"completion datasets: {x}" for x in pr]
    if problems:
        raise Refused(problems)
    cfgs, _sum, pr = dl.b_prime_inputs(doc, None, dl.ONLINE_DWELL_WINDOWS, "the controller's dwell")
    hcfgs, _hsum, hpr = dl.b_prime_inputs(hold, None, dl.ONLINE_DWELL_WINDOWS, "the controller's dwell")
    if pr or hpr:
        raise Refused(pr + [f"hold-only freeze: {x}" for x in hpr])
    work.mkdir(parents=True, exist_ok=False)
    models = {}
    for model in sorted(doc["models"]):
        entry, hentry = doc["models"][model], hold["models"][model]
        hyb = work / f"{model}_hybrid.csv"
        com = work / f"{model}_completion.csv"
        dl._write_validation_csv(hyb, inp["m"]["header"][model], inp["m"]["rows"][model])
        dl._write_validation_csv(com, cm["header"][model], cm["rows"][model])

        def ev(e, csv_path, d, bp):
            return summary(dl.evaluate_model(e, csv_path, n_resamples=n_resamples, seed=seed,
                                             b_prime_cfg=bp, gate_cfg=dl.accept_gate_of(d)))

        models[model] = {
            "frozen_hybrid_reference": {"note": "the accept arithmetic on the accept's own inputs (what accept "
                                                "reports; repeated here for the side-by-side only)",
                                        **ev(entry, hyb, doc, cfgs[model])},
            "completion_label": {"note": ("completion (label v1) windows; theta / tau / lambda / w_p and the "
                                          "severity cut are the frozen hybrid ones"),
                                 **ev(completion_entry(entry), com, doc, cfgs[model])},
            "holdonly_theta": {"note": "hybrid windows; parameters of the hold-only hybrid freeze (sealed before M2)",
                               "theta": hentry["published"]["theta"], "tau_crit": hentry["published"]["tau_crit"],
                               "lambda_wait": hentry["published"]["lambda_wait"], "w_p": hentry["published"]["w_p"],
                               **ev(hentry, hyb, hold, hcfgs[model])},
        }
    return {"what": WHAT, "code": dl.code_state(),
            "freeze": {"path": str(freeze_file), "sha256": dl.sha256_file(Path(freeze_file)),
                       "freeze_sha256": doc.get("freeze_sha256")},
            "holdonly_freeze": {"path": str(holdonly_file), "sha256": dl.sha256_file(Path(holdonly_file)),
                                "freeze_sha256": hold.get("freeze_sha256"), "created_at_utc": hold.get("created_at_utc")},
            "datasets": {"hybrid": list(datasets), "completion": list(completion)},
            "bootstrap": {"n_resamples": n_resamples, "seed": seed},
            "changes_the_verdict": False, "models": models}


def main(argv: Optional[Sequence[str]] = None) -> int:
    from scripts import dline_refit as dl

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--freeze-file", type=Path, required=True)
    ap.add_argument("--holdonly-freeze-file", type=Path, required=True)
    ap.add_argument("--dataset", action="append", required=True, metavar="[RUN=]DIR",
                    help="an M2 dataset of the frozen label's (hybrid) attribution; repeatable")
    ap.add_argument("--completion-dataset", action="append", required=True, metavar="[RUN=]DIR",
                    help="the same runs' completion (label v1) dataset; repeatable")
    ap.add_argument("--m-manifest", action="append", required=True)
    ap.add_argument("--resamples", type=int, default=dl.ACCEPT_RESAMPLES)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    work = args.out.with_name(args.out.name + ".d")
    if args.out.exists() or work.exists():
        ap.error(f"{args.out} or {work} exists (written once)")
    try:
        doc = disclose(args.freeze_file, args.holdonly_freeze_file, args.dataset, args.completion_dataset,
                       args.m_manifest, work, n_resamples=args.resamples, seed=dl.SEED)
    except Refused as exc:
        print("m2_disclosure REFUSED - nothing was written:")
        for x in exc.problems:
            print(f"  - {x}")
        return dl.EXIT_REFUSED
    args.out.write_text(json.dumps(doc, indent=1, sort_keys=True, default=str) + "\n", encoding="utf-8")
    for p in [args.out, *work.iterdir()]:
        os.chmod(p, 0o444)
    for model, r in doc["models"].items():
        print(f"[{model}] " + "; ".join(f"{k}: {v['verdict']}" for k, v in r.items()))
    print(f"wrote {args.out} (DISCLOSURE, not acceptance)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
