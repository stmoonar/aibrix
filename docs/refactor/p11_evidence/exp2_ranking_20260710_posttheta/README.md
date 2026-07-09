# Experiment 2 rerun — ranking separation, post-recalibration (2026-07-10)

Rerun of the scenario-level train/test holdout ranking-separation eval (`deploy/scripts/eval_ranking_separation.py`,
same method as the original P8 experiment: `scenario_hash_holdout`, test_fraction=0.2, dual seed
`tre-v2-ranking` / `tre-v2-ranking-b`) on the post-recalibration sliding-window CSVs — the same
datasets the converged thetas were fit on (`r3_7b_slide_convprobe2.csv` 64 cells,
`r3_llama_slide_supp.csv` 51 cells, `r3_14b_slide_supp3.csv` 62 cells).

Files: `ranking2_{7b,llama,14b}_{a,b}.json`.

## Cell-count caveat (not a bug)

`scenario_hash_holdout` redraws its 80/20 split from the model's current scenario pool, which is
materially larger post-recalibration (7b 49→59 scenarios, llama 33→51, 14b 46→56, from the
densification/convergence-probe supplements). Before/after are two independent snapshots of the
splitter, not a paired comparison — expected, not a defect.

## Headline finding (Fable 5): AUROC alone hides the important result

| model/seed | test AUROC before | test AUROC after | true_violation caught / violations in test — before | — after |
|---|---|---|---|---|
| 7b a | 0.817 | 0.968 | 0/50 | 119/201 |
| 7b b | 0.972 | 0.944 | 0/62 | 77/189 |
| llama a | 0.898 | 0.929 | 0/16 | 35/127 |
| llama b | 0.771 | 0.935 | 0/19 | 87/109 |
| 14b a | 0.979 (train; test degenerate) | 0.807 | 0/0 (no violations in test) | 0/7 |
| 14b b | " | 0.988 | 0/0 (no violations in test) | 31/64 |

**Before recalibration, the published theta caught zero real violations in every held-out test
split that contained any.** The AUROC numbers (0.817-0.979) looked respectable but described pure
*ranking* quality on a threshold that, applied operationally, missed 100% of the danger it was
meant to catch. 14b's before-numbers are a degenerate case, not a real measurement: both test
splits had zero violation windows, so `generalizes=True` was a vacuous pass (`false_healthy==0` is
trivially true when there's nothing to miss), not a safety confirmation.

After recalibration, 5 of 6 model/seed pairs get nonzero recall for the first time (up to 79.8% at
llama-b). **14b-a is still 0/7** — a real gap, but n=7 is too small to be conclusive either way;
treat 14b's ranking validation as still-open pending more violation-bearing test draws, not as
"worse than before" (the `generalizes: True→False` flip for 14b is the flag firing meaningfully for
the first time, not a regression).

New cost: `false_violation` (false alarms) went from 0 in 5/6 before-pairs to nonzero in 3/6
after-pairs (7b-b: 38, llama-b: 36, 14b-a: 2) — the expected trade-off of a threshold that now
actually fires, not a red flag on its own.

## Known confound to fix before the next round

7b's post-recalibration seed-A test split draws `i1024_o512_c80` into its test fold — the cell
confirmed as containing a ramp-edge artifact window (see the convergence-probe section of
`../theta_stability_diagnostic_20260709/README.md`). The ramp-edge trim fast-follow should land
before the next bootstrap *or* ranking-separation run, since it now touches both downstream
artifacts.
