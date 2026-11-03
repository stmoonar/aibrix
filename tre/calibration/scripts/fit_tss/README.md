# fit_tss: refit of the TSS parameters (theta, tau_crit, tau_high, lambda, w_p)

The offline tools that fit the per-model TSS parameters from calibration datasets, and the
entry point that reproduces the parameters published on 2026-11-03 (`Frecf_all`, registry
`f75155148eba1e61`). CPU only; read-only on the data; every output goes to a directory you name.

Method (the estimator, the label, the attribution, the lambda / w_p formula, the tau_high
ladder): local workspace `docs/tss-param-fit-method-20261103.md`; the preregistration of the
2026-11-03 supplement: `docs/prereg-calib-supplement-20261103.md`.

## Files

| file | role |
|---|---|
| `refit_supp.py` | the fit: one model x one analysis configuration -> `fit.json`, `candidate.json` (publishable values), `summary.txt`, `per_shape.csv`, `tau_high_by_shape.csv` |
| `supp_lib.py` | shared code: data loading, sealed-set guard, vectorised TSS, kept-window filter, attribution, tau_high ladder, lambda / w_p formula |
| `refit_fixedslo.py` | estimator library loaded by `supp_lib.py` (`glm`, `local_root`, `boot_ci`, `auroc`, `antitonic`); its own `main()` is the 10-29 / 10-31 refit |
| `validate_supp.py` | acceptance metrics of one candidate on one dataset (dev mode: training data; final mode: the one-time evaluation, ledger) |
| `selfcheck.py` | reproduces known numbers on the old data (tune F, review §8, the lambda / w_p formula) |
| `smoke_guard.py`, `smoke_newdata.py` | smoke tests on the real data (sealed-set refusals, final-mode ledger; the new-source code paths) |
| `run_frecipe_formula.sh` | **the Frecf_all fit** (and its controls f_old, f_le1100) |
| `run_after_campaign.sh` | the supplement pipeline after the campaign: build, fit (primary, S1-S5), dev, final-build, final |
| `run_before.sh` | the "before" reference on the old data (selfcheck, primary / S1-S5, dev validation) |
| `build_datasets.sh` | calibration datasets with the recipe of the old roots (`scripts.calibration_dataset`) |
| `run_alphafit_w10.sh` | the 10-30 run: its step 1 built the old 10 s root (`supp_lib.OLD_ROOTS[10000]`); steps 2-3 the D4' tau fit |
| `fit_env.sh` | shared shell environment (sourced) |
| `requirements.txt` | the package versions of the 2026-11-03 fits |

Hermetic tests: `tre/calibration/tests/test_fit_tss.py` (the fit's TSS equals
`recompute_tss_rows`; the sealed guard works from the path string alone; no site path).

## Configuration

Nothing site-specific is written in the tools; the environment names every location.

| variable | meaning |
|---|---|
| `TRE_FIT_DATA_ROOT` | the directory that holds the calibration data roots (`calib_20261003/`, `calib_supp_20261103/`, `analysis-tss-params-20261030/`, `refit-fixedslo-w10-20261031/`, `freeze-ctr20-20261028/`). The `old` training roots, the freeze parameters and the sealed-root prefixes are named relative to it |
| `TRE_FIT_PYLIB` | extra `PYTHONPATH` entries put first (a site-packages directory with the pinned numpy / pandas) |
| `TRE_FIT_NICE`, `TRE_FIT_CPUS` | nice level (default 19) and `taskset -c` list (default: no pinning) |
| `TRE_EXCLUSIVE_WINDOW_FILE`, `TRE_EXCLUSIVE_WINDOW_DENY_HOSTS` | the exclusive-window marker (default `<data root>/TRE_EXCLUSIVE_WINDOW`) and the hostname substrings that refuse to run while it exists |
| `TRE` | the `tre/` tree whose `scripts.calibration_dataset` builds datasets (default: this repo) |
| `TRE_FIT_LEDGER` | the one-time ledger of `validate_supp.py --i-am-the-final-evaluation` (2026-11-03: `<data root>/calib_supp_20261103_fit_tools/final_eval_ledger.jsonl`) |
| `GIT_CONFIG_GLOBAL` | for a host whose git refuses the repo ("dubious ownership"): a gitconfig with `[safe] directory = <repo>`, else the dataset manifests record `code_commit` null and `build_datasets.sh` refuses |

Dependencies: python 3.10.12 with the versions in `requirements.txt` (numpy 1.26.4,
pandas 2.2.3). On 2026-11-03 the fits ran on a host without pandas, from a copy of another
host's site-packages put first on `PYTHONPATH` (that host's own numpy is 2.2.6 and was not
used). Bit-identical results were checked only with exactly these versions.

## Reproduce Frecf_all (the 2026-11-03 parameters)

```bash
TRE_FIT_DATA_ROOT=<data root> TRE_FIT_PYLIB=<site-packages with requirements.txt> \
TRE_FIT_NICE=10 TRE_FIT_CPUS=0-7 \
  bash tre/calibration/scripts/fit_tss/run_frecipe_formula.sh <new dir> f_all [f_old f_le1100]
```

About 25 s per variant. The core call is

```
refit_supp.py --variant custom --attribution completion --lam-wp formula --r-star 0.05 --window-ms 10000 \
              --tag Frecf_all --out <dir>/f_all --data old --data A=$DS/A --data B=$DS/B
```

with `DS = <data root>/calib_supp_20261103/datasets/w10000`. Expected result
(`<dir>/f_all/Frecf_all/candidate.json`; equal to the live registry `f75155148eba1e61`):

| model | theta [CI90] | tau_crit | tau_high | lambda | w_p |
|---|---|---|---|---|---|
| dsqwen-7b | 382.4 [348.7, 445.5] | 0.65 | 1.47 | 32.91 | 0.009472 |
| dsllama-8b | 291.0 [268.3, 344.4] | 0.64 | 1.60 | 31.44 | 0.009446 |
| dsqwen-14b | 409.7 [343.4, 443.9] | 0.49 | 1.27 | 32.93 | 0.009425 |

Inputs (their sizes, mtimes and sha256 are in `fit.json` `data.<model>.sources[].files`):

* old: `<data root>/analysis-tss-params-20261030/win10/alphafit-w10000-s10000/datasets/<src>/<model>/dataset_hybrid`,
  sources run1 / run2 (14b: run2b) / supp / p1 (7b: p1r2), built 10-30 by `run_alphafit_w10.sh`
  step 1 (`WINDOW_MS=10000 STEP_MS=10000`, builder main `9f2856a1`) from `<data root>/calib_20261003`
  (`build_datasets.sh --old <root> 10000` builds the same sources).
* A, B: `<data root>/calib_supp_20261103/datasets/w10000/{A,B}/<model>/dataset_hybrid`, built 11-03 by
  `build_datasets.sh` (builder `97a4d4ab`) from the supplement campaign `<data root>/calib_supp_20261103/{A,B}`.
* label registry of both builds: `tre/deploy/registry.yaml` at commit `e891d29b` (`fit_label_registry`).

### Equivalence check (2026-11-03)

`run_frecipe_formula.sh <dir> f_all f_old f_le1100` from this directory, on the host and with
the packages of the original run (nice 10, 8 cores), against the original outputs
(`<data root>/calib_supp_20261103/fit/adhoc/frecipe-20261103/`). The check's outputs:
`<data root>/calib_supp_20261103/fit/adhoc/frecipe-repo-20261103/` and `.../fit-tss-repo-checks-20261103/`
(selfcheck, smoke tests, rebuilt datasets and the fit on them).

* `summary.txt`, `per_shape.csv`, `tau_high_by_shape.csv`: byte-identical, all three variants
  (f_all sha256 `8535e3ef…`, `d395c1ca…`, `d216cbf9…`).
* `fit.json`: every value identical except `meta.tool`, `meta.argv[0]` (the tool's path),
  `meta.argv` `--out`, `meta.time` and `meta.tool_sha256` of the three changed files (below);
  the input-file hashes and every number are equal. `candidate.json`: only `fit_json` (path)
  and `fit_json_sha256` differ (it hashes `fit.json`).
* logs: equal except the exclusive-window note (the original ran inside a window) and the
  path / line number in a numpy overflow warning of `refit_fixedslo.py`.
* `selfcheck.py` (which also runs `refit_fixedslo.py`'s own loading / filter / counting path):
  90/90 ok; `selfcheck.txt`, `.csv`, `.json` byte-identical with the original run's.
  `smoke_guard.py` and `smoke_newdata.py`: OK.
* From the raw campaign data: `build_datasets.sh` (builder = this repo at `ff6398f9`) rebuilt
  all 18 datasets (A, B and, with `--old`, the 12 old sources). A / B: `windows.csv`,
  `requests.csv`, `cells.csv`, `DATASET.md` byte-identical. Old: the same except
  `windows.csv`, which has the alternative-signal columns added after `9f2856a1` (`tpot_*`);
  for byte-identical old roots build with `TRE` = a checkout of `9f2856a1`. `manifest.json`
  differs in builder commit, build time and paths (the label registry's sha256 is equal).
  Frecf_all fitted on the rebuilt datasets: `per_shape.csv`, `tau_high_by_shape.csv` and the
  `candidate.json` values identical; `summary.txt` differs only in its header's data paths.

## Provenance

The tools were written on 2026-11-03 outside git; the originals are kept verbatim (with a
sha256 list) in `<data root>/backups/fit-tools-20261103/`. Paths below are relative to the
data root. The changes are paths and host rules only; the fit logic is unchanged.

| repo file | original | original sha256 | change |
|---|---|---|---|
| `refit_supp.py` | `calib_supp_20261103_fit_tools/refit_supp.py` | `6d71f11c7d44e4b3f205dd01cc959135d6bcc09fcfeddabc3203aa79f78390f7` | none (byte-identical) |
| `selfcheck.py` | `calib_supp_20261103_fit_tools/selfcheck.py` | `0a4d7d450db687a2bd710e7b4902e48cf87da11b7b672b9f92327593a729f667` | none (byte-identical) |
| `supp_lib.py` | `calib_supp_20261103_fit_tools/supp_lib.py` | `0a6d0b1b9325c1785dce28399c553a9956cd339987fd7a167d8082c5b5d42595` | data roots relative to `TRE_FIT_DATA_ROOT`; `refit_fixedslo.py` loaded from this directory; marker / host rule from the environment; freeze lambda check skipped (logged) without a data root |
| `validate_supp.py` | `calib_supp_20261103_fit_tools/validate_supp.py` | `181cbb8197fea9710d5ce9898a556786726ba5515976802b53cd37b41eaa0334` | ledger from `--ledger` / `TRE_FIT_LEDGER` (final mode refuses without one) |
| `refit_fixedslo.py` | `refit-fixedslo-w10-20261031/refit_fixedslo.py` | `0cdd5ba8fde42f83b4a92d1d8f78c15bc71ed4194206775574e5a3eca63988d9` | data roots relative to `TRE_FIT_DATA_ROOT`, `--out` without default; `tre_calibration.dataset` from this repo instead of the build tree of `e891d29b` (the raw-TSS path, `ema_tau_ms` None, of `recompute_tss_rows` / `tss_terms` is unchanged since that commit; `supp_lib.assert_tss` checks it on every fit) |
| `smoke_guard.py` | `calib_supp_20261103_fit_tools/test_guard.py` | `408ac121c32baead46850e5577e874171b27c1949ae86b88a796281eef28aa4c` | `--work`, `--candidate` instead of the tools directory |
| `smoke_newdata.py` | `calib_supp_20261103_fit_tools/test_newdata.py` | `4b7690672e81230536886115df83da470b284755395e22d8583ef3a9e0237b03` | `--work` instead of the tools directory |
| `build_datasets.sh` | `calib_supp_20261103_fit_tools/build_datasets.sh` | `65a136d18d0b88eb84daf6d95121175374f4a44fe8b5c370e72b933ef80d1035` | `fit_env.sh`; label registry from git (`e891d29b`) or `LABEL_REGISTRY`; logs under the output root |
| `run_after_campaign.sh` | `calib_supp_20261103_fit_tools/run_after_campaign.sh` | `56864d6c53f467212c6b9bb368fdcfa724a04a7f5bb874fdb4a5c4c19c402282` | `fit_env.sh`; `CALIB_ROOT`, `RAW_*`, `DS`, `FIT`, `TRE_FIT_LEDGER` from the environment |
| `run_before.sh` | `calib_supp_20261103_fit_tools/run_before.sh` | `1059db6313d516bd660906de2baa25f2fd472f646956aae17d6364a057849afe` | `fit_env.sh`; writes under `OUT_DIR` (was the tools directory) |
| `run_alphafit_w10.sh` | `analysis-tss-params-20261030/win10/run_alphafit_w10.sh` | `68f9b339c2c967feca0ed5b59dbda88dd0ab7d64b6628e04fff8edb5e08103ba` | `fit_env.sh`; `OUT` required; label registry as `build_datasets.sh` |
| `run_frecipe_formula.sh` | `calib_supp_20261103/fit/adhoc/frecipe-20261103/run_frecipe_formula.sh` | `33f9d0fcf39b6ee59326ecf8d501544304dce596621aae294a61489b27f6b769` | `fit_env.sh`; output directory and variants as arguments; refuses an existing output |

Not moved: `rstar_F.py` (`campaign-icse-tune-20261031/fit/`, sha256 `cca9746f…`; the tune F
fit, not imported: its ladder code is copied in `supp_lib.py`, `selfcheck.py` reproduces its
numbers), the bundled site-packages (`pylib75/`: versions in `requirements.txt`, file hashes
in the backup) and a host-specific gitconfig (`[safe] directory`, see `GIT_CONFIG_GLOBAL`).
