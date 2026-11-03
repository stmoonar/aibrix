# Campaign drivers, guards and the files they came from (2026-10-31 .. 2026-11-03 runs)

Repo copies of the specs (`tre/eval/runner/campaigns/`), drivers and guards of the campaigns whose data is in use. Everything is
parameterised by three environment variables, nothing else is site-specific except the node name noted below:

| variable | meaning | default |
|---|---|---|
| `CAMPAIGN_DATA_ROOT` | directory with the results roots (`<name>-run`), the trace sets (`traces-*`), `baseline-traces/`, the exclusive-window marker | none: required (specs: `${CAMPAIGN_DATA_ROOT}`) |
| `CAMPAIGN_WORK_DIR` | where a driver writes its log, STOP file and the runner stdout | `$CAMPAIGN_DATA_ROOT/<original campaign dir name>` |
| `TRE_REPO` | the `tre/` tree the runner, client and guards come from | the `tre/` of this checkout |

Example: `CAMPAIGN_DATA_ROOT=/data/nfs_shared_data/xxy bash tre/eval/runner/run_campaign.sh tre/eval/runner/campaigns/campaign-bl-rep2-r3e-20261103.yaml --dry-run`,
or `NAME=campaign-bl-rep2-r3e-20261103 bash tre/eval/runner/drivers/bl-rep2-20261103/run_campaign-bl-rep2-20261103.sh --check` (read only).

| directory | contents | runs |
|---|---|---|
| `bl-rep2-20261103/` | driver, `guard_cluster.py`, `registry_guard.py`, trace sha256 lists, `source_runs.tsv` | campaign-bl-rep2-r3e-20261103 (APA / TokenScale / PreServe, second sample; the primary spec campaign-bl-rep2-20261103 was never run) |
| `tre-rerun-20261103/` | driver, `guard_cluster_tre.py`, `tss_only_guard.py`, `make_rep2.sh`, `CHECKLIST.md`, trace sha256 list, `source_tre_runs.tsv` | campaign-tre-rerun-r3e-20261103 (TRE with the recalibrated TSS parameters; rep 1, stopped after 5 of 7 arms; rep 2 not run) |
| `icse-r3c-20261101/` | `run_r3c_full_D.sh` (tune D, 4 arms), `run_campaign-icse-r3c-full-F-20261101.sh`, `run_campaign-icse-ktrace-full-F-20261101.sh` (tune F, TRE only) | first batches |
| `icse-r3e-20261101/` | `run_campaign-icse-r3e-full-20261101.sh` | first batch (tune F, 4 arms, Real_conv / Real_code r3e) |
| `icse-ktrace-20261031/` | `run_all3.sh` | first batch (win10 registry for all arms) |

Registries: `tre/eval/registries/`. Controller env: `tre/eval/controller_env/`. Traces: `tre/loadgen_v1/configs/trace_sets/`.

## Originals and differences

`originals.tsv`: for each repo file the original (path relative to the data root, sha256 as it was run on 2026-10-31 .. 11-03) and
the repo sha256. For the specs the sha256 is the `campaign_sha256` the runner wrote into every arm's `campaign_manifest.json`.
Relation: `identical` (byte for byte), `spec-param` (the data root is `${CAMPAIGN_DATA_ROOT}` and 5 `# [repo]` comment lines are added on top: drop
those lines, put the root back, and the original comes back byte for byte - tested), `script-param` / `doc-param` (paths only, see
`DIFFS.txt`, the unified diffs), `data-relative-paths` (sha256 lists with paths relative to the data root), `regenerated` (below).
For every spec the runner's `--dry-run` output with the repo copy is identical to the one with the original, except for a time stamp.

Not path-only:
* `campaign-tre-rerun-r3e-rep2-20261103.yaml` is regenerated with `make_rep2.sh` from the repo rep 1. The original (07:26) was stale: it
  still had `registry_id: "FILL_REGISTRY_ID"` and lacked the PARAMETERS header, because rep 1 was edited at 13:48 - 13:52 and `make_rep2.sh` was
  not re-run. It never ran.
* The two 11-03 drivers create the work dir (`mkdir -p`) only on a real launch, so `--check` still writes nothing.

Notes: the drivers pin `MAIN_SHA` (`d17bccea` for the 11-03 drivers; the first-batch drivers carry theirs in the script) and refuse another
checkout; to run from a later main give `MAIN_SHA=<8 hex>`, as the driver's own message says.

## Gaps (not reproducible from the repo alone)

* `icse-ktrace-20261031/run_all3.sh` applies and rolls back the win10 registry + 10 s window with `apply_win10.sh`,
  `rollback_win10.sh`, `cm_replace.py` and `registry/` from the cluster directories `analysis-tss-params-20261030/win10` and
  `backups/tss-win10-20261030` (variables `WIN10_DIR`, `WIN10_BACKUP_DIR`). They are not copied. The registry half is
  `tre/eval/registries/5028a1209a5fe38d.yaml`, the env half `tre/eval/controller_env/`.
* How tune D / F (`ee37...`, `f3ac...`) were produced and published (`campaign-icse-tune-20261031/`, `backups/tss-tune-20261031/`, the
  `make_tune*.py` scripts) is not in the repo; the registry files are the result.
* The 11-03 TRE parameters (`f75155148eba1e61`) were published with `params_set.py` from `fix/lambda-bound-20261103`; main still refuses
  `lambda_wait` > 20.
* Node name `nscc-ds-4a100-node9` in `precheck.disk_min_free_gb.nodes` of every spec (YAML keys are not expanded; same as the other specs of main).
* Images are named by tag in the specs / guards (controller 20261029-d9fa1571, 20261101-c8279678, 20261102-420c69ef; SM 20261029-d9fa1571; gateway
  20261029-8b60253f-nozmq2; baseline-scaler 20261029-8b60253f) and must be built from the shas in `tre/eval/controller_env/runs.tsv` / the specs' headers.
