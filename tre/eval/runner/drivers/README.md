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
| `icse-ktrace-20261031/state-records/win10/` | RECORDS (not code), see below | how the win10 state of that batch was applied and rolled back |
| `icse-r3c-20261101/state-records/w10refit/` | RECORDS (not code), see below | how the 10 s window became the live state (every batch from 2026-11-01 on ran on it) |

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
* Work dir: the first-batch drivers (`icse-*/`) run `mkdir -p "$C"` unconditionally, as the originals did, so they create
  `$CAMPAIGN_WORK_DIR` (default `$CAMPAIGN_DATA_ROOT/<campaign dir>`) even for a read-only look. The two 11-03 drivers create it only on a
  real launch (`--check` writes nothing).

Notes: the drivers pin `MAIN_SHA` (`d17bccea` for the 11-03 drivers; the first-batch drivers carry theirs in the script) and refuse another
checkout; to run from a later main give `MAIN_SHA=<8 hex>`, as the driver's own message says.

## State records (`state-records/`)

These directories hold files that changed the CLUSTER STATE the runs started from. They are records of how a past state was produced,
copied verbatim (sha256 in `originals.tsv`, relation `record`). They are NOT code to run: they contain the paths, namespaces and
resource versions of the run host, and the first lines of each script say what it did. To bring a cluster to such a state use the
repo tools instead: the registry file from `tre/eval/registries/` with `params_set.py` / `merge_live_registry.py`, and the controller env
file from `tre/eval/controller_env/` with `controller_env.py apply`.

* `icse-ktrace-20261031/state-records/win10/`: `apply_win10.sh` (registry `5028a1209a5fe38d` + `TRE_METRICS_WINDOW_MS=10000`, temporary),
  `apply_win10.ktrace.sh` (the copy `run_all3.sh` generated with sed and executed), `rollback_win10.sh`, `cm_replace.py`, `make_registry.py`
  (how the win10 registry was derived), `registry-win10.yaml`, `registry-standard-w20s.yaml` (the registry that was live before: id
  `af9aef4701804743`, 20 s window; what the rollback restores), `diff-win10-vs-standard.txt`.
* `icse-r3c-20261101/state-records/w10refit/`: `apply_w10refit.sh` (the permanent switch to the 10 s window, 2026-10-31 about 19:09: registry
  `b1eb8f0c7593a86c` and `TRE_METRICS_WINDOW_MS=10000` in one controller patch), `rollback_w10refit.sh`, `cm_replace.py`,
  `registry-w10refit-fixed.yaml`, `meta.json`, `diff-vs-win10.txt`, `diff-vs-standard.txt`. The batches of 2026-11-01 and later ran on the
  state after this script plus later registry publications (tune D / F, merged theta, 11-03 parameters); those publications are the registry
  files in `tre/eval/registries/` (their production scripts are listed under Gaps).

## Gaps (not reproducible from the repo alone)

* `icse-ktrace-20261031/run_all3.sh` applies and rolls back the win10 registry + 10 s window with `apply_win10.sh`,
  `rollback_win10.sh`, `cm_replace.py` and `registry/` (variables `WIN10_DIR`, `WIN10_BACKUP_DIR`; defaults are cluster directories).
  Verbatim copies are in `icse-ktrace-20261031/state-records/win10/` as records; they still point to the paths of the run host, so
  the driver keeps using `WIN10_DIR` / `WIN10_BACKUP_DIR`. The registry half is `tre/eval/registries/5028a1209a5fe38d.yaml`, the env
  half `tre/eval/controller_env/`.
* How tune D / F (`ee37...`, `f3ac...`) were produced and published (`campaign-icse-tune-20261031/`, `backups/tss-tune-20261031/`, the
  `make_tune*.py` scripts) is not in the repo; the registry files are the result.
* The 11-03 TRE parameters (`f75155148eba1e61`) were published with `params_set.py`; the registry file is the result.
* Node name `nscc-ds-4a100-node9` in `precheck.disk_min_free_gb.nodes` of every spec (YAML keys are not expanded; same as the other specs of main).
* Images are named by tag in the specs / guards (controller 20261029-d9fa1571, 20261101-c8279678, 20261102-420c69ef; SM 20261029-d9fa1571; gateway
  20261029-8b60253f-nozmq2; baseline-scaler 20261029-8b60253f) and must be built from the shas in `tre/eval/controller_env/runs.tsv` / the specs' headers.
