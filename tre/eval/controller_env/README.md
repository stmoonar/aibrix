# Controller env of the recorded runs

`controller_env.py` writes the env of the live controller Deployment to a file (`snapshot`), shows what a file would change (`diff`), and
applies it with one `kubectl patch` = one Recreate rollout (`apply --backup-dir`, which also writes the rollback script). The patch
carries the resourceVersion of the Deployment it was planned on, so a concurrent edit fails the apply instead of being overwritten.
`apply` is refused (exit 2, nothing written) when
- the exclusive-window marker cannot be located (neither `TRE_EXCLUSIVE_WINDOW_FILE` nor `CAMPAIGN_DATA_ROOT` is set: a window cannot be
  ruled out; plain shells on the servers have neither), or the marker exists. Only the session that owns the window may apply inside it:
  `--window-owner X` must equal the `owner=` field of the marker line;
- a campaign runner or driver shell is active on this host (`ps`; `run_campaign*.sh`, `run_all*.sh`, `run_arm_pilot.sh`, `campaign.py run`,
  scripts under `drivers/`, also `bash ./x.sh` started inside `drivers/`; a `bash -c` wrapper that names them also matches: refusing is
  the safe direction);
- the file does not name `TRE_METRICS_WINDOW_MS` while the live Deployment sets it. Applying would unset it, and the controller image
  falls back to ITS code default, which can differ from this checkout's (image 420c69ef: 20000; this checkout: 10000) and refuses a registry
  that stamps another window; being `Recreate`, the cluster would have no controller. Name the key in the file. Only when neither the file nor
  the live Deployment names it is this checkout's code default used (with a note that an older image may differ);
- the effective `TRE_METRICS_WINDOW_MS` disagrees with `trs.theta_window_ms` of the registry (live ConfigMap, or `--registry-file`).

If the API server refuses the patch (a concurrent edit changed the resourceVersion), nothing was applied: `rollback.sh` is renamed to
`rollback.sh.not-applied` (it would overwrite the concurrent edit) and the tool prints "not applied". If the patch went through but the
rollout is not confirmed, `rollback.sh` stays. The rollback script passes its arguments on (`rollback.sh --window-owner X`).
Tested with a fake kubectl in `tre/eval/tests/test_repro_assets_20261103.py`. The server-side rejection of a stale resourceVersion is
not tested (the fake models it).

| file | runs (`runs.tsv`) | controller image |
|---|---|---|
| `controller-env-20261031-w10.env` | campaign-icse-ktrace-20261031, -r3c-full-D, -r3c-full-F, -ktrace-full-F, -r3e-full (10-31 .. 11-02) | 20261029-d9fa1571 (ktrace), 20261101-c8279678 (others) |
| `controller-env-20261103-live.env` | campaign-bl-rep2-r3e-20261103, campaign-tre-rerun-r3e-20261103 | 20261102-420c69ef |

The two files differ in one value: `TRE_SAFESCALE_SUPPRESS_HOT_PROACTIVE` `0` vs `false` (same meaning). The runner-managed switches
(`TRE_SAFESCALE_SCOPE`, `TRE_SIGNAL_SOURCE`, `TRE_ABLATION_*`, `TRE_SAFESCALE_SUPPRESS_HOT_PROACTIVE`;
`campaign.py` CONTROLLER_ENV_KEYS; `TRE_DISABLE_ETA_GATE` was one of them until eta was removed, see below) are written per arm from the spec's `arm_defs.<arm>.controller_env` and put back to the canonical value
afterwards; the files hold the canonical state between arms.

## Live env against `deploy/overlays/tre-v2/controller.yaml` (main + the 11-03 window change)

| variable | live | overlay | read by the controller (image 420c69ef, `config.py`) | effect |
|---|---|---|---|---|
| `TRE_METRICS_WINDOW_MS` | 10000 | 10000 (was 20000 until 11-03) | yes, default 10000 (20000 until 11-03) | real: the TSS window; must equal every model's `trs.theta_window_ms` |
| `TRE_ACTION_COOLDOWN` | 1 | absent | no: in `REMOVED_TIMER_ENV`, logged once and ignored since 10-29 (TSS-only release) | none (leftover on the live Deployment) |
| `TRE_DISABLE_ETA_GATE` | false | absent | yes (`_get_bool`, default false) | none for `TRE_SIGNAL_SOURCE=zm` (false is the default; it must be true for the alt-signal arms); written `false` by the runner's reset. Removed with eta (chore/remove-eta-20261102): a controller built after that removal logs a warning and ignores `false`, refuses to start on `true`; the runner unsets it (`REMOVED_CONTROLLER_ENV_KEYS` / `CTL_REMOVED_KEYS`). The files above are records of the earlier runs and keep it |
| `TRE_SIGNAL_SOURCE` | unset | zm | yes, default `zm` | none (unset = zm; the runner's canonical state is unset) |
| `TRE_SAFESCALE_SUPPRESS_HOT_PROACTIVE` | false (0 before 11-02) | "0" | yes (`_get_bool`) | none (0 and false are both false) |

Every other variable is equal in name and value. How the 10 s window got onto the live Deployment: `TRE_METRICS_WINDOW_MS` is not a
campaign `controller_env` key (the drivers only check it: `WINDOW_EXPECT=10000`). It was patched together with the registry in one
step by the 10-30 `apply_win10.sh` (temporary, rolled back by `run_all3.sh` after the ktrace batch) and then left at 10000 by the
10-31 `apply_w10refit.sh`. Both scripts are recorded verbatim under `tre/eval/runner/drivers/*/state-records/` (records, not code);
`controller_env.py apply` is the repo-side replacement for the env half, `tre/eval/registries/` holds the registry half.
