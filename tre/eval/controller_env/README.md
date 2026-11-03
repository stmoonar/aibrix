# Controller env of the recorded runs

`controller_env.py` writes the env of the live controller Deployment to a file (`snapshot`), shows what a file would change (`diff`), and
applies it with one `kubectl patch` = one Recreate rollout (`apply --backup-dir`, which also writes the rollback script). The patch
carries the resourceVersion of the Deployment it was planned on, so a concurrent edit fails the apply instead of being overwritten.
`apply` refuses while an exclusive-window marker exists (`TRE_EXCLUSIVE_WINDOW_FILE`, default `$CAMPAIGN_DATA_ROOT/TRE_EXCLUSIVE_WINDOW`),
while a campaign runner or driver shell is active, and when the EFFECTIVE `TRE_METRICS_WINDOW_MS` disagrees with `trs.theta_window_ms`
of the registry (the controller would not start). Effective = the value in the file; a file without the key unsets it, and the
controller then falls back to the default of its code (read from this checkout; name the key in the file when the image may be older).
Tested with a fake kubectl in `tre/eval/tests/test_repro_assets_20261103.py`.

| file | runs (`runs.tsv`) | controller image |
|---|---|---|
| `controller-env-20261031-w10.env` | campaign-icse-ktrace-20261031, -r3c-full-D, -r3c-full-F, -ktrace-full-F, -r3e-full (10-31 .. 11-02) | 20261029-d9fa1571 (ktrace), 20261101-c8279678 (others) |
| `controller-env-20261103-live.env` | campaign-bl-rep2-r3e-20261103, campaign-tre-rerun-r3e-20261103 | 20261102-420c69ef |

The two files differ in one value: `TRE_SAFESCALE_SUPPRESS_HOT_PROACTIVE` `0` vs `false` (same meaning). The runner-managed switches
(`TRE_SAFESCALE_SCOPE`, `TRE_SIGNAL_SOURCE`, `TRE_ABLATION_*`, `TRE_DISABLE_ETA_GATE`, `TRE_SAFESCALE_SUPPRESS_HOT_PROACTIVE`;
`campaign.py` CONTROLLER_ENV_KEYS) are written per arm from the spec's `arm_defs.<arm>.controller_env` and put back to the canonical value
afterwards; the files hold the canonical state between arms.

## Live env against `deploy/overlays/tre-v2/controller.yaml` (main + the 11-03 window change)

| variable | live | overlay | read by the controller (image 420c69ef, `config.py`) | effect |
|---|---|---|---|---|
| `TRE_METRICS_WINDOW_MS` | 10000 | 10000 (was 20000 until 11-03) | yes, default 10000 (20000 until 11-03) | real: the TSS window; must equal every model's `trs.theta_window_ms` |
| `TRE_ACTION_COOLDOWN` | 1 | absent | no: in `REMOVED_TIMER_ENV`, logged once and ignored since 10-29 (TSS-only release) | none (leftover on the live Deployment) |
| `TRE_DISABLE_ETA_GATE` | false | absent | yes (`_get_bool`, default false) | none for `TRE_SIGNAL_SOURCE=zm` (false is the default; it must be true for the alt-signal arms); written `false` by the runner's reset |
| `TRE_SIGNAL_SOURCE` | unset | zm | yes, default `zm` | none (unset = zm; the runner's canonical state is unset) |
| `TRE_SAFESCALE_SUPPRESS_HOT_PROACTIVE` | false (0 before 11-02) | "0" | yes (`_get_bool`) | none (0 and false are both false) |

Every other variable is equal in name and value. How the 10 s window got onto the live Deployment: `TRE_METRICS_WINDOW_MS` is not a
campaign `controller_env` key (the drivers only check it: `WINDOW_EXPECT=10000`). It was patched together with the registry in one
step by the 10-30 `apply_win10.sh` (temporary, rolled back by `run_all3.sh` after the ktrace batch) and then left at 10000 by the
10-31 `apply_w10refit.sh`. Both scripts are recorded verbatim under `tre/eval/runner/drivers/*/state-records/` (records, not code);
`controller_env.py apply` is the repo-side replacement for the env half, `tre/eval/registries/` holds the registry half.
