# Registry snapshots of the 2026-10-31 .. 2026-11-03 campaigns

One file per registry id: `<id>.yaml` is the `registry.yaml` of the ConfigMap `tre-v2-registry` that was live for the runs
(the `live-registry.yaml` every arm directory records), byte for byte. The id is `arm_registry.registry_id`
(`tre/eval/runner/arm_registry.py id`): sha256 of the parsed content (key order, comments and number spelling do not
count), first 16 hex digits. `index.tsv` lists file sha256, source arm and the runs that carried each id;
`tre/eval/tests/test_repro_assets_20261103.py` recomputes every id from its file.

| id | what it is | carried by (arms) | trs window / EMA | 7b theta, tau_crit, tau_high, lambda, w_p |
|---|---|---|---|---|
| `5028a1209a5fe38d` | "win10": TSS window 10 s, EMA tau 0, theta x0.5, slo_mode labels | campaign-icse-ktrace-20261031 (all 4 arms) | 10000 / 0 | 302.8, 0.52, 1.645, 2.125, 0.025 |
| `ee3712855753d0e8` | tune D (r* 5 %), slo_mode fixed | campaign-icse-r3c-full-D-20261101 (all 4 arms) | 10000 / 0 | 411.1, 0.66, 1.38, 2.125, 0 |
| `f3ac339a6e5de9be` | tune F (lambda_wait 20, w_p 0.01) | campaign-icse-r3e-full-20261101 (4 arms), campaign-icse-r3c-full-F-20261101 and campaign-icse-ktrace-full-F-20261101 (TRE only) | 10000 / 0 | 404.9, 0.65, 1.39, 20, 0.01 |
| `0ab5352747bdd70e` | tune F + the 11-02 `alt_thresholds` (10 s basis; refit queue_len / kv_usage / ttft_mean, new tpot_mean) | campaign-bl-rep2-r3e-20261103 (baselines only); the reference of both 11-03 drivers | 10000 / 0 | 404.9, 0.65, 1.39, 20, 0.01 |
| `f75155148eba1e61` | "Frecf_all": recalibrated TSS parameters, lambda from the SLO-budget formula (spec header of campaign-tre-rerun-r3e-20261103) = the live ConfigMap on 2026-11-03 (resourceVersion 237290695) | campaign-tre-rerun-r3e-20261103 (TRE) | 10000 / 0 | 382.4, 0.65, 1.47, 32.91, 0.009472 |

All five stamp `trs.numerator: step_counters` and `trs.theta_window_ms: 10000`, so each needs `TRE_METRICS_WINDOW_MS=10000` on the
controller (the controller refuses to start otherwise). The overlay says 10000 since 2026-11-03; the bootstrap registry
(`deploy/registry.yaml`, `deploy/overlays/tre-v2/params.yaml`) still stamps 20000 and is NOT one of these.

Putting a snapshot live: check `arm_registry.py id` against the file name first. Tunables (`trs.*`, `slo.*`, `alt_thresholds.*`,
replica bounds) go through `deploy/scripts/release/params_set.py --set ...` (bounds-checked, `--apply --restart --backup-dir`, SM
before controller); `lambda_wait` above 20 (`f75155148eba1e61`: 31-33) is refused by `registry_params.py` of main until the
`fix/lambda-bound-20261103` change (bound 20 -> 100) is merged. A whole-file replace of the ConfigMap key `registry.yaml` was done by
`cm_replace.py` of the 10-30 / 10-31 apply scripts, which are not in the repo (`tre/eval/runner/drivers/README.md`, gaps).
Set `TRE_METRICS_WINDOW_MS` together with the registry (`tre/eval/controller_env/`).
