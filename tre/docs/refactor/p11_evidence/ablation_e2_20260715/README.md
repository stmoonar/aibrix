# E2 signal ablation 2026-07-15 seed: frozen setup and run ledger

Signal-source ablation over the canonical traces t2/t4/t5 with three active-TRE arms:
`zm` (TRE_DISABLE_ETA_GATE=true per the E2 symmetry requirement), `queue_len`, and
`decode_tps`. Same frozen identity as canonical E1: FROZEN_SHA
`ca61e485d229ecfa9aff209d7f29d24baf2db18f`, images `tre-v2-*:20260710-ca61e485`, params
`328bcfd2b54ca2db`, seed 20260715, node9 1/1/1 exact reset, 600 s cooldown, 30 s post-drain,
trim=1. Manifest: `deploy/campaigns/ablation_e2.json`. All arms use the TRE gateway 31094;
no APA arm exists in E2, so the APA-baseline defect documented in
`../canonical_rerun_20260715/README.md` does not affect these runs.

## Launch incident (resolved, no data impact)

The first launch died at t2_zm_seed1 replay start: the queue was launched with
`PYTHONPATH=deploy:common`, which lacks `replayer`, so `python3 -m tre_replayer.run_trace`
failed with ModuleNotFoundError before any request was sent (models never saw load; the
`finally` restore returned the cluster to observe + safe env). The failed directory is
archived as `t2_zm_seed1_failed_env_20260712/`. Relaunch used the full
`PYTHONPATH=common:deploy:ui:controller:service-manager:replayer`; all 9 runs completed.

## Audit

Full-queue audit BAD=[]: 26 required artifacts per run, gzip+uncompressed SHA and line
counts match `run.json` and `queue_status.json`, trim=1, freeze SHA and params everywhere,
arm env `TRE_SIGNAL_SOURCE` matches the run id arm, stderr empty, all three models'
queue/decode/prefill signals finite, node9 reset in every `reset_state.json`.
`SHA256SUMS` pins every artifact; verify with `sha256sum -c SHA256SUMS` here.

## Results: system V_req (request-weighted, trim=1)

| trace | zm | queue_len | decode_tps | best |
| --- | ---: | ---: | ---: | --- |
| t2 | 17.371% | 16.949% | 19.263% | queue_len |
| t4 | 4.810% | 4.823% | 5.493% | zm |
| t5 | 30.714% | 32.303% | 35.291% | zm |

zm is best or tied-best on t4/t5 and within 0.43 pp of queue_len on t2; decode_tps is
uniformly worst. The spread is largest on t5 (tp_pressure), the trace with genuine
cross-model capacity contention — consistent with Z_m's cross-model normalization being
most valuable exactly there. queue_len proposes markedly more actions than zm on t2
(152 vs 89 proposed; both actualized 5), i.e. it is a noisier decision signal at equal
actuation. Comparisons against the E1 `tre` arm (zm + eta gate on): t2 17.22% / t4 6.00% /
t5 31.15% — differences vs the E2 zm arm are within a still-unmeasured noise floor;
eta-gate attribution must wait for the noise-seed campaign.

## Campaign ledger

| run | seed | start | end | params hash | verdict | evidence |
| --- | ---: | --- | --- | --- | --- | --- |
| `t2_zm_seed1` | 20260715 | 2026-07-11T17:34:03Z | 2026-07-11T17:53:16Z | `328bcfd2b54ca2db` | DONE, V_req=17.371% | `t2_zm_seed1/` |
| `t2_queue_len_seed1` | 20260715 | 2026-07-11T18:05:12Z | 2026-07-11T18:24:25Z | `328bcfd2b54ca2db` | DONE, V_req=16.949% | `t2_queue_len_seed1/` |
| `t2_decode_tps_seed1` | 20260715 | 2026-07-11T18:36:22Z | 2026-07-11T18:55:35Z | `328bcfd2b54ca2db` | DONE, V_req=19.263% | `t2_decode_tps_seed1/` |
| `t4_zm_seed1` | 20260715 | 2026-07-11T19:07:30Z | 2026-07-11T19:20:23Z | `328bcfd2b54ca2db` | DONE, V_req=4.810% | `t4_zm_seed1/` |
| `t4_queue_len_seed1` | 20260715 | 2026-07-11T19:32:16Z | 2026-07-11T19:45:09Z | `328bcfd2b54ca2db` | DONE, V_req=4.823% | `t4_queue_len_seed1/` |
| `t4_decode_tps_seed1` | 20260715 | 2026-07-11T19:57:03Z | 2026-07-11T20:09:56Z | `328bcfd2b54ca2db` | DONE, V_req=5.493% | `t4_decode_tps_seed1/` |
| `t5_zm_seed1` | 20260715 | 2026-07-11T20:21:51Z | 2026-07-11T20:37:24Z | `328bcfd2b54ca2db` | DONE, V_req=30.714% | `t5_zm_seed1/` |
| `t5_queue_len_seed1` | 20260715 | 2026-07-11T20:49:18Z | 2026-07-11T21:04:52Z | `328bcfd2b54ca2db` | DONE, V_req=32.303% | `t5_queue_len_seed1/` |
| `t5_decode_tps_seed1` | 20260715 | 2026-07-11T21:16:45Z | 2026-07-11T21:32:18Z | `328bcfd2b54ca2db` | DONE, V_req=35.291% | `t5_decode_tps_seed1/` |
