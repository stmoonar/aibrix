# Canonical rerun 2026-07-15: frozen setup and run ledger

Status: **Phase 3 and E3 complete; Phase 4 queue canary passed; E1/E2 canonical runs not started.** This directory is
the authoritative ledger for runs performed from the frozen runtime below. Operator for the
freeze and smoke checks: `root` via Codex, 2026-07-10 Asia/Shanghai.

## Freeze identity

- `FROZEN_SHA`: `ca61e485d229ecfa9aff209d7f29d24baf2db18f`
- Deployment-manifest HEAD: `4a3b8b9678200a7fe0db4f64b3eef6bb5541c5df`
- Controller: `tre-v2-controller:20260710-ca61e485`
- Service manager: `tre-v2-service-manager:20260710-ca61e485`
- UI: `tre-v2-ui:20260710-ca61e485`
- Registry params hash: `328bcfd2b54ca2db`; applied hash identical;
  `pending_restart=false` after `/api/params` PUT and controller restart.
- Final controller mode: `observe`; hidden-orphan alerts: 0; safescale probes: 0.

Exact image IDs and build timestamps are in `phase3_checks/freeze.txt`. Three images were
built, correcting the plan's two-image estimate: `alt_thresholds` GET/PUT persistence lives
in the UI backend, so deploying only controller/SM would have left the required API stale.

## Frozen parameters and traces

`params.json` is the post-rollout `GET /api/params` response. It includes all exact
queue/decode/prefill thresholds and `lower_is_healthier` directions for the three models.
`trace_manifest.sha256` pins t1-t9. Replayer seed is the run ledger seed; scoring always uses
`--trim-ramp-windows 1`.

Canonical arms remain:

- TRE: `signal_source=zm`, eta gate enabled (default).
- APA: existing APA arm, same baseline and trace seed.
- Ablation only: t2/t4/t5 with `TRE_DISABLE_ETA_GATE=true` for every arm including zm;
  first-cut signals are zm, queue_len, decode_tps, then prefill_tps if time permits.

## Phase 3 live gates

| gate | result | evidence |
| --- | --- | --- |
| Unit/integration gate | 512 passed | git history + `make check` |
| Params round-trip | PUT exact alt thresholds, restart, hashes converge | `params.json` |
| H1 multi-wake | controlled equal-node test wakes node9 then node10; 2:2 final | `phase3_checks/h1_*` |
| H2 orphan detector | hidden sleeping pod alerts at simulated post-grace scan; unhide clears watch+alert | `phase3_checks/h2_fire_drill.json` |
| H5 signal stream | 5-minute observe harvest, 138 rows, all 18 fields, all 3 models | `phase3_checks/h5_timeline.csv` |
| H6 7b cap | t1 tight 180 s: 7b reaches 5 replicas; 7,643 requests | `phase3_checks/h6_*` |
| A4 t8 smoke | canonical t8 prefix 600 s: 11,023/11,023 successful; P99 schedule delay 2.36 ms | `phase3_checks/t8_smoke_*` |

H1 required one post-deploy correction. Inspection showed the multi-wake loop sorted sleeping
candidates once using initial node counts. Commit `45f7bb81` updates the count and reselects
after every wake; the new unit regression and a controlled live `1 -> 3` test both pass.

## Phase 4 re-freeze and queue canary

The pre-campaign audit found that `SignalLogWriter` declared `decode_tps` and `prefill_tps`
but hard-coded both to `nan`. Commit `ca61e485` now writes the already-computed tick-context
values and the active signal threshold. Tests verify both zm rows and alternate-signal rows.
A 45-second observe-mode live gate harvested 33/33 finite decode/prefill rows across all three
models.

The same commit adds `deploy/scripts/campaign_queue.py`. It enforces exact node9 reset by
serve_id, preserves `tre:v2:sm:{state,version}` while clearing per-run controller keys, checks
all model/TRE pods Ready and guard hashes zero, takes the locked cooldown, pins SHA/images/
params/arm env, uses gateway 31094 for TRE and 31592 for APA, harvests signals, actual SM
layout transitions, proposed controller actions, events and params, scores with trim=1, and
restores observe mode in `finally`.

APA needs counterfactual signal logging even though `ENABLE_TRE_SCALING=false` disables the
controller decision loops entirely. During an APA run the queue therefore keeps the loops
enabled but sets controller mode to `observe`; APA CRs are the sole actuator and TRE dispatch
is impossible. A two-arm 30-second live canary verified this contract: 44/44 requests per arm,
12 finite signal rows per arm, complete artifact sets, TRE `active/APA=0`, APA `observe/APA=3`,
and final restoration to the exact baseline. See `phase4_checks/queue_canary.json` and
`phase4_checks/h5_counterfactual.csv`.

E3 remains pinned to `ee882da0671e80d650a659dbd4a64215f3d7ac68`. The only later runtime
change is controller evidence serialization; the service-manager exact power path, vLLM calls,
readiness polling and GPU sampling are unchanged, so the E3 switch measurements are not rerun.
This is the required post-freeze impact assessment, not an assumption that the SHAs are equal.

The canonical queue is pinned in `deploy/campaigns/canonical_e1.json`: 18 ordered runs,
t1-t9 x TRE/APA, seed value 20260715, 600-second cooldown, 30-second post-drain.

## Baseline layout

The post-smoke reset has one awake replica per model, no hidden bindings, all on node9:
7b GPU0, llama GPU1, 14b GPU2-3. This is the node-mirrored equivalent of the earlier
all-node10 baseline and is now the reset target for both arms; natural serve-id downscale
makes it deterministic. Every campaign run must verify this exact awake layout, all system
pods Ready, alert/probe counts zero, then take the locked 10-minute cooldown. Placement
symmetry must still be checked across TRE/APA with the H4 audit before publication.

## Campaign ledger

| run | seed | start | end | params hash | verdict | evidence |
| --- | ---: | --- | --- | --- | --- | --- |
| `t1_tre_seed1` | 20260715 | 2026-07-10T10:18:32Z | 2026-07-10T10:32:25Z | `328bcfd2b54ca2db` | DONE, V_req=43.300% | `t1_tre_seed1/` |
| `t1_apa_seed1` | 20260715 | 2026-07-10T10:44:21Z | 2026-07-10T10:58:14Z | `328bcfd2b54ca2db` | DONE, V_req=55.810% | `t1_apa_seed1/` |
| `t2_tre_seed1` | 20260715 | 2026-07-10T11:10:03Z | 2026-07-10T11:29:16Z | `328bcfd2b54ca2db` | DONE, V_req=17.224% | `t2_tre_seed1/` |
| `t2_apa_seed1` | 20260715 | 2026-07-10T11:41:14Z | 2026-07-10T12:00:28Z | `328bcfd2b54ca2db` | DONE, V_req=39.885% | `t2_apa_seed1/` |
| `t3_tre_seed1` | 20260715 | 2026-07-10T12:12:17Z | 2026-07-10T12:30:59Z | `328bcfd2b54ca2db` | DONE, V_req=0.616% | `t3_tre_seed1/` |
| `t3_apa_seed1` | 20260715 | 2026-07-10T12:42:57Z | 2026-07-10T13:01:38Z | `328bcfd2b54ca2db` | DONE, V_req=48.163% | `t3_apa_seed1/` |
| `t4_tre_seed1` | 20260715 | 2026-07-10T13:13:28Z | 2026-07-10T13:26:20Z | `328bcfd2b54ca2db` | DONE, V_req=6.003% | `t4_tre_seed1/` |
| `t4_apa_seed1` | 20260715 | 2026-07-10T13:38:20Z | 2026-07-10T13:51:12Z | `328bcfd2b54ca2db` | DONE, V_req=19.904% | `t4_apa_seed1/` |
| `t5_tre_seed1` | 20260715 | 2026-07-10T14:02:59Z | 2026-07-10T14:18:33Z | `328bcfd2b54ca2db` | DONE, V_req=31.152% | `t5_tre_seed1/` |
| `t5_apa_seed1` | 20260715 | 2026-07-10T14:30:28Z | 2026-07-10T14:46:01Z | `328bcfd2b54ca2db` | DONE, V_req=55.025% | `t5_apa_seed1/` |
| `t6_tre_seed1` | 20260715 | 2026-07-10T14:57:49Z | 2026-07-10T15:12:47Z | `328bcfd2b54ca2db` | DONE, V_req=0.004% | `t6_tre_seed1/` |
| `t6_apa_seed1` | 20260715 | 2026-07-10T15:24:36Z | 2026-07-10T15:39:34Z | `328bcfd2b54ca2db` | DONE, V_req=0.004% | `t6_apa_seed1/` |
| `t7_tre_seed1` | 20260715 | 2026-07-10T15:51:23Z | 2026-07-10T16:10:37Z | `328bcfd2b54ca2db` | DONE, V_req=38.818% | `t7_tre_seed1/` |
| `t7_apa_seed1` | 20260715 | 2026-07-10T16:22:33Z | 2026-07-10T16:41:47Z | `328bcfd2b54ca2db` | DONE, V_req=52.850% | `t7_apa_seed1/` |
| `t8_tre_seed1` | 20260715 | 2026-07-10T16:53:38Z | 2026-07-10T17:12:50Z | `328bcfd2b54ca2db` | DONE, V_req=0.850% | `t8_tre_seed1/` |
| `t8_apa_seed1` | 20260715 | 2026-07-10T17:24:49Z | 2026-07-10T17:44:18Z | `328bcfd2b54ca2db` | DONE, V_req=67.707% | `t8_apa_seed1/` |
| `t9_tre_seed1` | 20260715 | 2026-07-10T17:56:06Z | 2026-07-10T18:15:22Z | `328bcfd2b54ca2db` | DONE, V_req=0.000% | `t9_tre_seed1/` |
| `t9_apa_seed1` | 20260715 | 2026-07-10T18:27:19Z | 2026-07-10T18:46:35Z | `328bcfd2b54ca2db` | DONE, V_req=7.663% | `t9_apa_seed1/` |

Per-run directories must be named `t<k>_<arm>_seed<j>/` and contain request JSONL,
`timeline_signals.csv`, action journal, pod events, params dump, exact command, timestamps,
image tags, and scored verdict. Figure regeneration commands will be added when E1/E2 data
exists; no paper figure may cite the Phase 3 smoke as a canonical result.

## E1 results: system V_req (request-weighted, trim=1)

Audit over all 18 runs: 26 required artifacts each, gzip/uncompressed SHA and line
counts match `run.json` and `queue_status.json`, `trim_ramp_windows=1`, freeze SHA
`ca61e485` and params `328bcfd2b54ca2db` everywhere, stderr empty, all three models'
queue/decode/prefill signal values finite. Sole flag: both t6 arms have an empty
`proposed_actions.jsonl` — legitimate zero-action runs (`run.json` proposed=actual=0,
220 controller decision rows alive, V_req~0.004% both arms), not a harvest defect.

| trace | TRE V_req | APA V_req | diff (APA-TRE) | winner |
| --- | ---: | ---: | ---: | --- |
| t1 | 43.300% | 55.810% | +12.509pp | TRE |
| t2 | 17.224% | 39.885% | +22.661pp | TRE |
| t3 | 0.616% | 48.163% | +47.546pp | TRE |
| t4 | 6.003% | 19.904% | +13.901pp | TRE |
| t5 | 31.152% | 55.025% | +23.873pp | TRE |
| t6 | 0.004% | 0.004% | +0.000pp | tie |
| t7 | 38.818% | 52.850% | +14.032pp | TRE |
| t8 | 0.850% | 67.707% | +66.857pp | TRE |
| t9 | 0.000% | 7.663% | +7.663pp | TRE |

TRE wins 8/9 with 1 tie (t6: both arms ~0.004%, zero-pressure trace;
both arms proposed zero actions). Largest gaps: t8 (+66.86pp) and t3 (+47.55pp).
`SHA256SUMS` in this directory pins every per-run artifact plus queue plan/status;
verify with `sha256sum -c SHA256SUMS` from this directory.

## Diagnosis: the "7b capped at 4 on t1/t7" backlog item is GPU contention, not a planner bug

Pre-rerun reports had 7b peaking at 4 awake replicas on t1/t7 with "5 physically feasible".
This rerun resolves it. On t7 7b in fact reached 5 awake (target 6) while llama stayed at 1.
On t1, llama legitimately woke `dsllama-8b-...-node10-gpu-0` at 10:19:11Z — before any of
7b's three wakes (10:21:03Z+) — after which all 8 GPUs were occupied (7b x4, llama x2,
14b x1 on 2 GPUs). A 5th 7b replica was physically infeasible for the rest of the run; the
rescue loop kept proposing `delta=1` for 7b (53 proposed vs 6 actual actions) and each
proposal correctly failed placement. The old "feasible 5" claim assumed llama pinned at 1,
which the demand-shift trace violates. Remaining open question (policy, not bug): whether
rescue should become donor-aware under full-GPU pressure and preempt a lower-utility replica;
that is a post-campaign design decision, out of scope for the frozen runtime.
