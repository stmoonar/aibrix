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
| _pending_ | | | | `328bcfd2b54ca2db` | | |

Per-run directories must be named `t<k>_<arm>_seed<j>/` and contain request JSONL,
`timeline_signals.csv`, action journal, pod events, params dump, exact command, timestamps,
image tags, and scored verdict. Figure regeneration commands will be added when E1/E2 data
exists; no paper figure may cite the Phase 3 smoke as a canonical result.