# Canonical rerun 2026-07-15: frozen setup and run ledger

Status: **Phase 3 freeze complete; E3/E1/E2 campaign runs not started.** This directory is
the authoritative ledger for runs performed from the frozen runtime below. Operator for the
freeze and smoke checks: `root` via Codex, 2026-07-10 Asia/Shanghai.

## Freeze identity

- `FROZEN_SHA`: `45f7bb814b214976b63292cd23789446f07be447`
- Deployment-manifest HEAD: `00290ed0e7f0b553eef737718159126b5897cf43`
- Controller: `tre-v2-controller:20260710-45f7bb81`
- Service manager: `tre-v2-service-manager:20260710-45f7bb81`
- UI: `tre-v2-ui:20260710-45f7bb81`
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
| Unit/integration gate | 493 passed | git history + `make check` |
| Params round-trip | PUT exact alt thresholds, restart, hashes converge | `params.json` |
| H1 multi-wake | controlled equal-node test wakes node9 then node10; 2:2 final | `phase3_checks/h1_*` |
| H2 orphan detector | hidden sleeping pod alerts at simulated post-grace scan; unhide clears watch+alert | `phase3_checks/h2_fire_drill.json` |
| H5 signal stream | 5-minute observe harvest, 138 rows, all 18 fields, all 3 models | `phase3_checks/h5_timeline.csv` |
| H6 7b cap | t1 tight 180 s: 7b reaches 5 replicas; 7,643 requests | `phase3_checks/h6_*` |
| A4 t8 smoke | canonical t8 prefix 600 s: 11,023/11,023 successful; P99 schedule delay 2.36 ms | `phase3_checks/t8_smoke_*` |

H1 required one post-deploy correction. Inspection showed the multi-wake loop sorted sleeping
candidates once using initial node counts. Commit `45f7bb81` updates the count and reselects
after every wake; the new unit regression and a controlled live `1 -> 3` test both pass.

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