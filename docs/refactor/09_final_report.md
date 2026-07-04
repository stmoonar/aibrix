# P9 Final Report

## Scope

This report closes the offline refactor run for the TRE v2 tree under `/data/nfs_shared_data/xxy/aibrix` on server 76. Phases P0-P8 are tagged through `p8-done`; P9 adds the final integration evidence and residual run list.

## Phase Tags

| Phase | Tag | Status |
| --- | --- | --- |
| P0 | `p0-done` | Complete |
| P1 | `p1-done` | Complete |
| P2 | `p2-done` | Complete |
| P3 | `p3-done` | Complete |
| P4 | `p4-done` | Complete |
| P5 | `p5-done` | Complete |
| P6 | `p6-done` | Complete |
| P7 | `p7-done` | Complete with trace-capacity limitation recorded |
| P8 | `p8-done` | Complete; screenshot skipped because Playwright browser binaries are absent |
| P9 | p9-done | Complete |

## P9 Verification

| Requirement | Evidence | Status |
| --- | --- | --- |
| `make manifests` artifacts complete | Final gate ran `cd tre && make manifests`; it wrote 12 model deployment manifests to `tre/deploy/models`. | Complete |
| Offline e2e integration | A temporary localhost service-manager FastAPI process and a separate controller-driver process ran 60 ticks over 296.309 seconds. The first fixture tick produced a `critical_idle_capacity` scale decision, dispatched through the real `ServiceManagerClient`/`ActionQueue` to `PUT /v2/models/m1/target`, and moved the service-manager state to 2 awake replicas. Evidence: `docs/refactor/p9_evidence/offline_e2e_5min.jsonl`. | Complete |
| Controller integration test guard | `tre/controller/tests/test_p9_offline_integration.py` covers metrics refresh -> rescue decision -> Redis decision snapshot -> queue dispatch -> service-manager v2 app. | Complete |
| `make check` | Final gate passed: 176 tests. | Complete |
| `make smoke` | Final gate passed: `tre smoke ok`. | Complete |
| L3 cluster smoke | Skipped. Read-only preflight showed node10 GPUs idle, but the live cluster already has active `aibrix-system` service-manager and TRE controller pods, `redis-cli` is not installed on the host for memory preflight, and no new image/deploy artifact was built for a safe isolated `tre-v2` smoke. No Kubernetes write operations were performed. | Skipped with reason |

## Offline E2E Evidence Summary

Command shape used for P9 L2:

```bash
PYTHONPATH=tre/common:tre/service-manager python3 -m uvicorn tre_p9_sm_app:app --host 127.0.0.1 --port 18081
PYTHONPATH=tre/common:tre/controller:tre/service-manager python3 /tmp/tre_p9_controller_driver.py --ticks 60 --interval-s 5 --min-duration-s 295
```

Final driver result:

```json
{"ok": true, "ticks": 60, "elapsed_s": 296.309, "final_awake": 2}
```

The temporary processes were stopped after the run; no `tre_p9` processes or listener on port 18081 remained.

## Known Limitations Carried Forward

- P7 trace reports use placeholder capacity derived from observed trace RPS. They validate the lint/oracle pipeline but do not qualify final traces.
- P8 UI screenshot was not captured because Playwright browser binaries are not installed on server 76.
- L3 deployment smoke was skipped to avoid mutating an active shared cluster without a verified new image and Redis memory preflight.

## Residual Run List

| # | Item | Preconditions | Estimate | Command Entry |
| --- | --- | --- | --- | --- |
| R1 | Baseline comparison experiments, old system TRE/APA one trace each | Cluster idle; old system frozen at `/root/aibrix-main` | ~2h | Old system `run_experiment.sh` |
| R2 | New-system smoke plus formal 7-trace TRE/APA regression | R1 complete; images built and pushed; isolated namespace ready | ~8h | `tre/replayer orchestrate` |
| R3 | Refit with new percentile semantics: training grid, fit, registry threshold update | Cluster idle; calibration image/env ready | ~10h/model | `tre-calib collect` then `tre-calib fit` |
| R4 | Switch `percentile_mode` to `interpolated` with R3 parameters and rerun R2 | R3 complete | ~8h | Same as R2 |
| R5 | Ablation matrix, four switches by three traces | R4 complete | ~6h | `tre/deploy/overlays/ablation-*` plus replayer orchestration |
| R6 | Replayer versus `vllm bench serve` real-machine timing comparison | One model pod available and isolated | ~0.5h | Command to be finalized in `docs/refactor/07_replayer_audit.md` |
| R7 | Regenerate or repair traces under section 12 methodology and freeze final trace set | Real capacity surfaces from calibration | ~1h pure generation | `tre_replayer design` plus `tre_replayer lint` |
