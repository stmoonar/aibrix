# TRE v2 Experiments Log (N5)

Per endgame plan §6. One entry per R-item run, appended in execution order.
Dependency order: R1 → R3 → R7 → R2 → R4 → R5 (R6 any gap).
All experiment output goes to 76 **local disk** (`/root/tre-experiments/`,
`/root/tre-n4b-soak/`), not NFS; only this log + final summary tables live in git.

> Status 2026-07-06: N5 is GATED on F4 (ADR-0008 isolated data plane) reaching a
> stable cutover (Phase B) so experiments run on the reproducible, traffic-isolated
> TRE data plane. Entries below are added as each R-item executes.

Entry template:

```markdown
## R<x> <name>  <YYYY-MM-DD>
- System version: git <sha> / images <digest list> / traceset <tag>
- Command(s) and params: <verbatim>
- Output dir: <path on 76 local disk>
- Result summary: <key numbers — oracle-normalized score, SLO viol %, switches, avg awake replicas>
- Anomalies / handling: <none | ...>
```

---

## Pending R-items (scaffold)

| # | Name | Depends on | Est | Driver |
|---|---|---|---|---|
| R1 | Old-system baseline (V_baseline, V_static) | isolated plane stable | ~2h+切换 | old system run_experiment.sh (secondary/prior-work) |
| R3 | Real re-fit (theta_m + capacity面), 2 percentile口径 | isolated plane | ~10h/model ×3 | `tre/deploy/scripts/r3_grid.py` + calibration CLI |
| R7 | Trace regenerate + freeze (design/lint/oracle) | R3 capacity面 | ~1h | `tre_replayer` design/lint/oracle; tag `traceset-v1` |
| R2 | New-system 7-trace regression (bucket_upper) | R7 | ~8h | `tre_replayer orchestrate`; reset_between_traces.sh |
| R4 | interpolated口径 re-run | R2 | ~8h | overlays/ablation-interpolated |
| R5 | Ablation matrix (no-fastloop / no-safescale / queue_len) × 3 traces | R4 | ~6h | overlays/ablation-* |
| R6 | Replayer timing precision (real vs dry-run) | any gap | ~0.5h | replayer dual-mode |

N5 gate: this log has R1–R7 entries, each reproducible; `git tag results-v1`;
main comparison table {old, new bucket_upper, interpolated, ablation arms} × oracle-normalized score.
