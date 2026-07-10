# Post-theta exp3 node-placement audit (2026-07-11)

## Purpose and verdict rule

This audit tests whether the pre-H1 lexicographic wake bias affected the TRE and
APA arms symmetrically. Existing exp3 numbers are placement-clean only when,
for the same trace:

1. the absolute TRE-vs-APA difference in mean node10 awake-replica share is at
   most 0.10; and
2. the maximum number of simultaneously awake bindings on node10 is equal in
   both arms.

`0.10` is the explicit operational definition of "small" used by the committed
command below. A trace failing either condition is `FLAG` and should be
superseded by the H1-based canonical rerun or cited with a placement caveat.

## Data sources

The committed exp3 timeline contains aggregate awake counts but not node or
serve-id placement. Placement is reconstructed from the richer physical source:
all 20 model pods' vLLM logs, whose serve IDs encode the node and GPU set.

- `pod_power_events.csv`: pod creation (`awake`) plus every vLLM
  `wake up ... complete` and `fall asleep ... complete` transition. Logs were
  collected with `kubectl -n default logs <pod> --timestamps` on 2026-07-10;
  no model pod had been recreated since before the campaign.
- `run_bounds.csv`: min/max `actual_send_ts_ms` and request count from each raw
  `comparison_v2_posttheta/{tre,apa}/t*/requests.jsonl` file.
- `../exp3_comparison_20260710_posttheta/timeline_{tre,apa}.csv`: committed
  per-arm sampling timestamps and aggregate awake counts.

The script replays physical power events at every timeline sample in each run,
then counts awake bindings per model and node. The aggregate timeline count is
used only as a cross-check. `aggregate_count_mismatches` records samples near
power transitions where the controller's aggregate sample and the vLLM
completion timestamp differ; physical completion events remain the placement
truth source. These mismatches are reported but are not part of the two-part
paper decision rule above.

## Regeneration

From the AIBrix repository root:

```bash
python3 tre/deploy/scripts/analysis/audit_node_placement.py \
  --timeline-tre docs/refactor/p11_evidence/exp3_comparison_20260710_posttheta/timeline_tre.csv \
  --timeline-apa docs/refactor/p11_evidence/exp3_comparison_20260710_posttheta/timeline_apa.csv \
  --run-bounds docs/refactor/p11_evidence/placement_audit_20260711/run_bounds.csv \
  --pod-events docs/refactor/p11_evidence/placement_audit_20260711/pod_power_events.csv \
  --output-dir docs/refactor/p11_evidence/placement_audit_20260711
```

The command regenerates:

- `placement_audit.csv`: 2,544 rows covering 7 traces x 2 arms x 3 models at
  848 timeline samples.
- `placement_summary.csv`: per-run mean node10 share, maximum node10
  co-residency, sample count, and aggregate cross-check mismatch count.
- `placement_verdicts.csv`: per-trace PASS/FLAG decision.

## Results

| Trace | Mean node10 share diff | Max node10 TRE/APA | Verdict | Reason |
|---|---:|---:|---|---|
| t1 | 0.364 | 3 / 3 | FLAG | node10 share differs |
| t2 | 0.519 | 3 / 3 | FLAG | node10 share differs |
| t3 | 0.420 | 3 / 3 | FLAG | node10 share differs |
| t4 | 0.354 | 3 / 3 | FLAG | node10 share differs |
| t5 | 0.341 | 3 / 3 | FLAG | node10 share differs |
| t6 | 0.051 | 3 / 3 | PASS | placement symmetric by rule |
| t7 | 0.402 | 3 / 3 | FLAG | node10 share differs |

APA stayed close to the three-replica node10 baseline for most traces. TRE's
additional wakes frequently landed on node9, so the historical wake-order bias
did not hit both arms identically. The maximum node10 co-residency was equal in
all traces, but the share criterion flags t1-t5 and t7.

## Citation decision

Only t6 passes this retrospective placement check. The post-theta t1-t5 and t7
TRE-vs-APA numbers must not be cited as placement-symmetric results; use the
canonical rerun on the H1 image as the replacement. If historical values are
shown for continuity, attach the node-placement caveat and reference this
audit.
