# H7 ramp-edge trim evidence (2026-07-10)

## Result

The offline scorer now defaults to `--trim-ramp-windows 1`. The trim is anchored to the
whole trace, never to a model's first request or to phase boundaries. With the canonical
30s window / 5s step, trim 1 advances the scoring origin by one 5s sliding-window position.
R3 calibration CSVs are already windowized, so the calibration, bootstrap-CI, and exp2
loaders drop the earliest row of each `scenario_id` by `window_start_ms`.

All emitted score reports record `trim_ramp_windows` and `trim_scope=trace_start_only`.
Future experiment evidence READMEs must state the trim count explicitly.

## Existing-run rescore

Source: `/root/tre-experiments/comparison_v2_posttheta/tre/t1/requests.jsonl`, the TRE arm
of the post-theta exp3 t1 run. That historical report used trim 0.

| metric | trim 0 | trim 1 |
|---|---:|---:|
| requests scored | 24,182 | 24,103 |
| requests trimmed | 0 | 79 |
| system V_req | 0.422670 | 0.424055 |
| system success rate | 0.816062 | 0.815459 |
| request-weighted violation-window fraction | 0.323470 | 0.325772 |

The first window was healthy for all three models in this particular run, so removing it
slightly worsens the aggregate rather than improving it. This is expected and confirms the
trim is a fixed preprocessing rule, not a result-dependent cleanup.

`trim1` produced 154 windows per model versus 155 for `trim0`. For every model, all 154
remaining CSV rows exactly equal `trim0` rows 1..154 after normalizing only the recorded trim
field: `mismatch_count=0`. Therefore only the trace-start window was removed; no phase-boundary
or later window changed. See `t1_window_alignment.json`.

## Regeneration

From `tre/` on node10:

```bash
export PYTHONPATH=common:deploy:replayer
SRC=/root/tre-experiments/comparison_v2_posttheta/tre/t1/requests.jsonl
for N in 0 1; do
  python3 deploy/scripts/analysis/score_request_trace.py \
    --input "$SRC" --registry deploy/registry.yaml \
    --trim-ramp-windows "$N" \
    --output "/tmp/h7_t1_trim${N}.json" \
    --windows-output "/tmp/h7_t1_trim${N}_windows.csv"
done
```

The committed summaries are `t1_trim0.json` and `t1_trim1.json`. The alignment artifact was
computed by grouping both window CSVs by model and comparing each trim-1 row with trim-0 row
at index `i+1`.

## Verification

- Focused H7 tests cover request scoring, global trace anchoring, CSV trimming, exp2 splits,
  and aligned parameter-refit inputs.
- Authoritative `make check`: 467 passed.
- No image build or cluster rollout is required; H7 is script-only.
