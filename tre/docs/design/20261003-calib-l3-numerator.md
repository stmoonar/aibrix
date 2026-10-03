# L3: the TSS numerator from vLLM token counters (2026-10-03)

Branch `feat/calib-l3-20261003` (on `calib/theta-20261003`). Offline only: it re-reads
what the calibration capture already keeps; nothing on the cluster changes.

## What changes

TSS = (w_p · prefill tokens + decode tokens) / (running + λ · waiting), per 30 s window.

| numerator | counts | source |
|---|---|---|
| `gateway` (default) | tokens of the requests that **completed** in the window | per-request `usage` (offline) = the gateway's per-request histograms (online) |
| `vllm_counter` (L3, paper §5.2) | tokens the engine **processed** in the window | increase of `vllm:prompt_tokens_total` / `vllm:generation_tokens_total`, summed over the model's pods, from `cells/<stem>/vllm_metrics_1hz/` |

Only the numerator changes. Windows (30 s / 10 s grid, `(start, end]`), the denominator
(the same queue sidecar samples), the EMA, the idle rule, the labels and the cell verdicts
are the shared code's (`rewindow_from_raw.label_cell(..., token_source=...)`,
`r3_grid.compute_window_results`, `tre_common.tss`).

## Rules (scripts/l3_numerator.py)

Per pod and window:

1. Boundary value = the last 1 Hz sample at or before the boundary, at most `max_gap_ms`
   (default 2000 ms) old; otherwise the window is void (`no_sample`).
2. A sample spacing above `max_gap_ms`, or a failed scrape, between the two boundary
   samples: void (`gap`).
3. A counter that decreases between those samples: void (`reset`; pod or engine restart).
4. A sample without the counter: void (`missing_series`); a pod listed in
   `cell_meta.json` without a file: every window void (`no_metrics`).
5. Model value = sum over pods; one void pod voids the window.

Void windows are dropped before the EMA and counted per reason (dataset manifest
`numerator`, and per cell). **Why void and not interpolate:** the counters are cumulative,
so a hole does not bias a delta read across it, but a hole is where a reset hides, and
interpolation assumes a uniform token rate - false exactly in the bursty windows TSS
misreads. Expected cost: the first window of each cell (it starts before the capture's
first sample; inside the warm-up of every ladder cell) and windows around a restart.

## Guards against mixing

- `calibration_dataset --numerator vllm_counter` writes `<run>/dataset_l3` by default (never
  replaces `<run>/dataset`), adds `numerator_source`, `prompt_tokens_gateway`,
  `generation_tokens_gateway`, and records `numerator` in `manifest.json`.
- `dline_refit trainset` refuses sources of two numerators; `trainset.json` records it.
- `freeze` records each model's `numerator`; `accept` refuses an M dataset built with
  another numerator than the freeze's.

## RUN §E: the two fits (on 76, after D; CPU only)

Per model `m`, after the D stages finished. Each numerator gets its **own** fit and refit
directories (the trainset stage rewrites `trainset.json`; per-model directories also keep
one model's trainset from replacing another's).

```bash
cd $T/deploy   # PYTHONPATH as in RUN §0
for m in $MODELS; do
  # 1. L3 datasets next to the default ones (the default `dataset/` is the gateway numerator)
  for s in run1 run2 supp p1; do
    python3 -m scripts.calibration_dataset $CALIB_ROOT/$s/$m --numerator vllm_counter
    python3 -c 'import json,sys; d=json.load(open(sys.argv[1]))["numerator"]; print(sys.argv[1], d["windows_kept"], d["windows_void"])' \
        $CALIB_ROOT/$s/$m/dataset_l3/manifest.json
  done
  # 2. one fit per numerator (N = gateway | l3; D = dataset | dataset_l3)
  for N in gateway l3; do
    D=$([ $N = l3 ] && echo dataset_l3 || echo dataset)
    F=$CALIB_ROOT/fit/$N/$m/fit; O=$CALIB_ROOT/fit/$N/refit
    python3 -m scripts.dline_refit trainset --fit-dir $F --model $m \
        --h2-dataset run1=$CALIB_ROOT/run1/$m/$D --h2-dataset run2=$CALIB_ROOT/run2/$m/$D \
        --dataset supp=$CALIB_ROOT/supp/$m/$D --dataset p1=$CALIB_ROOT/p1/$m/$D
    python3 -m scripts.dline_refit alpha --fit-dir $F --out-dir $O --model $m --publish-tau-s 10
    python3 -m scripts.dline_refit wp    --fit-dir $F --out-dir $O --model $m
    python3 -m scripts.dline_refit final --fit-dir $F --out-dir $O --model $m --no-holdout
  done
done
# (the v1-λ variant: the same loop with --lambda-method v1 on wp, into fit/$N-v1lambda)
```

Report per model and numerator: θ, CI half width (D13), training BA, B′ on the training
set (dwell 1 gate, dwell 2 disclosed), and the L3 void counts. The run names (`run1=`,
`run2=` ...) are required: two `dataset_l3` directories would otherwise get the same name.

After the owner picks the numerator, §F freezes from that numerator's `--fit-dir` template
(`$CALIB_ROOT/fit/<N>/{model}/fit`, `--out-dir $CALIB_ROOT/fit/<N>/refit`). With L3, §G's M
and T14 datasets are rebuilt with `--numerator vllm_counter` before §H, and `accept` /
`calibration_decision` are pointed at `dataset_l3`; accept refuses the default ones.

**Going live with an L3 θ** needs the controller to compute the numerator the same way
(counter deltas, not the gateway's completion histograms); a θ fitted on L3 and applied to
the gateway numerator is miscalibrated. That is a separate controller change.

## Smoke (2026-10-03, existing captures)

Numerator ratio L3 / gateway per window (registry w_p = 0, so it is the decode ratio;
prompt ratio separately):

- 09-30 capture smoke, dsqwen-7b hold `i256_o64` (60 s, 1 pod): 3 kept windows, ratio
  1.000 / 0.993 / 0.966 (prompt 1.000 / 0.974 / 0.946); 1 window `no_sample` (the first).
- 10-03 idle-TTFT capture, 3 models x 418 s (one short request at a time): 117 kept
  windows, every ratio exactly 1.000 (prompt and decode); 3 `no_sample` (the first of each).

No hold cell of the 10-03 round existed yet; rerun the ratio on run1's hold cells (the
`prompt_tokens_gateway` / `generation_tokens_gateway` columns of `dataset_l3` give it
directly) before reading the §E table.
