# Alternative queue thresholds fit (generated 2026-07-10)

## Result

A1 replaces the shared `qsat=4.0` queue placeholder with model-specific fitted thresholds.
The native lower-is-healthier reliability fit operates in raw queue units:

`z = theta / max(queue_len, 1e-6)`, capped at `10.0`.

| model | fitted theta | support | healthy attainment | windows | cells |
|---|---:|---:|---:|---:|---:|
| dsqwen-7b | 78.333333 | 2,860 | 0.903147 | 3,113 | 59 |
| dsllama-8b | 46.333333 | 1,767 | 0.923033 | 2,621 | 51 |
| dsqwen-14b | 191.666667 | 2,905 | 0.918072 | 3,045 | 56 |

All three fits publish, pass the multi-family coverage gate, exceed the 0.90 reliability
target, and are model-distinct. The 14B threshold is materially higher, consistent with its
TP=2 queue occupancy scale; this is why a shared queue threshold was not a fair ablation.

`queue_len_reliability.svg` plots healthy attainment versus raw queue threshold for every
model. The dashed horizontal line is the 0.90 target and the marked points are the selected
thresholds. The underlying values are committed in `curves/`.

## Inputs

The fit uses the final supplemented R3 window CSV for each currently published model theta.
Each scenario's first ramp window is removed (`trim_ramp_windows=1`).

| model | input on node10 | SHA-256 |
|---|---|---|
| dsqwen-7b | `/root/tre-experiments/r3_7b_slide_convprobe2.csv` | `91674fa707d9d9576725f0e7f4736f09edd29b4ad9cfd9bb725680f9a4e39584` |
| dsllama-8b | `/root/tre-experiments/r3_llama_slide_supp.csv` | `49efbc466cbb8b61e0c5800140b86a56c6cb17992589fe5fa1629acc8e2a99c4` |
| dsqwen-14b | `/root/tre-experiments/r3_14b_slide_supp3.csv` | `98df06e2e59c16706cdaadc2732619e7d872fb7621158d4cf954aac27f28150f` |

Fit-code SHA: `6ee1899b98e0b74210fde47d02d33f7559dd06a5`.

## Regeneration

From `tre/` on node10:

```bash
PYTHONPATH=common:calibration python3 calibration/scripts/fit_alt_thresholds.py \
  --registry deploy/registry.yaml \
  --signal queue_len \
  --model-input dsqwen-7b=/root/tre-experiments/r3_7b_slide_convprobe2.csv \
  --model-input dsllama-8b=/root/tre-experiments/r3_llama_slide_supp.csv \
  --model-input dsqwen-14b=/root/tre-experiments/r3_14b_slide_supp3.csv \
  --trim-ramp-windows 1 \
  --generated-at 2026-07-10T00:00:00+08:00 \
  --output ../docs/refactor/p11_evidence/alt_thresholds_fit_20260713/queue_len_fit.yaml \
  --curve-dir ../docs/refactor/p11_evidence/alt_thresholds_fit_20260713/curves \
  --plot-output ../docs/refactor/p11_evidence/alt_thresholds_fit_20260713/queue_len_reliability.svg
```

## Application State

The exact queue thresholds are present in `tre/deploy/registry.yaml` and the bootstrap
ConfigMap payload. The live `/api/params` PUT and controller restart are intentionally deferred
to the single merged Phase 3 rollout, so current cluster behavior is unchanged.

Verification at the fit-code SHA: authoritative `make check` passed 478 tests.
