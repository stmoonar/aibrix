# PreServe mLSTM predictors

Created 2026-11-03T07:52:47+00:00; code 1c661670f2d27d0e8566a3374b47439e72c18bf2 (tre_preserve/train dirty: False); torch 2.13.0+cu130, numpy 2.2.6, python 3.10.12.

Window W = 10 s (real time). seq_len 25, hidden 64, epochs 50, batch 32, Adam lr 0.001, seed 42. upstream PreServe mLSTM (mLSTM_predict.py): 1 mLSTMCell + Linear(H->1); m computed but unused in the gates (effectively LSTM), kept as a faithful copy.

## Data

| stream | file sha256 | windows | excluded (UTC) | excluded windows | train windows | held-out windows | train sequences | held-out targets | held-out from |
|---|---|---|---|---|---|---|---|---|---|
| code | `71de5c55cbc3…` | 60480 | 2024-05-11T00:00:00+00:00 – 2024-05-11T00:30:00+00:00 | 180 | 48204 | 12096 | 48154 | 12096 | 2024-05-15T14:24:00+00:00 |
| conv | `a0cc9b969a9b…` | 60480 | 2024-05-13T09:00:00+00:00 – 2024-05-13T11:00:00+00:00 | 720 | 47664 | 12096 | 47614 | 12096 | 2024-05-17T14:24:00+00:00 |

## Protocol paper_past (train on the past, predict the future)

The per-stream split columns above are not used; each set's split:

| set | stream | rule | train until (UTC) | train windows | train sequences | held-out targets |
|---|---|---|---|---|---|---|
| code | code | windows before 2024-05-11T00:00:00+00:00 (Real_code slices) | 2024-05-11T00:00:00+00:00 | 8640 | 8615 | 51635 |
| conv | conv | windows before 2024-05-13T09:00:00+00:00 (Real_conv slices) | 2024-05-13T09:00:00+00:00 | 11880 | 11855 | 47855 |
| pooled | code | chronological 1:1 (first half) | 2024-05-13T12:00:00+00:00 | 30060 | 30010 | 30240 |
| pooled | conv | chronological 1:1 (first half) | 2024-05-15T12:00:00+00:00 | 29520 | 29470 | 30240 |

Deviation: upstream's request-length filter and sub-range selection are not applied: the test slices are unfiltered.

## Held-out MAPE (%)

mean |y - yhat| / y * 100 over windows with y > 0 (upstream train_predictor.py); total = (P + D) vs (Phat + Dhat).

| set | evaluated on | normalization | prompt | response | total |
|---|---|---|---|---|---|
| code | code | per_sequence_mean (shipped) | 17.4 | 28.7 | 17.3 |
| code | code | global_minmax | 23.1 | 36.7 | 23.0 |
| conv | conv | per_sequence_mean (shipped) | 7.3 | 9.2 | 7.3 |
| conv | conv | global_minmax | 7.4 | 9.1 | 7.4 |
| pooled | code | per_sequence_mean (shipped) | 12.8 | 22.1 | 12.7 |
| pooled | conv | per_sequence_mean (shipped) | 7.6 | 9.3 | 7.6 |
| pooled | code | global_minmax | 12.8 | 20.5 | 12.7 |
| pooled | conv | global_minmax | 7.6 | 9.3 | 7.6 |

Upstream reference (total MAPE, their windows): code 5 s 17.8 / 600 s 9.2; conv 5 s 10.4 / 600 s 4.6.

numpy vs torch (shipped models, max relative diff on held-out histories): code 1.6e-07, conv 1.8e-07, pooled 1.4e-07.
