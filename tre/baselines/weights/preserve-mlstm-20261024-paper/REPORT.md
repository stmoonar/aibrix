# PreServe mLSTM predictors

Created 2026-10-24T13:50:45+00:00; code 80571402c94b326e3e7d34fbdbbdacfc3a791784 (tre_preserve/train dirty: False); torch 2.13.0+cu130, numpy 2.2.6, python 3.10.12.

Window W = 30 s (real time). seq_len 25, hidden 64, epochs 50, batch 32, Adam lr 0.001, seed 42. upstream PreServe mLSTM (mLSTM_predict.py): 1 mLSTMCell + Linear(H->1); m computed but unused in the gates (effectively LSTM), kept as a faithful copy.

## Data

| stream | file sha256 | windows | excluded (UTC) | excluded windows | train windows | held-out windows | train sequences | held-out targets | held-out from |
|---|---|---|---|---|---|---|---|---|---|
| code | `71de5c55cbc3…` | 20160 | 2024-05-11T00:00:00+00:00 – 2024-05-11T00:30:00+00:00 | 60 | 16068 | 4032 | 16018 | 4032 | 2024-05-15T14:24:00+00:00 |
| conv | `a0cc9b969a9b…` | 20160 | 2024-05-13T09:00:00+00:00 – 2024-05-13T11:00:00+00:00 | 240 | 15888 | 4032 | 15838 | 4032 | 2024-05-17T14:24:00+00:00 |

## Protocol paper_past (train on the past, predict the future)

The per-stream split columns above are not used; each set's split:

| set | stream | rule | train until (UTC) | train windows | train sequences | held-out targets |
|---|---|---|---|---|---|---|
| code | code | windows before 2024-05-11T00:00:00+00:00 (Real_code slices) | 2024-05-11T00:00:00+00:00 | 2880 | 2855 | 17195 |
| conv | conv | windows before 2024-05-13T09:00:00+00:00 (Real_conv slices) | 2024-05-13T09:00:00+00:00 | 3960 | 3935 | 15935 |
| pooled | code | chronological 1:1 (first half) | 2024-05-13T12:00:00+00:00 | 10020 | 9970 | 10080 |
| pooled | conv | chronological 1:1 (first half) | 2024-05-15T12:00:00+00:00 | 9840 | 9790 | 10080 |

Deviation: upstream's request-length filter and sub-range selection are not applied: the test slices are unfiltered.

## Held-out MAPE (%)

mean |y - yhat| / y * 100 over windows with y > 0 (upstream train_predictor.py); total = (P + D) vs (Phat + Dhat).

| set | evaluated on | normalization | prompt | response | total |
|---|---|---|---|---|---|
| code | code | per_sequence_mean (shipped) | 12.9 | 19.2 | 12.9 |
| code | code | global_minmax | 13.6 | 21.8 | 13.6 |
| conv | conv | per_sequence_mean (shipped) | 6.5 | 7.0 | 6.4 |
| conv | conv | global_minmax | 6.7 | 7.1 | 6.6 |
| pooled | code | per_sequence_mean (shipped) | 10.4 | 15.4 | 10.3 |
| pooled | conv | per_sequence_mean (shipped) | 6.8 | 7.4 | 6.8 |
| pooled | code | global_minmax | 10.5 | 14.9 | 10.5 |
| pooled | conv | global_minmax | 6.8 | 7.4 | 6.8 |

Upstream reference (total MAPE, their windows): code 5 s 17.8 / 600 s 9.2; conv 5 s 10.4 / 600 s 4.6.

numpy vs torch (shipped models, max relative diff on held-out histories): code 1.4e-07, conv 1.4e-07, pooled 1.4e-07.
