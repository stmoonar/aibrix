# PreServe mLSTM predictors, W = 10 s

Created 2026-11-03T07:50:40+00:00 by `preserve_train/pipeline.py finish` (aibrix 1c661670f2d27d0e8566a3374b47439e72c18bf2, dirty False). Trained by `preserve_train/train_mlstm.py` (aibrix 1c661670f2d27d0e8566a3374b47439e72c18bf2, preserve_train dirty: False; the v1 trainer tre_code 80571402 with imports adjusted and `window_s` exported, see preserve_train/README.md), torch 2.13.0+cu130, numpy 2.2.6, python 3.10.12, CPU; argv `/data/nfs_shared_data/xxy/aibrix-wt/bl-window10-20261103/tre/baselines/preserve_train/pipeline.py train --train-dir /data/nfs_shared_data/xxy/bl-mlstm-w10-20261103-repo --window-s 10 --azure-root /data/nfs_shared_data/xxy/datasets/azure-llm-inference --file-suffix=-paper-w10 --jobs 12 --threads 1`. Trainer output: `/data/nfs_shared_data/xxy/bl-mlstm-w10-20261103-repo`; its own report is `REPORT.trainer.md` here.

Protocol: the v1 `paper_past` protocol of `preserve-mlstm-20261024-paper` (seq_len, hidden, epochs, batch, lr, seed, normalization, splits, exclusions, dataset files) at W = 10 s. Each file declares `window_s` = 10; the policy refuses to start when it differs from `params.window_s`.

Run notes:

- trained by the repo copy preserve_train (aibrix 1c661670) with the settings of the tre_code 80571402 run of the same day (/data/nfs_shared_data/xxy/bl-mlstm-w10-20261103): every array of the three files is identical to that run (window_s aside, which the tre_code export lacks); window caches and held-out MAPE identical too (/data/nfs_shared_data/xxy/bl-mlstm-w10-20261103-repo/compare_vs_tre_code.json, compare_caches_mape_vs_tre_code.json)
- --jobs 12 (10-24 run: --jobs 2): parallel training processes only; each job trains with 1 torch thread as before; the trainer main process (held-out forward passes) used min(8, jobs x threads) = 8 threads (was 2)
- window caches windows_{code,conv}_10s.npz: pipeline.py windows (no pandas on host 75); it first reproduced both v1 30 s caches of preserve-mlstm-20261024-paper exactly (windows_check_10s.json in the training dir)
- host 75 (as the 10-24 run), CPU, nice 10, taskset 48-63, 2026-11-03 15:38-15:53 (75 clock, about 160 s ahead of 76); git safe.directory via GIT_CONFIG_GLOBAL only

| set | file | sha256 |
|---|---|---|
| code | `code-paper-w10.npz` | `f8b74a2b91cd570e3b9b565ae8ac0fb5d47a7e9366699bf138d40f91509340c2` |
| conv | `conv-paper-w10.npz` | `f21dad6b84f43ff1fc354b35590f0f62ff69e1ab9b7f3f7c66559ca9bc93636b` |
| pooled | `pooled-paper-w10.npz` | `e69c2dc188b46e8fa0368e47ea42962697578613745e6c1621dcbb3728f1d7ad` |

## Held-out MAPE (%): mLSTM (shipped per_sequence_mean) vs naive last window

mean |y - yhat| / y * 100 over windows with y > 0 (upstream train_predictor.py); total = (P + D) vs (Phat + Dhat). Naive: the prediction of window t is the value of window t-1 (same held-out targets).

| W (s) | set | evaluated on | predictor | prompt | response | total | targets |
|---|---|---|---|---|---|---|---|
| 10 | code | code | mLSTM | 17.36 | 28.69 | 17.26 | 51635 |
| 10 | code | code | naive | 19.41 | 34.15 | 19.3 | 51635 |
| 10 | conv | conv | mLSTM | 7.35 | 9.15 | 7.33 | 47855 |
| 10 | conv | conv | naive | 8.29 | 10.68 | 8.26 | 47855 |
| 10 | pooled | code | mLSTM | 12.76 | 22.14 | 12.71 | 30240 |
| 10 | pooled | code | naive | 14.8 | 27.02 | 14.72 | 30240 |
| 10 | pooled | conv | mLSTM | 7.63 | 9.34 | 7.61 | 30240 |
| 10 | pooled | conv | naive | 8.63 | 11.02 | 8.6 | 30240 |
| 30 | code | code | mLSTM | 12.94 | 19.2 | 12.88 | 17195 |
| 30 | code | code | naive | 15.16 | 23.11 | 15.09 | 17195 |
| 30 | conv | conv | mLSTM | 6.46 | 7.05 | 6.44 | 15935 |
| 30 | conv | conv | naive | 7.68 | 8.54 | 7.66 | 15935 |
| 30 | pooled | code | mLSTM | 10.37 | 15.37 | 10.33 | 10080 |
| 30 | pooled | code | naive | 12.14 | 18.74 | 12.1 | 10080 |
| 30 | pooled | conv | mLSTM | 6.8 | 7.42 | 6.78 | 10080 |
| 30 | pooled | conv | naive | 7.98 | 8.9 | 7.96 | 10080 |
