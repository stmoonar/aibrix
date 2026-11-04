# Audit: traces-icse-r3e-20261101 (audit_r3e.py, generated files; Real_* round 3)

Calibrated demand model (r3c_model KAPPA real_code 7b 0.85 / 8b 0.855 / 14b 1.2; real_conv 1.105 / 1.045 / 1.2; knee = request violation > 5 % at TTFT 500 / TPOT 75 ms on observed runs, `_checks/cap_calib.txt`). 30 s windows, 10 s step, from t = 30 s. Gate definitions: audit_r3e.py docstring. 'old kappa' = the r3d model (Real 0.85 / 0.95 / 1.0) for reference only.

## Gates (full 720 s | first 400 s)

| set | trace | best fixed full layout over knee: full / 400 s | rho* <= 1 share full / 400 | oracle moves U.85 full / 400 (knee oracle) | max rise (limit) | load cap | verdict |
|---|---|---|---|---|---|---|---|
| r3e | Real_code_2024_slice_a_tok70 | 1/3/2 0.343 PASS / 3/1/2 0.057 FAIL | 1.0 / 1.0 PASS | 2 FAIL / 1 FAIL (0 / 0) | 1.78 (2.16) PASS | PASS | 4/7 |
| r3e | Real_conv_2023_slice_a_tok70 | 1/3/2 0.433 PASS / 3/1/2 0.114 FAIL | 1.0 / 1.0 PASS | 3 FAIL / 3 PASS (0 / 0) | 1.79 (1.8) PASS | PASS | 5/7 |
| r3d | Real_code_2024_slice_a_tok70 | 3/3/1 0.0 FAIL / 3/3/1 0.0 FAIL | 1.0 / 1.0 PASS | 2 FAIL / 0 FAIL (0 / 0) | 1.54 (2.16) PASS | FAIL | 2/7 |
| r3d | Real_conv_2023_slice_a_tok70 | 2/2/2 0.0 FAIL / 2/2/2 0.0 FAIL | 1.0 / 1.0 PASS | 0 FAIL / 0 FAIL (0 / 0) | 2.11 (1.8) FAIL | PASS | 2/7 |
| ktrace | Real_code_2024_slice_a_tok70 | 3/3/1 0.0 FAIL / 3/3/1 0.0 FAIL | 1.0 / 1.0 PASS | 0 FAIL / 0 FAIL (0 / 0) | 1.29 (2.16) PASS | FAIL | 2/7 |
| ktrace | Real_conv_2023_slice_a_tok70 | 1/3/2 0.0 FAIL / 1/3/2 0.0 FAIL | 1.0 / 1.0 PASS | 0 FAIL / 0 FAIL (0 / 0) | 1.22 (1.8) PASS | PASS | 3/7 |

## Load vs references (total req/s and tokens/s: mean over [30, 720) / peak 30 s window)

Code reference = the ICSE Real_code x1.25 version (traces-icse-ktrace-r2-20261031; APA 8 % client failures), cap = 1.0x; conv reference = ICSE Real_conv x1 (traces-icse-20261030), cap = 1.6x. ICSE x1.6 code (APA 21 % failures) shown for scale.

| set | trace | req/s mean / peak | tok/s mean / peak | ratio to reference (rps mean / peak, tok mean / peak) | first 400 s req/s mean / peak |
|---|---|---|---|---|---|
| r3e | Real_code_2024_slice_a_tok70 | 64.39 / 79.17 | 30320.1 / 37314.7 | 0.867 / 0.995, 0.871 / 0.981 | 72.56 / 79.17 |
| r3e | Real_conv_2023_slice_a_tok70 | 73.01 / 94.13 | 32665.8 / 42049.9 | 0.727 / 0.842, 0.736 / 0.854 | 81.34 / 94.13 |
| r3d | Real_code_2024_slice_a_tok70 | 79.43 / 101.5 | 37588.4 / 48309.4 | 1.069 / 1.276, 1.079 / 1.27 | 82.51 / 101.5 |
| r3d | Real_conv_2023_slice_a_tok70 | 62.63 / 78.23 | 28947.3 / 35761.0 | 0.624 / 0.7, 0.652 / 0.727 | 59.89 / 78.23 |
| ktrace | Real_code_2024_slice_a_tok70 | 95.25 / 101.87 | 44828.3 / 48682.7 | 1.282 / 1.28, 1.287 / 1.28 | 96.23 / 101.73 |
| ktrace | Real_conv_2023_slice_a_tok70 | 62.76 / 69.87 | 27749.9 / 30771.7 | 0.625 / 0.625, 0.625 / 0.625 | 64.1 / 69.87 |
| ref | code | 74.27 / 79.57 | 34826.4 / 38030.8 | 1 | |
| ref | conv | 62.76 / 69.87 | 27750.4 / 30757.2 | 1 | |
| ref | code_x1.6 | 95.25 / 101.87 | 44828.3 / 48682.7 | 1 | |

## Old kappa (r3d model) for reference: best-by-worst static layout / over share, oracle moves, oracle pool-full, rho* peak, need mean

| set | trace | static | moves | pool full | rho* peak | need mean |
|---|---|---|---|---|---|---|
| r3e | Real_code_2024_slice_a_tok70 | 2/2/2 0.448 | 2 | 0.478 | 1.094 | 7.12 |
| r3e | Real_conv_2023_slice_a_tok70 | 2/2/2 0.851 | 3 | 0.522 | 1.186 | 8.03 |
| r3d | Real_code_2024_slice_a_tok70 | 2/2/2 0.045 | 3 | 0.537 | 0.932 | 7.67 |
| r3d | Real_conv_2023_slice_a_tok70 | 2/2/2 0.045 | 3 | 0.254 | 0.936 | 6.81 |
| ktrace | Real_code_2024_slice_a_tok70 | 3/3/1 0.0 | 0 | 0.791 | 0.922 | 7.99 |
| ktrace | Real_conv_2023_slice_a_tok70 | 2/2/2 0.0 | 0 | 0.612 | 0.659 | 7.22 |

## Fixed full-pool layouts (7b/8b/14b): share of windows over the knee, full / first 400 s (calibrated)

| set | trace | 2/2/2 | 3/3/1 | 3/1/2 | 1/3/2 | 4/2/1 | 2/4/1 | 1/1/3 |
|---|---|---|---|---|---|---|---|---|
| r3e | Real_code_2024_slice_a_tok70 | 0.433 / 0.429 | 0.403 / 0.686 | 0.463 / 0.057 | 0.343 / 0.657 | 0.612 / 0.686 | 0.522 / 0.914 | 0.746 / 0.6 |
| r3e | Real_conv_2023_slice_a_tok70 | 0.552 / 0.543 | 0.552 / 0.857 | 0.463 / 0.114 | 0.433 / 0.8 | 0.806 / 0.857 | 0.582 / 0.914 | 0.746 / 0.686 |
| r3d | Real_code_2024_slice_a_tok70 | 0.149 / 0.2 | 0.0 / 0.0 | 1.0 / 1.0 | 0.896 / 1.0 | 0.149 / 0.2 | 0.045 / 0.086 | 1.0 / 1.0 |
| r3d | Real_conv_2023_slice_a_tok70 | 0.0 / 0.0 | 0.0 / 0.0 | 0.955 / 0.914 | 0.06 / 0.114 | 0.0 / 0.0 | 0.0 / 0.0 | 0.955 / 0.914 |
| ktrace | Real_code_2024_slice_a_tok70 | 0.761 / 0.886 | 0.0 / 0.0 | 1.0 / 1.0 | 1.0 / 1.0 | 0.612 / 0.771 | 0.343 / 0.371 | 1.0 / 1.0 |
| ktrace | Real_conv_2023_slice_a_tok70 | 0.0 / 0.0 | 0.0 / 0.0 | 0.91 / 0.886 | 0.0 / 0.0 | 0.0 / 0.0 | 0.0 / 0.0 | 0.91 / 0.886 |

## Orientation sims (late % / GPUs mean / pool full % / moves), full | first 400 s late %

| set | trace | static best | free_only | donor | first 400: static / free_only / donor |
|---|---|---|---|---|---|
| r3e | Real_code_2024_slice_a_tok70 | 34.8 / 8.0 / 100.0 / 0 | 1.2 / 6.7 / 35.7 / 0 | 0.0 / 6.92 / 39.4 / 23 | 44.3 / 1.9 / 0.0 |
| r3e | Real_conv_2023_slice_a_tok70 | 37.8 / 8.0 / 100.0 / 0 | 0.4 / 7.02 / 46.7 / 0 | 0.0 / 7.18 / 48.6 / 11 | 47.6 / 0.7 / 0.0 |
| r3d | Real_code_2024_slice_a_tok70 | 0.0 / 8.0 / 100.0 / 0 | 0.1 / 6.86 / 29.6 / 0 | 0.0 / 7.52 / 64.6 / 27 | 0.0 / 0.1 / 0.0 |
| r3d | Real_conv_2023_slice_a_tok70 | 0.0 / 8.0 / 100.0 / 0 | 0.7 / 5.49 / 2.1 / 0 | 0.3 / 6.26 / 18.5 / 3 | 0.0 / 1.3 / 0.5 |
| ktrace | Real_code_2024_slice_a_tok70 | 0.3 / 8.0 / 100.0 / 0 | 0.6 / 7.8 / 81.9 / 0 | 0.3 / 7.94 / 97.5 / 32 | 0.6 / 1.0 / 0.6 |
| ktrace | Real_conv_2023_slice_a_tok70 | 3.2 / 8.0 / 100.0 / 0 | 0.3 / 6.04 / 13.5 / 0 | 0.0 / 7.76 / 88.9 / 2 | 3.2 / 0.3 / 0.0 |

## Per model

| set | trace | requests 7b/8b/14b | in mean/p50/p95 | out mean/p50/p95 | rho peak (calibrated) | rise |
|---|---|---|---|---|---|---|
| r3e | Real_code_2024_slice_a_tok70 | 16132/14829/15044 | 382.0/351/684; 395.5/359/692; 386.5/352/682 | 81.3/79/91; 81.4/81/85; 86.3/85/90 | [2.07, 2.11, 2.0] | [1.61, 1.37, 1.78] |
| r3e | Real_conv_2023_slice_a_tok70 | 19592/16716/15873 | 336.0/299/499; 331.2/289/513; 324.3/283/496 | 112.5/92/221; 116.0/94/232; 121.7/100/237 | [2.09, 2.13, 2.02] | [1.79, 1.74, 1.28] |
| r3d | Real_code_2024_slice_a_tok70 | 23250/22567/11096 | 385.7/349/694; 400.6/355/694; 384.9/347/683 | 80.6/79/88; 81.7/81/91; 86.3/85/90 | [2.08, 2.14, 0.96] | [1.39, 1.54, 1.16] |
| r3d | Real_conv_2023_slice_a_tok70 | 14016/21438/9178 | 346.1/297/544; 345.8/300/555; 333.3/285/530 | 114.5/92/242; 118.8/94/252; 125.9/100/263 | [1.39, 1.84, 0.91] | [2.11, 1.51, 1.23] |
| ktrace | Real_code_2024_slice_a_tok70 | 30048/29272/9444 | 394.3/359/687; 388.7/354/676; 377.9/341/675 | 78.9/78/90; 81.3/81/92; 86.2/86/98 | [2.27, 2.37, 0.77] | [1.18, 1.19, 1.29] |
| ktrace | Real_conv_2023_slice_a_tok70 | 16944/17360/10940 | 341.0/311/529; 333.1/302/523; 319.6/288/512 | 105.8/88/206; 109.6/91/211; 114.0/95/222 | [0.99, 1.2, 0.95] | [1.21, 1.17, 1.22] |

## Series (r3e, 30 s windows every 10 s from t = 30 s; calibrated model)

- Real_code_2024_slice_a_tok70 need (U 0.85): [7, 7, 6, 6, 6, 8, 8, 8, 8, 8, 8, 8, 6, 6, 8, 8, 8, 8, 8, 7, 9, 9, 9, 9, 9, 9, 9, 9, 8, 8, 9, 9, 9, 9, 9, 8, 7, 7, 7, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6]
  - oracle layout (7b/8b/14b): 2/1/2 2/1/2 3/1/1 3/1/1 3/1/1 3/1/2 3/1/2 3/1/2 3/1/2 3/1/2 3/1/2 3/1/2 3/1/1 3/1/1 3/1/2 3/1/2 3/1/2 3/1/2 3/1/2 2/1/2 2/1/2 2/1/2 1/1/3 1/1/3 1/1/3 1/1/3 1/1/3 1/1/3 1/1/3 1/1/3 1/1/3 1/1/3 1/1/3 1/1/3 1/1/3 1/1/3 1/2/2 1/2/2 1/2/2 1/2/1 1/2/1 1/2/1 1/2/1 1/2/1 1/2/1 1/2/1 1/2/1 1/2/1 1/2/1 1/3/1 1/3/1 1/3/1 1/3/1 1/3/1 1/3/1 1/3/1 1/3/1 1/3/1 1/3/1 1/3/1 1/3/1 1/3/1 1/3/1 1/3/1 1/3/1 1/3/1 1/3/1
  - oracle move times (s): [250, 390]
- Real_conv_2023_slice_a_tok70 need (U 0.85): [8, 7, 7, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 9, 9, 9, 9, 9, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 6, 7, 7, 5, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 7, 7, 7, 6, 6, 6, 6, 6]
  - oracle layout (7b/8b/14b): 2/2/2 3/2/1 3/2/1 3/1/2 3/1/2 3/1/2 3/1/2 3/1/2 3/1/2 3/1/2 3/1/2 3/1/2 3/1/2 3/1/2 3/1/2 3/1/2 3/1/2 3/1/2 3/1/2 3/1/2 2/1/2 2/1/2 2/1/2 2/1/2 1/1/3 1/1/3 1/1/3 1/1/3 1/1/3 1/1/3 1/1/3 1/1/3 1/1/3 1/1/3 1/1/3 1/1/3 1/1/3 1/1/3 1/1/3 1/1/2 1/2/2 1/2/2 1/2/1 1/2/1 1/2/1 1/2/1 1/2/1 1/2/1 1/3/1 1/3/1 1/3/1 1/3/1 1/3/1 1/3/1 1/3/1 1/3/1 1/3/1 1/3/1 1/3/1 2/3/1 2/3/1 2/3/1 1/3/1 1/3/1 1/3/1 1/3/1 1/3/1
  - oracle move times (s): [40, 60, 270]
