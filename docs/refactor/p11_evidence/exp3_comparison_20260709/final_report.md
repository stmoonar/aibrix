# Experiment 3 - TRE vs APA full 7x2 (traceset-v2)

- **Status**: DONE 2026-07-09T01:49:47Z; all 14 runs passed the infra gate (UF=0, SOCKFAIL=0).
- **Traceset**: traceset-v2 (commit 04de30a1, integer-feasible GPU occupancy, total_slots=8). Model slot widths: dsqwen-7b=1, dsllama-8b=1, dsqwen-14b=2 (total 8).
- **SLO** (registry): ttft_p95 500ms, tpot_p95 75ms, e2e_p95 12000ms (7b/llama) / 15000ms (14b).
- **Scoring**: `tre_replayer.scoring.compute_v_sys` (30s window / 5s step / min_samples 5). Recomputed per-model values match run.log summaries exactly.
- **Terminal state**: controller=observe, signal_source=zm (TRE active), 0 PodAutoscaler CRs, routable/awake 1/1/1, console 200.

> **APA scaling caveat**: APA scales via AIBrix PodAutoscaler (KVCache) on separate anchor deployments; the timeline `awake` column is the TRE observer view and reads 1/1/1 + `NA` for APA - it does NOT reflect APA replica count. APA behaviour below is read from its 503-over-time signal (authoritative JSONL), not the awake column.

## 1. Comparison table (system-level, request-weighted)

Lower is better for every metric. delta = TRE - APA (negative => TRE better).

| Trace | Axis | Tier | Metric | TRE | APA | delta(TRE-APA) | Verdict |
|---|---|---|---|---|---|---|---|
| t1 | A1 demand_shift | tight(1.0) | V_req (violation req frac) | 42.72% | 55.48% | -12.76pp | TRE |
|  |  |  | V_time (violation window frac) | 32.35% | 33.89% | -1.54pp | TRE |
|  |  |  | 503 unload rate | 23.40% | 35.86% | -12.46pp | TRE |
| t2 | A2 anticorrelated | loose | V_req (violation req frac) | 20.01% | 40.22% | -20.21pp | TRE |
|  |  |  | V_time (violation window frac) | 33.17% | 41.62% | -8.45pp | TRE |
|  |  |  | 503 unload rate | 5.08% | 16.73% | -11.65pp | TRE |
| t3 | A3 io_drift | loose | V_req (violation req frac) | 2.87% | 50.14% | -47.27pp | TRE |
|  |  |  | V_time (violation window frac) | 9.82% | 52.80% | -42.98pp | TRE |
|  |  |  | 503 unload rate | 0.01% | 16.33% | -16.32pp | TRE |
| t4 | A4 spike_vs_burst | loose | V_req (violation req frac) | 8.83% | 19.48% | -10.65pp | TRE |
|  |  |  | V_time (violation window frac) | 13.54% | 17.48% | -3.94pp | TRE |
|  |  |  | 503 unload rate | 0.27% | 7.28% | -7.01pp | TRE |
| t5 | A5 tp_pressure | tight(1.0) | V_req (violation req frac) | 55.34% | 55.33% | +0.01pp | tie |
|  |  |  | V_time (violation window frac) | 51.08% | 50.09% | +0.99pp | APA |
|  |  |  | 503 unload rate | 26.58% | 26.56% | +0.03pp | tie |
| t6 | A6 control | control | V_req (violation req frac) | 0.01% | 0.01% | +0.00pp | tie |
|  |  |  | V_time (violation window frac) | 3.26% | 2.36% | +0.90pp | APA |
|  |  |  | 503 unload rate | 0.01% | 0.01% | +0.00pp | tie |
| t7 | A2b anticorrelated_hot | tight(1.0) | V_req (violation req frac) | 40.46% | 52.93% | -12.47pp | TRE |
|  |  |  | V_time (violation window frac) | 43.23% | 44.46% | -1.23pp | TRE |
|  |  |  | 503 unload rate | 22.90% | 33.80% | -10.90pp | TRE |

### Per-model p95 latency (200-only) and 503 count

| Trace | Model | TRE p95 ttft/e2e (ms) | APA p95 ttft/e2e (ms) | TRE 503 | APA 503 |
|---|---|---|---|---|---|
| t1 | dsqwen-7b | 11194.56 / 25025.69 | 22293.74 / 48548.04 | 3392 | 6104 |
| t1 | dsllama-8b | 29846.46 / 60658.73 | 33527.9 / 64675.29 | 2267 | 2567 |
| t1 | dsqwen-14b | 97.0 / 2221.24 | 61.53 / 2192.15 | 0 | 0 |
| t2 | dsqwen-7b | 4291.32 / 17327.99 | 17005.97 / 44070.8 | 214 | 4196 |
| t2 | dsllama-8b | 26829.34 / 58985.78 | 26603.93 / 58935.17 | 1481 | 1386 |
| t2 | dsqwen-14b | 103.52 / 2222.01 | 66.44 / 2195.4 | 0 | 0 |
| t3 | dsqwen-7b | 186.07 / 9988.48 | 18108.44 / 41861.25 | 1 | 4888 |
| t3 | dsllama-8b | 68.76 / 2061.78 | 62.58 / 2060.15 | 1 | 0 |
| t3 | dsqwen-14b | 63.45 / 2188.28 | 56.01 / 2186.57 | 0 | 3 |
| t4 | dsqwen-7b | 7708.01 / 22773.86 | 11451.94 / 36182.17 | 41 | 1096 |
| t4 | dsllama-8b | 95.12 / 4225.91 | 77.83 / 4172.31 | 0 | 0 |
| t4 | dsqwen-14b | 62.31 / 2187.33 | 49.95 / 2179.87 | 0 | 0 |
| t5 | dsqwen-7b | 64.29 / 1954.88 | 63.68 / 1953.78 | 0 | 1 |
| t5 | dsllama-8b | 69.38 / 2160.74 | 68.31 / 2163.42 | 2 | 1 |
| t5 | dsqwen-14b | 17777.01 / 35833.58 | 17725.1 / 35718.54 | 8321 | 8313 |
| t6 | dsqwen-7b | 54.88 / 2141.73 | 54.47 / 2141.39 | 0 | 1 |
| t6 | dsllama-8b | 61.07 / 2358.17 | 60.32 / 2355.84 | 3 | 1 |
| t6 | dsqwen-14b | 48.84 / 2231.7 | 49.17 / 2232.48 | 0 | 0 |
| t7 | dsqwen-7b | 10131.71 / 24523.95 | 18847.63 / 44828.18 | 5407 | 10058 |
| t7 | dsllama-8b | 30452.8 / 61695.86 | 31234.98 / 61613.68 | 4323 | 4307 |
| t7 | dsqwen-14b | 119.24 / 2235.57 | 70.36 / 2202.74 | 1 | 0 |

## 2. Oracle-normalized score (TRE vs APA baseline)

V_oracle=0 by construction (INDEX feasibility + lint guard). Score=(V_apa-V_tre)/V_apa on request-frac; 1.0 closes the whole APA->oracle gap, 0 = no better than APA, <0 = worse than APA.

| Trace | TRE V_req | APA V_req | Oracle-norm score |
|---|---|---|---|
| t1 | 0.427 | 0.555 | 0.230 |
| t2 | 0.200 | 0.402 | 0.502 |
| t3 | 0.029 | 0.501 | 0.943 |
| t4 | 0.088 | 0.195 | 0.547 |
| t5 | 0.553 | 0.553 | -0.000 |
| t6 | 0.000 | 0.000 | degenerate (baseline~0, raw tie) |
| t7 | 0.405 | 0.529 | 0.236 |

## 3. Behavior analysis

Sources: TRE scaling from `timeline.csv` (faithful for TRE only); both arms' 503-over-time from authoritative JSONL (`http_status`).

### Tight traces (t1, t5, t7)

**t1 (A1 demand_shift).** 7b demand jumps at ~113s. TRE fires its first scale at **141s** (rescue loop, reason `critical_sleeping_capacity`), then the fairness loop pushes 7b to **peak 4** replicas by 157s; llama scaled to 2 at 611s. The `qwen7b_saturate` phase actually needs 5 slots (5+1+2 = 8, feasible) but TRE topped out at **4** — one short — leaving residual 7b shedding. 503 timeline: both arms shed 7b through the 120–390s saturation window and both drop to 0 at **420s when the demand phase ends (400s), not because of scaling**. The difference is magnitude: TRE sheds ~260–380 / 30s, APA ~600–730 / 30s (~2x). Net 7b 503: TRE 3392 vs APA 6104; p95 ttft 11.2s vs 22.3s. llama ~tie.

**t7 (A2b anticorrelated_hot).** Two 7b hot waves (120–360s, 600–810s). TRE scaled 7b to 4 at 157s and **held it across both waves** (final awake 7b=4), so wave-2 shedding is ~half APA's (~340/30s vs ~700/30s). Net 7b 503: TRE 5407 vs APA 10058. llama (waves 360–570, 810–1050s) ~tie.

**t5 (A5 tp_pressure) — ANOMALY / tie.** The 2-slot 14b saturates. **Neither arm scaled 14b**: TRE awake stayed 1/1/1, and APA's 14b 503 curve is *nearly identical* to TRE's (e.g. 407/413, 745/762 per 30s). Both shed ~8.3k on 14b; system tie (TRE V_req 0.553 vs APA 0.553; V_time 0.511 vs 0.501, TRE marginally worse = noise). TRE *had* 4 free slots (7b+llama+14b = 1+1+2 of 8) to place a 2nd 14b replica but did not. No scale-freeze exists in the controller env (`ENABLE_TRE_SCALING=true`, no per-model gate). **待深挖**: why TRE's fast/slow loop never scaled the 2-slot 14b under TP pressure — candidate causes are 2-GPU contiguous placement gating, or the 14b Z_m/theta_m signal not crossing `delta_high`. Not treated as explained.

### t3 (A3 io_drift) — rate-signal-lag hypothesis CONFIRMED

RPS is held constant while per-request output grows heavier over the run. TRE scaled 7b to 4 at **172s** proactively (TSS caught the rising work-per-request before queues blew up); 7b 503 is **~0 across the whole run (1 total)**. APA's KVCache signal lagged: 7b 503 onset at **330s** and ramps **monotonically 15 → 99 → 130 → … → 291 per 30s** through the entire heavy-output phase, never recovering (demand never recedes). Net 7b 503: TRE **1** vs APA **4888**; system V_req 2.9% vs 50.1%; oracle-norm 0.943. This is the sharpest single-trace win and the cleanest validation that a throughput/queue signal (TSS) leads a cache-utilization signal (KVCache) when work-per-request drifts under constant RPS.

### t6 (A6 control) — fairness / 打平 confirmed

Neither model needs scaling; both arms sit at 1/1/1, V_req ~0.0001 both, 503 ~2–3 total each. TRE's V_time (3.26%) vs APA (2.36%) differ only by a handful of floor-level 503s (TRE llama 3 vs APA 1) each flipping one window; not meaningful. Both arms tie at the feasible-at-rest optimum — TRE imposes no fairness/overhead penalty when scaling is unnecessary.

### Loose traces (t2, t4)

TRE scaled 7b (peak 4 / 3) and cut 7b 503 by ~20x (t2: 214 vs 4196) / ~27x (t4: 41 vs 1096); llama ~tie. TRE wins both on every headline metric.

## 4. Conclusions

**TRE beats APA on 5 of 7 traces (t1, t2, t3, t4, t7), ties the control (t6), and ties on t5 (where both fail).**

- **Where TRE wins and why**: every win is driven by TRE **proactively scaling the hot 1-slot model (7b) via the TSS / Z_m signal**, cutting 7b 503-shedding from ~2x (tight t1/t7) to 20–4888x (loose/drift t2/t4/t3) and roughly halving 7b p95 latency. APA's KVCache-driven PodAutoscaler reacts later and adds less capacity for the same demand.
- **Sharpest axis (A3, t3)**: constant-RPS output drift is where TSS's lead over KVCache is largest — TRE 1 vs APA 4888 7b 503s (oracle-norm 0.943). Confirms the design thesis.
- **Fairness (t6)**: exact tie at the at-rest optimum; TRE adds no penalty when no scaling is needed.
- **Where TRE does NOT win (t5, tp_pressure)**: both arms fail to scale the 2-slot 14b and tie (~0.55 V_req). This is a **real TRE gap, not an APA win** — TRE had free capacity and did not use it. Flagged 待深挖.
- **Honest caveat on tight-trace "recovery"**: in t1/t7 both arms recover simultaneously when the demand phase ends, so TRE's measured advantage is *lower loss during* saturation, not *faster recovery*.

### Leftover / open items

1. **t5 14b (2-slot) never scaled by either arm** — investigate TRE's 2-GPU placement / signal path (待深挖).
2. **t1 7b topped out at 4 replicas where 5 is feasible** (8 slots) — one-short under-scale, minor residual 7b shedding.
3. **envoy_5xx_permodel_minute.csv are partial captures** (e.g. tre/t1 = 9029 of 24182 lines); all 503 analysis used the authoritative JSONL. CSVs kept only as supplementary evidence.
