# Fixed-APA rerun of the canonical E1 APA arm (2026-07-12)

Re-execution of all nine `t*_apa_seed1` runs after the APA baseline fix
(`fix(apa-baseline): scope anchor selectors to awake pods via routable label`; live canary
in `../apa_fix_canary_20260712/`). FROZEN_SHA
`1457e54ae20da445eb1e11aef7c17a0ce884f612`; runtime images unchanged from
`tre-v2-*:20260710-ca61e485` (the diff from the E1 freeze is campaigns/docs/baseline
yaml/analysis scripts only, so in-cluster runtime is identical). Same traces, seed
20260715, node9 1/1/1 exact reset, 600 s cooldown, trim=1, gateway 31592 for the APA arm.
Manifest: `deploy/campaigns/apa_rerun_e1.json`.

These runs SUPERSEDE the `t*_apa_seed1` directories in `../canonical_rerun_20260715/`
(see the invalidation notice there): those measured a baseline that could never scale.
The TRE arm of canonical E1 is unaffected and is compared against directly below.

## Audit

Full-queue audit BAD=[]: 26 artifacts per run, gzip+uncompressed SHA and line counts
match, trim=1, freeze SHA and params everywhere, stderr empty, all signals finite,
node9 reset verified per run. `SHA256SUMS` pins every artifact.

## APA now actuates

The fixed APA scaled on 4/9 traces: t1/t2/t7 (llama 1->5) and t8 (7b 1->3 plus
llama 1->5). On t3/t4/t5/t9 it legitimately never scaled: `gpu_cache_usage_perc`
stayed below the 0.5 target across the awake pods even while queue-driven SLO
violations ran as high as 47-55% — KVCache utilization is a poor proxy for
queue-driven violations, now demonstrated with a functional baseline rather than a
broken one. Notably t5 (tp_pressure): the 14b KV signal never crossed target while
TRE's Z_m path scaled 14b to 3 and cut V_req to 31.2% vs APA's 55.0%.

## Rebuilt E1 verdict (TRE from ../canonical_rerun_20260715, APA from this rerun)

| trace | TRE V_req | APA v2 V_req | APA old (broken) | diff (APAv2-TRE) | winner |
| --- | ---: | ---: | ---: | ---: | --- |
| t1 | 43.300% | 48.811% | 55.810% | +5.510pp | TRE |
| t2 | 17.224% | 32.451% | 39.885% | +15.227pp | TRE |
| t3 | 0.616% | 47.536% | 48.163% | +46.920pp | TRE |
| t4 | 6.003% | 19.957% | 19.904% | +13.955pp | TRE |
| t5 | 31.152% | 55.021% | 55.025% | +23.869pp | TRE |
| t6 | 0.004% | 0.013% | 0.004% | +0.009pp | tie (noise-level) |
| t7 | 38.818% | 45.323% | 52.850% | +6.504pp | TRE |
| t8 | 0.850% | 36.532% | 67.707% | +35.682pp | TRE |
| t9 | 0.000% | 7.698% | 7.663% | +7.698pp | TRE |

TRE leads on all nine traces (8 clear wins; t6 is noise-level on both arms,
0.004% vs 0.013%). The fixed APA improves materially where it does scale
(t8 67.7% -> 36.5%, t1 55.8% -> 48.8%, t2 39.9% -> 32.5%, t7 52.9% -> 45.3%) and the
TRE margin narrows accordingly — these, not the broken-baseline numbers, are the
citable comparisons.

## Campaign ledger

| run | seed | start | end | params hash | verdict | evidence |
| --- | ---: | --- | --- | --- | --- | --- |
| `t1_apa_seed1` | 20260715 | 2026-07-11T22:04:05Z | 2026-07-11T22:17:58Z | `328bcfd2b54ca2db` | DONE, V_req=48.811%, peaks 1/5/1 | `t1_apa_seed1/` |
| `t2_apa_seed1` | 20260715 | 2026-07-11T22:29:48Z | 2026-07-11T22:49:01Z | `328bcfd2b54ca2db` | DONE, V_req=32.451%, peaks 1/5/1 | `t2_apa_seed1/` |
| `t3_apa_seed1` | 20260715 | 2026-07-11T23:00:53Z | 2026-07-11T23:19:34Z | `328bcfd2b54ca2db` | DONE, V_req=47.536%, peaks 1/1/1 | `t3_apa_seed1/` |
| `t4_apa_seed1` | 20260715 | 2026-07-11T23:31:28Z | 2026-07-11T23:44:21Z | `328bcfd2b54ca2db` | DONE, V_req=19.957%, peaks 1/1/1 | `t4_apa_seed1/` |
| `t5_apa_seed1` | 20260715 | 2026-07-11T23:56:11Z | 2026-07-12T00:11:45Z | `328bcfd2b54ca2db` | DONE, V_req=55.021%, peaks 1/1/1 | `t5_apa_seed1/` |
| `t6_apa_seed1` | 20260715 | 2026-07-12T00:23:36Z | 2026-07-12T00:38:34Z | `328bcfd2b54ca2db` | DONE, V_req=0.013%, peaks 1/1/1 | `t6_apa_seed1/` |
| `t7_apa_seed1` | 20260715 | 2026-07-12T00:50:26Z | 2026-07-12T01:09:40Z | `328bcfd2b54ca2db` | DONE, V_req=45.323%, peaks 1/5/1 | `t7_apa_seed1/` |
| `t8_apa_seed1` | 20260715 | 2026-07-12T01:21:32Z | 2026-07-12T01:40:47Z | `328bcfd2b54ca2db` | DONE, V_req=36.532%, peaks 3/5/1 | `t8_apa_seed1/` |
| `t9_apa_seed1` | 20260715 | 2026-07-12T01:52:40Z | 2026-07-12T02:11:57Z | `328bcfd2b54ca2db` | DONE, V_req=7.698%, peaks 1/1/1 | `t9_apa_seed1/` |
