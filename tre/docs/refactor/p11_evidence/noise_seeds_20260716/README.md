# Noise seeds campaign (seeds 20260716/20260717) — t1/t3/t6 x TRE/APA

Two extra seeds per (trace, arm) cell to establish the seed-to-seed noise floor for
E1/E2 margin judgments. FROZEN_SHA `1457e54ae20da445eb1e11aef7c17a0ce884f612` (post
APA fix; runtime images unchanged from 20260710-ca61e485). Seed1 references: TRE arm
`../canonical_rerun_20260715/`, APA arm `../canonical_e1_apa_v2_20260712/` (the fixed
baseline; broken-APA seed1 numbers are not comparable). Manifest:
`deploy/campaigns/noise_seeds.json`. Audit over all 12 runs: BAD=[].

## Noise floor (see `noise_floor.txt` for the full 3-seed table)

- TRE arm: max seed range 1.707pp (t1); t3 1.379pp; t6 negligible.
- APA arm: max seed range 6.681pp (t3); t1 4.925pp — the fixed APA is 3-4x noisier
  than TRE seed-to-seed, because whether/when the KV threshold trips varies by seed.

## Consequences

- E1 verdicts are seed-robust: the 3-seed ranges do not overlap on t1
  (TRE 41.6-43.3% vs APA 48.8-53.7%) or t3 (TRE 0.6-2.0% vs APA 42.7-49.4%);
  t6 is a genuine noise-level tie on both arms.
- The E1-tre vs E2-zm deltas (<=1.2pp) and the t2 zm-vs-queue_len gap (0.42pp) are
  within the TRE noise floor: the eta gate has no measurable effect at this seed
  budget, and zm vs queue_len is not separable on t2. The t5 zm advantage over
  decode_tps (4.58pp) exceeds the floor and stands.

## Campaign ledger

| run | seed | start | end | params hash | verdict | evidence |
| --- | ---: | --- | --- | --- | --- | --- |
| `t1_tre_seed2` | 20260716 | 2026-07-12T02:27:50Z | 2026-07-12T02:41:43Z | `328bcfd2b54ca2db` | DONE, V_req=42.187% | `t1_tre_seed2/` |
| `t1_apa_seed2` | 20260716 | 2026-07-12T02:53:40Z | 2026-07-12T03:07:33Z | `328bcfd2b54ca2db` | DONE, V_req=49.190% | `t1_apa_seed2/` |
| `t3_tre_seed2` | 20260716 | 2026-07-12T03:19:21Z | 2026-07-12T03:38:02Z | `328bcfd2b54ca2db` | DONE, V_req=1.995% | `t3_tre_seed2/` |
| `t3_apa_seed2` | 20260716 | 2026-07-12T03:49:57Z | 2026-07-12T04:08:39Z | `328bcfd2b54ca2db` | DONE, V_req=49.361% | `t3_apa_seed2/` |
| `t6_tre_seed2` | 20260716 | 2026-07-12T04:20:27Z | 2026-07-12T04:35:25Z | `328bcfd2b54ca2db` | DONE, V_req=0.022% | `t6_tre_seed2/` |
| `t6_apa_seed2` | 20260716 | 2026-07-12T04:47:17Z | 2026-07-12T05:02:15Z | `328bcfd2b54ca2db` | DONE, V_req=0.004% | `t6_apa_seed2/` |
| `t1_tre_seed3` | 20260717 | 2026-07-12T05:14:03Z | 2026-07-12T05:27:57Z | `328bcfd2b54ca2db` | DONE, V_req=41.593% | `t1_tre_seed3/` |
| `t1_apa_seed3` | 20260717 | 2026-07-12T05:39:54Z | 2026-07-12T05:53:47Z | `328bcfd2b54ca2db` | DONE, V_req=53.736% | `t1_apa_seed3/` |
| `t3_tre_seed3` | 20260717 | 2026-07-12T06:05:37Z | 2026-07-12T06:24:18Z | `328bcfd2b54ca2db` | DONE, V_req=0.756% | `t3_tre_seed3/` |
| `t3_apa_seed3` | 20260717 | 2026-07-12T06:36:16Z | 2026-07-12T06:54:57Z | `328bcfd2b54ca2db` | DONE, V_req=42.680% | `t3_apa_seed3/` |
| `t6_tre_seed3` | 20260717 | 2026-07-12T07:06:46Z | 2026-07-12T07:21:44Z | `328bcfd2b54ca2db` | DONE, V_req=0.036% | `t6_tre_seed3/` |
| `t6_apa_seed3` | 20260717 | 2026-07-12T07:33:36Z | 2026-07-12T07:48:34Z | `328bcfd2b54ca2db` | DONE, V_req=0.009% | `t6_apa_seed3/` |
