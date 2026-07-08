# S1.2 authoritative W-freeze evidence (2026-07-08)

Real-machine acceptance that freezes the control window **W = `TRE_METRICS_WINDOW_MS` = 30000**
(sliding, 5s refresh), per `docs/refactor/15_signal_and_window_plan.md` §1.2 and
`DECISIONS.md` ADR-0012 (+ its same-day correction). This is the R3 hard-gate precondition:
R3 `theta_m` refit must not start until W is frozen on the isolated `tre-v2` plane — done here.

## What was measured
Lag from a step-load onset to the first controller decision entry whose `z_m` becomes non-null,
i.e. the moment the shared TSS signal first **reflects** the added load. The pre-change value of
this quantity (old 60s *tumbling* window) was 60–120s; the S1 change (sliding window + 5s refresh)
targets `<= W + refresh + write granularity ~= 35s`.

Controller ran in **observe** mode the whole time: decisions computed and logged, no fleet
actuation. The fleet (3 awake, 17 sleeping) was never mutated.

## Result
`lag_first` (onset -> first non-null z_m), 12 trials pooled over 3 models x 4 rounds:
**P50 = 11.0s, P95 = 12.7s, max = 14.9s** — well under 35s. All trials: baseline idle confirmed,
0 request errors. **PASS => W frozen at 30000.**

## Files
| file | contents |
|---|---|
| `summary.txt` | human-readable method + per-trial table + aggregate + judgment |
| `summary.json` | machine-readable summary (metrics, P50/P95/max, per-trial rows) |
| `trials.json` | per-trial onset/load-end timestamps + client load stats (ok/err/rps/p50/p95) |
| `raw_decision_hist.json` | full per-model decision-history entries spanning the run (z_m/trs/state/signal_warm/window_end_ms) |
| `measure.py` | the load driver + redis reader used to produce trials.json / raw_decision_hist.json |
| `analyze.py` | computes the lag metrics + P50/P95/max from the two data files |
| `pilot_load.json`, `pilot_hist.jsonl` | one earlier pilot trial (dsqwen-7b, 20 workers, 80s) used to shape the metric/load level |

## Reproduce
On node10 host (76): `python3 measure.py` (writes /tmp/s1wf_*.json), then `python3 analyze.py`.
Requires controller in observe mode, redis pod name pinned in `measure.py`, and the fleet awake:1 per model.

## Metric note
`lag_first` (first-reflection) is the acceptance gate — apples-to-apples with the pre-change
60–120s number. A secondary `lag_settle` (onset -> z_m >= 90% of the saturated plateau, P95 ~51s)
is recorded for context only; it is inherently >= W and dominated by single-replica queue growth
under sustained saturation, so it is not the criterion the freeze is judged on.
