# RUN: θ recalibration on vLLM 0.30 (2026-10-03)

Status: **plan only, nothing executed.** Every step that changes the cluster needs the
owner's go-ahead for this run (approval list at the end). Rules: preregistration
`tre/docs/preregistration-20261003-calibration-v030.md`; background
`docs/theta-recalibration.md` (local workspace) §5.

- System under test = the live control plane `integ/tre-v2-20261001b` (bdea7a72) and
  `vllm-openai-tre:0.30.0-ts-8dc0f2a7`. Calibration code = branch `calib/theta-20261003`
  (worktree `aibrix-wt/calib-theta-20261003`). Nothing is deployed from this branch: θ
  depends only on the engine and the gateway signal.
- Launchers: `tre/deploy/scripts/calibration/run_*.sh` (shared guards in `lib.sh`: one driver
  per model, empty out-dir, Redis + Envoy probed, run mode `observe observe`, exclusive-window
  marker present, engines idle; `CALIB_TASKSET` pins the driver).
- Wall clock (three models in parallel, 14b is the critical path): **~28-30 h in total
  plus the λ/L3 pause**. Breakdown in §T.

## 0. Conventions (on 76)

```bash
REPO=/data/nfs_shared_data/xxy/aibrix
WT=$REPO-wt/calib-theta-20261003
T=$WT/tre
CAL=$T/deploy/scripts/calibration
export CALIB_ROOT=/data/nfs_shared_data/xxy/calib_20261003       # NEW root; never reuse
export TRE_EXCLUSIVE_WINDOW_FILE=/data/nfs_shared_data/xxy/TRE_EXCLUSIVE_WINDOW
export TRE_CALIBRATION_GATEWAY_URL=http://192.168.223.76:31094/v1/chat/completions
export CALIB_TASKSET=48-63            # NUMA1 cores; 14b's GPUs 0-1 sit on NUMA0 (0-31)
export DESIGN_SEED=20261003
MODELS="dsqwen-7b dsllama-8b dsqwen-14b"
SM=http://$(kubectl -n tre-v2 get svc tre-v2-service-manager -o jsonpath='{.spec.clusterIP}'):8000
cd $T/deploy && export PYTHONPATH="../common:.:../controller:../service-manager:../calibration:../replayer:../ui"
mkdir -p $CALIB_ROOT && touch $CALIB_ROOT/ATTEMPTS.md   # every void / repeated attempt is registered here
```

Every long stage is launched detached, one process per model, and polled:

```bash
# launch pattern (stage <S>, out root $R)
for m in $MODELS; do nohup bash $CAL/run_<S>.sh $m > $R.$m.log 2>&1 & done
# poll (no pkill, ever; pgrep anchored)
pgrep -af '^(\S*/)?python3 -m scripts\.(calibration_campaign|r3_grid)'; tail -3 $R.*.log
```

Before each long stage: write the stage, its roots and its expected end into HANDOFF.md
(local workspace) and `$CALIB_ROOT/ATTEMPTS.md`.

## A. Calibration state (~15 min) - cluster change, needs go-ahead

1. Exclusive window (refuse if someone holds it):
   ```bash
   [ -e $TRE_EXCLUSIVE_WINDOW_FILE ] && { cat $TRE_EXCLUSIVE_WINDOW_FILE; echo "window held - stop"; }
   [ -e $TRE_EXCLUSIVE_WINDOW_FILE ] || echo "calibration $(date -Iseconds) $(date -Iseconds -d '+36 hours') calib/theta-20261003" > $TRE_EXCLUSIVE_WINDOW_FILE
   ```
2. Nothing else drives load (both nodes): `pgrep -af '^python3 -m'` on 76 and 75 shows only
   `vllm` api servers (a static `http.server` of another session is harmless).
3. Save the awake set and run mode (needed by §I):
   ```bash
   curl -s $SM/v2/state > $CALIB_ROOT/sm-state-before.json
   bash $T/deploy/scripts/set_run_mode.sh status | tee $CALIB_ROOT/run-mode-before.txt
   ```
4. Run mode: `bash $T/deploy/scripts/set_run_mode.sh observe observe && bash $T/deploy/scripts/set_run_mode.sh status`
5. One routable replica per model. The baseline awake set already is one per model
   (7b node9/GPU0, 8b node9/GPU1, 14b node10/GPU0-1), so **no wake/sleep is needed**:
   ```bash
   for m in $MODELS; do echo "$m $(kubectl -n default get pods -l model.aibrix.ai/name=$m,tre.aibrix.io/routable=true --no-headers | wc -l)"; done   # each 1
   python3 scripts/release/awake_ctl.py show
   ```
   If not: `python3 scripts/release/awake_ctl.py restore-ids <the three baseline binding ids>`
   (via an empty GPU if a target GPU is held; see HANDOFF 10-02 note on `99_restore.sh`).
6. Health: 20/20 pods `2/2 Running`; `python3 scripts/release/release_checks.py gpu-truth`
   (re-check after ~15 s if it reports a seq reset); controller log has no
   `breakpoint_window_suspended`.
7. Sidecar baseline: record `tre_reissue_total` of the three awake pods (§R) into
   `$CALIB_ROOT/reissue-before.txt`.

## B. Idle TTFT c/b on 0.30 (~0.5 h) - load

```bash
R=$CALIB_ROOT/cb_idle; mkdir -p $R
for m in $MODELS; do OUT_ROOT=$R bash $CAL/run_cb_idle.sh $m --dry-run; done     # plan + estimate
for m in $MODELS; do OUT_ROOT=$R nohup bash $CAL/run_cb_idle.sh $m > $R.$m.log 2>&1 & done
# ~10 min each (245 requests x 2 s gap). After all three exit 0:
python3 -m scripts.ttft_idle_fit --root $R --models dsqwen-7b,dsllama-8b,dsqwen-14b \
    --min-gap-ms 500 --out $R/ttft_idle_fit.json --registry-patch
```

Check: n isolated requests ≥ 25 per L and model; c within [20, 80] ms, b within
[0.02, 0.15] ms/token (old engine: 36.4-43.4 ms, 0.053-0.068); look at the per-L
`residual_median_ms` at L = 3072 / 4096. **Chunked prefill**: the engines run
`--max-num-batched-tokens 2048`, so L > 2048 prefills in two chunks and TTFT(L) may bend at
2048 (the old fit only had L ≤ 2048). If the residuals there are clearly off, stop and let
the owner choose: fit L ≤ 2048 only (`--lengths`), or the full range. Outside the ranges
above: stop and show the owner. The fit tool reproduces the 09-22 values exactly from the
old run1 raw (36.4/0.0527, 39.3/0.0555, 43.4/0.0683; n 94/77/161).

## C. c/b into the registry (~20 min)

1. **Branch registry (what every campaign reads):** write c/b into
   `$T/deploy/registry.yaml` (`python3 -m scripts.ttft_idle_fit --root $R --min-gap-ms 500
   --out $R/ttft_idle_fit.json --write-registry $T/deploy/registry.yaml`; it changes only the
   two keys and refuses n < 20 or fewer than 3 lengths), update the provenance comment next
   to them by hand, run `pytest deploy/tests/test_registry_smoke.py -x`, commit on
   `calib/theta-20261003`. All
   later stages run from that commit (the manifests record it and the label sha256).
2. **Live registry (cluster change, needs go-ahead).** The console cannot do it:
   `PUT /api/params` only takes `slo.ttft_p95_ms / tpot_p95_ms / e2e_p95_ms`, and
   `merge_live_registry.py` lets LIVE `slo.*` win over the release. So edit the live copy
   directly:
   ```bash
   B=/data/nfs_shared_data/xxy/backups/pre-calib-cb-$(date +%Y%m%d-%H%M); mkdir -p $B
   kubectl -n tre-v2 get cm tre-v2-registry -o yaml > $B/cm.yaml
   kubectl -n tre-v2 get cm tre-v2-registry -o jsonpath='{.data.registry\.yaml}' > $B/live.yaml
   cp $B/live.yaml $B/new.yaml
   python3 -m scripts.ttft_idle_fit --root $R --min-gap-ms 500 --out $B/fit.json --write-registry $B/new.yaml
   diff $B/live.yaml $B/new.yaml          # only ttft_idle_c_ms / ttft_idle_b_ms_per_token of 3 models
   kubectl -n tre-v2 create cm tre-v2-registry --from-file=registry.yaml=$B/new.yaml --dry-run=client -o yaml | kubectl replace -f -
   kubectl -n tre-v2 rollout restart deploy/tre-v2-controller deploy/tre-v2-service-manager
   kubectl -n tre-v2 rollout status deploy/tre-v2-service-manager; kubectl -n tre-v2 rollout status deploy/tre-v2-controller
   ```
   Rollback: `kubectl replace -f $B/cm.yaml` + the same restarts.
   It is not needed by the collection (labels are computed offline from the branch
   registry; SafeScale does not act in `observe`). Doing it now keeps live and branch
   equal and lets SafeScale labels mode and baseline μ export read the new c/b. Alternative:
   defer to the θ go-live (§after H) and update θ/λ/τ/w_p/c/b in one step - **owner picks**.
3. After any controller/SM restart: run mode still `observe observe` (re-run
   `set_run_mode.sh status`), awake set unchanged (`awake_ctl.py show`), wait **≥ 60 s idle**
   (EMA state is lost, ~30 s blind spot) before §D.

## D. Training collection (~19.5 h) - load

All three models in parallel (one replica each, different GPUs). CPU: 64 cores per node,
~87 % idle now; one lane needs ≤ 1.6 cores at its burst peak (4 sender processes), three
lanes ≤ 5 cores, pinned to 48-63. The 50 ms send-lateness p99 guard voids any cell where
the client fell behind; a void twice stops that model's run. 7b and 8b engines run on 75,
the 14b engine on 76 (NUMA0): the client never shares cores with an engine that is
being measured, only with idle api servers and Envoy.

| Step | Command | Per model (7b / 8b / 14b) | Output |
|---|---|---|---|
| D1 run1 | `OUT_ROOT=$CALIB_ROOT/run1 bash $CAL/run_primitives.sh $m` | ~5.5 h | `run1/<m>/` + dataset |
| D2 priors | `python3 -m scripts.analysis.calibration_priors $CALIB_ROOT/run1 --out-dir $CALIB_ROOT/prereg` then `sha256sum $CALIB_ROOT/prereg/*.json > $CALIB_ROOT/prereg/SHA256SUMS`, record in the preregistration's deviation-free addendum (commit) **before D3** | min | `prereg/` |
| D3 run2 | `OUT_ROOT=$CALIB_ROOT/run2 PRIORS_DIR=$CALIB_ROOT/prereg bash $CAL/run_ladder.sh $m` | 10.4 / 10.4 / 11.4 h | `run2/<m>/` |
| D4 S3 supplement | `OUT_ROOT=$CALIB_ROOT/supp BASE_RUN=$CALIB_ROOT/run2 REPROBE_GRID=<from run2> bash $CAL/run_supplement.sh $m` | 0.3-0.7 h | `supp/<m>/` |
| D5 P1 | `OUT_ROOT=$CALIB_ROOT/p1 BASE_RUN=$CALIB_ROOT/run2 SUPP_RUN=$CALIB_ROOT/supp TRAINING_PLAN=p1-deep-overload bash $CAL/run_stage3.sh $m` | 1.34 h expected, 2.66 h upper (18 cells) | `p1/<m>/` |

- Dry-run every stage first (`... run_X.sh $m --dry-run`): prints the plan and the
  estimate, drives nothing.
- D4 grid: the old D6' priors put the S3 flip at ~1.4x / 1.7x / >1.74x rho*_run2; read the
  new run2's S3 boundary record and pick 4 multiples bracketing it (write them into
  ATTEMPTS.md before launching).
- D4 exit 3 (a rho* is a bound or the smoke is out of band) = stop for the owner.
- D5 P1 = 15 hold cells (S2/S3/T8/S4/S5 x {1.5, 2, 3} rho*_run2, 150 s) + 3 S2 drift
  sentinels; each cell first waits for the engine to drain (running + waiting = 0, up to
  300 s; a state gate, not a timer). At 3x the client holds up to ~3000 requests in
  flight (7b S4; cap 4096) - **run the three models' P1 one after another** (or 14b alone
  in parallel with one of the others) unless the dry-run shows no in-flight WARNING; the
  50 ms lateness guard voids a cell if the client falls behind. Exit 3 = a P1 cell had no
  labelled window. Whether client-side timeouts reach vLLM through the sidecar is not
  verified; if not, the 300 s drain gate catches it and marks the next cell.
- HANDOFF before D1 and before D3 (both are long; the KV cache expires).

## E. Dataset and offline fits (~1 h, CPU only)

1. Training set and the four variants (λ: v2 rule / v1-λ; numerator: D2 gateway tokens /
   L3 vLLM counter deltas):
   ```bash
   F=$CALIB_ROOT/fit
   python3 -m scripts.dline_refit trainset --fit-dir $F --model $m \
       --h2-dataset run1=$CALIB_ROOT/run1/$m/dataset --h2-dataset run2=$CALIB_ROOT/run2/$m/dataset \
       --dataset supp=$CALIB_ROOT/supp/$m/dataset --dataset p1=$CALIB_ROOT/p1/$m/dataset
   python3 -m scripts.dline_refit alpha --fit-dir $F --model $m --publish-tau-s 10
   python3 -m scripts.dline_refit wp    --fit-dir $F --model $m                     # λ = v2 rule
   python3 -m scripts.dline_refit wp    --fit-dir $F/v1lambda --model $m --lambda-method v1
   python3 -m scripts.dline_refit final --fit-dir $F --model $m --no-holdout
   ```
2. L3 (numerator from `vllm_metrics_1hz/` counter deltas): **not implemented yet**. It is
   an offline re-windowing of data the capture already keeps (all `vllm:*_total`
   counters at 1 Hz per pod, running and waiting separately). Write it on a branch during
   D (no cluster needed), then rerun step 1 with the L3 numerator.
3. Report per model and variant: θ, CI half width (D13 ≤ 20 %), training BA, B′ on the
   training set at dwell 1 and 2, and the λ / numerator curves. Training data only - no M,
   no T14.

## ⏸ PAUSE: the owner picks λ and the numerator (D2 or L3)

Send the §E table. Nothing is frozen and nothing else is collected until the answer.
The cluster stays in `observe observe` under the window (or close the window and restore
§I if the pause is long; re-open §A before §G).

## F. Freeze (~10 min)

```bash
python3 -m scripts.dline_refit freeze --out-dir $CALIB_ROOT/freeze --fit-dir $F \
    --model dsqwen-7b --model dsllama-8b --model dsqwen-14b --freeze-file $CALIB_ROOT/freeze/params_freeze.json
python3 -m scripts.dline_refit verify-freeze --freeze-file $CALIB_ROOT/freeze/params_freeze.json
```
`freeze` refuses when D13 fails or a model has no violating training window (stop for the
owner). Freeze format revision 3 seals, per model, `b_prime.severity_cut` (.65 quantile of
the training violating windows' severity) and `b_prime_gate` (.80 / .70 / .05 / .08) under
the freeze's own hash, i.e. before M / T14 exist. Check:
`python3 -c 'import json,sys; d=json.load(open(sys.argv[1])); print({m: v.get("b_prime") for m, v in d.items() if isinstance(v, dict) and "b_prime" in v})' $CALIB_ROOT/freeze/params_freeze.json`

## G. M (3 models) and T14 (14b) (~6 h) - load

1. b50 table and T14 capacity prior (CPU, from training data only):
   ```bash
   R2=$CALIB_ROOT/run2
   python3 -m scripts.analysis.boundary_b50_table --base-run $R2 --out $CALIB_ROOT/b50.csv \
       --windows $R2/dsqwen-7b/dataset/windows.csv --windows $R2/dsllama-8b/dataset/windows.csv \
       --windows $R2/dsqwen-14b/dataset/windows.csv
   # (reproduces the 09-23 boundary_d6prime_run2.csv byte for byte from the old inputs)
   python3 -m scripts.calibration_t14 capacity-prior --model dsqwen-14b --base-run $CALIB_ROOT/run2 \
       --boundary-supplement-run $CALIB_ROOT/supp --boundary-table $CALIB_ROOT/b50.csv \
       --out $CALIB_ROOT/t14/capacity_prior_dsqwen-14b.json
   ```
2. T14 preregistration JSON (new clean file, schema of the 2026-09-24 one, no amendment):
   write `$CALIB_ROOT/t14/preregistration.json` + `.sha256` (sha256sum format), commit a
   copy on the branch. Bound by the campaign (must be present): `t14.capacity_prior.sha256`,
   `t14.design_seed` (new, e.g. 20261004), `t14.cell_serial_base` 80500, `t14.factors`
   [0.9, 1.0, 1.1], `t14.hold_s` 240, `t14.shapes` (the 8 held-out shapes). Checked when
   present (write all): `t14.model`, `t14.api` "chat", `t14.gateway.url` ending in
   `/v1/chat/completions`, `parameter_sets.freeze.sha256`, `parameter_sets.v1lambda.sha256`
   (or the second set's), `t14.forbidden_roots` = this round's run1 / run2 / supp / p1 /
   fit / freeze / M roots (a real run refuses with no roots). Also write: label sha256
   (new c/b), engine image tag, prompt corpus (mix 0.5), B′ cut, expected wall clock.
   The campaign needs **two** parameter sets (`--refit-params-file`); with one frozen set,
   name it twice and say so in the file.
3. M, all three models in parallel:
   `OUT_ROOT=$CALIB_ROOT/M BASE_RUN=$CALIB_ROOT/run2 SUPP_RUN=$CALIB_ROOT/supp BOUNDARY_TABLE=$CALIB_ROOT/b50.csv RETAINED_DATASET=$CALIB_ROOT/run1/$m/dataset FREEZE_FILE=$CALIB_ROOT/freeze/params_freeze.json bash $CAL/run_M.sh $m`
   (2.6-3.0 h; exit 3 = stop for the owner).
4. T14 after 14b's M (one routable 14b pod; 7b/8b M may still run):
   `OUT_ROOT=$CALIB_ROOT/T14 PREREG_JSON=$CALIB_ROOT/t14/preregistration.json CAPACITY_PRIOR=$CALIB_ROOT/t14/capacity_prior_dsqwen-14b.json FREEZE_FILE=$CALIB_ROOT/freeze/params_freeze.json REFIT_PARAMS_FILE=<second set> DESIGN_SEED=<t14.design_seed> bash $CAL/run_T14.sh dsqwen-14b`
   (24 cells: 2.2 h expected, 2.9 h upper).
5. HANDOFF before G.

## H. Accept and decision (~0.5 h, CPU)

```bash
python3 -m scripts.dline_refit accept --freeze-file $CALIB_ROOT/freeze/params_freeze.json \
    --dataset M=$CALIB_ROOT/M/<m>/dataset --dataset run1=$CALIB_ROOT/run1/<m>/dataset \
    --m-manifest $CALIB_ROOT/M/<m>/M_manifest.json --dwell-windows 1   # one --m-manifest per model; T14_manifest the same way
# --dwell-windows 1 = the live TRE_DWELL_WINDOWS (a guard test keeps the default equal to the overlay);
# the result also reports B' at dwell 2, old B, all-violation recall and the LOW-band share (not gating).
python3 -m scripts.analysis.calibration_decision <dataset> --regime-groups $CALIB_ROOT/prereg/regime_groups.json --out-dir $CALIB_ROOT/decision
```
Gate = A, B′ (dwell = live value 1), D; disclosures as the preregistration §6. Old-vs-new
parameter table to the owner. **Going live with θ is a separate release** (θ, λ, τ, w_p,
c/b atomically; console PUT + restart controller and SM, idle ≥ 60 s).

## I. Restore (~10 min) - cluster change

```bash
bash $T/deploy/scripts/set_run_mode.sh observe active && bash $T/deploy/scripts/set_run_mode.sh status
python3 scripts/release/awake_ctl.py expect $CALIB_ROOT/sm-state-before.json <7b node9/0> <8b node9/1> <14b node10/0-1>
python3 scripts/release/awake_ctl.py show       # 7b node9/0, 8b node9/1, 14b node10/0,1
rm -f $TRE_EXCLUSIVE_WINDOW_FILE
```
HANDOFF: roots, verdicts, attempts.

## R. Checks run at every stage boundary

- **Reissue contamination (automatic):** each cell reads `tre_reissue_total` of ALL the
  model's pods (port 8000, `/tre-reissue/metrics`) before and after the load. `continue`
  delta > 0 = contaminated; a pod not read twice, a counter that went down (sidecar
  restart) or no pod = unmeasured. Both void the cell (`--reissue-check void`, default: rerun
  once, a second void stops the run) and the dataset excludes them (`cells.csv` columns,
  manifest `reissue_check`). At each stage end also compare the three awake pods' counters
  with `$CALIB_ROOT/reissue-before.txt`:
  ```bash
  for p in $(kubectl -n default get pods -l tre.aibrix.io/routable=true -o name); do
    echo "$p $(kubectl -n default exec ${p#pod/} -c vllm-openai -- curl -s localhost:8000/tre-reissue/metrics | grep '^tre_reissue_total')"; done
  ```
  Any `kind="continue"` increase = find the cell(s) and register them in ATTEMPTS.md. In
  `observe observe` nothing sleeps a routable pod, so the expected delta is 0.
- Run mode still `observe observe`; awake set unchanged; gpu-truth fresh; no other load
  process (`pgrep -af '^python3 -m'` on both nodes).
- Voids per stage (`design_result.json`, the campaign log).

## Failure and resume

- A stage stops (exit 1: a cell voided twice, driver failure) or the host drops: the
  partial directory is **void evidence**. Keep it, write it into ATTEMPTS.md (what, when,
  why), and rerun **that model's** stage into a new root, e.g.
  `OUT_ROOT=$CALIB_ROOT/run2_r2`. Later stages read per model, so point that model's
  `BASE_RUN` / `SUPP_RUN` / dataset paths at its accepted attempt.
- There is no in-run resume. Long stages (run1 5.5 h, run2 ~11 h) lose their progress on
  a stop - keep them detached (`nohup`) so an ssh drop does not stop them.
- Controller or SM restarted mid-stage (EMA lost): the cells of the next ~60 s are
  suspect; register it, and rerun that model's stage if a cell overlapped.
- Exit 3 (a check failed after the run finished) never auto-continues: stop for the owner.

## T. Wall clock

| Stage | Wall clock (parallel; 14b) |
|---|---|
| A | 0.25 h |
| B | 0.5 h |
| C | 0.3 h |
| D1 run1 | 5.5 h |
| D3 run2 | 11.4 h |
| D4 supp | 0.7 h |
| D5 P1 (serial across models: 3 x 1.34 h; parallel: 1.34 h) | 1.3-4.0 h |
| E fits (+ L3 code, written during D) | 1 h |
| pause | owner |
| F | 0.2 h |
| G M then T14 (14b) | 3.0 + 2.9 h |
| H | 0.5 h |
| I | 0.2 h |
| **Total** | **~28-30 h + pause** (P1 parallel vs serial; without run1: ~5.5 h less) |

## Approvals needed (cluster changes)

1. A: create the exclusive-window marker; `set_run_mode.sh observe observe`.
2. B, D, G: drive calibration load through the TRE gateway (~25 h in total).
3. C.2: replace the live registry ConfigMap (c/b only) and restart controller + SM - or
   defer to the θ go-live.
4. I: `set_run_mode.sh observe active`, remove the marker.
5. Later (separate release): θ/λ/τ/w_p (+ c/b) go-live.
