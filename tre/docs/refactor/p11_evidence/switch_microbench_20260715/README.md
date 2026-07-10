# E3 switch-latency microbenchmark

## Frozen setup

- Formal run: 2026-07-10T09:08:54Z to 2026-07-10T09:29:18Z (UTC).
- Code SHA: `ee882da0671e80d650a659dbd4a64215f3d7ac68`.
- Deployment manifest HEAD: `54373f1a19b4621a78e7417264af353aecb8e262`.
- Controller, service-manager, and UI image tag: `20260710-ee882da0`.
- Image IDs: controller `sha256:84a51584b784...`, service-manager `sha256:af220c083b9f...`, UI `sha256:9057237d2980...`.
- Registry params hash and applied hash: `328bcfd2b54ca2db`; `pending_restart=false`.
- Controller mode: `observe`.
- vLLM image: `vllm/vllm-openai:0.10.1-sleep`; `/version` returned `0.10.1` for all targets.
- Test gate before the run: `make check` -> 502 passed.

Docker Hub was unreachable while resolving `python:3.11-slim`. The three images were therefore built from the immediately preceding frozen local images, after deleting and fully replacing every application source directory copied by the component Dockerfile. Container-side `py_compile` checks passed before rollout. No dependency files changed between the base and code SHAs.

## Targets

| Model | serve_id | Node | GPU IDs |
|---|---|---|---|
| dsllama-8b | `dsllama-8b-nscc-ds-4a100-node9-gpu-1-5cb98fdbb6-mr5b7` | `nscc-ds-4a100-node9` | 1 |
| dsqwen-7b | `dsqwen-7b-nscc-ds-4a100-node9-gpu-0-546d5d9f88-f94nf` | `nscc-ds-4a100-node9` | 0 |
| dsqwen-14b | `dsqwen-14b-nscc-ds-4a100-node9-gpu-2-3-69c86d8db7-vnxtl` | `nscc-ds-4a100-node9` | 2,3 |

The existing model-count target API applies H1 placement and cannot keep one node9 binding fixed across repeated cycles. Commit `f0703a4b` added the exact-binding `PUT /v2/bindings/{serve_id}/power` API. It reuses the production vLLM power action, Kubernetes annotation writer, GPU conflict check, and versioned SM state store. The benchmark does not edit Redis, Pod metadata, or vLLM state directly.

## Method

Each of 75 cycles contains one sleep row and one wake row, so `cycles.csv` has 150 transition rows. Cycle 1 for each model is marked cold and is never included in steady-state aggregates. Cycles 2-20 are unloaded steady state. Cycles 21-25 run 2 rps through the TRE gateway from 30 seconds before sleep until 30 seconds after wake.

Sleep readiness is `/is_sleeping=true`; wake readiness is a successful direct 1-token completion, polled every 100 ms. `t_log_marker` and `dur_engine_s` come from the vLLM executor marker. A 1.2 second post-transition settle interval guarantees at least one stable 1 Hz GPU sample before the next transition. TP=2 memory is the sum over GPUs 2 and 3.

Node9 was 149.708 seconds ahead of the 76/controller host. `clock_sync.json` records five SSH midpoint probes. Marker and nvidia-smi timestamps in `cycles.csv` are normalized by the median offset; the raw logs retain node9 timestamps.

Exact formal command:

```bash
cd /data/nfs_shared_data/xxy/aibrix/tre
PYTHONPATH=common:deploy:ui:controller:service-manager:replayer \
python3 deploy/scripts/switch_microbench.py \
  --output-dir docs/refactor/p11_evidence/switch_microbench_20260715 \
  --target dsllama-8b=dsllama-8b-nscc-ds-4a100-node9-gpu-1-5cb98fdbb6-mr5b7 \
  --target dsqwen-7b=dsqwen-7b-nscc-ds-4a100-node9-gpu-0-546d5d9f88-f94nf \
  --target dsqwen-14b=dsqwen-14b-nscc-ds-4a100-node9-gpu-2-3-69c86d8db7-vnxtl \
  --cycles 25 --load-start-cycle 21 --load-rps 2 \
  --load-pre-s 30 --load-post-s 30
```

## Results

Unloaded steady-state end-to-end latency, cycles 2-20:

| Model | Sleep median / p95 (s) | Wake median / p95 (s) |
|---|---:|---:|
| dsllama-8b | 1.090 / 1.109 | 0.731 / 0.787 |
| dsqwen-7b | 1.096 / 1.156 | 0.693 / 0.734 |
| dsqwen-14b | 1.806 / 2.561 | 0.752 / 0.781 |

Cold sleep/wake end-to-end values were 1.226/0.734 s for llama-8b, 1.181/0.679 s for qwen-7b, and 2.340/0.745 s for qwen-14b. They are reported separately in `latency_summary.csv`.

| Model | Loaded cycles | Requests | Errors | Error rate | Median / max per-cycle successful p99 (ms) |
|---|---:|---:|---:|---:|---:|
| dsllama-8b | 5 | 650 | 20 | 3.077% | 2255.752 / 2284.918 |
| dsqwen-7b | 5 | 650 | 20 | 3.077% | 2230.840 / 2276.491 |
| dsqwen-14b | 5 | 655 | 20 | 3.053% | 2853.411 / 2927.065 |

Every loaded cycle recorded exactly four request errors while the sole routable replica was asleep. This is measured behavior, not filtered from the result. Request-level outcomes are in `inflight_requests.csv`.

Median GPU memory moved from 37,932 to 4,070 MiB for llama-8b, 37,892 to 4,070 MiB for qwen-7b, and about 74,764 to 8,140 MiB for qwen-14b. Wake restored the inverse values.

## Acceptance

- PASS: 75 unique complete cycles, 25 per model.
- PASS: 150 transition rows with the required columns and no missing timing or memory fields.
- PASS: 15 loaded cycles and 1,955 request-level outcomes.
- PASS: Redis orphan-alert hash remained empty and the controller session log contains zero `TRE_ORPHAN_HIDDEN` events.
- PASS: `sm_state_before.json` and `sm_state_after.json` have identical per-binding layout/state; all three original node9 bindings are awake and routable after the run.
- PASS: safescale probes remain 0 and controller mode remains `observe`.

## Artifacts

- `cycles.csv`: canonical 150 transition rows.
- `latency_summary.csv`: cold, unloaded steady, and loaded aggregates.
- `inflight_requests.csv` and `inflight_summary.csv`: request-level and model-level loaded behavior.
- `nvidia_smi_node9.csv`: raw 1 Hz four-GPU session.
- `pod_log_*.txt`: the 150 raw vLLM executor marker lines.
- `controller_session.log`: controller log covering the formal run.
- `clock_sync.json`, `targets.json`, `sm_state_before.json`, `sm_state_after.json`, and `run_summary.json`: provenance and acceptance state.
- `SHA256SUMS.txt`: artifact checksums.