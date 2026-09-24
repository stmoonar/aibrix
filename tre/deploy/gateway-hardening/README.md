# Gateway hardening — upstream socket-exhaustion protection

Protects the experiment data paths (the shared Envoy gateways in front of the
vLLM model Services) from the upstream local-connection / socket-creation
exhaustion that produced the t1 503 storm. See `docs/refactor/WORKLOG.md`
("Envoy gateway upstream socket-exhaustion hardening", 2026-07-09) for the full
diagnosis and validation.

## The bug (t1)

Under tight load t1, 49.6% of 45902 requests returned 503. 10309/10310 of those
were envoy `response_flags=UF` + `upstream_reset_before_response_started{local_
connection_failure|socket_creation_failure}`: **envoy could not open the upstream
socket** — requests never reached vLLM. An innocent model (14b, 6rps) took 47%
collateral 503s. Two compounding root causes:

1. The aibrix-system envoy container had `RLIMIT_NOFILE` **soft=1024** (node
   container-runtime default). One HTTP/1.1 upstream socket is held per in-flight
   request; the saturated model's long requests (p50 9s / p99 34s) pile these up.
2. **No per-cluster connection cap** — only `max_retries` was set, so
   max_connections/max_pending/max_requests defaulted to 1024 *per cluster*. With
   3 model clusters sharing one 1024-fd budget, the pool is exhausted globally, so
   socket creation fails for every model → innocent models are collateral damage.

## The fix

| File | What | Layer |
| --- | --- | --- |
| `backendtrafficpolicy-aibrix-system.yaml` | per-model circuit breakers on the shared gateway (Gateway aibrix-system/aibrix-eg, NodePort 31592) - the APA arm's path until 2026-09-24, no longer used by either arm | additive, aibrix-system ns |
| `backendtrafficpolicy-tre-v2.yaml` | per-model circuit breakers on the serving path of BOTH arms, TRE and APA (Gateway tre-v2/tre-aibrix-eg, NodePort 31094); the per-model ORIGINAL_DST clusters of `overlays/tre-v2/gateway-extproc.yaml` carry the same values | additive, tre-v2 ns |
| `envoyproxy-nofile-patch.yaml` | raise envoy RLIMIT_NOFILE 1024→65536 | **modifies shared aibrix-system EnvoyProxy** (class-level → both envoys) |

Circuit-breaker values (per model, per gateway): maxConnections 4096,
maxPendingRequests 1024, maxParallelRequests 4096, maxParallelRetries 16. Each
model gets its own bounded quota, so overload fast-fails as 503 **UO** (overflow,
returned in ms before any socket attempt) confined to the saturated model, instead
of UF socket_creation_failure that spreads across all models. The cap is a
ceiling, not a steady target; it only stops the runaway pile-up.

### The two arms must carry identical values

Since 2026-09-24 both arms, TRE and APA, are served by the **same** tre-v2 gateway
(NodePort 31094; `deploy/scripts/campaign_queue.py` GATEWAYS), routed as in v1
(`routing-strategy: least-gpu-cache`, ext_proc, per-model ORIGINAL_DST clusters
whose limits `deploy/tests/test_gateway_extproc.py` pins to
`backendtrafficpolicy-tre-v2.yaml`). The arms therefore admit traffic under
identical rules by construction. An A/B result is only interpretable that way:
otherwise "TRE served more requests" is indistinguishable from "APA was shed
sooner".

Before that date the APA arm went through aibrix-system (NodePort 31592,
`backendtrafficpolicy-aibrix-system.yaml`). That file is kept identical to the
tre-v2 one (**change the two files together or not at all** -
`deploy/tests/test_gateway_arm_symmetry.py` fails if they drift) so the legacy
path stays comparable for re-analysis of older runs.

This is not hypothetical. On 2026-09-20 the TRE arm was raised to 4096/1024 while
the APA arm was left at 256/64; the asymmetry was caught before the re-run, but a
comparison made in that window would have been meaningless.

### Why the values moved (2026-09-21)

The original 256/64 was sized against an assumed `RLIMIT_NOFILE` of 1024
(3×256=768 upstream conns). That assumption is stale: the fd raise in
`envoyproxy-nofile-patch.yaml` is live on **both** envoy pods — `/proc/1/limits`
reports 65536 soft and hard, with ~414 fds in use at idle. So the old ceiling sat
two orders of magnitude below what the proxy can hold.

Because the cap is per Envoy **cluster**, it is shared by every replica of a model
and does **not** grow when TRE scales out. Two consequences, both measured:

- calibration fitted `theta_m` against that ceiling and mistook it for capacity,
  which is why the 1718/1494/1414 values are invalid by construction;
- in E1 a large share of the 58,907 errors were Envoy shed rather than real SLO
  failures, and removing them flipped the winner on trace t1.

Aggregate ceiling 3×4096=12288 upstream conns still sits well inside the 65536 fd
budget. Real admission control now lives where it belongs: vLLM
`--max-num-seqs 256` per replica, which **does** scale with replica count.

## Apply

```bash
kubectl apply -f backendtrafficpolicy-aibrix-system.yaml
kubectl apply -f backendtrafficpolicy-tre-v2.yaml
kubectl patch envoyproxy aibrix-custom-proxy-config -n aibrix-system   --type merge --patch-file envoyproxy-nofile-patch.yaml
kubectl -n envoy-gateway-system rollout status deploy/envoy-aibrix-system-aibrix-eg-903790dc
kubectl -n envoy-gateway-system rollout status deploy/envoy-tre-v2-tre-aibrix-eg-161007f9
```

## Verify / rollback

- `kubectl get backendtrafficpolicy -A` → all Accepted.
- `kubectl -n envoy-gateway-system exec <envoy-pod> -c envoy -- cat /proc/1/limits | grep 'open files'` → 65536.
- Rollback fd raise: re-apply `envoyproxy-nofile-patch.yaml` with the `command:`
  block removed from the envoy container. Rollback circuit breakers:
  `kubectl delete -f backendtrafficpolicy-*.yaml`.

## Validation (2026-07-08)

Flood 7b @ conc 500 for 140s + probe 14b @ 2rps. Result: 7b returned 78085 × 503
**flags=UO** (fast-fail, p50 125ms) with **zero UF** under load heavier than t1;
14b probe **52/52 = 200** (perfect isolation). Post-test envoy fd back to 414 idle
baseline. Test script archived at `76:/tmp/isolation_test.py`.
