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
| `backendtrafficpolicy-aibrix-system.yaml` | per-model circuit breakers on the APA / experiment-3 control path (Gateway aibrix-system/aibrix-eg, NodePort 31592) | additive, aibrix-system ns |
| `backendtrafficpolicy-tre-v2.yaml` | per-model circuit breakers on the TRE serving path (Gateway tre-v2/tre-aibrix-eg, NodePort 31094) | additive, tre-v2 ns |
| `envoyproxy-nofile-patch.yaml` | raise envoy RLIMIT_NOFILE 1024→65536 | **modifies shared aibrix-system EnvoyProxy** (class-level → both envoys) |

Circuit-breaker values (per model, per gateway): maxConnections 256,
maxPendingRequests 64, maxParallelRequests 256, maxParallelRetries 16. Each model
gets its own bounded quota, so overload fast-fails as 503 **UO** (overflow,
returned in ms before any socket attempt) confined to the saturated model, instead
of UF socket_creation_failure that spreads across all models. The cap is a
ceiling, not a steady target (a single A100-40G vLLM replica's productive
concurrency is well below 256); it only stops the runaway pile-up. Aggregate
ceiling 3×256=768 upstream conns sits well inside the raised 65536 fd budget. The
cap is cluster-wide (shared across a model's replicas) — retune as replicas scale.

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
