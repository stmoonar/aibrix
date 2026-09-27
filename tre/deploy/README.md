# TRE deploy: registry and service-manager configuration

## The registry is the single source of configuration

`deploy/registry.yaml` declares the cluster (nodes, GPUs, `max_bound_per_gpu`),
the models, the gateway (`gateway.route_timeout_s`) and the service-manager
behaviour (`service_manager:`). Every key of the `service_manager:` section is
optional; the shipped file spells out the built-in defaults with comments.

### Changing the live registry

The live registry is the `tre-v2-registry` ConfigMap (mounted at `/etc/tre`
by the controller and the service-manager).

- Change it **only through the console**: `PUT /api/params`.
- **Never** `kubectl apply` `overlays/tre-v2/params.yaml`: it is a bootstrap
  copy for a fresh install and would overwrite the live values (for example
  calibrated `theta_m`).
- The service-manager reads `service_manager:` and `gateway:` once, at start.
  After changing them, restart it:
  `kubectl -n tre-v2 rollout restart deploy/tre-v2-service-manager`.
  The controller restart is offered by the console.

### Checked at start

- The service-manager refuses to start when `service_manager:` / `gateway:` is
  invalid. One check is the worst-case duration of one sleeping call, for any
  number of targets (they drain and commit in parallel):
  `writer_lock_wait_s + sleep.ack_timeout_s + sleep.hard_cap_s +
  commit-lock wait + 2 x sleep.sleep_call_timeout_s + 6 x sleep.probe_timeout_s +
  sleep.physical_confirm_timeout_s + sleep.io_margin_s` must be below
  `service_manager.api_call_timeout_s`. Another: `sleep.reservation_ttl_s` must
  exceed `commit-lock wait + sleep.poll_interval_s + sleep.probe_timeout_s +
  sleep.io_margin_s` (the longest gap between two reservation renewals).
- The controller uses `service_manager.api_call_timeout_s` as its timeout for
  slow service-manager calls, unless `TRE_SM_SLOW_TIMEOUT_SECONDS` overrides it.
  It refuses to start when that timeout does not exceed the same worst case.

### Route timeout

`gateway.route_timeout_s` is the request timeout of every model route:

- `make manifests` writes it into each model HTTPRoute;
- the service-manager drain hard cap (`service_manager.sleep.hard_cap_s`)
  defaults to it and may not exceed it;
- the ext_proc route timeout in `overlays/tre-v2/gateway-extproc.yaml` must
  carry the same value (a guard test in `deploy/tests` enforces it).

### Gateway plugin contract

The service-manager and the gateway plugin share Redis keys keyed by pod
**name** only (`tre:v2:gw:seen:<pod>`, `tre:v2:gw:inflight:<pod>`). TRE model
pods must therefore have unique names across namespaces. The generated
Deployments put model, node and GPUs in the name, which guarantees this.

### Service-manager rollout

The service-manager Deployment uses `strategy: Recreate`, so only one writer
runs at a time. `terminationGracePeriodSeconds` must exceed the SIGTERM wait
`ServiceManagerConfig.shutdown_timeout_s()` (commit-lock wait + a parallel
commit + one poll round, computed from the same values as the call timeout); a
guard test enforces it. On SIGTERM the service-manager:

1. stops accepting new sleeps;
2. rolls back every drain that has not reached `/sleep`;
3. lets a sleep that is already past `/sleep` finish.

On start (and on every supervisor pass) it resolves any sleep journal entries
that a dead instance left behind. A pod whose `/sleep` may still be running is
re-opened for routing only after it read awake twice, more than
`sleep.sleep_call_timeout_s` apart.

### Tests

`make check` is hermetic: the Lua scripts (sleep reservations, the fair writer
lock) run against Python models of them. `make check-redis` runs the same tests
against a real throwaway Redis container (`REDIS_TEST_IMAGE`, default
`redis:7.2-alpine`) and removes it afterwards.
