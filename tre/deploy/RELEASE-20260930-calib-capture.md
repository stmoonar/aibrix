# Release 20260930-f8ccb0ca: calibration capture (control plane only)

Plan only, not executed. Code: main merge f8ccb0ca (feat/calib-capture-20260930).
Images (tag `20260930-f8ccb0ca`, present on both nodes, IDs identical): gateway-plugins
`aibrix/gateway-plugins:20260930-f8ccb0ca-nozmq2`, `tre-v2-service-manager`,
`tre-v2-controller`, `tre-v2-ui`. Model Deployments, registry ConfigMap and vLLM images
are NOT touched; no model pod is rebuilt.

Confirm with the user before running (other sessions may use the cluster).

## 1. Backup (new directory, same recipe as backups/pre-integ-20260930-1214)

```bash
B=/data/nfs_shared_data/xxy/backups/tre-v2-state-$(date +%Y%m%d-%H%M)-pre-calib-capture
mkdir -p $B
kubectl -n tre-v2 get deploy,svc,cm,sa,role,rolebinding -o yaml > $B/tre-v2-ns.yaml
kubectl -n tre-v2 get cm tre-v2-registry -o jsonpath="{.data.registry\.yaml}" > $B/live-registry.yaml
kubectl -n tre-v2 exec deploy/tre-v2-redis -- redis-cli SAVE
kubectl -n tre-v2 exec deploy/tre-v2-redis -- cat /data/dump.rdb > $B/tre-v2-redis.rdb
kubectl -n tre-v2 get deploy tre-gateway-plugins tre-v2-service-manager tre-v2-controller tre-v2-ui -o yaml > $B/control-plane-deploys.yaml
scripts/set_run_mode.sh status > $B/run-mode.txt 2>&1
git -C /data/nfs_shared_data/xxy/aibrix rev-parse main > $B/main-sha.txt
```

Write and review `$B/rollback.sh` (section 4) before step 2. Record the current run mode
(controller/SM) and images in `$B/README.txt`.

## 2. Swap (order: plugin -> SM -> controller -> UI)

```bash
cd /data/nfs_shared_data/xxy/aibrix/tre
kubectl apply -f deploy/overlays/tre-v2/gateway-plugins.yaml && kubectl -n tre-v2 rollout status deploy/tre-gateway-plugins
kubectl apply -f deploy/overlays/tre-v2/service-manager.yaml && kubectl -n tre-v2 rollout status deploy/tre-v2-service-manager   # Recreate
kubectl apply -f deploy/overlays/tre-v2/controller.yaml     && kubectl -n tre-v2 rollout status deploy/tre-v2-controller
kubectl apply -f deploy/overlays/tre-v2/ui.yaml             && kubectl -n tre-v2 rollout status deploy/tre-v2-ui
```

Do not change run mode; do not apply overlays/tre-v2/params.yaml; do not touch `default` ns.

## 3. Verification (stop at first failure, then roll back)

1. Run mode unchanged: same controller/SM values as recorded in `$B/run-mode.txt`.
2. Pods Running/Ready, no restarts; images match the tag.
3. A controller decision (log or console decision view) contains the field `trs_raw`.
4. `GET /v2/audit` once (never polled): healthy/empty.
5. Fleet still 20/20 Ready; gateway 31094 answers one request.

## 4. Rollback (whole Deployment object, env included)

Never `kubectl set image` alone. Order: controller -> SM -> plugin -> UI. For each of
tre-v2-controller, tre-v2-service-manager (stays Recreate), tre-gateway-plugins,
tre-v2-ui: take the object from `$B/tre-v2-ns.yaml` / `$B/control-plane-deploys.yaml`,
strip `resourceVersion`, `uid`, `creationTimestamp`, `generation`, `managedFields`,
`status`, `kubectl replace -f`, then `rollout status`. Restore run mode to the recorded
value with `set_run_mode.sh`. Redis rdb only if desired state is corrupted.
