# California Housing XGBoost Pipeline — Local Sandbox Runbook

All commands run on the **Mac** (`/Users/pzimme1/GitHub/michelangelo`).

---

## 1. After a Mac restart — bring the sandbox back

```bash
cd /Users/pzimme1/GitHub/michelangelo/python
ma sandbox start
```

Check whether Michelangelo pods came back (they usually don't after a restart):

```bash
kubectl get pods -n default
```

If you see Michelangelo pods (`Running`): skip to step 2.

If you see only `kube-system` pods, run sync to redeploy the Helm chart:

```bash
ma sandbox sync
```

**If sync fails** with `conflict with "kubectl-set" using apps/v1` (Helm SSA field manager conflict):

```bash
# Find the conflicting deployment name in the error message, then:
kubectl delete deployment michelangelo-controllermgr   # or whichever is named
ma sandbox sync
```

Wait for all pods to reach `Running`:

```bash
kubectl get pods -n default
```

---

## 2. Pre-run cleanup (always do this before submitting a pipeline run)

Zombie RayCluster objects accumulate across failed runs and eventually cause
`create_cluster` to return nil with no obvious error. Clean them up first:

```bash
kubectl delete raycluster -n default --all
kubectl delete pod -n default --field-selector=status.phase=Failed
```

---

## 3. Verify one-time prerequisites

These survive `stop/start` but are lost on `ma sandbox delete`:

```bash
# Namespace
kubectl get namespace ma-examples || kubectl create namespace ma-examples

# Project CR
kubectl get project ma-examples -n ma-examples 2>/dev/null || \
  kubectl apply -f /Users/pzimme1/GitHub/michelangelo/python/examples/config/project.yaml
```

---

## 4. Pull latest changes from fork (if needed)

```bash
cd /Users/pzimme1/GitHub/michelangelo
git fetch paulzim feat/pipeline-local-run-example
git checkout feat/pipeline-local-run-example
git merge paulzim/feat/pipeline-local-run-example
```

---

## 5. Build and import the pipeline image

Only needed when you change the Dockerfile or pipeline Python code.
The image survives `sandbox stop/start` — skip this step if nothing changed.

```bash
cd /Users/pzimme1/GitHub/michelangelo/python
docker build -t california-housing-xgb-local:latest \
  -f examples/pipelines/california_housing_xgb/.docker/Dockerfile .

k3d image import california-housing-xgb-local:latest -c michelangelo-sandbox
```

---

## 6. Rebuild the uniflowTar (only when @uniflow.task config changes)

Required when you change the `@uniflow.task(config=...)` decorator on any task
(e.g. switching between RayTask and SparkTask, or changing resource limits).
**Not** required for changes inside the task function body.

```bash
cd /Users/pzimme1/GitHub/michelangelo/python
poetry run python examples/pipelines/california_housing_xgb/.docker/rebuild_tar.py
```

---

## 7. Submit the pipeline run

```bash
kubectl apply -f /Users/pzimme1/GitHub/michelangelo/python/examples/pipelines/california_housing_xgb/pipeline.yaml

kubectl delete pipelinerun california-housing-xgb-run -n ma-examples --ignore-not-found

kubectl apply -f /Users/pzimme1/GitHub/michelangelo/python/examples/pipelines/california_housing_xgb/pipelinerun.yaml
```

---

## 8. Watch the run

```bash
kubectl logs -n default deployment/michelangelo-worker --tail=50 -f | \
  grep -E "task_state|SUCCEEDED|FAILED|full traceback|Error|Traceback"
```

Expected sequence (each ~1 min apart):
1. `feature_prep` → `SUCCEEDED`
2. `preprocess` → `SUCCEEDED`
3. `train` → `SUCCEEDED` (validation-rmse ~1.39)
4. `push_step` → `SUCCEEDED`

---

## 9. Verify model registration

```bash
cd /Users/pzimme1/GitHub/michelangelo/python
poetry run ma model get --namespace ma-examples
```

Expected output:
```
 NAMESPACE    NAME                   LAST_UPDATED_SPEC
 ma-examples  california-housing-xgb <timestamp>
```

---

## Quick reference — common failure modes

| Symptom | Cause | Fix |
|---------|-------|-----|
| `ma sandbox sync` fails: `conflict with "kubectl-set"` | Helm SSA field manager conflict | `kubectl delete deployment <name>` then re-sync |
| `create_cluster` returns nil, task fails before Python runs | Zombie RayClusters filling namespace | Step 2: delete all rayclusters + failed pods |
| `ma sandbox sync` fails: `CalledProcessError` on MySQL exec | MySQL pod not running (chart never deployed) | `ma sandbox delete` then `ma sandbox create` |
| PipelineRun disappears immediately after apply | Missing Project CR in namespace | Step 3: apply project.yaml |
| push_step fails: `PusherPluginError` | Proto/module mismatch in image | Rebuild image (step 5); check diagnostic.py output |
| Stale cache error: `failed to read object: key does not exist` | Cache entry from a previous failed run | `kubectl delete configmap -n ma-examples -l michelangelo/uniflow-task-path` |
| MA Studio tables show "Unable to fetch data", DevTools shows HTTP 415 | Envoy `http_filters` regressed to `grpc_web`, but the browser client uses the Connect protocol | See "MA Studio 415 errors" below |

---

## MA Studio 415 errors (all tables fail to load)

**Symptom**: `http://localhost:8090/ma-examples` loads navigation, but every table
shows "Unable to fetch data for table". Browser DevTools → Network tab shows HTTP
415 (Unsupported Media Type) on the XHR/fetch calls. The backend itself is healthy
(`poetry run ma model get --namespace ma-examples` works fine via gRPC).

**Cause**: The Studio frontend (`javascript/packages/rpc/services.ts`) uses
`createConnectTransport` — the Connect protocol, sent as `application/json`. Envoy's
`michelangelo-envoy` deployment must run the `envoy.filters.http.connect_grpc_bridge`
filter to translate that into native gRPC for the apiserver. If that filter is instead
`envoy.filters.http.grpc_web` (which only accepts `application/grpc-web+proto`),
Connect's JSON requests aren't recognized and get rejected with 415.

This is a one-line regression that has recurred at least once before (an earlier
sandbox debugging session swapped the filter to `grpc_web` for an unrelated fix and
never swapped it back) — check this first before assuming a new bug.

**Diagnose**:

```bash
kubectl get configmap -n default -o yaml | grep -A2 "http_filters"
```

Should show `envoy.filters.http.connect_grpc_bridge`. If it shows `envoy.filters.http.grpc_web`,
that's the bug.

**Fix**: edit `helm/michelangelo/templates/core/envoy-configmap.yaml` on the devpod so the
`http_filters` block reads:

```yaml
http_filters:
  - name: envoy.filters.http.connect_grpc_bridge
    typed_config:
      "@type": type.googleapis.com/envoy.extensions.filters.http.connect_grpc_bridge.v3.FilterConfig
```

Commit and push it, then on the Mac:

```bash
cd /Users/pzimme1/GitHub/michelangelo
git fetch paulzim feat/pipeline-local-run-example
git merge paulzim/feat/pipeline-local-run-example
ma sandbox sync
kubectl get configmap -n default -o yaml | grep -A2 "http_filters"   # confirm connect_grpc_bridge
```

**Note**: `ma sandbox sync`'s Helm upgrade does restart `michelangelo-envoy` as part of its
flow, so a manual `kubectl rollout restart` typically isn't needed after `sync` — but if
you ever patch the ConfigMap directly (`kubectl edit configmap` / `kubectl apply` outside
of `sandbox sync`), you must restart the deployment manually, since there's no checksum
annotation on `envoy-deployment.yaml` to trigger an automatic rollout:

```bash
kubectl rollout restart deployment/michelangelo-envoy -n default
```
