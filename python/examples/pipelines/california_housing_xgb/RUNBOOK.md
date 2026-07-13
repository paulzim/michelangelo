# California Housing XGBoost Pipeline — Local Sandbox Runbook

All commands run on the **Mac** (`/Users/pzimme1/GitHub/michelangelo`).

---

## 1. Activate the Python environment

```bash
cd /Users/pzimme1/GitHub/michelangelo/python
source .venv/bin/activate
```

All `ma` and `kubectl` commands below assume the venv is active. Alternatively,
prefix any `ma` command with `poetry run` instead of activating.

---

## 2. After a Mac restart — bring the sandbox back

```bash
ma sandbox start
```

Check whether Michelangelo pods came back (they usually don't after a restart):

```bash
kubectl get pods -n default
```

If you see Michelangelo pods (`Running`): skip to step 3.

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

**If the k3d server container is missing** (k3d reports "All servers already running" but
`kubectl get pods` refuses to connect), the cluster is unrecoverable — go to step 3.

Wait for all pods to reach `Running`:

```bash
kubectl get pods -n default
```

---

## 3. Deploy the fixed controllermgr (required once per sandbox — do this before the first run)

The published `ghcr.io/michelangelo-ai/controllermgr:main` image predates a fix for
`HeadPodNotFound` being incorrectly treated as a terminal pod error. Without this fix, every
task after the first one fails immediately: kuberay sets `HeadPodNotFound` on the `HeadPodReady`
condition as a transient state right after cluster creation (informer cache lag), but the old
binary treats it as fatal and marks the cluster FAILED before the head pod ever starts.

The fix is in the fork branch (`cfeb1dd0 fix(ray): treat HeadPodNotFound as transient
provisioning state`) but not yet in the upstream published image.

**One-time setup per sandbox lifecycle** (survives `stop/start`, lost on `delete/create`):

```bash
# On Mac — pull the fixed binary from the devpod
scp pzimme1-codebuddy.devpod-us-or:/tmp/controllermgr_arm64 ~/controllermgr-build/controllermgr

# Create the build dir and Dockerfile (if not already present)
mkdir -p ~/controllermgr-build
cat > ~/controllermgr-build/Dockerfile << 'EOF'
FROM gcr.io/distroless/static-debian12:nonroot
COPY controllermgr /controllermgr
ENTRYPOINT ["/controllermgr"]
EOF

# Build and import
cd ~/controllermgr-build
docker build --platform linux/arm64 -t michelangelo-controllermgr-local:fixed .
k3d image import michelangelo-controllermgr-local:fixed -c michelangelo-sandbox

# Update the deployment
kubectl set image deployment/michelangelo-controllermgr \
  app=michelangelo-controllermgr-local:fixed -n default
kubectl rollout status deployment/michelangelo-controllermgr -n default
```

**Note**: `kubectl set image` takes field ownership from Helm, so `ma sandbox sync` will fail
with an SSA conflict on the controllermgr deployment. Fix: `kubectl delete deployment
michelangelo-controllermgr -n default` then `ma sandbox sync`, then re-run the
`kubectl set image` command above.

**If the devpod binary is stale** (e.g. after a devpod rebuild): rebuild it on the devpod:

```bash
# On devpod
cd /home/user/michelangelo/go
CGO_ENABLED=0 GOOS=linux GOARCH=arm64 go build -o /tmp/controllermgr_arm64 ./cmd/controllermgr/...
```

---

## 4. After a full sandbox recreation (if step 2 fails unrecoverably)

If the sandbox is too broken to recover with `sync`, delete and recreate:

```bash
k3d cluster delete michelangelo-sandbox   # or: ma sandbox delete
ma sandbox create
```

`ma sandbox create` takes 10–20 minutes. If helm times out on kuberay-operator or
spark-operator, check `kubectl get pods --all-namespaces` — if those pods are `Running`,
continue with `ma sandbox sync` rather than retrying create.

After a full recreation, three extra steps are required because MinIO and the k3d
image store are wiped:

**Re-apply one-time prerequisites** (namespace and Project CR are lost on delete):

```bash
kubectl create namespace ma-examples
kubectl apply -f /Users/pzimme1/GitHub/michelangelo/python/examples/config/project.yaml
```

**Register the Cadence domain** (`ma sandbox sync` registers it after `_helm_wait()`, but if either sync timed out the domain will be absent):

```bash
kubectl run cadence-reg --restart=Never --image ubercadence/cli:v1.2.6 \
  --env=CADENCE_CLI_ADDRESS=michelangelo-cadence-frontend:7933 \
  --command -- cadence --domain default domain register --rd 1
kubectl logs cadence-reg   # should print: Domain default successfully registered.
kubectl delete pod cadence-reg
```

**Reimport the pipeline image** (k3d's image store is wiped on cluster delete):

```bash
docker tag california-housing-xgb-local:latest california-housing-xgb-local:latest 2>/dev/null || true
k3d image import california-housing-xgb-local:latest -c michelangelo-sandbox
```

If you don't have the image locally, rebuild it first (step 8 below).

**Patch the michelangelo-config ConfigMap** to add the registry endpoint (fresh clusters are missing it, causing `push_step` to use in-memory registration instead of the API):

```bash
kubectl patch configmap michelangelo-config -n default \
  --patch '{"data":{"REGISTRY_ENDPOINT":"michelangelo-apiserver:15566","REGISTRY_NAMESPACE":"ma-examples"}}'
```

**Rebuild the uniflowTar** (MinIO is empty on a fresh cluster):

```bash
cd /Users/pzimme1/GitHub/michelangelo/python
poetry run python examples/pipelines/california_housing_xgb/.docker/rebuild_tar.py
```

Then continue from step 6.

---

## 5. Pre-run cleanup (always do this before submitting a pipeline run)

Zombie RayCluster objects accumulate across failed runs and eventually cause
`create_cluster` to return nil with no obvious error. There are **two CRD groups**
that must both be cleaned up — `rayclusters.michelangelo.api` (Michelangelo's own
CRDs) and `rayclusters.ray.io` (kuberay's CRDs). `kubectl delete raycluster` without
a group qualifier only deletes the `ray.io` ones; the Michelangelo CRDs pile up
silently and continuously saturate the controllermgr reconcile queue.

```bash
kubectl delete raycluster.michelangelo.api -n default --all
kubectl delete raycluster.ray.io -n default --all
kubectl delete pod -n default --field-selector=status.phase=Failed
```

---

## 6. Verify one-time prerequisites

These survive `stop/start` but are lost on `ma sandbox delete`:

```bash
# Namespace
kubectl get namespace ma-examples || kubectl create namespace ma-examples

# Project CR
kubectl get project ma-examples -n ma-examples 2>/dev/null || \
  kubectl apply -f /Users/pzimme1/GitHub/michelangelo/python/examples/config/project.yaml
```

---

## 7. Pull latest changes from fork (if needed)

```bash
cd /Users/pzimme1/GitHub/michelangelo
git fetch paulzim feat/pipeline-local-run-example
git checkout feat/pipeline-local-run-example
git merge paulzim/feat/pipeline-local-run-example
```

---

## 8. Build and import the pipeline image

Only needed when you change the Dockerfile or pipeline Python code.
The image survives `sandbox stop/start` — skip this step if nothing changed.

```bash
cd /Users/pzimme1/GitHub/michelangelo/python
docker build -t california-housing-xgb-local:latest \
  -f examples/pipelines/california_housing_xgb/.docker/Dockerfile .

k3d image import california-housing-xgb-local:latest -c michelangelo-sandbox
```

---

## 9. Rebuild the uniflowTar (only when @uniflow.task config changes)

Required when you change the `@uniflow.task(config=...)` decorator on any task
(e.g. switching between RayTask and SparkTask, or changing resource limits).
**Not** required for changes inside the task function body.

```bash
cd /Users/pzimme1/GitHub/michelangelo/python
poetry run python examples/pipelines/california_housing_xgb/.docker/rebuild_tar.py
```

---

## 10. Submit the pipeline run

```bash
kubectl apply -f /Users/pzimme1/GitHub/michelangelo/python/examples/pipelines/california_housing_xgb/pipeline.yaml

kubectl delete pipelinerun california-housing-xgb-run -n ma-examples --ignore-not-found

kubectl apply -f /Users/pzimme1/GitHub/michelangelo/python/examples/pipelines/california_housing_xgb/pipelinerun.yaml
```

---

## 11. Watch the run

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

## 12. Verify model registration

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
| PipelineRun fails immediately: `EntityNotExistsError Domain default does not exist` | Cadence domain not registered (`_helm_wait` timed out during sync, skipping registration) | Run `kubectl run cadence-reg --restart=Never --image ubercadence/cli:v1.2.6 --env=CADENCE_CLI_ADDRESS=michelangelo-cadence-frontend:7933 --command -- cadence --domain default domain register --rd 1`; check logs; delete pod; resubmit run |
| Task 2+ immediately fail: `internal error: nil (not None) returned from create_cluster` | Deployed controllermgr treats `HeadPodNotFound` as terminal (bug predating `cfeb1dd0`) | Step 3: deploy the fixed controllermgr image |
| `ma model get --namespace ma-examples` returns empty after successful push_step | `REGISTRY_ENDPOINT` missing from `michelangelo-config` ConfigMap on fresh cluster | Step 4: patch configmap with `REGISTRY_ENDPOINT` and `REGISTRY_NAMESPACE`, resubmit run |
| `ma sandbox sync` fails: `conflict with "kubectl-set"` | Helm SSA field manager conflict | `kubectl delete deployment <name>` then re-sync |
| `create_cluster` returns nil, task fails before Python runs | Zombie RayClusters filling namespace | Step 5: delete both `raycluster.michelangelo.api` and `raycluster.ray.io` in default namespace |
| `ma sandbox sync` fails: `CalledProcessError` on MySQL exec | MySQL pod not running (chart never deployed) | `ma sandbox delete` then `ma sandbox create` |
| PipelineRun disappears immediately after apply | Missing Project CR in namespace | Step 4: apply project.yaml |
| push_step fails: `PusherPluginError` | Proto/module mismatch in image | Rebuild image (step 8); check diagnostic.py output |
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
