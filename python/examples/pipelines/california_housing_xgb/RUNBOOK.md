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
