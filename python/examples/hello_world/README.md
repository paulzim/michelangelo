# Hello World

Minimal two-task Ray-only pipeline: generate random numbers, then compute
their mean/std. No external data, no Spark, no storage/registry pushes —
just enough to exercise the `@uniflow.task` / `@uniflow.workflow` decorators
and the DAG execution path end to end.

## Pipeline

```
generate_data  →  compute_stats
     (Ray)             (Ray)
```

## Local Run

No cluster needed — runs entirely in-process:

```bash
cd python
PYTHONPATH=. poetry run python examples/hello_world/hello_world.py
```

## k3d Sandbox Run

### 1. Prerequisites

A Michelangelo sandbox running (`ma sandbox create` or `ma sandbox sync`), and
the `ma-dev-test` namespace + Project CR bootstrapped once per sandbox:

```bash
kubectl get project ma-dev-test -n ma-dev-test 2>/dev/null || \
  ma sandbox demo pipeline
```

`ma sandbox demo pipeline` creates the `ma-dev-test` namespace and applies its
Project CR as a side effect (it also submits its own bundled demo pipelines,
which is harmless here). Both survive `ma sandbox stop`/`start` but are lost
on `ma sandbox delete`.

### 2. Build and import the image

```bash
cd python
docker build -t michelangelo-hello-world:local \
  -f examples/hello_world/.docker/Dockerfile .

k3d image import michelangelo-hello-world:local -c michelangelo-sandbox
```

Redo this after every `ma sandbox delete`/`create` and after a Mac restart
(the k3d node's local image cache doesn't survive either).

### 3. Build and upload the uniflow tar

The controller fetches the workflow's Starlark tarball from the `uniflowTar`
URI in `pipeline.yaml` before it will run — `manifest.filePath` alone is only
used to build that tar, not to execute it. Without a valid `uniflowTar`, the
PipelineRun fails immediately with `failed to get client: scheme  is not
supported` and gets garbage-collected within seconds (no pod, no lingering
PipelineRun CR — easy to mistake for a namespace/Project problem).

```bash
kubectl port-forward svc/minio 9091:9091 -n default &
cd python
PYTHONPATH=. poetry run python examples/hello_world/.docker/rebuild_tar.py
```

Re-run this whenever the workflow DAG or `@uniflow.task(config=...)` changes
(not needed for changes inside a task function body). `pipeline.yaml` already
points at the resulting `s3://michelangelo/uniflow/ma-dev-test_hello-world.tar.gz`.

### 4. Submit the pipeline run

```bash
kubectl apply -f examples/hello_world/pipeline.yaml
kubectl apply -f examples/hello_world/pipelinerun.yaml
```

`pipeline.yaml` sets `michelangelo/uniflow-image-pull-policy: IfNotPresent` —
without it the cluster tries to pull `michelangelo-hello-world:local` from a
registry that doesn't have it, and the pod sits in `ImagePullBackOff`.

### 5. Watch it run

```bash
kubectl get pods -n ma-dev-test -w
kubectl logs -n default deployment/michelangelo-worker --tail=50 -f | \
  grep -E "task_state|SUCCEEDED|FAILED"
```

Expected: `generate_data` then `compute_stats` both `SUCCEEDED`, with the
computed `StatsResult(n=200, mean=..., std=...)` in the logs.

## Troubleshooting

For sandbox-level issues not specific to this pipeline (recovering from a
failed `ma sandbox create`, zombie RayCluster cleanup, MA Studio 415 errors,
etc.), see the failure-mode table in
[`../pipelines/california_housing_xgb/RUNBOOK.md`](../pipelines/california_housing_xgb/RUNBOOK.md).
