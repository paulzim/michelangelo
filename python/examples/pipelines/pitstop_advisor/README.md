# Pit Crew Advisor

Training + serving pipeline for the "Michelangelo Pit Stop" KubeCon demo's
Pit Crew Advisor: an XGBoost model that recommends a pit-lane speed cap and
caution buffer for a given lane and track grip. Consumed by the `LaneRun`
controller (`go/components/lanerun`) when a `LaneRun`'s `mode` is
`LANE_RUN_MODE_RECOMMENDED`.

Unlike `california_housing_xgb`, which registers a bare training checkpoint,
this pipeline completes the full serve path: package as a deployable Triton
artifact, upload it, and register both URIs with the model registry.

## Pipeline

```
generate_data  →  train
    (Ray)          (Ray)
```

| Step | File | Description |
|---|---|---|
| `generate_data` | `generate_data.py` | Synthetic `[lane, track_grip] -> [speed_cap_cms, caution_buffer_cm]` rows |
| `train` | `train.py` | Train two XGBoost regressors, package with `CustomTritonPackager`, upload to MinIO, register with the model registry |

The workflow is orchestrated in `pitstop_advisor.py`.

## Model contract

`model.py`'s `PitStopAdvisorModel` takes a single `"features"` input of shape
`[2]` (`[lane, track_grip]`, lane encoded as `0.0`/`1.0`) and returns a single
`"settings"` output of shape `[2]` (`[speed_cap_cms, caution_buffer_cm]`).
This must exactly match `advisorInputName`/`advisorOutputName` in
`go/components/lanerun/controller.go` — the controller builds its KServe v2
request against these hardcoded names, not against the schema at runtime.

## One-time setup per sandbox

```bash
# On the Mac, from the repo root:
cd python
docker build -t pitstop-advisor-local:latest \
  -f examples/pipelines/pitstop_advisor/.docker/Dockerfile .
k3d image import pitstop-advisor-local:latest -c michelangelo-sandbox
```

`ma sandbox demo inference` must already have been run once (provisions
`inference-server-example` in the `default` namespace — see
[deploy-a-model.md](/docs/user-guides/train-and-deploy-models/deploy-a-model.md)).

## Train

```bash
ma pipeline apply -f examples/pipelines/pitstop_advisor/pipeline.yaml
ma pipelinerun apply -f examples/pipelines/pitstop_advisor/pipelinerun.yaml
ma pipelinerun get -n ma-examples --name pitstop-advisor-run
```

Wait for `COMPLETED`. This registers a `Model` named `pitstop-advisor` in the
`ma-examples` namespace, with both a raw checkpoint URI and a deployable
Triton bundle URI uploaded to `s3://deploy-models/pitstop-advisor/<run_id>/`.

## Deploy (first time)

```bash
ma revision apply -f examples/pipelines/pitstop_advisor/revision.yaml
ma deployment apply -f examples/pipelines/pitstop_advisor/deployment.yaml
ma deployment get -n ma-examples --name pitstop-advisor-deployment
```

Wait for `Rollout Complete`, then verify:

```bash
curl -X POST http://localhost:8080/inference-server-example/pitstop-advisor-deployment/infer \
  -H "Content-Type: application/json" \
  -d '{"inputs": [{"name": "features", "shape": [2], "datatype": "FP32", "data": [1.0, 0.75]}]}'
```

## Retrain and promote

Re-running the `PipelineRun` trains a new model but does **not** change what
the `Deployment` serves — there's no auto-promote. After a retrain:

```bash
./promote_revision.sh
```

This creates a new `Revision` from the current `Model` state and re-points
`deployment.yaml`'s `desiredRevision.name` at it.

## Exercise the LaneRun controller

```bash
kubectl apply -f examples/pipelines/pitstop_advisor/lanerun-example.yaml
kubectl get lanerun -n default -o wide
```

`lane-b-recommended` should progress to `DEPLOY_PHASE_ADVISOR_QUERIED` then
`DEPLOY_PHASE_READY`, with `status.recommendedSpeedCapCms` /
`status.recommendedCautionBufferCm` populated from the deployment above. The
controller's advisor endpoint is configured in
`go/cmd/controllermgr/config/base.yaml`'s `lanerun:` section.
