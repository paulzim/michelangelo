#!/usr/bin/env bash
# Promote the latest pitstop-advisor training run to serving.
#
# There is no auto-promote: register_model() (called from train.py during the
# PipelineRun) only updates the Model resource's registry entry. A Deployment
# keeps serving whatever Revision it was last pointed at until this script
# (or the equivalent `ma revision apply` + `ma deployment apply` by hand) runs.
#
# Usage (run on the Mac, after a PipelineRun reaches COMPLETED):
#   ./promote_revision.sh
set -euo pipefail

cd "$(dirname "$0")"

NAMESPACE="ma-examples"
MODEL_NAME="pitstop-advisor"
REVISION_NAME="pitstop-advisor-$(date +%Y%m%d-%H%M%S)"

echo "Creating revision '${REVISION_NAME}' from model '${MODEL_NAME}'..."
cat > /tmp/pitstop-advisor-revision.yaml <<EOF
apiVersion: michelangelo.api/v2
kind: Revision
metadata:
  name: ${REVISION_NAME}
  namespace: ${NAMESPACE}
spec:
  baseType:
    kind: Model
    apiVersion: michelangelo.api/v2
  baseResource:
    name: ${MODEL_NAME}
    namespace: ${NAMESPACE}
  owner:
    name: "${USER}"
EOF
ma revision apply -f /tmp/pitstop-advisor-revision.yaml

echo "Pointing deployment.yaml at '${REVISION_NAME}' and applying..."
# Only replace the `name:` line immediately under `desiredRevision:` — the
# file also has metadata.name and inferenceServer.name, which must stay put.
awk -v new="${REVISION_NAME}" '
  /^[[:space:]]*desiredRevision:/ { in_block=1 }
  in_block && /^[[:space:]]*name:/ && !done {
    sub(/name:.*/, "name: " new)
    done=1
    in_block=0
  }
  { print }
' deployment.yaml > deployment.yaml.tmp && mv deployment.yaml.tmp deployment.yaml

ma deployment apply -f deployment.yaml

echo "Done. Check rollout status with:"
echo "  ma deployment get -n ${NAMESPACE} --name pitstop-advisor-deployment"
