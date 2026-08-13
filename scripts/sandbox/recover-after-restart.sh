#!/usr/bin/env bash
#
# recover-after-restart.sh — Collapse the manual recovery checklist for
# `ma sandbox start` after a Mac restart into one idempotent script.
#
# SCOPE: this is a *restart* recovery script — it assumes the k3d cluster and
# its volumes already exist (you ran `ma sandbox stop` then `ma sandbox
# start`, or the Mac itself rebooted and k3d came back on its own). It does
# NOT perform the Zscaler CA docker-cp fix into the k3d node containers —
# that is only needed after a full `ma sandbox delete` + `ma sandbox create`
# (see the `sandbox_zscaler_ca_fixes` note), because a stop/start cycle never
# touches the node containers' trust store. If you just ran delete+create,
# do the CA fix first, then this script.
#
# What this covers (see CLAUDE.md "Sandbox install gotchas" for the full
# writeup of each):
#   1. `mysql`/`minio` are bare Pods with no Deployment/PVC — if a node
#      restart evicts them, nothing recreates them, and `ma sandbox sync`
#      cannot repair this itself: gotcha #22 confirms `_sync()` deliberately
#      never touches infra resources, and gotcha #9 confirms sync outright
#      fails if MySQL isn't already reachable. Reapply their raw manifests
#      defensively before sync runs (see `michelangelo-examples`'s
#      `docs/sandbox-troubleshooting.md` for the incident this came from —
#      MySQL data loss here also wipes any model-registry Project CRs).
#   2. `ma sandbox start` only restarts k3d nodes — Michelangelo pods do not
#      come back on their own (gotcha #8). `ma sandbox sync` redeploys them.
#   3. Pods can take a few minutes to settle after sync; poll until healthy
#      instead of racing the very next step.
#   4. A `sync`-based recovery never re-runs the compute-cluster CRD/RBAC/
#      secrets setup that `create` does (gotcha #4) — backfill defensively.
#   5. The `ma-examples` namespace and its Project CR are lost on `delete`
#      but NOT on `stop`/`start` — however if the cluster was ever recovered
#      via `sync` after a wipe, they may be missing (gotchas #10/#11).
#   6. If `_helm_wait()` timed out on this or a previous sync, Cadence domain
#      registration can be skipped (gotcha #5) — verify/register it.
#   7. Zombie RayCluster CRs (both `ray.io` and `michelangelo.api` groups)
#      accumulate across failed runs and saturate the reconcile queue
#      (gotcha #13) — clean them defensively on every run.
#   8. Locally-built images (controllermgr overrides, example pipeline
#      images) live in the k3d node's containerd content store, which does
#      survive stop/start — but re-import defensively in case the node
#      volume was recreated.
#
# Usage:
#   scripts/sandbox/recover-after-restart.sh
#
# Environment overrides:
#   CLUSTER_NAME         — k3d cluster name (default: michelangelo-sandbox)
#   NAMESPACE            — namespace the Michelangelo app runs in (default: default)
#   EXAMPLES_NAMESPACE   — namespace demo PipelineRuns target (default: ma-examples)
#   POD_WAIT_TIMEOUT     — seconds to poll for pods to settle (default: 300)
#   LOCAL_IMAGE_PATTERN  — grep pattern for local images to re-import (default: -local)
#
# Exit codes:
#   0 = completed (see final pod summary for actual health)
#   1 = a required precondition failed (cluster not found, poetry/kubectl missing)

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_DIR="$REPO_ROOT/python"
PROJECT_YAML="$PYTHON_DIR/examples/config/project.yaml"
MYSQL_YAML="$PYTHON_DIR/michelangelo/cli/sandbox/resources/mysql.yaml"
MINIO_YAML="$PYTHON_DIR/michelangelo/cli/sandbox/resources/minio.yaml"

CLUSTER_NAME="${CLUSTER_NAME:-michelangelo-sandbox}"
NAMESPACE="${NAMESPACE:-default}"
EXAMPLES_NAMESPACE="${EXAMPLES_NAMESPACE:-ma-examples}"
POD_WAIT_TIMEOUT="${POD_WAIT_TIMEOUT:-300}"
LOCAL_IMAGE_PATTERN="${LOCAL_IMAGE_PATTERN:--local}"

log() { echo "[recover] $*"; }

if ! command -v kubectl >/dev/null 2>&1; then
  echo "kubectl not found on PATH" >&2
  exit 1
fi
if ! command -v poetry >/dev/null 2>&1; then
  echo "poetry not found on PATH (needed for \`ma sandbox\` commands)" >&2
  exit 1
fi
if ! k3d cluster get "$CLUSTER_NAME" >/dev/null 2>&1; then
  echo "k3d cluster '$CLUSTER_NAME' not found. This script only handles" >&2
  echo "restart recovery; run \`ma sandbox create\` for a fresh cluster." >&2
  exit 1
fi

# 1. Reapply mysql/minio bare-Pod manifests if a node eviction dropped them —
# `ma sandbox sync` cannot recover this on its own (gotchas #9/#22).
log "Step 1/8: checking mysql/minio pods in namespace '$NAMESPACE'"
mysql_missing=0
minio_missing=0
kubectl get pod mysql -n "$NAMESPACE" >/dev/null 2>&1 || mysql_missing=1
kubectl get pod minio -n "$NAMESPACE" >/dev/null 2>&1 || minio_missing=1
if [ "$mysql_missing" -eq 0 ] && [ "$minio_missing" -eq 0 ]; then
  log "  mysql and minio pods already present — skipping."
else
  if [ "$mysql_missing" -eq 1 ]; then
    if [ -f "$MYSQL_YAML" ]; then
      log "  mysql pod missing — reapplying $MYSQL_YAML"
      kubectl apply -f "$MYSQL_YAML" >/dev/null
    else
      log "  WARNING — mysql pod missing and $MYSQL_YAML not found, skipping."
    fi
  fi
  if [ "$minio_missing" -eq 1 ]; then
    if [ -f "$MINIO_YAML" ]; then
      log "  minio pod missing — reapplying $MINIO_YAML"
      kubectl apply -f "$MINIO_YAML" >/dev/null
    else
      log "  WARNING — minio pod missing and $MINIO_YAML not found, skipping."
    fi
  fi
  log "  waiting for mysql/minio pods to become ready (timeout 120s)"
  kubectl wait --for=condition=ready pod/mysql pod/minio -n "$NAMESPACE" --timeout=120s || \
    log "  WARNING — mysql/minio did not report ready in time; continuing anyway."
fi

# 2. Redeploy the Michelangelo app pods (gotcha #8).
log "Step 2/8: ma sandbox sync"
(cd "$PYTHON_DIR" && poetry run ma sandbox sync)

# 3. Poll until nothing is Pending/ContainerCreating, surfacing crash-loops.
log "Step 3/8: waiting for pods in namespace '$NAMESPACE' to settle (timeout ${POD_WAIT_TIMEOUT}s)"
deadline=$(( $(date +%s) + POD_WAIT_TIMEOUT ))
while true; do
  pod_lines="$(kubectl get pods -n "$NAMESPACE" --no-headers 2>/dev/null || true)"
  if [ -z "$pod_lines" ]; then
    log "  no pods found yet in '$NAMESPACE'..."
  else
    unsettled="$(echo "$pod_lines" | awk '$3 ~ /Pending|ContainerCreating|PodInitializing|Init:/ {print $1}')"
    crashing="$(echo "$pod_lines" | awk '$3 ~ /CrashLoopBackOff|Error|ImagePullBackOff|ErrImagePull/ {print $1 ": " $3}')"
    if [ -n "$crashing" ]; then
      log "  WARNING — pods in a failure state (will keep polling until timeout):"
      echo "$crashing" | sed 's/^/[recover]   /'
    fi
    if [ -z "$unsettled" ]; then
      log "  all pods have left Pending/ContainerCreating."
      break
    fi
    log "  still settling: $(echo "$unsettled" | tr '\n' ' ')"
  fi
  if [ "$(date +%s)" -ge "$deadline" ]; then
    log "  timed out after ${POD_WAIT_TIMEOUT}s — continuing anyway, check the final summary."
    break
  fi
  sleep 5
done

# 4. Backfill compute-cluster CRD/RBAC/secrets if a sync-recovery skipped them (gotcha #4).
log "Step 4/8: checking compute-cluster CRD/RBAC/secrets"
if kubectl get secret "cluster-${CLUSTER_NAME}-ca-data" -n "$NAMESPACE" >/dev/null 2>&1 \
  && kubectl get secret "cluster-${CLUSTER_NAME}-client-token" -n "$NAMESPACE" >/dev/null 2>&1; then
  log "  compute-cluster secrets already present — skipping backfill."
else
  log "  compute-cluster secrets missing — backfilling via sandbox.py helpers."
  (cd "$PYTHON_DIR" && poetry run python3 -c "
from michelangelo.cli.sandbox.sandbox import (
    _create_compute_cluster_crd, _apply_compute_cluster_rbac,
    _create_compute_cluster_secrets, _michelangelo_sandbox_kube_cluster_name,
)
name = _michelangelo_sandbox_kube_cluster_name
_create_compute_cluster_crd(name)
_apply_compute_cluster_rbac(name)
_create_compute_cluster_secrets(name)
")
fi

# 5. Ensure the ma-examples namespace and Project CR exist (gotchas #10/#11).
log "Step 5/8: ensuring '$EXAMPLES_NAMESPACE' namespace and Project CR"
kubectl create namespace "$EXAMPLES_NAMESPACE" --dry-run=client -o yaml | kubectl apply -f - >/dev/null
if [ -f "$PROJECT_YAML" ]; then
  kubectl apply -f "$PROJECT_YAML" >/dev/null
  log "  applied $PROJECT_YAML"
else
  log "  WARNING — $PROJECT_YAML not found, skipping Project CR apply."
fi

# 6. Verify/register the Cadence domain if this sandbox uses Cadence (gotcha #5).
log "Step 6/8: checking Cadence domain registration"
if kubectl get deployment -n "$NAMESPACE" \
  -l app.kubernetes.io/name=cadence,app.kubernetes.io/component=frontend \
  --no-headers 2>/dev/null | grep -q .; then
  (cd "$PYTHON_DIR" && poetry run python3 -c "
from michelangelo.cli.sandbox.sandbox import _create_cadence_domain
_create_cadence_domain([])
")
else
  log "  no Cadence frontend deployment found — assuming Temporal, skipping."
fi

# 7. Clean zombie RayClusters across both CRD groups + failed pods (gotcha #13).
log "Step 7/8: cleaning zombie RayCluster CRs and failed pods"
kubectl delete raycluster.michelangelo.api -n "$NAMESPACE" --all --ignore-not-found >/dev/null 2>&1 || true
kubectl delete raycluster.ray.io -n "$NAMESPACE" --all --ignore-not-found >/dev/null 2>&1 || true
kubectl delete pod -n "$NAMESPACE" --field-selector=status.phase=Failed --ignore-not-found >/dev/null 2>&1 || true

# 8. Re-import any locally-built images into k3d (survives stop/start, but
# re-import defensively in case the node volume was recreated).
log "Step 8/8: re-importing local images matching '*${LOCAL_IMAGE_PATTERN}*' into k3d"
local_images="$(docker images --format '{{.Repository}}:{{.Tag}}' 2>/dev/null | grep -- "$LOCAL_IMAGE_PATTERN" || true)"
if [ -n "$local_images" ]; then
  # shellcheck disable=SC2086
  k3d image import $local_images -c "$CLUSTER_NAME"
  echo "$local_images" | sed 's/^/[recover]   imported: /'
else
  log "  no local images found matching '*${LOCAL_IMAGE_PATTERN}*'."
fi

log "Done. Final pod summary for namespace '$NAMESPACE':"
kubectl get pods -n "$NAMESPACE"
