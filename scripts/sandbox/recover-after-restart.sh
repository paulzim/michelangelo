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
#   1. After a node restart, `kubectl` can fail with `Unable to connect to
#      the server: EOF` because the k3d `serverlb` proxy holds stale backend
#      connections (gotcha #23) — detect this and restart serverlb before
#      anything else, since every later step depends on kubectl working.
#   2. `mysql`/`minio` are bare Pods with no Deployment/PVC — if a node
#      restart evicts them, nothing recreates them, and `ma sandbox sync`
#      cannot repair this itself: gotcha #22 confirms `_sync()` deliberately
#      never touches infra resources, and gotcha #9 confirms sync outright
#      fails if MySQL isn't already reachable. In the worst case (gotcha
#      #28) both pods are entirely absent — reapply their raw manifests,
#      plus minio's credentials Secret and bucket-setup Job, defensively
#      before sync runs.
#   3. `ma sandbox start` only restarts k3d nodes — Michelangelo pods do not
#      come back on their own (gotcha #8). `ma sandbox sync` redeploys them
#      — though if the prior release was broken/absent, sync silently falls
#      back to a full `helm uninstall` + fresh install (gotcha #29), which
#      matters for step 5 below.
#   4. Pods can take a few minutes to settle after sync; poll until healthy
#      instead of racing the very next step.
#   5. A fresh Helm install (see #3) points apiserver/controllermgr/ui/worker
#      at the published `ghcr.io/michelangelo-ai/*:main` images, and the k3d
#      node's own containerd has a never-fully-diagnosed x509 trust failure
#      pulling from ghcr.io specifically, even with the right CA present
#      (gotchas #24/#25) — so expect `ImagePullBackOff`/`ErrImagePull` on
#      all of them. Work around it the same way gotcha #26 does: pull each
#      affected image by exact platform digest via the Mac's own (Zscaler-
#      clean) Docker daemon, tag it back to the original ref, and
#      `k3d image import` it directly — bypassing node-internal pulls
#      entirely — then delete the stuck pods so they pick up the local copy.
#   6. A `sync`-based recovery never re-runs the compute-cluster CRD/RBAC/
#      secrets setup that `create` does (gotcha #4) — backfill defensively.
#   7. The `ma-examples` namespace and its Project CR are lost on `delete`
#      but NOT on `stop`/`start` — however if the cluster was ever recovered
#      via `sync` after a wipe, they may be missing (gotchas #10/#11).
#   8. If `_helm_wait()` timed out on this or a previous sync, Cadence domain
#      registration can be skipped (gotcha #5) — verify/register it.
#   9. Zombie RayCluster CRs (both `ray.io` and `michelangelo.api` groups)
#      accumulate across failed runs and saturate the reconcile queue
#      (gotcha #13) — clean them defensively on every run.
#  10. Locally-built images (controllermgr overrides, example pipeline
#      images) live in the k3d node's containerd content store, which does
#      survive stop/start — but re-import defensively in case the node
#      volume was recreated.
#
# NOT covered: the root cause of #5's node-internal ghcr.io trust failure —
# it's a workaround, not a fix. Also does not redo the Zscaler CA docker-cp
# fix (see SCOPE above) or attempt anything for registries other than
# ghcr.io, since that's the only one seen to hit this failure mode so far.
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
#   SERVERLB_CONTAINER   — k3d serverlb container name (default: k3d-<CLUSTER_NAME>-serverlb)
#   IMAGE_PLATFORM       — os/arch to resolve ghcr.io digests for (default: linux/arm64)
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
MINIO_CREDS_YAML="$PYTHON_DIR/michelangelo/cli/sandbox/resources/minio-credentials.yaml"
BUCKET_SETUP_YAML="$PYTHON_DIR/michelangelo/cli/sandbox/resources/sandbox-bucket-setup.yaml"

CLUSTER_NAME="${CLUSTER_NAME:-michelangelo-sandbox}"
NAMESPACE="${NAMESPACE:-default}"
EXAMPLES_NAMESPACE="${EXAMPLES_NAMESPACE:-ma-examples}"
POD_WAIT_TIMEOUT="${POD_WAIT_TIMEOUT:-300}"
LOCAL_IMAGE_PATTERN="${LOCAL_IMAGE_PATTERN:--local}"
SERVERLB_CONTAINER="${SERVERLB_CONTAINER:-k3d-${CLUSTER_NAME}-serverlb}"
IMAGE_PLATFORM="${IMAGE_PLATFORM:-linux/arm64}"

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

# 1. Verify kubectl can actually reach the API server before anything else —
# after a node restart, the k3d serverlb proxy commonly holds stale backend
# connections and every kubectl call fails with EOF until it's restarted
# (gotcha #23).
log "Step 1/10: verifying kubectl connectivity"
if kubectl get --raw='/healthz' --request-timeout=5s >/dev/null 2>&1; then
  log "  kubectl connectivity OK."
else
  log "  kubectl cannot reach the API server — likely a stale serverlb proxy."
  if docker ps --format '{{.Names}}' 2>/dev/null | grep -qx "$SERVERLB_CONTAINER"; then
    log "  restarting $SERVERLB_CONTAINER"
    docker restart "$SERVERLB_CONTAINER" >/dev/null
    deadline=$(( $(date +%s) + 30 ))
    until kubectl get --raw='/healthz' --request-timeout=5s >/dev/null 2>&1; do
      if [ "$(date +%s)" -ge "$deadline" ]; then
        log "  WARNING — kubectl still unreachable after restarting serverlb; continuing anyway, later steps will likely fail."
        break
      fi
      sleep 2
    done
    kubectl get --raw='/healthz' --request-timeout=5s >/dev/null 2>&1 && log "  kubectl connectivity restored."
  else
    log "  WARNING — container '$SERVERLB_CONTAINER' not found, cannot auto-fix. Continuing anyway."
  fi
fi

# 2. Reapply mysql/minio bare-Pod manifests (and, if minio itself was gone,
# its credentials Secret + bucket-setup Job too) if a node eviction dropped
# them — `ma sandbox sync` cannot recover this on its own (gotchas #9/#22),
# and in the worst case both are entirely absent, not just unhealthy
# (gotcha #28).
log "Step 2/10: checking mysql/minio pods in namespace '$NAMESPACE'"
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
    for extra in "$MINIO_CREDS_YAML" "$BUCKET_SETUP_YAML"; do
      if [ -f "$extra" ]; then
        log "  minio was missing — also reapplying $extra (gotcha #28)"
        kubectl apply -f "$extra" >/dev/null
      fi
    done
  fi
  log "  waiting for mysql/minio pods to become ready (timeout 120s)"
  kubectl wait --for=condition=ready pod/mysql pod/minio -n "$NAMESPACE" --timeout=120s || \
    log "  WARNING — mysql/minio did not report ready in time; continuing anyway."
fi

# 3. Redeploy the Michelangelo app pods (gotcha #8). Note: if the prior Helm
# release was broken/absent, sync silently does a full uninstall+reinstall
# instead of an upgrade (gotcha #29) — step 5 below handles the fallout.
log "Step 3/10: ma sandbox sync"
(cd "$PYTHON_DIR" && poetry run ma sandbox sync)

# 4. Poll until nothing is Pending/ContainerCreating, surfacing crash-loops
# and image-pull failures (the latter handled by step 5).
log "Step 4/10: waiting for pods in namespace '$NAMESPACE' to settle (timeout ${POD_WAIT_TIMEOUT}s)"
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
    log "  timed out after ${POD_WAIT_TIMEOUT}s — continuing anyway, step 5 will try to fix image-pull failures."
    break
  fi
  sleep 5
done

# 5. Auto-fix ghcr.io images stuck in ImagePullBackOff/ErrImagePull, across
# ALL namespaces (a fresh sync-triggered reinstall hits this on apiserver/
# controllermgr/ui/worker; other flows have hit it on spark-operator and
# kuberay-historyserver too). The k3d node's containerd has a never-fully-
# diagnosed x509 trust failure against ghcr.io specifically (gotchas
# #24/#25) — the reliable workaround is to bypass node-internal pulls
# entirely: pull by exact platform digest via the Mac's host Docker daemon
# (gotcha #26), tag back to the original ref, and `k3d image import` it.
log "Step 5/10: checking for ghcr.io images stuck in ImagePullBackOff/ErrImagePull"
mapfile -t bad_pods < <(kubectl get pods --all-namespaces --no-headers 2>/dev/null \
  | awk '$4 ~ /ImagePullBackOff|ErrImagePull/ {print $1"/"$2}')
if [ "${#bad_pods[@]}" -eq 0 ]; then
  log "  none found — skipping."
else
  log "  found ${#bad_pods[@]} pod(s) stuck on image pull: ${bad_pods[*]}"
  images_to_fix=()
  for entry in "${bad_pods[@]}"; do
    ns="${entry%%/*}"
    pod="${entry#*/}"
    while IFS= read -r img; do
      if [ -n "$img" ] && [[ "$img" == ghcr.io/* ]]; then
        images_to_fix+=("$img")
      fi
    done < <(kubectl get pod "$pod" -n "$ns" -o \
      jsonpath='{range .spec.initContainers[*]}{.image}{"\n"}{end}{range .spec.containers[*]}{.image}{"\n"}{end}' \
      2>/dev/null)
  done
  if [ "${#images_to_fix[@]}" -gt 0 ]; then
    mapfile -t images_to_fix < <(printf '%s\n' "${images_to_fix[@]}" | sort -u)
  fi
  if [ "${#images_to_fix[@]}" -eq 0 ]; then
    log "  stuck pods found but none reference ghcr.io — not an auto-fixable case here, leaving as-is."
  else
    log "  resolving and pulling ${#images_to_fix[@]} ghcr.io image(s) by $IMAGE_PLATFORM digest via host Docker"
    platform_os="${IMAGE_PLATFORM%%/*}"
    platform_arch="${IMAGE_PLATFORM##*/}"
    for img in "${images_to_fix[@]}"; do
      base="${img%:*}"
      log "    inspecting $img"
      digest="$(docker buildx imagetools inspect "$img" --raw 2>/dev/null | python3 -c "
import json, sys
try:
    data = json.load(sys.stdin)
except Exception:
    sys.exit(0)
for m in data.get('manifests', []):
    p = m.get('platform', {})
    if p.get('architecture') == '$platform_arch' and p.get('os') == '$platform_os':
        print(m['digest'])
        break
")"
      if [ -z "$digest" ]; then
        log "    WARNING — could not resolve a $IMAGE_PLATFORM digest for $img, skipping."
        continue
      fi
      log "    $IMAGE_PLATFORM digest: $digest"
      docker pull "${base}@${digest}" >/dev/null
      docker tag "${base}@${digest}" "$img"
    done
    log "  importing fixed image(s) into k3d cluster '$CLUSTER_NAME'"
    k3d image import "${images_to_fix[@]}" -c "$CLUSTER_NAME"
    log "  deleting stuck pods so they are recreated against the now-local image"
    for entry in "${bad_pods[@]}"; do
      ns="${entry%%/*}"
      pod="${entry#*/}"
      kubectl delete pod "$pod" -n "$ns" --ignore-not-found >/dev/null 2>&1 || true
    done
    log "  waiting 15s for replacement pods to be scheduled"
    sleep 15
  fi
fi

# 6. Backfill compute-cluster CRD/RBAC/secrets if a sync-recovery skipped them (gotcha #4).
log "Step 6/10: checking compute-cluster CRD/RBAC/secrets"
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

# 7. Ensure the ma-examples namespace and Project CR exist (gotchas #10/#11).
log "Step 7/10: ensuring '$EXAMPLES_NAMESPACE' namespace and Project CR"
kubectl create namespace "$EXAMPLES_NAMESPACE" --dry-run=client -o yaml | kubectl apply -f - >/dev/null
if [ -f "$PROJECT_YAML" ]; then
  kubectl apply -f "$PROJECT_YAML" >/dev/null
  log "  applied $PROJECT_YAML"
else
  log "  WARNING — $PROJECT_YAML not found, skipping Project CR apply."
fi

# 8. Verify/register the Cadence domain if this sandbox uses Cadence (gotcha #5).
log "Step 8/10: checking Cadence domain registration"
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

# 9. Clean zombie RayClusters across both CRD groups + failed pods (gotcha #13).
log "Step 9/10: cleaning zombie RayCluster CRs and failed pods"
kubectl delete raycluster.michelangelo.api -n "$NAMESPACE" --all --ignore-not-found >/dev/null 2>&1 || true
kubectl delete raycluster.ray.io -n "$NAMESPACE" --all --ignore-not-found >/dev/null 2>&1 || true
kubectl delete pod -n "$NAMESPACE" --field-selector=status.phase=Failed --ignore-not-found >/dev/null 2>&1 || true

# 10. Re-import any locally-built images into k3d (survives stop/start, but
# re-import defensively in case the node volume was recreated).
log "Step 10/10: re-importing local images matching '*${LOCAL_IMAGE_PATTERN}*' into k3d"
local_images="$(docker images --format '{{.Repository}}:{{.Tag}}' 2>/dev/null | grep -- "$LOCAL_IMAGE_PATTERN" || true)"
if [ -n "$local_images" ]; then
  mapfile -t local_images_arr <<< "$local_images"
  k3d image import "${local_images_arr[@]}" -c "$CLUSTER_NAME"
  printf '%s\n' "${local_images_arr[@]}" | sed 's/^/[recover]   imported: /'
else
  log "  no local images found matching '*${LOCAL_IMAGE_PATTERN}*'."
fi

log "Done. Final pod summary across all namespaces:"
kubectl get pods --all-namespaces
