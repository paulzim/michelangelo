#!/usr/bin/env bash
#
# recover-after-restart.sh — Collapse the manual recovery checklist for
# `ma sandbox start` after a Mac restart into one idempotent script.
#
# SCOPE: this is a *restart* recovery script — it assumes the k3d cluster and
# its volumes already exist (you ran `ma sandbox stop` then `ma sandbox
# start`, or the Mac itself rebooted). It does NOT perform the full Zscaler CA
# setup a fresh `ma sandbox delete` + `create` needs (see the
# `sandbox_zscaler_ca_fixes` note) — it only repairs the k3d CA bind-mount
# source file when macOS has garbage-collected it (step 1).
#
# Steps (see CLAUDE.md "Sandbox install gotchas" for the full writeup of each):
#   1. Node containers: rebuild the k3d CA bind-mount source in $TMPDIR if a
#      reboot deleted it (gotcha #32), then start stopped containers in
#      order — server, agents, serverlb.
#   2. Fail fast if Docker Desktop's port forwarding is wedged: serverlb has
#      configured host ports that aren't live (gotcha #35). Only a Mac reboot
#      clears this, so stop with that instruction instead of cascading.
#   3. kubectl connectivity — restart serverlb on stale-EOF (gotcha #23).
#   4. Nodes Ready *and stable* — a freshly booted node can report Ready and
#      then flip to NotReady ~50s later; restart any node container that stays
#      NotReady (gotcha #32 addendum). Must run before step 5: a node NotReady
#      for >300s evicts the mysql/minio bare Pods.
#   5. mysql/minio bare Pods: delete evicted/terminal ones and reapply missing
#      manifests (gotchas #9/#22/#28).
#   6. Re-import locally-built images into k3d, before sync references them.
#   7. Delete app Deployments carrying a kubectl field-manager lock (e.g. from
#      `kubectl set image`) so Helm's server-side apply can't conflict
#      (gotcha #12). A failed upgrade would also mark the release "failed" and
#      push every later sync into a full reinstall (gotcha #29).
#   8. `ma sandbox sync` (gotcha #8), passing values-local.yaml (gitignored;
#      copy values-local.example.yaml) as `--set` flags so custom images apply
#      inside sync's own helm call on both the upgrade and reinstall paths.
#      If sync's helm step succeeds but its readiness wait fails, continue and
#      let the final health check report it.
#   9. Poll until pods leave Pending/ContainerCreating.
#  10. Auto-fix ghcr.io images stuck in ImagePullBackOff via host-Docker
#      digest pulls + `k3d image import` (gotchas #24-27/#29).
#  11. Refresh cached ghcr.io `:main` app images that are behind the registry,
#      compared by image config digest, then restart their Deployments
#      (gotcha #36). Nodes never refresh a cached `:main` on their own.
#  12. Backfill compute-cluster CRD/RBAC/secrets (gotcha #4).
#  13. Make the michelangelo-config ConfigMap's S3 credentials, which Ray and
#      Spark task pods use, match what MinIO actually boots with (gotcha #38).
#      Only `ma sandbox create` writes that ConfigMap.
#  14. Ensure the ma-examples namespace + Project CR (gotchas #10/#11).
#  15. Verify/register the Cadence domain (gotcha #5).
#  16. Clean zombie RayClusters (both CRD groups) and failed pods (gotcha #13).
#   Finally: report pods still unhealthy (minus IGNORE_PODS) and app images
#   still stale, and exit 2 if any.
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
#   VALUES_LOCAL         — Helm overrides file (default: <script dir>/values-local.yaml)
#   ZSCALER_CA_PEM       — Zscaler root CA appended to a rebuilt CA bundle
#                          (default: <repo>/../michelangelo-examples/.docker/zscaler-root-ca.pem)
#   IGNORE_PODS          — regex of pod names excluded from the step 9 warning and the
#                          final health check (default: ^model-sync- ; empty string = ignore nothing)
#   REFRESH_STALE_IMAGES — 0 = only report stale ghcr.io :main app images in step 11 (default: 1).
#                          A fresh :main can be ahead of this checkout's chart if the fork
#                          hasn't synced upstream lately.
#   NODE_READY_TIMEOUT   — seconds to wait for nodes Ready+stable (default: 300)
#   NODE_STABLE_SECS     — seconds a Ready transition must hold (default: 90)
#   NODE_RESTART_AFTER   — seconds NotReady before restarting a node container (default: 60)
#
# Exit codes:
#   0 = completed, all pods healthy (excluding IGNORE_PODS)
#   1 = blocked: missing tool/cluster, wedged Docker ports, nodes not Ready, or sync's helm step failed
#   2 = completed, but some pods are still unhealthy or app images still stale (listed at the end)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
PYTHON_DIR="$REPO_ROOT/python"
PROJECT_YAML="$PYTHON_DIR/examples/config/project.yaml"
MYSQL_YAML="$PYTHON_DIR/michelangelo/cli/sandbox/resources/mysql.yaml"
MINIO_YAML="$PYTHON_DIR/michelangelo/cli/sandbox/resources/minio.yaml"
MINIO_CREDS_YAML="$PYTHON_DIR/michelangelo/cli/sandbox/resources/minio-credentials.yaml"
BUCKET_SETUP_YAML="$PYTHON_DIR/michelangelo/cli/sandbox/resources/sandbox-bucket-setup.yaml"
MA_CONFIG_YAML="$PYTHON_DIR/michelangelo/cli/sandbox/resources/michelangelo-config.yaml"

CLUSTER_NAME="${CLUSTER_NAME:-michelangelo-sandbox}"
NAMESPACE="${NAMESPACE:-default}"
EXAMPLES_NAMESPACE="${EXAMPLES_NAMESPACE:-ma-examples}"
POD_WAIT_TIMEOUT="${POD_WAIT_TIMEOUT:-300}"
LOCAL_IMAGE_PATTERN="${LOCAL_IMAGE_PATTERN:--local}"
SERVERLB_CONTAINER="${SERVERLB_CONTAINER:-k3d-${CLUSTER_NAME}-serverlb}"
IMAGE_PLATFORM="${IMAGE_PLATFORM:-linux/arm64}"
VALUES_LOCAL="${VALUES_LOCAL:-$SCRIPT_DIR/values-local.yaml}"
ZSCALER_CA_PEM="${ZSCALER_CA_PEM:-$REPO_ROOT/../michelangelo-examples/.docker/zscaler-root-ca.pem}"
IGNORE_PODS="${IGNORE_PODS-^model-sync-}"
NODE_READY_TIMEOUT="${NODE_READY_TIMEOUT:-300}"
NODE_STABLE_SECS="${NODE_STABLE_SECS:-90}"
NODE_RESTART_AFTER="${NODE_RESTART_AFTER:-60}"
REFRESH_STALE_IMAGES="${REFRESH_STALE_IMAGES:-1}"

CA_MOUNT_DEST="/etc/ssl/certs/ca-certificates.crt"
APP_DEPLOYMENTS=(michelangelo-apiserver michelangelo-controllermgr michelangelo-worker michelangelo-ui)
TOTAL_STEPS=16
STALE_IMAGES=()
STEP=0

log() { echo "[recover] $*"; }
step() { STEP=$((STEP + 1)); log "Step ${STEP}/${TOTAL_STEPS}: $*"; }
die() { log "ERROR — $*"; exit 1; }

for tool in kubectl poetry helm docker k3d python3; do
  command -v "$tool" >/dev/null 2>&1 || die "'$tool' not found on PATH"
done
docker info >/dev/null 2>&1 || die "Docker daemon not reachable — is Docker Desktop running (and fully started)?"
if ! k3d cluster get "$CLUSTER_NAME" >/dev/null 2>&1; then
  die "k3d cluster '$CLUSTER_NAME' not found. This script only handles restart recovery; run 'ma sandbox create' for a fresh cluster."
fi

SERVER_CONTAINERS=()
AGENT_CONTAINERS=()
while IFS= read -r name; do
  case "$name" in
    "k3d-${CLUSTER_NAME}-server-"*) SERVER_CONTAINERS+=("$name") ;;
    "k3d-${CLUSTER_NAME}-agent-"*) AGENT_CONTAINERS+=("$name") ;;
  esac
done < <(docker ps -a --format '{{.Names}}' 2>/dev/null | sort)
[ "${#SERVER_CONTAINERS[@]}" -gt 0 ] || die "no container named k3d-${CLUSTER_NAME}-server-* found"
NODE_CONTAINERS=("${SERVER_CONTAINERS[@]}")
if [ "${#AGENT_CONTAINERS[@]}" -gt 0 ]; then
  NODE_CONTAINERS+=("${AGENT_CONTAINERS[@]}")
fi

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

port_wedge_abort() {
  log "ERROR — Docker Desktop is not publishing host ports for $1 (gotcha #35)."
  if [ -n "${2:-}" ]; then
    log "  detail: $2"
  fi
  log "  If 'lsof -nP -iTCP:<port> -sTCP:LISTEN' shows nothing holding the port, Docker Desktop's"
  log "  port-forwarding layer is stuck. Restarting containers or Docker Desktop does NOT clear it:"
  log "  reboot the Mac, then rerun this script (step 1 repairs the CA mount a reboot breaks)."
  exit 1
}

check_serverlb_ports() {
  local configured live missing
  configured="$(docker inspect -f '{{json .HostConfig.PortBindings}}' "$SERVERLB_CONTAINER" 2>/dev/null || echo '{}')"
  live="$(docker inspect -f '{{json .NetworkSettings.Ports}}' "$SERVERLB_CONTAINER" 2>/dev/null || echo '{}')"
  missing="$(CONFIGURED="$configured" LIVE="$live" python3 -c '
import json, os
def bound(raw):
    try:
        data = json.loads(raw) or {}
    except Exception:
        return set()
    return {port for port, bindings in data.items() if bindings}
print(" ".join(sorted(bound(os.environ["CONFIGURED"]) - bound(os.environ["LIVE"]))))
')"
  if [ -n "$missing" ]; then
    port_wedge_abort "$SERVERLB_CONTAINER" "configured but not live: $missing"
  fi
}

container_running() {
  [ "$(docker inspect -f '{{.State.Running}}' "$1" 2>/dev/null || echo false)" = "true" ]
}

STARTED_ANY=0
start_container() {
  local err
  if container_running "$1"; then
    return 0
  fi
  log "  starting $1"
  if ! err="$(docker start "$1" 2>&1 >/dev/null)"; then
    case "$err" in
      *"ports are not available"*) port_wedge_abort "$1" "$err" ;;
    esac
    die "failed to start $1: $err"
  fi
  STARTED_ANY=1
}

ca_mount_source() {
  docker inspect "$1" \
    --format "{{range .Mounts}}{{if eq .Destination \"$CA_MOUNT_DEST\"}}{{.Source}}{{end}}{{end}}" \
    2>/dev/null || true
}

build_ca_bundle() {
  command -v security >/dev/null 2>&1 || return 1
  security export -t certs -f pemseq /Library/Keychains/System.keychain > "$1"
  if [ -f "$ZSCALER_CA_PEM" ]; then
    cat "$ZSCALER_CA_PEM" >> "$1"
  else
    log "  WARNING — $ZSCALER_CA_PEM not found; bundle has System keychain certs only (gotcha #24: may lack the Zscaler root)."
  fi
}

repair_ca_mount() {
  local src tmp
  src="$(ca_mount_source "$1")"
  if [ -z "$src" ] || [ -f "$src" ]; then
    return 0
  fi
  log "  $1: CA bind-mount source is missing or not a file (gotcha #32): $src"
  if [ -d "$src" ]; then
    if [ -n "$(ls -A "$src" 2>/dev/null)" ]; then
      die "$src is a non-empty directory — not touching it; inspect it manually."
    fi
    rmdir "$src"
  fi
  tmp="$(mktemp)"
  if ! build_ca_bundle "$tmp"; then
    rm -f "$tmp"
    die "cannot rebuild the CA bundle (macOS 'security' tool unavailable) — see CLAUDE.md gotcha #32."
  fi
  cp "$tmp" "$src"
  chmod 644 "$src"
  rm -f "$tmp"
  log "  rebuilt $src ($(grep -c 'BEGIN CERTIFICATE' "$src" || true) certs)"
}

kubectl_reachable() {
  kubectl get --raw='/healthz' --request-timeout=5s >/dev/null 2>&1
}

restart_serverlb() {
  local err deadline
  if ! docker inspect "$SERVERLB_CONTAINER" >/dev/null 2>&1; then
    log "  WARNING — container '$SERVERLB_CONTAINER' not found, cannot restart it."
    return 0
  fi
  log "  restarting $SERVERLB_CONTAINER (gotcha #23)"
  if ! err="$(docker restart "$SERVERLB_CONTAINER" 2>&1 >/dev/null)"; then
    case "$err" in
      *"ports are not available"*) port_wedge_abort "$SERVERLB_CONTAINER" "$err" ;;
    esac
    log "  WARNING — docker restart failed: $err"
    return 0
  fi
  check_serverlb_ports
  deadline=$(( $(date +%s) + 30 ))
  until kubectl_reachable; do
    if [ "$(date +%s)" -ge "$deadline" ]; then
      log "  WARNING — kubectl still unreachable 30s after restarting serverlb."
      return 0
    fi
    sleep 2
  done
  log "  kubectl connectivity restored."
}

# Prints NotReady node names, or __apiserver_unreachable__ if kubectl fails.
nodes_not_ready() {
  local out
  if ! out="$(kubectl get nodes --no-headers 2>/dev/null)" || [ -z "$out" ]; then
    echo "__apiserver_unreachable__"
    return 0
  fi
  echo "$out" | awk '$2 !~ /^Ready/ {print $1}'
}

# Seconds since the most recent node Ready-condition transition.
newest_ready_age() {
  local json
  json="$(kubectl get nodes -o json 2>/dev/null || true)"
  printf '%s' "$json" | python3 -c '
import json, sys
from datetime import datetime, timezone
try:
    items = json.load(sys.stdin)["items"]
except Exception:
    print(0)
    sys.exit(0)
now = datetime.now(timezone.utc)
ages = []
for node in items:
    for cond in node.get("status", {}).get("conditions", []):
        if cond.get("type") == "Ready" and cond.get("lastTransitionTime"):
            try:
                t = datetime.strptime(cond["lastTransitionTime"], "%Y-%m-%dT%H:%M:%SZ")
            except ValueError:
                continue
            ages.append(int((now - t.replace(tzinfo=timezone.utc)).total_seconds()))
print(min(ages) if ages else 0)
'
}

# Succeeds if the bare Pod is Running/Pending and not being deleted.
pod_usable() {
  local state
  state="$(kubectl get pod "$1" -n "$NAMESPACE" \
    -o jsonpath='{.status.phase}/{.metadata.deletionTimestamp}' 2>/dev/null || true)"
  [ "$state" = "Running/" ] || [ "$state" = "Pending/" ]
}

# Prints "<revision> <status>" for the Helm release, or "0 missing".
release_state() {
  local json
  json="$(helm status michelangelo -n "$NAMESPACE" -o json 2>/dev/null || true)"
  printf '%s' "$json" | python3 -c '
import json, sys
try:
    r = json.load(sys.stdin)
    print(r.get("version", 0), r.get("info", {}).get("status") or "unknown")
except Exception:
    print(0, "missing")
'
}

# Prints IMG's $IMAGE_PLATFORM manifest digest from the registry, or nothing.
remote_platform_digest() {
  docker buildx imagetools inspect "$1" --raw 2>/dev/null | PLATFORM="$IMAGE_PLATFORM" python3 -c '
import json, os, sys
want = os.environ["PLATFORM"].split("/")
try:
    data = json.load(sys.stdin)
except Exception:
    sys.exit(0)
for m in data.get("manifests", []):
    p = m.get("platform", {})
    if p.get("os") == want[0] and p.get("architecture") == want[-1]:
        print(m["digest"])
        break
' || true
}

# Prints the image config digest of the manifest BASE@DIGEST, or nothing.
remote_config_digest() {
  docker buildx imagetools inspect "$1@$2" --raw 2>/dev/null | python3 -c '
import json, sys
try:
    print(json.load(sys.stdin)["config"]["digest"])
except Exception:
    pass
' || true
}

# Prints the config digest (image ID) of IMG in node container NODE's
# containerd store, or nothing if the node doesn't have it.
node_config_digest() {
  docker exec "$1" crictl inspecti -o json "$2" 2>/dev/null | python3 -c '
import json, sys
try:
    print(json.load(sys.stdin)["status"]["id"])
except Exception:
    pass
' || true
}

# Prints MinIO's root user and password, one per line, as the live minio Pod
# boots with them (literal env value or secretKeyRef). Falls back to the
# minio-credentials Secret; prints nothing if neither can be read.
minio_root_creds() {
  kubectl get pod minio -n "$NAMESPACE" -o json 2>/dev/null | NS="$NAMESPACE" python3 -c '
import base64, json, os, subprocess, sys

def secret_value(name, key):
    out = subprocess.run(
        ["kubectl", "get", "secret", name, "-n", os.environ["NS"], "-o", "jsonpath={.data." + key + "}"],
        capture_output=True, text=True,
    )
    if out.returncode != 0 or not out.stdout.strip():
        return None
    return base64.b64decode(out.stdout.strip()).decode()

def from_pod():
    try:
        containers = json.load(sys.stdin)["spec"]["containers"]
    except Exception:
        return None
    if not containers:
        return None
    c = next((c for c in containers if c.get("name") == "minio"), containers[0])
    env = {e.get("name"): e for e in c.get("env", [])}
    if "MINIO_ROOT_USER" not in env and "MINIO_ROOT_PASSWORD" not in env:
        # MinIO uses its built-in default when neither is set (minio.yaml before #2219).
        return None if c.get("envFrom") else ["minioadmin", "minioadmin"]
    values = []
    for var in ("MINIO_ROOT_USER", "MINIO_ROOT_PASSWORD"):
        e = env.get(var, {})
        if "value" in e:
            values.append(e["value"])
            continue
        ref = e.get("valueFrom", {}).get("secretKeyRef")
        value = secret_value(ref.get("name", ""), ref.get("key", "")) if ref else None
        if value is None:
            return None
        values.append(value)
    return values

creds = from_pod()
if creds is None:
    creds = [secret_value("minio-credentials", k) for k in ("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY")]
if all(creds):
    print(creds[0])
    print(creds[1])
' || true
}

# ---------------------------------------------------------------------------
# 1. Node containers
# ---------------------------------------------------------------------------
step "checking k3d node containers (CA bind-mount, start order)"
for c in "${NODE_CONTAINERS[@]}"; do
  repair_ca_mount "$c"
done
for c in "${SERVER_CONTAINERS[@]}"; do
  start_container "$c"
done
if [ "$STARTED_ANY" -eq 1 ] && [ "${#AGENT_CONTAINERS[@]}" -gt 0 ]; then
  log "  giving the server 15s head start before agents"
  sleep 15
fi
if [ "${#AGENT_CONTAINERS[@]}" -gt 0 ]; then
  for c in "${AGENT_CONTAINERS[@]}"; do
    start_container "$c"
  done
fi
if docker inspect "$SERVERLB_CONTAINER" >/dev/null 2>&1; then
  start_container "$SERVERLB_CONTAINER"
fi
if [ "$STARTED_ANY" -eq 0 ]; then
  log "  all node containers already running."
fi

# ---------------------------------------------------------------------------
# 2. Docker Desktop port publishing
# ---------------------------------------------------------------------------
step "verifying Docker publishes serverlb's host ports"
if docker inspect "$SERVERLB_CONTAINER" >/dev/null 2>&1; then
  check_serverlb_ports
  log "  all configured host ports are live."
else
  log "  WARNING — $SERVERLB_CONTAINER not found; skipping."
fi

# ---------------------------------------------------------------------------
# 3. kubectl connectivity
# ---------------------------------------------------------------------------
step "verifying kubectl connectivity"
if kubectl_reachable; then
  log "  kubectl connectivity OK."
else
  log "  kubectl cannot reach the API server — likely a stale serverlb proxy."
  restart_serverlb
fi

# ---------------------------------------------------------------------------
# 4. Nodes Ready and stable
# ---------------------------------------------------------------------------
step "waiting for nodes to be Ready and stable (timeout ${NODE_READY_TIMEOUT}s)"
start_ts=$(date +%s)
deadline=$((start_ts + NODE_READY_TIMEOUT))
restarted_nodes=0
while true; do
  not_ready="$(nodes_not_ready)"
  now=$(date +%s)
  if [ -z "$not_ready" ]; then
    age="$(newest_ready_age)"
    age="${age:-0}"
    if [ "$age" -ge "$NODE_STABLE_SECS" ]; then
      log "  all nodes Ready (newest Ready transition ${age}s ago)."
      break
    fi
    log "  all nodes Ready, but newest Ready transition was ${age}s ago — waiting until ${NODE_STABLE_SECS}s to rule out a post-boot flip back to NotReady"
  else
    log "  not Ready: $(echo "$not_ready" | tr '\n' ' ')"
    if [ "$restarted_nodes" -eq 0 ] && [ $((now - start_ts)) -ge "$NODE_RESTART_AFTER" ]; then
      restarted_nodes=1
      if [ "$not_ready" = "__apiserver_unreachable__" ]; then
        restart_serverlb
      else
        log "  still NotReady after ${NODE_RESTART_AFTER}s — restarting those node containers once (gotcha #32 addendum)"
        for n in $not_ready; do
          if docker inspect "$n" >/dev/null 2>&1; then
            docker restart "$n" >/dev/null
          else
            log "  WARNING — no container named $n"
          fi
        done
        sleep 10
        restart_serverlb
      fi
    fi
  fi
  if [ "$now" -ge "$deadline" ]; then
    log "ERROR — nodes not Ready+stable after ${NODE_READY_TIMEOUT}s; 'ma sandbox sync' would fail its own node wait."
    log "  Inspect: kubectl describe node <name> | tail -30"
    log "           docker inspect -f '{{json .State}}' <node container>"
    exit 1
  fi
  sleep 5
done

# ---------------------------------------------------------------------------
# 5. mysql/minio bare Pods
# ---------------------------------------------------------------------------
step "checking mysql/minio pods in namespace '$NAMESPACE'"
mysql_missing=0
minio_missing=0
for p in mysql minio; do
  if pod_usable "$p"; then
    continue
  fi
  if kubectl get pod "$p" -n "$NAMESPACE" >/dev/null 2>&1; then
    log "  $p pod exists but is not Running/Pending (likely evicted while a node was NotReady) — deleting it"
    kubectl delete pod "$p" -n "$NAMESPACE" --ignore-not-found --wait=true --timeout=60s >/dev/null 2>&1 || true
  fi
  if [ "$p" = "mysql" ]; then
    mysql_missing=1
  else
    minio_missing=1
  fi
done
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

# ---------------------------------------------------------------------------
# 6. Local images
# ---------------------------------------------------------------------------
step "re-importing local images matching '*${LOCAL_IMAGE_PATTERN}*' into k3d"
local_images="$(docker images --format '{{.Repository}}:{{.Tag}}' 2>/dev/null | grep -- "$LOCAL_IMAGE_PATTERN" || true)"
if [ -n "$local_images" ]; then
  local_images_arr=()
  while IFS= read -r line; do
    [ -n "$line" ] && local_images_arr+=("$line")
  done <<< "$local_images"
  k3d image import "${local_images_arr[@]}" -c "$CLUSTER_NAME"
  printf '%s\n' "${local_images_arr[@]}" | sed 's/^/[recover]   imported: /'
else
  log "  no local images found matching '*${LOCAL_IMAGE_PATTERN}*'."
fi

# ---------------------------------------------------------------------------
# 7. Field-manager locks + release state
# ---------------------------------------------------------------------------
step "checking app Deployments for kubectl field-manager locks (gotcha #12)"
locked=()
while IFS=$'\t' read -r name managers; do
  [ -n "$name" ] || continue
  case ",$managers," in
    *,kubectl-set,*|*,kubectl-edit,*|*,kubectl-patch,*) locked+=("$name") ;;
  esac
done < <(kubectl get deployment -n "$NAMESPACE" -l app.kubernetes.io/instance=michelangelo \
  -o jsonpath='{range .items[*]}{.metadata.name}{"\t"}{range .metadata.managedFields[*]}{.manager}{","}{end}{"\n"}{end}' \
  2>/dev/null || true)
if [ "${#locked[@]}" -gt 0 ]; then
  for d in "${locked[@]}"; do
    img="$(kubectl get deployment "$d" -n "$NAMESPACE" -o jsonpath='{.spec.template.spec.containers[0].image}' 2>/dev/null || true)"
    log "  $d has fields owned by a kubectl edit (image: ${img:-?}) — deleting so Helm can recreate it."
    kubectl delete deployment "$d" -n "$NAMESPACE" --wait=true >/dev/null
  done
  log "  pin custom images in $VALUES_LOCAL instead of 'kubectl set image' (see values-local.example.yaml)."
else
  log "  no locks found."
fi
release_before="$(release_state)"
case "${release_before#* }" in
  deployed) log "  Helm release revision ${release_before%% *} is 'deployed' — sync will upgrade in place." ;;
  *) log "  Helm release is '${release_before#* }' — sync will uninstall + reinstall (gotcha #29)." ;;
esac

# ---------------------------------------------------------------------------
# 8. ma sandbox sync, with local overrides as --set flags
# ---------------------------------------------------------------------------
# Overrides go through sync's own --set passthrough rather than a separate
# helm upgrade afterwards: --set outranks sync's `-f values-k3d.yaml` (which
# pins every app image tag to "main"), and it applies on both the upgrade and
# reinstall paths, so a recreated Deployment comes up on the right image
# before sync's readiness wait runs.
step "ma sandbox sync"
SYNC_SET_ARGS=()
if [ -f "$VALUES_LOCAL" ]; then
  overrides="$(cd "$PYTHON_DIR" && VALUES_LOCAL="$VALUES_LOCAL" poetry run python3 -c '
import os, sys, yaml
path = os.environ["VALUES_LOCAL"]
try:
    with open(path) as f:
        data = yaml.safe_load(f) or {}
except yaml.YAMLError as e:
    sys.exit(f"invalid YAML in {path}: {e}")
if not isinstance(data, dict):
    sys.exit(f"{path} must be a YAML mapping")
def esc(s, chars):
    s = s.replace("\\", "\\\\")
    for c in chars:
        s = s.replace(c, "\\" + c)
    return s
def walk(prefix, node):
    for k, v in node.items():
        key = prefix + esc(str(k), ".,")
        if isinstance(v, dict):
            walk(key + ".", v)
        elif isinstance(v, list):
            sys.exit(f"{path}: lists are not supported (key {key})")
        else:
            s = "null" if v is None else ("true" if v is True else "false" if v is False else str(v))
            print(key + "=" + esc(s, ","))
walk("", data)
')" || die "could not read $VALUES_LOCAL (see error above)."
  while IFS= read -r kv; do
    if [ -n "$kv" ]; then SYNC_SET_ARGS+=(--set "$kv"); fi
  done <<< "$overrides"
  if [ "${#SYNC_SET_ARGS[@]}" -gt 0 ]; then
    log "  local overrides from $VALUES_LOCAL:"
    echo "$overrides" | sed 's/^/[recover]     --set /'
  fi
else
  log "  no $VALUES_LOCAL — chart default images (template: $SCRIPT_DIR/values-local.example.yaml)."
fi
if ! (cd "$PYTHON_DIR" && poetry run ma sandbox sync ${SYNC_SET_ARGS[@]+"${SYNC_SET_ARGS[@]}"}); then
  # A new 'deployed' revision means helm itself succeeded and only sync's
  # readiness wait failed — keep going so the final health check can report.
  release_after="$(release_state)"
  if [ "$release_after" != "$release_before" ] && [ "${release_after#* }" = deployed ]; then
    log "  WARNING — sync's helm step succeeded (revision ${release_after%% *}) but its readiness wait failed;"
    log "  continuing. The final health check lists what's unhealthy (stale cached image? gotcha #36)."
  elif [ "$release_after" = "$release_before" ]; then
    die "'ma sandbox sync' failed without producing a new Helm release revision — see its output above."
  else
    die "'ma sandbox sync' failed; Helm release is now '${release_after#* }' — see its output above (field-manager conflict: gotcha #12)."
  fi
fi
log "  app images now:"
kubectl get deployment -n "$NAMESPACE" "${APP_DEPLOYMENTS[@]}" \
  -o jsonpath='{range .items[*]}{.metadata.name}{": "}{.spec.template.spec.containers[0].image}{"\n"}{end}' \
  2>/dev/null | sed 's/^/[recover]     /' || true

# ---------------------------------------------------------------------------
# 9. Pods settle
# ---------------------------------------------------------------------------
step "waiting for pods in namespace '$NAMESPACE' to settle (timeout ${POD_WAIT_TIMEOUT}s)"
deadline=$(( $(date +%s) + POD_WAIT_TIMEOUT ))
while true; do
  pod_lines="$(kubectl get pods -n "$NAMESPACE" --no-headers 2>/dev/null || true)"
  if [ -z "$pod_lines" ]; then
    log "  no pods found yet in '$NAMESPACE'..."
  else
    unsettled="$(echo "$pod_lines" | awk '$3 ~ /Pending|ContainerCreating|PodInitializing|Init:/ {print $1}')"
    crashing="$(echo "$pod_lines" | awk -v ignore="$IGNORE_PODS" '
      $3 ~ /CrashLoopBackOff|Error|ImagePullBackOff|ErrImagePull/ &&
        (ignore == "" || $1 !~ ignore) {print $1 ": " $3}')"
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
    log "  timed out after ${POD_WAIT_TIMEOUT}s — continuing anyway, the next step tries to fix image-pull failures."
    break
  fi
  sleep 5
done

# ---------------------------------------------------------------------------
# 10. ghcr.io image-pull failures
# ---------------------------------------------------------------------------
step "checking for ghcr.io images stuck in ImagePullBackOff/ErrImagePull"
bad_pods=()
while IFS= read -r line; do
  [ -n "$line" ] && bad_pods+=("$line")
done < <(kubectl get pods --all-namespaces --no-headers 2>/dev/null \
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
    sorted_images=()
    while IFS= read -r line; do
      [ -n "$line" ] && sorted_images+=("$line")
    done < <(printf '%s\n' "${images_to_fix[@]}" | sort -u)
    images_to_fix=("${sorted_images[@]}")
  fi
  if [ "${#images_to_fix[@]}" -eq 0 ]; then
    log "  stuck pods found but none reference ghcr.io — not an auto-fixable case here, leaving as-is."
  else
    log "  resolving and pulling ${#images_to_fix[@]} ghcr.io image(s) by $IMAGE_PLATFORM digest via host Docker"
    for img in "${images_to_fix[@]}"; do
      base="${img%:*}"
      log "    inspecting $img"
      digest="$(remote_platform_digest "$img")"
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

# ---------------------------------------------------------------------------
# 11. Stale cached ghcr.io :main app images
# ---------------------------------------------------------------------------
# Compared by image config digest, which survives the host pull + `k3d image
# import` path unchanged; a manifest digest may not.
step "checking cached ghcr.io :main app images against the registry (gotcha #36)"
refresh_deploys=()
refresh_images=()
checked=0
for d in "${APP_DEPLOYMENTS[@]}"; do
  img="$(kubectl get deployment "$d" -n "$NAMESPACE" -o jsonpath='{.spec.template.spec.containers[0].image}' 2>/dev/null || true)"
  case "$img" in
    ghcr.io/michelangelo-ai/*:main) ;;
    "") continue ;;
    *) log "  $d: $img is not a ghcr.io :main image — skipping."; continue ;;
  esac
  checked=$((checked + 1))
  base="${img%:*}"
  digest="$(remote_platform_digest "$img")"
  remote_cfg=""
  if [ -n "$digest" ]; then
    remote_cfg="$(remote_config_digest "$base" "$digest")"
  fi
  if [ -z "$remote_cfg" ]; then
    log "  WARNING — $d: could not read $img's $IMAGE_PLATFORM config from the registry; not checked."
    continue
  fi
  stale_on=""
  read_any=0
  for n in "${NODE_CONTAINERS[@]}"; do
    node_cfg="$(node_config_digest "$n" "$img")"
    if [ -n "$node_cfg" ]; then
      read_any=1
      if [ "$node_cfg" != "$remote_cfg" ]; then
        stale_on="$stale_on ${n} (${node_cfg:7:12})"
      fi
    fi
  done
  if [ "$read_any" -eq 0 ]; then
    log "  WARNING — $d: couldn't read $img from any node's image store (crictl inspecti); not checked."
    continue
  fi
  if [ -z "$stale_on" ]; then
    log "  $d: $img is current (${remote_cfg:7:12})."
    continue
  fi
  log "  $d: $img is stale on$stale_on; the registry has ${remote_cfg:7:12}."
  if [ "$REFRESH_STALE_IMAGES" = "0" ]; then
    STALE_IMAGES+=("$d: $img")
  elif docker pull "${base}@${digest}" >/dev/null && docker tag "${base}@${digest}" "$img"; then
    refresh_deploys+=("$d")
    refresh_images+=("$img")
  else
    log "  WARNING — could not pull ${base}@${digest}."
    STALE_IMAGES+=("$d: $img")
  fi
done
if [ "$checked" -eq 0 ]; then
  log "  no app Deployments use ghcr.io :main images — skipping."
fi
if [ "${#refresh_images[@]}" -gt 0 ]; then
  log "  importing ${#refresh_images[@]} refreshed image(s) into k3d cluster '$CLUSTER_NAME'"
  if k3d image import "${refresh_images[@]}" -c "$CLUSTER_NAME"; then
    for d in "${refresh_deploys[@]}"; do
      kubectl rollout restart "deployment/$d" -n "$NAMESPACE" >/dev/null
    done
    for d in "${refresh_deploys[@]}"; do
      if kubectl rollout status "deployment/$d" -n "$NAMESPACE" --timeout=180s >/dev/null 2>&1; then
        log "  $d restarted on the refreshed image."
      else
        log "  WARNING — $d's rollout didn't finish within 180s; the final check lists its pods."
      fi
    done
  else
    log "  WARNING — k3d image import failed; the stale images are still in use."
    i=0
    while [ "$i" -lt "${#refresh_deploys[@]}" ]; do
      STALE_IMAGES+=("${refresh_deploys[$i]}: ${refresh_images[$i]}")
      i=$((i + 1))
    done
  fi
fi

# ---------------------------------------------------------------------------
# 12. Compute-cluster CRD/RBAC/secrets
# ---------------------------------------------------------------------------
step "checking compute-cluster CRD/RBAC/secrets"
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

# ---------------------------------------------------------------------------
# 13. Task-pod S3 credentials
# ---------------------------------------------------------------------------
# Ray and Spark task pods read S3 settings from michelangelo-config via
# envFrom. Only `ma sandbox create` writes it, so a sandbox created before the
# minioadmin -> michelangeloadmin rename keeps the old key. Only the two
# credential keys are synced; REGISTRY_NAMESPACE and the rest are left alone.
step "checking michelangelo-config S3 credentials against MinIO's (gotcha #38)"
minio_creds="$(minio_root_creds)"
minio_user="$(printf '%s\n' "$minio_creds" | sed -n 1p)"
minio_password="$(printf '%s\n' "$minio_creds" | sed -n 2p)"
if [ -z "$minio_user" ] || [ -z "$minio_password" ]; then
  log "  WARNING — could not read MinIO's root credentials (minio pod or minio-credentials Secret); skipping."
else
  if ! kubectl get configmap michelangelo-config -n "$NAMESPACE" >/dev/null 2>&1; then
    if [ -f "$MA_CONFIG_YAML" ]; then
      log "  michelangelo-config is missing — applying $MA_CONFIG_YAML"
      kubectl apply -n "$NAMESPACE" -f "$MA_CONFIG_YAML" >/dev/null
    else
      log "  WARNING — michelangelo-config is missing and $MA_CONFIG_YAML not found; task pods can't start."
    fi
  fi
  if kubectl get configmap michelangelo-config -n "$NAMESPACE" >/dev/null 2>&1; then
    cm_user="$(kubectl get configmap michelangelo-config -n "$NAMESPACE" -o jsonpath='{.data.AWS_ACCESS_KEY_ID}' 2>/dev/null || true)"
    cm_password="$(kubectl get configmap michelangelo-config -n "$NAMESPACE" -o jsonpath='{.data.AWS_SECRET_ACCESS_KEY}' 2>/dev/null || true)"
    if [ "$cm_user" = "$minio_user" ] && [ "$cm_password" = "$minio_password" ]; then
      log "  credentials match MinIO's (access key '$minio_user')."
    else
      if [ "$cm_user" = "$minio_user" ]; then
        log "  access key '$minio_user' matches, but the secret key differs — patching."
      else
        log "  ConfigMap has access key '${cm_user:-<unset>}', MinIO uses '$minio_user' — patching."
      fi
      patch_file="$(mktemp)"
      MINIO_USER="$minio_user" MINIO_PASSWORD="$minio_password" python3 -c '
import json, os
print(json.dumps({"data": {
    "AWS_ACCESS_KEY_ID": os.environ["MINIO_USER"],
    "AWS_SECRET_ACCESS_KEY": os.environ["MINIO_PASSWORD"],
}}))
' > "$patch_file"
      kubectl patch configmap michelangelo-config -n "$NAMESPACE" --type merge --patch-file "$patch_file" >/dev/null
      rm -f "$patch_file"
      log "  patched. No restarts needed: every task creates new pods."
    fi
  fi
fi

# ---------------------------------------------------------------------------
# 14. ma-examples namespace + Project CR
# ---------------------------------------------------------------------------
step "ensuring '$EXAMPLES_NAMESPACE' namespace and Project CR"
kubectl create namespace "$EXAMPLES_NAMESPACE" --dry-run=client -o yaml | kubectl apply -f - >/dev/null
if [ -f "$PROJECT_YAML" ]; then
  kubectl apply -f "$PROJECT_YAML" >/dev/null
  log "  applied $PROJECT_YAML"
else
  log "  WARNING — $PROJECT_YAML not found, skipping Project CR apply."
fi

# ---------------------------------------------------------------------------
# 15. Cadence domain
# ---------------------------------------------------------------------------
step "checking Cadence domain registration"
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

# ---------------------------------------------------------------------------
# 16. Zombie RayClusters + failed pods
# ---------------------------------------------------------------------------
step "cleaning zombie RayCluster CRs and failed pods"
kubectl delete raycluster.michelangelo.api -n "$NAMESPACE" --all --ignore-not-found >/dev/null 2>&1 || true
kubectl delete raycluster.ray.io -n "$NAMESPACE" --all --ignore-not-found >/dev/null 2>&1 || true
kubectl delete pod -n "$NAMESPACE" --field-selector=status.phase=Failed --ignore-not-found >/dev/null 2>&1 || true

# ---------------------------------------------------------------------------
# Final health check
# ---------------------------------------------------------------------------
unhealthy_pods() {
  kubectl get pods --all-namespaces --no-headers 2>/dev/null | awk -v ignore="$IGNORE_PODS" '
    $4 ~ /CrashLoopBackOff|Error|ImagePullBackOff|ErrImagePull|CreateContainerConfigError|InvalidImageName|RunContainerError|Pending/ &&
      (ignore == "" || $2 !~ ignore) {print $1"/"$2": "$4}' || true
}

log "Final pod summary across all namespaces:"
kubectl get pods --all-namespaces || true
unhealthy="$(unhealthy_pods)"
if [ -n "$unhealthy" ]; then
  log "  re-checking unhealthy pods in 20s (rollouts may still be finishing)..."
  sleep 20
  unhealthy="$(unhealthy_pods)"
fi
problems=0
if [ -n "$unhealthy" ]; then
  log "FINISHED WITH PROBLEMS — unhealthy pods (ignoring '${IGNORE_PODS}'):"
  echo "$unhealthy" | sed 's/^/[recover]   /'
  log "  Next: kubectl logs -n <ns> <pod> --previous ; kubectl describe pod -n <ns> <pod> | tail -30"
  problems=1
fi
if [ "${#STALE_IMAGES[@]}" -gt 0 ]; then
  log "FINISHED WITH PROBLEMS — app images still behind the registry (step 11, gotcha #36):"
  printf '%s\n' "${STALE_IMAGES[@]}" | sed 's/^/[recover]   /'
  if [ "$REFRESH_STALE_IMAGES" = "0" ]; then
    log "  Rerun without REFRESH_STALE_IMAGES=0 to refresh them."
  fi
  problems=1
fi
if [ "$problems" -eq 1 ]; then
  exit 2
fi
log "Done — all pods healthy (ignoring '${IGNORE_PODS}')."
