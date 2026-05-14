#!/usr/bin/env bash
#
# build-kuberay-images.sh — Build and import the KubeRay collector and
# History Server images into a local k3d cluster, ready for the sandbox
# to pull as `kuberay-collector:v0.1.0` and `kuberay-historyserver:v0.1.0`
# (the tags referenced by python/michelangelo/cli/sandbox/resources/).
#
# Mirrors what `.github/workflows/build-kuberay-images.yaml` does in CI,
# but for a local developer running `ma sandbox create`. The upstream
# Dockerfile at https://github.com/ray-project/kuberay/tree/master/historyserver/
# hardcodes GOARCH=amd64 in its `make build...` step; we patch it to
# honor `${TARGETARCH:-amd64}` so docker buildx can produce a real arm64
# binary on Apple Silicon hosts.
#
# Usage:
#   scripts/kuberay/build-kuberay-images.sh
#
# Environment overrides:
#   KUBERAY_DIR        — where to clone/find the upstream repo
#                        (default: ~/Code/kuberay)
#   KUBERAY_REF        — git ref to check out (default: v1.6.1)
#   IMAGE_TAG          — image tag (default: v0.1.0; matches sandbox manifests)
#   K3D_CLUSTERS       — space-separated list of k3d clusters to import into
#                        (default: "michelangelo-sandbox michelangelo-compute-0")
#   SKIP_IMPORT        — set to 1 to skip k3d import (build only)
#
# Exit codes:
#   0 = success
#   1 = build/import failure or unsupported host arch

set -euo pipefail

KUBERAY_DIR="${KUBERAY_DIR:-$HOME/Code/kuberay}"
KUBERAY_REF="${KUBERAY_REF:-v1.6.1}"
IMAGE_TAG="${IMAGE_TAG:-v0.1.0}"
K3D_CLUSTERS="${K3D_CLUSTERS:-michelangelo-sandbox michelangelo-compute-0 michelangelo-compute-1}"
SKIP_IMPORT="${SKIP_IMPORT:-0}"

# 1. Clone or update kuberay
if [ ! -d "$KUBERAY_DIR/.git" ]; then
  echo "Cloning kuberay to $KUBERAY_DIR (ref: $KUBERAY_REF)"
  git clone https://github.com/ray-project/kuberay.git "$KUBERAY_DIR"
fi
# Pull tags too — `git fetch origin <tag>` only writes FETCH_HEAD; the local
# tag ref is not updated, so a subsequent `git checkout <tag>` fails with
# "pathspec did not match" on clones that didn't already have the tag.
git -C "$KUBERAY_DIR" fetch --quiet --tags origin
git -C "$KUBERAY_DIR" checkout --quiet "$KUBERAY_REF"

cd "$KUBERAY_DIR/historyserver"

# 2. Patch Dockerfiles to honor TARGETARCH (idempotent — safe to re-run).
#    Without this, `docker buildx build --platform linux/arm64` produces an
#    image labeled arm64 that contains an amd64 binary (because GOARCH is
#    hardcoded), and k3d on Apple Silicon refuses it with
#    "no match for platform in manifest".
sed -i.bak \
  -e 's|^ARG ENABLE_RACE=false$|ARG ENABLE_RACE=false\nARG TARGETARCH|' \
  -e 's|GOARCH=amd64|GOARCH=${TARGETARCH:-amd64}|g' \
  Dockerfile.collector
sed -i.bak \
  -e 's|^ARG GOPROXY=https://proxy.golang.org,direct$|ARG GOPROXY=https://proxy.golang.org,direct\nARG TARGETARCH|' \
  -e 's|GOARCH=amd64|GOARCH=${TARGETARCH:-amd64}|g' \
  Dockerfile.historyserver
rm -f Dockerfile.*.bak

# 3. Detect host arch and build native single-arch images via buildx.
HOST_ARCH=$(uname -m)
case "$HOST_ARCH" in
  arm64|aarch64) BUILD_PLATFORM=linux/arm64 ;;
  x86_64|amd64)  BUILD_PLATFORM=linux/amd64 ;;
  *) echo "Unsupported host arch: $HOST_ARCH" >&2; exit 1 ;;
esac
echo "Building kuberay images for $BUILD_PLATFORM (tag: $IMAGE_TAG)"

docker buildx build --platform "$BUILD_PLATFORM" \
  -t "kuberay-collector:$IMAGE_TAG" --load -f Dockerfile.collector .
docker buildx build --platform "$BUILD_PLATFORM" \
  -t "kuberay-historyserver:$IMAGE_TAG" --load -f Dockerfile.historyserver .

# 4. Import into k3d clusters so the sandbox manifests can find them.
if [ "$SKIP_IMPORT" = "1" ]; then
  echo "SKIP_IMPORT=1 — skipping k3d import"
  exit 0
fi

for cluster in $K3D_CLUSTERS; do
  if k3d cluster list "$cluster" >/dev/null 2>&1; then
    echo "Importing into k3d cluster: $cluster"
    k3d image import "kuberay-collector:$IMAGE_TAG" "kuberay-historyserver:$IMAGE_TAG" \
      -c "$cluster"
  else
    echo "k3d cluster '$cluster' not found — skipping (run \`ma sandbox create\` first)"
  fi
done

# 5. Sanity-check binaries match the host arch.
#    ELF e_machine field (bytes 18-19): b7 00 = aarch64, 3e 00 = x86_64.
echo "Binary arch sanity check:"
docker run --rm --entrypoint sh "kuberay-collector:$IMAGE_TAG" \
  -c 'head -c 22 /usr/local/bin/collector | od -An -tx1 -j18 -N2'
docker run --rm --entrypoint sh "kuberay-historyserver:$IMAGE_TAG" \
  -c 'head -c 22 /usr/local/bin/historyserver | od -An -tx1 -j18 -N2'

echo "Done. Sandbox manifests reference these tags directly — no further action needed."