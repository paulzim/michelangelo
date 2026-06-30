"""
Register the california_housing_xgb pipeline tar to MinIO.

Builds the Starlark workflow tarball, opens a temporary kubectl port-forward
to MinIO, uploads via s3fs, and prints the kubectl patch command you need next.

Run from /Users/pzimme1/GitHub/michelangelo/python/ :

    PYTHONPATH=. poetry run python examples/pipelines/california_housing_xgb/register_tar.py
"""

import logging
import os
import socket
import subprocess
import sys
import tempfile
import time

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

# ── 1. Build tarball ──────────────────────────────────────────────────────────
try:
    from examples.pipelines.california_housing_xgb.california_housing_xgb import (
        train_workflow,
    )
except ImportError as e:
    print(f"ERROR: could not import train_workflow: {e}", file=sys.stderr)
    print(
        "Make sure you are running from the python/ directory with PYTHONPATH=.",
        file=sys.stderr,
    )
    sys.exit(1)

from michelangelo.uniflow.core.build import build

logging.info("Building Starlark tarball from train_workflow...")
tarball_bytes = build(train_workflow).to_tarball_bytes()
logging.info("Built tarball: %d bytes", len(tarball_bytes))

# ── 2. Start port-forward (background) ───────────────────────────────────────
LOCAL_PORT = 19000  # use a non-standard port to avoid conflicts
PF_CMD = [
    "kubectl", "port-forward", "svc/minio", f"{LOCAL_PORT}:9091", "-n", "default",
]
logging.info("Starting port-forward: %s", " ".join(PF_CMD))
pf = subprocess.Popen(PF_CMD, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)

# Wait up to 15 s for the port to open
deadline = time.time() + 15
ready = False
while time.time() < deadline:
    try:
        with socket.create_connection(("127.0.0.1", LOCAL_PORT), timeout=1):
            ready = True
            break
    except OSError:
        time.sleep(0.5)

if not ready:
    pf.kill()
    stderr_out = pf.stderr.read().decode(errors="replace")
    print(f"ERROR: port-forward did not become ready within 15 s.", file=sys.stderr)
    print(f"kubectl output: {stderr_out}", file=sys.stderr)
    print(
        "\nMake sure the minio pod is Running in the default namespace:",
        file=sys.stderr,
    )
    print("  kubectl get pod minio -n default", file=sys.stderr)
    sys.exit(1)

logging.info("Port-forward ready on localhost:%d", LOCAL_PORT)

# ── 3. Upload via s3fs ────────────────────────────────────────────────────────
MINIO_ENDPOINT = f"http://127.0.0.1:{LOCAL_PORT}"
os.environ["AWS_ACCESS_KEY_ID"] = "minioadmin"
os.environ["AWS_SECRET_ACCESS_KEY"] = "minioadmin"
os.environ["AWS_ENDPOINT_URL"] = MINIO_ENDPOINT
os.environ["FSSPEC_S3_ENDPOINT_URL"] = MINIO_ENDPOINT
os.environ["AWS_DEFAULT_REGION"] = "us-east-1"

TAR_NAME = "ma-examples_california-housing-xgb.tar.gz"
S3_PATH = f"s3://michelangelo/uniflow/{TAR_NAME}"

try:
    import fsspec

    logging.info("Uploading to %s ...", S3_PATH)
    with fsspec.open(S3_PATH, "wb", anon=False) as f:
        f.write(tarball_bytes)
    logging.info("Upload complete.")
finally:
    pf.kill()

print("\n=== SUCCESS ===")
print(f"Tar uploaded to: {S3_PATH}")
print("\nNow run these two commands on the Mac:")
print()
print(
    f"kubectl patch pipeline california-housing-xgb -n ma-examples "
    f"--type=merge "
    f"-p '{{\"spec\":{{\"manifest\":{{\"uniflowTar\":\"{S3_PATH}\"}}}}}}'"
)
print()
print(
    "kubectl apply -f "
    "/Users/pzimme1/GitHub/michelangelo/python/examples/pipelines/"
    "california_housing_xgb/pipelinerun.yaml"
)
