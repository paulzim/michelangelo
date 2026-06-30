"""
Register the california_housing_xgb pipeline tar to MinIO.

Run from /Users/pzimme1/GitHub/michelangelo/python/ :

    PYTHONPATH=. poetry run python examples/pipelines/california_housing_xgb/register_tar.py

MinIO NodePort (localhost:30007) is used for the upload; the controllermgr
reads the same object via minio:9091 (same MinIO instance, same bucket).
"""

import logging
import os
import sys
import tempfile

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

# MinIO connection for k3d sandbox from Mac (NodePort)
os.environ.setdefault("AWS_ACCESS_KEY_ID", "minioadmin")
os.environ.setdefault("AWS_SECRET_ACCESS_KEY", "minioadmin")
os.environ.setdefault("AWS_ENDPOINT_URL", "http://localhost:30007")
os.environ.setdefault("AWS_DEFAULT_REGION", "us-east-1")
# Some versions of fsspec/s3fs read this variant instead of AWS_ENDPOINT_URL
os.environ.setdefault("FSSPEC_S3_ENDPOINT_URL", "http://localhost:30007")

try:
    from examples.pipelines.california_housing_xgb.california_housing_xgb import (
        train_workflow,
    )
except ImportError as e:
    print(f"ERROR: could not import train_workflow: {e}", file=sys.stderr)
    print("Make sure you are running from the python/ directory with PYTHONPATH=.", file=sys.stderr)
    sys.exit(1)

from michelangelo.uniflow.registration.uniflow_tar import prepare_uniflow_tar

with tempfile.TemporaryDirectory() as tmpdir:
    remote_path = prepare_uniflow_tar(
        project_name="ma-examples",
        pipeline_name="california-housing-xgb",
        output_dir=tmpdir,
        workflow_function=(
            "examples.pipelines.california_housing_xgb"
            ".california_housing_xgb.train_workflow"
        ),
        workflow_function_obj=train_workflow,
        storage_base_url="s3://michelangelo/uniflow",
    )

print("\n=== SUCCESS ===")
print(f"Tar uploaded to: {remote_path}")
print("\nRun this to patch the Pipeline CR, then resubmit the PipelineRun:")
print(
    f"kubectl patch pipeline california-housing-xgb -n ma-examples "
    f"--type=merge "
    f"-p '{{\"spec\":{{\"manifest\":{{\"uniflowTar\":\"{remote_path}\"}}}}}}'"
)
print(
    "\nkubectl apply -f "
    "/Users/pzimme1/GitHub/michelangelo/python/examples/pipelines/"
    "california_housing_xgb/pipelinerun.yaml"
)
