"""Build and upload the uniflowTar for hello-world.

Run this from the python/ directory after port-forwarding MinIO:

    kubectl port-forward svc/minio 9091:9091 -n default &
    cd /Users/pzimme1/GitHub/michelangelo/python
    poetry run python examples/hello_world/.docker/rebuild_tar.py
"""

import os
import sys

sys.path.insert(0, ".")

os.environ.setdefault("AWS_ENDPOINT_URL", "http://localhost:9091")
os.environ.setdefault("AWS_ACCESS_KEY_ID", "minioadmin")
os.environ.setdefault("AWS_SECRET_ACCESS_KEY", "minioadmin")

from examples.hello_world.hello_world import hello_world_workflow
from michelangelo.uniflow.core.build import build
import s3fs

print("Building uniflow tar...")
package = build(hello_world_workflow)
tarball_bytes = package.to_tarball_bytes()
print(f"Built {len(tarball_bytes):,} bytes (main function: {package.main_function})")

target = "michelangelo/uniflow/ma-dev-test_hello-world.tar.gz"
fs = s3fs.S3FileSystem(
    key=os.environ["AWS_ACCESS_KEY_ID"],
    secret=os.environ["AWS_SECRET_ACCESS_KEY"],
    endpoint_url=os.environ["AWS_ENDPOINT_URL"],
    use_ssl=False,
)
with fs.open(target, "wb") as f:
    f.write(tarball_bytes)

print(f"Uploaded to s3://{target}")
print("Done. Re-apply pipeline.yaml and apply pipelinerun.yaml next.")
