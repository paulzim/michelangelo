FROM ghcr.io/michelangelo-ai/examples:main
COPY examples/pipelines/california_housing_xgb /app/examples/pipelines/california_housing_xgb/
# Patch ref.py and codec.py to filter __init__ params when reconstructing
# dataclasses. The base image predates the fix for DatasetVariable.__init__
# not accepting _io_metadata, causing TypeError at task boundaries.
COPY examples/pipelines/california_housing_xgb/.docker/ref.py \
     /app/michelangelo/uniflow/core/ref.py
COPY examples/pipelines/california_housing_xgb/.docker/patch_codec.py /tmp/patch_codec.py
RUN python3 /tmp/patch_codec.py
# Overlay SDK modules added after the GHCR base image was published
# (PRs #1261 #1276 #1282 #1429: pusher, artifact_manager, model_manager, variables).
# The base image has PYTHONPATH=/app and the SDK lives at /app/michelangelo/.
COPY michelangelo/workflow/tasks/pusher/ /app/michelangelo/workflow/tasks/pusher/
COPY michelangelo/workflow/schema/pusher.py /app/michelangelo/workflow/schema/pusher.py
COPY michelangelo/workflow/variables/types.py /app/michelangelo/workflow/variables/types.py
COPY michelangelo/workflow/variables/metadata.py /app/michelangelo/workflow/variables/metadata.py
COPY michelangelo/lib/artifact_manager/ /app/michelangelo/lib/artifact_manager/
COPY michelangelo/lib/model_manager/ /app/michelangelo/lib/model_manager/
# evaluation_report_pb2 added in PR #1236 — not present in the base image.
# Required at module-load time by pusher.plugins.__init__ and push.py.
COPY michelangelo/gen/api/v2/evaluation_report_pb2.py /app/michelangelo/gen/api/v2/evaluation_report_pb2.py
COPY michelangelo/gen/api/v2/evaluation_report_svc_pb2.py /app/michelangelo/gen/api/v2/evaluation_report_svc_pb2.py
# io.py: base image calls _ensure_s3a_config() at module level (old version),
# which invokes SparkSession.builder.getOrCreate() at import time and fails in
# the k3d driver pod. Current version is lazy (called only from read_data /
# SparkIO.write). Overlay with the current lazy version to fix the import crash.
# (Kept as a precaution; push_step now uses RayTask so SparkTask is not imported
# by any example code, but other SDK code may still pull in spark/io.py.)
COPY michelangelo/uniflow/plugins/spark/io.py /app/michelangelo/uniflow/plugins/spark/io.py
# Diagnostic: verify non-JVM import chain works in the container at build time.
# SparkTask / push module are skipped here (they need a live JVM/SparkContext).
COPY examples/pipelines/california_housing_xgb/.docker/diagnostic.py /tmp/diagnostic.py
RUN python3 /tmp/diagnostic.py
