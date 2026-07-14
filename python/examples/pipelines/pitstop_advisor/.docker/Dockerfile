FROM ghcr.io/michelangelo-ai/examples:main
# minio: raw S3 upload of the packaged Triton model directory (train.py's
# _upload_dir_to_s3), so the Triton model-repo layout isn't tarred like
# MinioStorageBackend.upload() would do.
RUN python3 -m pip install --no-cache-dir \
    --trusted-host pypi.org \
    --trusted-host files.pythonhosted.org \
    "minio>=7,<8"
COPY examples/pipelines/pitstop_advisor /app/examples/pipelines/pitstop_advisor/
# Overlay model_manager (CustomTritonPackager, ModelSchema, APIRegistryClient)
# added after the base image was published — same gap documented in
# california_housing_xgb/.docker/Dockerfile, but pitstop_advisor doesn't use
# DatasetVariable/Pusher/sinks/artifact_manager, so none of those overlays
# are needed here (confirmed via `grep -rhoE "^\s*(from|import)
# michelangelo[a-zA-Z0-9_.]*" michelangelo/lib/model_manager/` — no
# dependency on artifact_manager, workflow.variables, or workflow.tasks.pusher).
COPY michelangelo/lib/model_manager/ /app/michelangelo/lib/model_manager/
# model_pb2 / model_svc_pb2 / model_svc_pb2_grpc: base image predates the
# model_artifact_uri / deployable_artifact_uri fields on ModelSpec that
# api_client.py (overlaid above) requires. Same fix as california_housing_xgb.
COPY michelangelo/gen/api/v2/model_pb2.py /app/michelangelo/gen/api/v2/model_pb2.py
COPY michelangelo/gen/api/v2/model_svc_pb2.py /app/michelangelo/gen/api/v2/model_svc_pb2.py
COPY michelangelo/gen/api/v2/model_svc_pb2_grpc.py /app/michelangelo/gen/api/v2/model_svc_pb2_grpc.py
# Diagnostic: verify the overlay + xgboost import chain works in the container
# at build time, before paying for a k3d image import + PipelineRun round trip.
COPY examples/pipelines/pitstop_advisor/.docker/diagnostic.py /tmp/diagnostic.py
RUN python3 /tmp/diagnostic.py
