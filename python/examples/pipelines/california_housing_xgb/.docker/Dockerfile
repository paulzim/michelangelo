FROM ghcr.io/michelangelo-ai/examples:main
COPY examples/pipelines/california_housing_xgb /app/examples/pipelines/california_housing_xgb/
# Patch ref.py to filter __init__ params when reconstructing dataclasses.
# The base image predates the fix for DatasetVariable.__init__ not accepting
# _io_metadata, causing TypeError at task result serialization time.
COPY examples/pipelines/california_housing_xgb/.docker/ref.py \
     /app/michelangelo/uniflow/core/ref.py
