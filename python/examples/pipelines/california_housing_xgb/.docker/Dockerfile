FROM ghcr.io/michelangelo-ai/examples:main
COPY examples/pipelines/california_housing_xgb /app/examples/pipelines/california_housing_xgb/
# Pre-cache the dataset so runtime tasks don't need network access.
# PYTHONHTTPSVERIFY=0 works around Zscaler MITM in the build environment.
RUN PYTHONHTTPSVERIFY=0 python3 -c \
    "from sklearn.datasets import fetch_california_housing; fetch_california_housing(as_frame=True)"
