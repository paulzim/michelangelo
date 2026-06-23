FROM ghcr.io/michelangelo-ai/examples:main
COPY examples/pipelines/california_housing_xgb /app/examples/pipelines/california_housing_xgb/
# Pre-cache the dataset so runtime tasks don't need network access.
# ssl._create_unverified_context works around Zscaler MITM in the build env.
RUN python3 -c "\
import ssl; ssl._create_default_https_context = ssl._create_unverified_context; \
from sklearn.datasets import fetch_california_housing; \
fetch_california_housing(as_frame=True)"
