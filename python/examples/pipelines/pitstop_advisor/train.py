"""Train + package + register task for the Pit Crew Advisor demo model.

Trains two independent XGBoost regressors on the synthetic lap dataset,
packages them as a deployable Triton model via `CustomTritonPackager`, and
registers both the raw checkpoint and the packaged bundle with the MA model
registry — unlike california_housing_xgb's train.py (which only registers a
bare training checkpoint), this produces something `ma revision apply` /
`ma deployment apply` can actually serve.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import TYPE_CHECKING

import michelangelo.uniflow.core as uniflow
from michelangelo.uniflow.plugins.ray import RayTask

if TYPE_CHECKING:
    from examples.pipelines.pitstop_advisor.generate_data import GeneratedData

log = logging.getLogger(__name__)

__all__ = ["TrainResult", "train"]

_MODEL_NAME = "pitstop-advisor"
_MODEL_CLASS = "examples.pipelines.pitstop_advisor.model.PitStopAdvisorModel"


@dataclass
class TrainResult:
    """Container for training results.

    Attributes:
        artifact_uri: s3:// URI of the packaged (deployable) Triton model
            directory, suitable for a Revision to point at once promoted.
        metrics: Evaluation RMSE per target.
    """

    artifact_uri: str
    metrics: dict | None = None


@uniflow.task(
    config=RayTask(
        head_cpu=2,
        head_memory="2Gi",
        worker_instances=0,
    ),
)
def train(data: GeneratedData) -> TrainResult:
    """Train, package, and register the Pit Crew Advisor model.

    Args:
        data: Synthetic (lane, track_grip) -> (speed_cap_cms, caution_buffer_cm)
            rows from `generate_data`.

    Returns:
        TrainResult with the packaged model's storage URI and eval metrics.
    """
    import shutil
    import tempfile

    import numpy as np
    import xgboost

    from examples.pipelines.pitstop_advisor.model import PitStopAdvisorModel

    features = np.array(data.features, dtype=np.float32)
    n = len(features)
    split = max(1, int(n * 0.8))

    def _train_one(target: list[float]) -> tuple[xgboost.Booster, float]:
        y = np.array(target, dtype=np.float32)
        dtrain = xgboost.DMatrix(features[:split], label=y[:split])
        dvalid = xgboost.DMatrix(features[split:], label=y[split:])
        booster = xgboost.train(
            {"objective": "reg:squarederror", "max_depth": 4, "eta": 0.1},
            dtrain=dtrain,
            num_boost_round=50,
            evals=[(dvalid, "validation")],
            verbose_eval=False,
        )
        preds = booster.predict(dvalid)
        rmse = float(np.sqrt(np.mean((preds - y[split:]) ** 2)))
        return booster, rmse

    speed_booster, speed_rmse = _train_one(data.speed_cap_cms)
    caution_booster, caution_rmse = _train_one(data.caution_buffer_cm)

    model = PitStopAdvisorModel(
        speed_booster=speed_booster, caution_booster=caution_booster
    )

    tmp_dir = tempfile.mkdtemp(prefix="pitstop_advisor_")
    try:
        artifacts_dir = os.path.join(tmp_dir, "artifacts")
        model.save(artifacts_dir)

        package_dir = os.path.join(tmp_dir, "package")
        raw_uri, deployable_uri = _package_and_upload(artifacts_dir, package_dir)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    metrics = {
        "speed_cap_rmse": round(speed_rmse, 4),
        "caution_buffer_rmse": round(caution_rmse, 4),
    }

    _register_model(raw_uri=raw_uri, deployable_uri=deployable_uri, metrics=metrics)

    return TrainResult(artifact_uri=deployable_uri, metrics=metrics)


def _package_and_upload(artifacts_dir: str, package_dir: str) -> tuple[str, str]:
    """Package the model as a Triton artifact and upload it plus the raw checkpoint.

    Returns:
        (raw_uri, deployable_uri) — s3:// URIs for the raw checkpoint and the
        packaged Triton directory respectively. Both live under a
        timestamp-versioned prefix so each training run's artifacts stay
        distinct and a Revision snapshot remains reproducible (see
        docs/user-guides/train-and-deploy-models/model-registry-guide.md's
        "When to create a new Revision vs. update").
    """
    import time

    from michelangelo.lib.model_manager.packager.custom_triton import (
        CustomTritonPackager,
    )
    from michelangelo.lib.model_manager.schema import (
        DataType,
        ModelSchema,
        ModelSchemaItem,
    )

    schema = ModelSchema(
        input_schema=[
            ModelSchemaItem(name="features", data_type=DataType.FLOAT, shape=[2])
        ],
        output_schema=[
            ModelSchemaItem(name="settings", data_type=DataType.FLOAT, shape=[2])
        ],
    )

    packager = CustomTritonPackager()
    created_package_dir = packager.create_model_package(
        model_path=artifacts_dir,
        model_class=_MODEL_CLASS,
        model_schema=schema,
        model_name=_MODEL_NAME,
        dest_model_path=package_dir,
        include_import_prefixes=["examples.pipelines.pitstop_advisor"],
    )

    run_id = time.strftime("%Y%m%d-%H%M%S")
    raw_uri = _upload_dir_to_s3(artifacts_dir, f"{_MODEL_NAME}/{run_id}/raw/")
    # Deployable bundle must land at the prefix root (config.pbtxt alongside
    # it), matching the sandbox model-sync daemon's `aws s3 sync <path> ...`
    # expectation of a raw Triton model-repo layout.
    deployable_uri = _upload_dir_to_s3(created_package_dir, f"{_MODEL_NAME}/{run_id}/")

    return raw_uri, deployable_uri


def _upload_dir_to_s3(local_dir: str, prefix: str) -> str:
    """Upload a directory tree file-by-file to `s3://<bucket>/<prefix>`.

    Not tarred: MinioStorageBackend.upload() archives directories into a
    single object, which would break Triton's expected raw multi-file model
    repo layout (config.pbtxt + a version subdirectory) that the sandbox's
    model-sync daemon (`aws s3 sync`) reads directly.
    """
    from urllib.parse import urlparse

    from minio import Minio

    s3_endpoint = os.environ.get("AWS_ENDPOINT_URL", "")
    parsed = urlparse(s3_endpoint) if s3_endpoint else None
    endpoint = parsed.netloc if parsed else None
    if not endpoint:
        raise ValueError(
            f"AWS_ENDPOINT_URL={s3_endpoint!r} is missing a scheme or unset. "
            "Expected a full URL like http://minio:9091 (set automatically in "
            "the sandbox via the michelangelo-config ConfigMap)."
        )
    secure = parsed.scheme == "https"
    access_key = os.environ.get("AWS_ACCESS_KEY_ID", "")
    secret_key = os.environ.get("AWS_SECRET_ACCESS_KEY", "")
    bucket = os.environ.get("AWS_S3_BUCKET", "deploy-models")

    client = Minio(
        endpoint, access_key=access_key, secret_key=secret_key, secure=secure
    )
    if not client.bucket_exists(bucket):
        client.make_bucket(bucket)

    for dirpath, _dirnames, filenames in os.walk(local_dir):
        for filename in filenames:
            file_path = os.path.join(dirpath, filename)
            rel_path = os.path.relpath(file_path, local_dir)
            object_name = prefix + rel_path.replace(os.sep, "/")
            client.fput_object(bucket, object_name, file_path)

    log.info("Uploaded %s to s3://%s/%s", local_dir, bucket, prefix)
    return f"s3://{bucket}/{prefix}"


def _register_model(raw_uri: str, deployable_uri: str, metrics: dict) -> None:
    """Register the packaged model with the MA model registry.

    Non-fatal: a registry failure is logged as a warning so the pipeline task
    still reports success and the TrainResult is returned to the caller.
    """
    from michelangelo.lib.model_manager.registry.api_client import APIRegistryClient

    endpoint = os.environ.get(
        "MA_API_SERVER",
        "michelangelo-apiserver.default.svc.cluster.local:15566",
    )
    namespace = os.environ.get("MA_NAMESPACE", "ma-examples")

    try:
        with APIRegistryClient(
            endpoint=endpoint, namespace=namespace, insecure=True
        ) as registry:
            registered = registry.register_model(
                name=_MODEL_NAME,
                artifact_uri=raw_uri,
                deployable_artifact_uri=deployable_uri,
                description=(
                    "Pit Crew Advisor: recommends pit-lane speed cap and caution buffer"
                ),
                labels={"training_framework": "xgboost", "demo": "kubecon-pit-stop"},
                metadata=metrics,
            )
        log.info(
            "Model registered: %s v%s at %s",
            registered.name,
            registered.version,
            registered.registry_uri,
        )
    except Exception as exc:
        log.warning("Model registration failed (non-fatal): %s", exc)
