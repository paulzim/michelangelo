"""Native transform Ray task.

Applies the native transform spec (LogTransform, Normalization, MinMax,
Bucketization, Stack) to train and val datasets via Ray map_batches,
then writes the transformed data back to Parquet as DatasetVariables.

Mirrors the tabular_native_transform step in the production pipeline.
"""

import logging
import os
import uuid
from typing import Any

from michelangelo.lib.native_transform.runner import apply_native_transforms
from michelangelo.uniflow.core.decorator import task
from michelangelo.uniflow.plugins.ray.task import RayTask
from michelangelo.workflow.variables import DatasetVariable

logger = logging.getLogger(__name__)


def _transform_and_save(ray_ds, transform_specs: list[dict], columns_to_keep: list[str] | None) -> DatasetVariable:
    """Apply transforms, write to Parquet, return a DatasetVariable."""
    transformed = apply_native_transforms(ray_ds, transform_specs, columns_to_keep)

    storage_url = os.environ.get("UF_STORAGE_URL", "/tmp/uf_storage")
    # Strip file:// prefix for Ray's write_parquet (expects a local path or s3://)
    path = f"{storage_url.removeprefix('file://')}/{uuid.uuid4().hex}"
    os.makedirs(path, exist_ok=True)

    logger.info("Writing transformed dataset to %s", path)
    transformed.write_parquet(path)

    return DatasetVariable(path=path, metadata=None)


@task(config=RayTask())
def native_transform_task(
    config: dict[str, Any],
    train_data: DatasetVariable,
    val_data: DatasetVariable,
) -> dict[str, DatasetVariable]:
    """Apply native transforms to train and val datasets.

    Args:
        config: Dict with keys:
            - ``transform_specs``: list of transform spec dicts
              (mirrors native_transform_specs.yaml).
            - ``columns_to_keep``: optional list of column names to retain
              after transforms (drops intermediates like *_padded columns).
        train_data: Training dataset (post-feature-prep / post-DSL schema).
        val_data: Validation dataset (same schema).

    Returns:
        Dict with ``train_dataset`` and ``val_dataset`` DatasetVariables
        in the post-native-transform schema ready for tabular_trainer.
    """
    transform_specs: list[dict] = config["transform_specs"]
    columns_to_keep: list[str] | None = config.get("columns_to_keep")

    train_data.load_ray_dataset()
    val_data.load_ray_dataset()

    logger.info("Applying %d transforms to train dataset", len(transform_specs))
    train_var = _transform_and_save(train_data.value, transform_specs, columns_to_keep)

    logger.info("Applying %d transforms to val dataset", len(transform_specs))
    val_var = _transform_and_save(val_data.value, transform_specs, columns_to_keep)

    return {"train_dataset": train_var, "val_dataset": val_var}
