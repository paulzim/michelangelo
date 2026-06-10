"""Ray preprocessing task for the California Housing XGBoost workflow.

Casts selected columns to float type using Ray and returns the preprocessed
training and validation datasets.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import michelangelo.uniflow.core as uniflow
from michelangelo.uniflow.plugins.ray import RayTask
from michelangelo.workflow.variables import DatasetVariable

log = logging.getLogger(__name__)

__all__ = ["PreprocessResult", "preprocess"]


@dataclass
class PreprocessResult:
    """Container for preprocessing results.

    Attributes:
        train_data: Training dataset.
        validation_data: Validation dataset.
    """

    train_data: DatasetVariable
    validation_data: DatasetVariable


@uniflow.task(
    config=RayTask(
        head_cpu=1,
        head_memory="2Gi",
        worker_instances=0,
    ),
    cache_enabled=False,  # off for tutorial simplicity; enable in production
)
def preprocess(
    cast_float_columns: list[str],
    train_dv: DatasetVariable,
    validation_dv: DatasetVariable,
) -> PreprocessResult:
    """Preprocess datasets using Ray to cast columns to float type.

    Args:
        cast_float_columns: List of column names to cast to float type.
        train_dv: Training DatasetVariable containing Ray Dataset.
        validation_dv: Validation DatasetVariable containing Ray Dataset.

    Returns:
        PreprocessResult containing preprocessed training and validation datasets.
    """
    train_dv.load_ray_dataset()
    train_data = train_dv.value

    validation_dv.load_ray_dataset()
    validation_data = validation_dv.value

    def cast_float(batch):
        for col in cast_float_columns:
            if col in batch.columns:
                batch[col] = batch[col].astype("float32")
        return batch

    train_data = train_data.map_batches(cast_float, batch_format="pandas")
    validation_data = validation_data.map_batches(cast_float, batch_format="pandas")

    train_dv_pr = DatasetVariable.create(train_data)
    train_dv_pr.save_ray_dataset()

    validation_dv_pr = DatasetVariable.create(validation_data)
    validation_dv_pr.save_ray_dataset()

    log.info("Processed train dataset schema: %s", train_data.schema())

    return PreprocessResult(
        train_data=train_dv_pr,
        validation_data=validation_dv_pr,
    )
